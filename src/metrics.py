"""Paper-aligned HV and normalized IGD+ metric construction.

Xue et al. (2024) report HV, after per-objective min-max normalization, using
the raw-space reference points in Tables 3 and 6 and Appendix B.  The paper
discusses IGD but does not report it because a true PF is unavailable for many
real-world tasks.  We therefore expose IGD+ as an explicitly documented
extension: its reference front and candidate set are transformed with exactly
the same offline-data bounds used by HV.
"""

from __future__ import annotations

import numpy as np
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from src.problem_specs import get_paper_reference_point


def _finite_objective_rows(objective_values):
    if objective_values is None:
        return None
    values = np.asarray(objective_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("objective_values must be a 2D array.")
    values = values[np.all(np.isfinite(values), axis=1)]
    return values if len(values) else None


def _objective_bounds_from_values(objective_values):
    values = _finite_objective_rows(objective_values)
    if values is None:
        return None, None
    return np.min(values, axis=0), np.max(values, axis=0)


def _problem_point(problem, method_name):
    method = getattr(problem, method_name, None)
    if method is None:
        return None
    try:
        value = method()
    except (NotImplementedError, TypeError, ValueError, FileNotFoundError):
        return None
    if value is None:
        return None
    value = np.asarray(value, dtype=float).reshape(-1)
    return value if np.all(np.isfinite(value)) else None


def get_problem_y_bounds(problem_name, problem=None, objective_values=None, n_var=None):
    """Return the min-max bounds used for metric normalization.

    Offline-data bounds are authoritative, matching the paper's normalization
    protocol. Problem ideal/nadir points are only a fallback for callers that
    do not provide the offline objective matrix.
    """

    obj_min, obj_max = _objective_bounds_from_values(objective_values)
    if obj_min is None and problem is not None:
        obj_min = _problem_point(problem, "get_ideal_point")
        obj_max = _problem_point(problem, "get_nadir_point")
    if obj_min is None or obj_max is None:
        raise ValueError(
            f"Objective normalization bounds require offline objective values "
            f"for problem '{problem_name}'."
        )
    if obj_min.shape != obj_max.shape:
        raise ValueError("Objective minimum and maximum shapes do not match.")
    scale = obj_max - obj_min
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError(
            f"Every objective must have a positive finite offline-data range; "
            f"got ranges {scale} for '{problem_name}'."
        )
    return obj_min, obj_max


def normalize_objectives(values, obj_min, obj_max):
    values = np.asarray(values, dtype=float)
    obj_min = np.asarray(obj_min, dtype=float)
    obj_max = np.asarray(obj_max, dtype=float)
    if values.ndim != 2 or values.shape[1] != obj_min.shape[0]:
        raise ValueError(
            f"Expected objective array with {obj_min.shape[0]} columns; "
            f"received shape {values.shape}."
        )
    return (values - obj_min) / (obj_max - obj_min)


def _simplex_reference_directions(n_obj, n_partitions=24):
    """Create deterministic simplex-lattice directions without extra state."""

    if n_obj == 2:
        first = np.arange(n_partitions + 1, dtype=float)
        return np.column_stack([first, n_partitions - first]) / n_partitions
    if n_obj == 3:
        directions = [
            (i, j, n_partitions - i - j)
            for i in range(n_partitions + 1)
            for j in range(n_partitions + 1 - i)
        ]
        return np.asarray(directions, dtype=float) / n_partitions
    return None


def _call_pareto_front(method, n_obj):
    """Handle both zero-argument and ref-dir based pymoo PF APIs."""

    try:
        front = method()
        if front is not None:
            return front
    except (NotImplementedError, TypeError, ValueError, FileNotFoundError):
        pass

    ref_dirs = _simplex_reference_directions(n_obj)
    if ref_dirs is None:
        return None
    for args, kwargs in (
        ((ref_dirs,), {}),
        ((), {"ref_dirs": ref_dirs}),
        ((), {"n_pareto_points": len(ref_dirs)}),
    ):
        try:
            return method(*args, **kwargs)
        except (NotImplementedError, TypeError, ValueError, FileNotFoundError):
            continue
    return None


def _problem_reference_front(problem):
    for method_name in ("pareto_front", "get_pareto_front"):
        method = getattr(problem, method_name, None)
        if method is None or not callable(method):
            continue
        front = _call_pareto_front(method, int(getattr(problem, "n_obj", 0)))
        front = _finite_objective_rows(front)
        if front is not None:
            return front, "problem_pareto_front"
    return None, None


def get_reference_front(problem, objective_values, fallback_reference_values=None):
    """Return a true/reference PF, falling back to one fixed offline ND front.

    For official-pool experiments, ``fallback_reference_values`` should be the
    full official training pool so tasks such as MO-Portfolio and Molecule use
    one fixed evaluation front across sample sizes, seeds, and methods.  These
    values are evaluation-only: they may define the fixed metric space, but
    must never enter surrogate fitting or optimizer normalization.
    """

    front, source = _problem_reference_front(problem)
    expected_n_obj = int(getattr(problem, "n_obj", 0))
    if (
        front is not None
        and expected_n_obj > 0
        and front.shape[1] == expected_n_obj
    ):
        return front, source

    values = _finite_objective_rows(
        fallback_reference_values
        if fallback_reference_values is not None
        else objective_values
    )
    if values is None:
        raise ValueError("IGD+ requires a problem Pareto front or offline objectives.")
    indices = NonDominatedSorting().do(values, only_non_dominated_front=True)
    source = (
        "official_training_pool_non_dominated_front"
        if fallback_reference_values is not None
        else "offline_non_dominated_front"
    )
    return values[np.asarray(indices, dtype=int)], source


def get_metrics(problem_name, problem, n_var=None, n_obj=None, objective_values=None):
    """Construct normalized-space HV using the paper's raw reference point."""

    obj_min, obj_max = get_problem_y_bounds(
        problem_name,
        problem=problem,
        objective_values=objective_values,
        n_var=n_var,
    )
    raw_ref_point = get_paper_reference_point(problem_name)
    if raw_ref_point.shape != obj_min.shape:
        raise ValueError(
            f"Paper reference point for '{problem_name}' has "
            f"{raw_ref_point.shape[0]} objectives, but the problem has "
            f"{obj_min.shape[0]}."
        )
    ref_point = (raw_ref_point - obj_min) / (obj_max - obj_min)
    hv = HV(ref_point=ref_point)
    return hv, obj_min, obj_max, ref_point


def get_igd_plus(
    problem,
    obj_min,
    obj_max,
    objective_values,
    fallback_reference_values=None,
):
    """Construct IGD+ in the same normalized objective space as HV."""

    reference_front, source = get_reference_front(
        problem,
        objective_values,
        fallback_reference_values=fallback_reference_values,
    )
    normalized_front = normalize_objectives(reference_front, obj_min, obj_max)
    return IGDPlus(normalized_front), source
