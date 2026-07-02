import numpy as np
from pymoo.indicators.hv import HV

from src.real_world_problems import get_real_world_problem_y_bounds


OBJECTIVE_BOUND_OVERRIDES = {
    "__default_2obj__": {
        "obj_min": [0.0, 0.0],
    },
    "re22": {
        "obj_min": [0.0, -1500.0],
        "obj_max": [None, 1.2e10],
    },
    "re24": {
        "obj_min": [-1.0, -1.0],
        "obj_max": [6000.0, 100.0],
    },
    "re25": {
        "obj_min": [-10.0, -60000.0],
    },
    "welded-beam": {
        "obj_min": [0.0, -200.0],
        "obj_max": [300.0, 160.0],
    },
    "welded_beam": {
        "obj_min": [0.0, -200.0],
        "obj_max": [300.0, 160.0],
    },
    "mo-portfolio": {
        "obj_min": [0.0, -0.5],
    },
    "mo_portfolio": {
        "obj_min": [0.0, -0.5],
    },
    "portfolio": {
        "obj_min": [0.0, -0.5],
    },
}


SUPPORTED_METRICS_PROBLEMS = {
    "truss2d",
    "welded-beam",
    "welded_beam",
    "re21",
    "re21-exact-v0",
    "re22",
    "re22-exact-v0",
    "re23",
    "re23-exact-v0",
    "re24",
    "re24-exact-v0",
    "re25",
    "re25-exact-v0",
    "mo-portfolio",
    "mo_portfolio",
    "portfolio",
    "portfolio-exact-v0",
}


def _objective_bounds_from_values(objective_values):
    if objective_values is None:
        return None, None
    values = np.asarray(objective_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("objective_values must be a 2D array.")
    if values.shape[0] == 0:
        return None, None
    finite_rows = np.all(np.isfinite(values), axis=1)
    if not np.any(finite_rows):
        return None, None
    values = values[finite_rows]
    return np.min(values, axis=0), np.max(values, axis=0)


def _is_supported_metrics_problem(problem_name):
    problem_name = str(problem_name).strip().lower()
    problem_key = problem_name.replace("_", "-")
    return (
        problem_name in SUPPORTED_METRICS_PROBLEMS
        or problem_key in SUPPORTED_METRICS_PROBLEMS
    )


def _apply_component_override(values, override):
    values = np.asarray(values, dtype=float).copy()
    if override is None:
        return values

    override = list(override)
    if len(override) != values.shape[0]:
        raise ValueError(
            f"Objective bound override length {len(override)} does not match "
            f"objective dimension {values.shape[0]}."
        )
    for idx, value in enumerate(override):
        if value is not None:
            values[idx] = float(value)
    return values


def _apply_objective_bound_overrides(problem_name, obj_min, obj_max):
    if obj_min is None or obj_max is None:
        return obj_min, obj_max

    problem_name = str(problem_name).strip().lower()
    obj_min = np.asarray(obj_min, dtype=float)
    obj_max = np.asarray(obj_max, dtype=float)

    override = {}
    if obj_min.shape[0] == 2:
        override.update(OBJECTIVE_BOUND_OVERRIDES["__default_2obj__"])
    override.update(OBJECTIVE_BOUND_OVERRIDES.get(problem_name, {}))

    obj_min = _apply_component_override(obj_min, override.get("obj_min"))
    obj_max = _apply_component_override(obj_max, override.get("obj_max"))
    return obj_min, obj_max


def get_problem_y_bounds(problem_name, n_var=None):
    problem_name = str(problem_name).strip().lower()
    problem_key = problem_name.replace("_", "-")

    if not _is_supported_metrics_problem(problem_name):
        obj_min, obj_max = None, None
    elif problem_key == 'truss2d':
        obj_min = np.array([0,0])
        obj_max = np.array([0.06,1.5e10])
    elif problem_key == 'welded-beam':
        obj_min = np.array([0,0])
        obj_max = np.array([30,160])
    else:
        obj_min, obj_max = get_real_world_problem_y_bounds(problem_name)

    return _apply_objective_bound_overrides(problem_name, obj_min, obj_max)


def get_metrics(problem_name, problem, n_var=None, n_obj=None, objective_values=None):
    problem_name = str(problem_name).strip().lower()
    supported_problem = _is_supported_metrics_problem(problem_name)
    if n_var is None and hasattr(problem, "n_var"):
        n_var = problem.n_var

    obj_min, obj_max = get_problem_y_bounds(problem_name, n_var=n_var)
    if (
        supported_problem
        and (obj_min is None or obj_max is None)
        and hasattr(problem, "get_ideal_point")
    ):
        obj_min = problem.get_ideal_point()
        obj_max = problem.get_nadir_point()
    if supported_problem and (obj_min is None or obj_max is None):
        obj_min, obj_max = _objective_bounds_from_values(objective_values)
    obj_min, obj_max = _apply_objective_bound_overrides(problem_name, obj_min, obj_max)

    if obj_min is None or obj_max is None:
        raise ValueError(
            f"Objective bounds are not configured for problem '{problem_name}'."
        )

    ref_point = np.array([1.1, 1.1])
    hv = HV(ref_point=ref_point)

    return hv, obj_min, obj_max, ref_point
