import contextlib
import io
import os
import sys
import warnings
from pathlib import Path

import numpy as np


OFFLINE_MOO_PROBLEM_NAMES = {
    "re21": "re21",
    "re21-exact-v0": "re21",
    "re22": "re22",
    "re22-exact-v0": "re22",
    "re23": "re23",
    "re23-exact-v0": "re23",
    "re24": "re24",
    "re24-exact-v0": "re24",
    "re25": "re25",
    "re25-exact-v0": "re25",
    "re31": "re31",
    "re31-exact-v0": "re31",
    "re32": "re32",
    "re32-exact-v0": "re32",
    "re33": "re33",
    "re33-exact-v0": "re33",
    "re34": "re34",
    "re34-exact-v0": "re34",
    "re35": "re35",
    "re35-exact-v0": "re35",
    "re36": "re36",
    "re36-exact-v0": "re36",
    "re37": "re37",
    "re37-exact-v0": "re37",
    "mo-portfolio": "portfolio",
    "mo_portfolio": "portfolio",
    "portfolio": "portfolio",
    "portfolio-exact-v0": "portfolio",
    "molecule": "molecule",
    "molecule-exact-v0": "molecule",
    "vlmop1": "vlmop1",
    "vlmop2": "vlmop2",
    "vlmop3": "vlmop3",
}


def canonical_offline_moo_problem_name(problem_name):
    key = str(problem_name).strip().lower().replace("_", "-")
    return OFFLINE_MOO_PROBLEM_NAMES.get(key)


def is_offline_moo_problem(problem_name):
    return canonical_offline_moo_problem_name(problem_name) is not None


def offline_moo_root():
    configured_root = os.getenv("OFFLINE_MOO_ROOT")
    if configured_root:
        root = Path(configured_root)
    else:
        root = Path(__file__).resolve().parents[1] / "external" / "offline-moo"
    if not root.exists():
        raise FileNotFoundError(
            f"offline-moo clone not found at {root}. "
            "Clone https://github.com/lamda-bbo/offline-moo into external/offline-moo "
            "or set OFFLINE_MOO_ROOT."
        )
    return root


def ensure_offline_moo_on_path():
    root = str(offline_moo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def build_offline_moo_problem(problem_name):
    canonical_name = canonical_offline_moo_problem_name(problem_name)
    if canonical_name is None:
        raise ValueError(f"Unknown offline-moo problem: {problem_name}")
    ensure_offline_moo_on_path()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Failed to config .* module.*",
                category=UserWarning,
            )
            # Some optional offline-moo dependencies write advisory text to stderr at import time.
            with contextlib.redirect_stderr(io.StringIO()):
                from off_moo_bench.problem import get_problem
    except ImportError as err:
        raise ImportError(
            "Could not import offline-moo problem code. Install the dependencies used "
            "by lamda-bbo/offline-moo, especially numpy, pandas, torch, and pymoo."
        ) from err
    return get_problem(canonical_name)


def is_offline_moo_problem_object(problem):
    module_name = getattr(problem.__class__, "__module__", "")
    return module_name.startswith("off_moo_bench.")


def is_offline_moo_portfolio(problem):
    if not is_offline_moo_problem_object(problem):
        return False
    return getattr(problem, "name", "").lower() == "moportfolio"


def get_offline_moo_repair(problem):
    if not is_offline_moo_portfolio(problem):
        return None
    ensure_offline_moo_on_path()
    from off_moo_bench.problem.comb_opt.mo_portfolio import PortfolioRepair

    return PortfolioRepair()


def repair_offline_moo_decisions(problem, X):
    repair = get_offline_moo_repair(problem)
    if repair is None:
        return X
    X = np.asarray(X, dtype=float).copy()
    # The official repair divides by each row sum after thresholding values
    # below 1e-3.  A degenerate all-zero offspring would otherwise produce
    # NaNs; use the neutral uniform portfolio for that pathological row.
    X[~np.isfinite(X)] = 0.0
    X[X < 1e-3] = 0.0
    zero_rows = np.sum(X, axis=1) <= 0.0
    if np.any(zero_rows):
        X[zero_rows] = 1.0 / X.shape[1]
    repaired = np.asarray(repair._do(problem, X), dtype=float)
    if not np.all(np.isfinite(repaired)):
        raise ValueError("Portfolio repair produced non-finite decisions.")
    return repaired


def evaluate_offline_moo_objectives_and_feasibility(problem, X):
    """Evaluate true objectives and expose Molecule's custom feasibility flag.

    Off-MOO-Bench's Molecule task writes ``out["feasible"]`` without declaring
    a Pymoo constraint, so ``Problem.evaluate(..., ["F"])`` silently discards
    that information.  Other tasks retain their normal evaluation path.
    """

    X = np.asarray(X, dtype=float)
    is_molecule = (
        is_offline_moo_problem_object(problem)
        and getattr(problem, "name", "").strip().lower() == "molecule"
    )
    if not is_molecule:
        objectives = problem.evaluate(X, return_values_of=["F"])
        return np.asarray(objectives, dtype=float), None

    out = {}
    problem._evaluate(X, out, mode="eval")
    objectives = np.asarray(out["F"], dtype=float)
    feasible = np.asarray(out["feasible"]).reshape(-1).astype(bool)
    if len(feasible) != len(X):
        raise ValueError(
            "Molecule feasibility output length does not match candidate count."
        )
    return objectives, feasible
