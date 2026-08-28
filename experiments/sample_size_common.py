"""Shared machinery for the training-size sensitivity experiment.

The unit of work is one (problem, train size, LHS seed, method) group. A group
trains its surrogate once, evaluates all requested optimizer seeds, and then
releases the surrogate instead of persisting it to disk.
"""

from __future__ import annotations

import ast
import csv
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_PACKAGES = PROJECT_ROOT / "python_packages"
if PYTHON_PACKAGES.is_dir() and str(PYTHON_PACKAGES) not in sys.path:
    sys.path.insert(0, str(PYTHON_PACKAGES))

import numpy as np
try:
    import yaml
except ImportError:
    yaml = None

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.problem_specs import EXPERIMENT_PROBLEMS


PROBLEMS = EXPERIMENT_PROBLEMS
TRAIN_SIZES = (50, 100, 200, 400, 1000)
LHS_SEEDS = tuple(range(1, 11))
OPT_SEEDS = tuple(range(1, 11))
TEST_SIZE = 100
LHS_PROTOCOL_VERSION = "lhs_full_offline_fixed_quantile_v5"
OFFICIAL_PROTOCOL_VERSION = "official_pool_full_offline_fixed_quantile_v9"


def current_protocol_version(dataset_source):
    return (
        OFFICIAL_PROTOCOL_VERSION
        if str(dataset_source).strip().lower() == "official_pool"
        else LHS_PROTOCOL_VERSION
    )


def result_protocol_version(row):
    value = str(row.get("protocol_version") or "").strip()
    if value:
        return value
    source = str(row.get("dataset_source") or "lhs").strip().lower()
    # Unversioned results predate the current fixed-quantile uncertainty protocol
    # and must not be resumed under it.
    return f"{source}_unversioned"


def result_optimizer_settings(row):
    """Read versioned optimizer settings, defaulting legacy rows to 100/100."""

    return (
        int(row.get("configured_n_gen") or 100),
        int(row.get("configured_pop_size") or 100),
    )

def _scalar(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def load_config_file(path):
    if yaml is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    root, stack = {}, [(-1, {})]
    stack[0] = (-1, root)
    for index, raw in enumerate(lines):
        clean = raw.split("#", 1)[0].rstrip()
        if not clean.strip():
            continue
        indent, text = len(clean) - len(clean.lstrip()), clean.strip()
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if text.startswith("- "):
            parent.append(_scalar(text[2:].strip()))
            continue
        key, value = (part.strip() for part in text.split(":", 1))
        if value:
            parent[key] = _scalar(value)
            continue
        next_text = ""
        for following in lines[index + 1:]:
            following = following.split("#", 1)[0].strip()
            if following:
                next_text = following
                break
        child = [] if next_text.startswith("- ") else {}
        parent[key] = child
        stack.append((indent, child))
    return root


_root_config = load_config_file(EXPERIMENTS_DIR / "config.yaml")
_ablation_config = _root_config.get("sample_size_ablation", {})
PROBLEMS = tuple(_root_config.get("problem_names", PROBLEMS))
TRAIN_SIZES = tuple(int(value) for value in _ablation_config.get("train_sizes", TRAIN_SIZES))
LHS_SEEDS = tuple(int(value) for value in _ablation_config.get("lhs_seeds", LHS_SEEDS))
OPT_SEEDS = tuple(int(value) for value in _ablation_config.get("opt_seeds", OPT_SEEDS))
TEST_SIZE = int(_ablation_config.get("test_size", TEST_SIZE))
DUAL_RANKING_QUANTILE = float(_root_config.get("dual_ranking_quantile", 0.90))


@dataclass(frozen=True)
class MethodSpec:
    name: str
    family: str
    dual_ranking: bool = False


# This is the single source of truth used by the CLI and summary tools.  Names
# follow the existing scripts/results; no baseline implementation is duplicated.
METHOD_REGISTRY = {
    spec.name: spec for spec in (
        MethodSpec("GPR-RBF + NSGA-II", "gpr_rbf"),
        MethodSpec("GPR-RBF + NSGA-II + DR", "gpr_rbf", True),
        MethodSpec("GPR-Matern + NSGA-II", "gpr_matern"),
        MethodSpec("GPR-Matern + NSGA-II + DR", "gpr_matern", True),
        MethodSpec("QR + NSGA-II", "qr"),
        MethodSpec("QR + NSGA-II + DR", "qr", True),
        MethodSpec("BNN + NSGA-II", "bnn"),
        MethodSpec("BNN + NSGA-II + DR", "bnn", True),
        MethodSpec("XGBoost + NSGA-II", "xgboost"),
        MethodSpec("WeightedEnsemble L2 + NSGA-II", "ensemble"),
        MethodSpec("TGPR-MO", "tgpr_mo"),
        MethodSpec("DDMOEA-GAN", "ddmoea_gan"),
        MethodSpec("Prob-RVEA", "prob_rvea"),
        MethodSpec("Prob-MOEA/D", "prob_moead"),
        MethodSpec("TabPFN + NSGA-II", "tabpfn"),
    )
}
BASELINE_FAMILIES = {"prob_rvea", "prob_moead", "tgpr_mo", "ddmoea_gan"}

RESULT_FIELDS = (
    "problem", "method", "dataset_source", "protocol_version",
    "configured_n_gen", "configured_pop_size", "training_size",
    "offline_sample_size", "fit_size", "test_size", "offline_seed", "lhs_seed",
    "model_seed", "opt_seed", "subset_indices_hash", "run_id", "MSEpre",
    "MSEsur_real", "HVreal", "IGDplus", "IGDplus_sur",
    "surrogate_normalization_source", "objective_normalization_source",
    "objective_min", "objective_max", "hv_reference_point_normalized",
    "igdplus_reference_source",
    "final_output_target", "submitted_solution_count",
    "number_of_feasible_solutions",
    "optimizer_generation_count", "surrogate_evaluation_count",
    "runtime_surrogate_training", "runtime_optimization", "status",
    "error_message",
)


def set_global_seed(seed: int) -> None:
    """Seed every installed RNG used by the experiment, without requiring it."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    try:
        import pyro
        pyro.set_rng_seed(seed)
    except ImportError:
        pass


def _stable_problem_seed(problem: str, offset: int) -> int:
    digest = hashlib.sha256(problem.encode("utf-8")).digest()
    return offset + int.from_bytes(digest[:4], "little") % 100_000


def _rows_overlap(a: np.ndarray, b: np.ndarray) -> bool:
    if not len(a) or not len(b):
        return False
    return bool({row.tobytes() for row in np.ascontiguousarray(a)} &
                {row.tobytes() for row in np.ascontiguousarray(b)})


def dataset_path(output_dir: Path, problem: str, training_size: int, lhs_seed: int) -> Path:
    safe_problem = problem.upper().replace("-", "_")
    filename = f"{safe_problem}_train_test_N{training_size}_lhs{lhs_seed}.npz"
    output_dir = Path(output_dir)
    return output_dir / "npz" / filename


def load_or_create_dataset(
    output_dir: Path,
    problem_name: str,
    training_size: int,
    lhs_seed: int,
    dataset_source="lhs",
    subset_cache_root=None,
    all_sample_sizes=None,
) -> dict[str, np.ndarray]:
    """Load one shared dataset without changing the legacy LHS default."""

    dataset_source = str(dataset_source).strip().lower()
    if dataset_source == "official_pool":
        from src.official_pool import load_official_subset

        if subset_cache_root is None:
            subset_cache_root = Path(output_dir).parent / "data_subsets"
        return load_official_subset(
            cache_root=subset_cache_root,
            problem_name=problem_name,
            sample_size=training_size,
            offline_seed=lhs_seed,
            all_sample_sizes=all_sample_sizes or (training_size,),
        )
    if dataset_source != "lhs":
        raise ValueError(
            "dataset_source must be either 'lhs' or 'official_pool'."
        )

    # Legacy LHS mode below is intentionally unchanged.
    from pymoo.operators.sampling.lhs import LHS
    from src.offline_moo_adapter import repair_offline_moo_decisions
    from src.opt_problem import build_problem

    path = dataset_path(output_dir, problem_name, training_size, lhs_seed)
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            arrays = {key: saved[key] for key in saved.files}
        arrays.setdefault("offline_seed", np.asarray(lhs_seed))
        arrays.setdefault("model_seed", np.asarray(lhs_seed))
        arrays.setdefault("offline_sample_size", np.asarray(training_size))
        arrays.setdefault("fit_size", np.asarray(len(arrays["X_train"])))
        arrays.setdefault("test_size", np.asarray(len(arrays["X_test"])))
        arrays.setdefault("dataset_source", np.asarray("lhs"))
        return arrays

    problem = build_problem(problem_name=problem_name)
    sampler = LHS()
    set_global_seed(lhs_seed)
    x_train = sampler(problem, int(training_size), seed=int(lhs_seed)).get("X")
    test_seed = _stable_problem_seed(problem_name, 400_000)
    x_test = sampler(problem, TEST_SIZE, seed=test_seed).get("X")
    x_train = np.asarray(repair_offline_moo_decisions(problem, x_train), dtype=float)
    x_test = np.asarray(repair_offline_moo_decisions(problem, x_test), dtype=float)
    if _rows_overlap(x_train, x_test):
        raise RuntimeError("Generated train/test datasets overlap")
    arrays = {
        "X_train": x_train,
        "y_train": np.asarray(problem.evaluate(x_train, return_values_of=["F"]), dtype=float),
        "X_test": x_test,
        "y_test": np.asarray(problem.evaluate(x_test, return_values_of=["F"]), dtype=float),
        "lhs_seed": np.asarray(lhs_seed),
        "test_seed": np.asarray(test_seed),
        "offline_seed": np.asarray(lhs_seed),
        "model_seed": np.asarray(lhs_seed),
        "offline_sample_size": np.asarray(training_size),
        "fit_size": np.asarray(len(x_train)),
        "test_size": np.asarray(len(x_test)),
        "dataset_source": np.asarray("lhs"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    return arrays


def _predictor_models(spec: MethodSpec, data: dict[str, np.ndarray], model_seed: int):
    from src import models

    x, y = data["X_train"], data["y_train"]
    n_obj = y.shape[1]
    if spec.family == "gpr_rbf":
        pair = tuple(models.GPR_RBF() for _ in range(n_obj))
        for objective_index, model in enumerate(pair):
            model.fit(x, y[:, objective_index])
        return pair, "GPR_uncertainty"
    if spec.family == "gpr_matern":
        pair = tuple(models.GPR_Matern() for _ in range(n_obj))
        for objective_index, model in enumerate(pair):
            model.fit(x, y[:, objective_index])
        return pair, "GPR_uncertainty"
    if spec.family == "qr":
        pair = tuple(models.autogluon_qr_fit_predict(
            x, y[:, i], data["X_test"], random_state=model_seed)[1] for i in range(n_obj))
        return pair, "QR_uncertainty"
    if spec.family == "bnn":
        pair = tuple(models.BNNRegressor(random_state=model_seed) for _ in range(n_obj))
        for objective_index, model in enumerate(pair):
            model.fit(x, y[:, objective_index])
        return pair, "BNN_uncertainty"
    if spec.family in {"xgboost", "ensemble"}:
        hyperparameters = {"XGB": {}} if spec.family == "xgboost" else None
        fit_kwargs = {"fit_weighted_ensemble": False} if spec.family == "xgboost" else {}
        pair = tuple(models.autogluon_fit_predict(
            x, y[:, i], data["X_test"], hyperparameters=hyperparameters,
            fit_kwargs=fit_kwargs, random_state=model_seed)[1] for i in range(n_obj))
        return pair, "Autogluon"
    if spec.family == "tabpfn":
        pair = tuple(
            models.tabpfn_fit_predict(
                x,
                y[:, i],
                data["X_test"],
                random_state=model_seed,
            )[1]
            for i in range(n_obj)
        )
        return pair, "TabPFN"
    raise ValueError(f"Not a predictor surrogate: {spec.family}")


def _train_predictor(lhs_seed: int, spec: MethodSpec, data):
    """Train one method-local surrogate; no model is persisted to disk."""
    model_seed = int(lhs_seed)
    start = time.perf_counter()
    set_global_seed(model_seed)
    models_pair, use_surrogate = _predictor_models(spec, data, model_seed)
    elapsed = time.perf_counter() - start
    return models_pair, use_surrogate, elapsed, model_seed


def _survival(spec, models_pair, data):
    from src.survival import Survival_dual_ranking, Survival_standard
    from src.uncertainty import gaussian_upper_scale

    if not spec.dual_ranking:
        return Survival_standard()

    if spec.family in {"gpr_rbf", "gpr_matern"}:
        return Survival_dual_ranking(
            alphas=[gaussian_upper_scale(DUAL_RANKING_QUANTILE)] * len(models_pair),
        )

    quantile = float(DUAL_RANKING_QUANTILE)
    if quantile not in {0.8, 0.9, 0.95}:
        raise ValueError("dual_ranking_quantile must be one of 0.8, 0.9, 0.95.")
    if spec.family not in {"qr", "bnn"}:
        raise ValueError(f"Unsupported dual-ranking family: {spec.family}")
    return Survival_dual_ranking(alpha=quantile)


def optimization_initial_population(x_train, population_size, optimization_seed):
    """Select a deterministic initial population controlled only by opt seed."""

    x_train = np.asarray(x_train, dtype=float)
    population_size = int(population_size)
    if x_train.ndim != 2 or len(x_train) == 0:
        raise ValueError("x_train must be a non-empty 2D array.")
    rng = np.random.default_rng(int(optimization_seed))
    if len(x_train) >= population_size:
        indices = rng.permutation(len(x_train))[:population_size]
    else:
        prefix = rng.permutation(len(x_train))
        indices = np.resize(prefix, population_size)
    return x_train[indices].copy()


def _dataset_scalar(data, name, default):
    value = data.get(name, default)
    return np.asarray(value).reshape(-1)[0]


def _subset_indices_hash(data):
    indices = data.get("offline_indices")
    if indices is None:
        return ""
    indices = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def _base_row(
    problem,
    method,
    training_size,
    lhs_seed,
    model_seed,
    opt_seed,
    training_time,
    data=None,
    dataset_source="lhs",
    n_gen=100,
    pop_size=100,
):
    data = {} if data is None else data
    dataset_source = str(_dataset_scalar(data, "dataset_source", dataset_source))
    protocol_version = current_protocol_version(dataset_source)
    test_size = int(_dataset_scalar(data, "test_size", TEST_SIZE))
    fit_size = int(_dataset_scalar(data, "fit_size", training_size))
    offline_sample_size = int(
        _dataset_scalar(data, "offline_sample_size", training_size)
    )
    return {
        "problem": problem, "method": method, "dataset_source": dataset_source,
        "protocol_version": protocol_version,
        "configured_n_gen": int(n_gen),
        "configured_pop_size": int(pop_size),
        "training_size": int(training_size),
        "offline_sample_size": offline_sample_size,
        "fit_size": fit_size,
        "test_size": test_size,
        "offline_seed": int(lhs_seed), "lhs_seed": int(lhs_seed),
        "model_seed": int(model_seed),
        "opt_seed": int(opt_seed),
        "subset_indices_hash": _subset_indices_hash(data),
        "run_id": (
            f"{dataset_source}|{protocol_version}|G{int(n_gen)}|P{int(pop_size)}|"
            f"{problem}|{method}|N{training_size}|"
            f"offline{lhs_seed}|opt{opt_seed}"
        ),
        "MSEpre": np.nan, "MSEsur_real": np.nan, "HVreal": np.nan,
        "IGDplus": np.nan, "IGDplus_sur": np.nan,
        "surrogate_normalization_source": (
            "selected_offline_subset_N_only"
            if dataset_source == "official_pool"
            else "model_fit_subset_only"
        ),
        "objective_normalization_source": (
            "official_training_pool_evaluation_only"
            if dataset_source == "official_pool"
            else "current_surrogate_fit_subset"
        ),
        "objective_min": "", "objective_max": "",
        "hv_reference_point_normalized": "",
        "igdplus_reference_source": "",
        "final_output_target": int(pop_size) if dataset_source == "official_pool" else np.nan,
        "submitted_solution_count": 0,
        "number_of_feasible_solutions": 0,
        "optimizer_generation_count": np.nan,
        "surrogate_evaluation_count": np.nan,
        "runtime_surrogate_training": float(training_time),
        "runtime_optimization": np.nan, "status": "failed", "error_message": "",
    }


def _run_predictor_group(
    output_dir,
    problem_name,
    training_size,
    lhs_seed,
    spec,
    opt_seeds,
    n_gen,
    pop_size,
    dataset_source="lhs",
    subset_cache_root=None,
    all_sample_sizes=None,
):
    from src.experiment import compute_surrogate_test_mse, run_experiment
    from src.metrics import get_igd_plus, get_metrics
    from src.opt_problem import build_problem

    data = load_or_create_dataset(
        output_dir,
        problem_name,
        training_size,
        lhs_seed,
        dataset_source=dataset_source,
        subset_cache_root=subset_cache_root,
        all_sample_sizes=all_sample_sizes,
    )
    problem = build_problem(problem_name=problem_name)
    pair, use_surrogate, training_time, model_seed = _train_predictor(
        lhs_seed, spec, data)
    mse_pre = float(compute_surrogate_test_mse(
        problem, problem_name, pair[0], pair[1], use_surrogate,
        data["X_test"], data["y_test"], models=pair))
    objective_values = data.get("metric_reference_values", data["y_train"])
    hv, obj_min, obj_max, normalized_ref_point = get_metrics(
        problem_name=problem_name, problem=problem, n_var=problem.n_var,
        n_obj=problem.n_obj, objective_values=objective_values)
    igd_plus, igd_plus_source = get_igd_plus(
        problem,
        obj_min,
        obj_max,
        objective_values,
        fallback_reference_values=data.get("igd_reference_values"),
    )
    survival = _survival(spec, pair, data)
    rows = []
    # The trained surrogate is shared; optimizer failures are recorded per seed.
    for opt_seed in opt_seeds:
        row = _base_row(
            problem_name,
            spec.name,
            training_size,
            lhs_seed,
            model_seed,
            opt_seed,
            training_time,
            data=data,
            dataset_source=dataset_source,
            n_gen=n_gen,
            pop_size=pop_size,
        )
        started = time.perf_counter()
        try:
            set_global_seed(
                int(opt_seed)
                if dataset_source == "official_pool"
                else int(lhs_seed) * 1000 + int(opt_seed)
            )
            initial_population = (
                optimization_initial_population(
                    data.get("X_offline", data["X_train"]), pop_size, opt_seed
                )
                if dataset_source == "official_pool"
                else None
            )
            results = run_experiment(
                problem=problem, problem_name=problem_name, n_gen=n_gen, pop_size=pop_size,
                use_surrogate=use_surrogate, model_f1=pair[0], model_f2=pair[1],
                survival_function=survival, obj_min=obj_min, obj_max=obj_max, hv=hv,
                use_callback=False, seeds=[int(opt_seed)], optimizer_name="NSGA-II",
                print_normalization_info=False, mse_test=mse_pre,
                plot_seed_objectives=False, models=pair,
                initial_population=initial_population,
                final_output_size=(
                    int(pop_size) if dataset_source == "official_pool" else None
                ),
                igd_plus_indicator=igd_plus, igd_plus_source=igd_plus_source)
            detail = results["run_details"][0]
            no_feasible = bool(detail.get("no_feasible_solution"))
            required_metrics = (
                mse_pre,
                detail["mse_sur_real"],
                detail["hv_real"],
                detail["igd_plus_real"],
            )
            metrics_finite = all(np.isfinite(float(value)) for value in required_metrics)
            failed_reason = (
                str(detail.get("no_feasible_reason") or "no feasible final solution")
                if no_feasible
                else "non-finite required evaluation metric"
            )
            row.update({
                "MSEpre": mse_pre,
                "MSEsur_real": float(detail["mse_sur_real"]),
                "HVreal": float(detail["hv_real"]),
                "IGDplus": float(detail["igd_plus_real"]),
                "IGDplus_sur": float(detail["igd_plus_surrogate"]),
                "objective_min": json.dumps(np.asarray(obj_min).tolist()),
                "objective_max": json.dumps(np.asarray(obj_max).tolist()),
                "hv_reference_point_normalized": json.dumps(
                    np.asarray(normalized_ref_point).tolist()
                ),
                "igdplus_reference_source": igd_plus_source,
                "final_output_target": detail.get("final_output_target", np.nan),
                "submitted_solution_count": int(
                    detail.get("submitted_solution_count", 0)
                ),
                "number_of_feasible_solutions": int(detail["solution_count"]),
                "optimizer_generation_count": int(
                    detail.get("optimizer_generation_count", n_gen)
                ),
                "surrogate_evaluation_count": int(
                    detail.get("surrogate_evaluation_count", n_gen * pop_size)
                ),
                "runtime_optimization": float(detail["time"]),
                "status": "success" if metrics_finite and not no_feasible else "failed",
                "error_message": "" if metrics_finite and not no_feasible else failed_reason,
            })
        except Exception as error:
            row["MSEpre"] = mse_pre
            row["runtime_optimization"] = time.perf_counter() - started
            row["error_message"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        rows.append(row)
    return rows, pair


def _run_baseline_group(
    output_dir,
    problem_name,
    training_size,
    lhs_seed,
    spec,
    opt_seeds,
    n_gen=100,
    pop_size=100,
    dataset_source="lhs",
    subset_cache_root=None,
    all_sample_sizes=None,
):
    # Reuse the exact uploaded baseline runners, replacing only their data hooks
    # so every method sees the shared paired dataset.
    baseline_dir = REPO_ROOT / "experiments" / "baseline"
    if str(baseline_dir) not in sys.path:
        sys.path.insert(0, str(baseline_dir))
    import batch_experiments as batch
    from src.metrics import get_igd_plus, get_metrics
    from src.opt_problem import build_problem

    data = load_or_create_dataset(
        output_dir,
        problem_name,
        training_size,
        lhs_seed,
        dataset_source=dataset_source,
        subset_cache_root=subset_cache_root,
        all_sample_sizes=all_sample_sizes,
    )
    problem = build_problem(problem_name=problem_name)
    _, metric_obj_min, metric_obj_max, metric_ref_point = get_metrics(
        problem_name=problem_name,
        problem=problem,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        objective_values=data.get("metric_reference_values", data["y_train"]),
    )
    _, metric_igd_source = get_igd_plus(
        problem,
        metric_obj_min,
        metric_obj_max,
        data.get("metric_reference_values", data["y_train"]),
        fallback_reference_values=data.get("igd_reference_values"),
    )
    metric_row = {
        "objective_min": json.dumps(np.asarray(metric_obj_min).tolist()),
        "objective_max": json.dumps(np.asarray(metric_obj_max).tolist()),
        "hv_reference_point_normalized": json.dumps(
            np.asarray(metric_ref_point).tolist()
        ),
        "igdplus_reference_source": metric_igd_source,
    }
    config = {
        "sample_size": int(training_size), "train_seed": int(lhs_seed),
        "model_seed": int(_dataset_scalar(data, "model_seed", lhs_seed)),
        "test_seed": int(np.asarray(data.get("test_seed", 0))),
        "dataset_source": dataset_source,
        "n_gen": int(n_gen),
        "pop_size": int(pop_size),
        "metric_reference_values": data.get("metric_reference_values"),
        "igd_reference_values": data.get("igd_reference_values"),
    }
    runner = {
        "prob_rvea": batch._run_prob_rvea_problem,
        "prob_moead": batch._run_prob_moead_problem,
        "tgpr_mo": batch._run_tgpr_mo_problem,
        "ddmoea_gan": batch._run_ddmoea_gan_problem,
    }[spec.family]
    original_train, original_test = batch.generate_offline_dataset, batch.generate_offline_test_dataset
    original_initial = batch.initial_population_from_offline_dataset
    def paired_initial_population(
        x_data,
        population_size=batch.POPULATION_SIZE,
        seed=None,
    ):
        if dataset_source == "official_pool":
            if seed is None:
                raise ValueError(
                    "official_pool initial populations require optimization seed."
                )
            return optimization_initial_population(
                data.get("X_offline", x_data), population_size, seed
            )
        x_data = np.asarray(x_data, dtype=float)
        population_size = int(population_size)
        if len(x_data) >= population_size:
            return x_data[:population_size].copy()
        repeats = int(np.ceil(population_size / len(x_data)))
        return np.tile(x_data, (repeats, 1))[:population_size].copy()
    batch.generate_offline_dataset = lambda *args, **kwargs: (data["X_train"], data["y_train"])
    batch.generate_offline_test_dataset = lambda *args, **kwargs: (data["X_test"], data["y_test"])
    batch.initial_population_from_offline_dataset = paired_initial_population
    set_global_seed(lhs_seed)
    started = time.perf_counter()
    try:
        details = runner(problem_name, problem, config, list(opt_seeds))
    finally:
        batch.generate_offline_dataset, batch.generate_offline_test_dataset = original_train, original_test
        batch.initial_population_from_offline_dataset = original_initial
    total = time.perf_counter() - started
    training_time = max(0.0, total - sum(float(item.get("time", 0.0)) for item in details))
    rows = []
    for detail in details:
        opt_seed = int(detail["seed"])
        row = _base_row(
            problem_name,
            spec.name,
            training_size,
            lhs_seed,
            lhs_seed,
            opt_seed,
            training_time,
            data=data,
            dataset_source=dataset_source,
            n_gen=n_gen,
            pop_size=pop_size,
        )
        counter_values = {
            "optimizer_generation_count": detail.get(
                "optimizer_generation_count", np.nan
            ),
            "surrogate_evaluation_count": detail.get(
                "surrogate_evaluation_count", np.nan
            ),
            "final_output_target": detail.get("final_output_target", np.nan),
            "submitted_solution_count": detail.get(
                "submitted_solution_count", 0
            ),
        }
        if detail.get("error_message"):
            row.update({
                "MSEpre": float(detail["offline_test_mse"]),
                "runtime_optimization": float(detail["time"]),
                "error_message": str(detail["error_message"]),
                **counter_values,
                **metric_row,
            })
        elif detail.get("no_feasible_solution"):
            row.update({
                "MSEpre": float(detail.get("offline_test_mse", np.nan)),
                "runtime_optimization": float(detail["time"]),
                "error_message": str(
                    detail.get("no_feasible_reason")
                    or "no feasible final solution"
                ),
                **counter_values,
                **metric_row,
            })
        else:
            required_metrics = (
                detail.get("offline_test_mse"),
                detail.get("mse_sur_real"),
                detail.get("hv_real"),
                detail.get("igd_plus_real"),
            )
            metrics_finite = all(
                value is not None and np.isfinite(float(value))
                for value in required_metrics
            )
            row.update({
                "MSEpre": float(detail["offline_test_mse"]),
                "MSEsur_real": float(detail["mse_sur_real"]),
                "HVreal": float(detail["hv_real"]),
                "IGDplus": float(detail.get("igd_plus_real", np.nan)),
                "IGDplus_sur": float(detail.get("igd_plus_surrogate", np.nan)),
                "number_of_feasible_solutions": int(detail["solution_count"]),
                "runtime_optimization": float(detail["time"]),
                "status": "success" if metrics_finite else "failed",
                "error_message": (
                    "" if metrics_finite else "non-finite required evaluation metric"
                ),
                **counter_values,
                **metric_row,
            })
        rows.append(row)
    return rows


def run_group(
    output_dir: Path,
    problem: str,
    training_size: int,
    lhs_seed: int,
    method: str,
    opt_seeds: Iterable[int],
    n_gen=100,
    pop_size=100,
    dataset_source="lhs",
    subset_cache_root=None,
    all_sample_sizes=None,
):
    """Run one cache group and stop immediately on the first error."""
    output_dir = Path(output_dir)
    spec = METHOD_REGISTRY[method]
    opt_seeds = tuple(int(seed) for seed in opt_seeds)
    try:
        if spec.family in BASELINE_FAMILIES:
            rows = _run_baseline_group(
                output_dir,
                problem,
                training_size,
                lhs_seed,
                spec,
                opt_seeds,
                n_gen=int(n_gen),
                pop_size=int(pop_size),
                dataset_source=dataset_source,
                subset_cache_root=subset_cache_root,
                all_sample_sizes=all_sample_sizes,
            )
            return rows, ()
        return _run_predictor_group(
            output_dir,
            problem,
            training_size,
            lhs_seed,
            spec,
            opt_seeds,
            int(n_gen),
            int(pop_size),
            dataset_source=dataset_source,
            subset_cache_root=subset_cache_root,
            all_sample_sizes=all_sample_sizes,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        rows = [
            dict(
                _base_row(
                    problem,
                    method,
                    training_size,
                    lhs_seed,
                    lhs_seed,
                    seed,
                    0.0,
                    dataset_source=dataset_source,
                    n_gen=n_gen,
                    pop_size=pop_size,
                ),
                error_message=message,
            )
            for seed in opt_seeds
        ]
        return rows, ()


def cleanup_model_storage(model_objects) -> None:
    """Delete temporary model directories after their result rows are durable."""
    for model in model_objects:
        finalizer = getattr(model, "_experiment_cleanup_finalizer", None)
        if finalizer is not None and finalizer.alive:
            finalizer()


def valid_success(row: dict) -> bool:
    if row.get("status") != "success":
        return False
    try:
        return all(np.isfinite(float(row[name])) for name in
                   ("MSEpre", "MSEsur_real", "HVreal", "IGDplus"))
    except (KeyError, TypeError, ValueError):
        return False


def read_result_rows(output_dir: Path) -> list[dict]:
    rows = []
    output_dir = Path(output_dir)
    paths = sorted(output_dir.glob("exp*_results.csv"))
    paths += sorted((output_dir / "csv").glob("exp*_results.csv"))
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _result_key(row: dict) -> tuple:
    configured_n_gen, configured_pop_size = result_optimizer_settings(row)
    return (
        str(row.get("dataset_source") or "lhs"),
        result_protocol_version(row),
        configured_n_gen,
        configured_pop_size,
        str(row["problem"]),
        str(row["method"]),
        int(row["training_size"]),
        int(row["lhs_seed"]),
        int(row["opt_seed"]),
    )


def append_rows(output_dir: Path, rows: Iterable[dict]) -> int:
    """Append missing completed rows, safely and without duplicate successes."""
    output_dir = Path(output_dir)
    raw_dir = output_dir / "csv"
    raw_dir.mkdir(parents=True, exist_ok=True)
    problem_index = {name: i + 1 for i, name in enumerate(PROBLEMS)}
    rows_by_path = {}
    for row in rows:
        path = raw_dir / f"exp{problem_index[row['problem']]}_results.csv"
        rows_by_path.setdefault(path, []).append(row)
    written = 0
    for path, pending_rows in rows_by_path.items():
        with path.open("a+", newline="", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.seek(0)
                reader = csv.DictReader(handle)
                existing_rows = list(reader)
                existing_fields = tuple(reader.fieldnames or ())
                if existing_fields and existing_fields != RESULT_FIELDS:
                    # Upgrade legacy result files before appending the wider
                    # official-pool schema. Existing values are preserved and
                    # newly introduced columns are left blank.
                    handle.seek(0)
                    handle.truncate()
                    upgrade_writer = csv.DictWriter(
                        handle,
                        fieldnames=RESULT_FIELDS,
                        extrasaction="ignore",
                    )
                    upgrade_writer.writeheader()
                    for existing in existing_rows:
                        upgrade_writer.writerow(
                            {name: existing.get(name, "") for name in RESULT_FIELDS}
                        )
                    handle.flush()
                    os.fsync(handle.fileno())
                successful_keys = {
                    _result_key(row) for row in existing_rows if valid_success(row)
                }
                failed_fingerprints = {
                    (_result_key(row), row.get("status", ""), row.get("error_message", ""))
                    for row in existing_rows if not valid_success(row)
                }
                handle.seek(0, os.SEEK_END)
                is_empty = handle.tell() == 0
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
                if is_empty:
                    writer.writeheader()
                for row in pending_rows:
                    key = _result_key(row)
                    if key in successful_keys:
                        continue
                    fingerprint = (key, row.get("status", ""), row.get("error_message", ""))
                    if not valid_success(row) and fingerprint in failed_fingerprints:
                        continue
                    writer.writerow({name: row.get(name, "") for name in RESULT_FIELDS})
                    written += 1
                    if valid_success(row):
                        successful_keys.add(key)
                    else:
                        failed_fingerprints.add(fingerprint)
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return written


def reconcile_result_csvs(output_dir: Path) -> int:
    """Copy rows missing from legacy root CSV files into ``results/csv``."""
    output_dir = Path(output_dir)
    legacy_rows = []
    for path in sorted(output_dir.glob("exp*_results.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            legacy_rows.extend(csv.DictReader(handle))
    return append_rows(output_dir, legacy_rows) if legacy_rows else 0


def organize_cache_files(output_dir: Path) -> None:
    """Move NPZ data into place and remove obsolete persistent model caches."""
    output_dir = Path(output_dir)
    npz_directory = output_dir / "npz"
    npz_directory.mkdir(parents=True, exist_ok=True)
    for source in output_dir.glob("*_train_N*_lhs*.npz"):
        target = npz_directory / source.name
        if not target.exists():
            try:
                os.replace(source, target)
            except FileNotFoundError:
                pass

    for directory in (output_dir, output_dir / "pkl"):
        if directory.exists():
            for source in directory.glob("surrogate_*.pkl*"):
                source.unlink(missing_ok=True)
    pkl_directory = output_dir / "pkl"
    if pkl_directory.is_dir():
        try:
            pkl_directory.rmdir()
        except OSError:
            pass
    shutil.rmtree(EXPERIMENTS_DIR / "AutogluonModels", ignore_errors=True)


def write_manifest(output_dir: Path, payload: dict) -> None:
    path = Path(output_dir) / "experiment_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
