"""Shared data, proxy, evaluation, and result utilities.

The generative methods intentionally reuse the repository's official-pool and
metric implementations.  The true oracle is called only after a method has
finished selecting its final candidates.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.generative_baseline import BASELINE_NAMES, PROTOCOL_VERSION


RESULT_FIELDS = (
    "problem", "method", "dataset_source", "protocol_version",
    "configuration_hash", "training_size", "offline_sample_size", "fit_size",
    "test_size", "offline_seed", "model_seed", "opt_seed",
    "subset_indices_hash", "configured_output_size", "MSEpre", "MSEsur_real",
    "HVsur", "HVreal", "IGDplus_sur", "IGDplus", "objective_min",
    "objective_max", "hv_reference_point_normalized",
    "igdplus_reference_source", "submitted_solution_count",
    "number_of_feasible_solutions", "runtime_training", "runtime_generation",
    "candidate_file", "status", "error_message",
)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values):
        values = matrix(values, "values")
        scale = np.std(values, axis=0)
        return cls(np.mean(values, axis=0), np.where(scale > 1e-12, scale, 1.0))

    def transform(self, values):
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse(self, values):
        return np.asarray(values, dtype=float) * self.scale + self.mean


def matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty 2D array; got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    return values


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("PyYAML is required to read generative baseline config.") from error
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    missing = [key for key in ("methods", "problems") if not config.get(key)]
    if missing:
        raise ValueError(f"Configuration is missing non-empty keys: {missing}.")
    unknown = sorted(set(config["methods"]) - set(BASELINE_NAMES))
    if unknown:
        raise ValueError(f"Unknown generative baselines: {unknown}.")
    if config.get("dataset_source", "official_pool") != "official_pool":
        raise ValueError("Generative baselines require dataset_source=official_pool.")
    return config


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str):
    import torch

    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {requested!r}, but CUDA is unavailable.")
    return torch.device(requested)


def load_data(problem, training_size, offline_seed, all_training_sizes, cache_dir):
    from src.offline_moo_adapter import ensure_offline_moo_on_path
    from src.official_pool import load_official_subset, official_task_name

    ensure_offline_moo_on_path()
    import off_moo_bench

    data = load_official_subset(
        cache_root=cache_dir,
        problem_name=problem,
        sample_size=int(training_size),
        offline_seed=int(offline_seed),
        all_sample_sizes=tuple(int(value) for value in all_training_sizes),
    )
    return off_moo_bench.make(official_task_name(problem)), data


def fit_shared_proxy(data, proxy_config, model_seed, device):
    """Reuse the existing MultipleModels-Vallina implementation."""

    from experiments.DL_baseline.core import fit_neural_predictor

    return fit_neural_predictor(
        "MultipleModels-Vallina",
        data,
        proxy_config,
        {},
        int(model_seed),
        device,
    )


def proxy_tensor(proxy, x, *, x_is_scaled=False, y_scaled=False):
    """Differentiable prediction through a reused DL-baseline proxy."""

    import torch

    dtype, device = x.dtype, x.device
    if x_is_scaled:
        scaled_x = x
    else:
        mean = torch.as_tensor(proxy.x_scaler.mean, dtype=dtype, device=device)
        scale = torch.as_tensor(proxy.x_scaler.scale, dtype=dtype, device=device)
        scaled_x = (x - mean) / scale
    if proxy.joint:
        prediction = proxy.models[0](scaled_x)
    else:
        prediction = torch.cat([model(scaled_x) for model in proxy.models], dim=1)
    if y_scaled:
        return prediction
    mean = torch.as_tensor(proxy.y_scaler.mean, dtype=dtype, device=device)
    scale = torch.as_tensor(proxy.y_scaler.scale, dtype=dtype, device=device)
    return prediction * scale + mean


def rank_and_crowding_indices(values, target_size):
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    values = matrix(values, "objectives")
    target_size = min(int(target_size), len(values))
    selected: list[int] = []
    for front in NonDominatedSorting().do(values):
        front = np.asarray(front, dtype=int)
        remaining = target_size - len(selected)
        if len(front) <= remaining:
            selected.extend(front.tolist())
        else:
            front_values = values[front]
            crowding = np.zeros(len(front), dtype=float)
            if len(front) <= 2:
                crowding[:] = np.inf
            else:
                for objective in range(values.shape[1]):
                    order = np.argsort(front_values[:, objective], kind="mergesort")
                    crowding[order[[0, -1]]] = np.inf
                    span = front_values[order[-1], objective] - front_values[order[0], objective]
                    if span > 0 and np.isfinite(span):
                        crowding[order[1:-1]] += (
                            front_values[order[2:], objective]
                            - front_values[order[:-2], objective]
                        ) / span
            order = np.lexsort((front, -crowding))
            selected.extend(front[order[:remaining]].tolist())
            break
        if len(selected) == target_size:
            break
    return np.asarray(selected, dtype=int)


def reference_directions(n_obj, count, seed):
    from pymoo.util.ref_dirs import get_reference_directions

    return np.asarray(
        get_reference_directions("energy", int(n_obj), int(count), seed=int(seed)),
        dtype=float,
    )


def _perpendicular_distances(points, directions):
    points = np.asarray(points, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    unit = directions / np.where(norms > 1e-12, norms, 1.0)
    projections = points @ unit.T
    closest = projections[:, :, None] * unit[None, :, :]
    return np.linalg.norm(points[:, None, :] - closest, axis=2)


def _largest_divisor_at_most(value, limit):
    value, limit = int(value), min(int(limit), int(value))
    return max(candidate for candidate in range(1, limit + 1) if value % candidate == 0)


def condition_points(
    y_scaled,
    count,
    seed,
    alpha_range=(0.1, 0.4),
    noise=0.05,
    max_base_points=32,
):
    """Official PCD reference-direction extrapolation in z-score space."""

    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    y_scaled = matrix(y_scaled, "y_scaled").astype(np.float64)
    count = int(count)
    if count < 1:
        raise ValueError("count must be positive.")
    rng = np.random.default_rng(int(seed))
    k = _largest_divisor_at_most(count, min(max_base_points, len(y_scaled)))
    directions = reference_directions(y_scaled.shape[1], k, seed)
    distances = _perpendicular_distances(y_scaled, directions)
    niches = np.argmin(distances, axis=1)
    niche_distances = distances[np.arange(len(y_scaled)), niches]
    fronts = NonDominatedSorting().do(y_scaled)

    selected = np.asarray(fronts[0], dtype=int)
    if len(selected) > k:
        selected = rng.choice(selected, size=k, replace=False)
    else:
        selected = selected.copy()

    niche_count = np.bincount(niches[selected], minlength=k)
    for front in fronts[1:]:
        if len(selected) >= k:
            break
        candidates = np.asarray(front, dtype=int)
        available = np.ones(len(candidates), dtype=bool)
        while len(selected) < k and np.any(available):
            candidate_niches = np.unique(niches[candidates[available]])
            minimum = np.min(niche_count[candidate_niches])
            least_used = candidate_niches[niche_count[candidate_niches] == minimum]
            niche = int(rng.choice(least_used))
            positions = np.where(available & (niches[candidates] == niche))[0]
            if niche_count[niche] == 0:
                position = positions[np.argmin(niche_distances[candidates[positions]])]
            else:
                position = int(rng.choice(positions))
            selected = np.append(selected, candidates[position])
            available[position] = False
            niche_count[niche] += 1

    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(len(y_scaled)), selected)
        selected = np.append(selected, rng.choice(remaining, k - len(selected), replace=False))

    point_directions = directions[niches[selected]]
    tiling_factor = count // k
    base = np.tile(y_scaled[selected], (tiling_factor, 1))
    tiled_directions = np.tile(point_directions, (tiling_factor, 1))
    alpha = rng.uniform(*alpha_range, size=(count, 1))
    targets = base - alpha * tiled_directions
    if abs(float(noise)) >= 1e-10:
        targets = rng.normal(targets, scale=float(noise))
    return targets


def evaluate_candidates(problem_name, task, data, candidates, surrogate_y=None):
    from src.metrics import get_igd_plus, get_metrics, normalize_objectives
    from src.offline_moo_adapter import (
        evaluate_offline_moo_objectives_and_feasibility,
        repair_offline_moo_decisions,
    )

    candidates = repair_offline_moo_decisions(task.problem, matrix(candidates, "candidates"))
    real_y, explicit_feasible = evaluate_offline_moo_objectives_and_feasibility(
        task.problem, candidates
    )
    real_y = np.asarray(real_y, dtype=float)
    surrogate = None if surrogate_y is None else np.asarray(surrogate_y, dtype=float)
    finite = np.all(np.isfinite(candidates), axis=1) & np.all(np.isfinite(real_y), axis=1)
    if surrogate is not None:
        if surrogate.shape != real_y.shape:
            raise ValueError("surrogate_y must match the true objective shape.")
        finite &= np.all(np.isfinite(surrogate), axis=1)
    if explicit_feasible is not None:
        finite &= np.asarray(explicit_feasible, dtype=bool)
    candidates, real_y = candidates[finite], real_y[finite]
    if surrogate is not None:
        surrogate = surrogate[finite]
    if len(candidates) == 0:
        raise RuntimeError("No finite feasible generated candidates remained.")

    reference = data["metric_reference_values"]
    hv, obj_min, obj_max, normalized_ref = get_metrics(
        problem_name, task.problem, objective_values=reference
    )
    igd, igd_source = get_igd_plus(
        task.problem, obj_min, obj_max, reference,
        fallback_reference_values=data.get("igd_reference_values"),
    )
    real_normalized = normalize_objectives(real_y, obj_min, obj_max)
    result = {
        "candidates": candidates,
        "real_y": real_y,
        "surrogate_y": surrogate,
        "HVreal": float(hv.do(real_normalized)),
        "IGDplus": float(igd.do(real_normalized)),
        "objective_min": obj_min,
        "objective_max": obj_max,
        "normalized_ref": normalized_ref,
        "igd_source": igd_source,
        "MSEsur_real": np.nan,
        "HVsur": np.nan,
        "IGDplus_sur": np.nan,
    }
    if surrogate is not None:
        surrogate_normalized = normalize_objectives(surrogate, obj_min, obj_max)
        result.update({
            "MSEsur_real": float(np.mean((surrogate - real_y) ** 2)),
            "HVsur": float(hv.do(surrogate_normalized)),
            "IGDplus_sur": float(igd.do(surrogate_normalized)),
        })
    return result


def subset_hash(data):
    indices = np.ascontiguousarray(data["offline_indices"], dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def configuration_hash(method, config):
    payload = json.dumps({"method": method, "config": config}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def base_row(problem, method, size, offline_seed, opt_seed, data, output_size, config_hash):
    return {
        "problem": problem,
        "method": method,
        "dataset_source": "official_pool",
        "protocol_version": PROTOCOL_VERSION,
        "configuration_hash": config_hash,
        "training_size": int(size),
        "offline_sample_size": int(size),
        "fit_size": int(len(data["X_train"])),
        "test_size": int(len(data["X_test"])),
        "offline_seed": int(offline_seed),
        "model_seed": int(offline_seed),
        "opt_seed": int(opt_seed),
        "subset_indices_hash": subset_hash(data),
        "configured_output_size": int(output_size),
        "status": "failed",
        "error_message": "",
    }


def result_key(row):
    return tuple(row[name] for name in (
        "protocol_version", "configuration_hash", "problem", "method",
        "training_size", "offline_seed", "opt_seed", "configured_output_size",
    ))


def read_success_keys(path: Path):
    if not Path(path).exists():
        return set()
    keys = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success":
                continue
            for name in ("training_size", "offline_seed", "opt_seed", "configured_output_size"):
                row[name] = int(row[name])
            keys.add(result_key(row))
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", newline="", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0, os.SEEK_END)
            empty = handle.tell() == 0
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
            if empty:
                writer.writeheader()
            writer.writerow({name: row.get(name, "") for name in RESULT_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def complete_row(row, evaluation, training_time, generation_time, candidate_path, mse_pre=np.nan):
    row.update({
        "MSEpre": mse_pre,
        "MSEsur_real": evaluation["MSEsur_real"],
        "HVsur": evaluation["HVsur"],
        "HVreal": evaluation["HVreal"],
        "IGDplus_sur": evaluation["IGDplus_sur"],
        "IGDplus": evaluation["IGDplus"],
        "objective_min": json.dumps(evaluation["objective_min"].tolist()),
        "objective_max": json.dumps(evaluation["objective_max"].tolist()),
        "hv_reference_point_normalized": json.dumps(evaluation["normalized_ref"].tolist()),
        "igdplus_reference_source": evaluation["igd_source"],
        "submitted_solution_count": int(row["configured_output_size"]),
        "number_of_feasible_solutions": int(len(evaluation["candidates"])),
        "runtime_training": float(training_time),
        "runtime_generation": float(generation_time),
        "candidate_file": str(candidate_path),
        "status": "success",
        "error_message": "",
    })
    return row
