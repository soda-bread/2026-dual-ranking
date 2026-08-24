"""Deterministic small-data subsets from the Off-MOO-Bench training pool."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from src.offline_moo_adapter import ensure_offline_moo_on_path
from src.problem_specs import canonical_problem_name


OFFICIAL_TASK_NAMES = {
    **{f"re{number}": f"RE{number}-Exact-v0"
       for number in (*range(21, 26), *range(31, 38))},
    **{f"zdt{number}": f"ZDT{number}-Exact-v0" for number in (1, 2, 3, 4, 6)},
    **{f"dtlz{number}": f"DTLZ{number}-Exact-v0" for number in range(1, 8)},
    **{f"vlmop{number}": f"VLMOP{number}-Exact-v0" for number in range(1, 4)},
    "omnitest": "OmniTest-Exact-v0",
    "mo-portfolio": "Portfolio-Exact-v0",
    "molecule": "Molecule-Exact-v0",
}


def official_task_name(problem_name):
    canonical = canonical_problem_name(problem_name)
    try:
        return OFFICIAL_TASK_NAMES[canonical]
    except KeyError as error:
        raise ValueError(
            f"No Off-MOO-Bench training-pool task is configured for '{problem_name}'."
        ) from error


def load_official_pool(problem_name):
    """Load untouched official train/test pools through ``off_moo_bench.make``."""

    ensure_offline_moo_on_path()
    import off_moo_bench

    task_name = official_task_name(problem_name)
    task = off_moo_bench.make(task_name)
    arrays = {
        "X_pool": np.asarray(task.x).copy(),
        "Y_pool": np.asarray(task.y).copy(),
        "X_test": np.asarray(task.x_test).copy(),
        "Y_test": np.asarray(task.y_test).copy(),
    }
    for name, values in arrays.items():
        if values.ndim != 2:
            raise ValueError(
                f"Official {task_name} {name} must be 2D; got {values.shape}."
            )
        if len(values) == 0:
            raise ValueError(f"Official {task_name} {name} is empty.")
    if len(arrays["X_pool"]) != len(arrays["Y_pool"]):
        raise ValueError(f"Official {task_name} training X/Y lengths do not match.")
    if len(arrays["X_test"]) != len(arrays["Y_test"]):
        raise ValueError(f"Official {task_name} test X/Y lengths do not match.")
    return task, arrays


def _permutation_hash(permutation):
    permutation = np.ascontiguousarray(permutation, dtype=np.int64)
    return hashlib.sha256(permutation.tobytes()).hexdigest()


def _atomic_save_npy(path, values):
    temporary = path.with_name(f"{path.stem}.tmp.{os.getpid()}.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_write_json(path, payload):
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def cache_official_subset_indices(
    cache_root,
    problem_name,
    offline_seed,
    sample_sizes,
    x_pool_shape,
    y_pool_shape,
):
    """Create one permutation and every nested prefix under an exclusive lock."""

    canonical = canonical_problem_name(problem_name)
    sample_sizes = tuple(sorted({int(size) for size in sample_sizes}))
    if not sample_sizes or sample_sizes[0] < 1:
        raise ValueError("sample_sizes must contain positive integers.")
    pool_size = int(x_pool_shape[0])
    if int(y_pool_shape[0]) != pool_size:
        raise ValueError("Official training X/Y pool lengths do not match.")
    if sample_sizes[-1] > pool_size:
        raise ValueError(
            f"Requested N={sample_sizes[-1]} for '{canonical}', but the official "
            f"training pool contains only {pool_size} rows."
        )

    directory = Path(cache_root) / canonical / f"offline_seed_{int(offline_seed)}"
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".subset.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            permutation_path = directory / "permutation.npy"
            if permutation_path.exists():
                permutation = np.load(permutation_path, allow_pickle=False)
            else:
                rng = np.random.default_rng(int(offline_seed))
                permutation = rng.permutation(pool_size)
                _atomic_save_npy(permutation_path, permutation)

            permutation = np.asarray(permutation, dtype=np.int64)
            if permutation.shape != (pool_size,):
                raise ValueError(
                    f"Cached permutation shape {permutation.shape} does not match "
                    f"official pool size {pool_size} for '{canonical}'."
                )
            if not np.array_equal(np.sort(permutation), np.arange(pool_size)):
                raise ValueError(f"Cached permutation for '{canonical}' is invalid.")

            for size in sample_sizes:
                expected = permutation[:size]
                indices_path = directory / f"indices_N{size}.npy"
                if indices_path.exists():
                    cached = np.load(indices_path, allow_pickle=False)
                    if not np.array_equal(cached, expected):
                        raise ValueError(
                            f"Cached {indices_path.name} is not the prefix of "
                            "permutation.npy."
                        )
                else:
                    _atomic_save_npy(indices_path, expected)

            metadata = {
                "problem": canonical,
                "dataset_source": "official_pool",
                "official_pool_size": pool_size,
                "offline_seed": int(offline_seed),
                "sample_sizes": list(sample_sizes),
                "permutation_hash": _permutation_hash(permutation),
                "x_pool_shape": [int(value) for value in x_pool_shape],
                "y_pool_shape": [int(value) for value in y_pool_shape],
            }
            metadata_path = directory / "metadata.json"
            if metadata_path.exists():
                existing = json.loads(metadata_path.read_text(encoding="utf-8"))
                immutable_keys = (
                    "problem",
                    "dataset_source",
                    "official_pool_size",
                    "offline_seed",
                    "permutation_hash",
                    "x_pool_shape",
                    "y_pool_shape",
                )
                mismatches = [
                    key for key in immutable_keys if existing.get(key) != metadata[key]
                ]
                if mismatches:
                    raise ValueError(
                        f"Official subset metadata mismatch for '{canonical}': "
                        f"{mismatches}."
                    )
            _atomic_write_json(metadata_path, metadata)
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    selected_path = directory / f"indices_N{int(sample_sizes[-1])}.npy"
    return directory, permutation, selected_path


def load_official_subset(
    cache_root,
    problem_name,
    sample_size,
    offline_seed,
    all_sample_sizes,
    validation_fraction=0.2,
):
    """Load a nested subset, split validation internally, and keep test isolated."""

    _, pools = load_official_pool(problem_name)
    directory, permutation, _ = cache_official_subset_indices(
        cache_root=cache_root,
        problem_name=problem_name,
        offline_seed=offline_seed,
        sample_sizes=all_sample_sizes,
        x_pool_shape=pools["X_pool"].shape,
        y_pool_shape=pools["Y_pool"].shape,
    )
    sample_size = int(sample_size)
    indices = np.load(directory / f"indices_N{sample_size}.npy", allow_pickle=False)
    if not np.array_equal(indices, permutation[:sample_size]):
        raise ValueError("Official subset is not a prefix of the cached permutation.")

    validation_fraction = float(validation_fraction)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    validation_size = max(1, int(round(sample_size * validation_fraction)))
    if validation_size >= sample_size:
        raise ValueError("Validation split leaves no surrogate-fitting samples.")

    fit_indices = np.asarray(indices[:-validation_size], dtype=np.int64)
    validation_indices = np.asarray(indices[-validation_size:], dtype=np.int64)
    x_pool, y_pool = pools["X_pool"], pools["Y_pool"]
    return {
        # The selected N-row offline dataset is the model-training dataset.
        # Every method therefore receives exactly the configured sample size.
        "X_offline": x_pool[indices].copy(),
        "y_offline": y_pool[indices].copy(),
        "X_train": x_pool[indices].copy(),
        "y_train": y_pool[indices].copy(),
        # This internal split is used only to choose BNN early-stopping steps.
        # The final BNN is reinitialized and fitted on all N rows afterward.
        "X_fit": x_pool[fit_indices].copy(),
        "y_fit": y_pool[fit_indices].copy(),
        "X_val": x_pool[validation_indices].copy(),
        "y_val": y_pool[validation_indices].copy(),
        "X_test": pools["X_test"].copy(),
        "y_test": pools["Y_test"].copy(),
        # Fixed evaluation-only normalization bounds.  Keeping the full pool
        # here makes HV/IGD+ comparable across N, offline seeds, and methods;
        # this matrix is never passed to model fitting or optimization.
        "metric_reference_values": y_pool.copy(),
        # Evaluation-only fallback for tasks without a true Pareto front.  It
        # is never passed to model fitting, normalization, or optimization.
        "igd_reference_values": y_pool.copy(),
        "igd_reference_source": np.asarray(
            "official_training_pool_non_dominated_front"
        ),
        "offline_indices": np.asarray(indices, dtype=np.int64),
        "fit_indices": fit_indices,
        "validation_indices": validation_indices,
        "offline_seed": np.asarray(int(offline_seed)),
        "model_seed": np.asarray(int(offline_seed)),
        "offline_sample_size": np.asarray(sample_size),
        "fit_size": np.asarray(sample_size),
        "early_stopping_fit_size": np.asarray(len(fit_indices)),
        "validation_size": np.asarray(validation_size),
        "test_size": np.asarray(len(pools["X_test"])),
        "dataset_source": np.asarray("official_pool"),
    }
