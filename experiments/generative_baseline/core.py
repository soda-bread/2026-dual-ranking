"""Unified runner for PCD and ParetoFlow.

Each model is fitted once for an official-pool subset/model seed and then
sampled independently for every optimization seed.  True objectives are only
queried after the final candidate set has been selected.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from experiments.generative_baseline.common import (
    base_row,
    complete_row,
    configuration_hash,
    evaluate_candidates,
    load_data,
    resolve_device,
    result_key,
)


def _fit(method, problem, data, method_config, proxy_config, model_seed, device):
    if method == "PCD":
        from experiments.generative_baseline.pcd import fit_pcd

        return fit_pcd(
            data, method_config, model_seed, device, problem_name=problem
        )
    if method == "ParetoFlow":
        from experiments.generative_baseline.paretoflow import fit_paretoflow

        return fit_paretoflow(
            data, method_config, proxy_config, model_seed, device
        )
    raise ValueError(f"Unknown generative baseline: {method!r}.")


def _generate(method, model, data, method_config, opt_seed, output_size, task):
    if method == "PCD":
        from experiments.generative_baseline.pcd import generate_pcd

        return generate_pcd(
            model, data, method_config, opt_seed, output_size, task
        )
    if method == "ParetoFlow":
        from experiments.generative_baseline.paretoflow import generate_paretoflow

        return generate_paretoflow(
            model, data, method_config, opt_seed, output_size, task
        )
    raise ValueError(f"Unknown generative baseline: {method!r}.")


def _proxy_mse(method, model, data):
    if method == "PCD":
        return np.nan
    prediction = model.proxy.predict(data["X_test"])
    return float(np.mean((prediction - data["y_test"]) ** 2))


def _candidate_path(output_dir, method, problem, training_size, offline_seed, opt_seed):
    directory = Path(output_dir) / "candidates" / method
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (
        f"{problem}_N{training_size}_offline{offline_seed}_opt{opt_seed}.npz"
    )


def _save_candidates(path, evaluation):
    payload = {
        "X": evaluation["candidates"],
        "F_real": evaluation["real_y"],
    }
    if evaluation["surrogate_y"] is not None:
        payload["F_surrogate"] = evaluation["surrogate_y"]
    np.savez_compressed(path, **payload)


def run_group(
    *,
    method: str,
    problem: str,
    training_size: int,
    offline_seed: int,
    opt_seeds: Iterable[int],
    all_training_sizes: Sequence[int],
    subset_cache_dir: Path,
    output_dir: Path,
    output_size: int,
    method_config: dict[str, Any],
    proxy_config: dict[str, Any],
    device_name: str,
    completed: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    task, data = load_data(
        problem,
        training_size,
        offline_seed,
        all_training_sizes,
        subset_cache_dir,
    )
    relevant_config = {
        "algorithm": method_config,
        "proxy": proxy_config if method == "ParetoFlow" else None,
    }
    config_hash = configuration_hash(method, relevant_config)
    pending = []
    for opt_seed in opt_seeds:
        probe = base_row(
            problem,
            method,
            training_size,
            offline_seed,
            int(opt_seed),
            data,
            output_size,
            config_hash,
        )
        if result_key(probe) not in completed:
            pending.append(int(opt_seed))
    if not pending:
        return []

    device = resolve_device(device_name)
    training_started = time.perf_counter()
    try:
        model = _fit(
            method,
            problem,
            data,
            method_config,
            proxy_config,
            offline_seed,
            device,
        )
        training_time = time.perf_counter() - training_started
        mse_pre = _proxy_mse(method, model, data)
    except Exception as error:
        training_time = time.perf_counter() - training_started
        rows = []
        for opt_seed in pending:
            row = base_row(
                problem,
                method,
                training_size,
                offline_seed,
                opt_seed,
                data,
                output_size,
                config_hash,
            )
            row["runtime_training"] = training_time
            row["error_message"] = f"{type(error).__name__}: {error}"
            rows.append(row)
        return rows

    rows = []
    for opt_seed in pending:
        row = base_row(
            problem,
            method,
            training_size,
            offline_seed,
            opt_seed,
            data,
            output_size,
            config_hash,
        )
        generation_started = time.perf_counter()
        try:
            candidates, surrogate_y = _generate(
                method,
                model,
                data,
                method_config,
                opt_seed,
                output_size,
                task,
            )
            candidates = np.asarray(candidates, dtype=float)
            if candidates.ndim != 2 or len(candidates) != int(output_size):
                raise ValueError(
                    f"{method} must return exactly {output_size} candidates; "
                    f"received shape {candidates.shape}."
                )
            evaluation = evaluate_candidates(
                problem, task, data, candidates, surrogate_y
            )
            generation_time = time.perf_counter() - generation_started
            path = _candidate_path(
                output_dir,
                method,
                problem,
                training_size,
                offline_seed,
                opt_seed,
            )
            _save_candidates(path, evaluation)
            relative_path = path.relative_to(output_dir)
            complete_row(
                row,
                evaluation,
                training_time,
                generation_time,
                relative_path,
                mse_pre,
            )
        except Exception as error:
            row["runtime_training"] = training_time
            row["runtime_generation"] = time.perf_counter() - generation_started
            row["MSEpre"] = mse_pre
            row["error_message"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return rows
