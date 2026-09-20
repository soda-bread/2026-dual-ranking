#!/usr/bin/env python3
"""Run PCD and ParetoFlow under the dual-ranking protocol."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import itertools
import multiprocessing as mp
import sys
import traceback
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.worker_runtime import (  # noqa: E402
    group_log_path,
    initialize_worker_threads,
    redirect_process_output,
    set_worker_thread_environment,
    suppress_known_optional_dependency_warnings,
)
from experiments.progress import ProgressReporter  # noqa: E402

set_worker_thread_environment(1)
suppress_known_optional_dependency_warnings()

from experiments.generative_baseline import (  # noqa: E402
    BASELINE_NAMES,
    PROTOCOL_VERSION,
)
from experiments.generative_baseline.common import (  # noqa: E402
    append_row,
    configuration_hash,
    load_config,
    read_success_keys,
    result_key,
)
from experiments.generative_baseline.core import run_group  # noqa: E402
from src.problem_specs import PROBLEM_SPECS  # noqa: E402


METHOD_CONFIG_KEYS = {
    "PCD": "pcd",
    "ParetoFlow": "paretoflow",
}


def _csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _resolve_relative(value, config_path):
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def _smoke_config(config):
    config = copy.deepcopy(config)
    config["proxy"].update(
        trainer="paretoflow_upstream",
        hidden_sizes=[32, 32],
        epochs=2,
        batch_size=16,
        validation_fraction=0.25,
        min_validation_rows=2,
    )
    config["pcd"].update(
        width=32,
        depth=1,
        time_dim=16,
        diffusion_steps=4,
        train_steps=2,
        batch_size=16,
    )
    config["pcd"].setdefault("re_overrides", {}).update(
        width=32,
        depth=1,
        time_dim=16,
        diffusion_steps=4,
        train_steps=2,
        batch_size=16,
    )
    config["paretoflow"].update(
        hidden_size=32,
        train_steps=4,
        min_train_steps=2,
        batch_size=16,
        validation_fraction=0.25,
        min_validation_rows=2,
        validation_interval=1,
        validation_repeats=2,
        patience=1,
        sampling_steps=4,
        offspring_count=2,
    )
    return config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--methods", help="comma-separated method names")
    parser.add_argument("--problems", help="comma-separated canonical problem names")
    parser.add_argument("--training-sizes", help="comma-separated official-pool N values")
    parser.add_argument("--offline-seeds", help="comma-separated subset/model seeds")
    parser.add_argument("--optimization-seeds", help="comma-separated generation seeds")
    parser.add_argument("--output-size", type=int)
    parser.add_argument("--proxy-epochs", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subset-cache-dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use tiny model/training settings while preserving the selected plan",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.smoke:
        config = _smoke_config(config)
    args.methods = _csv(args.methods) or list(config["methods"])
    args.problems = _csv(args.problems) or list(config["problems"])
    args.training_sizes = _csv(args.training_sizes, int) or list(
        config["training_sizes"]
    )
    args.offline_seeds = _csv(args.offline_seeds, int) or list(
        config["offline_seeds"]
    )
    args.optimization_seeds = _csv(args.optimization_seeds, int) or list(
        config["optimization_seeds"]
    )
    output_size_was_set = args.output_size is not None
    args.output_size = int(
        args.output_size if output_size_was_set else config["output_size"]
    )
    if args.smoke and not output_size_was_set:
        args.output_size = min(args.output_size, 8)
    if args.proxy_epochs is not None:
        config["proxy"]["epochs"] = int(args.proxy_epochs)
    args.max_workers = int(
        args.max_workers
        if args.max_workers is not None
        else config.get("max_workers", 1)
    )
    args.output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else _resolve_relative(config["output_dir"], config_path)
    )
    args.subset_cache_dir = (
        args.subset_cache_dir.resolve()
        if args.subset_cache_dir
        else _resolve_relative(config["subset_cache_dir"], config_path)
    )

    unknown_methods = sorted(set(args.methods) - set(BASELINE_NAMES))
    unknown_problems = sorted(set(args.problems) - set(PROBLEM_SPECS))
    if unknown_methods:
        parser.error(f"unknown methods: {unknown_methods}")
    if unknown_problems:
        parser.error(f"unknown problems: {unknown_problems}")
    if any(size < 2 for size in args.training_sizes):
        parser.error("training sizes must be at least 2")
    if args.output_size < 2:
        parser.error("--output-size must be at least 2")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    for method in args.methods:
        key = METHOD_CONFIG_KEYS[method]
        if key not in config:
            parser.error(f"configuration is missing the {key!r} section")
    if "proxy" not in config:
        parser.error("configuration is missing the 'proxy' section")
    return args, config


def build_plan(args):
    return list(
        itertools.product(
            args.methods,
            args.problems,
            args.training_sizes,
            args.offline_seeds,
        )
    )


def _execute_group(payload):
    """Run one independent model/data group in a worker process."""

    initialize_worker_threads(1)
    log_path = group_log_path(
        payload["output_dir"],
        payload["problem"],
        payload["training_size"],
        payload["offline_seed"],
        payload["method"],
    )
    with redirect_process_output(log_path) as log_handle:
        print(
            f"\n=== source=official_pool | problem={payload['problem']} | "
            f"N={payload['training_size']} | "
            f"offline_seed={payload['offline_seed']} | "
            f"method={payload['method']} | opt={list(payload['opt_seeds'])} ===",
            file=log_handle,
            flush=True,
        )
        try:
            return run_group(**payload)
        except Exception:
            traceback.print_exc(file=log_handle)
            raise


def main(argv=None):
    args, config = parse_args(argv)
    plan = build_plan(args)
    total = len(plan) * len(args.optimization_seeds)
    if args.dry_run:
        print(f"Protocol: {PROTOCOL_VERSION}")
        print(
            f"Plan: {len(plan)} model/data groups, {total} generation runs; "
            f"output_size={args.output_size}; device={args.device}; "
            f"max_workers={args.max_workers}"
        )
        print(f"Output: {args.output_dir}")
        print(f"Subset cache: {args.subset_cache_dir}")
        for method, problem, size, offline_seed in plan:
            print(
                f"{method} | {problem} | N={size} | offline_seed={offline_seed} "
                f"| opt_seeds={args.optimization_seeds}"
            )
        return 0

    results_path = args.output_dir / "generative_baselines.csv"
    completed = read_success_keys(results_path) if args.resume else set()

    def configuration_hash_for(method):
        relevant_config = {
            "algorithm": config[METHOD_CONFIG_KEYS[method]],
            "proxy": config["proxy"] if method == "ParetoFlow" else None,
        }
        return configuration_hash(method, relevant_config)

    def pending_seeds(task):
        method, problem, size, offline_seed = task
        config_hash = configuration_hash_for(method)
        return tuple(
            int(opt_seed)
            for opt_seed in args.optimization_seeds
            if (
                PROTOCOL_VERSION,
                config_hash,
                problem,
                method,
                int(size),
                int(offline_seed),
                int(opt_seed),
                int(args.output_size),
            )
            not in completed
        )

    scheduled = [
        (task, seeds) for task in plan if (seeds := pending_seeds(task))
    ]
    skipped = total - sum(len(seeds) for _, seeds in scheduled)
    progress = ProgressReporter(total, skipped)

    def payload_for(task, seeds):
        method, problem, size, offline_seed = task
        return {
            "method": method,
            "problem": problem,
            "training_size": size,
            "offline_seed": offline_seed,
            "opt_seeds": seeds,
            "all_training_sizes": args.training_sizes,
            "subset_cache_dir": args.subset_cache_dir,
            "output_dir": args.output_dir,
            "output_size": args.output_size,
            "method_config": config[METHOD_CONFIG_KEYS[method]],
            "proxy_config": config["proxy"],
            "device_name": args.device,
            "completed": completed,
        }

    def record_group(task, seeds, rows=None, error=None):
        method, problem, size, offline_seed = task
        label = (
            f"{problem} | N={size} | offline_seed={offline_seed} | {method}"
        )
        if error is not None:
            progress.record(0, len(seeds), label)
            return
        success_count = 0
        failed_count = 0
        for row in rows:
            append_row(results_path, row)
            if row["status"] == "success":
                success_count += 1
                completed.add(result_key(row))
            else:
                failed_count += 1
        progress.record(success_count, failed_count, label)

    progress.start()
    for method_name in args.methods:
        stage = [item for item in scheduled if item[0][0] == method_name]
        worker_count = min(args.max_workers, len(stage)) if stage else 1
        if worker_count == 1:
            for task, seeds in stage:
                try:
                    rows = _execute_group(payload_for(task, seeds))
                except Exception as error:
                    record_group(task, seeds, error=error)
                else:
                    record_group(task, seeds, rows=rows)
            continue

        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
            initializer=initialize_worker_threads,
            initargs=(1,),
        ) as executor:
            futures = {
                executor.submit(_execute_group, payload_for(task, seeds)): (task, seeds)
                for task, seeds in stage
            }
            for future in concurrent.futures.as_completed(futures):
                task, seeds = futures[future]
                try:
                    rows = future.result()
                except Exception as error:
                    record_group(task, seeds, error=error)
                else:
                    record_group(task, seeds, rows=rows)
    progress.complete(args.output_dir)
    return 1 if progress.total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
