#!/usr/bin/env python3
"""Run Off-MOO End2End, MultipleModels, COM, and MOBO baselines."""

from __future__ import annotations

import argparse
import concurrent.futures
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

from experiments.DL_MOBO_baseline.core import (  # noqa: E402
    BASELINE_NAMES,
    PROTOCOL_VERSION,
    append_row,
    load_config,
    method_configuration_hash,
    read_success_keys,
    result_key,
    run_group,
)


def _csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _resolve_relative(value, config_path: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--methods", help="comma-separated baseline names")
    parser.add_argument("--problems", help="comma-separated canonical problem names")
    parser.add_argument("--training-sizes", help="comma-separated offline N values")
    parser.add_argument("--offline-seeds", help="comma-separated subset/model seeds")
    parser.add_argument("--optimization-seeds", help="comma-separated optimizer seeds")
    parser.add_argument("--n-gen", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--subset-cache-dir", type=Path)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="skip successful rows already present in the results CSV (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="re-run configurations already present in the results CSV",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    config = load_config(config_path)
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
    args.n_gen = int(
        args.n_gen if args.n_gen is not None else config["optimizer"]["n_gen"]
    )
    args.pop_size = int(
        args.pop_size
        if args.pop_size is not None
        else config["optimizer"]["pop_size"]
    )
    if args.epochs is not None:
        config["neural"]["epochs"] = int(args.epochs)
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
    unknown = sorted(set(args.methods) - set(BASELINE_NAMES))
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    if not args.problems:
        parser.error("at least one problem is required")
    if any(size < 2 for size in args.training_sizes):
        parser.error("training sizes must be at least 2")
    if args.n_gen < 1 or args.pop_size < 2:
        parser.error("--n-gen must be >=1 and --pop-size must be >=2")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
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


def main(argv=None) -> int:
    args, config = parse_args(argv)
    plan = build_plan(args)
    run_count = len(plan) * len(args.optimization_seeds)
    if args.dry_run:
        print(f"Protocol: {PROTOCOL_VERSION}")
        print(
            f"Plan: {len(plan)} model/data groups, {run_count} optimizer runs; "
            f"device={args.device}; max_workers={args.max_workers}"
        )
        print(f"Output: {args.output_dir}")
        print(f"Subset cache: {args.subset_cache_dir}")
        for method, problem, size, offline_seed in plan:
            print(
                f"{method} | {problem} | N={size} | offline_seed={offline_seed} "
                f"| opt_seeds={args.optimization_seeds}"
            )
        return 0

    results_path = args.output_dir / "dl_baselines.csv"
    completed = read_success_keys(results_path) if args.resume else set()

    def configuration_hash_for(method):
        return method_configuration_hash(
            method, config["neural"], config["com"], config["mobo"]
        )

    def pending_seeds(task):
        method, problem, size, offline_seed = task
        configuration_hash = configuration_hash_for(method)
        return tuple(
            int(opt_seed)
            for opt_seed in args.optimization_seeds
            if (
                PROTOCOL_VERSION,
                configuration_hash,
                problem,
                method,
                int(size),
                int(offline_seed),
                int(opt_seed),
                int(args.n_gen),
                int(args.pop_size),
            )
            not in completed
        )

    scheduled = [
        (task, seeds) for task in plan if (seeds := pending_seeds(task))
    ]
    skipped = run_count - sum(len(seeds) for _, seeds in scheduled)
    progress = ProgressReporter(run_count, skipped)

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
            "n_gen": args.n_gen,
            "pop_size": args.pop_size,
            "neural_config": config["neural"],
            "com_config": config["com"],
            "mobo_config": config["mobo"],
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
