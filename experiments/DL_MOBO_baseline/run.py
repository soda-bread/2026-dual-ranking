#!/usr/bin/env python3
"""Run Off-MOO End2End, MultipleModels, COM, and MOBO baselines."""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import multiprocessing as mp
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.DL_MOBO_baseline.core import (  # noqa: E402
    BASELINE_NAMES,
    PROTOCOL_VERSION,
    append_row,
    load_config,
    read_success_keys,
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

    return run_group(**payload)


def main(argv=None) -> int:
    args, config = parse_args(argv)
    plan = build_plan(args)
    run_count = len(plan) * len(args.optimization_seeds)
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(
        f"Plan: {len(plan)} model/data groups, {run_count} optimizer runs; "
        f"device={args.device}; max_workers={args.max_workers}"
    )
    print(f"Output: {args.output_dir}")
    print(f"Subset cache: {args.subset_cache_dir}")
    if args.dry_run:
        for method, problem, size, offline_seed in plan:
            print(
                f"{method} | {problem} | N={size} | offline_seed={offline_seed} "
                f"| opt_seeds={args.optimization_seeds}"
            )
        return 0

    results_path = args.output_dir / "dl_baselines.csv"
    completed = read_success_keys(results_path) if args.resume else set()
    succeeded = failed = 0

    def payload_for(task):
        method, problem, size, offline_seed = task
        return {
            "method": method,
            "problem": problem,
            "training_size": size,
            "offline_seed": offline_seed,
            "opt_seeds": args.optimization_seeds,
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

    def record_group(group_index, task, rows=None, error=None):
        nonlocal succeeded, failed
        method, problem, size, offline_seed = task
        print(
            f"[{group_index}/{len(plan)}] {method} | {problem} | N={size} "
            f"| offline_seed={offline_seed}"
        )
        if error is not None:
            failed += len(args.optimization_seeds)
            print(f"  group failed: {type(error).__name__}: {error}", file=sys.stderr)
            return
        if not rows:
            print("  skipped: all requested runs already succeeded")
            return
        for row in rows:
            append_row(results_path, row)
            if row["status"] == "success":
                succeeded += 1
                completed.add(
                    (
                        row["protocol_version"],
                        row["configuration_hash"],
                        row["problem"],
                        row["method"],
                        row["training_size"],
                        row["offline_seed"],
                        row["opt_seed"],
                        row["configured_n_gen"],
                        row["configured_pop_size"],
                    )
                )
            else:
                failed += 1
            print(
                f"  opt_seed={row['opt_seed']} status={row['status']} "
                f"HVreal={row.get('HVreal', '')}"
            )

    indexed_plan = list(enumerate(plan, 1))
    for method_name in args.methods:
        stage = [item for item in indexed_plan if item[1][0] == method_name]
        worker_count = min(args.max_workers, len(stage)) if stage else 1
        if worker_count == 1:
            for group_index, task in stage:
                try:
                    rows = _execute_group(payload_for(task))
                except Exception as error:
                    record_group(group_index, task, error=error)
                else:
                    record_group(group_index, task, rows=rows)
            continue

        print(
            f"Running {method_name} groups with {worker_count} worker processes"
        )
        context = mp.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_execute_group, payload_for(task)): (index, task)
                for index, task in stage
            }
            for future in concurrent.futures.as_completed(futures):
                group_index, task = futures[future]
                try:
                    rows = future.result()
                except Exception as error:
                    record_group(group_index, task, error=error)
                else:
                    record_group(group_index, task, rows=rows)
    print(f"Finished: {succeeded} succeeded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
