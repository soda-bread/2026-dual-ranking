#!/usr/bin/env python3
"""Run PCD and ParetoFlow under the dual-ranking protocol."""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.generative_baseline import (  # noqa: E402
    BASELINE_NAMES,
    PROTOCOL_VERSION,
)
from experiments.generative_baseline.common import (  # noqa: E402
    append_row,
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
        hidden_sizes=[32, 32], epochs=1, batch_size=16
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
        epochs=1,
        batch_size=16,
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


def main(argv=None):
    args, config = parse_args(argv)
    plan = build_plan(args)
    total = len(plan) * len(args.optimization_seeds)
    print(f"Protocol: {PROTOCOL_VERSION}")
    print(
        f"Plan: {len(plan)} model/data groups, {total} generation runs; "
        f"output_size={args.output_size}; device={args.device}"
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

    results_path = args.output_dir / "generative_baselines.csv"
    completed = read_success_keys(results_path) if args.resume else set()
    succeeded = failed = 0
    for index, (method, problem, size, offline_seed) in enumerate(plan, 1):
        print(
            f"[{index}/{len(plan)}] {method} | {problem} | N={size} "
            f"| offline_seed={offline_seed}"
        )
        try:
            rows = run_group(
                method=method,
                problem=problem,
                training_size=size,
                offline_seed=offline_seed,
                opt_seeds=args.optimization_seeds,
                all_training_sizes=args.training_sizes,
                subset_cache_dir=args.subset_cache_dir,
                output_dir=args.output_dir,
                output_size=args.output_size,
                method_config=config[METHOD_CONFIG_KEYS[method]],
                proxy_config=config["proxy"],
                device_name=args.device,
                completed=completed,
            )
        except Exception as error:
            failed += len(args.optimization_seeds)
            print(f"  group failed: {type(error).__name__}: {error}", file=sys.stderr)
            continue
        if not rows:
            print("  skipped: all requested runs already succeeded")
            continue
        for row in rows:
            append_row(results_path, row)
            if row["status"] == "success":
                succeeded += 1
                completed.add(result_key(row))
            else:
                failed += 1
            print(
                f"  opt_seed={row['opt_seed']} status={row['status']} "
                f"HVreal={row.get('HVreal', '')}"
            )
    print(f"Finished: {succeeded} succeeded, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
