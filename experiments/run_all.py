#!/usr/bin/env python3
"""Run the complete training-sample-size sensitivity experiment."""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
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

# Spawned workers inherit these limits before importing NumPy, BLAS, or Torch.
set_worker_thread_environment(1)
suppress_known_optional_dependency_warnings()
# Batch experiments only write summary figures; never initialize an interactive
# display backend on login nodes, compute nodes, or local headless test runs.
os.environ.setdefault("MPLBACKEND", "Agg")

from sample_size_common import (
    LHS_SEEDS, METHOD_REGISTRY, OPT_SEEDS, PROBLEMS, TRAIN_SIZES, TEST_SIZE,
    append_rows, cleanup_model_storage, configure_method_settings,
    load_config_file, organize_cache_files,
    current_protocol_version, read_result_rows, reconcile_result_csvs,
    result_optimizer_settings, result_protocol_version, run_group,
    run_predictor_family_group,
    valid_success, write_manifest,
)
from sample_size_summary import summarize


OFFICIAL_SAMPLE_SIZES = (50, 100, 200, 400, 1000)
OFFICIAL_OFFLINE_SEEDS = tuple(range(1, 11))
OFFICIAL_OPTIMIZATION_SEEDS = tuple(range(1, 11))


def comma_values(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml")
    parser.add_argument("--problems", help="comma-separated problem names")
    parser.add_argument("--methods", help="comma-separated registry method names")
    parser.add_argument("--train-sizes")
    parser.add_argument("--lhs-seeds")
    parser.add_argument(
        "--offline-seeds",
        help="alias for --lhs-seeds; controls official-pool subset/model seeds",
    )
    parser.add_argument("--opt-seeds")
    parser.add_argument(
        "--dataset-source",
        choices=("lhs", "official_pool"),
        help="data source (default: config; legacy default is lhs)",
    )
    parser.add_argument("--subset-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "deprecated compatibility flag; failed rows are always retried "
            "when --resume is enabled"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--tabpfn-max-workers",
        type=int,
        help="maximum concurrent TabPFN groups (default: config value, capped separately)",
    )
    parser.add_argument("--n-gen", type=int,
                        help="optimizer generations (exposed for smoke testing)")
    parser.add_argument("--pop-size", type=int,
                        help="optimizer population size (exposed for smoke testing)")
    args = parser.parse_args(argv)
    root_config = load_config_file(args.config)
    configure_method_settings(root_config)
    args.root_config = root_config
    config = root_config.get("sample_size_ablation", {})
    args.dataset_source = (
        args.dataset_source
        or str(config.get("dataset_source", root_config.get("dataset_source", "lhs")))
    ).strip().lower()
    args.problems = comma_values(args.problems) or list(root_config.get("problem_names", PROBLEMS))
    requested_methods = comma_values(args.methods)
    disabled_methods = {
        str(method) for method in config.get("disabled_methods", ())
    }
    args.methods = requested_methods or [
        method for method in METHOD_REGISTRY if method not in disabled_methods
    ]
    if args.dataset_source == "official_pool":
        configured_train_sizes = config.get(
            "offline_sample_sizes",
            OFFICIAL_SAMPLE_SIZES,
        )
        configured_offline_seeds = config.get(
            "offline_seeds",
            OFFICIAL_OFFLINE_SEEDS,
        )
        configured_opt_seeds = config.get(
            "optimization_seeds",
            OFFICIAL_OPTIMIZATION_SEEDS,
        )
    else:
        configured_train_sizes = config.get("train_sizes", TRAIN_SIZES)
        configured_offline_seeds = config.get("lhs_seeds", LHS_SEEDS)
        configured_opt_seeds = config.get("opt_seeds", OPT_SEEDS)
    args.train_sizes = comma_values(args.train_sizes, int) or list(configured_train_sizes)
    cli_offline_seeds = args.offline_seeds or args.lhs_seeds
    args.lhs_seeds = comma_values(cli_offline_seeds, int) or list(configured_offline_seeds)
    args.opt_seeds = comma_values(args.opt_seeds, int) or list(configured_opt_seeds)
    args.n_gen = args.n_gen if args.n_gen is not None else int(config.get("n_gen", root_config.get("n_gen", 100)))
    args.pop_size = args.pop_size if args.pop_size is not None else int(config.get("pop_size", root_config.get("pop_size", 100)))
    args.max_workers = args.max_workers if args.max_workers is not None else int(config.get("max_workers", 1))
    args.tabpfn_max_workers = (
        args.tabpfn_max_workers
        if args.tabpfn_max_workers is not None
        else int(config.get("tabpfn_max_workers", 5))
    )
    if args.output_dir is None:
        configured_output = Path(config.get("output_dir", "results"))
        args.output_dir = configured_output if configured_output.is_absolute() else args.config.parent / configured_output
    if args.subset_cache_dir is None:
        configured_cache = Path(config.get("subset_cache_dir", "data_subsets"))
        args.subset_cache_dir = (
            configured_cache
            if configured_cache.is_absolute()
            else args.config.parent / configured_cache
        )
    unknown_problems = sorted(set(args.problems) - set(PROBLEMS))
    unknown_methods = sorted(set(args.methods) - set(METHOD_REGISTRY))
    if unknown_problems:
        parser.error(f"unknown problems: {unknown_problems}")
    if unknown_methods:
        parser.error(f"unknown methods: {unknown_methods}")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if args.tabpfn_max_workers < 1:
        parser.error("--tabpfn-max-workers must be at least 1")
    if args.n_gen < 1:
        parser.error("--n-gen must be at least 1")
    if args.pop_size < 2:
        parser.error("--pop-size must be at least 2")
    if args.dataset_source not in {"lhs", "official_pool"}:
        parser.error("dataset_source must be 'lhs' or 'official_pool'")
    return args


def _key(
    dataset_source,
    protocol_version,
    n_gen,
    pop_size,
    problem,
    method,
    training_size,
    lhs_seed,
    opt_seed,
):
    return (
        str(dataset_source),
        str(protocol_version),
        int(n_gen),
        int(pop_size),
        problem,
        method,
        int(training_size),
        int(lhs_seed),
        int(opt_seed),
    )


def build_plan(args):
    existing = read_result_rows(args.output_dir) if args.resume else []
    successful = {
        _key(
            row.get("dataset_source") or "lhs",
            result_protocol_version(row),
            *result_optimizer_settings(row),
            row["problem"],
            row["method"],
            row["training_size"],
            row["lhs_seed"],
            row["opt_seed"],
        )
        for row in existing if valid_success(row)
    }
    groups = []
    skipped = 0
    for problem, size, lhs_seed, method in itertools.product(
            args.problems, args.train_sizes, args.lhs_seeds, args.methods):
        pending = []
        for opt_seed in args.opt_seeds:
            key = _key(
                args.dataset_source,
                current_protocol_version(args.dataset_source, method),
                args.n_gen,
                args.pop_size,
                problem,
                method,
                size,
                lhs_seed,
                opt_seed,
            )
            if args.resume and key in successful:
                skipped += 1
            else:
                pending.append(opt_seed)
        if pending:
            groups.append((problem, size, lhs_seed, method, tuple(pending)))
    return groups, skipped


def build_execution_stages(args, payloads):
    """Bundle primary methods by surrogate family and offline-data cell."""

    payloads = tuple(payloads)
    payloads_by_method = {
        method: tuple(payload for payload in payloads if payload[4] == method)
        for method in args.methods
    }
    stages = []
    seen = set()
    category_order = {"normal": 0, "dr": 1, "ebu_dr": 2}
    for method in args.methods:
        spec = METHOD_REGISTRY[method]
        is_primary = spec.category in category_order
        stage_key = ("family", spec.family) if is_primary else ("method", method)
        if stage_key in seen:
            continue
        seen.add(stage_key)
        if not is_primary:
            method_payloads = payloads_by_method[method]
            if method_payloads:
                stages.append({
                    "kind": "method",
                    "name": method,
                    "payloads": method_payloads,
                    "worker_limit": (
                        args.tabpfn_max_workers
                        if spec.family == "tabpfn"
                        else args.max_workers
                    ),
                })
            continue

        family_methods = sorted(
            (
                selected
                for selected in args.methods
                if METHOD_REGISTRY[selected].family == spec.family
                and METHOD_REGISTRY[selected].category in category_order
            ),
            key=lambda selected: category_order[METHOD_REGISTRY[selected].category],
        )
        cell_payloads = {}
        for selected in family_methods:
            for payload in payloads_by_method[selected]:
                cell_payloads.setdefault(payload[1:4], {})[selected] = payload
        bundled = []
        for cell in sorted(cell_payloads):
            methods_for_cell = cell_payloads[cell]
            first = next(iter(methods_for_cell.values()))
            method_opt_seeds = tuple(
                (selected, methods_for_cell[selected][5])
                for selected in family_methods
                if selected in methods_for_cell
            )
            bundled.append((
                first[0],
                first[1],
                first[2],
                first[3],
                spec.family,
                method_opt_seeds,
                *first[6:],
            ))
        if bundled:
            stages.append({
                "kind": "family",
                "name": spec.family,
                "payloads": tuple(bundled),
                "worker_limit": args.max_workers,
            })
    return stages


def _release_worker_memory():
    """Release cyclic garbage and any already-loaded PyTorch CUDA cache."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    try:
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()
    except Exception:
        # Cleanup must never turn a completed experiment into a failed run.
        pass


def _execute(payload):
    initialize_worker_threads(1)
    (
        output_dir,
        problem,
        size,
        lhs_seed,
        method,
        seeds,
        n_gen,
        pop_size,
        dataset_source,
        subset_cache_dir,
        all_sample_sizes,
        root_config,
    ) = payload
    configure_method_settings(root_config)
    log_path = group_log_path(output_dir, problem, size, lhs_seed, method)
    with redirect_process_output(log_path) as log_handle:
        print(
            f"\n=== source={dataset_source} | problem={problem} | N={size} | "
            f"offline_seed={lhs_seed} | "
            f"method={method} | opt={list(seeds)} ===",
            file=log_handle,
            flush=True,
        )
        try:
            result = run_group(
                Path(output_dir),
                problem,
                size,
                lhs_seed,
                method,
                seeds,
                n_gen,
                pop_size,
                dataset_source=dataset_source,
                subset_cache_root=Path(subset_cache_dir),
                all_sample_sizes=all_sample_sizes,
            )
            rows = result[0]
            success_count = sum(row.get("status") == "success" for row in rows)
            print(
                f"=== worker completed | success={success_count} | "
                f"failed={len(rows) - success_count} ===",
                file=log_handle,
                flush=True,
            )
            return result
        except Exception:
            _release_worker_memory()
            raise


def _execute_primary_family(payload):
    initialize_worker_threads(1)
    (
        output_dir,
        problem,
        size,
        lhs_seed,
        family,
        method_opt_seeds,
        n_gen,
        pop_size,
        dataset_source,
        subset_cache_dir,
        all_sample_sizes,
        root_config,
    ) = payload
    configure_method_settings(root_config)
    method_label = " -> ".join(method for method, _ in method_opt_seeds)
    log_path = group_log_path(
        output_dir,
        problem,
        size,
        lhs_seed,
        f"{family}_shared_normal_dr_ebu_dr",
    )
    with redirect_process_output(log_path) as log_handle:
        print(
            f"\n=== source={dataset_source} | problem={problem} | N={size} | "
            f"offline_seed={lhs_seed} | surrogate={family} | "
            f"methods={method_label} ===",
            file=log_handle,
            flush=True,
        )
        try:
            result = run_predictor_family_group(
                Path(output_dir),
                problem,
                size,
                lhs_seed,
                method_opt_seeds,
                n_gen,
                pop_size,
                dataset_source=dataset_source,
                subset_cache_root=Path(subset_cache_dir),
                all_sample_sizes=all_sample_sizes,
            )
            rows = result[0]
            success_count = sum(row.get("status") == "success" for row in rows)
            print(
                f"=== shared-surrogate worker completed | success={success_count} | "
                f"failed={len(rows) - success_count} ===",
                file=log_handle,
                flush=True,
            )
            return result
        except Exception:
            _release_worker_memory()
            raise


def _execute_sequence(payloads, kind="method"):
    rows = []
    execute = _execute_primary_family if kind == "family" else _execute
    for payload in payloads:
        method_rows, model_objects = execute(payload)
        try:
            append_rows(Path(payload[0]), method_rows)
        finally:
            cleanup_model_storage(model_objects)
            del model_objects
            _release_worker_memory()
        rows.extend(method_rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    total = (len(args.problems) * len(args.train_sizes) * len(args.lhs_seeds) *
             len(args.opt_seeds) * len(args.methods))
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "logs").mkdir(exist_ok=True)
        (args.output_dir / "csv").mkdir(exist_ok=True)
        organize_cache_files(args.output_dir)
        reconcile_result_csvs(args.output_dir)
    groups, skipped = build_plan(args)
    pending = sum(len(group[-1]) for group in groups)
    payloads = [
        (
            str(args.output_dir),
            *group,
            args.n_gen,
            args.pop_size,
            args.dataset_source,
            str(args.subset_cache_dir),
            tuple(args.train_sizes),
            args.root_config,
        )
        for group in groups
    ]
    execution_stages = build_execution_stages(args, payloads)
    if args.dry_run:
        print(f"Problems: {len(args.problems)} | methods: {len(args.methods)}")
        print(f"Config: {args.config}")
        print(f"Dataset source: {args.dataset_source}")
        print(
            f"Optimizer: max_gen={args.n_gen} | population_size={args.pop_size}"
        )
        print(f"Total optimization tasks: {total} = {len(args.problems)} problems × "
              f"{len(args.train_sizes)} sizes × {len(args.lhs_seeds)} offline seeds × "
              f"{len(args.opt_seeds)} optimizer seeds × {len(args.methods)} methods")
        worker_groups = sum(len(stage["payloads"]) for stage in execution_stages)
        shared_groups = sum(
            len(stage["payloads"])
            for stage in execution_stages
            if stage["kind"] == "family"
        )
        print(
            f"Pending: {pending} | skipped by resume: {skipped} | "
            f"worker groups: {worker_groups} | "
            f"shared-surrogate fits: {shared_groups}"
        )
        for problem, size, lhs_seed, method, seeds in groups:
            print(
                f"  {problem} | N={size} | offline_seed={lhs_seed} | "
                f"{method} | opt={list(seeds)}"
            )
        return 0

    write_manifest(args.output_dir, {
        "config": str(args.config.resolve()),
        "dataset_source": args.dataset_source,
        "protocol_version": current_protocol_version(args.dataset_source),
        "subset_cache_dir": str(args.subset_cache_dir.resolve()),
        "problems": args.problems, "methods": args.methods,
        "train_sizes": args.train_sizes, "offline_seeds": args.lhs_seeds,
        "opt_seeds": args.opt_seeds,
        "test_size": (
            "official_test_pool"
            if args.dataset_source == "official_pool"
            else TEST_SIZE
        ),
        "n_gen": args.n_gen,
        "pop_size": args.pop_size, "total_tasks": total,
        "max_workers": args.max_workers,
        "tabpfn_max_workers": args.tabpfn_max_workers,
    })
    progress = ProgressReporter(total, skipped)

    def record_progress(rows, label, write_results=True):
        if write_results:
            append_rows(args.output_dir, rows)
        success_count = sum(row.get("status") == "success" for row in rows)
        failed_count = len(rows) - success_count
        progress.record(success_count, failed_count, label)

    progress.start()
    # Primary stages are grouped by surrogate family. Each worker trains one
    # final surrogate per offline-data cell, then runs normal -> DR -> EBU-DR.
    # Non-primary methods retain their independent method stages.
    for stage in execution_stages:
        stage_payloads = stage["payloads"]
        stage_workers = min(stage["worker_limit"], len(stage_payloads))
        execute = _execute_primary_family if stage["kind"] == "family" else _execute

        if stage_workers == 1:
            for payload in stage_payloads:
                rows, model_objects = execute(payload)
                try:
                    method_label = (
                        " -> ".join(method for method, _ in payload[5])
                        if stage["kind"] == "family"
                        else payload[4]
                    )
                    label = (
                        f"{payload[1]} | N={payload[2]} | "
                        f"offline_seed={payload[3]} | {method_label}"
                    )
                    record_progress(rows, label)
                finally:
                    cleanup_model_storage(model_objects)
                    del model_objects
                    _release_worker_memory()
            continue

        context = mp.get_context("spawn")
        active_label = "not-yet-reported task"
        try:
            with ProcessPoolExecutor(
                max_workers=stage_workers,
                mp_context=context,
                initializer=initialize_worker_threads,
                initargs=(1,),
            ) as executor:
                future_labels = {}
                for payload in stage_payloads:
                    future = executor.submit(
                        _execute_sequence,
                        (payload,),
                        stage["kind"],
                    )
                    method_label = (
                        " -> ".join(method for method, _ in payload[5])
                        if stage["kind"] == "family"
                        else payload[4]
                    )
                    future_labels[future] = (
                        f"{payload[1]} | N={payload[2]} | "
                        f"offline_seed={payload[3]} | {method_label}"
                    )
                for future in as_completed(future_labels):
                    active_label = future_labels[future]
                    record_progress(
                        future.result(),
                        active_label,
                        write_results=False,
                    )
        except BrokenProcessPool:
            print(
                "[worker-pool-failed] A worker exited without a Python "
                f"exception while processing the {stage['name']} stage "
                f"(observed while waiting for: {active_label}).",
                file=sys.stderr,
                flush=True,
            )
            print(
                "Completed CSV rows were already saved. Check the matching "
                f"{args.output_dir / 'logs'}/*.log and the SLURM job's "
                "MaxRSS/OUT_OF_MEMORY status, then rerun the same command "
                "with --resume. If SLURM reports OOM, reduce --max-workers.",
                file=sys.stderr,
                flush=True,
            )
            return 2
    summarize(args.output_dir, args.output_dir)
    progress.complete(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
