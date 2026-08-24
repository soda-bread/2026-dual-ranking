#!/usr/bin/env python3
"""Run the complete training-sample-size sensitivity experiment."""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
import re
import sys
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


from sample_size_common import (
    LHS_SEEDS, METHOD_REGISTRY, OPT_SEEDS, PROBLEMS, TRAIN_SIZES, VALIDATION_SIZE, TEST_SIZE,
    append_rows, cleanup_model_storage, load_config_file, organize_cache_files,
    current_protocol_version, read_result_rows, reconcile_result_csvs,
    result_optimizer_settings, result_protocol_version, run_group,
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
    parser.add_argument("--retry-failed", action="store_true")
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
    args.validation_fraction = float(config.get("validation_fraction", 0.2))
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
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("validation_fraction must be in (0, 1)")
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
    failed = {
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
        for row in existing if row.get("status") == "failed"
    }
    groups = []
    skipped = 0
    for problem, size, lhs_seed, method in itertools.product(
            args.problems, args.train_sizes, args.lhs_seeds, args.methods):
        pending = []
        for opt_seed in args.opt_seeds:
            key = _key(
                args.dataset_source,
                current_protocol_version(args.dataset_source),
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
            elif args.resume and key in failed and not args.retry_failed:
                skipped += 1
            else:
                pending.append(opt_seed)
        if pending:
            groups.append((problem, size, lhs_seed, method, tuple(pending)))
    return groups, skipped


def _safe_log_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


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


@contextlib.contextmanager
def _redirect_process_output(log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(log_handle.fileno(), 1)
            os.dup2(log_handle.fileno(), 2)
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
                yield log_handle
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)


def _execute(payload):
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
        validation_fraction,
    ) = payload
    log_name = (
        f"{_safe_log_component(problem)}_N{size}_lhs{lhs_seed}_"
        f"{_safe_log_component(method)}.log"
    )
    log_path = Path(output_dir) / "logs" / log_name
    with _redirect_process_output(log_path) as log_handle:
        print(
            f"\n=== source={dataset_source} | problem={problem} | N={size} | "
            f"offline_seed={lhs_seed} | "
            f"method={method} | opt={list(seeds)} ===",
            file=log_handle,
            flush=True,
        )
        try:
            return run_group(
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
                validation_fraction=validation_fraction,
            )
        except Exception:
            _release_worker_memory()
            raise


def _execute_sequence(payloads):
    rows = []
    for payload in payloads:
        method_rows, model_objects = _execute(payload)
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
        print(f"Pending: {pending} | skipped by resume: {skipped} | group executions: {len(groups)}")
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
        "validation_size": (
            None if args.dataset_source == "official_pool" else VALIDATION_SIZE
        ),
        "validation_source": (
            "within_selected_training_subset"
            if args.dataset_source == "official_pool"
            else "fixed_independent_lhs"
        ),
        "test_size": (
            "official_test_pool"
            if args.dataset_source == "official_pool"
            else TEST_SIZE
        ),
        "n_gen": args.n_gen,
        "pop_size": args.pop_size, "total_tasks": total,
        "validation_fraction": args.validation_fraction,
        "max_workers": args.max_workers,
        "tabpfn_max_workers": args.tabpfn_max_workers,
    })
    payloads = [
        (
            str(args.output_dir),
            *group,
            args.n_gen,
            args.pop_size,
            args.dataset_source,
            str(args.subset_cache_dir),
            tuple(args.train_sizes),
            args.validation_fraction,
        )
        for group in groups
    ]
    processed = int(skipped)
    executed_success = 0
    executed_failed = 0

    def record_progress(rows, label, write_results=True):
        nonlocal processed, executed_success, executed_failed
        if write_results:
            append_rows(args.output_dir, rows)
        success_count = sum(row.get("status") == "success" for row in rows)
        failed_count = len(rows) - success_count
        processed += len(rows)
        executed_success += success_count
        executed_failed += failed_count
        percentage = 100.0 * processed / total if total else 100.0
        print(
            f"[progress] {processed}/{total} ({percentage:.2f}%) | "
            f"success={success_count} | failed={failed_count} | "
            f"total_success={executed_success} | total_failed={executed_failed} | "
            f"skipped={skipped} | {label}",
            flush=True,
        )

    print(f"[progress] {processed}/{total} | skipped={skipped}", flush=True)
    # Execute complete method stages in the configured/CLI order. Regular methods
    # can use the full CPU allocation; only the TabPFN stage receives the smaller
    # API-concurrency cap.
    for method in args.methods:
        method_payloads = [payload for payload in payloads if payload[4] == method]
        if not method_payloads:
            continue
        family = METHOD_REGISTRY[method].family
        worker_limit = args.tabpfn_max_workers if family == "tabpfn" else args.max_workers
        method_workers = min(worker_limit, len(method_payloads))

        if method_workers == 1:
            for payload in method_payloads:
                rows, model_objects = _execute(payload)
                try:
                    label = (
                        f"{payload[1]} | N={payload[2]} | "
                        f"offline_seed={payload[3]} | {method}"
                    )
                    record_progress(rows, label)
                finally:
                    cleanup_model_storage(model_objects)
                    del model_objects
                    _release_worker_memory()
            continue

        with ProcessPoolExecutor(max_workers=method_workers) as executor:
            future_labels = {}
            for payload in method_payloads:
                future = executor.submit(_execute_sequence, (payload,))
                future_labels[future] = (
                    f"{payload[1]} | N={payload[2]} | "
                    f"offline_seed={payload[3]} | {method}"
                )
            for future in as_completed(future_labels):
                record_progress(future.result(), future_labels[future], write_results=False)
    summarize(args.output_dir, args.output_dir)
    print(f"[progress] complete | results={args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
