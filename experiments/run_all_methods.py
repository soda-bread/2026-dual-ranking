#!/usr/bin/env python3
"""Dispatch all repository methods through one top-level command."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from experiments.DL_MOBO_baseline import (  # noqa: E402
    BASELINE_NAMES as DL_MOBO_METHODS,
)
from experiments.generative_baseline import (  # noqa: E402
    BASELINE_NAMES as GENERATIVE_METHODS,
)
from experiments.project_environment import (  # noqa: E402
    environment_issues,
    format_environment_report,
    project_python,
)
from experiments.method_registry import (  # noqa: E402
    DEFAULT_DISABLED_METHODS,
    METHOD_REGISTRY,
)


HIDDEN_METHODS = tuple(DEFAULT_DISABLED_METHODS)
MAIN_METHODS = tuple(
    method for method in METHOD_REGISTRY if method not in HIDDEN_METHODS
)
METHOD_GROUPS = {
    "main": MAIN_METHODS,
    "dl_mobo": tuple(DL_MOBO_METHODS),
    "generative": tuple(GENERATIVE_METHODS),
}
ALL_METHODS = tuple(
    method for methods in METHOD_GROUPS.values() for method in methods
)
DEFAULT_METHODS = ALL_METHODS
PRIMARY_EXPERIMENT_METHODS = tuple(
    method
    for method in MAIN_METHODS
    if METHOD_REGISTRY[method].family in {"gpr_rbf", "gpr_matern", "qr", "bnn"}
)
MAIN_BASELINE_METHODS = (
    "TGPR-MO",
    "DDMOEA-GAN",
    "Prob-RVEA",
    "Prob-MOEA/D",
)
BASELINE_EXPERIMENT_METHODS = (
    *MAIN_BASELINE_METHODS,
    *DL_MOBO_METHODS,
    *GENERATIVE_METHODS,
)


def _csv(value, cast=str):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def _append_option(command, option, value):
    if value is not None:
        command.extend((option, str(value)))


def fixed_group_main(group_name, methods, argv=None, default_output_dir=None):
    """Run one fixed experiment family while allowing an in-family subset."""

    argv = list(sys.argv[1:] if argv is None else argv)
    if "--list-methods" in argv:
        print(f"{group_name} ({len(methods)}):")
        for method in methods:
            print(f"  {method}")
        return 0

    requested = None
    for index, argument in enumerate(argv):
        if argument == "--methods":
            if index + 1 >= len(argv):
                raise SystemExit("--methods requires a comma-separated value")
            requested = _csv(argv[index + 1])
            break
        if argument.startswith("--methods="):
            requested = _csv(argument.split("=", 1)[1])
            break

    if requested is not None:
        outside_group = sorted(set(requested) - set(methods))
        if outside_group:
            raise SystemExit(
                f"{group_name} entry does not include methods: {outside_group}"
            )
    forwarded = list(argv)
    if requested is None:
        forwarded[0:0] = ["--methods", ",".join(methods)]
    has_output_dir = any(
        argument == "--output-dir" or argument.startswith("--output-dir=")
        for argument in forwarded
    )
    if default_output_dir is not None and not has_output_dir:
        forwarded[2:2] = ["--output-dir", str(default_output_dir)]
    return main(forwarded)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        help=(
            "comma-separated methods from exactly one experiment group; prefer "
            "run_primary_methods.py or run_baselines.py"
        ),
    )
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument(
        "--check-environment",
        action="store_true",
        help="check the repository-local .venv and selected method dependencies",
    )
    parser.add_argument("--problems", help="comma-separated canonical problem names")
    parser.add_argument(
        "--training-sizes",
        "--train-sizes",
        dest="training_sizes",
        help="comma-separated offline training sizes",
    )
    parser.add_argument(
        "--offline-seeds",
        "--lhs-seeds",
        dest="offline_seeds",
        help="comma-separated offline subset/model seeds",
    )
    parser.add_argument(
        "--optimization-seeds",
        "--opt-seeds",
        dest="optimization_seeds",
        help="comma-separated optimizer/generation seeds",
    )
    parser.add_argument(
        "--dataset-source",
        choices=("official_pool", "lhs"),
        default="official_pool",
        help="shared data protocol; DL and generative methods require official_pool",
    )
    parser.add_argument("--main-config", type=Path)
    parser.add_argument(
        "--dl-mobo-config",
        "--dl-config",
        dest="dl_mobo_config",
        type=Path,
        default=HERE / "DL_MOBO_baseline" / "config.yaml",
    )
    parser.add_argument(
        "--generative-config",
        type=Path,
        default=HERE / "generative_baseline" / "config.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument(
        "--subset-cache-dir", type=Path, default=HERE / "data_subsets"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--n-gen", type=int)
    parser.add_argument("--pop-size", type=int)
    parser.add_argument("--epochs", type=int, help="DL neural training epochs")
    parser.add_argument(
        "--proxy-epochs", type=int, help="generative ParetoFlow proxy epochs"
    )
    parser.add_argument("--output-size", type=int, help="generative output size")
    parser.add_argument("--max-workers", type=int, help="main-runner worker count")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help=(
            "deprecated compatibility flag; failed rows are always retried "
            "when --resume is enabled"
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use the generative runner's reduced smoke configuration",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)

    requested = _csv(args.methods)
    if requested is None and not args.list_methods:
        parser.error(
            "no combined default run is available; use run_primary_methods.py "
            "or run_baselines.py"
        )
    requested = requested or []
    args.methods = list(dict.fromkeys(requested))
    unknown = sorted(set(args.methods) - set(ALL_METHODS))
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    if not args.methods and not args.list_methods:
        parser.error("at least one method is required")

    selected = set(args.methods)
    uses_primary = bool(selected.intersection(PRIMARY_EXPERIMENT_METHODS))
    uses_baselines = bool(selected.intersection(BASELINE_EXPERIMENT_METHODS))
    if uses_primary and uses_baselines:
        parser.error(
            "primary methods and baselines must run separately; use "
            "run_primary_methods.py and run_baselines.py"
        )

    selected_non_main = set(args.methods) & set(
        DL_MOBO_METHODS + GENERATIVE_METHODS
    )
    if args.dataset_source != "official_pool" and selected_non_main:
        parser.error(
            "DL and generative methods require --dataset-source official_pool; "
            f"incompatible methods: {sorted(selected_non_main)}"
        )

    if args.main_config is None:
        filename = (
            "config_official_pool.yaml"
            if args.dataset_source == "official_pool"
            else "config.yaml"
        )
        args.main_config = HERE / filename

    for label, path in (
        ("main", args.main_config),
        ("DL/MOBO", args.dl_mobo_config),
        ("generative", args.generative_config),
    ):
        if not path.is_file():
            parser.error(f"{label} configuration does not exist: {path}")

    args.output_dir = args.output_dir.resolve()
    args.subset_cache_dir = args.subset_cache_dir.resolve()
    return args


def selected_groups(methods):
    selected = set(methods)
    return {
        group: [method for method in group_methods if method in selected]
        for group, group_methods in METHOD_GROUPS.items()
        if selected.intersection(group_methods)
    }


def build_commands(args):
    groups = selected_groups(args.methods)
    commands = []
    resume_flag = "--resume" if args.resume else "--no-resume"
    # Dry runs intentionally remain usable before dependencies are installed.
    # Every real child process is pinned to this repository's own .venv so a
    # sibling project's environment can never be used accidentally.
    local_python = project_python(require_exists=False)
    python = str(local_python) if local_python.is_file() else sys.executable

    if "main" in groups:
        command = [
            python,
            str(HERE / "run_all.py"),
            "--config",
            str(args.main_config.resolve()),
            "--methods",
            ",".join(groups["main"]),
            "--dataset-source",
            args.dataset_source,
            "--output-dir",
            str(args.output_dir),
            "--subset-cache-dir",
            str(args.subset_cache_dir),
            resume_flag,
        ]
        _append_option(command, "--problems", args.problems)
        _append_option(command, "--train-sizes", args.training_sizes)
        _append_option(command, "--offline-seeds", args.offline_seeds)
        _append_option(command, "--opt-seeds", args.optimization_seeds)
        _append_option(command, "--n-gen", args.n_gen)
        _append_option(command, "--pop-size", args.pop_size)
        _append_option(command, "--max-workers", args.max_workers)
        if args.retry_failed:
            command.append("--retry-failed")
        if args.dry_run:
            command.append("--dry-run")
        commands.append(("main", command))

    if "dl_mobo" in groups:
        command = [
            python,
            str(HERE / "DL_MOBO_baseline" / "run.py"),
            "--config",
            str(args.dl_mobo_config.resolve()),
            "--methods",
            ",".join(groups["dl_mobo"]),
            "--device",
            args.device,
            "--output-dir",
            str(args.output_dir),
            "--subset-cache-dir",
            str(args.subset_cache_dir),
            resume_flag,
        ]
        _append_option(command, "--problems", args.problems)
        _append_option(command, "--training-sizes", args.training_sizes)
        _append_option(command, "--offline-seeds", args.offline_seeds)
        _append_option(command, "--optimization-seeds", args.optimization_seeds)
        _append_option(command, "--n-gen", args.n_gen)
        _append_option(command, "--pop-size", args.pop_size)
        _append_option(command, "--epochs", args.epochs)
        _append_option(command, "--max-workers", args.max_workers)
        if args.dry_run:
            command.append("--dry-run")
        commands.append(("dl_mobo", command))

    if "generative" in groups:
        command = [
            python,
            str(HERE / "generative_baseline" / "run.py"),
            "--config",
            str(args.generative_config.resolve()),
            "--methods",
            ",".join(groups["generative"]),
            "--device",
            args.device,
            "--output-dir",
            str(args.output_dir),
            "--subset-cache-dir",
            str(args.subset_cache_dir),
            resume_flag,
        ]
        _append_option(command, "--problems", args.problems)
        _append_option(command, "--training-sizes", args.training_sizes)
        _append_option(command, "--offline-seeds", args.offline_seeds)
        _append_option(command, "--optimization-seeds", args.optimization_seeds)
        _append_option(command, "--output-size", args.output_size)
        _append_option(command, "--proxy-epochs", args.proxy_epochs)
        _append_option(command, "--max-workers", args.max_workers)
        if args.smoke:
            command.append("--smoke")
        if args.dry_run:
            command.append("--dry-run")
        commands.append(("generative", command))

    return commands


def main(argv=None):
    args = parse_args(argv)
    if args.list_methods:
        for group, methods in (
            ("primary methods", PRIMARY_EXPERIMENT_METHODS),
            ("baselines", BASELINE_EXPERIMENT_METHODS),
        ):
            print(f"{group} ({len(methods)}):")
            for method in methods:
                print(f"  {method}")
        return 0

    if args.check_environment:
        print(format_environment_report(args.methods, args.problems))
        return 1 if environment_issues(args.methods, args.problems) else 0

    if not args.dry_run:
        issues = environment_issues(args.methods, args.problems)
        if issues:
            print(
                format_environment_report(args.methods, args.problems),
                file=sys.stderr,
            )
            return 2

    commands = build_commands(args)
    print(
        f"Unified dispatch: {len(args.methods)} methods across "
        f"{len(commands)} runner(s)",
        flush=True,
    )
    failed = []
    for group, command in commands:
        print(f"\n[{group}] {shlex.join(command)}", flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failed.append((group, completed.returncode))

    if failed:
        details = ", ".join(f"{group}=exit {code}" for group, code in failed)
        print(f"Unified dispatch completed with failures: {details}", file=sys.stderr)
        return 1
    print("Unified dispatch completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
