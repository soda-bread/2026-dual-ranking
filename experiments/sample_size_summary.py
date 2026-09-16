#!/usr/bin/env python3
"""Create hierarchical summaries for sample-size ablation results."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("MSEpre", "MSEsur_real", "HVreal", "IGDplus")


def _expected_protocol_version(method, dataset_source):
    """Return the repository's current protocol for a known method."""

    method = str(method)
    dataset_source = str(dataset_source)
    from experiments.method_registry import METHOD_REGISTRY

    if method in METHOD_REGISTRY:
        from experiments.sample_size_common import current_protocol_version

        return current_protocol_version(dataset_source, method)

    from experiments.DL_MOBO_baseline import (
        BASELINE_NAMES as DL_BASELINES,
        PROTOCOL_VERSION as DL_PROTOCOL_VERSION,
    )

    if method in DL_BASELINES:
        return DL_PROTOCOL_VERSION

    from experiments.generative_baseline import (
        BASELINE_NAMES as GENERATIVE_BASELINES,
        PROTOCOL_VERSION as GENERATIVE_PROTOCOL_VERSION,
    )

    if method in GENERATIVE_BASELINES:
        return GENERATIVE_PROTOCOL_VERSION
    return None


def bootstrap_ci(values, seed=2026, samples=10_000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def _result_paths(input_dirs):
    if isinstance(input_dirs, (str, Path)):
        input_dirs = [Path(input_dirs)]
    paths = []
    for input_dir in (Path(value) for value in input_dirs):
        paths.extend(sorted((input_dir / "csv").glob("exp*_results.csv")))
        paths.extend(sorted(input_dir.glob("exp*_results.csv")))
        for filename in ("dl_baselines.csv", "generative_baselines.csv"):
            path = input_dir / filename
            if path.is_file():
                paths.append(path)
    return list(dict.fromkeys(path.resolve() for path in paths))


def _normalized_frame(path):
    frame = pd.read_csv(path)
    frame["result_source"] = path.name
    if "lhs_seed" not in frame.columns:
        frame["lhs_seed"] = frame.get("offline_seed")
    else:
        frame["lhs_seed"] = frame["lhs_seed"].fillna(frame.get("offline_seed"))
    if "configured_pop_size" not in frame.columns:
        frame["configured_pop_size"] = frame.get("configured_output_size", 100)
    if "configured_n_gen" not in frame.columns:
        # Generative methods use a different internal sampler, but participate
        # in the shared 100-solution/10,000-evaluation comparison protocol.
        frame["configured_n_gen"] = 100
    return frame


def summarize(input_dir: Path | list[Path], output_dir: Path, *, write_plots=True):
    paths = _result_paths(input_dir)
    if not paths:
        raise FileNotFoundError(f"No experiment result CSV files under {input_dir}")
    raw = pd.concat((_normalized_frame(path) for path in paths), ignore_index=True)
    if "dataset_source" not in raw.columns:
        raw["dataset_source"] = "lhs"
    else:
        raw["dataset_source"] = raw["dataset_source"].fillna("lhs")
        raw.loc[raw["dataset_source"].astype(str).str.strip() == "", "dataset_source"] = "lhs"
    if "protocol_version" not in raw.columns:
        raw["protocol_version"] = "unversioned"
    else:
        raw["protocol_version"] = raw["protocol_version"].fillna("unversioned")
        raw.loc[
            raw["protocol_version"].astype(str).str.strip() == "",
            "protocol_version",
        ] = "unversioned"
    if "configuration_hash" not in raw.columns:
        raw["configuration_hash"] = ""
    else:
        raw["configuration_hash"] = raw["configuration_hash"].fillna("")

    raw["expected_protocol_version"] = [
        _expected_protocol_version(method, source)
        for method, source in zip(raw["method"], raw["dataset_source"])
    ]
    protocol_inventory = (
        raw.assign(
            result_identity=raw["protocol_version"].astype(str)
            + raw["configuration_hash"].astype(str).map(
                lambda value: f"|cfg={value}" if value else ""
            )
        )
        .groupby(["dataset_source", "method"], as_index=False)
        .agg(
            protocol_versions=(
                "protocol_version",
                lambda values: " | ".join(sorted(set(map(str, values)))),
            ),
            result_identities=(
                "result_identity",
                lambda values: " | ".join(sorted(set(map(str, values)))),
            ),
            identity_count=("result_identity", "nunique"),
        )
    )
    mixed = protocol_inventory[protocol_inventory["identity_count"] > 1]
    if not mixed.empty:
        details = ", ".join(
            f"{row.method} ({row.result_identities})"
            for row in mixed.itertuples()
        )
        warnings.warn(
            "Multiple protocol/configuration identities were found; stale rows "
            f"will not be mixed silently: {details}",
            RuntimeWarning,
        )

    known_protocol = raw["expected_protocol_version"].notna()
    stale_protocol = known_protocol & (
        raw["protocol_version"].astype(str)
        != raw["expected_protocol_version"].astype(str)
    )
    stale = raw[stale_protocol].copy()
    raw = raw[~stale_protocol].copy()
    if not stale.empty:
        warnings.warn(
            f"Excluded {len(stale)} rows whose protocol_version is not current. "
            "See stale_protocol_rows.csv.",
            RuntimeWarning,
        )
    for column in ("configured_n_gen", "configured_pop_size"):
        if column not in raw.columns:
            raw[column] = 100
        else:
            raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(100)
    run_keys = [
        "dataset_source", "configured_n_gen", "configured_pop_size",
        "problem", "method", "training_size", "lhs_seed", "opt_seed",
    ]
    # Result files are append-only. The last row is the active configuration
    # for a method/run key, so stale protocol/configuration rows are not mixed.
    latest = raw.drop_duplicates(run_keys, keep="last")
    successful = latest[latest["status"] == "success"].copy()
    failed = latest[latest["status"] != "success"].copy()
    for metric in METRICS:
        successful[metric] = pd.to_numeric(successful[metric], errors="coerce")

    # Stage 1: optimizer variability within each LHS dataset.
    lhs_keys = [
        "dataset_source", "configured_n_gen", "configured_pop_size", "problem",
        "method", "training_size", "lhs_seed",
    ]
    lhs = successful.groupby(lhs_keys, as_index=False).agg(
        protocol_version=("protocol_version", "last"),
        result_source=("result_source", "last"),
        optimization_runs=("opt_seed", "nunique"),
        MSEpre=("MSEpre", "first"),
        MSEsur_real_opt_mean=("MSEsur_real", "mean"),
        MSEsur_real_opt_std=("MSEsur_real", "std"),
        HVreal_opt_mean=("HVreal", "mean"),
        HVreal_opt_std=("HVreal", "std"),
        IGDplus_opt_mean=("IGDplus", "mean"),
        IGDplus_opt_std=("IGDplus", "std"),
    )

    # Stage 2: LHS-level variability. MSEpre is already one observation per LHS;
    # optimization-dependent metrics use the per-LHS optimizer mean.
    records = []
    for keys, group in lhs.groupby(
        [
            "dataset_source", "configured_n_gen", "configured_pop_size",
            "problem", "method", "training_size",
        ]
    ):
        for metric, column in (
            ("MSEpre", "MSEpre"),
            ("MSEsur_real", "MSEsur_real_opt_mean"),
            ("HVreal", "HVreal_opt_mean"),
            ("IGDplus", "IGDplus_opt_mean"),
        ):
            values = group[column].dropna().to_numpy(float)
            low, high = bootstrap_ci(values)
            records.append({
                "dataset_source": keys[0],
                "configured_n_gen": keys[1], "configured_pop_size": keys[2],
                "problem": keys[3],
                "method": keys[4], "training_size": keys[5],
                "protocol_version": group["protocol_version"].iloc[-1],
                "result_source": group["result_source"].iloc[-1],
                "metric": metric, "lhs_count": len(values),
                "overall_mean": np.mean(values) if len(values) else np.nan,
                "std": np.std(values, ddof=1) if len(values) > 1 else 0.0 if len(values) else np.nan,
                "median": np.median(values) if len(values) else np.nan,
                "q25": np.quantile(values, 0.25) if len(values) else np.nan,
                "q75": np.quantile(values, 0.75) if len(values) else np.nan,
                "bootstrap_ci95_low": low, "bootstrap_ci95_high": high,
            })
    problem_summary = pd.DataFrame(records)

    ranks = []
    for (source, n_gen, pop_size, size, metric), group in problem_summary.groupby(
        [
            "dataset_source", "configured_n_gen", "configured_pop_size",
            "training_size", "metric",
        ]
    ):
        ascending = metric != "HVreal"
        ranked = group.copy()
        ranked["rank"] = ranked.groupby("problem")["overall_mean"].rank(
            ascending=ascending, method="average")
        average = ranked.groupby("method", as_index=False)["rank"].mean()
        average.insert(0, "metric", metric)
        average.insert(0, "training_size", size)
        average.insert(0, "configured_pop_size", pop_size)
        average.insert(0, "configured_n_gen", n_gen)
        average.insert(0, "dataset_source", source)
        ranks.append(average.rename(columns={"rank": "average_rank"}))
    average_ranks = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir = output_dir / "csv"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    protocol_inventory.to_csv(
        csv_output_dir / "protocol_inventory.csv", index=False
    )
    stale.to_csv(csv_output_dir / "stale_protocol_rows.csv", index=False)
    lhs.to_csv(csv_output_dir / "lhs_level_summary.csv", index=False)
    problem_summary.to_csv(csv_output_dir / "sample_size_summary.csv", index=False)
    value_columns = ["lhs_count", "overall_mean", "std", "median", "q25", "q75",
                     "bootstrap_ci95_low", "bootstrap_ci95_high"]
    method_problem = problem_summary.pivot(
        index=[
            "dataset_source", "configured_n_gen", "configured_pop_size",
            "problem", "method", "training_size",
        ], columns="metric",
        values=value_columns).reset_index()
    method_problem.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
        for column in method_problem.columns
    ]
    average_ranks.to_csv(csv_output_dir / "average_rank_by_training_size.csv", index=False)
    method_problem.to_csv(csv_output_dir / "method_problem_summary.csv", index=False)
    failed.to_csv(csv_output_dir / "failed_runs.csv", index=False)

    if write_plots and not average_ranks.empty:
        import matplotlib.pyplot as plt
        for (source, n_gen, pop_size, metric), source_metric in average_ranks.groupby(
            [
                "dataset_source", "configured_n_gen", "configured_pop_size",
                "metric",
            ]
        ):
            figure, axis = plt.subplots(figsize=(10, 6))
            for method, group in source_metric.groupby("method"):
                group = group.sort_values("training_size")
                axis.plot(group["training_size"], group["average_rank"], marker="o", label=method)
            axis.set_xscale("log")
            axis.set_xlabel("Training size")
            axis.set_ylabel("Average rank (lower is better)")
            axis.set_title(
                f"Average rank by training size: {source} | "
                f"G={n_gen} | P={pop_size} | {metric}"
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            figure.tight_layout()
            figure.savefig(
                output_dir
                / f"average_rank_{source}_G{n_gen}_P{pop_size}_{metric}.png",
                dpi=180,
            )
            plt.close(figure)
    return lhs, problem_summary, average_ranks, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        help="result directory; repeat to merge primary and baseline outputs",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    default_root = Path(__file__).resolve().parent
    inputs = args.input_dir or [
        default_root / "results_primary_methods",
        default_root / "results_baselines",
    ]
    output = args.output_dir or default_root / "results_summary"
    lhs, problem, ranks, failed = summarize(inputs, output)
    print(f"Wrote {len(lhs)} LHS summaries, {len(problem)} method/problem summaries, "
          f"{len(ranks)} average ranks, and {len(failed)} failed runs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
