#!/usr/bin/env python3
"""Create hierarchical summaries for sample-size ablation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ("MSEpre", "MSEsur_real", "HVreal", "IGDplus")


def bootstrap_ci(values, seed=2026, samples=10_000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (samples, len(values)), replace=True), axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(input_dir: Path, output_dir: Path):
    paths = sorted((input_dir / "csv").glob("exp*_results.csv"))
    paths += sorted(input_dir.glob("exp*_results.csv"))
    if not paths:
        raise FileNotFoundError(f"No exp*_results.csv files under {input_dir}")
    raw = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
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
    for column in ("configured_n_gen", "configured_pop_size"):
        if column not in raw.columns:
            raw[column] = 100
        else:
            raw[column] = pd.to_numeric(raw[column], errors="coerce").fillna(100)
    run_keys = [
        "dataset_source", "protocol_version", "configured_n_gen",
        "configured_pop_size", "problem", "method",
        "training_size", "lhs_seed", "opt_seed",
    ]
    successful = (raw[raw["status"] == "success"].copy()
                  .drop_duplicates(run_keys, keep="last"))
    successful_keys = set(successful[run_keys].itertuples(index=False, name=None))
    failed = raw[raw["status"] != "success"].copy()
    failed_keys = failed[run_keys].apply(tuple, axis=1)
    failed = failed[~failed_keys.isin(successful_keys)].copy()
    for metric in METRICS:
        successful[metric] = pd.to_numeric(successful[metric], errors="coerce")

    # Stage 1: optimizer variability within each LHS dataset.
    lhs_keys = [
        "dataset_source", "protocol_version", "configured_n_gen",
        "configured_pop_size", "problem", "method",
        "training_size", "lhs_seed",
    ]
    lhs = successful.groupby(lhs_keys, as_index=False).agg(
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
            "dataset_source", "protocol_version", "configured_n_gen",
            "configured_pop_size", "problem", "method", "training_size",
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
                "dataset_source": keys[0], "protocol_version": keys[1],
                "configured_n_gen": keys[2], "configured_pop_size": keys[3],
                "problem": keys[4],
                "method": keys[5], "training_size": keys[6],
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
    for (source, protocol, n_gen, pop_size, size, metric), group in problem_summary.groupby(
        [
            "dataset_source", "protocol_version", "configured_n_gen",
            "configured_pop_size", "training_size", "metric",
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
        average.insert(0, "protocol_version", protocol)
        average.insert(0, "dataset_source", source)
        ranks.append(average.rename(columns={"rank": "average_rank"}))
    average_ranks = pd.concat(ranks, ignore_index=True) if ranks else pd.DataFrame()

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_output_dir = output_dir / "csv"
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    lhs.to_csv(csv_output_dir / "lhs_level_summary.csv", index=False)
    problem_summary.to_csv(csv_output_dir / "sample_size_summary.csv", index=False)
    value_columns = ["lhs_count", "overall_mean", "std", "median", "q25", "q75",
                     "bootstrap_ci95_low", "bootstrap_ci95_high"]
    method_problem = problem_summary.pivot(
        index=[
            "dataset_source", "protocol_version", "configured_n_gen",
            "configured_pop_size", "problem", "method", "training_size",
        ], columns="metric",
        values=value_columns).reset_index()
    method_problem.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
        for column in method_problem.columns
    ]
    average_ranks.to_csv(csv_output_dir / "average_rank_by_training_size.csv", index=False)
    method_problem.to_csv(csv_output_dir / "method_problem_summary.csv", index=False)
    failed.to_csv(csv_output_dir / "failed_runs.csv", index=False)

    if not average_ranks.empty:
        import matplotlib.pyplot as plt
        for (source, protocol, n_gen, pop_size, metric), source_metric in average_ranks.groupby(
            [
                "dataset_source", "protocol_version", "configured_n_gen",
                "configured_pop_size", "metric",
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
                f"Average rank by training size: {source} | {protocol} | "
                f"G={n_gen} | P={pop_size} | {metric}"
            )
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
            figure.tight_layout()
            figure.savefig(
                output_dir
                / f"average_rank_{source}_{protocol}_G{n_gen}_P{pop_size}_{metric}.png",
                dpi=180,
            )
            plt.close(figure)
    return lhs, problem_summary, average_ranks, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output = args.output_dir or args.input_dir
    lhs, problem, ranks, failed = summarize(args.input_dir, output)
    print(f"Wrote {len(lhs)} LHS summaries, {len(problem)} method/problem summaries, "
          f"{len(ranks)} average ranks, and {len(failed)} failed runs to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
