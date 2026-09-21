#!/usr/bin/env python3
"""Create paired normal/DR/EBU-DR comparison tables from primary results."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.method_registry import METHOD_REGISTRY  # noqa: E402
from experiments.sample_size_common import (  # noqa: E402
    configure_method_settings,
    current_protocol_version,
    load_config_file,
)


PAIR_KEYS = (
    "dataset_source",
    "problem",
    "training_size",
    "offline_seed",
    "opt_seed",
    "configured_n_gen",
    "configured_pop_size",
)
CELL_KEYS = (
    "dataset_source",
    "problem",
    "family",
    "training_size",
    "offline_seed",
    "configured_n_gen",
    "configured_pop_size",
)
MAXIMIZE_METRICS = {"HVreal"}


def _cliffs_delta(left, right):
    left = np.asarray(left, dtype=float).reshape(-1)
    right = np.asarray(right, dtype=float).reshape(-1)
    if not len(left) or not len(right):
        return np.nan
    comparison = left[:, None] - right[None, :]
    return float((np.sum(comparison > 0) - np.sum(comparison < 0)) / comparison.size)


def _one_sided_wilcoxon(improvements):
    improvements = np.asarray(improvements, dtype=float)
    improvements = improvements[np.isfinite(improvements)]
    if not len(improvements) or np.all(improvements == 0):
        return 1.0
    return float(
        wilcoxon(
            improvements,
            alternative="greater",
            zero_method="wilcox",
        ).pvalue
    )


def _load_results(results_dir):
    results_dir = Path(results_dir)
    manifest_path = results_dir / "experiment_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_path = Path(str(manifest.get("config") or ""))
        if config_path.is_file():
            configure_method_settings(load_config_file(config_path))
    patterns = (
        str(results_dir / "csv" / "exp*_results.csv"),
        str(results_dir / "exp*_results.csv"),
    )
    files = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    if not files:
        raise FileNotFoundError(f"No exp*_results.csv files under {results_dir}")
    frame = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    frame = frame.loc[frame["status"].astype(str).str.lower() == "success"].copy()
    frame["family"] = frame["method"].map(
        lambda name: METHOD_REGISTRY[name].family if name in METHOD_REGISTRY else None
    )
    frame["category"] = frame["method"].map(
        lambda name: METHOD_REGISTRY[name].category if name in METHOD_REGISTRY else None
    )
    frame = frame.loc[frame["category"].isin({"normal", "dr", "ebu_dr"})]
    expected_protocol = frame.apply(
        lambda row: current_protocol_version(row["dataset_source"], row["method"]),
        axis=1,
    )
    frame = frame.loc[frame["protocol_version"].astype(str) == expected_protocol]
    # Resume can leave historical rows in place. For an identical experiment
    # identity, the most recently appended successful row is authoritative.
    frame["_row_order"] = np.arange(len(frame))
    frame = frame.sort_values("_row_order").drop_duplicates(
        [*PAIR_KEYS, "family", "category"], keep="last"
    )
    return frame


def analyze(results, metric, controls):
    if metric not in results:
        raise ValueError(f"Result column is missing: {metric}")
    results = results.copy()
    results[metric] = pd.to_numeric(results[metric], errors="coerce")
    results = results.loc[np.isfinite(results[metric])]
    maximize = metric in MAXIMIZE_METRICS
    cell_rows = []
    paired_tables = []

    for control in controls:
        if control == "ctrl_duplicate":
            target_category = comparator_category = "normal"
        else:
            target_category = "ebu_dr"
            comparator_category = control
        target = results.loc[results["category"] == target_category]
        comparator = results.loc[results["category"] == comparator_category]
        paired = target.merge(
            comparator,
            on=[*PAIR_KEYS, "family"],
            suffixes=("_target", "_control"),
            validate="one_to_one",
        )
        if control == "ctrl_duplicate":
            paired[f"{metric}_control"] = paired[f"{metric}_target"]
        raw_difference = (
            paired[f"{metric}_target"] - paired[f"{metric}_control"]
        )
        paired["improvement"] = raw_difference if maximize else -raw_difference
        paired["control"] = control
        paired_tables.append(paired)

        for keys, group in paired.groupby(list(CELL_KEYS), dropna=False):
            row = dict(zip(CELL_KEYS, keys))
            target_values = group[f"{metric}_target"].to_numpy(float)
            control_values = group[f"{metric}_control"].to_numpy(float)
            delta = _cliffs_delta(target_values, control_values)
            row.update({
                "control": control,
                "metric": metric,
                "paired_runs": len(group),
                "target_mean": float(np.mean(target_values)),
                "control_mean": float(np.mean(control_values)),
                "mean_improvement": float(group["improvement"].mean()),
                "cliffs_delta": delta if maximize else -delta,
            })
            cell_rows.append(row)

    paired_all = pd.concat(paired_tables, ignore_index=True)
    aggregate_rows = []
    aggregate_keys = ("control", "family", "training_size")
    for keys, group in paired_all.groupby(list(aggregate_keys), dropna=False):
        row = dict(zip(aggregate_keys, keys))
        row.update({
            "metric": metric,
            "paired_runs": len(group),
            "mean_improvement": float(group["improvement"].mean()),
            "median_improvement": float(group["improvement"].median()),
            "one_sided_wilcoxon_p": _one_sided_wilcoxon(group["improvement"]),
        })
        aggregate_rows.append(row)
    return pd.DataFrame(cell_rows), pd.DataFrame(aggregate_rows), paired_all


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_primary_methods",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--metric", default="HVreal")
    parser.add_argument(
        "--controls",
        default="normal,dr,ctrl_duplicate",
        help="comma-separated controls: normal, dr, ctrl_duplicate",
    )
    args = parser.parse_args(argv)
    controls = tuple(value.strip() for value in args.controls.split(",") if value.strip())
    unknown = sorted(set(controls) - {"normal", "dr", "ctrl_duplicate"})
    if unknown:
        parser.error(f"unknown controls: {unknown}")
    output_dir = args.output_dir or args.results_dir / "dr_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    cell, aggregate, paired = analyze(
        _load_results(args.results_dir), args.metric, controls
    )
    cell.to_csv(output_dir / "cell_cliffs_delta.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate_wilcoxon.csv", index=False)
    paired.to_csv(output_dir / "paired_runs.csv", index=False)
    print(
        f"Wrote {len(cell)} cell comparisons and {len(aggregate)} aggregate "
        f"tests to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
