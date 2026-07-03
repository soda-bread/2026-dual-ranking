from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fcntl
except ImportError:
    fcntl = None


def _mean_std(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.any(np.isfinite(values)):
        return np.nan, np.nan
    return float(np.nanmean(values)), float(np.nanstd(values))


def _summary_rows(method_name, optimizer_names, problem_names, all_results):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for optimizer_name in optimizer_names:
        for problem_name in problem_names:
            results = all_results[problem_name][optimizer_name]
            mse_mean, mse_std = _mean_std(
                results.get("mse_test_list", results.get("sur_real_mse_list", []))
            )
            mse_sur_real_mean, mse_sur_real_std = _mean_std(
                results.get("mse_sur_real_list", results.get("sur_real_mse_list", []))
            )
            hv_sur_mean, hv_sur_std = _mean_std(results.get("hv_surrogate_list", []))
            hv_real_mean, hv_real_std = _mean_std(results.get("hv_real_list", []))
            rows.append(
                {
                    "timestamp": timestamp,
                    "method": method_name,
                    "optimizer": optimizer_name,
                    "problem": problem_name,
                    "MSE_test_mean": mse_mean,
                    "MSE_test_std": mse_std,
                    "MSE_sur_real_mean": mse_sur_real_mean,
                    "MSE_sur_real_std": mse_sur_real_std,
                    "HV_sur_mean": hv_sur_mean,
                    "HV_sur_std": hv_sur_std,
                    "HV_real_mean": hv_real_mean,
                    "HV_real_std": hv_real_std,
                }
            )
    return rows


def _seed_detail_rows(method_name, optimizer_names, problem_names, all_results):
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = []
    for optimizer_name in optimizer_names:
        for problem_name in problem_names:
            results = all_results[problem_name][optimizer_name]
            for detail in results.get("run_details", []):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "method": method_name,
                        "optimizer": optimizer_name,
                        "problem": problem_name,
                        "seed": detail.get("seed"),
                        "time": detail.get("time"),
                        "MSE_test": detail.get("mse_test", detail.get("offline_test_mse")),
                        "MSE_sur_real": detail.get(
                            "mse_sur_real",
                            detail.get("sur_real_mse"),
                        ),
                        "HV_sur": detail.get("hv_surrogate"),
                        "HV_real": detail.get("hv_real"),
                        "HV_bounds_check": detail.get("hv_bounds_check"),
                        "solution_count": (
                            detail.get("solution_count")
                            if detail.get("solution_count") is not None
                            else (
                                None
                                if detail.get("solution") is None
                                else int(np.asarray(detail.get("solution")).shape[0])
                            )
                        ),
                    }
                )
    return rows


def _fmt_value(value):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    return f"{value:.3e}"


def _raw_value(value):
    if value is None:
        return "None"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _format_result_table(table):
    table = table.copy()
    metadata_columns = {"timestamp", "method", "optimizer", "problem"}
    for column in table.columns:
        if column not in metadata_columns:
            table[column] = table[column].map(_fmt_value)
    return table


def _safe_filename(name):
    return (
        str(name)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
    )


def append_seed_txt(method_name, optimizer_names, problem_names, all_results, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_filename(method_name)}.txt"
    rows = _seed_detail_rows(
        method_name=method_name,
        optimizer_names=optimizer_names,
        problem_names=problem_names,
        all_results=all_results,
    )
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n# {datetime.now().astimezone().isoformat(timespec='seconds')} | {method_name}\n")
        if not rows:
            handle.write("No seed records.\n")
        for row in rows:
            handle.write(
                "method={method} | optimizer={optimizer} | problem={problem} | "
                "seed={seed} | time={time} | MSE_test={MSE_test} | "
                "MSE_sur_real={MSE_sur_real} | HV_sur={HV_sur} | "
                "HV_real={HV_real} | HV_bounds_check={HV_bounds_check} | "
                "solution_count={solution_count}\n".format(
                    method=row["method"],
                    optimizer=row["optimizer"],
                    problem=row["problem"],
                    seed=row["seed"],
                    time=_raw_value(row["time"]),
                    MSE_test=_raw_value(row["MSE_test"]),
                    MSE_sur_real=_raw_value(row["MSE_sur_real"]),
                    HV_sur=_raw_value(row["HV_sur"]),
                    HV_real=_raw_value(row["HV_real"]),
                    HV_bounds_check=row["HV_bounds_check"],
                    solution_count=row["solution_count"],
                )
            )
    print(f"Appended seed detail TXT to: {output_path}")
    return output_path


def append_single_seed_txt(method_name, optimizer_name, problem_name, detail, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{_safe_filename(method_name)}.txt"
    row = {
        "method": method_name,
        "optimizer": optimizer_name,
        "problem": problem_name,
        "seed": detail.get("seed"),
        "time": detail.get("time"),
        "MSE_test": detail.get("mse_test", detail.get("offline_test_mse")),
        "MSE_sur_real": detail.get("mse_sur_real", detail.get("sur_real_mse")),
        "HV_sur": detail.get("hv_surrogate"),
        "HV_real": detail.get("hv_real"),
        "HV_bounds_check": detail.get("hv_bounds_check"),
        "solution_count": detail.get("solution_count"),
        "no_feasible_solution": detail.get("no_feasible_solution", False),
        "no_feasible_reason": detail.get("no_feasible_reason"),
    }
    line = (
        "{timestamp} | method={method} | optimizer={optimizer} | problem={problem} | "
        "seed={seed} | time={time} | MSE_test={MSE_test} | "
        "MSE_sur_real={MSE_sur_real} | HV_sur={HV_sur} | "
        "HV_real={HV_real} | HV_bounds_check={HV_bounds_check} | "
        "solution_count={solution_count} | no_feasible_solution={no_feasible_solution} | "
        "no_feasible_reason={no_feasible_reason}\n"
    ).format(
        timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
        method=row["method"],
        optimizer=row["optimizer"],
        problem=row["problem"],
        seed=row["seed"],
        time=_raw_value(row["time"]),
        MSE_test=_raw_value(row["MSE_test"]),
        MSE_sur_real=_raw_value(row["MSE_sur_real"]),
        HV_sur=_raw_value(row["HV_sur"]),
        HV_real=_raw_value(row["HV_real"]),
        HV_bounds_check=row["HV_bounds_check"],
        solution_count=row["solution_count"],
        no_feasible_solution=row["no_feasible_solution"],
        no_feasible_reason=row["no_feasible_reason"],
    )
    with output_path.open("a", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(line)
            handle.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return output_path


def append_result_csv(
    method_name,
    optimizer_names,
    problem_names,
    all_results,
    result_csv_path,
    write_seed_txt=True,
):
    result_csv_path = Path(result_csv_path)
    table = pd.DataFrame(
        _summary_rows(
            method_name=method_name,
            optimizer_names=optimizer_names,
            problem_names=problem_names,
            all_results=all_results,
        )
    )
    formatted_table = _format_result_table(table)
    result_csv_path.parent.mkdir(parents=True, exist_ok=True)
    formatted_table.to_csv(
        result_csv_path,
        mode="a",
        header=not result_csv_path.exists(),
        index=False,
    )
    if write_seed_txt:
        append_seed_txt(
            method_name=method_name,
            optimizer_names=optimizer_names,
            problem_names=problem_names,
            all_results=all_results,
            output_dir=result_csv_path.parent,
        )
    print("Final summary")
    print(formatted_table.to_string(index=False))
    print(f"Appended final result CSV to: {result_csv_path}")
    return formatted_table


append_bluelear_result_csv = append_result_csv
append_bluelear_seed_txt = append_seed_txt
append_bluelear_single_seed_txt = append_single_seed_txt
