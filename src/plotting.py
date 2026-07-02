import numpy as np
from datetime import datetime
from pathlib import Path


EXP1_METRICS_SUMMARY_FILENAME = "result_exp1_metrics_summary.txt"
EXP1_METRICS_SUMMARY_CSV_FILENAME = "result_exp1_metrics_summary.csv"
EXP1_REQUESTED_METRICS = ("MSE_test", "MSE_sur_real", "HV_sur", "HV_real")


def _format_summary_float(value, digits=3):
    if value is None or not np.isfinite(value):
        return "nan"
    value = float(value)
    if value == 0.0:
        return "0"
    return f"{value:.{int(digits)}e}"


def _format_summary_scientific(value, digits=3):
    if value is None or not np.isfinite(value):
        return "nan"
    value = float(value)
    if value == 0.0:
        return "0"
    return f"{value:.{int(digits)}e}"


def _summary_values(values):
    values = np.asarray(values, dtype=float)
    return values.reshape(-1)


def _nanmean_or_nan(values):
    values = _summary_values(values)
    if values.size == 0:
        return np.nan
    return float(np.nanmean(values))


def build_exp1_metrics_summary_table(
    method_name,
    optimizer_names,
    problem_names,
    all_results,
    problem_contexts,
):
    import pandas as pd

    summary_columns = [("Method", "")]
    for problem_name in problem_names:
        summary_columns.extend(
            (problem_name, metric) for metric in EXP1_REQUESTED_METRICS
        )

    summary_rows = []
    for optimizer_name in optimizer_names:
        row = {("Method", ""): f"{method_name}+{optimizer_name}"}
        for problem_name in problem_names:
            results = all_results[problem_name][optimizer_name]
            row[(problem_name, "MSE_test")] = _format_summary_scientific(
                _nanmean_or_nan(
                    results.get(
                        "mse_test_list",
                        results.get("sur_real_mse_list", []),
                    )
                )
            )
            row[(problem_name, "MSE_sur_real")] = _format_summary_scientific(
                _nanmean_or_nan(
                    results.get(
                        "mse_sur_real_list",
                        results.get("sur_real_mse_list", []),
                    )
                )
            )
            row[(problem_name, "HV_sur")] = _format_summary_float(
                _nanmean_or_nan(results["hv_surrogate_list"])
            )
            row[(problem_name, "HV_real")] = _format_summary_float(
                _nanmean_or_nan(results["hv_real_list"])
            )
        summary_rows.append(row)

    return pd.DataFrame(
        summary_rows,
        columns=pd.MultiIndex.from_tuples(summary_columns),
    )


def _append_compact_summary_csv(table, csv_path, timestamp):
    csv_table = table.copy()
    csv_table.columns = [
        first if not second else f"{first}_{second}"
        for first, second in csv_table.columns.to_flat_index()
    ]
    csv_table.insert(0, "timestamp", timestamp)
    csv_path = Path(csv_path)
    csv_table.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )


def append_compact_summary_outputs(table, title, txt_path, csv_path):
    txt_path = Path(txt_path)
    csv_path = Path(csv_path)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    summary_text = table.to_string(index=False)
    with open(txt_path, "a", encoding="utf-8") as summary_record_file:
        summary_record_file.write(f"[{timestamp}] {title}\n")
        summary_record_file.write(summary_text)
        summary_record_file.write("\n\n")
    _append_compact_summary_csv(table, csv_path, timestamp)

    print(title)
    print(summary_text)
    print(f"Appended compact summary to: {txt_path}")
    print(f"Appended compact summary CSV to: {csv_path}")
    return table


def append_exp1_metrics_summary_outputs(
    method_name,
    optimizer_names,
    problem_names,
    all_results,
    problem_contexts,
    output_dir,
):
    table = build_exp1_metrics_summary_table(
        method_name=method_name,
        optimizer_names=optimizer_names,
        problem_names=problem_names,
        all_results=all_results,
        problem_contexts=problem_contexts,
    )
    output_dir = Path(output_dir)
    return append_compact_summary_outputs(
        table=table,
        title=f"Metrics summary | {method_name}",
        txt_path=output_dir / EXP1_METRICS_SUMMARY_FILENAME,
        csv_path=output_dir / EXP1_METRICS_SUMMARY_CSV_FILENAME,
    )
