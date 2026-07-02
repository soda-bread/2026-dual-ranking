# Run with: python -u Case_2_initial_objectives.py

import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:jupyter_client.session"
sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"jupyter_client\.session")
warnings.filterwarnings("ignore", message=r".*datetime\.datetime\.utcnow\(\) is deprecated.*", category=DeprecationWarning)


DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "pythermalcomfort": "pythermalcomfort",
    "tabpfn_client": "tabpfn-client",
}

missing = []
for module_name, pip_name in DEPENDENCIES.items():
    try:
        __import__(module_name)
    except ImportError as err:
        print(f"{module_name} is missing: {err}", flush=True)
        missing.append(pip_name)

if missing:
    print("Installing missing packages:", missing, flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])
    raise RuntimeError("Packages were installed. Restart the runtime, then run this script again.")

import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from pythermalcomfort.utilities import v_relative
except ImportError:
    def v_relative(v, met):
        return v + 0.3 * max(met - 1.0, 0.0)


def extract_pmv(result):
    if isinstance(result, dict):
        return float(result["pmv"])
    if hasattr(result, "pmv"):
        return float(result.pmv)
    return float(result)


try:
    from pythermalcomfort.models import pmv_ppd_iso

    def pmv_value(tdb, tr, vr, rh, met, clo):
        return extract_pmv(pmv_ppd_iso(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo))

except ImportError:
    from pythermalcomfort.models import pmv

    def pmv_value(tdb, tr, vr, rh, met, clo):
        return extract_pmv(pmv(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo))


np.set_printoptions(precision=6, suppress=True)


def resolve_building_space_dir():
    candidates = [
        Path("/content/drive/MyDrive/2026 Real-wrold problem/building_space_opt"),
        Path("/rds/projects/w/wangsu-building-automation/Huanbo/2026_real_world_problem/building_space_opt"),
        Path.cwd().resolve(),
        Path.cwd().resolve() / "building_space_opt",
        Path.cwd().resolve().parent / "building_space_opt",
    ]
    for candidate in candidates:
        if (candidate / "Dataset" / "data_office_1.csv").exists():
            return candidate
    searched = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find Dataset/data_office_1.csv. Searched:\n{searched}")


BASE_DIR = resolve_building_space_dir()
DATA_PATH = BASE_DIR / "Dataset" / "data_office_1.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "case_2_initial_objectives.txt"

FEATURE_COLUMNS = [
    "occupant_count [number]",
    "air_temperature [Celsius]",
    "indoor_relative_humidity [%]",
    "dry_bulb_temp [Celsius]",
    "outdoor_relative_humidity [%]",
    "wind_speed [m/s]",
    "global_horizontal_solar_radiation [W/m2]",
]
ENERGY_COLUMNS = [
    "ceiling_fan_energy [kWh]",
    "lighting_energy [kWh]",
    "plug_load_energy [kWh]",
    "chilled_water_energy [kWh]",
    "ahu_fan_energy [kWh]",
]
TARGET_COLUMN = "total_hvac_plug_lighting_energy [kWh]"
TRAIN_ROWS = 288
CALIBRATION_ROWS = 288


def load_data():
    dataset = pd.read_csv(DATA_PATH)
    dataset[TARGET_COLUMN] = dataset[ENERGY_COLUMNS].sum(axis=1)
    model_data = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    train_df = model_data.iloc[:TRAIN_ROWS].copy()
    val_df = model_data.iloc[TRAIN_ROWS:TRAIN_ROWS + CALIBRATION_ROWS].copy()
    test_df = model_data.iloc[TRAIN_ROWS:].copy()
    dual_test_df = model_data.iloc[TRAIN_ROWS + CALIBRATION_ROWS:].copy()
    if len(train_df) != TRAIN_ROWS:
        raise ValueError(f"Expected {TRAIN_ROWS} training rows, got {len(train_df)}")
    if len(val_df) == 0:
        raise ValueError("Dual-ranking needs a non-empty calibration set after the first training day.")
    return model_data, train_df, val_df, test_df, dual_test_df


# Building-space problem settings copied from Case 2.
OUTDOOR_T = 27.5
OUTDOOR_RH = 85.0
WIND_SPEED = 1.0
SOLAR_RADIATION = 650.0
CLO_DEFAULT = 0.5
MET_DEFAULT = 1.0
V_DEFAULT = 0.1

INITIAL_SOLUTION = np.array([0.0, 3.9, 7.8, 11.7, 15.6, 23.5, 27.4, 31.7, 34.3])
ROOM_TYPES = np.array(["office", "office", "office", "office", "office", "office", "office", "office"])
SPACE_WIDTH = 4.86
INITIAL_OCC_LIST = np.array([1, 1, 1, 1, 5, 1, 3, 1], dtype=float)
TOTAL_OCC = float(np.sum(INITIAL_OCC_LIST))
OCC_MIN_AREA = 6.0
TH_ZONE_LIST = np.array([
    [0.0, 3.9, 28.0, 60.1],
    [3.9, 7.8, 27.9, 60.0],
    [7.8, 11.7, 27.1, 71.3],
    [11.7, 15.6, 27.5, 74.0],
    [15.6, 23.5, 27.2, 72.0],
    [23.5, 27.4, 26.9, 57.5],
    [27.4, 31.7, 27.1, 61.0],
    [31.7, 34.3, 26.7, 58.8],
])


DUAL_RANKING_TARGET_COVERAGE = 0.90
DUAL_RANKING_ALPHA_MAX = 500.0
DUAL_RANKING_ALPHA_STEP = 0.01


def update_room_t_and_rh(wall_position):
    zone_start = TH_ZONE_LIST[:, 0]
    zone_end = TH_ZONE_LIST[:, 1]
    zone_t = TH_ZONE_LIST[:, 2]
    zone_rh = TH_ZONE_LIST[:, 3]
    left_boundaries = wall_position[:-1]
    right_boundaries = wall_position[1:]
    room_lengths = right_boundaries - left_boundaries
    room_t_list = np.full(len(room_lengths), 32.0)
    room_rh_list = np.full(len(room_lengths), 100.0)
    valid_mask = room_lengths > 0
    if not np.any(valid_mask):
        return room_t_list, room_rh_list
    overlap_start = np.maximum(left_boundaries[valid_mask, None], zone_start)
    overlap_end = np.minimum(right_boundaries[valid_mask, None], zone_end)
    overlap_length = np.clip(overlap_end - overlap_start, 0, None)
    overlap_ratios = overlap_length / room_lengths[valid_mask, None]
    room_t_list[valid_mask] = np.sum(overlap_ratios * zone_t, axis=1)
    room_rh_list[valid_mask] = np.sum(overlap_ratios * zone_rh, axis=1)
    return room_t_list, room_rh_list


def occ_allocate(room_area_list):
    office_area_list = room_area_list * (ROOM_TYPES == "office")
    max_possible_occ = np.floor(office_area_list / OCC_MIN_AREA).astype(int)
    excess_occupancy = np.maximum(INITIAL_OCC_LIST.astype(int) - max_possible_occ, 0)
    occ_remain = int(np.sum(excess_occupancy))
    new_occ_list = INITIAL_OCC_LIST.astype(int) - excess_occupancy
    for j, single_room_area in enumerate(office_area_list):
        max_occ_addable = int(single_room_area // OCC_MIN_AREA) - int(new_occ_list[j])
        if max_occ_addable >= 1 and occ_remain > 0:
            occ_added = min(max_occ_addable, occ_remain)
            new_occ_list[j] += occ_added
            occ_remain -= occ_added
        if occ_remain == 0:
            break
    return new_occ_list.astype(float)


def build_initial_features():
    room_area_list = np.diff(INITIAL_SOLUTION) * SPACE_WIDTH
    room_area_list = np.where(room_area_list < 0, 0.0, room_area_list)
    room_occ_list = occ_allocate(room_area_list)
    room_t_list, room_rh_list = update_room_t_and_rh(INITIAL_SOLUTION)

    energy_features = []
    pmv_features = []
    for room_t, room_rh, room_occ in zip(room_t_list, room_rh_list, room_occ_list):
        energy_features.append([
            room_occ,
            room_t,
            room_rh,
            OUTDOOR_T,
            OUTDOOR_RH,
            WIND_SPEED,
            SOLAR_RADIATION,
        ])
        pmv_features.append([room_t, room_t, V_DEFAULT, room_rh, MET_DEFAULT, CLO_DEFAULT])
    return np.asarray(energy_features, dtype=float), np.asarray(pmv_features, dtype=float), room_area_list, room_occ_list


def weighted_energy(predictions, room_area_list):
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    positive_area = np.maximum(np.asarray(room_area_list, dtype=float), 0.0)
    if positive_area.sum() <= 0:
        return 1e6
    weights = positive_area / positive_area.sum()
    return float(np.sum(predictions * weights))


def weighted_pmv(pmv_features, room_occ_list):
    pmv_results = []
    for indoor_t, tr, v, indoor_rh, met, clo in pmv_features:
        v_r = v_relative(v=v, met=met)
        pmv_results.append(pmv_value(tdb=indoor_t, tr=tr, vr=v_r, rh=indoor_rh, met=met, clo=clo))
    pmv_results = np.asarray(pmv_results, dtype=float)
    room_occ_list = np.asarray(room_occ_list, dtype=float)
    if room_occ_list.sum() <= 0:
        return float(np.mean(np.abs(pmv_results)))
    return float(np.sum(np.abs(pmv_results) * room_occ_list) / room_occ_list.sum())


def make_gpr_matern_pipeline():
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(len(FEATURE_COLUMNS)), length_scale_bounds=(1e-2, 1e3), nu=2.5)
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
    )
    return Pipeline([
        ("scale", StandardScaler()),
        ("gpr", GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42,
        )),
    ])


def find_upper_alpha_from_predictions(mean, std, y_true, target_coverage=0.90, alpha_max=500.0, alpha_step=0.01):
    mean = np.asarray(mean, dtype=float).reshape(-1)
    std = np.asarray(std, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if mean.shape != y_true.shape or std.shape != y_true.shape:
        raise ValueError("Prediction and target shapes must match.")
    if not 0.0 < float(target_coverage) <= 1.0:
        raise ValueError("target_coverage must be in (0, 1].")
    if alpha_step <= 0.0 or alpha_max < 0.0:
        raise ValueError("alpha_step must be positive and alpha_max must be non-negative.")

    required_scores = np.zeros_like(mean)
    positive_residual = y_true > mean
    positive_std = std > 0.0
    scalable = positive_residual & positive_std
    required_scores[scalable] = (y_true[scalable] - mean[scalable]) / std[scalable]
    required_scores[positive_residual & ~positive_std] = np.inf

    target_count = int(np.ceil(float(target_coverage) * len(required_scores)))
    required_alpha = float(np.sort(required_scores)[target_count - 1])
    if np.isfinite(required_alpha):
        alpha = float(np.ceil(required_alpha / alpha_step) * alpha_step)
        coverage = float(np.mean(y_true <= mean + alpha * std))
        return alpha, coverage

    alpha = float(alpha_max)
    coverage = float(np.mean(y_true <= mean + alpha * std))
    return alpha, coverage


def train_tabpfn(train_df, test_df):
    from tabpfn_client import TabPFNRegressor, set_access_token

    tabpfn_token = os.environ.get("TABPFN_TOKEN")
    if tabpfn_token:
        set_access_token(tabpfn_token)
    else:
        print("TABPFN_TOKEN is not set. Using any token already configured for tabpfn_client.", flush=True)

    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model = TabPFNRegressor()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    metrics = {"training_time_s": elapsed}
    if len(X_test) > 0:
        y_test_pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        metrics["test_mse"] = float(mean_squared_error(y_test, y_test_pred))
        metrics["test_r2"] = float(r2_score(y_test, y_test_pred))
    return model, metrics


def train_gpr_matern(train_df, test_df):
    gpr_train_df = train_df.sample(n=min(1500, len(train_df)), random_state=42).reset_index(drop=True)
    gpr_test_df = test_df.sample(n=min(5000, len(test_df)), random_state=42).reset_index(drop=True)

    X_train = gpr_train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = gpr_train_df[TARGET_COLUMN].to_numpy(dtype=float)
    X_test = gpr_test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = gpr_test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model = make_gpr_matern_pipeline()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    metrics = {
        "training_time_s": elapsed,
        "kernel": str(model.named_steps["gpr"].kernel_),
    }
    if len(X_test) > 0:
        y_test_pred, y_test_std = model.predict(X_test, return_std=True)
        metrics["test_mse"] = float(mean_squared_error(y_test, y_test_pred))
        metrics["test_r2"] = float(r2_score(y_test, y_test_pred))
        metrics["test_std_mean"] = float(np.mean(y_test_std))
    return model, metrics


def train_gpr_matern_dual(train_df, val_df, test_df):
    X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
    X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=float)
    X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

    model = make_gpr_matern_pipeline()
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    val_pred, val_std = model.predict(X_val, return_std=True)
    alpha_f1, coverage_f1 = find_upper_alpha_from_predictions(
        val_pred,
        val_std,
        y_val,
        target_coverage=DUAL_RANKING_TARGET_COVERAGE,
        alpha_max=DUAL_RANKING_ALPHA_MAX,
        alpha_step=DUAL_RANKING_ALPHA_STEP,
    )

    metrics = {
        "training_time_s": elapsed,
        "kernel": str(model.named_steps["gpr"].kernel_),
        "alpha_f1": alpha_f1,
        "coverage_f1": coverage_f1,
        "alpha_f2": 0.0,
    }
    if len(X_test) > 0:
        y_test_pred, y_test_std = model.predict(X_test, return_std=True)
        metrics["test_mse"] = float(mean_squared_error(y_test, y_test_pred))
        metrics["test_r2"] = float(r2_score(y_test, y_test_pred))
        metrics["test_std_mean"] = float(np.mean(y_test_std))
    return model, metrics


def format_metrics(metrics):
    lines = []
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.6f}")
        else:
            lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def main():
    model_data, train_df, val_df, test_df, dual_test_df = load_data()
    energy_features, pmv_features, room_area_list, room_occ_list = build_initial_features()
    f2 = weighted_pmv(pmv_features, room_occ_list)

    rows = []
    log_lines = [
        "Case 2 initial solution objectives",
        f"data_path: {DATA_PATH}",
        f"output_path: {OUTPUT_PATH}",
        f"model_data_shape: {model_data.shape}",
        f"training_rows: {train_df.shape}",
        f"calibration_rows_for_dual_ranking: {val_df.shape}",
        f"initial_solution: {np.array2string(INITIAL_SOLUTION, precision=6, separator=', ')}",
        f"room_area_list: {np.array2string(room_area_list, precision=6, separator=', ')}",
        f"room_occ_list: {np.array2string(room_occ_list, precision=6, separator=', ')}",
        f"room_energy_features_columns: {FEATURE_COLUMNS}",
        "",
    ]

    print("Training TabPFN and evaluating initial solution...", flush=True)
    tabpfn_model, tabpfn_metrics = train_tabpfn(train_df, test_df)
    tabpfn_pred = tabpfn_model.predict(energy_features)
    rows.append({
        "method": "TabPFN + NSGA-II",
        "f1": weighted_energy(tabpfn_pred, room_area_list),
        "f2": f2,
        "f1_std": np.nan,
        "f1_upper": np.nan,
        "metrics": tabpfn_metrics,
    })

    print("Training GPR-Matern and evaluating initial solution...", flush=True)
    gpr_model, gpr_metrics = train_gpr_matern(train_df, test_df)
    gpr_pred = gpr_model.predict(energy_features)
    rows.append({
        "method": "GPR-Matern + NSGA-II",
        "f1": weighted_energy(gpr_pred, room_area_list),
        "f2": f2,
        "f1_std": np.nan,
        "f1_upper": np.nan,
        "metrics": gpr_metrics,
    })

    print("Training GPR-Matern dual-ranking and evaluating initial solution...", flush=True)
    dual_model, dual_metrics = train_gpr_matern_dual(train_df, val_df, dual_test_df)
    dual_mean, dual_std = dual_model.predict(energy_features, return_std=True)
    dual_f1 = weighted_energy(dual_mean, room_area_list)
    dual_f1_std = weighted_energy(dual_std, room_area_list)
    dual_f1_upper = dual_f1 + dual_metrics["alpha_f1"] * dual_f1_std
    rows.append({
        "method": "GPR-Matern + Dual-Ranking NSGA-II",
        "f1": dual_f1,
        "f2": f2,
        "f1_std": dual_f1_std,
        "f1_upper": dual_f1_upper,
        "metrics": dual_metrics,
    })

    for row in rows:
        log_lines.extend([
            row["method"],
            f"  f1: {row['f1']:.10f}",
            f"  f2: {row['f2']:.10f}",
        ])
        if np.isfinite(row["f1_std"]):
            log_lines.append(f"  f1_std: {row['f1_std']:.10f}")
        if np.isfinite(row["f1_upper"]):
            log_lines.append(f"  f1_upper_mean_plus_alpha_std: {row['f1_upper']:.10f}")
        log_lines.append(format_metrics(row["metrics"]))
        log_lines.append("")

    OUTPUT_PATH.write_text("\n".join(log_lines), encoding="utf-8")
    print("\n".join(log_lines), flush=True)
    print(f"Saved txt: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
