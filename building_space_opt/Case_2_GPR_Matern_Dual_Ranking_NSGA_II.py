# Auto-generated from Case_2_GPR_Matern_Dual_Ranking_NSGA_II.ipynb.
# Run with: python -u Case_2_GPR_Matern_Dual_Ranking_NSGA_II.py


# %% [markdown]
# # Case 2 GPR-Matern + Dual-Ranking+NSGA-II
# 
# This notebook implements the building-space flexible-wall optimisation problem with a Gaussian Process Regression surrogate using a Matern kernel and Dual-Ranking+NSGA-II. The energy objective uses GPR mean and uncertainty; the PMV objective remains deterministic.

# %% [markdown]
# ## 1. Packages

# %% cell 2
import os
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning:jupyter_client.session"

import importlib
import subprocess
import sys
import warnings

sys.dont_write_bytecode = True
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"jupyter_client\.session")
warnings.filterwarnings("ignore", message=r".*datetime\.datetime\.utcnow\(\) is deprecated.*", category=DeprecationWarning)

DEPENDENCIES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "sklearn": "scikit-learn",
    "pymoo": "pymoo==0.6.1.6",
    "pythermalcomfort": "pythermalcomfort",
}

missing = []
for module_name, pip_name in DEPENDENCIES.items():
    try:
        importlib.import_module(module_name)
        print(f"{module_name} is available.")
    except ImportError as err:
        print(f"{module_name} is missing: {err}")
        missing.append(pip_name)

if missing:
    print("Installing missing packages:", missing)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *missing])
    raise RuntimeError("Packages were installed. Restart the notebook kernel/runtime, then run this cell again.")

print(sys.version)

# %% [markdown]
# ## 2. Imports and dataset

# %% cell 4
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from pymoo.core.problem import Problem
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination import get_termination
from pymoo.visualization.scatter import Scatter
from pymoo.core.survival import Survival
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.operators.survival.rank_and_crowding.metrics import get_crowding_function

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

np.set_printoptions(precision=3, suppress=True)

# %% cell 5
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ModuleNotFoundError:
    pass

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

dataset = pd.read_csv(DATA_PATH)
dataset[TARGET_COLUMN] = dataset[ENERGY_COLUMNS].sum(axis=1)
model_data = dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()

train_df = model_data.iloc[:TRAIN_ROWS].copy()
val_df = model_data.iloc[TRAIN_ROWS:TRAIN_ROWS + CALIBRATION_ROWS].copy()
test_df = model_data.iloc[TRAIN_ROWS + CALIBRATION_ROWS:].copy()
if len(train_df) != TRAIN_ROWS:
    raise ValueError(f"Expected {TRAIN_ROWS} training rows, got {len(train_df)}")
if len(val_df) == 0:
    raise ValueError("Dual-ranking needs a non-empty calibration set after the first training day.")
print("Data path:", DATA_PATH)
print("Model data shape:", model_data.shape)
print("Training rows:", train_df.shape, "| Calibration rows:", val_df.shape, "| Test rows:", test_df.shape)
print(model_data[TARGET_COLUMN].describe().round(4))

# %% [markdown]
# ## 3. Train energy surrogate

# %% cell 7
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DUAL_RANKING_TARGET_COVERAGE = 0.90
DUAL_RANKING_ALPHA_MAX = 500.0
DUAL_RANKING_ALPHA_STEP = 0.01

X_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=float)
y_train = train_df[TARGET_COLUMN].to_numpy(dtype=float)
X_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=float)
y_val = val_df[TARGET_COLUMN].to_numpy(dtype=float)
X_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=float)
y_test = test_df[TARGET_COLUMN].to_numpy(dtype=float)

kernel = (
    ConstantKernel(1.0, (1e-3, 1e3))
    * Matern(length_scale=np.ones(len(FEATURE_COLUMNS)), length_scale_bounds=(1e-2, 1e3), nu=2.5)
    + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e1))
)

energy_predictor = Pipeline([
    ("scale", StandardScaler()),
    ("gpr", GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=42,
    )),
])

start = time.time()
energy_predictor.fit(X_train, y_train)
print(f"GPR-Matern training time: {time.time() - start:.2f} s")
print("Optimized kernel:", energy_predictor.named_steps["gpr"].kernel_)


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

val_pred, val_std = energy_predictor.predict(X_val, return_std=True)
dual_alpha_f1, dual_coverage_f1 = find_upper_alpha_from_predictions(
    val_pred,
    val_std,
    y_val,
    target_coverage=DUAL_RANKING_TARGET_COVERAGE,
    alpha_max=DUAL_RANKING_ALPHA_MAX,
    alpha_step=DUAL_RANKING_ALPHA_STEP,
)
dual_alpha_f2 = 0.0
print(
    f"Dual-ranking upper bound calibration: "
    f"alpha_f1={dual_alpha_f1:.3f}, coverage_f1={dual_coverage_f1:.3%}, "
    f"alpha_f2={dual_alpha_f2:.3f} (PMV deterministic)"
)

if len(X_test) > 0:
    y_test_pred, y_test_std = energy_predictor.predict(X_test, return_std=True)
    print(f"GPR-Matern test MSE: {mean_squared_error(y_test, y_test_pred):.6f}")
    print(f"GPR-Matern test R2:  {r2_score(y_test, y_test_pred):.6f}")
    print("Predictive std summary:", pd.Series(y_test_std).describe().round(4).to_dict())
else:
    print("No held-out test rows after training and calibration windows.")

# %% [markdown]
# ## 4. Objectives, layout display, and NSGA-II problem

# %% cell 9
class EnergyCostObjective:
    def __init__(self, predictor, model_kind):
        self.predictor = predictor
        self.model_kind = model_kind
        self.columns = FEATURE_COLUMNS

    def predict_energy(self, features_batch):
        mean, _ = self.predict_energy_mean_std(features_batch)
        return mean

    def predict_energy_mean_std(self, features_batch):
        data = pd.DataFrame(features_batch, columns=self.columns)
        if self.model_kind == "gpr_matern":
            mean, std = self.predictor.predict(data.to_numpy(dtype=float), return_std=True)
            return np.asarray(mean, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1)
        raise ValueError(f"Unknown model_kind for dual-ranking: {self.model_kind}")

    def calculate_cost(self, features_batch, room_area_list):
        mean, _ = self.calculate_cost_mean_std(features_batch, room_area_list)
        return mean

    def calculate_cost_mean_std(self, features_batch, room_area_list):
        mean, std = self.predict_energy_mean_std(features_batch)
        room_area_list = np.asarray(room_area_list, dtype=float)
        positive_area = np.maximum(room_area_list, 0.0)
        if positive_area.sum() <= 0:
            return 1e6, 1e6
        weights = positive_area / positive_area.sum()
        mean_weighted = float(np.sum(mean * weights))
        std_weighted = float(np.sum(std * weights))
        return mean_weighted, std_weighted


class PMVObjective:
    def calculate_pmv(self, features_batch, room_occ_list):
        pmv_results = []
        for feature in features_batch:
            indoor_t, tr, v, indoor_rh, met, clo = feature
            v_r = v_relative(v=v, met=met)
            pmv_results.append(pmv_value(tdb=indoor_t, tr=tr, vr=v_r, rh=indoor_rh, met=met, clo=clo))
        pmv_results = np.asarray(pmv_results, dtype=float)
        room_occ_list = np.asarray(room_occ_list, dtype=float)
        if room_occ_list.sum() <= 0:
            return float(np.mean(np.abs(pmv_results)))
        return float(np.sum(np.abs(pmv_results) * room_occ_list) / room_occ_list.sum())


class Plot2DDisplay:
    def __init__(self, wall_position, room_types, space_width, title="2D Room Plot"):
        self.wall_position = np.asarray(wall_position, dtype=float)
        self.room_types = np.asarray(room_types)
        self.space_width = float(space_width)
        self.title = title
        self.room_colors = {
            "office": "#c8dbf3",
            "meeting_room": "#d5e8d4",
            "closed": "#f8cecc",
        }

    def plot(self, save_path=None):
        y0 = np.zeros_like(self.wall_position)
        y1 = np.full_like(self.wall_position, self.space_width)
        plt.figure(figsize=(9, 4.8))
        plt.scatter(self.wall_position, y0, color="black", s=8)
        plt.scatter(self.wall_position, y1, color="black", s=8)
        plt.plot(self.wall_position, y0, color="black", linewidth=1)
        plt.plot(self.wall_position, y1, color="black", linewidth=1)
        for x in self.wall_position:
            plt.plot([x, x], [0, self.space_width], color="black", linewidth=1)
        room_idx = 0
        for i in range(len(self.wall_position) - 1):
            x0, x1 = self.wall_position[i], self.wall_position[i + 1]
            if x1 <= x0:
                continue
            room_type = self.room_types[room_idx]
            color = self.room_colors.get(room_type, "lightgray")
            plt.fill_between([x0, x1], 0, self.space_width, color=color, alpha=1)
            area = (x1 - x0) * self.space_width
            plt.text((x0 + x1) / 2, self.space_width * 0.78, str(room_idx + 1), ha="center", fontsize=9)
            plt.text((x0 + x1) / 2, self.space_width * 0.12, f"{area:.1f} m2", ha="center", fontsize=8)
            room_idx += 1
        plt.axis("equal")
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.title(self.title)
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print("Saved figure:", save_path, flush=True)
        plt.show()


class FlexibleWallProblem(Problem):
    def __init__(
        self,
        num_decision_variables,
        initial_position,
        fixed_wall_list,
        room_types,
        required_num_meeting_room,
        space_width,
        room_area_min,
        initial_occ_list,
        total_occ,
        occ_min_area,
        th_zone_list,
        energy_predictor,
        model_kind,
        outdoor_t,
        outdoor_rh,
        wind_speed,
        solar_radiation,
        clo,
        met,
        v,
    ):
        self.num_decision_variables = int(num_decision_variables)
        self.position = np.asarray(initial_position, dtype=float)
        self.fixed_wall_list = np.asarray(fixed_wall_list, dtype=float)
        self.room_types = np.asarray(room_types)
        self.required_num_meeting_room = int(required_num_meeting_room)
        self.space_width = float(space_width)
        self.room_area_min = float(room_area_min)
        self.initial_occ_list = np.asarray(initial_occ_list, dtype=float)
        self.total_occ = float(total_occ)
        self.occ_min_area = float(occ_min_area)
        self.th_zone_list = np.asarray(th_zone_list, dtype=float)
        self.energy_objective = EnergyCostObjective(energy_predictor, model_kind=model_kind)
        self.pmv_objective = PMVObjective()
        self.outdoor_t = float(outdoor_t)
        self.outdoor_rh = float(outdoor_rh)
        self.wind_speed = float(wind_speed)
        self.solar_radiation = float(solar_radiation)
        self.clo = float(clo)
        self.met = float(met)
        self.v = float(v)

        if self.position.shape[1] != self.num_decision_variables:
            raise ValueError("initial_position width must match num_decision_variables")
        if len(self.fixed_wall_list) != self.num_decision_variables:
            raise ValueError("fixed_wall_list length must match num_decision_variables")

        xl = np.zeros(self.num_decision_variables)
        xu = np.ones(self.num_decision_variables) * np.max(self.position)
        n_intervals = self.num_decision_variables - 1
        fixed_count = int(np.sum(self.fixed_wall_list >= 0))
        n_constr = n_intervals + fixed_count + 1 + n_intervals + 1 + n_intervals
        super().__init__(n_var=self.num_decision_variables, n_obj=2, n_constr=n_constr, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = np.asarray(X, dtype=float).copy()
        fixed_indices = np.where(self.fixed_wall_list >= 0)[0]
        X[:, fixed_indices] = self.fixed_wall_list[fixed_indices]

        result_f1 = []
        result_f2 = []
        result_f1_std = []
        result_f2_std = []
        occ_record = np.zeros(X.shape[0])
        area_occ_record = np.zeros((X.shape[0], X.shape[1] - 1))
        room_area_list_record = np.zeros((X.shape[0], X.shape[1] - 1))

        for i, single_solution in enumerate(X):
            room_area_list = np.diff(single_solution) * self.space_width
            room_area_list = np.where(room_area_list < 0, 0.0, room_area_list)
            room_occ_list = self.occ_allocate(room_area_list)
            occ_record[i] = abs(np.sum(room_occ_list) - self.total_occ)
            area_occ_record[i, :] = room_occ_list * self.occ_min_area - room_area_list
            room_area_list_record[i, :] = room_area_list - self.room_area_min
            room_t_list, room_rh_list = self.update_room_t_and_rh(single_solution)

            features_batch_f1 = []
            features_batch_f2 = []
            for room_t, room_rh, room_occ in zip(room_t_list, room_rh_list, room_occ_list):
                features_batch_f1.append([
                    room_occ,
                    room_t,
                    room_rh,
                    self.outdoor_t,
                    self.outdoor_rh,
                    self.wind_speed,
                    self.solar_radiation,
                ])
                features_batch_f2.append([room_t, room_t, self.v, room_rh, self.met, self.clo])

            f1_mean, f1_std = self.energy_objective.calculate_cost_mean_std(features_batch_f1, room_area_list)
            result_f1.append(f1_mean)
            result_f1_std.append(f1_std)
            result_f2.append(self.pmv_objective.calculate_pmv(features_batch_f2, room_occ_list))
            result_f2_std.append(0.0)

        constraints_x = np.diff(X)
        constraints_fixed_walls = np.column_stack([
            X[:, i] - self.fixed_wall_list[i]
            for i in range(len(self.fixed_wall_list))
            if self.fixed_wall_list[i] >= 0
        ])
        constraints_total_occ = occ_record.reshape(-1, 1)
        num_meeting_room = int(np.count_nonzero(self.room_types == "meeting_room"))
        constraints_num_meeting_room = np.full((X.shape[0], 1), self.required_num_meeting_room - num_meeting_room)

        out["F"] = np.column_stack([result_f1, result_f2])
        out["std"] = np.column_stack([result_f1_std, result_f2_std])
        out["G"] = np.column_stack([
            -constraints_x,
            constraints_fixed_walls,
            constraints_total_occ,
            area_occ_record,
            constraints_num_meeting_room,
            -room_area_list_record,
        ])

    def update_room_t_and_rh(self, single_solution):
        zone_start = self.th_zone_list[:, 0]
        zone_end = self.th_zone_list[:, 1]
        zone_t = self.th_zone_list[:, 2]
        zone_rh = self.th_zone_list[:, 3]
        left_boundaries = single_solution[:-1]
        right_boundaries = single_solution[1:]
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

    def occ_allocate(self, room_area_list):
        office_area_list = room_area_list * (self.room_types == "office")
        max_possible_occ = np.floor(office_area_list / self.occ_min_area).astype(int)
        excess_occupancy = np.maximum(self.initial_occ_list.astype(int) - max_possible_occ, 0)
        occ_remain = int(np.sum(excess_occupancy))
        new_occ_list = self.initial_occ_list.astype(int) - excess_occupancy
        for j, single_room_area in enumerate(office_area_list):
            max_occ_addable = int(single_room_area // self.occ_min_area) - int(new_occ_list[j])
            if max_occ_addable >= 1 and occ_remain > 0:
                occ_added = min(max_occ_addable, occ_remain)
                new_occ_list[j] += occ_added
                occ_remain -= occ_added
            if occ_remain == 0:
                break
        return new_occ_list.astype(float)


class CustomMutation(Mutation):
    def __init__(self, prob, eta, fixed_wall_list):
        super().__init__()
        self.prob = float(prob)
        self.eta = float(eta)
        self.fixed_wall_list = np.asarray(fixed_wall_list, dtype=float)

    def _do(self, problem, X, **kwargs):
        X = np.asarray(X, dtype=float).copy()
        movable_indices = np.where(self.fixed_wall_list < 0)[0]
        mutation_mask = np.random.random(X[:, movable_indices].shape) < self.prob
        X[:, movable_indices] += mutation_mask * np.random.normal(0, self.eta, size=mutation_mask.shape)
        X = np.clip(X, problem.xl, problem.xu)
        X = np.sort(X, axis=1)
        fixed_indices = np.where(self.fixed_wall_list >= 0)[0]
        X[:, fixed_indices] = self.fixed_wall_list[fixed_indices]
        return X


class FromArraySampling(Sampling):
    def __init__(self, initial_population):
        super().__init__()
        self.initial_population = np.asarray(initial_population, dtype=float)

    def _do(self, problem, n_samples, **kwargs):
        n_initial = self.initial_population.shape[0]
        if n_initial < n_samples:
            extra = n_samples - n_initial
            random_samples = np.random.uniform(problem.xl, problem.xu, (extra, problem.n_var))
            random_samples = np.sort(random_samples, axis=1)
            return np.vstack([self.initial_population, random_samples])
        return self.initial_population[:n_samples]


class DualRankingSurvival(Survival):
    def __init__(self, alpha_f1, alpha_f2, nds=None, crowding_func="cd"):
        crowding_func_ = get_crowding_function(crowding_func)
        super().__init__(filter_infeasible=True)
        self.nds = nds if nds is not None else NonDominatedSorting()
        self.crowding_func = crowding_func_
        self.alpha_f1 = float(alpha_f1)
        self.alpha_f2 = float(alpha_f2)

    def _do(self, problem, pop, *args, random_state=None, n_survive=None, **kwargs):
        F = pop.get("F").astype(float, copy=False)
        F_std = pop.get("std")
        if F_std is None:
            raise ValueError("DualRankingSurvival requires problem output `std`.")
        F_std = F_std.astype(float, copy=False)
        alphas = np.array([self.alpha_f1, self.alpha_f2], dtype=float)
        F_upper = F + alphas * F_std
        F_hybrid = np.concatenate([F, F_upper], axis=1)
        fronts_hybrid = self.nds.do(F_hybrid, n_stop_if_ranked=n_survive)

        survivors = []
        for k, front in enumerate(fronts_hybrid):
            I = np.arange(len(front))
            if len(survivors) + len(I) > n_survive:
                n_remove = len(survivors) + len(front) - n_survive
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=n_remove)
                I = randomized_argsort(crowding_of_front, order="descending", method="numpy", random_state=random_state)
                I = I[:-n_remove]
            else:
                crowding_of_front = self.crowding_func.do(F[front, :], n_remove=0)

            for j, i in enumerate(front):
                pop[i].set("rank", k)
                pop[i].set("crowding", crowding_of_front[j])
            survivors.extend(front[I])
        return pop[survivors]


class HVMonitor:
    def __init__(self, ref_point, print_every=1):
        self.hv = HV(ref_point=np.asarray(ref_point, dtype=float))
        self.values = []
        self.print_every = int(print_every)
        self.start_time = None
        self.last_time = None
        self.generation_times = []

    def __call__(self, algorithm):
        now = time.time()
        if self.start_time is None:
            self.start_time = now
            self.last_time = now
        gen_time = now - self.last_time
        elapsed_time = now - self.start_time
        self.last_time = now
        self.generation_times.append(gen_time)

        F = algorithm.pop.get("F")
        hv_value = self.hv(F)
        self.values.append(hv_value)

        n_gen = getattr(algorithm, "n_gen", len(self.values))
        if self.print_every > 0 and n_gen % self.print_every != 0:
            return

        evaluator = getattr(algorithm, "evaluator", None)
        n_eval = getattr(evaluator, "n_eval", "NA")
        CV = algorithm.pop.get("CV")
        if CV is None:
            cv_min = np.nan
            cv_avg = np.nan
        else:
            CV = np.asarray(CV, dtype=float).reshape(-1)
            cv_min = float(np.nanmin(CV)) if CV.size else np.nan
            cv_avg = float(np.nanmean(CV)) if CV.size else np.nan

        print(
            f"[progress] gen={n_gen} eval={n_eval} "
            f"gen_time={gen_time:.3f}s elapsed={elapsed_time:.3f}s "
            f"n_nds={len(F) if F is not None else 'NA'} "
            f"cv_min={cv_min:.3e} cv_avg={cv_avg:.3e} hv={hv_value:.6e}",
            flush=True,
        )

# %% [markdown]
# ## 5. Run NSGA-II

# %% cell 11
METHOD_NAME = "GPR-Matern"
METHOD_SLUG = "gpr_matern_dual_ranking"
OPTIMIZER_NAME = "Dual-Ranking+NSGA-II"
MODEL_KIND = "gpr_matern"

# Building-space problem settings copied from Case 2.
outdoor_t = 27.5
outdoor_rh = 85.0
wind_speed = 1.0
solar_radiation = 650.0
clo_default = 0.5
met_default = 1.0
v_default = 0.1

num_decision_variables = 9
position_first_wall = 0.0
position_last_wall = 34.3
initial_solution = np.array([0.0, 3.9, 7.8, 11.7, 15.6, 23.5, 27.4, 31.7, 34.3])
fixed_wall_list = np.array([position_first_wall, -1, -1, -1, -1, -1, -1, -1, position_last_wall], dtype=float)
room_types_list = np.array(["office", "office", "office", "office", "office", "office", "office", "office"])
required_num_meeting_room = 0
space_width = 4.86
room_length_min = 1.5
room_area_min = room_length_min * space_width
initial_occ_list = np.array([1, 1, 1, 1, 5, 1, 3, 1])
total_occ = int(np.sum(initial_occ_list))
occ_min_area = 6.0
th_zone_list = np.array([
    [0.0, 3.9, 28.0, 60.1],
    [3.9, 7.8, 27.9, 60.0],
    [7.8, 11.7, 27.1, 71.3],
    [11.7, 15.6, 27.5, 74.0],
    [15.6, 23.5, 27.2, 72.0],
    [23.5, 27.4, 26.9, 57.5],
    [27.4, 31.7, 27.1, 61.0],
    [31.7, 34.3, 26.7, 58.8],
])

N_GEN = 200
POP_SIZE = 50
RANDOM_SEED = 0
MUTATION_PROB = 0.7
MUTATION_ETA = 1.5
CROSSOVER_PROB = 0.7
REF_POINT = np.array([0.5, 0.8])

solutions = np.tile(initial_solution, (POP_SIZE, 1))
problem = FlexibleWallProblem(
    num_decision_variables=num_decision_variables,
    initial_position=solutions,
    fixed_wall_list=fixed_wall_list,
    room_types=room_types_list,
    required_num_meeting_room=required_num_meeting_room,
    space_width=space_width,
    room_area_min=room_area_min,
    initial_occ_list=initial_occ_list,
    total_occ=total_occ,
    occ_min_area=occ_min_area,
    th_zone_list=th_zone_list,
    energy_predictor=energy_predictor,
    model_kind=MODEL_KIND,
    outdoor_t=outdoor_t,
    outdoor_rh=outdoor_rh,
    wind_speed=wind_speed,
    solar_radiation=solar_radiation,
    clo=clo_default,
    met=met_default,
    v=v_default,
)

initial_result = problem.evaluate(initial_solution.reshape(1, -1), return_values_of=["F"])
initial_f1, initial_f2 = initial_result[0]
print("Initial solution:", initial_solution)
print(f"Initial objectives | f1 energy: {initial_f1:.6f}, f2 PMV: {initial_f2:.6f}")

mutation = CustomMutation(prob=MUTATION_PROB, eta=MUTATION_ETA, fixed_wall_list=fixed_wall_list)
crossover = SBX(prob=CROSSOVER_PROB)
survival_function = DualRankingSurvival(alpha_f1=dual_alpha_f1, alpha_f2=dual_alpha_f2)
algorithm = NSGA2(
    pop_size=POP_SIZE,
    mutation=mutation,
    crossover=crossover,
    survival=survival_function,
    eliminate_duplicates=True,
    sampling=FromArraySampling(solutions),
)
termination = get_termination("n_gen", N_GEN)
ref_point = REF_POINT

print("\nAlgorithm parameters", flush=True)
print(f"method: {METHOD_NAME} + {OPTIMIZER_NAME}")
print(f"n_gen: {N_GEN}")
print(f"pop_size: {POP_SIZE}")
print(f"random_seed: {RANDOM_SEED}")
print(f"mutation: CustomMutation(prob={MUTATION_PROB}, eta={MUTATION_ETA})")
print(f"crossover: SBX(prob={CROSSOVER_PROB})")
print(f"survival: DualRankingSurvival(alpha_f1={dual_alpha_f1:.3f}, alpha_f2={dual_alpha_f2:.3f})")
print(f"eliminate_duplicates: True")
print(f"sampling: FromArraySampling(initial_solution repeated {POP_SIZE} times)")
print(f"hypervolume ref_point: {ref_point}")

start = time.time()
res = minimize(
    problem,
    algorithm,
    termination,
    seed=RANDOM_SEED,
    save_history=True,
    verbose=True,
    callback=HVMonitor(ref_point=ref_point),
)
print(f"Running time: {time.time() - start:.3f} s")

print("Pareto-optimal solutions (X):")
print(res.X)
print("\nOptimal objective values")
for i, objective in enumerate(res.F):
    f1_reduce_percent = ((initial_f1 - objective[0]) / initial_f1) * 100 if initial_f1 != 0 else np.nan
    f2_reduce_percent = ((initial_f2 - objective[1]) / initial_f2) * 100 if initial_f2 != 0 else np.nan
    print(f"solution_{i+1} f1: {objective[0]:.6f} ({f1_reduce_percent:.3f}%), f2: {objective[1]:.6f} ({f2_reduce_percent:.3f}%)")

# %% [markdown]
# ## 6. Plot and save results

# %% cell 13
F = np.asarray(res.F, dtype=float)
X = np.atleast_2d(res.X).astype(float)
optimizer_name_for_output = globals().get("OPTIMIZER_NAME", "NSGA-II")

idx_best_f1 = int(np.argmin(F[:, 0]))
idx_best_f2 = int(np.argmin(F[:, 1]))
sorted_by_f1 = np.argsort(F[:, 0])
idx_middle_pareto = int(sorted_by_f1[len(sorted_by_f1) // 2])
abc_indices = {
    "A": idx_best_f1,          # left/top: best energy
    "B": idx_middle_pareto,    # middle/balanced point on Pareto front
    "C": idx_best_f2,          # right/bottom: best comfort
}
abc_descriptions = {
    "A": "best_energy",
    "B": "middle_pareto_balance",
    "C": "best_comfort",
}


def plot_pareto_with_abc(save_path, xlim=None, ylim=None):
    plt.figure(figsize=(6, 4))
    plt.scatter(F[:, 0], F[:, 1], s=24, color="tab:blue", alpha=0.75)
    label_offsets = {
        "A": (-12, 10),
        "B": (8, 10),
        "C": (8, -12),
    }
    for label, idx in abc_indices.items():
        plt.scatter(
            F[idx, 0],
            F[idx, 1],
            s=30,
            marker="o",
            color="red",
            edgecolors="red",
            linewidths=0.8,
            zorder=4,
        )
        plt.annotate(
            label,
            xy=(F[idx, 0], F[idx, 1]),
            xytext=label_offsets[label],
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            color="red",
            zorder=5,
        )
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print("Saved figure:", save_path, flush=True)
    plt.show()


def draw_layout_axis(ax, wall_position, row_label):
    wall_position = np.asarray(wall_position, dtype=float)
    y0 = 0.0
    y1 = float(space_width)
    room_colors = {
        "office": "#c8dbf3",
        "meeting_room": "#d5e8d4",
        "closed": "#f8cecc",
    }
    room_idx = 0
    for i in range(len(wall_position) - 1):
        x0, x1 = wall_position[i], wall_position[i + 1]
        if x1 <= x0:
            continue
        room_type = room_types_list[room_idx]
        color = room_colors.get(room_type, "lightgray")
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=color, edgecolor="black", linewidth=1.0))
        area = (x1 - x0) * space_width
        ax.text((x0 + x1) / 2, y1 * 0.63, str(room_idx + 1), ha="center", va="center", fontsize=8)
        ax.text((x0 + x1) / 2, y1 * 0.27, f"{area:.1f} m2", ha="center", va="center", fontsize=7)
        room_idx += 1
    for x in wall_position:
        ax.plot([x, x], [y0, y1], color="black", linewidth=1.0)
    ax.set_xlim(position_first_wall, position_last_wall)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(position_first_wall - 1.4, y1 / 2, row_label, ha="right", va="center", fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_layout_abc(save_path):
    layout_rows = [
        ("Initial", initial_solution),
        ("A", X[abc_indices["A"]]),
        ("B", X[abc_indices["B"]]),
        ("C", X[abc_indices["C"]]),
    ]
    fig, axes = plt.subplots(len(layout_rows), 1, figsize=(10, 7.2), constrained_layout=True)
    for ax, (row_label, wall_position) in zip(axes, layout_rows):
        draw_layout_axis(ax, wall_position, row_label)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print("Saved figure:", save_path, flush=True)
    plt.show()


pareto_path = OUTPUT_DIR / f"{METHOD_SLUG}_pareto_front.png"
plot_pareto_with_abc(pareto_path)

pareto_zoom_path = OUTPUT_DIR / f"{METHOD_SLUG}_pareto_front_zoom.png"
plot_pareto_with_abc(pareto_zoom_path, xlim=(0.185, 0.220), ylim=(0.45, 0.65))

layout_abc_path = OUTPUT_DIR / f"{METHOD_SLUG}_layout_ABC.png"
plot_layout_abc(layout_abc_path)

hv = HV(ref_point=ref_point)
hypervolume = hv(F)
print("Reference point:", ref_point)
print("Hypervolume:", hypervolume)

solution_columns = [f"wall_{i}" for i in range(num_decision_variables)]
results_df = pd.DataFrame(X, columns=solution_columns)
results_df["f1_energy"] = F[:, 0]
results_df["f2_pmv"] = F[:, 1]
results_df["distance_to_origin"] = np.linalg.norm(F, axis=1)
results_df["pareto_order_by_f1"] = np.nan
for order, idx in enumerate(sorted_by_f1):
    results_df.loc[int(idx), "pareto_order_by_f1"] = order
results_df["point_label"] = ""
for label, idx in abc_indices.items():
    existing = results_df.loc[idx, "point_label"]
    results_df.loc[idx, "point_label"] = label if existing == "" else f"{existing},{label}"
results_df["is_best_energy"] = False
results_df["is_best_comfort"] = False
results_df["is_middle_pareto_balance"] = False
results_df.loc[idx_best_f1, "is_best_energy"] = True
results_df.loc[idx_best_f2, "is_best_comfort"] = True
results_df.loc[idx_middle_pareto, "is_middle_pareto_balance"] = True
results_df["method"] = METHOD_NAME
results_df["optimizer"] = optimizer_name_for_output


def format_solution_block(label, idx):
    walls = np.array2string(X[idx], precision=6, separator=", ")
    return "\n".join([
        f"[{label}: {abc_descriptions[label]}]",
        f"index: {idx}",
        f"pareto_order_by_f1: {int(results_df.loc[idx, 'pareto_order_by_f1'])}",
        f"f1_energy: {F[idx, 0]:.10f}",
        f"f2_pmv: {F[idx, 1]:.10f}",
        f"distance_to_origin: {np.linalg.norm(F[idx]):.10f}",
        f"walls: {walls}",
    ])

summary_lines = [
    f"method: {METHOD_NAME}",
    f"optimizer: {optimizer_name_for_output}",
    f"n_gen: {N_GEN}",
    f"pop_size: {POP_SIZE}",
    f"mutation_prob: {MUTATION_PROB}",
    f"mutation_eta: {MUTATION_ETA}",
    f"crossover_prob: {CROSSOVER_PROB}",
    f"ref_point: {np.array2string(ref_point, precision=6, separator=', ')}",
    f"hypervolume: {hypervolume:.10f}",
    "",
    "Representative Pareto solutions",
    "A: best_energy (left/top on Pareto front)",
    "B: middle_pareto_balance (middle point after sorting Pareto solutions by f1)",
    "C: best_comfort (right/bottom on Pareto front)",
    "",
    format_solution_block("A", abc_indices["A"]),
    "",
    format_solution_block("B", abc_indices["B"]),
    "",
    format_solution_block("C", abc_indices["C"]),
    "",
    "All Pareto-optimal solutions",
]
for i in range(len(F)):
    label_text = results_df.loc[i, "point_label"]
    label_prefix = f" label={label_text}," if label_text else ""
    summary_lines.append(
        f"solution_{i}:{label_prefix} pareto_order_by_f1={int(results_df.loc[i, 'pareto_order_by_f1'])}, "
        f"f1_energy={F[i, 0]:.10f}, f2_pmv={F[i, 1]:.10f}, "
        f"distance_to_origin={np.linalg.norm(F[i]):.10f}, "
        f"walls={np.array2string(X[i], precision=6, separator=', ')}"
    )

summary_text = "\n".join(summary_lines) + "\n"
summary_path = OUTPUT_DIR / f"{METHOD_SLUG}_optimal_solutions.txt"
summary_path.write_text(summary_text, encoding="utf-8")
print(summary_text, flush=True)
print("Saved optimal solutions to:", summary_path, flush=True)
print(results_df.to_string(index=False), flush=True)
