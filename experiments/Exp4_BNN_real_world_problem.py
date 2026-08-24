# Auto-generated from Exp4_BNN_real_world_problem.ipynb.
# Run with: python Exp4_BNN_real_world_problem.py

from pathlib import Path as _ExperimentsPath
import atexit as _experiments_atexit
import os as _experiments_os
import sys as _experiments_sys

for _experiments_thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    _experiments_os.environ.setdefault(_experiments_thread_env, "1")

EXPERIMENTS_DIR = _ExperimentsPath(__file__).resolve().parent
EXPERIMENTS_LOG_DIR = EXPERIMENTS_DIR / "logs"
EXPERIMENTS_RESULTS_DIR = EXPERIMENTS_DIR / "results"
EXPERIMENTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_EXPERIMENTS_METHOD_NAME = "BNN"
_EXPERIMENTS_LOG_PATH = EXPERIMENTS_LOG_DIR / (_EXPERIMENTS_METHOD_NAME + ".log")
_EXPERIMENTS_LOG_FILE = open(_EXPERIMENTS_LOG_PATH, "a", encoding="utf-8", buffering=1)
_EXPERIMENTS_STDOUT = _experiments_sys.stdout
_EXPERIMENTS_STDERR = _experiments_sys.stderr


class _ExperimentsTee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


_experiments_sys.stdout = _ExperimentsTee(_EXPERIMENTS_STDOUT, _EXPERIMENTS_LOG_FILE)
_experiments_sys.stderr = _ExperimentsTee(_EXPERIMENTS_STDERR, _EXPERIMENTS_LOG_FILE)


def _close_experiments_log():
    _experiments_sys.stdout = _EXPERIMENTS_STDOUT
    _experiments_sys.stderr = _EXPERIMENTS_STDERR
    _EXPERIMENTS_LOG_FILE.flush()
    _EXPERIMENTS_LOG_FILE.close()


_experiments_atexit.register(_close_experiments_log)
print(f"[experiments] logging to: {_EXPERIMENTS_LOG_PATH}")
print(f"[experiments] results dir: {EXPERIMENTS_RESULTS_DIR}")

# # Exp4 BNN real-world problem
#
# Run BNN across the configured real-world problems and print one summary block per problem.

# ### **Package**


# ---- notebook cell 2 ----
try:
    from google.colab import drive
except ModuleNotFoundError:
    drive = None

if drive is not None:
    drive.mount('/content/drive')

import importlib
import subprocess
import sys
sys.dont_write_bytecode = True
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*datetime\.datetime\.utcnow\(\) is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"paramz(\..*)?")
warnings.filterwarnings("ignore", message=r"Failed to config .* module.*", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Gym has been unmaintained.*", category=UserWarning)

print(sys.version)

DEPENDENCIES = {
    'pymoo': {
        'pip': 'pymoo==0.6.1.6',
        'checks': ('pymoo', 'pymoo.gradient.toolbox', 'pymoo.core.problem', 'pymoo.operators.sampling.lhs'),
        'pip_args': ('--force-reinstall',),
    },
    'pyro': {
        'pip': 'pyro-ppl',
        'checks': ('torch', 'pyro'),
    },
    'yaml': {
        'pip': 'pyyaml',
        'checks': ('yaml',),
    },
    'pandas': {
        'pip': 'pandas',
        'checks': ('pandas',),
    },
    'scipy': {
        'pip': 'scipy',
        'checks': ('scipy',),
    },
    'sklearn': {
        'pip': 'scikit-learn',
        'checks': ('sklearn',),
    },
}

def first_failed_import(modules):
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as err:
            return module_name, err
    return None, None

def install_dependency(package_name, config):
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        *config.get("pip_args", ()),
        config["pip"],
    ]
    print(f"Installing/updating {package_name}: {config['pip']}")
    subprocess.check_call(command)

packages_to_install = []
for package_name, config in DEPENDENCIES.items():
    failed_module, error = first_failed_import(config["checks"])
    if failed_module is None:
        print(f"{package_name} is available.")
    else:
        print(f"{package_name} check failed at {failed_module}: {error}")
        packages_to_install.append(package_name)

for package_name in packages_to_install:
    install_dependency(package_name, DEPENDENCIES[package_name])

if packages_to_install:
    raise RuntimeError(
        "Packages were installed or repaired. Restart the Colab runtime, "
        "then run again from this package cell before continuing."
    )

warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")


from pathlib import Path
import contextlib
import io

server_code_path = Path('/rds/projects/w/wangsu-building-automation/Huanbo/2026_real_world_problem')
colab_code_path = Path('/content/drive/MyDrive/2026 Real-wrold problem')
code_path = server_code_path if server_code_path.exists() else colab_code_path
repo_candidates = [Path.cwd().resolve(), Path.cwd().resolve().parent, code_path, server_code_path, colab_code_path]

for repo_root in repo_candidates:
    if (repo_root / 'src').exists() and str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))

import yaml
import numpy as np
from pymoo.operators.sampling.lhs import LHS

from src.data import generate_data
from src.experiment import (
    compute_surrogate_test_mse,
    run_experiment,
)
from src.metrics import get_igd_plus, get_metrics
from src.models import BNNRegressor, bnn_pred_mean_quantiles, train_bnn_models_for_calibration
from src.opt_problem import build_problem
from src.other_functions import mean_std
from result_recording import append_result_csv
from src.survival import Survival_dual_ranking, Survival_standard

warnings.filterwarnings("ignore", message=".*load_learner.*pickle.*")
np.set_printoptions(precision=3, suppress=True)

# ### **Main**

# ###### 1. Initial settings


# ---- notebook cell 5 ----
CONFIG_FILE_NAME = "config.yaml"


def _resolve_experiment_config_path(filename=CONFIG_FILE_NAME):
    roots = [
        EXPERIMENTS_DIR,
        Path.cwd().resolve(),
        Path.cwd().resolve().parent,
    ]
    try:
        roots.append(Path(code_path))
    except NameError:
        pass

    seen = set()
    for root in roots:
        for candidate in (root / "experiments" / filename, root / filename):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

    searched = "\n".join(str(path) for path in sorted(seen))
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n{searched}")


config_path = _resolve_experiment_config_path()
with open(config_path, "r", encoding="utf-8") as config_file:
    experiment_config = yaml.safe_load(config_file)

problem_names = experiment_config["problem_names"]
n_gen = experiment_config["n_gen"]
pop_size = experiment_config["pop_size"]
seed_start = experiment_config["seed_start"]
seed_end = experiment_config["seed_end"]
train_seed = experiment_config["train_seed"]
test_seed = experiment_config["test_seed"]
sample_size = experiment_config.get("sample_size")
def _sample_size_values(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


sample_sizes_by_problem = {
    str(name).strip().lower().replace("_", "-"): _sample_size_values(size)
    for name, size in experiment_config.get("sample_sizes_by_problem", {}).items()
}
show_seed_output = experiment_config["show_seed_output"]
optimizer_run_specs = [
    {"result_name": "NSGA-II", "optimizer_name": "NSGA-II", "use_dual_ranking": False},
    {"result_name": "Dual-Ranking+NSGA-II", "optimizer_name": "NSGA-II", "use_dual_ranking": True},
]
optimizer_names = [spec["result_name"] for spec in optimizer_run_specs]
dual_ranking_quantile = experiment_config.get("dual_ranking_quantile", 0.90)
method_name = "BNN"

print(f"Loaded experiment config: {config_path}")
print(f"Problems: {len(problem_names)} | seeds: range({seed_start}, {seed_end}) | n_gen: {n_gen} | pop_size: {pop_size}")
print(f"Optimizers: {optimizer_names}")

# ###### 2. Surrogate model and summary functions


# ---- notebook cell 7 ----
use_surrogate = "BNN_uncertainty"
bnn_hidden = (6, 3)
bnn_lr = 5e-4
bnn_max_steps = 50000
bnn_patience = 15
bnn_fixed_sigma = 0.05




def reset_experiment_random_state(seed, label=None):
    seed = int(seed)
    np.random.seed(seed)
def train_model_for_calibration(problem, sample_size, train_seed=42, test_seed=1):
    return train_bnn_models_for_calibration(
        problem=problem,
        sample_size=sample_size,
        train_seed=train_seed,
        test_seed=test_seed,
        hidden=bnn_hidden,
        lr=bnn_lr,
        max_steps=bnn_max_steps,
        patience=bnn_patience,
        fixed_sigma=bnn_fixed_sigma,
    )


# ---- notebook cell 8 ----
def build_benchmark_problem(problem_name):
    return build_problem(problem_name=problem_name)

def get_training_sample_sizes(problem_name, problem):
    if sample_size is not None:
        return [int(sample_size)]

    key = str(problem_name).strip().lower().replace("_", "-")
    if key in sample_sizes_by_problem:
        return sample_sizes_by_problem[key]

    return [max(11 * problem.n_var - 1, 100)]

def _format_result_value(value):
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


def _mean_std_text(values):
    values = np.asarray(values, dtype=float)
    return f"Mean = {_format_result_value(np.nanmean(values))}, Std = {_format_result_value(np.nanstd(values))}"


def print_problem_summary(problem_name, results):
    print(f"MSE_test: {_mean_std_text(results['mse_test_list'])}")
    print(f"MSE_sur_real: {_mean_std_text(results.get('mse_sur_real_list', results.get('sur_real_mse_list', [])))}")
    print(f"HV_sur: {_mean_std_text(results['hv_surrogate_list'])}")
    print(f"HV_real: {_mean_std_text(results['hv_real_list'])}")
    print(f"IGD+_sur: {_mean_std_text(results['igd_plus_surrogate_list'])}")
    print(f"IGD+_real: {_mean_std_text(results['igd_plus_real_list'])}")


def print_problem_header(problem_name, optimizer_name):
    print(f"\n============================== {problem_name} | {optimizer_name} ==============================")


def print_problem_result(problem_name, optimizer_name, results):
    try:
        print_problem_summary(problem_name, results)
    except Exception as err:
        print(f"Problem finished, but summary printing failed: {type(err).__name__}: {err}")
        print(f"Available result keys: {sorted(results.keys())}")
        raise


def run_problem(problem_name, sample_size_override=None, result_problem_name=None):
    result_problem_name = result_problem_name or problem_name
    current_use_surrogate = "BNN_uncertainty"
    reset_experiment_random_state(train_seed, f"{problem_name} surrogate/data")
    problem = build_benchmark_problem(
        problem_name,
    )
    current_sample_size = int(sample_size_override) if sample_size_override is not None else get_training_sample_sizes(problem_name, problem)[0]
    print(f"[{result_problem_name}] training sample size: {current_sample_size}")
    models, X_train, y_train, X_val, y_val, X_test, y_test = train_model_for_calibration(
        problem=problem,
        sample_size=current_sample_size,
        train_seed=train_seed,
        test_seed=test_seed,
    )
    model_f1, model_f2 = models[:2]

    metric_objective_values = y_train
    hv, obj_min, obj_max, _ = get_metrics(
        problem_name=problem_name,
        problem=problem,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        objective_values=metric_objective_values,
    )
    igd_plus, igd_plus_source = get_igd_plus(
        problem, obj_min, obj_max, metric_objective_values
    )
    print(f"[{result_problem_name}] obj_min: {np.array2string(np.asarray(obj_min, dtype=float), precision=3, suppress_small=False)}")
    print(f"[{result_problem_name}] obj_max: {np.array2string(np.asarray(obj_max, dtype=float), precision=3, suppress_small=False)}")

    offline_test_mse = compute_surrogate_test_mse(
        problem=problem,
        problem_name=problem_name,
        model_f1=model_f1,
        model_f2=model_f2,
        use_surrogate=current_use_surrogate,
        x_test=X_test,
        y_test=y_test,
        models=models,
    )

    problem_results = {}
    dual_ranking_survival = None
    for optimizer_spec in optimizer_run_specs:
        result_name = optimizer_spec["result_name"]
        optimizer_name = optimizer_spec["optimizer_name"]
        if optimizer_spec["use_dual_ranking"]:
            if dual_ranking_survival is None:
                print(f"Dual-ranking upper quantile: q={dual_ranking_quantile:.3f}")
                dual_ranking_survival = Survival_dual_ranking(alpha=dual_ranking_quantile)
            survival_function = dual_ranking_survival
        else:
            survival_function = Survival_standard()

        run_kwargs = dict(
            problem=problem,
            problem_name=problem_name,
            n_gen=n_gen,
            pop_size=pop_size,
            model_f1=model_f1,
            model_f2=model_f2,
            obj_min=obj_min,
            obj_max=obj_max,
            hv=hv,
            use_surrogate=current_use_surrogate,
            survival_function=survival_function,
            use_callback=False,
            seeds=range(seed_start, seed_end),
            optimizer_name=optimizer_name,
            print_normalization_info=False,
            mse_test=offline_test_mse,
            plot_seed_objectives=False,
            models=models,
            igd_plus_indicator=igd_plus,
            igd_plus_source=igd_plus_source,
        )

        print_problem_header(result_problem_name, result_name)
        results = run_experiment(**run_kwargs)

        problem_results[result_name] = results
        print_problem_result(result_problem_name, result_name, results)

    return problem_results

# ###### 3. Batch optimization


# ---- notebook cell 10 ----
import concurrent.futures
import gc
import multiprocessing as mp

EXPERIMENTS_MAX_PROBLEM_WORKERS = 4


def _sanitize_parallel_results(problem_results):
    sanitized = {}
    for optimizer_name, results in problem_results.items():
        run_details = []
        for detail in results.get("run_details", []):
            run_details.append(
                {
                    "seed": detail.get("seed"),
                    "time": detail.get("time"),
                    "mse_test": detail.get("mse_test"),
                    "offline_test_mse": detail.get("offline_test_mse", detail.get("mse_test")),
                    "mse_sur_real": detail.get("mse_sur_real", detail.get("sur_real_mse")),
                    "sur_real_mse": detail.get("mse_sur_real", detail.get("sur_real_mse")),
                    "hv_surrogate": detail.get("hv_surrogate"),
                    "hv_real": detail.get("hv_real"),
                    "hv_bounds_check": detail.get("hv_bounds_check"),
                    "solution_count": detail.get("solution_count"),
                    "no_feasible_solution": detail.get("no_feasible_solution", False),
                    "no_feasible_reason": detail.get("no_feasible_reason"),
                }
            )
        sanitized[optimizer_name] = {
            "hv_surrogate_list": list(results.get("hv_surrogate_list", [])),
            "hv_real_list": list(results.get("hv_real_list", [])),
            "hv_real_count_list": list(results.get("hv_real_count_list", [])),
            "mse_test_list": list(results.get("mse_test_list", [])),
            "mse_sur_real_list": list(results.get("mse_sur_real_list", results.get("sur_real_mse_list", []))),
            "sur_real_mse_list": list(results.get("mse_sur_real_list", results.get("sur_real_mse_list", []))),
            "run_details": run_details,
        }
    return sanitized


def _run_problem_task(task):
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    problem_name, configured_sample_size, result_problem_name = task
    problem_results = run_problem(
        problem_name,
        sample_size_override=configured_sample_size,
        result_problem_name=result_problem_name,
    )
    return result_problem_name, _sanitize_parallel_results(problem_results)


all_results = {}
metrics_summary_tables = []
result_problem_names = []
result_csv_path = EXPERIMENTS_RESULTS_DIR / "results_real_world.csv"
problem_tasks = []

for problem_index, problem_name in enumerate(problem_names, start=1):
    problem = build_benchmark_problem(
        problem_name,
    )
    configured_sample_sizes = get_training_sample_sizes(problem_name, problem)
    for configured_sample_size in configured_sample_sizes:
        result_problem_name = (
            problem_name
            if len(configured_sample_sizes) == 1
            else f"{problem_name}_n{configured_sample_size}"
        )
        problem_tasks.append((problem_name, configured_sample_size, result_problem_name))
    del problem

worker_count = min(EXPERIMENTS_MAX_PROBLEM_WORKERS, len(problem_tasks))
print(f"[experiments] running {method_name} problems with {worker_count} worker processes")


def _record_problem_result(result_problem_name, problem_results):
    result_problem_names.append(result_problem_name)
    one_problem_results = {result_problem_name: problem_results}
    metrics_summary_tables.append(
        append_result_csv(
            method_name=method_name,
            optimizer_names=optimizer_names,
            problem_names=[result_problem_name],
            all_results=one_problem_results,
            result_csv_path=result_csv_path,
        )
    )
    del one_problem_results, problem_results
    gc.collect()


if worker_count <= 1:
    for task in problem_tasks:
        _record_problem_result(*_run_problem_task(task))
else:
    mp_context = mp.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=mp_context,
    ) as executor:
        future_to_task = {executor.submit(_run_problem_task, task): task for task in problem_tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            _record_problem_result(*future.result())

gc.collect()

problem_names = result_problem_names
if metrics_summary_tables:
    import pandas as pd
    metrics_summary_table = pd.concat(metrics_summary_tables, ignore_index=True)
else:
    import pandas as pd
    metrics_summary_table = pd.DataFrame()
all_results = {}


# ---- notebook cell 11 ----
# Results were written per problem during batch optimization.
print(metrics_summary_table.to_string(index=False))
