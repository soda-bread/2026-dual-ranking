"""Batch runners for Exp5-8 using the uploaded baseline method implementations."""

from __future__ import annotations

import importlib.util
import concurrent.futures
import gc
import multiprocessing as mp
import random
import sys
import time
import types
from datetime import datetime
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from sklearn.metrics import mean_squared_error


REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_ROOT = REPO_ROOT / "experiments" / "baseline"
PROB_VENDOR_ROOT = BASELINE_ROOT / "Prob-RVEA and Prob-MOEA-D 2022"
TGPR_VENDOR_ROOT = BASELINE_ROOT / "TGPR-MO 2023"
DEFAULT_CONFIG_PATH = REPO_ROOT / "experiments" / "config.yaml"

TOTAL_FUNCTION_EVALUATIONS = 10_000
POPULATION_SIZE = 100
LEGACY_PROB_MOEAD_POPULATION_SIZE = 50
DDMOEA_GAN_N_GEN = 100
_REFERENCE_DIRECTION_CACHE = {}


def _desdeo_termination_kwargs(n_gen, official_pool_mode):
    """Keep legacy FE termination or match pymoo's generation semantics."""

    n_gen = int(n_gen)
    if n_gen < 1:
        raise ValueError("n_gen must be positive.")
    if official_pool_mode:
        return {
            "n_iterations": 1,
            "n_gen_per_iter": n_gen - 1,
            "total_function_evaluations": 0,
        }
    return {"total_function_evaluations": TOTAL_FUNCTION_EVALUATIONS}


def _desdeo_run_counters(evolver):
    """Report the initial generation plus completed offspring generations."""

    return {
        "optimizer_generation_count": 1
        + int(getattr(evolver, "_current_gen_count", 0)),
        "surrogate_evaluation_count": int(
            getattr(evolver, "_function_evaluation_count", 0)
        ),
    }


def _crowding_distance(objectives):
    """Deterministic NSGA-II crowding distance for one Pareto front."""

    objectives = np.asarray(objectives, dtype=float)
    n_points = len(objectives)
    if n_points <= 2:
        return np.full(n_points, np.inf)
    distance = np.zeros(n_points, dtype=float)
    for objective_index in range(objectives.shape[1]):
        order = np.argsort(objectives[:, objective_index], kind="mergesort")
        distance[order[[0, -1]]] = np.inf
        span = (
            objectives[order[-1], objective_index]
            - objectives[order[0], objective_index]
        )
        if not np.isfinite(span) or span <= 0.0:
            continue
        distance[order[1:-1]] += (
            objectives[order[2:], objective_index]
            - objectives[order[:-2], objective_index]
        ) / span
    return distance


def _rank_and_crowding_indices(objectives, target_size):
    """Select a fixed number by Pareto rank, then deterministic crowding."""

    objectives = np.asarray(objectives, dtype=float)
    target_size = int(target_size)
    if target_size < 1 or target_size > len(objectives):
        raise ValueError("target_size must be within the candidate count.")
    selected = []
    for front in NonDominatedSorting().do(objectives):
        front = np.asarray(front, dtype=int)
        remaining = target_size - len(selected)
        if len(front) <= remaining:
            selected.extend(front.tolist())
        else:
            crowding = _crowding_distance(objectives[front])
            order = np.lexsort((front, -crowding))
            selected.extend(front[order[:remaining]].tolist())
            break
        if len(selected) == target_size:
            break
    return np.asarray(selected, dtype=int)


def _selection_safe_objectives(objectives):
    """Make deterministic fallback selection robust to invalid predictions."""

    objectives = np.asarray(objectives, dtype=float)
    if objectives.ndim != 2:
        raise ValueError("Selection objectives must be a two-dimensional array.")
    finite_rows = np.all(np.isfinite(objectives), axis=1)
    if np.all(finite_rows):
        return objectives
    finite_values = objectives[finite_rows]
    replacement = (
        np.max(finite_values, axis=0) + 1.0
        if len(finite_values)
        else np.ones(objectives.shape[1], dtype=float)
    )
    safe = objectives.copy()
    safe[~finite_rows] = replacement
    return safe


def _install_exact_reference_vectors(evolver, n_objectives, population_size):
    """Install exactly one deterministic reference vector per live individual."""

    population_size = int(population_size)
    cache_key = (int(n_objectives), population_size)
    if cache_key not in _REFERENCE_DIRECTION_CACHE:
        _REFERENCE_DIRECTION_CACHE[cache_key] = np.asarray(
            get_reference_directions(
                "energy",
                cache_key[0],
                population_size,
                seed=1,
            ),
            dtype=float,
        )
    directions = _REFERENCE_DIRECTION_CACHE[cache_key].copy()
    if directions.shape != (population_size, int(n_objectives)):
        raise ValueError(
            f"Expected {population_size} reference vectors with "
            f"{n_objectives} objectives; got {directions.shape}."
        )

    vectors = evolver.reference_vectors
    vectors.values = directions.copy()
    vectors.values_planar = directions.copy()
    vectors.normalize()
    vectors.initial_values = vectors.values.copy()
    vectors.initial_values_planar = vectors.values_planar.copy()
    vectors.neighbouring_angles()

    # MOEA/D stores an index-based neighborhood matrix during construction.
    # Rebuild it after replacing the default 50/105-vector simplex lattice.
    if hasattr(evolver, "neighborhoods"):
        n_neighbors = min(int(evolver.n_neighbors), population_size)
        distances = np.linalg.norm(
            vectors.values[:, None, :] - vectors.values[None, :, :],
            axis=2,
        )
        evolver.n_neighbors = n_neighbors
        evolver.neighborhoods = np.argsort(
            distances,
            axis=1,
            kind="mergesort",
        )[:, :n_neighbors]


def _install_population_size_preserving_selection(evolver, population_size):
    """Preserve APD selections and deterministically fill empty vectors."""

    operator = evolver.selection_operator
    if getattr(operator, "_experiment_fixed_population_selection", False):
        return
    target_size = int(population_size)
    original_do = operator.do

    def fixed_do(self, pop, vectors):
        raw = np.asarray(original_do(pop, vectors), dtype=int).reshape(-1)
        selected = []
        seen = set()
        for index in raw:
            index = int(index)
            if 0 <= index < len(pop.individuals) and index not in seen:
                seen.add(index)
                selected.append(index)

        objectives = _selection_safe_objectives(pop.fitness)
        if len(selected) > target_size:
            local = _rank_and_crowding_indices(
                objectives[np.asarray(selected, dtype=int)],
                target_size,
            )
            selected = np.asarray(selected, dtype=int)[local].tolist()
        elif len(selected) < target_size:
            remaining = np.asarray(
                [i for i in range(len(objectives)) if i not in seen],
                dtype=int,
            )
            needed = target_size - len(selected)
            if len(remaining) < needed:
                raise ValueError(
                    f"APD population has only {len(selected) + len(remaining)} "
                    f"candidates; {target_size} are required."
                )
            local = _rank_and_crowding_indices(objectives[remaining], needed)
            selected.extend(remaining[local].tolist())
        return np.asarray(selected, dtype=int)

    operator.do = types.MethodType(fixed_do, operator)
    operator._experiment_fixed_population_selection = True


def _install_complete_generation_time_penalty(
    evolver,
    *,
    restore_rvea_objective_factor=False,
):
    """Evaluate APD at generations 1..T instead of 0..T-1.

    DESDEO invokes selection before incrementing ``_current_gen_count``.  The
    official-pool protocol therefore needs the upcoming generation in the
    RVEA time ratio.  Some TGPR vendor code also clips the *whole* ``M*t^alpha``
    factor to one; only ``t`` should be clipped according to the RVEA formula.
    """

    total_generations = int(getattr(evolver, "total_gen_count", 0))
    if total_generations <= 0:
        return

    def generation_time():
        return float(
            np.clip(
                (int(getattr(evolver, "_current_gen_count", 0)) + 1)
                / total_generations,
                0.0,
                1.0,
            )
        )

    evolver.time_penalty_function = generation_time
    operator = evolver.selection_operator
    operator.time_penalty_function = generation_time

    if restore_rvea_objective_factor:
        def partial_penalty_factor(self):
            time_ratio = float(np.clip(self.time_penalty_function(), 0.0, 1.0))
            return (time_ratio ** self.alpha) * self.n_of_objectives

        operator._partial_penalty_factor = types.MethodType(
            partial_penalty_factor,
            operator,
        )


def _repair_values_for_problem(benchmark_problem, values):
    """Apply finite box repair plus any official problem-specific repair."""

    from src.offline_moo_adapter import repair_offline_moo_decisions

    values = np.asarray(values, dtype=float)
    was_one_dimensional = values.ndim == 1
    values_2d = values.reshape(1, -1) if was_one_dimensional else values.copy()
    values_2d = _repair_decision_vectors(
        values_2d,
        benchmark_problem.xl,
        benchmark_problem.xu,
    )
    values_2d = np.asarray(
        repair_offline_moo_decisions(benchmark_problem, values_2d),
        dtype=float,
    )
    return values_2d[0] if was_one_dimensional else values_2d


def _install_problem_decision_repair(
    population,
    benchmark_problem,
    *,
    repair_population_add,
):
    """Attach the same official-domain repair used by the Pymoo methods."""

    def repair(self, individual):
        return _repair_values_for_problem(benchmark_problem, individual)

    population.repair = types.MethodType(repair, population)
    if repair_population_add:
        original_add = population.add

        def add(self, offsprings, use_surrogates=False):
            repaired = _repair_values_for_problem(benchmark_problem, offsprings)
            return original_add(repaired, use_surrogates)

        population.add = types.MethodType(add, population)


def _validate_official_desdeo_budget(evolver, n_gen, population_size):
    """Fail loudly if a baseline silently violates the shared protocol."""

    counters = _desdeo_run_counters(evolver)
    expected_generations = int(n_gen)
    expected_evaluations = expected_generations * int(population_size)
    live_count = int(len(evolver.population.individuals))
    if counters["optimizer_generation_count"] != expected_generations:
        raise RuntimeError(
            f"Expected {expected_generations} generations; got "
            f"{counters['optimizer_generation_count']}."
        )
    if counters["surrogate_evaluation_count"] != expected_evaluations:
        raise RuntimeError(
            f"Expected {expected_evaluations} surrogate evaluations; got "
            f"{counters['surrogate_evaluation_count']}."
        )
    if live_count != int(population_size):
        raise RuntimeError(
            f"Expected final live population {population_size}; got {live_count}."
        )
    return counters


def _fixed_desdeo_output(evolver, surrogate_problem, target_size):
    """Return the strict final live population used by official-pool runs."""

    target_size = int(target_size)
    solution = np.asarray(evolver.population.individuals, dtype=float)
    if solution.ndim == 1:
        solution = solution.reshape(1, -1)
    if len(solution) != target_size:
        raise ValueError(
            f"Expected {target_size} final DESDEO candidates; got {len(solution)}."
        )
    objectives = _predict_desdeo_surrogate(surrogate_problem, solution)
    return solution, objectives

def set_optimization_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    try:
        import pyro
        pyro.set_rng_seed(seed)
    except ImportError:
        pass
def install_pydoe2_imp_compatibility():
    """Allow the last pyDOE2 release to import on Python 3.12 and newer."""
    if "imp" not in sys.modules and importlib.util.find_spec("imp") is None:
        sys.modules["imp"] = types.ModuleType("imp")


install_pydoe2_imp_compatibility()


def _activate_vendor(vendor_root):
    vendor_root = str(Path(vendor_root).resolve())
    for existing_root in (str(PROB_VENDOR_ROOT.resolve()), str(TGPR_VENDOR_ROOT.resolve())):
        while existing_root in sys.path:
            sys.path.remove(existing_root)
    sys.path.insert(0, vendor_root)

    prefixes = ("desdeo_emo", "desdeo_problem", "desdeo_tools", "framework")
    for module_name in list(sys.modules):
        if any(
            module_name == prefix or module_name.startswith(prefix + ".")
            for prefix in prefixes
        ):
            del sys.modules[module_name]


def load_config(config_path=None):
    resolved_path = Path(config_path or DEFAULT_CONFIG_PATH).resolve()
    with open(resolved_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    print(f"Loaded baseline experiment config: {resolved_path}")
    return config


def build_benchmark_problem(problem_name, config):
    from src.opt_problem import build_problem

    return build_problem(problem_name=problem_name)


def configured_sample_size(problem, config, problem_name=None):
    sample_size = config.get("sample_size")
    if sample_size is not None:
        return int(sample_size)

    problem_key = str(problem_name or getattr(problem, "name", "")).strip().lower().replace("_", "-")
    sample_sizes_by_problem = {
        str(name).strip().lower().replace("_", "-"): (
            int(size[0]) if isinstance(size, (list, tuple)) else int(size)
        )
        for name, size in config.get("sample_sizes_by_problem", {}).items()
    }
    return sample_sizes_by_problem.get(
        problem_key,
        max(11 * int(problem.n_var) - 1, POPULATION_SIZE),
    )


def generate_offline_dataset(problem, config, problem_name=None):
    sample_size = configured_sample_size(problem, config, problem_name=problem_name)
    train_seed = int(config["train_seed"])
    np.random.seed(train_seed)
    x_data = LHS()(problem, sample_size, seed=train_seed).get("X")
    y_data = problem.evaluate(x_data, return_values_of=["F"])
    print(
        f"Offline dataset: LHS | sample_size={sample_size} | "
        f"train_seed={train_seed}"
    )
    return x_data, y_data


def generate_offline_test_dataset(problem, config, problem_name=None):
    sample_size = configured_sample_size(problem, config, problem_name=problem_name)
    test_seed = int(config["test_seed"])
    np.random.seed(test_seed)
    x_test = LHS()(problem, sample_size, seed=test_seed).get("X")
    y_test = problem.evaluate(x_test, return_values_of=["F"])
    return x_test, y_test


def initial_population_from_offline_dataset(
    x_data,
    population_size=POPULATION_SIZE,
    seed=None,
):
    x_data = np.asarray(x_data)
    population_size = int(population_size)
    if seed is None:
        if x_data.shape[0] < population_size:
            raise ValueError(
                f"Offline dataset contains {x_data.shape[0]} points, but "
                f"{population_size} initial solutions are required."
            )
        return x_data[:population_size].copy()

    rng = np.random.default_rng(int(seed))
    if len(x_data) >= population_size:
        indices = rng.permutation(len(x_data))[:population_size]
    else:
        prefix = rng.permutation(len(x_data))
        indices = np.resize(prefix, population_size)
    return x_data[indices].copy()


def _repair_decision_vectors(decision_vectors, lower_bounds, upper_bounds):
    decision_vectors = np.asarray(decision_vectors, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    midpoint = (lower_bounds + upper_bounds) / 2
    decision_vectors = np.where(np.isfinite(decision_vectors), decision_vectors, midpoint)
    return np.minimum(np.maximum(decision_vectors, lower_bounds), upper_bounds)


def _repair_decision_vectors_for_evaluation(
    decision_vectors,
    lower_bounds,
    upper_bounds,
    relative_epsilon=1e-9,
):
    decision_vectors = np.asarray(decision_vectors, dtype=float)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    span = upper_bounds - lower_bounds
    epsilon = np.where(np.isfinite(span) & (span > 0.0), span * relative_epsilon, 0.0)
    safe_lower = lower_bounds + epsilon
    safe_upper = upper_bounds - epsilon
    return _repair_decision_vectors(decision_vectors, safe_lower, safe_upper)


def _repair_and_predict_final_population(surrogate_problem, individuals):
    solution = _repair_decision_vectors_for_evaluation(
        individuals,
        surrogate_problem.get_variable_lower_bounds(),
        surrogate_problem.get_variable_upper_bounds(),
    )
    obj = _predict_desdeo_surrogate(surrogate_problem, solution)
    return solution, obj


def _install_prob_population_numeric_compatibility():
    population_module = importlib.import_module("desdeo_emo.population.Population")
    population_class = population_module.Population
    if getattr(population_class, "_offline_finite_repair_installed", False):
        return

    original_add = population_class.add

    def add(self, offsprings, use_surrogates=False):
        offsprings = _repair_decision_vectors(
            offsprings,
            self.lower_limits,
            self.upper_limits,
        )
        return original_add(self, offsprings, use_surrogates)

    population_class.add = add
    population_class._offline_finite_repair_installed = True


def _install_gpy_numpy_compatibility():
    """Restore the NumPy module alias imported by older GPy releases."""
    sys.modules.setdefault("numpy.linalg.linalg", np.linalg)


def _install_tgpr_numeric_compatibility():
    population_module = importlib.import_module("desdeo_emo.population.Population")
    population_class = population_module.Population
    if not getattr(population_class, "_offline_finite_repair_installed", False):
        original_add = population_class.add

        def add(self, offsprings, use_surrogates=False):
            offsprings = _repair_decision_vectors(
                offsprings,
                self.lower_limits,
                self.upper_limits,
            )
            return original_add(self, offsprings, use_surrogates)

        population_class.add = add
        population_class._offline_finite_repair_installed = True

    surrogate_module = importlib.import_module(
        "desdeo_problem.surrogatemodels.surrogate_treedGP"
    )
    surrogate_class = surrogate_module.treeGP
    if not getattr(surrogate_class, "_offline_finite_repair_installed", False):
        original_predict = surrogate_class.predict

        def predict(self, x):
            x = np.asarray(x, dtype=float)
            training_x = np.asarray(self.X, dtype=float)
            training_min = np.nanmin(training_x, axis=0)
            training_max = np.nanmax(training_x, axis=0)
            training_midpoint = (training_min + training_max) / 2
            x = np.where(np.isfinite(x), x, training_midpoint)
            x = np.minimum(np.maximum(x, training_min), training_max)
            model_x = (
                self._transform_inputs(x)
                if hasattr(self, "_transform_inputs")
                else x
            )
            try:
                y_mean, y_stdev = original_predict(self, x)
            except ValueError:
                y_mean = np.asarray(self.regr.predict(X=model_x), dtype=float)
                y_stdev = None
            y_mean = np.asarray(y_mean, dtype=float)
            non_finite = ~np.isfinite(y_mean)
            if np.any(non_finite):
                tree_prediction = np.asarray(
                    self.regr.predict(X=model_x),
                    dtype=float,
                )
                y_mean = np.where(non_finite, tree_prediction, y_mean)
            return y_mean, y_stdev

        surrogate_class.predict = predict
        surrogate_class._offline_finite_repair_installed = True


def _install_prob_moead_numeric_compatibility(population):
    def repair(self, individual):
        return _repair_decision_vectors(
            individual,
            self.problem.get_variable_lower_bounds(),
            self.problem.get_variable_upper_bounds(),
        )

    population.repair = types.MethodType(repair, population)


def _tgpr_default_population_size(surrogate_problem):
    n_objectives = int(surrogate_problem.n_of_objectives)
    lattice_res_options = [49, 13, 7, 5, 4, 3, 3, 3, 3]
    if n_objectives < 11:
        lattice_resolution = lattice_res_options[n_objectives - 2]
    else:
        lattice_resolution = 3
    return comb(lattice_resolution + n_objectives - 1, n_objectives - 1)


def _build_tgpr_rvea(
    RVEA,
    surrogate_problem,
    initial_population,
    population_size=None,
    total_function_evaluations=TOTAL_FUNCTION_EVALUATIONS,
    n_iterations=10,
    n_gen_per_iter=100,
):
    population_size = (
        _tgpr_default_population_size(surrogate_problem)
        if population_size is None
        else int(population_size)
    )
    init_pop = np.asarray(initial_population, dtype=float)[:population_size].copy()
    return RVEA(
        surrogate_problem,
        use_surrogates=True,
        population_size=population_size,
        population_params={
            "design": "InitSamples",
            "init_pop": init_pop,
        },
        n_iterations=int(n_iterations),
        n_gen_per_iter=int(n_gen_per_iter),
        total_function_evaluations=int(total_function_evaluations),
    )


def _build_kriging_surrogate(
    benchmark_problem,
    x_data,
    y_data,
    *,
    official_pool_mode=False,
    model_seed=None,
):
    _activate_vendor(PROB_VENDOR_ROOT)
    from desdeo_problem.Problem import DataProblem
    from desdeo_problem.surrogatemodels.SurrogateKriging import SurrogateKriging

    n_var = benchmark_problem.n_var
    n_obj = benchmark_problem.n_obj
    x_names = [f"x{i}" for i in range(1, n_var + 1)]
    y_names = [f"f{i}" for i in range(1, n_obj + 1)]
    bounds = pd.DataFrame(
        np.vstack((benchmark_problem.xl, benchmark_problem.xu)),
        columns=x_names,
        index=["lower_bound", "upper_bound"],
    )
    data = pd.DataFrame(np.hstack((x_data, y_data)), columns=x_names + y_names)
    surrogate_problem = DataProblem(
        data=data,
        variable_names=x_names,
        objective_names=y_names,
        bounds=bounds,
    )
    model_parameters = None
    if official_pool_mode:
        model_parameters = {
            "normalize_inputs": True,
            "normalize_y": True,
            "alpha": 1e-8,
            "random_state": int(model_seed),
        }
    surrogate_problem.train(
        SurrogateKriging,
        model_parameters=model_parameters,
    )
    return surrogate_problem


def _metric_context(
    problem_name,
    benchmark_problem,
    objective_values=None,
    fallback_reference_values=None,
):
    from src.metrics import get_igd_plus, get_metrics

    hv, obj_min, obj_max, _ = get_metrics(
        problem_name=problem_name,
        problem=benchmark_problem,
        n_var=benchmark_problem.n_var,
        n_obj=benchmark_problem.n_obj,
        objective_values=objective_values,
    )
    igd_plus, igd_plus_source = get_igd_plus(
        benchmark_problem,
        obj_min,
        obj_max,
        objective_values,
        fallback_reference_values=fallback_reference_values,
    )
    return hv, igd_plus, igd_plus_source, obj_min, obj_max, obj_min


def _evaluate_solution(
    benchmark_problem,
    solution,
    obj,
    hv,
    igd_plus,
    igd_plus_source,
    obj_min,
    obj_max,
    final_output_size=None,
):
    n_var = int(getattr(benchmark_problem, "n_var", 0))
    n_obj = int(getattr(benchmark_problem, "n_obj", 0))
    submitted_solution_count = 0

    def zero_result(reason, non_finite_candidate_count=0):
        print(f"No feasible final solution for metric evaluation: {reason}. Metrics set to NaN.")
        return {
            "solution_count": 0,
            "submitted_solution_count": int(submitted_solution_count),
            "final_output_target": final_output_size,
            "mse_sur_real": np.nan,
            "sur_real_mse": np.nan,
            "hv_surrogate": np.nan,
            "hv_real": np.nan,
            "igd_plus_surrogate": np.nan,
            "igd_plus_real": np.nan,
            "igd_plus_source": igd_plus_source,
            "hv_sur_gap": np.nan,
            "non_finite_candidate_count": int(non_finite_candidate_count),
            "no_feasible_solution": True,
            "no_feasible_reason": str(reason),
        }

    if solution is None or obj is None:
        return zero_result("missing final X or F")

    solution = np.asarray(solution, dtype=float)
    obj = np.asarray(obj, dtype=float)
    if solution.size == 0 or obj.size == 0:
        return zero_result("empty final X or F")
    if solution.ndim == 1:
        solution = solution.reshape(1, -1)
    if obj.ndim == 1:
        obj = obj.reshape(1, -1)
    submitted_solution_count = int(len(solution))
    if len(obj) != submitted_solution_count:
        return zero_result("final X and F row counts do not match")
    if (
        final_output_size is not None
        and submitted_solution_count != int(final_output_size)
    ):
        return zero_result(
            f"expected {int(final_output_size)} final candidates, got "
            f"{submitted_solution_count}"
        )

    try:
        from src.offline_moo_adapter import (
            evaluate_offline_moo_objectives_and_feasibility,
        )

        f_real, custom_feasible = evaluate_offline_moo_objectives_and_feasibility(
            benchmark_problem,
            solution,
        )
        f_real = np.asarray(f_real, dtype=float)
    except Exception as err:
        return zero_result(f"real evaluation failed: {type(err).__name__}: {err}")
    if f_real.ndim == 1:
        f_real = f_real.reshape(1, -1)

    finite_mask = (
        np.all(np.isfinite(solution), axis=1)
        & np.all(np.isfinite(obj), axis=1)
        & np.all(np.isfinite(f_real), axis=1)
    )
    if custom_feasible is not None:
        finite_mask &= np.asarray(custom_feasible, dtype=bool)
    non_finite_candidate_count = int(np.count_nonzero(~finite_mask))
    if non_finite_candidate_count:
        solution = solution[finite_mask]
        obj = obj[finite_mask]
        f_real = f_real[finite_mask]
    if solution.shape[0] == 0:
        return zero_result(
            "all optimized candidates are non-finite or infeasible",
            non_finite_candidate_count,
        )
    obj_normalized = (obj - obj_min) / (obj_max - obj_min)
    f_real_normalized = (f_real - obj_min) / (obj_max - obj_min)
    result = {
        "solution_count": int(solution.shape[0]),
        "submitted_solution_count": submitted_solution_count,
        "final_output_target": final_output_size,
        "mse_sur_real": _finite_mean_squared_error(f_real, obj),
        "sur_real_mse": _finite_mean_squared_error(f_real, obj),
        "hv_surrogate": float(hv.do(obj_normalized)),
        "hv_real": float(hv.do(f_real_normalized)),
        "igd_plus_surrogate": float(igd_plus.do(obj_normalized)),
        "igd_plus_real": float(igd_plus.do(f_real_normalized)),
        "igd_plus_source": igd_plus_source,
        "non_finite_candidate_count": non_finite_candidate_count,
    }
    result["hv_sur_gap"] = abs(result["hv_real"] - result["hv_surrogate"])
    return result


def _predict_desdeo_surrogate(surrogate_problem, x_data):
    evaluation = surrogate_problem.evaluate(x_data, use_surrogate=True)
    return np.asarray(evaluation.objectives, dtype=float)


def _finite_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    finite_mask = np.all(np.isfinite(y_true), axis=1) & np.all(
        np.isfinite(y_pred),
        axis=1,
    )
    if not np.any(finite_mask):
        return float("nan")
    return float(mean_squared_error(y_true[finite_mask], y_pred[finite_mask]))


def _format_result_value(value, digits=3):
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
    return f"{value:.{int(digits)}e}"


def _compute_offline_test_mse(benchmark_problem, config, predict_fn, problem_name=None):
    x_test, y_test = generate_offline_test_dataset(
        benchmark_problem,
        config,
        problem_name=problem_name,
    )
    y_pred = np.asarray(predict_fn(x_test), dtype=float)
    return _finite_mean_squared_error(y_test, y_pred)


def print_seed_result(result, seed, elapsed, problem_name, method_name):
    print(
        f"Seed {seed} | {problem_name} | {method_name} | "
        f"Time: {_format_result_value(elapsed)}s | "
        f"MSE_test: {_format_result_value(result.get('offline_test_mse', float('nan')))} | "
        f"MSE_sur_real: {_format_result_value(result.get('mse_sur_real', float('nan')))} | "
        f"HV_sur: {_format_result_value(result.get('hv_surrogate', float('nan')))} | "
        f"HV_real: {_format_result_value(result.get('hv_real', float('nan')))} | "
        f"IGD+_sur: {_format_result_value(result.get('igd_plus_surrogate', float('nan')))} | "
        f"IGD+_real: {_format_result_value(result.get('igd_plus_real', float('nan')))}"
    )



def _aggregate_seed_results(run_results):
    sanitized_details = []
    for item in run_results:
        sanitized_details.append(
            {
                "seed": item.get("seed"),
                "time": item.get("time"),
                "offline_test_mse": item.get("offline_test_mse"),
                "mse_test": item.get("mse_test", item.get("offline_test_mse")),
                "mse_sur_real": item.get("mse_sur_real", item.get("sur_real_mse")),
                "sur_real_mse": item.get("mse_sur_real", item.get("sur_real_mse")),
                "hv_surrogate": item.get("hv_surrogate"),
                "hv_real": item.get("hv_real"),
                "igd_plus_surrogate": item.get("igd_plus_surrogate"),
                "igd_plus_real": item.get("igd_plus_real"),
                "igd_plus_source": item.get("igd_plus_source"),
                "hv_bounds_check": item.get("hv_bounds_check"),
                "solution_count": item.get("solution_count"),
                "no_feasible_solution": item.get("no_feasible_solution", False),
                "no_feasible_reason": item.get("no_feasible_reason"),
                "optimizer_generation_count": item.get(
                    "optimizer_generation_count"
                ),
                "surrogate_evaluation_count": item.get(
                    "surrogate_evaluation_count"
                ),
            }
        )
    return {
        "hv_surrogate_list": [item.get("hv_surrogate", float("nan")) for item in run_results],
        "hv_real_list": [item.get("hv_real", float("nan")) for item in run_results],
        "igd_plus_surrogate_list": [
            item.get("igd_plus_surrogate", float("nan")) for item in run_results
        ],
        "igd_plus_real_list": [
            item.get("igd_plus_real", float("nan")) for item in run_results
        ],
        "mse_test_list": [item.get("offline_test_mse", float("nan")) for item in run_results],
        "mse_sur_real_list": [item.get("mse_sur_real", item.get("sur_real_mse", float("nan"))) for item in run_results],
        "sur_real_mse_list": [item.get("mse_sur_real", item.get("sur_real_mse", float("nan"))) for item in run_results],
        "run_details": sanitized_details,
    }


def _run_suite_problem(task):
    problem_name, method_name, run_problem, config, seeds = task
    print(f"\nPreparing {problem_name} for {method_name}")
    benchmark_problem = build_benchmark_problem(problem_name, config)
    run_results = run_problem(problem_name, benchmark_problem, config, seeds)
    problem_results = {method_name: _aggregate_seed_results(run_results)}
    del run_results, benchmark_problem
    gc.collect()
    return problem_name, problem_results


def _run_suite(method_name, run_problem, config_path=None, max_workers=None):
    try:
        from result_recording import append_result_csv
    except ImportError:
        experiments_dir = REPO_ROOT / "experiments"
        if str(experiments_dir) not in sys.path:
            sys.path.insert(0, str(experiments_dir))
        from result_recording import append_result_csv

    config = load_config(config_path)
    output_dir = Path(config_path or DEFAULT_CONFIG_PATH).resolve().parent
    seeds = range(int(config["seed_start"]), int(config["seed_end"]))
    all_results = {}
    problem_tasks = [
        (problem_name, method_name, run_problem, config, seeds)
        for problem_name in config["problem_names"]
    ]
    worker_count = int(max_workers or 1)
    worker_count = max(1, min(worker_count, len(problem_tasks)))

    def record_problem_result(problem_name, problem_results):
        all_results[problem_name] = problem_results
        append_result_csv(
            method_name=method_name,
            optimizer_names=[method_name],
            problem_names=[problem_name],
            all_results={problem_name: all_results[problem_name]},
            result_csv_path=output_dir / "results" / "results_real_world.csv",
        )
        del all_results[problem_name]
        del problem_results
        gc.collect()

    if worker_count > 1:
        print(f"Running {method_name} problems with {worker_count} worker processes")
        mp_context = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_context,
        ) as executor:
            future_to_problem = {
                executor.submit(_run_suite_problem, task): task[0]
                for task in problem_tasks
            }
            for future in concurrent.futures.as_completed(future_to_problem):
                record_problem_result(*future.result())
    else:
        for task in problem_tasks:
            record_problem_result(*_run_suite_problem(task))

    all_results.clear()
    gc.collect()
    return all_results


def _run_prob_rvea_problem(problem_name, benchmark_problem, config, seeds):
    x_data, y_data = generate_offline_dataset(
        benchmark_problem,
        config,
        problem_name=problem_name,
    )
    official_pool_mode = config.get("dataset_source") == "official_pool"
    population_size = int(config.get("pop_size", POPULATION_SIZE)) if official_pool_mode else POPULATION_SIZE
    termination_kwargs = _desdeo_termination_kwargs(
        config.get("n_gen", 100), official_pool_mode
    )
    initial_population = (
        None
        if official_pool_mode
        else initial_population_from_offline_dataset(x_data, population_size)
    )
    model_seed = int(config.get("model_seed", config["train_seed"]))
    if official_pool_mode:
        set_optimization_seed(model_seed)
    surrogate_problem = _build_kriging_surrogate(
        benchmark_problem,
        x_data,
        y_data,
        official_pool_mode=official_pool_mode,
        model_seed=model_seed,
    )
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: _predict_desdeo_surrogate(surrogate_problem, x_test),
        problem_name=problem_name,
    )
    from desdeo_emo.EAs.ProbRVEA import ProbRVEA_v3

    use_truss2d_repair = str(problem_name).lower() == "truss2d"
    if use_truss2d_repair:
        _install_prob_population_numeric_compatibility()
    hv, igd_plus, igd_plus_source, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
        objective_values=config.get("metric_reference_values", y_data),
        fallback_reference_values=config.get("igd_reference_values"),
    )

    run_results = []
    for seed in seeds:
        seed_started = time.time()
        try:
            set_optimization_seed(seed)
            seed_initial_population = (
                initial_population_from_offline_dataset(
                    x_data, population_size=population_size, seed=seed
                )
                if official_pool_mode
                else initial_population
            )
            start_time = time.time()
            evolver = ProbRVEA_v3(
                surrogate_problem,
                use_surrogates=True,
                population_size=population_size,
                population_params={
                    "design": "InitSamples",
                    "init_pop": seed_initial_population.copy(),
                },
                **termination_kwargs,
            )
            if official_pool_mode:
                _install_exact_reference_vectors(
                    evolver,
                    benchmark_problem.n_obj,
                    population_size,
                )
                _install_population_size_preserving_selection(
                    evolver,
                    population_size,
                )
                _install_complete_generation_time_penalty(evolver)
                _install_problem_decision_repair(
                    evolver.population,
                    benchmark_problem,
                    repair_population_add=True,
                )
            while evolver.continue_evolution():
                evolver.iterate()
            elapsed = time.time() - start_time
            run_counters = (
                _validate_official_desdeo_budget(
                    evolver,
                    config.get("n_gen", 100),
                    population_size,
                )
                if official_pool_mode
                else _desdeo_run_counters(evolver)
            )
            if official_pool_mode:
                final_solution, final_obj = _fixed_desdeo_output(
                    evolver, surrogate_problem, population_size
                )
            elif use_truss2d_repair:
                final_solution, final_obj = _repair_and_predict_final_population(
                    surrogate_problem,
                    evolver.population.individuals,
                )
            else:
                final_solution = evolver.population.individuals
                final_obj = evolver.population.objectives
            result = _evaluate_solution(
                benchmark_problem,
                final_solution,
                final_obj,
                hv,
                igd_plus,
                igd_plus_source,
                obj_min,
                obj_max,
                final_output_size=population_size if official_pool_mode else None,
            )
            result.update(run_counters)
            result["offline_test_mse"] = float(offline_test_mse)
            result["seed"] = seed
            result["time"] = elapsed
            run_results.append(result)
            print_seed_result(
                result,
                seed=seed,
                elapsed=elapsed,
                problem_name=problem_name,
                method_name="Prob-RVEA",
            )
        except Exception as error:
            elapsed = time.time() - seed_started
            result = {
                "seed": seed,
                "time": elapsed,
                "solution_count": 0,
                "offline_test_mse": float(offline_test_mse),
                "mse_sur_real": np.nan,
                "sur_real_mse": np.nan,
                "hv_surrogate": np.nan,
                "hv_real": np.nan,
                "error_message": f"{type(error).__name__}: {error}",
            }
            run_results.append(result)
            print(f"Seed {seed} failed: {type(error).__name__}: {error}")
    return run_results


def _run_prob_moead_problem(problem_name, benchmark_problem, config, seeds):
    x_data, y_data = generate_offline_dataset(
        benchmark_problem,
        config,
        problem_name=problem_name,
    )
    official_pool_mode = config.get("dataset_source") == "official_pool"
    population_size = (
        int(config.get("pop_size", POPULATION_SIZE))
        if official_pool_mode
        else LEGACY_PROB_MOEAD_POPULATION_SIZE
    )
    termination_kwargs = _desdeo_termination_kwargs(
        config.get("n_gen", 100), official_pool_mode
    )
    if official_pool_mode:
        # ProbMOEAD uses this value as the denominator of its adaptive theta
        # schedule. Keep it positive while controlling generations explicitly.
        termination_kwargs["total_function_evaluations"] = (
            int(config.get("n_gen", 100)) * population_size
        )
    initial_population = (
        None
        if official_pool_mode
        else initial_population_from_offline_dataset(
            x_data,
            population_size=population_size,
        )
    )
    model_seed = int(config.get("model_seed", config["train_seed"]))
    if official_pool_mode:
        set_optimization_seed(model_seed)
    surrogate_problem = _build_kriging_surrogate(
        benchmark_problem,
        x_data,
        y_data,
        official_pool_mode=official_pool_mode,
        model_seed=model_seed,
    )
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: _predict_desdeo_surrogate(surrogate_problem, x_test),
        problem_name=problem_name,
    )
    from desdeo_emo.EAs.ProbMOEAD import ProbMOEAD_v3

    hv, igd_plus, igd_plus_source, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
        objective_values=config.get("metric_reference_values", y_data),
        fallback_reference_values=config.get("igd_reference_values"),
    )

    run_results = []
    for seed in seeds:
        seed_started = time.time()
        try:
            set_optimization_seed(seed)
            seed_initial_population = (
                initial_population_from_offline_dataset(
                    x_data, population_size=population_size, seed=seed
                )
                if official_pool_mode
                else initial_population
            )
            start_time = time.time()
            evolver = ProbMOEAD_v3(
                surrogate_problem,
                use_surrogates=True,
                population_size=population_size,
                population_params={
                    "design": "InitSamples",
                    "init_pop": seed_initial_population.copy(),
                },
                **termination_kwargs,
            )
            if official_pool_mode:
                _install_exact_reference_vectors(
                    evolver,
                    benchmark_problem.n_obj,
                    population_size,
                )
                _install_problem_decision_repair(
                    evolver.population,
                    benchmark_problem,
                    repair_population_add=False,
                )
                # The initial population is generation 1, not part of the
                # adaptive PBI-theta schedule.  Advance theta over offspring
                # evaluations only, reaching theta_max on the final one.
                evolver._theta_evaluation_offset = population_size
                evolver._theta_evaluation_budget = (
                    (int(config.get("n_gen", 100)) - 1) * population_size
                )
                evolver.selection_operator.use_absolute_pbi_projection = True
                # Exactly one iteration containing n_gen - 1 offspring
                # generations; the initial population is generation 1.
                evolver.iterate()
            else:
                _install_prob_moead_numeric_compatibility(evolver.population)
                while evolver.continue_evolution():
                    evolver.iterate()
            elapsed = time.time() - start_time
            run_counters = (
                _validate_official_desdeo_budget(
                    evolver,
                    config.get("n_gen", 100),
                    population_size,
                )
                if official_pool_mode
                else _desdeo_run_counters(evolver)
            )
            result = _evaluate_solution(
                benchmark_problem,
                evolver.population.individuals,
                evolver.population.objectives,
                hv,
                igd_plus,
                igd_plus_source,
                obj_min,
                obj_max,
                final_output_size=population_size if official_pool_mode else None,
            )
            result.update(run_counters)
            result["offline_test_mse"] = float(offline_test_mse)
            result["seed"] = seed
            result["time"] = elapsed
            run_results.append(result)
            print_seed_result(
                result,
                seed=seed,
                elapsed=elapsed,
                problem_name=problem_name,
                method_name="Prob-MOEA/D",
            )
        except Exception as error:
            elapsed = time.time() - seed_started
            result = {
                "seed": seed,
                "time": elapsed,
                "solution_count": 0,
                "offline_test_mse": float(offline_test_mse),
                "mse_sur_real": np.nan,
                "sur_real_mse": np.nan,
                "hv_surrogate": np.nan,
                "hv_real": np.nan,
                "error_message": f"{type(error).__name__}: {error}",
            }
            run_results.append(result)
            print(f"Seed {seed} failed: {type(error).__name__}: {error}")
    return run_results


def _run_tgpr_mo_problem(problem_name, benchmark_problem, config, seeds):
    _activate_vendor(TGPR_VENDOR_ROOT)
    _install_gpy_numpy_compatibility()
    _install_tgpr_numeric_compatibility()
    from desdeo_emo.EAs.RVEA import RVEA
    from framework.treedGP_framework import run_treed_GP

    x_data, y_data = generate_offline_dataset(
        benchmark_problem,
        config,
        problem_name=problem_name,
    )
    official_pool_mode = config.get("dataset_source") == "official_pool"
    model_seed = int(config.get("model_seed", config["train_seed"]))
    if official_pool_mode:
        set_optimization_seed(model_seed)
    surrogate_problem, _, _ = run_treed_GP(
        x_data,
        y_data,
        benchmark_problem.xl,
        benchmark_problem.xu,
        robust_small_data=official_pool_mode,
        random_state=model_seed if official_pool_mode else None,
    )
    population_size = (
        int(config.get("pop_size", POPULATION_SIZE))
        if official_pool_mode
        else _tgpr_default_population_size(surrogate_problem)
    )
    termination_kwargs = _desdeo_termination_kwargs(
        config.get("n_gen", 100), official_pool_mode
    )
    initial_population = (
        None
        if official_pool_mode
        else initial_population_from_offline_dataset(x_data)
    )
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: _predict_desdeo_surrogate(surrogate_problem, x_test),
        problem_name=problem_name,
    )
    hv, igd_plus, igd_plus_source, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
        objective_values=config.get("metric_reference_values", y_data),
        fallback_reference_values=config.get("igd_reference_values"),
    )

    run_results = []
    for seed in seeds:
        seed_started = time.time()
        try:
            set_optimization_seed(seed)
            seed_initial_population = (
                initial_population_from_offline_dataset(
                    x_data,
                    population_size=population_size,
                    seed=seed,
                )
                if official_pool_mode
                else initial_population
            )
            start_time = time.time()
            evolver = _build_tgpr_rvea(
                RVEA,
                surrogate_problem,
                seed_initial_population,
                population_size=population_size if official_pool_mode else None,
                **termination_kwargs,
            )
            if official_pool_mode:
                _install_exact_reference_vectors(
                    evolver,
                    benchmark_problem.n_obj,
                    population_size,
                )
                _install_population_size_preserving_selection(
                    evolver,
                    population_size,
                )
                _install_complete_generation_time_penalty(
                    evolver,
                    restore_rvea_objective_factor=True,
                )
                _install_problem_decision_repair(
                    evolver.population,
                    benchmark_problem,
                    repair_population_add=True,
                )
            while evolver.continue_evolution():
                evolver.iterate()
            elapsed = time.time() - start_time
            run_counters = (
                _validate_official_desdeo_budget(
                    evolver,
                    config.get("n_gen", 100),
                    population_size,
                )
                if official_pool_mode
                else _desdeo_run_counters(evolver)
            )
            if official_pool_mode:
                final_solution, final_obj = _fixed_desdeo_output(
                    evolver, surrogate_problem, population_size
                )
            else:
                final_solution = evolver.population.individuals
                final_obj = evolver.population.objectives
            result = _evaluate_solution(
                benchmark_problem,
                final_solution,
                final_obj,
                hv,
                igd_plus,
                igd_plus_source,
                obj_min,
                obj_max,
                final_output_size=population_size if official_pool_mode else None,
            )
            result.update(run_counters)
            result["offline_test_mse"] = float(offline_test_mse)
            result["seed"] = seed
            result["time"] = elapsed
            run_results.append(result)
            print_seed_result(
                result,
                seed=seed,
                elapsed=elapsed,
                problem_name=problem_name,
                method_name="TGPR-MO",
            )
        except Exception as error:
            elapsed = time.time() - seed_started
            result = {
                "seed": seed,
                "time": elapsed,
                "solution_count": 0,
                "offline_test_mse": float(offline_test_mse),
                "mse_sur_real": np.nan,
                "sur_real_mse": np.nan,
                "hv_surrogate": np.nan,
                "hv_real": np.nan,
                "error_message": f"{type(error).__name__}: {error}",
            }
            run_results.append(result)
            print(f"Seed {seed} failed: {type(error).__name__}: {error}")
    return run_results


def _run_ddmoea_gan_problem(problem_name, benchmark_problem, config, seeds):
    from ddmoea_gan import (
        DDMOEAGANProblem,
        construct_surrogate_pool_with_gan,
        surrogate_predict_with_ensemble,
        train_wgan_gp,
    )

    x_init, f_init = generate_offline_dataset(
        benchmark_problem,
        config,
        problem_name=problem_name,
    )
    official_pool_mode = config.get("dataset_source") == "official_pool"
    population_size = int(config.get("pop_size", POPULATION_SIZE)) if official_pool_mode else POPULATION_SIZE
    n_generations = int(config.get("n_gen", DDMOEA_GAN_N_GEN)) if official_pool_mode else DDMOEA_GAN_N_GEN
    initial_population = (
        None
        if official_pool_mode
        else initial_population_from_offline_dataset(x_init, population_size)
    )
    x_min = x_init.min(axis=0)
    x_max = x_init.max(axis=0)
    f_min = f_init.min(axis=0)
    f_max = f_init.max(axis=0)
    x_normalized = 2.0 * (x_init - x_min) / (x_max - x_min + 1e-12) - 1.0
    f_normalized = 2.0 * (f_init - f_min) / (f_max - f_min + 1e-12) - 1.0
    joint_init = np.hstack([x_normalized, f_normalized])

    # Train the stochastic surrogate exactly once per offline dataset.  Its
    # randomness is tied to model_seed (offline_seed by default), never to an
    # optimization seed used below.
    model_seed = int(config.get("model_seed", config["train_seed"]))
    set_optimization_seed(model_seed)
    generator, discriminator, device = train_wgan_gp(
        joint_init=joint_init,
        d_dim=joint_init.shape[1],
        n_obj=benchmark_problem.n_obj,
        n_epochs=2000,
        batch_size=64,
        z_dim=32,
        lambda_gp=10.0,
        n_critic=5,
        lr=1e-4,
        verbose=False,
    )
    surrogate_pools, norm_bounds = construct_surrogate_pool_with_gan(
        X_init=x_init,
        F_init=f_init,
        generator=generator,
        discriminator=discriminator,
        device=device,
        n_models=benchmark_problem.n_var,
        select_ratio=0.2,
        poly_degree=2,
        gamma_rbfn=0.5,
        lambda_rbfn=1e-6,
        verbose=False,
    )
    x_min, x_max, f_min, f_max = norm_bounds
    dd_problem = DDMOEAGANProblem(
        n_var=benchmark_problem.n_var,
        n_obj=benchmark_problem.n_obj,
        xl=benchmark_problem.xl,
        xu=benchmark_problem.xu,
        surrogate_pools=surrogate_pools,
        discriminator=discriminator,
        device=device,
        x_min=x_min,
        x_max=x_max,
        f_min=f_min,
        f_max=f_max,
        alpha_critic=0.1,
    )
    offline_test_mse = _compute_offline_test_mse(
        benchmark_problem,
        config,
        lambda x_test: surrogate_predict_with_ensemble(x_test, surrogate_pools),
        problem_name=problem_name,
    )
    hv, igd_plus, igd_plus_source, obj_min, obj_max, problem_y_min = _metric_context(
        problem_name,
        benchmark_problem,
        objective_values=config.get("metric_reference_values", f_init),
        fallback_reference_values=config.get("igd_reference_values"),
    )

    run_results = []
    for seed in seeds:
        seed_started = time.time()
        try:
            set_optimization_seed(seed)
            seed_initial_population = (
                initial_population_from_offline_dataset(
                    x_init, population_size=population_size, seed=seed
                )
                if official_pool_mode
                else initial_population
            )
            start_time = time.time()
            from src.offline_moo_adapter import get_offline_moo_repair

            ddmoea_repair = (
                get_offline_moo_repair(benchmark_problem)
                if official_pool_mode
                else None
            )
            algorithm_kwargs = dict(
                pop_size=population_size,
                sampling=seed_initial_population.copy(),
                crossover=SBX(prob=1.0, eta=20),
                mutation=PM(prob=1.0 / benchmark_problem.n_var, eta=20),
                eliminate_duplicates=True,
            )
            if ddmoea_repair is not None:
                algorithm_kwargs["repair"] = ddmoea_repair
            algorithm = NSGA2(**algorithm_kwargs)
            result_minimize = minimize(
                dd_problem,
                algorithm,
                ("n_gen", n_generations),
                seed=seed,
                verbose=False,
                save_history=False,
            )
            elapsed = time.time() - start_time
            final_opt = (
                result_minimize.pop if official_pool_mode else result_minimize.opt
            )
            final_x = final_opt.get("X")
            actual_evaluations = int(result_minimize.algorithm.evaluator.n_eval)
            if official_pool_mode:
                expected_evaluations = population_size * n_generations
                if actual_evaluations != expected_evaluations:
                    raise RuntimeError(
                        f"Expected {expected_evaluations} surrogate evaluations; "
                        f"got {actual_evaluations}."
                    )
                if len(final_x) != population_size:
                    raise RuntimeError(
                        f"Expected final live population {population_size}; "
                        f"got {len(final_x)}."
                    )
            # DDMOEAGANProblem.F is the critic-adjusted selection fitness.  It
            # must remain internal to optimization and is not a surrogate
            # objective prediction.  Recompute the ensemble mean at the fixed
            # final candidates for MSE/HV/IGD+ evaluation.
            final_surrogate_objectives = surrogate_predict_with_ensemble(
                final_x, surrogate_pools
            )
            result = _evaluate_solution(
                benchmark_problem,
                final_x,
                final_surrogate_objectives,
                hv,
                igd_plus,
                igd_plus_source,
                obj_min,
                obj_max,
                final_output_size=population_size if official_pool_mode else None,
            )
            result["optimizer_generation_count"] = int(n_generations)
            result["surrogate_evaluation_count"] = actual_evaluations
            result["offline_test_mse"] = float(offline_test_mse)
            result["seed"] = seed
            result["time"] = elapsed
            run_results.append(result)
            print_seed_result(
                result,
                seed=seed,
                elapsed=elapsed,
                problem_name=problem_name,
                method_name="DDMOEA-GAN",
            )
        except Exception as error:
            elapsed = time.time() - seed_started
            result = {
                "seed": seed,
                "time": elapsed,
                "solution_count": 0,
                "offline_test_mse": float(offline_test_mse),
                "mse_sur_real": np.nan,
                "sur_real_mse": np.nan,
                "hv_surrogate": np.nan,
                "hv_real": np.nan,
                "error_message": f"{type(error).__name__}: {error}",
            }
            run_results.append(result)
            print(f"Seed {seed} failed: {type(error).__name__}: {error}")
    return run_results


def run_prob_rvea_suite(config_path=None):
    return _run_suite("Prob-RVEA", _run_prob_rvea_problem, config_path=config_path)


def run_prob_moead_suite(config_path=None, max_workers=None):
    return _run_suite(
        "Prob-MOEA/D",
        _run_prob_moead_problem,
        config_path=config_path,
        max_workers=max_workers,
    )


def run_tgpr_mo_suite(config_path=None):
    return _run_suite("TGPR-MO", _run_tgpr_mo_problem, config_path=config_path)


def run_ddmoea_gan_suite(config_path=None):
    return _run_suite(
        "DDMOEA-GAN",
        _run_ddmoea_gan_problem,
        config_path=config_path,
    )
