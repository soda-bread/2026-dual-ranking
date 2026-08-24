import random
import time
import gc
import numpy as np
from sklearn.metrics import mean_squared_error
from pymoo.algorithms.moo.moead import ParallelMOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.constraints.as_penalty import ConstraintsAsPenalty
from pymoo.core.individual import calc_cv
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.util.misc import from_dict
from src.survival import Survival_standard 
from src.opt_problem import Benchmark_Problem, EvaluatePreRealCallback, evaluate_pre_real
from src.offline_moo_adapter import (
    evaluate_offline_moo_objectives_and_feasibility,
    get_offline_moo_repair,
    repair_offline_moo_decisions,
)
from src.metrics import normalize_objectives


class BroadcastConstraintsAsPenalty(ConstraintsAsPenalty):
    def do(self, X, return_values_of, *args, **kwargs):
        out = self.__object__.do(X, return_values_of, *args, **kwargs)
        F, G, H = from_dict(out, "F", "G", "H")
        out["__F__"], out["__G__"], out["__H__"] = F, G, H

        CV = calc_cv(G=G, H=H)
        penalty = np.asarray(CV, dtype=float).reshape(-1, 1)
        out["F"] = F + self.penalty * penalty

        out.pop("G", None)
        out.pop("H", None)
        return out


def _normalize_optimizer_name(optimizer_name):
    name = str(optimizer_name).strip().lower()
    return name.replace("_", "").replace("-", "").replace(" ", "")


def _prepare_initial_population(problem, pop_size, initial_population=None):
    configured_pop_size = int(pop_size)
    if configured_pop_size < 2:
        raise ValueError("pop_size must be at least 2.")
    if initial_population is None:
        return configured_pop_size, None

    initial_population = np.asarray(initial_population, dtype=float)
    if initial_population.ndim != 2:
        raise ValueError("initial_population must be a 2D array.")
    if initial_population.shape[1] != problem.n_var:
        raise ValueError(
            "initial_population must have one column per problem variable: "
            f"expected {problem.n_var}, got {initial_population.shape[1]}."
        )
    if initial_population.shape[0] < 2:
        raise ValueError("initial_population must contain at least 2 points.")
    if not np.all(np.isfinite(initial_population)):
        raise ValueError("initial_population must contain only finite values.")

    initial_population = np.array(
        initial_population[:configured_pop_size],
        dtype=float,
        copy=True,
    )
    return initial_population.shape[0], initial_population


def objectives_within_hv_bounds(
    *objective_sets,
    obj_min,
    obj_max,
    atol=1e-12,
    only_non_dominated=False,
):
    obj_min = np.asarray(obj_min, dtype=float).reshape(1, -1)
    obj_max = np.asarray(obj_max, dtype=float).reshape(1, -1)
    for values in objective_sets:
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != obj_min.shape[1]:
            return False
        if not np.all(np.isfinite(values)):
            return False
        if only_non_dominated:
            values = values[non_dominated_mask(values)]
        if not np.all(values >= obj_min - atol):
            return False
        if not np.all(values <= obj_max + atol):
            return False
    return True


def build_optimization_algorithm(
    optimizer_name,
    problem,
    pop_size,
    survival_function=None,
    initial_population=None,
):
    if initial_population is not None:
        initial_population = repair_offline_moo_decisions(problem, initial_population)
    effective_pop_size, initial_population = _prepare_initial_population(
        problem,
        pop_size,
        initial_population=initial_population,
    )
    # Official-pool experiments can intentionally use fewer offline rows than
    # the optimizer population (for example, N=50 with pop_size=100). Their
    # deterministic initializer then samples with replacement. If pymoo removed
    # those repeated rows, generation 1 would be undersized and the shared
    # population/evaluation budget would be violated.
    repeated_initial_rows = (
        initial_population is not None
        and len(np.unique(initial_population, axis=0)) < effective_pop_size
    )
    optimizer_key = _normalize_optimizer_name(optimizer_name)
    crossover = SBX(prob=1.0, eta=20)
    mutation = PM(prob=1 / problem.n_var, eta=20)
    repair = get_offline_moo_repair(problem)
    sampling_kwargs = (
        {}
        if initial_population is None
        else {"sampling": initial_population}
    )

    if optimizer_key in {"nsga2", "nsgaii"}:
        kwargs = {
            "pop_size": effective_pop_size,
            "crossover": crossover,
            "mutation": mutation,
            "eliminate_duplicates": not repeated_initial_rows,
            **sampling_kwargs,
        }
        if repair is not None:
            kwargs["repair"] = repair
        if survival_function is not None:
            kwargs["survival"] = survival_function
        return NSGA2(**kwargs)

    if optimizer_key == "moead":
        ref_dirs = get_reference_directions(
            "energy",
            problem.n_obj,
            effective_pop_size,
            seed=1,
        )
        # Use pymoo's synchronous variant so surrogate inference is evaluated
        # once per generation as a population-sized batch.
        kwargs = {
            "ref_dirs": ref_dirs,
            "n_neighbors": min(20, len(ref_dirs)),
            "n_offsprings": effective_pop_size,
            "prob_neighbor_mating": 0.9,
            "crossover": crossover,
            "mutation": mutation,
            **sampling_kwargs,
        }
        if repair is not None:
            kwargs["repair"] = repair
        return ParallelMOEAD(**kwargs)

    if optimizer_key in {"smsemoa", "sms", "sms emoa"}:
        kwargs = {
            "pop_size": effective_pop_size,
            "crossover": crossover,
            "mutation": mutation,
            "eliminate_duplicates": True,
            **sampling_kwargs,
        }
        if repair is not None:
            kwargs["repair"] = repair
        return SMSEMOA(**kwargs)

    raise ValueError(
        "optimizer_name must be one of 'NSGA-II', 'MOEAD', or 'SMS-EMOA'."
    )


def normalized_hv(hv, F, obj_min, obj_max):
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return np.nan
    F_normalization = normalize_objectives(F, obj_min, obj_max)
    return float(hv.do(F_normalization))


def normalized_igd_plus(igd_plus, F, obj_min, obj_max):
    if igd_plus is None:
        return np.nan
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return np.nan
    return float(igd_plus.do(normalize_objectives(F, obj_min, obj_max)))


def non_dominated_mask(F):
    F = np.asarray(F, dtype=float)
    mask = np.zeros(F.shape[0], dtype=bool)
    if F.size == 0:
        return mask
    nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
    mask[np.asarray(nd_idx, dtype=int)] = True
    return mask


def plot_seed_sur_real_hv_reference(
    f_sur,
    f_real,
    hv,
    obj_min,
    obj_max,
    title=None,
    width=700,
    height=650,
    point_size=7,
):
    f_sur = np.asarray(f_sur, dtype=float)
    f_real = np.asarray(f_real, dtype=float)
    obj_min = np.asarray(obj_min, dtype=float)
    obj_max = np.asarray(obj_max, dtype=float)

    if f_sur.ndim != 2 or f_sur.shape[1] != 2:
        raise ValueError("f_sur must have shape (n, 2).")
    if f_real.ndim != 2 or f_real.shape[1] != 2:
        raise ValueError("f_real must have shape (n, 2).")
    if f_sur.shape != f_real.shape:
        raise ValueError("f_sur and f_real must have the same shape.")

    hv_ref_norm = np.asarray(
        getattr(hv, "ref_point", np.ones(f_sur.shape[1])),
        dtype=float,
    )
    hv_ref = obj_min + hv_ref_norm * (obj_max - obj_min)
    all_points = np.vstack([
        f_sur,
        f_real,
        hv_ref.reshape(1, -1),
        obj_min.reshape(1, -1),
        obj_max.reshape(1, -1),
    ])
    x_min, x_max = np.nanmin(all_points[:, 0]), np.nanmax(all_points[:, 0])
    y_min, y_max = np.nanmin(all_points[:, 1]), np.nanmax(all_points[:, 1])
    x_pad = 0.06 * (x_max - x_min) if x_max > x_min else 1.0
    y_pad = 0.06 * (y_max - y_min) if y_max > y_min else 1.0

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=f_sur[:, 0],
            y=f_sur[:, 1],
            mode="markers",
            name="f_sur",
            marker=dict(size=point_size, color="#87CEEB", opacity=0.8),
            customdata=np.arange(f_sur.shape[0]),
            hovertemplate=(
                "f_sur[%{customdata}]<br>"
                "f1=%{x:.6g}<br>"
                "f2=%{y:.6g}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=f_real[:, 0],
            y=f_real[:, 1],
            mode="markers",
            name="f_real",
            marker=dict(size=point_size, color="#FF7F0E", opacity=0.8),
            customdata=np.arange(f_real.shape[0]),
            hovertemplate=(
                "f_real[%{customdata}]<br>"
                "f1=%{x:.6g}<br>"
                "f2=%{y:.6g}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[hv_ref[0]],
            y=[hv_ref[1]],
            mode="markers",
            name="HV ref",
            marker=dict(size=point_size, color="#D62728", opacity=1.0),
            hovertemplate=(
                "HV ref<br>"
                "f1=%{x:.6g}<br>"
                "f2=%{y:.6g}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[obj_min[0]],
            y=[obj_min[1]],
            mode="markers",
            name="HV obj_min",
            marker=dict(
                size=point_size,
                color="#2CA02C",
                symbol="square",
                line=dict(width=1.5, color="#1B6E1B"),
                opacity=1.0,
            ),
            hovertemplate=(
                "HV obj_min<br>"
                "f1=%{x:.6g}<br>"
                "f2=%{y:.6g}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[obj_max[0]],
            y=[obj_max[1]],
            mode="markers",
            name="HV obj_max",
            marker=dict(
                size=point_size,
                color="#9467BD",
                symbol="square",
                line=dict(width=1.5, color="#5D3B7D"),
                opacity=1.0,
            ),
            hovertemplate=(
                "HV obj_max<br>"
                "f1=%{x:.6g}<br>"
                "f2=%{y:.6g}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="f1",
        yaxis_title="f2",
        width=width,
        height=height,
        xaxis=dict(range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
        legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle"),
        margin=dict(r=140),
    )
    fig.show()
    return fig


def mse_or_nan(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0 or y_pred.size == 0:
        return np.nan
    if y_true.shape != y_pred.shape:
        raise ValueError(
            "MSE arrays must have the same shape: "
            f"got {y_true.shape} and {y_pred.shape}."
        )
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(finite_mask):
        return np.nan
    if not np.all(finite_mask):
        return float(np.mean((y_true[finite_mask] - y_pred[finite_mask]) ** 2))
    return mean_squared_error(y_true, y_pred)


def format_percent(value, digits=1):
    value = np.asarray(value, dtype=float)
    if value.size == 0:
        return "nan%"
    value = float(np.ravel(value)[0])
    if not np.isfinite(value):
        return "nan%"
    return f"{100.0 * value:.{digits}f}%"


def format_result_value(value, digits=3):
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


def compute_surrogate_test_mse(
    problem,
    problem_name,
    model_f1,
    model_f2,
    use_surrogate,
    x_test,
    y_test,
    models=None,
):
    surrogate_problem = Benchmark_Problem(
        model_f1=model_f1,
        model_f2=model_f2,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        xl=problem.xl,
        xu=problem.xu,
        problem_name=problem_name,
        use_surrogate=use_surrogate,
        models=models,
    )
    f_test_pred = surrogate_problem.evaluate(
        np.asarray(x_test, dtype=float),
        return_values_of=["F"],
    )
    return mse_or_nan(y_test, f_test_pred)


def run_experiment(
    problem,
    problem_name,
    n_gen,
    pop_size,
    use_surrogate,
    model_f1,
    model_f2,
    survival_function,
    obj_min,
    obj_max,
    hv,
    use_callback,
    seeds,
    optimizer_name="NSGA-II",
    print_normalization_info=True,
    initial_population=None,
    mse_test=None,
    plot_seed_objectives=True,
    seed_result_callback=None,
    models=None,
    igd_plus_indicator=None,
    igd_plus_source=None,
    final_output_size=None,
):

    minimize_kwargs = dict(
        termination=get_termination("n_gen", n_gen),
        save_history=False,
        verbose=False,
    )

    if use_callback:
        callback_standard = EvaluatePreRealCallback(
            true_problem=problem,
            plot_every=10,
            use_opt=True,
            dynamic_show=False,
            prefix=f"{optimizer_name}-standard",
            obj_min=obj_min,
            obj_max=obj_max,
            hv_indicator=hv,
        )
        minimize_kwargs["callback"] = callback_standard

    hv_surrogate_list = []
    hv_real_list = []
    igd_plus_surrogate_list = []
    igd_plus_real_list = []
    hv_real_count_list = []
    mse_test_list = []
    sur_real_mse_list = []
    run_details = []

    if initial_population is not None:
        initial_population_source = "explicit_first_pop_size"
    else:
        initial_population_source = "optimizer_default_random_sampling"
    initial_population_array = None if initial_population is None else np.asarray(initial_population)
    initial_population_available_count = (
        None
        if initial_population_array is None or initial_population_array.ndim == 0
        else int(initial_population_array.shape[0])
    )
    effective_pop_size, initial_population = _prepare_initial_population(
        problem,
        pop_size,
        initial_population=repair_offline_moo_decisions(problem, initial_population)
        if initial_population is not None
        else None,
    )
    initialization_info = {
        "source": initial_population_source,
        "selection": None if initial_population is None else "first_configured_pop_size_points",
        "configured_pop_size": int(pop_size),
        "available_offline_points": initial_population_available_count,
        "effective_pop_size": int(effective_pop_size),
        "initial_population_x": (
            None
            if initial_population is None
            else np.array(initial_population, dtype=float, copy=True)
        ),
    }
    normalization_info = {
        "hv_obj_min": np.asarray(obj_min, dtype=float).tolist(),
        "hv_obj_max": np.asarray(obj_max, dtype=float).tolist(),
        "optimizer_name": optimizer_name,
    }
    if print_normalization_info:
        print(
            f"[{problem_name}] initial population: {initial_population_source}, "
            f"configured pop_size={int(pop_size)}, "
            f"effective pop_size={effective_pop_size}"
        )
        print(f"[{problem_name}] HV obj_min: {_array_text(obj_min)}")
        print(f"[{problem_name}] HV obj_max: {_array_text(obj_max)}")

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        benchmark_problem_GPR = Benchmark_Problem(
            model_f1=model_f1,
            model_f2=model_f2,
            n_var=problem.n_var,
            n_obj=problem.n_obj,
            xl=problem.xl,
            xu=problem.xu,
            problem_name=problem_name,
            use_surrogate=use_surrogate,
            models=models,
        )

        start_time = time.time()
        algorithm = build_optimization_algorithm(
            optimizer_name=optimizer_name,
            problem=problem,
            pop_size=pop_size,
            survival_function=survival_function,
            initial_population=initial_population,
        )
        optimization_problem = benchmark_problem_GPR
        constraint_handling = None
        if (
            _normalize_optimizer_name(optimizer_name) == "moead"
            and benchmark_problem_GPR.has_constraints()
        ):
            optimization_problem = BroadcastConstraintsAsPenalty(
                benchmark_problem_GPR,
                penalty=1e6,
            )
            constraint_handling = "BroadcastConstraintsAsPenalty(penalty=1e6)"

        res = minimize(
            optimization_problem,
            algorithm,
            seed=seed,
            **minimize_kwargs,
        )
        end_time = time.time()

        actual_evaluations = int(res.algorithm.evaluator.n_eval)
        if final_output_size is not None:
            expected_evaluations = int(n_gen) * int(pop_size)
            if effective_pop_size != int(pop_size):
                raise RuntimeError(
                    f"Expected optimizer population {int(pop_size)}; initialized "
                    f"{effective_pop_size}."
                )
            if actual_evaluations != expected_evaluations:
                raise RuntimeError(
                    f"Expected {expected_evaluations} surrogate evaluations; "
                    f"got {actual_evaluations}."
                )
            if res.pop is None or len(res.pop) != int(final_output_size):
                live_count = 0 if res.pop is None else len(res.pop)
                raise RuntimeError(
                    f"Expected final live population {int(final_output_size)}; "
                    f"got {live_count}."
                )

        no_feasible_solution = False
        no_feasible_reason = None
        submitted_solution_count = 0
        opt = res.pop if final_output_size is not None else res.opt
        solution = None if opt is None else opt.get("X")
        try:
            solution = np.asarray(solution, dtype=float)
            if solution.size == 0:
                raise ValueError("empty final X")
            if solution.ndim == 1:
                solution = solution.reshape(1, -1)
            submitted_solution_count = int(len(solution))
            if final_output_size is not None:
                target_size = int(final_output_size)
                if target_size < 1:
                    raise ValueError("final_output_size must be positive.")
                cv = opt.get("CV")
                if cv is not None:
                    cv = np.asarray(cv, dtype=float).reshape(-1)
                    feasible = np.isfinite(cv) & (cv <= 0.0)
                    solution = solution[feasible]
                solution = solution[:target_size]
                if len(solution) == 0:
                    raise ValueError("final population contains no feasible solution")
            obj = benchmark_problem_GPR.evaluate(solution, return_values_of=["F"])
            f_real, custom_feasible = (
                evaluate_offline_moo_objectives_and_feasibility(problem, solution)
            )
            obj = np.asarray(obj, dtype=float)
            f_real = np.asarray(f_real, dtype=float)
            if obj.ndim == 1:
                obj = obj.reshape(1, -1)
            if f_real.ndim == 1:
                f_real = f_real.reshape(1, -1)
            finite_mask = (
                np.all(np.isfinite(solution), axis=1)
                & np.all(np.isfinite(obj), axis=1)
                & np.all(np.isfinite(f_real), axis=1)
            )
            if custom_feasible is not None:
                finite_mask &= np.asarray(custom_feasible, dtype=bool)
            if not np.any(finite_mask):
                raise ValueError("all final candidates are non-finite or infeasible")
            if not np.all(finite_mask):
                solution = solution[finite_mask]
                obj = obj[finite_mask]
                f_real = f_real[finite_mask]

            sur_real_mse = float(mse_or_nan(f_real, obj))
            seed_mse_test = sur_real_mse if mse_test is None else float(mse_test)
            hv_real = normalized_hv(hv, f_real, obj_min, obj_max)
            hv_surrogate = normalized_hv(hv, obj, obj_min, obj_max)
            igd_plus_real = normalized_igd_plus(
                igd_plus_indicator, f_real, obj_min, obj_max
            )
            igd_plus_surrogate = normalized_igd_plus(
                igd_plus_indicator, obj, obj_min, obj_max
            )
            hv_bounds_check = objectives_within_hv_bounds(
                obj,
                f_real,
                obj_min=obj_min,
                obj_max=obj_max,
                only_non_dominated=True,
            )
            hv_real_count = int(f_real.shape[0])
        except Exception as err:
            no_feasible_solution = True
            no_feasible_reason = f"{type(err).__name__}: {err}"
            print(
                f"Seed {seed} | no feasible final solution for metric evaluation: "
                f"{no_feasible_reason}. Metrics set to NaN."
            )
            solution = np.empty((0, problem.n_var))
            obj = np.empty((0, problem.n_obj))
            f_real = np.empty((0, problem.n_obj))
            sur_real_mse = np.nan
            seed_mse_test = (
                np.nan if mse_test is None else float(mse_test)
            )
            hv_real = np.nan
            hv_surrogate = np.nan
            igd_plus_real = np.nan
            igd_plus_surrogate = np.nan
            hv_bounds_check = False
            hv_real_count = 0

        hv_real_list.append(hv_real)
        hv_surrogate_list.append(hv_surrogate)
        igd_plus_real_list.append(igd_plus_real)
        igd_plus_surrogate_list.append(igd_plus_surrogate)
        hv_real_count_list.append(hv_real_count)
        mse_test_list.append(seed_mse_test)
        sur_real_mse_list.append(sur_real_mse)

        if no_feasible_solution:
            max_obj = np.zeros(problem.n_obj, dtype=float)
            max_obj_real = np.zeros(problem.n_obj, dtype=float)
        else:
            max_obj = np.max(obj, axis=0)
            max_obj_real = np.max(f_real, axis=0)
        print(
            f"Seed {seed} | "
            f"Time: {format_result_value(end_time - start_time)}s | "
            f"MSE_test: {format_result_value(seed_mse_test)} | "
            f"MSE_sur_real: {format_result_value(sur_real_mse)} | "
            f"HV_sur: {format_result_value(hv_surrogate)} | "
            f"HV_real: {format_result_value(hv_real)} | "
            f"IGD+_sur: {format_result_value(igd_plus_surrogate)} | "
            f"IGD+_real: {format_result_value(igd_plus_real)} | "
            f"HV_bounds_check: {'yes' if hv_bounds_check else 'no'}"
        )
        if plot_seed_objectives and int(seed) == 1 and problem.n_obj == 2:
            plot_seed_sur_real_hv_reference(
                obj,
                f_real,
                hv,
                obj_min,
                obj_max,
                title=f"{problem_name} | {optimizer_name} | Seed {seed}",
            )

        detail = {
            "seed": seed,
            "time": end_time - start_time,
            "solution_count": hv_real_count,
            "submitted_solution_count": submitted_solution_count,
            "mse_test": seed_mse_test,
            "mse_sur_real": sur_real_mse,
            "sur_real_mse": sur_real_mse,
            "hv_surrogate": hv_surrogate,
            "hv_bounds_check": hv_bounds_check,
            "hv_real": hv_real,
            "igd_plus_surrogate": igd_plus_surrogate,
            "igd_plus_real": igd_plus_real,
            "igd_plus_source": igd_plus_source,
            "hv_real_count": hv_real_count,
            "final_output_target": final_output_size,
            "no_feasible_solution": no_feasible_solution,
            "no_feasible_reason": no_feasible_reason,
            "normalization_info": normalization_info,
            "constraint_handling": constraint_handling,
            "optimizer_generation_count": int(n_gen),
            "surrogate_evaluation_count": actual_evaluations,
            "max_obj": max_obj,
            "max_f_real": max_obj_real,
        }
        if use_callback:
            detail.update(
                {
                    "gen_history": callback_standard.gen_list,
                    "hv_sur_history": callback_standard.hv_sur_list,
                    "hv_real_history": callback_standard.hv_real_list,
                }
            )
        run_details.append(detail)
        if seed_result_callback is not None:
            seed_result_callback(detail)
        del res, algorithm, benchmark_problem_GPR, optimization_problem
        gc.collect()

    return {
        "hv_surrogate_list": hv_surrogate_list,
        "hv_real_list": hv_real_list,
        "igd_plus_surrogate_list": igd_plus_surrogate_list,
        "igd_plus_real_list": igd_plus_real_list,
        "hv_real_count_list": hv_real_count_list,
        "mse_test_list": mse_test_list,
        "mse_sur_real_list": sur_real_mse_list,
        "sur_real_mse_list": sur_real_mse_list,
        "normalization_info": normalization_info,
        "initialization_info": initialization_info,
        "run_details": run_details,
    }
