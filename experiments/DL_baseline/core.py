"""Reproducible adapters for four baselines from lamda-bbo/offline-moo.

The upstream experiment entry points mix data loading, tracking, model fitting,
optimization, and evaluation.  This module keeps the algorithms but plugs them
into this repository's paired official-pool protocol.  Heavy optional imports
are intentionally lazy so ``run.py --dry-run`` works before the DL environment
is installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml


BASELINE_NAMES = (
    "End2End-Vallina",
    "MultipleModels-Vallina",
    "MultipleModels-COM",
    "MOBO-Vallina",
)
PROTOCOL_VERSION = "off_moo_dl_baselines_official_pool_v3"
REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_MOO_ROOT = REPO_ROOT / "external" / "offline-moo"


RESULT_FIELDS = (
    "problem",
    "method",
    "dataset_source",
    "protocol_version",
    "configuration_hash",
    "training_size",
    "offline_sample_size",
    "fit_size",
    "test_size",
    "offline_seed",
    "model_seed",
    "opt_seed",
    "subset_indices_hash",
    "configured_n_gen",
    "configured_pop_size",
    "MSEpre",
    "MSEsur_real",
    "HVsur",
    "HVreal",
    "IGDplus_sur",
    "IGDplus",
    "objective_min",
    "objective_max",
    "hv_reference_point_normalized",
    "igdplus_reference_source",
    "submitted_solution_count",
    "number_of_feasible_solutions",
    "optimizer_generation_count",
    "surrogate_evaluation_count",
    "runtime_surrogate_training",
    "runtime_optimization",
    "candidate_file",
    "status",
    "error_message",
)


@dataclass(frozen=True)
class Standardizer:
    """Column-wise z-score transform fitted only on the selected N rows."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        values = _matrix(values, "values")
        mean = np.mean(values, axis=0)
        scale = np.std(values, axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.scale + self.mean


@dataclass(frozen=True)
class BoundsScaler:
    """Affine normalization from the true problem bounds to [0, 1]."""

    lower: np.ndarray
    span: np.ndarray

    @classmethod
    def fit(cls, lower: np.ndarray, upper: np.ndarray) -> "BoundsScaler":
        lower = np.asarray(lower, dtype=float).reshape(-1)
        upper = np.asarray(upper, dtype=float).reshape(-1)
        if lower.shape != upper.shape or not np.all(np.isfinite([lower, upper])):
            raise ValueError("MOBO requires matching finite decision bounds.")
        span = upper - lower
        if np.any(span <= 0.0):
            raise ValueError("MOBO requires strictly increasing decision bounds.")
        return cls(lower=lower, span=span)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values, dtype=float) - self.lower) / self.span

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=float) * self.span + self.lower


@dataclass
class Predictor:
    models: Sequence[Any]
    x_scaler: Standardizer
    y_scaler: Standardizer
    device: Any
    joint: bool

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        x_scaled = self.x_scaler.transform(_matrix(x, "x"))
        tensor = torch.as_tensor(x_scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.joint:
                prediction = self.models[0](tensor)
            else:
                prediction = torch.cat([model(tensor) for model in self.models], dim=1)
        return self.y_scaler.inverse(prediction.detach().cpu().numpy())


@dataclass
class MoboPredictor:
    model: Any
    x_scaler: BoundsScaler
    y_scaler: Standardizer
    device: Any
    dtype: Any
    baseline_x: Any
    baseline_y: Any

    def predict(self, x: np.ndarray) -> np.ndarray:
        import torch

        scaled = self.x_scaler.transform(_matrix(x, "x"))
        tensor = torch.as_tensor(scaled, device=self.device, dtype=self.dtype)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model.posterior(tensor).mean
        # The model is trained on negated standardized minimization objectives.
        return self.y_scaler.inverse(-prediction.detach().cpu().numpy())


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty 2D array; got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    return values


def ensure_import_paths() -> None:
    import sys

    if not OFFLINE_MOO_ROOT.exists():
        raise FileNotFoundError(
            "external/offline-moo is missing. Run "
            "`git submodule update --init external/offline-moo`."
        )
    for path in (REPO_ROOT, OFFLINE_MOO_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_config(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    missing = [name for name in ("methods", "problems") if not config.get(name)]
    if missing:
        raise ValueError(f"Configuration is missing non-empty keys: {missing}.")
    unknown = sorted(set(config["methods"]) - set(BASELINE_NAMES))
    if unknown:
        raise ValueError(f"Unknown baseline names: {unknown}.")
    if str(config.get("dataset_source", "official_pool")) != "official_pool":
        raise ValueError("DL baselines currently require dataset_source=official_pool.")
    return config


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str):
    import torch

    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable.")
    return torch.device(requested)


def subset_hash(data: dict[str, np.ndarray]) -> str:
    indices = np.ascontiguousarray(data["offline_indices"], dtype=np.int64)
    return hashlib.sha256(indices.tobytes()).hexdigest()


def load_data(
    problem: str,
    training_size: int,
    offline_seed: int,
    all_training_sizes: Sequence[int],
    subset_cache_dir: Path,
):
    ensure_import_paths()
    import off_moo_bench

    from src.official_pool import load_official_subset, official_task_name

    data = load_official_subset(
        cache_root=subset_cache_dir,
        problem_name=problem,
        sample_size=int(training_size),
        offline_seed=int(offline_seed),
        all_sample_sizes=tuple(int(value) for value in all_training_sizes),
    )
    task = off_moo_bench.make(official_task_name(problem))
    return task, data


def _make_mlp(input_size: int, output_size: int, hidden_sizes: Sequence[int]):
    import torch

    layers: list[Any] = []
    width = int(input_size)
    for hidden in hidden_sizes:
        layers.extend((torch.nn.Linear(width, int(hidden)), torch.nn.LeakyReLU()))
        width = int(hidden)
    layers.append(torch.nn.Linear(width, int(output_size)))
    return torch.nn.Sequential(*layers)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int):
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=True,
        drop_last=False,
        generator=generator,
    )


def _fit_vallina_model(model, loader, epochs: int, learning_rate: float, device):
    import torch

    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=0.0
    )
    for _ in range(int(epochs)):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            # Matches the upstream Vallina trainer's batch-summed MSE.
            loss = torch.sum(torch.mean((prediction - batch_y) ** 2, dim=1))
            loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()


def _fit_com_model(model, loader, epochs: int, config: dict[str, Any], device):
    """Train one objective with the upstream COM Lagrangian and particles."""

    import torch
    import torch.nn.functional as functional

    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["learning_rate"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, int(epochs)), eta_min=0.0
    )
    log_alpha = torch.nn.Parameter(
        torch.log(torch.tensor(float(config["alpha"]), device=device))
    )
    alpha_optimizer = torch.optim.Adam(
        [log_alpha], lr=float(config["alpha_learning_rate"])
    )
    input_width = int(loader.dataset.tensors[0].shape[1])
    particle_lr = float(config["particle_learning_rate"]) * np.sqrt(input_width)
    particle_steps = int(config["particle_gradient_steps"])
    entropy_coefficient = float(config["entropy_coefficient"])
    noise_std = float(config["noise_std"])
    overestimation_limit = float(config["overestimation_limit"])

    for _ in range(int(epochs)):
        model.train()
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            positive_x = batch_x + noise_std * torch.randn_like(batch_x)
            positive_prediction = model(positive_x)

            negative_x = positive_x.detach().clone()
            for _ in range(particle_steps):
                negative_x.requires_grad_(True)
                shuffled = negative_x[torch.randperm(len(negative_x), device=device)]
                entropy = torch.mean((negative_x - shuffled) ** 2)
                score = model(negative_x)
                particle_objective = (
                    entropy_coefficient * entropy * len(score) + score.sum()
                )
                gradient = torch.autograd.grad(particle_objective, negative_x)[0]
                negative_x = (negative_x - particle_lr * gradient).detach()

            negative_prediction = model(negative_x)
            overestimation = (
                positive_prediction[:, 0] - negative_prediction[:, 0]
            ).mean()
            mse = functional.mse_loss(positive_prediction, batch_y)

            alpha = torch.exp(log_alpha)
            alpha_loss = alpha * (overestimation_limit - overestimation.detach())
            alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            alpha_optimizer.step()

            model_loss = mse + alpha.detach() * overestimation
            optimizer.zero_grad(set_to_none=True)
            model_loss.backward()
            optimizer.step()
        scheduler.step()
    model.eval()


def fit_neural_predictor(
    method: str,
    data: dict[str, np.ndarray],
    neural_config: dict[str, Any],
    com_config: dict[str, Any],
    model_seed: int,
    device,
) -> Predictor:
    import torch

    set_seed(model_seed)
    x_train = _matrix(data["X_train"], "X_train")
    y_train = _matrix(data["y_train"], "y_train")
    x_scaler = Standardizer.fit(x_train)
    y_scaler = Standardizer.fit(y_train)
    x_scaled = x_scaler.transform(x_train)
    y_scaled = y_scaler.transform(y_train)
    hidden_sizes = tuple(int(value) for value in neural_config["hidden_sizes"])
    epochs = int(neural_config["epochs"])
    batch_size = int(neural_config["batch_size"])

    if method == "End2End-Vallina":
        model = _make_mlp(x_train.shape[1], y_train.shape[1], hidden_sizes)
        loader = _loader(x_scaled, y_scaled, batch_size, model_seed)
        _fit_vallina_model(
            model,
            loader,
            epochs,
            float(neural_config["learning_rate"]),
            device,
        )
        models = (model,)
        joint = True
    else:
        models_list = []
        for objective_index in range(y_train.shape[1]):
            objective_seed = int(model_seed) + 10_000 * objective_index
            set_seed(objective_seed)
            model = _make_mlp(x_train.shape[1], 1, hidden_sizes)
            loader = _loader(
                x_scaled,
                y_scaled[:, objective_index : objective_index + 1],
                batch_size,
                objective_seed,
            )
            if method == "MultipleModels-COM":
                _fit_com_model(model, loader, epochs, com_config, device)
            else:
                _fit_vallina_model(
                    model,
                    loader,
                    epochs,
                    float(neural_config["learning_rate"]),
                    device,
                )
            models_list.append(model)
        models = tuple(models_list)
        joint = False

    return Predictor(
        models=models,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        joint=joint,
    )


def _nondominated_training_indices(y: np.ndarray, limit: int) -> np.ndarray:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

    y = _matrix(y, "y_train")
    limit = min(int(limit), len(y))
    fronts = NonDominatedSorting().do(y)
    ordered = np.concatenate([np.asarray(front, dtype=int) for front in fronts])
    return ordered[:limit]


def upstream_nds_initial_population(
    x_train: np.ndarray,
    y_train: np.ndarray,
    population_size: int,
    opt_seed: int,
    problem,
) -> np.ndarray:
    """Reproduce Off-MOO's NDS initialization with a small-data LHS fill.

    Upstream takes offline designs in Pareto-front order and fills a population
    shortage with Latin-hypercube samples.  As a deliberate deviation, this
    adapter samples the true problem bounds.  Upstream instead samples the
    dataset min-max envelope in [0, 1], which also includes its test pool.
    """

    from pymoo.operators.sampling.lhs import LHS

    from src.offline_moo_adapter import repair_offline_moo_decisions

    x_train = _matrix(x_train, "x_train")
    y_train = _matrix(y_train, "y_train")
    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must contain the same number of rows.")
    population_size = int(population_size)
    if population_size < 2:
        raise ValueError("population_size must be at least 2.")

    ordered = _nondominated_training_indices(y_train, len(y_train))
    initial = x_train[ordered[:population_size]].copy()
    shortage = population_size - len(initial)
    if shortage > 0:
        saved_state = np.random.get_state()
        np.random.seed(int(opt_seed))
        try:
            # pymoo 0.5 uses NumPy's global RNG and silently ignores ``seed``.
            # Passing that same RNG explicitly also preserves this behavior on
            # pymoo 0.6+, where an omitted random_state creates a fresh RNG.
            lhs = LHS().do(problem, shortage, random_state=np.random).get("X")
        finally:
            np.random.set_state(saved_state)
        initial = np.vstack((initial, np.asarray(lhs, dtype=float)))
    return np.asarray(
        repair_offline_moo_decisions(problem, initial), dtype=float
    )


def fit_mobo_predictor(
    data: dict[str, np.ndarray],
    mobo_config: dict[str, Any],
    model_seed: int,
    device,
    x_lower: np.ndarray,
    x_upper: np.ndarray,
) -> MoboPredictor:
    import torch
    from botorch import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    set_seed(model_seed)
    x_train = _matrix(data["X_train"], "X_train")
    y_train = _matrix(data["y_train"], "y_train")
    gp_limit = mobo_config.get("train_gp_data_size")
    indices = (
        np.arange(len(y_train), dtype=int)
        if gp_limit is None
        else _nondominated_training_indices(y_train, int(gp_limit))
    )
    # Deviation from upstream: normalize designs once using the true problem
    # bounds.  Upstream first uses the full dataset min-max envelope (including
    # its test pool) and then applies problem-bound normalization inside MOBO;
    # that double transform can map RE-task candidates outside their bounds.
    # Our NDS cap also keeps final-front order, whereas upstream uses crowding
    # distance when the last admitted front must be truncated.
    x_scaler = BoundsScaler.fit(x_lower, x_upper)
    # Fit the objective z-score on the selected offline dataset and use this
    # same transform for both GP targets and the raw-scale reference point.
    y_scaler = Standardizer.fit(y_train)
    dtype = torch.double
    train_x = torch.as_tensor(
        x_scaler.transform(x_train[indices]), dtype=dtype, device=device
    )
    train_y = torch.as_tensor(
        -y_scaler.transform(y_train[indices]), dtype=dtype, device=device
    )
    model = SingleTaskGP(train_x, train_y)
    likelihood = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(likelihood)
    model.eval()
    return MoboPredictor(
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        dtype=dtype,
        baseline_x=train_x,
        baseline_y=train_y,
    )


def optimize_neural(
    predictor: Predictor,
    task,
    x_train: np.ndarray,
    y_train: np.ndarray,
    n_gen: int,
    pop_size: int,
    opt_seed: int,
):
    set_seed(opt_seed)

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.core.repair import Repair
    from pymoo.optimize import minimize

    from src.offline_moo_adapter import repair_offline_moo_decisions

    true_problem = task.problem

    class OfflineRepair(Repair):
        def _do(self, problem, x, **kwargs):
            return repair_offline_moo_decisions(true_problem, x)

    class SurrogateProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=int(true_problem.n_var),
                n_obj=int(true_problem.n_obj),
                xl=np.asarray(true_problem.xl, dtype=float),
                xu=np.asarray(true_problem.xu, dtype=float),
            )

        def _evaluate(self, x, out, *args, **kwargs):
            repaired = repair_offline_moo_decisions(true_problem, x)
            out["F"] = predictor.predict(repaired)

    initial = upstream_nds_initial_population(
        x_train, y_train, pop_size, opt_seed, true_problem
    )
    algorithm = NSGA2(
        pop_size=int(pop_size),
        sampling=initial,
        repair=OfflineRepair(),
        eliminate_duplicates=True,
    )
    result = minimize(
        SurrogateProblem(),
        algorithm,
        termination=("n_gen", int(n_gen)),
        seed=int(opt_seed),
        verbose=False,
        save_history=False,
    )
    candidates = repair_offline_moo_decisions(
        true_problem, result.algorithm.pop.get("X")
    )
    evaluations = int(result.algorithm.evaluator.n_eval)
    return np.asarray(candidates, dtype=float), evaluations


def optimize_mobo(
    predictor: MoboPredictor,
    task,
    raw_reference_point: np.ndarray,
    pop_size: int,
    opt_seed: int,
    config: dict[str, Any],
):
    import inspect
    import torch
    from botorch.acquisition.multi_objective.monte_carlo import (
        qNoisyExpectedHypervolumeImprovement,
    )
    from botorch.optim import optimize_acqf
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.box_decompositions.non_dominated import (
        FastNondominatedPartitioning,
    )

    from src.offline_moo_adapter import repair_offline_moo_decisions

    set_seed(opt_seed)
    dtype, device = predictor.dtype, predictor.device
    train_x = predictor.baseline_x
    train_y = predictor.baseline_y
    requested_ref = -predictor.y_scaler.transform(
        np.asarray(raw_reference_point, dtype=float).reshape(1, -1)
    )[0]
    # The raw reference is upstream's 1.1*nadir, but unlike upstream we apply
    # the GP target's z-score transform before comparing it with normalized Y.
    # The optional small-data safeguard moves only invalid coordinates below
    # observed maximization values so qNEHVI remains well-defined.
    if bool(config.get("ensure_dominated_reference", True)):
        requested_ref = np.minimum(
            requested_ref,
            train_y.detach().cpu().numpy().min(axis=0) - 1e-6,
        )
    ref_point = torch.as_tensor(requested_ref, dtype=dtype, device=device)

    sampler = SobolQMCNormalSampler(
        sample_shape=torch.Size([int(config["mc_samples"])])
    )
    acquisition_kwargs = {
        "model": predictor.model,
        "ref_point": ref_point.tolist(),
        "X_baseline": train_x,
        "sampler": sampler,
        "prune_baseline": False,
    }
    # Some older Off-MOO/BoTorch combinations accepted an explicit partitioning.
    if "partitioning" in inspect.signature(
        qNoisyExpectedHypervolumeImprovement.__init__
    ).parameters:
        with torch.no_grad():
            posterior_mean = predictor.model.posterior(train_x).mean
        acquisition_kwargs["partitioning"] = FastNondominatedPartitioning(
            ref_point=torch.zeros_like(ref_point), Y=posterior_mean
        )
    acquisition = qNoisyExpectedHypervolumeImprovement(**acquisition_kwargs)

    bounds = torch.stack(
        (
            torch.zeros(train_x.shape[1], dtype=dtype, device=device),
            torch.ones(train_x.shape[1], dtype=dtype, device=device),
        )
    )
    candidates, _ = optimize_acqf(
        acq_function=acquisition,
        bounds=bounds,
        q=int(pop_size),
        num_restarts=int(config["num_restarts"]),
        raw_samples=int(config["raw_samples"]),
        options={
            "batch_limit": int(config["batch_limit"]),
            "maxiter": int(config["maxiter"]),
        },
        sequential=True,
    )
    raw_candidates = predictor.x_scaler.inverse(candidates.detach().cpu().numpy())
    raw_candidates = repair_offline_moo_decisions(task.problem, raw_candidates)
    # Acquisition optimization makes many posterior calls internally, so the
    # NSGA-style surrogate evaluation count is not defined for this method.
    return np.asarray(raw_candidates, dtype=float), float("nan")


def evaluate_candidates(
    problem_name: str,
    task,
    predictor,
    data: dict[str, np.ndarray],
    candidates: np.ndarray,
):
    from src.metrics import get_igd_plus, get_metrics, normalize_objectives
    from src.offline_moo_adapter import evaluate_offline_moo_objectives_and_feasibility

    surrogate_y = predictor.predict(candidates)
    real_y, explicit_feasible = evaluate_offline_moo_objectives_and_feasibility(
        task.problem, candidates
    )
    finite = (
        np.all(np.isfinite(candidates), axis=1)
        & np.all(np.isfinite(surrogate_y), axis=1)
        & np.all(np.isfinite(real_y), axis=1)
    )
    if explicit_feasible is not None:
        finite &= np.asarray(explicit_feasible, dtype=bool)
    candidates, surrogate_y, real_y = (
        values[finite] for values in (candidates, surrogate_y, real_y)
    )
    if len(candidates) == 0:
        raise RuntimeError("No finite feasible candidates remained after true evaluation.")

    reference_values = data["metric_reference_values"]
    hv, objective_min, objective_max, normalized_ref = get_metrics(
        problem_name=problem_name,
        problem=task.problem,
        objective_values=reference_values,
    )
    igd_plus, igd_source = get_igd_plus(
        task.problem,
        objective_min,
        objective_max,
        reference_values,
        fallback_reference_values=data.get("igd_reference_values"),
    )
    surrogate_normalized = normalize_objectives(
        surrogate_y, objective_min, objective_max
    )
    real_normalized = normalize_objectives(real_y, objective_min, objective_max)
    return {
        "candidates": candidates,
        "surrogate_y": surrogate_y,
        "real_y": real_y,
        "MSEsur_real": float(np.mean((surrogate_y - real_y) ** 2)),
        "HVsur": float(hv.do(surrogate_normalized)),
        "HVreal": float(hv.do(real_normalized)),
        "IGDplus_sur": float(igd_plus.do(surrogate_normalized)),
        "IGDplus": float(igd_plus.do(real_normalized)),
        "objective_min": objective_min,
        "objective_max": objective_max,
        "normalized_ref": normalized_ref,
        "igd_source": igd_source,
    }


def test_mse(predictor, data: dict[str, np.ndarray]) -> float:
    prediction = predictor.predict(data["X_test"])
    return float(np.mean((prediction - data["y_test"]) ** 2))


def result_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        row[name]
        for name in (
            "protocol_version",
            "configuration_hash",
            "problem",
            "method",
            "training_size",
            "offline_seed",
            "opt_seed",
            "configured_n_gen",
            "configured_pop_size",
        )
    )


def read_success_keys(path: Path) -> set[tuple[Any, ...]]:
    if not Path(path).exists():
        return set()
    keys = set()
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success":
                continue
            for name in (
                "training_size",
                "offline_seed",
                "opt_seed",
                "configured_n_gen",
                "configured_pop_size",
            ):
                row[name] = int(row[name])
            keys.add(result_key(row))
    return keys


def append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in RESULT_FIELDS})
        handle.flush()
        os.fsync(handle.fileno())


def _base_row(
    problem: str,
    method: str,
    training_size: int,
    offline_seed: int,
    opt_seed: int,
    data: dict[str, np.ndarray],
    n_gen: int,
    pop_size: int,
    configuration_hash: str,
) -> dict[str, Any]:
    return {
        "problem": problem,
        "method": method,
        "dataset_source": "official_pool",
        "protocol_version": PROTOCOL_VERSION,
        "configuration_hash": configuration_hash,
        "training_size": int(training_size),
        "offline_sample_size": int(training_size),
        "fit_size": int(len(data["X_train"])),
        "test_size": int(len(data["X_test"])),
        "offline_seed": int(offline_seed),
        "model_seed": int(offline_seed),
        "opt_seed": int(opt_seed),
        "subset_indices_hash": subset_hash(data),
        "configured_n_gen": int(n_gen),
        "configured_pop_size": int(pop_size),
        "status": "failed",
        "error_message": "",
    }


def run_group(
    *,
    method: str,
    problem: str,
    training_size: int,
    offline_seed: int,
    opt_seeds: Iterable[int],
    all_training_sizes: Sequence[int],
    subset_cache_dir: Path,
    output_dir: Path,
    n_gen: int,
    pop_size: int,
    neural_config: dict[str, Any],
    com_config: dict[str, Any],
    mobo_config: dict[str, Any],
    device_name: str,
    completed: set[tuple[Any, ...]],
) -> list[dict[str, Any]]:
    ensure_import_paths()
    task, data = load_data(
        problem,
        training_size,
        offline_seed,
        all_training_sizes,
        subset_cache_dir,
    )
    relevant_config = {
        "method": method,
        "neural": neural_config if method != "MOBO-Vallina" else None,
        "com": com_config if method == "MultipleModels-COM" else None,
        "mobo": mobo_config if method == "MOBO-Vallina" else None,
    }
    configuration_hash = hashlib.sha256(
        json.dumps(relevant_config, sort_keys=True).encode("utf-8")
    ).hexdigest()
    pending = []
    for opt_seed in opt_seeds:
        probe = _base_row(
            problem,
            method,
            training_size,
            offline_seed,
            int(opt_seed),
            data,
            n_gen,
            pop_size,
            configuration_hash,
        )
        if result_key(probe) not in completed:
            pending.append(int(opt_seed))
    if not pending:
        return []

    device = resolve_device(device_name)
    training_started = time.perf_counter()
    try:
        if method == "MOBO-Vallina":
            predictor = fit_mobo_predictor(
                data,
                mobo_config,
                offline_seed,
                device,
                task.problem.xl,
                task.problem.xu,
            )
        else:
            predictor = fit_neural_predictor(
                method,
                data,
                neural_config,
                com_config,
                offline_seed,
                device,
            )
        training_time = time.perf_counter() - training_started
        mse_pre = test_mse(predictor, data)
    except Exception as error:
        training_time = time.perf_counter() - training_started
        failed_rows = []
        for opt_seed in pending:
            row = _base_row(
                problem,
                method,
                training_size,
                offline_seed,
                opt_seed,
                data,
                n_gen,
                pop_size,
                configuration_hash,
            )
            row["runtime_surrogate_training"] = training_time
            row["error_message"] = f"{type(error).__name__}: {error}"
            failed_rows.append(row)
        return failed_rows

    rows = []
    for opt_seed in pending:
        row = _base_row(
            problem,
            method,
            training_size,
            offline_seed,
            opt_seed,
            data,
            n_gen,
            pop_size,
            configuration_hash,
        )
        if method == "MOBO-Vallina":
            row["fit_size"] = int(len(predictor.baseline_x))
        row["runtime_surrogate_training"] = training_time
        row["MSEpre"] = mse_pre
        optimization_started = time.perf_counter()
        try:
            if method == "MOBO-Vallina":
                candidates, evaluation_count = optimize_mobo(
                    predictor,
                    task,
                    1.1 * np.asarray(task.nadir_point, dtype=float),
                    pop_size,
                    opt_seed,
                    mobo_config,
                )
                generation_count = 1
            else:
                candidates, evaluation_count = optimize_neural(
                    predictor,
                    task,
                    data["X_train"],
                    data["y_train"],
                    n_gen,
                    pop_size,
                    opt_seed,
                )
                generation_count = n_gen
            evaluation = evaluate_candidates(
                problem, task, predictor, data, candidates
            )
            candidate_dir = output_dir / "candidates" / method
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate_path = candidate_dir / (
                f"{problem}_N{training_size}_offline{offline_seed}_opt{opt_seed}.npz"
            )
            np.savez_compressed(
                candidate_path,
                X=evaluation["candidates"],
                F_surrogate=evaluation["surrogate_y"],
                F_real=evaluation["real_y"],
            )
            row.update(
                {
                    "MSEsur_real": evaluation["MSEsur_real"],
                    "HVsur": evaluation["HVsur"],
                    "HVreal": evaluation["HVreal"],
                    "IGDplus_sur": evaluation["IGDplus_sur"],
                    "IGDplus": evaluation["IGDplus"],
                    "objective_min": json.dumps(
                        evaluation["objective_min"].tolist()
                    ),
                    "objective_max": json.dumps(
                        evaluation["objective_max"].tolist()
                    ),
                    "hv_reference_point_normalized": json.dumps(
                        evaluation["normalized_ref"].tolist()
                    ),
                    "igdplus_reference_source": evaluation["igd_source"],
                    "submitted_solution_count": int(len(candidates)),
                    "number_of_feasible_solutions": int(
                        len(evaluation["candidates"])
                    ),
                    "optimizer_generation_count": int(generation_count),
                    "surrogate_evaluation_count": (
                        int(evaluation_count)
                        if np.isfinite(evaluation_count)
                        else np.nan
                    ),
                    "candidate_file": str(candidate_path.relative_to(output_dir)),
                    "status": "success",
                }
            )
        except Exception as error:
            row["error_message"] = f"{type(error).__name__}: {error}"
        row["runtime_optimization"] = time.perf_counter() - optimization_started
        rows.append(row)
    return rows
