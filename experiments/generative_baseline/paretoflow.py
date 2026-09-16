"""ParetoFlow adapter using the implementation already vendored by Off-MOO."""

from __future__ import annotations

import copy

import numpy as np

from experiments.generative_baseline.common import (
    fit_shared_proxy,
    matrix,
    proxy_tensor,
    rank_and_crowding_indices,
    set_seed,
)
from experiments.generative_baseline.proxy import seeded_validation_split


def _flow_classes():
    """Load, rather than duplicate, the complete vendored ParetoFlow method."""

    from src.offline_moo_adapter import ensure_offline_moo_on_path

    ensure_offline_moo_on_path()
    import off_moo_baselines.paretoflow.paretoflow_nets as paretoflow_nets

    return paretoflow_nets


class ParetoFlowModel:
    def __init__(
        self,
        flow,
        proxy,
        device,
        epochs_trained,
        best_validation_loss,
        *,
        validation_enabled,
        last_validation_loss=float("nan"),
        validation_loss_history=(),
    ):
        self.flow = flow
        self.proxy = proxy
        self.device = device
        self.epochs_trained = int(epochs_trained)
        self.best_validation_loss = float(best_validation_loss)
        self.validation_enabled = bool(validation_enabled)
        self.last_validation_loss = float(last_validation_loss)
        self.validation_loss_history = tuple(validation_loss_history)


def fit_paretoflow(data, config, proxy_config, model_seed, device):
    import torch

    paretoflow_nets = _flow_classes()
    paretoflow_nets.device = torch.device(device)
    set_seed(model_seed)
    proxy = fit_shared_proxy(data, proxy_config, model_seed, device)
    for objective_model in proxy.models:
        objective_model.eval()
        for parameter in objective_model.parameters():
            parameter.requires_grad_(False)
    x = matrix(data["X_train"], "X_train")
    x_scaled = torch.as_tensor(
        proxy.x_scaler.transform(x), dtype=torch.float32, device=device
    )
    # Proxy training consumes RNG state, so reset before the flow split and
    # initialization to keep both controlled only by model_seed.
    set_seed(model_seed)
    train_indices, validation_indices, validation_enabled = seeded_validation_split(
        len(x),
        config.get("validation_fraction", 0.1),
        config.get("min_validation_rows", 10),
        model_seed,
    )
    train_indices = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    validation_indices = torch.as_tensor(
        validation_indices, dtype=torch.long, device=device
    )
    x_train = x_scaled[train_indices]
    x_validation = x_scaled[validation_indices] if validation_enabled else None
    flow = paretoflow_nets.FlowMatching(
        paretoflow_nets.VectorFieldNet(x.shape[1], int(config["hidden_size"])),
        float(config["sigma"]),
        x.shape[1],
        int(config["sampling_steps"]),
        prob_path=str(config["probability_path"]),
    ).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=float(config["learning_rate"]))
    epochs = int(config["epochs"])
    batch_size = min(int(config["batch_size"]), len(x_train))
    validation_repeats = int(config.get("validation_repeats", 4))
    patience_limit = int(config.get("patience", 20))
    if epochs < 1 or batch_size < 1 or validation_repeats < 1:
        raise ValueError(
            "ParetoFlow epochs, batch_size, and validation_repeats must be positive."
        )
    if patience_limit < 0:
        raise ValueError("ParetoFlow patience must be non-negative.")
    best_state = copy.deepcopy(flow.state_dict())
    best_validation_loss = float("inf")
    validation_history = []
    epochs_trained = 0
    patience = 0
    for epoch in range(epochs):
        flow.train()
        permutation = torch.randperm(len(x_train), device=device)
        for start in range(0, len(x_train), batch_size):
            batch = x_train[permutation[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = flow(batch)
            loss.backward()
            optimizer.step()
        epochs_trained = epoch + 1
        if validation_enabled:
            flow.eval()
            with torch.no_grad():
                validation_loss = float(
                    np.mean(
                        [
                            float(flow(x_validation).item())
                            for _ in range(validation_repeats)
                        ]
                    )
                )
            validation_history.append(validation_loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_state = copy.deepcopy(flow.state_dict())
                patience = 0
            else:
                patience += 1
            if patience > patience_limit:
                break
    if validation_enabled:
        flow.load_state_dict(best_state)
    else:
        best_validation_loss = float("nan")
    flow.eval()
    return ParetoFlowModel(
        flow,
        proxy,
        device,
        epochs_trained,
        best_validation_loss,
        validation_enabled=validation_enabled,
        last_validation_loss=(
            validation_history[-1] if validation_history else float("nan")
        ),
        validation_loss_history=validation_history,
    )


class _OfficialPoolTaskAdapter:
    """Expose only the selected official-pool subset to vendored ParetoFlow."""

    def __init__(self, task, data, proxy):
        self.problem = task.problem
        self.proxy = proxy
        self.x = matrix(data["X_train"], "X_train")
        self.y = matrix(data["y_train"], "y_train")
        self.xl = np.asarray(self.problem.xl, dtype=float)
        self.xu = np.asarray(self.problem.xu, dtype=float)
        self.nadir_point = np.max(self.y, axis=0)
        self.is_discrete = False
        self.is_sequence = False
        self._y_min = np.min(self.y, axis=0)
        self._y_span = np.maximum(np.max(self.y, axis=0) - self._y_min, 1e-12)

    def normalize_x(self, values):
        return self.proxy.x_scaler.transform(values)

    def denormalize_x(self, values):
        # paretoflow_sample immediately sends this temporary diagnostic set
        # back through standardized-input classifiers. Keep it standardized;
        # the returned archive is converted to raw task coordinates outside.
        return np.asarray(values, dtype=float)

    def normalize_y(self, values, normalization_method="z-score"):
        if normalization_method == "min-max":
            return (np.asarray(values, dtype=float) - self._y_min) / self._y_span
        return self.proxy.y_scaler.transform(values)

    def predict(self, values):
        import torch

        inputs = torch.as_tensor(
            values, dtype=torch.float32, device=self.proxy.device
        )
        with torch.no_grad():
            return proxy_tensor(
                self.proxy, inputs, x_is_scaled=True, y_scaled=False
            ).cpu().numpy()

    def get_N_non_dominated_solutions(self, N, return_x=True, return_y=True):
        requested = int(N)
        indices = rank_and_crowding_indices(self.y, min(requested, len(self.y)))
        if len(indices) < requested:
            indices = np.resize(indices, requested)
        selected_x, selected_y = self.x[indices], self.y[indices]
        return (
            selected_x if return_x else None,
            selected_y if return_y else None,
        )


def generate_paretoflow(model, data, config, opt_seed, output_size, task):
    import torch

    from src.offline_moo_adapter import repair_offline_moo_decisions

    set_seed(opt_seed)
    paretoflow_nets = _flow_classes()
    paretoflow_nets.device = torch.device(model.device)
    adapter = _OfficialPoolTaskAdapter(task, data, model.proxy)
    neighborhood_size = int(config.get("neighborhood_size", 0))
    if neighborhood_size <= 0:
        neighborhood_size = data["y_train"].shape[1] + 1
    normalized, _ = model.flow.paretoflow_sample(
        list(model.proxy.models),
        T=int(config["sampling_steps"]),
        O=int(config.get("offspring_count", 5)),
        K=neighborhood_size,
        num_solutions=int(output_size),
        distance=str(config.get("distance", "cosine")),
        g_t=float(config.get("stochastic_step", 0.1)),
        init_method=str(config.get("init_method", "d_best")),
        task=adapter,
        task_name="official-pool",
        t_threshold=float(config["guidance_threshold"]),
        adaptive=bool(config.get("adaptive", False)),
        gamma=float(config["guidance_scale"]),
    )
    raw_lower = np.asarray(task.problem.xl, dtype=float)
    raw_upper = np.asarray(task.problem.xu, dtype=float)
    raw = model.proxy.x_scaler.inverse(np.asarray(normalized, dtype=float))
    raw = np.clip(raw, raw_lower, raw_upper)
    raw = repair_offline_moo_decisions(task.problem, raw)
    predicted = model.proxy.predict(raw)
    selected = rank_and_crowding_indices(predicted, output_size)
    return raw[selected], predicted[selected]
