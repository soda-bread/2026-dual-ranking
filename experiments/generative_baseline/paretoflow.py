"""ParetoFlow adapter using the implementation already vendored by Off-MOO."""

from __future__ import annotations

import numpy as np

from experiments.generative_baseline.common import (
    fit_shared_proxy,
    matrix,
    proxy_tensor,
    rank_and_crowding_indices,
    set_seed,
)


def _flow_classes():
    """Load, rather than duplicate, the complete vendored ParetoFlow method."""

    from src.offline_moo_adapter import ensure_offline_moo_on_path

    ensure_offline_moo_on_path()
    import off_moo_baselines.paretoflow.paretoflow_nets as paretoflow_nets

    return paretoflow_nets


class ParetoFlowModel:
    def __init__(self, flow, proxy, device):
        self.flow = flow
        self.proxy = proxy
        self.device = device


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
    flow = paretoflow_nets.FlowMatching(
        paretoflow_nets.VectorFieldNet(x.shape[1], int(config["hidden_size"])),
        float(config["sigma"]),
        x.shape[1],
        int(config["sampling_steps"]),
        prob_path=str(config["probability_path"]),
    ).to(device)
    optimizer = torch.optim.Adam(flow.parameters(), lr=float(config["learning_rate"]))
    batch_size = min(int(config["batch_size"]), len(x))
    for _ in range(int(config["epochs"])):
        permutation = torch.randperm(len(x_scaled), device=device)
        for start in range(0, len(x_scaled), batch_size):
            batch = x_scaled[permutation[start : start + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss = flow(batch)
            loss.backward()
            optimizer.step()
    flow.eval()
    return ParetoFlowModel(flow, proxy, device)


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
