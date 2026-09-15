"""ParetoFlow adapter using the implementation already vendored by Off-MOO."""

from __future__ import annotations

import numpy as np

from experiments.generative_baseline.common import (
    fit_shared_proxy,
    matrix,
    proxy_tensor,
    rank_and_crowding_indices,
    reference_directions,
    set_seed,
)


def _flow_classes():
    """Load, rather than duplicate, the vendored ParetoFlow network classes."""

    from src.offline_moo_adapter import ensure_offline_moo_on_path

    ensure_offline_moo_on_path()
    from off_moo_baselines.paretoflow.paretoflow_nets import FlowMatching, VectorFieldNet

    return FlowMatching, VectorFieldNet


class ParetoFlowModel:
    def __init__(self, flow, proxy, device):
        self.flow = flow
        self.proxy = proxy
        self.device = device


def fit_paretoflow(data, config, proxy_config, model_seed, device):
    import torch

    FlowMatching, VectorFieldNet = _flow_classes()
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
    flow = FlowMatching(
        VectorFieldNet(x.shape[1], int(config["hidden_size"])),
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
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 10.0)
            optimizer.step()
    flow.eval()
    return ParetoFlowModel(flow, proxy, device)


def _guided_sample(model, weights, config, raw_lower, raw_upper):
    import torch

    flow, proxy, device = model.flow, model.proxy, model.device
    count = len(weights)
    x = torch.randn((count, flow.D), dtype=torch.float32, device=device)
    weights_tensor = torch.as_tensor(weights, dtype=x.dtype, device=device)
    lower = torch.as_tensor(
        proxy.x_scaler.transform(np.asarray(raw_lower).reshape(1, -1))[0],
        dtype=x.dtype,
        device=device,
    )
    upper = torch.as_tensor(
        proxy.x_scaler.transform(np.asarray(raw_upper).reshape(1, -1))[0],
        dtype=x.dtype,
        device=device,
    )
    steps = int(config["sampling_steps"])
    times = torch.linspace(0.0, 1.0, steps, device=device)
    delta = 1.0 / max(steps - 1, 1)
    threshold = float(config["guidance_threshold"])
    gamma = float(config["guidance_scale"])
    for time in times[1:]:
        time_value = float(time)
        with torch.no_grad():
            embedding = flow.time_embedding(time.reshape(1, 1)).expand_as(x)
            velocity = flow.vnet(x + embedding)
        if time_value >= threshold and gamma:
            guided_x = x.detach().requires_grad_(True)
            with torch.no_grad():
                embedding = flow.time_embedding(time.reshape(1, 1)).expand_as(guided_x)
                base_velocity = flow.vnet(guided_x.detach() + embedding)
            if flow.prob_path == "icfm":
                estimated_final = guided_x + (1.0 - time_value) * base_velocity
            else:
                estimated_final = (
                    base_velocity * (1.0 - (1.0 - flow.sigma) * time_value)
                    + (1.0 - flow.sigma) * guided_x
                )
            prediction = proxy_tensor(
                proxy, estimated_final, x_is_scaled=True, y_scaled=True
            )
            utility = -torch.sum(prediction * weights_tensor)
            gradient = torch.autograd.grad(utility, guided_x)[0]
            velocity = velocity + gamma * (1.0 - time_value) * gradient
        x = torch.clamp(x + delta * velocity, lower, upper).detach()
    return x


def generate_paretoflow(model, data, config, opt_seed, output_size, task):
    from src.offline_moo_adapter import repair_offline_moo_decisions

    set_seed(opt_seed)
    oversample = max(1, int(config["oversample_factor"]))
    base_directions = reference_directions(
        data["y_train"].shape[1], output_size, opt_seed
    )
    weights = np.repeat(base_directions, oversample, axis=0)
    raw_lower = np.asarray(task.problem.xl, dtype=float)
    raw_upper = np.asarray(task.problem.xu, dtype=float)
    samples = _guided_sample(model, weights, config, raw_lower, raw_upper)
    raw = model.proxy.x_scaler.inverse(samples.detach().cpu().numpy())
    raw = np.clip(raw, raw_lower, raw_upper)
    raw = repair_offline_moo_decisions(task.problem, raw)
    predicted = model.proxy.predict(raw)
    selected = rank_and_crowding_indices(predicted, output_size)
    return raw[selected], predicted[selected]
