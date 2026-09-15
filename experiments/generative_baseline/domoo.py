"""DOMOO risk-controlled nested Pareto-set learning adapter."""

from __future__ import annotations

import numpy as np

from experiments.generative_baseline.common import (
    fit_shared_proxy,
    matrix,
    nondominated_indices,
    proxy_tensor,
    select_domoo_candidates,
    set_seed,
)


def _make_mlp(input_dim, hidden_sizes, output_dim):
    import torch.nn as nn

    layers = []
    width = int(input_dim)
    for hidden in hidden_sizes:
        layers.extend((nn.Linear(width, int(hidden)), nn.LeakyReLU()))
        width = int(hidden)
    layers.append(nn.Linear(width, int(output_dim)))
    return nn.Sequential(*layers)


def _langevin(values, objective_model, step_size, steps, noise=True):
    import torch

    current = values.detach().clone()
    for index in range(int(steps)):
        if noise:
            current = current + 0.001 * (int(steps) - index) / max(int(steps), 1) * torch.randn_like(current)
        current.requires_grad_(True)
        score = objective_model(current).sum()
        gradient = torch.autograd.grad(score, current)[0]
        current = (current - float(step_size) * gradient).detach()
    return current


class DOMOOModel:
    def __init__(self, proxy, pareto_set, energy_models, energy_bounds, x_lower, x_upper, device):
        self.proxy = proxy
        self.pareto_set = pareto_set
        self.energy_models = energy_models
        self.energy_bounds = energy_bounds
        self.x_lower = x_lower
        self.x_upper = x_upper
        self.device = device

    def decode(self, preferences):
        import torch

        unit = self.pareto_set(preferences)
        return self.x_lower + unit * (self.x_upper - self.x_lower)


def _risk_confidence(energy_models, energy_bounds, x):
    import torch

    confidence = []
    for model, (low, high) in zip(energy_models, energy_bounds):
        energy = model(x).reshape(-1)
        value = (float(high) - energy) / max(float(high) - float(low), 1e-8)
        confidence.append(torch.clamp(value, 0.0, 1.0))
    return torch.stack(confidence, dim=1)


def fit_domoo(data, config, proxy_config, model_seed, device):
    import torch

    set_seed(model_seed)
    proxy = fit_shared_proxy(data, proxy_config, model_seed, device)
    for objective_model in proxy.models:
        objective_model.eval()
        for parameter in objective_model.parameters():
            parameter.requires_grad_(False)
    x = matrix(data["X_train"], "X_train")
    y = matrix(data["y_train"], "y_train")
    x_scaled_np = proxy.x_scaler.transform(x)
    y_scaled_np = proxy.y_scaler.transform(y)
    x_scaled = torch.as_tensor(x_scaled_np, dtype=torch.float32, device=device)
    nd = nondominated_indices(y)
    nd_x = x_scaled[torch.as_tensor(nd, dtype=torch.long, device=device)]
    step_size = float(config["langevin_step_size"]) * np.sqrt(x.shape[1])

    energy_models = []
    energy_bounds = []
    batch_size = min(int(config["batch_size"]), len(x))
    for objective, objective_model in enumerate(proxy.models):
        set_seed(int(model_seed) + 10_000 * objective)
        energy = _make_mlp(x.shape[1], config["energy_hidden_sizes"], 1).to(device)
        optimizer = torch.optim.Adam(energy.parameters(), lr=float(config["energy_learning_rate"]))
        for _ in range(int(config["energy_epochs"])):
            permutation = torch.randperm(len(x_scaled), device=device)
            for start in range(0, len(x_scaled), batch_size):
                positive = x_scaled[permutation[start : start + batch_size]]
                negative = _langevin(
                    positive, objective_model, step_size,
                    config["langevin_steps"], noise=True,
                )
                positive_energy, negative_energy = energy(positive), energy(negative)
                loss = (
                    positive_energy.mean() - negative_energy.mean()
                    + positive_energy.square().mean() + negative_energy.square().mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        energy.eval()
        for parameter in energy.parameters():
            parameter.requires_grad_(False)
        boundary = _langevin(
            nd_x, objective_model, step_size,
            config["boundary_langevin_steps"], noise=False,
        )
        with torch.no_grad():
            first = float(energy(nd_x).mean().cpu())
            second = float(energy(boundary).mean().cpu())
        energy_models.append(energy)
        energy_bounds.append((min(first, second), max(first, second) + 1e-8))

    import torch.nn as nn

    class ParetoSet(nn.Module):
        def __init__(self, n_obj, n_dim, width):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(n_obj, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
                nn.Linear(width, n_dim), nn.Sigmoid(),
            )

        def forward(self, preference):
            return self.network(preference)

    x_low_np, x_high_np = np.min(x_scaled_np, axis=0), np.max(x_scaled_np, axis=0)
    x_lower = torch.as_tensor(x_low_np, dtype=torch.float32, device=device)
    x_upper = torch.as_tensor(x_high_np, dtype=torch.float32, device=device)
    pareto_set = ParetoSet(y.shape[1], x.shape[1], int(config["pareto_width"])).to(device)
    z = np.min(y_scaled_np, axis=0) - 0.1
    differences = np.maximum(y_scaled_np[nd] - z, 1e-6)
    preferences = 1.0 / differences
    preferences /= np.sum(preferences, axis=1, keepdims=True)
    preference_tensor = torch.as_tensor(preferences, dtype=torch.float32, device=device)
    target_unit = (nd_x - x_lower) / torch.clamp(x_upper - x_lower, min=1e-8)
    pre_optimizer = torch.optim.Adam(pareto_set.parameters(), lr=float(config["pareto_learning_rate"]))
    for _ in range(int(config["pretrain_epochs"])):
        prediction = pareto_set(preference_tensor)
        loss = torch.mean((prediction - target_unit) ** 2)
        pre_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        pre_optimizer.step()

    z_tensor = torch.as_tensor(z, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(pareto_set.parameters(), lr=float(config["pareto_learning_rate"]))
    preference_count = int(config["preference_batch_size"])
    for step in range(int(config["training_steps"])):
        raw_pref = np.random.dirichlet(np.ones(y.shape[1]), size=preference_count)
        logits = torch.log(torch.as_tensor(raw_pref, dtype=torch.float32, device=device) + 1e-8)
        if step >= int(config["exploration_steps"]) and int(config["preference_steps"]) > 0:
            logits = logits.detach().requires_grad_(True)
            preference_optimizer = torch.optim.Adam([logits], lr=float(config["preference_learning_rate"]))
            for _ in range(int(config["preference_steps"])):
                preference = torch.softmax(logits, dim=1)
                generated = x_lower + pareto_set(preference) * (x_upper - x_lower)
                objective = proxy_tensor(proxy, generated, x_is_scaled=True, y_scaled=True)
                confidence = _risk_confidence(energy_models, energy_bounds, generated)
                scalar = torch.max(preference * (objective - z_tensor), dim=1).values
                # Refine preferences toward under-covered, low-risk trade-offs.
                diversity = torch.pdist(objective).mean() if len(objective) > 1 else 0.0
                pref_loss = scalar.mean() - 0.01 * diversity + float(config["risk_ratio"]) * (1 - confidence).mean()
                preference_optimizer.zero_grad(set_to_none=True)
                pref_loss.backward()
                preference_optimizer.step()
            preference = torch.softmax(logits.detach(), dim=1)
        else:
            preference = torch.softmax(logits, dim=1)

        generated = x_lower + pareto_set(preference) * (x_upper - x_lower)
        objective = proxy_tensor(proxy, generated, x_is_scaled=True, y_scaled=True)
        confidence = _risk_confidence(energy_models, energy_bounds, generated)
        scalar = torch.max(preference * (objective - z_tensor), dim=1).values
        loss = scalar.mean() + float(config["risk_ratio"]) * (1.0 - confidence).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pareto_set.parameters(), 10.0)
        optimizer.step()

    pareto_set.eval()
    return DOMOOModel(
        proxy, pareto_set, tuple(energy_models), tuple(energy_bounds),
        x_lower, x_upper, device,
    )


def generate_domoo(model, data, config, opt_seed, output_size, task):
    import torch

    from experiments.DL_baseline.core import optimize_neural
    from src.offline_moo_adapter import repair_offline_moo_decisions

    set_seed(opt_seed)
    preference = np.random.dirichlet(
        np.ones(data["y_train"].shape[1]), size=int(config["candidate_count"])
    )
    preference_tensor = torch.as_tensor(
        preference, dtype=torch.float32, device=model.device
    )
    with torch.no_grad():
        psl_scaled = model.decode(preference_tensor)
        psl_raw = model.proxy.x_scaler.inverse(psl_scaled.cpu().numpy())
    psl_raw = np.clip(psl_raw, np.asarray(task.problem.xl), np.asarray(task.problem.xu))
    psl_raw = repair_offline_moo_decisions(task.problem, psl_raw)
    psl_y = model.proxy.predict(psl_raw)

    surrogate_x, _ = optimize_neural(
        model.proxy,
        task,
        data["X_train"],
        data["y_train"],
        int(config["surrogate_generations"]),
        max(int(output_size), int(config["surrogate_population"])),
        int(opt_seed),
    )
    surrogate_y = model.proxy.predict(surrogate_x)
    all_x = np.vstack((psl_raw, surrogate_x))
    all_y = np.vstack((psl_y, surrogate_y))
    return select_domoo_candidates(all_x, all_y, data["y_train"], output_size)
