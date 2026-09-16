"""Pareto-Conditioned Diffusion adapted to the shared official-pool protocol."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np

from experiments.generative_baseline.common import (
    Standardizer,
    condition_points,
    matrix,
    rank_and_crowding_indices,
    set_seed,
)


def pareto_reweight(scores, bins=30, k=10.0, tau=0.05):
    """Official PCD dominance-count and objective-bin sample weights."""

    scores = matrix(scores, "scores")
    # dominates[j, i] means sample j dominates sample i (minimization).
    less_equal = scores[:, None, :] <= scores[None, :, :]
    strictly_less = scores[:, None, :] < scores[None, :, :]
    dominates = np.all(less_equal, axis=2) & np.any(strictly_less, axis=2)
    dominance_count = np.sum(dominates, axis=0, dtype=float)
    span = float(np.max(dominance_count) - np.min(dominance_count))
    if span > 1e-12:
        dominance_count = (dominance_count - np.min(dominance_count)) / span
    else:
        dominance_count = np.zeros_like(dominance_count)

    bin_count = int(bins)
    if bin_count < 1:
        raise ValueError("bins must be positive.")
    cell_columns = []
    for objective in range(scores.shape[1]):
        lo, hi = np.min(scores[:, objective]), np.max(scores[:, objective])
        if hi <= lo:
            cell_columns.append(np.zeros(len(scores), dtype=int))
        else:
            edges = np.linspace(lo, hi, bin_count + 1)
            cell_columns.append(
                np.clip(
                    np.digitize(scores[:, objective], edges[1:-1]),
                    0,
                    bin_count - 1,
                )
            )
    cells = np.column_stack(cell_columns)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    weights = np.zeros(len(scores), dtype=float)
    for cell in np.unique(inverse):
        mask = inverse == cell
        population = int(np.sum(mask))
        weights[mask] = population / (population + float(k)) * np.exp(
            -float(np.mean(dominance_count[mask])) / max(float(tau), 1e-8)
        )
    return weights


def _torch_components():
    import torch
    import torch.nn as nn

    class SinusoidalTimeEmbedding(nn.Module):
        def __init__(self, width):
            super().__init__()
            self.width = int(width)

        def forward(self, timesteps):
            half = max(1, self.width // 2)
            frequencies = torch.exp(
                -math.log(10_000.0)
                * torch.arange(half, device=timesteps.device, dtype=timesteps.dtype)
                / max(half - 1, 1)
            )
            angles = timesteps.reshape(-1, 1) * frequencies.reshape(1, -1)
            embedding = torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)
            if embedding.shape[1] < self.width:
                embedding = torch.nn.functional.pad(embedding, (0, self.width - embedding.shape[1]))
            return embedding[:, : self.width]

    class RandomOrLearnedSinusoidalEmbedding(nn.Module):
        def __init__(self, width, random_features):
            super().__init__()
            if int(width) % 2:
                raise ValueError("learned_sinusoidal_dim must be even.")
            self.weights = nn.Parameter(
                torch.randn(int(width) // 2), requires_grad=not bool(random_features)
            )

        def forward(self, timesteps):
            values = timesteps.reshape(-1, 1)
            frequencies = values * self.weights.reshape(1, -1) * (2.0 * math.pi)
            return torch.cat((values, frequencies.sin(), frequencies.cos()), dim=1)

    class ResidualBlock(nn.Module):
        def __init__(self, width, layer_norm):
            super().__init__()
            self.norm = nn.LayerNorm(width) if layer_norm else nn.Identity()
            self.linear = nn.Linear(width, width)

        def forward(self, values):
            return values + self.linear(torch.relu(self.norm(values)))

    class ConditionalDenoiser(nn.Module):
        def __init__(
            self,
            x_dim,
            y_dim,
            width=512,
            depth=4,
            time_dim=128,
            learned_sinusoidal_cond=False,
            random_fourier_features=True,
            learned_sinusoidal_dim=16,
            layer_norm=False,
        ):
            super().__init__()
            if learned_sinusoidal_cond or random_fourier_features:
                embedding = RandomOrLearnedSinusoidalEmbedding(
                    learned_sinusoidal_dim, random_fourier_features
                )
                embedding_dim = int(learned_sinusoidal_dim) + 1
            else:
                embedding = SinusoidalTimeEmbedding(time_dim)
                embedding_dim = int(time_dim)
            self.time = nn.Sequential(
                embedding,
                nn.Linear(embedding_dim, int(time_dim)),
                nn.SiLU(),
                nn.Linear(int(time_dim), int(time_dim)),
            )
            self.projection = nn.Linear(int(x_dim) + int(y_dim), int(time_dim))
            self.input = nn.Linear(int(time_dim), int(width))
            self.blocks = nn.Sequential(
                *(
                    ResidualBlock(int(width), bool(layer_norm))
                    for _ in range(int(depth))
                )
            )
            self.final_norm = nn.LayerNorm(int(width)) if layer_norm else nn.Identity()
            self.output = nn.Linear(int(width), int(x_dim))

        def forward(self, noisy_x, noise_condition, condition):
            embedded = self.time(noise_condition)
            hidden = self.projection(torch.cat((noisy_x, condition), dim=1)) + embedded
            hidden = self.blocks(self.input(hidden))
            return self.output(torch.relu(self.final_norm(hidden)))

    class ConditionalDiffusion(nn.Module):
        """PCD's elucidated-diffusion objective and Heun sampler."""

        def __init__(
            self,
            denoiser,
            steps,
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=1.0,
            rho=7.0,
            p_mean=-1.2,
            p_std=1.2,
            s_churn=80.0,
            s_tmin=0.05,
            s_tmax=50.0,
            s_noise=1.003,
        ):
            super().__init__()
            self.denoiser = denoiser
            self.num_sample_steps = int(steps)
            if self.num_sample_steps < 2:
                raise ValueError("PCD diffusion_steps must be at least 2.")
            self.sigma_min = float(sigma_min)
            self.sigma_max = float(sigma_max)
            self.sigma_data = float(sigma_data)
            self.rho = float(rho)
            self.p_mean = float(p_mean)
            self.p_std = float(p_std)
            self.s_churn = float(s_churn)
            self.s_tmin = float(s_tmin)
            self.s_tmax = float(s_tmax)
            self.s_noise = float(s_noise)

        @property
        def device(self):
            return next(self.parameters()).device

        def _denoise(self, noisy, sigma, condition, guidance_scale=1.0):
            padded = sigma.reshape(-1, 1)
            c_skip = self.sigma_data**2 / (padded.square() + self.sigma_data**2)
            c_out = padded * self.sigma_data / torch.sqrt(
                self.sigma_data**2 + padded.square()
            )
            c_in = torch.rsqrt(padded.square() + self.sigma_data**2)
            c_noise = torch.log(torch.clamp(sigma, min=1e-20)) * 0.25
            conditional = self.denoiser(c_in * noisy, c_noise, condition)
            if guidance_scale != 1.0:
                unconditional = self.denoiser(
                    c_in * noisy, c_noise, torch.zeros_like(condition)
                )
                network_output = (
                    float(guidance_scale) * conditional
                    + (1.0 - float(guidance_scale)) * unconditional
                )
            else:
                network_output = conditional
            return c_skip * noisy + c_out * network_output

        def loss(self, clean_x, condition, weights, cond_drop_prob):
            batch = len(clean_x)
            sigma = torch.exp(
                self.p_mean
                + self.p_std * torch.randn(batch, device=clean_x.device)
            )
            noisy = clean_x + sigma.reshape(-1, 1) * torch.randn_like(clean_x)
            dropped = torch.rand(batch, device=clean_x.device) < float(cond_drop_prob)
            used_condition = condition.clone()
            used_condition[dropped] = 0.0
            denoised = self._denoise(noisy, sigma, used_condition)
            edm_weight = (
                sigma.square() + self.sigma_data**2
            ) / (sigma * self.sigma_data).square()
            point_loss = torch.mean((denoised - clean_x) ** 2, dim=1)
            return torch.mean(point_loss * edm_weight * weights)

        def _schedule(self):
            steps = torch.arange(
                self.num_sample_steps, device=self.device, dtype=torch.float32
            )
            inverse_rho = 1.0 / self.rho
            schedule = (
                self.sigma_max**inverse_rho
                + steps
                / (self.num_sample_steps - 1)
                * (self.sigma_min**inverse_rho - self.sigma_max**inverse_rho)
            ) ** self.rho
            return torch.cat((schedule, schedule.new_zeros(1)))

        @torch.no_grad()
        def sample(self, condition, guidance_scale=2.5):
            schedule = self._schedule()
            values = schedule[0] * torch.randn(
                (len(condition), self.denoiser.output.out_features),
                dtype=condition.dtype,
                device=condition.device,
            )
            for index in range(self.num_sample_steps):
                sigma = float(schedule[index])
                sigma_next = float(schedule[index + 1])
                gamma = (
                    min(self.s_churn / self.num_sample_steps, math.sqrt(2.0) - 1.0)
                    if self.s_tmin <= sigma <= self.s_tmax
                    else 0.0
                )
                sigma_hat = sigma * (1.0 + gamma)
                if gamma:
                    values_hat = values + math.sqrt(
                        max(sigma_hat**2 - sigma**2, 0.0)
                    ) * self.s_noise * torch.randn_like(values)
                else:
                    values_hat = values
                sigma_tensor = torch.full(
                    (len(values),), sigma_hat, device=values.device
                )
                denoised = self._denoise(
                    values_hat, sigma_tensor, condition, guidance_scale
                )
                derivative = (values_hat - denoised) / sigma_hat
                proposal = values_hat + (sigma_next - sigma_hat) * derivative
                if sigma_next:
                    next_tensor = torch.full(
                        (len(values),), sigma_next, device=values.device
                    )
                    denoised_next = self._denoise(
                        proposal, next_tensor, condition, guidance_scale
                    )
                    derivative_next = (proposal - denoised_next) / sigma_next
                    values = values_hat + 0.5 * (sigma_next - sigma_hat) * (
                        derivative + derivative_next
                    )
                else:
                    values = proposal
            return values

    return torch, ConditionalDenoiser, ConditionalDiffusion


class PCDModel:
    def __init__(self, diffusion, x_scaler, y_scaler, device):
        self.diffusion = diffusion
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.device = device

    def generate(self, conditions, guidance_scale, *, conditions_are_scaled=False):
        import torch

        values = (
            np.asarray(conditions, dtype=float)
            if conditions_are_scaled
            else self.y_scaler.transform(conditions)
        )
        condition = torch.as_tensor(
            values, dtype=torch.float32, device=self.device
        )
        self.diffusion.eval()
        generated = self.diffusion.sample(condition, guidance_scale=guidance_scale)
        return self.x_scaler.inverse(generated.detach().cpu().numpy())


def _resolved_config(config, problem_name=None):
    resolved = dict(config)
    overrides = resolved.pop("re_overrides", {}) or {}
    if str(problem_name or "").lower().startswith("re"):
        resolved.update(overrides)
    return resolved


def fit_pcd(data, config, model_seed, device, problem_name=None):
    torch, ConditionalDenoiser, ConditionalDiffusion = _torch_components()
    config = _resolved_config(config, problem_name)
    set_seed(model_seed)
    x, y = matrix(data["X_train"], "X_train"), matrix(data["y_train"], "y_train")
    x_scaler, y_scaler = Standardizer.fit(x), Standardizer.fit(y)
    x_tensor = torch.as_tensor(x_scaler.transform(x), dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(y_scaler.transform(y), dtype=torch.float32, device=device)
    weights = torch.as_tensor(
        pareto_reweight(y, config["bins"], config["density_k"], config["tau"]),
        dtype=torch.float32,
        device=device,
    )
    denoiser = ConditionalDenoiser(
        x.shape[1],
        y.shape[1],
        config["width"],
        config["depth"],
        config["time_dim"],
        config.get("learned_sinusoidal_cond", False),
        config.get("random_fourier_features", True),
        config.get("learned_sinusoidal_dim", 16),
        config.get("layer_norm", False),
    )
    diffusion = ConditionalDiffusion(
        denoiser,
        config["diffusion_steps"],
        config["sigma_min"],
        config["sigma_max"],
        config["sigma_data"],
        config["rho"],
        config["p_mean"],
        config["p_std"],
        config["s_churn"],
        config["s_tmin"],
        config["s_tmax"],
        config["s_noise"],
    ).to(device)
    no_decay = ("bias", "LayerNorm.weight", "norm.weight", ".g")
    parameter_groups = [
        {
            "params": [
                parameter
                for name, parameter in diffusion.named_parameters()
                if not any(token in name for token in no_decay)
            ],
            "weight_decay": float(config["weight_decay"]),
        },
        {
            "params": [
                parameter
                for name, parameter in diffusion.named_parameters()
                if any(token in name for token in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config.get("adam_betas", (0.9, 0.99))),
    )
    train_steps = int(config["train_steps"])
    if train_steps < 1:
        raise ValueError("PCD train_steps must be positive.")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_steps)
    ema_decay = float(config.get("ema_decay", 0.995))
    ema_update_every = int(config.get("ema_update_every", 10))
    if ema_update_every < 1:
        raise ValueError("PCD ema_update_every must be positive.")
    ema_diffusion = deepcopy(diffusion).eval()
    for parameter in ema_diffusion.parameters():
        parameter.requires_grad_(False)
    batch_size = min(int(config["batch_size"]), len(x))
    for step in range(train_steps):
        indices = torch.randint(0, len(x), (batch_size,), device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.loss(
            x_tensor[indices], y_tensor[indices], weights[indices], config["cond_drop_prob"]
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            diffusion.parameters(), float(config.get("gradient_clip_norm", 1.0))
        )
        optimizer.step()
        scheduler.step()
        if step == 0:
            ema_diffusion.load_state_dict(diffusion.state_dict())
        elif step % ema_update_every == 0:
            with torch.no_grad():
                for ema_parameter, parameter in zip(
                    ema_diffusion.parameters(), diffusion.parameters()
                ):
                    ema_parameter.mul_(ema_decay).add_(
                        parameter.detach(), alpha=1.0 - ema_decay
                    )
                for ema_buffer, buffer in zip(
                    ema_diffusion.buffers(), diffusion.buffers()
                ):
                    ema_buffer.copy_(buffer)
    return PCDModel(ema_diffusion, x_scaler, y_scaler, device)


def generate_pcd(model, data, config, opt_seed, output_size, task):
    from src.offline_moo_adapter import repair_offline_moo_decisions

    set_seed(opt_seed)
    y_train = matrix(data["y_train"], "y_train")
    d_best = rank_and_crowding_indices(y_train, min(256, len(y_train)))
    scaled_objectives = model.y_scaler.transform(y_train[d_best])
    targets = condition_points(
        scaled_objectives,
        output_size,
        opt_seed,
        alpha_range=tuple(config["alpha_range"]),
        noise=config["condition_noise"],
        max_base_points=int(config.get("condition_base_points", 32)),
    )
    candidates = model.generate(
        targets, config["guidance_scale"], conditions_are_scaled=True
    )
    candidates = np.clip(
        candidates, np.asarray(task.problem.xl), np.asarray(task.problem.xu)
    )
    return repair_offline_moo_decisions(task.problem, candidates), None
