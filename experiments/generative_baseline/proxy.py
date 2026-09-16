"""ParetoFlow-specific objective proxy training with validation checkpoints."""

from __future__ import annotations

import copy
import math

import numpy as np

from experiments.DL_MOBO_baseline.core import (
    Predictor,
    Standardizer,
    _make_mlp,
    set_seed,
)


def seeded_validation_split(
    row_count,
    validation_fraction,
    min_validation_rows,
    seed,
):
    """Return deterministic train/validation indices or disable validation."""

    import torch

    row_count = int(row_count)
    minimum = int(min_validation_rows)
    fraction = float(validation_fraction)
    if minimum < 1:
        raise ValueError("min_validation_rows must be positive.")
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    permutation = torch.randperm(row_count, generator=generator).numpy()
    if row_count < 2 * minimum:
        return permutation, np.empty(0, dtype=int), False
    validation_rows = max(minimum, int(math.ceil(fraction * row_count)))
    validation_rows = min(validation_rows, row_count - minimum)
    return permutation[validation_rows:], permutation[:validation_rows], True


def compute_pcc(prediction, target):
    """Pearson correlation used by the upstream ParetoFlow proxy trainer."""

    import torch

    vx = prediction - torch.mean(prediction)
    vy = target - torch.mean(target)
    return torch.sum(vx * vy) / (
        torch.sqrt(torch.sum(vx**2) + 1e-12)
        * torch.sqrt(torch.sum(vy**2) + 1e-12)
    )


def _matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError(f"{name} must be a non-empty 2D array; got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values.")
    return values


def fit_paretoflow_proxy(data, config, model_seed, device):
    """Train one objective MLP at a time using ParetoFlow's proxy protocol."""

    import torch

    set_seed(model_seed)
    x = _matrix(data["X_train"], "X_train")
    y = _matrix(data["y_train"], "y_train")
    x_scaler = Standardizer.fit(x)
    y_scaler = Standardizer.fit(y)
    x_scaled = torch.as_tensor(
        x_scaler.transform(x), dtype=torch.float32, device=device
    )
    y_scaled = torch.as_tensor(
        y_scaler.transform(y), dtype=torch.float32, device=device
    )
    train_indices, validation_indices, validation_enabled = seeded_validation_split(
        len(x),
        config.get("validation_fraction", 0.1),
        config.get("min_validation_rows", 10),
        model_seed,
    )
    train_indices = torch.as_tensor(train_indices, dtype=torch.long, device=device)
    validation_indices_tensor = torch.as_tensor(
        validation_indices, dtype=torch.long, device=device
    )
    hidden_sizes = tuple(int(value) for value in config["hidden_sizes"])
    epochs = int(config["epochs"])
    batch_size = int(config["batch_size"])
    learning_rate = float(config.get("learning_rate", 1e-3))
    lr_decay = float(config.get("lr_decay", 0.98))
    if epochs < 1 or batch_size < 1:
        raise ValueError("ParetoFlow proxy epochs and batch_size must be positive.")
    if learning_rate <= 0.0 or not 0.0 < lr_decay <= 1.0:
        raise ValueError("ParetoFlow proxy learning rates are invalid.")

    models = []
    histories = []
    best_values = []
    epochs_trained = []
    for objective in range(y.shape[1]):
        objective_seed = int(model_seed) + 10_000 * objective
        set_seed(objective_seed)
        model = _make_mlp(x.shape[1], 1, hidden_sizes).to(
            device=device, dtype=torch.float32
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_pcc = -float("inf")
        best_state = copy.deepcopy(model.state_dict())
        history = []
        completed_epochs = 0
        for epoch in range(epochs):
            model.train()
            permutation = train_indices[
                torch.randperm(len(train_indices), device=device)
            ]
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start : start + batch_size]
                prediction = model(x_scaled[indices])
                target = y_scaled[indices, objective : objective + 1]
                loss = torch.sum(torch.mean((prediction - target) ** 2, dim=1))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            completed_epochs = epoch + 1
            if validation_enabled:
                model.eval()
                with torch.no_grad():
                    validation_prediction = model(x_scaled[validation_indices_tensor])
                    validation_target = y_scaled[
                        validation_indices_tensor, objective : objective + 1
                    ]
                    pcc = float(
                        compute_pcc(validation_prediction, validation_target).item()
                    )
                history.append(pcc)
                if pcc > best_pcc:
                    best_pcc = pcc
                    best_state = copy.deepcopy(model.state_dict())

            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] *= lr_decay
            if validation_enabled and history[-1] == 1.0:
                break

        if validation_enabled:
            model.load_state_dict(best_state)
        else:
            best_pcc = float("nan")
        model.eval()
        models.append(model)
        histories.append(tuple(history))
        best_values.append(best_pcc)
        epochs_trained.append(completed_epochs)

    predictor = Predictor(
        models=tuple(models),
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        device=device,
        joint=False,
    )
    predictor.validation_enabled = validation_enabled
    predictor.validation_indices = np.asarray(validation_indices, dtype=int)
    predictor.validation_pcc_history = tuple(histories)
    predictor.best_validation_pcc = tuple(best_values)
    predictor.epochs_trained = tuple(epochs_trained)
    return predictor
