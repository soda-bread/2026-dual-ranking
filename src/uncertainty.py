"""Fixed upper-quantile utilities shared by dual-ranking surrogates."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np


def validate_upper_quantile(quantile):
    """Validate a one-sided upper Gaussian quantile."""

    quantile = float(quantile)
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1.0).")
    return quantile


def gaussian_upper_scale(quantile=0.90):
    """Return the Gaussian z-score for a one-sided upper quantile."""

    return float(NormalDist().inv_cdf(validate_upper_quantile(quantile)))


def _as_objective_matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError(f"{name} must contain at least two rows.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite.")
    return values


def _positive_part(estimate, se):
    """Return the one-standard-error positive part of an estimate."""

    estimate = float(estimate)
    se = float(se)
    if not np.isfinite(estimate) or not np.isfinite(se) or se < 0:
        raise ValueError("estimate and standard error must be finite; se >= 0.")
    return max(estimate - se, 0.0)


def _mean_and_standard_deviation(prediction, residual_rms):
    if isinstance(prediction, tuple) and len(prediction) >= 2:
        mean, std = prediction[:2]
    else:
        mean, std = prediction, None
    mean = np.asarray(mean, dtype=float).reshape(-1)
    if std is None:
        std = np.full_like(mean, float(residual_rms), dtype=float)
    else:
        std = np.asarray(std, dtype=float).reshape(-1)
        if std.shape != mean.shape:
            raise ValueError("predictive mean and std must have matching shapes.")
        std = np.where(np.isfinite(std), np.maximum(std, 0.0), residual_rms)
        # Preserve a data-scaled numerical floor without introducing a tuning
        # constant. This matters for crossed QR quantiles that yield zero.
        floor = np.sqrt(np.finfo(float).eps) * max(float(residual_rms), np.finfo(float).tiny)
        std = np.maximum(std, floor)
    if not np.all(np.isfinite(mean)):
        raise ValueError("OOF predictive means must be finite.")
    return mean, std


def cv_weakness(oof_mean, y, fold_ids):
    """Return fold-wise and aggregate normalized OOF prediction weakness."""

    oof_mean = _as_objective_matrix(oof_mean, "oof_mean")
    y = _as_objective_matrix(y, "y")
    fold_ids = np.asarray(fold_ids, dtype=int).reshape(-1)
    if oof_mean.shape != y.shape or len(fold_ids) != len(y):
        raise ValueError("OOF predictions, targets, and fold ids must align.")
    fold_values = []
    for fold in np.unique(fold_ids):
        selected = fold_ids == fold
        target = y[selected]
        prediction = oof_mean[selected]
        centered = target - target.mean(axis=0, keepdims=True)
        scale = np.mean(centered**2, axis=0)
        scale = np.maximum(scale, np.finfo(float).tiny)
        fold_values.append(np.mean((target - prediction) ** 2, axis=0) / scale)
    fold_values = np.asarray(fold_values, dtype=float)
    return fold_values, np.mean(fold_values, axis=0)


def cv_oof_predictions(model_factory, X, y, n_folds=5, seed=0):
    """Fit deterministic K-fold models and return out-of-fold uncertainty.

    ``model_factory(objective_index, fold_index, fold_seed)`` must return a
    fresh estimator exposing ``fit(X, y)`` and ``predict(X)``. Predict may
    return either a mean or ``(mean, std)``. Mean-only estimators use the
    validation-fold residual RMS as their uncertainty fallback.
    """

    X = np.asarray(X, dtype=float)
    y = _as_objective_matrix(y, "y")
    if X.ndim != 2 or len(X) != len(y) or not np.all(np.isfinite(X)):
        raise ValueError("X must be a finite 2D array aligned with y.")
    n_folds = min(int(n_folds), len(X))
    if n_folds < 2:
        raise ValueError("n_folds must be at least two.")

    permutation = np.random.default_rng(int(seed)).permutation(len(X))
    folds = np.array_split(permutation, n_folds)
    fold_ids = np.empty(len(X), dtype=int)
    oof_mean = np.empty_like(y, dtype=float)
    oof_std = np.empty_like(y, dtype=float)

    all_indices = np.arange(len(X))
    for fold_index, validation_indices in enumerate(folds):
        fold_ids[validation_indices] = fold_index
        training_indices = np.setdiff1d(
            all_indices, validation_indices, assume_unique=True
        )
        for objective_index in range(y.shape[1]):
            fold_seed = int(seed) + 1009 * fold_index + 9176 * objective_index
            model = model_factory(objective_index, fold_index, fold_seed)
            try:
                fitted = model.fit(X[training_indices], y[training_indices, objective_index])
                if fitted is not None:
                    model = fitted
                raw_prediction = model.predict(X[validation_indices])
                if isinstance(raw_prediction, tuple):
                    raw_mean = np.asarray(raw_prediction[0], dtype=float).reshape(-1)
                else:
                    raw_mean = np.asarray(raw_prediction, dtype=float).reshape(-1)
                residual_rms = float(
                    np.sqrt(np.mean((y[validation_indices, objective_index] - raw_mean) ** 2))
                )
                mean, std = _mean_and_standard_deviation(
                    raw_prediction, residual_rms
                )
                oof_mean[validation_indices, objective_index] = mean
                oof_std[validation_indices, objective_index] = std
            finally:
                cleanup = getattr(model, "cleanup", None)
                if callable(cleanup):
                    cleanup()

    weakness = cv_weakness(oof_mean, y, fold_ids)[1]
    return oof_mean, oof_std, fold_ids, weakness


def eb_shrinkage_params(
    oof_mean,
    oof_std,
    y,
    fold_ids,
    *,
    tau2_rule="floor",
    c_rule="dispersion",
):
    """Estimate the three-regime empirical-Bayes shrinkage parameters."""

    oof_mean = _as_objective_matrix(oof_mean, "oof_mean")
    oof_std = _as_objective_matrix(oof_std, "oof_std")
    y = _as_objective_matrix(y, "y")
    fold_ids = np.asarray(fold_ids, dtype=int).reshape(-1)
    if oof_mean.shape != y.shape or oof_std.shape != y.shape:
        raise ValueError("OOF means, standard deviations, and y must align.")
    if len(fold_ids) != len(y):
        raise ValueError("fold_ids must contain one entry per row.")
    if tau2_rule not in {"floor", "positive_part"}:
        raise ValueError("tau2_rule must be 'floor' or 'positive_part'.")
    if c_rule not in {"dispersion", "calibrated"}:
        raise ValueError("c_rule must be 'dispersion' or 'calibrated'.")

    prediction_centered = np.empty_like(oof_mean)
    target_centered = np.empty_like(y)
    for fold in np.unique(fold_ids):
        selected = fold_ids == fold
        prediction_centered[selected] = (
            oof_mean[selected] - oof_mean[selected].mean(axis=0, keepdims=True)
        )
        target_centered[selected] = (
            y[selected] - y[selected].mean(axis=0, keepdims=True)
        )

    n = len(y)
    m = y.mean(axis=0)
    tau2 = np.empty(y.shape[1], dtype=float)
    c = np.empty(y.shape[1], dtype=float)
    slope = np.empty(y.shape[1], dtype=float)
    s2max = np.max(oof_std**2, axis=0)
    signal = np.empty(y.shape[1], dtype=bool)

    for objective_index in range(y.shape[1]):
        p = prediction_centered[:, objective_index]
        target = target_centered[:, objective_index]
        covariance_samples = p * target
        covariance = float(np.mean(covariance_samples))
        se_covariance = float(np.std(covariance_samples, ddof=1) / np.sqrt(n))
        prediction_variance = float(np.mean(p**2))
        target_variance = float(np.mean(target**2))
        floor = 1e-12 * max(target_variance, np.finfo(float).tiny)
        signal[objective_index] = covariance > se_covariance
        if tau2_rule == "floor":
            tau2[objective_index] = max(covariance, floor)
        else:
            tau2[objective_index] = max(
                _positive_part(covariance, se_covariance), floor
            )
        slope[objective_index] = covariance / max(
            prediction_variance, np.finfo(float).tiny
        )

        sigma2_mean = float(np.mean(oof_std[:, objective_index] ** 2))
        if c_rule == "dispersion":
            excess_samples = p**2 - p * target
            excess = float(np.mean(excess_samples))
            se_excess = float(np.std(excess_samples, ddof=1) / np.sqrt(n))
            numerator = _positive_part(excess, se_excess)
        else:
            numerator = float(np.mean((p - target) ** 2))
        c[objective_index] = numerator / max(
            sigma2_mean, np.finfo(float).tiny
        )

    return m, tau2, c, slope, s2max, signal
