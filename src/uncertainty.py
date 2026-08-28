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


def reflect_upper_quantile(center, raw_upper):
    """Preserve the original dual-ranking upper-spread construction.

    A crossed upper quantile is reflected around the center, so the bound is
    exactly ``center + abs(raw_upper - center)``.
    """

    center = np.asarray(center, dtype=float)
    raw_upper = np.asarray(raw_upper, dtype=float)
    if center.shape != raw_upper.shape:
        raise ValueError("Center and upper prediction shapes must match.")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(raw_upper)):
        raise ValueError("Center and upper predictions must be finite.")
    return center + np.abs(raw_upper - center)
