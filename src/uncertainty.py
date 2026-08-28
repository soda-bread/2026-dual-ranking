"""Fixed upper-quantile utilities shared by dual-ranking surrogates."""

from __future__ import annotations

from statistics import NormalDist


def validate_upper_quantile(quantile):
    """Validate a one-sided upper Gaussian quantile."""

    quantile = float(quantile)
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must be in (0.5, 1.0).")
    return quantile


def gaussian_upper_scale(quantile=0.90):
    """Return the Gaussian z-score for a one-sided upper quantile."""

    return float(NormalDist().inv_cdf(validate_upper_quantile(quantile)))
