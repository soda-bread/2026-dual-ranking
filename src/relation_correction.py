"""Experimental OOF coverage-bias + Margin-U pair classification.

All objectives are minimized. This predicts context-dependent pair relations,
not adjusted objective vectors or a transitive Pareto order. It is deliberately
not wired into an optimizer; see ``experiments/wfg_sample_size/README.md``.
"""

from dataclasses import dataclass

import numpy as np
from scipy.special import log_ndtr

__all__ = ["OOFCoverageMarginU", "RelationPrediction"]


def _matrix(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite 2-D matrix")
    return values


def _relation(values, first, second):
    left, right = values[first], values[second]
    forward = np.all(left <= right + 1e-12, axis=1) & np.any(
        left < right - 1e-12, axis=1
    )
    reverse = np.all(right <= left + 1e-12, axis=1) & np.any(
        right < left - 1e-12, axis=1
    )
    return forward.astype(np.int8) - reverse.astype(np.int8)


@dataclass(frozen=True)
class RelationPrediction:
    """Pair labels and diagnostics returned by the experimental rule."""

    relation: np.ndarray  # +1: first dominates; -1: second; 0: neither
    score: np.ndarray  # log score, not a calibrated probability
    requested_count: int
    actual_count: int


@dataclass(frozen=True)
class OOFCoverageMarginU:
    """Training-only coverage correction followed by Margin-U ranking."""

    calibration: np.ndarray
    coverage_bias: float

    @classmethod
    def from_oof(cls, y_train, oof_mean, oof_std, fold_ids):
        """Fit using predictions made by models excluding each complete fold.

        The caller must ensure that every OOF prediction, including any
        preprocessing, excludes the row's entire fold. Arrays alone cannot
        establish that provenance.
        """
        truth = _matrix(y_train, "y_train")
        mean = _matrix(oof_mean, "oof_mean")
        std = _matrix(oof_std, "oof_std")
        folds = np.asarray(fold_ids)
        if truth.shape != mean.shape or truth.shape != std.shape or np.any(std < 0):
            raise ValueError(
                "Training arrays must share shape and std must be nonnegative"
            )
        if (
            folds.shape != (len(truth),)
            or folds.dtype.kind not in "iu"
            or np.any(folds < 0)
            or len(np.unique(folds)) < 2
        ):
            raise ValueError("Provide one nonnegative integer fold ID per row")

        first_parts, second_parts = [], []
        for fold in np.unique(folds):
            rows = np.flatnonzero(folds == fold)
            first, second = np.triu_indices(len(rows), 1)
            first_parts.append(rows[first])
            second_parts.append(rows[second])
        first = np.concatenate(first_parts)
        second = np.concatenate(second_parts)
        if not len(first):
            raise ValueError("No within-fold pairs are available")

        true_coverage = np.mean(_relation(truth, first, second) != 0)
        predicted_coverage = np.mean(_relation(mean, first, second) != 0)
        calibration = np.sqrt(
            np.sum((truth - mean) ** 2, axis=0)
            / np.maximum(np.sum(std**2, axis=0), 1e-24)
        )
        return cls(calibration, float(true_coverage - predicted_coverage))

    def predict_relations(self, mean, std, first, second):
        """Classify one fixed collection of unique unordered pairs.

        Candidate truth is not an input. The dominance-call budget depends on
        this pair collection, so a pair's label is context dependent. Exact
        direction ties abstain; equal score ties preserve input pair order.
        """
        mean = _matrix(mean, "mean")
        std = _matrix(std, "std")
        first, second = np.asarray(first), np.asarray(second)
        calibration = np.asarray(self.calibration, dtype=float)
        if std.shape != mean.shape or np.any(std < 0):
            raise ValueError("std must match mean and be nonnegative")
        if (
            calibration.shape != (mean.shape[1],)
            or not np.isfinite(calibration).all()
            or np.any(calibration < 0)
            or not np.isfinite(self.coverage_bias)
        ):
            raise ValueError("Invalid fitted calibration")
        if (
            first.ndim != 1
            or second.shape != first.shape
            or not first.size
            or first.dtype.kind not in "iu"
            or second.dtype.kind not in "iu"
            or np.any(first < 0)
            or np.any(second < 0)
            or np.any(first >= len(mean))
            or np.any(second >= len(mean))
            or np.any(first == second)
        ):
            raise ValueError(
                "Provide nonempty integer indices for distinct valid endpoints"
            )
        canonical_pairs = np.sort(np.column_stack([first, second]), axis=1)
        if len(np.unique(canonical_pairs, axis=0)) != len(first):
            raise ValueError("Each unordered pair must occur only once")

        centre = _relation(mean, first, second)
        requested = int(
            np.rint(
                len(first)
                * np.clip(np.mean(centre != 0) + self.coverage_bias, 0.0, 1.0)
            )
        )
        pair_std = calibration * np.sqrt(std[first] ** 2 + std[second] ** 2)
        z_score = (mean[second] - mean[first]) / np.maximum(pair_std, 1e-12)
        forward = log_ndtr(z_score).sum(axis=1)
        reverse = log_ndtr(-z_score).sum(axis=1)
        score = np.maximum(forward, reverse)
        direction = np.sign(forward - reverse).astype(np.int8)

        eligible = np.flatnonzero(direction)
        chosen = eligible[
            np.argsort(-score[eligible], kind="stable")[:requested]
        ]
        relation = np.zeros(len(first), dtype=np.int8)
        relation[chosen] = direction[chosen]
        return RelationPrediction(relation, score, requested, len(chosen))
