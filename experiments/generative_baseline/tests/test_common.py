from __future__ import annotations

import unittest

import numpy as np

from experiments.generative_baseline.common import (
    Standardizer,
    condition_points,
    rank_and_crowding_indices,
)
from experiments.generative_baseline.pcd import pareto_reweight


class CommonTest(unittest.TestCase):
    def test_standardizer_round_trip(self):
        values = np.array([[1.0, 7.0], [3.0, 7.0], [5.0, 7.0]])
        scaler = Standardizer.fit(values)
        np.testing.assert_allclose(scaler.inverse(scaler.transform(values)), values)
        np.testing.assert_allclose(scaler.scale, [np.std([1.0, 3.0, 5.0]), 1.0])

    def test_rank_and_crowding_is_deterministic(self):
        values = np.array(
            [[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]]
        )
        first = rank_and_crowding_indices(values, 3)
        second = rank_and_crowding_indices(values, 3)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 3)
        self.assertIn(0, first)
        self.assertIn(4, first)

    def test_conditions_follow_reference_directions_without_clipping(self):
        values = np.array(
            [[0.1, 0.2], [0.2, 0.1]]
        )
        conditions = condition_points(values, 10, seed=3, noise=0.0)
        repeated = condition_points(values, 10, seed=3, noise=0.0)
        self.assertEqual(conditions.shape, (10, 2))
        self.assertTrue(np.all(np.isfinite(conditions)))
        np.testing.assert_allclose(conditions, repeated)
        self.assertTrue(np.any(conditions < values.min(axis=0)))

    def test_pcd_reweighting_uses_dominance_count_without_mean_normalization(self):
        values = np.array([[0.0, 1.0], [1.0, 0.0], [1.2, 1.2], [1.4, 1.4]])
        weights = pareto_reweight(values, bins=4, k=10.0, tau=0.1)
        self.assertLess(float(np.mean(weights)), 1.0)
        self.assertGreater(np.min(weights[:2]), np.max(weights[2:]))

    def test_pcd_bin_members_share_their_weight(self):
        values = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        weights = pareto_reweight(values, bins=1, k=10.0, tau=0.1)
        np.testing.assert_allclose(weights, np.repeat(weights[0], 3))


if __name__ == "__main__":
    unittest.main()
