import unittest

import numpy as np

from src.relation_correction import OOFCoverageMarginU


class RelationCorrectionChecks(unittest.TestCase):
    def test_oof_uses_only_within_fold_pairs(self):
        truth = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
        mean = np.array([[0.0, 1.0], [1.0, 0.0], [5.0, 6.0], [6.0, 5.0]])
        model = OOFCoverageMarginU.from_oof(
            truth, mean, np.ones_like(truth), [0, 0, 1, 1]
        )
        self.assertEqual(model.coverage_bias, 1.0)
        np.testing.assert_allclose(model.calibration, [0.0, 1.0])

    def test_endpoint_swap_reverses_relation(self):
        rng = np.random.default_rng(4)
        mean = rng.normal(size=(40, 3))
        std = rng.uniform(0.1, 1.0, mean.shape)
        model = OOFCoverageMarginU(np.ones(3), 0.1)
        first, second = np.arange(20), np.arange(20, 40)
        forward = model.predict_relations(mean, std, first, second)
        reverse = model.predict_relations(mean, std, second, first)
        np.testing.assert_array_equal(forward.relation, -reverse.relation)
        np.testing.assert_array_equal(forward.score, reverse.score)

    def test_positive_objective_rescaling_is_invariant(self):
        rng = np.random.default_rng(8)
        mean = rng.normal(size=(40, 3))
        std = rng.uniform(0.1, 1.0, mean.shape)
        factor = np.array([100.0, 0.1, 3.0])
        first, second = np.arange(20), np.arange(20, 40)
        model = OOFCoverageMarginU(np.ones(3), 0.2)
        original = model.predict_relations(mean, std, first, second)
        scaled = model.predict_relations(
            mean * factor, std * factor, first, second
        )
        np.testing.assert_array_equal(original.relation, scaled.relation)

    def test_ties_abstain_and_report_shortfall(self):
        model = OOFCoverageMarginU(np.ones(2), 1.0)
        prediction = model.predict_relations(
            np.zeros((3, 2)), np.ones((3, 2)), [0, 0, 1], [1, 2, 2]
        )
        self.assertEqual(prediction.requested_count, 3)
        self.assertEqual(prediction.actual_count, 0)

    def test_duplicate_pairs_are_rejected(self):
        model = OOFCoverageMarginU(np.ones(2), 0.0)
        with self.assertRaises(ValueError):
            model.predict_relations(
                np.ones((2, 2)), np.ones((2, 2)), [0, 1], [1, 0]
            )

    def test_single_fold_is_rejected(self):
        with self.assertRaises(ValueError):
            OOFCoverageMarginU.from_oof(
                np.ones((4, 2)), np.ones((4, 2)), np.ones((4, 2)), [0, 0, 0, 0]
            )


if __name__ == "__main__":
    unittest.main()
