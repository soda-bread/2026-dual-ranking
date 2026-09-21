"""Regression tests for the formal EBU-DR primary-method path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from experiments.method_registry import MethodSpec
from experiments.sample_size_common import _cached_ebu_survival
from src.models import qr_prediction_mean_std
from src.survival import (
    Survival_eb_shrinkage,
    Survival_standard,
)
from src.uncertainty import _positive_part, cv_oof_predictions, eb_shrinkage_params


class _LinearUncertaintyModel:
    fit_count = 0

    def fit(self, X, y):
        type(self).fit_count += 1
        design = np.column_stack([np.ones(len(X)), X])
        self.coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
        residual = y - design @ self.coefficients
        self.std = max(float(np.sqrt(np.mean(residual**2))), 1e-6)
        return self

    def predict(self, X):
        design = np.column_stack([np.ones(len(X)), X])
        mean = design @ self.coefficients
        return mean, np.full(len(X), self.std)


def _factory(objective_index, fold_index, fold_seed):
    del objective_index, fold_index, fold_seed
    return _LinearUncertaintyModel()


class EBUEstimatorTests(unittest.TestCase):
    def test_positive_part_is_james_stein(self):
        self.assertAlmostEqual(
            _positive_part(1.3, 0.15),
            1.3 * (1 - 0.15**2 / 1.3**2),
        )
        self.assertEqual(_positive_part(0.10, 0.15), 0.0)
        self.assertEqual(_positive_part(-0.2, 0.05), 0.0)
        self.assertEqual(_positive_part(0.7, 0.0), 0.7)

    def test_oof_split_and_predictions_are_deterministic(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(30, 2))
        y = np.column_stack([X[:, 0] + 0.1 * X[:, 1], X[:, 1] ** 2])
        first = cv_oof_predictions(_factory, X, y, n_folds=5, seed=7)
        second = cv_oof_predictions(_factory, X, y, n_folds=5, seed=7)
        for left, right in zip(first, second):
            np.testing.assert_allclose(left, right)

    def test_noise_without_detectable_signal_marks_objective_uninformative(self):
        rng = np.random.default_rng(11)
        n = 200
        folds = np.arange(n) % 5
        y = rng.normal(size=(n, 1))
        prediction = rng.normal(size=(n, 1))
        for fold in np.unique(folds):
            selected = folds == fold
            target = y[selected, 0] - y[selected, 0].mean()
            candidate = prediction[selected, 0] - prediction[selected, 0].mean()
            candidate -= target * (
                np.dot(candidate, target) / np.dot(target, target)
            )
            prediction[selected, 0] = candidate
        std = np.ones((n, 1))
        *_, signal = eb_shrinkage_params(prediction, std, y, folds)
        self.assertFalse(bool(signal[0]))

    def test_frequency_simulation_produces_finite_nonnegative_parameters(self):
        rng = np.random.default_rng(9)
        n = 2000
        latent = rng.normal(size=n)
        sigma = rng.uniform(0.2, 0.8, size=n)
        prediction = latent + rng.normal(scale=sigma)
        values = eb_shrinkage_params(
            prediction[:, None],
            sigma[:, None],
            latent[:, None],
            np.arange(n) % 5,
        )
        _, tau2, c, slope, s2max, signal = values
        self.assertTrue(bool(signal[0]))
        self.assertGreater(tau2[0], 0.0)
        self.assertGreaterEqual(c[0], 0.0)
        self.assertLess(slope[0], 1.0)
        self.assertGreater(s2max[0], 0.0)

    def test_exact_signal_has_zero_dispersion_correction(self):
        rng = np.random.default_rng(17)
        y = rng.normal(size=(100, 1))
        std = np.full_like(y, 0.4)
        *_, c, _, _, signal = eb_shrinkage_params(
            y.copy(), std, y, np.arange(len(y)) % 5
        )
        self.assertTrue(bool(signal[0]))
        self.assertAlmostEqual(float(c[0]), 0.0, places=12)

    def test_fold_mean_predictor_has_zero_dispersion_correction(self):
        rng = np.random.default_rng(23)
        y = rng.normal(size=(50, 1))
        folds = np.arange(len(y)) % 5
        prediction = np.empty_like(y)
        for fold in np.unique(folds):
            prediction[folds == fold] = y[folds == fold].mean(axis=0)
        _, _, c, _, _, signal = eb_shrinkage_params(
            prediction, np.ones_like(y), y, folds
        )
        self.assertFalse(bool(signal[0]))
        self.assertAlmostEqual(float(c[0]), 0.0, places=12)

    def test_calibrated_rule_uses_raw_oof_residual(self):
        y = np.array([[0.0], [1.0], [10.0], [11.0]])
        prediction = y + 5.0
        _, _, c, *_ = eb_shrinkage_params(
            prediction,
            np.ones_like(y),
            y,
            np.array([0, 0, 1, 1]),
            c_rule="calibrated",
        )
        self.assertAlmostEqual(float(c[0]), 25.0)

    def test_qr_std_proxy_is_nonnegative_under_quantile_crossing(self):
        prediction = pd.DataFrame(
            {"y_q0.5": [1.0, 2.0], "y_q0.9": [0.5, 3.0]}
        )
        mean, std = qr_prediction_mean_std(prediction)
        np.testing.assert_allclose(mean, [1.0, 2.0])
        self.assertEqual(std[0], 0.0)
        self.assertGreater(std[1], 0.0)


class EBUSurvivalTests(unittest.TestCase):
    def test_c_zero_matches_standard_survivors(self):
        rng = np.random.default_rng(5)
        F = rng.normal(size=(40, 2))
        std = rng.uniform(0.1, 1.0, size=(40, 2))
        standard_pop = Population.new(F=F, std=std)
        ebu_pop = Population.new(F=F, std=std)
        standard = Survival_standard()._do(
            None,
            standard_pop,
            n_survive=20,
            random_state=np.random.RandomState(3),
        )
        ebu = Survival_eb_shrinkage(
            m=[0.0, 0.0],
            tau2=[1.0, 1.0],
            c=[0.0, 0.0],
            s2max=[1.0, 1.0],
            signal=[True, True],
        )._do(
            None,
            ebu_pop,
            n_survive=20,
            random_state=np.random.RandomState(3),
        )
        np.testing.assert_allclose(standard.get("F"), ebu.get("F"))

    def test_shrinkage_is_monotone_and_saturates_at_s2max(self):
        survival = Survival_eb_shrinkage(
            m=[0.0], tau2=[1.0], c=[1.0], s2max=[4.0], signal=[True]
        )
        F = np.full((4, 1), 2.0)
        adjusted = survival.adjusted(F, [[0.0], [1.0], [2.0], [4.0]])[:, 0]
        self.assertTrue(np.all(np.diff(adjusted) <= 0.0))
        self.assertEqual(adjusted[-1], adjusted[-2])

    def test_homoscedastic_shrinkage_matches_standard_survivors(self):
        rng = np.random.default_rng(19)
        F = rng.normal(size=(50, 2))
        std = np.full_like(F, 0.5)
        standard = Survival_standard()._do(
            None,
            Population.new(F=F, std=std),
            n_survive=25,
            random_state=np.random.RandomState(8),
        )
        ebu = Survival_eb_shrinkage(
            m=[0.1, -0.2],
            tau2=[1.0, 2.0],
            c=[0.7, 1.3],
            s2max=[1.0, 1.0],
            signal=[True, True],
        )._do(
            None,
            Population.new(F=F, std=std),
            n_survive=25,
            random_state=np.random.RandomState(8),
        )
        np.testing.assert_allclose(standard.get("F"), ebu.get("F"))

    def test_uninformative_objective_uses_sigma_view(self):
        survival = Survival_eb_shrinkage(
            m=[2.0], tau2=[1.0], c=[1.0], s2max=[1.0], signal=[False]
        )
        std = np.array([[0.2], [0.7]])
        np.testing.assert_allclose(survival.adjusted([[9.0], [3.0]], std), std)

    def test_n_survive_is_exact(self):
        rng = np.random.default_rng(2)
        pop = Population.new(F=rng.normal(size=(17, 2)), std=np.ones((17, 2)))
        survival = Survival_eb_shrinkage(
            m=[0.0, 0.0],
            tau2=[1.0, 1.0],
            c=[1.0, 1.0],
            s2max=[1.0, 1.0],
            signal=[True, True],
        )
        selected = survival._do(
            None, pop, n_survive=7, random_state=np.random.RandomState(1)
        )
        self.assertEqual(len(selected), 7)

    def test_c_zero_matches_standard_three_generation_trajectory(self):
        class _QuadraticProblem(Problem):
            def __init__(self):
                super().__init__(n_var=2, n_obj=2, xl=-2.0, xu=2.0)

            def _evaluate(self, X, out, *args, **kwargs):
                out["F"] = np.column_stack([
                    np.sum((X - 0.5) ** 2, axis=1),
                    np.sum((X + 0.5) ** 2, axis=1),
                ])
                out["std"] = np.full((len(X), 2), 0.25)

        problem = _QuadraticProblem()
        standard = minimize(
            problem,
            NSGA2(pop_size=20, survival=Survival_standard()),
            termination=("n_gen", 3),
            seed=31,
            verbose=False,
        )
        ebu = minimize(
            problem,
            NSGA2(
                pop_size=20,
                survival=Survival_eb_shrinkage(
                    m=[0.0, 0.0],
                    tau2=[1.0, 1.0],
                    c=[0.0, 0.0],
                    s2max=[1.0, 1.0],
                    signal=[True, True],
                ),
            ),
            termination=("n_gen", 3),
            seed=31,
            verbose=False,
        )
        np.testing.assert_allclose(standard.pop.get("X"), ebu.pop.get("X"))
        np.testing.assert_allclose(standard.pop.get("F"), ebu.pop.get("F"))
class EBUCacheTests(unittest.TestCase):
    def test_cache_hit_does_not_refit_oof_models(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(20, 2))
        y = np.column_stack([X[:, 0], X[:, 1]])
        data = {
            "X_train": X,
            "y_train": y,
            "dataset_source": np.asarray("official_pool"),
            "offline_indices": np.arange(len(X)),
        }
        spec = MethodSpec("test EBU-DR", "gpr_rbf", "ebu_dr")
        _LinearUncertaintyModel.fit_count = 0
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "experiments.sample_size_common._oof_model_factory",
                return_value=_factory,
            ):
                _cached_ebu_survival(directory, spec, data, "zdt1", 20, 3)
                first_count = _LinearUncertaintyModel.fit_count
                _cached_ebu_survival(directory, spec, data, "zdt1", 20, 3)
        self.assertEqual(first_count, 10)
        self.assertEqual(_LinearUncertaintyModel.fit_count, first_count)


if __name__ == "__main__":
    unittest.main()
