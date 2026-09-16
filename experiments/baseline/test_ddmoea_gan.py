"""Focused tests for the paper and small-data DDMOEA/GAN switches."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, RidgeCV

from experiments.baseline.ddmoea_gan import (
    Discriminator,
    Generator,
    RBFN,
    build_poly_models,
    construct_surrogate_pool_with_gan,
    discriminator_confidence_score,
    initial_population_has_repeated_rows,
)


class ArchitectureTests(unittest.TestCase):
    def test_paper_generator_and_discriminator(self):
        generator = Generator(n_var=30, n_obj=2)
        discriminator = Discriminator(n_var=30, n_obj=2)
        generator_linears = [
            layer for layer in generator.net if isinstance(layer, torch.nn.Linear)
        ]
        discriminator_linears = [
            layer
            for layer in discriminator.net
            if isinstance(layer, torch.nn.Linear)
        ]
        self.assertEqual(generator.latent_dim, 30)
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in generator_linears],
            [(30, 30), (30, 32)],
        )
        self.assertEqual(
            [
                (layer.in_features, layer.out_features)
                for layer in discriminator_linears
            ],
            [(32, 30), (30, 1)],
        )


class SurrogateSwitchTests(unittest.TestCase):
    def test_polynomial_ridge_switch(self):
        rng = np.random.default_rng(3)
        x = rng.uniform(size=(50, 6))
        y = rng.normal(size=(50, 2))
        bounds = x.min(axis=0), x.max(axis=0)
        _, paper_regs = build_poly_models(x, y, *bounds, ridge=False)
        _, ridge_regs = build_poly_models(x, y, *bounds, ridge=True)
        self.assertTrue(all(isinstance(model, LinearRegression) for model in paper_regs))
        self.assertTrue(all(isinstance(model, RidgeCV) for model in ridge_regs))
        self.assertTrue(all(model.alpha_ > 0.0 for model in ridge_regs))

    def test_mean_distance_width_and_center_cap(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(5, 8))
        y = rng.normal(size=5)
        model = RBFN(
            n_centers=8,
            width="mean_distance",
            center_cap="half_train",
        ).fit(x, y)
        self.assertEqual(model.n_centers_, 2)
        self.assertGreater(model.sigma_, 0.0)
        self.assertAlmostEqual(model.gamma_, 1.0 / (2.0 * model.sigma_**2))
        self.assertGreater(model.activation_mean_, 0.0)

    def test_rbfn_fits_paper_bias_term(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(12, 3))
        y = np.full(12, 4.25)
        model = RBFN(n_centers=3, lambda_reg=1e-6).fit(x, y)
        prediction, _ = model.predict(x)
        np.testing.assert_allclose(prediction[:, 0], y, atol=1e-6)
        self.assertAlmostEqual(float(model.bias[0]), 4.25, places=5)

    def test_repeated_initializer_disables_pymoo_duplicate_elimination(self):
        unique = np.arange(12, dtype=float).reshape(4, 3)
        repeated = np.vstack((unique, unique[:2]))
        self.assertFalse(initial_population_has_repeated_rows(unique))
        self.assertTrue(initial_population_has_repeated_rows(repeated))

    def test_repeated_n50_initializer_preserves_10000_evaluation_budget(self):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem
        from pymoo.optimize import minimize

        class ToyProblem(Problem):
            def __init__(self):
                super().__init__(
                    n_var=2,
                    n_obj=2,
                    xl=np.zeros(2),
                    xu=np.ones(2),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                out["F"] = np.column_stack(
                    (x[:, 0], 1.0 - x[:, 0] + x[:, 1])
                )

        rng = np.random.default_rng(10)
        fifty_rows = rng.random((50, 2))
        initial_population = np.vstack((fifty_rows, fifty_rows))
        algorithm = NSGA2(
            pop_size=100,
            sampling=initial_population,
            eliminate_duplicates=not initial_population_has_repeated_rows(
                initial_population
            ),
        )
        result = minimize(
            ToyProblem(),
            algorithm,
            ("n_gen", 100),
            seed=1,
            verbose=False,
        )
        self.assertEqual(result.algorithm.evaluator.n_eval, 10_000)
        self.assertEqual(len(result.pop), 100)

    def test_bagging_and_literal_accumulation(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(size=(10, 2))
        y = np.column_stack((x[:, 0] ** 2, x[:, 1] ** 2))
        generator = Generator(2, 2)
        discriminator = Discriminator(2, 2)

        torch.manual_seed(7)
        bagged, _ = construct_surrogate_pool_with_gan(
            x,
            y,
            generator,
            discriminator,
            torch.device("cpu"),
            n_models=3,
            accumulate_synthetic=False,
            verbose=False,
        )
        torch.manual_seed(7)
        accumulated, _ = construct_surrogate_pool_with_gan(
            x,
            y,
            generator,
            discriminator,
            torch.device("cpu"),
            n_models=3,
            accumulate_synthetic=True,
            verbose=False,
        )
        self.assertEqual(
            [model.model.n_samples_fit_ for model in bagged[0]], [12, 12, 12]
        )
        self.assertEqual(
            [model.model.n_samples_fit_ for model in accumulated[0]],
            [12, 14, 16],
        )

    def test_data_relative_score_normalization(self):
        class FirstCoordinate(torch.nn.Module):
            def forward(self, joint):
                return joint[:, :1]

        confidence = discriminator_confidence_score(
            x=np.array([[0.0], [0.5], [1.0]]),
            y_pred=np.zeros((3, 1)),
            discriminator=FirstCoordinate(),
            device=torch.device("cpu"),
            x_min=np.array([0.0]),
            x_max=np.array([1.0]),
            f_min=np.array([0.0]),
            f_max=np.array([1.0]),
            score_norm="data_minmax",
            score_reference=(-1.0, 1.0),
        )
        np.testing.assert_allclose(confidence[:, 0], [0.0, 0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
