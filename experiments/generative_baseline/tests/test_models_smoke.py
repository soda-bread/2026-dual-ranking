from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from pymoo.core.problem import Problem

from experiments.generative_baseline.domoo import fit_domoo, generate_domoo
from experiments.generative_baseline.paretoflow import (
    fit_paretoflow,
    generate_paretoflow,
)
from experiments.generative_baseline.pcd import fit_pcd, generate_pcd


class ToyProblem(Problem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, xl=np.zeros(3), xu=np.ones(3))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = objectives(x)


def objectives(x):
    x = np.asarray(x)
    return np.column_stack((x[:, 0] + 0.1 * x[:, 2], (1.0 - x[:, 0]) + 0.1 * x[:, 1]))


class ModelSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        x = rng.uniform(size=(32, 3))
        cls.data = {
            "X_train": x[:24],
            "y_train": objectives(x[:24]),
            "X_test": x[24:],
            "y_test": objectives(x[24:]),
        }
        cls.task = SimpleNamespace(problem=ToyProblem())
        cls.proxy = {
            "hidden_sizes": [16, 16],
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.001,
        }

    def test_domoo_fit_and_generate(self):
        config = {
            "energy_hidden_sizes": [16, 16],
            "energy_learning_rate": 0.001,
            "energy_epochs": 1,
            "batch_size": 8,
            "langevin_step_size": 0.02,
            "langevin_steps": 1,
            "boundary_langevin_steps": 1,
            "pareto_width": 16,
            "pareto_learning_rate": 0.001,
            "pretrain_epochs": 1,
            "training_steps": 1,
            "exploration_steps": 1,
            "preference_batch_size": 4,
            "preference_steps": 0,
            "preference_learning_rate": 0.0001,
            "risk_ratio": 0.001,
            "candidate_count": 8,
            "surrogate_generations": 1,
            "surrogate_population": 4,
        }
        model = fit_domoo(self.data, config, self.proxy, 1, "cpu")
        x, y = generate_domoo(model, self.data, config, 2, 4, self.task)
        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(y.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_pcd_fit_and_generate(self):
        config = {
            "width": 16,
            "depth": 1,
            "time_dim": 8,
            "diffusion_steps": 4,
            "train_steps": 2,
            "batch_size": 8,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "cond_drop_prob": 0.15,
            "guidance_scale": 2.5,
            "sigma_min": 0.002,
            "sigma_max": 8.0,
            "sigma_data": 1.0,
            "rho": 7.0,
            "p_mean": -1.2,
            "p_std": 1.2,
            "s_churn": 1.0,
            "s_tmin": 0.05,
            "s_tmax": 5.0,
            "s_noise": 1.003,
            "bins": 4,
            "density_k": 10.0,
            "tau": 0.05,
            "alpha_range": [0.1, 0.4],
            "condition_noise": 0.0,
        }
        model = fit_pcd(self.data, config, 1, "cpu")
        x, y = generate_pcd(model, self.data, config, 2, 4, self.task)
        self.assertEqual(x.shape, (4, 3))
        self.assertIsNone(y)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_paretoflow_fit_and_generate(self):
        config = {
            "hidden_size": 16,
            "sigma": 0.0,
            "probability_path": "icfm",
            "learning_rate": 0.001,
            "epochs": 1,
            "batch_size": 8,
            "sampling_steps": 4,
            "guidance_scale": 2.0,
            "guidance_threshold": 0.5,
            "oversample_factor": 1,
        }
        model = fit_paretoflow(self.data, config, self.proxy, 1, "cpu")
        x, y = generate_paretoflow(model, self.data, config, 2, 4, self.task)
        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(y.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(x)))


if __name__ == "__main__":
    unittest.main()
