from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from pymoo.core.problem import Problem

from experiments.generative_baseline.paretoflow import (
    fit_paretoflow,
    generate_paretoflow,
)
from experiments.generative_baseline.common import configuration_hash
from experiments.generative_baseline.pcd import (
    _ema_decay_at_step,
    _resolved_config,
    _torch_components,
    fit_pcd,
    generate_pcd,
)
from experiments.generative_baseline.proxy import (
    compute_pcc,
    fit_paretoflow_proxy,
)


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
            "trainer": "paretoflow_upstream",
            "hidden_sizes": [16, 16],
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 0.001,
            "lr_decay": 0.98,
            "validation_fraction": 0.25,
            "min_validation_rows": 4,
        }

    def test_pcd_fit_and_generate(self):
        config = {
            "width": 16,
            "depth": 1,
            "time_dim": 8,
            "learned_sinusoidal_cond": False,
            "random_fourier_features": True,
            "learned_sinusoidal_dim": 8,
            "layer_norm": False,
            "diffusion_steps": 4,
            "train_steps": 2,
            "batch_size": 8,
            "learning_rate": 0.0003,
            "weight_decay": 0.0,
            "adam_betas": [0.9, 0.99],
            "gradient_clip_norm": 1.0,
            "ema_beta": 0.995,
            "ema_update_every": 10,
            "ema_update_after_step": 100,
            "ema_power": 2.0 / 3.0,
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
            "condition_base_points": 4,
            "alpha_range": [0.1, 0.4],
            "condition_noise": 0.0,
        }
        model = fit_pcd(self.data, config, 1, "cpu")
        x, y = generate_pcd(model, self.data, config, 2, 4, self.task)
        self.assertEqual(x.shape, (4, 3))
        self.assertIsNone(y)
        self.assertTrue(np.all(np.isfinite(x)))

    def test_pcd_official_network_layout_and_re_override(self):
        import torch

        _, ConditionalDenoiser, _ = _torch_components()
        synthetic = ConditionalDenoiser(
            3, 2, width=16, depth=2, time_dim=8,
            learned_sinusoidal_cond=False,
            random_fourier_features=True,
            learned_sinusoidal_dim=8,
            layer_norm=False,
        )
        self.assertEqual(synthetic.projection.in_features, 5)
        self.assertEqual(synthetic.projection.out_features, 8)
        self.assertEqual(synthetic.input.in_features, 8)
        self.assertFalse(synthetic.time[0].weights.requires_grad)
        self.assertFalse(
            any(isinstance(module, torch.nn.LayerNorm) for module in synthetic.modules())
        )

        resolved = _resolved_config(
            {"width": 256, "re_overrides": {"width": 512}}, "re21"
        )
        self.assertEqual(resolved["width"], 512)

    def test_pcd_official_ema_decay_schedule(self):
        self.assertAlmostEqual(
            _ema_decay_at_step(101, 100, 0.995, 2.0 / 3.0), 0.0
        )
        self.assertAlmostEqual(
            _ema_decay_at_step(110, 100, 0.995, 2.0 / 3.0),
            1.0 - 10.0 ** (-2.0 / 3.0),
        )
        self.assertEqual(
            _ema_decay_at_step(100_000, 100, 0.995, 2.0 / 3.0), 0.995
        )

    def test_paretoflow_fit_and_generate(self):
        config = {
            "hidden_size": 16,
            "sigma": 0.0,
            "probability_path": "icfm",
            "learning_rate": 0.001,
            "train_steps": 2,
            "min_train_steps": 1,
            "batch_size": 8,
            "validation_fraction": 0.25,
            "min_validation_rows": 4,
            "validation_interval": 1,
            "validation_repeats": 2,
            "patience": 1,
            "sampling_steps": 4,
            "guidance_scale": 2.0,
            "guidance_threshold": 0.5,
            "offspring_count": 2,
            "neighborhood_size": 0,
            "stochastic_step": 0.1,
            "distance": "cosine",
            "init_method": "d_best",
            "adaptive": False,
        }
        model = fit_paretoflow(self.data, config, self.proxy, 1, "cpu")
        x, y = generate_paretoflow(model, self.data, config, 2, 4, self.task)
        self.assertEqual(x.shape, (4, 3))
        self.assertEqual(y.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(x)))

    def test_paretoflow_early_stopping_waits_for_minimum_steps(self):
        config = {
            "hidden_size": 16,
            "sigma": 0.0,
            "probability_path": "icfm",
            # A zero rate makes the deterministic validation loss constant:
            # step 5 is best, then patience is exhausted after steps 6 and 7.
            "learning_rate": 0.0,
            "train_steps": 20,
            "min_train_steps": 5,
            "batch_size": 8,
            "validation_fraction": 0.25,
            "min_validation_rows": 4,
            "validation_interval": 1,
            "validation_repeats": 2,
            "patience": 1,
            "sampling_steps": 4,
        }
        model = fit_paretoflow(self.data, config, self.proxy, 3, "cpu")
        self.assertTrue(model.validation_enabled)
        self.assertEqual(model.training_steps, 7)
        self.assertTrue(np.isnan(model.epochs_trained))
        self.assertLessEqual(
            model.best_validation_loss, model.last_validation_loss
        )

    def test_paretoflow_small_data_fallback_runs_all_steps(self):
        config = {
            "hidden_size": 16,
            "sigma": 0.0,
            "probability_path": "icfm",
            "learning_rate": 0.001,
            "train_steps": 4,
            "min_train_steps": 2,
            "batch_size": 8,
            "validation_fraction": 0.25,
            "min_validation_rows": 20,
            "validation_interval": 1,
            "validation_repeats": 2,
            "patience": 1,
            "sampling_steps": 4,
        }
        model = fit_paretoflow(self.data, config, self.proxy, 3, "cpu")
        self.assertFalse(model.validation_enabled)
        self.assertEqual(model.training_steps, 4)
        self.assertTrue(np.isnan(model.best_validation_loss))

    def test_paretoflow_proxy_reloads_maximum_validation_pcc(self):
        import torch

        rng = np.random.default_rng(21)
        x = rng.normal(size=(48, 3))
        y = np.column_stack((2.0 * x[:, 0] - x[:, 1], x[:, 2] + x[:, 0]))
        data = {
            "X_train": x,
            "y_train": y,
        }
        config = {
            "hidden_sizes": [16, 16],
            "epochs": 40,
            "batch_size": 8,
            "learning_rate": 0.001,
            "lr_decay": 0.98,
            "validation_fraction": 0.25,
            "min_validation_rows": 4,
        }
        predictor = fit_paretoflow_proxy(data, config, 5, "cpu")
        indices = predictor.validation_indices
        x_validation = torch.as_tensor(
            predictor.x_scaler.transform(x[indices]), dtype=torch.float32
        )
        y_validation = torch.as_tensor(
            predictor.y_scaler.transform(y[indices]), dtype=torch.float32
        )
        for objective, model in enumerate(predictor.models):
            with torch.no_grad():
                restored_pcc = float(
                    compute_pcc(
                        model(x_validation),
                        y_validation[:, objective : objective + 1],
                    ).item()
                )
            history = predictor.validation_pcc_history[objective]
            self.assertAlmostEqual(restored_pcc, max(history), places=6)
            self.assertAlmostEqual(
                predictor.best_validation_pcc[objective], max(history), places=6
            )

    def test_paretoflow_new_keys_change_configuration_hash(self):
        old = {
            "algorithm": {"epochs": 1000, "patience": 20},
            "proxy": {"epochs": 200},
        }
        new = {
            "algorithm": {
                "train_steps": 10000,
                "min_train_steps": 1000,
                "validation_interval": 100,
                "validation_repeats": 4,
                "patience": 20,
            },
            "proxy": {
                "epochs": 200,
                "trainer": "paretoflow_upstream",
                "lr_decay": 0.98,
            },
        }
        self.assertNotEqual(
            configuration_hash("ParetoFlow", old),
            configuration_hash("ParetoFlow", new),
        )


if __name__ == "__main__":
    unittest.main()
