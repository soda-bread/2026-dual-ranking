"""Dependency-light checks for configuration, normalization, and resume keys."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.DL_baseline.core import (
    BoundsScaler,
    PROTOCOL_VERSION,
    Standardizer,
    append_row,
    load_config,
    read_success_keys,
    result_key,
    upstream_nds_initial_population,
)


class StandardizerTests(unittest.TestCase):
    def test_round_trip_and_constant_column(self):
        values = np.asarray([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]])
        scaler = Standardizer.fit(values)
        transformed = scaler.transform(values)
        np.testing.assert_allclose(scaler.inverse(transformed), values)
        np.testing.assert_allclose(transformed[:, 1], 0.0)

    def test_bounds_scaler_round_trip(self):
        scaler = BoundsScaler.fit(np.array([-2.0, 10.0]), np.array([2.0, 20.0]))
        values = np.array([[-2.0, 10.0], [0.0, 15.0], [2.0, 20.0]])
        normalized = scaler.transform(values)
        np.testing.assert_allclose(normalized, [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        np.testing.assert_allclose(scaler.inverse(normalized), values)


class UpstreamCompatibilityTests(unittest.TestCase):
    def test_default_budget_matches_upstream(self):
        config = load_config(Path(__file__).with_name("config.yaml"))
        self.assertEqual(config["optimizer"], {"n_gen": 50, "pop_size": 256})
        self.assertEqual(config["mobo"]["train_gp_data_size"], 256)

    def test_nds_initialization_and_small_data_fill(self):
        from pymoo.core.problem import Problem

        problem = Problem(n_var=2, n_obj=2, xl=np.zeros(2), xu=np.ones(2))
        x = np.array([[0.9, 0.9], [0.1, 0.9], [0.9, 0.1]])
        y = x.copy()
        initial = upstream_nds_initial_population(x, y, 5, 7, problem)
        self.assertEqual(initial.shape, (5, 2))
        np.testing.assert_allclose(initial[:2], x[[1, 2]])
        np.testing.assert_allclose(initial[2], x[0])
        self.assertTrue(np.all((initial[3:] >= 0.0) & (initial[3:] <= 1.0)))


class ResumeTests(unittest.TestCase):
    def test_successful_row_round_trip(self):
        row = {
            "protocol_version": PROTOCOL_VERSION,
            "configuration_hash": "abc123",
            "problem": "zdt1",
            "method": "End2End-Vallina",
            "training_size": 50,
            "offline_seed": 1,
            "opt_seed": 2,
            "configured_n_gen": 3,
            "configured_pop_size": 10,
            "status": "success",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            append_row(path, row)
            self.assertEqual(read_success_keys(path), {result_key(row)})


if __name__ == "__main__":
    unittest.main()
