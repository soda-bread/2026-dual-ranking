"""Dependency-light checks for configuration, normalization, and resume keys."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.DL_baseline.core import (
    PROTOCOL_VERSION,
    Standardizer,
    append_row,
    read_success_keys,
    result_key,
)


class StandardizerTests(unittest.TestCase):
    def test_round_trip_and_constant_column(self):
        values = np.asarray([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]])
        scaler = Standardizer.fit(values)
        transformed = scaler.transform(values)
        np.testing.assert_allclose(scaler.inverse(transformed), values)
        np.testing.assert_allclose(transformed[:, 1], 0.0)


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
