import unittest
import sys
import types

import numpy as np
import pandas as pd

# The lightweight protocol tests do not instantiate the BNN paths that require
# StandardScaler. Keep them runnable in the repository's minimal system Python.
try:
    import sklearn.preprocessing  # noqa: F401
except ImportError:
    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    preprocessing_module.StandardScaler = object
    sklearn_module.preprocessing = preprocessing_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module

from src.models import (
    GPR_Matern,
    GPR_RBF,
    autogluon_qr_predict,
)
from src.uncertainty import (
    gaussian_upper_scale,
    reflect_upper_quantile,
)


class _FakeGPyModel:
    def __init__(self):
        self.include_likelihood_calls = []

    def predict(self, X, include_likelihood):
        self.include_likelihood_calls.append(include_likelihood)
        variance = 9.0 if include_likelihood else 4.0
        return np.zeros((len(X), 1)), np.full((len(X), 1), variance)


class _FakeQuantileModel:
    def predict(self, frame):
        self.columns = tuple(frame.columns)
        return pd.DataFrame(
            {
                0.5: [2.0, 2.0],
                0.8: [1.5, 2.4],
                0.9: [1.0, 2.2],
                0.95: [2.5, 2.1],
            }
        )


class UncertaintyProtocolTests(unittest.TestCase):
    def test_gpr_default_is_epistemic_only(self):
        for model_class in (GPR_RBF, GPR_Matern):
            model = model_class()
            model.model = _FakeGPyModel()
            _, epistemic_std = model.predict(np.zeros((3, 2)))
            self.assertEqual(model.model.include_likelihood_calls, [False])
            np.testing.assert_allclose(epistemic_std, 2.0)

    def test_quantile_crossing_uses_original_reflection(self):
        q50 = np.array([2.0])
        q90 = np.array([1.0])
        reflected = reflect_upper_quantile(q50, q90)
        np.testing.assert_allclose(reflected, [3.0])

    def test_autogluon_preserves_raw_crossed_quantiles(self):
        model = _FakeQuantileModel()
        prediction = autogluon_qr_predict(model, np.zeros((2, 3)))
        self.assertEqual(model.columns, ("x0", "x1", "x2"))
        self.assertEqual(prediction.loc[0, "y_q0.5"], 2.0)
        self.assertEqual(prediction.loc[0, "y_q0.9"], 1.0)

    def test_gaussian_bound_uses_matching_quantile_weight(self):
        expected = {0.80: 0.841621, 0.90: 1.281552, 0.95: 1.644854}
        for quantile, weight in expected.items():
            self.assertAlmostEqual(
                gaussian_upper_scale(quantile), weight, places=6
            )

if __name__ == "__main__":
    unittest.main()
