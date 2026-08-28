import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

try:
    import sklearn.preprocessing  # noqa: F401
except ImportError:
    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    preprocessing_module.StandardScaler = object
    sklearn_module.preprocessing = preprocessing_module
    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module

from experiments.sample_size_common import METHOD_REGISTRY, _predictor_models
from src import models


class _RecordingModel:
    instances = []

    def __init__(self, *args, **kwargs):
        self.fit_rows = None
        type(self).instances.append(self)

    def fit(self, X, y):
        self.fit_rows = len(X)
        return self


class FullOfflineTrainingTests(unittest.TestCase):
    def setUp(self):
        self.X_train = np.zeros((10, 3))
        self.y_train = np.zeros((10, 2))
        self.X_test = np.zeros((4, 3))
        self.y_test = np.zeros((4, 2))
        self.data_tuple = (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        )

    def test_standalone_training_helpers_use_every_offline_row(self):
        _RecordingModel.instances = []
        with patch.object(models, "_generate_experiment_data", return_value=self.data_tuple), \
                patch.object(models, "GPR_RBF", _RecordingModel):
            fitted, *_ = models.train_gpr_models(object(), 10, kernel="rbf")
        self.assertEqual([model.fit_rows for model in fitted], [10, 10])

        qr_rows = []

        def fake_qr_fit(X, y, X_test, random_state=42):
            qr_rows.append(len(X))
            return None, object()

        with patch.object(models, "_generate_experiment_data", return_value=self.data_tuple), \
                patch.object(models, "autogluon_qr_fit_predict", fake_qr_fit):
            fitted, *_ = models.train_autogluon_qr_models(object(), 10)
        self.assertEqual(len(fitted), 2)
        self.assertEqual(qr_rows, [10, 10])

        _RecordingModel.instances = []
        with patch.object(models, "_generate_experiment_data", return_value=self.data_tuple), \
                patch.object(models, "BNNRegressor", _RecordingModel):
            fitted, *_ = models.train_bnn_models(object(), 10)
        self.assertEqual([model.fit_rows for model in fitted], [10, 10])

    def test_unified_runner_uses_every_offline_row(self):
        data = {
            "X_train": self.X_train,
            "y_train": self.y_train,
            "X_test": self.X_test,
        }
        qr_rows = []

        def fake_qr_fit(X, y, X_test, random_state=42):
            qr_rows.append(len(X))
            return None, object()

        with patch.object(models, "GPR_RBF", _RecordingModel), \
                patch.object(models, "GPR_Matern", _RecordingModel), \
                patch.object(models, "BNNRegressor", _RecordingModel), \
                patch.object(models, "autogluon_qr_fit_predict", fake_qr_fit):
            for method in (
                "GPR-RBF + NSGA-II",
                "GPR-Matern + NSGA-II",
                "QR + NSGA-II",
                "BNN + NSGA-II",
            ):
                fitted, _ = _predictor_models(METHOD_REGISTRY[method], data, 7)
                if method != "QR + NSGA-II":
                    self.assertEqual([model.fit_rows for model in fitted], [10, 10])
        self.assertEqual(qr_rows, [10, 10])


if __name__ == "__main__":
    unittest.main()
