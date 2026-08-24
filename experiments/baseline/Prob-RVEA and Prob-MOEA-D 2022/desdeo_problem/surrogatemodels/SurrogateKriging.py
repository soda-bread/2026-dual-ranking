import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class SurrogateKriging(BaseRegressor):
    def __init__(
        self,
        normalize_inputs=False,
        normalize_y=False,
        alpha=0.0,
        n_restarts_optimizer=9,
        random_state=None,
    ):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.m = None
        self.normalize_inputs = bool(normalize_inputs)
        self.normalize_y = bool(normalize_y)
        self.alpha = float(alpha)
        self.n_restarts_optimizer = int(n_restarts_optimizer)
        self.random_state = random_state
        self.x_min = None
        self.x_scale = None

    def _transform_inputs(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not np.all(np.isfinite(X)):
            raise ModelError("SurrogateKriging received non-finite inputs.")
        if not self.normalize_inputs:
            return X
        return (X - self.x_min) / self.x_scale

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)

        # Make a 2-D array if needed
        X = np.asarray(X, dtype=float)
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.ndim != 2 or len(X) != len(y):
            raise ModelError("SurrogateKriging requires aligned two-dimensional X and y.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ModelError("SurrogateKriging training data must be finite.")

        self.X = X.copy()
        self.y = y.copy()
        if self.normalize_inputs:
            self.x_min = np.min(X, axis=0)
            x_max = np.max(X, axis=0)
            self.x_scale = x_max - self.x_min
            self.x_scale = np.where(self.x_scale > 0.0, self.x_scale, 1.0)
            X_model = self._transform_inputs(X)
            length_scale = np.ones(X.shape[1], dtype=float)
            length_bounds = (1e-3, 1e3)
        else:
            X_model = X
            length_scale = 10
            length_bounds = (1e-2, 1e2)

        kernel = C(1.0, (1e-3, 1e3)) * RBF(
            length_scale,
            length_bounds,
        )
        self.m = GaussianProcessRegressor(
            alpha=self.alpha,
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        )
        self.m.fit(X_model, y)

    def predict(self, X):
        #y_mean, y_stdev = np.asarray(self.m.predict(X, return_std=True)).reshape(1,-1)
        if self.m is None:
            raise ModelError("SurrogateKriging must be fitted before prediction.")
        y_mean, y_stdev = self.m.predict(
            self._transform_inputs(X),
            return_std=True,
        )
        y_mean = (y_mean.reshape(1,-1))
        y_stdev = (y_stdev.reshape(1,-1))
        return (y_mean, y_stdev)
