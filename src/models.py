import copy
import contextlib
import io
import logging
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


TABPFN_TOKEN = "tabpfn_sk_oYB3VV-fcuoIOMmO48LDhBSN4lbyzmfEMV91tGJwDmA"


def _set_tabpfn_token():
    os.environ["TABPFN_TOKEN"] = TABPFN_TOKEN
    return TABPFN_TOKEN


def _safe_std_from_variance(y_var):
    y_var = np.asarray(y_var, dtype=float)
    return np.sqrt(np.maximum(y_var, 0.0))


def _require_gpy():
    try:
        import GPy
    except ImportError as err:
        raise ImportError(
            "GPy is required for GPR_RBF/GPR_Matern. "
            "Install it only for GPR notebooks, then restart the runtime before continuing."
        ) from err
    return GPy


# Model: GPR_RBF
class GPR_RBF:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        GPy = _require_gpy()
        y = y.reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1],ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

    def fit_high_bias_variance(self, X, y, lengthscale_weight):
        GPy = _require_gpy()
        y = y.reshape(-1, 1)
        kernel = GPy.kern.RBF(input_dim=X.shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

        optimized_noise = float(self.model.Gaussian_noise.variance.values[0])
        kernel = GPy.kern.RBF(
            input_dim=X.shape[1],
            ARD=True,
            variance=kernel.variance,
            lengthscale=kernel.lengthscale * lengthscale_weight)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.Gaussian_noise.variance = optimized_noise

    def predict(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=True)
        y_std = _safe_std_from_variance(y_var)
        return y_mean.flatten(), y_std.flatten()

    def predict_noiseless(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=False)
        y_std = _safe_std_from_variance(y_var)
        return y_mean.flatten(), y_std.flatten()


class GPR_Matern:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        GPy = _require_gpy()
        y = y.reshape(-1, 1)
        kernel = GPy.kern.Matern52(input_dim=X.shape[1], ARD=True)
        self.model = GPy.models.GPRegression(X, y, kernel, normalizer=True)
        self.model.optimize(messages=False)

    def predict(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=True)
        y_std = _safe_std_from_variance(y_var)
        return y_mean.flatten(), y_std.flatten()

    def predict_noiseless(self, X):
        y_mean, y_var = self.model.predict(X, include_likelihood=False)
        y_std = _safe_std_from_variance(y_var)
        return y_mean.flatten(), y_std.flatten()

def gpr_pred_mean_std(model_f1, model_f2, X_test, noiseless=False, verbose=True):
    if noiseless:
        mean_f1, std_f1 = model_f1.predict_noiseless(X_test)
        mean_f2, std_f2 = model_f2.predict_noiseless(X_test)
    else:
        mean_f1, std_f1 = model_f1.predict(X_test)
        mean_f2, std_f2 = model_f2.predict(X_test)

    mean_f1 = np.asarray(mean_f1).reshape(-1)
    std_f1 = np.asarray(std_f1).reshape(-1)
    mean_f2 = np.asarray(mean_f2).reshape(-1)
    std_f2 = np.asarray(std_f2).reshape(-1)

    pred_mean = np.stack([mean_f1, mean_f2], axis=1)
    pred_std = np.stack([std_f1, std_f2], axis=1)

    if verbose:
        tag = "noiseless" if noiseless else "with_noise"

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2


def autogluon_qr_fit_predict(X_train, y_train, X_test, quantile_levels=None, random_state=42):
    from autogluon.tabular import TabularPredictor

    if quantile_levels is None:
        quantile_levels = [0.5, 0.8, 0.9, 0.95]

    logging.getLogger("autogluon").setLevel(logging.ERROR)
    train_df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    train_df["target"] = y_train

    model = TabularPredictor(
        label="target",
        problem_type="quantile",
        quantile_levels=quantile_levels,
        path=tempfile.mkdtemp(prefix="autogluon_qr_"),
        verbosity=0,
    ).fit(
        train_data=train_df,
        verbosity=0,
        excluded_model_types=["CAT"],
        ag_args_fit={"random_state": random_state},
    )

    quantile_pred = autogluon_qr_predict(model, X_test)
    return quantile_pred, model


def autogluon_qr_predict(model, X):
    test_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    pred = model.predict(test_df)
    pred.columns = [f"y_q{q}" for q in pred.columns]
    return pred


def autogluon_qr_pred_mean_quantiles(model_f1, model_f2, X_test, verbose=True):
    pred_y1 = autogluon_qr_predict(model_f1, X_test)
    pred_y2 = autogluon_qr_predict(model_f2, X_test)

    mean_q = np.stack([pred_y1["y_q0.5"].values, pred_y2["y_q0.5"].values], axis=1)
    q80 = np.stack([pred_y1["y_q0.8"].values, pred_y2["y_q0.8"].values], axis=1)
    q90 = np.stack([pred_y1["y_q0.9"].values, pred_y2["y_q0.9"].values], axis=1)
    q95 = np.stack([pred_y1["y_q0.95"].values, pred_y2["y_q0.95"].values], axis=1)

    if verbose:
        print("[QR] y_q50\n", mean_q[:5])
        print("[QR] y_q80\n", q80[:5])
        print("[QR] y_q90\n", q90[:5])
        print("[QR] y_q95\n", q95[:5])

    return mean_q, q80, q90, q95


def autogluon_fit_predict(
    X_train,
    y_train,
    X_test,
    hyperparameters=None,
    fit_kwargs=None,
    random_state=42,
):
    from autogluon.tabular import TabularPredictor

    train_df = pd.DataFrame(X_train, columns=[f"x{i}" for i in range(X_train.shape[1])])
    train_df["target"] = y_train
    fit_kwargs = {} if fit_kwargs is None else dict(fit_kwargs)
    fit_kwargs.setdefault("excluded_model_types", ["CAT"])

    model = TabularPredictor(
        label="target",
        problem_type="regression",
    ).fit(
        train_data=train_df,
        hyperparameters=hyperparameters,
        verbosity=0,
        ag_args_fit={"random_state": random_state},
        **fit_kwargs,
    )

    pred = autogluon_predict(model, X_test)
    return pred, model


def autogluon_predict(model, X):
    test_df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    return np.asarray(model.predict(test_df), dtype=float).reshape(-1)


def autogluon_pred_mean(model_f1, model_f2, X_test):
    mean_f1 = autogluon_predict(model_f1, X_test)
    mean_f2 = autogluon_predict(model_f2, X_test)
    return np.stack([mean_f1, mean_f2], axis=1)


def _require_tabpfn():
    _set_tabpfn_token()
    try:
        from tabpfn_client import TabPFNRegressor, set_access_token
    except ImportError as err:
        raise ImportError(
            "tabpfn-client is required for TabPFN-3 surrogate models. "
            "Install it with `pip install tabpfn-client`, then restart the runtime before continuing."
        ) from err
    set_access_token(TABPFN_TOKEN)
    return TabPFNRegressor


def tabpfn_fit_predict(X_train, y_train, X_test):
    _set_tabpfn_token()
    TabPFNRegressor = _require_tabpfn()
    model = TabPFNRegressor()
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).reshape(-1)
    _set_tabpfn_token()
    model.fit(X_train, y_train)
    pred = tabpfn_predict(model, X_test)
    return pred, model


def tabpfn_predict(model, X):
    X = np.asarray(X, dtype=float)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return np.asarray(model.predict(X), dtype=float).reshape(-1)


def tabpfn_pred_mean(model_f1, model_f2, X_test):
    mean_f1 = tabpfn_predict(model_f1, X_test)
    mean_f2 = tabpfn_predict(model_f2, X_test)
    return np.stack([mean_f1, mean_f2], axis=1)


def _require_pyro():
    try:
        import torch
        import torch.nn as nn
        from torch.distributions import constraints
        import pyro
        import pyro.distributions as dist
        from pyro.infer import Predictive, SVI, Trace_ELBO
        from pyro.infer.autoguide import AutoDiagonalNormal
        from pyro.nn import PyroModule, PyroSample
        from pyro.optim import Adam
    except ImportError as err:
        raise ImportError(
            "PyTorch and pyro-ppl are required for BNNRegressor. "
            "Install pyro-ppl, then restart the runtime before continuing."
        ) from err
    return {
        "torch": torch,
        "nn": nn,
        "constraints": constraints,
        "pyro": pyro,
        "dist": dist,
        "Predictive": Predictive,
        "SVI": SVI,
        "Trace_ELBO": Trace_ELBO,
        "AutoDiagonalNormal": AutoDiagonalNormal,
        "Adam": Adam,
        "PyroModule": PyroModule,
        "PyroSample": PyroSample,
    }


def _build_bayesian_network(
    modules,
    in_dim,
    hidden,
    prior_scale,
    fixed_sigma,
):
    torch = modules["torch"]
    nn = modules["nn"]
    dist = modules["dist"]
    PyroModule = modules["PyroModule"]
    PyroSample = modules["PyroSample"]
    constraints = modules["constraints"]
    if isinstance(hidden, (tuple, list)):
        if len(hidden) != 2:
            raise ValueError("BNN hidden must be an int or a two-element sequence.")
        hidden_1, hidden_2 = (int(hidden[0]), int(hidden[1]))
    else:
        hidden_1 = hidden_2 = int(hidden)
    if hidden_1 <= 0 or hidden_2 <= 0:
        raise ValueError("BNN hidden layer sizes must be positive.")
    if fixed_sigma <= 0:
        raise ValueError("BNN fixed_sigma must be positive because it initializes obs_sigma.")

    class BayesianNeuralNetwork(PyroModule):
        def __init__(self):
            super().__init__()
            self.fc1 = PyroModule[nn.Linear](in_dim, hidden_1)
            self.fc1.weight = PyroSample(
                dist.Normal(0.0, prior_scale).expand([hidden_1, in_dim]).to_event(2)
            )
            self.fc1.bias = PyroSample(
                dist.Normal(0.0, prior_scale).expand([hidden_1]).to_event(1)
            )

            self.fc2 = PyroModule[nn.Linear](hidden_1, hidden_2)
            self.fc2.weight = PyroSample(
                dist.Normal(0.0, prior_scale).expand([hidden_2, hidden_1]).to_event(2)
            )
            self.fc2.bias = PyroSample(
                dist.Normal(0.0, prior_scale).expand([hidden_2]).to_event(1)
            )

            self.out = PyroModule[nn.Linear](hidden_2, 1)
            self.out.weight = PyroSample(
                dist.Normal(0.0, prior_scale).expand([1, hidden_2]).to_event(2)
            )
            self.out.bias = PyroSample(
                dist.Normal(0.0, prior_scale).expand([1]).to_event(1)
            )

        def forward(self, x, y=None):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            mean = self.out(x).squeeze(-1)
            obs_sigma = modules["pyro"].param(
                "obs_sigma",
                torch.tensor(float(fixed_sigma), dtype=mean.dtype, device=mean.device),
                constraint=constraints.positive,
            )
            modules["pyro"].deterministic("mean", mean)
            with modules["pyro"].plate("data", x.shape[0]):
                modules["pyro"].sample(
                    "obs",
                    dist.Normal(mean, obs_sigma),
                    obs=y,
                )
            return mean

    return BayesianNeuralNetwork()


class BNNRegressor:
    """SVI-trained Bayesian neural-network regressor with posterior quantiles."""

    def __init__(
        self,
        hidden=16,
        lr=1e-3,
        max_steps=50000,
        patience=10,
        validation_interval=200,
        validation_samples=50,
        prediction_samples=100,
        prior_scale=0.5,
        fixed_sigma=1e-3,
        random_state=42,
        device=None,
        verbose=False,
    ):
        if isinstance(hidden, (tuple, list)):
            if len(hidden) != 2:
                raise ValueError("BNN hidden must be an int or a two-element sequence.")
            self.hidden = (int(hidden[0]), int(hidden[1]))
        else:
            self.hidden = int(hidden)
        self.lr = float(lr)
        self.max_steps = int(max_steps)
        self.patience = int(patience)
        self.validation_interval = int(validation_interval)
        self.validation_samples = int(validation_samples)
        self.prediction_samples = int(prediction_samples)
        self.prior_scale = float(prior_scale)
        self.fixed_sigma = float(fixed_sigma)
        self.random_state = int(random_state)
        self.device_name = device
        self.verbose = bool(verbose)
        self.model = None
        self.guide = None
        self.param_store_state = None
        self.target_scaler = StandardScaler()

    def __deepcopy__(self, memo):
        # pymoo deep-copies optimization problems. The fitted Pyro network and
        # variational tensors are read-only during prediction, so sharing them
        # avoids copying local Pyro classes or large parameter stores.
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        result = object.__new__(type(self))
        memo[id(self)] = result
        for name, value in self.__dict__.items():
            if name in ("model", "guide", "param_store_state"):
                setattr(result, name, value)
            else:
                setattr(result, name, copy.deepcopy(value, memo))
        return result

    def _torch_rng_devices(self):
        if self.device.type != "cuda":
            return []
        return [self.device.index if self.device.index is not None else 0]

    def fit(self, X, y, X_val=None, y_val=None):
        modules = _require_pyro()
        torch = modules["torch"]
        pyro = modules["pyro"]
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.shape[0] != X.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if X.shape[0] < 2:
            raise ValueError("BNNRegressor needs at least two training samples.")
        if X_val is None or y_val is None:
            raise ValueError("BNNRegressor.fit requires X_val and y_val for early stopping.")
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)
        if X_val.ndim != 2:
            raise ValueError("X_val must be a 2D array.")
        if X_val.shape[1] != X.shape[1]:
            raise ValueError("X and X_val must have the same number of columns.")
        if y_val.shape[0] != X_val.shape[0]:
            raise ValueError("X_val and y_val must have the same number of rows.")

        self.target_scaler.fit(y.reshape(-1, 1))
        y_scaled = self.target_scaler.transform(y.reshape(-1, 1)).reshape(-1)
        y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).reshape(-1)

        self.device = torch.device(
            self.device_name
            if self.device_name is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32, device=self.device)
        param_store = pyro.get_param_store()

        with torch.random.fork_rng(devices=self._torch_rng_devices()):
            torch.manual_seed(self.random_state)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.random_state)
            with param_store.scope() as param_store_state:
                self.model = _build_bayesian_network(
                    modules,
                    in_dim=X.shape[1],
                    hidden=self.hidden,
                    prior_scale=self.prior_scale,
                    fixed_sigma=self.fixed_sigma,
                ).to(self.device)
                self.guide = modules["AutoDiagonalNormal"](self.model)
                optimizer = modules["Adam"]({"lr": self.lr})
                svi = modules["SVI"](
                    self.model,
                    self.guide,
                    optimizer,
                    loss=modules["Trace_ELBO"](),
                )

                best_val_mse = float("inf")
                best_state = None
                no_improve = 0
                for step in range(1, self.max_steps + 1):
                    loss = svi.step(X_tensor, y_tensor)
                    if step % self.validation_interval != 0:
                        continue

                    predictive = modules["Predictive"](
                        self.model,
                        guide=self.guide,
                        num_samples=self.validation_samples,
                        return_sites=("mean",),
                    )
                    with torch.no_grad():
                        val_mean = predictive(X_val_tensor)["mean"].mean(dim=0)
                        val_mse = ((val_mean - y_val_tensor) ** 2).mean().item()

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_state = copy.deepcopy(param_store.get_state())
                        no_improve = 0
                    else:
                        no_improve += 1

                    if self.verbose:
                        print(
                            f"[{step}] train ELBO={loss:.2e}, "
                            f"val MSE={val_mse:.2e}, "
                            f"no_improve={no_improve}/{self.patience}"
                        )
                    if no_improve >= self.patience:
                        if self.verbose:
                            print(
                                f"Early stopping at step {step}, "
                                f"best val={best_val_mse:.2e}"
                            )
                        break

                if best_state is not None:
                    param_store.clear()
                    param_store.set_state(best_state)

            self.param_store_state = param_store_state
            self.best_val_mse = best_val_mse
            self.training_steps = step
        return self

    def _predictive_samples(self, X, num_samples=None):
        if self.model is None or self.guide is None or self.param_store_state is None:
            raise ValueError("Model is not fitted yet.")
        modules = _require_pyro()
        torch = modules["torch"]
        pyro = modules["pyro"]
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        sample_count = self.prediction_samples if num_samples is None else int(num_samples)

        with torch.random.fork_rng(devices=self._torch_rng_devices()):
            torch.manual_seed(self.random_state)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.random_state)
            with pyro.get_param_store().scope(self.param_store_state):
                predictive = modules["Predictive"](
                    self.model,
                    guide=self.guide,
                    num_samples=sample_count,
                    return_sites=("mean", "obs"),
                )
                with torch.no_grad():
                    samples = predictive(X_tensor)
                    mean_samples = samples["mean"].detach().cpu().numpy()
                    obs_samples = samples["obs"].detach().cpu().numpy()

        mean_samples = self.target_scaler.inverse_transform(
            mean_samples.reshape(-1, 1)
        ).reshape(mean_samples.shape)
        obs_samples = self.target_scaler.inverse_transform(
            obs_samples.reshape(-1, 1)
        ).reshape(obs_samples.shape)
        return mean_samples, obs_samples

    def predict_distribution(self, X, num_samples=None):
        mean_samples, obs_samples = self._predictive_samples(X, num_samples=num_samples)
        predictions = (
            mean_samples.mean(axis=0),
            obs_samples.std(axis=0),
            np.percentile(obs_samples, 80, axis=0),
            np.percentile(obs_samples, 90, axis=0),
            np.percentile(obs_samples, 95, axis=0),
        )
        return tuple(np.asarray(values).reshape(-1) for values in predictions)

    def predict(self, X, num_samples=None):
        mean, std, _, _, _ = self.predict_distribution(X, num_samples=num_samples)
        return mean, std

    def predict_quantiles(self, X, num_samples=None):
        mean, _, q80, q90, q95 = self.predict_distribution(X, num_samples=num_samples)
        return mean, q80, q90, q95


def bnn_pred_mean_std(model_f1, model_f2, X_test, verbose=True):
    mean_f1, std_f1 = model_f1.predict(X_test)
    mean_f2, std_f2 = model_f2.predict(X_test)

    mean_f1 = np.asarray(mean_f1).reshape(-1)
    std_f1 = np.asarray(std_f1).reshape(-1)
    mean_f2 = np.asarray(mean_f2).reshape(-1)
    std_f2 = np.asarray(std_f2).reshape(-1)

    pred_mean = np.stack([mean_f1, mean_f2], axis=1)
    pred_std = np.stack([std_f1, std_f2], axis=1)

    if verbose:
        print("[BNN] pred_mean\n", pred_mean[:5])
        print("[BNN] pred_std\n", pred_std[:5])
        print("[BNN] Max pred_std\n", np.max(pred_std, axis=0))

    return pred_mean, pred_std, mean_f1, std_f1, mean_f2, std_f2


def bnn_pred_mean_quantiles(model_f1, model_f2, X_test, verbose=True):
    mean_f1, q80_f1, q90_f1, q95_f1 = model_f1.predict_quantiles(X_test)
    mean_f2, q80_f2, q90_f2, q95_f2 = model_f2.predict_quantiles(X_test)

    mean_f1 = np.asarray(mean_f1).reshape(-1)
    q80_f1 = np.asarray(q80_f1).reshape(-1)
    q90_f1 = np.asarray(q90_f1).reshape(-1)
    q95_f1 = np.asarray(q95_f1).reshape(-1)
    mean_f2 = np.asarray(mean_f2).reshape(-1)
    q80_f2 = np.asarray(q80_f2).reshape(-1)
    q90_f2 = np.asarray(q90_f2).reshape(-1)
    q95_f2 = np.asarray(q95_f2).reshape(-1)

    mean_q = np.stack([mean_f1, mean_f2], axis=1)
    q80 = np.stack([q80_f1, q80_f2], axis=1)
    q90 = np.stack([q90_f1, q90_f2], axis=1)
    q95 = np.stack([q95_f1, q95_f2], axis=1)

    if verbose:
        print("[BNN] mean\n", mean_q[:5])
        print("[BNN] q80\n", q80[:5])
        print("[BNN] q90\n", q90[:5])
        print("[BNN] q95\n", q95[:5])

    return mean_q, q80, q90, q95


def _generate_calibration_data(problem, sample_size, train_seed, test_seed):
    from pymoo.operators.sampling.lhs import LHS
    from src.data import generate_data

    return generate_data(
        problem=problem,
        sample_size=sample_size,
        sampling=LHS(),
        train_seed=train_seed,
        val_size=100,
        test_size=100,
        test_seed=test_seed,
    )


def train_gpr_rbf_for_calibration(problem, sample_size, train_seed=42, test_seed=1):
    X_train, y_train, X_val, y_val, X_test, y_test = _generate_calibration_data(
        problem,
        sample_size,
        train_seed,
        test_seed,
    )
    model_f1 = GPR_RBF()
    model_f2 = GPR_RBF()
    model_f1.fit(X_train, y_train[:, 0])
    model_f2.fit(X_train, y_train[:, 1])
    return model_f1, model_f2, X_train, y_train, X_val, y_val, X_test, y_test


def train_gpr_matern_for_calibration(problem, sample_size, train_seed=42, test_seed=1):
    X_train, y_train, X_val, y_val, X_test, y_test = _generate_calibration_data(
        problem,
        sample_size,
        train_seed,
        test_seed,
    )
    model_f1 = GPR_Matern()
    model_f2 = GPR_Matern()
    model_f1.fit(X_train, y_train[:, 0])
    model_f2.fit(X_train, y_train[:, 1])
    return model_f1, model_f2, X_train, y_train, X_val, y_val, X_test, y_test


def train_autogluon_qr_for_calibration(problem, sample_size, train_seed=42, test_seed=1):
    X_train, y_train, X_val, y_val, X_test, y_test = _generate_calibration_data(
        problem,
        sample_size,
        train_seed,
        test_seed,
    )
    _, model_f1 = autogluon_qr_fit_predict(
        X_train,
        y_train[:, 0],
        X_test,
        random_state=train_seed,
    )
    _, model_f2 = autogluon_qr_fit_predict(
        X_train,
        y_train[:, 1],
        X_test,
        random_state=train_seed,
    )
    return model_f1, model_f2, X_train, y_train, X_test, y_test


def train_autogluon_for_calibration(
    problem,
    sample_size,
    train_seed=42,
    test_seed=1,
    hyperparameters=None,
    fit_kwargs=None,
):
    X_train, y_train, X_val, y_val, X_test, y_test = _generate_calibration_data(
        problem,
        sample_size,
        train_seed,
        test_seed,
    )
    _, model_f1 = autogluon_fit_predict(
        X_train,
        y_train[:, 0],
        X_test,
        hyperparameters=hyperparameters,
        fit_kwargs=fit_kwargs,
        random_state=train_seed,
    )
    _, model_f2 = autogluon_fit_predict(
        X_train,
        y_train[:, 1],
        X_test,
        hyperparameters=hyperparameters,
        fit_kwargs=fit_kwargs,
        random_state=train_seed,
    )
    return model_f1, model_f2, X_train, y_train, X_test, y_test


def train_tabpfn_for_calibration(problem, sample_size, train_seed=42, test_seed=1):
    X_train, y_train, X_val, y_val, X_test, y_test = _generate_calibration_data(
        problem,
        sample_size,
        train_seed,
        test_seed,
    )
    _, model_f1 = tabpfn_fit_predict(
        X_train,
        y_train[:, 0],
        X_test,
    )
    _, model_f2 = tabpfn_fit_predict(
        X_train,
        y_train[:, 1],
        X_test,
    )
    return model_f1, model_f2, X_train, y_train, X_test, y_test


def train_bnn_for_calibration(
    problem,
    sample_size,
    train_seed=42,
    test_seed=1,
    hidden=16,
    lr=1e-3,
    max_steps=50000,
    patience=10,
    fixed_sigma=1e-3,
):
    X_train, y_train, X_val, y_val, X_test, y_test = _generate_calibration_data(
        problem,
        sample_size,
        train_seed,
        test_seed,
    )
    model_f1 = BNNRegressor(
        hidden=hidden,
        lr=lr,
        max_steps=max_steps,
        patience=patience,
        fixed_sigma=fixed_sigma,
        random_state=train_seed,
    )
    model_f2 = BNNRegressor(
        hidden=hidden,
        lr=lr,
        max_steps=max_steps,
        patience=patience,
        fixed_sigma=fixed_sigma,
        random_state=train_seed,
    )
    model_f1.fit(X_train, y_train[:, 0], X_val, y_val[:, 0])
    model_f2.fit(X_train, y_train[:, 1], X_val, y_val[:, 1])
    return model_f1, model_f2, X_train, y_train, X_test, y_test
