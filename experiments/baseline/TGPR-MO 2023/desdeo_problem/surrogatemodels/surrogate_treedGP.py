import numpy as np
import pandas as pd
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError
from sklearn import tree
import GPy
import graphviz
import datetime
class treeGP(BaseRegressor):
    def __init__(
        self,
        min_samples_leaf,
        robust_small_data=False,
        random_state=None,
    ):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.regr = None
        self.dict_gps = {}
        self.model_htgp = None
        self.error_leaves = None
        self.total_point = 0
        self.total_point_gps = 0
        self.min_samples_leaf = min_samples_leaf
        self.robust_small_data = bool(robust_small_data)
        self.random_state = random_state
        self.x_min = None
        self.x_scale = None
        self._X_model = None
        self.failed_leaves = set()
        self.gp_output_stats = {}
        self.last_gp_error = None

    def _transform_inputs(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not self.robust_small_data:
            return X
        if not np.all(np.isfinite(X)):
            raise ValueError("treeGP received non-finite decision vectors.")
        return (X - self.x_min) / self.x_scale

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        X = np.asarray(X, dtype=float)
        y = np.atleast_1d(np.asarray(y, dtype=float))
        if y.ndim == 1:
            y = y
        if X.ndim != 2 or len(X) != len(y):
            raise ModelError("treeGP requires aligned two-dimensional X and y.")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise ModelError("treeGP training data must be finite.")
        self.X = X.copy()
        self.y = y.copy()
        if self.robust_small_data:
            self.x_min = np.min(X, axis=0)
            self.x_scale = np.max(X, axis=0) - self.x_min
            self.x_scale = np.where(self.x_scale > 0.0, self.x_scale, 1.0)
        self._X_model = self._transform_inputs(X)
        self.regr = tree.DecisionTreeRegressor(
            max_depth=100,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.regr = self.regr.fit(self._X_model, y)
        n_nodes = self.regr.tree_.node_count
        children_left = self.regr.tree_.children_left
        children_right = self.regr.tree_.children_right
        feature = self.regr.tree_.feature
        threshold = self.regr.tree_.threshold
        rmse = self.regr.tree_.impurity
        rmse_threshold = 1
        samples_leaf_nodes = self.regr.apply(self._X_model)
        self.samples_leaf_nodes = samples_leaf_nodes
        error_leaves = None


    # add GPs to leaf nodes with max MSE
    def addGPs(self, X_solutions):
        X_solutions_model = self._transform_inputs(X_solutions)
        Y_solution_leaf = self.regr.apply(X_solutions_model)
        unique_solutions, count_solutions = np.unique(Y_solution_leaf, return_counts=True)
        unique_solutions = np.setdiff1d(unique_solutions,self.error_leaves)
        if self.robust_small_data and self.failed_leaves:
            unique_solutions = np.setdiff1d(
                unique_solutions,
                np.fromiter(self.failed_leaves, dtype=int),
            )
        self.total_point_gps = unique_solutions.size

        if unique_solutions.size > 0:
            # Taking max MSE
            mse_solutions = self.regr.tree_.impurity[unique_solutions]
            arg_max_mse = np.argmax(mse_solutions)
            selected_leaf = int(unique_solutions[arg_max_mse])
            loc_leaf = np.where(self.samples_leaf_nodes == selected_leaf)[0]
            X_leaf = self._X_model[loc_leaf]
            Y_leaf = self.y[loc_leaf]
            kernel = GPy.kern.Matern52(np.shape(X_leaf)[1],ARD=True)
            if self.robust_small_data:
                y_center = float(np.mean(Y_leaf))
                y_scale = float(np.std(Y_leaf))
                if not np.isfinite(y_scale) or y_scale <= np.finfo(float).eps:
                    y_scale = 1.0
                Y_model = (Y_leaf - y_center) / y_scale
                try:
                    m = GPy.models.GPRegression(
                        X_leaf,
                        Y_model.reshape(-1, 1),
                        kernel=kernel,
                    )
                    m.Gaussian_noise.variance = 1e-6
                    m.Gaussian_noise.variance.constrain_bounded(
                        1e-10,
                        1e-2,
                        warning=False,
                    )
                    m.optimize('bfgs', messages=False, max_iters=200)
                except Exception as error:
                    # A failed leaf GP must not be marked as successfully
                    # modelled: prediction continues with the fitted tree.
                    self.failed_leaves.add(selected_leaf)
                    self.last_gp_error = f"{type(error).__name__}: {error}"
                    return False
                key = str(selected_leaf)
                self.dict_gps[key] = m
                self.gp_output_stats[key] = (y_center, y_scale)
            else:
                m = GPy.models.GPRegression(
                    X_leaf,
                    Y_leaf.reshape(-1, 1),
                    kernel=kernel,
                )
                m.optimize('bfgs')
                self.dict_gps[str(selected_leaf)] = m

            if self.error_leaves is None:
                self.error_leaves = [selected_leaf]
            else:
                self.error_leaves = np.append(self.error_leaves, selected_leaf)
            self.total_point += np.shape(X_leaf)[0]
            return True
        return False

    def predict(self, X):
        X_model = self._transform_inputs(X)
        Y_predict = np.asarray(self.regr.predict(X=X_model), dtype=float)
        Y_test_leaf = self.regr.apply(X_model)
        unique_solutions, count_solutions = np.unique(Y_test_leaf, return_counts=True)
        Y_predict_mod = Y_predict
        count=0
        if self.error_leaves is not None:            
            for i in range(np.shape(X)[0]):
                key = str(Y_test_leaf[i])
                if Y_test_leaf[i] in self.error_leaves and key in self.dict_gps:
                    gp_mean = self.dict_gps[key].predict(
                        X_model[i].reshape(1, -1)
                    )[0][0]
                    if self.robust_small_data:
                        y_center, y_scale = self.gp_output_stats[key]
                        gp_mean = y_center + y_scale * gp_mean
                    if np.isfinite(gp_mean):
                        Y_predict_mod[i] = gp_mean
                    count += 1  
        y_mean = Y_predict_mod
        y_stdev = None
        return (y_mean, y_stdev)

