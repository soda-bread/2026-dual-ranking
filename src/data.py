import numpy as np
from pymoo.operators.sampling.lhs import LHS

from src.offline_moo_adapter import repair_offline_moo_decisions


def generate_data(problem, sample_size, sampling, train_seed, test_size=100, test_seed=1):
    # Training data
    X_train = sampling(problem, sample_size, seed=train_seed).get("X")
    X_train = repair_offline_moo_decisions(problem, X_train)
    y_train = problem.evaluate(X_train, return_values_of=["F"])

    # Testing data
    X_test = sampling(problem, test_size, seed=test_seed).get("X")
    X_test = repair_offline_moo_decisions(problem, X_test)
    y_test = problem.evaluate(X_test, return_values_of=["F"])

    return X_train, y_train, X_test, y_test
