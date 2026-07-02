import numpy as np
from pymoo.operators.sampling.lhs import LHS

from src.offline_moo_adapter import repair_offline_moo_decisions


def generate_data(problem, sample_size, sampling, train_seed, val_size=100, test_size=100, test_seed=1):
    if hasattr(problem, "load_offline_data"):
        return problem.load_offline_data(
            sample_size=sample_size,
            val_size=val_size,
            test_size=test_size,
            train_seed=train_seed,
            test_seed=test_seed,
        )

    # Training data
    X_train = sampling(problem, sample_size, seed=train_seed).get("X")
    X_train = repair_offline_moo_decisions(problem, X_train)
    y_train = problem.evaluate(X_train, return_values_of=["F"])
    
    # Validation data
    X_val = sampling(problem, val_size, seed=train_seed).get("X")
    X_val = repair_offline_moo_decisions(problem, X_val)
    y_val = problem.evaluate(X_val, return_values_of=["F"])
    
    # Testing data
    X_test = sampling(problem, test_size, seed=test_seed).get("X")
    X_test = repair_offline_moo_decisions(problem, X_test)
    y_test = problem.evaluate(X_test, return_values_of=["F"])
    
    return X_train, y_train, X_val, y_val, X_test, y_test
