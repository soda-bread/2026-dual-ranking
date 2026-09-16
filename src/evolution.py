"""Shared evolutionary operators for the comparison protocol."""

from __future__ import annotations

from pymoo.operators.mutation.pm import PM


def polynomial_mutation(n_var: int, eta: float = 20.0) -> PM:
    """Return polynomial mutation with the classic per-variable rate 1 / D.

    In pymoo 0.6, ``prob`` controls whether an individual enters the mutation
    operator and ``prob_var`` controls each variable.  Setting only
    ``prob=1 / D`` would therefore apply pymoo's additional default per-variable
    probability after rejecting most individuals.
    """

    n_var = int(n_var)
    if n_var < 1:
        raise ValueError("n_var must be positive.")
    return PM(prob=1.0, prob_var=1.0 / n_var, eta=float(eta))
