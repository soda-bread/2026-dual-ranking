"""Regression tests for the shared pymoo polynomial-mutation semantics."""

from __future__ import annotations

import unittest

import numpy as np
from pymoo.core.problem import Problem

from src.evolution import polynomial_mutation
from src.experiment import build_optimization_algorithm
from experiments.sample_size_common import current_protocol_version


class ToyProblem(Problem):
    def __init__(self, n_var=30):
        super().__init__(
            n_var=n_var,
            n_obj=2,
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )


class MutationProtocolTests(unittest.TestCase):
    def test_operator_uses_individual_probability_one_and_variable_rate(self):
        mutation = polynomial_mutation(30)
        self.assertAlmostEqual(mutation.prob.value, 1.0)
        self.assertAlmostEqual(mutation.prob_var.value, 1.0 / 30.0)
        self.assertAlmostEqual(mutation.eta.value, 20.0)

    def test_all_shared_pymoo_optimizers_receive_correct_mutation(self):
        problem = ToyProblem()
        for optimizer_name in ("NSGA-II", "MOEAD", "SMS-EMOA"):
            with self.subTest(optimizer=optimizer_name):
                algorithm = build_optimization_algorithm(
                    optimizer_name, problem, pop_size=20
                )
                mutation = algorithm.mating.mutation
                self.assertAlmostEqual(mutation.prob.value, 1.0)
                self.assertAlmostEqual(mutation.prob_var.value, 1.0 / 30.0)

    def test_invalid_dimension_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "n_var must be positive"):
            polynomial_mutation(0)

    def test_only_affected_methods_receive_new_resume_protocol(self):
        primary = current_protocol_version(
            "official_pool", "GPR-RBF + NSGA-II"
        )
        ddmoea = current_protocol_version("official_pool", "DDMOEA-GAN")
        unaffected = current_protocol_version("official_pool", "TGPR-MO")
        self.assertIn("pm_individual1_variable1overd_v1", primary)
        self.assertIn("pm_individual1_variable1overd_v1", ddmoea)
        self.assertIn("+cfg-", ddmoea)
        self.assertNotIn("pm_individual1_variable1overd_v1", unaffected)


if __name__ == "__main__":
    unittest.main()
