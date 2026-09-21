"""Tests for the paired DR analysis tables."""

import unittest

import pandas as pd

from experiments.analysis_dr import analyze


class DRAnalysisTests(unittest.TestCase):
    def test_paired_analysis_handles_three_controls(self):
        rows = []
        method_by_category = {
            "normal": "GPR-RBF + NSGA-II",
            "dr": "GPR-RBF + NSGA-II + DR",
            "ebu_dr": "GPR-RBF + NSGA-II + EBU-DR",
        }
        values = {
            "normal": (0.2, 0.3, 0.4),
            "dr": (0.3, 0.4, 0.5),
            "ebu_dr": (0.5, 0.6, 0.7),
        }
        for category, method in method_by_category.items():
            for opt_seed, value in enumerate(values[category], start=1):
                rows.append({
                    "dataset_source": "official_pool",
                    "problem": "zdt1",
                    "training_size": 50,
                    "offline_seed": 1,
                    "opt_seed": opt_seed,
                    "configured_n_gen": 100,
                    "configured_pop_size": 100,
                    "family": "gpr_rbf",
                    "category": category,
                    "HVreal": value,
                })
        cell, aggregate, paired = analyze(
            pd.DataFrame(rows),
            "HVreal",
            ("normal", "dr", "ctrl_duplicate"),
        )
        self.assertEqual(len(cell), 3)
        self.assertEqual(len(aggregate), 3)
        self.assertEqual(len(paired), 9)
        duplicate = aggregate.loc[aggregate["control"] == "ctrl_duplicate"].iloc[0]
        self.assertEqual(float(duplicate["mean_improvement"]), 0.0)
        self.assertEqual(float(duplicate["one_sided_wilcoxon_p"]), 1.0)


if __name__ == "__main__":
    unittest.main()
