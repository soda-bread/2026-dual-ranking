from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from experiments.sample_size_summary import summarize


def _row(method, **overrides):
    row = {
        "dataset_source": "official_pool",
        "protocol_version": f"{method}-protocol",
        "problem": "zdt1",
        "method": method,
        "training_size": 100,
        "offline_seed": 1,
        "opt_seed": 1,
        "MSEpre": 1.0,
        "MSEsur_real": 1.0,
        "HVreal": 1.0,
        "IGDplus": 1.0,
        "status": "success",
    }
    row.update(overrides)
    return row


class CombinedSummaryTests(unittest.TestCase):
    def test_primary_dl_and_generative_columns_are_merged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            baseline = root / "baseline"
            (primary / "csv").mkdir(parents=True)
            baseline.mkdir()

            pd.DataFrame(
                [
                    _row(
                        "GPR-RBF + NSGA-II",
                        lhs_seed=1,
                        configured_n_gen=100,
                        configured_pop_size=100,
                    )
                ]
            ).to_csv(primary / "csv" / "exp1_results.csv", index=False)
            pd.DataFrame(
                [
                    _row(
                        "End2End-Vallina",
                        configured_n_gen=100,
                        configured_pop_size=100,
                    )
                ]
            ).to_csv(baseline / "dl_baselines.csv", index=False)
            pd.DataFrame(
                [_row("ParetoFlow", configured_output_size=100)]
            ).to_csv(baseline / "generative_baselines.csv", index=False)

            lhs, problem, ranks, failed = summarize(
                [primary, baseline], root / "summary", write_plots=False
            )

            expected = {
                "GPR-RBF + NSGA-II",
                "End2End-Vallina",
                "ParetoFlow",
            }
            self.assertEqual(set(lhs["method"]), expected)
            self.assertEqual(set(problem["method"]), expected)
            self.assertEqual(set(ranks["method"]), expected)
            self.assertEqual(set(lhs["configured_pop_size"]), {100})
            self.assertEqual(set(lhs["configured_n_gen"]), {100})
            self.assertEqual(set(lhs["lhs_seed"]), {1})
            self.assertTrue(failed.empty)


if __name__ == "__main__":
    unittest.main()
