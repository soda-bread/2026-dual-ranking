from __future__ import annotations

import tempfile
import unittest
import warnings
from pathlib import Path

import pandas as pd

from experiments.sample_size_summary import summarize
from experiments.sample_size_common import current_protocol_version
from experiments.DL_MOBO_baseline import PROTOCOL_VERSION as DL_PROTOCOL_VERSION
from experiments.generative_baseline import (
    PROTOCOL_VERSION as GENERATIVE_PROTOCOL_VERSION,
)


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
                        protocol_version=current_protocol_version(
                            "official_pool", "GPR-RBF + NSGA-II"
                        ),
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
                        protocol_version=DL_PROTOCOL_VERSION,
                        configured_n_gen=100,
                        configured_pop_size=100,
                    )
                ]
            ).to_csv(baseline / "dl_baselines.csv", index=False)
            pd.DataFrame(
                [
                    _row(
                        "ParetoFlow",
                        protocol_version=GENERATIVE_PROTOCOL_VERSION,
                        configured_output_size=100,
                    )
                ]
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

    def test_stale_protocol_is_excluded_even_when_it_is_the_last_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            (results / "csv").mkdir(parents=True)
            method = "GPR-RBF + NSGA-II"
            current = current_protocol_version("official_pool", method)
            rows = [
                _row(
                    method,
                    protocol_version=current,
                    lhs_seed=1,
                    configured_n_gen=100,
                    configured_pop_size=100,
                    HVreal=2.0,
                ),
                _row(
                    method,
                    protocol_version="old-mutation-protocol",
                    lhs_seed=1,
                    configured_n_gen=100,
                    configured_pop_size=100,
                    HVreal=99.0,
                ),
            ]
            pd.DataFrame(rows).to_csv(
                results / "csv" / "exp1_results.csv", index=False
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lhs, _, _, _ = summarize(
                    results, root / "summary", write_plots=False
                )
            self.assertEqual(float(lhs.iloc[0]["HVreal_opt_mean"]), 2.0)
            self.assertTrue(
                any("Excluded 1 rows" in str(item.message) for item in caught)
            )
            stale = pd.read_csv(
                root / "summary" / "csv" / "stale_protocol_rows.csv"
            )
            self.assertEqual(len(stale), 1)
            inventory = pd.read_csv(
                root / "summary" / "csv" / "protocol_inventory.csv"
            )
            self.assertEqual(int(inventory.iloc[0]["identity_count"]), 2)


if __name__ == "__main__":
    unittest.main()
