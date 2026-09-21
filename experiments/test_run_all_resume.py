"""Regression tests for the main runner's resume policy."""

from __future__ import annotations

import unittest
import sys
import csv
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiments import run_all
from sample_size_common import (
    RESULT_FIELDS,
    current_protocol_version,
    reconcile_result_csvs,
)


class MainRunnerResumeTests(unittest.TestCase):
    @staticmethod
    def _args():
        return SimpleNamespace(
            output_dir=Path("unused"),
            resume=True,
            retry_failed=False,
            dataset_source="official_pool",
            n_gen=100,
            pop_size=100,
            problems=["zdt1"],
            train_sizes=[50],
            lhs_seeds=[1],
            methods=["TGPR-MO"],
            opt_seeds=[1],
        )

    @staticmethod
    def _row(status):
        return {
            "dataset_source": "official_pool",
            "protocol_version": current_protocol_version(
                "official_pool", "TGPR-MO"
            ),
            "configured_n_gen": 100,
            "configured_pop_size": 100,
            "problem": "zdt1",
            "method": "TGPR-MO",
            "training_size": 50,
            "lhs_seed": 1,
            "opt_seed": 1,
            "MSEpre": 1.0,
            "MSEsur_real": 1.0,
            "HVreal": 1.0,
            "IGDplus": 1.0,
            "status": status,
        }

    def test_resume_retries_failed_row_without_retry_flag(self):
        with patch.object(run_all, "read_result_rows", return_value=[self._row("failed")]):
            groups, skipped = run_all.build_plan(self._args())

        self.assertEqual(skipped, 0)
        self.assertEqual(groups, [("zdt1", 50, 1, "TGPR-MO", (1,))])

    def test_resume_still_skips_successful_row(self):
        with patch.object(run_all, "read_result_rows", return_value=[self._row("success")]):
            groups, skipped = run_all.build_plan(self._args())

        self.assertEqual(skipped, 1)
        self.assertEqual(groups, [])

    def test_reconcile_keeps_only_current_protocol_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            csv_dir = output_dir / "csv"
            csv_dir.mkdir()
            current = self._row("success")
            current["method"] = "GPR-RBF + NSGA-II"
            current["protocol_version"] = current_protocol_version(
                "official_pool", current["method"]
            )
            old = dict(current, protocol_version="old-rank-and-crowd-protocol")
            path = csv_dir / "exp1_results.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
                writer.writeheader()
                writer.writerow({name: old.get(name, "") for name in RESULT_FIELDS})
                writer.writerow(
                    {name: current.get(name, "") for name in RESULT_FIELDS}
                )

            reconcile_result_csvs(output_dir)

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["protocol_version"], current["protocol_version"]
            )


if __name__ == "__main__":
    unittest.main()
