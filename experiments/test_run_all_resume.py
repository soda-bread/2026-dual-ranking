"""Regression tests for the main runner's resume policy."""

from __future__ import annotations

import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from experiments import run_all
from sample_size_common import current_protocol_version


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


if __name__ == "__main__":
    unittest.main()
