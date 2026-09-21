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
import sample_size_common
from sample_size_common import (
    RESULT_FIELDS,
    append_rows,
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

            split_path = csv_dir / "results_gpr_rbf_normal.csv"
            with split_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["protocol_version"], current["protocol_version"]
            )

    def test_primary_categories_use_surrogate_category_csvs(self):
        methods = (
            "GPR-RBF + NSGA-II",
            "GPR-RBF + NSGA-II + DR",
            "GPR-RBF + NSGA-II + EBU-DR",
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            rows = []
            for method in methods:
                row = self._row("success")
                row["method"] = method
                row["protocol_version"] = current_protocol_version(
                    "official_pool", method
                )
                rows.append(row)
            append_rows(output_dir, rows)

            csv_dir = output_dir / "csv"
            self.assertEqual(
                {path.name for path in csv_dir.glob("*_results.csv")},
                set(),
            )
            expected = {
                "results_gpr_rbf_normal.csv",
                "results_gpr_rbf_dr.csv",
                "results_gpr_rbf_ebu_dr.csv",
            }
            self.assertEqual(
                {path.name for path in csv_dir.glob("results_*.csv")},
                expected,
            )
            for filename in expected:
                with (csv_dir / filename).open(
                    newline="", encoding="utf-8"
                ) as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 1)


class SharedSurrogateSchedulingTests(unittest.TestCase):
    METHODS = (
        "GPR-RBF + NSGA-II",
        "GPR-RBF + NSGA-II + DR",
        "GPR-RBF + NSGA-II + EBU-DR",
    )

    @staticmethod
    def _payload(method, seeds):
        return (
            "results",
            "zdt1",
            100,
            6,
            method,
            tuple(seeds),
            100,
            100,
            "official_pool",
            "subsets",
            (50, 100, 200, 400, 1000),
            {},
        )

    def test_three_categories_become_one_family_worker_group(self):
        args = SimpleNamespace(
            methods=list(reversed(self.METHODS)),
            max_workers=72,
            tabpfn_max_workers=5,
        )
        payloads = [
            self._payload(method, range(1, 11)) for method in self.METHODS
        ]
        stages = run_all.build_execution_stages(args, payloads)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["kind"], "family")
        self.assertEqual(stages[0]["name"], "gpr_rbf")
        self.assertEqual(len(stages[0]["payloads"]), 1)
        method_opt_seeds = stages[0]["payloads"][0][5]
        self.assertEqual(
            tuple(method for method, _ in method_opt_seeds),
            self.METHODS,
        )
        self.assertTrue(all(seeds == tuple(range(1, 11)) for _, seeds in method_opt_seeds))

    def test_family_group_prepares_surrogate_once(self):
        prepared = {"pair": (object(), object())}

        def method_result(
            prepared_arg,
            output_dir,
            problem,
            size,
            offline_seed,
            spec,
            seeds,
            n_gen,
            pop_size,
            dataset_source="lhs",
        ):
            del (
                prepared_arg,
                output_dir,
                problem,
                size,
                offline_seed,
                seeds,
                n_gen,
                pop_size,
                dataset_source,
            )
            return [{"method": spec.name}], prepared["pair"]

        method_opt_seeds = tuple(
            (method, tuple(range(1, 11))) for method in self.METHODS
        )
        with patch.object(
            sample_size_common,
            "_prepare_predictor_group",
            return_value=prepared,
        ) as prepare, patch.object(
            sample_size_common,
            "_run_prepared_predictor_method",
            side_effect=method_result,
        ) as run_method:
            rows, pair = sample_size_common._run_predictor_family_group(
                Path("unused"),
                "zdt1",
                100,
                6,
                method_opt_seeds,
                100,
                100,
                dataset_source="official_pool",
            )

        prepare.assert_called_once()
        self.assertEqual(run_method.call_count, 3)
        self.assertEqual([row["method"] for row in rows], list(self.METHODS))
        self.assertIs(pair, prepared["pair"])


if __name__ == "__main__":
    unittest.main()
