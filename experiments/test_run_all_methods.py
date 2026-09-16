from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_all_methods
from experiments.project_environment import PROJECT_PYTHON


class UnifiedDispatcherTest(unittest.TestCase):
    def test_registry_contains_all_21_unique_methods(self):
        self.assertEqual(len(run_all_methods.ALL_METHODS), 21)
        self.assertEqual(len(set(run_all_methods.ALL_METHODS)), 21)
        self.assertEqual(
            {key: len(value) for key, value in run_all_methods.METHOD_GROUPS.items()},
            {"main": 15, "dl_mobo": 4, "generative": 2},
        )

    def test_default_plan_excludes_three_temporarily_disabled_methods(self):
        args = run_all_methods.parse_args(["--dry-run"])
        self.assertEqual(len(args.methods), 18)
        self.assertTrue(
            set(args.methods).isdisjoint(run_all_methods.DEFAULT_DISABLED_METHODS)
        )

    def test_two_experiment_groups_partition_the_18_active_methods(self):
        primary = set(run_all_methods.PRIMARY_EXPERIMENT_METHODS)
        baselines = set(run_all_methods.BASELINE_EXPERIMENT_METHODS)
        self.assertEqual(len(primary), 8)
        self.assertEqual(len(baselines), 10)
        self.assertTrue(primary.isdisjoint(baselines))
        self.assertEqual(primary | baselines, set(run_all_methods.DEFAULT_METHODS))
        self.assertEqual(
            set(run_all_methods.MAIN_BASELINE_METHODS),
            {"TGPR-MO", "DDMOEA-GAN", "Prob-RVEA", "Prob-MOEA/D"},
        )

    @patch("experiments.run_all_methods.main")
    def test_fixed_group_entry_injects_its_methods(self, unified_main):
        unified_main.return_value = 0
        result = run_all_methods.fixed_group_main(
            "primary methods",
            run_all_methods.PRIMARY_EXPERIMENT_METHODS,
            ["--dry-run"],
            default_output_dir="primary-results",
        )
        self.assertEqual(result, 0)
        forwarded = unified_main.call_args.args[0]
        self.assertEqual(forwarded[0], "--methods")
        self.assertEqual(
            forwarded[1].split(","),
            list(run_all_methods.PRIMARY_EXPERIMENT_METHODS),
        )
        self.assertIn("primary-results", forwarded)

    def test_fixed_group_rejects_methods_from_the_other_group(self):
        with self.assertRaises(SystemExit):
            run_all_methods.fixed_group_main(
                "primary methods",
                run_all_methods.PRIMARY_EXPERIMENT_METHODS,
                ["--methods", "DDMOEA-GAN"],
            )

    def test_disabled_method_can_still_be_selected_explicitly(self):
        args = run_all_methods.parse_args(
            ["--methods", "XGBoost + NSGA-II", "--dry-run"]
        )
        self.assertEqual(args.methods, ["XGBoost + NSGA-II"])

    def test_mixed_selection_routes_to_each_existing_runner(self):
        args = run_all_methods.parse_args(
            [
                "--methods",
                "DDMOEA-GAN,MultipleModels-COM,PCD",
                "--problems",
                "zdt1",
                "--training-sizes",
                "50",
                "--offline-seeds",
                "1",
                "--optimization-seeds",
                "2",
                "--dry-run",
            ]
        )
        commands = dict(run_all_methods.build_commands(args))
        self.assertEqual(set(commands), {"main", "dl_mobo", "generative"})
        self.assertIn("DDMOEA-GAN", commands["main"])
        self.assertIn("MultipleModels-COM", commands["dl_mobo"])
        self.assertIn("PCD", commands["generative"])
        self.assertIn("--dry-run", commands["main"])
        self.assertIn("--dry-run", commands["dl_mobo"])
        self.assertIn("--dry-run", commands["generative"])

    def test_real_commands_use_only_the_repository_venv(self):
        args = run_all_methods.parse_args(
            ["--methods", "DDMOEA-GAN,MultipleModels-COM,PCD"]
        )
        for _, command in run_all_methods.build_commands(args):
            self.assertEqual(command[0], str(PROJECT_PYTHON))

    @patch("experiments.run_all_methods.subprocess.run")
    def test_dispatch_continues_and_reports_child_failure(self, run):
        run.side_effect = [
            type("Result", (), {"returncode": 0})(),
            type("Result", (), {"returncode": 3})(),
        ]
        exit_code = run_all_methods.main(
            ["--methods", "DDMOEA-GAN,PCD", "--dry-run"]
        )
        self.assertEqual(exit_code, 1)
        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
