from __future__ import annotations

import unittest
from unittest.mock import patch

from experiments import run_all_methods
from experiments import project_environment
from experiments.project_environment import PROJECT_PYTHON


class UnifiedDispatcherTest(unittest.TestCase):
    def test_molecule_dependencies_follow_problem_selection(self):
        self.assertFalse(project_environment._includes_molecule(None))
        self.assertTrue(project_environment._includes_molecule("zdt1,molecule"))
        self.assertFalse(project_environment._includes_molecule("zdt1,re21"))

    def test_unified_registry_contains_only_the_22_experiment_methods(self):
        self.assertEqual(len(run_all_methods.ALL_METHODS), 22)
        self.assertEqual(len(set(run_all_methods.ALL_METHODS)), 22)
        self.assertEqual(
            {key: len(value) for key, value in run_all_methods.METHOD_GROUPS.items()},
            {"main": 16, "dl_mobo": 4, "generative": 2},
        )
        self.assertTrue(
            set(run_all_methods.ALL_METHODS).isdisjoint(
                run_all_methods.HIDDEN_METHODS
            )
        )

    def test_combined_default_run_is_rejected(self):
        with self.assertRaises(SystemExit):
            run_all_methods.parse_args(["--dry-run"])

    def test_two_experiment_groups_partition_the_22_active_methods(self):
        primary = set(run_all_methods.PRIMARY_EXPERIMENT_METHODS)
        baselines = set(run_all_methods.BASELINE_EXPERIMENT_METHODS)
        self.assertEqual(len(primary), 12)
        self.assertEqual(len(baselines), 10)
        self.assertTrue(primary.isdisjoint(baselines))
        self.assertEqual(primary | baselines, set(run_all_methods.DEFAULT_METHODS))
        self.assertEqual(
            set(run_all_methods.MAIN_BASELINE_METHODS),
            {"TGPR-MO", "DDMOEA-GAN", "Prob-RVEA", "Prob-MOEA/D"},
        )

    def test_primary_categories_are_normal_dr_and_ebu_dr(self):
        self.assertEqual(
            set(run_all_methods.PRIMARY_METHOD_GROUPS),
            {"normal", "dr", "ebu_dr"},
        )
        self.assertTrue(
            all(
                len(methods) == 4
                for methods in run_all_methods.PRIMARY_METHOD_GROUPS.values()
            )
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

    def test_hidden_method_is_not_selectable_from_unified_entries(self):
        with self.assertRaises(SystemExit):
            run_all_methods.parse_args(
                ["--methods", "XGBoost + NSGA-II", "--dry-run"]
            )

    def test_primary_and_baseline_methods_cannot_be_mixed(self):
        with self.assertRaises(SystemExit):
            run_all_methods.parse_args(
                [
                    "--methods",
                    "GPR-RBF + NSGA-II,DDMOEA-GAN",
                    "--dry-run",
                ]
            )

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
                "--max-workers",
                "7",
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
        for command in commands.values():
            self.assertIn("--max-workers", command)
            self.assertIn("7", command)

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
