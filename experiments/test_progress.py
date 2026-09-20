"""Tests for the shared server progress format."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from experiments.progress import ProgressReporter


class ProgressReporterTests(unittest.TestCase):
    def test_progress_format_matches_main_runner(self):
        output = io.StringIO()
        reporter = ProgressReporter(total=100, skipped=20)
        with redirect_stdout(output):
            reporter.start()
            reporter.record(
                9,
                1,
                "zdt1 | N=100 | offline_seed=3 | ParetoFlow",
            )
            reporter.complete("results")

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "[progress] 20/100 | skipped=20",
                "[progress] 30/100 (30.00%) | success=9 | failed=1 | "
                "total_success=9 | total_failed=1 | skipped=20 | "
                "zdt1 | N=100 | offline_seed=3 | ParetoFlow",
                "[progress] complete | results=results",
            ],
        )


if __name__ == "__main__":
    unittest.main()
