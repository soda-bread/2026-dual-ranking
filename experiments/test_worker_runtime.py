"""Tests for process-level CPU thread limiting in all experiment runners."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from experiments.worker_runtime import (
    THREAD_ENVIRONMENT_VARIABLES,
    initialize_worker_threads,
    set_worker_thread_environment,
)


class WorkerRuntimeTests(unittest.TestCase):
    def test_native_and_torch_thread_pools_are_capped(self):
        fake_torch = SimpleNamespace(
            set_num_threads=Mock(),
            set_num_interop_threads=Mock(),
        )
        original = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT_VARIABLES}
        try:
            with patch.dict(sys.modules, {"torch": fake_torch}):
                initialize_worker_threads(1)
            self.assertTrue(
                all(os.environ[name] == "1" for name in THREAD_ENVIRONMENT_VARIABLES)
            )
            fake_torch.set_num_threads.assert_called_once_with(1)
            fake_torch.set_num_interop_threads.assert_called_once_with(1)
        finally:
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def test_invalid_thread_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_threads must be positive"):
            set_worker_thread_environment(0)

    def test_all_three_process_pools_use_the_initializer(self):
        root = Path(__file__).resolve().parents[1]
        paths = (
            root / "experiments" / "run_all.py",
            root / "experiments" / "DL_MOBO_baseline" / "run.py",
            root / "experiments" / "generative_baseline" / "run.py",
        )
        for path in paths:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("initializer=initialize_worker_threads", source)
                self.assertIn("initargs=(1,)", source)


if __name__ == "__main__":
    unittest.main()
