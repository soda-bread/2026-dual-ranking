"""Per-process CPU thread limits for parallel experiment workers."""

from __future__ import annotations

import contextlib
import os
import re
import sys
from pathlib import Path


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def suppress_known_optional_dependency_warnings() -> None:
    """Hide only Off-MOO warnings for task families outside this experiment."""

    from src.offline_moo_adapter import suppress_offline_moo_optional_warnings

    suppress_offline_moo_optional_warnings()


def set_worker_thread_environment(num_threads: int = 1) -> None:
    """Set native-library limits before NumPy/Torch are imported in workers."""

    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError("num_threads must be positive.")
    value = str(num_threads)
    for name in THREAD_ENVIRONMENT_VARIABLES:
        os.environ[name] = value


def initialize_worker_threads(num_threads: int = 1) -> None:
    """Initialize one worker and cap both native and Torch thread pools."""

    set_worker_thread_environment(num_threads)
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(int(num_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting inter-op threads only before parallel work
        # starts. Repeated calls in a serial worker are harmless to ignore.
        pass


def safe_log_component(value) -> str:
    """Return a filesystem-safe component for one experiment-group log."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "item"


def group_log_path(output_dir, problem, size, offline_seed, method) -> Path:
    """Return the common per-group log path used by every runner."""

    filename = (
        f"{safe_log_component(problem)}_N{int(size)}_lhs{int(offline_seed)}_"
        f"{safe_log_component(method)}.log"
    )
    return Path(output_dir) / "logs" / filename


@contextlib.contextmanager
def redirect_process_output(log_path):
    """Redirect Python and native stdout/stderr from one worker to its log."""

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout = os.dup(1)
        saved_stderr = os.dup(2)
        try:
            os.dup2(log_handle.fileno(), 1)
            os.dup2(log_handle.fileno(), 2)
            with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(
                log_handle
            ):
                yield log_handle
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_stdout, 1)
            os.dup2(saved_stderr, 2)
            os.close(saved_stdout)
            os.close(saved_stderr)
