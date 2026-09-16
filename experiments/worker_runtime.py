"""Per-process CPU thread limits for parallel experiment workers."""

from __future__ import annotations

import os


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


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
