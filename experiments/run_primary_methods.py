#!/usr/bin/env python3
"""Run the eight active GPR, QR, and BNN primary methods."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_all_methods import (  # noqa: E402
    PRIMARY_EXPERIMENT_METHODS,
    fixed_group_main,
)


def main(argv=None):
    return fixed_group_main(
        "primary methods",
        PRIMARY_EXPERIMENT_METHODS,
        argv,
        default_output_dir=REPO_ROOT / "experiments" / "results_primary_methods",
    )


if __name__ == "__main__":
    raise SystemExit(main())
