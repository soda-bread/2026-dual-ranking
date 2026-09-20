"""Consistent terminal progress reporting for every experiment runner."""

from __future__ import annotations


class ProgressReporter:
    """Track completed optimizer runs and print the shared progress format."""

    def __init__(self, total: int, skipped: int = 0):
        self.total = int(total)
        self.skipped = int(skipped)
        self.processed = int(skipped)
        self.total_success = 0
        self.total_failed = 0

    def start(self) -> None:
        print(
            f"[progress] {self.processed}/{self.total} | "
            f"skipped={self.skipped}",
            flush=True,
        )

    def record(self, success_count: int, failed_count: int, label: str) -> None:
        success_count = int(success_count)
        failed_count = int(failed_count)
        self.processed += success_count + failed_count
        self.total_success += success_count
        self.total_failed += failed_count
        percentage = 100.0 * self.processed / self.total if self.total else 100.0
        print(
            f"[progress] {self.processed}/{self.total} ({percentage:.2f}%) | "
            f"success={success_count} | failed={failed_count} | "
            f"total_success={self.total_success} | "
            f"total_failed={self.total_failed} | skipped={self.skipped} | "
            f"{label}",
            flush=True,
        )

    def complete(self, output_dir) -> None:
        print(f"[progress] complete | results={output_dir}", flush=True)

