"""Command-line entry point for the Task 02A falsification experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .task02a import Task02AConfig, run_task02a


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 02A fixed-particle SAAS reuse.")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-path", type=Path, default=Path("TASK_02A_SUMMARY.md"))
    args = parser.parse_args()
    if args.profile == "smoke":
        config = Task02AConfig.smoke(tuple(args.seeds or (0,)))
    else:
        config = Task02AConfig.full(tuple(args.seeds or (0, 1, 2)))
    output_dir = args.output_dir or Path(f"artifacts/task02a_{args.profile}")
    run_task02a(config, output_dir, args.summary_path)


if __name__ == "__main__":
    main()
