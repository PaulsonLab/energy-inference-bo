"""Command-line entry point for Task 02B diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

from .task02b import Task02BConfig, retrospective_task02a, run_task02b


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task 02B decision diagnostics.")
    parser.add_argument("--profile", choices=("retrospective", "smoke", "full"), default="smoke")
    parser.add_argument(
        "--task02a-results", type=Path, default=Path("results/task02a/full")
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--retrospective-dir", type=Path)
    parser.add_argument("--summary-path", type=Path)
    args = parser.parse_args()
    if args.profile == "retrospective":
        retrospective_dir = (
            args.output_dir
            or args.retrospective_dir
            or Path("artifacts/task02b/retrospective")
        )
        retrospective_task02a(args.task02a_results, retrospective_dir)
        return
    config = (
        Task02BConfig.smoke()
        if args.profile == "smoke"
        else Task02BConfig.full(args.task02a_results)
    )
    output_dir = args.output_dir or Path(f"artifacts/task02b/{args.profile}")
    retrospective_dir = args.retrospective_dir or Path("artifacts/task02b/retrospective")
    summary_path = args.summary_path or output_dir / "SUMMARY.md"
    run_task02b(
        config,
        args.task02a_results,
        output_dir,
        retrospective_dir,
        summary_path,
    )


if __name__ == "__main__":
    main()
