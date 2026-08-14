"""Command-line driver for Task 04A."""

from __future__ import annotations

import argparse
from pathlib import Path

from .task04a import Task04AConfig, run_task04a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "preflight", "full"), default="smoke")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    config = {"smoke":Task04AConfig.smoke,"preflight":Task04AConfig.preflight,"full":Task04AConfig.full}[args.profile]()
    run_task04a(config, args.output_dir or Path(f"artifacts/task04a/{args.profile}"), device=args.device)


if __name__ == "__main__":
    main()
