"""Command-line driver for Task 04A-E."""

from __future__ import annotations

import argparse
from pathlib import Path

from .task04ae import Task04AEConfig, run_task04ae


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",choices=("smoke","full"),default="smoke")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    config=Task04AEConfig.smoke() if args.profile=="smoke" else Task04AEConfig.full()
    run_task04ae(config,args.output_dir or Path(f"artifacts/task04ae/{args.profile}"),device=args.device)


if __name__=="__main__":main()
