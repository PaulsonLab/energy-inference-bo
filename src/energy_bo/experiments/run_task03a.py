"""Command-line driver for the bounded Task 03A study."""

from __future__ import annotations
import argparse
from pathlib import Path
from .task03a import Task03AConfig,run_task03a

def main()->None:
    parser=argparse.ArgumentParser(); parser.add_argument("--profile",choices=("smoke","full"),default="smoke"); parser.add_argument("--output-dir",type=Path); parser.add_argument("--skip-nuts",action="store_true"); args=parser.parse_args()
    config=Task03AConfig.smoke() if args.profile=="smoke" else Task03AConfig.full(); output=args.output_dir or Path(f"artifacts/task03a/{args.profile}")
    run_task03a(config,output,run_nuts=not args.skip_nuts)
if __name__=="__main__": main()
