"""Command-line driver for the Task 04A-D decision diagnostic."""

from __future__ import annotations

import argparse
from pathlib import Path

from .task04ad import DecisionDiagnosticConfig, run_decision_diagnostic


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",choices=("wiring","local"),default="wiring")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    config=DecisionDiagnosticConfig.wiring() if args.profile=="wiring" else DecisionDiagnosticConfig.local()
    run_decision_diagnostic(config,args.output_dir or Path(f"artifacts/task04a/decision_diagnostic/{args.profile}"))


if __name__=="__main__":
    main()
