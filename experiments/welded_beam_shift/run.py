"""Run the frozen Welded Beam q=1 decision-shift experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from decision_tilt.welded_beam_experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).with_name("outputs")
    )
    arguments = parser.parse_args()
    summary = run_experiment(arguments.config, arguments.output_dir)
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
