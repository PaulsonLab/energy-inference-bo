"""Run the preregistered E3 redundant-preference-bank follow-up pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_minimal_pilot import run


EXPERIMENT_DIRECTORY = Path(__file__).resolve().parent
CONFIG_PATH = EXPERIMENT_DIRECTORY / "configs" / "redundant_bank_pilot.json"
DEFAULT_OUTPUT_DIRECTORY = (
    EXPERIMENT_DIRECTORY / "outputs" / "redundant_bank_pilot"
)
DEFAULT_SMOKE_OUTPUT_DIRECTORY = (
    EXPERIMENT_DIRECTORY / "outputs" / "redundant_bank_pilot_smoke"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run the preregistered reduced-count mechanical smoke profile",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="optional new output directory; existing paths are never overwritten",
    )
    arguments = parser.parse_args()
    run(
        smoke=arguments.smoke,
        output_directory=arguments.output_directory,
        config_path=CONFIG_PATH,
        default_output_directory=DEFAULT_OUTPUT_DIRECTORY,
        default_smoke_output_directory=DEFAULT_SMOKE_OUTPUT_DIRECTORY,
        result_title="E3 Preference-BO Redundant-Bank Pilot Results",
        entrypoint_path=Path(__file__),
    )


if __name__ == "__main__":
    main()
