"""Command-line entry point for Task 02C."""

from __future__ import annotations

import argparse
from pathlib import Path

from .task02c import (
    run_full_transport,
    run_saved_teacher_preflight,
    run_smoke_transport,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded Task 02C study.")
    parser.add_argument(
        "--profile", choices=("preflight", "smoke", "full"), default="smoke"
    )
    parser.add_argument(
        "--signature-dir",
        type=Path,
        default=Path("artifacts/task02b/full/signatures"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-path", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path(f"artifacts/task02c/{args.profile}")
    if args.profile == "preflight":
        run_saved_teacher_preflight(args.signature_dir, output)
    elif args.profile == "smoke":
        preflight_path = args.preflight_path or Path(
            "artifacts/task02c/preflight/task02c_preflight.json"
        )
        run_smoke_transport(args.signature_dir, preflight_path, output)
    else:
        run_full_transport(output, args.signature_dir)


if __name__ == "__main__":
    main()
