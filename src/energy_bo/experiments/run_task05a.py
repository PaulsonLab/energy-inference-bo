"""CLI for Task 05A smoke, full shards, and aggregation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .task05a import Task05AConfig, aggregate_task05a, run_task05a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full", "aggregate"), required=True)
    parser.add_argument("--dataset", choices=("trpb", "creilov"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--data-dir", type=Path, default=Path("data/task05a"))
    parser.add_argument("--shards-dir", type=Path, default=Path("artifacts/task05a/full_shards"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.profile == "aggregate":
        result = aggregate_task05a(args.shards_dir, args.output_dir or Path("results/task05a/full"))
    else:
        if args.profile == "full":
            if args.dataset is None or args.seed is None:
                parser.error("full profile requires --dataset and --seed")
            config = Task05AConfig.full_shard(args.dataset, args.seed, args.device)
            output = args.output_dir or args.shards_dir / f"{args.dataset}_seed{args.seed}"
        else:
            config = Task05AConfig.smoke(args.device)
            output = args.output_dir or Path("artifacts/task05a/smoke")
        result = run_task05a(config, output, args.data_dir)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
