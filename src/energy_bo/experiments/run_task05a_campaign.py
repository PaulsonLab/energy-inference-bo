"""CLI for the resumable Task 05A Colab campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .task05a import Task05AConfig
from .task05a_campaign import (
    DEFAULT_SESSION_BUDGET_SECONDS,
    DEFAULT_SHARD_TIMEOUT_SECONDS,
    aggregate_if_complete,
    package_campaign,
    refresh_campaign_status,
    run_campaign,
    write_environment_manifest,
    write_profile_review,
)


def _git_sha(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("status", "profile", "campaign", "package", "diagnostic"), required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, default=Path.cwd())
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--session-budget-seconds", type=int, default=DEFAULT_SESSION_BUDGET_SECONDS)
    parser.add_argument("--shard-timeout-seconds", type=int, default=DEFAULT_SHARD_TIMEOUT_SECONDS)
    args = parser.parse_args()

    repo = args.repo_dir.resolve()
    root = args.campaign_root.resolve()
    data = (args.data_dir or root / "data").resolve()
    git_sha = _git_sha(repo)
    config_hash = Task05AConfig.full_shard("trpb", 0, args.device).config_hash
    write_environment_manifest(root, git_sha, config_hash, args.device)

    if args.mode == "status":
        result = refresh_campaign_status(root, git_sha, config_hash)
    elif args.mode in {"profile", "campaign"}:
        result = run_campaign(
            root, repo, data, git_sha, config_hash, mode=args.mode, device=args.device,
            session_budget_seconds=args.session_budget_seconds,
            shard_timeout_seconds=args.shard_timeout_seconds,
        )
        if args.mode == "profile" and result["counts"]["COMPLETE"] >= 1:
            result["profile_review"] = write_profile_review(root, git_sha, config_hash)
        if args.mode == "campaign":
            aggregate = aggregate_if_complete(root, repo, git_sha, config_hash)
            result["aggregate"] = str(aggregate) if aggregate else None
            result["zip"] = str(package_campaign(root)) if aggregate else None
    elif args.mode == "package":
        aggregate = aggregate_if_complete(root, repo, git_sha, config_hash)
        if aggregate is None:
            raise RuntimeError("all twenty validated shards are required before packaging")
        result = {"aggregate": str(aggregate), "zip": str(package_campaign(root))}
    else:
        result = {"zip": str(package_campaign(root, diagnostic=True))}
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
