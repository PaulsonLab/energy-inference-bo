"""Resumable Drive-backed orchestration for the frozen Task 05A shards.

This module contains no scientific logic.  It validates and schedules the twenty
frozen dataset/seed shards, delegates each shard to ``run_task05a``, and invokes
the existing aggregate evaluator only after every shard is complete.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from .task05a import DATASETS, MODELS, PROTOCOL_VERSION, Task05AConfig


SHARD_STATES = ("PENDING", "PARTIAL", "RUNNING", "COMPLETE", "FAILED", "INCOMPATIBLE")
EXPECTED_OFFLINE_ROWS = 9
EXPECTED_SEQUENTIAL_ROWS = 3
EXPECTED_TRACE_ROWS = 96
EXPECTED_BO_STEPS = 32
DEFAULT_SHARD_TIMEOUT_SECONDS = 2 * 60 * 60 + 55 * 60
DEFAULT_SESSION_BUDGET_SECONDS = 8 * 60 * 60


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def shard_keys() -> tuple[tuple[str, int], ...]:
    """Return the frozen deterministic campaign order."""

    return tuple((dataset, seed) for dataset in DATASETS for seed in range(10))


def shard_name(dataset: str, seed: int) -> str:
    return f"{dataset}_seed{seed}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


@dataclass(frozen=True)
class ShardStatus:
    dataset: str
    seed: int
    state: str
    offline_fits: int
    bo_steps: int
    elapsed_seconds: float | None
    last_update: str | None
    error: str | None

    @property
    def name(self) -> str:
        return shard_name(self.dataset, self.seed)


def _last_update(directory: Path) -> str | None:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if not files:
        return None
    timestamp = max(path.stat().st_mtime for path in files)
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def _marker_error(directory: Path) -> str | None:
    marker = directory / "campaign_shard_status.json"
    if not marker.exists():
        return None
    try:
        value = _read_json(marker)
    except Exception as error:  # corrupted orchestration state is a visible failure
        return f"invalid campaign marker: {error}"
    if value.get("state") == "FAILED":
        return str(value.get("error") or "shard process failed")
    return None


def inspect_shard(directory: Path, dataset: str, seed: int, git_sha: str, config_hash: str) -> ShardStatus:
    """Validate a shard and summarize resumable progress.

    A stale ``RUNNING`` marker is deliberately reported as ``PARTIAL`` on a new
    inspection.  Only the live controller's in-memory display reports RUNNING.
    """

    if not directory.exists() or not any(directory.iterdir()):
        return ShardStatus(dataset, seed, "PENDING", 0, 0, None, None, None)

    offline_rows: list[dict[str, str]] = []
    for n in (48, 96, 192):
        offline_rows.extend(_read_csv(directory / f"offline_{dataset}_seed{seed}_n{n}.csv"))
    offline_keys = {
        (row.get("dataset"), row.get("seed"), row.get("n"), row.get("model"))
        for row in offline_rows
    }
    offline_fits = len(offline_keys)

    bo_steps = 0
    checkpoint_error: str | None = None
    for model in MODELS:
        checkpoint = directory / "checkpoints" / f"{dataset}_seed{seed}_{model}.pt"
        if not checkpoint.exists():
            continue
        try:
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved.get("config_hash") != config_hash or saved.get("git_sha") != git_sha:
                checkpoint_error = f"incompatible checkpoint: {checkpoint.name}"
                break
            bo_steps += int(saved.get("completed_steps", 0))
        except Exception as error:
            checkpoint_error = f"invalid checkpoint {checkpoint.name}: {error}"
            break

    last_update = _last_update(directory)
    elapsed: float | None = None
    metadata_path = directory / "run_metadata.json"
    if metadata_path.exists():
        try:
            elapsed = float(_read_json(metadata_path)["elapsed_seconds"])
        except Exception:
            pass

    if checkpoint_error:
        return ShardStatus(dataset, seed, "INCOMPATIBLE", offline_fits, bo_steps, elapsed, last_update, checkpoint_error)

    for path in (directory / "run_state.json", directory / "config.yaml"):
        if path.exists():
            try:
                value = _read_json(path)
            except Exception as error:
                return ShardStatus(dataset, seed, "FAILED", offline_fits, bo_steps, elapsed, last_update, f"invalid {path.name}: {error}")
            if value.get("git_sha", git_sha) != git_sha or value.get("config_hash") != config_hash:
                return ShardStatus(dataset, seed, "INCOMPATIBLE", offline_fits, bo_steps, elapsed, last_update, f"incompatible {path.name}")
            if path.name == "config.yaml":
                if value.get("datasets") != [dataset] or value.get("seeds") != [seed] or value.get("profile") != "full":
                    return ShardStatus(dataset, seed, "INCOMPATIBLE", offline_fits, bo_steps, elapsed, last_update, "wrong shard coordinates in config.yaml")

    required = (
        "run_state.json", "config.yaml", "offline_metrics.csv", "sequential_metrics.csv",
        "metrics.csv", "bo_trace.csv", "gate_result.json", "run_metadata.json",
        "TASK_05A_RUN_SUMMARY.md",
    )
    if all((directory / name).exists() for name in required):
        try:
            run_state = _read_json(directory / "run_state.json")
            metadata = _read_json(directory / "run_metadata.json")
            gate = _read_json(directory / "gate_result.json")
            offline = _read_csv(directory / "offline_metrics.csv")
            sequential = _read_csv(directory / "sequential_metrics.csv")
            traces = _read_csv(directory / "bo_trace.csv")
            complete = (
                run_state == {"git_sha": git_sha, "config_hash": config_hash, "protocol_version": PROTOCOL_VERSION}
                and metadata.get("git_sha") == git_sha
                and metadata.get("config_hash") == config_hash
                and gate.get("status") == "INCONCLUSIVE"
                and len(offline) == EXPECTED_OFFLINE_ROWS
                and len(sequential) == EXPECTED_SEQUENTIAL_ROWS
                and len(traces) == EXPECTED_TRACE_ROWS
                and offline_fits == EXPECTED_OFFLINE_ROWS
                and bo_steps == len(MODELS) * EXPECTED_BO_STEPS
                and all(row.get("dataset") == dataset and int(row.get("seed", -1)) == seed for row in offline + sequential + traces)
            )
        except Exception as error:
            return ShardStatus(dataset, seed, "FAILED", offline_fits, bo_steps, elapsed, last_update, f"invalid completion files: {error}")
        if complete:
            return ShardStatus(dataset, seed, "COMPLETE", offline_fits, bo_steps, elapsed, last_update, None)
        return ShardStatus(dataset, seed, "FAILED", offline_fits, bo_steps, elapsed, last_update, "completion files failed frozen-protocol validation")

    error = _marker_error(directory)
    state = "FAILED" if error else "PARTIAL"
    return ShardStatus(dataset, seed, state, offline_fits, bo_steps, elapsed, last_update, error)


def inspect_campaign(root: Path, git_sha: str, config_hash: str) -> list[ShardStatus]:
    shards = root / "full_shards"
    return [inspect_shard(shards / shard_name(dataset, seed), dataset, seed, git_sha, config_hash) for dataset, seed in shard_keys()]


def next_incomplete(statuses: Iterable[ShardStatus]) -> ShardStatus | None:
    return next((status for status in statuses if status.state != "COMPLETE"), None)


def _status_markdown(statuses: list[ShardStatus], git_sha: str, config_hash: str) -> str:
    counts = {state: sum(status.state == state for status in statuses) for state in SHARD_STATES}
    upcoming = next_incomplete(statuses)
    lines = [
        "# Task 05A campaign status", "",
        f"Updated: `{utc_now()}`  ",
        f"Git SHA: `{git_sha}`  ",
        f"Configuration hash: `{config_hash}`", "",
        f"Complete: **{counts['COMPLETE']}/20**; partial: **{counts['PARTIAL']}**; "
        f"failed: **{counts['FAILED']}**; incompatible: **{counts['INCOMPATIBLE']}**; "
        f"pending: **{counts['PENDING']}**.", "",
        f"Next shard: **{upcoming.name if upcoming else 'none — campaign complete'}**.", "",
        "| Shard | State | Offline fits | BO steps | Runtime (s) | Last update | Error |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for status in statuses:
        elapsed = "" if status.elapsed_seconds is None else f"{status.elapsed_seconds:.1f}"
        lines.append(
            f"| {status.name} | {status.state} | {status.offline_fits}/9 | {status.bo_steps}/96 | "
            f"{elapsed} | {status.last_update or ''} | {(status.error or '').replace('|', '/')} |"
        )
    return "\n".join(lines) + "\n"


def write_campaign_status(root: Path, statuses: list[ShardStatus], git_sha: str, config_hash: str) -> dict[str, Any]:
    counts = {state: sum(status.state == state for status in statuses) for state in SHARD_STATES}
    upcoming = next_incomplete(statuses)
    payload = {
        "task_id": "05A", "protocol_version": PROTOCOL_VERSION, "git_sha": git_sha,
        "config_hash": config_hash, "updated_at": utc_now(), "counts": counts,
        "next_shard": upcoming.name if upcoming else None,
        "shards": [asdict(status) | {"name": status.name} for status in statuses],
    }
    _atomic_json(root / "campaign_status.json", payload)
    rows = [asdict(status) | {"name": status.name} for status in statuses]
    columns = ("name", "dataset", "seed", "state", "offline_fits", "bo_steps", "elapsed_seconds", "last_update", "error")
    temporary = root / "campaign_status.csv.tmp"
    root.mkdir(parents=True, exist_ok=True)
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, root / "campaign_status.csv")
    _atomic_text(root / "CAMPAIGN_STATUS.md", _status_markdown(statuses, git_sha, config_hash))
    return payload


def refresh_campaign_status(root: Path, git_sha: str, config_hash: str) -> dict[str, Any]:
    return write_campaign_status(root, inspect_campaign(root, git_sha, config_hash), git_sha, config_hash)


def _as_bool(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in {"true", "1"})


def write_profile_review(root: Path, git_sha: str, config_hash: str) -> dict[str, Any]:
    """Summarize the technical profile shard without making a research decision."""

    directory = root / "full_shards" / "trpb_seed0"
    status = inspect_shard(directory, "trpb", 0, git_sha, config_hash)
    if status.state != "COMPLETE":
        raise RuntimeError(f"profile review requires COMPLETE trpb_seed0, got {status.state}")
    offline = _read_csv(directory / "offline_metrics.csv")
    sequential = _read_csv(directory / "sequential_metrics.csv")
    traces = _read_csv(directory / "bo_trace.csv")
    metadata = _read_json(directory / "run_metadata.json")
    elapsed = float(metadata["elapsed_seconds"])
    payload = {
        "task_id": "05A",
        "purpose": "runtime and technical-integrity profile only; not gate evidence by itself",
        "git_sha": git_sha,
        "config_hash": config_hash,
        "shard": "trpb_seed0",
        "state": status.state,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600,
        "under_shard_timeout": elapsed < DEFAULT_SHARD_TIMEOUT_SECONDS,
        "offline_finite": sum(_as_bool(row.get("finite")) for row in offline),
        "offline_converged": sum(_as_bool(row.get("converged")) for row in offline),
        "offline_expected": EXPECTED_OFFLINE_ROWS,
        "bo_fit_converged": sum(_as_bool(row.get("fit_converged")) for row in traces),
        "bo_fit_expected": EXPECTED_TRACE_ROWS,
        "trajectory_converged": sum(_as_bool(row.get("converged")) for row in sequential),
        "trajectory_expected": EXPECTED_SEQUENTIAL_ROWS,
        "peak_cuda_mb": float(metadata.get("peak_cuda_mb", 0.0)),
        "peak_rss_mb": float(metadata.get("peak_rss_mb", 0.0)),
        "naive_twenty_shard_gpu_hours": 20 * elapsed / 3600,
        "notes": (
            "Use elapsed time to plan Colab sessions. One seed cannot establish calibration or BO gains; "
            "only the twenty-shard aggregate can evaluate the frozen gate."
        ),
    }
    _atomic_json(root / "profile_review.json", payload)
    return payload


def estimated_next_seconds(statuses: Iterable[ShardStatus], timeout_seconds: int) -> float:
    completed = [status.elapsed_seconds for status in statuses if status.state == "COMPLETE" and status.elapsed_seconds]
    if not completed:
        return float(timeout_seconds)
    return min(float(timeout_seconds), max(30 * 60.0, 1.5 * max(completed)))


Launcher = Callable[[list[str], Path, int], tuple[int, str | None]]


def _subprocess_launcher(command: list[str], repo: Path, timeout_seconds: int) -> tuple[int, str | None]:
    output = Path(command[command.index("--output-dir") + 1])
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output / "campaign_stdout.log").open("a") as stdout, (output / "campaign_stderr.log").open("a") as stderr:
            completed = subprocess.run(command, cwd=repo, text=True, timeout=timeout_seconds, check=False, stdout=stdout, stderr=stderr)
        return completed.returncode, None if completed.returncode == 0 else f"process returned {completed.returncode}"
    except subprocess.TimeoutExpired:
        return 124, f"shard exceeded {timeout_seconds} seconds"
    except Exception as error:
        return 1, f"could not launch shard: {error}"


def _write_marker(directory: Path, state: str, error: str | None = None) -> None:
    _atomic_json(directory / "campaign_shard_status.json", {"state": state, "updated_at": utc_now(), "error": error})


def run_campaign(
    root: Path,
    repo: Path,
    data_dir: Path,
    git_sha: str,
    config_hash: str,
    *,
    mode: str,
    device: str,
    session_budget_seconds: int = DEFAULT_SESSION_BUDGET_SECONDS,
    shard_timeout_seconds: int = DEFAULT_SHARD_TIMEOUT_SECONDS,
    launcher: Launcher = _subprocess_launcher,
) -> dict[str, Any]:
    """Run the profile shard or as many resumable shards as safely fit."""

    if mode not in {"profile", "campaign"}:
        raise ValueError("mode must be profile or campaign")
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    statuses = inspect_campaign(root, git_sha, config_hash)
    write_campaign_status(root, statuses, git_sha, config_hash)

    profile = statuses[0]
    if mode == "campaign" and profile.state != "COMPLETE":
        raise RuntimeError("profiling shard trpb_seed0 is not complete; run profile mode first")
    targets = ([profile] if profile.state != "COMPLETE" else []) if mode == "profile" else [status for status in statuses if status.state != "COMPLETE"]
    completed_this_session: list[str] = []
    stop_reason = "profile_complete" if mode == "profile" and profile.state == "COMPLETE" else None

    for target in targets:
        if target.state in {"FAILED", "INCOMPATIBLE"}:
            stop_reason = f"{target.name} is {target.state}: {target.error}"
            break
        statuses = inspect_campaign(root, git_sha, config_hash)
        estimate = estimated_next_seconds(statuses, shard_timeout_seconds)
        elapsed = time.monotonic() - started
        if mode == "campaign" and elapsed + estimate > session_budget_seconds:
            stop_reason = f"soft session budget reached before {target.name}"
            break
        directory = root / "full_shards" / target.name
        directory.mkdir(parents=True, exist_ok=True)
        _write_marker(directory, "RUNNING")
        live = inspect_campaign(root, git_sha, config_hash)
        live = [ShardStatus(s.dataset, s.seed, "RUNNING", s.offline_fits, s.bo_steps, s.elapsed_seconds, s.last_update, s.error) if s.name == target.name else s for s in live]
        write_campaign_status(root, live, git_sha, config_hash)
        command = [
            sys.executable, "-m", "energy_bo.experiments.run_task05a", "--profile", "full",
            "--dataset", target.dataset, "--seed", str(target.seed), "--device", device,
            "--data-dir", str(data_dir), "--output-dir", str(directory),
        ]
        returncode, error = launcher(command, repo, shard_timeout_seconds)
        checked = inspect_shard(directory, target.dataset, target.seed, git_sha, config_hash)
        if returncode != 0 or checked.state != "COMPLETE":
            detail = error or checked.error or f"post-run validation returned {checked.state}"
            _write_marker(directory, "FAILED", detail)
            stop_reason = f"{target.name} failed: {detail}"
            break
        _write_marker(directory, "COMPLETE")
        completed_this_session.append(target.name)
        if mode == "profile":
            stop_reason = "profile_complete"
            break

    statuses = inspect_campaign(root, git_sha, config_hash)
    payload = write_campaign_status(root, statuses, git_sha, config_hash)
    payload.update({"mode": mode, "completed_this_session": completed_this_session, "stop_reason": stop_reason})
    return payload


def aggregate_if_complete(root: Path, repo: Path, git_sha: str, config_hash: str) -> Path | None:
    statuses = inspect_campaign(root, git_sha, config_hash)
    write_campaign_status(root, statuses, git_sha, config_hash)
    if any(status.state != "COMPLETE" for status in statuses):
        return None
    aggregate = root / "aggregate"
    command = [
        sys.executable, "-m", "energy_bo.experiments.run_task05a", "--profile", "aggregate",
        "--shards-dir", str(root / "full_shards"), "--output-dir", str(aggregate),
    ]
    subprocess.run(command, cwd=repo, check=True)
    required = ("config.yaml", "metrics.csv", "gate_result.json", "run_metadata.json", "TASK_05A_AGGREGATE_SUMMARY.md")
    if not all((aggregate / name).exists() for name in required):
        raise RuntimeError("aggregate completed without all required outputs")
    gate = _read_json(aggregate / "gate_result.json")
    if gate.get("git_sha") != git_sha or gate.get("config_hash") != config_hash:
        raise RuntimeError("aggregate provenance does not match campaign")
    return aggregate


def write_environment_manifest(root: Path, git_sha: str, config_hash: str, device: str) -> Path:
    payload = {
        "task_id": "05A", "protocol_version": PROTOCOL_VERSION, "git_sha": git_sha,
        "config_hash": config_hash, "created_at": utc_now(), "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__, "device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
    path = root / "colab_manifest.json"
    _atomic_json(path, payload)
    return path


def _included_files(root: Path, diagnostic: bool) -> list[Path]:
    roots = [root / "full_shards", root / "aggregate"] if not diagnostic else [root / "full_shards"]
    files = [
        path for base in roots if base.exists() for path in base.rglob("*")
        if path.is_file() and not path.name.endswith(".tmp") and (not diagnostic or path.suffix != ".pt")
    ]
    for name in ("campaign_status.json", "campaign_status.csv", "CAMPAIGN_STATUS.md", "colab_manifest.json", "profile_review.json"):
        path = root / name
        if path.exists():
            files.append(path)
    return sorted(set(files))


def package_campaign(root: Path, *, diagnostic: bool = False) -> Path:
    if not diagnostic and not (root / "aggregate" / "gate_result.json").exists():
        raise RuntimeError("final ZIP requires a completed aggregate")
    files = _included_files(root, diagnostic)
    inventory = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    inventory_path = root / ("DIAGNOSTIC_SHA256SUMS.json" if diagnostic else "SHA256SUMS.json")
    _atomic_json(inventory_path, inventory)
    files.append(inventory_path)
    zip_path = root / ("task05a_diagnostic.zip" if diagnostic else "task05a_full_results.zip")
    temporary = zip_path.with_suffix(".zip.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, Path("task05a") / path.relative_to(root))
    os.replace(temporary, zip_path)
    return zip_path
