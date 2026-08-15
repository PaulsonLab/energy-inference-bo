from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import torch

from energy_bo.experiments.task05a import MODELS, PROTOCOL_VERSION, Task05AConfig
from energy_bo.experiments.task05a_campaign import (
    aggregate_if_complete,
    inspect_campaign,
    inspect_shard,
    next_incomplete,
    package_campaign,
    run_campaign,
    shard_keys,
    write_campaign_status,
    write_profile_review,
)


GIT_SHA = "a" * 40
CONFIG_HASH = Task05AConfig.full_shard("trpb", 0, "cpu").config_hash


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _complete_shard(directory: Path, dataset: str, seed: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _json(directory / "run_state.json", {"git_sha": GIT_SHA, "config_hash": CONFIG_HASH, "protocol_version": PROTOCOL_VERSION})
    _json(directory / "config.yaml", {
        "profile": "full", "datasets": [dataset], "seeds": [seed], "config_hash": CONFIG_HASH,
    })
    offline = [
        {"dataset": dataset, "seed": seed, "n": n, "model": model, "finite": True, "converged": True, "fit_seconds": 1.0}
        for n in (48, 96, 192) for model in MODELS
    ]
    sequential = [{"dataset": dataset, "seed": seed, "model": model, "converged": True, "fit_seconds": 2.0} for model in MODELS]
    traces = [
        {"dataset": dataset, "seed": seed, "model": model, "step": step, "fit_converged": True}
        for model in MODELS for step in range(1, 33)
    ]
    for n in (48, 96, 192):
        _csv(directory / f"offline_{dataset}_seed{seed}_n{n}.csv", [row for row in offline if row["n"] == n])
    _csv(directory / "offline_metrics.csv", offline)
    _csv(directory / "sequential_metrics.csv", sequential)
    _csv(directory / "metrics.csv", offline + sequential)
    _csv(directory / "bo_trace.csv", traces)
    _json(directory / "gate_result.json", {"status": "INCONCLUSIVE"})
    _json(directory / "run_metadata.json", {"git_sha": GIT_SHA, "config_hash": CONFIG_HASH, "elapsed_seconds": 12.5, "peak_cuda_mb": 64.0, "peak_rss_mb": 128.0})
    (directory / "TASK_05A_RUN_SUMMARY.md").write_text("complete")
    for model in MODELS:
        checkpoint = directory / "checkpoints" / f"{dataset}_seed{seed}_{model}.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"git_sha": GIT_SHA, "config_hash": CONFIG_HASH, "completed_steps": 32}, checkpoint)


def test_campaign_grid_and_order_are_frozen() -> None:
    keys = shard_keys()
    assert len(keys) == 20
    assert keys[:2] == (("trpb", 0), ("trpb", 1))
    assert keys[-1] == ("creilov", 9)


def test_status_distinguishes_pending_partial_complete_and_incompatible(tmp_path: Path) -> None:
    pending = inspect_shard(tmp_path / "pending", "trpb", 0, GIT_SHA, CONFIG_HASH)
    assert pending.state == "PENDING"

    partial_dir = tmp_path / "partial"
    _json(partial_dir / "campaign_shard_status.json", {"state": "RUNNING"})
    partial = inspect_shard(partial_dir, "trpb", 0, GIT_SHA, CONFIG_HASH)
    assert partial.state == "PARTIAL"

    complete_dir = tmp_path / "complete"
    _complete_shard(complete_dir, "trpb", 0)
    complete = inspect_shard(complete_dir, "trpb", 0, GIT_SHA, CONFIG_HASH)
    assert complete.state == "COMPLETE"
    assert complete.offline_fits == 9 and complete.bo_steps == 96

    checkpoint = complete_dir / "checkpoints" / "trpb_seed0_S0.pt"
    saved = torch.load(checkpoint, weights_only=False)
    saved["git_sha"] = "b" * 40
    torch.save(saved, checkpoint)
    incompatible = inspect_shard(complete_dir, "trpb", 0, GIT_SHA, CONFIG_HASH)
    assert incompatible.state == "INCOMPATIBLE"

    broken_dir = tmp_path / "broken"
    broken_dir.mkdir(); (broken_dir / "run_state.json").write_text("not-json")
    broken = inspect_shard(broken_dir, "trpb", 0, GIT_SHA, CONFIG_HASH)
    assert broken.state == "FAILED" and "invalid run_state.json" in str(broken.error)


def test_campaign_status_is_human_and_machine_readable(tmp_path: Path) -> None:
    statuses = inspect_campaign(tmp_path, GIT_SHA, CONFIG_HASH)
    payload = write_campaign_status(tmp_path, statuses, GIT_SHA, CONFIG_HASH)
    assert payload["counts"]["PENDING"] == 20
    assert payload["next_shard"] == "trpb_seed0"
    assert "Complete: **0/20**" in (tmp_path / "CAMPAIGN_STATUS.md").read_text()
    assert len(list(csv.DictReader((tmp_path / "campaign_status.csv").open()))) == 20


def test_profile_review_is_explicitly_technical_not_gate_evidence(tmp_path: Path) -> None:
    _complete_shard(tmp_path / "full_shards" / "trpb_seed0", "trpb", 0)
    review = write_profile_review(tmp_path, GIT_SHA, CONFIG_HASH)
    assert review["state"] == "COMPLETE"
    assert review["offline_converged"] == 9
    assert review["bo_fit_converged"] == 96
    assert review["trajectory_converged"] == 3
    assert "not gate evidence" in review["purpose"]


def test_profile_resumes_stale_running_shard_without_repeating_complete_work(tmp_path: Path) -> None:
    root, repo = tmp_path / "campaign", tmp_path / "repo"
    repo.mkdir()
    shard = root / "full_shards" / "trpb_seed0"
    _json(shard / "campaign_shard_status.json", {"state": "RUNNING"})
    calls: list[str] = []

    def launcher(command: list[str], _repo: Path, _timeout: int):
        calls.append(command[command.index("--output-dir") + 1])
        _complete_shard(shard, "trpb", 0)
        return 0, None

    result = run_campaign(
        root, repo, root / "data", GIT_SHA, CONFIG_HASH, mode="profile", device="cpu", launcher=launcher,
    )
    assert calls == [str(shard)]
    assert result["counts"]["COMPLETE"] == 1
    assert result["stop_reason"] == "profile_complete"

    second = run_campaign(
        root, repo, root / "data", GIT_SHA, CONFIG_HASH, mode="profile", device="cpu", launcher=launcher,
    )
    assert len(calls) == 1
    assert second["stop_reason"] == "profile_complete"


def test_campaign_requires_profile_and_respects_deterministic_next_shard(tmp_path: Path) -> None:
    root, repo = tmp_path / "campaign", tmp_path / "repo"
    repo.mkdir()
    try:
        run_campaign(root, repo, root / "data", GIT_SHA, CONFIG_HASH, mode="campaign", device="cpu")
    except RuntimeError as error:
        assert "profiling shard" in str(error)
    else:
        raise AssertionError("campaign should require its profile shard")

    _complete_shard(root / "full_shards" / "trpb_seed0", "trpb", 0)
    calls: list[str] = []

    def launcher(command: list[str], _repo: Path, _timeout: int):
        dataset = command[command.index("--dataset") + 1]
        seed = int(command[command.index("--seed") + 1])
        output = Path(command[command.index("--output-dir") + 1])
        calls.append(f"{dataset}_seed{seed}")
        _complete_shard(output, dataset, seed)
        return 0, None

    result = run_campaign(
        root, repo, root / "data", GIT_SHA, CONFIG_HASH, mode="campaign", device="cpu",
        session_budget_seconds=60 * 60, shard_timeout_seconds=10, launcher=launcher,
    )
    assert calls[0] == "trpb_seed1"
    assert result["counts"]["COMPLETE"] == 20
    assert next_incomplete(inspect_campaign(root, GIT_SHA, CONFIG_HASH)) is None


def test_aggregate_and_final_package_require_all_shards(tmp_path: Path, monkeypatch) -> None:
    root, repo = tmp_path / "campaign", tmp_path / "repo"
    repo.mkdir()
    assert aggregate_if_complete(root, repo, GIT_SHA, CONFIG_HASH) is None

    for dataset, seed in shard_keys():
        _complete_shard(root / "full_shards" / f"{dataset}_seed{seed}", dataset, seed)

    def fake_run(command: list[str], cwd: Path, check: bool):
        assert cwd == repo and check
        output = Path(command[command.index("--output-dir") + 1])
        output.mkdir(parents=True)
        for name in ("config.yaml", "metrics.csv", "run_metadata.json", "TASK_05A_AGGREGATE_SUMMARY.md"):
            (output / name).write_text("{}")
        _json(output / "gate_result.json", {"git_sha": GIT_SHA, "config_hash": CONFIG_HASH, "status": "PASS"})

    monkeypatch.setattr("energy_bo.experiments.task05a_campaign.subprocess.run", fake_run)
    aggregate = aggregate_if_complete(root, repo, GIT_SHA, CONFIG_HASH)
    assert aggregate == root / "aggregate"
    package = package_campaign(root)
    assert package.exists()
    assert (root / "SHA256SUMS.json").exists()
    diagnostic = package_campaign(root, diagnostic=True)
    with zipfile.ZipFile(diagnostic) as archive:
        assert not any(name.endswith(".pt") for name in archive.namelist())
