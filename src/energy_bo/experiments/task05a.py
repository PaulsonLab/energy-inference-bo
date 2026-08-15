"""Task 05A measured-protein belief experiment.

The runner deliberately separates shard production from full gate aggregation.  A
single full shard is one dataset/seed pair, making Colab interruption recovery
cheap and preventing partial evidence from being mistaken for a gate result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from energy_bo.protein.data import ProteinLandscape, frozen_permutation, load_landscape, smoke_subset
from energy_bo.protein.gate import evaluate_task05a_gate
from energy_bo.protein.metrics import offline_metrics, trajectory_metrics
from energy_bo.protein.models import FitResult, fit_protein_gp, log_ei, predict_raw


PROTOCOL_VERSION = "task05a-v1"
MODELS = ("S0", "S1", "S2")
DATASETS = ("trpb", "creilov")


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux/Colab reports KiB.
    return value / (1024**2 if sys.platform == "darwin" else 1024)


@dataclass(frozen=True)
class Task05AConfig:
    profile: str
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    offline_sizes: tuple[int, ...]
    initial_size: int
    bo_steps: int
    candidate_limit: int | None
    max_iterations: int
    prediction_chunk: int
    device: str
    source_commit: str = "290fa8a4cc99d50980cb8d7cf85ae76744552ead"
    lock_reference_commit: str = "df384fe24c26ebc3ac4a8aab49809d66104f7e8e"
    protocol_version: str = PROTOCOL_VERSION

    @classmethod
    def smoke(cls, device: str = "cpu") -> "Task05AConfig":
        return cls("smoke", DATASETS, (0,), (48,), 48, 2, 2048, 25, 512, device)

    @classmethod
    def full_shard(cls, dataset: str, seed: int, device: str) -> "Task05AConfig":
        if dataset not in DATASETS or seed not in range(10):
            raise ValueError("full shards require dataset trpb/creilov and seed 0..9")
        return cls("full", (dataset,), (seed,), (48, 96, 192), 48, 32, None, 200, 2048, device)

    def protocol_payload(self) -> dict[str, Any]:
        # Shard coordinates are deliberately omitted: all twenty full shards share
        # one frozen protocol hash.
        return {
            "profile": self.profile,
            "datasets": list(DATASETS if self.profile == "full" else self.datasets),
            "seeds": list(range(10) if self.profile == "full" else self.seeds),
            "offline_sizes": list(self.offline_sizes),
            "initial_size": self.initial_size,
            "bo_steps": self.bo_steps,
            "candidate_limit": self.candidate_limit,
            "max_iterations": self.max_iterations,
            "prediction_chunk": self.prediction_chunk,
            "source_commit": self.source_commit,
            "lock_reference_commit": self.lock_reference_commit,
            "protocol_version": self.protocol_version,
        }

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self.protocol_payload(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.read_text().strip():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _candidate_indices(landscape: ProteinLandscape, permutation: np.ndarray, n: int, limit: int | None) -> np.ndarray:
    remaining = permutation[n:]
    if limit is not None:
        remaining = remaining[:limit]
    return remaining


def _fit_and_predict(
    landscape: ProteinLandscape,
    observed: np.ndarray,
    candidates: np.ndarray,
    model_name: str,
    config: Task05AConfig,
    initial_state: dict[str, torch.Tensor] | None = None,
) -> tuple[FitResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(config.device)
    # BoTorch analytic acquisitions call ``module.to(X)`` and therefore require
    # floating inputs.  Kernels convert these exact integer-valued codes to long.
    train_x = landscape.encoded[observed].to(device=device, dtype=torch.double)
    train_y = landscape.fitness[observed].to(device=device, dtype=torch.double)
    fit = fit_protein_gp(train_x, train_y, model_name, max_iterations=config.max_iterations, initial_state=initial_state)
    candidate_x = landscape.encoded[candidates].to(device=device, dtype=torch.double)
    observed_mean, observed_variance = predict_raw(fit, candidate_x, observation_noise=True, chunk_size=config.prediction_chunk)
    latent_values = log_ei(fit, candidate_x, float(train_y.max()), config.prediction_chunk)
    return fit, observed_mean, observed_variance, latent_values


def _offline_case(config: Task05AConfig, landscape: ProteinLandscape, seed: int, n: int, output: Path) -> list[dict[str, Any]]:
    permutation = frozen_permutation(len(landscape.fitness), landscape.name, seed)
    observed = permutation[:n]
    candidates = _candidate_indices(landscape, permutation, n, config.candidate_limit)
    case_path = output / f"offline_{landscape.name}_seed{seed}_n{n}.csv"
    rows: list[dict[str, Any]] = _read_csv(case_path)
    for model_name in MODELS:
        if any(row["model"] == model_name for row in rows):
            continue
        fit, mean, variance, acquisition = _fit_and_predict(landscape, observed, candidates, model_name, config)
        values = landscape.fitness[candidates].cpu()
        metrics = offline_metrics(landscape.fitness[observed].cpu(), values, mean, variance, acquisition)
        rows.append({
            "record_type": "offline",
            "dataset": landscape.name,
            "seed": seed,
            "n": n,
            "model": model_name,
            "finite": bool(torch.isfinite(mean).all() and torch.isfinite(variance).all()),
            "converged": fit.converged,
            "fit_message": fit.message,
            "fit_iterations": fit.iterations,
            "fit_evaluations": fit.function_evaluations,
            "fit_seconds": fit.wall_seconds,
            "objective": fit.objective,
            "candidate_count": len(candidates),
            **metrics,
        })
        _write_csv(case_path, rows)
    return rows


def _state_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def _sequential_case(config: Task05AConfig, landscape: ProteinLandscape, seed: int, output: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    permutation = frozen_permutation(len(landscape.fitness), landscape.name, seed)
    pool = permutation if config.candidate_limit is None else permutation[: config.initial_size + config.candidate_limit]
    global_best = float(landscape.fitness[pool].max())
    top1 = float(torch.quantile(landscape.fitness[pool], 0.99))
    top5 = float(torch.quantile(landscape.fitness[pool], 0.95))
    summaries, traces = [], []
    for model_name in MODELS:
        checkpoint_path = output / "checkpoints" / f"{landscape.name}_seed{seed}_{model_name}.pt"
        observed = list(map(int, pool[: config.initial_size]))
        best_values = [float(landscape.fitness[observed].max())]
        warm_state = None
        model_traces: list[dict[str, Any]] = []
        fit_seconds = 0.0
        start_step = 0
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if saved["config_hash"] != config.config_hash or saved["git_sha"] != _git_sha():
                raise RuntimeError(f"incompatible checkpoint: {checkpoint_path}")
            observed, best_values, warm_state = saved["observed"], saved["best_values"], saved["model_state"]
            model_traces = saved.get("traces", [])
            fit_seconds = float(saved.get("fit_seconds", 0.0))
            start_step = int(saved["completed_steps"])
        for step in range(start_step, config.bo_steps):
            observed_set = set(observed)
            candidates = np.asarray([int(index) for index in pool if int(index) not in observed_set], dtype=np.int64)
            fit, _, _, acquisition = _fit_and_predict(landscape, np.asarray(observed), candidates, model_name, config, warm_state)
            fit_seconds += fit.wall_seconds
            chosen_offset = int(torch.argmax(acquisition))
            chosen = int(candidates[chosen_offset])
            observed.append(chosen)
            best_values.append(max(best_values[-1], float(landscape.fitness[chosen])))
            warm_state = _state_cpu(fit.model)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint_path.with_suffix(".tmp")
            trace = {"dataset": landscape.name, "seed": seed, "model": model_name, "step": step + 1, "chosen_index": chosen, "chosen_fitness": float(landscape.fitness[chosen]), "best_fitness": best_values[-1], "fit_converged": fit.converged, "fit_seconds": fit.wall_seconds}
            model_traces.append(trace)
            torch.save({"config_hash": config.config_hash, "git_sha": _git_sha(), "completed_steps": step + 1, "observed": observed, "best_values": best_values, "model_state": warm_state, "traces": model_traces, "fit_seconds": fit_seconds}, temporary)
            os.replace(temporary, checkpoint_path)
        traces.extend(model_traces)
        metrics = trajectory_metrics(best_values, global_best)
        summaries.append({"record_type": "sequential", "dataset": landscape.name, "seed": seed, "model": model_name, "steps": config.bo_steps, "fit_seconds": fit_seconds, "converged": all(bool(row["fit_converged"]) for row in model_traces), "top1_hit": best_values[-1] >= top1, "top5_hit": best_values[-1] >= top5, **metrics})
    return summaries, traces


def run_task05a(config: Task05AConfig, output: Path, data_dir: Path) -> dict[str, Any]:
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Task 05A requested CUDA but torch.cuda.is_available() is false")
    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    git_sha = _git_sha()
    run_state_path = output / "run_state.json"
    run_state = {"git_sha": git_sha, "config_hash": config.config_hash, "protocol_version": PROTOCOL_VERSION}
    if run_state_path.exists() and json.loads(run_state_path.read_text()) != run_state:
        raise RuntimeError(f"incompatible existing Task 05A output directory: {output}")
    _atomic_json(run_state_path, run_state)
    _atomic_json(output / "config.yaml", {**asdict(config), "config_hash": config.config_hash})
    offline_rows: list[dict[str, Any]] = []
    sequential_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    for dataset in config.datasets:
        landscape = load_landscape(dataset, data_dir, download=True)
        if config.profile == "smoke":
            landscape = smoke_subset(
                landscape,
                config.seeds[0],
                config.initial_size + int(config.candidate_limit or 0),
            )
        for seed in config.seeds:
            for n in config.offline_sizes:
                offline_rows.extend(_offline_case(config, landscape, seed, n, output))
            summaries, traces = _sequential_case(config, landscape, seed, output)
            sequential_rows.extend(summaries); trace_rows.extend(traces)
    _write_csv(output / "offline_metrics.csv", offline_rows)
    _write_csv(output / "sequential_metrics.csv", sequential_rows)
    _write_csv(output / "metrics.csv", offline_rows + sequential_rows)
    _write_csv(output / "bo_trace.csv", trace_rows)
    gate = evaluate_task05a_gate(offline_rows, sequential_rows, profile=config.profile, git_sha=git_sha, config_hash=config.config_hash)
    _atomic_json(output / "gate_result.json", gate)
    metadata = {
        "task_id": "05A", "protocol_version": PROTOCOL_VERSION, "git_sha": git_sha,
        "config_hash": config.config_hash, "elapsed_seconds": time.perf_counter() - started,
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "device": config.device, "cuda_device": torch.cuda.get_device_name() if config.device == "cuda" and torch.cuda.is_available() else None,
        "peak_rss_mb": _peak_rss_mb(),
        "peak_cuda_mb": torch.cuda.max_memory_allocated() / 1024**2 if config.device == "cuda" else 0.0,
    }
    _atomic_json(output / "run_metadata.json", metadata)
    (output / "TASK_05A_RUN_SUMMARY.md").write_text(
        "# Task 05A run summary\n\n"
        f"Profile: `{config.profile}`; offline rows: {len(offline_rows)}; "
        f"sequential rows: {len(sequential_rows)}; elapsed: {metadata['elapsed_seconds']:.2f} s.\n\n"
        f"Gate status: `{gate['status']}`. {gate['notes']}\n"
    )
    return {"gate": gate, "metadata": metadata, "offline_rows": len(offline_rows), "sequential_rows": len(sequential_rows)}


def aggregate_task05a(shards: Path, output: Path) -> dict[str, Any]:
    offline, sequential = [], []
    metadata = []
    expected_hash = Task05AConfig.full_shard("trpb", 0, "cpu").config_hash
    for dataset in DATASETS:
        for seed in range(10):
            shard = shards / f"{dataset}_seed{seed}"
            config = json.loads((shard / "config.yaml").read_text())
            if config["config_hash"] != expected_hash:
                raise RuntimeError(f"protocol mismatch in {shard}")
            offline.extend(_read_csv(shard / "offline_metrics.csv"))
            sequential.extend(_read_csv(shard / "sequential_metrics.csv"))
            metadata.append(json.loads((shard / "run_metadata.json").read_text()))
    git_shas = sorted({item["git_sha"] for item in metadata})
    if len(git_shas) != 1:
        raise RuntimeError(f"shards use different Git revisions: {git_shas}")
    output.mkdir(parents=True, exist_ok=True)
    frozen = Task05AConfig.full_shard("trpb", 0, "cpu").protocol_payload()
    _atomic_json(output / "config.yaml", {**frozen, "config_hash": expected_hash})
    _write_csv(output / "metrics.csv", offline + sequential)
    _write_csv(output / "offline_metrics.csv", offline)
    _write_csv(output / "sequential_metrics.csv", sequential)
    gate = evaluate_task05a_gate(offline, sequential, profile="full", git_sha=git_shas[0], config_hash=expected_hash)
    _atomic_json(output / "gate_result.json", gate)
    _atomic_json(output / "run_metadata.json", {"task_id": "05A", "protocol_version": PROTOCOL_VERSION, "git_sha": git_shas[0], "config_hash": expected_hash, "shards": metadata})
    _aggregate_figures(offline, sequential, output / "figures")
    path = gate["gate_checks"].get("path", gate["status"])
    selected = gate["gate_checks"].get("selected_model")
    (output / "TASK_05A_AGGREGATE_SUMMARY.md").write_text(
        "# Task 05A full aggregate\n\n"
        f"All 20 frozen shards were validated at Git `{git_shas[0]}`. "
        f"The mechanical gate is **{gate['status']}** (`{path}`)"
        f"{f' with selected belief {selected}' if selected else ''}.\n\n"
        "Use `gate_result.json` and the canonical task summary for interpretation; "
        "do not authorize Task 05B from this generated sentence alone.\n"
    )
    return gate


def _aggregate_figures(offline: list[dict[str, Any]], sequential: list[dict[str, Any]], directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for axis, dataset in zip(axes, DATASETS, strict=True):
        for model in MODELS:
            rows = [row for row in offline if row["dataset"] == dataset and row["model"] == model]
            sizes = sorted({int(row["n"]) for row in rows})
            values = [np.median([float(row["one_step_regret"]) for row in rows if int(row["n"]) == n]) for n in sizes]
            axis.plot(sizes, values, marker="o", label=model)
        axis.set(title=dataset, xlabel="training size", ylabel="median one-step regret")
        axis.set_ylim(bottom=0); axis.legend()
    fig.tight_layout(); fig.savefig(directory / "offline_decision_regret.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3.6))
    labels, values = [], []
    for dataset in DATASETS:
        for model in MODELS:
            labels.append(f"{dataset}\n{model}")
            values.append(np.median([float(row["regret_auc"]) for row in sequential if row["dataset"] == dataset and row["model"] == model]))
    ax.bar(labels, values); ax.set_ylabel("median normalized-regret AUC"); ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout(); fig.savefig(directory / "sequential_auc.png", dpi=180); plt.close(fig)
