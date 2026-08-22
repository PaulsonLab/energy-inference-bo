"""Frozen Gp2 P1 preference-informed Bayesian-optimization gate.

This module keeps target-blind preprocessing separate from retrospective yield
evaluation.  The graph and preference constructors deliberately do not accept
the scalar target.  Importance-sampling diagnostics are empirical numerical
checks, not rigorous certificates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
from urllib.request import urlopen

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from conditioned_bo.preference_bo import (
    GPPosterior,
    LaplaceResult,
    analytic_expected_improvement,
    laplace_preference_mode,
    select_unobserved_argmax,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ASSAYS = ("Sort1", "Sort8")
REPLICATE_COLUMNS = {
    assay: tuple(f"{assay}_{replicate}_score" for replicate in (1, 2, 3))
    for assay in ASSAYS
}
USED_COLUMNS = (
    "Paratope",
    "Stop",
    "SH_Average_bc",
    *(column for assay in ASSAYS for column in REPLICATE_COLUMNS[assay]),
)
TRAJECTORY_FIELDS = {
    "profile",
    "seed",
    "method",
    "bo_iteration",
    "initial_action_indices",
    "initial_observation_sha256",
    "selected_action_index",
    "selected_paratope",
    "observed_target",
    "best_target_observed",
    "normalized_simple_regret",
    "top_10_percent_hit",
    "config_sha256",
}
INFERENCE_FIELDS = {
    "profile",
    "seed",
    "method",
    "bo_iteration",
    "draws",
    "draw_stage_index",
    "ess",
    "ess_fraction",
    "maximum_split_half_ei_discrepancy",
    "laplace_iterations",
    "laplace_gradient_infinity_norm",
    "factor_likelihood_evaluation_count",
    "wall_time_seconds",
    "numerically_reliable",
    "config_sha256",
}


class PreprocessingAmbiguity(RuntimeError):
    """Raised when duplicate sequences disagree on a gate-used field."""


class PreprocessingInvalid(RuntimeError):
    """Raised when a mechanical preprocessing construction is impossible."""


@dataclass(frozen=True)
class PinnedSource:
    path: str
    sha256: str
    local_path: Path
    row_count: int


@dataclass(frozen=True)
class HammingGraph:
    edges: IntArray
    component_sizes: tuple[int, ...]
    retained_indices: IntArray
    largest_component_fraction: float


@dataclass(frozen=True)
class PreparedGp2Data:
    candidates: pd.DataFrame
    edges: IntArray
    factors: pd.DataFrame
    preprocessing_summary: dict[str, Any]
    source_provenance: tuple[PinnedSource, ...]


@dataclass(frozen=True)
class TargetScaling:
    mean: float
    standard_deviation: float


@dataclass(frozen=True)
class StreamedInferenceResult:
    acquisition: FloatArray
    split_half_acquisition_1: FloatArray
    split_half_acquisition_2: FloatArray
    draws: int
    draw_stage_index: int
    ess: float
    ess_fraction: float
    maximum_split_half_discrepancy: float
    laplace: LaplaceResult
    factor_likelihood_evaluations: int
    wall_time_seconds: float
    reliable: bool


@dataclass(frozen=True)
class GateRunResult:
    trajectory_rows: list[dict[str, Any]]
    inference_rows: list[dict[str, Any]]
    per_seed_rows: list[dict[str, Any]]
    verdict: str
    gate_summary: dict[str, Any] | None
    numerically_reliable: bool


@dataclass
class _WeightedAccumulator:
    dimension: int
    log_weight_sum: float = -np.inf
    log_squared_weight_sum: float = -np.inf
    acquisition: FloatArray | None = None

    def add(self, log_weights: FloatArray, utilities: FloatArray) -> None:
        chunk_log_sum = float(logsumexp(log_weights))
        chunk_mean = np.asarray(
            np.exp(log_weights - chunk_log_sum) @ utilities, dtype=float
        )
        new_log_sum = float(np.logaddexp(self.log_weight_sum, chunk_log_sum))
        if self.acquisition is None:
            self.acquisition = chunk_mean
        else:
            self.acquisition = np.asarray(
                np.exp(self.log_weight_sum - new_log_sum) * self.acquisition
                + np.exp(chunk_log_sum - new_log_sum) * chunk_mean,
                dtype=float,
            )
        self.log_weight_sum = new_log_sum
        self.log_squared_weight_sum = float(
            np.logaddexp(
                self.log_squared_weight_sum, float(logsumexp(2.0 * log_weights))
            )
        )


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_sha256(path: str | Path) -> str:
    return sha256_file(path)


def load_gate_config(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the frozen scientific choices."""

    config = json.loads(Path(path).read_text())
    if config.get("status") != "preregistered_frozen":
        raise ValueError("configuration is not marked preregistered_frozen")
    if config["external_data"]["repository"] != "HackelLab-UMN/DevRep":
        raise ValueError("external repository is not the frozen DevRep source")
    if (
        config["external_data"]["commit"]
        != "e05023a8abe7be6c2e22f42d523b20bd76cd8da5"
    ):
        raise ValueError("external commit differs from the frozen commit")
    if [item["path"] for item in config["external_data"]["files"]] != [
        "datasets/assay_to_yield_training_sequences.csv",
        "datasets/test_sequences.csv",
    ]:
        raise ValueError("external paths differ from the frozen paths")
    if config["target_column"] != "SH_Average_bc":
        raise ValueError("target column differs from the frozen target")
    if config["preference"]["assays"] != ["Sort1", "Sort8"]:
        raise ValueError("only Sort1 and Sort8 are permitted")
    if float(config["preference"]["tau_pref"]) != 1.0:
        raise ValueError("tau_pref must equal 1.0")
    if int(config["graph"]["k"]) != 8:
        raise ValueError("the graph degree parameter must equal eight")
    bo = config["bo"]
    if bo["methods"] != ["scalar_only", "full_preference"]:
        raise ValueError("the P1 gate has exactly two frozen methods")
    if bo["seeds"] != list(range(20)):
        raise ValueError("scientific seeds must be 0,...,19")
    if int(bo["initial_action_count"]) != 5 or int(bo["post_initial_horizon"]) != 5:
        raise ValueError("the frozen BO protocol is five initial plus five BO actions")
    if config["inference"]["scientific_draw_schedule"] != [
        8192,
        16384,
        32768,
        65536,
    ]:
        raise ValueError("scientific draw schedule differs from the handoff")
    if config["inference"]["smoke_draw_schedule"] != [1024, 2048, 4096]:
        raise ValueError("smoke draw schedule differs from the handoff")
    return config


def _download(url: str, destination: Path) -> None:
    with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def ensure_pinned_sources(
    config: Mapping[str, Any],
    repository_root: str | Path,
    *,
    downloader: Callable[[str, Path], None] = _download,
) -> tuple[PinnedSource, ...]:
    """Download missing pinned files and reject every hash mismatch."""

    root = Path(repository_root)
    external = config["external_data"]
    cache = root / str(external["cache_directory"])
    cache.mkdir(parents=True, exist_ok=True)
    commit = str(external["commit"])
    base = str(external["raw_base_url"]).rstrip("/")
    sources: list[PinnedSource] = []
    for item in external["files"]:
        source_path = str(item["path"])
        expected = str(item["sha256"])
        local = cache / Path(source_path).name
        if local.exists() and sha256_file(local) != expected:
            raise RuntimeError(f"cached pinned-data hash mismatch: {local}")
        if not local.exists():
            temporary = local.with_suffix(local.suffix + ".download")
            if temporary.exists():
                temporary.unlink()
            downloader(f"{base}/{commit}/{source_path}", temporary)
            actual = sha256_file(temporary)
            if actual != expected:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"downloaded pinned-data hash mismatch for {source_path}: {actual}"
                )
            os.replace(temporary, local)
        frame = pd.read_csv(local, low_memory=False)
        sources.append(
            PinnedSource(
                path=source_path,
                sha256=expected,
                local_path=local,
                row_count=int(len(frame)),
            )
        )
    return tuple(sources)


def _stop_is_false(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: isinstance(value, (bool, np.bool_)) and not bool(value)
        or isinstance(value, str) and value.strip().lower() == "false"
    )


def canonical_candidates(
    frames: Sequence[pd.DataFrame],
    source_paths: Sequence[str],
    *,
    duplicate_rtol: float = 1e-12,
    duplicate_atol: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the frozen row filters and deterministic duplicate policy."""

    if len(frames) != len(source_paths) or not frames:
        raise ValueError("one source path is required per nonempty frame")
    tagged: list[pd.DataFrame] = []
    source_row_counts: dict[str, int] = {}
    for frame, source_path in zip(frames, source_paths, strict=True):
        missing = set(USED_COLUMNS).difference(frame.columns)
        if missing:
            raise PreprocessingInvalid(
                f"source {source_path} lacks required columns: {sorted(missing)}"
            )
        copy = frame.loc[:, USED_COLUMNS].copy()
        copy["source_path"] = source_path
        copy["source_row"] = np.arange(len(copy), dtype=np.int64)
        tagged.append(copy)
        source_row_counts[source_path] = int(len(copy))
    union = pd.concat(tagged, ignore_index=True)
    mask = np.ones(len(union), dtype=bool)
    filtering: dict[str, int] = {"union_rows": int(len(union))}

    stop_false = _stop_is_false(union["Stop"]).to_numpy(dtype=bool)
    filtering["rows_with_stop_false"] = int(stop_false.sum())
    mask &= stop_false
    filtering["after_stop_filter"] = int(mask.sum())

    present = (
        union["Paratope"].notna()
        & union["Paratope"].astype(str).str.len().gt(0)
    ).to_numpy(dtype=bool)
    filtering["rows_with_present_paratope"] = int(present.sum())
    mask &= present
    filtering["after_paratope_filter"] = int(mask.sum())

    numeric_columns = ("SH_Average_bc",) + tuple(
        column for assay in ASSAYS for column in REPLICATE_COLUMNS[assay]
    )
    numeric: dict[str, FloatArray] = {}
    for column in numeric_columns:
        converted = pd.to_numeric(union[column], errors="coerce").to_numpy(float)
        numeric[column] = converted
        finite = np.isfinite(converted)
        filtering[f"rows_with_finite_{column}"] = int(finite.sum())
        mask &= finite
        filtering[f"after_{column}_filter"] = int(mask.sum())

    retained = union.loc[mask].copy()
    for column in numeric_columns:
        retained[column] = numeric[column][mask]
    retained["Paratope"] = retained["Paratope"].astype(str)
    rows_before_duplicates = int(len(retained))

    kept_rows: list[pd.Series] = []
    duplicate_groups = 0
    for _, group in retained.groupby("Paratope", sort=True, dropna=False):
        first = group.iloc[0]
        if len(group) > 1:
            duplicate_groups += 1
            for column in numeric_columns:
                if not np.allclose(
                    group[column].to_numpy(float),
                    float(first[column]),
                    rtol=duplicate_rtol,
                    atol=duplicate_atol,
                ):
                    raise PreprocessingAmbiguity(
                        f"duplicate Paratope disagrees in {column}: {first['Paratope']}"
                    )
            if not bool((_stop_is_false(group["Stop"])).all()):
                raise PreprocessingAmbiguity("duplicate Stop values disagree")
        kept_rows.append(first)
    candidates = pd.DataFrame(kept_rows, columns=retained.columns)
    candidates = candidates.sort_values("Paratope", kind="stable").reset_index(drop=True)
    candidates.insert(0, "candidate_index", np.arange(len(candidates), dtype=np.int64))
    lengths = candidates["Paratope"].str.len().to_numpy(dtype=np.int64)
    if lengths.size == 0 or np.unique(lengths).size != 1:
        raise PreprocessingInvalid("retained Paratope strings do not have one length")
    summary = {
        "source_row_counts": source_row_counts,
        "filtering_counts": filtering,
        "retained_rows_before_duplicates": rows_before_duplicates,
        "duplicate_row_count": rows_before_duplicates - int(len(candidates)),
        "duplicate_group_count": duplicate_groups,
        "canonical_candidate_count_before_component_rule": int(len(candidates)),
        "sequence_length": int(lengths[0]),
        "sequence_length_check": True,
    }
    return candidates, summary


def hamming_knn_graph(paratopes: Sequence[str], k: int = 8) -> HammingGraph:
    """Build the deterministic union-of-directed-kNN Hamming graph."""

    sequences = np.asarray(tuple(str(value) for value in paratopes), dtype=str)
    n_actions = int(sequences.size)
    if n_actions <= k or k < 1:
        raise PreprocessingInvalid("candidate count must exceed positive k")
    lengths = np.char.str_len(sequences)
    if np.unique(lengths).size != 1:
        raise PreprocessingInvalid("Hamming graph requires equal sequence lengths")
    encoded = np.frombuffer("".join(sequences.tolist()).encode("utf-8"), dtype="S1")
    encoded = encoded.reshape(n_actions, int(lengths[0]))
    indices = np.arange(n_actions, dtype=np.int64)
    directed: list[tuple[int, int]] = []
    for source in range(n_actions):
        distances = np.count_nonzero(encoded != encoded[source], axis=1)
        eligible = indices != source
        ordered = np.lexsort(
            (indices[eligible], sequences[eligible], distances[eligible])
        )
        neighbors = indices[eligible][ordered[:k]]
        directed.extend((source, int(neighbor)) for neighbor in neighbors)
    edges = np.asarray(
        sorted({tuple(sorted(edge)) for edge in directed}), dtype=np.int64
    ).reshape(-1, 2)

    adjacency: list[list[int]] = [[] for _ in range(n_actions)]
    for left, right in edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    unseen = set(range(n_actions))
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        stack = [start]
        unseen.remove(start)
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in sorted(adjacency[node], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda values: (-len(values), values[0]))
    sizes = tuple(len(values) for values in components)
    largest = np.asarray(components[0], dtype=np.int64)
    return HammingGraph(
        edges=edges,
        component_sizes=sizes,
        retained_indices=largest,
        largest_component_fraction=float(largest.size / n_actions),
    )


def apply_component_rule(
    candidates: pd.DataFrame,
    graph: HammingGraph,
    *,
    minimum_largest_fraction: float = 0.90,
) -> tuple[pd.DataFrame, IntArray, bool]:
    """Keep a connected graph or deterministically retain its largest component."""

    if len(graph.component_sizes) == 1:
        return candidates.copy(), graph.edges.copy(), False
    if graph.largest_component_fraction < minimum_largest_fraction:
        raise PreprocessingInvalid("largest Hamming component contains under 90%")
    retained_old = graph.retained_indices
    subset = candidates.iloc[retained_old].copy()
    subset = subset.sort_values("Paratope", kind="stable").reset_index(drop=True)
    old_indices = subset["candidate_index"].to_numpy(dtype=np.int64)
    remap = {int(old): new for new, old in enumerate(old_indices)}
    retained_set = set(remap)
    edges = np.asarray(
        sorted(
            (remap[int(left)], remap[int(right)])
            for left, right in graph.edges
            if int(left) in retained_set and int(right) in retained_set
        ),
        dtype=np.int64,
    ).reshape(-1, 2)
    subset["candidate_index"] = np.arange(len(subset), dtype=np.int64)
    return subset, edges, True


def preference_vote(left: ArrayLike, right: ArrayLike) -> int | None:
    """Return +1/-1 for a strict two-of-three vote, else abstain."""

    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    if differences.shape != (3,) or not np.all(np.isfinite(differences)):
        raise ValueError("each preference vote requires three finite replicates")
    if int(np.count_nonzero(differences > 0.0)) >= 2:
        return 1
    if int(np.count_nonzero(differences < 0.0)) >= 2:
        return -1
    return None


def build_preference_bank(
    paratopes: Sequence[str],
    assay_scores: Mapping[str, ArrayLike],
    edges: ArrayLike,
) -> pd.DataFrame:
    """Construct canonical yield-blind Sort1/Sort8 factors."""

    sequences = tuple(str(value) for value in paratopes)
    edge_array = np.asarray(
        sorted(
            {
                tuple(sorted((int(left), int(right))))
                for left, right in np.asarray(edges, dtype=np.int64)
            }
        ),
        dtype=np.int64,
    ).reshape(-1, 2)
    scores = {assay: np.asarray(assay_scores[assay], dtype=float) for assay in ASSAYS}
    for assay in ASSAYS:
        if scores[assay].shape != (len(sequences), 3):
            raise ValueError(f"{assay} scores must have shape (N, 3)")
    rows: list[dict[str, Any]] = []
    for left, right in edge_array:
        a, b = int(min(left, right)), int(max(left, right))
        for assay in ASSAYS:
            sign = preference_vote(scores[assay][a], scores[assay][b])
            if sign is not None:
                rows.append(
                    {
                        "factor_index": len(rows),
                        "left_action_index": a,
                        "right_action_index": b,
                        "left_paratope": sequences[a],
                        "right_paratope": sequences[b],
                        "assay": assay,
                        "preference_sign": sign,
                        "provenance": f"{assay}_strict_2_of_3_replicate_vote",
                    }
                )
    return pd.DataFrame(rows)


def normalized_laplacian_precision(n_actions: int, edges: ArrayLike) -> FloatArray:
    """Return Q0 = I + L_sym for the unweighted frozen graph."""

    edge_array = np.asarray(edges, dtype=np.int64)
    adjacency = np.zeros((n_actions, n_actions), dtype=float)
    for left, right in edge_array:
        adjacency[int(left), int(right)] = 1.0
        adjacency[int(right), int(left)] = 1.0
    degrees = adjacency.sum(axis=1)
    if np.any(degrees <= 0.0):
        raise PreprocessingInvalid("normalized Laplacian has an isolated vertex")
    inverse_sqrt = 1.0 / np.sqrt(degrees)
    normalized_adjacency = inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    return np.asarray(2.0 * np.eye(n_actions) - normalized_adjacency, dtype=float)


def fit_target_scaling(initial_targets: ArrayLike) -> TargetScaling:
    values = np.asarray(initial_targets, dtype=float)
    if values.shape != (5,) or not np.all(np.isfinite(values)):
        raise ValueError("target scaling requires exactly five finite observations")
    return TargetScaling(
        mean=float(np.mean(values)),
        standard_deviation=max(float(np.std(values, ddof=1)), 1e-6),
    )


def graph_gaussian_posterior(
    prior_precision: ArrayLike,
    observed_indices: Sequence[int],
    standardized_observations: ArrayLike,
    observation_noise_standard_deviation: float = 0.05,
) -> GPPosterior:
    """Condition the graph-Gaussian reference on unique scalar observations."""

    precision = np.asarray(prior_precision, dtype=float).copy()
    indices = np.asarray(tuple(observed_indices), dtype=np.int64)
    observations = np.asarray(standardized_observations, dtype=float)
    if indices.shape != observations.shape or np.unique(indices).size != indices.size:
        raise ValueError("unique observed indices must match observations")
    if observation_noise_standard_deviation <= 0.0:
        raise ValueError("observation noise must be positive")
    natural = np.zeros(precision.shape[0], dtype=float)
    variance = observation_noise_standard_deviation**2
    precision[indices, indices] += 1.0 / variance
    natural[indices] += observations / variance
    precision = 0.5 * (precision + precision.T)
    mean = np.linalg.solve(precision, natural)
    covariance = np.linalg.solve(precision, np.eye(precision.shape[0]))
    covariance = 0.5 * (covariance + covariance.T)
    return GPPosterior(mean=mean, covariance=covariance, precision=precision)


def initial_design(n_actions: int, seed: int, count: int = 5) -> IntArray:
    if n_actions < count:
        raise ValueError("initial design exceeds the action set")
    return np.asarray(
        np.random.default_rng(int(seed)).choice(n_actions, size=count, replace=False),
        dtype=np.int64,
    )


def inference_rng(
    seed: int, bo_iteration: int, draw_stage_index: int, split_half: int
) -> np.random.Generator:
    """Create independent deterministic inference streams."""

    if split_half not in (0, 1):
        raise ValueError("split_half must be zero or one")
    sequence = np.random.SeedSequence(
        [271828, int(seed), int(bo_iteration), int(draw_stage_index), split_half]
    )
    return np.random.default_rng(sequence)


def _preference_energy_chunked(
    samples: FloatArray,
    endpoint_pairs: IntArray,
    signs: IntArray,
    temperature: float,
    factor_chunk_size: int,
) -> FloatArray:
    energy = np.zeros(samples.shape[0], dtype=float)
    for start in range(0, endpoint_pairs.shape[0], factor_chunk_size):
        selected = endpoint_pairs[start : start + factor_chunk_size]
        selected_signs = signs[start : start + factor_chunk_size]
        margins = (
            selected_signs[None, :]
            * (samples[:, selected[:, 0]] - samples[:, selected[:, 1]])
            / temperature
        )
        energy += np.logaddexp(0.0, -margins).sum(axis=1)
    return energy


def _importance_half(
    *,
    posterior: GPPosterior,
    endpoint_pairs: IntArray,
    signs: IntArray,
    temperature: float,
    incumbent: float,
    draws: int,
    rng: np.random.Generator,
    laplace: LaplaceResult,
    draw_chunk_size: int,
    factor_chunk_size: int,
) -> _WeightedAccumulator:
    accumulator = _WeightedAccumulator(posterior.mean.size)
    cholesky = np.linalg.cholesky(laplace.hessian)
    remaining = draws
    while remaining:
        chunk = min(draw_chunk_size, remaining)
        standard = rng.standard_normal((chunk, posterior.mean.size))
        deviations = solve_triangular(
            cholesky.T, standard.T, lower=False, check_finite=False
        ).T
        samples = laplace.mode + deviations
        centered = samples - posterior.mean
        gaussian_energy = 0.5 * np.einsum(
            "ni,ij,nj->n", centered, posterior.precision, centered, optimize=True
        )
        preference_energy = _preference_energy_chunked(
            samples, endpoint_pairs, signs, temperature, factor_chunk_size
        )
        proposal_energy = 0.5 * np.square(standard).sum(axis=1)
        log_weights = -gaussian_energy - preference_energy + proposal_energy
        utilities = np.maximum(samples - incumbent, 0.0)
        accumulator.add(log_weights, utilities)
        remaining -= chunk
    return accumulator


def streamed_laplace_is_ei(
    *,
    posterior: GPPosterior,
    endpoint_pairs: ArrayLike,
    signs: ArrayLike,
    temperature: float,
    incumbent: float,
    observed_indices: Sequence[int],
    draw_schedule: Sequence[int],
    seed: int,
    bo_iteration: int,
    laplace_settings: Mapping[str, Any],
    minimum_ess_fraction: float = 0.20,
    maximum_split_half_discrepancy: float = 0.01,
    draw_chunk_size: int = 512,
    factor_chunk_size: int = 128,
) -> StreamedInferenceResult:
    """Run frozen staged, split-stream, chunked Laplace-preconditioned IS."""

    started = time.perf_counter()
    pairs = np.asarray(endpoint_pairs, dtype=np.int64)
    sign_values = np.asarray(signs, dtype=np.int64)
    active = np.arange(pairs.shape[0], dtype=np.int64)
    laplace = laplace_preference_mode(
        posterior, pairs, sign_values, temperature, active, laplace_settings
    )
    if not laplace.converged:
        raise RuntimeError("Laplace mode failed the frozen gradient tolerance")
    eligible = np.ones(posterior.mean.size, dtype=bool)
    eligible[np.asarray(tuple(observed_indices), dtype=np.int64)] = False
    total_factor_evaluations = int(laplace.factor_likelihood_evaluations)
    final: StreamedInferenceResult | None = None
    for stage_index, draws_value in enumerate(draw_schedule):
        draws = int(draws_value)
        if draws < 2 or draws % 2:
            raise ValueError("each total draw count must be positive and even")
        halves = [
            _importance_half(
                posterior=posterior,
                endpoint_pairs=pairs,
                signs=sign_values,
                temperature=temperature,
                incumbent=incumbent,
                draws=draws // 2,
                rng=inference_rng(seed, bo_iteration, stage_index, split_half),
                laplace=laplace,
                draw_chunk_size=draw_chunk_size,
                factor_chunk_size=factor_chunk_size,
            )
            for split_half in (0, 1)
        ]
        if any(half.acquisition is None for half in halves):
            raise RuntimeError("empty importance half")
        log_total = float(
            np.logaddexp(halves[0].log_weight_sum, halves[1].log_weight_sum)
        )
        acquisition = np.asarray(
            np.exp(halves[0].log_weight_sum - log_total) * halves[0].acquisition
            + np.exp(halves[1].log_weight_sum - log_total) * halves[1].acquisition,
            dtype=float,
        )
        log_squared_total = float(
            np.logaddexp(
                halves[0].log_squared_weight_sum,
                halves[1].log_squared_weight_sum,
            )
        )
        ess = min(float(draws), float(np.exp(2.0 * log_total - log_squared_total)))
        discrepancy = float(
            np.max(np.abs(halves[0].acquisition[eligible] - halves[1].acquisition[eligible]))
        )
        total_factor_evaluations += draws * pairs.shape[0]
        reliable = (
            ess / draws >= minimum_ess_fraction
            and discrepancy <= maximum_split_half_discrepancy
        )
        final = StreamedInferenceResult(
            acquisition=acquisition,
            split_half_acquisition_1=np.asarray(halves[0].acquisition, dtype=float),
            split_half_acquisition_2=np.asarray(halves[1].acquisition, dtype=float),
            draws=draws,
            draw_stage_index=stage_index,
            ess=ess,
            ess_fraction=ess / draws,
            maximum_split_half_discrepancy=discrepancy,
            laplace=laplace,
            factor_likelihood_evaluations=total_factor_evaluations,
            wall_time_seconds=time.perf_counter() - started,
            reliable=reliable,
        )
        if reliable:
            return final
    assert final is not None
    return final


def prepare_gp2_data(
    config: Mapping[str, Any], repository_root: str | Path
) -> PreparedGp2Data:
    sources = ensure_pinned_sources(config, repository_root)
    frames = [pd.read_csv(source.local_path, low_memory=False) for source in sources]
    candidates, summary = canonical_candidates(
        frames,
        [source.path for source in sources],
        duplicate_rtol=float(config["preprocessing"]["duplicate_rtol"]),
        duplicate_atol=float(config["preprocessing"]["duplicate_atol"]),
    )
    graph = hamming_knn_graph(candidates["Paratope"].tolist(), int(config["graph"]["k"]))
    final_candidates, final_edges, component_was_restricted = apply_component_rule(
        candidates,
        graph,
        minimum_largest_fraction=float(
            config["preprocessing"]["minimum_largest_component_fraction"]
        ),
    )
    assay_scores = {
        assay: final_candidates.loc[:, REPLICATE_COLUMNS[assay]].to_numpy(float)
        for assay in ASSAYS
    }
    factors = build_preference_bank(
        final_candidates["Paratope"].tolist(), assay_scores, final_edges
    )
    endpoint_degrees = np.zeros(len(final_candidates), dtype=np.int64)
    for left, right in factors[["left_action_index", "right_action_index"]].to_numpy(int):
        endpoint_degrees[left] += 1
        endpoint_degrees[right] += 1
    factor_counts = {
        assay: int((factors["assay"] == assay).sum()) for assay in ASSAYS
    }
    thresholds = config["preprocessing"]
    criteria = {
        "candidate_count_at_least_250": len(final_candidates)
        >= int(thresholds["minimum_action_count"]),
        "largest_component_at_least_90_percent": graph.largest_component_fraction
        >= float(thresholds["minimum_largest_component_fraction"]),
        "factor_count_at_least_1000": len(factors)
        >= int(thresholds["minimum_factor_count"]),
        "factor_provenance_sort1_sort8_only": set(factors["assay"]).issubset(ASSAYS)
        and factors["provenance"].str.endswith("strict_2_of_3_replicate_vote").all(),
        "graph_factor_generation_target_blind_by_interface": True,
    }
    summary.update(
        {
            "final_candidate_count": int(len(final_candidates)),
            "graph_k": int(config["graph"]["k"]),
            "graph_edge_count": int(len(final_edges)),
            "graph_component_count_before_rule": len(graph.component_sizes),
            "graph_component_sizes_before_rule": list(graph.component_sizes),
            "largest_component_fraction": graph.largest_component_fraction,
            "component_was_restricted": component_was_restricted,
            "factor_count_by_assay": factor_counts,
            "total_factor_count": int(len(factors)),
            "endpoint_factor_degree_summary": {
                "minimum": int(endpoint_degrees.min()),
                "median": float(np.median(endpoint_degrees)),
                "maximum": int(endpoint_degrees.max()),
                "mean": float(endpoint_degrees.mean()),
            },
            "preprocessing_criteria": {key: bool(value) for key, value in criteria.items()},
            "verdict": "PREPROCESSING_VALID" if all(criteria.values()) else "PREPROCESSING_INVALID",
        }
    )
    return PreparedGp2Data(
        candidates=final_candidates,
        edges=final_edges,
        factors=factors,
        preprocessing_summary=summary,
        source_provenance=sources,
    )


def _initial_observation_hash(indices: IntArray, targets: FloatArray) -> str:
    payload = json.dumps(
        {"indices": indices.tolist(), "targets": targets.tolist()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _run_method(
    *,
    prepared: PreparedGp2Data,
    config: Mapping[str, Any],
    config_hash: str,
    seed: int,
    method: str,
    horizon: int,
    draw_schedule: Sequence[int],
    profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    targets = prepared.candidates["SH_Average_bc"].to_numpy(float)
    initial = initial_design(len(targets), seed, int(config["bo"]["initial_action_count"]))
    scaling = fit_target_scaling(targets[initial])
    observed = initial.tolist()
    observed_raw = targets[initial].tolist()
    initial_hash = _initial_observation_hash(initial, targets[initial])
    q0 = normalized_laplacian_precision(len(targets), prepared.edges)
    factor_pairs = prepared.factors[["left_action_index", "right_action_index"]].to_numpy(int)
    factor_signs = prepared.factors["preference_sign"].to_numpy(int)
    trajectory: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    reliable = True
    global_range = float(targets.max() - targets.min())
    top_threshold = float(np.quantile(targets, 0.90))
    for iteration in range(1, horizon + 1):
        standardized = (np.asarray(observed_raw) - scaling.mean) / scaling.standard_deviation
        posterior = graph_gaussian_posterior(
            q0,
            observed,
            standardized,
            float(config["reference"]["observation_noise_standard_deviation"]),
        )
        incumbent = float(np.max(standardized))
        if method == "scalar_only":
            acquisition = analytic_expected_improvement(
                posterior.mean, np.diag(posterior.covariance), incumbent
            )
        elif method == "full_preference":
            inference = streamed_laplace_is_ei(
                posterior=posterior,
                endpoint_pairs=factor_pairs,
                signs=factor_signs,
                temperature=float(config["preference"]["tau_pref"]),
                incumbent=incumbent,
                observed_indices=observed,
                draw_schedule=draw_schedule,
                seed=seed,
                bo_iteration=iteration,
                laplace_settings=config["laplace"],
                minimum_ess_fraction=float(config["inference"]["minimum_ess_fraction"]),
                maximum_split_half_discrepancy=float(
                    config["inference"]["maximum_split_half_ei_discrepancy"]
                ),
                draw_chunk_size=int(config["inference"]["draw_chunk_size"]),
                factor_chunk_size=int(config["inference"]["factor_chunk_size"]),
            )
            acquisition = inference.acquisition
            diagnostics.append(
                {
                    "profile": profile,
                    "seed": seed,
                    "method": method,
                    "bo_iteration": iteration,
                    "draws": inference.draws,
                    "draw_stage_index": inference.draw_stage_index,
                    "ess": inference.ess,
                    "ess_fraction": inference.ess_fraction,
                    "maximum_split_half_ei_discrepancy": inference.maximum_split_half_discrepancy,
                    "laplace_iterations": inference.laplace.iterations,
                    "laplace_gradient_infinity_norm": inference.laplace.gradient_infinity_norm,
                    "factor_likelihood_evaluation_count": inference.factor_likelihood_evaluations,
                    "wall_time_seconds": inference.wall_time_seconds,
                    "numerically_reliable": inference.reliable,
                    "config_sha256": config_hash,
                }
            )
            if not inference.reliable:
                reliable = False
                break
        else:
            raise ValueError(f"unknown method: {method}")
        selected = select_unobserved_argmax(acquisition, observed)
        observed.append(selected)
        observed_raw.append(float(targets[selected]))
        best = float(np.max(observed_raw))
        trajectory.append(
            {
                "profile": profile,
                "seed": seed,
                "method": method,
                "bo_iteration": iteration,
                "initial_action_indices": json.dumps(initial.tolist()),
                "initial_observation_sha256": initial_hash,
                "selected_action_index": selected,
                "selected_paratope": str(prepared.candidates.iloc[selected]["Paratope"]),
                "observed_target": float(targets[selected]),
                "best_target_observed": best,
                "normalized_simple_regret": float((targets.max() - best) / global_range),
                "top_10_percent_hit": bool(best >= top_threshold),
                "config_sha256": config_hash,
            }
        )
    return trajectory, diagnostics, reliable


def evaluate_p1_gate(per_seed_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scalar = np.asarray(
        [row["r_5"] for row in per_seed_rows if row["method"] == "scalar_only"],
        dtype=float,
    )
    preference = np.asarray(
        [row["r_5"] for row in per_seed_rows if row["method"] == "full_preference"],
        dtype=float,
    )
    if scalar.size != preference.size or scalar.size == 0:
        raise ValueError("paired scalar/full r_5 values are required")
    median_scalar = float(np.median(scalar))
    median_preference = float(np.median(preference))
    if median_scalar == 0.0:
        verdict = "GATE_UNINFORMATIVE_SCALAR_CEILING"
        improvement = None
    else:
        improvement = float(1.0 - median_preference / median_scalar)
        verdict = "PASS" if improvement >= 0.25 else "FAIL_P1"
    return {
        "median_r_5_scalar_only": median_scalar,
        "median_r_5_full_preference": median_preference,
        "relative_median_regret_improvement": improvement,
        "pass_threshold": 0.25,
        "verdict": verdict,
    }


def run_gate(
    prepared: PreparedGp2Data,
    config: Mapping[str, Any],
    *,
    config_hash: str,
    smoke: bool,
    progress: Callable[[str], None] | None = None,
) -> GateRunResult:
    """Run the mechanical smoke or the frozen 20-seed scientific P1 gate."""

    if prepared.preprocessing_summary["verdict"] != "PREPROCESSING_VALID":
        raise PreprocessingInvalid("scientific/smoke execution requires valid preprocessing")
    seeds = [int(config["smoke"]["seed"])] if smoke else list(config["bo"]["seeds"])
    horizon = int(config["smoke"]["post_initial_horizon"] if smoke else config["bo"]["post_initial_horizon"])
    draw_schedule = config["inference"]["smoke_draw_schedule" if smoke else "scientific_draw_schedule"]
    profile = "smoke" if smoke else "scientific"
    trajectories: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    all_reliable = True
    for seed in seeds:
        for method in config["bo"]["methods"]:
            if progress is not None:
                progress(f"seed={seed} method={method}")
            rows, inference_rows, reliable = _run_method(
                prepared=prepared,
                config=config,
                config_hash=config_hash,
                seed=seed,
                method=str(method),
                horizon=horizon,
                draw_schedule=draw_schedule,
                profile=profile,
            )
            trajectories.extend(rows)
            diagnostics.extend(inference_rows)
            all_reliable &= reliable
            if not reliable:
                break
        if not all_reliable:
            break
    per_seed: list[dict[str, Any]] = []
    if all_reliable:
        targets = prepared.candidates["SH_Average_bc"].to_numpy(float)
        denominator = float(targets.max() - targets.min())
        for seed in seeds:
            for method in config["bo"]["methods"]:
                rows = sorted(
                    [row for row in trajectories if row["seed"] == seed and row["method"] == method],
                    key=lambda row: row["bo_iteration"],
                )
                if len(rows) != horizon:
                    raise RuntimeError("a complete method trajectory is missing")
                initial = np.asarray(json.loads(rows[0]["initial_action_indices"]), dtype=int)
                selected = np.asarray([row["selected_action_index"] for row in rows], dtype=int)
                best = float(targets[np.concatenate((initial, selected))].max())
                per_seed.append(
                    {
                        "seed": seed,
                        "method": method,
                        "r_5": float((targets.max() - best) / denominator),
                        "selected_action_indices": selected.tolist(),
                        "initial_action_indices": initial.tolist(),
                    }
                )
    if smoke:
        verdict = "SMOKE_PASS" if all_reliable else "IMPLEMENTATION_BLOCKED"
        gate_summary = None
    elif not all_reliable:
        verdict = "INCONCLUSIVE_NUMERICAL"
        gate_summary = None
    else:
        gate_summary = evaluate_p1_gate(per_seed)
        verdict = str(gate_summary["verdict"])
    validate_output_schema(trajectories, diagnostics, allow_partial=not all_reliable)
    return GateRunResult(
        trajectory_rows=trajectories,
        inference_rows=diagnostics,
        per_seed_rows=per_seed,
        verdict=verdict,
        gate_summary=gate_summary,
        numerically_reliable=all_reliable,
    )


def validate_output_schema(
    trajectory_rows: Sequence[Mapping[str, Any]],
    inference_rows: Sequence[Mapping[str, Any]],
    *,
    allow_partial: bool = False,
) -> None:
    if not trajectory_rows and not allow_partial:
        raise ValueError("trajectory output is empty")
    seen: dict[tuple[int, str], set[int]] = {}
    initial_hashes: dict[int, set[str]] = {}
    for row in trajectory_rows:
        missing = TRAJECTORY_FIELDS.difference(row)
        if missing:
            raise ValueError(f"trajectory fields missing: {sorted(missing)}")
        key = (int(row["seed"]), str(row["method"]))
        selected = int(row["selected_action_index"])
        if selected in seen.setdefault(key, set()):
            raise ValueError("a BO method repeated a post-initial action")
        initial = set(json.loads(str(row["initial_action_indices"])))
        if selected in initial:
            raise ValueError("a BO method repeated an initial action")
        seen[key].add(selected)
        initial_hashes.setdefault(int(row["seed"]), set()).add(
            str(row["initial_observation_sha256"])
        )
    if any(len(values) != 1 for values in initial_hashes.values()):
        raise ValueError("methods did not share identical initial observations")
    for row in inference_rows:
        missing = INFERENCE_FIELDS.difference(row)
        if missing:
            raise ValueError(f"inference fields missing: {sorted(missing)}")


def create_immutable_output_directory(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)
    return destination


__all__ = [
    "ASSAYS",
    "GateRunResult",
    "HammingGraph",
    "INFERENCE_FIELDS",
    "PinnedSource",
    "PreparedGp2Data",
    "PreprocessingAmbiguity",
    "PreprocessingInvalid",
    "REPLICATE_COLUMNS",
    "TRAJECTORY_FIELDS",
    "TargetScaling",
    "apply_component_rule",
    "build_preference_bank",
    "canonical_candidates",
    "config_sha256",
    "create_immutable_output_directory",
    "ensure_pinned_sources",
    "evaluate_p1_gate",
    "fit_target_scaling",
    "graph_gaussian_posterior",
    "hamming_knn_graph",
    "inference_rng",
    "initial_design",
    "load_gate_config",
    "normalized_laplacian_precision",
    "preference_vote",
    "prepare_gp2_data",
    "run_gate",
    "sha256_file",
    "streamed_laplace_is_ei",
    "validate_output_schema",
]
