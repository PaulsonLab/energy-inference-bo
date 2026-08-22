"""Target-blind structural preflight for the Gp2 proxy-conditioned E3 case.

The historical table is used only to standardize yield and calibrate the two
frozen proxy assays.  Prospective target magnitudes are consumed only by the
finite-value filter and duplicate-consistency check; they are deliberately
absent from every action, graph, factor, theory, and structural interface.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any
from urllib.request import urlopen

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import sklearn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

HISTORICAL_SOURCE_PATH = "datasets/assay_to_yield_training_sequences.csv"
PROSPECTIVE_SOURCE_PATH = "datasets/test_sequences.csv"
TARGET_COLUMN = "SH_Average_bc"
PROXY_COLUMNS = ("Sort1_mean_score", "Sort8_mean_score")
BASE_COLUMNS = ("Stop", "Paratope", TARGET_COLUMN, *PROXY_COLUMNS)


class PreflightFailure(RuntimeError):
    """Base class for frozen terminal outcomes before the sparsity verdict."""

    verdict = "PREPROCESSING_INVALID"

    def __init__(self, reason: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(reason)
        self.reason = reason
        self.details = dict(details or {})


class PreprocessingAmbiguity(PreflightFailure):
    verdict = "PREPROCESSING_AMBIGUITY"


class PreprocessingInvalid(PreflightFailure):
    verdict = "PREPROCESSING_INVALID"


class TheoryInstantiationInvalid(PreflightFailure):
    verdict = "THEORY_INSTANTIATION_INVALID"


@dataclass(frozen=True)
class PinnedSource:
    path: str
    sha256: str
    local_path: Path
    row_count: int


@dataclass(frozen=True)
class HistoricalRows:
    rows: pd.DataFrame
    paratopes: frozenset[str]
    sequence_length: int
    summary: dict[str, Any]


@dataclass(frozen=True)
class ProxyCalibration:
    pipeline: Pipeline
    mu_hist: float
    s_hist: float
    oof_rmse: float
    s_proxy: float
    historical_target_scale_count: int
    calibration_count: int
    summary: dict[str, Any]


@dataclass(frozen=True)
class HammingGraph:
    edges: IntArray
    component_sizes: tuple[int, ...]
    largest_component_indices: IntArray
    largest_component_fraction: float


@dataclass(frozen=True)
class TheoryConstruction:
    q0: FloatArray
    a: FloatArray
    k0: FloatArray
    c: FloatArray
    graph_distances: FloatArray
    summary: dict[str, Any]


@dataclass(frozen=True)
class PreparedStructuralPreflight:
    actions: pd.DataFrame
    edges: IntArray
    factors: pd.DataFrame
    calibration: ProxyCalibration
    theory: TheoryConstruction
    preprocessing_summary: dict[str, Any]
    graph_summary: dict[str, Any]
    source_provenance: tuple[PinnedSource, ...]


@dataclass(frozen=True)
class StructuralSparsityResult:
    pairwise: pd.DataFrame
    summary: dict[str, Any]
    verdict: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def ensure_pinned_sources(
    config: Mapping[str, Any],
    repository_root: str | Path,
    *,
    downloader: Callable[[str, Path], None] = _download,
) -> tuple[PinnedSource, ...]:
    """Materialize and verify the two source-role-pinned DevRep tables."""

    root = Path(repository_root)
    external = config["external_data"]
    cache = root / str(external["cache_directory"])
    cache.mkdir(parents=True, exist_ok=True)
    commit = str(external["commit"])
    raw_base = str(external["raw_base_url"]).rstrip("/")
    sources: list[PinnedSource] = []
    for item in external["files"]:
        source_path = str(item["path"])
        expected_hash = str(item["sha256"])
        local_path = cache / Path(source_path).name
        if local_path.exists() and sha256_file(local_path) != expected_hash:
            raise PreprocessingInvalid(
                f"cached pinned-data hash mismatch for {source_path}"
            )
        if not local_path.exists():
            temporary = local_path.with_suffix(local_path.suffix + ".download")
            temporary.unlink(missing_ok=True)
            downloader(f"{raw_base}/{commit}/{source_path}", temporary)
            actual_hash = sha256_file(temporary)
            if actual_hash != expected_hash:
                temporary.unlink(missing_ok=True)
                raise PreprocessingInvalid(
                    f"downloaded pinned-data hash mismatch for {source_path}"
                )
            os.replace(temporary, local_path)
        row_count = int(len(pd.read_csv(local_path, low_memory=False)))
        sources.append(
            PinnedSource(source_path, expected_hash, local_path, row_count)
        )
    return tuple(sources)


def load_structural_preflight_config(path: str | Path) -> dict[str, Any]:
    """Load the config and reject deviations from the handoff-frozen science."""

    import json

    config = json.loads(Path(path).read_text())
    if config.get("status") != "preregistered_frozen":
        raise ValueError("configuration is not marked preregistered_frozen")
    external = config["external_data"]
    if external["repository"] != "HackelLab-UMN/DevRep":
        raise ValueError("external repository differs from the frozen source")
    if external["commit"] != "e05023a8abe7be6c2e22f42d523b20bd76cd8da5":
        raise ValueError("external commit differs from the frozen commit")
    if [item["path"] for item in external["files"]] != [
        HISTORICAL_SOURCE_PATH,
        PROSPECTIVE_SOURCE_PATH,
    ]:
        raise ValueError("historical and prospective source roles are not frozen")
    if config["source_roles"] != {
        "historical_calibration": HISTORICAL_SOURCE_PATH,
        "prospective_actions": PROSPECTIVE_SOURCE_PATH,
        "union_sources_for_actions": False,
    }:
        raise ValueError("source-role separation differs from the handoff")
    if config["columns"]["target"] != TARGET_COLUMN:
        raise ValueError("target column differs from the handoff")
    if config["columns"]["proxies"] != list(PROXY_COLUMNS):
        raise ValueError("exactly Sort1_mean_score and Sort8_mean_score are required")
    calibration = config["calibration"]
    if calibration["pipeline"] != ["StandardScaler", "Ridge"]:
        raise ValueError("calibration pipeline differs from StandardScaler + Ridge")
    if float(calibration["ridge_alpha"]) != 1.0:
        raise ValueError("Ridge alpha must equal 1.0")
    if calibration["features_in_order"] != list(PROXY_COLUMNS):
        raise ValueError("calibration feature order differs from the handoff")
    if calibration["kfold"] != {
        "n_splits": 10,
        "shuffle": True,
        "random_state": 0,
    }:
        raise ValueError("OOF KFold differs from the handoff")
    if int(config["graph"]["k"]) != 8:
        raise ValueError("Hamming graph k must equal eight")
    if config["reference"]["precision"] != "I_plus_symmetric_normalized_laplacian":
        raise ValueError("reference precision differs from Q0 = I + L_sym")
    structural = config["structural_gate"]
    if (
        float(structural["epsilon_struct"]) != 0.05
        or float(structural["maximum_active_factor_fraction"]) != 0.50
        or float(structural["minimum_passing_pair_fraction"]) != 0.90
    ):
        raise ValueError("structural sparsity thresholds differ from the handoff")
    expected_outcomes = [
        "PREPROCESSING_AMBIGUITY",
        "PREPROCESSING_INVALID",
        "THEORY_INSTANTIATION_INVALID",
        "FAIL_STRUCTURAL_SPARSITY",
        "PASS_STRUCTURAL_PREFLIGHT",
    ]
    if config["outcome_precedence"] != expected_outcomes:
        raise ValueError("outcome precedence differs from the handoff")
    return config


def _stop_is_false(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: (
            isinstance(value, (bool, np.bool_)) and not bool(value)
        )
        or (isinstance(value, str) and value.strip().lower() == "false")
    )


def _numeric_group_agrees(
    values: pd.Series, reference: float, *, rtol: float, atol: float
) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return bool(
        np.all(np.isclose(numeric, reference, rtol=rtol, atol=atol, equal_nan=True))
    )


def _require_columns(frame: pd.DataFrame, source_role: str) -> None:
    missing = set(BASE_COLUMNS).difference(frame.columns)
    if missing:
        raise PreprocessingInvalid(
            f"{source_role} source lacks required columns: {sorted(missing)}"
        )


def prepare_historical_rows(
    frame: pd.DataFrame,
    *,
    duplicate_rtol: float = 1e-12,
    duplicate_atol: float = 1e-12,
) -> HistoricalRows:
    """Prepare only the historical target-scale and calibration source."""

    _require_columns(frame, "historical calibration")
    working = frame.loc[:, BASE_COLUMNS].copy()
    working["source_row"] = np.arange(len(working), dtype=np.int64)
    stop_false = _stop_is_false(working["Stop"]).to_numpy(dtype=bool)
    present = (
        working["Paratope"].notna()
        & working["Paratope"].astype(str).str.len().gt(0)
    ).to_numpy(dtype=bool)
    target = pd.to_numeric(working[TARGET_COLUMN], errors="coerce").to_numpy(float)
    target_finite = np.isfinite(target)
    retained_mask = stop_false & present & target_finite
    retained = working.loc[retained_mask].copy()
    retained[TARGET_COLUMN] = target[retained_mask]
    for proxy in PROXY_COLUMNS:
        retained[proxy] = pd.to_numeric(retained[proxy], errors="coerce")
    retained["Paratope"] = retained["Paratope"].astype(str)

    rows_before_duplicates = int(len(retained))
    kept: list[pd.Series] = []
    duplicate_groups = 0
    for paratope, group in retained.groupby("Paratope", sort=True, dropna=False):
        first = group.sort_values("source_row", kind="stable").iloc[0]
        if len(group) > 1:
            duplicate_groups += 1
            for column in (TARGET_COLUMN, *PROXY_COLUMNS):
                if not _numeric_group_agrees(
                    group[column],
                    float(first[column]),
                    rtol=duplicate_rtol,
                    atol=duplicate_atol,
                ):
                    raise PreprocessingAmbiguity(
                        f"historical duplicate Paratope disagrees in {column}: {paratope}"
                    )
        kept.append(first)
    canonical = pd.DataFrame(kept, columns=retained.columns)
    canonical = canonical.sort_values("Paratope", kind="stable").reset_index(drop=True)
    lengths = canonical["Paratope"].str.len().to_numpy(dtype=np.int64)
    if lengths.size == 0 or np.unique(lengths).size != 1:
        raise PreprocessingInvalid(
            "historical target-scale paratopes do not have one common length"
        )
    summary = {
        "source_path": HISTORICAL_SOURCE_PATH,
        "source_row_count": int(len(frame)),
        "rows_with_stop_false": int(stop_false.sum()),
        "rows_with_present_paratope": int(present.sum()),
        "rows_with_finite_target": int(target_finite.sum()),
        "retained_before_duplicates": rows_before_duplicates,
        "historical_target_scale_count": int(len(canonical)),
        "duplicate_group_count": duplicate_groups,
        "duplicate_row_count": rows_before_duplicates - int(len(canonical)),
        "sequence_length": int(lengths[0]),
    }
    return HistoricalRows(
        rows=canonical,
        paratopes=frozenset(canonical["Paratope"].tolist()),
        sequence_length=int(lengths[0]),
        summary=summary,
    )


def _new_calibration_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, fit_intercept=True)),
        ]
    )


def fit_proxy_calibration(
    historical: HistoricalRows,
    *,
    minimum_calibration_count: int = 100,
    minimum_target_standard_deviation: float = 1e-6,
    minimum_proxy_scale: float = 1e-6,
) -> ProxyCalibration:
    """Fit the frozen historical scaling, 10-fold OOF scale, and full pipeline."""

    targets = historical.rows[TARGET_COLUMN].to_numpy(dtype=float)
    mu_hist = float(np.mean(targets))
    raw_scale = float(np.std(targets, ddof=1))
    if not np.isfinite(mu_hist) or not np.isfinite(raw_scale):
        raise PreprocessingInvalid("historical target mean or sample SD is nonfinite")
    s_hist = max(raw_scale, float(minimum_target_standard_deviation))
    proxy_values = historical.rows.loc[:, PROXY_COLUMNS].to_numpy(dtype=float)
    calibration_mask = np.all(np.isfinite(proxy_values), axis=1)
    features = proxy_values[calibration_mask]
    response = (targets[calibration_mask] - mu_hist) / s_hist
    calibration_count = int(features.shape[0])
    if calibration_count < int(minimum_calibration_count):
        raise PreprocessingInvalid(
            f"historical calibration count {calibration_count} is below 100"
        )

    splitter = KFold(n_splits=10, shuffle=True, random_state=0)
    oof_predictions = np.empty(calibration_count, dtype=float)
    for train_indices, validation_indices in splitter.split(features):
        fold_pipeline = _new_calibration_pipeline()
        fold_pipeline.fit(features[train_indices], response[train_indices])
        oof_predictions[validation_indices] = fold_pipeline.predict(
            features[validation_indices]
        )
    oof_rmse = float(np.sqrt(np.mean(np.square(response - oof_predictions))))
    s_proxy = float(np.sqrt(3.0) / np.pi * oof_rmse)
    if not np.isfinite(s_proxy) or s_proxy <= float(minimum_proxy_scale):
        raise PreprocessingInvalid("frozen OOF proxy scale is nonfinite or at most 1e-6")

    full_pipeline = _new_calibration_pipeline()
    full_pipeline.fit(features, response)
    scaler = full_pipeline.named_steps["scale"]
    ridge = full_pipeline.named_steps["ridge"]
    summary = {
        "historical_target_scale_count": int(len(targets)),
        "calibration_count": calibration_count,
        "mu_hist": mu_hist,
        "s_hist": s_hist,
        "target_scale_ddof": 1,
        "target_scale_floor": float(minimum_target_standard_deviation),
        "features_in_order": list(PROXY_COLUMNS),
        "pipeline": "StandardScaler + Ridge(alpha=1.0, fit_intercept=True)",
        "kfold": {"n_splits": 10, "shuffle": True, "random_state": 0},
        "oof_rmse": oof_rmse,
        "s_proxy": s_proxy,
        "standard_scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "standard_scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "ridge_coefficients": np.asarray(ridge.coef_, dtype=float).tolist(),
        "ridge_intercept": float(ridge.intercept_),
        "scikit_learn_version": sklearn.__version__,
    }
    return ProxyCalibration(
        pipeline=full_pipeline,
        mu_hist=mu_hist,
        s_hist=s_hist,
        oof_rmse=oof_rmse,
        s_proxy=s_proxy,
        historical_target_scale_count=int(len(targets)),
        calibration_count=calibration_count,
        summary=summary,
    )


def prepare_prospective_actions(
    frame: pd.DataFrame,
    *,
    historical_paratopes: frozenset[str],
    historical_sequence_length: int,
    duplicate_rtol: float = 1e-12,
    duplicate_atol: float = 1e-12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare test-only actions while dropping held-out target magnitudes."""

    _require_columns(frame, "prospective action")
    working = frame.loc[:, BASE_COLUMNS].copy()
    working["source_row"] = np.arange(len(working), dtype=np.int64)
    stop_false = _stop_is_false(working["Stop"]).to_numpy(dtype=bool)
    present = (
        working["Paratope"].notna()
        & working["Paratope"].astype(str).str.len().gt(0)
    ).to_numpy(dtype=bool)
    heldout_target = pd.to_numeric(
        working[TARGET_COLUMN], errors="coerce"
    ).to_numpy(dtype=float)
    target_finite = np.isfinite(heldout_target)
    retained_mask = stop_false & present & target_finite
    retained = working.loc[retained_mask].copy()
    retained["_target_duplicate_check"] = heldout_target[retained_mask]
    for proxy in PROXY_COLUMNS:
        retained[proxy] = pd.to_numeric(retained[proxy], errors="coerce")
    retained["Paratope"] = retained["Paratope"].astype(str)

    rows_before_duplicates = int(len(retained))
    kept: list[pd.Series] = []
    duplicate_groups = 0
    duplicate_fields = ("_target_duplicate_check", *PROXY_COLUMNS)
    for paratope, group in retained.groupby("Paratope", sort=True, dropna=False):
        first = group.sort_values("source_row", kind="stable").iloc[0]
        if len(group) > 1:
            duplicate_groups += 1
            for column in duplicate_fields:
                if not _numeric_group_agrees(
                    group[column],
                    float(first[column]),
                    rtol=duplicate_rtol,
                    atol=duplicate_atol,
                ):
                    public_column = (
                        TARGET_COLUMN if column == "_target_duplicate_check" else column
                    )
                    raise PreprocessingAmbiguity(
                        f"prospective duplicate Paratope disagrees in {public_column}: {paratope}"
                    )
        kept.append(first)
    canonical_internal = pd.DataFrame(kept, columns=retained.columns)
    canonical_internal = canonical_internal.sort_values(
        "Paratope", kind="stable"
    ).reset_index(drop=True)

    overlaps = sorted(set(canonical_internal["Paratope"]).intersection(historical_paratopes))
    if overlaps:
        raise PreprocessingAmbiguity(
            f"historical/prospective Paratope overlap detected ({len(overlaps)} sequences)"
        )
    lengths = canonical_internal["Paratope"].str.len().to_numpy(dtype=np.int64)
    if (
        lengths.size == 0
        or np.unique(lengths).size != 1
        or int(lengths[0]) != int(historical_sequence_length)
    ):
        raise PreprocessingInvalid(
            "prospective paratopes do not match the common historical sequence length"
        )

    actions = canonical_internal.loc[:, ["Paratope", *PROXY_COLUMNS, "source_row"]].copy()
    actions.insert(0, "action_index", np.arange(len(actions), dtype=np.int64))
    summary = {
        "source_path": PROSPECTIVE_SOURCE_PATH,
        "source_row_count": int(len(frame)),
        "rows_with_stop_false": int(stop_false.sum()),
        "rows_with_present_paratope": int(present.sum()),
        "rows_with_finite_target": int(target_finite.sum()),
        "retained_before_duplicates": rows_before_duplicates,
        "canonical_action_count_before_component_rule": int(len(actions)),
        "duplicate_group_count": duplicate_groups,
        "duplicate_row_count": rows_before_duplicates - int(len(actions)),
        "sequence_length": int(lengths[0]),
        "historical_action_overlap_count": 0,
        "prospective_action_output_columns": list(actions.columns),
        "heldout_target_magnitude_in_action_output": False,
    }
    return actions, summary


def hamming_knn_graph(paratopes: Sequence[str], k: int = 8) -> HammingGraph:
    """Build the deterministic union of directed Hamming k-NN choices."""

    sequences = np.asarray(tuple(str(value) for value in paratopes), dtype=str)
    n_actions = int(sequences.size)
    if k < 1 or n_actions <= k:
        raise PreprocessingInvalid("action count must exceed the positive graph k")
    lengths = np.char.str_len(sequences)
    if np.unique(lengths).size != 1:
        raise PreprocessingInvalid("Hamming graph requires equal-length paratopes")
    encoded = np.asarray([list(sequence) for sequence in sequences], dtype="U1")
    indices = np.arange(n_actions, dtype=np.int64)
    directed: list[tuple[int, int]] = []
    for source in range(n_actions):
        distances = np.count_nonzero(encoded != encoded[source], axis=1)
        eligible = indices != source
        order = np.lexsort(
            (indices[eligible], sequences[eligible], distances[eligible])
        )
        for neighbor in indices[eligible][order[:k]]:
            directed.append((source, int(neighbor)))
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
    components.sort(key=lambda component: (-len(component), component[0]))
    largest = np.asarray(components[0], dtype=np.int64)
    return HammingGraph(
        edges=edges,
        component_sizes=tuple(len(component) for component in components),
        largest_component_indices=largest,
        largest_component_fraction=float(largest.size / n_actions),
    )


def apply_component_rule(
    actions: pd.DataFrame,
    graph: HammingGraph,
    *,
    minimum_largest_fraction: float = 0.90,
) -> tuple[pd.DataFrame, IntArray, bool]:
    if len(graph.component_sizes) == 1:
        return actions.copy(), graph.edges.copy(), False
    if graph.largest_component_fraction < minimum_largest_fraction:
        raise PreprocessingInvalid(
            "largest Hamming component contains less than 90% of actions"
        )
    subset = actions.iloc[graph.largest_component_indices].copy()
    subset = subset.sort_values("Paratope", kind="stable").reset_index(drop=True)
    old_indices = subset["action_index"].to_numpy(dtype=np.int64)
    remap = {int(old): new for new, old in enumerate(old_indices)}
    retained = set(remap)
    edges = np.asarray(
        sorted(
            (remap[int(left)], remap[int(right)])
            for left, right in graph.edges
            if int(left) in retained and int(right) in retained
        ),
        dtype=np.int64,
    ).reshape(-1, 2)
    subset["action_index"] = np.arange(len(subset), dtype=np.int64)
    return subset, edges, True


def build_proxy_factor_bank(
    actions: pd.DataFrame, calibration_pipeline: Pipeline
) -> pd.DataFrame:
    """Create one local factor per action with both frozen proxy means."""

    required = {"action_index", "Paratope", *PROXY_COLUMNS}
    missing = required.difference(actions.columns)
    if missing:
        raise ValueError(f"target-blind action table lacks columns: {sorted(missing)}")
    proxy_values = actions.loc[:, PROXY_COLUMNS].to_numpy(dtype=float)
    available = np.all(np.isfinite(proxy_values), axis=1)
    selected = actions.loc[available].copy()
    predictions = calibration_pipeline.predict(proxy_values[available])
    return pd.DataFrame(
        {
            "factor_index": np.arange(len(selected), dtype=np.int64),
            "action_index": selected["action_index"].to_numpy(dtype=np.int64),
            "Paratope": selected["Paratope"].to_numpy(dtype=str),
            PROXY_COLUMNS[0]: selected[PROXY_COLUMNS[0]].to_numpy(dtype=float),
            PROXY_COLUMNS[1]: selected[PROXY_COLUMNS[1]].to_numpy(dtype=float),
            "mu_proxy": np.asarray(predictions, dtype=float),
        }
    )


def normalized_laplacian_precision(n_actions: int, edges: ArrayLike) -> FloatArray:
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    adjacency = np.zeros((n_actions, n_actions), dtype=float)
    for left, right in edge_array:
        adjacency[int(left), int(right)] = 1.0
        adjacency[int(right), int(left)] = 1.0
    degrees = adjacency.sum(axis=1)
    if np.any(degrees <= 0.0):
        raise PreprocessingInvalid("normalized Laplacian has an isolated action")
    inverse_sqrt_degree = 1.0 / np.sqrt(degrees)
    normalized_adjacency = (
        inverse_sqrt_degree[:, None]
        * adjacency
        * inverse_sqrt_degree[None, :]
    )
    return np.asarray(2.0 * np.eye(n_actions) - normalized_adjacency, dtype=float)


def all_pairs_graph_distances(n_actions: int, edges: ArrayLike) -> FloatArray:
    edge_array = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    rows = np.concatenate((edge_array[:, 0], edge_array[:, 1]))
    columns = np.concatenate((edge_array[:, 1], edge_array[:, 0]))
    adjacency = csr_matrix(
        (np.ones(rows.size, dtype=float), (rows, columns)),
        shape=(n_actions, n_actions),
    )
    distances = np.asarray(
        shortest_path(adjacency, directed=False, unweighted=True), dtype=float
    )
    if not np.all(np.isfinite(distances)):
        raise TheoryInstantiationInvalid("final action graph is disconnected")
    return distances


def local_proxy_factor_energy(z: ArrayLike, mu_proxy: float, s_proxy: float) -> FloatArray:
    values = np.asarray(z, dtype=float)
    scaled = (values - float(mu_proxy)) / (2.0 * float(s_proxy))
    return np.asarray(
        2.0 * (np.logaddexp(scaled, -scaled) - np.log(2.0)), dtype=float
    )


def local_proxy_factor_gradient(z: ArrayLike, mu_proxy: float, s_proxy: float) -> FloatArray:
    values = np.asarray(z, dtype=float)
    scaled = (values - float(mu_proxy)) / (2.0 * float(s_proxy))
    return np.asarray(np.tanh(scaled) / float(s_proxy), dtype=float)


def local_proxy_factor_hessian(z: ArrayLike, mu_proxy: float, s_proxy: float) -> FloatArray:
    values = np.asarray(z, dtype=float)
    scaled = (values - float(mu_proxy)) / (2.0 * float(s_proxy))
    return np.asarray(
        (1.0 - np.square(np.tanh(scaled))) / (2.0 * float(s_proxy) ** 2),
        dtype=float,
    )


def _factor_derivative_regression(
    proxy_means: ArrayLike, s_proxy: float
) -> dict[str, Any]:
    means = np.asarray(proxy_means, dtype=float)
    probe_means = means[np.unique(np.linspace(0, len(means) - 1, 3, dtype=int))]
    step = max(1e-4 * float(s_proxy), 1e-10)
    gradient_errors: list[float] = []
    hessian_errors: list[float] = []
    gradient_references: list[float] = []
    hessian_references: list[float] = []
    for proxy_mean in probe_means:
        z = float(proxy_mean + 0.37 * s_proxy)
        energy_plus = float(local_proxy_factor_energy(z + step, proxy_mean, s_proxy))
        energy_minus = float(local_proxy_factor_energy(z - step, proxy_mean, s_proxy))
        finite_gradient = (energy_plus - energy_minus) / (2.0 * step)
        analytic_gradient = float(local_proxy_factor_gradient(z, proxy_mean, s_proxy))
        gradient_errors.append(abs(finite_gradient - analytic_gradient))
        gradient_references.append(abs(analytic_gradient))
        gradient_plus = float(local_proxy_factor_gradient(z + step, proxy_mean, s_proxy))
        gradient_minus = float(local_proxy_factor_gradient(z - step, proxy_mean, s_proxy))
        finite_hessian = (gradient_plus - gradient_minus) / (2.0 * step)
        analytic_hessian = float(local_proxy_factor_hessian(z, proxy_mean, s_proxy))
        hessian_errors.append(abs(finite_hessian - analytic_hessian))
        hessian_references.append(abs(analytic_hessian))
    gradient_error = max(gradient_errors)
    hessian_error = max(hessian_errors)
    gradient_tolerance = 1e-7 * max(1.0, max(gradient_references))
    hessian_tolerance = 1e-7 * max(1.0, max(hessian_references))
    if gradient_error > gradient_tolerance or hessian_error > hessian_tolerance:
        raise TheoryInstantiationInvalid(
            "local proxy factor derivative finite-difference regression failed"
        )
    return {
        "factor_derivative_probe_count": int(len(probe_means)),
        "factor_gradient_maximum_absolute_error": gradient_error,
        "factor_gradient_tolerance": gradient_tolerance,
        "factor_hessian_maximum_absolute_error": hessian_error,
        "factor_hessian_tolerance": hessian_tolerance,
    }


def construct_and_verify_theory(
    n_actions: int,
    edges: ArrayLike,
    *,
    proxy_means: ArrayLike,
    s_proxy: float,
    tolerance: float = 1e-10,
) -> TheoryConstruction:
    """Construct Q0, A, K0, C and run every frozen theory regression."""

    q0 = normalized_laplacian_precision(n_actions, edges)
    symmetry_error = float(np.max(np.abs(q0 - q0.T)))
    minimum_eigenvalue = float(np.linalg.eigvalsh(q0).min())
    off_diagonal = q0 - np.diag(np.diag(q0))
    maximum_off_diagonal = float(np.max(off_diagonal))
    if symmetry_error > tolerance or minimum_eigenvalue <= 0.0:
        raise TheoryInstantiationInvalid("Q0 is not symmetric positive definite")
    if maximum_off_diagonal > tolerance:
        raise TheoryInstantiationInvalid("Q0 has a positive off-diagonal entry")

    a = np.diag(np.diag(q0)) - np.abs(off_diagonal)
    a_q0_error = float(np.max(np.abs(a - q0)))
    if a_q0_error > tolerance:
        raise TheoryInstantiationInvalid("implemented Menz matrix A does not equal Q0")
    identity = np.eye(n_actions)
    k0 = np.linalg.solve(q0, identity)
    k0 = 0.5 * (k0 + k0.T)
    c = np.linalg.solve(a, identity)
    c = 0.5 * (c + c.T)
    inverse_residual = float(np.max(np.abs(q0 @ k0 - identity)))
    c_k0_error = float(np.max(np.abs(c - k0)))
    minimum_k0_entry = float(np.min(k0))
    if inverse_residual > tolerance:
        raise TheoryInstantiationInvalid("K0 is not Q0 inverse to tolerance")
    if c_k0_error > tolerance:
        raise TheoryInstantiationInvalid("C = A^-1 does not equal K0")
    if minimum_k0_entry < -tolerance:
        raise TheoryInstantiationInvalid("K0 has a materially negative entry")

    distances = all_pairs_graph_distances(n_actions, edges)
    distance_bound = np.exp2(-distances)
    maximum_distance_bound_violation = float(np.max(k0 - distance_bound))
    if maximum_distance_bound_violation > tolerance:
        raise TheoryInstantiationInvalid(
            "graph-distance covariance bound K0_ij <= 2^-d(i,j) failed"
        )
    derivative_summary = _factor_derivative_regression(proxy_means, s_proxy)
    summary = {
        "matrix_tolerance": tolerance,
        "q0_symmetry_maximum_absolute_error": symmetry_error,
        "q0_minimum_eigenvalue": minimum_eigenvalue,
        "q0_maximum_off_diagonal_entry": maximum_off_diagonal,
        "a_equals_q0_maximum_absolute_error": a_q0_error,
        "q0_k0_inverse_maximum_absolute_residual": inverse_residual,
        "c_equals_k0_maximum_absolute_error": c_k0_error,
        "k0_minimum_entry": minimum_k0_entry,
        "graph_distance_covariance_maximum_violation": maximum_distance_bound_violation,
        "a_equals_q0": True,
        "c_equals_k0_equals_q0_inverse": True,
        "k0_entrywise_nonnegative_to_tolerance": True,
        "graph_distance_covariance_bound_passed": True,
        **derivative_summary,
    }
    return TheoryConstruction(q0, a, k0, c, distances, summary)


def structural_tail_count(contributions: ArrayLike, epsilon: float = 0.05) -> int:
    """Return the smallest active count whose descending omitted tail is <= epsilon."""

    values = np.asarray(contributions, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("factor contributions must be one-dimensional and finite")
    if np.any(values < 0.0):
        raise ValueError("factor contributions must be nonnegative")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    descending = np.sort(values)[::-1]
    total = float(descending.sum())
    if total <= epsilon:
        return 0
    required_removed = total - float(epsilon)
    return int(np.searchsorted(np.cumsum(descending), required_removed, side="left") + 1)


def compute_structural_sparsity(
    k0: ArrayLike,
    factor_action_indices: ArrayLike,
    s_proxy: float,
    graph_distances: ArrayLike,
    *,
    epsilon_struct: float = 0.05,
    maximum_active_factor_fraction: float = 0.50,
    minimum_passing_pair_fraction: float = 0.90,
) -> StructuralSparsityResult:
    """Compute the frozen all-unordered-action-pair R_0.05 distribution."""

    covariance = np.asarray(k0, dtype=float)
    factor_actions = np.asarray(factor_action_indices, dtype=np.int64)
    distances = np.asarray(graph_distances, dtype=float)
    n_actions = covariance.shape[0]
    n_factors = int(factor_actions.size)
    if covariance.shape != (n_actions, n_actions) or distances.shape != covariance.shape:
        raise ValueError("K0 and graph-distance matrices must be square and aligned")
    if n_factors == 0:
        raise ValueError("the structural distribution requires at least one factor")
    rows: list[dict[str, Any]] = []
    highest_distance_counts: Counter[int] = Counter()
    for x in range(n_actions):
        for x_hat in range(x + 1, n_actions):
            contributions = (
                covariance[x, factor_actions] + covariance[x_hat, factor_actions]
            ) / float(s_proxy)
            if np.min(contributions) < -1e-10:
                raise TheoryInstantiationInvalid(
                    "structural contribution is materially negative"
                )
            contributions = np.where(contributions < 0.0, 0.0, contributions)
            active_count = structural_tail_count(contributions, epsilon_struct)
            ratio = float(active_count / n_factors)
            highest_factor = int(np.argmax(contributions))
            highest_action = int(factor_actions[highest_factor])
            distance_to_x = int(distances[x, highest_action])
            distance_to_x_hat = int(distances[x_hat, highest_action])
            minimum_distance = min(distance_to_x, distance_to_x_hat)
            highest_distance_counts[minimum_distance] += 1
            rows.append(
                {
                    "x_action_index": x,
                    "x_hat_action_index": x_hat,
                    "M_0_05": active_count,
                    "N_factors": n_factors,
                    "R_0_05": ratio,
                    "highest_contribution_factor_index": highest_factor,
                    "highest_contribution_factor_action_index": highest_action,
                    "highest_factor_distance_to_x": distance_to_x,
                    "highest_factor_distance_to_x_hat": distance_to_x_hat,
                    "highest_factor_minimum_graph_distance": minimum_distance,
                }
            )
    pairwise = pd.DataFrame(rows)
    ratios = pairwise["R_0_05"].to_numpy(dtype=float)
    fraction_025 = float(np.mean(ratios <= 0.25))
    fraction_050 = float(np.mean(ratios <= maximum_active_factor_fraction))
    verdict = (
        "PASS_STRUCTURAL_PREFLIGHT"
        if fraction_050 >= minimum_passing_pair_fraction
        else "FAIL_STRUCTURAL_SPARSITY"
    )
    summary = {
        "epsilon_struct": float(epsilon_struct),
        "action_pair_count": int(len(pairwise)),
        "factor_count": n_factors,
        "fraction_pairs_R_0_05_at_most_0_25": fraction_025,
        "fraction_pairs_R_0_05_at_most_0_50": fraction_050,
        "median_R_0_05": float(np.quantile(ratios, 0.50)),
        "percentile_75_R_0_05": float(np.quantile(ratios, 0.75)),
        "percentile_90_R_0_05": float(np.quantile(ratios, 0.90)),
        "percentile_95_R_0_05": float(np.quantile(ratios, 0.95)),
        "maximum_active_factor_fraction": float(maximum_active_factor_fraction),
        "minimum_passing_pair_fraction": float(minimum_passing_pair_fraction),
        "highest_contribution_factor_minimum_graph_distance_counts": {
            str(distance): int(count)
            for distance, count in sorted(highest_distance_counts.items())
        },
        "quantile_method": "numpy_linear",
        "verdict": verdict,
    }
    return StructuralSparsityResult(pairwise, summary, verdict)


def _degree_summary(n_actions: int, edges: IntArray) -> dict[str, Any]:
    degrees = np.zeros(n_actions, dtype=np.int64)
    for left, right in edges:
        degrees[int(left)] += 1
        degrees[int(right)] += 1
    return {
        "minimum": int(degrees.min()),
        "median": float(np.median(degrees)),
        "mean": float(np.mean(degrees)),
        "maximum": int(degrees.max()),
    }


def prepare_structural_preflight(
    config: Mapping[str, Any], repository_root: str | Path
) -> PreparedStructuralPreflight:
    """Prepare and theory-check the frozen real-data structural construction."""

    sources = ensure_pinned_sources(config, repository_root)
    source_by_path = {source.path: source for source in sources}
    if set(source_by_path) != {HISTORICAL_SOURCE_PATH, PROSPECTIVE_SOURCE_PATH}:
        raise PreprocessingInvalid("the two frozen sources were not resolved by role")
    historical_frame = pd.read_csv(
        source_by_path[HISTORICAL_SOURCE_PATH].local_path, low_memory=False
    )
    prospective_frame = pd.read_csv(
        source_by_path[PROSPECTIVE_SOURCE_PATH].local_path, low_memory=False
    )
    preprocessing = config["preprocessing"]
    historical = prepare_historical_rows(
        historical_frame,
        duplicate_rtol=float(preprocessing["duplicate_rtol"]),
        duplicate_atol=float(preprocessing["duplicate_atol"]),
    )
    calibration_settings = config["calibration"]
    calibration = fit_proxy_calibration(
        historical,
        minimum_calibration_count=int(
            preprocessing["minimum_calibration_count"]
        ),
        minimum_target_standard_deviation=float(
            calibration_settings["minimum_target_standard_deviation"]
        ),
        minimum_proxy_scale=float(calibration_settings["minimum_proxy_scale"]),
    )
    actions_before_component, prospective_summary = prepare_prospective_actions(
        prospective_frame,
        historical_paratopes=historical.paratopes,
        historical_sequence_length=historical.sequence_length,
        duplicate_rtol=float(preprocessing["duplicate_rtol"]),
        duplicate_atol=float(preprocessing["duplicate_atol"]),
    )
    graph = hamming_knn_graph(actions_before_component["Paratope"].tolist(), k=8)
    actions, edges, component_was_restricted = apply_component_rule(
        actions_before_component,
        graph,
        minimum_largest_fraction=float(
            preprocessing["minimum_largest_component_fraction"]
        ),
    )
    if len(actions) < int(preprocessing["minimum_action_count"]):
        raise PreprocessingInvalid(
            f"final action count {len(actions)} is below 150",
            details={
                "historical": historical.summary,
                "prospective": prospective_summary,
                "graph_component_sizes": list(graph.component_sizes),
            },
        )
    factors = build_proxy_factor_bank(actions, calibration.pipeline)
    factor_count = int(len(factors))
    coverage = float(factor_count / len(actions))
    if factor_count < int(preprocessing["minimum_factor_count"]):
        raise PreprocessingInvalid(
            f"proxy factor count {factor_count} is below 75"
        )
    if coverage < float(preprocessing["minimum_factor_coverage"]):
        raise PreprocessingInvalid(
            f"proxy factor coverage {coverage} is below 0.40"
        )

    theory = construct_and_verify_theory(
        len(actions),
        edges,
        proxy_means=factors["mu_proxy"].to_numpy(dtype=float),
        s_proxy=calibration.s_proxy,
        tolerance=float(config["theory_checks"]["matrix_tolerance"]),
    )
    distances = theory.graph_distances
    graph_summary = {
        "k": 8,
        "action_count_before_component_rule": int(len(actions_before_component)),
        "component_count_before_rule": int(len(graph.component_sizes)),
        "component_sizes_before_rule": list(graph.component_sizes),
        "largest_component_fraction": graph.largest_component_fraction,
        "component_was_restricted": component_was_restricted,
        "final_component_count": 1,
        "final_action_count": int(len(actions)),
        "final_edge_count": int(len(edges)),
        "degree_summary": _degree_summary(len(actions), edges),
        "diameter": int(np.max(distances)),
    }
    preprocessing_summary = {
        "source_roles": {
            "historical_calibration": HISTORICAL_SOURCE_PATH,
            "prospective_actions": PROSPECTIVE_SOURCE_PATH,
            "sources_unioned_for_actions": False,
        },
        "historical": historical.summary,
        "prospective": prospective_summary,
        "historical_target_scale_count": calibration.historical_target_scale_count,
        "calibration_count": calibration.calibration_count,
        "final_action_count": int(len(actions)),
        "proxy_factor_count": factor_count,
        "factor_coverage_fraction": coverage,
        "actions_with_missing_proxy_factor": int(len(actions) - factor_count),
        "heldout_target_magnitude_available_to_graph_interface": False,
        "heldout_target_magnitude_available_to_factor_interface": False,
        "heldout_target_magnitude_available_to_structural_interface": False,
        "preprocessing_valid": True,
    }
    return PreparedStructuralPreflight(
        actions=actions,
        edges=edges,
        factors=factors,
        calibration=calibration,
        theory=theory,
        preprocessing_summary=preprocessing_summary,
        graph_summary=graph_summary,
        source_provenance=sources,
    )


def graph_edge_table(actions: pd.DataFrame, edges: ArrayLike) -> pd.DataFrame:
    sequences = actions["Paratope"].astype(str).tolist()
    rows = []
    for left, right in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        left_index = int(left)
        right_index = int(right)
        rows.append(
            {
                "left_action_index": left_index,
                "right_action_index": right_index,
                "left_paratope": sequences[left_index],
                "right_paratope": sequences[right_index],
                "hamming_distance": sum(
                    first != second
                    for first, second in zip(
                        sequences[left_index], sequences[right_index], strict=True
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def create_immutable_output_directory(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)
    return destination


__all__ = [
    "BASE_COLUMNS",
    "HISTORICAL_SOURCE_PATH",
    "HammingGraph",
    "PROSPECTIVE_SOURCE_PATH",
    "PROXY_COLUMNS",
    "PinnedSource",
    "PreflightFailure",
    "PreprocessingAmbiguity",
    "PreprocessingInvalid",
    "PreparedStructuralPreflight",
    "ProxyCalibration",
    "StructuralSparsityResult",
    "TARGET_COLUMN",
    "TheoryConstruction",
    "TheoryInstantiationInvalid",
    "all_pairs_graph_distances",
    "apply_component_rule",
    "build_proxy_factor_bank",
    "compute_structural_sparsity",
    "construct_and_verify_theory",
    "create_immutable_output_directory",
    "ensure_pinned_sources",
    "fit_proxy_calibration",
    "graph_edge_table",
    "hamming_knn_graph",
    "load_structural_preflight_config",
    "local_proxy_factor_energy",
    "local_proxy_factor_gradient",
    "local_proxy_factor_hessian",
    "normalized_laplacian_precision",
    "prepare_historical_rows",
    "prepare_prospective_actions",
    "prepare_structural_preflight",
    "sha256_file",
    "structural_tail_count",
]
