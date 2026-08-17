"""Frozen rare-mode mechanism experiment and paper Figure 1 generation."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc

from .mixture import (
    GaussianMixture1D,
    chi_square_decision_shift,
    mc_relative_variance,
    population_ess_fraction,
    softplus_utility,
)


@dataclass(frozen=True)
class RareModeConfig:
    raw: dict[str, Any]
    source_path: Path

    @classmethod
    def load(cls, path: str | Path) -> "RareModeConfig":
        source = Path(path).resolve()
        return cls(json.loads(source.read_text()), source)

    @property
    def model(self) -> dict[str, Any]:
        return self.raw["model"]

    @property
    def numerics(self) -> dict[str, Any]:
        return self.raw["numerics"]

    @property
    def gate(self) -> dict[str, Any]:
        return self.raw["prospective_gate"]

    @property
    def config_hash(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def component_means(x: np.ndarray | float, config: RareModeConfig) -> np.ndarray:
    x_array = np.asarray(x, dtype=np.float64)
    common = config.model["common_mean"]
    rare = config.model["rare_mean"]

    def bell(parameters: dict[str, float]) -> np.ndarray:
        offset = (x_array - parameters["center"]) / parameters["width"]
        return parameters["baseline"] + parameters["amplitude"] * np.exp(
            -0.5 * np.square(offset)
        )

    return np.stack([bell(common), bell(rare)], axis=-1)


def mixture_at(x: float, config: RareModeConfig) -> GaussianMixture1D:
    rare_weight = config.model["rare_weight"]
    return GaussianMixture1D(
        weights=np.array([1.0 - rare_weight, rare_weight]),
        means=component_means(float(x), config),
        stds=np.array([config.model["common_std"], config.model["rare_std"]]),
    )


def _base_samples(method: str, sample_count: int, seed: int) -> np.ndarray:
    if method == "iid":
        return np.random.default_rng(seed).random((sample_count, 2))
    if method == "qmc":
        power = int(round(math.log2(sample_count)))
        if 2**power != sample_count:
            raise ValueError("scrambled Sobol sample_count must be a power of two")
        return qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(power)
    raise ValueError(f"unknown sampling method: {method}")


def _component_indices_and_scores(
    base: np.ndarray, config: RareModeConfig
) -> tuple[np.ndarray, np.ndarray]:
    rare = base[..., 0] >= 1.0 - config.model["rare_weight"]
    tiny = np.nextafter(0.0, 1.0)
    scores = ndtri(np.clip(base[..., 1], tiny, np.nextafter(1.0, 0.0)))
    return rare.astype(np.int64), scores


def _outcomes(
    x: np.ndarray | float,
    indices: np.ndarray,
    scores: np.ndarray,
    config: RareModeConfig,
) -> np.ndarray:
    means = component_means(x, config)
    stds = np.array([config.model["common_std"], config.model["rare_std"]])
    if means.ndim == 1:
        return means[indices] + stds[indices] * scores
    return means[:, indices] + stds[indices][None, :] * scores[None, :]


def _stable_log_mean(utility: np.ndarray, axis: int = -1) -> np.ndarray:
    positive = utility > 0.0
    logs = np.full_like(utility, -np.inf, dtype=np.float64)
    np.log(utility, out=logs, where=positive)
    return logsumexp(logs, axis=axis) - np.log(utility.shape[axis])


def estimate_landscape(
    x: np.ndarray,
    base: np.ndarray,
    config: RareModeConfig,
    *,
    utility: str = "improvement",
    chunk_size: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    indices, scores = _component_indices_and_scores(base, config)
    acquisitions: list[np.ndarray] = []
    log_acquisitions: list[np.ndarray] = []
    for start in range(0, len(x), chunk_size):
        batch = np.asarray(x[start : start + chunk_size], dtype=np.float64)
        outcomes = _outcomes(batch, indices, scores, config)
        if utility == "improvement":
            values = np.maximum(outcomes - config.model["incumbent"], 0.0)
        elif utility == "softplus":
            values = softplus_utility(
                outcomes,
                config.model["incumbent"],
                config.model["smooth_temperature"],
            )
        else:
            raise ValueError(f"unknown utility: {utility}")
        log_mean = _stable_log_mean(values, axis=1)
        acquisitions.append(np.exp(log_mean))
        log_acquisitions.append(log_mean)
    return np.concatenate(acquisitions), np.concatenate(log_acquisitions)


def exact_landscape(
    x: np.ndarray, config: RareModeConfig
) -> dict[str, np.ndarray]:
    raw_first: list[float] = []
    raw_second: list[float] = []
    smooth_first: list[float] = []
    smooth_second: list[float] = []
    for location in np.asarray(x, dtype=np.float64):
        mixture = mixture_at(float(location), config)
        first, second = mixture.improvement_moments(config.model["incumbent"])
        smooth_1, smooth_2 = mixture.softplus_moments(
            config.model["incumbent"],
            config.model["smooth_temperature"],
            order=config.numerics["gauss_hermite_order"],
        )
        raw_first.append(first)
        raw_second.append(second)
        smooth_first.append(smooth_1)
        smooth_second.append(smooth_2)
    return {
        "raw_first": np.asarray(raw_first),
        "raw_second": np.asarray(raw_second),
        "smooth_first": np.asarray(smooth_first),
        "smooth_second": np.asarray(smooth_second),
    }


def _candidate_summary(location: float, config: RareModeConfig) -> dict[str, Any]:
    mixture = mixture_at(location, config)
    incumbent = config.model["incumbent"]
    raw_first, raw_second = mixture.improvement_moments(incumbent)
    smooth_first, smooth_second = mixture.softplus_moments(
        incumbent,
        config.model["smooth_temperature"],
        order=config.numerics["gauss_hermite_order"],
    )
    return {
        "x": location,
        "component_means": mixture.means.tolist(),
        "raw": {
            "acquisition": raw_first,
            "second_moment": raw_second,
            "chi_square": chi_square_decision_shift(raw_first, raw_second),
            "ess_fraction": population_ess_fraction(raw_first, raw_second),
            "tilted_component_weights": mixture.tilted_component_weights(
                incumbent
            ).tolist(),
        },
        "smooth": {
            "acquisition": smooth_first,
            "second_moment": smooth_second,
            "chi_square": chi_square_decision_shift(smooth_first, smooth_second),
            "ess_fraction": population_ess_fraction(smooth_first, smooth_second),
            "tilted_component_weights": mixture.tilted_component_weights(
                incumbent,
                utility="softplus",
                temperature=config.model["smooth_temperature"],
                order=config.numerics["gauss_hermite_order"],
            ).tolist(),
        },
        "predictive_shape": mixture.standardized_shape(),
    }


def _iid_repetitions(
    sample_count: int, config: RareModeConfig
) -> dict[str, np.ndarray]:
    repetitions = config.numerics["iid_repetitions"]
    chunk_size = config.numerics["repetition_chunk_size"]
    seed = config.numerics["experiment_seed"] + 100_000 * sample_count
    generator = np.random.default_rng(seed)
    x_a = config.model["candidate_a"]
    x_b = config.model["candidate_b"]
    output = {key: [] for key in ("raw_a", "raw_b", "smooth_a", "smooth_b")}
    for start in range(0, repetitions, chunk_size):
        size = min(chunk_size, repetitions - start)
        base = generator.random((size, sample_count, 2))
        indices, scores = _component_indices_and_scores(base, config)
        y_a = _outcomes(x_a, indices, scores, config)
        y_b = _outcomes(x_b, indices, scores, config)
        raw_a = np.maximum(y_a - config.model["incumbent"], 0.0)
        raw_b = np.maximum(y_b - config.model["incumbent"], 0.0)
        smooth_a = softplus_utility(
            y_a, config.model["incumbent"], config.model["smooth_temperature"]
        )
        smooth_b = softplus_utility(
            y_b, config.model["incumbent"], config.model["smooth_temperature"]
        )
        output["raw_a"].append(raw_a.mean(axis=1))
        output["raw_b"].append(raw_b.mean(axis=1))
        output["smooth_a"].append(smooth_a.mean(axis=1))
        output["smooth_b"].append(smooth_b.mean(axis=1))
    return {key: np.concatenate(value) for key, value in output.items()}


def _qmc_repetitions(
    sample_count: int, config: RareModeConfig
) -> dict[str, np.ndarray]:
    repetitions = config.numerics["qmc_repetitions"]
    x_a = config.model["candidate_a"]
    x_b = config.model["candidate_b"]
    output = {
        key: np.empty(repetitions, dtype=np.float64)
        for key in ("raw_a", "raw_b", "smooth_a", "smooth_b")
    }
    for repetition in range(repetitions):
        seed = (
            config.numerics["experiment_seed"]
            + 1_000_000
            + sample_count * 10_000
            + repetition
        )
        base = _base_samples("qmc", sample_count, seed)
        indices, scores = _component_indices_and_scores(base, config)
        y_a = _outcomes(x_a, indices, scores, config)
        y_b = _outcomes(x_b, indices, scores, config)
        raw_a = np.maximum(y_a - config.model["incumbent"], 0.0)
        raw_b = np.maximum(y_b - config.model["incumbent"], 0.0)
        smooth_a = softplus_utility(
            y_a, config.model["incumbent"], config.model["smooth_temperature"]
        )
        smooth_b = softplus_utility(
            y_b, config.model["incumbent"], config.model["smooth_temperature"]
        )
        output["raw_a"][repetition] = raw_a.mean()
        output["raw_b"][repetition] = raw_b.mean()
        output["smooth_a"][repetition] = smooth_a.mean()
        output["smooth_b"][repetition] = smooth_b.mean()
    return output


def repeated_sampling(
    config: RareModeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_a = _candidate_summary(config.model["candidate_a"], config)
    candidate_b = _candidate_summary(config.model["candidate_b"], config)
    exact = {
        "raw_a": candidate_a["raw"]["acquisition"],
        "raw_b": candidate_b["raw"]["acquisition"],
        "smooth_a": candidate_a["smooth"]["acquisition"],
        "smooth_b": candidate_b["smooth"]["acquisition"],
    }
    variance_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    for sample_count in config.numerics["sample_counts"]:
        for method, sampler in (("iid", _iid_repetitions), ("qmc", _qmc_repetitions)):
            estimates = sampler(sample_count, config)
            repetitions = len(estimates["raw_a"])
            for utility in ("raw", "smooth"):
                key_a, key_b = f"{utility}_a", f"{utility}_b"
                ranking_rows.append(
                    {
                        "method": method,
                        "utility": utility,
                        "sample_count": sample_count,
                        "repetitions": repetitions,
                        "ranking_accuracy": float(
                            np.mean(estimates[key_b] > estimates[key_a])
                        ),
                        "exact_a": exact[key_a],
                        "exact_b": exact[key_b],
                    }
                )
                for candidate, key in (("A", key_a), ("B", key_b)):
                    empirical = float(
                        np.var(estimates[key], ddof=1) / np.square(exact[key])
                    )
                    first = (
                        candidate_a[utility]["acquisition"]
                        if candidate == "A"
                        else candidate_b[utility]["acquisition"]
                    )
                    second = (
                        candidate_a[utility]["second_moment"]
                        if candidate == "A"
                        else candidate_b[utility]["second_moment"]
                    )
                    theoretical = mc_relative_variance(first, second, sample_count)
                    variance_rows.append(
                        {
                            "method": method,
                            "utility": utility,
                            "candidate": candidate,
                            "sample_count": sample_count,
                            "repetitions": repetitions,
                            "empirical_relative_variance": empirical,
                            "iid_theoretical_relative_variance": theoretical,
                            "relative_error_to_iid_theory": abs(empirical - theoretical)
                            / theoretical,
                        }
                    )
    return pd.DataFrame(variance_rows), pd.DataFrame(ranking_rows)


def _representative_landscapes(
    x_grid: np.ndarray, config: RareModeConfig
) -> pd.DataFrame:
    output: dict[str, np.ndarray] = {"x": x_grid}
    count = config.numerics["representative_sample_count"]
    for method in ("iid", "qmc"):
        for seed in config.numerics["representative_seeds"]:
            base = _base_samples(method, count, seed)
            acquisition, log_acquisition = estimate_landscape(x_grid, base, config)
            output[f"{method}_seed{seed}_ei"] = acquisition
            output[f"{method}_seed{seed}_log_ei"] = log_acquisition
    high_count = 2 ** config.numerics["high_budget_qmc_power"]
    high_base = _base_samples(
        "qmc", high_count, config.numerics["high_budget_qmc_seed"]
    )
    high, high_log = estimate_landscape(x_grid, high_base, config, chunk_size=8)
    output["high_budget_qmc_ei"] = high
    output["high_budget_qmc_log_ei"] = high_log
    return pd.DataFrame(output)


def _evaluate_gate(
    summary: dict[str, Any], variance: pd.DataFrame, ranking: pd.DataFrame, config: RareModeConfig
) -> dict[str, Any]:
    gate = config.gate
    a = summary["candidate_a"]
    b = summary["candidate_b"]

    def accuracy(method: str, utility: str, sample_count: int) -> float:
        row = ranking[
            (ranking.method == method)
            & (ranking.utility == utility)
            & (ranking.sample_count == sample_count)
        ]
        return float(row.iloc[0].ranking_accuracy)

    iid_raw = variance[(variance.method == "iid") & (variance.utility == "raw")]
    checks = {
        "raw_ei_margin": b["raw"]["acquisition"] / a["raw"]["acquisition"]
        >= gate["minimum_raw_ei_ratio_b_over_a"],
        "smooth_margin": b["smooth"]["acquisition"] / a["smooth"]["acquisition"]
        >= gate["minimum_smooth_acquisition_ratio_b_over_a"],
        "large_decision_shift": b["raw"]["ess_fraction"]
        <= gate["maximum_candidate_b_ess_fraction"],
        "rare_mode_dominates_tilt": b["raw"]["tilted_component_weights"][1]
        >= gate["minimum_candidate_b_tilted_rare_weight"],
        "iid_variance_identity": float(iid_raw.relative_error_to_iid_theory.max())
        <= gate["maximum_iid_variance_relative_error"],
        "iid_raw_failure_at_512": accuracy("iid", "raw", 512)
        <= gate["maximum_iid_raw_ranking_accuracy_at_512"],
        "qmc_raw_failure_at_512": accuracy("qmc", "raw", 512)
        <= gate["maximum_qmc_raw_ranking_accuracy_at_512"],
        "smooth_failure_at_512": accuracy("iid", "smooth", 512)
        <= gate["maximum_iid_smooth_ranking_accuracy_at_512"],
        "eventual_iid_recovery": accuracy("iid", "raw", 8192)
        >= gate["minimum_iid_raw_ranking_accuracy_at_8192"],
    }
    return {
        "protocol_version": config.raw["protocol_version"],
        "config_hash": config.config_hash,
        "status": "GO_EXPECTATIONS_MET" if all(checks.values()) else "NO_GO",
        "checks": checks,
        "observed": {
            "raw_ei_ratio_b_over_a": b["raw"]["acquisition"]
            / a["raw"]["acquisition"],
            "smooth_ratio_b_over_a": b["smooth"]["acquisition"]
            / a["smooth"]["acquisition"],
            "candidate_b_ess_fraction": b["raw"]["ess_fraction"],
            "candidate_b_tilted_rare_weight": b["raw"]["tilted_component_weights"][1],
            "maximum_iid_raw_variance_relative_error": float(
                iid_raw.relative_error_to_iid_theory.max()
            ),
            "iid_raw_accuracy_at_512": accuracy("iid", "raw", 512),
            "qmc_raw_accuracy_at_512": accuracy("qmc", "raw", 512),
            "iid_smooth_accuracy_at_512": accuracy("iid", "smooth", 512),
            "iid_raw_accuracy_at_8192": accuracy("iid", "raw", 8192),
        },
    }


def make_figure(
    summary: dict[str, Any],
    exact: pd.DataFrame,
    representative: pd.DataFrame,
    variance: pd.DataFrame,
    ranking: pd.DataFrame,
    config: RareModeConfig,
    output_directory: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )
    figure = plt.figure(figsize=(12.0, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2)
    density_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=grid[0, 0], wspace=0.28)
    axes_a = [figure.add_subplot(density_grid[0, index]) for index in range(2)]
    incumbent = config.model["incumbent"]
    for axis, key, bounds in zip(
        axes_a,
        ("candidate_a", "candidate_b"),
        ((-0.55, 0.23), (-0.55, 7.75)),
        strict=True,
    ):
        location = summary[key]["x"]
        mixture = mixture_at(location, config)
        values = np.linspace(*bounds, config.numerics["density_grid_size"])
        axis.plot(values, mixture.density(values), color="#4C78A8", label="posterior $P_x$")
        axis.plot(
            values,
            mixture.tilted_density(values, incumbent),
            color="#E45756",
            label="decision tilt $\\Pi_x$",
        )
        axis.set_yscale("log")
        axis.set_ylim(1e-4, 30.0)
        axis.axvline(incumbent, color="0.35", linestyle=":", linewidth=0.9)
        rare_tilt = summary[key]["raw"]["tilted_component_weights"][1]
        axis.set_title(f"$x={location:.2f}$: rare 0.5% $\\to$ {100*rare_tilt:.1f}%")
        axis.set_xlabel("outcome $y$")
        axis.grid(alpha=0.18)
    axes_a[0].set_ylabel("density (log scale)")
    axes_a[0].legend(loc="upper left")
    axes_a[0].text(-0.2, 1.12, "A", transform=axes_a[0].transAxes, fontweight="bold", fontsize=13)

    axis_b = figure.add_subplot(grid[0, 1])
    exact_log = exact["log_ei"].to_numpy()
    floor = min(float(np.nanmin(exact_log)) - 1.0, -12.0)
    axis_b.plot(exact.x, exact_log, color="black", linewidth=2.2, label="exact LogEI")
    axis_b.plot(
        representative.x,
        np.maximum(representative.high_budget_qmc_log_ei, floor),
        color="#54A24B",
        linestyle="--",
        linewidth=1.5,
        label=f"Sobol reference ($2^{{{config.numerics['high_budget_qmc_power']}}}$)",
    )
    for index, seed in enumerate(config.numerics["representative_seeds"]):
        axis_b.plot(
            representative.x,
            np.maximum(representative[f"iid_seed{seed}_log_ei"], floor),
            color="#F58518",
            alpha=0.45,
            linewidth=0.9,
            label="iid LogEI, N=128" if index == 0 else None,
        )
        axis_b.plot(
            representative.x,
            np.maximum(representative[f"qmc_seed{seed}_log_ei"], floor),
            color="#4C78A8",
            alpha=0.45,
            linewidth=0.9,
            label="scrambled Sobol, N=128" if index == 0 else None,
        )
    axis_b.axvline(config.model["candidate_a"], color="0.5", linestyle=":")
    axis_b.axvline(config.model["candidate_b"], color="0.5", linestyle=":")
    axis_b.set_xlabel("candidate $x$")
    axis_b.set_ylabel("log expected improvement")
    axis_b.set_title("Finite posterior samples create the wrong acquisition mode")
    axis_b.grid(alpha=0.18)
    axis_b.legend(loc="lower left")
    axis_b.text(-0.12, 1.04, "B", transform=axis_b.transAxes, fontweight="bold", fontsize=13)

    axis_c = figure.add_subplot(grid[1, 0])
    styles = {"A": ("#9D755D", "o"), "B": ("#E45756", "o")}
    for candidate in ("A", "B"):
        subset = variance[
            (variance.method == "iid")
            & (variance.utility == "raw")
            & (variance.candidate == candidate)
        ]
        color, marker = styles[candidate]
        axis_c.loglog(
            subset.sample_count,
            subset.iid_theoretical_relative_variance,
            color=color,
            linewidth=1.8,
            label=f"exact $\\chi^2/N$, $x_{candidate}$",
        )
        axis_c.scatter(
            subset.sample_count,
            subset.empirical_relative_variance,
            color=color,
            marker=marker,
            facecolors="none",
            zorder=3,
            label=f"iid empirical, $x_{candidate}$",
        )
    smooth_b = variance[
        (variance.method == "iid")
        & (variance.utility == "smooth")
        & (variance.candidate == "B")
    ]
    axis_c.loglog(
        smooth_b.sample_count,
        smooth_b.empirical_relative_variance,
        color="#B279A2",
        marker="s",
        linestyle=":",
        label="iid empirical, smooth utility at $x_B$",
    )
    qmc_b = variance[
        (variance.method == "qmc")
        & (variance.utility == "raw")
        & (variance.candidate == "B")
    ]
    axis_c.loglog(
        qmc_b.sample_count,
        qmc_b.empirical_relative_variance,
        color="#4C78A8",
        marker="^",
        linestyle="--",
        label="scrambled Sobol empirical, $x_B$",
    )
    axis_c.set_xlabel("posterior sample count $N$")
    axis_c.set_ylabel("relative variance")
    axis_c.set_title("Empirical iid variance follows the exact decision-shift law")
    axis_c.grid(which="both", alpha=0.18)
    axis_c.legend(ncol=2, loc="lower left")
    axis_c.text(-0.12, 1.04, "C", transform=axis_c.transAxes, fontweight="bold", fontsize=13)

    axis_d = figure.add_subplot(grid[1, 1])
    line_styles = {
        ("iid", "raw"): ("#F58518", "o", "iid, improvement"),
        ("qmc", "raw"): ("#4C78A8", "^", "scrambled Sobol, improvement"),
        ("iid", "smooth"): ("#E45756", "s", "iid, positive softplus"),
        ("qmc", "smooth"): ("#72B7B2", "D", "scrambled Sobol, positive softplus"),
    }
    for (method, utility), (color, marker, label) in line_styles.items():
        subset = ranking[(ranking.method == method) & (ranking.utility == utility)]
        axis_d.semilogx(
            subset.sample_count,
            subset.ranking_accuracy,
            base=2,
            color=color,
            marker=marker,
            linewidth=1.7,
            label=label,
        )
    axis_d.axhline(0.5, color="0.5", linestyle=":", linewidth=1.0)
    axis_d.axhline(0.9, color="0.7", linestyle="--", linewidth=0.8)
    axis_d.set_ylim(0.0, 1.03)
    axis_d.set_xlabel("posterior sample count $N$")
    axis_d.set_ylabel("$P(\\widehat{EI}(x_B)>\\widehat{EI}(x_A))$")
    axis_d.set_title("Stable log arithmetic cannot supply missing decision worlds")
    axis_d.grid(alpha=0.18)
    axis_d.legend(loc="lower right")
    axis_d.text(-0.12, 1.04, "D", transform=axis_d.transAxes, fontweight="bold", fontsize=13)

    figure.suptitle(
        "Posterior-to-decision shift makes finite-sample acquisition inference unreliable",
        fontsize=13,
        fontweight="bold",
    )
    for suffix in ("png", "pdf", "svg"):
        figure_path = output_directory / f"figure1_rare_mode_mechanism.{suffix}"
        figure.savefig(figure_path)
        if suffix == "svg":
            # Matplotlib writes insignificant trailing spaces in SVG path data.
            # Normalize them so generated evidence passes repository diff checks.
            lines = figure_path.read_text().splitlines()
            figure_path.write_text("\n".join(line.rstrip() for line in lines) + "\n")
    plt.close(figure)


def run_experiment(
    config_path: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    started = time.perf_counter()
    config = RareModeConfig.load(config_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    x_grid = np.linspace(
        config.model["domain"][0],
        config.model["domain"][1],
        config.numerics["landscape_grid_size"],
    )
    exact_values = exact_landscape(x_grid, config)
    exact_frame = pd.DataFrame(
        {
            "x": x_grid,
            "ei": exact_values["raw_first"],
            "log_ei": np.log(exact_values["raw_first"]),
            "second_improvement_moment": exact_values["raw_second"],
            "chi_square": exact_values["raw_second"]
            / np.square(exact_values["raw_first"])
            - 1.0,
            "ess_fraction": np.square(exact_values["raw_first"])
            / exact_values["raw_second"],
            "smooth_acquisition": exact_values["smooth_first"],
            "smooth_second_moment": exact_values["smooth_second"],
        }
    )
    representative = _representative_landscapes(x_grid, config)
    variance, ranking = repeated_sampling(config)
    candidate_a = _candidate_summary(config.model["candidate_a"], config)
    candidate_b = _candidate_summary(config.model["candidate_b"], config)
    optimum_index = int(np.argmax(exact_values["raw_first"]))
    top_mask = exact_values["raw_first"] >= 0.1 * exact_values["raw_first"].max()
    high_log_error = np.abs(
        representative.high_budget_qmc_log_ei.to_numpy()[top_mask]
        - exact_frame.log_ei.to_numpy()[top_mask]
    )
    summary: dict[str, Any] = {
        "protocol_version": config.raw["protocol_version"],
        "config_hash": config.config_hash,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
        "raw_ei_ratio_b_over_a": candidate_b["raw"]["acquisition"]
        / candidate_a["raw"]["acquisition"],
        "smooth_ratio_b_over_a": candidate_b["smooth"]["acquisition"]
        / candidate_a["smooth"]["acquisition"],
        "exact_grid_optimum": {
            "x": float(x_grid[optimum_index]),
            "ei": float(exact_values["raw_first"][optimum_index]),
        },
        "high_budget_qmc": {
            "sample_count": 2 ** config.numerics["high_budget_qmc_power"],
            "maximum_absolute_log_error_in_top_10_percent_region": float(
                high_log_error.max()
            ),
            "root_mean_square_log_error_in_top_10_percent_region": float(
                np.sqrt(np.mean(np.square(high_log_error)))
            ),
        },
    }
    gate = _evaluate_gate(summary, variance, ranking, config)
    summary["gate"] = gate
    summary["runtime_seconds"] = time.perf_counter() - started
    exact_frame.to_csv(output / "exact_landscape.csv", index=False)
    representative.to_csv(output / "representative_landscapes.csv", index=False)
    variance.to_csv(output / "variance_results.csv", index=False)
    ranking.to_csv(output / "ranking_results.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "gate_result.json").write_text(json.dumps(gate, indent=2) + "\n")
    make_figure(summary, exact_frame, representative, variance, ranking, config, output)
    return summary
