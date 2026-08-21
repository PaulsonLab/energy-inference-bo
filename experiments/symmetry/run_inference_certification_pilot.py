"""Run the preregistered reflection-symmetry certification pilot exactly once."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from conditioned_bo.inference_certification import (
    CertificationBatchRegistry,
    ProposalCapExceeded,
    active_set_sha256,
    certify_symmetry_grid_round,
    draw_ou_reference_ar1,
    make_batch_id,
    rejection_sample_symmetry_target,
)
from conditioned_bo.symmetry_influence import (
    ei_action_coefficients,
    ei_block_decision_footprint,
    omitted_factor_load,
    ou_symmetry_comparison,
    ranked_omitted_contributions,
)


CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "inference_certification_pilot.json"
)
TEST_RECORD_PATH = (
    Path(__file__).resolve().parent
    / "configs"
    / "inference_certification_test_record.json"
)
OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "outputs"
    / "inference_certification_pilot"
)


@dataclass(frozen=True)
class ActionConditionals:
    actions: np.ndarray
    coefficients: np.ndarray
    variances: np.ndarray
    means: np.ndarray


class WorkingSymmetryFactors:
    """Cache only active factor vectors for the non-certifying working stage."""

    def __init__(
        self,
        samples: np.ndarray,
        blocks: np.ndarray,
        gamma: float,
        tau: float,
    ) -> None:
        self.samples = samples
        self.blocks = blocks
        self.gamma = gamma
        self.tau = tau
        self._cache: dict[int, np.ndarray] = {}

    def evaluate(self, factor_index: int) -> np.ndarray:
        if factor_index not in self._cache:
            left, right = self.blocks[factor_index]
            scaled = (self.samples[:, right] - self.samples[:, left]) / self.tau
            self._cache[factor_index] = self.gamma * (
                np.logaddexp(scaled, -scaled) - np.log(2.0)
            )
        return self._cache[factor_index]

    def log_weights(self, active: list[int]) -> np.ndarray:
        values = np.zeros(self.samples.shape[0], dtype=float)
        for factor_index in active:
            values -= self.evaluate(factor_index)
        return values


def reference_mean(x: np.ndarray) -> np.ndarray:
    """The committed clean-EI validation mean, not the archived notebook mean."""

    values = np.asarray(x, dtype=float)
    return 0.36 * np.exp(-0.5 * ((values + 0.22) / 0.095) ** 2) + 0.31 * np.exp(
        -0.5 * ((values - 0.25) / 0.18) ** 2
    )


def latent_grid(n_factors: int, spacing: float) -> np.ndarray:
    radii = (np.arange(n_factors) + 0.5) * spacing
    return np.concatenate((-radii[::-1], radii))


def build_action_conditionals(
    actions: np.ndarray,
    points: np.ndarray,
    lengthscale: float,
    precision: scipy.sparse.spmatrix,
) -> ActionConditionals:
    coefficients = np.empty((actions.size, points.size), dtype=float)
    variances = np.empty(actions.size, dtype=float)
    for index, action in enumerate(actions):
        coefficients[index], variances[index] = ei_action_coefficients(
            float(action), points, lengthscale, precision
        )
    if np.count_nonzero(np.abs(coefficients) > 2e-13, axis=1).max() > 2:
        raise RuntimeError("OU Markov coefficients unexpectedly lost local support")
    return ActionConditionals(
        actions=actions,
        coefficients=coefficients,
        variances=variances,
        means=reference_mean(actions),
    )


def conditional_ei_matrix(
    samples: np.ndarray,
    latent_mean: np.ndarray,
    conditionals: ActionConditionals,
    incumbent: float,
) -> np.ndarray:
    """Working-stage conditional EI matrix; never used for certification."""

    centered_samples = samples - latent_mean
    values = np.empty((samples.shape[0], conditionals.actions.size), dtype=float)
    for action_index in range(conditionals.actions.size):
        coefficients = conditionals.coefficients[action_index]
        support = np.flatnonzero(np.abs(coefficients) > 2e-13)
        conditional_mean = conditionals.means[action_index] + (
            centered_samples[:, support] @ coefficients[support]
        )
        variance = float(conditionals.variances[action_index])
        if variance == 0.0:
            values[:, action_index] = np.maximum(
                conditional_mean - incumbent, 0.0
            )
        else:
            sigma = float(np.sqrt(variance))
            centered = conditional_mean - incumbent
            z = centered / sigma
            density = np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi)
            values[:, action_index] = centered * scipy.special.ndtr(z) + sigma * density
    return values


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / weights.sum()


def _run_git(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_preregistration_state() -> str:
    """Refuse to run unless the locked config/test record are clean and committed."""

    if OUTPUT_DIRECTORY.exists():
        raise RuntimeError(
            "prospective output directory already exists; refusing to overwrite or rerun"
        )
    status = _run_git("status", "--porcelain")
    if status:
        raise RuntimeError("working tree must be clean before prospective sampling")
    for path in (CONFIG_PATH, TEST_RECORD_PATH):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        committed = subprocess.run(
            ("git", "show", f"HEAD:{relative}"),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if committed != path.read_bytes():
            raise RuntimeError(f"{relative} does not match the committed preregistration")
    return _run_git("rev-parse", "HEAD")


def validate_locked_config(config: dict[str, Any]) -> None:
    locked = {
        "n_factors": config["model"]["n_factors"],
        "action_count": config["action_grid"]["count"],
        "action_min": config["action_grid"]["minimum"],
        "action_max": config["action_grid"]["maximum"],
        "incumbent": config["decision"]["incumbent"],
        "epsilon": config["decision"]["epsilon"],
        "delta": config["decision"]["delta"],
        "activation_batch_size": config["refinement"]["activation_batch_size"],
        "r_max": config["refinement"]["maximum_rounds"],
        "working_samples": config["working_stage"]["reference_samples"],
        "working_seed": config["working_stage"]["seed"],
        "certification_root_seed": config["certification_stage"]["root_seed"],
        "proposal_chunk_size": config["certification_stage"]["proposal_chunk_size"],
        "n_low": config["certification_stage"]["sample_schedule"][
            "accepted_samples_below_threshold"
        ],
        "n_high": config["certification_stage"]["sample_schedule"][
            "accepted_samples_at_or_above_threshold"
        ],
        "max_active": config["pass_criteria"]["maximum_active_factors"],
        "max_b_infer": config["pass_criteria"][
            "maximum_b_infer_at_worst_challenger"
        ],
        "proposal_cap": config["pass_criteria"][
            "maximum_cumulative_gaussian_proposals"
        ],
        "min_acceptance": config["pass_criteria"][
            "minimum_final_acceptance_rate"
        ],
    }
    expected = {
        "n_factors": 40,
        "action_count": 401,
        "action_min": -0.58,
        "action_max": -0.06,
        "incumbent": 0.50,
        "epsilon": 0.01,
        "delta": 0.05,
        "activation_batch_size": 3,
        "r_max": 15,
        "working_samples": 80_000,
        "working_seed": 123,
        "certification_root_seed": 314159265,
        "proposal_chunk_size": 25_000,
        "n_low": 100_000,
        "n_high": 1_500_000,
        "max_active": 18,
        "max_b_infer": 0.0045,
        "proposal_cap": 20_000_000,
        "min_acceptance": 0.20,
    }
    if locked != expected:
        raise RuntimeError(f"pilot configuration is not locked: {locked!r}")


def json_seed_state(seed_sequence: np.random.SeedSequence) -> dict[str, Any]:
    state = seed_sequence.state
    return {
        "entropy": int(state["entropy"]),
        "spawn_key": [int(value) for value in state["spawn_key"]],
        "pool_size": int(state["pool_size"]),
        "n_children_spawned": int(state["n_children_spawned"]),
        "generated_state_uint32": [
            int(value) for value in seed_sequence.generate_state(4)
        ],
    }


def hardware_manifest() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def save_progress(
    history: list[dict[str, Any]],
    sampler_rows: list[dict[str, Any]],
    batch_manifest: dict[str, Any],
) -> None:
    write_csv(OUTPUT_DIRECTORY / "round_history.csv", history)
    write_csv(OUTPUT_DIRECTORY / "sampler_diagnostics.csv", sampler_rows)
    write_json(OUTPUT_DIRECTORY / "batch_manifest.json", batch_manifest)


def certificate_plot(history: list[dict[str, Any]]) -> None:
    complete = [row for row in history if row["estimated_gap"] != ""]
    if not complete:
        fig, axis = plt.subplots(figsize=(7.2, 3.8))
        axis.text(0.5, 0.5, "No complete certification round", ha="center", va="center")
        axis.axis("off")
    else:
        active_counts = np.array([row["active_count"] for row in complete])
        gaps = np.array([row["estimated_gap"] for row in complete], dtype=float)
        inference = np.array([row["b_infer"] for row in complete], dtype=float)
        structural = np.array([row["b_struct"] for row in complete], dtype=float)
        totals = np.array([row["u_cert"] for row in complete], dtype=float)
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        axis.plot(active_counts, gaps, "o-", label=r"$\widehat G_S$")
        axis.plot(active_counts, inference, "o-", label=r"$B_{\rm infer}$")
        axis.plot(active_counts, structural, "o-", label=r"$B_{\rm struct}$")
        axis.plot(active_counts, totals, "o-", linewidth=2.2, label=r"$U_{\rm cert}$")
        axis.axhline(0.01, color="black", linestyle="--", linewidth=1, label=r"$\epsilon$")
        axis.set_xlabel("Active symmetry factors")
        axis.set_ylabel("Worst-challenger certificate component")
        axis.set_title("Prospective finite-sample certificate decomposition")
        axis.grid(alpha=0.25)
        axis.legend(ncol=3, fontsize=8)
        fig.tight_layout()
    fig.savefig(OUTPUT_DIRECTORY / "certificate_decomposition.png", dpi=180)
    plt.close(fig)


def render_results(summary: dict[str, Any]) -> str:
    final = summary.get("final_round") or {}
    conditions = summary["pass_conditions"]
    condition_lines = "\n".join(
        f"- `{name}`: **{'PASS' if value else 'FAIL'}**"
        for name, value in conditions.items()
    )
    return f"""# Reflection-Symmetry Finite-Sample Certification Pilot

Prospective verdict: **{summary['pilot_verdict']}**

This was run once from preregistration commit
`{summary['preregistration_commit']}` with frozen configuration SHA-256
`{summary['config_sha256']}`. The guarantee is for this reflection-symmetry
finite action grid and this exact rejection-sampling backend only.

## Mechanical conditions

{condition_lines}

## Final reached round

- active factors: `{final.get('active_count')}` / `40`;
- omitted fraction: `{final.get('omitted_fraction')}`;
- leader: `{final.get('leader_action')}`;
- worst optimistic challenger: `{final.get('worst_challenger_action')}`;
- estimated active-target gap: `{final.get('estimated_gap')}`;
- inference bound: `{final.get('b_infer')}`;
- structural bound: `{final.get('b_struct')}`;
- certificate: `{final.get('u_cert')}`;
- final acceptance rate: `{final.get('acceptance_rate')}`;
- cumulative generated Gaussian proposals: `{summary['cumulative_proposals_generated']}`.

## Interpretation

{summary['interpretation']}

This result does not transfer the finite-sample guarantee to HMC, SMC,
FlowGP, importance sampling, or any other inference backend.
"""


def run() -> dict[str, Any]:
    preregistration_commit = verify_preregistration_state()
    config_bytes = CONFIG_PATH.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    config = json.loads(config_bytes)
    test_record = json.loads(TEST_RECORD_PATH.read_text())
    validate_locked_config(config)
    if test_record.get("status") != "PASS":
        raise RuntimeError("committed pre-pilot test record is not PASS")

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=False)
    (OUTPUT_DIRECTORY / "frozen_config.json").write_bytes(config_bytes)
    start_time = time.perf_counter()

    model = config["model"]
    grid = config["action_grid"]
    decision = config["decision"]
    refinement = config["refinement"]
    working = config["working_stage"]
    certification = config["certification_stage"]
    schedule = certification["sample_schedule"]
    criteria = config["pass_criteria"]

    n_factors = int(model["n_factors"])
    spacing = float(model["spacing"])
    lengthscale = float(model["lengthscale"])
    gamma = float(model["gamma"])
    tau = float(model["tau"])
    correlation = float(np.exp(-spacing / lengthscale))
    incumbent = float(decision["incumbent"])
    epsilon = float(decision["epsilon"])
    delta = float(decision["delta"])
    r_max = int(refinement["maximum_rounds"])
    proposal_cap = int(criteria["maximum_cumulative_gaussian_proposals"])

    precision_q, blocks, _, _, comparison_a = ou_symmetry_comparison(
        n_factors, spacing, lengthscale
    )
    points = latent_grid(n_factors, spacing)
    latent_mean = reference_mean(points)
    actions = np.linspace(
        float(grid["minimum"]), float(grid["maximum"]), int(grid["count"])
    )
    conditionals = build_action_conditionals(
        actions, points, lengthscale, precision_q
    )

    working_rng = np.random.default_rng(int(working["seed"]))
    working_samples = draw_ou_reference_ar1(
        working_rng,
        int(working["reference_samples"]),
        latent_mean,
        correlation,
    )
    working_ei = conditional_ei_matrix(
        working_samples, latent_mean, conditionals, incumbent
    )
    working_factors = WorkingSymmetryFactors(
        working_samples, blocks, gamma, tau
    )

    root_seed = np.random.SeedSequence(int(certification["root_seed"]))
    child_seeds = root_seed.spawn(r_max)
    child_metadata = [json_seed_state(child) for child in child_seeds]
    registry = CertificationBatchRegistry()
    batch_manifest: dict[str, Any] = {
        "preregistration_commit": preregistration_commit,
        "config_sha256": config_sha256,
        "root_seed": json_seed_state(root_seed),
        "planned_children": [
            {"round": index, "seed": metadata, "used": False}
            for index, metadata in enumerate(child_metadata)
        ],
        "batches": [],
    }

    active = [int(value) for value in refinement["initial_active_factors"]]
    history: list[dict[str, Any]] = []
    sampler_rows: list[dict[str, Any]] = []
    cumulative_generated = 0
    cumulative_consumed = 0
    certificate_reached = False
    cap_failure = False
    final_completed: dict[str, Any] | None = None

    for round_index in range(r_max):
        active_mask = np.zeros(n_factors, dtype=bool)
        active_mask[active] = True
        weights = normalized_weights(working_factors.log_weights(active))
        working_curve = np.asarray(weights @ working_ei, dtype=float)
        leader_index = int(np.argmax(working_curve))
        active_hash = active_set_sha256(active)
        child_state = child_metadata[round_index]
        batch_id = make_batch_id(
            config_sha256=config_sha256,
            round_index=round_index,
            child_seed_state=child_state,
            active_factors=active,
            leader_index=leader_index,
        )
        batch_manifest["planned_children"][round_index]["used"] = True

        accepted_samples = (
            int(schedule["accepted_samples_below_threshold"])
            if len(active) < int(schedule["active_count_less_than"])
            else int(schedule["accepted_samples_at_or_above_threshold"])
        )
        remaining_proposals = proposal_cap - cumulative_generated
        child_rng = np.random.default_rng(child_seeds[round_index])
        print(
            f"[CERT] round={round_index} active={len(active):2d} "
            f"leader={actions[leader_index]:.6f} requested={accepted_samples:,}",
            flush=True,
        )

        def reference_draw(
            generator: np.random.Generator, count: int
        ) -> np.ndarray:
            return draw_ou_reference_ar1(
                generator, count, latent_mean, correlation
            )

        try:
            sample_result = rejection_sample_symmetry_target(
                rng=child_rng,
                reference_draw=reference_draw,
                n_accepted=accepted_samples,
                proposal_chunk_size=int(certification["proposal_chunk_size"]),
                proposal_cap=remaining_proposals,
                blocks=blocks,
                active_factors=active,
                gamma=gamma,
                tau=tau,
            )
        except ProposalCapExceeded as error:
            cumulative_generated += error.proposals_generated
            cumulative_consumed += error.proposals_consumed
            for chunk in error.chunks:
                sampler_rows.append(
                    {
                        "round": round_index,
                        "batch_id": batch_id,
                        **chunk.to_dict(),
                    }
                )
            batch_manifest["batches"].append(
                {
                    "round": round_index,
                    "batch_id": batch_id,
                    "active_factors": list(active),
                    "active_set_hash": active_hash,
                    "leader_index": leader_index,
                    "leader_action": float(actions[leader_index]),
                    "child_seed": child_state,
                    "status": "PROPOSAL_CAP_EXCEEDED",
                    "accepted_samples": error.accepted_samples,
                    "requested_samples": error.requested_samples,
                    "proposals_generated": error.proposals_generated,
                    "proposals_consumed": error.proposals_consumed,
                }
            )
            history.append(
                {
                    "round": round_index,
                    "active_count": len(active),
                    "active_factors": json.dumps(active),
                    "active_set_hash": active_hash,
                    "leader_index": leader_index,
                    "leader_action": float(actions[leader_index]),
                    "worst_challenger_index": "",
                    "worst_challenger_action": "",
                    "accepted_samples": error.accepted_samples,
                    "requested_samples": error.requested_samples,
                    "gaussian_proposals_generated": error.proposals_generated,
                    "gaussian_proposals_consumed": error.proposals_consumed,
                    "acceptance_rate": (
                        error.accepted_candidates / error.proposals_generated
                    ),
                    "estimated_gap": "",
                    "b_infer": "",
                    "b_struct": "",
                    "u_cert": "",
                    "next_activated_factors": "[]",
                    "batch_id": batch_id,
                    "pass_fail_state": "FAIL_PROPOSAL_CAP",
                }
            )
            cap_failure = True
            save_progress(history, sampler_rows, batch_manifest)
            break

        cumulative_generated += sample_result.proposals_generated
        cumulative_consumed += sample_result.proposals_consumed
        for chunk in sample_result.chunks:
            sampler_rows.append(
                {
                    "round": round_index,
                    "batch_id": batch_id,
                    **chunk.to_dict(),
                }
            )

        omitted_load = omitted_factor_load(active_mask, gamma, tau)
        certification_result = certify_symmetry_grid_round(
            samples=sample_result.samples,
            batch_id=batch_id,
            registry=registry,
            active_factors=active,
            actions=actions,
            leader_index=leader_index,
            latent_mean=latent_mean,
            action_means=conditionals.means,
            action_coefficients=conditionals.coefficients,
            conditional_variances=conditionals.variances,
            incumbent=incumbent,
            precision_q=precision_q,
            reflection_blocks=blocks,
            comparison_matrix_a=comparison_a,
            omitted_load=omitted_load,
            r_max=r_max,
            delta=delta,
            sample_chunk_size=int(certification["proposal_chunk_size"]),
        )
        worst_index = certification_result.worst_index
        certificate_reached = certification_result.u_cert <= epsilon
        next_factors: list[int] = []
        if not certificate_reached and len(active) < n_factors:
            footprint = ei_block_decision_footprint(
                conditionals.coefficients[worst_index],
                conditionals.coefficients[leader_index],
                blocks,
            )
            scores = ranked_omitted_contributions(
                comparison_a, footprint, active_mask, gamma, tau
            )
            candidates = [
                int(index)
                for index in np.argsort(-scores, kind="stable")
                if np.isfinite(scores[index])
            ]
            next_factors = candidates[: int(refinement["activation_batch_size"])]

        challenger_path = OUTPUT_DIRECTORY / f"challenger_bounds_round_{round_index:02d}.csv"
        write_csv(challenger_path, certification_result.challenger_rows(actions))
        batch_manifest["batches"].append(
            {
                "round": round_index,
                "batch_id": batch_id,
                "active_factors": list(active),
                "active_set_hash": active_hash,
                "leader_index": leader_index,
                "leader_action": float(actions[leader_index]),
                "child_seed": child_state,
                "status": "CERTIFIED" if certificate_reached else "EXPENDED",
                "accepted_samples": accepted_samples,
                "accepted_candidates": sample_result.accepted_candidates,
                "proposals_generated": sample_result.proposals_generated,
                "proposals_consumed": sample_result.proposals_consumed,
                "acceptance_rate": sample_result.acceptance_rate,
                "challenger_file": challenger_path.name,
            }
        )
        final_completed = {
            "round": round_index,
            "active_count": len(active),
            "active_factors": list(active),
            "omitted_fraction": (n_factors - len(active)) / n_factors,
            "leader_index": leader_index,
            "leader_action": float(actions[leader_index]),
            "worst_challenger_index": worst_index,
            "worst_challenger_action": float(actions[worst_index]),
            "accepted_samples": accepted_samples,
            "proposals_generated": sample_result.proposals_generated,
            "proposals_consumed": sample_result.proposals_consumed,
            "acceptance_rate": sample_result.acceptance_rate,
            "estimated_gap": float(certification_result.estimated_gaps[worst_index]),
            "b_infer": float(certification_result.inference_bounds[worst_index]),
            "b_struct": float(certification_result.structural_bounds[worst_index]),
            "u_cert": certification_result.u_cert,
            "batch_id": batch_id,
        }
        history.append(
            {
                "round": round_index,
                "active_count": len(active),
                "active_factors": json.dumps(active),
                "active_set_hash": active_hash,
                "leader_index": leader_index,
                "leader_action": float(actions[leader_index]),
                "worst_challenger_index": worst_index,
                "worst_challenger_action": float(actions[worst_index]),
                "accepted_samples": accepted_samples,
                "requested_samples": accepted_samples,
                "gaussian_proposals_generated": sample_result.proposals_generated,
                "gaussian_proposals_consumed": sample_result.proposals_consumed,
                "acceptance_rate": sample_result.acceptance_rate,
                "estimated_gap": final_completed["estimated_gap"],
                "b_infer": final_completed["b_infer"],
                "b_struct": final_completed["b_struct"],
                "u_cert": final_completed["u_cert"],
                "next_activated_factors": json.dumps(next_factors),
                "batch_id": batch_id,
                "pass_fail_state": "CERTIFIED" if certificate_reached else "CONTINUE",
            }
        )
        print(
            f"[CERT] round={round_index} proposals={sample_result.proposals_generated:,} "
            f"accept={sample_result.acceptance_rate:.4f} "
            f"worst={actions[worst_index]:.6f} U={certification_result.u_cert:.8f}",
            flush=True,
        )
        save_progress(history, sampler_rows, batch_manifest)
        del sample_result

        if certificate_reached:
            break
        active.extend(next_factors)

    conditions = {
        "certificate_within_epsilon": bool(
            certificate_reached
            and final_completed is not None
            and final_completed["u_cert"] <= float(criteria["maximum_u_cert"])
        ),
        "active_count_at_most_18": bool(
            certificate_reached
            and final_completed is not None
            and final_completed["active_count"]
            <= int(criteria["maximum_active_factors"])
        ),
        "worst_challenger_b_infer_at_most_0_0045": bool(
            certificate_reached
            and final_completed is not None
            and final_completed["b_infer"]
            <= float(criteria["maximum_b_infer_at_worst_challenger"])
        ),
        "cumulative_generated_proposals_at_most_20m": bool(
            cumulative_generated <= proposal_cap and not cap_failure
        ),
        "final_acceptance_rate_at_least_0_20": bool(
            certificate_reached
            and final_completed is not None
            and final_completed["acceptance_rate"]
            >= float(criteria["minimum_final_acceptance_rate"])
        ),
        "required_tests_passed": test_record["status"] == "PASS",
    }
    pilot_pass = all(conditions.values())
    interpretation = (
        "The locked reflection-symmetry finite-grid pilot satisfied every "
        "predeclared condition; the end-to-end finite-sample blocker is closed "
        "for this exact instantiation."
        if pilot_pass
        else "The locked prospective pilot failed at least one predeclared "
        "condition; the end-to-end empirical blocker remains open."
    )
    summary: dict[str, Any] = {
        "pilot_verdict": "PASS" if pilot_pass else "FAIL",
        "preregistration_commit": preregistration_commit,
        "config_sha256": config_sha256,
        "test_record": test_record,
        "pass_conditions": conditions,
        "final_round": final_completed,
        "rounds_reached": len(history),
        "cumulative_proposals_generated": cumulative_generated,
        "cumulative_proposals_consumed": cumulative_consumed,
        "hardware": hardware_manifest(),
        "wall_seconds": time.perf_counter() - start_time,
        "interpretation": interpretation,
        "guarantee_scope": "reflection_symmetry_finite_grid_exact_rejection_samples",
        "not_guaranteed_backends": ["HMC", "SMC", "FlowGP", "importance_sampling"],
    }
    write_json(OUTPUT_DIRECTORY / "summary.json", summary)
    (OUTPUT_DIRECTORY / "RESULTS.md").write_text(render_results(summary))
    certificate_plot(history)
    save_progress(history, sampler_rows, batch_manifest)
    print(f"[RESULT] prospective pilot: {summary['pilot_verdict']}", flush=True)
    return summary


if __name__ == "__main__":
    run()
