"""CLI for the frozen constrained-batch shift diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from decision_tilt.constrained import load_protocol
from decision_tilt.constrained_experiment import (
    aggregate_results,
    environment_record,
    package_results,
    run_state,
    smoke_report,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "preflight", "state", "full", "aggregate", "package", "status"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "constrained_batch_shift")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--archive", type=Path, default=ROOT / "constrained_batch_shift_gpu_results.zip")
    return parser


def main() -> None:
    args = _parser().parse_args()
    protocol, protocol_hash = load_protocol(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "status":
        complete = [seed for seed in protocol["states"]["seeds"] if (args.output_dir / f"state_{seed}" / "state_summary.json").exists()]
        print(json.dumps({"protocol_hash": protocol_hash, "complete": complete, "remaining": [s for s in protocol["states"]["seeds"] if s not in complete]}, indent=2))
        return
    if args.mode == "smoke":
        summary = run_state(args.config, args.output_dir / "smoke", protocol["states"]["seeds"][0], "smoke", "cpu")
        print(smoke_report(summary))
        return
    if args.mode == "preflight":
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("GPU preflight requires CUDA")
        print("=" * 60 + "\nGPU PREFLIGHT\n" + "=" * 60)
        print("[1/5] Loading frozen state ...", end=" ", flush=True)
        summary = run_state(args.config, args.output_dir / "preflight", protocol["states"]["seeds"][0], "preflight", "cuda")
        print("OK\n[2/5] Gaussian posterior/QMC ... OK\n[3/5] Student-t posterior/QMC ... OK\n[4/5] Acquisition + gradients ... OK\n[5/5] Checkpoint write/read ... OK")
        marker = {"passed": True, "protocol_hash": protocol_hash, "environment": environment_record(torch.device("cuda")), "summary": summary}
        from decision_tilt.constrained import atomic_write_json
        atomic_write_json(args.output_dir / "preflight_pass.json", marker)
        print("\nGPU PREFLIGHT: PASS\nSafe to run FULL EXPERIMENT.\n" + "=" * 60)
        return
    if args.mode == "state":
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Full frozen state execution requires CUDA")
        if args.seed is None or args.seed not in protocol["states"]["seeds"]:
            raise ValueError("--seed must be one frozen state seed")
        print(json.dumps(run_state(args.config, args.output_dir, args.seed, "full", args.device), indent=2))
        return
    if args.mode == "full":
        if args.device != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Full frozen experiment requires CUDA")
        marker = args.output_dir / "preflight_pass.json"
        if not marker.exists() or json.loads(marker.read_text()).get("protocol_hash") != protocol_hash:
            raise RuntimeError("Full run refused: matching GPU preflight has not passed")
        print("=" * 60 + f"\nCONSTRAINED BATCH SHIFT — FULL RUN\nProtocol: {protocol_hash}\nStates: 8\n" + "=" * 60)
        for index, seed in enumerate(protocol["states"]["seeds"], 1):
            print(f"\n[STATE {index}/8] seed={seed}")
            summary = run_state(args.config, args.output_dir, seed, "full", args.device)
            print(f"State {index} summary\n----------------")
            for belief in ("gaussian", "student_t"):
                values = summary["beliefs"][belief]
                qmc = values.get("qmc_512", {})
                print(f"{belief}: ESS median={values['top_decile_median_ess_fraction']:.4g}; 512 ranking disagreement={qmc.get('mean_high_pairwise_ranking_disagreement', float('nan')):.4g}; gradient cosine={qmc.get('median_high_gradient_cosine', float('nan')):.4g}")
            print(f"Saved: {args.output_dir / f'state_{seed}'}\nState complete.")
        result = aggregate_results(args.config, args.output_dir)
        print("\nAll states complete. Mechanical aggregate (requires final audit):")
        print(json.dumps(result, indent=2))
        return
    if args.mode == "aggregate":
        print(json.dumps(aggregate_results(args.config, args.output_dir), indent=2))
        return
    if args.mode == "package":
        archive, digest = package_results(args.config, args.output_dir, args.archive)
        print(json.dumps({"archive": str(archive), "sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
