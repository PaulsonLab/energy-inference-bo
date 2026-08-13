# Task 03A Colab summary

**Decision: NO_GO**

The full package contains 240 primary method rows and 30 configurations.

## Eight completion questions

1. Cross-fit provenance is enforced in code and all PIT clamp counts/times are recorded.
2. Gaussian safety is reported by Gate A below.
3. Warped predictive flexibility against all five reference/calibration baselines is Gate B.
4. Paired n=64 decision wins and absolute/relative regret gains are in Gate B diagnostics.
5. NUTS decision quality and fully charged cost are Gate C.
6. Residual optimization fractions are included in Gate C.
7. Evidence-dependent correction behavior is Gate D.
8. Numerical identities are gated by the notebook test suite; GPU/CPU profiling is in the manifest and device profile.

## Frozen gate evaluation

```json
{
  "eligible": true,
  "decision": "NO_GO",
  "qualifying_energy": null,
  "gates": {
    "A": false,
    "B": false,
    "C": false,
    "D": false
  },
  "diagnostics": {
    "A_safety": {
      "E0": {
        "mean_excess_nll": -0.25713584751171475,
        "median_regret_increase": 0.0,
        "regret_degradation_over_0_10": 1,
        "median_correction_kl": 0.04433477078228802,
        "passed": false
      },
      "E1": {
        "mean_excess_nll": -0.32693339931637394,
        "median_regret_increase": 0.0,
        "regret_degradation_over_0_10": 1,
        "median_correction_kl": 0.065819284734639,
        "passed": false
      }
    },
    "B_flexibility": {
      "qualifier": null,
      "variants": {
        "E1": {
          "nll": {
            "32": {
              "strongest": "B4",
              "baseline_mean_nll": 1.741728125991326,
              "energy_mean_nll": 2.8525746533136194,
              "gain": -1.1108465273222934
            },
            "64": {
              "strongest": "B4",
              "baseline_mean_nll": 0.28366628475573175,
              "energy_mean_nll": 0.3557173813543583,
              "gain": -0.07205109659862657
            }
          },
          "n64_regret_absolute_gain": 0.025855170772757452,
          "n64_regret_relative_gain": 0.0860966837257254,
          "paired_seed_wins": 1,
          "passed": false
        },
        "E0": {
          "nll": {
            "32": {
              "strongest": "B4",
              "baseline_mean_nll": 1.741728125991326,
              "energy_mean_nll": 2.9087038898632445,
              "gain": -1.1669757638719185
            },
            "64": {
              "strongest": "B4",
              "baseline_mean_nll": 0.28366628475573175,
              "energy_mean_nll": 0.3437071482027163,
              "gain": -0.06004086344698456
            }
          },
          "n64_regret_absolute_gain": 0.025855170772757452,
          "n64_regret_relative_gain": 0.0860966837257254,
          "paired_seed_wins": 1,
          "passed": false
        }
      }
    },
    "C_quality_cost": {
      "passed": false
    },
    "D_unlocking": {
      "passed": false
    }
  },
  "structural_only": {
    "regret_condition": false,
    "median_final_map_to_nuts_ratio": 0.23634270973260796,
    "passed": false
  }
}
```

This automatically generated report requires local import audit before any next task. It does not authorize Task 03B by itself.
