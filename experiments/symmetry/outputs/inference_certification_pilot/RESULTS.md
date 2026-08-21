# Reflection-Symmetry Finite-Sample Certification Pilot

Prospective verdict: **PASS**

This was run once from preregistration commit
`5da6fec6c0a645ba56f555062a3adb4139a1782d` with frozen configuration SHA-256
`c006a02581a6e586c793ce116476ccdd311509c31ece85790deedf7fdb33b639`. The guarantee is for this reflection-symmetry
finite action grid and this exact rejection-sampling backend only.

## Mechanical conditions

- `certificate_within_epsilon`: **PASS**
- `active_count_at_most_18`: **PASS**
- `worst_challenger_b_infer_at_most_0_0045`: **PASS**
- `cumulative_generated_proposals_at_most_20m`: **PASS**
- `final_acceptance_rate_at_least_0_20`: **PASS**
- `required_tests_passed`: **PASS**

## Final reached round

- active factors: `15` / `40`;
- omitted fraction: `0.625`;
- leader: `-0.2082`;
- worst optimistic challenger: `-0.2173`;
- estimated active-target gap: `-0.0003099863882800591`;
- inference bound: `0.003848511075377576`;
- structural bound: `0.005771786534479693`;
- certificate: `0.00931031122157721`;
- final acceptance rate: `0.3342708888888889`;
- cumulative generated Gaussian proposals: `5375000`.

## Interpretation

The locked reflection-symmetry finite-grid pilot satisfied every predeclared condition; the end-to-end finite-sample blocker is closed for this exact instantiation.

This result does not transfer the finite-sample guarantee to HMC, SMC,
FlowGP, importance sampling, or any other inference backend.
