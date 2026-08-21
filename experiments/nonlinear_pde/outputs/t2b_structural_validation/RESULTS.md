# T2-B Nonlinear-PDE Structural Validation

Status: **PASS**

This artifact validates the accepted nonlinear-PDE comparison construction. It
does not evaluate factor energies, perform posterior inference, or establish a
finite-sample end-to-end action certificate.

## Matrix regression

- minimum `rho`: `4.6333333333`;
- maximum `kappa`: `0.8666666667`;
- minimum row-dominance margin: `2.2130666667`;
- `lambda_min(A)`: `2.2366796511`;
- `lambda_max(A)`: `9.1204570358`;
- `cond_2(A)`: `4.0776769402`;
- maximum nonzeros per row: `13`;
- maximum absolute clean/notebook matrix difference: `1.110e-16`.

## Locked structural replay

- active factors: `40/576`;
- leader/challenger: `(14, 12)` / `(9, 12)`;
- structural value: `0.03874403301354686`;
- literal notebook-construction value: `0.03874403301354685`;
- locked replay value: `0.03874403301354687`;
- rigorous correction factor: `1.0`.

The theorem-backed structural term is separate from the archived empirical
inference allowance `0.0235285003` and total
envelope `0.0495224797`. Inference-error
certification remains open.
