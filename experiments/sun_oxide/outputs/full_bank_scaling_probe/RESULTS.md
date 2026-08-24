# Full-bank scaling probe — DEVELOPMENT ONLY

Development classification: `FULL_ARCHIVE_NOT_HELPFUL`.

This is not scientific evidence or a fresh-seed preregistration. It used
only the three fixed, already-consumed seed-0 FULL_PBE states. Seeds
12--31 were not accessed, and the immutable failed adaptive smokes were
not modified or reinterpreted.

- Implementation SHA: `aa39cb19b1198bd433a70d63e00dc0dc39ec1fb1`
- GitHub `main` baseline SHA: `d3318151ebd8fa6b317d31a69c9f1fa508218758`
- Config SHA-256: `046c9358f06d16377e790cf4af112be37d5c9458669c0c1b7a31b3a23b9af392`
- Peak process RSS: `0.425639936` GB

`PASS_PBE_VALUE` remains valid. The decision-reset adaptive smoke remains
`ADAPTIVE_ENGINEERING_PATHOLOGICAL`. Fresh seeds 12--31 have never been
scientifically executed, and the earlier fresh preregistration at
`70a9686b143c09f9f970306cc4489a2ce2b6e173` remains superseded and must not
be run.

## Support, theory, and signal

| m | strict factors | tie pairs omitted | omega | max weighted degree | lambda_min(A0) | action/support Spearman |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | 124718 | 32 | 0.002004008016 | 1 | 0.9003424447 | 0.988180 / 0.989744 |
| 1000 | 499361 | 139 | 0.001001001001 | 1 | 0.8583029303 | 0.985232 / 0.987452 |
| 2142 | 2292440 | 571 | 0.0004670714619 | 1 | 0.7520475515 | 0.971347 / 0.975638 |

All three A0 operators remained SPD above the existing analytic 0.75
floor. No dense Menz inverse was used in routine calculations.

## FULL runtime scaling

| m | state | FULL leader | MAP s | factor E/G s | Hessian s | Cholesky s | variance s | total s | peak RSS GB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | seed_0_initial | 13 | 0.2202 | 0.2143 | 0.0030 | 0.0005 | 0.0006 | 0.2250 | 0.2084 |
| 500 | seed_0_after_6_queries | 134 | 0.2399 | 0.2333 | 0.0030 | 0.0005 | 0.0005 | 0.2449 | 0.2084 |
| 500 | seed_0_after_12_queries | 133 | 0.1910 | 0.1832 | 0.0032 | 0.0005 | 0.0006 | 0.1963 | 0.2084 |
| 1000 | seed_0_initial | 13 | 0.8254 | 0.8080 | 0.0115 | 0.0022 | 0.0017 | 0.8442 | 0.2795 |
| 1000 | seed_0_after_6_queries | 134 | 0.8862 | 0.8676 | 0.0111 | 0.0024 | 0.0017 | 0.9049 | 0.2853 |
| 1000 | seed_0_after_12_queries | 133 | 0.8644 | 0.8464 | 0.0116 | 0.0024 | 0.0018 | 0.8838 | 0.2887 |
| 2142 | seed_0_initial | 10 | 5.5143 | 5.4887 | 0.0543 | 0.0226 | 0.0094 | 5.6256 | 0.3171 |
| 2142 | seed_0_after_6_queries | 71 | 5.4713 | 5.4436 | 0.0561 | 0.0191 | 0.0087 | 5.5806 | 0.3171 |
| 2142 | seed_0_after_12_queries | 10 | 3.2600 | 3.2434 | 0.0540 | 0.0178 | 0.0092 | 3.3603 | 0.3171 |

## Structural scaling at epsilon 0.02

| m | median pre-fallback count | fraction | final max-Psi | stage leader agreement | first-agreement fraction | ADAPT/FULL time | ADAPT/FULL factor work | fallbacks |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 116195 | 0.931662 | 0.084635 | 0.925926 | 0.925071 | 3.4565 | 3.0677 | 3/3 |
| 1000 | 474338 | 0.949890 | 0.059355 | 0.925926 | 0.942442 | 4.3244 | 4.0389 | 3/3 |
| 2142 | 2230019 | 0.972771 | 0.064963 | 0.888889 | 0.969701 | 4.1526 | 3.9278 | 3/3 |

From m=500 to 2142, total factors grew by `18.3810x`, while the median pre-fallback active count grew by `19.1920x`. The implied active-count power versus m is `2.0307`.

## Epsilon sensitivity at the middle state

| m | epsilon | active fraction | stages | fallback | shadow agreement | EI regret | conditioning s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 0.05 | 0.858248 | 8 | True | True | 0 | 0.8965 |
| 500 | 0.10 | 0.740430 | 8 | True | True | 0 | 0.8893 |
| 1000 | 0.05 | 0.895591 | 8 | True | True | 0 | 4.0920 |
| 1000 | 0.10 | 0.800467 | 8 | True | True | 0 | 4.2419 |
| 2142 | 0.05 | 0.943380 | 8 | True | True | 0 | 27.2238 |
| 2142 | 0.10 | 0.888722 | 8 | True | True | 0 | 23.4899 |

## Interpretation

1. **Mathematical health.** The 2,142-support operator remained healthy:
   `lambda_min(A0)=0.7520475515` exceeded the existing analytic `0.75` floor,
   its eigen residual was `2.57e-13`, and every implicit influence-solve
   residual was at most `9.86e-12`. No new theorem or dense Menz inverse was
   introduced.
2. **PBE signal.** The normalized full archive retained a strong,
   nondegenerate target-blind signal. On actions/support, Spearman correlation
   was `0.971347/0.975638`, strict-order accuracy was `0.926702/0.930222`, MAP
   standard deviation was `0.155058/0.185705`, range was
   `0.672735/0.798877`, and top-minus-bottom-decile contrast was
   `0.484354/0.573184`.
3. **FULL scaling.** Median FULL conditioning time rose from `0.2250` s at
   500 to `0.8838` s at 1,000 and `5.5806` s at 2,142: `3.93x` and `24.80x`
   the 500-support median on this machine.
4. **Active-count scaling.** Median pre-fallback active counts were `116195`,
   `474338`, and `2230019`. The 2,142 count was `19.19x` the 500 count while
   total factors grew `18.38x`; fitted active-count power versus support size
   was `2.0307`, not subquadratic over these three sizes.
5. **Active-fraction scaling.** Median pre-fallback fractions increased from
   `0.931662` to `0.949890` to `0.972771`; enlarging the bank made the relative
   sparsity worse.
6. **Action agreement.** Median smallest recorded agreement fractions were
   `0.925071`, `0.942442`, and `0.969701`. At 2,142 support, all three active
   leaders first matched FULL after the first bound-selected batch, at
   fractions `0.969912`, `0.969701`, and `0.964427`; all later active regrets
   were zero. The frozen batch rule jumps directly from an empty bank to these
   fractions, so this probe does not identify the unobserved minimum subset
   inside that first batch.
7. **Immediate blocker.** The theorem-backed structural envelope and its batch
   rule are the immediate engineering blocker: they force a near-full first
   fit, and none of the nine primary states certified after eight stages.
   Because the protocol did not evaluate intermediate subsets inside that
   first batch, these results do not prove that the intrinsically
   decision-relevant PBE information itself is dense. They do show that the
   current certified algorithm cannot exploit any such sparsity.
8. **Tolerance sensitivity.** At the 2,142 middle state, epsilon `0.05` and
   `0.10` reduced the pre-fallback fractions to `0.943380` and `0.888722`, but
   both still exhausted eight stages and full-fallbacked. They did not create
   a conditioning-time saving and are not new scientific settings.
9. **Fresh-validation path.** There is no credible fresh adaptive E3 path from
   merely replacing the 500-support bank with all 2,142 materials under the
   current certificate. Any future path first needs a separate development
   plan that tightens or diagnoses the structural activation step; fresh seeds
   must remain unspent.

Therefore the development classification is `FULL_ARCHIVE_NOT_HELPFUL`. This
is not a paper verdict or permission to spend fresh seeds. The raw stage CSV
preserves the exact point at which every active leader first matched shadow
FULL and the remaining theorem-backed envelope.
