# Task 04A Colab status

**Do not launch the full A100 study.** The standardized-child CPU preflight returned
`PAUSE_FULL_STUDY`: density learning was promising at n=256, but P had no q=1
decision wins over U. The guarded [Task 04A notebook](../../notebooks/task04a_colab.ipynb)
is retained for reproducibility and has full execution disabled.

## Cell order

1. Leave `REPO_REF="main"` and `RUN_FULL=False`; run setup only if reproducing checks.
2. Confirm the printed GPU name contains `A100`, PyTorch CUDA is available, and the
   full test suite passes. Stop on any identity failure.
3. Optionally reproduce the CPU preflight with `--profile preflight`; its frozen gate
   must remain unchanged.
4. Do not set `RUN_FULL=True`. A new approved contract is required before the full
   cell may be re-enabled.

The disabled full profile contains 5 seeds × 3 regimes × 4 sizes, 120 learned energy fits,
1,024 predictive test points, and 2,048 EI candidates. Runtime is empirical profiling
data; for the first run reserve a 1–3 hour A100 session rather than assuming the
millisecond CPU smoke fit times extrapolate to the high-accuracy evaluation workload.
The likely bottlenecks are truth integration and prediction metrics, not the 56-
parameter optimizer or GPU memory. The runner writes partial CSVs after every case,
so an interrupted session can resume from copied case files. Keep the browser session
open until the ZIP downloads.

## Future output policy

If a later approved contract re-enables the study, extract its ZIP locally as
`artifacts/task04a/full/`. Keep all case JSON and automatic
outputs ignored. Ask for an import audit that checks completeness, finite values,
frozen gates, timing scaling, and checksums. Only after review should compact summary,
aggregate tables, manifest, audit, and decisive figures be copied into
`results/task04a/full/` and committed. The notebook never authenticates to or pushes
to GitHub.
