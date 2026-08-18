# Notebooks

Notebooks are readable scientific records, not thin launchers. Reusable functions belong in `src/decision_tilt/`; notebooks own figures, numerical summaries, and interpretation.

Every substantive notebook must contain sections for the question, paper relevance, mathematics, frozen protocol, GO/NO-GO expectation, results, interpretation, and next action. Colab notebooks must detect CPU/GPU, print device and package-version metadata, and run top-to-bottom without manual editing.

- [`rare_mode_mechanism.ipynb`](rare_mode_mechanism.ipynb): executed, CPU-compatible scientific record for the completed synthetic mechanism experiment.
- [`constrained_batch_shift_colab.ipynb`](constrained_batch_shift_colab.ipynb): guarded, Drive-resumable A100 driver and human-readable protocol for the frozen constrained-batch diagnostic.
- [`welded_beam_shift.ipynb`](welded_beam_shift.ipynb): executed CPU-compatible record for the completed three-state q=1 Welded Beam diagnostic.
