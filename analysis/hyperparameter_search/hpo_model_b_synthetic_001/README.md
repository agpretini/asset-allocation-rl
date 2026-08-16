# Model B Synthetic HPO 001

This was a synthetic-only Model B search based on the grid defined by
`hpo_mode_comparison_001`, with episode counts 8,000, 22,000, 42,000, and
54,000. It produced 216 trials.

Trial 76 (22,000 episodes, learning rate 0.10 constant, epsilon 0.30, epsilon
decay 0.95) was selected for the product analysis run. The generated HPO CSVs
are intentionally ignored: only the selected model's production artifacts are
versioned under `analysis/runs/model_b_synthetic_trial_076/`.
