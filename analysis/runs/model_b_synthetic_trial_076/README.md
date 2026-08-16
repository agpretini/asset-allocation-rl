# Model B Synthetic Trial 076 (Selected Product Model)

This directory contains the complete production-style analysis for trial 76
from `analysis/hyperparameter_search/hpo_model_b_synthetic_001`.

The fixed configuration is 22,000 synthetic episodes, constant learning rate
0.10, epsilon 0.30 with decay 0.95, discount factor 0.95, seed 42, and the
compact market-only state. The reward is the existing CRRA net-return reward;
the deployed policy scores each target as Q-value less immediate rebalance
cost, then publishes the modal target across the 286 permitted origin
allocations.

Run the complete analysis from the repository root:

```bash
caffeinate -i uv run python analysis/runs/run.py
```

The runner writes four self-contained groups of artifacts:

- `walk_forward/`: validation (train only) and historical test
  (train plus validation) performance for both walk-forward splits.
- `comparison/`: each real test window compared with passive and rule-based
  baselines, including cumulative-return SVGs and metric tables.
- `policy/`: the 108-market-state modal policy, state/action visit coverage,
  the all-regime policy heatmap, and the latest-market sensitivity chart over
  every starting allocation.
- `forecast/2026-08/`: full-data Model B snapshot, training history, and the
  August 2026 decision-support allocation signal.

The generic runner derives this destination from `run_label` in
`src/config/selected_synthetic_model.yaml`; it contains no trial-specific path
or implementation name. Pass `--output-dir` only when deliberately evaluating
another candidate in a separate directory.

The next invocation will also create `status.json` and `run.log` at this
directory's root. They show the live stage and split as the four walk-forward
training/evaluation phases progress. They cannot be added retrospectively to a
process that was already running when this feature was introduced.

Trial 76 is the selected product configuration. Its completed HPO ranking was
calculated before the cost-aware Model B evaluator was fixed; this committed
production analysis uses the corrected cost-aware policy. It is therefore the
authoritative record for this selected configuration, not an untouched final
holdout claim.
