# Model B Synthetic Trial 169 (Selected Product Model)

This is the selected Model B product configuration: 54,000 synthetic episodes,
constant learning rate 0.05, epsilon 0.50 with decay 0.95, discount factor
0.95, and seed 42. The learned state has 108 market regimes; the published
target applies Model B's cost-aware modal decision rule.

Use the generic runner from repository root:

```bash
caffeinate -i uv run python analysis/runs/run.py
```

- `walk_forward/`: validation and real historical test results.
- `comparison/`: per-test-split comparisons against passive and rule-based
  baselines.
- `policy/`: policy map, visit coverage, current-market sensitivity, and
  matched-context policy-response artifacts.
- `forecast/`: dated full-data decision-support signals.

The HPO ranking preceded the corrected cost-aware evaluator. This directory is
the authoritative evaluation record for the selected configuration under the
corrected policy, not an untouched final holdout claim.
