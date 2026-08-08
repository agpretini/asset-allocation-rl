# Selected Synthetic Trial 371

This is the selected compact-regime synthetic Q-learning configuration:
18,000 episodes, visit-decay learning rate starting at 0.30, and the documented
exploration schedule. `run.py` can run the complete protocol or refresh only
policy/comparison artifacts from the saved full-data model.

- `walk_forward/`: train-only validation and train-plus-validation test results.
- `comparison/`: per-test-split baseline tables and charts.
- `policy/`: 30,888-state greedy table, visit coverage, and heatmap.
- `forecast/`: dated full-data allocation predictions and model snapshots.

The trial was selected using HPO test metrics, so these results are operational
analysis rather than an unbiased final holdout estimate.
