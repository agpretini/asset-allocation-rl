# Legacy Synthetic Trial 371

This directory retains historical Model A compact-regime synthetic Q-learning artifacts:
18,000 episodes, visit-decay learning rate starting at 0.30, and the documented
exploration schedule. It is not the current selected production configuration;
use `../run.py` for the generic selected-model runner.

- `walk_forward/`: train-only validation and train-plus-validation test results.
- `comparison/`: per-test-split baseline tables and charts.
- `policy/`: 30,888-state greedy table, visit coverage, and heatmap.
- `forecast/`: dated full-data allocation predictions and model snapshots.

These artifacts are retained for research comparison only. The selected
production-style Model B run is `../model_b_synthetic_trial_076/`.
