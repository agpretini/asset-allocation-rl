# Model B Synthetic HPO Runner

`run.py` runs the Model B synthetic-only HPO. By default it uses the learning
and exploration grid recorded in `hpo_mode_comparison_001`, replaces its
episode counts with `8000`, `22000`, `42000`, and `54000`, and creates a new
resumable `hpo_model_b_synthetic_*` directory under
`analysis/hyperparameter_search/`.

Model B uses the market-only compact state and cost-aware historical evaluation.
The reference experiment is never modified.

```bash
caffeinate -i uv run python experiments/mode_comparison/run.py
```

The default grid has 216 trials: 4 episode counts × 3 learning rates × 2
learning-rate schedules × 3 epsilon values × 3 epsilon decays, all synthetic.
Use `--dry-run` to inspect it and `--resume-dir <directory>` to resume an
interrupted run.
