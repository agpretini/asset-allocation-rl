# Model Runs

Each subdirectory is a self-contained trained-model evaluation. A run keeps its
configuration context, walk-forward outputs, comparisons, policy artifacts, and
forecast together so results are not duplicated by analytical topic.

`run.py` executes the selected Model B production-style protocol and delegates
to `production.py`, whose implementation is intentionally independent of any
trial ID. Its destination comes from `run_label` in the selected-model YAML;
use `--output-dir` to keep a different candidate isolated. Historical
trial-371 artifacts remain retained as legacy research evidence and are not
overwritten.

Every new run writes two root-level progress artifacts while it executes:

- `status.json`: current machine-readable state, stage, split (when relevant),
  and UTC update timestamp.
- `run.log`: append-only human-readable event log.

For example:

```bash
cat analysis/runs/<run_label>/status.json
tail -f analysis/runs/<run_label>/run.log
```

After full-data training, the runner also writes a finite-state
partial-dependence-style policy-response package under `policy/`: level means,
matched-context effects, an interpretation Markdown file, and an SVG profile
chart. It is an exact summary of the discrete policy, not SHAP or a causal
estimate. `render_policy_feature_effects.py` can regenerate that package from a
saved policy table without retraining.
