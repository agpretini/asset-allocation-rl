# Historical Comparisons

Each `test_split_<id>_metrics.csv` compares the selected Model B policy with
passive and rule-based baselines on the same real historical test window.
`test_split_<id>_cumulative_paths.csv` and its SVG show normalized cumulative
portfolio value. Test windows overlap, so they must be read split by split and
not stitched into one aggregate path.

`walk_forward_model_metrics.csv` is the model-only copy of the validation/test
metrics used as input to these comparisons.
