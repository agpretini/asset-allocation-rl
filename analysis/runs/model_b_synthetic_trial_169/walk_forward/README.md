# Walk-Forward Evaluation

The CSV records two chronological splits. Validation trains only on the train
window; test retrains from scratch on train plus validation. Synthetic episodes
are training-only, while both reported phases use real historical observations
and the frozen cost-aware Model B policy.
