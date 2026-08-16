# Walk-Forward Evaluation

`walk_forward_train_validation_test.csv` records two chronological splits.
For each split, validation trains only on its train window; test retrains from
scratch on train plus validation. Synthetic simulator parameters are fitted
only on the corresponding historical training window, while validation and
test observations are always real historical data.

The committed results use the selected Model B trial 76 configuration and the
cost-aware frozen decision policy.
