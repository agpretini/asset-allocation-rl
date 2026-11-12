# Configuration

YAML files define assets, simulation parameters, and the fixed selected-model
configuration. `paths.py` centralizes repository locations for the canonical
dataset and analytical output roots.

`models.py` loads named operational YAML files for the API and the standard
prediction runner. Both default to `selected_synthetic_model.yaml` (trial 169).
Each operational YAML must contain `episodes` and `training_mode`, plus any
`QLearningTrainingConfig` overrides and an optional `feature_mode` (default:
`compact_regime`). `run_label` and `walk_forward_splits` are research metadata
and are excluded from training parameters. HPO search-space YAMLs cannot be
used as individual model configurations.
