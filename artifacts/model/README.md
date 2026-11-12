# Saved models

Each successful retraining writes a `run_*/` snapshot containing
`trained_model.json`, `monthly_dataset.parquet`, and `metadata.json`.
`current.json` points to the active snapshot and is replaced only after all
files have been saved. Failed saves remove their incomplete snapshot.

Snapshots are retained locally. To remove an old snapshot manually, keep the
one named in `current.json` and any snapshots referenced by predictions you
want to reproduce. Generated files are ignored by Git.
