# Saved predictions

Every successful `POST /predict` writes a uniquely named JSON containing the
target month, allocation, policy diagnostics, model reference and metadata,
creation time, output path, and confirmation message. Repeated requests do not
overwrite earlier predictions. Generated JSON files are ignored by Git.
