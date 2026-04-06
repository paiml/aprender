# Training Data

## Datasets

| File | Format | Purpose |
|------|--------|---------|
| `train.parquet` | Parquet | Training data |
| `val.parquet` | Parquet | Validation data |
| `pretokenized-2048/` | Parquet | Pre-tokenized sequences (seq_len=2048) |

## Data Versioning

Data is tracked via `inventory.yaml` with retention policies.
Model checkpoints are managed by entrenar's checkpoint pruning.

## Classification

All training data is classified PUBLIC (open datasets).
Model weights are classified INTERNAL.
