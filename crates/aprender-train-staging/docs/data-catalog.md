# Data Catalog

## Training Datasets

| Dataset | Format | Classification | Purpose | Lifecycle |
|---------|--------|---------------|---------|-----------|
| Pre-tokenized Parquet | `.parquet` | Internal | Pretraining | Retained during training |
| Validation Split | `.parquet` | Internal | Evaluation | Retained during training |
| Tokenizer Vocabulary | `.json` | Public | Tokenization | Permanent |

## Model Artifacts

| Artifact | Format | Classification | Purpose | Lifecycle |
|----------|--------|---------------|---------|-----------|
| Model Weights | `.safetensors` | Confidential | Inference | Versioned |
| Optimizer State | `.safetensors` | Internal | Training resume | Temporary |
| Training Config | `.yaml` | Public | Reproducibility | Permanent |
| Checkpoints | `.safetensors` | Confidential | Fault tolerance | Pruned to max_checkpoints |

## Data Classification Levels

- **Public**: Can be freely shared (configs, schemas, documentation)
- **Internal**: Organization-internal (training data, logs)
- **Confidential**: Restricted (model weights, embeddings)
- **Sovereign**: Subject to data residency requirements
