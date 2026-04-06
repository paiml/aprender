# Introduction

**Pacha** is a unified registry for machine learning artifacts—models, datasets, and training recipes—with full lineage tracking, semantic versioning, and cryptographic integrity.

## The Problem

Machine learning projects face critical challenges in artifact management:

1. **Model Versioning** - How do you track which version of a model is in production?
2. **Dataset Provenance** - Where did your training data come from? How was it processed?
3. **Reproducibility** - Can you recreate a model from 6 months ago exactly?
4. **Lineage Tracking** - Was this model fine-tuned from another? Quantized? Pruned?

## The Solution

Pacha provides a content-addressed registry that:

- **Deduplicates** artifacts using BLAKE3 hashing
- **Versions** models, datasets, and recipes with semantic versioning
- **Documents** artifacts with Model Cards and Datasheets
- **Tracks** lineage through fine-tuning, distillation, quantization, and merging
- **Verifies** integrity with cryptographic hashes

## Core Principles

### Toyota Way Methodology

Pacha follows Toyota Way principles:

- **Muda (Waste Elimination)** - Content-addressed deduplication eliminates duplicate storage
- **Jidoka (Built-in Quality)** - Cryptographic integrity verification catches corruption
- **Kaizen (Continuous Improvement)** - Incremental lineage tracking enables iterative improvement

### Sovereign AI Stack

Pacha integrates with the Pragmatic AI Labs Sovereign AI Stack:

| Component | Purpose | Format |
|-----------|---------|--------|
| **alimentar** | Data loading | `.ald` encrypted |
| **aprender** | Model training | `.apr` encrypted |
| **entrenar** | Training pipelines | TOML configs |
| **realizar** | Model serving | gRPC/REST |
| **pacha** | Artifact registry | SQLite + CAS |

## Quick Start

```bash
# Initialize registry
pacha init

# Register a model
pacha model register fraud-detector model.apr -v 1.0.0 -d "Fraud detection model"

# Register a dataset
pacha data register transactions data.ald -v 1.0.0 -p "Transaction data for fraud detection"

# Check model stage
pacha model get fraud-detector -v 1.0.0

# Promote to production
pacha model stage fraud-detector -v 1.0.0 -t production
```

## Architecture Overview

```
~/.pacha/
├── registry.db      # SQLite metadata
└── objects/         # Content-addressed storage
    ├── ab/
    │   └── cdef1234...
    ├── cd/
    │   └── ef5678...
    └── ...
```

## Key Features

### Model Registry
- Semantic versioning (MAJOR.MINOR.PATCH)
- Model Cards (Mitchell et al., 2019)
- Lifecycle stages (development → staging → production → archived)
- Lineage tracking (fine-tuning, distillation, quantization, merging)

### Data Registry
- Dataset versioning
- Datasheets (Gebru et al., 2021)
- Provenance tracking (W3C PROV-DM)

### Recipe Registry
- Training recipes in TOML
- Hyperparameter specifications
- Environment dependencies
- Hardware requirements

### Experiment Tracking
- Run tracking with metrics
- Artifact association
- Git commit tracking
- Duration and status logging

## Next Steps

- [What is Pacha?](./methodology/what-is-pacha.md) - Deep dive into the design philosophy
- [Quick Start](./examples/quick-start.md) - Hands-on tutorial
- [CLI Reference](./cli/installation.md) - Complete command reference
