# Analysis Flags

Command-line flags for ML anomaly detection and model persistence.

## ML Anomaly Detection

| Flag | Description | Default |
|------|-------------|---------|
| `--ml-anomaly` | Enable ML-based anomaly detection | disabled |
| `--ml-clusters N` | Number of KMeans clusters | 3 |
| `--ml-compare` | Compare ML vs z-score results | disabled |

### Examples

```bash
# Basic ML anomaly detection
renacer -c --ml-anomaly -- cargo build

# Custom cluster count
renacer -c --ml-anomaly --ml-clusters 5 -- ./app

# Compare with z-score
renacer -c --ml-anomaly --ml-compare -- ./app
```

## Model Persistence (Sprint 48)

| Flag | Description | Example |
|------|-------------|---------|
| `--save-model FILE` | Save trained model to .apr | `--save-model baseline.apr` |
| `--load-model FILE` | Load pre-trained model | `--load-model baseline.apr` |
| `--baseline FILE` | Compare against baseline | `--baseline release-1.0.apr` |

### Examples

```bash
# Save model after training
renacer -c --ml-anomaly --save-model baseline.apr -- cargo build

# Load existing model (skip training)
renacer -c --ml-anomaly --load-model baseline.apr -- cargo test

# Regression detection
renacer -c --ml-anomaly --baseline baseline.apr -- cargo build
```

### Output with --save-model

```text
=== ML Anomaly Detection Report ===
Clusters: 3
Silhouette Score: 0.847

Model saved: baseline.apr
  - Training samples: 47 syscalls
  - Compression: Zstd
  - Size: 1.2 KB
```

### Output with --baseline

```text
=== Regression Analysis ===
Baseline: baseline.apr (v0.6.3, 47 samples)
Current:  52 syscalls

New anomalies not in baseline:
  - futex (avg: 1250µs) - REGRESSION

Silhouette change: 0.847 → 0.723 (-14.6%)
```

## Statistical Analysis

| Flag | Description | Default |
|------|-------------|---------|
| `--anomaly-realtime` | Real-time z-score monitoring | disabled |
| `--anomaly-threshold N` | Z-score threshold | 2.0 |

## Related

- [ML Pipeline](../advanced/ml-pipeline.md) - Detailed ML documentation
- [Model Persistence](../advanced/model-persistence.md) - .apr format details
- [Machine Learning](../advanced/machine-learning.md) - KMeans basics
