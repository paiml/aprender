# Model Persistence (.apr Format)

Eliminate MUDA (waste) by persisting trained ML models. Skip retraining on every run.

> **Toyota Way:** 無駄 (Muda) - Waste elimination through model reuse

## Quick Start

```bash
# Save model after training
renacer -c --ml-anomaly --save-model baseline.apr -- cargo build

# Load model (no retraining)
renacer -c --ml-anomaly --load-model baseline.apr -- cargo test

# Compare against baseline
renacer -c --ml-anomaly --baseline baseline.apr -- cargo build
```

## The .apr Format

aprender's binary format with:
- **Zstd compression** - 60-80% size reduction
- **Version tracking** - Detects incompatible models
- **Metadata storage** - Hyperparameters, training info

### File Structure

```text
┌────────────────────────────────┐
│ Magic: "APR\x00"               │ 4 bytes
├────────────────────────────────┤
│ Version: u32                   │ 4 bytes
├────────────────────────────────┤
│ Model Type: u8                 │ 1 byte
├────────────────────────────────┤
│ Compression: u8                │ 1 byte
├────────────────────────────────┤
│ Metadata Length: u32           │ 4 bytes
├────────────────────────────────┤
│ Metadata (JSON)                │ variable
├────────────────────────────────┤
│ Model Data (compressed)        │ variable
└────────────────────────────────┘
```

## API Reference

### ModelMetadata

```rust,ignore
let metadata = ModelMetadata::new(1000)  // training samples
    .with_hyperparameter("n_clusters", "5")
    .with_hyperparameter("eps", "0.5")
    .with_description("Production baseline v1.0");
```

| Field | Type | Description |
|-------|------|-------------|
| `renacer_version` | String | Auto-populated |
| `trained_at` | String | Unix timestamp |
| `training_samples` | usize | Sample count |
| `hyperparameters` | HashMap | Model config |
| `description` | Option | User notes |

### PersistenceOptions

```rust,ignore
let options = PersistenceOptions::new()
    .with_compression(true)      // default: true
    .with_name("baseline-v1")
    .with_description("Release candidate");
```

### Save/Load Functions

```rust,ignore
// KMeans
save_kmeans_model(&model, "model.apr", options)?;
let model = load_kmeans_model("model.apr")?;

// IsolationForest
save_isolation_forest_model(&model, "iforest.apr", options)?;
let model = load_isolation_forest_model("iforest.apr")?;

// Validation only
let metadata = validate_model_file("model.apr")?;
```

## Error Handling

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFound` | Path doesn't exist | Check path |
| `InvalidFormat` | Not .apr file | Use correct file |
| `VersionMismatch` | Old model format | Retrain model |
| `LoadError` | Corrupted file | Retrain model |

## Performance

| Model Size | Save Time | Load Time | Compressed Size |
|------------|-----------|-----------|-----------------|
| 10 clusters | <1ms | <1ms | ~500 bytes |
| 100 clusters | ~2ms | ~1ms | ~5 KB |
| 1000 clusters | ~10ms | ~5ms | ~50 KB |

## Workflow: CI/CD Integration

```yaml
# .github/workflows/perf.yml
jobs:
  performance:
    steps:
      - name: Download baseline
        uses: actions/download-artifact@v3
        with:
          name: baseline-model

      - name: Run regression check
        run: |
          renacer -c --ml-anomaly --baseline baseline.apr \
            -- cargo build 2>&1 | tee results.txt

          if grep -q "REGRESSION" results.txt; then
            echo "::error::Performance regression detected"
            exit 1
          fi
```

## Related

- [ML Pipeline with EXTREME TDD](./ml-pipeline.md)
- [Machine Learning](./machine-learning.md)
- [Anomaly Detection](./anomaly-detection.md)
