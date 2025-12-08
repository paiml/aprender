# Case Study: APR Model Inspection

This example demonstrates the inspection tooling for `.apr` model files, following the Toyota Way principle of Genchi Genbutsu (go and see).

## Overview

The inspection module provides comprehensive tooling to analyze `.apr` model files:
- Header inspection (magic, version, flags, compression)
- Metadata extraction (hyperparameters, training info, license)
- Weight statistics with health assessment
- Model diff for version comparison

## Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| Genchi Genbutsu | Go and see - inspect actual model data |
| Visualization | Make problems visible for debugging |
| Jidoka | Built-in quality checks with health assessment |

## Running the Example

```bash
cargo run --example apr_inspection
```

## Header Inspection

Inspect the binary header of `.apr` files:

```rust
let mut header = HeaderInspection::new();
header.version = (1, 2);
header.model_type = 3;  // RandomForest
header.compressed_size = 5 * 1024 * 1024;
header.uncompressed_size = 12 * 1024 * 1024;

println!("Compression Ratio: {:.2}x", header.compression_ratio());
println!("Header Valid: {}", header.is_valid());
```

### Header Flags

| Flag | Description |
|------|-------------|
| compressed | Model weights are compressed |
| signed | Ed25519 signature present |
| encrypted | AES-256-GCM encryption |
| streaming | Supports streaming loading |
| licensed | License restrictions apply |
| quantized | Weights are quantized |

## Metadata Inspection

Extract model metadata including hyperparameters and provenance:

```rust
let mut meta = MetadataInspection::new("RandomForestClassifier");
meta.n_parameters = 50_000;
meta.n_features = 13;
meta.n_outputs = 3;

meta.hyperparameters.insert("n_estimators".to_string(), "100".to_string());
meta.hyperparameters.insert("max_depth".to_string(), "10".to_string());
```

### Training Info

Track training provenance for reproducibility:

```rust
meta.training_info = Some(TrainingInfo {
    trained_at: Some("2024-12-08T10:30:00Z".to_string()),
    duration: Some(Duration::from_secs(120)),
    dataset_name: Some("iris_extended".to_string()),
    n_samples: Some(10000),
    final_loss: Some(0.0234),
    framework: Some("aprender".to_string()),
    framework_version: Some("0.15.0".to_string()),
});
```

## Weight Statistics

Analyze model weights for health issues:

```rust
let stats = WeightStats::from_slice(&weights);

println!("Count: {}", stats.count);
println!("Min: {:.4}", stats.min);
println!("Max: {:.4}", stats.max);
println!("Mean: {:.4}", stats.mean);
println!("Std: {:.4}", stats.std);
println!("NaN Count: {}", stats.nan_count);  // CRITICAL if > 0
println!("Inf Count: {}", stats.inf_count);  // CRITICAL if > 0
println!("Sparsity: {:.2}%", stats.sparsity * 100.0);
println!("Health: {:?}", stats.health_status());
```

### Health Status Levels

| Status | Description |
|--------|-------------|
| Healthy | All weights finite, reasonable distribution |
| Warning | High sparsity or unusual distribution |
| Critical | Contains NaN or Infinity values |

## Model Diff

Compare two model versions:

```rust
let mut diff = DiffResult::new("model_v1.apr", "model_v2.apr");

diff.header_diff.push(DiffItem::new("version", "1.0", "1.1"));
diff.metadata_diff.push(DiffItem::new("n_estimators", "100", "150"));

let weight_diff = WeightDiff::from_slices(&weights_a, &weights_b);
println!("Changed Count: {}", weight_diff.changed_count);
println!("Max Diff: {:.6}", weight_diff.max_diff);
println!("Cosine Similarity: {:.4}", weight_diff.cosine_similarity);
```

## Inspection Options

Configure inspection behavior:

```rust
// Quick inspection (no weights, no quality)
let quick = InspectOptions::quick();

// Full inspection (all checks, verbose output)
let full = InspectOptions::full();

// Default (balanced)
let default = InspectOptions::default();
```

## Source Code

- Example: `examples/apr_inspection.rs`
- Module: `src/inspect/mod.rs`
