# aprender Integration

[aprender](https://github.com/paiml/aprender) is a next-generation machine learning library in pure Rust. trueno integrates with aprender to provide ML-based kernel selection and throughput prediction.

## Overview

The integration provides:

- **RandomForestRegressor** for throughput prediction
- **RandomForestClassifier** for kernel selection
- Training on benchmark data for hardware-specific optimization

## Enabling the Integration

Add the `ml-tuner` feature to your `Cargo.toml`:

```toml
[dependencies]
trueno = { version = "0.13", features = ["ml-tuner"] }
```

## Feature Matrix

| Feature | Default | ml-tuner |
|---------|---------|----------|
| TunerFeatures (42-dim) | Yes | Yes |
| Heuristic prediction | Yes | Yes |
| Roofline clamping | Yes | Yes |
| RandomForest regressor | No | Yes |
| RandomForest classifier | No | Yes |
| Custom model training | No | Yes |

## Usage Example

```rust
use trueno::tuner::{ThroughputRegressor, TunerFeatures, QuantType};

// Create RF-backed regressor
let mut regressor = ThroughputRegressor::with_random_forest(100);

// Collect benchmark data
let training_data: Vec<(TunerFeatures, f32)> = collect_benchmarks();

// Train the model
regressor.train_random_forest(&training_data)?;

// Use trained model for predictions
let features = TunerFeatures::builder()
    .model_params_b(7.0)
    .batch_size(4)
    .quant_type(QuantType::Q4K)
    .gpu_mem_bw_gbs(1000.0)
    .build();

let pred = regressor.predict(&features);
println!("Predicted throughput: {:.1} tok/s", pred.predicted_tps);
```

## Why aprender?

1. **Pure Rust** - No Python or C++ dependencies
2. **SIMD-accelerated** - Uses trueno for tensor operations (circular dependency resolved via feature flags)
3. **Production-ready** - Used in PAIML showcase demos
4. **Minimal API** - Simple fit/predict interface

## Training Data Collection

For best results, train on benchmark data from your target hardware:

```rust
use trueno::tuner::{TunerFeatures, QuantType};
use std::time::Instant;

fn benchmark_throughput(features: &TunerFeatures) -> f32 {
    // Run actual inference and measure tokens/second
    let start = Instant::now();
    let tokens = run_inference(features);
    let elapsed = start.elapsed().as_secs_f32();
    tokens as f32 / elapsed
}

fn collect_training_data() -> Vec<(TunerFeatures, f32)> {
    let mut data = Vec::new();

    // Sweep batch sizes
    for batch in [1, 2, 4, 8, 16] {
        // Sweep model sizes
        for params_b in [0.5, 1.5, 7.0, 13.0] {
            let features = TunerFeatures::builder()
                .model_params_b(params_b)
                .batch_size(batch)
                .quant_type(QuantType::Q4K)
                .gpu_mem_bw_gbs(1000.0)
                .build();

            let throughput = benchmark_throughput(&features);
            data.push((features, throughput));
        }
    }

    data
}
```

## Model Persistence

Save trained models for reuse:

```rust
use trueno::tuner::ThroughputRegressor;
use std::fs;

// Save model
let model_json = serde_json::to_string(&regressor)?;
fs::write("throughput_model.json", model_json)?;

// Load model
let model_json = fs::read_to_string("throughput_model.json")?;
let regressor: ThroughputRegressor = serde_json::from_str(&model_json)?;
```

**Note:** RandomForest models are not serialized (marked `#[serde(skip)]`). After loading, you must retrain or use heuristic fallback.

## Further Reading

- [ML Tuner Chapter](../performance/ml-tuner.md)
- [aprender Documentation](https://docs.rs/aprender)
- SHOWCASE-BRICK-001 Specification (not yet published)
