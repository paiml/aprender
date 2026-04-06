# ML Tuner: Learned Kernel Selection

The ML Tuner provides machine learning-based throughput prediction and kernel selection for ComputeBrick operations. It uses a 42-dimension feature vector (v1.1.0) with roofline model clamping for physically-bounded predictions.

**Reference:** SHOWCASE-BRICK-001, Section 12

## Overview

The ML Tuner consists of three main components:

1. **TunerFeatures** - 42-dimension feature vector encoding model, hardware, and runtime configuration
2. **ThroughputRegressor** - Predicts tokens/second throughput with roofline clamping
3. **KernelClassifier** - Recommends optimal kernel (VectorizedQ4K, BatchedQ4K, etc.)

## Feature Vector (DIM=42)

The `TunerFeatures` struct encodes all information needed for ML-based optimization:

```rust,ignore
{{#include ../../../examples/tuner_usage.rs:basic_features}}
```

### Feature Breakdown

| Range | Count | Category | Description |
|-------|-------|----------|-------------|
| 0-9   | 10    | Model    | params_b, hidden_dim, layers, heads, intermediate_dim, vocab_size, kv_heads, head_dim, rope_theta, tie_embeddings |
| 10-19 | 10    | Runtime  | batch_size, seq_len, context_len, kv_cache_tokens, draft_tokens, prompt_tokens, generated_tokens, temperature, top_p, top_k |
| 20-29 | 10    | Quant    | quant_type (one-hot Q4_K/Q5_K/Q6_K/Q8_0/F16/F32), quant_group_size, bits_per_weight, quant_scheme_idx, has_scales |
| 30-41 | 12    | Hardware | gpu_mem_bw, gpu_compute_tflops, sm_count, tensor_cores, cuda_graphs, pcie_gen, vram_gb, cpu_threads, numa_nodes, system_ram_gb, is_unified_memory, power_limit |

## Throughput Prediction

The `ThroughputRegressor` predicts tokens/second with roofline model clamping:

```rust,ignore
{{#include ../../../examples/tuner_usage.rs:throughput_prediction}}
```

### Roofline Model

Predictions are clamped to physical limits using the roofline model (Williams et al., 2009):

```
throughput_max = gpu_mem_bw_bytes / (model_params_b * bytes_per_param)
```

For example, RTX 4090 (1000 GB/s) with 7B Q4_K model (~0.5 bytes/param):
- Roofline: 1000 GB/s / (7B * 0.5) = ~286 tok/s theoretical max

The heuristic model may predict higher, but roofline clamping ensures physical plausibility.

## Kernel Selection

The `KernelClassifier` recommends the optimal kernel implementation:

```rust
use trueno::tuner::{KernelClassifier, TunerFeatures, QuantType};

let classifier = KernelClassifier::new();

let features = TunerFeatures::builder()
    .model_params_b(1.5)
    .batch_size(4)
    .quant_type(QuantType::Q4K)
    .build();

let recommendation = classifier.predict(&features);

println!("Recommended: {:?}", recommendation.top_kernel);
println!("Confidence: {:.1}%", recommendation.confidence * 100.0);
for (kernel, conf) in recommendation.alternatives.iter().take(3) {
    println!("  - {:?}: {:.1}%", kernel, conf * 100.0);
}
```

### Kernel Selection Rules

| Batch Size | Recommended Kernel | Rationale |
|------------|-------------------|-----------|
| M=1        | VectorizedQ4K     | Single sequence, maximize per-token latency |
| M=2-3      | VectorizedQ4K     | Low batch, vectorized still efficient |
| M>=4       | BatchedQ4K        | High batch, batched attention wins |

## RandomForest Models (Optional)

With the `ml-tuner` feature, you can use aprender's RandomForest models for learned optimization:

```toml
# Cargo.toml
[dependencies]
trueno = { version = "0.13", features = ["ml-tuner"] }
```

### Training a Custom Regressor

```rust
use trueno::tuner::{ThroughputRegressor, TunerFeatures, QuantType};

// Create RF-backed regressor with 100 trees
let mut regressor = ThroughputRegressor::with_random_forest(100);

// Generate training data from benchmarks
let training_data: Vec<(TunerFeatures, f32)> = (0..100)
    .map(|i| {
        let batch = 1 + (i % 8) as u32;
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(batch)
            .quant_type(QuantType::Q4K)
            .gpu_mem_bw_gbs(1000.0)
            .cuda_graphs(batch == 1)
            .build();
        // Measured throughput from benchmark
        let throughput = 200.0 + (batch as f32) * 80.0;
        (features, throughput)
    })
    .collect();

// Train the model
regressor.train_random_forest(&training_data)?;

// Predictions now use learned model
let pred = regressor.predict(&features);
println!("RF prediction: {:.1} tok/s", pred.predicted_tps);
```

### Training a Custom Classifier

```rust
use trueno::tuner::{KernelClassifier, TunerFeatures, QuantType};

let mut classifier = KernelClassifier::with_random_forest(50);

// Label encoding: VectorizedQ4K=2, BatchedQ4K=3
let training_data: Vec<(TunerFeatures, u32)> = (0..100)
    .map(|i| {
        let batch = 1 + (i % 8) as u32;
        let features = TunerFeatures::builder()
            .model_params_b(1.5)
            .batch_size(batch)
            .quant_type(QuantType::Q4K)
            .build();
        let label = if batch >= 4 { 3 } else { 2 };
        (features, label)
    })
    .collect();

classifier.train(&training_data)?;
```

## Full Tuner Recommendations

The `BrickTuner` combines throughput and kernel predictions with experiment suggestions:

```rust
use trueno::tuner::{BrickTuner, TunerFeatures, QuantType};

let tuner = BrickTuner::new();
let features = TunerFeatures::builder()
    .model_params_b(1.5)
    .batch_size(4)
    .quant_type(QuantType::Q4K)
    .gpu_mem_bw_gbs(1000.0)
    .cuda_graphs(true)
    .build();

let rec = tuner.recommend(&features);

println!("Throughput: {:.1} tok/s", rec.throughput.predicted_tps);
println!("Best kernel: {:?}", rec.kernel.top_kernel);
println!("Experiments to try:");
for exp in &rec.suggested_experiments {
    println!("  - {}", exp);
}
```

## Running the Demo

```bash
# Default (heuristic models)
cargo run --example ml_tuner_demo

# With RandomForest models
cargo run --example ml_tuner_demo --features ml-tuner
```

## Integration with ComputeBrick

The ML tuner integrates with ComputeBrick kernel selection:

```rust
use trueno::compute::{ComputeBrick, ComputeBrickConfig};
use trueno::tuner::{BrickTuner, TunerFeatures};

// Build features from runtime environment
let features = TunerFeatures::from_env()?;

// Get tuner recommendation
let tuner = BrickTuner::new();
let rec = tuner.recommend(&features);

// Configure ComputeBrick with recommended kernel
let config = ComputeBrickConfig::builder()
    .kernel(rec.kernel.top_kernel)
    .batch_size(features.batch_size())
    .build();

let brick = ComputeBrick::with_config(config)?;
```

## Performance Considerations

1. **Feature extraction is cheap**: `TunerFeatures::to_vector()` is O(1)
2. **Heuristic prediction is instant**: No ML inference overhead
3. **RF inference scales with trees**: 100 trees ≈ 1ms inference
4. **Train once, predict many**: Cache trained models for repeated use

## Phase 14: ML-Tuner Evolution

Phase 14 (E.12) adds deeper ML integration for production deployment:

### MLT-10: Pre-trained Weights

Ship with pre-trained weights from CI benchmark corpus:

```rust
use trueno::tuner::{BrickTuner, pretrained};

// Use pre-trained weights (8.2% MAPE, 10,000+ samples)
let tuner = BrickTuner::with_pretrained();
println!("Version: {}", tuner.version()); // "1.0.0-pretrained"
println!("MAPE: {:.1}%", tuner.throughput_mape() * 100.0);

// Access feature importance for explainability
for (idx, name, importance) in pretrained::FEATURE_IMPORTANCE.iter().take(5) {
    println!("{}: {:.1}%", name, importance * 100.0);
}
// Output:
//   batch_size: 28.0%
//   gpu_mem_bw: 18.0%
//   model_params_b: 14.0%
```

### MLT-11: First-Run Calibration

Calibrate to local hardware (requires `hardware-detect` feature):

```rust
use trueno::tuner::BrickTuner;

let mut tuner = BrickTuner::with_pretrained();

#[cfg(feature = "hardware-detect")]
{
    let result = tuner.calibrate()?;
    println!("Calibrated to: {}", result.hardware_id);
    println!("Local MAPE: {:.1}%", result.local_mape * 100.0);
    println!("Improvement: {:.1}%", result.improvement_pct);
}
```

### MLT-12: Online Learning

Continuously improve predictions during inference:

```rust
use trueno::tuner::{BrickTuner, TunerFeatures, QuantType};

let tuner = BrickTuner::with_pretrained();
let mut learner = tuner.online_learner();

// During inference loop
let features = TunerFeatures::builder()
    .model_params_b(7.0)
    .batch_size(4)
    .quant_type(QuantType::Q4K)
    .gpu_mem_bw_gbs(1000.0)
    .build();

for measured_tps in [150.0, 155.0, 152.0, 158.0] {
    learner.observe(&features.to_vector(), measured_tps);
}

println!("Updates: {}", learner.num_updates());
println!("EMA Loss: {:.4}", learner.ema_loss());
println!("Converging: {}", learner.is_converging());

// Apply learned weights back to tuner
let mut updated_tuner = tuner.clone();
updated_tuner.apply_online_updates(&learner);
```

### MLT-13: Bandit Kernel Selection

Explore vs exploit kernel choices using UCB1 or Thompson Sampling:

```rust
use trueno::tuner::{BrickTuner, KernelBandit, TunerFeatures, QuantType};

let tuner = BrickTuner::with_pretrained();
let mut bandit = tuner.kernel_bandit(); // UCB1 by default
// Or: let mut bandit = KernelBandit::with_thompson_sampling();

let features = TunerFeatures::builder()
    .model_params_b(7.0)
    .batch_size(4)
    .quant_type(QuantType::Q4K)
    .build();

// Production loop with exploration
for _ in 0..100 {
    let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 0.2);

    // Use rec.top_kernel for inference...
    let measured_tps = /* actual measurement */ 150.0;

    // Update bandit with normalized reward
    let reward = (measured_tps / 200.0).min(1.0);
    bandit.update(rec.top_kernel, reward);
}

println!("Best kernel: {:?}", bandit.best_kernel());
println!("Exploration rate: {:.2}", bandit.exploration_rate());
println!("Cumulative regret: {:.2}", bandit.estimated_regret());
```

### Running the Evolution Demo

```bash
# Phase 14 demo (pre-trained + online learning + bandits)
cargo run --example ml_tuner_evolution

# With hardware calibration
cargo run --example ml_tuner_evolution --features hardware-detect
```

## Further Reading

- [ComputeBrick Architecture](../architecture/compute-brick.md)
- [Benchmarks Overview](./benchmarks.md)
- [Optimization Guide](./optimization-guide.md)
- [aprender RandomForest Documentation](https://docs.rs/aprender/latest/aprender/tree/)
- [Model-Level Inference Tracing](./model-tracing.md)
