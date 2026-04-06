# Model-Level Inference Tracing (Phase 13)

Model-level tracing provides deep visibility into transformer inference behavior, complementing brick-level profiling. While `BrickProfiler` tracks *computational* performance (timing, throughput), `ModelTracer` tracks *semantic* behavior—what the model is computing and why.

## Overview

Five complementary tracing systems can be enabled independently:

| Trace Type | Purpose | Overhead | Output |
|------------|---------|----------|--------|
| **LayerActivationTrace** | Detect NaN/explosion/vanishing | ~2% | Statistics per layer |
| **AttentionWeightTrace** | Debug context/repetition | ~5% | Sparse attention matrix |
| **LogitEvolutionTrace** | Understand token selection | ~3% | Per-layer logit ranks |
| **QuantizationErrorTrace** | Measure quantization impact | ~10% | MSE vs FP32 reference |
| **KvCacheStateTrace** | Debug context window | ~1% | Cache utilization stats |

## Quick Start

```rust
use trueno::brick::{ModelTracer, ModelTracerConfig, LayerActivationTrace, TensorStats};

// Create tracer with lightweight config (activations + KV cache)
let config = ModelTracerConfig::lightweight();
let mut tracer = ModelTracer::new(config);

// During inference forward pass
tracer.begin_forward(position);

// After each layer
let mut layer_trace = LayerActivationTrace::new(layer_idx);
layer_trace.input_stats = TensorStats::from_slice(&input_tensor);
layer_trace.output_stats = TensorStats::from_slice(&output_tensor);
tracer.record_layer_activation(layer_trace);

// End forward and check for anomalies
if let Some(anomaly) = tracer.end_forward() {
    log::warn!("Anomaly detected: {}", anomaly);
}

// Get summary
println!("{}", tracer.summary());
```

## TensorStats (MLT-01)

Computes tensor statistics in a single pass without storing the tensor:

```rust
use trueno::brick::TensorStats;

let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
let stats = TensorStats::from_slice(&data);

println!("count: {}", stats.count);       // 5
println!("min: {}", stats.min);           // 1.0
println!("max: {}", stats.max);           // 5.0
println!("mean: {}", stats.mean);         // 3.0
println!("std: {}", stats.std);           // 1.58
println!("l2_norm: {}", stats.l2_norm);   // 7.42
println!("nan_count: {}", stats.nan_count); // 0
println!("inf_count: {}", stats.inf_count); // 0
```

### Anomaly Detection

```rust
// Detect NaN
let nan_data = vec![1.0, f32::NAN, 3.0];
let stats = TensorStats::from_slice(&nan_data);
assert!(stats.has_anomaly());
assert!(stats.anomaly_description().unwrap().contains("NaN"));

// Detect explosion (values > 1e6)
let explode_data = vec![1.0, 1e7, 3.0];
let stats = TensorStats::from_slice(&explode_data);
assert!(stats.has_anomaly());

// Detect vanishing (std < 1e-6)
let vanish_data = vec![1.0, 1.0, 1.0];
let stats = TensorStats::from_slice(&vanish_data);
assert!(stats.is_vanishing());
```

## LayerActivationTrace

Track activations through transformer layers:

```rust
use trueno::brick::{LayerActivationTrace, ModelActivationTrace, TensorStats};

let mut model_trace = ModelActivationTrace::with_capacity(32);

for layer_idx in 0..32 {
    let mut layer = LayerActivationTrace::new(layer_idx);

    // Record stats at each stage
    layer.input_stats = TensorStats::from_slice(&input);
    layer.post_norm_stats = TensorStats::from_slice(&after_norm);
    layer.post_attn_stats = TensorStats::from_slice(&after_attn);
    layer.post_ffn_stats = TensorStats::from_slice(&after_ffn);
    layer.output_stats = TensorStats::from_slice(&output);

    // Track residual connection health
    layer.residual_ratio = compute_residual_ratio(&output, &after_attn);

    model_trace.add_layer(layer);
}

model_trace.finalize();
if model_trace.has_anomaly {
    println!("Anomaly: {}", model_trace.anomaly_desc.unwrap());
}
```

### Anomaly Rules

- **NaN detected**: `nan_count > 0`
- **Explosion**: `max.abs() > 1e6` or `std > 1e4`
- **Vanishing**: `std < 1e-6` (after warmup layers)
- **Residual dominance**: `residual_ratio > 0.99` (skip connection bypass)

## AttentionWeightTrace (MLT-02)

Debug attention patterns without storing full matrices:

```rust
use trueno::brick::{AttentionWeightTrace, AttentionTraceConfig};

// Create trace from attention weights
let weights = vec![0.4, 0.1, 0.05, 0.05, 0.2, 0.1, 0.1];
let trace = AttentionWeightTrace::from_weights(
    0,       // layer_idx
    0,       // head_idx
    6,       // query_pos (current token)
    &weights,
    5,       // top_k
);

println!("Top-5 positions: {:?}", trace.top_k_positions);
println!("Top-5 weights: {:?}", trace.top_k_weights);
println!("Tail mass: {}", trace.tail_mass);
println!("Entropy: {}", trace.entropy);

// Diagnostic patterns
if trace.is_attention_sink(0.3) {
    println!("Warning: Attention sink on BOS token");
}
if trace.has_recency_bias(5, 0.7) {
    println!("Warning: Strong recency bias (repetition risk)");
}
```

### Configure Selective Tracing

```rust
let config = AttentionTraceConfig {
    top_k: 10,
    layers: Some(vec![0, 15, 31]),  // Only trace specific layers
    heads: Some(vec![0, 1]),         // Only trace specific heads
    weight_threshold: 0.01,
};

if config.should_trace_layer(15) && config.should_trace_head(0) {
    // Record trace
}
```

## LogitEvolutionTrace (MLT-03)

Track how token probabilities evolve through layers:

```rust
use trueno::brick::{LogitEvolutionTrace, TokenLogitEvolution};

let mut trace = LogitEvolutionTrace::new(100, 0.7, 0.9);

// Track specific tokens
let token = trace.track_token(42, "hello".to_string());
token.record_layer(0.5, 500);  // logit, rank at layer 0
token.record_layer(1.0, 200);  // layer 1
token.record_layer(3.0, 10);   // layer 2
token.record_layer(5.0, 1);    // layer 3

// Find where the token's fate was decided
if let Some(layer) = token.decisive_layer() {
    println!("Token 'hello' rank changed most at layer {}", layer);
}

// Compute rank directly
let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
let rank = LogitEvolutionTrace::compute_rank(&logits, 1); // token 1 has highest logit
assert_eq!(rank, 0);
```

## QuantizationErrorTrace (MLT-04)

Measure quantization impact:

```rust
use trueno::brick::{QuantizationErrorTrace, QuantType, BrickId};

let reference = vec![1.0, 2.0, 3.0, 4.0];
let quantized = vec![1.02, 1.98, 3.05, 3.95];

let trace = QuantizationErrorTrace::compute(
    BrickId::QkvProjection,
    5,
    &quantized,
    &reference,
    QuantType::Q4_K,
);

println!("MSE: {:.6}", trace.mse);
println!("Cosine similarity: {:.6}", trace.cosine_similarity);
println!("SNR: {:.1} dB", trace.snr_db);

// Thresholds (from llama.cpp Q4_K validation)
if trace.is_acceptable() {
    println!("Quantization acceptable (cosine > 0.995)");
} else if trace.is_warning() {
    println!("Quantization warning (0.99 < cosine < 0.995)");
} else {
    println!("Quantization CRITICAL (cosine < 0.99)");
}
```

### Quantization Types

```rust
use trueno::brick::QuantType;

// Get bits per element
assert_eq!(QuantType::F32.bits_per_element(), 32.0);
assert_eq!(QuantType::Q4_K.bits_per_element(), 4.5);

// Compression ratios
println!("Q4_K compression: {:.1}x", QuantType::Q4_K.compression_ratio());
// Output: Q4_K compression: 7.1x
```

## KvCacheStateTrace (MLT-05)

Monitor KV cache behavior:

```rust
use trueno::brick::{KvCacheStateTrace, KvCacheSessionTrace};

let mut session = KvCacheSessionTrace::default();

for step in 0..100 {
    let mut trace = KvCacheStateTrace::new(step, 2048);
    trace.valid_positions = step + 1;
    trace.cache_size_bytes = (step + 1) * 4096;
    trace.cache_hit_rate = 0.95;
    trace.evictions_this_step = if step > 90 { 1 } else { 0 };

    session.add_step(trace);
}

println!("Total evictions: {}", session.total_evictions);
println!("Peak memory: {} KB", session.peak_memory_bytes / 1024);
println!("Avg hit rate: {:.1}%", session.avg_hit_rate * 100.0);

// Detect thrashing
if session.has_thrashing(50, 0.5) {
    println!("WARNING: Context thrashing detected");
}
```

## Unified ModelTracer

Combine all trace types:

```rust
use trueno::brick::{ModelTracer, ModelTracerConfig};

// Full tracing (debugging)
let config = ModelTracerConfig::full();

// Lightweight (production)
let config = ModelTracerConfig::lightweight();

// Disabled (zero overhead)
let config = ModelTracerConfig::default();
assert!(!config.is_enabled());

// Custom
let config = ModelTracerConfig {
    trace_activations: true,
    trace_attention: false,  // Skip attention tracing
    trace_logits: true,
    trace_quant_error: false,  // Too expensive for production
    trace_kv_cache: true,
    ..Default::default()
};

let mut tracer = ModelTracer::new(config);

// During inference
tracer.begin_forward(position);
// ... record traces ...
if let Some(anomaly) = tracer.end_forward() {
    log::warn!("Anomaly: {}", anomaly);
}

// Summary
println!("{}", tracer.summary());
```

## Running the Example

```bash
cargo run --example model_tracing
```

## Performance Considerations

1. **Zero-cost when disabled**: `ModelTracerConfig::default()` produces no overhead
2. **Lightweight for production**: Use `ModelTracerConfig::lightweight()` (~3% overhead)
3. **TensorStats is single-pass**: Uses Welford's algorithm, no extra allocations
4. **AttentionWeightTrace is sparse**: Only stores top-k, not full attention matrix
5. **Enable selectively**: Use `AttentionTraceConfig` to trace only specific layers/heads

## Falsification Tests

The implementation includes comprehensive tests (F250-F275):

- F250: TensorStats correctness
- F251: NaN detection (100% recall)
- F252: Explosion detection
- F253: Attention top-k sorting
- F258: Cosine similarity range
- F263: Overhead bounds
- F272: Bit-exactness

Run with:

```bash
cargo test test_f250 --lib
cargo test test_f251 --lib
# ... etc
```

## See Also

- [BrickProfiler](./profiling.md) - Computational performance profiling
- [ML Tuner](./ml-tuner.md) - Learned kernel selection
- [Spec E.11](../../../docs/specifications/ml-tuner-bricks.md) - Full specification
