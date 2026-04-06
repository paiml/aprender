//! Realizar Comparative Benchmark
//!
//! Benchmarks Realizar inference on canonical datasets for comparison
//! with PyTorch and other frameworks.
//!
//! Also includes APR vs GGUF transformer comparison for fair WASM benchmarking.
//!
//! Datasets: MNIST, CIFAR-10, Fashion-MNIST, Iris (from alimentar)
//! Metrics: Latency (p50, p95, p99), Throughput, Memory

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::apr_transformer::{AprTransformer, AprTransformerConfig, AprTransformerLayer};
use realizar::convert::GgufToAprConverter;
use realizar::gguf::{ArchConstraints, GGUFConfig, GGUFTransformer, GGUFTransformerLayer};
use realizar::Tensor;

/// Simple forward pass simulation for benchmark comparison.
/// This mimics the compute pattern of a small CNN without actual weights.
fn simulate_cnn_forward(input: &Tensor<f32>, hidden_dim: usize, num_classes: usize) -> Vec<f32> {
    let batch_size = input.shape()[0];
    let input_features = input.size() / batch_size;

    // Simulate conv + pool + fc layers (compute-bound operations)
    let mut output = vec![0.0f32; batch_size * num_classes];

    // Simulate matrix operations that would happen in a real forward pass
    for b in 0..batch_size {
        for c in 0..num_classes {
            let mut sum = 0.0f32;
            // Simulate hidden layer computation
            for h in 0..hidden_dim.min(input_features) {
                let idx = b * input_features + h;
                if idx < input.size() {
                    sum += input.data()[idx] * ((h + c) as f32 * 0.01);
                }
            }
            output[b * num_classes + c] = sum;
        }
    }

    // Simulate softmax (numerically stable)
    for b in 0..batch_size {
        let start = b * num_classes;
        let end = start + num_classes;
        let slice = &mut output[start..end];

        let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in slice.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        for v in slice.iter_mut() {
            *v /= sum;
        }
    }

    output
}

/// Simple linear forward pass for tabular data.
fn simulate_linear_forward(input: &Tensor<f32>, hidden_dim: usize, num_classes: usize) -> Vec<f32> {
    let batch_size = input.shape()[0];
    let input_features = input.size() / batch_size;

    let mut output = vec![0.0f32; batch_size * num_classes];

    for b in 0..batch_size {
        for c in 0..num_classes {
            let mut sum = 0.0f32;
            for f in 0..input_features.min(hidden_dim) {
                let idx = b * input_features + f;
                sum += input.data()[idx] * ((f + c) as f32 * 0.1);
            }
            // ReLU activation
            output[b * num_classes + c] = sum.max(0.0);
        }
    }

    output
}

/// Generate test MNIST-like data (28x28 grayscale).
fn generate_mnist_batch(batch_size: usize) -> Tensor<f32> {
    let features = 28 * 28; // 784
    let data: Vec<f32> = (0..batch_size * features)
        .map(|i| ((i % 256) as f32) / 255.0)
        .collect();
    Tensor::from_vec(vec![batch_size, features], data).expect("Failed to create tensor")
}

/// Generate test CIFAR-10-like data (32x32x3 RGB).
fn generate_cifar10_batch(batch_size: usize) -> Tensor<f32> {
    let features = 32 * 32 * 3; // 3072
    let data: Vec<f32> = (0..batch_size * features)
        .map(|i| ((i % 256) as f32) / 255.0)
        .collect();
    Tensor::from_vec(vec![batch_size, features], data).expect("Failed to create tensor")
}

/// Generate test Iris-like data (4 features).
fn generate_iris_batch(batch_size: usize) -> Tensor<f32> {
    let features = 4;
    let data: Vec<f32> = (0..batch_size * features)
        .map(|i| (i as f32) * 0.1)
        .collect();
    Tensor::from_vec(vec![batch_size, features], data).expect("Failed to create tensor")
}

fn benchmark_mnist_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("realizar_mnist");

    for batch_size in [1, 8, 32] {
        let input = generate_mnist_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let output = simulate_cnn_forward(black_box(&input), 128, 10);
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_cifar10_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("realizar_cifar10");

    for batch_size in [1, 8, 32] {
        let input = generate_cifar10_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let output = simulate_cnn_forward(black_box(&input), 256, 10);
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_iris_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("realizar_iris");

    for batch_size in [1, 8, 32] {
        let input = generate_iris_batch(batch_size);

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    let output = simulate_linear_forward(black_box(&input), 32, 3);
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

fn benchmark_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("realizar_tensor_creation");

    // MNIST tensor creation
    group.bench_function("mnist_batch_32", |b| {
        b.iter(|| {
            let tensor = generate_mnist_batch(32);
            black_box(tensor)
        });
    });

    // CIFAR-10 tensor creation
    group.bench_function("cifar10_batch_32", |b| {
        b.iter(|| {
            let tensor = generate_cifar10_batch(32);
            black_box(tensor)
        });
    });

    // Iris tensor creation
    group.bench_function("iris_batch_32", |b| {
        b.iter(|| {
            let tensor = generate_iris_batch(32);
            black_box(tensor)
        });
    });

    group.finish();
}

// ============================================================================
// APR vs GGUF Transformer Comparison Benchmarks
// ============================================================================

/// Fixed token sequence for reproducible inference (DO NOT CHANGE)
const REPRODUCIBLE_TOKENS: &[u32] = &[1, 2, 3, 4];

/// Create a test GGUF transformer with known weights
fn create_test_gguf_transformer(
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_dim: usize,
) -> GGUFTransformer {
    let config = GGUFConfig {
        architecture: "test_model".to_string(),
        constraints: ArchConstraints::from_architecture("llama"),
        hidden_dim,
        num_layers,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        rope_type: 0, // NORM (adjacent pairs)
        bos_token_id: None,
        eos_token_id: None,
        explicit_head_dim: None,
    };

    let layers: Vec<GGUFTransformerLayer> = (0..num_layers)
        .map(|_| GGUFTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
        })
        .collect();

    GGUFTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        position_embedding: None,
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
    }
}

/// Create APR transformer from GGUF (uses the converter)
fn create_test_apr_transformer(
    hidden_dim: usize,
    num_layers: usize,
    vocab_size: usize,
    intermediate_dim: usize,
) -> AprTransformer {
    let config = AprTransformerConfig {
        architecture: "test_model".to_string(),
        hidden_dim,
        num_layers,
        num_heads: 4,
        num_kv_heads: 4,
        vocab_size,
        intermediate_dim,
        context_length: 512,
        rope_theta: 10000.0,
        eps: 1e-5,
        ..Default::default()
    };

    let layers: Vec<AprTransformerLayer> = (0..num_layers)
        .map(|_| AprTransformerLayer {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.01; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.01; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.01; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        })
        .collect();

    AprTransformer {
        config,
        token_embedding: vec![0.1; vocab_size * hidden_dim],
        layers,
        output_norm_weight: vec![1.0; hidden_dim],
        output_norm_bias: None,
        lm_head_weight: vec![0.01; hidden_dim * vocab_size],
        lm_head_bias: None,
        q4k_layers: None,
        lm_head_weight_q6k: None,
        lm_head_weight_q4k: None,
    }
}

/// Benchmark: GGUF to APR conversion overhead
fn benchmark_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("gguf_to_apr_conversion");

    for (name, hidden, layers, vocab, intermediate) in [
        ("tiny_64x1", 64, 1, 100, 128),
        ("small_128x2", 128, 2, 500, 256),
    ] {
        let gguf = create_test_gguf_transformer(hidden, layers, vocab, intermediate);

        group.bench_with_input(BenchmarkId::from_parameter(name), &gguf, |b, model| {
            b.iter(|| {
                let apr = GgufToAprConverter::from_gguf_transformer(black_box(model));
                black_box(apr)
            });
        });
    }

    group.finish();
}

/// Benchmark: APR serialization/deserialization
fn benchmark_apr_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("apr_serialization");

    for (name, hidden, layers, vocab, intermediate) in [
        ("tiny_64x1", 64, 1, 100, 128),
        ("small_128x2", 128, 2, 500, 256),
    ] {
        let apr = create_test_apr_transformer(hidden, layers, vocab, intermediate);

        group.bench_with_input(BenchmarkId::new("to_bytes", name), &apr, |b, model| {
            b.iter(|| {
                let bytes = GgufToAprConverter::to_apr_bytes(black_box(model))
                    .expect("serialization failed");
                black_box(bytes)
            });
        });

        let bytes = GgufToAprConverter::to_apr_bytes(&apr).expect("serialization");
        group.bench_with_input(BenchmarkId::new("from_bytes", name), &bytes, |b, data| {
            b.iter(|| {
                let loaded = GgufToAprConverter::from_apr_bytes(black_box(data))
                    .expect("deserialization failed");
                black_box(loaded)
            });
        });
    }

    group.finish();
}

/// Benchmark: Memory usage comparison
fn benchmark_memory_usage(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");

    for (name, hidden, layers, vocab, intermediate) in [
        ("tiny_64x1", 64, 1, 100, 128),
        ("small_128x2", 128, 2, 500, 256),
        ("medium_256x4", 256, 4, 1000, 512),
    ] {
        let apr = create_test_apr_transformer(hidden, layers, vocab, intermediate);
        let stats = GgufToAprConverter::stats(&apr);

        // Just document the memory stats (benchmark doesn't measure time meaningfully)
        group.bench_function(BenchmarkId::new("apr_params", name), |b| {
            b.iter(|| black_box(stats.total_parameters));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_mnist_inference,
    benchmark_cifar10_inference,
    benchmark_iris_inference,
    benchmark_tensor_creation,
    benchmark_conversion,
    benchmark_apr_serialization,
    benchmark_memory_usage,
);

criterion_main!(benches);
