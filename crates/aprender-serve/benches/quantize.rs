//! Benchmark suite for quantization/dequantization operations
//!
//! Measures dequantization performance for different quantization formats:
//! - Q4_0 (4-bit, block size 32)
//! - Q8_0 (8-bit, block size 32)
//! - Q4_K (4-bit K-quant, super-block size 256)
//! - Q5_K (5-bit K-quant, super-block size 256)
//! - Q6_K (6-bit K-quant, super-block size 256)
//!
//! Performance targets:
//! - Q4_0/Q8_0: <1μs per block (32 values)
//! - Q4_K/Q5_K/Q6_K: <10μs per super-block (256 values)

#![allow(clippy::same_item_push)] // Demo benchmark data uses simple push patterns

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::quantize::{
    dequantize_q4_0, dequantize_q4_k, dequantize_q5_k, dequantize_q6_k, dequantize_q8_0,
};

/// Create Q4_0 test data (20 bytes per block = 2 bytes scale + 16 bytes quantized + 2 bytes padding)
fn create_q4_0_data(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 20);
    for _ in 0..num_blocks {
        // Scale (f16 as 2 bytes, simplified)
        data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        // 16 bytes of 4-bit quantized data (32 values packed)
        for _ in 0..16 {
            data.push(0x12); // Demo data
        }
        // 2 bytes padding for alignment
        data.extend_from_slice(&[0, 0]);
    }
    data
}

/// Create Q8_0 test data (36 bytes per block = 2 bytes scale + 32 bytes quantized + 2 bytes padding)
fn create_q8_0_data(num_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_blocks * 36);
    for _ in 0..num_blocks {
        // Scale (f16 as 2 bytes, simplified)
        data.extend_from_slice(&0.5f32.to_le_bytes()[..2]);
        // 32 bytes of 8-bit quantized data
        for i in 0..32 {
            data.push(i as u8);
        }
        // 2 bytes padding for alignment
        data.extend_from_slice(&[0, 0]);
    }
    data
}

/// Create Q4_K test data (144 bytes per super-block)
fn create_q4_k_data(num_super_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_super_blocks * 144);
    for _ in 0..num_super_blocks {
        // Scales and mins (2 bytes each, 12 scales + 12 mins = 48 bytes)
        for _ in 0..24 {
            data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        }
        // d (4 bytes f32) + dmin (4 bytes f32) = 8 bytes
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // Quantized data (256 values * 4 bits / 8 = 128 bytes - 48 scales/mins - 8 d/dmin = 72 bytes)
        // Actually: 144 - 48 - 8 = 88 bytes for quantized data
        for _ in 0..88 {
            data.push(0x12);
        }
    }
    data
}

/// Create Q5_K test data (176 bytes per super-block)
fn create_q5_k_data(num_super_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_super_blocks * 176);
    for _ in 0..num_super_blocks {
        // Similar structure to Q4_K but with 5-bit encoding
        // Scales (12 * 2 = 24 bytes)
        for _ in 0..12 {
            data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        }
        // Mins (12 * 2 = 24 bytes)
        for _ in 0..12 {
            data.extend_from_slice(&0.0f32.to_le_bytes()[..2]);
        }
        // d (4 bytes) + dmin (4 bytes) = 8 bytes
        data.extend_from_slice(&1.0f32.to_le_bytes());
        data.extend_from_slice(&0.0f32.to_le_bytes());
        // Quantized data (176 - 24 - 24 - 8 = 120 bytes)
        for _ in 0..120 {
            data.push(0x12);
        }
    }
    data
}

/// Create Q6_K test data (210 bytes per super-block)
fn create_q6_k_data(num_super_blocks: usize) -> Vec<u8> {
    let mut data = Vec::with_capacity(num_super_blocks * 210);
    for _ in 0..num_super_blocks {
        // Scales (16 * 2 = 32 bytes for Q6_K)
        for _ in 0..16 {
            data.extend_from_slice(&1.0f32.to_le_bytes()[..2]);
        }
        // d (4 bytes f32)
        data.extend_from_slice(&1.0f32.to_le_bytes());
        // Quantized data (210 - 32 - 4 = 174 bytes)
        for _ in 0..174 {
            data.push(0x12);
        }
    }
    data
}

// Benchmark: Q4_0 dequantization
fn benchmark_q4_0_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4_0_dequantize");

    for num_blocks in &[1, 10, 100, 1000] {
        let data = create_q4_0_data(*num_blocks);
        let num_values = num_blocks * 32; // 32 values per block

        group.bench_with_input(BenchmarkId::new("blocks", num_blocks), &data, |b, data| {
            b.iter(|| {
                let result = dequantize_q4_0(black_box(data)).expect("Dequantization failed");
                black_box(result)
            });
        });

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q4_0(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q8_0 dequantization
fn benchmark_q8_0_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q8_0_dequantize");

    for num_blocks in &[1, 10, 100, 1000] {
        let data = create_q8_0_data(*num_blocks);
        let num_values = num_blocks * 32;

        group.bench_with_input(BenchmarkId::new("blocks", num_blocks), &data, |b, data| {
            b.iter(|| {
                let result = dequantize_q8_0(black_box(data)).expect("Dequantization failed");
                black_box(result)
            });
        });

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q8_0(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q4_K dequantization
fn benchmark_q4_k_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q4_k_dequantize");

    for num_super_blocks in &[1, 10, 100] {
        let data = create_q4_k_data(*num_super_blocks);
        let num_values = num_super_blocks * 256; // 256 values per super-block

        group.bench_with_input(
            BenchmarkId::new("super_blocks", num_super_blocks),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = dequantize_q4_k(black_box(data)).expect("Dequantization failed");
                    black_box(result)
                });
            },
        );

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q4_k(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q5_K dequantization
fn benchmark_q5_k_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q5_k_dequantize");

    for num_super_blocks in &[1, 10, 100] {
        let data = create_q5_k_data(*num_super_blocks);
        let num_values = num_super_blocks * 256;

        group.bench_with_input(
            BenchmarkId::new("super_blocks", num_super_blocks),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = dequantize_q5_k(black_box(data)).expect("Dequantization failed");
                    black_box(result)
                });
            },
        );

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q5_k(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Q6_K dequantization
fn benchmark_q6_k_dequantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("q6_k_dequantize");

    for num_super_blocks in &[1, 10, 100] {
        let data = create_q6_k_data(*num_super_blocks);
        let num_values = num_super_blocks * 256;

        group.bench_with_input(
            BenchmarkId::new("super_blocks", num_super_blocks),
            &data,
            |b, data| {
                b.iter(|| {
                    let result = dequantize_q6_k(black_box(data)).expect("Dequantization failed");
                    black_box(result)
                });
            },
        );

        group.bench_function(format!("throughput_{}_values", num_values), |b| {
            b.iter(|| {
                let result = dequantize_q6_k(black_box(&data)).expect("Dequantization failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

// Benchmark: Compare all quantization formats
fn benchmark_quantization_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization_comparison");

    // All benchmarks dequantize ~1000 values for fair comparison
    let q4_0_data = create_q4_0_data(32); // 32 blocks * 32 values = 1024 values
    let q8_0_data = create_q8_0_data(32); // 32 blocks * 32 values = 1024 values
    let q4_k_data = create_q4_k_data(4); // 4 super-blocks * 256 values = 1024 values
    let q5_k_data = create_q5_k_data(4); // 4 super-blocks * 256 values = 1024 values
    let q6_k_data = create_q6_k_data(4); // 4 super-blocks * 256 values = 1024 values

    group.bench_function("q4_0_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q4_0(black_box(&q4_0_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q8_0_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q8_0(black_box(&q8_0_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q4_k_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q4_k(black_box(&q4_k_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q5_k_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q5_k(black_box(&q5_k_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.bench_function("q6_k_1k_values", |b| {
        b.iter(|| {
            let result = dequantize_q6_k(black_box(&q6_k_data)).expect("Dequantization failed");
            black_box(result)
        });
    });

    group.finish();
}

// =============================================================================
// BENCH-SPRINT-002: QuantizedLinear Forward Pass Benchmarks
// Per benchmark-model-runners-spec.md v2.0: Measure fused dequant+dot throughput
// =============================================================================

use realizar::layers::QuantizedLinear;
use realizar::tensor::Tensor;

/// Create Q4_K weight data for QuantizedLinear benchmarking
///
/// Per Q4_K spec: 144 bytes per super-block of 256 values
fn create_q4k_weight_data(in_features: usize, out_features: usize) -> Vec<u8> {
    const SUPER_BLOCK_VALUES: usize = 256;
    const SUPER_BLOCK_BYTES: usize = 144;

    let super_blocks_per_row = in_features.div_ceil(SUPER_BLOCK_VALUES);
    let bytes_per_row = super_blocks_per_row * SUPER_BLOCK_BYTES;
    let total_bytes = out_features * bytes_per_row;

    // Create test Q4_K data with realistic scale values
    let mut data = Vec::with_capacity(total_bytes);
    for _ in 0..out_features {
        for sb in 0..super_blocks_per_row {
            // d (f16 as 2 bytes) - scale factor
            let scale = 0.01f32 * (sb as f32 + 1.0);
            let scale_bytes = half::f16::from_f32(scale).to_le_bytes();
            data.extend_from_slice(&scale_bytes);

            // dmin (f16 as 2 bytes) - min factor
            data.extend_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

            // scales (12 bytes) - 6-bit quantized scales
            data.extend_from_slice(&[0x55u8; 12]);

            // qs (128 bytes) - 4-bit quantized values
            data.extend_from_slice(&[0x55u8; 128]);
        }
    }

    data
}

/// Benchmark QuantizedLinear forward pass (fused dequant+dot)
///
/// This measures the critical path for LLM inference:
/// - Memory-bound: ~4.5 bits/weight vs 32 bits for f32
/// - Target: Match or approach llama.cpp performance
fn benchmark_quantized_linear_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_linear_forward");

    // Test various realistic dimensions (phi-2 style)
    let test_cases = [
        (256, 256, "256x256"),       // Small layer
        (512, 512, "512x512"),       // Medium layer
        (2560, 2560, "2560x2560"),   // phi-2 hidden dim
        (2560, 10240, "2560x10240"), // phi-2 FFN up
    ];

    for (in_features, out_features, label) in &test_cases {
        let weight_data = create_q4k_weight_data(*in_features, *out_features);
        let bias = vec![0.0f32; *out_features];

        let layer = QuantizedLinear::new(*in_features, *out_features, weight_data, bias)
            .expect("Should create QuantizedLinear");

        // Create input tensor
        let input = Tensor::from_vec(vec![*in_features], vec![1.0f32; *in_features])
            .expect("Should create input");

        group.bench_function(*label, |b| {
            b.iter(|| {
                let result = layer.forward(black_box(&input)).expect("Forward failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark QuantizedLinear vs f32 Linear comparison
///
/// Per spec: Q4_K should be memory-bound, achieving ~7x memory reduction
/// while maintaining similar compute throughput.
fn benchmark_quantized_vs_f32(c: &mut Criterion) {
    use realizar::layers::Linear;

    let mut group = c.benchmark_group("quantized_vs_f32");

    let in_features = 2560; // phi-2 hidden dim
    let out_features = 2560;

    // Q4_K layer
    let q4k_weight_data = create_q4k_weight_data(in_features, out_features);
    let q4k_bias = vec![0.0f32; out_features];
    let q4k_layer = QuantizedLinear::new(in_features, out_features, q4k_weight_data, q4k_bias)
        .expect("Should create QuantizedLinear");

    // f32 layer
    let f32_layer = Linear::new(in_features, out_features).expect("Should create Linear");

    // Input tensor
    let input = Tensor::from_vec(vec![in_features], vec![1.0f32; in_features])
        .expect("Should create input");

    // Memory comparison
    let q4k_bytes = q4k_layer.memory_bytes();
    let f32_bytes = in_features * out_features * 4 + out_features * 4; // weights + bias
    let memory_ratio = f32_bytes as f64 / q4k_bytes as f64;
    eprintln!(
        "\nMemory comparison: Q4_K={} bytes, f32={} bytes, ratio={:.2}x",
        q4k_bytes, f32_bytes, memory_ratio
    );

    group.bench_function("q4k_2560x2560", |b| {
        b.iter(|| {
            let result = q4k_layer
                .forward(black_box(&input))
                .expect("Forward failed");
            black_box(result)
        });
    });

    group.bench_function("f32_2560x2560", |b| {
        b.iter(|| {
            let result = f32_layer
                .forward(black_box(&input))
                .expect("Forward failed");
            black_box(result)
        });
    });

    group.finish();
}

/// Benchmark batch throughput for QuantizedLinear
///
/// Simulates token generation where multiple positions need forward pass
fn benchmark_quantized_batch_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_batch_throughput");

    let in_features = 2560; // phi-2 hidden dim
    let out_features = 2560;

    let weight_data = create_q4k_weight_data(in_features, out_features);
    let bias = vec![0.0f32; out_features];
    let layer = QuantizedLinear::new(in_features, out_features, weight_data, bias)
        .expect("Should create QuantizedLinear");

    for batch_size in &[1, 4, 8, 16, 32] {
        let input = Tensor::from_vec(
            vec![*batch_size, in_features],
            vec![1.0f32; batch_size * in_features],
        )
        .expect("Should create batch input");

        group.bench_function(format!("batch_{}", batch_size), |b| {
            b.iter(|| {
                let result = layer.forward(black_box(&input)).expect("Forward failed");
                black_box(result)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_q4_0_dequantize,
    benchmark_q8_0_dequantize,
    benchmark_q4_k_dequantize,
    benchmark_q5_k_dequantize,
    benchmark_q6_k_dequantize,
    benchmark_quantization_comparison,
    benchmark_quantized_linear_forward,
    benchmark_quantized_vs_f32,
    benchmark_quantized_batch_throughput
);
criterion_main!(benches);
