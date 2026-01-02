//! Ollama-parity performance benchmarks.
//!
//! Measures performance against Ollama targets for CPU/GPU inference.
//!
//! ## Performance Targets (Ollama Parity)
//!
//! | Operation | Size | Target (CPU) | Target (GPU) |
//! |-----------|------|--------------|--------------|
//! | Matmul    | 4096×4096 | <5ms | <1ms |
//! | Matmul    | 4096×11008 | <15ms | <3ms |
//! | Attention | seq=2048, d=128 | <10ms | <2ms |
//! | Q4 Dequant+Matmul | 4096×4096 | <8ms | <1.5ms |

use aprender::autograd::Tensor;
use aprender::format::quantize::{quantize, QuantType};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// LLM-relevant matrix sizes
const SIZES: &[(usize, usize, usize)] = &[
    (512, 512, 512),     // Small baseline
    (1024, 1024, 1024),  // Medium
    (2048, 2048, 2048),  // Large
    (4096, 4096, 4096),  // Llama 7B hidden dim
    (4096, 11008, 4096), // Llama 7B FFN (typical up_proj)
];

// Attention benchmark sizes (batch, seq_len, n_heads, head_dim)
const ATTENTION_SIZES: &[(usize, usize, usize, usize)] = &[
    (1, 512, 32, 128),  // Short context, 7B config
    (1, 2048, 32, 128), // Medium context
    (1, 4096, 32, 128), // Long context
    (4, 512, 32, 128),  // Batched inference
];

fn generate_random_tensor(rows: usize, cols: usize) -> Tensor {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
        .collect();
    Tensor::new(&data, &[rows, cols])
}

fn generate_random_f32(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 1000.0 - 0.5)
        .collect()
}

/// Benchmark: Matrix multiplication (SIMD-accelerated via trueno)
fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_simd");
    group.sample_size(20); // Reduce samples for large matrices

    for &(m, k, n) in SIZES {
        let flops = 2 * m * k * n; // FLOPs for matmul
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_random_tensor(m, k);
        let b = generate_random_tensor(k, n);

        group.bench_with_input(
            BenchmarkId::new("trueno", format!("{m}x{k}x{n}")),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| black_box(a.matmul(&b)));
            },
        );
    }

    group.finish();
}

/// Benchmark: Quantized matmul dequantize + compute
fn bench_quantized_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantized_matmul");
    group.sample_size(20);

    for &(m, k, n) in &SIZES[..3] {
        // Skip largest sizes for quantize benchmarks
        let flops = 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a = generate_random_tensor(m, k);
        let b_data = generate_random_f32(k * n);

        // Pre-quantize weights to Q8_0
        let b_q8 = quantize(&b_data, &[k, n], QuantType::Q8_0).expect("Q8_0 quantization");

        // Pre-quantize weights to Q4_0
        let b_q4 = quantize(&b_data, &[k, n], QuantType::Q4_0).expect("Q4_0 quantization");

        group.bench_with_input(
            BenchmarkId::new("Q8_0_dequant+matmul", format!("{m}x{k}x{n}")),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    let b_deq = aprender::format::quantize::dequantize(&b_q8)
                        .expect("dequantize should succeed");
                    let b_tensor = Tensor::new(&b_deq, &[k, n]);
                    black_box(a.matmul(&b_tensor))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Q4_0_dequant+matmul", format!("{m}x{k}x{n}")),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| {
                    let b_deq = aprender::format::quantize::dequantize(&b_q4)
                        .expect("dequantize should succeed");
                    let b_tensor = Tensor::new(&b_deq, &[k, n]);
                    black_box(a.matmul(&b_tensor))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Softmax attention computation
fn bench_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention");
    group.sample_size(20);

    for &(batch, seq, n_heads, head_dim) in ATTENTION_SIZES {
        let total_dim = n_heads * head_dim;
        // Attention FLOPs: 2 * batch * n_heads * seq^2 * head_dim (Q@K^T + softmax@V)
        let flops = 2 * batch * n_heads * seq * seq * head_dim;
        group.throughput(Throughput::Elements(flops as u64));

        // Generate Q, K, V tensors
        let q = generate_random_tensor(batch * seq, total_dim);
        let k = generate_random_tensor(batch * seq, total_dim);
        let v = generate_random_tensor(batch * seq, total_dim);

        group.bench_with_input(
            BenchmarkId::new("scaled_dot_product", format!("b{batch}_s{seq}_h{n_heads}")),
            &(batch, seq, n_heads, head_dim),
            |bench, _| {
                bench.iter(|| {
                    // Simplified single-head attention for benchmarking
                    // In practice, this would be multi-head with reshape
                    let scores = q.matmul(&k.transpose());
                    let scale = 1.0 / (head_dim as f32).sqrt();
                    let scaled = scores.mul_scalar(scale);
                    let attn_weights = scaled.softmax(); // No dim arg in aprender
                    black_box(attn_weights.matmul(&v))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark: Tensor transpose (common in attention)
fn bench_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("transpose");

    for &(m, k, _) in SIZES {
        let a = generate_random_tensor(m, k);
        group.throughput(Throughput::Elements((m * k) as u64));

        group.bench_with_input(
            BenchmarkId::new("trueno", format!("{m}x{k}")),
            &(m, k),
            |bench, _| {
                bench.iter(|| black_box(a.transpose()));
            },
        );
    }

    group.finish();
}

/// Benchmark: Element-wise operations (GELU, ReLU)
fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");

    let sizes = [1024, 4096, 16384, 65536, 262144];

    for size in sizes {
        let a = generate_random_tensor(1, size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("relu", size), &size, |bench, _| {
            bench.iter(|| black_box(a.relu()));
        });

        group.bench_with_input(BenchmarkId::new("sigmoid", size), &size, |bench, _| {
            bench.iter(|| black_box(a.sigmoid()));
        });

        group.bench_with_input(BenchmarkId::new("gelu", size), &size, |bench, _| {
            bench.iter(|| black_box(aprender::nn::F::gelu(&a)));
        });
    }

    group.finish();
}

/// Benchmark: Quantization round-trip
fn bench_quantization(c: &mut Criterion) {
    let mut group = c.benchmark_group("quantization");

    let sizes = [1024, 4096, 16384, 65536];

    for size in sizes {
        let data = generate_random_f32(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Q8_0_quantize", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(quantize(&data, &[size], QuantType::Q8_0)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Q4_0_quantize", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(quantize(&data, &[size], QuantType::Q4_0)));
            },
        );

        // Pre-quantize for dequantize benchmark
        let q8 = quantize(&data, &[size], QuantType::Q8_0).expect("quantize");
        let q4 = quantize(&data, &[size], QuantType::Q4_0).expect("quantize");

        group.bench_with_input(
            BenchmarkId::new("Q8_0_dequantize", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(aprender::format::quantize::dequantize(&q8)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("Q4_0_dequantize", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(aprender::format::quantize::dequantize(&q4)));
            },
        );
    }

    group.finish();
}

/// Benchmark: End-to-end MLP forward (common in transformers)
fn bench_mlp_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_forward");
    group.sample_size(20);

    // Llama 7B MLP config: hidden=4096, intermediate=11008
    let configs = [
        (4096, 11008, "7B"),
        (5120, 13824, "13B"),
        (8192, 22016, "70B"),
    ];

    for (hidden, intermediate, name) in configs {
        // MLP: x -> up_proj -> GELU -> down_proj
        let batch_seq = 1; // Single token
        let x = generate_random_tensor(batch_seq, hidden);
        let up_proj = generate_random_tensor(hidden, intermediate);
        let down_proj = generate_random_tensor(intermediate, hidden);

        // FLOPs: 2*hidden*intermediate + intermediate + 2*intermediate*hidden
        let flops = 4 * hidden * intermediate + intermediate;
        group.throughput(Throughput::Elements(flops as u64));

        group.bench_with_input(
            BenchmarkId::new("mlp", name),
            &(hidden, intermediate),
            |bench, _| {
                bench.iter(|| {
                    let h = x.matmul(&up_proj);
                    let h = aprender::nn::F::gelu(&h);
                    black_box(h.matmul(&down_proj))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_quantized_matmul,
    bench_attention,
    bench_transpose,
    bench_activations,
    bench_quantization,
    bench_mlp_forward,
);
criterion_main!(benches);
