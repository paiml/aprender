//! Normalization benchmarks: aprender RMSNorm/LayerNorm vs scalar reference.
//!
//! RMSNorm runs once per layer per token — for a 32-layer 7B model,
//! that's 64 RMSNorm calls per token (pre-attn + pre-FFN). At 100 tok/s,
//! that's 6400 RMSNorm/s. Must be fast.
//!
//! ## Reference
//!
//! ndarray scalar implementation: compute mean-square, divide, scale.
//! aprender should beat this via fused SIMD operations.
//!
//! ## Throughput metric
//!
//! Reports elements/s (hidden_dim elements per norm).

use aprender::autograd::Tensor;
use aprender::nn::F;
use aprender_bench_compute::sizes::NORM_SIZES;
use aprender_bench_compute::{deterministic_f32, deterministic_ndarray_1d};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_rmsnorm_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm_aprender");

    for &size in NORM_SIZES {
        let data = deterministic_f32(size);
        let x = Tensor::new(&data, &[1, size]);
        // Weight (gamma) is typically ones for initial state
        let weight = Tensor::new(&vec![1.0f32; size], &[size]);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("trueno", size), &size, |bench, _| {
            bench.iter(|| black_box(F::rms_norm(&x, &weight, 1e-5)));
        });
    }

    group.finish();
}

fn bench_rmsnorm_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm_ndarray");

    for &size in NORM_SIZES {
        let arr = deterministic_ndarray_1d(size);
        let weight: ndarray::Array1<f32> = ndarray::Array1::ones(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
                let sq_mean = arr.mapv(|x| x * x).mean().unwrap();
                let rms = (sq_mean + 1e-5).sqrt();
                black_box(&arr / rms * &weight)
            });
        });
    }

    group.finish();
}

fn bench_layernorm_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm_aprender");

    for &size in NORM_SIZES {
        let data = deterministic_f32(size);
        let x = Tensor::new(&data, &[1, size]);
        let weight = Tensor::new(&vec![1.0f32; size], &[size]);
        let bias = Tensor::new(&vec![0.0f32; size], &[size]);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("trueno", size), &size, |bench, _| {
            bench.iter(|| black_box(F::layer_norm(&x, &weight, &bias, 1e-5)));
        });
    }

    group.finish();
}

fn bench_layernorm_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("layernorm_ndarray");

    for &size in NORM_SIZES {
        let arr = deterministic_ndarray_1d(size);
        let weight: ndarray::Array1<f32> = ndarray::Array1::ones(size);
        let bias: ndarray::Array1<f32> = ndarray::Array1::zeros(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                // LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
                let mean = arr.mean().unwrap();
                let centered = &arr - mean;
                let var = centered.mapv(|x| x * x).mean().unwrap();
                let std = (var + 1e-5).sqrt();
                black_box(&centered / std * &weight + &bias)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_rmsnorm_aprender,
    bench_rmsnorm_ndarray,
    bench_layernorm_aprender,
    bench_layernorm_ndarray,
);
criterion_main!(benches);
