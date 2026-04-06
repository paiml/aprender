#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Benchmarks for activation functions: relu, softmax, log_softmax, clip, sigmoid, gelu, swish, tanh

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::{Backend, Vector};

/// Benchmark ReLU activation function
pub fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu");

    // Test various sizes: small (100), medium (1K, 10K), large (100K, 1M - GPU candidates)
    for size in [100, 1000, 10000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.5 - (*size as f32) * 0.25).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.relu().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.relu().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.relu().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark softmax activation function
pub fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    // Softmax is more expensive (exp, sum, div) - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.softmax().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.softmax().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.softmax().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark log_softmax activation function
pub fn bench_log_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("log_softmax");

    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.01).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.log_softmax().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.log_softmax().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.log_softmax().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark clip (clamp) operation
pub fn bench_clip(c: &mut Criterion) {
    let mut group = c.benchmark_group("clip");

    for size in [100, 1000, 10000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.5).collect();
        let min_val = 100.0;
        let max_val = 5000.0;

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.clip(min_val, max_val).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.clip(min_val, max_val).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.clip(min_val, max_val).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark sigmoid activation function
pub fn bench_sigmoid(c: &mut Criterion) {
    let mut group = c.benchmark_group("sigmoid");

    // Sigmoid requires exp() - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data in range [-6, 6] for realistic sigmoid values
        // Previous range [-500, 500] caused scalar fast-path (returns 0/1 without exp())
        // while SIMD computed full exp(), creating misleading benchmarks
        let data: Vec<f32> = (0..*size).map(|i| (i as f32 / *size as f32) * 12.0 - 6.0).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.sigmoid().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.sigmoid().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.sigmoid().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark GELU activation function
pub fn bench_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu");

    // GELU requires tanh() - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.1 - (*size as f32) * 0.05).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.gelu().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.gelu().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.gelu().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark swish activation function
pub fn bench_swish(c: &mut Criterion) {
    let mut group = c.benchmark_group("swish");

    // Swish requires exp() for sigmoid - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) * 0.1 - (*size as f32) * 0.05).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.swish().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.swish().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.swish().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark tanh activation function
pub fn bench_tanh(c: &mut Criterion) {
    let mut group = c.benchmark_group("tanh");

    // Tanh requires exp() - test up to 100K
    for size in [100, 1000, 10000, 100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate data with mix of positive and negative values in [-3.5, 3.5] range
        // (avoiding saturation region where tanh(x) ≈ ±1)
        let data: Vec<f32> = (0..*size).map(|i| (i as f32) / (*size as f32) * 7.0 - 3.5).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(v.tanh().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(v.tanh().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let v = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(v.tanh().unwrap());
            });
        });
    }

    group.finish();
}
