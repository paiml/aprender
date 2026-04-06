#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! GPU vector operation benchmarks (vec_add, dot product)

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Generate test data for benchmarks
fn generate_test_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.5).collect()
}

/// Benchmark GPU vector addition vs scalar baseline
pub fn bench_gpu_vec_add(c: &mut Criterion) {
    // Skip if GPU not available
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_vec_add");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // GPU backend
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.vec_add(&data_a, &data_b).unwrap());
            });
        });

        // Scalar baseline (for speedup comparison)
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);

            bencher.iter(|| {
                let result: Vec<f32> =
                    data_a.iter().zip(data_b.iter()).map(|(a, b)| a + b).collect();
                black_box(result);
            });
        });
    }

    group.finish();
}

/// Benchmark GPU dot product vs scalar baseline
pub fn bench_gpu_dot(c: &mut Criterion) {
    // Skip if GPU not available
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_dot");

    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // GPU backend
        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);
            let mut gpu = GpuBackend::new();

            bencher.iter(|| {
                black_box(gpu.dot(&data_a, &data_b).unwrap());
            });
        });

        // Scalar baseline (for speedup comparison)
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data_a = generate_test_data(size);
            let data_b = generate_test_data(size);

            bencher.iter(|| {
                let result: f32 = data_a.iter().zip(data_b.iter()).map(|(a, b)| a * b).sum();
                black_box(result);
            });
        });
    }

    group.finish();
}
