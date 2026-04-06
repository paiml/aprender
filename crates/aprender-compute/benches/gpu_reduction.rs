#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! GPU Tiled Reduction Benchmarks
//!
//! Validates tiled reduction shader performance on Metal/Vulkan/DX12.
//! Per issue #76: Benchmark tiled reduction at 1M, 10M, 32M elements.
//!
//! **EMPIRICAL FINDINGS (AMD Radeon Pro W5700X, Metal, 2026-01-03):**
//!
//! GPU tiled reduction achieves consistent ~150 Melem/s throughput on GPU.
//! However, transfer overhead makes standalone GPU reductions slower than CPU:
//!
//! | Operation | Size  | GPU Tiled   | Scalar CPU  | Winner    |
//! |-----------|-------|-------------|-------------|-----------|
//! | Sum       | 1M    | 8.25ms      | 0.92ms      | CPU 9x    |
//! | Sum       | 10M   | 67.2ms      | 9.46ms      | CPU 7x    |
//! | Sum       | 32M   | 215ms       | 30.7ms      | CPU 7x    |
//! | Max/Min   | 1M    | 8.3ms       | 0.22ms      | CPU 37x   |
//! | Max/Min   | 10M   | 67ms        | 3.25ms      | CPU 20x   |
//! | Max/Min   | 32M   | 215ms       | 10.7ms      | CPU 20x   |
//!
//! This is expected behavior - GPU wins for O(n³) operations (matmul),
//! but loses for O(n) reductions due to ~8ms transfer overhead baseline.
//!
//! GPU tiled reduction is optimal when:
//! - Data is already on GPU (no transfer cost)
//! - Reduction is part of larger GPU compute pipeline
//! - Latency hiding in async GPU workloads
//!
//! Metal buffer binding limit: 128MB (max ~32M f32 elements)

#![cfg(feature = "gpu")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Generate 2D test data for reduction benchmarks
fn generate_matrix_data(width: usize, height: usize) -> Vec<f32> {
    (0..width * height).map(|i| ((i % 1000) as f32) * 0.001).collect()
}

/// Benchmark GPU tiled sum reduction vs scalar baseline
fn bench_tiled_sum(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping tiled reduction benchmarks");
        return;
    }

    let mut group = c.benchmark_group("tiled_sum_reduction");
    group.sample_size(20); // Fewer samples for GPU

    // Test sizes: 1M, 10M, 32M (Metal buffer binding limit is 128MB)
    let sizes: Vec<(usize, usize, &str)> = vec![
        (1000, 1000, "1M"),  // 1M elements = 4MB
        (3162, 3163, "10M"), // ~10M elements = 40MB
        (5657, 5657, "32M"), // ~32M elements = 128MB (Metal limit)
    ];

    for (width, height, label) in sizes.iter() {
        let total = width * height;
        group.throughput(Throughput::Elements(total as u64));

        // GPU Tiled Reduction (measures transfer + compute)
        group.bench_with_input(
            BenchmarkId::new("GPU_Tiled", label),
            &(*width, *height),
            |bencher, &(w, h)| {
                let data = generate_matrix_data(w, h);
                let mut gpu = GpuBackend::new();

                bencher.iter(|| {
                    black_box(gpu.tiled_sum_2d_gpu(&data, w, h).unwrap());
                });
            },
        );

        // Scalar baseline (naive loop for comparison)
        group.bench_with_input(
            BenchmarkId::new("Scalar", label),
            &(*width, *height),
            |bencher, &(w, h)| {
                let data = generate_matrix_data(w, h);

                bencher.iter(|| {
                    let sum: f32 = data.iter().sum();
                    black_box(sum);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU tiled max reduction
fn bench_tiled_max(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping tiled reduction benchmarks");
        return;
    }

    let mut group = c.benchmark_group("tiled_max_reduction");
    group.sample_size(20);

    let sizes: Vec<(usize, usize, &str)> =
        vec![(1000, 1000, "1M"), (3162, 3163, "10M"), (5657, 5657, "32M")];

    for (width, height, label) in sizes.iter() {
        let total = width * height;
        group.throughput(Throughput::Elements(total as u64));

        // GPU Tiled Max
        group.bench_with_input(
            BenchmarkId::new("GPU_Tiled", label),
            &(*width, *height),
            |bencher, &(w, h)| {
                let data = generate_matrix_data(w, h);
                let mut gpu = GpuBackend::new();

                bencher.iter(|| {
                    black_box(gpu.tiled_max_2d_gpu(&data, w, h).unwrap());
                });
            },
        );

        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("Scalar", label),
            &(*width, *height),
            |bencher, &(w, h)| {
                let data = generate_matrix_data(w, h);

                bencher.iter(|| {
                    let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    black_box(max_val);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU tiled min reduction
fn bench_tiled_min(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping tiled reduction benchmarks");
        return;
    }

    let mut group = c.benchmark_group("tiled_min_reduction");
    group.sample_size(20);

    let sizes: Vec<(usize, usize, &str)> =
        vec![(1000, 1000, "1M"), (3162, 3163, "10M"), (5657, 5657, "32M")];

    for (width, height, label) in sizes.iter() {
        let total = width * height;
        group.throughput(Throughput::Elements(total as u64));

        // GPU Tiled Min
        group.bench_with_input(
            BenchmarkId::new("GPU_Tiled", label),
            &(*width, *height),
            |bencher, &(w, h)| {
                let data = generate_matrix_data(w, h);
                let mut gpu = GpuBackend::new();

                bencher.iter(|| {
                    black_box(gpu.tiled_min_2d_gpu(&data, w, h).unwrap());
                });
            },
        );

        // Scalar baseline
        group.bench_with_input(
            BenchmarkId::new("Scalar", label),
            &(*width, *height),
            |bencher, &(w, h)| {
                let data = generate_matrix_data(w, h);

                bencher.iter(|| {
                    let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
                    black_box(min_val);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_tiled_sum, bench_tiled_max, bench_tiled_min);
criterion_main!(benches);
