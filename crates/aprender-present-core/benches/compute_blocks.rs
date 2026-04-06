//! Benchmark tests for `ComputeBlock` aggregation primitives.
//!
//! Tests SIMD-friendly aggregation functions across different data sizes
//! to verify auto-vectorization performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use presentar_core::simd::{
    batch_mean_f64, batch_min_max_f64, batch_scale_f64, batch_sum_f64, batch_variance_f64,
    histogram_f64, normalize_f64, weighted_sum_f64,
};

// =============================================================================
// Sum Benchmarks
// =============================================================================

fn bench_batch_sum_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_sum_f64");

    for size in &[8, 64, 128, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(f64::from).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| batch_sum_f64(black_box(data)));
        });
    }

    group.finish();
}

fn bench_scalar_sum_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_sum_baseline");

    for size in &[8, 64, 128, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(f64::from).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let sum: f64 = black_box(data).iter().sum();
                sum
            });
        });
    }

    group.finish();
}

// =============================================================================
// Mean Benchmarks
// =============================================================================

fn bench_batch_mean_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_mean_f64");

    for size in &[8, 64, 128, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(f64::from).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| batch_mean_f64(black_box(data)));
        });
    }

    group.finish();
}

// =============================================================================
// Min/Max Benchmarks
// =============================================================================

fn bench_batch_min_max_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_min_max_f64");

    for size in &[8, 64, 128, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(|i| f64::from(i % 100)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| batch_min_max_f64(black_box(data)));
        });
    }

    group.finish();
}

fn bench_scalar_min_max_baseline(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar_min_max_baseline");

    for size in &[8, 64, 128, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(|i| f64::from(i % 100)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| {
                let data = black_box(data);
                let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                (min, max)
            });
        });
    }

    group.finish();
}

// =============================================================================
// Normalization Benchmarks
// =============================================================================

fn bench_normalize_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("normalize_f64");

    for size in &[64, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(f64::from).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| normalize_f64(black_box(data)));
        });
    }

    group.finish();
}

// =============================================================================
// Scale Benchmarks
// =============================================================================

fn bench_batch_scale_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_scale_f64");

    for size in &[64, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(f64::from).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| batch_scale_f64(black_box(data), 2.0));
        });
    }

    group.finish();
}

// =============================================================================
// Variance Benchmarks
// =============================================================================

fn bench_batch_variance_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_variance_f64");

    for size in &[64, 256, 1024] {
        let data: Vec<f64> = (0..*size).map(|i| f64::from(i % 100)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| batch_variance_f64(black_box(data)));
        });
    }

    group.finish();
}

// =============================================================================
// Weighted Sum Benchmarks
// =============================================================================

fn bench_weighted_sum_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_sum_f64");

    for size in &[64, 256, 1024] {
        let values: Vec<f64> = (0..*size).map(f64::from).collect();
        let weights: Vec<f64> = (0..*size).map(|i| 1.0 / f64::from(i + 1)).collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &(values.clone(), weights.clone()),
            |b, (v, w)| {
                b.iter(|| weighted_sum_f64(black_box(v), black_box(w)));
            },
        );
    }

    group.finish();
}

// =============================================================================
// Histogram Benchmarks
// =============================================================================

fn bench_histogram_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram_f64");

    for size in &[256, 1024, 4096] {
        let data: Vec<f64> = (0..*size).map(|i| f64::from(i % 100)).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &data, |b, data| {
            b.iter(|| histogram_f64(black_box(data), 0.0, 100.0, 10));
        });
    }

    group.finish();
}

// =============================================================================
// Groups
// =============================================================================

criterion_group!(
    benches,
    bench_batch_sum_f64,
    bench_scalar_sum_baseline,
    bench_batch_mean_f64,
    bench_batch_min_max_f64,
    bench_scalar_min_max_baseline,
    bench_normalize_f64,
    bench_batch_scale_f64,
    bench_batch_variance_f64,
    bench_weighted_sum_f64,
    bench_histogram_f64,
);
criterion_main!(benches);
