//! Aggregation benchmarks (SIMD performance baseline)
//!
//! CORE-004: GPU kernel implementations with baseline SIMD benchmarks
//!
//! Toyota Way: Genchi Genbutsu (measure, don't guess)
//!
//! Note: GPU vs SIMD comparison benchmarks will be added in CORE-008/CORE-009
//! This file establishes SIMD performance baseline for trueno integration.
//!
//! Run with: cargo bench --bench aggregations

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno::Vector;

const SMALL_SIZE: usize = 1_000; // 1K rows
const MEDIUM_SIZE: usize = 1_000_000; // 1M rows

/// Benchmark SUM aggregation with trueno SIMD (f32)
fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_aggregation_f32");

    // Small dataset
    let small_data: Vec<f32> = (0..SMALL_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(BenchmarkId::new("trueno_simd", SMALL_SIZE), &small_data, |b, data| {
        b.iter(|| {
            let vec = Vector::from_slice(black_box(data));
            vec.sum()
        });
    });

    // Medium dataset
    let medium_data: Vec<f32> = (0..MEDIUM_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(
        BenchmarkId::new("trueno_simd", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| {
                let vec = Vector::from_slice(black_box(data));
                vec.sum()
            });
        },
    );

    // Scalar baseline for comparison
    group.bench_with_input(
        BenchmarkId::new("scalar_baseline", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| black_box(data).iter().sum::<f32>());
        },
    );

    group.finish();
}

/// Benchmark MIN aggregation (f32)
fn bench_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("min_aggregation_f32");

    let small_data: Vec<f32> = (0..SMALL_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(BenchmarkId::new("trueno_simd", SMALL_SIZE), &small_data, |b, data| {
        b.iter(|| {
            let vec = Vector::from_slice(black_box(data));
            vec.min()
        });
    });

    let medium_data: Vec<f32> = (0..MEDIUM_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(
        BenchmarkId::new("trueno_simd", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| {
                let vec = Vector::from_slice(black_box(data));
                vec.min()
            });
        },
    );

    // Scalar baseline
    group.bench_with_input(
        BenchmarkId::new("scalar_baseline", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| black_box(data).iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap()));
        },
    );

    group.finish();
}

/// Benchmark MAX aggregation (f32)
fn bench_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max_aggregation_f32");

    let small_data: Vec<f32> = (0..SMALL_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(BenchmarkId::new("trueno_simd", SMALL_SIZE), &small_data, |b, data| {
        b.iter(|| {
            let vec = Vector::from_slice(black_box(data));
            vec.max()
        });
    });

    let medium_data: Vec<f32> = (0..MEDIUM_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(
        BenchmarkId::new("trueno_simd", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| {
                let vec = Vector::from_slice(black_box(data));
                vec.max()
            });
        },
    );

    // Scalar baseline
    group.bench_with_input(
        BenchmarkId::new("scalar_baseline", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| black_box(data).iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()));
        },
    );

    group.finish();
}

/// Benchmark AVG aggregation (f32)
fn bench_avg(c: &mut Criterion) {
    let mut group = c.benchmark_group("avg_aggregation_f32");

    let small_data: Vec<f32> = (0..SMALL_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(BenchmarkId::new("trueno_simd", SMALL_SIZE), &small_data, |b, data| {
        b.iter(|| {
            let vec = Vector::from_slice(black_box(data));
            vec.mean()
        });
    });

    let medium_data: Vec<f32> = (0..MEDIUM_SIZE).map(|i| i as f32).collect();
    group.bench_with_input(
        BenchmarkId::new("trueno_simd", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| {
                let vec = Vector::from_slice(black_box(data));
                vec.mean()
            });
        },
    );

    // Scalar baseline
    group.bench_with_input(
        BenchmarkId::new("scalar_baseline", MEDIUM_SIZE),
        &medium_data,
        |b, data| {
            b.iter(|| {
                let sum: f32 = black_box(data).iter().sum();
                sum / data.len() as f32
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_sum, bench_min, bench_max, bench_avg);
criterion_main!(benches);
