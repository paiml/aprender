//! SIMD Visualization Benchmark (Sprint 52-55)
//!
//! This benchmark validates the >4x speedup requirement for SIMD-accelerated
//! visualization functions using trueno-viz kernels.
//!
//! # Performance Targets
//!
//! - **SIMD aggregation:** >4x speedup vs scalar baseline
//! - **Batch normalization:** >4x speedup vs scalar baseline
//!
//! # Run Instructions
//!
//! ```bash
//! RUSTFLAGS="-C target-feature=+avx2" cargo bench --bench visualization_simd
//! ```
//!
//! # Expected Output
//!
//! ```text
//! simd_stats_300          time:   [50 ns 60 ns 70 ns]
//! scalar_stats_300        time:   [300 ns 350 ns 400 ns]
//! ```
//!
//! # Peer-Reviewed Foundation
//!
//! - **Lemire & Langdale (2019). "simdjson: Parsing Gigabytes of JSON per Second."**
//!   - Finding: SIMD can achieve 10x+ speedup for data-parallel operations
//!   - Application: Visualization aggregation is data-parallel

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use renacer::visualize::ring_buffer::HistoryBuffer;

/// Generate test data of specified size
fn generate_data(size: usize) -> Vec<f64> {
    (0..size).map(|i| i as f64 * 0.5).collect()
}

/// Benchmark: SIMD statistics via HistoryBuffer (uses trueno-viz kernels)
fn bench_simd_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_stats");

    for size in [100, 300, 1000, 10_000] {
        // Benchmark SIMD stats (via HistoryBuffer.stats())
        group.bench_with_input(BenchmarkId::new("simd_stats", size), &size, |b, &size| {
            let mut buf = HistoryBuffer::new(size);
            for v in generate_data(size) {
                buf.push(v);
            }

            b.iter(|| {
                black_box(buf.stats()) // Uses SIMD via trueno-viz
            });
        });

        // Benchmark scalar baseline
        group.bench_with_input(BenchmarkId::new("scalar_stats", size), &size, |b, &size| {
            let data = generate_data(size);

            b.iter(|| {
                let sum: f64 = data.iter().sum();
                let mean = sum / data.len() as f64;
                let min = data.iter().copied().fold(f64::INFINITY, f64::min);
                let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                black_box((min, max, mean))
            });
        });
    }

    group.finish();
}

/// Benchmark: SIMD sum
fn bench_simd_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_sum");

    for size in [100, 300, 1000, 10_000] {
        // Benchmark SIMD sum
        group.bench_with_input(BenchmarkId::new("simd_sum", size), &size, |b, &size| {
            let mut buf = HistoryBuffer::new(size);
            for v in generate_data(size) {
                buf.push(v);
            }

            b.iter(|| {
                black_box(buf.sum()) // Uses SIMD via trueno-viz
            });
        });

        // Benchmark scalar sum
        group.bench_with_input(BenchmarkId::new("scalar_sum", size), &size, |b, &size| {
            let data = generate_data(size);

            b.iter(|| {
                let sum: f64 = data.iter().sum();
                black_box(sum)
            });
        });
    }

    group.finish();
}

/// Benchmark: SIMD min/max
fn bench_simd_minmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar_minmax");

    for size in [100, 300, 1000, 10_000] {
        // Benchmark SIMD min/max
        group.bench_with_input(BenchmarkId::new("simd_minmax", size), &size, |b, &size| {
            let mut buf = HistoryBuffer::new(size);
            for v in generate_data(size) {
                buf.push(v);
            }

            b.iter(|| {
                let min = buf.min();
                let max = buf.max();
                black_box((min, max))
            });
        });

        // Benchmark scalar min/max
        group.bench_with_input(BenchmarkId::new("scalar_minmax", size), &size, |b, &size| {
            let data = generate_data(size);

            b.iter(|| {
                let min = data.iter().copied().fold(f64::INFINITY, f64::min);
                let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                black_box((min, max))
            });
        });
    }

    group.finish();
}

/// Benchmark: SIMD normalize batch (theme.rs)
fn bench_simd_normalize(c: &mut Criterion) {
    use renacer::visualize::theme::normalize_batch;

    let mut group = c.benchmark_group("simd_vs_scalar_normalize");

    for size in [100, 300, 1000, 10_000] {
        // Benchmark SIMD normalize
        group.bench_with_input(BenchmarkId::new("simd_normalize", size), &size, |b, &size| {
            let data = generate_data(size);

            b.iter(|| {
                black_box(normalize_batch(&data)) // Uses SIMD via trueno-viz
            });
        });

        // Benchmark scalar normalize
        group.bench_with_input(BenchmarkId::new("scalar_normalize", size), &size, |b, &size| {
            let data = generate_data(size);

            b.iter(|| {
                let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let normalized: Vec<f64> = if max == 0.0 {
                    vec![0.0; data.len()]
                } else {
                    data.iter().map(|v| v / max).collect()
                };
                black_box(normalized)
            });
        });
    }

    group.finish();
}

/// Benchmark: SIMD sparkline generation (uses SIMD min/max)
fn bench_simd_sparkline(c: &mut Criterion) {
    use renacer::visualize::theme::sparkline;

    let mut group = c.benchmark_group("simd_vs_scalar_sparkline");

    for size in [50, 100, 300] {
        // Benchmark SIMD sparkline
        group.bench_with_input(BenchmarkId::new("simd_sparkline", size), &size, |b, &size| {
            let data = generate_data(size);

            b.iter(|| {
                black_box(sparkline(&data, size)) // Uses SIMD for min/max
            });
        });

        // Benchmark scalar sparkline
        group.bench_with_input(BenchmarkId::new("scalar_sparkline", size), &size, |b, &size| {
            let data = generate_data(size);

            b.iter(|| {
                // Scalar min/max
                let min = data.iter().copied().fold(f64::INFINITY, f64::min);
                let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;

                let spark: String = data
                    .iter()
                    .take(size)
                    .map(|v| {
                        if range == 0.0 || v.is_nan() {
                            '▁'
                        } else {
                            let normalized = ((v - min) / range).clamp(0.0, 1.0);
                            let idx = (normalized * 7.0).round() as usize;
                            ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'][idx.min(7)]
                        }
                    })
                    .collect();
                black_box(spark)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_simd_stats,
    bench_simd_sum,
    bench_simd_minmax,
    bench_simd_normalize,
    bench_simd_sparkline,
);
criterion_main!(benches);
