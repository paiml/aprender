//! Backend Selection Benchmarks
//!
//! Validates the Mixture-of-Experts (MoE) backend selection algorithm
//! by measuring actual performance across CPU, SIMD, and GPU backends.
//!
//! Implements the 5× PCIe rule from Gregg & Hazelwood (2011):
//! GPU is beneficial when compute_time > 5× transfer_time
//!
//! Run with: cargo bench --bench backend_selection

use batuta::backend::{BackendSelector, OpComplexity};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;

/// Benchmark backend selection for different data sizes
fn bench_backend_selection_by_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("backend_selection_by_size");

    let selector = BackendSelector::new();
    let data_sizes = vec![
        1_000,      // 1K elements - expect CPU
        10_000,     // 10K elements - expect CPU/SIMD threshold
        100_000,    // 100K elements - expect SIMD
        1_000_000,  // 1M elements - expect SIMD/GPU threshold
        10_000_000, // 10M elements - expect GPU
    ];

    for size in data_sizes {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("matmul", size), &size, |b, &size| {
            b.iter(|| {
                let backend =
                    selector.select_for_matmul(black_box(size), black_box(size), black_box(size));
                black_box(backend);
            });
        });
    }

    group.finish();
}

/// Benchmark backend selection for different complexity levels (MoE)
fn bench_moe_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("moe_selection");

    let selector = BackendSelector::new();
    let complexities = vec![
        (OpComplexity::Low, "low"),
        (OpComplexity::Medium, "medium"),
        (OpComplexity::High, "high"),
    ];

    for (complexity, name) in complexities {
        group.bench_with_input(BenchmarkId::new("select", name), &complexity, |b, &complexity| {
            b.iter(|| {
                let backend = selector.select_with_moe(black_box(complexity), black_box(100_000));
                black_box(backend);
            });
        });
    }

    group.finish();
}

/// Benchmark selection overhead (how fast is the selection itself?)
fn bench_selection_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection_overhead");

    let selector = BackendSelector::new();

    // Benchmark raw selection calls
    group.bench_function("simple_select", |b| {
        b.iter(|| {
            let backend =
                selector.select_for_matmul(black_box(1000), black_box(1000), black_box(1000));
            black_box(backend);
        });
    });

    group.bench_function("moe_select", |b| {
        b.iter(|| {
            let backend =
                selector.select_with_moe(black_box(OpComplexity::High), black_box(1_000_000));
            black_box(backend);
        });
    });

    group.finish();
}

/// Benchmark PCIe transfer cost calculation
fn bench_pcie_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("pcie_calculation");

    let _selector = BackendSelector::new();
    let data_sizes = vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000];

    for size in data_sizes {
        group.bench_with_input(BenchmarkId::new("pcie_cost", size), &size, |b, &size| {
            b.iter(|| {
                // Calculate transfer cost (data_bytes / bandwidth)
                let data_bytes = black_box(size * 4); // f32 = 4 bytes
                let bandwidth = 16_000_000_000.0; // 16 GB/s PCIe 3.0
                let transfer_time = data_bytes as f64 / bandwidth;
                black_box(transfer_time);
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_backend_selection_by_size,
    bench_moe_selection,
    bench_selection_overhead,
    bench_pcie_calculation,
);

criterion_main!(benches);
