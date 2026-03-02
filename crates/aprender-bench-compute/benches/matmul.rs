//! Matrix multiplication benchmarks: aprender/trueno vs ndarray.
//!
//! The #1 hotspot in LLM inference. Single-token decode is a matvec
//! (M=1, K=hidden, N=output) — this is where 90%+ of time goes.
//!
//! ## What we measure
//!
//! - **Matvec** (M=1): The autoregressive decode hot path
//! - **Matmul** (square): Prefill/batched inference
//! - Both implementations are pure Rust (no external BLAS)
//!
//! ## Throughput metric
//!
//! Reports FLOP/s: `2 × M × K × N` floating-point operations per matmul.

use aprender_bench_compute::sizes::{MATMUL_SIZES, MATVEC_SIZES};
use aprender_bench_compute::{deterministic_ndarray, deterministic_tensor};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_matmul_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_aprender");
    group.sample_size(20);

    for &(m, k, n) in MATMUL_SIZES {
        let flops = 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a = deterministic_tensor(m, k);
        let b = deterministic_tensor(k, n);

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

fn bench_matmul_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_ndarray");
    group.sample_size(20);

    for &(m, k, n) in MATMUL_SIZES {
        let flops = 2 * m * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let a = deterministic_ndarray(m, k);
        let b = deterministic_ndarray(k, n);

        group.bench_with_input(
            BenchmarkId::new("pure_rust", format!("{m}x{k}x{n}")),
            &(m, k, n),
            |bench, _| {
                bench.iter(|| black_box(a.dot(&b)));
            },
        );
    }

    group.finish();
}

fn bench_matvec_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_aprender");
    group.sample_size(30);

    for &(k, n) in MATVEC_SIZES {
        let flops = 2 * k * n; // M=1
        group.throughput(Throughput::Elements(flops as u64));

        let x = deterministic_tensor(1, k);
        let w = deterministic_tensor(k, n);

        group.bench_with_input(
            BenchmarkId::new("trueno", format!("1x{k}x{n}")),
            &(k, n),
            |bench, _| {
                bench.iter(|| black_box(x.matmul(&w)));
            },
        );
    }

    group.finish();
}

fn bench_matvec_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("matvec_ndarray");
    group.sample_size(30);

    for &(k, n) in MATVEC_SIZES {
        let flops = 2 * k * n;
        group.throughput(Throughput::Elements(flops as u64));

        let x = deterministic_ndarray(1, k);
        let w = deterministic_ndarray(k, n);

        group.bench_with_input(
            BenchmarkId::new("pure_rust", format!("1x{k}x{n}")),
            &(k, n),
            |bench, _| {
                bench.iter(|| black_box(x.dot(&w)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_matmul_aprender,
    bench_matmul_ndarray,
    bench_matvec_aprender,
    bench_matvec_ndarray,
);
criterion_main!(benches);
