//! Softmax benchmarks: aprender vs ndarray scalar.
//!
//! Softmax runs on the vocabulary dimension (32K–152K) at the final
//! output layer, and on seq_len² in attention. Both are hot paths.
//!
//! ## Reference
//!
//! ndarray implementation: subtract max, exp, divide by sum.
//! This is the textbook numerically-stable softmax.
//!
//! ## Throughput metric
//!
//! Reports elements/s.

use aprender::nn::F as NnF;
use aprender_bench_compute::sizes::SOFTMAX_SIZES;
use aprender_bench_compute::{deterministic_f32, deterministic_ndarray_1d};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_softmax_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_aprender");

    for &size in SOFTMAX_SIZES {
        let data = deterministic_f32(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("trueno", size), &size, |bench, _| {
            bench.iter(|| black_box(NnF::softmax_1d(&data)));
        });
    }

    group.finish();
}

fn bench_softmax_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_ndarray");

    for &size in SOFTMAX_SIZES {
        let arr = deterministic_ndarray_1d(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                // Numerically stable softmax: exp(x - max) / sum(exp(x - max))
                let max = arr.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp = arr.mapv(|x| (x - max).exp());
                let sum = exp.sum();
                black_box(exp / sum)
            });
        });
    }

    group.finish();
}

fn bench_softmax_1d_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_1d_aprender");

    // Also benchmark smaller attention-sized softmax
    let attn_sizes = [128, 512, 2048, 4096];
    for size in attn_sizes {
        let data = deterministic_f32(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("trueno", size), &size, |bench, _| {
            bench.iter(|| black_box(NnF::softmax_1d(&data)));
        });
    }

    group.finish();
}

fn bench_softmax_1d_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_1d_ndarray");

    let attn_sizes = [128, 512, 2048, 4096];
    for size in attn_sizes {
        let arr = deterministic_ndarray_1d(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |bench, _| {
            bench.iter(|| {
                let max = arr.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp = arr.mapv(|x| (x - max).exp());
                let sum = exp.sum();
                black_box(exp / sum)
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_softmax_aprender,
    bench_softmax_ndarray,
    bench_softmax_1d_aprender,
    bench_softmax_1d_ndarray,
);
criterion_main!(benches);
