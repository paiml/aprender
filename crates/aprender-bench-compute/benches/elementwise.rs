//! Element-wise operation benchmarks: aprender vs ndarray.
//!
//! Benchmarks add and mul_scalar — the most common element-wise ops
//! in neural network forward/backward passes. These are pure memory-bound
//! operations where allocation and data movement dominate.
//!
//! ## Reference
//!
//! ndarray uses operator overloading with zero-cost views. aprender wraps
//! data in Tensor (Arc<TensorInner>) with autograd metadata.
//!
//! ## Throughput metric
//!
//! Reports elements/s (one op per element).

use aprender::autograd::Tensor;
use aprender_bench_compute::sizes::ELEMENTWISE_SIZES;
use aprender_bench_compute::{deterministic_f32, deterministic_ndarray_1d};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_add_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_aprender");

    for &size in ELEMENTWISE_SIZES {
        let data_a = deterministic_f32(size);
        let data_b = deterministic_f32(size);
        let a = Tensor::new(&data_a, &[1, size]);
        let b = Tensor::new(&data_b, &[1, size]);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("tensor", size), &size, |bench, _| {
            bench.iter(|| black_box(a.add(&b)));
        });
    }

    group.finish();
}

fn bench_add_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_ndarray");

    for &size in ELEMENTWISE_SIZES {
        let a = deterministic_ndarray_1d(size);
        let b = deterministic_ndarray_1d(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bench, _| {
            bench.iter(|| black_box(&a + &b));
        });
    }

    group.finish();
}

fn bench_mul_scalar_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_scalar_aprender");

    for &size in ELEMENTWISE_SIZES {
        let data = deterministic_f32(size);
        let t = Tensor::new(&data, &[1, size]);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("tensor", size), &size, |bench, _| {
            bench.iter(|| black_box(t.mul_scalar(2.5)));
        });
    }

    group.finish();
}

fn bench_mul_scalar_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_scalar_ndarray");

    for &size in ELEMENTWISE_SIZES {
        let arr = deterministic_ndarray_1d(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bench, _| {
            bench.iter(|| black_box(&arr * 2.5));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_add_aprender,
    bench_add_ndarray,
    bench_mul_scalar_aprender,
    bench_mul_scalar_ndarray,
);
criterion_main!(benches);
