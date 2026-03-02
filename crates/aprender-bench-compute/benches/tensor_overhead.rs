//! Tensor creation overhead benchmarks.
//!
//! Isolates the cost of Tensor::new() vs raw Vec<f32> allocation.
//! This is the systemic root cause for memory-bound ops being slower
//! than ndarray (#385): each intermediate result creates a new Tensor
//! with shape metadata, strides, and reference counting.
//!
//! ## What we measure
//!
//! 1. Tensor::new() from data — full construction cost
//! 2. Vec<f32> allocation — raw memory baseline
//! 3. ndarray::Array1 from vec — ndarray's equivalent overhead
//!
//! If Tensor::new() is significantly more expensive than Vec/Array1,
//! that explains the 2-4x gap on memory-bound operations.

use aprender::autograd::Tensor;
use aprender_bench_compute::deterministic_f32;
use aprender_bench_compute::sizes::ACTIVATION_SIZES;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_tensor_create(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_create");

    for &size in ACTIVATION_SIZES {
        let data = deterministic_f32(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Tensor_new", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(Tensor::new(&data, &[1, size])));
            },
        );

        group.bench_with_input(BenchmarkId::new("Vec_clone", size), &size, |bench, _| {
            bench.iter(|| black_box(data.clone()));
        });

        group.bench_with_input(
            BenchmarkId::new("ndarray_from_vec", size),
            &size,
            |bench, _| {
                bench.iter(|| {
                    let d = data.clone();
                    black_box(ndarray::Array1::from_vec(d))
                });
            },
        );
    }

    group.finish();
}

fn bench_tensor_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_add");

    for &size in ACTIVATION_SIZES {
        let data = deterministic_f32(size);
        let t1 = Tensor::new(&data, &[1, size]);
        let t2 = Tensor::new(&data, &[1, size]);
        let a1 = ndarray::Array1::from_vec(data.clone());
        let a2 = ndarray::Array1::from_vec(data.clone());
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Tensor_add", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(t1.add(&t2)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_add", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(&a1 + &a2));
            },
        );
    }

    group.finish();
}

fn bench_tensor_mul_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_mul_scalar");

    for &size in ACTIVATION_SIZES {
        let data = deterministic_f32(size);
        let t = Tensor::new(&data, &[1, size]);
        let a = ndarray::Array1::from_vec(data.clone());
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(
            BenchmarkId::new("Tensor_mul_scalar", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(t.mul_scalar(0.5)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("ndarray_mul_scalar", size),
            &size,
            |bench, _| {
                bench.iter(|| black_box(&a * 0.5f32));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tensor_create,
    bench_tensor_add,
    bench_tensor_mul_scalar,
);
criterion_main!(benches);
