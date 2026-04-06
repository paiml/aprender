#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Benchmarks for reduction operations: dot, sum, max, min, argmax, argmin

use crate::generate_test_data;
use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::{Backend, Vector};

/// Benchmark dot product
pub fn bench_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.dot(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.dot(&b).unwrap());
            });
        });

        // AVX2 backend (with FMA)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.dot(&b).unwrap());
            });
        });

        // AVX512 backend (with FMA, 16 elements per iteration)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.dot(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark sum reduction
pub fn bench_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.sum().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.sum().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.sum().unwrap());
            });
        });

        // AVX512 backend (16 elements per iteration)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.sum().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark max reduction
pub fn bench_max(c: &mut Criterion) {
    let mut group = c.benchmark_group("max");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.max().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.max().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.max().unwrap());
            });
        });

        // AVX512 backend (16 elements per iteration)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.max().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark min (find minimum value)
pub fn bench_min(c: &mut Criterion) {
    let mut group = c.benchmark_group("min");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.min().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.min().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.min().unwrap());
            });
        });

        // AVX512 backend (16 elements per iteration)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.min().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark argmax (find index of maximum value)
pub fn bench_argmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmax");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.argmax().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.argmax().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.argmax().unwrap());
            });
        });

        // AVX512 backend (16 elements per iteration)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.argmax().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark argmin (find index of minimum value)
pub fn bench_argmin(c: &mut Criterion) {
    let mut group = c.benchmark_group("argmin");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.argmin().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.argmin().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.argmin().unwrap());
            });
        });

        // AVX512 backend (16 elements per iteration)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.argmin().unwrap());
            });
        });
    }

    group.finish();
}
