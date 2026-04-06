#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Benchmarks for element-wise arithmetic operations: add, sub, mul, scale, div, fma

use crate::generate_test_data;
use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::{Backend, Vector};

/// Benchmark element-wise addition
pub fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.add(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.add(&b).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.add(&b).unwrap());
            });
        });

        // AVX-512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.add(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise subtraction
pub fn bench_sub(c: &mut Criterion) {
    let mut group = c.benchmark_group("sub");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&b_data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.sub(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&b_data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.sub(&b).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&b_data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.sub(&b).unwrap());
            });
        });

        // AVX-512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX512);
            let b = Vector::from_slice_with_backend(&b_data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.sub(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise multiplication
pub fn bench_mul(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.mul(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.mul(&b).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.mul(&b).unwrap());
            });
        });

        // AVX-512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            let b = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.mul(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark scalar multiplication (vector * scalar)
pub fn bench_scale(c: &mut Criterion) {
    let mut group = c.benchmark_group("scale");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::Scalar);
            let scalar = 2.5f32;

            bencher.iter(|| {
                black_box(a.scale(scalar).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::SSE2);
            let scalar = 2.5f32;

            bencher.iter(|| {
                black_box(a.scale(scalar).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX2);
            let scalar = 2.5f32;

            bencher.iter(|| {
                black_box(a.scale(scalar).unwrap());
            });
        });

        // AVX-512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX512);
            let scalar = 2.5f32;

            bencher.iter(|| {
                black_box(a.scale(scalar).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark element-wise division
pub fn bench_div(c: &mut Criterion) {
    let mut group = c.benchmark_group("div");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&b_data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.div(&b).unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&b_data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.div(&b).unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&b_data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.div(&b).unwrap());
            });
        });

        // AVX-512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX512);
            let b = Vector::from_slice_with_backend(&b_data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.div(&b).unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark fused multiply-add (a*b+c)
pub fn bench_fma(c: &mut Criterion) {
    let mut group = c.benchmark_group("fma");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let c_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::Scalar);
            let b = Vector::from_slice_with_backend(&b_data, Backend::Scalar);
            let c = Vector::from_slice_with_backend(&c_data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.fma(&b, &c).unwrap());
            });
        });

        // SSE2 backend (note: SSE2 doesn't have FMA, uses separate mul+add)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let c_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::SSE2);
            let b = Vector::from_slice_with_backend(&b_data, Backend::SSE2);
            let c = Vector::from_slice_with_backend(&c_data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.fma(&b, &c).unwrap());
            });
        });

        // AVX2 backend (uses hardware FMA instruction)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let c_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX2);
            let b = Vector::from_slice_with_backend(&b_data, Backend::AVX2);
            let c = Vector::from_slice_with_backend(&c_data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.fma(&b, &c).unwrap());
            });
        });

        // AVX-512 backend (uses hardware FMA instruction)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let a_data = generate_test_data(size);
            let b_data = generate_test_data(size);
            let c_data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&a_data, Backend::AVX512);
            let b = Vector::from_slice_with_backend(&b_data, Backend::AVX512);
            let c = Vector::from_slice_with_backend(&c_data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.fma(&b, &c).unwrap());
            });
        });
    }

    group.finish();
}
