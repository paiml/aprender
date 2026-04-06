#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Benchmarks for math/unary operations: norms, abs, exp, ln, log2, log10, sqrt, recip

use crate::generate_test_data;
use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::{Backend, Vector};

/// Benchmark L1 norm (sum of absolute values)
pub fn bench_norm_l1(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_l1");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.norm_l1().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.norm_l1().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.norm_l1().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark L2 norm (Euclidean norm, sqrt of sum of squares)
pub fn bench_norm_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_l2");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.norm_l2().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.norm_l2().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.norm_l2().unwrap());
            });
        });

        // AVX-512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);

            bencher.iter(|| {
                black_box(a.norm_l2().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark L-infinity norm (max absolute value) - currently uses temp allocation
pub fn bench_norm_linf(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_linf");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.norm_linf().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.norm_linf().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.norm_linf().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark absolute value (currently scalar-only, no SIMD backend)
pub fn bench_abs(c: &mut Criterion) {
    let mut group = c.benchmark_group("abs");

    for size in [100, 1000, 10000, 100000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.abs().unwrap());
            });
        });

        // SSE2 backend (currently uses same scalar code)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.abs().unwrap());
            });
        });

        // AVX2 backend (currently uses same scalar code)
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data = generate_test_data(size);
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.abs().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark exp() operation (transcendental function with range reduction)
pub fn bench_exp(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32) * 4.0 - 2.0).collect();
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);

            bencher.iter(|| {
                black_box(a.exp().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32) * 4.0 - 2.0).collect();
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);

            bencher.iter(|| {
                black_box(a.exp().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, &size| {
            let data: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32) * 4.0 - 2.0).collect();
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);

            bencher.iter(|| {
                black_box(a.exp().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark natural logarithm (ln)
pub fn bench_ln(c: &mut Criterion) {
    let mut group = c.benchmark_group("ln");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate positive values in range [0.1, 100.0] for logarithm
        let data: Vec<f32> = (0..*size).map(|i| 0.1 + (i as f32 / *size as f32) * 99.9).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(a.ln().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(a.ln().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(a.ln().unwrap());
            });
        });

        // AVX512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            bencher.iter(|| {
                black_box(a.ln().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark base-2 logarithm (log2)
pub fn bench_log2(c: &mut Criterion) {
    let mut group = c.benchmark_group("log2");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate positive values in range [0.1, 100.0] for logarithm
        let data: Vec<f32> = (0..*size).map(|i| 0.1 + (i as f32 / *size as f32) * 99.9).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(a.log2().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(a.log2().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(a.log2().unwrap());
            });
        });

        // AVX512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            bencher.iter(|| {
                black_box(a.log2().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark base-10 logarithm (log10)
pub fn bench_log10(c: &mut Criterion) {
    let mut group = c.benchmark_group("log10");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate positive values in range [0.1, 100.0] for logarithm
        let data: Vec<f32> = (0..*size).map(|i| 0.1 + (i as f32 / *size as f32) * 99.9).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(a.log10().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(a.log10().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(a.log10().unwrap());
            });
        });

        // AVX512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            bencher.iter(|| {
                black_box(a.log10().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark square root (sqrt)
pub fn bench_sqrt(c: &mut Criterion) {
    let mut group = c.benchmark_group("sqrt");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate positive values in range [0.1, 100.0] for sqrt
        let data: Vec<f32> = (0..*size).map(|i| 0.1 + (i as f32 / *size as f32) * 99.9).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(a.sqrt().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(a.sqrt().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(a.sqrt().unwrap());
            });
        });

        // AVX512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            bencher.iter(|| {
                black_box(a.sqrt().unwrap());
            });
        });
    }

    group.finish();
}

/// Benchmark reciprocal (1/x)
pub fn bench_recip(c: &mut Criterion) {
    let mut group = c.benchmark_group("recip");

    for size in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        // Generate non-zero values in range [0.1, 100.0] to avoid division by zero
        let data: Vec<f32> = (0..*size).map(|i| 0.1 + (i as f32 / *size as f32) * 99.9).collect();

        // Scalar backend
        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::Scalar);
            bencher.iter(|| {
                black_box(a.recip().unwrap());
            });
        });

        // SSE2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("SSE2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::SSE2);
            bencher.iter(|| {
                black_box(a.recip().unwrap());
            });
        });

        // AVX2 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX2", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX2);
            bencher.iter(|| {
                black_box(a.recip().unwrap());
            });
        });

        // AVX512 backend
        #[cfg(target_arch = "x86_64")]
        group.bench_with_input(BenchmarkId::new("AVX512", size), size, |bencher, _size| {
            let a = Vector::from_slice_with_backend(&data, Backend::AVX512);
            bencher.iter(|| {
                black_box(a.recip().unwrap());
            });
        });
    }

    group.finish();
}
