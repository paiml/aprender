//! Input-length scaling benchmarks: aprender vs ndarray.
//!
//! Verifies O(n) scaling for element-wise operations (GELU, RMSNorm)
//! and reveals fixed overhead (setup cost amortization).
//!
//! Plots throughput across 6 sizes (256 to 262K elements) for both
//! implementations. Both should scale linearly — any super-linear
//! behavior indicates cache effects or algorithmic issues.

use aprender::autograd::Tensor;
use aprender::nn::F;
use aprender_bench_compute::sizes::SCALING_SIZES;
use aprender_bench_compute::{deterministic_f32, deterministic_ndarray_1d};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_scaling_gelu(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_gelu");

    for &size in SCALING_SIZES {
        let data = deterministic_f32(size);
        let tensor = Tensor::new(&data, &[1, size]);
        let arr = deterministic_ndarray_1d(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| black_box(F::gelu(&tensor)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bench, _| {
            bench.iter(|| {
                black_box(arr.mapv(|x| {
                    let c = 0.797_884_6_f32;
                    x * 0.5 * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
                }))
            });
        });
    }

    group.finish();
}

fn bench_scaling_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_rmsnorm");

    for &size in SCALING_SIZES {
        let data = deterministic_f32(size);
        let tensor = Tensor::new(&data, &[1, size]);
        let weight = Tensor::new(&vec![1.0f32; size], &[size]);

        let arr = deterministic_ndarray_1d(size);
        let nd_weight: ndarray::Array1<f32> = ndarray::Array1::ones(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| black_box(F::rms_norm(&tensor, &weight, 1e-5)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bench, _| {
            bench.iter(|| {
                let sq_mean = arr.mapv(|x| x * x).mean().unwrap();
                let rms = (sq_mean + 1e-5).sqrt();
                black_box(&arr / rms * &nd_weight)
            });
        });
    }

    group.finish();
}

fn bench_scaling_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_softmax");

    for &size in SCALING_SIZES {
        let data = deterministic_f32(size);
        let arr = deterministic_ndarray_1d(size);

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("aprender", size), &size, |bench, _| {
            bench.iter(|| black_box(F::softmax_1d(&data)));
        });

        group.bench_with_input(BenchmarkId::new("ndarray", size), &size, |bench, _| {
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
    bench_scaling_gelu,
    bench_scaling_rmsnorm,
    bench_scaling_softmax,
);
criterion_main!(benches);
