//! Activation function benchmarks: aprender vs ndarray element-wise.
//!
//! Activation functions (GELU, SiLU, ReLU) run on the FFN intermediate
//! dimension — typically 11008 for 7B models. These are memory-bound
//! operations where SIMD vectorization matters.
//!
//! ## Reference
//!
//! ndarray's `mapv` applies a scalar function element-wise. This is the
//! baseline: no SIMD, no fused ops, just a tight loop over f32 values.
//! aprender/trueno should beat this via SIMD vectorization.
//!
//! ## Throughput metric
//!
//! Reports elements/s (one activation per element).

use aprender::autograd::Tensor;
use aprender::nn::F;
use aprender_bench_compute::sizes::ACTIVATION_SIZES;
use aprender_bench_compute::{deterministic_f32, deterministic_ndarray_1d};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_gelu_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu_aprender");

    for &size in ACTIVATION_SIZES {
        let data = deterministic_f32(size);
        let tensor = Tensor::new(&data, &[1, size]);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("trueno", size), &size, |bench, _| {
            bench.iter(|| black_box(F::gelu(&tensor)));
        });
    }

    group.finish();
}

fn bench_gelu_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("gelu_ndarray");

    for &size in ACTIVATION_SIZES {
        let arr = deterministic_ndarray_1d(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mapv", size), &size, |bench, _| {
            bench.iter(|| {
                // GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                black_box(arr.mapv(|x| {
                    let c = 0.797_884_6_f32; // sqrt(2/pi)
                    x * 0.5 * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
                }))
            });
        });
    }

    group.finish();
}

fn bench_silu_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_aprender");

    for &size in ACTIVATION_SIZES {
        let data = deterministic_f32(size);
        let tensor = Tensor::new(&data, &[1, size]);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("trueno", size), &size, |bench, _| {
            bench.iter(|| black_box(F::silu(&tensor)));
        });
    }

    group.finish();
}

fn bench_silu_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("silu_ndarray");

    for &size in ACTIVATION_SIZES {
        let arr = deterministic_ndarray_1d(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mapv", size), &size, |bench, _| {
            bench.iter(|| {
                // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
                black_box(arr.mapv(|x| x / (1.0 + (-x).exp())))
            });
        });
    }

    group.finish();
}

fn bench_relu_aprender(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu_aprender");

    for &size in ACTIVATION_SIZES {
        let data = deterministic_f32(size);
        let tensor = Tensor::new(&data, &[1, size]);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("trueno", size), &size, |bench, _| {
            bench.iter(|| black_box(tensor.relu()));
        });
    }

    group.finish();
}

fn bench_relu_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("relu_ndarray");

    for &size in ACTIVATION_SIZES {
        let arr = deterministic_ndarray_1d(size);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("mapv", size), &size, |bench, _| {
            bench.iter(|| black_box(arr.mapv(|x| x.max(0.0))));
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_gelu_aprender,
    bench_gelu_ndarray,
    bench_silu_aprender,
    bench_silu_ndarray,
    bench_relu_aprender,
    bench_relu_ndarray,
);
criterion_main!(benches);
