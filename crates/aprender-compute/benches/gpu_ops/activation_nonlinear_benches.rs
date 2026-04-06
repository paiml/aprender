#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! GPU non-linear activation benchmarks (sigmoid, tanh, swish, gelu)

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Sizes used for all activation benchmarks
const BENCH_SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];

/// Generate narrow-range test data: values in `[-size*0.0005, size*0.0005)` with step 0.001
fn make_narrow_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.001 - (size as f32) * 0.0005).collect()
}

/// Run a GPU-vs-Scalar activation benchmark over standard sizes.
fn run_bench(
    c: &mut Criterion,
    name: &str,
    data_fn: fn(usize) -> Vec<f32>,
    gpu_op: &dyn Fn(&mut GpuBackend, &[f32]) -> Vec<f32>,
    scalar_op: &dyn Fn(&[f32]) -> Vec<f32>,
) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group(name);

    for &size in BENCH_SIZES.iter() {
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), &size, |bencher, &size| {
            let data = data_fn(size);
            let mut gpu = GpuBackend::new();
            bencher.iter(|| black_box(gpu_op(&mut gpu, &data)));
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), &size, |bencher, &size| {
            let data = data_fn(size);
            bencher.iter(|| black_box(scalar_op(&data)));
        });
    }

    group.finish();
}

/// Benchmark GPU sigmoid activation vs scalar baseline
pub fn bench_gpu_sigmoid(c: &mut Criterion) {
    run_bench(
        c,
        "gpu_sigmoid",
        make_narrow_data,
        &|gpu, data| gpu.sigmoid(data).unwrap(),
        &|data| data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
    );
}

/// Benchmark GPU tanh activation vs scalar baseline
pub fn bench_gpu_tanh(c: &mut Criterion) {
    run_bench(c, "gpu_tanh", make_narrow_data, &|gpu, data| gpu.tanh(data).unwrap(), &|data| {
        data.iter().map(|&x| x.tanh()).collect()
    });
}

/// Benchmark GPU swish activation vs scalar baseline
pub fn bench_gpu_swish(c: &mut Criterion) {
    run_bench(c, "gpu_swish", make_narrow_data, &|gpu, data| gpu.swish(data).unwrap(), &|data| {
        data.iter().map(|&x| x / (1.0 + (-x).exp())).collect()
    });
}

/// Benchmark GPU GELU activation vs scalar baseline
pub fn bench_gpu_gelu(c: &mut Criterion) {
    run_bench(c, "gpu_gelu", make_narrow_data, &|gpu, data| gpu.gelu(data).unwrap(), &|data| {
        const SQRT_2_OVER_PI: f32 = 0.7978846;
        const COEFF: f32 = 0.044715;
        data.iter()
            .map(|&x| {
                let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect()
    });
}
