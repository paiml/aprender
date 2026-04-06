#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! GPU activation function benchmarks (relu, leaky_relu, elu, clip)

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Sizes used for all activation benchmarks
const BENCH_SIZES: [usize; 3] = [10_000, 100_000, 1_000_000];

/// Generate centered test data: values in `[-size*0.25, size*0.25)` with step 0.5
fn make_centered_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.5 - (size as f32) * 0.25).collect()
}

/// Generate positive-only test data: values in `[0, size*0.5)` with step 0.5
fn make_positive_data(size: usize) -> Vec<f32> {
    (0..size).map(|i| (i as f32) * 0.5).collect()
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

/// Benchmark GPU ReLU activation vs scalar baseline
pub fn bench_gpu_relu(c: &mut Criterion) {
    run_bench(c, "gpu_relu", make_centered_data, &|gpu, data| gpu.relu(data).unwrap(), &|data| {
        data.iter().map(|&x| x.max(0.0)).collect()
    });
}

/// Benchmark GPU leaky ReLU activation vs scalar baseline
pub fn bench_gpu_leaky_relu(c: &mut Criterion) {
    run_bench(
        c,
        "gpu_leaky_relu",
        make_centered_data,
        &|gpu, data| gpu.leaky_relu(data, 0.01).unwrap(),
        &|data| {
            let s = 0.01_f32;
            data.iter().map(|&x| if x > 0.0 { x } else { s * x }).collect()
        },
    );
}

/// Benchmark GPU ELU activation vs scalar baseline
pub fn bench_gpu_elu(c: &mut Criterion) {
    run_bench(
        c,
        "gpu_elu",
        make_centered_data,
        &|gpu, data| gpu.elu(data, 1.0).unwrap(),
        &|data| data.iter().map(|&x| if x > 0.0 { x } else { x.exp() - 1.0 }).collect(),
    );
}

/// Benchmark GPU clip operation vs scalar baseline
pub fn bench_gpu_clip(c: &mut Criterion) {
    run_bench(
        c,
        "gpu_clip",
        make_positive_data,
        &|gpu, data| gpu.clip(data, 100.0, 5000.0).unwrap(),
        &|data| data.iter().map(|&x| x.max(100.0).min(5000.0)).collect(),
    );
}
