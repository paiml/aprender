#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! GPU matrix multiplication benchmarks

use criterion::{BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use trueno::backends::gpu::GpuBackend;

/// Generate matrix data for matmul benchmarks.
fn matrix_data(size: usize) -> Vec<f32> {
    (0..(size * size)).map(|i| i as f32 * 0.5).collect()
}

/// Benchmark GPU matrix multiplication vs scalar baseline
pub fn bench_gpu_matmul(c: &mut Criterion) {
    if !GpuBackend::is_available() {
        eprintln!("GPU not available, skipping GPU benchmarks");
        return;
    }

    let mut group = c.benchmark_group("gpu_matmul");

    for size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements((size * size * size) as u64));

        group.bench_with_input(BenchmarkId::new("GPU", size), size, |bencher, &size| {
            let data = matrix_data(size);
            let mut gpu = GpuBackend::new();
            bencher.iter(|| black_box(gpu.matmul(&data, &data, size, size, size).unwrap()));
        });

        group.bench_with_input(BenchmarkId::new("Scalar", size), size, |bencher, &size| {
            let data = matrix_data(size);
            bencher.iter(|| {
                let mut result = vec![0.0f32; size * size];
                for i in 0..size {
                    for j in 0..size {
                        let mut sum = 0.0;
                        for k in 0..size {
                            sum += data[i * size + k] * data[k * size + j];
                        }
                        result[i * size + j] = sum;
                    }
                }
                black_box(result);
            });
        });
    }

    group.finish();
}
