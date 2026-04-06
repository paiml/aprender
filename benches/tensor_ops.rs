// Benchmark suite for Realizar tensor operations
// Uses Criterion.rs for statistical benchmarking

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use realizar::Tensor;

fn benchmark_tensor_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_creation");

    for size in &[10, 100, 1000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let data: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let shape = vec![size];
            b.iter(|| {
                let t = Tensor::from_vec(black_box(shape.clone()), black_box(data.clone()));
                black_box(t)
            });
        });
    }

    group.finish();
}

fn benchmark_tensor_properties(c: &mut Criterion) {
    let t = Tensor::from_vec(vec![100, 100], vec![0.0; 10_000]).expect("test");

    c.bench_function("tensor_shape", |b| {
        b.iter(|| {
            let shape = black_box(&t).shape();
            black_box(shape)
        });
    });

    c.bench_function("tensor_ndim", |b| {
        b.iter(|| {
            let ndim = black_box(&t).ndim();
            black_box(ndim)
        });
    });

    c.bench_function("tensor_size", |b| {
        b.iter(|| {
            let size = black_box(&t).size();
            black_box(size)
        });
    });
}

criterion_group!(
    benches,
    benchmark_tensor_creation,
    benchmark_tensor_properties
);
criterion_main!(benches);
