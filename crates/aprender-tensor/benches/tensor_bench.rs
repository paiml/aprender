#![allow(missing_docs, clippy::expect_used, clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_tensor::{einsum, matmul, Tensor};

fn make_matrix(m: usize, n: usize) -> Tensor {
    let data: Vec<f32> = (0..m * n).map(|i| ((i * 7 + 3) % 97) as f32 / 97.0).collect();
    Tensor::new(vec![m, n], data).expect("valid tensor")
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul");
    for &n in &[16, 64, 128, 256] {
        let a = make_matrix(n, n);
        let b = make_matrix(n, n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                black_box(matmul(black_box(&a), black_box(&b)).expect("matmul ok"));
            });
        });
    }
    group.finish();
}

fn bench_einsum_transpose(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_transpose");
    for &n in &[64, 128, 256] {
        let a = make_matrix(n, n);
        let b = make_matrix(n, n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                black_box(
                    einsum(black_box("ij,jk->ik"), black_box(&a), black_box(&b))
                        .expect("einsum ok"),
                );
            });
        });
    }
    group.finish();
}

fn bench_einsum_trace(c: &mut Criterion) {
    let mut group = c.benchmark_group("einsum_trace");
    for &n in &[64, 128, 256] {
        let a = make_matrix(n, n);
        let ident = {
            let mut data = vec![0.0f32; n * n];
            for i in 0..n {
                data[i * n + i] = 1.0;
            }
            Tensor::new(vec![n, n], data).expect("valid tensor")
        };
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                black_box(
                    einsum(black_box("ij,ji->"), black_box(&a), black_box(&ident))
                        .expect("trace ok"),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_einsum_transpose, bench_einsum_trace);
criterion_main!(benches);
