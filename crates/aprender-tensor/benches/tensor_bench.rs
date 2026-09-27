#![allow(missing_docs, clippy::expect_used, clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_tensor::{einsum, matmul, Matrix, Tensor};

fn make_matrix(m: usize, n: usize) -> Tensor {
    let data: Vec<f32> = (0..m * n)
        .map(|i| ((i * 7 + 3) % 97) as f32 / 97.0)
        .collect();
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
/// #3150 zero-cost evidence: the old dynamic route (`einsum "ij,jk->ik"`), the
/// migrated `matmul` (rank-checks and copies into `Matrix`, then back), and the
/// typed kernel called directly. The typed path carries no rank checks at all;
/// what `matmul` adds over `Matrix::matmul` is the `from_dynamic` copy.
fn bench_rank_typed_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_typed_matmul");
    for &n in &[16, 64, 256] {
        let a = make_matrix(n, n);
        let b = make_matrix(n, n);
        let ta = Matrix::from_dynamic(&a).expect("rank 2");
        let tb = Matrix::from_dynamic(&b).expect("rank 2");
        group.bench_with_input(BenchmarkId::new("einsum_route", n), &n, |bench, _| {
            bench
                .iter(|| black_box(einsum("ij,jk->ik", black_box(&a), black_box(&b)).expect("ok")));
        });
        group.bench_with_input(BenchmarkId::new("dynamic_matmul", n), &n, |bench, _| {
            bench.iter(|| black_box(matmul(black_box(&a), black_box(&b)).expect("ok")));
        });
        group.bench_with_input(BenchmarkId::new("typed_matmul", n), &n, |bench, _| {
            bench.iter(|| black_box(black_box(&ta).matmul(black_box(&tb)).expect("ok")));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_matmul,
    bench_einsum_transpose,
    bench_einsum_trace,
    bench_rank_typed_matmul
);
criterion_main!(benches);
