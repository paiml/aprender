#![allow(missing_docs, clippy::expect_used, clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_solve::{cholesky, lu_factorize, qr_factorize, svd};

fn make_spd_matrix(n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = if i == j {
                (n + 1) as f32
            } else {
                1.0 / (1 + (i as i32 - j as i32).unsigned_abs() as usize) as f32
            };
        }
    }
    a
}

fn make_general_matrix(m: usize, n: usize) -> Vec<f32> {
    (0..m * n).map(|i| ((i * 7 + 3) % 97) as f32 / 97.0).collect()
}

fn bench_lu(c: &mut Criterion) {
    let mut group = c.benchmark_group("lu_factorize");
    for &n in &[8, 32, 64, 128] {
        let a = make_spd_matrix(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(lu_factorize(black_box(&a), n).expect("lu ok"));
            });
        });
    }
    group.finish();
}

fn bench_qr(c: &mut Criterion) {
    let mut group = c.benchmark_group("qr_factorize");
    for &n in &[8, 32, 64, 128] {
        let a = make_general_matrix(n, n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(qr_factorize(black_box(&a), n, n).expect("qr ok"));
            });
        });
    }
    group.finish();
}

fn bench_cholesky(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky");
    for &n in &[8, 32, 64, 128] {
        let a = make_spd_matrix(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(cholesky(black_box(&a), n).expect("cholesky ok"));
            });
        });
    }
    group.finish();
}

fn bench_svd(c: &mut Criterion) {
    let mut group = c.benchmark_group("svd");
    for &n in &[8, 16, 32, 64] {
        let a = make_general_matrix(n, n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(svd(black_box(&a), n, n).expect("svd ok"));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_lu, bench_qr, bench_cholesky, bench_svd);
criterion_main!(benches);
