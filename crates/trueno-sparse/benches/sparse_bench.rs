#![allow(missing_docs, clippy::expect_used, clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_sparse::{CooMatrix, CsrMatrix, SparseOps};

fn make_sparse_system(n: usize, nnz_per_row: usize) -> (CsrMatrix<f32>, Vec<f32>) {
    let mut row_idx = Vec::new();
    let mut col_idx = Vec::new();
    let mut vals = Vec::new();
    for i in 0..n {
        for k in 0..nnz_per_row {
            let j = (i + k) % n;
            row_idx.push(i as u32);
            col_idx.push(j as u32);
            #[allow(clippy::cast_precision_loss)]
            vals.push(1.0 / (1 + k) as f32);
        }
    }
    let coo = CooMatrix::new(n, n, row_idx, col_idx, vals).expect("valid coo");
    let csr = CsrMatrix::from_coo(&coo);
    let x = vec![1.0f32; n];
    (csr, x)
}

fn bench_spmv(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmv");
    for &n in &[64, 256, 1024] {
        let (csr, x) = make_sparse_system(n, 5);
        let mut y = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                y.fill(0.0);
                csr.spmv(black_box(1.0), black_box(&x), 0.0, &mut y).expect("spmv ok");
                black_box(&y);
            });
        });
    }
    group.finish();
}

fn bench_spmm(c: &mut Criterion) {
    let mut group = c.benchmark_group("spmm");
    for &n in &[64, 256] {
        let (csr, _) = make_sparse_system(n, 5);
        let k = 8;
        let b_dense = vec![1.0f32; n * k];
        let mut c_out = vec![0.0f32; n * k];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bench, _| {
            bench.iter(|| {
                c_out.fill(0.0);
                csr.spmm(black_box(1.0), black_box(&b_dense), black_box(k), 0.0, &mut c_out)
                    .expect("spmm ok");
                black_box(&c_out);
            });
        });
    }
    group.finish();
}

fn bench_coo_to_csr(c: &mut Criterion) {
    let mut group = c.benchmark_group("coo_to_csr");
    for &n in &[64, 256, 1024] {
        let mut row_idx = Vec::new();
        let mut col_idx = Vec::new();
        let mut vals = Vec::new();
        for i in 0..n {
            for k in 0..5usize {
                row_idx.push(i as u32);
                col_idx.push(((i + k) % n) as u32);
                vals.push(1.0f32);
            }
        }
        let coo = CooMatrix::new(n, n, row_idx, col_idx, vals).expect("valid coo");
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                black_box(CsrMatrix::from_coo(black_box(&coo)));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_spmv, bench_spmm, bench_coo_to_csr);
criterion_main!(benches);
