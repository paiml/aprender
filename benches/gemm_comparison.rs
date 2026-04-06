#![allow(clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
fn gen(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect()
}

fn bench_transpose(c: &mut Criterion) {
    let mut g = c.benchmark_group("transpose");
    for &n in &[64, 128, 256, 512] {
        let d = gen(n * n);
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |b, &n| {
            let mut o = vec![0.0f32; n * n];
            b.iter(|| trueno::blis::transpose(n, n, black_box(&d), black_box(&mut o)).unwrap());
        });
        // FAIR comparison: as_standard_layout forces real data rearrangement
        // (not just memcpy with swapped strides)
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |b, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), d.clone()).unwrap();
            b.iter(|| black_box(black_box(&a).t().as_standard_layout().into_owned()));
        });
    }
    g.finish();
}

fn bench_gemm(c: &mut Criterion) {
    let mut g = c.benchmark_group("gemm");
    g.sample_size(20);
    for &n in &[64, 128, 256, 512, 1024] {
        let a = gen(n * n);
        let b = gen(n * n);
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut c = vec![0.0f32; n * n];
            bench.iter(|| {
                trueno::blis::gemm(n, n, n, black_box(&a), black_box(&b), black_box(&mut c))
                    .unwrap()
            });
        });
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), a.clone()).unwrap();
            let b = ndarray::Array2::from_shape_vec((n, n), b.clone()).unwrap();
            bench.iter(|| black_box(black_box(&a).dot(black_box(&b))));
        });
        // faer: fastest pure-Rust BLAS (uses runtime SIMD dispatch)
        // Pre-allocate output to measure compute only (same as trueno)
        g.bench_with_input(BenchmarkId::new("faer", n), &n, |bench, &n| {
            let fa = faer::Mat::<f32>::from_fn(n, n, |i, j| a[i * n + j]);
            let fb = faer::Mat::<f32>::from_fn(n, n, |i, j| b[i * n + j]);
            let mut fc = faer::Mat::<f32>::zeros(n, n);
            bench.iter(|| {
                faer::linalg::matmul::matmul(
                    black_box(fc.as_mut()),
                    faer::Accum::Replace,
                    black_box(fa.as_ref()),
                    black_box(fb.as_ref()),
                    1.0f32,
                    faer::Par::Seq,
                );
            });
        });
        // nalgebra: popular Rust linear algebra (uses matrixmultiply crate)
        g.bench_with_input(BenchmarkId::new("nalgebra", n), &n, |bench, &n| {
            let a = nalgebra::DMatrix::from_row_slice(n, n, &a);
            let b = nalgebra::DMatrix::from_row_slice(n, n, &b);
            bench.iter(|| black_box(black_box(&a) * black_box(&b)));
        });
        // matrixmultiply: raw sgemm (the engine behind ndarray GEMM)
        g.bench_with_input(BenchmarkId::new("matrixmultiply", n), &n, |bench, &n| {
            let mut c = vec![0.0f32; n * n];
            bench.iter(|| {
                // SAFETY: calling raw BLAS-style sgemm with valid pointers and strides
                unsafe {
                    matrixmultiply::sgemm(
                        n,
                        n,
                        n,
                        1.0,
                        black_box(a.as_ptr()),
                        n as isize,
                        1,
                        black_box(b.as_ptr()),
                        n as isize,
                        1,
                        0.0,
                        black_box(c.as_mut_ptr()),
                        n as isize,
                        1,
                    );
                }
            });
        });
    }
    g.finish();
}

fn bench_gemv(c: &mut Criterion) {
    let mut g = c.benchmark_group("gemv");
    for &n in &[64, 128, 256, 512, 1024] {
        let m = gen(n * n);
        let v = gen(n);
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut o = vec![0.0f32; n];
            bench.iter(|| {
                trueno::blis::gemv::gemv(n, n, black_box(&m), black_box(&v), black_box(&mut o));
            });
        });
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), m.clone()).unwrap();
            let x = ndarray::Array1::from_vec(v.clone());
            bench.iter(|| black_box(black_box(&a).dot(black_box(&x))));
        });
    }
    g.finish();
}

fn bench_add(c: &mut Criterion) {
    let mut g = c.benchmark_group("vec_add");
    for &n in &[1000, 10_000, 100_000, 1_000_000] {
        let a = gen(n);
        let b = gen(n);
        // Pre-allocated output for compute-only comparison.
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut o = vec![0.0f32; n];
            bench.iter(|| {
                trueno::blis::elementwise::add(black_box(&a), black_box(&b), black_box(&mut o))
                    .unwrap()
            });
        });
        // ndarray also pre-allocated for fair compute comparison.
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array1::from_vec(a.clone());
            let b = ndarray::Array1::from_vec(b.clone());
            let mut o = ndarray::Array1::zeros(n);
            bench.iter(|| {
                ndarray::Zip::from(black_box(&a))
                    .and(black_box(&b))
                    .and(black_box(&mut o))
                    .for_each(|&a, &b, o| *o = a + b);
            });
        });
    }
    g.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut g = c.benchmark_group("relu");
    for &n in &[1000, 10_000, 100_000, 1_000_000] {
        let d: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) / 100.0).collect();
        // Pre-allocated output for compute-only comparison.
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &n| {
            let mut o = vec![0.0f32; n];
            bench.iter(|| {
                trueno::blis::elementwise::relu(black_box(&d), black_box(&mut o)).unwrap()
            });
        });
        // ndarray also pre-allocated for fair compute comparison.
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &n| {
            let a = ndarray::Array1::from_vec(d.clone());
            let mut o = ndarray::Array1::zeros(n);
            bench.iter(|| {
                ndarray::Zip::from(black_box(&a))
                    .and(black_box(&mut o))
                    .for_each(|&a, o| *o = a.max(0.0));
            });
        });
    }
    g.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut g = c.benchmark_group("softmax");
    for &n in &[128, 1024, 4096, 32768] {
        let d = gen(n);
        g.bench_with_input(BenchmarkId::new("trueno", n), &n, |bench, &_n| {
            bench.iter(|| black_box(trueno::blis::softmax::softmax_1d_alloc(black_box(&d))));
        });
        g.bench_with_input(BenchmarkId::new("ndarray", n), &n, |bench, &_n| {
            let a = ndarray::Array1::from_vec(d.clone());
            bench.iter(|| {
                let mx = a.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: ndarray::Array1<f32> = a.mapv(|x| (x - mx).exp());
                let s = e.sum();
                black_box(e / s)
            });
        });
    }
    g.finish();
}

fn bench_fused_add_relu(c: &mut Criterion) {
    let mut g = c.benchmark_group("fused_add_relu");
    for &n in &[1000, 10_000, 100_000, 1_000_000] {
        let a = gen(n);
        let b = gen(n);
        // Trueno: single-pass fused add+relu
        g.bench_with_input(BenchmarkId::new("trueno_fused", n), &n, |bench, &n| {
            let mut o = vec![0.0f32; n];
            bench.iter(|| {
                trueno::blis::elementwise::fused_add_relu(
                    black_box(&a),
                    black_box(&b),
                    black_box(&mut o),
                )
                .unwrap()
            });
        });
        // ndarray: unfused add then relu (2 passes)
        g.bench_with_input(BenchmarkId::new("ndarray_unfused", n), &n, |bench, &_n| {
            let a = ndarray::Array1::from_vec(a.clone());
            let b = ndarray::Array1::from_vec(b.clone());
            bench.iter(|| {
                let sum = black_box(&a) + black_box(&b);
                black_box(sum.mapv(|x: f32| x.max(0.0)))
            });
        });
    }
    g.finish();
}

fn bench_gemm_bias_relu(c: &mut Criterion) {
    let mut g = c.benchmark_group("gemm_bias_relu");
    g.sample_size(20);
    for &n in &[64, 128] {
        let a = gen(n * n);
        let b = gen(n * n);
        let bias = gen(n);
        // Trueno: fused GEMM+bias+relu
        g.bench_with_input(BenchmarkId::new("trueno_fused", n), &n, |bench, &n| {
            let mut c = vec![0.0f32; n * n];
            bench.iter(|| {
                trueno::blis::gemm_bias_relu(
                    n,
                    n,
                    n,
                    black_box(&a),
                    black_box(&b),
                    black_box(&bias),
                    black_box(&mut c),
                )
                .unwrap()
            });
        });
        // ndarray: separate GEMM + bias + relu (3 passes)
        g.bench_with_input(BenchmarkId::new("ndarray_unfused", n), &n, |bench, &n| {
            let a = ndarray::Array2::from_shape_vec((n, n), a.clone()).unwrap();
            let b = ndarray::Array2::from_shape_vec((n, n), b.clone()).unwrap();
            let bias = ndarray::Array1::from_vec(bias.clone());
            bench.iter(|| {
                let mut c = black_box(&a).dot(black_box(&b));
                c += &bias;
                black_box(c.mapv(|x: f32| x.max(0.0)))
            });
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_transpose,
    bench_gemm,
    bench_gemv,
    bench_add,
    bench_relu,
    bench_softmax,
    bench_fused_add_relu,
    bench_gemm_bias_relu
);
criterion_main!(benches);
