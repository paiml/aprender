#![allow(missing_docs, clippy::expect_used, clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_fft::{Complex, FftPlan};

fn bench_fft_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_forward");
    for &n in &[64, 256, 1024, 4096] {
        let plan = FftPlan::new(n).expect("valid plan");
        let input: Vec<Complex> =
            (0..n).map(|i| Complex { re: (i as f32).sin(), im: 0.0 }).collect();
        let mut output = vec![Complex { re: 0.0, im: 0.0 }; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                plan.forward(black_box(&input), &mut output).expect("fft ok");
                black_box(&output);
            });
        });
    }
    group.finish();
}

fn bench_fft_inverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_inverse");
    for &n in &[64, 256, 1024] {
        let plan = FftPlan::new(n).expect("valid plan");
        let input: Vec<Complex> =
            (0..n).map(|i| Complex { re: (i as f32).cos(), im: (i as f32).sin() }).collect();
        let mut output = vec![Complex { re: 0.0, im: 0.0 }; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                plan.inverse(black_box(&input), &mut output).expect("ifft ok");
                black_box(&output);
            });
        });
    }
    group.finish();
}

fn bench_fft_r2c(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft_r2c");
    for &n in &[256, 1024] {
        let plan = FftPlan::new(n).expect("valid plan");
        let input: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
        let mut output = vec![Complex { re: 0.0, im: 0.0 }; n / 2 + 1];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                plan.forward_r2c(black_box(&input), &mut output).expect("r2c ok");
                black_box(&output);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fft_forward, bench_fft_inverse, bench_fft_r2c);
criterion_main!(benches);
