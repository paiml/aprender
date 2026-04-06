#![allow(missing_docs)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_rand::{Philox4x32, Threefry4x64};

fn bench_philox_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("philox_fill_uniform");
    for &n in &[1024, 16384, 131072] {
        let mut rng = Philox4x32::new(42);
        let mut buf = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                rng.fill_uniform(black_box(&mut buf));
                black_box(&buf);
            });
        });
    }
    group.finish();
}

fn bench_philox_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("philox_fill_normal");
    for &n in &[1024, 16384, 131072] {
        let mut rng = Philox4x32::new(42);
        let mut buf = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                rng.fill_normal(black_box(&mut buf));
                black_box(&buf);
            });
        });
    }
    group.finish();
}

fn bench_threefry_uniform(c: &mut Criterion) {
    let mut group = c.benchmark_group("threefry_fill_uniform");
    for &n in &[1024, 16384, 131072] {
        let mut rng = Threefry4x64::new(42);
        let mut buf = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                rng.fill_uniform(black_box(&mut buf));
                black_box(&buf);
            });
        });
    }
    group.finish();
}

fn bench_threefry_normal(c: &mut Criterion) {
    let mut group = c.benchmark_group("threefry_fill_normal");
    for &n in &[1024, 16384, 131072] {
        let mut rng = Threefry4x64::new(42);
        let mut buf = vec![0.0f32; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                rng.fill_normal(black_box(&mut buf));
                black_box(&buf);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_philox_uniform,
    bench_philox_normal,
    bench_threefry_uniform,
    bench_threefry_normal
);
criterion_main!(benches);
