#![allow(missing_docs, clippy::expect_used, clippy::disallowed_methods)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use trueno_image::{canny, conv2d, gaussian_blur, resize, BorderMode, Interpolation};

fn make_image(w: usize, h: usize) -> Vec<f32> {
    (0..w * h).map(|i| ((i * 13 + 7) % 256) as f32 / 255.0).collect()
}

fn bench_conv2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv2d");
    let kernel = vec![1.0 / 9.0; 9]; // 3x3 box blur
    for &size in &[64, 128, 256] {
        let img = make_image(size, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    conv2d(black_box(&img), size, size, &kernel, 3, 3, BorderMode::Zero)
                        .expect("conv2d ok"),
                );
            });
        });
    }
    group.finish();
}

fn bench_gaussian_blur(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blur");
    for &size in &[64, 128, 256] {
        let img = make_image(size, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(gaussian_blur(black_box(&img), size, size, 1.0).expect("blur ok"));
            });
        });
    }
    group.finish();
}

fn bench_canny(c: &mut Criterion) {
    let mut group = c.benchmark_group("canny");
    for &size in &[64, 128, 256] {
        let img = make_image(size, size);
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(canny(black_box(&img), size, size, 1.0, 0.1, 0.3).expect("canny ok"));
            });
        });
    }
    group.finish();
}

fn bench_resize(c: &mut Criterion) {
    let mut group = c.benchmark_group("resize_bilinear");
    for &size in &[64, 128, 256] {
        let img = make_image(size, size);
        let new_size = size * 2;
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                black_box(
                    resize(
                        black_box(&img),
                        size,
                        size,
                        new_size,
                        new_size,
                        Interpolation::Bilinear,
                    )
                    .expect("resize ok"),
                );
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_conv2d, bench_gaussian_blur, bench_canny, bench_resize);
criterion_main!(benches);
