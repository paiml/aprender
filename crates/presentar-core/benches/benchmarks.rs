//! Criterion benchmarks for presentar-core.
//!
//! Benchmarks color parsing, constraint computation, geometry operations,
//! and SIMD batch operations which are hot paths in the rendering pipeline.

#![allow(clippy::unwrap_used, clippy::disallowed_methods, clippy::cast_lossless)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use presentar_core::{Color, Constraints, Point, Rect, Size};

fn bench_color_from_hex(c: &mut Criterion) {
    c.bench_function("color_from_hex_rgb", |b| {
        b.iter(|| Color::from_hex(black_box("#ff8033")).unwrap());
    });
}

fn bench_color_from_hex_rgba(c: &mut Criterion) {
    c.bench_function("color_from_hex_rgba", |b| {
        b.iter(|| Color::from_hex(black_box("#ff803380")).unwrap());
    });
}

fn bench_color_create(c: &mut Criterion) {
    c.bench_function("color_rgb_create", |b| {
        b.iter(|| Color::rgb(black_box(0.5), black_box(0.3), black_box(0.8)));
    });
}

fn bench_constraints_constrain(c: &mut Criterion) {
    let constraints = Constraints::new(0.0, 200.0, 0.0, 100.0);
    c.bench_function("constraints_constrain", |b| {
        b.iter(|| {
            let _ = constraints.constrain(black_box(Size::new(300.0, 150.0)));
        });
    });
}

fn bench_rect_operations(c: &mut Criterion) {
    let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
    c.bench_function("rect_contains_point", |b| {
        b.iter(|| {
            let _ = rect.contains_point(&black_box(Point::new(50.0, 40.0)));
        });
    });
}

fn bench_simd_batch_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_batch_sum");
    for size in [64, 256, 1024, 4096] {
        let data: Vec<f64> = (0..size).map(|i| i as f64 * 0.1).collect();
        group.bench_with_input(BenchmarkId::new("elements", size), &data, |b, data| {
            b.iter(|| presentar_core::batch_sum_f64(black_box(data)));
        });
    }
    group.finish();
}

fn bench_simd_batch_min_max(c: &mut Criterion) {
    let data: Vec<f64> = (0..1024).map(|i| (i as f64 * 0.7).sin()).collect();
    c.bench_function("simd_batch_min_max_1024", |b| {
        b.iter(|| presentar_core::batch_min_max_f64(black_box(&data)));
    });
}

fn bench_simd_normalize(c: &mut Criterion) {
    let data: Vec<f64> = (0..1024).map(|i| i as f64 * 0.1).collect();
    c.bench_function("simd_normalize_1024", |b| {
        b.iter(|| presentar_core::normalize_f64(black_box(&data)));
    });
}

criterion_group!(
    benches,
    bench_color_from_hex,
    bench_color_from_hex_rgba,
    bench_color_create,
    bench_constraints_constrain,
    bench_rect_operations,
    bench_simd_batch_sum,
    bench_simd_batch_min_max,
    bench_simd_normalize,
);
criterion_main!(benches);
