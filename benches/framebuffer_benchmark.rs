#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
//! Benchmark for framebuffer operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno_viz::color::Rgba;
use trueno_viz::framebuffer::Framebuffer;

fn framebuffer_clear_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("framebuffer_clear");

    for (width, height) in [(800, 600), (1920, 1080), (3840, 2160)] {
        let mut fb = Framebuffer::new(width, height).expect("framebuffer creation should succeed");

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{width}x{height}")),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    fb.clear(black_box(Rgba::RED));
                });
            },
        );
    }

    group.finish();
}

fn framebuffer_blend_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("framebuffer_blend");

    let mut fb = Framebuffer::new(800, 600).expect("framebuffer creation should succeed");
    fb.clear(Rgba::WHITE);

    let semi_transparent = Rgba::new(255, 0, 0, 128);

    group.bench_function("blend_pixel_800x600", |b| {
        b.iter(|| {
            for y in 0..600 {
                for x in 0..800 {
                    fb.blend_pixel(black_box(x), black_box(y), semi_transparent);
                }
            }
        });
    });

    group.finish();
}

criterion_group!(benches, framebuffer_clear_benchmark, framebuffer_blend_benchmark);
criterion_main!(benches);
