#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
//! Benchmark for output encoders (PNG, SVG).

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno_viz::color::Rgba;
use trueno_viz::framebuffer::Framebuffer;
use trueno_viz::output::{PngEncoder, SvgEncoder};

fn png_encoder_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("png_encoder");

    for (width, height) in [(800, 600), (1920, 1080), (3840, 2160)] {
        let mut fb = Framebuffer::new(width, height).expect("framebuffer creation should succeed");
        // Create a gradient pattern for realistic encoding
        for y in 0..height {
            for x in 0..width {
                let r = ((x as f32 / width as f32) * 255.0) as u8;
                let g = ((y as f32 / height as f32) * 255.0) as u8;
                let b = 128;
                fb.set_pixel(x, y, Rgba::new(r, g, b, 255));
            }
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{width}x{height}")),
            &(width, height),
            |b, _| {
                b.iter(|| PngEncoder::to_bytes(black_box(&fb)).expect("encoding should succeed"));
            },
        );
    }

    group.finish();
}

fn svg_encoder_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("svg_encoder");

    for (width, height) in [(800, 600), (1920, 1080)] {
        let mut fb = Framebuffer::new(width, height).expect("framebuffer creation should succeed");
        fb.clear(Rgba::WHITE);

        // Draw some shapes for realistic SVG content
        for i in 0..100 {
            let x = (i * 7) % width;
            let y = (i * 11) % height;
            fb.set_pixel(x, y, Rgba::BLUE);
        }

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{width}x{height}")),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    let encoder = SvgEncoder::from_framebuffer(black_box(&fb))
                        .expect("operation should succeed");
                    encoder.render()
                });
            },
        );
    }

    group.finish();
}

fn png_compression_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("png_content_types");

    let width = 800;
    let height = 600;

    // Solid color (highly compressible)
    let mut fb_solid =
        Framebuffer::new(width, height).expect("framebuffer creation should succeed");
    fb_solid.clear(Rgba::RED);

    // Gradient (moderately compressible)
    let mut fb_gradient =
        Framebuffer::new(width, height).expect("framebuffer creation should succeed");
    for y in 0..height {
        for x in 0..width {
            let r = ((x as f32 / width as f32) * 255.0) as u8;
            let g = ((y as f32 / height as f32) * 255.0) as u8;
            fb_gradient.set_pixel(x, y, Rgba::new(r, g, 0, 255));
        }
    }

    // Noise-like (poorly compressible)
    let mut fb_noise =
        Framebuffer::new(width, height).expect("framebuffer creation should succeed");
    for y in 0..height {
        for x in 0..width {
            let v = ((x * 17 + y * 31) % 256) as u8;
            fb_noise.set_pixel(x, y, Rgba::new(v, v, v, 255));
        }
    }

    group.bench_function("solid", |b| {
        b.iter(|| PngEncoder::to_bytes(black_box(&fb_solid)).expect("encoding should succeed"));
    });

    group.bench_function("gradient", |b| {
        b.iter(|| PngEncoder::to_bytes(black_box(&fb_gradient)).expect("encoding should succeed"));
    });

    group.bench_function("noise", |b| {
        b.iter(|| PngEncoder::to_bytes(black_box(&fb_noise)).expect("encoding should succeed"));
    });

    group.finish();
}

criterion_group!(benches, png_encoder_benchmark, svg_encoder_benchmark, png_compression_benchmark);
criterion_main!(benches);
