#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
//! Benchmark for heatmap rendering.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno_viz::prelude::*;

fn heatmap_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("heatmap");

    for size in [10, 50, 100, 200] {
        // Generate 2D grid data
        let data: Vec<Vec<f32>> = (0..size)
            .map(|i| {
                (0..size)
                    .map(|j| {
                        let x = i as f32 / size as f32;
                        let y = j as f32 / size as f32;
                        (x * std::f32::consts::PI).sin() * (y * std::f32::consts::PI).cos() * 100.0
                    })
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{size}x{size}")),
            &size,
            |b, _| {
                b.iter(|| {
                    let heatmap = Heatmap::new()
                        .data_2d(black_box(&data))
                        .dimensions(800, 600)
                        .build()
                        .expect("operation should succeed");

                    heatmap.to_framebuffer().expect("framebuffer conversion should succeed")
                });
            },
        );
    }

    group.finish();
}

fn heatmap_palette_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("heatmap_palette");

    let size = 100;
    let data: Vec<Vec<f32>> = (0..size)
        .map(|i| (0..size).map(|j| ((i + j) as f32) / (2.0 * size as f32) * 100.0).collect())
        .collect();

    for palette in [
        HeatmapPalette::Viridis,
        HeatmapPalette::Blues,
        HeatmapPalette::RedBlue,
        HeatmapPalette::Magma,
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{palette:?}")),
            &palette,
            |b, &palette| {
                b.iter(|| {
                    let heatmap = Heatmap::new()
                        .data_2d(black_box(&data))
                        .palette(palette)
                        .dimensions(800, 600)
                        .build()
                        .expect("operation should succeed");

                    heatmap.to_framebuffer().expect("framebuffer conversion should succeed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, heatmap_benchmark, heatmap_palette_benchmark);
criterion_main!(benches);
