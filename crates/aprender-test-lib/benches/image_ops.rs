//! Image Operations Benchmarks
//!
//! Benchmarks for PNG heatmap generation and color operations.
//!
//! Run with: `cargo bench --bench image_ops`

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use jugar_probar::pixel_coverage::{ColorPalette, CoverageCell, PngHeatmap};
use jugar_probar::Color;

fn bench_color_contrast(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_contrast");

    let color_pairs = vec![
        (
            Color::new(0, 0, 0),
            Color::new(255, 255, 255),
            "black_white",
        ),
        (
            Color::new(33, 33, 33),
            Color::new(255, 255, 255),
            "dark_gray_white",
        ),
        (
            Color::new(255, 0, 0),
            Color::new(255, 255, 255),
            "red_white",
        ),
        (
            Color::new(0, 128, 0),
            Color::new(255, 255, 255),
            "green_white",
        ),
    ];

    for (fg, bg, name) in color_pairs {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(fg, bg),
            |bench, (f, b): &(Color, Color)| {
                bench.iter(|| {
                    let ratio = black_box(*f).contrast_ratio(black_box(b));
                    black_box(ratio);
                });
            },
        );
    }

    group.finish();
}

fn bench_color_luminance(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_luminance");

    let batch_sizes = vec![10, 100, 1000];

    for size in batch_sizes {
        let colors: Vec<Color> = (0..size)
            .map(|i| {
                Color::new(
                    (i % 256) as u8,
                    ((i * 7) % 256) as u8,
                    ((i * 13) % 256) as u8,
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_colors", size)),
            &colors,
            |bench, cols: &Vec<Color>| {
                bench.iter(|| {
                    let luminances: Vec<f32> = cols
                        .iter()
                        .map(|c| black_box(*c).relative_luminance())
                        .collect();
                    black_box(luminances);
                });
            },
        );
    }

    group.finish();
}

fn bench_color_palette_mapping(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_palette_mapping");

    let palettes = vec![
        (ColorPalette::viridis(), "viridis"),
        (ColorPalette::magma(), "magma"),
        (ColorPalette::heat(), "heat"),
    ];

    let sample_count = 1000;

    for (palette, name) in palettes {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &palette,
            |bench, pal: &ColorPalette| {
                bench.iter(|| {
                    let mut colors = Vec::with_capacity(sample_count);
                    for i in 0..sample_count {
                        let value = (i as f32) / (sample_count as f32);
                        colors.push(pal.interpolate(black_box(value)));
                    }
                    black_box(colors);
                });
            },
        );
    }

    group.finish();
}

fn bench_heatmap_render(c: &mut Criterion) {
    let mut group = c.benchmark_group("heatmap_render");

    let configs = vec![
        (10, 10, 200, 200, "10x10_to_200x200"),
        (20, 20, 400, 400, "20x20_to_400x400"),
        (50, 50, 800, 600, "50x50_to_800x600"),
    ];

    for (grid_w, grid_h, png_w, png_h, name) in configs {
        // Create coverage data as CoverageCell grid
        let coverage: Vec<Vec<CoverageCell>> = (0..grid_h)
            .map(|y| {
                (0..grid_w)
                    .map(|x| {
                        let cov = ((x + y) as f32) / ((grid_w + grid_h) as f32);
                        CoverageCell {
                            coverage: cov,
                            hit_count: (cov * 10.0) as u64,
                        }
                    })
                    .collect()
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &(coverage.clone(), png_w, png_h),
            |bench, (cov, w, h): &(Vec<Vec<CoverageCell>>, u32, u32)| {
                bench.iter(|| {
                    let heatmap = PngHeatmap::new(black_box(*w), black_box(*h))
                        .with_palette(ColorPalette::viridis());
                    let result = heatmap.export(black_box(cov));
                    let _ = black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn bench_wcag_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("wcag_validation");

    let color_counts = vec![10, 50, 100, 500];

    for count in color_counts {
        let colors: Vec<(Color, Color)> = (0..count)
            .map(|i| {
                (
                    Color::new(
                        (i % 256) as u8,
                        ((i * 3) % 256) as u8,
                        ((i * 7) % 256) as u8,
                    ),
                    Color::new(255, 255, 255),
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}_pairs", count)),
            &colors,
            |bench, pairs: &Vec<(Color, Color)>| {
                bench.iter(|| {
                    let results: Vec<bool> = pairs
                        .iter()
                        .map(|(fg, bg)| black_box(*fg).meets_wcag_aa_normal(black_box(bg)))
                        .collect();
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_color_contrast,
    bench_color_luminance,
    bench_color_palette_mapping,
    bench_heatmap_render,
    bench_wcag_validation
);
criterion_main!(benches);
