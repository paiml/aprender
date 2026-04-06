#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
//! Benchmark for scatter plot rendering.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno_viz::prelude::*;

fn scatter_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("scatter_plot");

    for size in [100, 1_000, 10_000, 100_000] {
        let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..size).map(|i| (i as f32).sin()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let plot = ScatterPlot::new()
                    .x(black_box(&x_data))
                    .y(black_box(&y_data))
                    .dimensions(800, 600)
                    .build()
                    .expect("operation should succeed");

                plot.to_framebuffer().expect("framebuffer conversion should succeed")
            });
        });
    }

    group.finish();
}

criterion_group!(benches, scatter_benchmark);
criterion_main!(benches);
