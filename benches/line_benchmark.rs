#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
//! Benchmark for line chart rendering.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno_viz::prelude::*;

fn line_chart_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_chart");

    for size in [100, 1_000, 10_000, 100_000] {
        let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = (0..size).map(|i| (i as f32 * 0.01).sin() * 100.0).collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let series = LineSeries::new("data").data(black_box(&x_data), black_box(&y_data));

                let chart = LineChart::new()
                    .add_series(series)
                    .dimensions(800, 600)
                    .build()
                    .expect("builder should produce valid result");

                chart.to_framebuffer().expect("framebuffer conversion should succeed")
            });
        });
    }

    group.finish();
}

fn line_chart_simplification_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("line_simplification");

    // High-frequency signal that benefits from simplification
    let size = 100_000;
    let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let y_data: Vec<f32> = (0..size)
        .map(|i| {
            let t = i as f32 * 0.001;
            t.sin() * 50.0 + (t * 10.0).sin() * 10.0 + (t * 100.0).sin() * 2.0
        })
        .collect();

    for epsilon in [0.1, 1.0, 5.0, 10.0] {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("eps_{epsilon}")),
            &epsilon,
            |b, &epsilon| {
                b.iter(|| {
                    let series =
                        LineSeries::new("data").data(black_box(&x_data), black_box(&y_data));

                    let chart = LineChart::new()
                        .add_series(series)
                        .simplify(epsilon)
                        .dimensions(800, 600)
                        .build()
                        .expect("operation should succeed");

                    chart.to_framebuffer().expect("framebuffer conversion should succeed")
                });
            },
        );
    }

    group.finish();
}

fn multi_series_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_series");

    let size = 1_000;
    let x_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

    for num_series in [1, 5, 10, 20] {
        group.bench_with_input(
            BenchmarkId::from_parameter(num_series),
            &num_series,
            |b, &num_series| {
                b.iter(|| {
                    let mut chart = LineChart::new().dimensions(800, 600);

                    for s in 0..num_series {
                        let y_data: Vec<f32> =
                            (0..size).map(|i| (i as f32 * 0.01 + s as f32).sin() * 100.0).collect();
                        let series = LineSeries::new(format!("series_{s}"))
                            .data(black_box(&x_data), black_box(&y_data));
                        chart = chart.add_series(series);
                    }

                    let chart = chart.build().expect("builder should produce valid result");
                    chart.to_framebuffer().expect("framebuffer conversion should succeed")
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    line_chart_benchmark,
    line_chart_simplification_benchmark,
    multi_series_benchmark
);
criterion_main!(benches);
