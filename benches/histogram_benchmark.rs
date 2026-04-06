#![allow(clippy::expect_used, clippy::unwrap_used, missing_docs)]
//! Benchmark for histogram rendering.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use trueno_viz::prelude::*;

fn histogram_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("histogram");

    for size in [100, 1_000, 10_000, 100_000] {
        // Generate random-ish data using deterministic formula
        let data: Vec<f32> = (0..size)
            .map(|i| {
                let x = i as f32 / size as f32;
                // Create bell-curve-like distribution
                (x * std::f32::consts::TAU).sin() * 50.0 + 50.0 + (i % 17) as f32
            })
            .collect();

        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, _| {
            b.iter(|| {
                let hist = Histogram::new()
                    .data(black_box(&data))
                    .dimensions(800, 600)
                    .build()
                    .expect("builder should produce valid result");

                hist.to_framebuffer().expect("framebuffer conversion should succeed")
            });
        });
    }

    group.finish();
}

criterion_group!(benches, histogram_benchmark);
criterion_main!(benches);
