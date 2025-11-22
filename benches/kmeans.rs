//! Benchmarks for K-Means clustering.

use aprender::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_kmeans_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_fit");

    for &n_samples in &[50, 100, 200] {
        // Create two clusters
        let mut data = Vec::with_capacity(n_samples * 2);
        for i in 0..n_samples {
            if i < n_samples / 2 {
                data.push(i as f32 * 0.1);
                data.push(i as f32 * 0.1);
            } else {
                data.push(10.0 + i as f32 * 0.1);
                data.push(10.0 + i as f32 * 0.1);
            }
        }

        let matrix = Matrix::from_vec(n_samples, 2, data).expect("Benchmark data should be valid");

        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    let mut kmeans = KMeans::new(2).with_random_state(42).with_max_iter(100);
                    kmeans
                        .fit(black_box(&matrix))
                        .expect("Benchmark data should be valid");
                });
            },
        );
    }

    group.finish();
}

fn bench_kmeans_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("kmeans_predict");

    for &n_samples in &[50, 100, 200] {
        let mut data = Vec::with_capacity(n_samples * 2);
        for i in 0..n_samples {
            if i < n_samples / 2 {
                data.push(i as f32 * 0.1);
                data.push(i as f32 * 0.1);
            } else {
                data.push(10.0 + i as f32 * 0.1);
                data.push(10.0 + i as f32 * 0.1);
            }
        }

        let matrix = Matrix::from_vec(n_samples, 2, data).expect("Benchmark data should be valid");
        let mut kmeans = KMeans::new(2).with_random_state(42);
        kmeans.fit(&matrix).expect("Benchmark data should be valid");

        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            &n_samples,
            |b, _| {
                b.iter(|| kmeans.predict(black_box(&matrix)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_kmeans_fit, bench_kmeans_predict);
criterion_main!(benches);
