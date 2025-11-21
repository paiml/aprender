//! Benchmarks for DataFrame operations.

use aprender::prelude::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_dataframe_to_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_to_matrix");

    for &n_rows in &[100, 500, 1000] {
        let columns = vec![
            (
                "a".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32).collect()),
            ),
            (
                "b".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32 * 2.0).collect()),
            ),
            (
                "c".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32 * 3.0).collect()),
            ),
        ];
        let df = DataFrame::new(columns).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n_rows), &n_rows, |b, _| {
            b.iter(|| black_box(&df).to_matrix());
        });
    }

    group.finish();
}

fn bench_dataframe_select(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataframe_select");

    for &n_rows in &[100, 500, 1000] {
        let columns = vec![
            (
                "a".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32).collect()),
            ),
            (
                "b".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32 * 2.0).collect()),
            ),
            (
                "c".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32 * 3.0).collect()),
            ),
            (
                "d".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32 * 4.0).collect()),
            ),
            (
                "e".to_string(),
                Vector::from_vec((0..n_rows).map(|i| i as f32 * 5.0).collect()),
            ),
        ];
        let df = DataFrame::new(columns).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(n_rows), &n_rows, |b, _| {
            b.iter(|| black_box(&df).select(&["a", "c", "e"]).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_dataframe_to_matrix, bench_dataframe_select);
criterion_main!(benches);
