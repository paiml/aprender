//! Benchmarks for linear regression.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use aprender::prelude::*;

fn bench_linear_regression_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_regression_fit");

    for size in [10, 50, 100, 500].iter() {
        // Create data: y = 2x + 1
        let x_data: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let x = Matrix::from_vec(*size, 1, x_data).unwrap();
        let y = Vector::from_vec(y_data);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mut model = LinearRegression::new();
                model.fit(black_box(&x), black_box(&y)).unwrap()
            });
        });
    }

    group.finish();
}

fn bench_linear_regression_predict(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_regression_predict");

    for size in [10, 50, 100, 500].iter() {
        let x_data: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

        let x = Matrix::from_vec(*size, 1, x_data).unwrap();
        let y = Vector::from_vec(y_data);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                model.predict(black_box(&x))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_linear_regression_fit, bench_linear_regression_predict);
criterion_main!(benches);
