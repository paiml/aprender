//! Benchmarks for hyperparameter sweep execution.
//!
//! #2519: these used to time `Sweeper::run()` under
//! `.expect("sweep must succeed")`. That measured the cost of evaluating a
//! closed-form parabola, not of running a sweep -- and it would have hidden the
//! defect twice over, since a benchmark of fabricated work looks exactly like a
//! benchmark of fast work. `run()` now refuses (see `sweep.rs`), so what is
//! left to time is the honest arithmetic around it: enumerating the points a
//! sweep would visit, and formatting a result someone else measured.

use criterion::{criterion_group, criterion_main, Criterion};
use entrenar_bench::sweep::{DataPoint, SweepConfig, SweepResult};
use std::hint::black_box;

fn bench_sweep_point_enumeration(c: &mut Criterion) {
    c.bench_function("temperature_values_15_points", |b| {
        let config = SweepConfig::temperature(1.0..8.0, 0.5);
        b.iter(|| black_box(config.parameter.values()));
    });
}

fn bench_alpha_point_enumeration(c: &mut Criterion) {
    c.bench_function("alpha_values_9_points", |b| {
        let config = SweepConfig::alpha(0.1..0.9, 0.1);
        b.iter(|| black_box(config.parameter.values()));
    });
}

fn bench_result_table_formatting(c: &mut Criterion) {
    // Values supplied here rather than invented by the crate under test.
    let data_points: Vec<DataPoint> = (0..15)
        .map(|i| DataPoint {
            parameter_value: 1.0 + f64::from(i) * 0.5,
            mean_loss: 0.9 - f64::from(i) * 0.01,
            std_loss: 0.003,
            mean_accuracy: 0.75 + f64::from(i) * 0.004,
            std_accuracy: 0.002,
            runs: 3,
        })
        .collect();
    let optimal = data_points.last().cloned();
    let result = SweepResult {
        parameter_name: "temperature".to_string(),
        data_points,
        optimal,
        config: SweepConfig::temperature(1.0..8.0, 0.5),
    };

    c.bench_function("sweep_result_to_table_15_rows", |b| {
        b.iter(|| black_box(result.to_table()));
    });
}

criterion_group!(
    benches,
    bench_sweep_point_enumeration,
    bench_alpha_point_enumeration,
    bench_result_table_formatting
);
criterion_main!(benches);
