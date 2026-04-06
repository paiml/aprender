//! Benchmarks for hyperparameter sweep execution.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use entrenar_bench::{SweepConfig, Sweeper};

fn bench_temperature_sweep(c: &mut Criterion) {
    c.bench_function("temperature_sweep_5_points", |b| {
        b.iter(|| {
            let config = SweepConfig::temperature(1.0..5.0, 1.0).with_runs(1);
            let sweeper = Sweeper::new(config);
            black_box(sweeper.run().expect("sweep must succeed"))
        });
    });
}

fn bench_alpha_sweep(c: &mut Criterion) {
    c.bench_function("alpha_sweep_9_points", |b| {
        b.iter(|| {
            let config = SweepConfig::alpha(0.1..0.9, 0.1).with_runs(1);
            let sweeper = Sweeper::new(config);
            black_box(sweeper.run().expect("sweep must succeed"))
        });
    });
}

fn bench_sweep_with_multiple_runs(c: &mut Criterion) {
    c.bench_function("temperature_sweep_3_runs", |b| {
        b.iter(|| {
            let config = SweepConfig::temperature(1.0..5.0, 1.0).with_runs(3);
            let sweeper = Sweeper::new(config);
            black_box(sweeper.run().expect("sweep must succeed"))
        });
    });
}

criterion_group!(
    benches,
    bench_temperature_sweep,
    bench_alpha_sweep,
    bench_sweep_with_multiple_runs
);
criterion_main!(benches);
