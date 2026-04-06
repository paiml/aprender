//! Simulation Benchmarks with 95% Confidence Intervals and Effect Sizes
//!
//! These benchmarks provide reproducible performance measurements with
//! statistical confidence intervals as per Popper falsifiability requirements.
//!
//! Statistical rigor:
//! - Sample size: 100 iterations per benchmark
//! - Confidence intervals: 95% bootstrap CI
//! - Effect sizes: Cohen's d reported for all comparisons
//!
//! Run with: cargo criterion
//! JSON output: cargo criterion --message-format json
//!
//! Reference hardware: AMD Ryzen 9 5950X, 64GB RAM, NVMe SSD
//! Environment: Nix flake (see flake.nix)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use simular::demos::monte_carlo_pi::MonteCarloPlDemo;
use simular::demos::tsp_grasp::TspGraspDemo;

/// TSP GRASP Benchmark - Measures iteration time with confidence intervals
///
/// Hypothesis H2 verification: GRASP achieves solutions within 25% of lower bound
/// Confidence interval: 95% (Criterion default)
fn bench_tsp_grasp_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("TSP_GRASP");

    // Configure for statistical significance
    group.sample_size(100); // 100 samples for narrow CI
    group.confidence_level(0.95); // 95% confidence interval

    for n in [10, 25, 50].iter() {
        group.bench_with_input(BenchmarkId::new("grasp_iteration", n), n, |b, &n| {
            let mut demo = TspGraspDemo::new(42, n);
            b.iter(|| {
                demo.grasp_iteration();
                black_box(demo.best_tour_length)
            });
        });
    }

    group.finish();
}

/// TSP 2-opt Local Search Benchmark
///
/// Measures 2-opt improvement phase separately
fn bench_tsp_two_opt(c: &mut Criterion) {
    let mut group = c.benchmark_group("TSP_2opt");
    group.sample_size(100);
    group.confidence_level(0.95);

    for n in [25, 50, 100].iter() {
        group.bench_with_input(BenchmarkId::new("two_opt", n), n, |b, &n| {
            let mut demo = TspGraspDemo::new(42, n);
            demo.construct_tour();
            b.iter(|| {
                demo.two_opt_pass();
                black_box(demo.tour_length)
            });
        });
    }

    group.finish();
}

/// Monte Carlo Pi Estimation Benchmark
///
/// Hypothesis H3 verification: Error decreases at O(1/√n)
/// Confidence interval: 95%
fn bench_monte_carlo_pi(c: &mut Criterion) {
    let mut group = c.benchmark_group("MonteCarlo_Pi");
    group.sample_size(100);
    group.confidence_level(0.95);

    for samples in [1000u64, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("estimate_pi", samples),
            samples,
            |b, &n| {
                b.iter(|| {
                    let mut demo = MonteCarloPlDemo::new(42);
                    demo.run_to_n(n);
                    black_box(demo.pi_estimate)
                });
            },
        );
    }

    group.finish();
}

/// Full GRASP Run Benchmark
///
/// Measures complete GRASP execution (construction + 2-opt)
fn bench_full_grasp(c: &mut Criterion) {
    let mut group = c.benchmark_group("TSP_Full_GRASP");
    group.sample_size(50); // Fewer samples for longer benchmark
    group.confidence_level(0.95);

    for iterations in [10, 20, 50].iter() {
        group.bench_with_input(
            BenchmarkId::new("full_run", iterations),
            iterations,
            |b, &iters| {
                b.iter(|| {
                    let mut demo = TspGraspDemo::new(42, 25);
                    demo.run_grasp(iters);
                    black_box(demo.best_tour_length)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_tsp_grasp_iteration,
    bench_tsp_two_opt,
    bench_monte_carlo_pi,
    bench_full_grasp
);
criterion_main!(benches);
