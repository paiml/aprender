//! Criterion benchmarks for TSP solvers.
//!
//! These benchmarks measure performance across different problem sizes
//! and algorithms for scientific reproducibility.

use aprender_tsp::{AcoSolver, Budget, GaSolver, HybridSolver, TabuSolver, TspInstance, TspSolver};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

/// Create a random instance with n cities
fn random_instance(n: usize, seed: u64) -> TspInstance {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut coords = Vec::with_capacity(n);
    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i, "x").hash(&mut hasher);
        let x = (hasher.finish() % 10000) as f64 / 100.0;

        let mut hasher = DefaultHasher::new();
        (seed, i, "y").hash(&mut hasher);
        let y = (hasher.finish() % 10000) as f64 / 100.0;

        coords.push((x, y));
    }
    TspInstance::from_coords(&format!("random_{n}"), coords).expect("should create")
}

fn bench_aco(c: &mut Criterion) {
    let mut group = c.benchmark_group("ACO");

    for size in [10, 20, 50].iter() {
        let instance = random_instance(*size, 42);

        group.bench_with_input(BenchmarkId::new("cities", size), size, |b, _| {
            b.iter(|| {
                let mut solver = AcoSolver::new().with_seed(42).with_num_ants(10);
                solver
                    .solve(black_box(&instance), Budget::Iterations(50))
                    .expect("should solve")
            });
        });
    }

    group.finish();
}

fn bench_tabu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Tabu");

    for size in [10, 20, 50].iter() {
        let instance = random_instance(*size, 42);

        group.bench_with_input(BenchmarkId::new("cities", size), size, |b, _| {
            b.iter(|| {
                let mut solver = TabuSolver::new().with_seed(42);
                solver
                    .solve(black_box(&instance), Budget::Iterations(50))
                    .expect("should solve")
            });
        });
    }

    group.finish();
}

fn bench_ga(c: &mut Criterion) {
    let mut group = c.benchmark_group("GA");

    for size in [10, 20, 50].iter() {
        let instance = random_instance(*size, 42);

        group.bench_with_input(BenchmarkId::new("cities", size), size, |b, _| {
            b.iter(|| {
                let mut solver = GaSolver::new().with_seed(42).with_population_size(20);
                solver
                    .solve(black_box(&instance), Budget::Iterations(50))
                    .expect("should solve")
            });
        });
    }

    group.finish();
}

fn bench_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("Hybrid");

    for size in [10, 20].iter() {
        let instance = random_instance(*size, 42);

        group.bench_with_input(BenchmarkId::new("cities", size), size, |b, _| {
            b.iter(|| {
                let mut solver = HybridSolver::new().with_seed(42).with_ga_population(15);
                solver
                    .solve(black_box(&instance), Budget::Iterations(30))
                    .expect("should solve")
            });
        });
    }

    group.finish();
}

fn bench_algorithm_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Algorithm_Comparison");
    let instance = random_instance(20, 42);
    let budget = Budget::Iterations(100);

    group.bench_function("ACO_20cities", |b| {
        b.iter(|| {
            let mut solver = AcoSolver::new().with_seed(42);
            solver
                .solve(black_box(&instance), budget)
                .expect("should solve")
        });
    });

    group.bench_function("Tabu_20cities", |b| {
        b.iter(|| {
            let mut solver = TabuSolver::new().with_seed(42);
            solver
                .solve(black_box(&instance), budget)
                .expect("should solve")
        });
    });

    group.bench_function("GA_20cities", |b| {
        b.iter(|| {
            let mut solver = GaSolver::new().with_seed(42).with_population_size(20);
            solver
                .solve(black_box(&instance), budget)
                .expect("should solve")
        });
    });

    group.bench_function("Hybrid_20cities", |b| {
        b.iter(|| {
            let mut solver = HybridSolver::new().with_seed(42).with_ga_population(15);
            solver
                .solve(black_box(&instance), budget)
                .expect("should solve")
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_aco,
    bench_tabu,
    bench_ga,
    bench_hybrid,
    bench_algorithm_comparison
);
criterion_main!(benches);
