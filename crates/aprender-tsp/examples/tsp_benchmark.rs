//! Scientific benchmark example for TSP solvers.
//!
//! This example demonstrates reproducible benchmarking across all
//! metaheuristic algorithms with deterministic seeding.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example tsp_benchmark --release
//! ```
//!
//! # Output Format (IEEE/ACM compatible)
//!
//! Results are formatted for direct inclusion in academic papers.

use aprender_tsp::{AcoSolver, Budget, GaSolver, HybridSolver, TabuSolver, TspInstance, TspSolver};
use std::time::Instant;

/// Benchmark configuration for reproducibility
struct BenchmarkConfig {
    seed: u64,
    iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            iterations: 1000,
        }
    }
}

fn main() {
    let config = BenchmarkConfig::default();

    println!("=== aprender-tsp Scientific Benchmark ===");
    println!();
    println!("Configuration:");
    println!("  Seed:       {}", config.seed);
    println!("  Iterations: {}", config.iterations);
    println!();

    // Test instances
    let instances = [
        ("square", create_square_instance()),
        ("pentagon", create_pentagon_instance()),
        ("random20", create_random_instance(20, 123)),
    ];

    // Header
    println!(
        "{:<12} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Instance", "n", "ACO", "Tabu", "GA", "Hybrid", "Time(ms)"
    );
    println!("{}", "-".repeat(76));

    for (name, instance) in &instances {
        let start = Instant::now();

        // Run all solvers with same seed
        let aco_result = {
            let mut solver = AcoSolver::new().with_seed(config.seed);
            solver
                .solve(instance, Budget::Iterations(config.iterations))
                .expect("ACO should solve")
        };

        let tabu_result = {
            let mut solver = TabuSolver::new().with_seed(config.seed);
            solver
                .solve(instance, Budget::Iterations(config.iterations))
                .expect("Tabu should solve")
        };

        let ga_result = {
            let mut solver = GaSolver::new()
                .with_seed(config.seed)
                .with_population_size(30);
            solver
                .solve(instance, Budget::Iterations(config.iterations))
                .expect("GA should solve")
        };

        let hybrid_result = {
            let mut solver = HybridSolver::new()
                .with_seed(config.seed)
                .with_ga_population(20);
            solver
                .solve(instance, Budget::Iterations(config.iterations))
                .expect("Hybrid should solve")
        };

        let elapsed = start.elapsed().as_millis();

        println!(
            "{:<12} {:>6} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10}",
            name,
            instance.dimension,
            aco_result.length,
            tabu_result.length,
            ga_result.length,
            hybrid_result.length,
            elapsed
        );
    }

    println!();
    println!(
        "Note: All results are deterministic with seed={}.",
        config.seed
    );
    println!("Reproduce with: cargo run --example tsp_benchmark --release");
}

fn create_square_instance() -> TspInstance {
    let coords = vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
    TspInstance::from_coords("square", coords).expect("should create")
}

fn create_pentagon_instance() -> TspInstance {
    let angle_step = 2.0 * std::f64::consts::PI / 5.0;
    let coords: Vec<(f64, f64)> = (0..5)
        .map(|i| {
            let angle = i as f64 * angle_step;
            (angle.cos(), angle.sin())
        })
        .collect();
    TspInstance::from_coords("pentagon", coords).expect("should create")
}

fn create_random_instance(n: usize, seed: u64) -> TspInstance {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut coords = Vec::with_capacity(n);
    for i in 0..n {
        let mut hasher = DefaultHasher::new();
        (seed, i, "x").hash(&mut hasher);
        let x = (hasher.finish() % 1000) as f64 / 1000.0;

        let mut hasher = DefaultHasher::new();
        (seed, i, "y").hash(&mut hasher);
        let y = (hasher.finish() % 1000) as f64 / 1000.0;

        coords.push((x, y));
    }
    TspInstance::from_coords("random", coords).expect("should create")
}
