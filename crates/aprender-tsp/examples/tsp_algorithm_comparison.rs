//! Algorithm comparison example for academic research.
//!
//! This example provides a comprehensive comparison of all TSP algorithms
//! with statistical analysis suitable for academic papers.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example tsp_algorithm_comparison --release
//! ```
//!
//! # Output
//!
//! - Mean tour length across multiple seeds
//! - Standard deviation
//! - Best/worst tour lengths
//! - Convergence characteristics

use aprender_tsp::{AcoSolver, Budget, GaSolver, HybridSolver, TabuSolver, TspInstance, TspSolver};

const NUM_RUNS: usize = 10;
const ITERATIONS: usize = 500;

fn main() {
    println!("=== TSP Algorithm Statistical Comparison ===");
    println!();
    println!("Configuration:");
    println!("  Runs per algorithm: {}", NUM_RUNS);
    println!("  Iterations per run: {}", ITERATIONS);
    println!();

    // Create test instance (10 cities for fast benchmarking)
    let instance = create_test_instance();
    println!("Instance: {} cities", instance.dimension);
    println!();

    // Run experiments
    let algorithms: Vec<(&str, Box<dyn Fn(u64) -> Box<dyn TspSolver>>)> = vec![
        (
            "ACO",
            Box::new(|seed| -> Box<dyn TspSolver> { Box::new(AcoSolver::new().with_seed(seed)) }),
        ),
        (
            "Tabu",
            Box::new(|seed| -> Box<dyn TspSolver> { Box::new(TabuSolver::new().with_seed(seed)) }),
        ),
        (
            "GA",
            Box::new(|seed| -> Box<dyn TspSolver> {
                Box::new(GaSolver::new().with_seed(seed).with_population_size(30))
            }),
        ),
        (
            "Hybrid",
            Box::new(|seed| -> Box<dyn TspSolver> {
                Box::new(HybridSolver::new().with_seed(seed).with_ga_population(20))
            }),
        ),
    ];

    // Header
    println!(
        "{:<10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Algorithm", "Mean", "Std", "Best", "Worst", "Evals"
    );
    println!("{}", "-".repeat(62));

    for (name, solver_fn) in &algorithms {
        let mut lengths: Vec<f64> = Vec::with_capacity(NUM_RUNS);
        let mut total_evals = 0usize;

        for run in 0..NUM_RUNS {
            let seed = 42 + run as u64;
            let mut solver = solver_fn(seed);
            let result = solver
                .solve(&instance, Budget::Iterations(ITERATIONS))
                .expect("should solve");
            lengths.push(result.length);
            total_evals += result.evaluations;
        }

        // Statistics
        let mean = lengths.iter().sum::<f64>() / lengths.len() as f64;
        let variance =
            lengths.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / lengths.len() as f64;
        let std = variance.sqrt();
        let best = lengths.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst = lengths.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_evals = total_evals / NUM_RUNS;

        println!(
            "{:<10} {:>10.2} {:>10.2} {:>10.2} {:>10.2} {:>10}",
            name, mean, std, best, worst, avg_evals
        );
    }

    println!();
    println!("Note: Lower tour length is better.");
    println!("Std = standard deviation across {} runs.", NUM_RUNS);
}

fn create_test_instance() -> TspInstance {
    // 10-city instance for fast benchmarking
    let coords = vec![
        (0.0, 0.0),
        (1.0, 0.0),
        (2.0, 0.0),
        (2.0, 1.0),
        (2.0, 2.0),
        (1.0, 2.0),
        (0.0, 2.0),
        (0.0, 1.0),
        (0.5, 0.5),
        (1.5, 1.5),
    ];
    TspInstance::from_coords("grid10", coords).expect("should create")
}
