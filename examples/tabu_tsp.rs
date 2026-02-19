#![allow(clippy::disallowed_methods)]
//! Tabu Search for the Traveling Salesman Problem
//!
//! This example demonstrates Tabu Search solving a TSP instance.
//! The algorithm explores the neighborhood via 2-opt swaps while
//! maintaining a tabu list to avoid cycling back to recent solutions.
//!
//! Run with: `cargo run --example tabu_tsp`

use aprender::metaheuristics::{
    Budget, ConstructiveMetaheuristic, OptimizationResult, SearchSpace, TabuSearch,
};

const CITY_NAMES: [&str; 8] = [
    "Paris",
    "Berlin",
    "Rome",
    "Madrid",
    "Vienna",
    "Amsterdam",
    "Prague",
    "Brussels",
];

fn main() {
    println!("=== Tabu Search: Traveling Salesman Problem ===\n");

    let distances = create_distance_matrix();
    let n = distances.len();
    let space = SearchSpace::Permutation { size: n };

    let objective = |tour: &Vec<usize>| -> f64 { compute_tour_length(tour, &distances) };

    print_problem_info(n);
    let result = run_tabu_search(&objective, &space);
    display_results(&result, &distances);
    run_sensitivity_analysis(&objective, &space);
}

fn create_distance_matrix() -> Vec<Vec<f64>> {
    vec![
        vec![0.0, 878.0, 1106.0, 1054.0, 1034.0, 430.0, 885.0, 265.0],
        vec![878.0, 0.0, 1181.0, 1870.0, 524.0, 577.0, 280.0, 651.0],
        vec![1106.0, 1181.0, 0.0, 1365.0, 765.0, 1293.0, 922.0, 1175.0],
        vec![1054.0, 1870.0, 1365.0, 0.0, 1809.0, 1482.0, 1773.0, 1316.0],
        vec![1034.0, 524.0, 765.0, 1809.0, 0.0, 938.0, 252.0, 915.0],
        vec![430.0, 577.0, 1293.0, 1482.0, 938.0, 0.0, 710.0, 173.0],
        vec![885.0, 280.0, 922.0, 1773.0, 252.0, 710.0, 0.0, 718.0],
        vec![265.0, 651.0, 1175.0, 1316.0, 915.0, 173.0, 718.0, 0.0],
    ]
}

fn compute_tour_length(tour: &[usize], distances: &[Vec<f64>]) -> f64 {
    tour.iter()
        .enumerate()
        .map(|(i, &from)| distances[from][tour[(i + 1) % tour.len()]])
        .sum()
}

fn print_problem_info(n: usize) {
    println!("Problem: Visit {} European capitals and return home", n);
    println!("Optimization: Minimize total distance\n");
    println!("Tabu Search Parameters:");
    println!("  Tabu tenure: 7");
    println!("  Max neighbors: 500");
    println!("  Iterations: 200\n");
}

fn run_tabu_search<F>(objective: &F, space: &SearchSpace) -> OptimizationResult<Vec<usize>>
where
    F: Fn(&Vec<usize>) -> f64,
{
    let mut ts = TabuSearch::new(7).with_max_neighbors(500).with_seed(42);
    ts.optimize(objective, space, Budget::Iterations(200))
}

fn display_results(result: &OptimizationResult<Vec<usize>>, distances: &[Vec<f64>]) {
    print_tour(&result.solution);
    println!("\nTotal distance: {:.0} km", result.objective_value);
    println!("Iterations: {}", result.iterations);
    println!("Termination: {:?}", result.termination);
    print_convergence(&result.history);
    validate_tour(&result.solution);
    print_leg_breakdown(&result.solution, distances);
}

fn print_tour(solution: &[usize]) {
    println!("=== Results ===\n");
    println!("Best tour found:");
    print!("  ");
    for (i, &city) in solution.iter().enumerate() {
        if i > 0 {
            print!(" -> ");
        }
        print!("{}", CITY_NAMES[city]);
    }
    println!(" -> {}", CITY_NAMES[solution[0]]);
}

fn print_convergence(history: &[f64]) {
    println!("\nConvergence (every 20 iterations):");
    for (i, &val) in history.iter().enumerate() {
        if i % 20 == 0 || i == history.len() - 1 {
            println!("  Iter {:3}: {:.0} km", i, val);
        }
    }
}

fn validate_tour(solution: &[usize]) {
    println!("\n=== Tour Validation ===");
    let n = solution.len();
    let mut visited = vec![false; n];
    for &city in solution {
        if visited[city] {
            println!("ERROR: City {} visited twice!", CITY_NAMES[city]);
        }
        visited[city] = true;
    }
    if visited.iter().all(|&v| v) {
        println!("✓ All {} cities visited exactly once", n);
    }
}

fn print_leg_breakdown(solution: &[usize], distances: &[Vec<f64>]) {
    println!("\n=== Leg-by-Leg Breakdown ===");
    let mut cumulative = 0.0;
    for i in 0..solution.len() {
        let from = solution[i];
        let to = solution[(i + 1) % solution.len()];
        let dist = distances[from][to];
        cumulative += dist;
        println!(
            "  {} -> {}: {:.0} km (cumulative: {:.0} km)",
            CITY_NAMES[from], CITY_NAMES[to], dist, cumulative
        );
    }
}

fn run_sensitivity_analysis<F>(objective: &F, space: &SearchSpace)
where
    F: Fn(&Vec<usize>) -> f64,
{
    println!("\n=== Sensitivity Analysis: Tabu Tenure ===");
    for tenure in [3, 5, 10, 15] {
        let mut ts_test = TabuSearch::new(tenure).with_seed(42);
        let res = ts_test.optimize(objective, space, Budget::Iterations(100));
        println!("  Tenure {:2}: {:.0} km", tenure, res.objective_value);
    }
}
