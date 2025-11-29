//! Tabu Search for the Traveling Salesman Problem
//!
//! This example demonstrates Tabu Search solving a TSP instance.
//! The algorithm explores the neighborhood via 2-opt swaps while
//! maintaining a tabu list to avoid cycling back to recent solutions.
//!
//! Run with: `cargo run --example tabu_tsp`

use aprender::metaheuristics::{Budget, ConstructiveMetaheuristic, SearchSpace, TabuSearch};

fn main() {
    println!("=== Tabu Search: Traveling Salesman Problem ===\n");

    // 8-city TSP: European capitals (distances in km)
    let city_names = [
        "Paris",
        "Berlin",
        "Rome",
        "Madrid",
        "Vienna",
        "Amsterdam",
        "Prague",
        "Brussels",
    ];

    // Approximate distances between cities (km)
    let distances: Vec<Vec<f64>> = vec![
        vec![0.0, 878.0, 1106.0, 1054.0, 1034.0, 430.0, 885.0, 265.0], // Paris
        vec![878.0, 0.0, 1181.0, 1870.0, 524.0, 577.0, 280.0, 651.0],  // Berlin
        vec![1106.0, 1181.0, 0.0, 1365.0, 765.0, 1293.0, 922.0, 1175.0], // Rome
        vec![1054.0, 1870.0, 1365.0, 0.0, 1809.0, 1482.0, 1773.0, 1316.0], // Madrid
        vec![1034.0, 524.0, 765.0, 1809.0, 0.0, 938.0, 252.0, 915.0],  // Vienna
        vec![430.0, 577.0, 1293.0, 1482.0, 938.0, 0.0, 710.0, 173.0],  // Amsterdam
        vec![885.0, 280.0, 922.0, 1773.0, 252.0, 710.0, 0.0, 718.0],   // Prague
        vec![265.0, 651.0, 1175.0, 1316.0, 915.0, 173.0, 718.0, 0.0],  // Brussels
    ];

    let n = distances.len();

    let space = SearchSpace::Permutation { size: n };

    // Objective: total tour length (closed loop)
    let objective = |tour: &Vec<usize>| -> f64 {
        let mut total = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total += distances[from][to];
        }
        total
    };

    println!("Problem: Visit {} European capitals and return home", n);
    println!("Optimization: Minimize total distance\n");

    // Configure Tabu Search
    let tenure = 7; // Moves stay tabu for 7 iterations
    let mut ts = TabuSearch::new(tenure)
        .with_max_neighbors(500) // Evaluate up to 500 swaps per iteration
        .with_seed(42);

    println!("Tabu Search Parameters:");
    println!("  Tabu tenure: {}", tenure);
    println!("  Max neighbors: 500");
    println!("  Iterations: 200\n");

    // Run optimization
    let result = ts.optimize(&objective, &space, Budget::Iterations(200));

    // Display results
    println!("=== Results ===\n");
    println!("Best tour found:");
    print!("  ");
    for (i, &city) in result.solution.iter().enumerate() {
        if i > 0 {
            print!(" -> ");
        }
        print!("{}", city_names[city]);
    }
    println!(" -> {}", city_names[result.solution[0]]); // Return to start

    println!("\nTotal distance: {:.0} km", result.objective_value);
    println!("Iterations: {}", result.iterations);
    println!("Termination: {:?}", result.termination);

    // Show convergence
    println!("\nConvergence (every 20 iterations):");
    for (i, &val) in result.history.iter().enumerate() {
        if i % 20 == 0 || i == result.history.len() - 1 {
            println!("  Iter {:3}: {:.0} km", i, val);
        }
    }

    // Verify tour validity
    println!("\n=== Tour Validation ===");
    let mut visited = vec![false; n];
    for &city in &result.solution {
        if visited[city] {
            println!("ERROR: City {} visited twice!", city_names[city]);
        }
        visited[city] = true;
    }
    if visited.iter().all(|&v| v) {
        println!("✓ All {} cities visited exactly once", n);
    }

    // Print leg-by-leg breakdown
    println!("\n=== Leg-by-Leg Breakdown ===");
    let mut cumulative = 0.0;
    for i in 0..result.solution.len() {
        let from = result.solution[i];
        let to = result.solution[(i + 1) % result.solution.len()];
        let dist = distances[from][to];
        cumulative += dist;
        println!(
            "  {} -> {}: {:.0} km (cumulative: {:.0} km)",
            city_names[from], city_names[to], dist, cumulative
        );
    }

    // Compare different tabu tenures
    println!("\n=== Sensitivity Analysis: Tabu Tenure ===");
    for tenure in [3, 5, 10, 15] {
        let mut ts_test = TabuSearch::new(tenure).with_seed(42);
        let res = ts_test.optimize(&objective, &space, Budget::Iterations(100));
        println!("  Tenure {:2}: {:.0} km", tenure, res.objective_value);
    }
}
