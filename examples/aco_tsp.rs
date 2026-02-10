//! Ant Colony Optimization for the Traveling Salesman Problem
//!
//! This example demonstrates ACO solving a classic TSP instance.
//! Ants deposit pheromones on good paths, guiding future ants toward
//! shorter tours through emergent swarm intelligence.
//!
//! Run with: `cargo run --example aco_tsp`

use aprender::metaheuristics::{AntColony, Budget, ConstructiveMetaheuristic, SearchSpace};

fn main() {
    println!("=== Ant Colony Optimization: Traveling Salesman Problem ===\n");

    // 10-city TSP instance (symmetric distances)
    // Cities arranged roughly in a circle for visualization
    let city_names = [
        "Atlanta",
        "Boston",
        "Chicago",
        "Denver",
        "El Paso",
        "Fresno",
        "Green Bay",
        "Houston",
        "Indianapolis",
        "Jacksonville",
    ];

    // Distance matrix (approximate driving distances in miles)
    let distances: Vec<Vec<f64>> = vec![
        vec![
            0.0, 1100.0, 720.0, 1400.0, 1400.0, 2200.0, 900.0, 800.0, 530.0, 350.0,
        ], // Atlanta
        vec![
            1100.0, 0.0, 980.0, 1960.0, 2300.0, 3100.0, 1100.0, 1850.0, 940.0, 1150.0,
        ], // Boston
        vec![
            720.0, 980.0, 0.0, 1000.0, 1700.0, 2100.0, 210.0, 1090.0, 180.0, 900.0,
        ], // Chicago
        vec![
            1400.0, 1960.0, 1000.0, 0.0, 680.0, 1200.0, 1300.0, 1030.0, 1060.0, 1900.0,
        ], // Denver
        vec![
            1400.0, 2300.0, 1700.0, 680.0, 0.0, 750.0, 1900.0, 750.0, 1500.0, 1700.0,
        ], // El Paso
        vec![
            2200.0, 3100.0, 2100.0, 1200.0, 750.0, 0.0, 2100.0, 1700.0, 2100.0, 2600.0,
        ], // Fresno
        vec![
            900.0, 1100.0, 210.0, 1300.0, 1900.0, 2100.0, 0.0, 1200.0, 400.0, 1100.0,
        ], // Green Bay
        vec![
            800.0, 1850.0, 1090.0, 1030.0, 750.0, 1700.0, 1200.0, 0.0, 1000.0, 850.0,
        ], // Houston
        vec![
            530.0, 940.0, 180.0, 1060.0, 1500.0, 2100.0, 400.0, 1000.0, 0.0, 800.0,
        ], // Indianapolis
        vec![
            350.0, 1150.0, 900.0, 1900.0, 1700.0, 2600.0, 1100.0, 850.0, 800.0, 0.0,
        ], // Jacksonville
    ];

    let n = distances.len();

    // Build adjacency list for SearchSpace::Graph
    let adjacency: Vec<Vec<(usize, f64)>> = distances
        .iter()
        .enumerate()
        .map(|(i, row)| {
            row.iter()
                .enumerate()
                .filter(|&(j, _)| i != j)
                .map(|(j, &d)| (j, d))
                .collect()
        })
        .collect();

    let space = SearchSpace::Graph {
        num_nodes: n,
        adjacency,
        heuristic: None, // ACO will compute 1/distance automatically
    };

    // Objective: total tour length
    let objective = |tour: &Vec<usize>| -> f64 {
        let mut total = 0.0;
        for i in 0..tour.len() {
            let from = tour[i];
            let to = tour[(i + 1) % tour.len()];
            total += distances[from][to];
        }
        total
    };

    println!("Problem: Visit all {} cities and return to start", n);
    println!("Optimization: Minimize total distance\n");

    // Configure ACO
    let mut aco = AntColony::new(20) // 20 ants
        .with_alpha(1.0) // Pheromone importance
        .with_beta(2.5) // Heuristic importance (prefer shorter edges)
        .with_rho(0.1) // Evaporation rate
        .with_seed(42);

    println!("ACO Parameters:");
    println!("  Ants: 20");
    println!("  Alpha (pheromone weight): 1.0");
    println!("  Beta (heuristic weight): 2.5");
    println!("  Rho (evaporation): 0.1");
    println!("  Iterations: 100\n");

    // Run optimization
    let result = aco.optimize(&objective, &space, Budget::Iterations(100));

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

    println!("\nTotal distance: {:.0} miles", result.objective_value);
    println!("Iterations: {}", result.iterations);
    println!("Termination: {:?}", result.termination);

    // Show convergence
    println!("\nConvergence (every 10 iterations):");
    for (i, &val) in result.history.iter().enumerate() {
        if i % 10 == 0 || i == result.history.len() - 1 {
            println!("  Iter {:3}: {:.0} miles", i, val);
        }
    }

    // Compare with greedy nearest neighbor
    println!("\n=== Comparison with Greedy Nearest Neighbor ===");
    let greedy_tour = greedy_tsp(&distances);
    let greedy_dist = objective(&greedy_tour);
    print!("Greedy tour: ");
    for (i, &city) in greedy_tour.iter().enumerate() {
        if i > 0 {
            print!(" -> ");
        }
        print!("{}", city_names[city]);
    }
    println!(" -> {}", city_names[greedy_tour[0]]);
    println!("Greedy distance: {:.0} miles", greedy_dist);

    let improvement = (greedy_dist - result.objective_value) / greedy_dist * 100.0;
    if improvement > 0.0 {
        println!(
            "\nACO improved over greedy by {:.1}% ({:.0} miles saved)",
            improvement,
            greedy_dist - result.objective_value
        );
    }
}

/// Simple greedy nearest neighbor heuristic for comparison
fn greedy_tsp(distances: &[Vec<f64>]) -> Vec<usize> {
    let n = distances.len();
    let mut tour = vec![0]; // Start from city 0
    let mut visited = vec![false; n];
    visited[0] = true;

    while tour.len() < n {
        let current = *tour.last().expect("tour should never be empty");
        let mut best_next = 0;
        let mut best_dist = f64::INFINITY;

        for j in 0..n {
            if !visited[j] && distances[current][j] < best_dist {
                best_dist = distances[current][j];
                best_next = j;
            }
        }

        tour.push(best_next);
        visited[best_next] = true;
    }

    tour
}
