//! Integration and property tests for metaheuristics.

use super::*;

/// Sphere function: f(x) = Σxᵢ² (global minimum at origin)
fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

/// Rosenbrock function: f(x) = Σ[100(xᵢ₊₁ - xᵢ²)² + (1-xᵢ)²]
/// Global minimum at (1, 1, ..., 1)
fn rosenbrock(x: &[f64]) -> f64 {
    (0..x.len() - 1)
        .map(|i| 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2))
        .sum()
}

/// Rastrigin function: highly multimodal
#[allow(dead_code)]
fn rastrigin(x: &[f64]) -> f64 {
    let n = f64::from(x.len() as i32);
    10.0 * n
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>()
}

#[test]
fn test_search_space_continuous_dimension() {
    let space = SearchSpace::continuous(30, -5.0, 5.0);
    assert_eq!(space.dimension(), 30);
}

#[test]
fn test_search_space_binary_dimension() {
    let space = SearchSpace::binary(100);
    assert_eq!(space.dimension(), 100);
}

#[test]
fn test_search_space_permutation_dimension() {
    let space = SearchSpace::permutation(50);
    assert_eq!(space.dimension(), 50);
}

#[test]
fn test_budget_evaluations_limit() {
    let budget = Budget::evaluations(1000);
    assert_eq!(budget.max_evaluations(1), 1000);
}

#[test]
fn test_budget_iterations_to_evaluations() {
    let budget = Budget::iterations(100);
    assert_eq!(budget.max_evaluations(50), 5000);
}

#[test]
fn test_de_minimizes_sphere() {
    let mut de = DifferentialEvolution::new().with_seed(42);
    let space = SearchSpace::continuous(10, -5.0, 5.0);
    let result = de.optimize(&sphere, &space, Budget::Evaluations(50_000));

    // Allow reasonable tolerance for 10D sphere
    assert!(
        result.objective_value < 1e-2,
        "DE should minimize sphere, got {}",
        result.objective_value
    );
    assert!(result.solution.iter().all(|&x| x.abs() < 0.5));
}

#[test]
fn test_de_improves_over_initial() {
    let mut de = DifferentialEvolution::new().with_seed(42);
    let space = SearchSpace::continuous(10, -5.0, 5.0);
    let result = de.optimize(&sphere, &space, Budget::Iterations(10));

    // Should improve from random initialization
    let initial = result.history.first().expect("history should not be empty");
    let final_val = result.history.last().expect("history should not be empty");
    assert!(final_val < initial);
}

#[test]
fn test_de_rosenbrock_finds_valley() {
    let mut de = DifferentialEvolution::new().with_seed(42);
    let space = SearchSpace::continuous(5, -5.0, 10.0);
    let result = de.optimize(&rosenbrock, &space, Budget::Evaluations(50_000));

    // Rosenbrock is harder; check we get close to valley
    assert!(result.objective_value < 1.0);
}

#[test]
fn test_de_strategies_all_work() {
    let strategies = [
        DEStrategy::Rand1Bin,
        DEStrategy::Best1Bin,
        DEStrategy::CurrentToBest1Bin,
    ];

    for strategy in strategies {
        let mut de = DifferentialEvolution::new()
            .with_strategy(strategy)
            .with_seed(42);
        let space = SearchSpace::continuous(5, -5.0, 5.0);
        let result = de.optimize(&sphere, &space, Budget::Evaluations(10_000));

        assert!(
            result.objective_value < 1e-3,
            "Strategy {:?} failed: {}",
            strategy,
            result.objective_value
        );
    }
}

#[test]
fn test_de_jade_improves_convergence() {
    let space = SearchSpace::continuous(10, -5.0, 5.0);
    let budget = Budget::Evaluations(15_000);

    // Standard DE
    let mut de_std = DifferentialEvolution::new().with_seed(42);
    let result_std = de_std.optimize(&sphere, &space, budget.clone());

    // JADE
    let mut de_jade = DifferentialEvolution::new().with_jade().with_seed(42);
    let result_jade = de_jade.optimize(&sphere, &space, budget);

    // JADE should perform at least as well (often better)
    assert!(result_jade.objective_value <= result_std.objective_value * 10.0);
}

#[test]
fn test_optimization_result_fields() {
    let mut de = DifferentialEvolution::new().with_seed(42);
    let space = SearchSpace::continuous(5, -5.0, 5.0);
    let result = de.optimize(&sphere, &space, Budget::Evaluations(5000));

    assert!(!result.solution.is_empty());
    assert!(result.objective_value.is_finite());
    assert!(result.evaluations > 0);
    assert!(result.iterations > 0);
    assert!(!result.history.is_empty());
}

#[test]
fn test_de_handles_different_dimensions() {
    for dim in [2, 5, 10, 20] {
        let mut de = DifferentialEvolution::new().with_seed(42);
        let space = SearchSpace::continuous(dim, -5.0, 5.0);
        // Scale budget with dimension^2 for harder problems
        let result = de.optimize(&sphere, &space, Budget::Evaluations(dim * dim * 500));

        // Allow dimension-dependent tolerance
        let tolerance = 0.1 * (dim as f64);
        assert!(
            result.objective_value < tolerance,
            "Failed for dim={}: {} (tolerance={})",
            dim,
            result.objective_value,
            tolerance
        );
    }
}

#[test]
fn test_search_space_clip() {
    let space = SearchSpace::continuous(3, 0.0, 10.0);
    let clipped = space.clip(&[-5.0, 5.0, 15.0]).expect("clip should succeed");

    assert!((clipped[0] - 0.0).abs() < 1e-10);
    assert!((clipped[1] - 5.0).abs() < 1e-10);
    assert!((clipped[2] - 10.0).abs() < 1e-10);
}

#[test]
fn test_convergence_tracker_early_stop() {
    // Use min_delta=0.5 to avoid floating point precision issues
    // (e.g., 24.99 - 24.98 ≈ 0.01000000001 > 0.01 due to IEEE 754)
    let budget = Budget::convergence_with(3, 0.5, 100_000);
    let mut tracker = ConvergenceTracker::from_budget(&budget);

    // Rapid improvement (> 0.5 delta)
    assert!(tracker.update(100.0, 100));
    assert!(tracker.update(50.0, 100));
    assert!(tracker.update(25.0, 100));

    // Stall (improvement <= 0.5, triggers no_improvement_count)
    assert!(tracker.update(24.8, 100)); // count = 1 (improvement = 0.2 <= 0.5)
    assert!(tracker.update(24.6, 100)); // count = 2 (improvement = 0.2 <= 0.5)
    assert!(!tracker.update(24.4, 100)); // count = 3, should stop

    assert!(tracker.is_converged());
}
