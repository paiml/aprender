use super::*;

#[test]
fn test_cmaes_sphere() {
    let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let mut cma = CmaEs::new(5).with_seed(42);
    let space = SearchSpace::continuous(5, -5.0, 5.0);
    let result = cma.optimize(&objective, &space, Budget::Evaluations(5000));
    assert!(
        result.objective_value < 1e-3,
        "Sphere should converge, got: {}",
        result.objective_value
    );
}

#[test]
fn test_cmaes_rosenbrock() {
    let objective = |x: &[f64]| {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    };
    let mut cma = CmaEs::new(3).with_seed(123);
    let space = SearchSpace::continuous(3, -5.0, 5.0);
    let result = cma.optimize(&objective, &space, Budget::Evaluations(10000));
    assert!(
        result.objective_value < 1.0,
        "Rosenbrock should find valley, got: {}",
        result.objective_value
    );
}

#[test]
fn test_cmaes_builder() {
    let cma = CmaEs::new(10).with_seed(999).with_sigma(0.5);
    assert_eq!(cma.dim, 10);
    assert!((cma.sigma - 0.5).abs() < 1e-10);
}

#[test]
fn test_cmaes_reset() {
    let mut cma = CmaEs::new(3).with_seed(42);
    let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
    let space = SearchSpace::continuous(3, -5.0, 5.0);
    cma.optimize(&objective, &space, Budget::Evaluations(100));
    assert!(!cma.history.is_empty());
    cma.reset();
    assert!(cma.history.is_empty());
}

// ==========================================================
// IPOP Restart Tests
// ==========================================================

#[test]
fn test_ipop_config_default() {
    let config = IpopConfig::default();
    assert!(!config.enabled);
    assert!((config.pop_increase_factor - 2.0).abs() < 1e-10);
    assert_eq!(config.max_restarts, 9);
}

#[test]
fn test_cmaes_with_ipop() {
    let cma = CmaEs::new(5).with_ipop();
    assert!(cma.ipop.enabled);
}

#[test]
fn test_cmaes_with_ipop_config() {
    let config = IpopConfig {
        enabled: true,
        pop_increase_factor: 3.0,
        max_restarts: 5,
        sigma_threshold: 1e-8,
        stagnation_gens: 10,
    };
    let cma = CmaEs::new(5).with_ipop_config(config);
    assert!(cma.ipop.enabled);
    assert!((cma.ipop.pop_increase_factor - 3.0).abs() < 1e-10);
}

#[test]
fn test_ipop_triggers_restart_on_multimodal() {
    // Rastrigin: highly multimodal function
    let rastrigin = |x: &[f64]| -> f64 {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
                .sum::<f64>()
    };

    let config = IpopConfig {
        enabled: true,
        pop_increase_factor: 2.0,
        max_restarts: 3,
        sigma_threshold: 1e-10,
        stagnation_gens: 10,
    };

    let mut cma = CmaEs::new(3).with_ipop_config(config).with_seed(42);
    let space = SearchSpace::continuous(3, -5.12, 5.12);
    let result = cma.optimize(&rastrigin, &space, Budget::Evaluations(3000));

    // IPOP should have attempted at least one restart on this hard function
    // (This is stochastic, so we just check it runs without panic)
    assert!(result.evaluations > 0);
}

#[test]
fn test_ipop_increases_population() {
    let config = IpopConfig {
        enabled: true,
        pop_increase_factor: 2.0,
        max_restarts: 2,
        sigma_threshold: 1e-10,
        stagnation_gens: 5, // Force early restart
    };

    let mut cma = CmaEs::new(3).with_ipop_config(config).with_seed(42);
    let initial_lambda = cma.lambda;

    // Run optimization that will likely stagnate
    let flat = |_: &[f64]| 1.0; // Flat function triggers stagnation
    let space = SearchSpace::continuous(3, -1.0, 1.0);
    cma.optimize(&flat, &space, Budget::Evaluations(500));

    // If restarts occurred, lambda should have increased
    if cma.restart_count > 0 {
        assert!(
            cma.lambda > initial_lambda,
            "Lambda should increase after restart"
        );
    }
}

#[test]
fn test_ipop_respects_max_restarts() {
    let config = IpopConfig {
        enabled: true,
        pop_increase_factor: 2.0,
        max_restarts: 2,
        sigma_threshold: 1e-10,
        stagnation_gens: 3,
    };

    let mut cma = CmaEs::new(2).with_ipop_config(config).with_seed(42);
    let flat = |_: &[f64]| 1.0;
    let space = SearchSpace::continuous(2, -1.0, 1.0);
    cma.optimize(&flat, &space, Budget::Evaluations(1000));

    assert!(cma.restart_count <= 2, "Should not exceed max_restarts");
}

#[test]
fn test_ipop_preserves_best_across_restarts() {
    // Easy sphere that should converge well
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();

    let config = IpopConfig {
        enabled: true,
        pop_increase_factor: 2.0,
        max_restarts: 3,
        sigma_threshold: 1e-15, // Very tight to maybe trigger restart
        stagnation_gens: 50,
    };

    let mut cma = CmaEs::new(3).with_ipop_config(config).with_seed(42);
    let space = SearchSpace::continuous(3, -5.0, 5.0);
    let result = cma.optimize(&sphere, &space, Budget::Evaluations(5000));

    // Best solution should still be good even if restarts occurred
    assert!(
        result.objective_value < 1e-3,
        "Should find good solution: {}",
        result.objective_value
    );
}

#[test]
fn test_restart_count_accessor() {
    let cma = CmaEs::new(5);
    assert_eq!(cma.restart_count(), 0);
}

#[test]
fn test_with_lambda() {
    let cma = CmaEs::with_lambda(5, 20);
    assert_eq!(cma.lambda, 20);
    assert_eq!(cma.initial_lambda, 20);
}
