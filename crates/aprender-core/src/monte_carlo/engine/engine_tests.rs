pub(crate) use super::*;

// Simple test model
pub(super) struct ConstantGrowthModel {
    initial: f64,
    growth_rate: f64,
}

impl SimulationModel for ConstantGrowthModel {
    fn name(&self) -> &'static str {
        "ConstantGrowth"
    }

    fn generate_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        let n = time_horizon.n_steps();
        let dt = time_horizon.dt();
        let noise = rng.standard_normal() * 0.01; // Small random component

        let values: Vec<f64> = (0..=n)
            .map(|i| {
                if i == 0 {
                    self.initial // No noise at t=0
                } else {
                    self.initial * (1.0 + self.growth_rate * i as f64 * dt + noise)
                }
            })
            .collect();

        SimulationPath::new(
            time_horizon.time_points(),
            values,
            PathMetadata {
                path_id,
                seed: rng.seed(),
                is_antithetic: false,
            },
        )
    }

    fn generate_antithetic_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        let n = time_horizon.n_steps();
        let dt = time_horizon.dt();
        let noise = -rng.standard_normal() * 0.01; // Negated noise

        let values: Vec<f64> = (0..=n)
            .map(|i| {
                if i == 0 {
                    self.initial // No noise at t=0
                } else {
                    self.initial * (1.0 + self.growth_rate * i as f64 * dt + noise)
                }
            })
            .collect();

        SimulationPath::new(
            time_horizon.time_points(),
            values,
            PathMetadata {
                path_id,
                seed: rng.seed(),
                is_antithetic: true,
            },
        )
    }
}

#[test]
fn test_engine_basic_simulation() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(100);

    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };

    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    assert_eq!(result.n_paths(), 100);
    assert_eq!(result.model_name, "ConstantGrowth");
    assert_eq!(result.seed, 42);
}

#[test]
fn test_engine_reproducibility() {
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);

    let result1 = MonteCarloEngine::new(42)
        .with_n_simulations(100)
        .simulate(&model, &horizon);

    let result2 = MonteCarloEngine::new(42)
        .with_n_simulations(100)
        .simulate(&model, &horizon);

    // Should be identical with same seed
    assert_eq!(result1.n_paths(), result2.n_paths());
    for (p1, p2) in result1.paths.iter().zip(result2.paths.iter()) {
        for (v1, v2) in p1.values.iter().zip(p2.values.iter()) {
            assert!((v1 - v2).abs() < 1e-10, "Values should match");
        }
    }
}

#[test]
fn test_engine_different_seeds() {
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);

    let result1 = MonteCarloEngine::new(42)
        .with_n_simulations(100)
        .simulate(&model, &horizon);

    let result2 = MonteCarloEngine::new(43)
        .with_n_simulations(100)
        .simulate(&model, &horizon);

    // Should differ with different seeds
    let v1 = result1.paths[0].values[result1.paths[0].values.len() - 1];
    let v2 = result2.paths[0].values[result2.paths[0].values.len() - 1];
    assert!(
        (v1 - v2).abs() > 1e-10,
        "Different seeds should give different results"
    );
}

#[test]
fn test_simulation_result_statistics() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(1000);

    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };

    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let stats = result.final_value_statistics();
    assert!(stats.mean > 100.0, "Mean should be above initial");
    assert!(stats.std > 0.0, "Should have some variance");
}

#[test]
fn test_convergence_budget() {
    let engine = MonteCarloEngine::new(42);

    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };

    let horizon = TimeHorizon::years(1);
    let budget = Budget::Convergence {
        patience: 5,
        min_delta: 0.001,
        max_simulations: 10_000,
    };

    let result = engine.simulate_with_budget(&model, &horizon, &budget);

    // Should have run some simulations
    assert!(result.n_paths() > 0);
    assert!(result.n_paths() <= 10_000);
}

#[test]
fn test_antithetic_variance_reduction() {
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);

    // Without antithetic - disable early convergence
    let result_std = MonteCarloEngine::new(42)
        .with_n_simulations(1000)
        .with_target_precision(0.0) // Disable early convergence
        .with_variance_reduction(VarianceReduction::None)
        .simulate(&model, &horizon);

    // With antithetic
    let result_anti = MonteCarloEngine::new(42)
        .with_n_simulations(1000)
        .with_target_precision(0.0) // Disable early convergence
        .with_variance_reduction(VarianceReduction::Antithetic)
        .simulate(&model, &horizon);

    // Both should have same number of paths (1000)
    assert_eq!(result_std.n_paths(), result_anti.n_paths());

    // Means should be similar (antithetic is also unbiased)
    let mean_std = result_std.final_value_statistics().mean;
    let mean_anti = result_anti.final_value_statistics().mean;
    assert!(
        (mean_std - mean_anti).abs() / mean_std < 0.05,
        "Means should be similar: std={mean_std}, anti={mean_anti}"
    );
}

#[test]
fn test_values_at_time() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(100);

    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };

    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let initial_values = result.values_at_time(0);
    assert_eq!(initial_values.len(), 100);

    // All initial values should be 100
    for v in initial_values {
        assert!((v - 100.0).abs() < 1e-6);
    }
}

#[test]
fn test_statistics_over_time() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(100);

    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };

    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let stats_over_time = result.statistics_over_time();

    // Should have statistics for each time point
    assert!(!stats_over_time.is_empty());

    // Mean should increase over time (positive growth)
    if stats_over_time.len() > 1 {
        assert!(stats_over_time.last().unwrap().mean > stats_over_time.first().unwrap().mean);
    }
}

#[test]
fn test_percentiles() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(1000);

    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };

    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let pcts = result.final_value_percentiles();
    assert!(pcts.p5 < pcts.p50);
    assert!(pcts.p50 < pcts.p95);
}

#[test]
fn test_engine_default() {
    let engine = MonteCarloEngine::default();
    assert_eq!(engine.seed, 42);
    assert_eq!(engine.n_simulations, 10_000);
    assert_eq!(engine.max_simulations, 100_000);
}

#[test]
fn test_engine_reproducible() {
    let engine = MonteCarloEngine::reproducible(99);
    assert_eq!(engine.seed, 99);
    assert_eq!(engine.n_simulations, 10_000);
}

#[test]
fn test_engine_with_max_simulations() {
    let engine = MonteCarloEngine::new(42).with_max_simulations(500);
    assert_eq!(engine.max_simulations, 500);
}

#[test]
fn test_engine_debug_clone() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(100);
    let debug_str = format!("{:?}", engine);
    assert!(debug_str.contains("MonteCarloEngine"));

    let cloned = engine.clone();
    assert_eq!(cloned.seed, 42);
    assert_eq!(cloned.n_simulations, 100);
}

#[test]
fn test_simulate_with_budget_simulations() {
    let engine = MonteCarloEngine::new(42);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let budget = Budget::Simulations(200);

    let result = engine.simulate_with_budget(&model, &horizon, &budget);
    assert!(result.n_paths() > 0);
    assert!(result.n_paths() <= 200);
}

#[test]
fn test_simulate_with_budget_evaluations() {
    let engine = MonteCarloEngine::new(42);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let budget = Budget::Evaluations(300);

    let result = engine.simulate_with_budget(&model, &horizon, &budget);
    assert!(result.n_paths() > 0);
    assert!(result.n_paths() <= 300);
}

#[test]
fn test_simulate_with_budget_no_antithetic() {
    let engine = MonteCarloEngine::new(42).with_variance_reduction(VarianceReduction::None);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let budget = Budget::Simulations(100);

    let result = engine.simulate_with_budget(&model, &horizon, &budget);
    assert_eq!(result.n_paths(), 100);
}

#[test]
fn test_simulate_no_antithetic() {
    let engine = MonteCarloEngine::new(42)
        .with_n_simulations(50)
        .with_variance_reduction(VarianceReduction::None);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);

    let result = engine.simulate(&model, &horizon);
    assert_eq!(result.n_paths(), 50);
}

#[test]
fn test_simulation_result_return_statistics() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(100);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let return_stats = result.return_statistics();
    assert!(return_stats.n > 0);
}

#[test]
fn test_simulation_result_return_percentiles() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(100);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let pcts = result.return_percentiles();
    assert!(pcts.p50.is_finite());
}

#[test]
fn test_simulation_result_total_returns() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(50);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let returns = result.total_returns();
    assert!(!returns.is_empty());
    // Growth rate is positive so returns should be positive
    for r in &returns {
        assert!(r.is_finite());
    }
}

#[test]
fn test_simulation_result_empty_paths() {
    let result = SimulationResult {
        paths: vec![],
        diagnostics: ConvergenceDiagnostics::new(),
        model_name: "empty".to_string(),
        seed: 0,
    };

    assert_eq!(result.n_paths(), 0);
    assert!(result.final_values().is_empty());
    assert!(result.total_returns().is_empty());
    assert!(result.statistics_over_time().is_empty());
    assert!(result.values_at_time(0).is_empty());
}

#[test]
fn test_simulation_result_debug_clone() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(10);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("SimulationResult"));

    let cloned = result.clone();
    assert_eq!(cloned.n_paths(), result.n_paths());
    assert_eq!(cloned.model_name, result.model_name);
}

#[path = "no_antithetic_model.rs"]
mod no_antithetic_model;
