use super::*;

// Test default implementation of generate_antithetic_path
struct NoAntitheticModel;

impl SimulationModel for NoAntitheticModel {
    fn name(&self) -> &'static str {
        "NoAntithetic"
    }

    fn generate_path(
        &self,
        _rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        let n = time_horizon.n_steps();
        SimulationPath::new(
            time_horizon.time_points(),
            vec![1.0; n + 1],
            PathMetadata {
                path_id,
                seed: 0,
                is_antithetic: false,
            },
        )
    }
}

#[test]
fn test_default_antithetic_path_impl() {
    let model = NoAntitheticModel;
    let mut rng = MonteCarloRng::new(42);
    let horizon = TimeHorizon::years(1);

    // Default antithetic just calls generate_path
    let path = model.generate_antithetic_path(&mut rng, &horizon, 0);
    assert_eq!(path.values.len(), horizon.n_steps() + 1);
}

#[test]
fn test_simulate_with_budget_convergence_not_converged() {
    // Use a budget where convergence won't happen quickly
    let engine = MonteCarloEngine::new(42).with_variance_reduction(VarianceReduction::None);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let budget = Budget::Convergence {
        patience: 100,  // Very high patience
        min_delta: 0.0, // Impossible to converge
        max_simulations: 200,
    };

    let result = engine.simulate_with_budget(&model, &horizon, &budget);
    // Should run all simulations since convergence won't happen
    assert!(result.n_paths() > 0);
}

#[test]
fn test_values_at_time_out_of_bounds() {
    let engine = MonteCarloEngine::new(42).with_n_simulations(10);
    let model = ConstantGrowthModel {
        initial: 100.0,
        growth_rate: 0.05,
    };
    let horizon = TimeHorizon::years(1);
    let result = engine.simulate(&model, &horizon);

    // Request a time index beyond path length
    let vals = result.values_at_time(99999);
    assert!(vals.is_empty());
}
