//! Core Monte Carlo simulation engine
//!
//! This module provides the foundation for Monte Carlo simulations,
//! including reproducible random number generation, variance reduction
//! techniques, and convergence diagnostics.

mod convergence;
mod rng;
mod types;
mod variance;

pub use convergence::{autocorrelation, calculate_ess, ess_autocorr, ConvergenceDiagnostics};
pub use rng::MonteCarloRng;
pub use types::{
    percentile, Budget, PathMetadata, Percentiles, SimulationPath, Statistics, TimeHorizon,
    TimeStep,
};
pub use variance::{empirical_variance_reduction, inverse_normal_cdf, VarianceReduction};

/// Trait for simulation models
pub trait SimulationModel: Send + Sync {
    /// Get the model name
    fn name(&self) -> &'static str;

    /// Generate a single simulation path
    fn generate_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath;

    /// Generate an antithetic path for variance reduction
    fn generate_antithetic_path(
        &self,
        rng: &mut MonteCarloRng,
        time_horizon: &TimeHorizon,
        path_id: usize,
    ) -> SimulationPath {
        // Default implementation just generates a regular path
        self.generate_path(rng, time_horizon, path_id)
    }
}

/// Monte Carlo simulation engine
#[derive(Debug, Clone)]
pub struct MonteCarloEngine {
    /// Number of simulation paths
    pub n_simulations: usize,

    /// Random seed for reproducibility
    pub seed: u64,

    /// Variance reduction technique
    pub variance_reduction: VarianceReduction,

    /// Target relative precision for convergence
    pub target_precision: f64,

    /// Maximum simulations before stopping
    pub max_simulations: usize,
}

impl MonteCarloEngine {
    /// Create a new engine with specified seed
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            n_simulations: 10_000,
            seed,
            variance_reduction: VarianceReduction::Antithetic,
            target_precision: 0.01,
            max_simulations: 100_000,
        }
    }

    /// Create a reproducible engine with specific seed
    #[must_use]
    pub fn reproducible(seed: u64) -> Self {
        Self::new(seed)
    }

    /// Set number of simulations
    #[must_use]
    pub fn with_n_simulations(mut self, n: usize) -> Self {
        self.n_simulations = n;
        self
    }

    /// Set variance reduction technique
    #[must_use]
    pub fn with_variance_reduction(mut self, vr: VarianceReduction) -> Self {
        self.variance_reduction = vr;
        self
    }

    /// Set target precision
    #[must_use]
    pub fn with_target_precision(mut self, precision: f64) -> Self {
        self.target_precision = precision;
        self
    }

    /// Set maximum simulations
    #[must_use]
    pub fn with_max_simulations(mut self, max: usize) -> Self {
        self.max_simulations = max;
        self
    }

    /// Run simulation with given model
    pub fn simulate<M: SimulationModel>(
        &self,
        model: &M,
        time_horizon: &TimeHorizon,
    ) -> SimulationResult {
        let mut rng = MonteCarloRng::new(self.seed);
        let use_antithetic = matches!(self.variance_reduction, VarianceReduction::Antithetic);

        let effective_n = if use_antithetic {
            self.n_simulations / 2
        } else {
            self.n_simulations
        };

        let mut paths = Vec::with_capacity(self.n_simulations);
        let mut diagnostics = ConvergenceDiagnostics::new();

        for i in 0..effective_n {
            // Generate primary path
            let path = model.generate_path(&mut rng, time_horizon, i);
            paths.push(path);

            // Generate antithetic path if enabled
            if use_antithetic {
                let antithetic = model.generate_antithetic_path(&mut rng, time_horizon, i);
                paths.push(antithetic);
            }

            // Update diagnostics periodically
            if (i + 1) % 100 == 0 || i == effective_n - 1 {
                diagnostics.update(&paths);

                // Check for early convergence
                if diagnostics.is_converged(self.target_precision) && i >= 100 {
                    break;
                }
            }
        }

        SimulationResult {
            paths,
            diagnostics,
            model_name: model.name().to_string(),
            seed: self.seed,
        }
    }

    /// Run simulation with custom budget
    pub fn simulate_with_budget<M: SimulationModel>(
        &self,
        model: &M,
        time_horizon: &TimeHorizon,
        budget: &Budget,
    ) -> SimulationResult {
        let mut rng = MonteCarloRng::new(self.seed);
        let use_antithetic = matches!(self.variance_reduction, VarianceReduction::Antithetic);

        let max_n = budget.max_simulations();
        let effective_n = if use_antithetic { max_n / 2 } else { max_n };

        let mut paths = Vec::with_capacity(max_n);
        let mut diagnostics = ConvergenceDiagnostics::new();
        let mut converged = false;

        for i in 0..effective_n {
            let path = model.generate_path(&mut rng, time_horizon, i);
            paths.push(path);

            if use_antithetic {
                let antithetic = model.generate_antithetic_path(&mut rng, time_horizon, i);
                paths.push(antithetic);
            }

            // Check budget-specific stopping criteria
            if let Budget::Convergence {
                patience,
                min_delta,
                ..
            } = &budget
            {
                if (i + 1) % 100 == 0 {
                    diagnostics.update(&paths);

                    // Check convergence with patience
                    if diagnostics.running_mean_history.len() >= *patience {
                        let recent = &diagnostics.running_mean_history
                            [diagnostics.running_mean_history.len() - patience..];

                        let range = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                            - recent.iter().copied().fold(f64::INFINITY, f64::min);

                        if range < *min_delta {
                            converged = true;
                            break;
                        }
                    }
                }
            }
        }

        if !converged {
            diagnostics.update(&paths);
        }

        SimulationResult {
            paths,
            diagnostics,
            model_name: model.name().to_string(),
            seed: self.seed,
        }
    }
}

impl Default for MonteCarloEngine {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Result of a Monte Carlo simulation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// All simulation paths
    pub paths: Vec<SimulationPath>,

    /// Convergence diagnostics
    pub diagnostics: ConvergenceDiagnostics,

    /// Model name
    pub model_name: String,

    /// Seed used for simulation
    pub seed: u64,
}

impl SimulationResult {
    /// Get number of paths
    #[must_use]
    pub fn n_paths(&self) -> usize {
        self.paths.len()
    }

    /// Get final values from all paths
    #[must_use]
    pub fn final_values(&self) -> Vec<f64> {
        self.paths
            .iter()
            .filter_map(SimulationPath::final_value)
            .collect()
    }

    /// Get total returns from all paths
    #[must_use]
    pub fn total_returns(&self) -> Vec<f64> {
        self.paths
            .iter()
            .filter_map(SimulationPath::total_return)
            .collect()
    }

    /// Calculate statistics for final values
    #[must_use]
    pub fn final_value_statistics(&self) -> Statistics {
        Statistics::from_values(&self.final_values())
    }

    /// Calculate statistics for returns
    #[must_use]
    pub fn return_statistics(&self) -> Statistics {
        Statistics::from_values(&self.total_returns())
    }

    /// Get values at specific time index across all paths
    #[must_use]
    pub fn values_at_time(&self, time_idx: usize) -> Vec<f64> {
        self.paths
            .iter()
            .filter_map(|p| p.values.get(time_idx).copied())
            .collect()
    }

    /// Calculate statistics at each time point
    #[must_use]
    pub fn statistics_over_time(&self) -> Vec<Statistics> {
        if self.paths.is_empty() {
            return Vec::new();
        }

        let n_times = self.paths.first().map_or(0, |p| p.values.len());

        (0..n_times)
            .map(|t| Statistics::from_values(&self.values_at_time(t)))
            .collect()
    }

    /// Get percentiles of final values
    #[must_use]
    pub fn final_value_percentiles(&self) -> Percentiles {
        Percentiles::from_values(&self.final_values())
    }

    /// Get percentiles of returns
    #[must_use]
    pub fn return_percentiles(&self) -> Percentiles {
        Percentiles::from_values(&self.total_returns())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple test model
    struct ConstantGrowthModel {
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
}
