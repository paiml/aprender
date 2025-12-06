//! HyperoptSearch: High-level Hyperparameter Optimization Wrapper
//!
//! Provides a scikit-learn style interface for hyperparameter tuning
//! using metaheuristic algorithms as the search backend.
//!
//! # Example
//!
//! ```
//! use aprender::metaheuristics::{HyperoptSearch, Hyperparameter, HyperparameterSet, Budget};
//!
//! // Define hyperparameter search space
//! let search = HyperoptSearch::new()
//!     .add_real("learning_rate", 1e-5, 1e-1, true)  // log scale
//!     .add_real("regularization", 1e-6, 1e-2, true)
//!     .add_int("n_estimators", 10, 500)
//!     .add_categorical("activation", &["relu", "tanh", "sigmoid"])
//!     .with_seed(42);
//!
//! // Define objective (returns loss to minimize)
//! let objective = |params: &HyperparameterSet| -> f64 {
//!     let lr = params.get_real("learning_rate").unwrap();
//!     let reg = params.get_real("regularization").unwrap();
//!     // Simulated validation loss
//!     (lr - 0.01).powi(2) + (reg - 0.001).powi(2)
//! };
//!
//! let result = search.minimize(objective, Budget::Evaluations(100));
//! println!("Best params: {:?}", result.best_params);
//! ```

use super::{
    Budget, DifferentialEvolution, OptimizationResult, PerturbativeMetaheuristic, SearchSpace,
};
use std::collections::HashMap;

/// Hyperparameter definition with bounds and scaling.
#[derive(Debug, Clone)]
pub enum Hyperparameter {
    /// Real-valued parameter with optional log scaling
    Real {
        name: String,
        lower: f64,
        upper: f64,
        log_scale: bool,
    },
    /// Integer-valued parameter
    Int {
        name: String,
        lower: i64,
        upper: i64,
    },
    /// Categorical parameter with discrete choices
    Categorical { name: String, choices: Vec<String> },
}

impl Hyperparameter {
    /// Get the parameter name.
    pub fn name(&self) -> &str {
        match self {
            Self::Real { name, .. } | Self::Int { name, .. } | Self::Categorical { name, .. } => {
                name
            }
        }
    }

    /// Get the dimensionality this parameter contributes.
    pub fn dim(&self) -> usize {
        match self {
            Self::Real { .. } | Self::Int { .. } => 1,
            Self::Categorical { choices, .. } => choices.len(),
        }
    }
}

/// A set of hyperparameter values.
#[derive(Debug, Clone, Default)]
pub struct HyperparameterSet {
    reals: HashMap<String, f64>,
    ints: HashMap<String, i64>,
    categoricals: HashMap<String, String>,
}

impl HyperparameterSet {
    /// Create empty parameter set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get a real-valued parameter.
    pub fn get_real(&self, name: &str) -> Option<f64> {
        self.reals.get(name).copied()
    }

    /// Get an integer parameter.
    pub fn get_int(&self, name: &str) -> Option<i64> {
        self.ints.get(name).copied()
    }

    /// Get a categorical parameter.
    pub fn get_categorical(&self, name: &str) -> Option<&str> {
        self.categoricals.get(name).map(String::as_str)
    }

    /// Set a real-valued parameter.
    pub fn set_real(&mut self, name: &str, value: f64) {
        self.reals.insert(name.to_string(), value);
    }

    /// Set an integer parameter.
    pub fn set_int(&mut self, name: &str, value: i64) {
        self.ints.insert(name.to_string(), value);
    }

    /// Set a categorical parameter.
    pub fn set_categorical(&mut self, name: &str, value: &str) {
        self.categoricals
            .insert(name.to_string(), value.to_string());
    }
}

/// Result of hyperparameter optimization.
#[derive(Debug, Clone)]
pub struct HyperoptResult {
    /// Best hyperparameter configuration found
    pub best_params: HyperparameterSet,
    /// Best objective value achieved
    pub best_score: f64,
    /// Number of objective evaluations
    pub evaluations: usize,
    /// History of best scores
    pub history: Vec<f64>,
}

/// Search algorithm backend.
#[derive(Debug, Clone, Copy, Default)]
pub enum SearchAlgorithm {
    /// Differential Evolution (default)
    #[default]
    DifferentialEvolution,
    /// Particle Swarm Optimization
    ParticleSwarm,
    /// Simulated Annealing (single-point)
    SimulatedAnnealing,
    /// CMA-ES (for smaller dimensions)
    CmaEs,
}

/// High-level hyperparameter optimization interface.
#[derive(Debug, Clone)]
pub struct HyperoptSearch {
    parameters: Vec<Hyperparameter>,
    algorithm: SearchAlgorithm,
    seed: Option<u64>,
    n_jobs: usize,
}

impl Default for HyperoptSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl HyperoptSearch {
    /// Create a new hyperparameter search.
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
            algorithm: SearchAlgorithm::DifferentialEvolution,
            seed: None,
            n_jobs: 1,
        }
    }

    /// Add a real-valued hyperparameter.
    pub fn add_real(mut self, name: &str, lower: f64, upper: f64, log_scale: bool) -> Self {
        self.parameters.push(Hyperparameter::Real {
            name: name.to_string(),
            lower,
            upper,
            log_scale,
        });
        self
    }

    /// Add an integer hyperparameter.
    pub fn add_int(mut self, name: &str, lower: i64, upper: i64) -> Self {
        self.parameters.push(Hyperparameter::Int {
            name: name.to_string(),
            lower,
            upper,
        });
        self
    }

    /// Add a categorical hyperparameter.
    pub fn add_categorical(mut self, name: &str, choices: &[&str]) -> Self {
        self.parameters.push(Hyperparameter::Categorical {
            name: name.to_string(),
            choices: choices.iter().map(|s| (*s).to_string()).collect(),
        });
        self
    }

    /// Set the search algorithm backend.
    pub fn with_algorithm(mut self, algorithm: SearchAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Set random seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set number of parallel jobs (reserved for future use).
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = n_jobs.max(1);
        self
    }

    /// Get total search space dimensionality.
    fn total_dim(&self) -> usize {
        self.parameters.iter().map(Hyperparameter::dim).sum()
    }

    /// Build internal search space for optimizer.
    fn build_search_space(&self) -> SearchSpace {
        let dim = self.total_dim();
        // All parameters are mapped to [0, 1] internally
        SearchSpace::Continuous {
            dim,
            lower: vec![0.0; dim],
            upper: vec![1.0; dim],
        }
    }

    /// Decode a continuous vector back to hyperparameters.
    fn decode(&self, x: &[f64]) -> HyperparameterSet {
        let mut params = HyperparameterSet::new();
        let mut idx = 0;

        for param in &self.parameters {
            match param {
                Hyperparameter::Real {
                    name,
                    lower,
                    upper,
                    log_scale,
                } => {
                    let t = x[idx].clamp(0.0, 1.0);
                    let value = if *log_scale {
                        // Log-uniform: exp(t * (ln(upper) - ln(lower)) + ln(lower))
                        let log_lower = lower.ln();
                        let log_upper = upper.ln();
                        (t * (log_upper - log_lower) + log_lower).exp()
                    } else {
                        // Linear interpolation
                        t * (upper - lower) + lower
                    };
                    params.set_real(name, value);
                    idx += 1;
                }
                Hyperparameter::Int { name, lower, upper } => {
                    let t = x[idx].clamp(0.0, 1.0);
                    let range = (*upper - *lower) as f64;
                    let value = (t * range + *lower as f64).round() as i64;
                    params.set_int(name, value.clamp(*lower, *upper));
                    idx += 1;
                }
                Hyperparameter::Categorical { name, choices } => {
                    // One-hot encoding: find max
                    let n = choices.len();
                    let mut max_idx = 0;
                    let mut max_val = x[idx];
                    for i in 1..n {
                        if x[idx + i] > max_val {
                            max_val = x[idx + i];
                            max_idx = i;
                        }
                    }
                    params.set_categorical(name, &choices[max_idx]);
                    idx += n;
                }
            }
        }

        params
    }

    /// Minimize an objective function over the hyperparameter space.
    pub fn minimize<F>(&self, objective: F, budget: Budget) -> HyperoptResult
    where
        F: Fn(&HyperparameterSet) -> f64,
    {
        let space = self.build_search_space();

        // Wrap objective to decode parameters
        let wrapped_objective = |x: &[f64]| -> f64 {
            let params = self.decode(x);
            objective(&params)
        };

        let result: OptimizationResult<Vec<f64>> = match self.algorithm {
            SearchAlgorithm::DifferentialEvolution => {
                let mut de = DifferentialEvolution::default();
                if let Some(seed) = self.seed {
                    de = de.with_seed(seed);
                }
                de.optimize(&wrapped_objective, &space, budget)
            }
            SearchAlgorithm::ParticleSwarm => {
                let mut pso = super::ParticleSwarm::default();
                if let Some(seed) = self.seed {
                    pso = pso.with_seed(seed);
                }
                pso.optimize(&wrapped_objective, &space, budget)
            }
            SearchAlgorithm::SimulatedAnnealing => {
                let mut sa = super::SimulatedAnnealing::default();
                if let Some(seed) = self.seed {
                    sa = sa.with_seed(seed);
                }
                sa.optimize(&wrapped_objective, &space, budget)
            }
            SearchAlgorithm::CmaEs => {
                let dim = self.total_dim();
                let mut cmaes = super::CmaEs::new(dim);
                if let Some(seed) = self.seed {
                    cmaes = cmaes.with_seed(seed);
                }
                cmaes.optimize(&wrapped_objective, &space, budget)
            }
        };

        HyperoptResult {
            best_params: self.decode(&result.solution),
            best_score: result.objective_value,
            evaluations: result.evaluations,
            history: result.history.clone(),
        }
    }

    /// Maximize an objective function (convenience wrapper).
    pub fn maximize<F>(&self, objective: F, budget: Budget) -> HyperoptResult
    where
        F: Fn(&HyperparameterSet) -> f64,
    {
        // Negate objective for maximization
        let mut result = self.minimize(|p| -objective(p), budget);
        result.best_score = -result.best_score;
        result.history = result.history.iter().map(|v| -v).collect();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==========================================================
    // EXTREME TDD: Tests written first
    // ==========================================================

    #[test]
    fn test_hyperopt_search_builder() {
        let search = HyperoptSearch::new()
            .add_real("lr", 1e-5, 1e-1, true)
            .add_int("epochs", 10, 100)
            .add_categorical("optimizer", &["sgd", "adam"])
            .with_seed(42);

        assert_eq!(search.parameters.len(), 3);
        assert!(search.seed.is_some());
    }

    #[test]
    fn test_hyperopt_real_parameter() {
        let search = HyperoptSearch::new()
            .add_real("x", 0.0, 10.0, false)
            .with_seed(42);

        let objective = |params: &HyperparameterSet| -> f64 {
            let x = params.get_real("x").unwrap();
            (x - 5.0).powi(2) // Minimum at x=5
        };

        let result = search.minimize(objective, Budget::Evaluations(200));

        assert!(result.best_params.get_real("x").is_some());
        let x = result.best_params.get_real("x").unwrap();
        assert!((x - 5.0).abs() < 1.0, "Expected x near 5, got {}", x);
    }

    #[test]
    fn test_hyperopt_log_scale() {
        let search = HyperoptSearch::new()
            .add_real("lr", 1e-5, 1e-1, true) // Log scale
            .with_seed(42);

        let objective = |params: &HyperparameterSet| -> f64 {
            let lr = params.get_real("lr").unwrap();
            // Optimal at lr = 0.001 (10^-3)
            (lr.log10() + 3.0).powi(2)
        };

        let result = search.minimize(objective, Budget::Evaluations(200));

        let lr = result.best_params.get_real("lr").unwrap();
        // Should find value near 0.001
        assert!(lr > 1e-5 && lr < 1e-1, "LR out of bounds: {}", lr);
    }

    #[test]
    fn test_hyperopt_int_parameter() {
        let search = HyperoptSearch::new().add_int("n", 1, 10).with_seed(42);

        let objective = |params: &HyperparameterSet| -> f64 {
            let n = params.get_int("n").unwrap();
            ((n - 7) as f64).powi(2) // Minimum at n=7
        };

        let result = search.minimize(objective, Budget::Evaluations(100));

        let n = result.best_params.get_int("n").unwrap();
        assert!(n >= 1 && n <= 10, "n out of bounds: {}", n);
    }

    #[test]
    fn test_hyperopt_categorical_parameter() {
        let search = HyperoptSearch::new()
            .add_categorical("color", &["red", "green", "blue"])
            .with_seed(42);

        let objective = |params: &HyperparameterSet| -> f64 {
            let color = params.get_categorical("color").unwrap();
            match color {
                "green" => 0.0, // Optimal choice
                "red" => 1.0,
                "blue" => 2.0,
                _ => 10.0,
            }
        };

        let result = search.minimize(objective, Budget::Evaluations(100));

        let color = result.best_params.get_categorical("color").unwrap();
        assert!(
            ["red", "green", "blue"].contains(&color),
            "Invalid color: {}",
            color
        );
    }

    #[test]
    fn test_hyperopt_maximize() {
        let search = HyperoptSearch::new()
            .add_real("x", 0.0, 10.0, false)
            .with_seed(42);

        let objective = |params: &HyperparameterSet| -> f64 {
            let x = params.get_real("x").unwrap();
            -(x - 5.0).powi(2) + 10.0 // Maximum at x=5
        };

        let result = search.maximize(objective, Budget::Evaluations(200));

        assert!(result.best_score > 9.0, "Score should be near 10");
    }

    #[test]
    fn test_hyperopt_mixed_space() {
        let search = HyperoptSearch::new()
            .add_real("lr", 0.001, 0.1, false)
            .add_int("hidden", 16, 256)
            .add_categorical("activation", &["relu", "tanh"])
            .with_seed(42);

        let objective = |params: &HyperparameterSet| -> f64 {
            let lr = params.get_real("lr").unwrap();
            let hidden = params.get_int("hidden").unwrap();
            let activation = params.get_categorical("activation").unwrap();

            let mut score = (lr - 0.01).powi(2);
            score += ((hidden - 64) as f64 / 100.0).powi(2);
            if activation == "relu" {
                score -= 0.1;
            }
            score
        };

        let result = search.minimize(objective, Budget::Evaluations(300));

        assert!(result.best_params.get_real("lr").is_some());
        assert!(result.best_params.get_int("hidden").is_some());
        assert!(result.best_params.get_categorical("activation").is_some());
    }

    #[test]
    fn test_hyperopt_result_structure() {
        let search = HyperoptSearch::new()
            .add_real("x", 0.0, 1.0, false)
            .with_seed(42);

        let objective = |_: &HyperparameterSet| 0.5;

        let result = search.minimize(objective, Budget::Evaluations(50));

        assert!(result.evaluations > 0);
        assert!(!result.history.is_empty());
    }

    #[test]
    fn test_hyperopt_different_algorithms() {
        let algorithms = [
            SearchAlgorithm::DifferentialEvolution,
            SearchAlgorithm::ParticleSwarm,
            SearchAlgorithm::SimulatedAnnealing,
        ];

        for algo in algorithms {
            let search = HyperoptSearch::new()
                .add_real("x", 0.0, 10.0, false)
                .with_algorithm(algo)
                .with_seed(42);

            let objective = |params: &HyperparameterSet| -> f64 {
                let x = params.get_real("x").unwrap();
                (x - 5.0).powi(2)
            };

            let result = search.minimize(objective, Budget::Evaluations(100));
            assert!(result.evaluations > 0, "Algorithm {:?} failed", algo);
        }
    }

    #[test]
    fn test_hyperparameter_set_operations() {
        let mut params = HyperparameterSet::new();

        params.set_real("lr", 0.01);
        params.set_int("epochs", 100);
        params.set_categorical("opt", "adam");

        assert_eq!(params.get_real("lr"), Some(0.01));
        assert_eq!(params.get_int("epochs"), Some(100));
        assert_eq!(params.get_categorical("opt"), Some("adam"));
        assert_eq!(params.get_real("missing"), None);
    }

    #[test]
    fn test_decode_consistency() {
        let search = HyperoptSearch::new()
            .add_real("x", 0.0, 1.0, false)
            .add_real("y", 10.0, 20.0, false);

        // Midpoint of [0,1] x [0,1] internal space
        let params = search.decode(&[0.5, 0.5]);

        let x = params.get_real("x").unwrap();
        let y = params.get_real("y").unwrap();

        assert!((x - 0.5).abs() < 0.01, "x should be 0.5, got {}", x);
        assert!((y - 15.0).abs() < 0.1, "y should be 15.0, got {}", y);
    }
}
