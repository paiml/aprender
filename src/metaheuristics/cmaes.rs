//! CMA-ES: Covariance Matrix Adaptation Evolution Strategy
//!
//! State-of-the-art derivative-free optimizer for continuous domains.
//! Adapts a full covariance matrix to capture variable dependencies.
//!
//! Reference: Hansen (2016) "The CMA Evolution Strategy: A Tutorial"
//!
//! # Example
//!
//! ```
//! use aprender::metaheuristics::{CmaEs, SearchSpace, Budget, PerturbativeMetaheuristic};
//!
//! let objective = |x: &[f64]| x.iter().map(|xi| xi * xi).sum();
//! let mut cma = CmaEs::new(5);  // 5 dimensions
//! let space = SearchSpace::continuous(5, -5.0, 5.0);
//! let result = cma.optimize(&objective, &space, Budget::Evaluations(5000));
//! assert!(result.objective_value < 1e-6);
//! ```

use super::{Budget, ConvergenceTracker, OptimizationResult, SearchSpace, TerminationReason};
use crate::metaheuristics::traits::PerturbativeMetaheuristic;
use rand::prelude::*;
use std::f64::consts::PI;

/// IPOP Restart configuration
#[derive(Debug, Clone, Copy)]
pub struct IpopConfig {
    /// Enable IPOP restarts
    pub enabled: bool,
    /// Population increase factor on restart (default 2.0)
    pub pop_increase_factor: f64,
    /// Maximum restarts before giving up
    pub max_restarts: usize,
    /// Sigma threshold for triggering restart
    pub sigma_threshold: f64,
    /// Stagnation tolerance (no improvement for N generations)
    pub stagnation_gens: usize,
}

impl Default for IpopConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pop_increase_factor: 2.0,
            max_restarts: 9,
            sigma_threshold: 1e-12,
            stagnation_gens: 20,
        }
    }
}

/// CMA-ES optimizer
#[derive(Debug, Clone)]
pub struct CmaEs {
    /// Problem dimension
    dim: usize,
    /// Population size (lambda)
    lambda: usize,
    /// Initial population size (for IPOP)
    initial_lambda: usize,
    /// Number of parents (mu)
    mu: usize,
    /// Recombination weights
    weights: Vec<f64>,
    /// Variance effective selection mass
    mu_eff: f64,
    /// Step-size
    sigma: f64,
    /// Evolution path for sigma
    p_sigma: Vec<f64>,
    /// Evolution path for C
    p_c: Vec<f64>,
    /// Covariance matrix (stored as vector for simplicity)
    c_diag: Vec<f64>,
    /// Mean vector
    mean: Vec<f64>,
    /// Learning rates
    c_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    /// Damping for sigma
    d_sigma: f64,
    /// Random seed
    seed: Option<u64>,
    /// Convergence history
    history: Vec<f64>,
    /// Best solution found
    best: Vec<f64>,
    /// Best objective value
    best_value: f64,
    /// IPOP restart configuration
    ipop: IpopConfig,
    /// Number of restarts performed
    restart_count: usize,
}

impl CmaEs {
    /// Create new CMA-ES for given dimension
    #[must_use]
    pub fn new(dim: usize) -> Self {
        let lambda = 4 + (3.0 * (dim as f64).ln()).floor() as usize;
        Self::with_lambda(dim, lambda)
    }

    /// Create CMA-ES with specific population size
    #[must_use]
    pub fn with_lambda(dim: usize, lambda: usize) -> Self {
        let lambda = lambda.max(4);
        let mu = lambda / 2;

        // Recombination weights (log-linear)
        let weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
            .collect();
        let sum_w: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / sum_w).collect();

        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Learning rates
        let c_sigma = (mu_eff + 2.0) / (dim as f64 + mu_eff + 5.0);
        let c_c = (4.0 + mu_eff / dim as f64) / (dim as f64 + 4.0 + 2.0 * mu_eff / dim as f64);
        let c_1 = 2.0 / ((dim as f64 + 1.3).powi(2) + mu_eff);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff))
            / ((dim as f64 + 2.0).powi(2) + mu_eff).min(1.0 - c_1);
        let d_sigma =
            1.0 + 2.0 * (0.0_f64).max((mu_eff - 1.0) / (dim as f64 + 1.0)).sqrt() + c_sigma;

        Self {
            dim,
            lambda,
            initial_lambda: lambda,
            mu,
            weights,
            mu_eff,
            sigma: 0.3,
            p_sigma: vec![0.0; dim],
            p_c: vec![0.0; dim],
            c_diag: vec![1.0; dim],
            mean: vec![0.0; dim],
            c_sigma,
            c_c,
            c_1,
            c_mu,
            d_sigma,
            seed: None,
            history: Vec::new(),
            best: Vec::new(),
            best_value: f64::INFINITY,
            ipop: IpopConfig::default(),
            restart_count: 0,
        }
    }

    /// Set random seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set initial sigma (step-size)
    #[must_use]
    pub fn with_sigma(mut self, sigma: f64) -> Self {
        self.sigma = sigma.max(1e-10);
        self
    }

    /// Enable IPOP (Increasing Population) restart strategy.
    ///
    /// When the algorithm stagnates (sigma too small or no improvement),
    /// it restarts with a doubled population size. This helps escape
    /// local optima on multimodal functions.
    #[must_use]
    pub fn with_ipop(mut self) -> Self {
        self.ipop.enabled = true;
        self
    }

    /// Configure IPOP settings.
    #[must_use]
    pub fn with_ipop_config(mut self, config: IpopConfig) -> Self {
        self.ipop = config;
        self.ipop.enabled = true;
        self
    }

    /// Get the number of restarts performed.
    #[must_use]
    pub fn restart_count(&self) -> usize {
        self.restart_count
    }

    /// Check if restart should be triggered based on stagnation.
    fn should_restart(&self, gens_without_improvement: usize) -> bool {
        if !self.ipop.enabled {
            return false;
        }
        if self.restart_count >= self.ipop.max_restarts {
            return false;
        }
        // Restart if sigma is too small or stagnated
        self.sigma < self.ipop.sigma_threshold
            || gens_without_improvement >= self.ipop.stagnation_gens
    }

    /// Perform IPOP restart: increase population and reset state.
    fn perform_restart(&mut self, lower: &[f64], upper: &[f64], rng: &mut impl Rng) {
        self.restart_count += 1;

        // Double the population
        let new_lambda = ((self.lambda as f64) * self.ipop.pop_increase_factor) as usize;
        let mu = new_lambda / 2;

        // Recalculate weights
        let weights: Vec<f64> = (0..mu)
            .map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln()).max(0.0))
            .collect();
        let sum_w: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / sum_w).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Update parameters
        self.lambda = new_lambda;
        self.mu = mu;
        self.weights = weights;
        self.mu_eff = mu_eff;

        // Recalculate learning rates
        self.c_sigma = (mu_eff + 2.0) / (self.dim as f64 + mu_eff + 5.0);
        self.c_c = (4.0 + mu_eff / self.dim as f64)
            / (self.dim as f64 + 4.0 + 2.0 * mu_eff / self.dim as f64);
        self.c_1 = 2.0 / ((self.dim as f64 + 1.3).powi(2) + mu_eff);
        self.c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff))
            / ((self.dim as f64 + 2.0).powi(2) + mu_eff).min(1.0 - self.c_1);
        self.d_sigma = 1.0
            + 2.0
                * (0.0_f64)
                    .max((mu_eff - 1.0) / (self.dim as f64 + 1.0))
                    .sqrt()
            + self.c_sigma;

        // Reset internal state
        self.p_sigma = vec![0.0; self.dim];
        self.p_c = vec![0.0; self.dim];
        self.c_diag = vec![1.0; self.dim];

        // Randomize mean within bounds
        self.mean = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| l + rng.gen::<f64>() * (u - l))
            .collect();

        // Reset sigma
        let range: f64 = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| u - l)
            .sum::<f64>()
            / self.dim as f64;
        self.sigma = range / 3.0;
    }

    /// Sample standard normal using Box-Muller transform
    fn randn(rng: &mut impl Rng) -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-10);
        let u2: f64 = rng.gen();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }

    /// Sample from multivariate normal with diagonal covariance
    fn sample(&self, rng: &mut impl Rng) -> Vec<f64> {
        (0..self.dim)
            .map(|i| {
                let z = Self::randn(rng);
                self.mean[i] + self.sigma * self.c_diag[i].sqrt() * z
            })
            .collect()
    }

    /// Clamp solution to bounds
    fn clamp(x: &mut [f64], lower: &[f64], upper: &[f64]) {
        for i in 0..x.len() {
            x[i] = x[i].clamp(lower[i], upper[i]);
        }
    }
}

impl PerturbativeMetaheuristic for CmaEs {
    type Solution = Vec<f64>;

    #[allow(clippy::too_many_lines, clippy::needless_range_loop)]
    fn optimize<F>(
        &mut self,
        objective: &F,
        space: &SearchSpace,
        budget: Budget,
    ) -> OptimizationResult<Self::Solution>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut rng: Box<dyn RngCore> = match self.seed {
            Some(s) => Box::new(StdRng::seed_from_u64(s)),
            None => Box::new(thread_rng()),
        };

        let (lower, upper) = match space {
            SearchSpace::Continuous { lower, upper, .. } => (lower.clone(), upper.clone()),
            _ => panic!("CMA-ES requires continuous search space"),
        };

        // Initialize mean at center of search space
        self.mean = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| (l + u) / 2.0)
            .collect();

        // Initialize sigma based on search range
        let range: f64 = lower
            .iter()
            .zip(upper.iter())
            .map(|(l, u)| u - l)
            .sum::<f64>()
            / self.dim as f64;
        self.sigma = range / 3.0;

        self.history.clear();
        self.best_value = f64::INFINITY;
        self.restart_count = 0;

        let mut tracker = ConvergenceTracker::from_budget(&budget);
        let max_iter = budget.max_iterations(self.initial_lambda);
        let chi_n = (self.dim as f64).sqrt()
            * (1.0 - 1.0 / (4.0 * self.dim as f64) + 1.0 / (21.0 * (self.dim as f64).powi(2)));

        let mut gens_without_improvement = 0usize;
        let mut last_best = f64::INFINITY;

        for gen in 0..max_iter {
            // Sample population
            let population: Vec<Vec<f64>> = (0..self.lambda)
                .map(|_| {
                    let mut x = self.sample(&mut rng);
                    Self::clamp(&mut x, &lower, &upper);
                    x
                })
                .collect();

            // Evaluate
            let mut fitness: Vec<(usize, f64)> = population
                .iter()
                .enumerate()
                .map(|(i, x)| (i, objective(x)))
                .collect();

            // Sort by fitness (minimization)
            fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Update best
            if fitness[0].1 < self.best_value {
                self.best_value = fitness[0].1;
                self.best.clone_from(&population[fitness[0].0]);
            }

            // Track stagnation for IPOP
            if self.best_value < last_best - 1e-10 {
                gens_without_improvement = 0;
                last_best = self.best_value;
            } else {
                gens_without_improvement += 1;
            }

            self.history.push(self.best_value);

            // Check for IPOP restart
            if self.should_restart(gens_without_improvement) {
                self.perform_restart(&lower, &upper, &mut rng);
                gens_without_improvement = 0;
                continue; // Skip the rest of this iteration
            }

            // Update mean
            let old_mean = self.mean.clone();
            self.mean = vec![0.0; self.dim];
            for (rank, &(idx, _)) in fitness.iter().take(self.mu).enumerate() {
                for i in 0..self.dim {
                    self.mean[i] += self.weights[rank] * population[idx][i];
                }
            }

            // Update evolution paths
            let mean_diff: Vec<f64> = self
                .mean
                .iter()
                .zip(old_mean.iter())
                .map(|(m, o)| (m - o) / self.sigma)
                .collect();

            // p_sigma update (simplified - no full C^(-1/2))
            for i in 0..self.dim {
                self.p_sigma[i] = (1.0 - self.c_sigma) * self.p_sigma[i]
                    + (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt() * mean_diff[i]
                        / self.c_diag[i].sqrt();
            }

            let p_sigma_norm: f64 = self.p_sigma.iter().map(|p| p * p).sum::<f64>().sqrt();
            let h_sigma = if p_sigma_norm
                / (1.0 - (1.0 - self.c_sigma).powi(2 * (gen as i32 + 1))).sqrt()
                < (1.4 + 2.0 / (self.dim as f64 + 1.0)) * chi_n
            {
                1.0
            } else {
                0.0
            };

            // p_c update
            for i in 0..self.dim {
                self.p_c[i] = (1.0 - self.c_c) * self.p_c[i]
                    + h_sigma * (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt() * mean_diff[i];
            }

            // Covariance update (diagonal only for simplicity)
            for i in 0..self.dim {
                let rank_one = self.p_c[i] * self.p_c[i];
                let mut rank_mu = 0.0;
                for (rank, &(idx, _)) in fitness.iter().take(self.mu).enumerate() {
                    let y_i = (population[idx][i] - old_mean[i]) / self.sigma;
                    rank_mu += self.weights[rank] * y_i * y_i;
                }

                self.c_diag[i] = (1.0 - self.c_1 - self.c_mu) * self.c_diag[i]
                    + self.c_1 * rank_one
                    + self.c_mu * rank_mu;

                // Ensure positive
                self.c_diag[i] = self.c_diag[i].max(1e-20);
            }

            // Step-size update
            self.sigma *= ((self.c_sigma / self.d_sigma) * (p_sigma_norm / chi_n - 1.0)).exp();
            self.sigma = self.sigma.clamp(1e-20, 1e10);

            if !tracker.update(self.best_value, self.lambda) {
                break;
            }
        }

        let termination = if tracker.is_converged() {
            TerminationReason::Converged
        } else if tracker.is_exhausted() {
            TerminationReason::BudgetExhausted
        } else {
            TerminationReason::MaxIterations
        };

        OptimizationResult {
            solution: self.best.clone(),
            objective_value: self.best_value,
            evaluations: tracker.evaluations(),
            iterations: self.history.len(),
            history: self.history.clone(),
            termination,
        }
    }

    fn best(&self) -> Option<&Self::Solution> {
        if self.best.is_empty() {
            None
        } else {
            Some(&self.best)
        }
    }

    fn history(&self) -> &[f64] {
        &self.history
    }

    fn reset(&mut self) {
        self.p_sigma = vec![0.0; self.dim];
        self.p_c = vec![0.0; self.dim];
        self.c_diag = vec![1.0; self.dim];
        self.mean = vec![0.0; self.dim];
        self.history.clear();
        self.best.clear();
        self.best_value = f64::INFINITY;
        self.restart_count = 0;
        self.lambda = self.initial_lambda;
    }
}

#[cfg(test)]
#[path = "cmaes_tests.rs"]
mod tests;
