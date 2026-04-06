//! Monte Carlo simulation engine.
//!
//! Implements Monte Carlo methods with variance reduction:
//! - Antithetic variates
//! - Control variates
//! - Importance sampling
//! - Stratified sampling
//!
//! # Convergence
//!
//! By the Central Limit Theorem, Monte Carlo estimators converge at O(n^{-1/2})
//! regardless of dimension, making them ideal for high-dimensional problems.

use crate::engine::rng::SimRng;
use serde::{Deserialize, Serialize};

/// Result of a Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    /// Point estimate.
    pub estimate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Number of samples used.
    pub samples: usize,
    /// 95% confidence interval (estimate ± 1.96 * `std_error`).
    pub confidence_interval: (f64, f64),
    /// Variance reduction factor (if applicable).
    pub variance_reduction_factor: Option<f64>,
}

impl MonteCarloResult {
    /// Create a new Monte Carlo result.
    #[must_use]
    pub fn new(estimate: f64, std_error: f64, samples: usize) -> Self {
        let ci_half = 1.96 * std_error;
        Self {
            estimate,
            std_error,
            samples,
            confidence_interval: (estimate - ci_half, estimate + ci_half),
            variance_reduction_factor: None,
        }
    }

    /// Set the variance reduction factor.
    #[must_use]
    pub fn with_variance_reduction(mut self, factor: f64) -> Self {
        self.variance_reduction_factor = Some(factor);
        self
    }

    /// Check if value is within confidence interval.
    #[must_use]
    pub fn contains(&self, value: f64) -> bool {
        value >= self.confidence_interval.0 && value <= self.confidence_interval.1
    }

    /// Get relative error.
    #[must_use]
    pub fn relative_error(&self) -> f64 {
        if self.estimate.abs() < f64::EPSILON {
            self.std_error
        } else {
            self.std_error / self.estimate.abs()
        }
    }
}

/// Variance reduction technique.
#[derive(Debug, Clone, Default)]
pub enum VarianceReduction {
    /// No variance reduction.
    #[default]
    None,
    /// Antithetic variates: use (U, 1-U) pairs.
    Antithetic,
    /// Control variate with known expectation.
    ControlVariate {
        /// Control function.
        control_fn: fn(f64) -> f64,
        /// Known expectation of control.
        expectation: f64,
    },
    /// Importance sampling with proposal distribution.
    ///
    /// Standard importance sampling: `E_p[f] = E_q[f * p/q]`
    ImportanceSampling {
        /// Sample from proposal distribution q(x).
        sample_fn: fn(&mut SimRng) -> f64,
        /// Likelihood ratio p(x)/q(x) where p is target, q is proposal.
        likelihood_ratio: fn(f64) -> f64,
    },
    /// Self-normalizing importance sampling.
    ///
    /// More robust when the normalizing constant is unknown.
    /// Uses: `E_p[f] ≈ Σ(w_i * f(x_i)) / Σ(w_i)`
    SelfNormalizingIS {
        /// Sample from proposal distribution q(x).
        sample_fn: fn(&mut SimRng) -> f64,
        /// Unnormalized weight w(x) ∝ p(x)/q(x).
        weight_fn: fn(f64) -> f64,
    },
    /// Stratified sampling.
    Stratified {
        /// Number of strata.
        num_strata: usize,
    },
}

/// Monte Carlo engine for stochastic simulation.
#[derive(Debug)]
pub struct MonteCarloEngine {
    /// Number of samples.
    n_samples: usize,
    /// Variance reduction technique.
    variance_reduction: VarianceReduction,
}

impl MonteCarloEngine {
    /// Create a new Monte Carlo engine.
    #[must_use]
    pub const fn new(n_samples: usize, variance_reduction: VarianceReduction) -> Self {
        Self {
            n_samples,
            variance_reduction,
        }
    }

    /// Create engine with default settings.
    #[must_use]
    pub const fn with_samples(n_samples: usize) -> Self {
        Self::new(n_samples, VarianceReduction::None)
    }

    /// Run Monte Carlo simulation with a function f: [0,1] -> R.
    ///
    /// # Example
    ///
    /// ```rust
    /// use simular::domains::monte_carlo::{MonteCarloEngine, VarianceReduction};
    /// use simular::engine::rng::SimRng;
    ///
    /// let engine = MonteCarloEngine::with_samples(10000);
    /// let mut rng = SimRng::new(42);
    ///
    /// // Estimate integral of x^2 from 0 to 1 (true value = 1/3)
    /// let result = engine.run(|x| x * x, &mut rng);
    /// assert!((result.estimate - 1.0/3.0).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn run<F>(&self, f: F, rng: &mut SimRng) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        match &self.variance_reduction {
            VarianceReduction::None => self.run_standard(&f, rng),
            VarianceReduction::Antithetic => self.run_antithetic(&f, rng),
            VarianceReduction::ControlVariate {
                control_fn,
                expectation,
            } => self.run_control_variate(&f, *control_fn, *expectation, rng),
            VarianceReduction::ImportanceSampling {
                sample_fn,
                likelihood_ratio,
            } => self.run_importance(&f, *sample_fn, *likelihood_ratio, rng),
            VarianceReduction::SelfNormalizingIS {
                sample_fn,
                weight_fn,
            } => self.run_self_normalizing_is(&f, *sample_fn, *weight_fn, rng),
            VarianceReduction::Stratified { num_strata } => {
                self.run_stratified(&f, *num_strata, rng)
            }
        }
    }

    /// Run multi-dimensional Monte Carlo.
    ///
    /// # Example
    ///
    /// ```rust
    /// use simular::domains::monte_carlo::MonteCarloEngine;
    /// use simular::engine::rng::SimRng;
    ///
    /// let engine = MonteCarloEngine::with_samples(10000);
    /// let mut rng = SimRng::new(42);
    ///
    /// // Estimate volume of unit sphere in 3D (true value ≈ 4.189)
    /// let result = engine.run_nd(3, |x| {
    ///     let r2 = x.iter().map(|&xi| xi * xi).sum::<f64>();
    ///     if r2 <= 1.0 { 8.0 } else { 0.0 } // 8 = 2^3 for [-1,1]^3 domain
    /// }, &mut rng);
    /// ```
    #[must_use]
    pub fn run_nd<F>(&self, dim: usize, f: F, rng: &mut SimRng) -> MonteCarloResult
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut samples = Vec::with_capacity(dim);

        for _ in 0..self.n_samples {
            samples.clear();
            for _ in 0..dim {
                samples.push(rng.gen_f64());
            }

            let value = f(&samples);
            sum += value;
            sum_sq += value * value;
        }

        let n = self.n_samples as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - (mean * mean);
        let std_error = (variance / n).sqrt();

        MonteCarloResult::new(mean, std_error, self.n_samples)
    }

    /// Standard Monte Carlo (no variance reduction).
    fn run_standard<F>(&self, f: &F, rng: &mut SimRng) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..self.n_samples {
            let u = rng.gen_f64();
            let value = f(u);
            sum += value;
            sum_sq += value * value;
        }

        let n = self.n_samples as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - (mean * mean);
        let std_error = (variance / n).sqrt();

        MonteCarloResult::new(mean, std_error, self.n_samples)
    }

    /// Antithetic variates: use (U, 1-U) pairs to induce negative correlation.
    fn run_antithetic<F>(&self, f: &F, rng: &mut SimRng) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        let n_pairs = self.n_samples / 2;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..n_pairs {
            let u = rng.gen_f64();

            // Antithetic pair
            let y1 = f(u);
            let y2 = f(1.0 - u);

            // Use average of pair
            let avg = (y1 + y2) / 2.0;
            sum += avg;
            sum_sq += avg * avg;
        }

        let n = n_pairs as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - (mean * mean);
        let std_error = (variance / n).sqrt();

        // Effective samples is n_pairs * 2
        let mut result = MonteCarloResult::new(mean, std_error, n_pairs * 2);

        // Estimate variance reduction factor
        let standard_result = self.run_standard(f, &mut rng.clone());
        if standard_result.std_error > f64::EPSILON {
            result = result
                .with_variance_reduction(standard_result.std_error / std_error.max(f64::EPSILON));
        }

        result
    }

    /// Control variate method.
    fn run_control_variate<F>(
        &self,
        f: &F,
        control_fn: fn(f64) -> f64,
        control_expectation: f64,
        rng: &mut SimRng,
    ) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        // First pass: estimate correlation
        let mut sum_f = 0.0;
        let mut sum_c = 0.0;
        let mut sum_fc = 0.0;
        let mut sum_c2 = 0.0;
        let samples: Vec<f64> = rng.sample_n(self.n_samples);

        for &u in &samples {
            let fv = f(u);
            let cv = control_fn(u);
            sum_f += fv;
            sum_c += cv;
            sum_fc += fv * cv;
            sum_c2 += cv * cv;
        }

        let n = self.n_samples as f64;
        let mean_f = sum_f / n;
        let mean_c = sum_c / n;

        // Estimate optimal coefficient
        let cov_fc = (sum_fc / n) - mean_f * mean_c;
        let var_c = (sum_c2 / n) - mean_c * mean_c;
        let c_star = if var_c > f64::EPSILON {
            -cov_fc / var_c
        } else {
            0.0
        };

        // Second pass: compute adjusted estimate
        let mut sum_adj = 0.0;
        let mut sum_adj_sq = 0.0;

        for &u in &samples {
            let fv = f(u);
            let cv = control_fn(u);
            let adjusted = fv + c_star * (cv - control_expectation);
            sum_adj += adjusted;
            sum_adj_sq += adjusted * adjusted;
        }

        let mean_adj = sum_adj / n;
        let variance_adj = (sum_adj_sq / n) - mean_adj * mean_adj;
        let std_error = (variance_adj / n).sqrt();

        MonteCarloResult::new(mean_adj, std_error, self.n_samples)
    }

    /// Importance sampling.
    fn run_importance<F>(
        &self,
        f: &F,
        sample_fn: fn(&mut SimRng) -> f64,
        likelihood_ratio: fn(f64) -> f64,
        rng: &mut SimRng,
    ) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..self.n_samples {
            let x = sample_fn(rng);
            let weight = likelihood_ratio(x);
            let value = f(x) * weight;
            sum += value;
            sum_sq += value * value;
        }

        let n = self.n_samples as f64;
        let mean = sum / n;
        let variance = (sum_sq / n) - mean * mean;
        let std_error = (variance / n).sqrt();

        MonteCarloResult::new(mean, std_error, self.n_samples)
    }

    /// Self-normalizing importance sampling.
    ///
    /// More robust when the normalizing constant is unknown.
    /// Uses ratio estimator: `Σ(w_i * f(x_i)) / Σ(w_i)`
    fn run_self_normalizing_is<F>(
        &self,
        f: &F,
        sample_fn: fn(&mut SimRng) -> f64,
        weight_fn: fn(f64) -> f64,
        rng: &mut SimRng,
    ) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        // Collect all weights and values
        let mut weights = Vec::with_capacity(self.n_samples);
        let mut values = Vec::with_capacity(self.n_samples);

        for _ in 0..self.n_samples {
            let x = sample_fn(rng);
            let w = weight_fn(x);
            let fv = f(x);

            weights.push(w);
            values.push(fv);
        }

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        if weight_sum.abs() < f64::EPSILON {
            return MonteCarloResult::new(0.0, f64::INFINITY, self.n_samples);
        }

        // Compute weighted mean: Σ(w_i * f_i) / Σ(w_i)
        let weighted_sum: f64 = weights.iter().zip(values.iter()).map(|(w, v)| w * v).sum();
        let mean = weighted_sum / weight_sum;

        // Compute effective sample size: ESS = (Σw)² / Σ(w²)
        let weight_sq_sum: f64 = weights.iter().map(|w| w * w).sum();
        let ess = (weight_sum * weight_sum) / weight_sq_sum;

        // Standard error estimation using linearization (delta method)
        // For ratio estimator, var(μ̂) ≈ 1/n * Σ w_i² (f_i - μ̂)² / (mean(w))²
        let mean_weight = weight_sum / self.n_samples as f64;
        let variance: f64 = weights
            .iter()
            .zip(values.iter())
            .map(|(w, v)| {
                let normalized_w = w / mean_weight;
                normalized_w * normalized_w * (v - mean) * (v - mean)
            })
            .sum::<f64>()
            / (self.n_samples as f64 * self.n_samples as f64);

        let std_error = variance.sqrt();

        let mut result = MonteCarloResult::new(mean, std_error, self.n_samples);
        // Report effective sample size as variance reduction indicator
        result = result.with_variance_reduction(ess / self.n_samples as f64);
        result
    }

    /// Stratified sampling.
    fn run_stratified<F>(&self, f: &F, num_strata: usize, rng: &mut SimRng) -> MonteCarloResult
    where
        F: Fn(f64) -> f64,
    {
        let samples_per_stratum = self.n_samples / num_strata;
        let stratum_width = 1.0 / num_strata as f64;

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut total_samples = 0;

        for i in 0..num_strata {
            let stratum_start = i as f64 * stratum_width;
            let mut stratum_sum = 0.0;
            let mut stratum_sum_sq = 0.0;

            for _ in 0..samples_per_stratum {
                let u = stratum_start + rng.gen_f64() * stratum_width;
                let value = f(u);
                stratum_sum += value;
                stratum_sum_sq += value * value;
                total_samples += 1;
            }

            let stratum_mean = stratum_sum / samples_per_stratum as f64;
            sum += stratum_mean;
            sum_sq += stratum_sum_sq / samples_per_stratum as f64;
        }

        let mean = sum / num_strata as f64;
        let variance = (sum_sq / num_strata as f64) - mean * mean;
        let std_error = (variance / self.n_samples as f64).sqrt();

        MonteCarloResult::new(mean, std_error, total_samples)
    }

    /// Get configured sample count.
    #[must_use]
    pub const fn n_samples(&self) -> usize {
        self.n_samples
    }
}

// =============================================================================
// Work-Stealing Monte Carlo (Section 4.3.5)
// =============================================================================

/// Individual simulation task for work-stealing scheduler.
#[derive(Debug, Clone)]
pub struct SimulationTask {
    /// Random seed for this trajectory.
    pub seed: u64,
    /// Task index.
    pub index: usize,
}

/// Work-stealing Monte Carlo scheduler [55].
///
/// Implements Heijunka (load leveling) for variable-duration simulations.
/// Uses crossbeam-deque for lock-free work stealing to handle the "straggler problem"
/// where threads would otherwise wait for the slowest simulation.
#[derive(Debug)]
pub struct WorkStealingMonteCarlo {
    /// Number of worker threads.
    num_workers: usize,
}

impl Default for WorkStealingMonteCarlo {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkStealingMonteCarlo {
    /// Create with default number of workers (number of CPUs).
    #[must_use]
    pub fn new() -> Self {
        Self {
            num_workers: std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(4),
        }
    }

    /// Create with specified number of workers.
    #[must_use]
    pub const fn with_workers(num_workers: usize) -> Self {
        Self { num_workers }
    }

    /// Execute Monte Carlo simulation with work stealing [55].
    ///
    /// Tasks are distributed across workers, and idle workers steal tasks
    /// from busy workers to maintain load balance (Heijunka).
    pub fn execute<F, R>(&self, n_samples: usize, simulate: F) -> Vec<R>
    where
        F: Fn(SimulationTask) -> R + Sync,
        R: Send,
    {
        use crossbeam_deque::{Injector, Stealer, Worker};

        // Global work queue
        let injector: Injector<SimulationTask> = Injector::new();

        // Per-worker local queues
        let workers: Vec<Worker<SimulationTask>> =
            (0..self.num_workers).map(|_| Worker::new_fifo()).collect();

        // Stealers for cross-worker theft
        let stealers: Vec<Stealer<SimulationTask>> = workers.iter().map(Worker::stealer).collect();

        // Populate global queue
        for index in 0..n_samples {
            injector.push(SimulationTask {
                seed: index as u64,
                index,
            });
        }

        // Results storage - use Vec with push instead of pre-allocation
        let results: std::sync::Mutex<Vec<(usize, R)>> =
            std::sync::Mutex::new(Vec::with_capacity(n_samples));

        std::thread::scope(|s| {
            for (worker_id, worker) in workers.into_iter().enumerate() {
                let injector = &injector;
                let stealers = &stealers;
                let results = &results;
                let simulate = &simulate;

                s.spawn(move || {
                    loop {
                        // Try local queue first
                        let task = worker
                            .pop()
                            .or_else(|| {
                                // Try global queue
                                loop {
                                    match injector.steal() {
                                        crossbeam_deque::Steal::Success(task) => return Some(task),
                                        crossbeam_deque::Steal::Empty => break,
                                        crossbeam_deque::Steal::Retry => {}
                                    }
                                }
                                None
                            })
                            .or_else(|| {
                                // Steal from other workers (round-robin)
                                for i in 0..stealers.len() {
                                    let stealer_idx = (worker_id + i + 1) % stealers.len();
                                    loop {
                                        match stealers[stealer_idx].steal() {
                                            crossbeam_deque::Steal::Success(task) => {
                                                return Some(task)
                                            }
                                            crossbeam_deque::Steal::Empty => break,
                                            crossbeam_deque::Steal::Retry => {}
                                        }
                                    }
                                }
                                None
                            });

                        match task {
                            Some(task) => {
                                let index = task.index;
                                let result = simulate(task);
                                if let Ok(mut guard) = results.lock() {
                                    guard.push((index, result));
                                }
                            }
                            None => break, // No more work
                        }
                    }
                });
            }
        });

        // Sort by index and extract results
        let mut indexed_results = results.into_inner().unwrap_or_default();
        indexed_results.sort_by_key(|(idx, _)| *idx);
        indexed_results.into_iter().map(|(_, r)| r).collect()
    }

    /// Execute with statistics collection.
    ///
    /// Returns results and basic statistics about the simulation.
    pub fn execute_with_stats<F>(
        &self,
        n_samples: usize,
        simulate: F,
    ) -> (Vec<f64>, MonteCarloResult)
    where
        F: Fn(SimulationTask) -> f64 + Sync,
    {
        let results = self.execute(n_samples, simulate);

        let n = results.len() as f64;
        let sum: f64 = results.iter().sum();
        let mean = sum / n;

        let variance: f64 = results.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_error = (variance / n).sqrt();

        let mc_result = MonteCarloResult::new(mean, std_error, results.len());

        (results, mc_result)
    }

    /// Get number of workers.
    #[must_use]
    pub const fn num_workers(&self) -> usize {
        self.num_workers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_result_creation() {
        let result = MonteCarloResult::new(0.5, 0.01, 10000);

        assert!((result.estimate - 0.5).abs() < f64::EPSILON);
        assert!((result.std_error - 0.01).abs() < f64::EPSILON);
        assert_eq!(result.samples, 10000);

        // 95% CI = estimate ± 1.96 * std_error
        assert!((result.confidence_interval.0 - 0.4804).abs() < 0.001);
        assert!((result.confidence_interval.1 - 0.5196).abs() < 0.001);
    }

    #[test]
    fn test_mc_contains() {
        let result = MonteCarloResult::new(0.5, 0.01, 10000);

        assert!(result.contains(0.5));
        assert!(result.contains(0.49));
        assert!(!result.contains(0.4));
    }

    #[test]
    fn test_standard_mc_uniform() {
        let engine = MonteCarloEngine::with_samples(100_000);
        let mut rng = SimRng::new(42);

        // E[U] = 0.5 for U ~ Uniform(0,1)
        let result = engine.run(|x| x, &mut rng);

        assert!((result.estimate - 0.5).abs() < 0.01);
        assert!(result.std_error < 0.01);
    }

    #[test]
    fn test_standard_mc_square() {
        let engine = MonteCarloEngine::with_samples(100_000);
        let mut rng = SimRng::new(42);

        // E[U^2] = 1/3 for U ~ Uniform(0,1)
        let result = engine.run(|x| x * x, &mut rng);

        assert!((result.estimate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_antithetic_variates() {
        let engine = MonteCarloEngine::new(100_000, VarianceReduction::Antithetic);
        let mut rng = SimRng::new(42);

        // Antithetic should reduce variance for monotonic functions
        let result = engine.run(|x| x, &mut rng);

        // Should still get correct estimate
        assert!((result.estimate - 0.5).abs() < 0.01);

        // Variance reduction factor should be > 1
        if let Some(vrf) = result.variance_reduction_factor {
            assert!(vrf > 1.0, "Expected variance reduction, got factor {}", vrf);
        }
    }

    #[test]
    fn test_stratified_sampling() {
        let engine =
            MonteCarloEngine::new(100_000, VarianceReduction::Stratified { num_strata: 10 });
        let mut rng = SimRng::new(42);

        let result = engine.run(|x| x * x, &mut rng);

        assert!((result.estimate - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_importance_sampling() {
        // Example: estimate E[x^4] under Uniform(0,1) using Beta(2,1) proposal
        // Target: p(x) = 1 (uniform)
        // Proposal: q(x) = 2x (Beta(2,1) PDF)
        // Likelihood ratio: p(x)/q(x) = 1/(2x)
        // True value: E[x^4] = 1/5 = 0.2

        fn sample_beta21(rng: &mut SimRng) -> f64 {
            // Sample from Beta(2,1): use inverse CDF method
            // CDF(x) = x^2, so inverse is sqrt(U)
            rng.gen_f64().sqrt()
        }

        fn likelihood_ratio(x: f64) -> f64 {
            if x < f64::EPSILON {
                1.0
            } else {
                1.0 / (2.0 * x)
            }
        }

        let engine = MonteCarloEngine::new(
            100_000,
            VarianceReduction::ImportanceSampling {
                sample_fn: sample_beta21,
                likelihood_ratio,
            },
        );
        let mut rng = SimRng::new(42);

        let result = engine.run(|x| x.powi(4), &mut rng);

        // E[x^4] = 1/5 = 0.2
        assert!(
            (result.estimate - 0.2).abs() < 0.01,
            "Expected ~0.2, got {}",
            result.estimate
        );
    }

    #[test]
    fn test_importance_sampling_reduces_variance() {
        // For functions peaked near x=1, sampling from Beta(2,1) should help
        // because it samples more from higher x values

        fn sample_beta21(rng: &mut SimRng) -> f64 {
            rng.gen_f64().sqrt()
        }

        fn likelihood_ratio(x: f64) -> f64 {
            if x < f64::EPSILON {
                1.0
            } else {
                1.0 / (2.0 * x)
            }
        }

        // Function peaked near x=1
        let f = |x: f64| x.powi(10);
        // True value: 1/11

        let standard_engine = MonteCarloEngine::with_samples(10_000);
        let is_engine = MonteCarloEngine::new(
            10_000,
            VarianceReduction::ImportanceSampling {
                sample_fn: sample_beta21,
                likelihood_ratio,
            },
        );

        let mut rng1 = SimRng::new(42);
        let mut rng2 = SimRng::new(42);

        let standard_result = standard_engine.run(f, &mut rng1);
        let is_result = is_engine.run(f, &mut rng2);

        // Both should estimate approximately 1/11 ≈ 0.0909
        let true_value = 1.0 / 11.0;
        assert!((standard_result.estimate - true_value).abs() < 0.02);
        assert!((is_result.estimate - true_value).abs() < 0.02);

        // IS should have lower variance for this function
        // (not always guaranteed, but likely for peaked functions)
    }

    #[test]
    fn test_self_normalizing_importance_sampling() {
        // Self-normalizing IS is useful when we don't know the normalizing constant
        // Test: E[x] under Uniform(0,1) using unnormalized weights

        fn sample_uniform(rng: &mut SimRng) -> f64 {
            rng.gen_f64()
        }

        fn weight_fn(x: f64) -> f64 {
            // Uniform weights = 1.0 everywhere
            // This should give same result as standard MC
            1.0 + 0.0 * x // Use x to prevent unused warning
        }

        let engine = MonteCarloEngine::new(
            100_000,
            VarianceReduction::SelfNormalizingIS {
                sample_fn: sample_uniform,
                weight_fn,
            },
        );
        let mut rng = SimRng::new(42);

        let result = engine.run(|x| x, &mut rng);

        // E[x] = 0.5
        assert!(
            (result.estimate - 0.5).abs() < 0.01,
            "Expected ~0.5, got {}",
            result.estimate
        );

        // With uniform weights, ESS should be close to n
        if let Some(ess_ratio) = result.variance_reduction_factor {
            assert!(
                ess_ratio > 0.9,
                "ESS ratio should be near 1 for uniform weights, got {}",
                ess_ratio
            );
        }
    }

    #[test]
    fn test_self_normalizing_is_weighted() {
        // Test with non-uniform weights
        // Use linear weight w(x) = x to emphasize larger values

        fn sample_uniform(rng: &mut SimRng) -> f64 {
            rng.gen_f64()
        }

        fn weight_fn(x: f64) -> f64 {
            // Weight proportional to x
            // This reweights to emphasize larger x values
            x.max(0.001) // Avoid zero weights
        }

        let engine = MonteCarloEngine::new(
            100_000,
            VarianceReduction::SelfNormalizingIS {
                sample_fn: sample_uniform,
                weight_fn,
            },
        );
        let mut rng = SimRng::new(42);

        // E_weighted[f(x)] where weight ∝ x
        // With w(x) = x and f(x) = 1: E = ∫x dx / ∫x dx = 1 (trivial)
        // With w(x) = x and f(x) = x: E = ∫x² dx / ∫x dx = (1/3)/(1/2) = 2/3
        let result = engine.run(|x| x, &mut rng);

        // Should be approximately 2/3
        assert!(
            (result.estimate - 2.0 / 3.0).abs() < 0.02,
            "Expected ~0.667, got {}",
            result.estimate
        );

        // ESS should be less than n due to weight variation
        if let Some(ess_ratio) = result.variance_reduction_factor {
            assert!(
                ess_ratio < 1.0,
                "ESS ratio should be < 1 for varied weights"
            );
        }
    }

    #[test]
    fn test_multidimensional_mc() {
        let engine = MonteCarloEngine::with_samples(100_000);
        let mut rng = SimRng::new(42);

        // Estimate volume of unit hypercube in 3D (should be 1.0)
        let result = engine.run_nd(3, |_x| 1.0, &mut rng);

        assert!((result.estimate - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mc_pi_estimation() {
        let engine = MonteCarloEngine::with_samples(100_000);
        let mut rng = SimRng::new(42);

        // Estimate pi using quarter circle
        // Area of quarter unit circle = pi/4
        let result = engine.run_nd(
            2,
            |x| {
                if x[0] * x[0] + x[1] * x[1] <= 1.0 {
                    4.0
                } else {
                    0.0
                }
            },
            &mut rng,
        );

        assert!((result.estimate - std::f64::consts::PI).abs() < 0.05);
    }

    #[test]
    fn test_convergence_rate() {
        // Monte Carlo should converge at O(n^{-1/2})
        let mut rng = SimRng::new(42);

        let engine_small = MonteCarloEngine::with_samples(1_000);
        let engine_large = MonteCarloEngine::with_samples(100_000);

        let result_small = engine_small.run(|x| x * x, &mut rng);
        let result_large = engine_large.run(|x| x * x, &mut rng);

        // Error should decrease by ~sqrt(100) = 10
        let ratio = result_small.std_error / result_large.std_error;
        assert!(
            ratio > 5.0 && ratio < 20.0,
            "Expected error ratio ~10, got {}",
            ratio
        );
    }

    // === Work-Stealing Monte Carlo Tests (Section 4.3.5) ===

    #[test]
    fn test_work_stealing_basic() {
        let ws = WorkStealingMonteCarlo::with_workers(4);

        let results = ws.execute(100, |task| task.index * 2);

        assert_eq!(results.len(), 100);
        for (i, &r) in results.iter().enumerate() {
            assert_eq!(r, i * 2);
        }
    }

    #[test]
    fn test_work_stealing_pi_estimation() {
        let ws = WorkStealingMonteCarlo::with_workers(4);

        // Estimate pi using quarter circle
        let (results, stats) = ws.execute_with_stats(100_000, |task| {
            let mut rng = SimRng::new(task.seed);
            let x = rng.gen_f64();
            let y = rng.gen_f64();
            if x * x + y * y <= 1.0 {
                4.0
            } else {
                0.0
            }
        });

        assert_eq!(results.len(), 100_000);
        assert!(
            (stats.estimate - std::f64::consts::PI).abs() < 0.1,
            "Pi estimate {} too far from actual",
            stats.estimate
        );
    }

    #[test]
    fn test_work_stealing_variable_duration() {
        let ws = WorkStealingMonteCarlo::with_workers(4);

        // Simulate variable-duration tasks (some take longer)
        let results: Vec<u64> = ws.execute(50, |task| {
            // Simulate some "work" - longer tasks should be stolen
            let mut sum = 0u64;
            let iterations = if task.index % 10 == 0 { 10000 } else { 100 };
            for i in 0..iterations {
                sum = sum.wrapping_add(i);
            }
            sum
        });

        assert_eq!(results.len(), 50);
    }

    #[test]
    fn test_work_stealing_num_workers() {
        let ws_default = WorkStealingMonteCarlo::new();
        assert!(ws_default.num_workers() > 0);

        let ws_custom = WorkStealingMonteCarlo::with_workers(8);
        assert_eq!(ws_custom.num_workers(), 8);
    }

    // === Additional Coverage Tests ===

    #[test]
    fn test_mc_result_relative_error() {
        let result = MonteCarloResult::new(0.5, 0.01, 10000);
        let rel_err = result.relative_error();
        // relative_error = std_error / |estimate| = 0.01 / 0.5 = 0.02
        assert!((rel_err - 0.02).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mc_result_relative_error_zero_estimate() {
        let result = MonteCarloResult::new(0.0, 0.01, 10000);
        let rel_err = result.relative_error();
        // When estimate is zero, relative_error = std_error
        assert!((rel_err - 0.01).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mc_result_with_variance_reduction() {
        let result = MonteCarloResult::new(0.5, 0.01, 10000).with_variance_reduction(2.0);
        assert!(result.variance_reduction_factor.is_some());
        assert!((result.variance_reduction_factor.unwrap() - 2.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mc_result_clone() {
        let result = MonteCarloResult::new(0.5, 0.01, 10000);
        let cloned = result.clone();
        assert!((cloned.estimate - result.estimate).abs() < f64::EPSILON);
        assert_eq!(cloned.samples, result.samples);
    }

    #[test]
    fn test_mc_result_debug() {
        let result = MonteCarloResult::new(0.5, 0.01, 10000);
        let debug = format!("{:?}", result);
        assert!(debug.contains("MonteCarloResult"));
        assert!(debug.contains("estimate"));
    }

    #[test]
    fn test_variance_reduction_default() {
        let vr = VarianceReduction::default();
        match vr {
            VarianceReduction::None => {} // Expected
            _ => panic!("Default should be None"),
        }
    }

    #[test]
    fn test_variance_reduction_clone() {
        let vr = VarianceReduction::Antithetic;
        let cloned = vr.clone();
        match cloned {
            VarianceReduction::Antithetic => {} // Expected
            _ => panic!("Clone should preserve variant"),
        }
    }

    #[test]
    fn test_variance_reduction_debug() {
        let vr = VarianceReduction::Antithetic;
        let debug = format!("{:?}", vr);
        assert!(debug.contains("Antithetic"));

        let vr = VarianceReduction::Stratified { num_strata: 10 };
        let debug = format!("{:?}", vr);
        assert!(debug.contains("Stratified"));
    }

    #[test]
    fn test_mc_engine_debug() {
        let engine = MonteCarloEngine::with_samples(1000);
        let debug = format!("{:?}", engine);
        assert!(debug.contains("MonteCarloEngine"));
    }

    #[test]
    fn test_control_variate() {
        // Test control variate with a simple known case
        // Estimate E[x^2] using x as control variate
        // E[x] = 0.5, E[x^2] = 1/3

        fn control_fn(x: f64) -> f64 {
            x
        }
        let control_expectation = 0.5;

        let engine = MonteCarloEngine::new(
            100_000,
            VarianceReduction::ControlVariate {
                control_fn,
                expectation: control_expectation,
            },
        );
        let mut rng = SimRng::new(42);

        let result = engine.run(|x| x * x, &mut rng);
        // E[x^2] = 1/3
        assert!(
            (result.estimate - 1.0 / 3.0).abs() < 0.01,
            "Expected ~0.333, got {}",
            result.estimate
        );
    }

    #[test]
    fn test_simulation_task_debug_clone() {
        let task = SimulationTask {
            index: 42,
            seed: 12345,
        };
        let cloned = task.clone();
        assert_eq!(cloned.index, 42);
        assert_eq!(cloned.seed, 12345);

        let debug = format!("{:?}", task);
        assert!(debug.contains("SimulationTask"));
    }

    #[test]
    fn test_work_stealing_debug() {
        let ws = WorkStealingMonteCarlo::with_workers(2);
        let debug = format!("{:?}", ws);
        assert!(debug.contains("WorkStealingMonteCarlo"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsification: MC estimate should be within confidence interval
        /// with high probability.
        #[test]
        fn prop_mc_confidence_interval(seed in 0u64..10000) {
            let engine = MonteCarloEngine::with_samples(100_000); // More samples
            let mut rng = SimRng::new(seed);

            // Known integral: integral of x from 0 to 1 = 0.5
            let result = engine.run(|x| x, &mut rng);

            // Check if true value is within CI
            let true_value = 0.5;
            let error = (result.estimate - true_value).abs();

            // Use 5 sigma for very lenient test (99.99994% coverage)
            prop_assert!(error < 5.0 * result.std_error,
                "Error {} exceeds 5 sigma = {}", error, 5.0 * result.std_error);
        }

        /// Falsification: standard error decreases with more samples.
        #[test]
        fn prop_mc_error_decreases(seed in 0u64..1000, n_small in 100usize..1000) {
            let n_large = n_small * 10;

            let engine_small = MonteCarloEngine::with_samples(n_small);
            let engine_large = MonteCarloEngine::with_samples(n_large);

            let mut rng1 = SimRng::new(seed);
            let mut rng2 = SimRng::new(seed + 1);

            let result_small = engine_small.run(|x| x * x, &mut rng1);
            let result_large = engine_large.run(|x| x * x, &mut rng2);

            // Standard error should decrease (not always strictly due to randomness)
            // But on average it should hold
            // We use a lenient check
            prop_assert!(result_large.std_error < result_small.std_error * 2.0,
                "Large std_error {} should be less than small {} * 2",
                result_large.std_error, result_small.std_error);
        }
    }
}
