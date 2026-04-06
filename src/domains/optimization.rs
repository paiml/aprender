//! Optimization engine with Bayesian optimization.
//!
//! Implements Bayesian optimization using Gaussian Process surrogates
//! for sample-efficient black-box optimization (Kaizen: continuous improvement).
//!
//! # Acquisition Functions
//!
//! - **Expected Improvement (EI)**: Balances exploration and exploitation
//! - **Upper Confidence Bound (UCB)**: Tunable exploration via kappa
//! - **Probability of Improvement (PI)**: Conservative improvement strategy
//!
//! # Example
//!
//! ```rust
//! use simular::domains::optimization::{BayesianOptimizer, OptimizerConfig, AcquisitionFunction};
//!
//! let config = OptimizerConfig {
//!     bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
//!     acquisition: AcquisitionFunction::ExpectedImprovement,
//!     ..Default::default()
//! };
//!
//! let mut optimizer = BayesianOptimizer::new(config);
//!
//! // Add initial observations
//! optimizer.observe(vec![0.0, 0.0], 1.5);
//! optimizer.observe(vec![1.0, 1.0], 0.8);
//!
//! // Get next suggested point
//! let next_point = optimizer.suggest();
//! ```

use crate::engine::rng::SimRng;
use crate::error::{SimError, SimResult};
use serde::{Deserialize, Serialize};

/// Acquisition functions for Bayesian optimization.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum AcquisitionFunction {
    /// Expected Improvement - balances exploration/exploitation.
    #[default]
    ExpectedImprovement,
    /// Upper Confidence Bound with exploration parameter kappa.
    UCB { kappa: f64 },
    /// Probability of Improvement - conservative strategy.
    ProbabilityOfImprovement,
}

/// Configuration for Bayesian optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Parameter bounds: (min, max) for each dimension.
    pub bounds: Vec<(f64, f64)>,
    /// Acquisition function to use.
    pub acquisition: AcquisitionFunction,
    /// Length scale for RBF kernel.
    pub length_scale: f64,
    /// Signal variance for GP.
    pub signal_variance: f64,
    /// Noise variance (observation noise).
    pub noise_variance: f64,
    /// Number of random samples for acquisition optimization.
    pub n_candidates: usize,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            bounds: vec![(-1.0, 1.0)],
            acquisition: AcquisitionFunction::default(),
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 1e-6,
            n_candidates: 1000,
            seed: 42,
        }
    }
}

/// Gaussian Process surrogate model.
///
/// Uses RBF (Radial Basis Function) kernel for smooth function approximation.
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Training inputs.
    x_train: Vec<Vec<f64>>,
    /// Training outputs.
    y_train: Vec<f64>,
    /// Length scale parameter.
    length_scale: f64,
    /// Signal variance.
    signal_variance: f64,
    /// Noise variance.
    noise_variance: f64,
    /// Cached inverse covariance matrix (for efficiency).
    k_inv_y: Option<Vec<f64>>,
    /// Cached Cholesky decomposition.
    l_matrix: Option<Vec<Vec<f64>>>,
}

impl GaussianProcess {
    /// Create a new Gaussian Process.
    #[must_use]
    pub fn new(length_scale: f64, signal_variance: f64, noise_variance: f64) -> Self {
        Self {
            x_train: Vec::new(),
            y_train: Vec::new(),
            length_scale,
            signal_variance,
            noise_variance,
            k_inv_y: None,
            l_matrix: None,
        }
    }

    /// Add training data point.
    pub fn add_observation(&mut self, x: Vec<f64>, y: f64) {
        self.x_train.push(x);
        self.y_train.push(y);
        // Invalidate cache
        self.k_inv_y = None;
        self.l_matrix = None;
    }

    /// Fit the GP to training data.
    ///
    /// # Errors
    ///
    /// Returns error if Cholesky decomposition fails.
    pub fn fit(&mut self) -> SimResult<()> {
        if self.x_train.is_empty() {
            return Ok(());
        }

        let n = self.x_train.len();

        // Compute covariance matrix K + noise*I
        let mut k_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                k_matrix[i][j] = self.rbf_kernel(&self.x_train[i], &self.x_train[j]);
                if i == j {
                    k_matrix[i][j] += self.noise_variance;
                }
            }
        }

        // Cholesky decomposition: K = L * L^T
        let l = Self::cholesky(&k_matrix)?;

        // Solve L * alpha = y, then L^T * alpha = alpha
        let alpha = Self::solve_triangular(&l, &self.y_train, false);
        let k_inv_y = Self::solve_triangular(&l, &alpha, true);

        self.l_matrix = Some(l);
        self.k_inv_y = Some(k_inv_y);

        Ok(())
    }

    /// Predict mean and variance at a point.
    #[must_use]
    pub fn predict(&self, x: &[f64]) -> (f64, f64) {
        if self.x_train.is_empty() {
            return (0.0, self.signal_variance);
        }

        let Some(k_inv_y) = &self.k_inv_y else {
            return (0.0, self.signal_variance);
        };

        let Some(l) = &self.l_matrix else {
            return (0.0, self.signal_variance);
        };

        // k* = kernel between x and training points
        let k_star: Vec<f64> = self
            .x_train
            .iter()
            .map(|xi| self.rbf_kernel(xi, x))
            .collect();

        // Mean: mu = k*^T * K^{-1} * y
        let mu: f64 = k_star.iter().zip(k_inv_y.iter()).map(|(k, a)| k * a).sum();

        // Variance: sigma^2 = k(x, x) - k*^T * K^{-1} * k*
        let k_xx = self.rbf_kernel(x, x);

        // Solve L * v = k*
        let v = Self::solve_triangular(l, &k_star, false);
        let variance = k_xx - v.iter().map(|vi| vi * vi).sum::<f64>();

        (mu, variance.max(1e-10))
    }

    /// RBF (squared exponential) kernel.
    fn rbf_kernel(&self, x1: &[f64], x2: &[f64]) -> f64 {
        let sq_dist: f64 = x1.iter().zip(x2.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        self.signal_variance * (-sq_dist / (2.0 * self.length_scale.powi(2))).exp()
    }

    /// Cholesky decomposition (lower triangular).
    fn cholesky(matrix: &[Vec<f64>]) -> SimResult<Vec<Vec<f64>>> {
        let n = matrix.len();
        let mut l = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }

                if i == j {
                    let val = matrix[i][i] - sum;
                    if val <= 0.0 {
                        return Err(SimError::optimization(
                            "Cholesky decomposition failed: matrix not positive definite",
                        ));
                    }
                    l[i][j] = val.sqrt();
                } else {
                    l[i][j] = (matrix[i][j] - sum) / l[j][j];
                }
            }
        }

        Ok(l)
    }

    /// Solve triangular system.
    fn solve_triangular(l: &[Vec<f64>], b: &[f64], transpose: bool) -> Vec<f64> {
        let n = b.len();
        let mut x = vec![0.0; n];

        if transpose {
            // Solve L^T * x = b (backward substitution)
            for i in (0..n).rev() {
                let mut sum = b[i];
                for j in (i + 1)..n {
                    sum -= l[j][i] * x[j];
                }
                x[i] = sum / l[i][i];
            }
        } else {
            // Solve L * x = b (forward substitution)
            for i in 0..n {
                let mut sum = b[i];
                for j in 0..i {
                    sum -= l[i][j] * x[j];
                }
                x[i] = sum / l[i][i];
            }
        }

        x
    }

    /// Get number of training points.
    #[must_use]
    pub fn n_observations(&self) -> usize {
        self.x_train.len()
    }
}

/// Bayesian optimizer using Gaussian Process surrogate.
#[derive(Debug)]
pub struct BayesianOptimizer {
    /// Configuration.
    config: OptimizerConfig,
    /// GP surrogate model.
    gp: GaussianProcess,
    /// RNG for candidate sampling.
    rng: SimRng,
    /// Best observed value.
    best_y: Option<f64>,
    /// Best observed point.
    best_x: Option<Vec<f64>>,
}

impl BayesianOptimizer {
    /// Create a new Bayesian optimizer.
    #[must_use]
    pub fn new(config: OptimizerConfig) -> Self {
        let gp = GaussianProcess::new(
            config.length_scale,
            config.signal_variance,
            config.noise_variance,
        );
        let rng = SimRng::new(config.seed);

        Self {
            config,
            gp,
            rng,
            best_y: None,
            best_x: None,
        }
    }

    /// Add an observation (x, y) pair.
    ///
    /// # Errors
    ///
    /// Returns error if GP fitting fails.
    pub fn observe(&mut self, x: Vec<f64>, y: f64) -> SimResult<()> {
        self.gp.add_observation(x.clone(), y);

        // Update best
        if self.best_y.is_none() || y < self.best_y.unwrap_or(f64::INFINITY) {
            self.best_y = Some(y);
            self.best_x = Some(x);
        }

        // Refit GP
        self.gp.fit()
    }

    /// Suggest the next point to evaluate (Kaizen: continuous improvement).
    ///
    /// Uses random candidate optimization of the acquisition function.
    #[must_use]
    pub fn suggest(&mut self) -> Vec<f64> {
        if self.gp.n_observations() == 0 {
            // No observations yet - sample random point
            return self.random_point();
        }

        // Generate candidates and evaluate acquisition function
        let mut best_acq = f64::NEG_INFINITY;
        let mut best_candidate = self.random_point();

        for _ in 0..self.config.n_candidates {
            let candidate = self.random_point();
            let acq_value = self.evaluate_acquisition(&candidate);

            if acq_value > best_acq {
                best_acq = acq_value;
                best_candidate = candidate;
            }
        }

        best_candidate
    }

    /// Evaluate acquisition function at a point.
    fn evaluate_acquisition(&self, x: &[f64]) -> f64 {
        let (mu, variance) = self.gp.predict(x);
        let sigma = variance.sqrt();

        match self.config.acquisition {
            AcquisitionFunction::ExpectedImprovement => self.expected_improvement(mu, sigma),
            AcquisitionFunction::UCB { kappa } => Self::upper_confidence_bound(mu, sigma, kappa),
            AcquisitionFunction::ProbabilityOfImprovement => {
                self.probability_of_improvement(mu, sigma)
            }
        }
    }

    /// Expected Improvement acquisition function.
    fn expected_improvement(&self, mu: f64, sigma: f64) -> f64 {
        let best = self.best_y.unwrap_or(0.0);

        if sigma < 1e-10 {
            return 0.0;
        }

        // Note: we're minimizing, so improvement is best - mu
        let z = (best - mu) / sigma;
        let pdf = Self::normal_pdf(z);
        let cdf = Self::normal_cdf(z);

        // EI is mathematically non-negative; use max to handle floating point precision
        (sigma * (z * cdf + pdf)).max(0.0)
    }

    /// Upper Confidence Bound acquisition function.
    fn upper_confidence_bound(mu: f64, sigma: f64, kappa: f64) -> f64 {
        // For minimization: lower confidence bound
        -mu + kappa * sigma
    }

    /// Probability of Improvement acquisition function.
    fn probability_of_improvement(&self, mu: f64, sigma: f64) -> f64 {
        let best = self.best_y.unwrap_or(0.0);

        if sigma < 1e-10 {
            return if mu < best { 1.0 } else { 0.0 };
        }

        let z = (best - mu) / sigma;
        Self::normal_cdf(z)
    }

    /// Standard normal PDF.
    fn normal_pdf(z: f64) -> f64 {
        const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7; // 1 / sqrt(2 * pi)
        INV_SQRT_2PI * (-0.5 * z * z).exp()
    }

    /// Standard normal CDF (approximation).
    fn normal_cdf(z: f64) -> f64 {
        // Abramowitz and Stegun approximation
        const A1: f64 = 0.254_829_592;
        const A2: f64 = -0.284_496_736;
        const A3: f64 = 1.421_413_741;
        const A4: f64 = -1.453_152_027;
        const A5: f64 = 1.061_405_429;
        const P: f64 = 0.327_591_1;

        let sign = if z < 0.0 { -1.0 } else { 1.0 };
        let z_abs = z.abs();
        let t = 1.0 / (1.0 + P * z_abs);
        let y = 1.0
            - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-z_abs * z_abs / 2.0).exp();

        0.5 * (1.0 + sign * y)
    }

    /// Generate a random point within bounds.
    fn random_point(&mut self) -> Vec<f64> {
        self.config
            .bounds
            .iter()
            .map(|(min, max)| self.rng.gen_range_f64(*min, *max))
            .collect()
    }

    /// Get the best observed point.
    #[must_use]
    pub fn best(&self) -> Option<(&[f64], f64)> {
        match (&self.best_x, self.best_y) {
            (Some(x), Some(y)) => Some((x.as_slice(), y)),
            _ => None,
        }
    }

    /// Get number of observations.
    #[must_use]
    pub fn n_observations(&self) -> usize {
        self.gp.n_observations()
    }

    /// Get the configuration.
    #[must_use]
    pub const fn config(&self) -> &OptimizerConfig {
        &self.config
    }
}

/// Result of optimization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Best found point.
    pub best_x: Vec<f64>,
    /// Best found value.
    pub best_y: f64,
    /// Number of function evaluations.
    pub n_evaluations: usize,
    /// History of (x, y) observations.
    pub history: Vec<(Vec<f64>, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gp_create() {
        let gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        assert_eq!(gp.n_observations(), 0);
    }

    #[test]
    fn test_gp_add_observation() {
        let mut gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        gp.add_observation(vec![0.0], 1.0);
        gp.add_observation(vec![1.0], 2.0);
        assert_eq!(gp.n_observations(), 2);
    }

    #[test]
    fn test_gp_fit() {
        let mut gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        gp.add_observation(vec![0.0], 0.0);
        gp.add_observation(vec![1.0], 1.0);
        gp.add_observation(vec![2.0], 4.0);

        assert!(gp.fit().is_ok());
    }

    #[test]
    fn test_gp_predict_empty() {
        let gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        let (mu, var) = gp.predict(&[0.5]);

        assert!((mu - 0.0).abs() < f64::EPSILON);
        assert!((var - 1.0).abs() < f64::EPSILON); // signal_variance
    }

    #[test]
    fn test_gp_predict_interpolation() {
        let mut gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        gp.add_observation(vec![0.0], 0.0);
        gp.add_observation(vec![1.0], 1.0);
        gp.fit().ok();

        // Predict at training points - should be close to observations
        let (mu0, _) = gp.predict(&[0.0]);
        let (mu1, _) = gp.predict(&[1.0]);

        assert!((mu0 - 0.0).abs() < 0.1);
        assert!((mu1 - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_gp_variance_decreases_near_data() {
        let mut gp = GaussianProcess::new(1.0, 1.0, 1e-6);
        gp.add_observation(vec![0.0], 0.0);
        gp.fit().ok();

        let (_, var_near) = gp.predict(&[0.0]);
        let (_, var_far) = gp.predict(&[5.0]);

        assert!(
            var_near < var_far,
            "Variance should be lower near observations"
        );
    }

    #[test]
    fn test_optimizer_create() {
        let config = OptimizerConfig::default();
        let optimizer = BayesianOptimizer::new(config);
        assert_eq!(optimizer.n_observations(), 0);
    }

    #[test]
    fn test_optimizer_observe() {
        let config = OptimizerConfig {
            bounds: vec![(-5.0, 5.0)],
            ..Default::default()
        };
        let mut optimizer = BayesianOptimizer::new(config);

        optimizer.observe(vec![0.0], 1.0).ok();
        optimizer.observe(vec![1.0], 0.5).ok();

        assert_eq!(optimizer.n_observations(), 2);

        let (best_x, best_y) = optimizer.best().expect("Should have best");
        assert!((best_y - 0.5).abs() < f64::EPSILON);
        assert!((best_x[0] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_optimizer_suggest_empty() {
        let config = OptimizerConfig {
            bounds: vec![(-5.0, 5.0)],
            ..Default::default()
        };
        let mut optimizer = BayesianOptimizer::new(config);

        // Should return random point within bounds
        let suggestion = optimizer.suggest();
        assert_eq!(suggestion.len(), 1);
        assert!(suggestion[0] >= -5.0 && suggestion[0] <= 5.0);
    }

    #[test]
    fn test_optimizer_suggest_with_observations() {
        let config = OptimizerConfig {
            bounds: vec![(-5.0, 5.0)],
            n_candidates: 100,
            ..Default::default()
        };
        let mut optimizer = BayesianOptimizer::new(config);

        optimizer.observe(vec![0.0], 1.0).ok();
        optimizer.observe(vec![1.0], 0.5).ok();
        optimizer.observe(vec![-1.0], 1.5).ok();

        let suggestion = optimizer.suggest();
        assert_eq!(suggestion.len(), 1);
        assert!(suggestion[0] >= -5.0 && suggestion[0] <= 5.0);
    }

    #[test]
    fn test_optimizer_multidimensional() {
        let config = OptimizerConfig {
            bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
            n_candidates: 100,
            ..Default::default()
        };
        let mut optimizer = BayesianOptimizer::new(config);

        optimizer.observe(vec![0.0, 0.0], 1.0).ok();
        optimizer.observe(vec![1.0, 1.0], 0.5).ok();

        let suggestion = optimizer.suggest();
        assert_eq!(suggestion.len(), 2);
    }

    #[test]
    fn test_acquisition_ei() {
        let config = OptimizerConfig {
            bounds: vec![(-5.0, 5.0)],
            acquisition: AcquisitionFunction::ExpectedImprovement,
            ..Default::default()
        };
        let optimizer = BayesianOptimizer::new(config);

        // Test that EI is positive for uncertain regions
        let ei = optimizer.expected_improvement(0.0, 1.0);
        assert!(ei > 0.0);

        // Test that EI is zero for zero variance
        let ei_zero_var = optimizer.expected_improvement(0.0, 0.0);
        assert!((ei_zero_var - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acquisition_ucb() {
        // UCB = -mu + kappa * sigma
        let ucb = BayesianOptimizer::upper_confidence_bound(1.0, 0.5, 2.0);
        let expected = -1.0 + 2.0 * 0.5;
        assert!((ucb - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn test_acquisition_pi() {
        let config = OptimizerConfig {
            bounds: vec![(-5.0, 5.0)],
            acquisition: AcquisitionFunction::ProbabilityOfImprovement,
            ..Default::default()
        };
        let mut optimizer = BayesianOptimizer::new(config);
        optimizer.best_y = Some(1.0);

        // PI should be 0.5 when mu = best
        let pi = optimizer.probability_of_improvement(1.0, 1.0);
        assert!((pi - 0.5).abs() < 0.01);

        // PI should be high when mu << best
        let pi_good = optimizer.probability_of_improvement(-1.0, 1.0);
        assert!(pi_good > 0.9);

        // PI should be low when mu >> best
        let pi_bad = optimizer.probability_of_improvement(3.0, 1.0);
        assert!(pi_bad < 0.1);
    }

    #[test]
    fn test_normal_pdf() {
        // PDF at z=0 should be ~0.399
        let pdf_0 = BayesianOptimizer::normal_pdf(0.0);
        assert!((pdf_0 - 0.3989).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        // CDF at z=0 should be 0.5
        let cdf_0 = BayesianOptimizer::normal_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < 0.01);

        // CDF at z=-3 should be very small
        let cdf_neg3 = BayesianOptimizer::normal_cdf(-3.0);
        assert!(cdf_neg3 < 0.01);

        // CDF at z=3 should be very large
        let cdf_pos3 = BayesianOptimizer::normal_cdf(3.0);
        assert!(cdf_pos3 > 0.99);
    }

    #[test]
    fn test_cholesky() {
        // Simple 2x2 positive definite matrix
        let matrix = vec![vec![4.0, 2.0], vec![2.0, 5.0]];

        let l = GaussianProcess::cholesky(&matrix).expect("Should succeed");

        // Verify L * L^T = matrix
        let reconstructed_00 = l[0][0] * l[0][0];
        let reconstructed_01 = l[1][0] * l[0][0];
        let reconstructed_11 = l[1][0] * l[1][0] + l[1][1] * l[1][1];

        assert!((reconstructed_00 - 4.0).abs() < 1e-10);
        assert!((reconstructed_01 - 2.0).abs() < 1e-10);
        assert!((reconstructed_11 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimizer_finds_minimum() {
        // Test on simple quadratic f(x) = (x - 2)^2
        let config = OptimizerConfig {
            bounds: vec![(-5.0, 5.0)],
            n_candidates: 500,
            seed: 42,
            ..Default::default()
        };
        let mut optimizer = BayesianOptimizer::new(config);

        // Initial random samples
        for x in [-4.0_f64, -2.0, 0.0, 2.0, 4.0] {
            let y = (x - 2.0).powi(2);
            optimizer.observe(vec![x], y).ok();
        }

        // Run optimization
        for _ in 0..20 {
            let suggestion = optimizer.suggest();
            let y = (suggestion[0] - 2.0).powi(2);
            optimizer.observe(suggestion, y).ok();
        }

        let (best_x, best_y) = optimizer.best().expect("Should have best");
        assert!(best_y < 0.1, "Should find near-minimum, got y={}", best_y);
        assert!(
            (best_x[0] - 2.0).abs() < 0.5,
            "Should find x near 2, got x={}",
            best_x[0]
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsification: GP variance is always non-negative.
        #[test]
        fn prop_gp_variance_nonnegative(
            x in -10.0f64..10.0,
            obs_x in prop::collection::vec(-10.0f64..10.0, 1..5),
            obs_y in prop::collection::vec(-10.0f64..10.0, 1..5),
        ) {
            if obs_x.len() != obs_y.len() {
                return Ok(());
            }

            let mut gp = GaussianProcess::new(1.0, 1.0, 1e-4);
            for (xi, yi) in obs_x.iter().zip(obs_y.iter()) {
                gp.add_observation(vec![*xi], *yi);
            }
            gp.fit().ok();

            let (_, var) = gp.predict(&[x]);
            prop_assert!(var >= 0.0, "Variance must be non-negative");
        }

        /// Falsification: suggestions are always within bounds.
        #[test]
        fn prop_suggest_within_bounds(
            min in -100.0f64..0.0,
            max in 0.0f64..100.0,
            seed in 0u64..10000,
        ) {
            let config = OptimizerConfig {
                bounds: vec![(min, max)],
                seed,
                n_candidates: 10,
                ..Default::default()
            };
            let mut optimizer = BayesianOptimizer::new(config);

            // Add some observations
            optimizer.observe(vec![(min + max) / 2.0], 1.0).ok();

            let suggestion = optimizer.suggest();
            prop_assert!(suggestion[0] >= min && suggestion[0] <= max);
        }

        /// Falsification: normal CDF is monotonic.
        #[test]
        fn prop_normal_cdf_monotonic(z1 in -5.0f64..5.0, z2 in -5.0f64..5.0) {
            let cdf1 = BayesianOptimizer::normal_cdf(z1);
            let cdf2 = BayesianOptimizer::normal_cdf(z2);

            if z1 < z2 {
                prop_assert!(cdf1 <= cdf2 + 1e-10, "CDF should be monotonic");
            }
        }

        /// Falsification: EI is non-negative.
        #[test]
        fn prop_ei_nonnegative(mu in -10.0f64..10.0, sigma in 0.01f64..10.0) {
            let config = OptimizerConfig::default();
            let mut optimizer = BayesianOptimizer::new(config);
            optimizer.best_y = Some(0.0);

            let ei = optimizer.expected_improvement(mu, sigma);
            prop_assert!(ei >= 0.0, "EI must be non-negative");
        }
    }
}
