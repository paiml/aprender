//! Reproducible random number generation for Monte Carlo simulations
//!
//! Uses `ChaCha20` PRNG for cryptographic-quality randomness with
//! explicit seeding for reproducibility.
//!
//! Reference: Bernstein (2008), "`ChaCha`, a variant of Salsa20"

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

/// Reproducible random number generator for Monte Carlo simulations
///
/// Wraps `ChaCha20Rng` with convenient methods for generating
/// various distributions needed in financial simulations.
///
/// # Example
/// ```
/// use aprender::monte_carlo::engine::MonteCarloRng;
///
/// let mut rng = MonteCarloRng::new(42);
///
/// // Same seed always produces same sequence
/// let u1 = rng.uniform();
/// rng.reset();
/// let u2 = rng.uniform();
/// assert!((u1 - u2).abs() < 1e-15);
/// ```
#[derive(Debug, Clone)]
pub struct MonteCarloRng {
    rng: ChaCha20Rng,
    seed: u64,
}

impl MonteCarloRng {
    /// Create a new RNG with the specified seed
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            rng: ChaCha20Rng::seed_from_u64(seed),
            seed,
        }
    }

    /// Get the seed used to initialize this RNG
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Reset the RNG to its initial state
    pub fn reset(&mut self) {
        self.rng = ChaCha20Rng::seed_from_u64(self.seed);
    }

    /// Generate a uniform random number in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        self.rng.random()
    }

    /// Generate a uniform random number in [low, high)
    pub fn uniform_range(&mut self, low: f64, high: f64) -> f64 {
        low + (high - low) * self.uniform()
    }

    /// Generate a standard normal random variable (mean=0, std=1)
    ///
    /// Uses the Box-Muller transform for efficiency.
    pub fn standard_normal(&mut self) -> f64 {
        // Box-Muller transform
        let u1: f64 = self.rng.random();
        let u2: f64 = self.rng.random();

        // Avoid log(0)
        let u1 = u1.max(1e-15_f64);

        (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos()
    }

    /// Generate a normal random variable with specified mean and std
    pub fn normal(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.standard_normal()
    }

    /// Generate a log-normal random variable
    ///
    /// If X ~ LogNormal(μ, σ), then log(X) ~ Normal(μ, σ)
    pub fn log_normal(&mut self, mean: f64, std: f64) -> f64 {
        (mean + std * self.standard_normal()).exp()
    }

    /// Generate an exponential random variable
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        let u: f64 = self.rng.random();
        -u.max(1e-15).ln() / lambda
    }

    /// Generate a Poisson random variable
    ///
    /// Uses inverse transform for small lambda, normal approximation for large.
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda < 30.0 {
            // Inverse transform method
            let l = (-lambda).exp();
            let mut k = 0u64;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= self.uniform();
                if p <= l {
                    break;
                }
            }
            k - 1
        } else {
            // Normal approximation for large lambda
            let x = self.normal(lambda, lambda.sqrt());
            x.max(0.0).round() as u64
        }
    }

    /// Generate a multivariate normal random vector
    ///
    /// Uses Cholesky decomposition of covariance matrix.
    ///
    /// # Arguments
    /// * `mean` - Mean vector
    /// * `cholesky_l` - Lower triangular Cholesky factor of covariance matrix
    #[must_use]
    pub fn multivariate_normal(&mut self, mean: &[f64], cholesky_l: &[Vec<f64>]) -> Vec<f64> {
        let n = mean.len();
        let z: Vec<f64> = (0..n).map(|_| self.standard_normal()).collect();

        // X = μ + L * Z
        let mut result = vec![0.0; n];
        for i in 0..n {
            result[i] = mean[i];
            for j in 0..=i {
                result[i] += cholesky_l[i][j] * z[j];
            }
        }

        result
    }

    /// Generate antithetic uniform pair
    ///
    /// Returns (U, 1-U) for variance reduction.
    pub fn antithetic_uniform(&mut self) -> (f64, f64) {
        let u = self.uniform();
        (u, 1.0 - u)
    }

    /// Generate antithetic normal pair
    ///
    /// Returns (Z, -Z) for variance reduction.
    pub fn antithetic_normal(&mut self) -> (f64, f64) {
        let z = self.standard_normal();
        (z, -z)
    }

    /// Generate gamma-distributed random variable
    ///
    /// Uses Marsaglia and Tsang's method for alpha >= 1,
    /// and transformation for alpha < 1.
    pub fn gamma(&mut self, alpha: f64, beta: f64) -> f64 {
        if alpha < 1.0 {
            // Use transformation: Gamma(alpha) = Gamma(1+alpha) * U^(1/alpha)
            let u = self.uniform();
            self.gamma(1.0 + alpha, 1.0) * u.powf(1.0 / alpha) / beta
        } else {
            // Marsaglia and Tsang's method
            let d = alpha - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();

            loop {
                let x = self.standard_normal();
                let v = (1.0 + c * x).powi(3);

                if v > 0.0 {
                    let u = self.uniform();
                    let x2 = x * x;

                    if u < 1.0 - 0.0331 * x2 * x2 {
                        return d * v / beta;
                    }

                    if u.ln() < 0.5 * x2 + d * (1.0 - v + v.ln()) {
                        return d * v / beta;
                    }
                }
            }
        }
    }

    /// Generate beta-distributed random variable
    pub fn beta(&mut self, alpha: f64, beta: f64) -> f64 {
        let x = self.gamma(alpha, 1.0);
        let y = self.gamma(beta, 1.0);
        x / (x + y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducibility() {
        let mut rng1 = MonteCarloRng::new(42);
        let mut rng2 = MonteCarloRng::new(42);

        for _ in 0..100 {
            assert!((rng1.uniform() - rng2.uniform()).abs() < 1e-15);
        }
    }

    #[test]
    fn test_reset() {
        let mut rng = MonteCarloRng::new(42);
        let first: Vec<f64> = (0..10).map(|_| rng.uniform()).collect();

        rng.reset();
        let second: Vec<f64> = (0..10).map(|_| rng.uniform()).collect();

        assert_eq!(first, second);
    }

    #[test]
    fn test_uniform_range() {
        let mut rng = MonteCarloRng::new(42);
        for _ in 0..1000 {
            let u = rng.uniform();
            assert!(u >= 0.0 && u < 1.0);
        }
    }

    #[test]
    fn test_uniform_custom_range() {
        let mut rng = MonteCarloRng::new(42);
        for _ in 0..1000 {
            let u = rng.uniform_range(10.0, 20.0);
            assert!(u >= 10.0 && u < 20.0);
        }
    }

    #[test]
    fn test_normal_statistics() {
        let mut rng = MonteCarloRng::new(42);
        let n = 10_000;
        let samples: Vec<f64> = (0..n).map(|_| rng.standard_normal()).collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        // Mean should be close to 0
        assert!(mean.abs() < 0.05, "Mean = {mean}");

        // Variance should be close to 1
        assert!((variance - 1.0).abs() < 0.1, "Variance = {variance}");
    }

    #[test]
    fn test_normal_custom() {
        let mut rng = MonteCarloRng::new(42);
        let n = 10_000;
        let mean_target = 5.0;
        let std_target = 2.0;

        let samples: Vec<f64> = (0..n)
            .map(|_| rng.normal(mean_target, std_target))
            .collect();

        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        assert!((mean - mean_target).abs() < 0.1, "Mean = {mean}");
        assert!(
            (variance.sqrt() - std_target).abs() < 0.15,
            "Std = {}",
            variance.sqrt()
        );
    }

    #[test]
    fn test_log_normal() {
        let mut rng = MonteCarloRng::new(42);
        for _ in 0..1000 {
            let x = rng.log_normal(0.0, 1.0);
            assert!(x > 0.0, "Log-normal must be positive");
        }
    }

    #[test]
    fn test_exponential() {
        let mut rng = MonteCarloRng::new(42);
        let lambda = 2.0;
        let n = 10_000;

        let samples: Vec<f64> = (0..n).map(|_| rng.exponential(lambda)).collect();
        let mean = samples.iter().sum::<f64>() / n as f64;

        // Mean of exponential should be 1/lambda
        assert!(
            (mean - 1.0 / lambda).abs() < 0.05,
            "Mean = {mean}, expected {}",
            1.0 / lambda
        );
    }

    #[test]
    fn test_poisson() {
        let mut rng = MonteCarloRng::new(42);
        let lambda = 5.0;
        let n = 10_000;

        let samples: Vec<u64> = (0..n).map(|_| rng.poisson(lambda)).collect();
        let mean = samples.iter().sum::<u64>() as f64 / n as f64;

        // Mean of Poisson should equal lambda
        assert!(
            (mean - lambda).abs() < 0.2,
            "Mean = {mean}, expected {lambda}"
        );
    }

    #[test]
    fn test_antithetic_uniform() {
        let mut rng = MonteCarloRng::new(42);
        for _ in 0..100 {
            let (u1, u2) = rng.antithetic_uniform();
            assert!((u1 + u2 - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_antithetic_normal() {
        let mut rng = MonteCarloRng::new(42);
        for _ in 0..100 {
            let (z1, z2) = rng.antithetic_normal();
            assert!((z1 + z2).abs() < 1e-15);
        }
    }

    #[test]
    fn test_multivariate_normal() {
        let mut rng = MonteCarloRng::new(42);
        let mean = vec![0.0, 0.0];
        // Identity covariance: L = I
        let cholesky_l = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let sample = rng.multivariate_normal(&mean, &cholesky_l);
        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_gamma() {
        let mut rng = MonteCarloRng::new(42);

        // Test with alpha >= 1
        for _ in 0..100 {
            let x = rng.gamma(2.0, 1.0);
            assert!(x > 0.0);
        }

        // Test with alpha < 1
        for _ in 0..100 {
            let x = rng.gamma(0.5, 1.0);
            assert!(x > 0.0);
        }
    }

    #[test]
    fn test_beta() {
        let mut rng = MonteCarloRng::new(42);
        for _ in 0..100 {
            let x = rng.beta(2.0, 5.0);
            assert!(x >= 0.0 && x <= 1.0);
        }
    }

    #[test]
    fn test_seed_getter() {
        let rng = MonteCarloRng::new(12345);
        assert_eq!(rng.seed(), 12345);
    }

    // Property-based tests
    #[cfg(test)]
    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #[test]
            fn prop_uniform_in_range(seed: u64) {
                let mut rng = MonteCarloRng::new(seed);
                for _ in 0..100 {
                    let u = rng.uniform();
                    prop_assert!(u >= 0.0 && u < 1.0);
                }
            }

            #[test]
            fn prop_normal_finite(seed: u64) {
                let mut rng = MonteCarloRng::new(seed);
                for _ in 0..100 {
                    let z = rng.standard_normal();
                    prop_assert!(z.is_finite());
                }
            }

            #[test]
            fn prop_exponential_positive(seed: u64, lambda in 0.01..10.0f64) {
                let mut rng = MonteCarloRng::new(seed);
                for _ in 0..100 {
                    let x = rng.exponential(lambda);
                    prop_assert!(x > 0.0);
                }
            }

            #[test]
            fn prop_gamma_positive(seed: u64, alpha in 0.1..5.0f64, beta in 0.1..5.0f64) {
                let mut rng = MonteCarloRng::new(seed);
                for _ in 0..20 {
                    let x = rng.gamma(alpha, beta);
                    prop_assert!(x > 0.0);
                }
            }

            #[test]
            fn prop_beta_bounded(seed: u64, alpha in 0.5..5.0f64, beta in 0.5..5.0f64) {
                let mut rng = MonteCarloRng::new(seed);
                for _ in 0..20 {
                    let x = rng.beta(alpha, beta);
                    prop_assert!(x >= 0.0 && x <= 1.0);
                }
            }
        }
    }
}
