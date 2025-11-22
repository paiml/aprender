//! Conjugate prior distributions for Bayesian inference.
//!
//! Conjugate priors allow closed-form posterior computation via Bayes' theorem.
//! Each conjugate family implements:
//! - Prior specification (uniform, Jeffrey's, informative)
//! - Posterior update from data
//! - Posterior statistics (mean, mode, variance, credible intervals)
//! - Posterior predictive distribution

use crate::{AprenderError, Result};

/// Beta-Binomial conjugate prior for Bernoulli/Binomial likelihood.
///
/// Models a probability parameter θ ∈ [0,1] for binary outcomes.
///
/// **Prior**: Beta(α, β)
/// **Likelihood**: Binomial(n, θ) or Bernoulli(θ)
/// **Posterior**: Beta(α + successes, β + failures)
///
/// # Mathematical Foundation
///
/// Given n trials with k successes:
/// - Prior: p(θ) = Beta(α, β) ∝ θ^(α-1) × (1-θ)^(β-1)
/// - Likelihood: p(k|θ,n) = Binomial(k|n,θ) ∝ θ^k × (1-θ)^(n-k)
/// - Posterior: p(θ|k,n) = Beta(α+k, β+n-k)
///
/// # Example
///
/// ```
/// use aprender::bayesian::BetaBinomial;
///
/// // Start with uniform prior Beta(1, 1)
/// let mut model = BetaBinomial::uniform();
///
/// // Observe 7 heads in 10 coin flips
/// model.update(7, 10);
///
/// // Posterior is Beta(8, 4)
/// assert!((model.alpha() - 8.0).abs() < 1e-6);
/// assert!((model.beta() - 4.0).abs() < 1e-6);
///
/// // Expected probability of heads
/// let mean = model.posterior_mean();
/// assert!((mean - 8.0/12.0).abs() < 1e-6);  // (α)/(α+β)
/// ```
#[derive(Debug, Clone)]
pub struct BetaBinomial {
    /// Shape parameter α (prior successes + 1)
    alpha: f32,
    /// Shape parameter β (prior failures + 1)
    beta: f32,
}

impl BetaBinomial {
    /// Creates a uniform prior Beta(1, 1).
    ///
    /// This represents complete ignorance: all probabilities θ ∈ [0,1] are equally likely.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let prior = BetaBinomial::uniform();
    /// assert_eq!(prior.alpha(), 1.0);
    /// assert_eq!(prior.beta(), 1.0);
    /// ```
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Creates Jeffrey's prior Beta(0.5, 0.5).
    ///
    /// This is the non-informative prior that is invariant under reparameterization.
    /// Recommended when no prior information is available.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let prior = BetaBinomial::jeffreys();
    /// assert_eq!(prior.alpha(), 0.5);
    /// assert_eq!(prior.beta(), 0.5);
    /// ```
    #[must_use]
    pub fn jeffreys() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Creates an informative prior Beta(α, β) from prior belief.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter α > 0 (prior successes + 1)
    /// * `beta` - Shape parameter β > 0 (prior failures + 1)
    ///
    /// # Interpretation
    ///
    /// - α = β: Belief that success and failure are equally likely
    /// - α > β: Belief that success is more likely
    /// - α < β: Belief that failure is more likely
    /// - α + β (total): Strength of prior belief (higher = stronger)
    ///
    /// # Errors
    ///
    /// Returns error if α ≤ 0 or β ≤ 0.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// // Strong prior belief: 80% success rate based on 100 trials
    /// let prior = BetaBinomial::new(80.0, 20.0).unwrap();
    /// assert!((prior.posterior_mean() - 0.8).abs() < 0.01);
    /// ```
    pub fn new(alpha: f32, beta: f32) -> Result<Self> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "alpha, beta".to_string(),
                value: format!("({alpha}, {beta})"),
                constraint: "both > 0".to_string(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Returns the current α parameter.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the current β parameter.
    #[must_use]
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Updates the posterior with observed data (Bayesian update).
    ///
    /// # Arguments
    ///
    /// * `successes` - Number of successful trials
    /// * `trials` - Total number of trials
    ///
    /// # Panics
    ///
    /// Panics if trials < successes.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let mut model = BetaBinomial::uniform();
    /// model.update(7, 10);  // 7 successes in 10 trials
    ///
    /// // Posterior is Beta(1+7, 1+3) = Beta(8, 4)
    /// assert_eq!(model.alpha(), 8.0);
    /// assert_eq!(model.beta(), 4.0);
    /// ```
    pub fn update(&mut self, successes: u32, trials: u32) {
        assert!(successes <= trials, "Successes cannot exceed total trials");
        let failures = trials - successes;
        #[allow(clippy::cast_precision_loss)]
        {
            self.alpha += successes as f32;
            self.beta += failures as f32;
        }
    }

    /// Computes the posterior mean E[θ|data] = α/(α+β).
    ///
    /// This is the expected value of the probability parameter under the posterior.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let mut model = BetaBinomial::uniform();
    /// model.update(7, 10);
    ///
    /// let mean = model.posterior_mean();
    /// assert!((mean - 8.0/12.0).abs() < 1e-6);  // 0.6667
    /// ```
    #[must_use]
    pub fn posterior_mean(&self) -> f32 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Computes the posterior mode (MAP estimate) = (α-1)/(α+β-2).
    ///
    /// This is the most probable value of θ under the posterior.
    /// Only defined for α > 1 and β > 1.
    ///
    /// # Returns
    ///
    /// - `Some(mode)` if α > 1 and β > 1
    /// - `None` if distribution is U-shaped (no unique mode)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let mut model = BetaBinomial::new(2.0, 2.0).unwrap();
    /// model.update(7, 10);
    ///
    /// // Posterior is Beta(9, 5), mode = (9-1)/(9+5-2) = 8/12
    /// let mode = model.posterior_mode().unwrap();
    /// assert!((mode - 8.0/12.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn posterior_mode(&self) -> Option<f32> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }

    /// Computes the posterior variance Var[θ|data] = αβ/[(α+β)²(α+β+1)].
    ///
    /// Measures uncertainty in the probability estimate.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let mut model = BetaBinomial::uniform();
    /// model.update(7, 10);
    ///
    /// // More data → lower variance (more certainty)
    /// let variance = model.posterior_variance();
    /// assert!(variance < 0.02);  // High certainty after 10 trials
    /// ```
    #[must_use]
    pub fn posterior_variance(&self) -> f32 {
        let sum = self.alpha + self.beta;
        (self.alpha * self.beta) / (sum * sum * (sum + 1.0))
    }

    /// Computes the posterior predictive probability P(success|data).
    ///
    /// This is the probability of success in the next trial, integrating
    /// over all possible values of θ weighted by the posterior.
    ///
    /// For Beta-Binomial, this equals the posterior mean.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let mut model = BetaBinomial::uniform();
    /// model.update(7, 10);
    ///
    /// let prob = model.posterior_predictive();
    /// assert!((prob - 8.0/12.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn posterior_predictive(&self) -> f32 {
        self.posterior_mean()
    }

    /// Computes the (1-α) credible interval using quantile approximation.
    ///
    /// Returns (lower, upper) bounds such that P(lower ≤ θ ≤ upper | data) = 1-α.
    ///
    /// # Arguments
    ///
    /// * `confidence` - Confidence level (e.g., 0.95 for 95% credible interval)
    ///
    /// # Returns
    ///
    /// (lower, upper) quantiles
    ///
    /// # Errors
    ///
    /// Returns error if confidence ∉ (0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BetaBinomial;
    ///
    /// let mut model = BetaBinomial::uniform();
    /// model.update(7, 10);
    ///
    /// let (lower, upper) = model.credible_interval(0.95).unwrap();
    /// assert!(lower < 0.6667 && 0.6667 < upper);
    /// assert!(upper - lower < 0.4);  // Reasonably narrow after 10 trials
    /// ```
    pub fn credible_interval(&self, confidence: f32) -> Result<(f32, f32)> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(AprenderError::InvalidHyperparameter {
                param: "confidence".to_string(),
                value: confidence.to_string(),
                constraint: "in (0, 1)".to_string(),
            });
        }

        // Simplified quantile approximation using normal approximation
        // For production, would use inverse beta CDF (requires special functions)
        let mean = self.posterior_mean();
        let std = self.posterior_variance().sqrt();

        // Normal approximation: μ ± z*σ where z ≈ 1.96 for 95%
        let z = match confidence {
            c if (c - 0.95).abs() < 0.01 => 1.96,
            c if (c - 0.99).abs() < 0.01 => 2.576,
            c if (c - 0.90).abs() < 0.01 => 1.645,
            _ => 1.96, // Default to 95%
        };

        let lower = (mean - z * std).max(0.0);
        let upper = (mean + z * std).min(1.0);

        Ok((lower, upper))
    }
}

// Placeholder structs for other conjugate priors
// Will be implemented in subsequent tasks

/// Gamma-Poisson conjugate prior for count data.
///
/// Models a rate parameter λ > 0 for Poisson-distributed count data.
/// Uses shape-rate parameterization: Gamma(α, β) where β is the rate parameter.
///
/// **Prior**: Gamma(α, β)
/// **Likelihood**: Poisson(λ)
/// **Posterior**: Gamma(α + Σxᵢ, β + n)
///
/// # Mathematical Foundation
///
/// Given n observations x₁, ..., xₙ from Poisson(λ):
/// - Prior: p(λ) = Gamma(α, β) ∝ λ^(α-1) × exp(-βλ)
/// - Likelihood: p(x|λ) = ∏ Poisson(xᵢ|λ) ∝ λ^(Σxᵢ) × exp(-nλ)
/// - Posterior: p(λ|x) = Gamma(α + Σxᵢ, β + n)
///
/// # Example
///
/// ```
/// use aprender::bayesian::GammaPoisson;
///
/// // Start with non-informative prior
/// let mut model = GammaPoisson::noninformative();
///
/// // Observe counts: [3, 5, 4, 6, 2] events per hour
/// model.update(&[3, 5, 4, 6, 2]);
///
/// // Expected event rate
/// let mean = model.posterior_mean();
/// assert!((mean - 4.0).abs() < 0.5);  // Should be around 4 events/hour
/// ```
#[derive(Debug, Clone)]
pub struct GammaPoisson {
    /// Shape parameter α (pseudo-count of prior observations)
    alpha: f32,
    /// Rate parameter β (pseudo-count of prior time intervals)
    beta: f32,
}

impl GammaPoisson {
    /// Creates a non-informative prior Gamma(0.001, 0.001).
    ///
    /// This weakly informative prior has minimal influence on the posterior,
    /// allowing the data to dominate.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// let prior = GammaPoisson::noninformative();
    /// assert_eq!(prior.alpha(), 0.001);
    /// assert_eq!(prior.beta(), 0.001);
    /// ```
    #[must_use]
    pub fn noninformative() -> Self {
        Self {
            alpha: 0.001,
            beta: 0.001,
        }
    }

    /// Creates an informative prior Gamma(α, β) from prior belief.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Shape parameter α > 0 (prior total count)
    /// * `beta` - Rate parameter β > 0 (prior number of intervals)
    ///
    /// # Interpretation
    ///
    /// - α/β: Prior mean rate
    /// - α: Pseudo-count of prior observations
    /// - β: Pseudo-count of prior time intervals
    /// - Larger α, β: Stronger prior belief
    ///
    /// # Errors
    ///
    /// Returns error if α ≤ 0 or β ≤ 0.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// // Prior belief: 50 events in 10 intervals (rate = 5)
    /// let prior = GammaPoisson::new(50.0, 10.0).unwrap();
    /// assert!((prior.posterior_mean() - 5.0).abs() < 0.01);
    /// ```
    pub fn new(alpha: f32, beta: f32) -> Result<Self> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "alpha, beta".to_string(),
                value: format!("({alpha}, {beta})"),
                constraint: "both > 0".to_string(),
            });
        }
        Ok(Self { alpha, beta })
    }

    /// Returns the current α parameter.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the current β parameter.
    #[must_use]
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Updates the posterior with observed count data (Bayesian update).
    ///
    /// # Arguments
    ///
    /// * `counts` - Slice of observed counts (non-negative integers)
    ///
    /// # Panics
    ///
    /// Panics if any count is negative.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// let mut model = GammaPoisson::noninformative();
    /// model.update(&[3, 5, 4, 6, 2]);
    ///
    /// // Posterior is Gamma(0.001 + 20, 0.001 + 5)
    /// assert!((model.alpha() - 20.001).abs() < 0.01);
    /// assert!((model.beta() - 5.001).abs() < 0.01);
    /// ```
    pub fn update(&mut self, counts: &[u32]) {
        let sum: u32 = counts.iter().sum();
        let n = counts.len();

        #[allow(clippy::cast_precision_loss)]
        {
            self.alpha += sum as f32;
            self.beta += n as f32;
        }
    }

    /// Computes the posterior mean E[λ|data] = α/β.
    ///
    /// This is the expected value of the rate parameter under the posterior.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// let mut model = GammaPoisson::noninformative();
    /// model.update(&[3, 5, 4, 6, 2]);  // Sum = 20, n = 5
    ///
    /// let mean = model.posterior_mean();
    /// assert!((mean - 20.001/5.001).abs() < 0.01);  // ≈ 4.0
    /// ```
    #[must_use]
    pub fn posterior_mean(&self) -> f32 {
        self.alpha / self.beta
    }

    /// Computes the posterior mode (MAP estimate) = (α-1)/β.
    ///
    /// This is the most probable value of λ under the posterior.
    /// Only defined for α > 1.
    ///
    /// # Returns
    ///
    /// - `Some(mode)` if α > 1
    /// - `None` if distribution is monotonically decreasing (α ≤ 1)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// let mut model = GammaPoisson::new(2.0, 1.0).unwrap();
    /// model.update(&[3, 5, 4, 6, 2]);
    ///
    /// let mode = model.posterior_mode().unwrap();
    /// assert!((mode - (22.0 - 1.0)/6.0).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn posterior_mode(&self) -> Option<f32> {
        if self.alpha > 1.0 {
            Some((self.alpha - 1.0) / self.beta)
        } else {
            None
        }
    }

    /// Computes the posterior variance Var[λ|data] = α/β².
    ///
    /// Measures uncertainty in the rate estimate.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// let mut model = GammaPoisson::noninformative();
    /// model.update(&[3, 5, 4, 6, 2]);
    ///
    /// let variance = model.posterior_variance();
    /// assert!(variance < 1.0);  // Low uncertainty with 5 observations
    /// ```
    #[must_use]
    pub fn posterior_variance(&self) -> f32 {
        self.alpha / (self.beta * self.beta)
    }

    /// Computes the posterior predictive distribution for next observation.
    ///
    /// For Gamma-Poisson, the posterior predictive is Negative Binomial.
    /// Returns the mean of the predictive distribution: α/β.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// let mut model = GammaPoisson::noninformative();
    /// model.update(&[3, 5, 4, 6, 2]);
    ///
    /// let pred_mean = model.posterior_predictive();
    /// assert!((pred_mean - 4.0).abs() < 0.5);
    /// ```
    #[must_use]
    pub fn posterior_predictive(&self) -> f32 {
        self.posterior_mean()
    }

    /// Computes the (1-α) credible interval using normal approximation.
    ///
    /// Returns (lower, upper) bounds such that P(lower ≤ λ ≤ upper | data) = 1-α.
    ///
    /// # Arguments
    ///
    /// * `confidence` - Confidence level (e.g., 0.95 for 95% credible interval)
    ///
    /// # Returns
    ///
    /// (lower, upper) quantiles, with lower ≥ 0 (rate cannot be negative)
    ///
    /// # Errors
    ///
    /// Returns error if confidence ∉ (0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::GammaPoisson;
    ///
    /// let mut model = GammaPoisson::noninformative();
    /// model.update(&[3, 5, 4, 6, 2]);
    ///
    /// let (lower, upper) = model.credible_interval(0.95).unwrap();
    /// assert!(lower < 4.0 && 4.0 < upper);
    /// ```
    pub fn credible_interval(&self, confidence: f32) -> Result<(f32, f32)> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(AprenderError::InvalidHyperparameter {
                param: "confidence".to_string(),
                value: confidence.to_string(),
                constraint: "in (0, 1)".to_string(),
            });
        }

        // Normal approximation for Gamma distribution
        // For large α, Gamma(α, β) ≈ N(α/β, α/β²)
        let mean = self.posterior_mean();
        let std = self.posterior_variance().sqrt();

        let z = match confidence {
            c if (c - 0.95).abs() < 0.01 => 1.96,
            c if (c - 0.99).abs() < 0.01 => 2.576,
            c if (c - 0.90).abs() < 0.01 => 1.645,
            _ => 1.96,
        };

        let lower = (mean - z * std).max(0.0); // Rate cannot be negative
        let upper = mean + z * std;

        Ok((lower, upper))
    }
}

/// Normal-InverseGamma conjugate prior for normal data with unknown mean and variance.
///
/// **Prior**: N-IG(μ₀, κ₀, α₀, β₀)
/// **Likelihood**: N(μ, σ²)
/// **Posterior**: N-IG with updated parameters
#[derive(Debug, Clone)]
#[allow(dead_code)] // Placeholder for future implementation
pub struct NormalInverseGamma {
    mu: f32,
    kappa: f32,
    alpha: f32,
    beta: f32,
}

impl NormalInverseGamma {
    /// Creates a non-informative prior.
    #[must_use]
    pub fn noninformative() -> Self {
        Self {
            mu: 0.0,
            kappa: 0.001,
            alpha: 0.001,
            beta: 0.001,
        }
    }
}

/// Dirichlet-Multinomial conjugate prior for categorical data.
///
/// **Prior**: Dirichlet(α₁, ..., αₖ)
/// **Likelihood**: Multinomial(n, θ₁, ..., θₖ)
/// **Posterior**: Dirichlet(α₁ + n₁, ..., αₖ + nₖ)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Placeholder for future implementation
pub struct DirichletMultinomial {
    alphas: Vec<f32>,
}

impl DirichletMultinomial {
    /// Creates a uniform prior Dirichlet(1, ..., 1) for k categories.
    #[must_use]
    pub fn uniform(k: usize) -> Self {
        Self {
            alphas: vec![1.0; k],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========== Beta-Binomial Tests ==========

    #[test]
    fn test_beta_binomial_uniform_prior() {
        let prior = BetaBinomial::uniform();
        assert_eq!(prior.alpha(), 1.0);
        assert_eq!(prior.beta(), 1.0);
        assert!((prior.posterior_mean() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_beta_binomial_jeffreys_prior() {
        let prior = BetaBinomial::jeffreys();
        assert_eq!(prior.alpha(), 0.5);
        assert_eq!(prior.beta(), 0.5);
    }

    #[test]
    fn test_beta_binomial_custom_prior() {
        let prior = BetaBinomial::new(2.0, 5.0).expect("Valid parameters");
        assert_eq!(prior.alpha(), 2.0);
        assert_eq!(prior.beta(), 5.0);
    }

    #[test]
    fn test_beta_binomial_invalid_prior() {
        assert!(BetaBinomial::new(0.0, 1.0).is_err());
        assert!(BetaBinomial::new(1.0, -1.0).is_err());
    }

    #[test]
    fn test_beta_binomial_update() {
        let mut model = BetaBinomial::uniform();
        model.update(7, 10);

        // Posterior should be Beta(1+7, 1+3) = Beta(8, 4)
        assert_eq!(model.alpha(), 8.0);
        assert_eq!(model.beta(), 4.0);
    }

    #[test]
    fn test_beta_binomial_posterior_mean() {
        let mut model = BetaBinomial::uniform();
        model.update(7, 10);

        let mean = model.posterior_mean();
        let expected = 8.0 / 12.0; // α/(α+β)
        assert!((mean - expected).abs() < 1e-6);
    }

    #[test]
    fn test_beta_binomial_posterior_mode() {
        let mut model = BetaBinomial::new(2.0, 2.0).expect("Valid parameters");
        model.update(7, 10);

        // Posterior is Beta(9, 5)
        // Mode = (α-1)/(α+β-2) = 8/12
        let mode = model.posterior_mode().expect("Mode should exist");
        let expected = 8.0 / 12.0;
        assert!((mode - expected).abs() < 1e-6);
    }

    #[test]
    fn test_beta_binomial_no_mode_for_uniform() {
        let model = BetaBinomial::uniform();
        // Beta(1, 1) has no unique mode (uniform on [0,1])
        assert!(model.posterior_mode().is_none());
    }

    #[test]
    fn test_beta_binomial_posterior_variance() {
        let mut model = BetaBinomial::uniform();
        model.update(70, 100);

        let variance = model.posterior_variance();

        // More data → lower variance
        assert!(variance < 0.01);
    }

    #[test]
    fn test_beta_binomial_predictive() {
        let mut model = BetaBinomial::uniform();
        model.update(7, 10);

        let prob = model.posterior_predictive();
        let mean = model.posterior_mean();

        // For Beta-Binomial, predictive equals posterior mean
        assert!((prob - mean).abs() < 1e-6);
    }

    #[test]
    fn test_beta_binomial_credible_interval() {
        let mut model = BetaBinomial::uniform();
        model.update(7, 10);

        let (lower, upper) = model
            .credible_interval(0.95)
            .expect("Valid confidence level");

        let mean = model.posterior_mean();

        // Mean should be within interval
        assert!(lower < mean);
        assert!(mean < upper);

        // Bounds should be in [0, 1]
        assert!((0.0..=1.0).contains(&lower));
        assert!((0.0..=1.0).contains(&upper));
    }

    #[test]
    fn test_beta_binomial_credible_interval_invalid() {
        let model = BetaBinomial::uniform();

        assert!(model.credible_interval(-0.1).is_err());
        assert!(model.credible_interval(1.1).is_err());
    }

    #[test]
    fn test_beta_binomial_sequential_updates() {
        let mut model = BetaBinomial::uniform();

        // First experiment: 7/10 successes
        model.update(7, 10);
        assert_eq!(model.alpha(), 8.0);
        assert_eq!(model.beta(), 4.0);

        // Second experiment: 3/5 successes
        model.update(3, 5);
        assert_eq!(model.alpha(), 11.0);
        assert_eq!(model.beta(), 6.0);
    }

    #[test]
    #[should_panic(expected = "Successes cannot exceed total trials")]
    fn test_beta_binomial_invalid_update() {
        let mut model = BetaBinomial::uniform();
        model.update(11, 10); // More successes than trials
    }

    // Property-based tests would go here using proptest
    // Example: Verify posterior mean is always in [0, 1]
    // Example: Verify variance decreases with more data

    // ========== Gamma-Poisson Tests ==========

    #[test]
    fn test_gamma_poisson_noninformative_prior() {
        let prior = GammaPoisson::noninformative();
        assert_eq!(prior.alpha(), 0.001);
        assert_eq!(prior.beta(), 0.001);
        assert!((prior.posterior_mean() - 1.0).abs() < 0.01); // α/β = 1
    }

    #[test]
    fn test_gamma_poisson_custom_prior() {
        let prior = GammaPoisson::new(50.0, 10.0).expect("Valid parameters");
        assert_eq!(prior.alpha(), 50.0);
        assert_eq!(prior.beta(), 10.0);
        assert!((prior.posterior_mean() - 5.0).abs() < 0.01); // 50/10 = 5
    }

    #[test]
    fn test_gamma_poisson_invalid_prior() {
        assert!(GammaPoisson::new(0.0, 1.0).is_err());
        assert!(GammaPoisson::new(1.0, -1.0).is_err());
    }

    #[test]
    fn test_gamma_poisson_update() {
        let mut model = GammaPoisson::noninformative();
        model.update(&[3, 5, 4, 6, 2]);

        // Sum = 20, n = 5
        // Posterior should be Gamma(0.001 + 20, 0.001 + 5)
        assert!((model.alpha() - 20.001).abs() < 0.01);
        assert!((model.beta() - 5.001).abs() < 0.01);
    }

    #[test]
    fn test_gamma_poisson_posterior_mean() {
        let mut model = GammaPoisson::noninformative();
        model.update(&[3, 5, 4, 6, 2]);

        let mean = model.posterior_mean();
        let expected = 20.001 / 5.001; // α/β ≈ 4.0
        assert!((mean - expected).abs() < 0.01);
    }

    #[test]
    fn test_gamma_poisson_posterior_mode() {
        let mut model = GammaPoisson::new(2.0, 1.0).expect("Valid parameters");
        model.update(&[3, 5, 4, 6, 2]);

        // Posterior is Gamma(22, 6)
        // Mode = (α-1)/β = 21/6 = 3.5
        let mode = model.posterior_mode().expect("Mode should exist");
        let expected = 21.0 / 6.0;
        assert!((mode - expected).abs() < 0.01);
    }

    #[test]
    fn test_gamma_poisson_no_mode_for_weak_prior() {
        let model = GammaPoisson::noninformative();
        // Gamma(0.001, 0.001) has α < 1, no unique mode
        assert!(model.posterior_mode().is_none());
    }

    #[test]
    fn test_gamma_poisson_posterior_variance() {
        let mut model = GammaPoisson::noninformative();
        model.update(&[3, 5, 4, 6, 2, 8, 7, 9, 1, 0]); // 10 observations

        let variance = model.posterior_variance();

        // More data → lower variance
        assert!(variance < 1.0);
    }

    #[test]
    fn test_gamma_poisson_predictive() {
        let mut model = GammaPoisson::noninformative();
        model.update(&[3, 5, 4, 6, 2]);

        let prob = model.posterior_predictive();
        let mean = model.posterior_mean();

        // For Gamma-Poisson, predictive mean equals posterior mean
        assert!((prob - mean).abs() < 1e-6);
    }

    #[test]
    fn test_gamma_poisson_credible_interval() {
        let mut model = GammaPoisson::noninformative();
        model.update(&[3, 5, 4, 6, 2]);

        let (lower, upper) = model
            .credible_interval(0.95)
            .expect("Valid confidence level");

        let mean = model.posterior_mean();

        // Mean should be within interval
        assert!(lower < mean);
        assert!(mean < upper);

        // Lower bound should be non-negative (rate cannot be negative)
        assert!(lower >= 0.0);
    }

    #[test]
    fn test_gamma_poisson_credible_interval_invalid() {
        let model = GammaPoisson::noninformative();

        assert!(model.credible_interval(-0.1).is_err());
        assert!(model.credible_interval(1.1).is_err());
    }

    #[test]
    fn test_gamma_poisson_sequential_updates() {
        let mut model = GammaPoisson::noninformative();

        // First batch: [3, 5, 4] sum=12, n=3
        model.update(&[3, 5, 4]);
        assert!((model.alpha() - 12.001).abs() < 0.01);
        assert!((model.beta() - 3.001).abs() < 0.01);

        // Second batch: [6, 2] sum=8, n=2
        model.update(&[6, 2]);
        assert!((model.alpha() - 20.001).abs() < 0.01);
        assert!((model.beta() - 5.001).abs() < 0.01);
    }

    #[test]
    fn test_gamma_poisson_empty_update() {
        let mut model = GammaPoisson::noninformative();
        let original_alpha = model.alpha();
        let original_beta = model.beta();

        // Empty data should not change parameters
        model.update(&[]);

        assert_eq!(model.alpha(), original_alpha);
        assert_eq!(model.beta(), original_beta);
    }
}
