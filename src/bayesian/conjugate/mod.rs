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
/// Models a probability parameter θ in the range `[0,1]` for binary outcomes.
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
    /// This represents complete ignorance: all probabilities θ in `[0,1]` are equally likely.
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
    /// let prior = BetaBinomial::new(80.0, 20.0).expect("valid alpha and beta parameters");
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
    /// let mut model = BetaBinomial::new(2.0, 2.0).expect("valid alpha and beta parameters");
    /// model.update(7, 10);
    ///
    /// // Posterior is Beta(9, 5), mode = (9-1)/(9+5-2) = 8/12
    /// let mode = model.posterior_mode().expect("mode exists when alpha > 1 and beta > 1");
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
    /// let (lower, upper) = model.credible_interval(0.95).expect("valid confidence level");
    /// assert!(lower < 0.6667 && 0.6667 < upper);
    /// assert!(upper - lower < 0.55);  // Reasonably narrow after 10 trials
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

include!("mod_part_02.rs");
include!("mod_part_03.rs");
include!("mod_part_04.rs");
