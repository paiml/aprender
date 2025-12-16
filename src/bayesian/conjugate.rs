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
/// Models both the mean μ and variance σ² of normally distributed data.
/// Uses the Normal-InverseGamma hierarchical structure:
/// - σ² ~ InverseGamma(α, β)
/// - μ | σ² ~ Normal(μ₀, σ²/κ)
///
/// **Prior**: N-IG(μ₀, κ₀, α₀, β₀)
/// **Likelihood**: N(μ, σ²)
/// **Posterior**: N-IG(μₙ, κₙ, αₙ, βₙ)
///
/// # Mathematical Foundation
///
/// Given n observations x₁, ..., xₙ from N(μ, σ²):
/// - μₙ = (κ₀μ₀ + nμ̄) / (κ₀ + n)
/// - κₙ = κ₀ + n
/// - αₙ = α₀ + n/2
/// - βₙ = β₀ + ½s² + κ₀n(μ̄ - μ₀)²/(2(κ₀ + n))
///
/// where μ̄ = sample mean, s² = sum of squared deviations
///
/// # Example
///
/// ```
/// use aprender::bayesian::NormalInverseGamma;
///
/// // Start with non-informative prior
/// let mut model = NormalInverseGamma::noninformative();
///
/// // Observe data from N(5, 4): [4.2, 5.8, 6.1, 4.5, 5.0]
/// model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
///
/// // Posterior mean of μ
/// let mean = model.posterior_mean_mu();
/// assert!((mean - 5.1).abs() < 0.5);
/// ```
#[derive(Debug, Clone)]
pub struct NormalInverseGamma {
    /// Location parameter μ₀ (prior mean)
    mu: f32,
    /// Precision parameter κ₀ (prior sample size for mean)
    kappa: f32,
    /// Shape parameter α₀ (prior sample size for variance)
    alpha: f32,
    /// Scale parameter β₀ (prior sum of squared deviations)
    beta: f32,
}

impl NormalInverseGamma {
    /// Creates a non-informative prior N-IG(0, 0.001, 0.001, 0.001).
    ///
    /// This weakly informative prior allows the data to dominate.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// let prior = NormalInverseGamma::noninformative();
    /// assert_eq!(prior.mu(), 0.0);
    /// assert_eq!(prior.kappa(), 0.001);
    /// ```
    #[must_use]
    pub fn noninformative() -> Self {
        Self {
            mu: 0.0,
            kappa: 0.001,
            alpha: 0.001,
            beta: 0.001,
        }
    }

    /// Creates an informative prior N-IG(μ₀, κ₀, α₀, β₀) from prior belief.
    ///
    /// # Arguments
    ///
    /// * `mu` - Prior mean μ₀
    /// * `kappa` - Prior precision κ₀ > 0 (pseudo sample size for mean)
    /// * `alpha` - Prior shape α₀ > 0 (pseudo sample size for variance)
    /// * `beta` - Prior scale β₀ > 0 (pseudo sum of squared deviations)
    ///
    /// # Interpretation
    ///
    /// - μ₀: Prior belief about the mean
    /// - κ₀: Confidence in prior mean (larger = stronger belief)
    /// - α₀: Prior degrees of freedom for variance
    /// - β₀: Prior scale for variance
    ///
    /// # Errors
    ///
    /// Returns error if κ ≤ 0, α ≤ 0, or β ≤ 0.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// // Prior belief: mean ≈ 5.0, variance ≈ 1.0, moderate confidence
    /// let prior = NormalInverseGamma::new(5.0, 10.0, 5.0, 5.0).unwrap();
    /// assert_eq!(prior.mu(), 5.0);
    /// ```
    pub fn new(mu: f32, kappa: f32, alpha: f32, beta: f32) -> Result<Self> {
        if kappa <= 0.0 || alpha <= 0.0 || beta <= 0.0 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "kappa, alpha, beta".to_string(),
                value: format!("({kappa}, {alpha}, {beta})"),
                constraint: "all > 0".to_string(),
            });
        }
        Ok(Self {
            mu,
            kappa,
            alpha,
            beta,
        })
    }

    /// Returns the current μ₀ parameter.
    #[must_use]
    pub fn mu(&self) -> f32 {
        self.mu
    }

    /// Returns the current κ₀ parameter.
    #[must_use]
    pub fn kappa(&self) -> f32 {
        self.kappa
    }

    /// Returns the current α₀ parameter.
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the current β₀ parameter.
    #[must_use]
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// Updates the posterior with observed data (Bayesian update).
    ///
    /// # Arguments
    ///
    /// * `data` - Slice of observed values
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// let mut model = NormalInverseGamma::noninformative();
    /// model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
    ///
    /// // Parameters have been updated via Bayesian inference
    /// assert!(model.kappa() > 0.001); // Precision increased
    /// ```
    pub fn update(&mut self, data: &[f32]) {
        if data.is_empty() {
            return;
        }

        let n = data.len() as f32;

        // Compute sample statistics
        let sample_mean = data.iter().sum::<f32>() / n;
        let sum_squared_deviations: f32 = data.iter().map(|&x| (x - sample_mean).powi(2)).sum();

        // Update parameters
        let kappa_n = self.kappa + n;
        let mu_n = (self.kappa * self.mu + n * sample_mean) / kappa_n;
        let alpha_n = self.alpha + n / 2.0;
        let beta_n = self.beta
            + sum_squared_deviations / 2.0
            + (self.kappa * n * (sample_mean - self.mu).powi(2)) / (2.0 * kappa_n);

        self.mu = mu_n;
        self.kappa = kappa_n;
        self.alpha = alpha_n;
        self.beta = beta_n;
    }

    /// Computes the posterior mean of μ: E[μ|data] = μₙ.
    ///
    /// This is the expected value of the mean parameter.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// let mut model = NormalInverseGamma::noninformative();
    /// model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
    ///
    /// let mean_mu = model.posterior_mean_mu();
    /// assert!((mean_mu - 5.1).abs() < 0.5); // Close to sample mean
    /// ```
    #[must_use]
    pub fn posterior_mean_mu(&self) -> f32 {
        self.mu
    }

    /// Computes the posterior mean of σ²: E[σ²|data] = β/(α-1) for α > 1.
    ///
    /// This is the expected value of the variance parameter.
    ///
    /// # Returns
    ///
    /// - `Some(variance)` if α > 1
    /// - `None` if α ≤ 1 (mean undefined)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// let mut model = NormalInverseGamma::noninformative();
    /// model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
    ///
    /// let mean_var = model.posterior_mean_variance().unwrap();
    /// assert!(mean_var > 0.0);
    /// ```
    #[must_use]
    pub fn posterior_mean_variance(&self) -> Option<f32> {
        if self.alpha > 1.0 {
            Some(self.beta / (self.alpha - 1.0))
        } else {
            None
        }
    }

    /// Computes the posterior variance of μ: Var[μ|data] = β/(κ(α-1)) for α > 1.
    ///
    /// Measures uncertainty in the mean estimate.
    ///
    /// # Returns
    ///
    /// - `Some(variance)` if α > 1
    /// - `None` if α ≤ 1 (variance undefined)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// let mut model = NormalInverseGamma::noninformative();
    /// model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
    ///
    /// let var_mu = model.posterior_variance_mu().unwrap();
    /// assert!(var_mu > 0.0); // Positive uncertainty
    /// ```
    #[must_use]
    pub fn posterior_variance_mu(&self) -> Option<f32> {
        if self.alpha > 1.0 {
            Some(self.beta / (self.kappa * (self.alpha - 1.0)))
        } else {
            None
        }
    }

    /// Computes the posterior predictive distribution mean: μₙ.
    ///
    /// For Normal-InverseGamma, the posterior predictive is Student's t-distribution.
    /// Returns the mean of the predictive distribution.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// let mut model = NormalInverseGamma::noninformative();
    /// model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
    ///
    /// let pred_mean = model.posterior_predictive();
    /// assert!((pred_mean - 5.1).abs() < 0.5);
    /// ```
    #[must_use]
    pub fn posterior_predictive(&self) -> f32 {
        self.mu
    }

    /// Computes a (1-α) credible interval for μ using Student's t-distribution.
    ///
    /// Returns (lower, upper) bounds such that P(lower ≤ μ ≤ upper | data) = 1-α.
    ///
    /// # Arguments
    ///
    /// * `confidence` - Confidence level (e.g., 0.95 for 95% credible interval)
    ///
    /// # Errors
    ///
    /// Returns error if confidence ∉ (0, 1) or if α ≤ 1 (variance undefined).
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::NormalInverseGamma;
    ///
    /// let mut model = NormalInverseGamma::noninformative();
    /// model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);
    ///
    /// let (lower, upper) = model.credible_interval_mu(0.95).unwrap();
    /// assert!(lower < 5.1 && 5.1 < upper);
    /// ```
    pub fn credible_interval_mu(&self, confidence: f32) -> Result<(f32, f32)> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(AprenderError::InvalidHyperparameter {
                param: "confidence".to_string(),
                value: confidence.to_string(),
                constraint: "in (0, 1)".to_string(),
            });
        }

        if self.alpha <= 1.0 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "alpha".to_string(),
                value: self.alpha.to_string(),
                constraint: "> 1 for credible interval".to_string(),
            });
        }

        // Approximate using normal distribution
        // For exact t-distribution, would need special functions
        let mean = self.mu;
        let std = self
            .posterior_variance_mu()
            .expect("Variance exists when alpha > 1")
            .sqrt();

        let z = match confidence {
            c if (c - 0.95).abs() < 0.01 => 1.96,
            c if (c - 0.99).abs() < 0.01 => 2.576,
            c if (c - 0.90).abs() < 0.01 => 1.645,
            _ => 1.96,
        };

        let lower = mean - z * std;
        let upper = mean + z * std;

        Ok((lower, upper))
    }
}

/// Dirichlet-Multinomial conjugate prior for categorical data.
///
/// Models probability parameters θ₁, ..., θₖ for k mutually exclusive categories
/// where Σθᵢ = 1 (probability simplex).
///
/// **Prior**: Dirichlet(α₁, ..., αₖ)
/// **Likelihood**: Multinomial(n, θ₁, ..., θₖ)
/// **Posterior**: Dirichlet(α₁ + n₁, ..., αₖ + nₖ)
///
/// # Mathematical Foundation
///
/// Given observations in k categories with counts n₁, ..., nₖ:
/// - Prior: p(θ) = Dirichlet(α) ∝ ∏θᵢ^(αᵢ-1)
/// - Likelihood: p(n|θ) = Multinomial(n|θ) ∝ ∏θᵢ^nᵢ
/// - Posterior: p(θ|n) = Dirichlet(α + n)
///
/// where α + n means element-wise addition: (α₁ + n₁, ..., αₖ + nₖ)
///
/// # Example
///
/// ```
/// use aprender::bayesian::DirichletMultinomial;
///
/// // 3-category classification: [A, B, C]
/// let mut model = DirichletMultinomial::uniform(3);
///
/// // Observe counts: 10 A's, 5 B's, 3 C's
/// model.update(&[10, 5, 3]);
///
/// // Posterior probabilities
/// let probs = model.posterior_mean();
/// assert!((probs[0] - 10.0/21.0).abs() < 0.1); // P(A) ≈ 0.476
/// ```
#[derive(Debug, Clone)]
pub struct DirichletMultinomial {
    /// Concentration parameters α₁, ..., αₖ (pseudo-counts for each category)
    alphas: Vec<f32>,
}

impl DirichletMultinomial {
    /// Creates a uniform prior Dirichlet(1, ..., 1) for k categories.
    ///
    /// This represents equal probability for all categories with minimal prior belief.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of categories (must be ≥ 2)
    ///
    /// # Panics
    ///
    /// Panics if k < 2.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// let prior = DirichletMultinomial::uniform(3);
    /// assert_eq!(prior.alphas().len(), 3);
    /// assert_eq!(prior.alphas()[0], 1.0);
    /// ```
    #[must_use]
    pub fn uniform(k: usize) -> Self {
        assert!(k >= 2, "Must have at least 2 categories");
        Self {
            alphas: vec![1.0; k],
        }
    }

    /// Creates an informative prior Dirichlet(α₁, ..., αₖ) from prior belief.
    ///
    /// # Arguments
    ///
    /// * `alphas` - Concentration parameters αᵢ > 0 for each category
    ///
    /// # Interpretation
    ///
    /// - αᵢ: Pseudo-count for category i
    /// - Σαᵢ: Total pseudo-count (strength of prior belief)
    /// - αᵢ / Σαⱼ: Prior mean probability for category i
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any αᵢ ≤ 0
    /// - Fewer than 2 categories
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// // Prior belief: category probabilities [0.5, 0.3, 0.2] with 10 pseudo-counts
    /// let prior = DirichletMultinomial::new(vec![5.0, 3.0, 2.0]).unwrap();
    /// let mean = prior.posterior_mean();
    /// assert!((mean[0] - 0.5).abs() < 0.01);
    /// ```
    pub fn new(alphas: Vec<f32>) -> Result<Self> {
        if alphas.len() < 2 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "alphas".to_string(),
                value: format!("{} categories", alphas.len()),
                constraint: "at least 2 categories".to_string(),
            });
        }

        if alphas.iter().any(|&a| a <= 0.0) {
            return Err(AprenderError::InvalidHyperparameter {
                param: "alphas".to_string(),
                value: format!("{alphas:?}"),
                constraint: "all > 0".to_string(),
            });
        }

        Ok(Self { alphas })
    }

    /// Returns the current concentration parameters.
    #[must_use]
    pub fn alphas(&self) -> &[f32] {
        &self.alphas
    }

    /// Returns the number of categories.
    #[must_use]
    pub fn num_categories(&self) -> usize {
        self.alphas.len()
    }

    /// Updates the posterior with observed category counts (Bayesian update).
    ///
    /// # Arguments
    ///
    /// * `counts` - Observed counts for each category
    ///
    /// # Panics
    ///
    /// Panics if counts.len() != num_categories().
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// let mut model = DirichletMultinomial::uniform(3);
    /// model.update(&[10, 5, 3]); // 10 A's, 5 B's, 3 C's
    ///
    /// // Posterior is Dirichlet(1+10, 1+5, 1+3) = Dirichlet(11, 6, 4)
    /// assert_eq!(model.alphas()[0], 11.0);
    /// ```
    pub fn update(&mut self, counts: &[u32]) {
        assert_eq!(
            counts.len(),
            self.alphas.len(),
            "Counts must match number of categories"
        );

        for (alpha, &count) in self.alphas.iter_mut().zip(counts.iter()) {
            #[allow(clippy::cast_precision_loss)]
            {
                *alpha += count as f32;
            }
        }
    }

    /// Computes the posterior mean E[θ|data] for all categories.
    ///
    /// Returns a vector where element i is E[θᵢ|data] = αᵢ / Σαⱼ.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// let mut model = DirichletMultinomial::uniform(3);
    /// model.update(&[10, 5, 3]);
    ///
    /// let mean = model.posterior_mean();
    /// assert!((mean[0] - 11.0/21.0).abs() < 0.01); // (1+10)/(1+1+1+10+5+3)
    /// assert!((mean.iter().sum::<f32>() - 1.0).abs() < 1e-6); // Sums to 1
    /// ```
    #[must_use]
    pub fn posterior_mean(&self) -> Vec<f32> {
        let sum: f32 = self.alphas.iter().sum();
        self.alphas.iter().map(|&a| a / sum).collect()
    }

    /// Computes the posterior mode (MAP estimate) for all categories.
    ///
    /// Returns a vector where element i is (αᵢ - 1) / (Σαⱼ - k).
    /// Only defined when all αᵢ > 1.
    ///
    /// # Returns
    ///
    /// - `Some(mode)` if all αᵢ > 1
    /// - `None` if any αᵢ ≤ 1 (distribution has no unique mode)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// let mut model = DirichletMultinomial::new(vec![2.0, 2.0, 2.0]).unwrap();
    /// model.update(&[10, 5, 3]);
    ///
    /// let mode = model.posterior_mode().unwrap();
    /// assert!((mode[0] - 11.0/21.0).abs() < 0.01); // (12-1)/(24-3)
    /// ```
    #[must_use]
    pub fn posterior_mode(&self) -> Option<Vec<f32>> {
        if self.alphas.iter().all(|&a| a > 1.0) {
            let k = self.alphas.len() as f32;
            let sum: f32 = self.alphas.iter().sum();
            Some(self.alphas.iter().map(|&a| (a - 1.0) / (sum - k)).collect())
        } else {
            None
        }
    }

    /// Computes the posterior variance `Var[θᵢ|data]` for all categories.
    ///
    /// Returns a vector where element i is:
    /// `Var[θᵢ] = αᵢ(α₀ - αᵢ) / (α₀²(α₀ + 1))`
    /// where `α₀ = Σαⱼ`
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// let mut model = DirichletMultinomial::uniform(3);
    /// model.update(&[10, 5, 3]);
    ///
    /// let variance = model.posterior_variance();
    /// assert!(variance[0] > 0.0); // Positive uncertainty
    /// ```
    #[must_use]
    pub fn posterior_variance(&self) -> Vec<f32> {
        let sum: f32 = self.alphas.iter().sum();
        self.alphas
            .iter()
            .map(|&a| a * (sum - a) / (sum * sum * (sum + 1.0)))
            .collect()
    }

    /// Computes the posterior predictive distribution for the next observation.
    ///
    /// For Dirichlet-Multinomial, the posterior predictive probabilities are:
    /// P(category i | data) = αᵢ / Σαⱼ
    ///
    /// This equals the posterior mean.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// let mut model = DirichletMultinomial::uniform(3);
    /// model.update(&[10, 5, 3]);
    ///
    /// let pred = model.posterior_predictive();
    /// assert!((pred[0] - 11.0/21.0).abs() < 0.01);
    /// assert!((pred.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    /// ```
    #[must_use]
    pub fn posterior_predictive(&self) -> Vec<f32> {
        self.posterior_mean()
    }

    /// Computes (1-α) credible intervals for all category probabilities.
    ///
    /// Returns a vector of (lower, upper) bounds for each category.
    /// Uses normal approximation for each marginal distribution.
    ///
    /// # Arguments
    ///
    /// * `confidence` - Confidence level (e.g., 0.95 for 95% credible intervals)
    ///
    /// # Errors
    ///
    /// Returns error if confidence ∉ (0, 1).
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::DirichletMultinomial;
    ///
    /// let mut model = DirichletMultinomial::uniform(3);
    /// model.update(&[10, 5, 3]);
    ///
    /// let intervals = model.credible_intervals(0.95).unwrap();
    /// let mean = model.posterior_mean();
    ///
    /// // Mean should be within interval for each category
    /// for i in 0..3 {
    ///     assert!(intervals[i].0 < mean[i] && mean[i] < intervals[i].1);
    /// }
    /// ```
    pub fn credible_intervals(&self, confidence: f32) -> Result<Vec<(f32, f32)>> {
        if !(0.0..1.0).contains(&confidence) {
            return Err(AprenderError::InvalidHyperparameter {
                param: "confidence".to_string(),
                value: confidence.to_string(),
                constraint: "in (0, 1)".to_string(),
            });
        }

        let means = self.posterior_mean();
        let variances = self.posterior_variance();

        let z = match confidence {
            c if (c - 0.95).abs() < 0.01 => 1.96,
            c if (c - 0.99).abs() < 0.01 => 2.576,
            c if (c - 0.90).abs() < 0.01 => 1.645,
            _ => 1.96,
        };

        let intervals = means
            .iter()
            .zip(variances.iter())
            .map(|(&mean, &var)| {
                let std = var.sqrt();
                let lower = (mean - z * std).max(0.0); // Probability cannot be negative
                let upper = (mean + z * std).min(1.0); // Probability cannot exceed 1
                (lower, upper)
            })
            .collect();

        Ok(intervals)
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

    // ========== Normal-InverseGamma Tests ==========

    #[test]
    fn test_normal_inverse_gamma_noninformative_prior() {
        let prior = NormalInverseGamma::noninformative();
        assert_eq!(prior.mu(), 0.0);
        assert_eq!(prior.kappa(), 0.001);
        assert_eq!(prior.alpha(), 0.001);
        assert_eq!(prior.beta(), 0.001);
    }

    #[test]
    fn test_normal_inverse_gamma_custom_prior() {
        let prior = NormalInverseGamma::new(5.0, 10.0, 5.0, 5.0).expect("Valid parameters");
        assert_eq!(prior.mu(), 5.0);
        assert_eq!(prior.kappa(), 10.0);
        assert_eq!(prior.alpha(), 5.0);
        assert_eq!(prior.beta(), 5.0);
    }

    #[test]
    fn test_normal_inverse_gamma_invalid_prior() {
        assert!(NormalInverseGamma::new(0.0, 0.0, 1.0, 1.0).is_err()); // kappa = 0
        assert!(NormalInverseGamma::new(0.0, 1.0, 0.0, 1.0).is_err()); // alpha = 0
        assert!(NormalInverseGamma::new(0.0, 1.0, 1.0, 0.0).is_err()); // beta = 0
    }

    #[test]
    fn test_normal_inverse_gamma_update() {
        let mut model = NormalInverseGamma::noninformative();
        model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

        // Parameters should have been updated
        assert!(model.kappa() > 0.001); // Precision increased
        assert!(model.alpha() > 0.001); // Shape increased
    }

    #[test]
    fn test_normal_inverse_gamma_posterior_mean_mu() {
        let mut model = NormalInverseGamma::noninformative();
        let data = vec![4.2, 5.8, 6.1, 4.5, 5.0];
        model.update(&data);

        let mean_mu = model.posterior_mean_mu();
        let sample_mean = data.iter().sum::<f32>() / data.len() as f32;

        // With weak prior, posterior mean should be close to sample mean
        assert!((mean_mu - sample_mean).abs() < 0.1);
    }

    #[test]
    fn test_normal_inverse_gamma_posterior_mean_variance() {
        let mut model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");
        model.update(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let mean_var = model.posterior_mean_variance().expect("Alpha > 1");
        assert!(mean_var > 0.0); // Variance is positive
    }

    #[test]
    fn test_normal_inverse_gamma_no_variance_for_weak_prior() {
        let model = NormalInverseGamma::noninformative();
        // Alpha = 0.001 < 1, so mean variance is undefined
        assert!(model.posterior_mean_variance().is_none());
        assert!(model.posterior_variance_mu().is_none());
    }

    #[test]
    fn test_normal_inverse_gamma_posterior_variance_mu() {
        let mut model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");
        model.update(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        let var_mu = model.posterior_variance_mu().expect("Alpha > 1");
        assert!(var_mu > 0.0); // Positive uncertainty
    }

    #[test]
    fn test_normal_inverse_gamma_predictive() {
        let mut model = NormalInverseGamma::noninformative();
        model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

        let pred_mean = model.posterior_predictive();
        let mean_mu = model.posterior_mean_mu();

        // Predictive mean equals posterior mean of μ
        assert!((pred_mean - mean_mu).abs() < 1e-6);
    }

    #[test]
    fn test_normal_inverse_gamma_credible_interval() {
        let mut model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");
        model.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

        let (lower, upper) = model
            .credible_interval_mu(0.95)
            .expect("Valid confidence level");

        let mean = model.posterior_mean_mu();

        // Mean should be within interval
        assert!(lower < mean);
        assert!(mean < upper);
    }

    #[test]
    fn test_normal_inverse_gamma_credible_interval_invalid() {
        let model = NormalInverseGamma::new(0.0, 1.0, 3.0, 2.0).expect("Valid parameters");

        assert!(model.credible_interval_mu(-0.1).is_err()); // Invalid confidence
        assert!(model.credible_interval_mu(1.1).is_err()); // Invalid confidence
    }

    #[test]
    fn test_normal_inverse_gamma_credible_interval_requires_alpha_gt_1() {
        let model = NormalInverseGamma::noninformative(); // Alpha = 0.001 < 1

        // Should fail because alpha <= 1
        assert!(model.credible_interval_mu(0.95).is_err());
    }

    #[test]
    fn test_normal_inverse_gamma_sequential_updates() {
        let mut model = NormalInverseGamma::noninformative();

        // First batch
        model.update(&[4.2, 5.8]);
        let kappa_1 = model.kappa();
        let alpha_1 = model.alpha();

        // Second batch
        model.update(&[6.1, 4.5, 5.0]);
        let kappa_2 = model.kappa();
        let alpha_2 = model.alpha();

        // Precision should increase with more data
        assert!(kappa_2 > kappa_1);
        assert!(alpha_2 > alpha_1);
    }

    #[test]
    fn test_normal_inverse_gamma_empty_update() {
        let mut model = NormalInverseGamma::noninformative();
        let original_mu = model.mu();
        let original_kappa = model.kappa();

        // Empty data should not change parameters
        model.update(&[]);

        assert_eq!(model.mu(), original_mu);
        assert_eq!(model.kappa(), original_kappa);
    }

    #[test]
    fn test_normal_inverse_gamma_strong_prior_influence() {
        // Strong prior centered at 10.0
        let mut strong_prior =
            NormalInverseGamma::new(10.0, 100.0, 50.0, 50.0).expect("Valid parameters");

        // Observe data far from prior (mean ≈ 5.0)
        strong_prior.update(&[4.2, 5.8, 6.1, 4.5, 5.0]);

        let posterior_mean = strong_prior.posterior_mean_mu();

        // Posterior should be pulled toward prior (closer to 10.0 than 5.0)
        assert!(posterior_mean > 7.0); // Strong prior dominates
    }

    // ========== Dirichlet-Multinomial Tests ==========

    #[test]
    fn test_dirichlet_multinomial_uniform_prior() {
        let prior = DirichletMultinomial::uniform(3);
        assert_eq!(prior.num_categories(), 3);
        assert_eq!(prior.alphas()[0], 1.0);
        assert_eq!(prior.alphas()[1], 1.0);
        assert_eq!(prior.alphas()[2], 1.0);
    }

    #[test]
    #[should_panic(expected = "Must have at least 2 categories")]
    fn test_dirichlet_multinomial_requires_2_categories() {
        let _ = DirichletMultinomial::uniform(1);
    }

    #[test]
    fn test_dirichlet_multinomial_custom_prior() {
        let prior = DirichletMultinomial::new(vec![5.0, 3.0, 2.0]).expect("Valid parameters");
        assert_eq!(prior.num_categories(), 3);
        assert_eq!(prior.alphas()[0], 5.0);
        assert_eq!(prior.alphas()[1], 3.0);
        assert_eq!(prior.alphas()[2], 2.0);
    }

    #[test]
    fn test_dirichlet_multinomial_invalid_prior() {
        assert!(DirichletMultinomial::new(vec![0.0, 1.0, 1.0]).is_err()); // Zero alpha
        assert!(DirichletMultinomial::new(vec![-1.0, 1.0, 1.0]).is_err()); // Negative alpha
        assert!(DirichletMultinomial::new(vec![1.0]).is_err()); // Only 1 category
    }

    #[test]
    fn test_dirichlet_multinomial_update() {
        let mut model = DirichletMultinomial::uniform(3);
        model.update(&[10, 5, 3]);

        // Posterior should be Dirichlet(1+10, 1+5, 1+3) = Dirichlet(11, 6, 4)
        assert_eq!(model.alphas()[0], 11.0);
        assert_eq!(model.alphas()[1], 6.0);
        assert_eq!(model.alphas()[2], 4.0);
    }

    #[test]
    #[should_panic(expected = "Counts must match number of categories")]
    fn test_dirichlet_multinomial_update_wrong_size() {
        let mut model = DirichletMultinomial::uniform(3);
        model.update(&[10, 5]); // Only 2 counts for 3 categories
    }

    #[test]
    fn test_dirichlet_multinomial_posterior_mean() {
        let mut model = DirichletMultinomial::uniform(3);
        model.update(&[10, 5, 3]);

        let mean = model.posterior_mean();

        // Should be [11/21, 6/21, 4/21]
        assert!((mean[0] - 11.0 / 21.0).abs() < 1e-6);
        assert!((mean[1] - 6.0 / 21.0).abs() < 1e-6);
        assert!((mean[2] - 4.0 / 21.0).abs() < 1e-6);

        // Sum should be 1.0
        assert!((mean.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dirichlet_multinomial_posterior_mode() {
        let mut model = DirichletMultinomial::new(vec![2.0, 2.0, 2.0]).expect("Valid parameters");
        model.update(&[10, 5, 3]);

        // Posterior is Dirichlet(12, 7, 5), all > 1
        // Mode = [(12-1)/(24-3), (7-1)/(24-3), (5-1)/(24-3)]
        let mode = model.posterior_mode().expect("Mode should exist");

        assert!((mode[0] - 11.0 / 21.0).abs() < 1e-6);
        assert!((mode[1] - 6.0 / 21.0).abs() < 1e-6);
        assert!((mode[2] - 4.0 / 21.0).abs() < 1e-6);
    }

    #[test]
    fn test_dirichlet_multinomial_no_mode_for_uniform() {
        let model = DirichletMultinomial::uniform(3);
        // Dirichlet(1, 1, 1) has no unique mode
        assert!(model.posterior_mode().is_none());
    }

    #[test]
    fn test_dirichlet_multinomial_posterior_variance() {
        let mut model = DirichletMultinomial::uniform(3);
        model.update(&[10, 5, 3]);

        let variance = model.posterior_variance();

        // All variances should be positive
        assert!(variance[0] > 0.0);
        assert!(variance[1] > 0.0);
        assert!(variance[2] > 0.0);

        // Higher probability categories should have higher variance
        assert!(variance[0] > variance[2]); // Category 0 has highest prob
    }

    #[test]
    fn test_dirichlet_multinomial_predictive() {
        let mut model = DirichletMultinomial::uniform(3);
        model.update(&[10, 5, 3]);

        let pred = model.posterior_predictive();
        let mean = model.posterior_mean();

        // Predictive should equal posterior mean
        for i in 0..3 {
            assert!((pred[i] - mean[i]).abs() < 1e-6);
        }

        // Sum should be 1.0
        assert!((pred.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dirichlet_multinomial_credible_intervals() {
        let mut model = DirichletMultinomial::uniform(3);
        model.update(&[10, 5, 3]);

        let intervals = model
            .credible_intervals(0.95)
            .expect("Valid confidence level");
        let mean = model.posterior_mean();

        // Mean should be within interval for each category
        for i in 0..3 {
            assert!(intervals[i].0 < mean[i]);
            assert!(mean[i] < intervals[i].1);

            // Bounds should be in [0, 1]
            assert!((0.0..=1.0).contains(&intervals[i].0));
            assert!((0.0..=1.0).contains(&intervals[i].1));
        }
    }

    #[test]
    fn test_dirichlet_multinomial_credible_intervals_invalid() {
        let model = DirichletMultinomial::uniform(3);

        assert!(model.credible_intervals(-0.1).is_err());
        assert!(model.credible_intervals(1.1).is_err());
    }

    #[test]
    fn test_dirichlet_multinomial_sequential_updates() {
        let mut model = DirichletMultinomial::uniform(3);

        // First batch: [10, 5, 3]
        model.update(&[10, 5, 3]);
        assert_eq!(model.alphas()[0], 11.0);

        // Second batch: [5, 10, 2]
        model.update(&[5, 10, 2]);
        assert_eq!(model.alphas()[0], 16.0); // 11 + 5
        assert_eq!(model.alphas()[1], 16.0); // 6 + 10
        assert_eq!(model.alphas()[2], 6.0); // 4 + 2
    }

    #[test]
    fn test_dirichlet_multinomial_empty_update() {
        let mut model = DirichletMultinomial::uniform(3);
        let original_alphas = model.alphas().to_vec();

        // Empty update (all zeros)
        model.update(&[0, 0, 0]);

        // Alphas should remain unchanged
        assert_eq!(model.alphas()[0], original_alphas[0]);
        assert_eq!(model.alphas()[1], original_alphas[1]);
        assert_eq!(model.alphas()[2], original_alphas[2]);
    }

    #[test]
    fn test_dirichlet_multinomial_many_categories() {
        let k = 10;
        let mut model = DirichletMultinomial::uniform(k);

        // Update with some counts
        let counts = vec![5, 3, 8, 2, 10, 1, 4, 6, 7, 9];
        model.update(&counts);

        let mean = model.posterior_mean();

        // Check properties
        assert_eq!(mean.len(), k);
        assert!((mean.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(mean.iter().all(|&p| p > 0.0 && p < 1.0));
    }

    #[test]
    fn test_dirichlet_multinomial_strong_prior_influence() {
        // Strong prior favoring category 0
        let mut strong_prior =
            DirichletMultinomial::new(vec![100.0, 10.0, 10.0]).expect("Valid parameters");

        // Observe data favoring category 1
        strong_prior.update(&[5, 20, 5]);

        let mean = strong_prior.posterior_mean();

        // Category 0 should still have highest probability due to strong prior
        assert!(mean[0] > mean[1]); // Prior dominates
        assert!(mean[0] > mean[2]);
    }
}
