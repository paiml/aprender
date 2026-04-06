
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
    /// let prior = GammaPoisson::new(50.0, 10.0).expect("valid alpha and beta parameters");
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
    /// let mut model = GammaPoisson::new(2.0, 1.0).expect("valid alpha and beta parameters");
    /// model.update(&[3, 5, 4, 6, 2]);
    ///
    /// let mode = model.posterior_mode().expect("mode exists when alpha > 1");
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
    /// let (lower, upper) = model.credible_interval(0.95).expect("valid confidence level");
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
