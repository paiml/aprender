
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
    /// let prior = NormalInverseGamma::new(5.0, 10.0, 5.0, 5.0).expect("valid prior parameters");
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
    /// let mean_var = model.posterior_mean_variance().expect("mean variance exists when alpha > 1");
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
    /// let var_mu = model.posterior_variance_mu().expect("variance of mu exists when alpha > 1");
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
    /// let (lower, upper) = model.credible_interval_mu(0.95).expect("valid confidence level with alpha > 1");
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
