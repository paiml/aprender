
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
    /// let prior = DirichletMultinomial::new(vec![5.0, 3.0, 2.0]).expect("valid concentration parameters");
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
    /// Panics if `counts.len()` != `num_categories()`.
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
    /// let mut model = DirichletMultinomial::new(vec![2.0, 2.0, 2.0]).expect("valid concentration parameters");
    /// model.update(&[10, 5, 3]);
    ///
    /// let mode = model.posterior_mode().expect("mode exists when all alphas > 1");
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
    /// let intervals = model.credible_intervals(0.95).expect("valid confidence level");
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
mod tests;
