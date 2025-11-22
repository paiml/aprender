//! Bayesian Regression Models
//!
//! This module implements Bayesian regression with analytical posteriors:
//! - Bayesian Linear Regression (conjugate Normal-InverseGamma)
//! - Ridge regression with Bayesian interpretation
//! - Prediction intervals and uncertainty quantification

use crate::error::{AprenderError, Result};

/// Bayesian Linear Regression with analytical posterior.
///
/// # Model
///
/// ```text
/// y = Xβ + ε,  ε ~ N(0, σ²I)
/// β ~ N(β₀, Σ₀)           # Prior on coefficients
/// σ² ~ InvGamma(α, β)     # Prior on noise variance
/// ```
///
/// # Posterior (Conjugate)
///
/// ```text
/// β|y,X,σ² ~ N(βₙ, Σₙ)
/// where:
///   Σₙ = (Σ₀⁻¹ + σ⁻²XᵀX)⁻¹
///   βₙ = Σₙ(Σ₀⁻¹β₀ + σ⁻²Xᵀy)
/// ```
///
/// # Example
///
/// ```ignore
/// use aprender::bayesian::BayesianLinearRegression;
///
/// // Create with weak prior
/// let mut model = BayesianLinearRegression::new(2); // 2 features
///
/// // Fit and predict methods to be implemented
/// ```
#[derive(Debug, Clone)]
#[allow(dead_code)] // Fields used after fit() implementation
pub struct BayesianLinearRegression {
    /// Number of features (including intercept if applicable)
    n_features: usize,

    /// Prior mean β₀ (default: zeros for weak prior)
    beta_prior_mean: Vec<f32>,

    /// Prior covariance Σ₀ (default: large diagonal for weak prior)
    beta_prior_precision: f32, // Simplified: Σ₀ = (1/precision) * I

    /// InverseGamma shape parameter for noise variance prior
    noise_alpha: f32,

    /// InverseGamma scale parameter for noise variance prior
    noise_beta: f32,

    /// Posterior mean βₙ (after fitting)
    posterior_mean: Option<Vec<f32>>,

    /// Posterior covariance Σₙ (stored as precision matrix for efficiency)
    posterior_precision: Option<Vec<Vec<f32>>>,

    /// Estimated noise variance σ²
    noise_variance: Option<f32>,
}

impl BayesianLinearRegression {
    /// Creates a new Bayesian Linear Regression with weakly informative priors.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features (not including intercept)
    ///
    /// # Returns
    ///
    /// Model with weak priors: β ~ N(0, 100²I), σ² ~ InvGamma(0.001, 0.001)
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BayesianLinearRegression;
    ///
    /// let model = BayesianLinearRegression::new(3); // 3 features
    /// ```
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            beta_prior_mean: vec![0.0; n_features],
            beta_prior_precision: 0.0001, // Very weak prior (variance = 10,000)
            noise_alpha: 0.001,           // Noninformative
            noise_beta: 0.001,            // Noninformative
            posterior_mean: None,
            posterior_precision: None,
            noise_variance: None,
        }
    }

    /// Creates model with custom prior parameters.
    ///
    /// # Arguments
    ///
    /// * `n_features` - Number of features
    /// * `beta_prior_mean` - Prior mean for coefficients
    /// * `beta_prior_precision` - Prior precision (inverse variance) for coefficients
    /// * `noise_alpha` - InverseGamma shape for noise variance
    /// * `noise_beta` - InverseGamma scale for noise variance
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BayesianLinearRegression;
    ///
    /// // Ridge-like prior with precision 1.0 (variance = 1.0)
    /// let model = BayesianLinearRegression::with_prior(
    ///     3,
    ///     vec![0.0, 0.0, 0.0],
    ///     1.0,  // Ridge regularization
    ///     3.0,
    ///     2.0,
    /// ).expect("Invalid prior parameters");
    /// ```
    pub fn with_prior(
        n_features: usize,
        beta_prior_mean: Vec<f32>,
        beta_prior_precision: f32,
        noise_alpha: f32,
        noise_beta: f32,
    ) -> Result<Self> {
        if beta_prior_mean.len() != n_features {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{n_features} features"),
                actual: format!("{} elements in beta_prior_mean", beta_prior_mean.len()),
            });
        }

        if beta_prior_precision <= 0.0 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "beta_prior_precision".to_string(),
                value: beta_prior_precision.to_string(),
                constraint: "must be > 0".to_string(),
            });
        }

        if noise_alpha <= 0.0 || noise_beta <= 0.0 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "noise_alpha or noise_beta".to_string(),
                value: format!("α={noise_alpha}, β={noise_beta}"),
                constraint: "both must be > 0".to_string(),
            });
        }

        Ok(Self {
            n_features,
            beta_prior_mean,
            beta_prior_precision,
            noise_alpha,
            noise_beta,
            posterior_mean: None,
            posterior_precision: None,
            noise_variance: None,
        })
    }

    /// Number of features (excluding intercept).
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Posterior mean coefficients (available after fitting).
    pub fn posterior_mean(&self) -> Option<&[f32]> {
        self.posterior_mean.as_deref()
    }

    /// Estimated noise variance σ² (available after fitting).
    pub fn noise_variance(&self) -> Option<f32> {
        self.noise_variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let model = BayesianLinearRegression::new(3);
        assert_eq!(model.n_features(), 3);
        assert_eq!(model.beta_prior_mean.len(), 3);
        assert!(model.posterior_mean().is_none());
    }

    #[test]
    fn test_with_prior_valid() {
        let model = BayesianLinearRegression::with_prior(2, vec![1.0, 2.0], 1.0, 3.0, 2.0);
        assert!(model.is_ok());
        let model = model.expect("Should be valid");
        assert_eq!(model.n_features(), 2);
    }

    #[test]
    fn test_with_prior_dimension_mismatch() {
        let result = BayesianLinearRegression::with_prior(
            3,
            vec![1.0, 2.0], // Only 2 elements, but n_features=3
            1.0,
            3.0,
            2.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_with_prior_invalid_precision() {
        let result = BayesianLinearRegression::with_prior(
            2,
            vec![1.0, 2.0],
            -1.0, // Invalid: must be > 0
            3.0,
            2.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_with_prior_invalid_noise_params() {
        let result = BayesianLinearRegression::with_prior(
            2,
            vec![1.0, 2.0],
            1.0,
            -1.0, // Invalid alpha
            2.0,
        );
        assert!(result.is_err());
    }
}
