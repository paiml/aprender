//! Bayesian Regression Models
//!
//! This module implements Bayesian regression with analytical posteriors:
//! - Bayesian Linear Regression (conjugate Normal-InverseGamma)
//! - Ridge regression with Bayesian interpretation
//! - Prediction intervals and uncertainty quantification

use crate::error::{AprenderError, Result};
use crate::primitives::{Matrix, Vector};

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

    /// `InverseGamma` shape parameter for noise variance prior
    noise_alpha: f32,

    /// `InverseGamma` scale parameter for noise variance prior
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
    #[must_use]
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
    /// * `noise_alpha` - `InverseGamma` shape for noise variance
    /// * `noise_beta` - `InverseGamma` scale for noise variance
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
    #[must_use]
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Posterior mean coefficients (available after fitting).
    #[must_use]
    pub fn posterior_mean(&self) -> Option<&[f32]> {
        self.posterior_mean.as_deref()
    }

    /// Estimated noise variance σ² (available after fitting).
    #[must_use]
    pub fn noise_variance(&self) -> Option<f32> {
        self.noise_variance
    }

    /// Fits the Bayesian Linear Regression using analytical posterior.
    ///
    /// Computes the posterior distribution over coefficients β given data (X, y)
    /// using the conjugate Normal-InverseGamma prior.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (n × p)
    /// * `y` - Target vector (n × 1)
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, or error if dimensions mismatch or matrix is singular
    ///
    /// # Example
    ///
    /// ```ignore
    /// use aprender::bayesian::BayesianLinearRegression;
    /// use aprender::primitives::{Matrix, Vector};
    ///
    /// let mut model = BayesianLinearRegression::new(2);
    ///
    /// let x = Matrix::from_vec(3, 2, vec![1.0, 1.0, 1.0, 2.0, 1.0, 3.0]).expect("valid matrix dimensions");
    /// let y = Vector::from_vec(vec![2.0, 3.0, 4.0]);
    ///
    /// model.fit(&x, &y).expect("fit should succeed with valid data");
    /// assert!(model.posterior_mean().is_some());
    /// ```
    pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let n = x.n_rows();
        let p = x.n_cols();

        // Validate dimensions
        if p != self.n_features {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{} features in X", self.n_features),
                actual: format!("{p} columns in X"),
            });
        }
        if n != y.len() {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{n} samples in X"),
                actual: format!("{} samples in y", y.len()),
            });
        }
        if n < p {
            return Err(AprenderError::Other(format!(
                "Need at least {p} samples for {p} features, got {n}"
            )));
        }

        // Step 1: Compute XᵀX and Xᵀy
        let xt = x.transpose();
        let xtx = xt.matmul(x).map_err(|e| AprenderError::Other(e.into()))?;
        let xty = xt.matvec(y).map_err(|e| AprenderError::Other(e.into()))?;

        // Step 2: Estimate noise variance σ² from OLS residuals
        // OLS solution: β_ols = (XᵀX)⁻¹Xᵀy
        let beta_ols = xtx
            .cholesky_solve(&xty)
            .map_err(|e| AprenderError::Other(format!("Cholesky decomposition failed: {e}")))?;

        // Compute residuals: r = y - Xβ_ols
        let y_pred = x.matvec(&beta_ols).map_err(|e| {
            AprenderError::Other(format!("Matrix-vector multiplication failed: {e}"))
        })?;

        let mut rss = 0.0_f32;
        for i in 0..n {
            let residual = y[i] - y_pred[i];
            rss += residual * residual;
        }

        // Estimate σ²: σ² = RSS / (n - p)
        let sigma2 = rss / ((n - p) as f32);

        // Step 3: Compute posterior precision
        // Σₙ⁻¹ = Σ₀⁻¹ + σ⁻²XᵀX
        // Since Σ₀ = (1/precision) * I, we have Σ₀⁻¹ = precision * I
        let prior_precision_matrix = Matrix::eye(p).mul_scalar(self.beta_prior_precision);
        let data_precision = xtx.mul_scalar(1.0 / sigma2);
        let posterior_precision_inv = prior_precision_matrix
            .add(&data_precision)
            .map_err(|e| AprenderError::Other(format!("Matrix addition failed: {e}")))?;

        // Step 4: Compute posterior mean
        // βₙ = Σₙ(Σ₀⁻¹β₀ + σ⁻²Xᵀy)
        // Right-hand side: Σ₀⁻¹β₀ + σ⁻²Xᵀy
        let mut rhs = Vec::with_capacity(p);
        for i in 0..p {
            let prior_term = self.beta_prior_mean[i] * self.beta_prior_precision;
            let data_term = xty[i] / sigma2;
            rhs.push(prior_term + data_term);
        }
        let rhs_vec = Vector::from_vec(rhs);

        // Solve Σₙ⁻¹ βₙ = rhs for βₙ
        let posterior_mean = posterior_precision_inv
            .cholesky_solve(&rhs_vec)
            .map_err(|e| AprenderError::Other(format!("Posterior mean computation failed: {e}")))?;

        // Step 5: Store results
        self.posterior_mean = Some(posterior_mean.as_slice().to_vec());
        self.noise_variance = Some(sigma2);

        // Store posterior precision matrix (for prediction intervals later)
        let precision_data: Vec<Vec<f32>> = posterior_precision_inv
            .as_slice()
            .chunks(p)
            .map(|row: &[f32]| row.to_vec())
            .collect();
        self.posterior_precision = Some(precision_data);

        Ok(())
    }

    /// Predicts target values for new data using posterior mean.
    ///
    /// # Arguments
    ///
    /// * `x_test` - Test feature matrix (`n_test` × p)
    ///
    /// # Returns
    ///
    /// Predicted values (`n_test` × 1)
    ///
    /// # Errors
    ///
    /// Returns error if model not fitted or dimensions mismatch
    ///
    /// # Example
    ///
    /// ```ignore
    /// let predictions = model.predict(&x_test).expect("prediction should succeed after fitting");
    /// ```
    pub fn predict(&self, x_test: &Matrix<f32>) -> Result<Vector<f32>> {
        let posterior_mean = self.posterior_mean.as_ref().ok_or_else(|| {
            AprenderError::Other("Model not fitted yet. Call fit() first.".into())
        })?;

        if x_test.n_cols() != self.n_features {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} columns in x_test", x_test.n_cols()),
            });
        }

        let beta = Vector::from_slice(posterior_mean);
        x_test
            .matvec(&beta)
            .map_err(|e| AprenderError::Other(format!("Prediction failed: {e}")))
    }

    /// Computes the log-likelihood of the data given current posterior parameters.
    ///
    /// Uses the posterior mean β and noise variance σ² to compute:
    /// log P(y | X, β, σ²) = -n/2 log(2π) - n/2 log(σ²) - 1/(2σ²) ||y - Xβ||²
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (n × p)
    /// * `y` - Target vector (n)
    ///
    /// # Returns
    ///
    /// Log-likelihood value, or error if model not fitted
    ///
    /// # Example
    ///
    /// ```ignore
    /// let log_lik = model.log_likelihood(&x_train, &y_train).expect("log-likelihood should succeed after fitting");
    /// ```
    pub fn log_likelihood(&self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<f32> {
        let posterior_mean = self.posterior_mean.as_ref().ok_or_else(|| {
            AprenderError::Other("Model not fitted yet. Call fit() first.".into())
        })?;

        let sigma2 = self.noise_variance.ok_or_else(|| {
            AprenderError::Other("Noise variance not available. Call fit() first.".into())
        })?;

        let n = x.n_rows() as f32;

        // Validate dimensions
        if x.n_cols() != self.n_features {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{} features", self.n_features),
                actual: format!("{} columns in x", x.n_cols()),
            });
        }
        if x.n_rows() != y.len() {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{} samples in x", x.n_rows()),
                actual: format!("{} samples in y", y.len()),
            });
        }

        // Compute predictions: ŷ = Xβ
        let beta = Vector::from_slice(posterior_mean);
        let y_pred = x
            .matvec(&beta)
            .map_err(|e| AprenderError::Other(format!("Prediction failed: {e}")))?;

        // Compute residual sum of squares: RSS = ||y - ŷ||²
        let mut rss = 0.0_f32;
        for i in 0..y.len() {
            let residual = y[i] - y_pred[i];
            rss += residual * residual;
        }

        // Log-likelihood: log P(y | X, β, σ²) = -n/2 log(2π) - n/2 log(σ²) - RSS/(2σ²)
        use std::f32::consts::PI;
        let log_lik = -0.5 * n * (2.0 * PI).ln() - 0.5 * n * sigma2.ln() - rss / (2.0 * sigma2);

        Ok(log_lik)
    }

    /// Computes the Bayesian Information Criterion (BIC).
    ///
    /// BIC = -2 * log L + k * log(n)
    ///
    /// where k is the number of parameters (p + 1 for noise variance)
    /// and n is the number of samples.
    ///
    /// Lower BIC indicates better model fit with penalty for complexity.
    ///
    /// # Arguments
    ///
    /// * `x` - Training feature matrix
    /// * `y` - Training target vector
    ///
    /// # Returns
    ///
    /// BIC value, or error if model not fitted
    ///
    /// # Example
    ///
    /// ```ignore
    /// let bic = model.bic(&x_train, &y_train).expect("BIC should succeed after fitting");
    /// ```
    pub fn bic(&self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<f32> {
        let log_lik = self.log_likelihood(x, y)?;
        let n = x.n_rows() as f32;
        let k = (self.n_features + 1) as f32; // p coefficients + 1 noise variance

        Ok(-2.0 * log_lik + k * n.ln())
    }

    /// Computes the Akaike Information Criterion (AIC).
    ///
    /// AIC = -2 * log L + 2k
    ///
    /// where k is the number of parameters (p + 1 for noise variance).
    ///
    /// Lower AIC indicates better model fit with penalty for complexity.
    /// AIC tends to prefer more complex models than BIC.
    ///
    /// # Arguments
    ///
    /// * `x` - Training feature matrix
    /// * `y` - Training target vector
    ///
    /// # Returns
    ///
    /// AIC value, or error if model not fitted
    ///
    /// # Example
    ///
    /// ```ignore
    /// let aic = model.aic(&x_train, &y_train).expect("AIC should succeed after fitting");
    /// ```
    pub fn aic(&self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<f32> {
        let log_lik = self.log_likelihood(x, y)?;
        let k = (self.n_features + 1) as f32; // p coefficients + 1 noise variance

        Ok(-2.0 * log_lik + 2.0 * k)
    }
}

#[cfg(test)]
#[path = "regression_tests.rs"]
mod tests;
