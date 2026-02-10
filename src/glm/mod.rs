//! Generalized Linear Models (GLM)
//!
//! GLMs extend linear regression to exponential family distributions with link functions.
//!
//! # Components
//!
//! - **Linear predictor**: η = Xβ
//! - **Link function**: g(μ) = η, maps mean to linear predictor
//! - **Family**: Distribution from exponential family
//!
//! # Families
//!
//! - **Poisson**: Count data, canonical link = log, V(μ) = μ
//! - **Negative Binomial**: Overdispersed count data, canonical link = log, V(μ) = μ + α*μ²
//! - **Gamma**: Positive continuous data, canonical link = inverse, V(μ) = μ²
//! - **Binomial**: Binary/proportion data, canonical link = logit, V(μ) = μ(1-μ)
//!
//! # Overdispersion in Count Data
//!
//! For count data where variance >> mean (overdispersion), use **Negative Binomial**
//! instead of Poisson. The dispersion parameter α controls extra variance.
//! See `notes-poisson.md` for detailed explanation with peer-reviewed references.
//!
//! # Example
//!
//! ```ignore
//! use aprender::glm::{GLM, Family};
//! use aprender::primitives::{Matrix, Vector};
//!
//! // Overdispersed count data - use Negative Binomial
//! let mut model = GLM::new(Family::NegativeBinomial).with_dispersion(0.5);
//! model.fit(&x, &y).expect("GLM fit should succeed");
//! let predictions = model.predict(&x_test).expect("GLM predict should succeed");
//! ```

use crate::error::{AprenderError, Result};
use crate::primitives::{Matrix, Vector};

/// Exponential family distribution for GLM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Family {
    /// Poisson distribution for count data.
    /// Canonical link: log, Variance: V(μ) = μ
    Poisson,

    /// Negative Binomial distribution for overdispersed count data.
    /// Canonical link: log, Variance: V(μ) = μ + α*μ² (α = dispersion parameter)
    /// Handles data where variance >> mean (overdispersion).
    NegativeBinomial,

    /// Gamma distribution for positive continuous data.
    /// Canonical link: inverse, Variance: V(μ) = μ²
    Gamma,

    /// Binomial distribution for binary/proportion data.
    /// Canonical link: logit, Variance: V(μ) = μ(1-μ)
    Binomial,
}

/// Link function for GLM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Link {
    /// Log link: η = log(μ), μ = exp(η)
    Log,

    /// Inverse link: η = 1/μ, μ = 1/η
    Inverse,

    /// Logit link: η = log(μ/(1-μ)), μ = 1/(1+exp(-η))
    Logit,

    /// Identity link: η = μ, μ = η
    Identity,
}

impl Family {
    /// Returns the canonical link function for this family.
    #[must_use]
    pub const fn canonical_link(&self) -> Link {
        match self {
            Self::Poisson | Self::NegativeBinomial => Link::Log,
            Self::Gamma => Link::Inverse,
            Self::Binomial => Link::Logit,
        }
    }

    /// Variance function V(μ).
    ///
    /// For Negative Binomial, requires dispersion parameter α.
    fn variance(self, mu: f32, dispersion: f32) -> f32 {
        match self {
            Self::Poisson => mu,                                 // V(μ) = μ
            Self::NegativeBinomial => mu + dispersion * mu * mu, // V(μ) = μ + α*μ²
            Self::Gamma => mu * mu,                              // V(μ) = μ²
            Self::Binomial => mu * (1.0 - mu),                   // V(μ) = μ(1-μ)
        }
    }

    /// Validates that y values are appropriate for this family.
    fn validate_response(self, y: &Vector<f32>) -> Result<()> {
        match self {
            Self::Poisson | Self::NegativeBinomial => {
                // Must be non-negative counts
                for &val in y.as_slice() {
                    if val < 0.0 {
                        let family_name = if matches!(self, Self::Poisson) {
                            "Poisson"
                        } else {
                            "Negative Binomial"
                        };
                        return Err(AprenderError::Other(format!(
                            "{family_name} requires non-negative counts, got {val}"
                        )));
                    }
                }
            }
            Self::Gamma => {
                // Must be positive
                for &val in y.as_slice() {
                    if val <= 0.0 {
                        return Err(AprenderError::Other(format!(
                            "Gamma requires positive values, got {val}"
                        )));
                    }
                }
            }
            Self::Binomial => {
                // Must be in [0, 1]
                for &val in y.as_slice() {
                    if !(0.0..=1.0).contains(&val) {
                        return Err(AprenderError::Other(format!(
                            "Binomial requires values in [0,1], got {val}"
                        )));
                    }
                }
            }
        }
        Ok(())
    }
}

impl Link {
    /// Applies the link function: η = g(μ).
    fn link(self, mu: f32) -> f32 {
        match self {
            Self::Log => mu.ln(),
            Self::Inverse => 1.0 / mu,
            Self::Logit => (mu / (1.0 - mu)).ln(),
            Self::Identity => mu,
        }
    }

    /// Inverse link function: μ = g⁻¹(η).
    fn inverse_link(self, eta: f32) -> f32 {
        match self {
            Self::Log => eta.exp(),
            Self::Inverse => 1.0 / eta,
            Self::Logit => 1.0 / (1.0 + (-eta).exp()),
            Self::Identity => eta,
        }
    }

    /// Derivative of inverse link: dμ/dη.
    fn derivative(self, eta: f32) -> f32 {
        match self {
            Self::Log => eta.exp(),
            Self::Inverse => -1.0 / (eta * eta),
            Self::Logit => {
                let mu = self.inverse_link(eta);
                mu * (1.0 - mu)
            }
            Self::Identity => 1.0,
        }
    }
}

/// Generalized Linear Model.
///
/// Fits regression models for exponential family distributions using IRLS
/// (Iteratively Reweighted Least Squares).
#[derive(Debug, Clone)]
pub struct GLM {
    /// Distribution family
    family: Family,

    /// Link function
    link: Link,

    /// Maximum iterations for IRLS
    max_iter: usize,

    /// Convergence tolerance
    tol: f32,

    /// Dispersion parameter for Negative Binomial (α in V(μ) = μ + α*μ²)
    /// Default: 1.0 (moderate overdispersion)
    dispersion: f32,

    /// Fitted coefficients
    coefficients: Option<Vec<f32>>,

    /// Intercept term
    intercept: Option<f32>,
}

impl GLM {
    /// Creates a new GLM with the specified family and its canonical link.
    #[must_use]
    pub fn new(family: Family) -> Self {
        Self {
            family,
            link: family.canonical_link(),
            max_iter: 1000,  // Increased for better convergence
            tol: 1e-3,       // More relaxed tolerance for practical convergence
            dispersion: 1.0, // Default dispersion for Negative Binomial
            coefficients: None,
            intercept: None,
        }
    }

    /// Creates a GLM with a custom link function.
    #[must_use]
    pub fn with_link(mut self, link: Link) -> Self {
        self.link = link;
        self
    }

    /// Sets the maximum number of IRLS iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the dispersion parameter for Negative Binomial family.
    ///
    /// The dispersion parameter α controls overdispersion: V(μ) = μ + α*μ².
    /// Higher α = more overdispersion. Default: 1.0.
    #[must_use]
    pub fn with_dispersion(mut self, dispersion: f32) -> Self {
        self.dispersion = dispersion;
        self
    }

    /// Returns the fitted coefficients.
    #[must_use]
    pub fn coefficients(&self) -> Option<&[f32]> {
        self.coefficients.as_deref()
    }

    /// Returns the intercept.
    #[must_use]
    pub fn intercept(&self) -> Option<f32> {
        self.intercept
    }

    /// Fits the GLM using IRLS (Iteratively Reweighted Least Squares).
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (n × p)
    /// * `y` - Response vector (n)
    ///
    /// # Returns
    ///
    /// `Ok(())` on successful convergence, error otherwise
    #[allow(clippy::needless_range_loop)]
    #[allow(clippy::too_many_lines)] // IRLS algorithm requires many steps
    pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let n = x.n_rows();
        let p = x.n_cols();

        // Validate dimensions
        if n != y.len() {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{n} samples in X"),
                actual: format!("{} samples in y", y.len()),
            });
        }

        // Validate response values for family
        self.family.validate_response(y)?;

        // Initialize coefficients to zero
        let mut beta = vec![0.0_f32; p];

        // Initialize intercept and eta based on response mean
        let y_mean = y.as_slice().iter().sum::<f32>() / n as f32;
        let y_mean_safe = y_mean.clamp(0.01, 0.99); // Avoid extreme values
        let mut intercept = self.link.link(y_mean_safe);

        // Initialize linear predictor η
        let mut eta = vec![intercept; n];

        // IRLS algorithm
        for _iter in 0..self.max_iter {
            // Compute μ = g⁻¹(η) and clamp to valid ranges
            let mu: Vec<f32> = eta
                .iter()
                .map(|&e| {
                    let mu_raw = self.link.inverse_link(e);
                    // Clamp mu to reasonable ranges based on family
                    match self.family {
                        Family::Poisson | Family::NegativeBinomial | Family::Gamma => {
                            mu_raw.max(1e-6) // Must be positive
                        }
                        Family::Binomial => mu_raw.clamp(1e-6, 1.0 - 1e-6), // Must be in (0,1)
                    }
                })
                .collect();

            // Compute working response z = η + (y - μ) * g'(η)
            let mut z = Vec::with_capacity(n);
            for i in 0..n {
                let deriv = self.link.derivative(eta[i]);
                z.push(eta[i] + (y[i] - mu[i]) * deriv);
            }

            // Compute weights W = 1 / (V(μ) * [g'(η)]²)
            let mut weights = Vec::with_capacity(n);
            for i in 0..n {
                let var = self.family.variance(mu[i], self.dispersion).max(1e-10); // Avoid zero variance
                let deriv = self.link.derivative(eta[i]);
                let weight = 1.0 / (var * deriv * deriv + 1e-10); // Add epsilon for stability
                                                                  // Clamp weights to reasonable range to avoid numerical issues
                let weight_clamped = weight.clamp(1e-6, 1e6);
                weights.push(weight_clamped);
            }

            // Weighted least squares: solve (X'WX + εI)β = X'Wz for numerical stability
            // Build augmented design matrix [1 X] to include intercept
            let mut x_aug_data = Vec::with_capacity(n * (p + 1));
            for i in 0..n {
                x_aug_data.push(1.0); // Intercept column
                for j in 0..p {
                    x_aug_data.push(x.get(i, j));
                }
            }
            let x_aug = Matrix::from_vec(n, p + 1, x_aug_data)
                .map_err(|e| AprenderError::Other(format!("Augmented matrix error: {e}")))?;

            // Compute X'W
            let mut xtw_data = Vec::with_capacity((p + 1) * n);
            for j in 0..=p {
                for i in 0..n {
                    xtw_data.push(x_aug.get(i, j) * weights[i].sqrt());
                }
            }
            let xtw = Matrix::from_vec(p + 1, n, xtw_data)
                .map_err(|e| AprenderError::Other(format!("X'W matrix error: {e}")))?;

            // Compute W^(1/2) X (for normal equations)
            let mut wx_data = Vec::with_capacity(n * (p + 1));
            for i in 0..n {
                for j in 0..=p {
                    wx_data.push(weights[i].sqrt() * x_aug.get(i, j));
                }
            }
            let wx = Matrix::from_vec(n, p + 1, wx_data)
                .map_err(|e| AprenderError::Other(format!("WX matrix error: {e}")))?;

            // Compute W^(1/2) z
            let wz = z
                .iter()
                .enumerate()
                .map(|(i, &zi)| weights[i].sqrt() * zi)
                .collect::<Vec<_>>();
            let wz_vec = Vector::from_vec(wz);

            // Solve (X'WX)β_aug = X'Wz using normal equations
            // (W^(1/2)X)' (W^(1/2)X) β_aug = (W^(1/2)X)' W^(1/2)z
            let mut xtwx = xtw
                .matmul(&wx)
                .map_err(|e| AprenderError::Other(format!("X'WX computation failed: {e}")))?;
            let xtwz = xtw
                .matvec(&wz_vec)
                .map_err(|e| AprenderError::Other(format!("X'Wz computation failed: {e}")))?;

            // Add ridge regularization: X'WX + λI for numerical stability
            // This ensures positive definiteness for Cholesky decomposition
            // Scale ridge based on diagonal magnitude
            let max_diag = (0..=p)
                .map(|i| xtwx.get(i, i).abs())
                .fold(0.0_f32, f32::max);
            let ridge = (max_diag * 1e-6).max(1e-8); // Larger adaptive ridge for stability
            for i in 0..=p {
                let old_val = xtwx.get(i, i);
                xtwx.set(i, i, old_val + ridge);
            }

            // Solve with Cholesky
            let beta_aug = xtwx
                .cholesky_solve(&xtwz)
                .map_err(|e| AprenderError::Other(format!("Cholesky solve failed: {e}")))?;

            // Extract intercept and coefficients
            let intercept_new = beta_aug[0];
            let beta_new = beta_aug.as_slice()[1..].to_vec();

            // Step damping for log link to prevent divergence
            // Log link can cause numerical instability regardless of family
            let step_size = match self.link {
                Link::Log => 0.5, // Damped steps for log link
                _ => 1.0,         // Full steps for other links
            };

            let intercept_damped = intercept + step_size * (intercept_new - intercept);
            let beta_damped: Vec<f32> = beta
                .iter()
                .zip(&beta_new)
                .map(|(old, new)| old + step_size * (new - old))
                .collect();

            // Update linear predictor with damped steps
            for i in 0..n {
                let mut new_eta = intercept_damped;
                for j in 0..p {
                    new_eta += x.get(i, j) * beta_damped[j];
                }
                eta[i] = new_eta;
            }

            // Check convergence
            let mut max_change = (intercept_damped - intercept).abs();
            for j in 0..p {
                max_change = max_change.max((beta_damped[j] - beta[j]).abs());
            }

            beta = beta_damped;
            intercept = intercept_damped;

            if max_change < self.tol {
                self.coefficients = Some(beta);
                self.intercept = Some(intercept);
                return Ok(());
            }
        }

        Err(AprenderError::Other(format!(
            "GLM IRLS did not converge in {} iterations",
            self.max_iter
        )))
    }

    /// Predicts mean response for test data.
    ///
    /// Returns μ = g⁻¹(Xβ + β₀).
    #[allow(clippy::needless_range_loop)]
    pub fn predict(&self, x_test: &Matrix<f32>) -> Result<Vector<f32>> {
        let beta = self.coefficients.as_ref().ok_or_else(|| {
            AprenderError::Other("Model not fitted yet. Call fit() first.".into())
        })?;

        let intercept = self.intercept.ok_or_else(|| {
            AprenderError::Other("Model not fitted yet. Call fit() first.".into())
        })?;

        let n = x_test.n_rows();
        let p = x_test.n_cols();

        if p != beta.len() {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{} features", beta.len()),
                actual: format!("{p} columns in x_test"),
            });
        }

        let mut predictions = Vec::with_capacity(n);
        for i in 0..n {
            let mut eta = intercept;
            for j in 0..p {
                eta += x_test.get(i, j) * beta[j];
            }
            let mu = self.link.inverse_link(eta);
            predictions.push(mu);
        }

        Ok(Vector::from_vec(predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Family canonical links
    #[test]
    fn test_family_canonical_links() {
        assert_eq!(Family::Poisson.canonical_link(), Link::Log);
        assert_eq!(Family::Gamma.canonical_link(), Link::Inverse);
        assert_eq!(Family::Binomial.canonical_link(), Link::Logit);
    }

    /// Test: Link functions and inverses
    #[test]
    fn test_link_functions() {
        // Log link
        let link = Link::Log;
        let mu = 5.0;
        let eta = link.link(mu);
        assert!((eta - mu.ln()).abs() < 1e-6);
        assert!((link.inverse_link(eta) - mu).abs() < 1e-6);

        // Inverse link
        let link = Link::Inverse;
        let mu = 2.0;
        let eta = link.link(mu);
        assert!((eta - 1.0 / mu).abs() < 1e-6);
        assert!((link.inverse_link(eta) - mu).abs() < 1e-6);

        // Logit link
        let link = Link::Logit;
        let mu = 0.7;
        let eta = link.link(mu);
        assert!((link.inverse_link(eta) - mu).abs() < 1e-6);

        // Identity link
        let link = Link::Identity;
        let mu = 3.0;
        assert_eq!(link.link(mu), mu);
        assert_eq!(link.inverse_link(mu), mu);
    }

    /// Test: Negative Binomial regression on overdispersed count data
    ///
    /// Negative Binomial handles count data where variance >> mean (overdispersion).
    /// This is the proper solution for overdispersed counts, as documented in notes-poisson.md.
    /// V(μ) = μ + α*μ² where α is the dispersion parameter.
    #[test]
    fn test_negative_binomial_regression() {
        // Gentle linear count data with slight overdispersion
        let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

        let mut model = GLM::new(Family::NegativeBinomial)
            .with_dispersion(0.1)
            .with_max_iter(5000); // More iterations for damped convergence
        let result = model.fit(&x, &y);

        assert!(
            result.is_ok(),
            "Negative Binomial GLM should fit, error: {:?}",
            result.err()
        );
        assert!(model.coefficients().is_some());
        assert!(model.intercept().is_some());

        // Verify predictions work
        let predictions = model.predict(&x).expect("Predictions should succeed");
        assert_eq!(predictions.len(), y.len());
    }

    /// Test: Negative Binomial with low dispersion (approaches Poisson)
    #[test]
    fn test_negative_binomial_low_dispersion() {
        // Low overdispersion data
        let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        let mut model = GLM::new(Family::NegativeBinomial)
            .with_dispersion(0.01) // Very low dispersion, close to Poisson
            .with_max_iter(5000);
        let result = model.fit(&x, &y);

        assert!(result.is_ok(), "Low dispersion NB should converge");
        assert!(model.coefficients().is_some());
    }

    /// Test: Negative Binomial validation rejects negative counts
    #[test]
    fn test_negative_binomial_validation() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![3.0, -1.0, 5.0]); // Negative count

        let mut model = GLM::new(Family::NegativeBinomial);
        let result = model.fit(&x, &y);

        assert!(
            result.is_err(),
            "Negative Binomial should reject negative counts"
        );
        assert!(result
            .expect_err("Should return error for negative counts")
            .to_string()
            .contains("Negative Binomial requires non-negative counts"));
    }

    /// Test: Gamma regression on positive continuous data
    #[test]
    fn test_gamma_regression() {
        // Simulated positive continuous data
        let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Valid matrix");
        let y = Vector::from_vec(vec![2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.75, 0.7]);

        let mut model = GLM::new(Family::Gamma); // Use default max_iter (1000)
        let result = model.fit(&x, &y);

        assert!(result.is_ok(), "Gamma GLM should fit");
        assert!(model.coefficients().is_some());

        // Predictions should be positive
        let predictions = model.predict(&x).expect("Predictions should succeed");
        for &pred in predictions.as_slice() {
            assert!(pred > 0.0, "Gamma predictions should be positive");
        }
    }

    /// Test: Binomial regression on proportions
    #[test]
    fn test_binomial_regression() {
        // Simulated binary/proportion data
        let x = Matrix::from_vec(8, 1, vec![-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
            .expect("Valid matrix");
        let y = Vector::from_vec(vec![0.1, 0.15, 0.25, 0.35, 0.65, 0.75, 0.85, 0.9]);

        let mut model = GLM::new(Family::Binomial); // Use default max_iter (1000)
        let result = model.fit(&x, &y);

        assert!(result.is_ok(), "Binomial GLM should fit");
        assert!(model.coefficients().is_some());

        // Predictions should be in [0, 1]
        let predictions = model.predict(&x).expect("Predictions should succeed");
        for &pred in predictions.as_slice() {
            assert!(
                (0.0..=1.0).contains(&pred),
                "Binomial predictions should be in [0,1], got {pred}"
            );
        }
    }

    /// Test: Invalid response for Poisson (negative values)
    #[test]
    fn test_poisson_invalid_response() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![1.0, -2.0, 3.0]); // Negative value!

        let mut model = GLM::new(Family::Poisson);
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }

    /// Test: Invalid response for Gamma (non-positive values)
    #[test]
    fn test_gamma_invalid_response() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![1.0, 0.0, 3.0]); // Zero value!

        let mut model = GLM::new(Family::Gamma);
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }

    /// Test: Invalid response for Binomial (out of [0,1])
    #[test]
    fn test_binomial_invalid_response() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.5, 1.2, 0.3]); // 1.2 out of range!

        let mut model = GLM::new(Family::Binomial);
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }

    /// Test: Dimension mismatch in fit
    #[test]
    fn test_fit_dimension_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Valid matrix");
        let y = Vector::from_vec(vec![1.0, 2.0]); // Wrong size!

        let mut model = GLM::new(Family::Poisson);
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: Predict before fit
    #[test]
    fn test_predict_not_fitted() {
        let model = GLM::new(Family::Poisson);
        let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix");

        let result = model.predict(&x_test);
        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }

    /// Test: Predict with dimension mismatch
    #[test]
    fn test_predict_dimension_mismatch() {
        // Simpler data for 2-feature model
        let x = Matrix::from_vec(
            6,
            2,
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0],
        )
        .expect("Valid matrix");
        let y = Vector::from_vec(vec![2.0, 3.0, 3.0, 4.0, 5.0, 6.0]);

        let mut model = GLM::new(Family::Poisson);
        model.fit(&x, &y).expect("Fit succeeds");

        // Try to predict with wrong number of features
        let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid test matrix");
        let result = model.predict(&x_test);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: Custom link function
    #[test]
    fn test_custom_link() {
        let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![2.0, 1.8, 1.6, 1.4, 1.2, 1.0]);

        // Gamma with log link (instead of canonical inverse)
        let mut model = GLM::new(Family::Gamma)
            .with_link(Link::Log)
            .with_max_iter(5000); // More iterations for non-canonical link

        let result = model.fit(&x, &y);
        assert!(
            result.is_ok(),
            "Custom link should work, error: {:?}",
            result.err()
        );
    }

    /// Test: Builder pattern
    #[test]
    fn test_builder_pattern() {
        let model = GLM::new(Family::Poisson)
            .with_max_iter(500)
            .with_tolerance(1e-8)
            .with_link(Link::Log);

        assert!(model.coefficients().is_none());
        assert!(model.intercept().is_none());
    }
}
