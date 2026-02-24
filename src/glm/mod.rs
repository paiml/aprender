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

    /// Returns the family name for error messages.
    const fn name(self) -> &'static str {
        match self {
            Self::Poisson => "Poisson",
            Self::NegativeBinomial => "Negative Binomial",
            Self::Gamma => "Gamma",
            Self::Binomial => "Binomial",
        }
    }

    /// Returns whether a single response value is valid for this family.
    fn is_valid_response(self, val: f32) -> bool {
        match self {
            Self::Poisson | Self::NegativeBinomial => val >= 0.0,
            Self::Gamma => val > 0.0,
            Self::Binomial => (0.0..=1.0).contains(&val),
        }
    }

    /// Returns the constraint description for error messages.
    const fn constraint_description(self) -> &'static str {
        match self {
            Self::Poisson | Self::NegativeBinomial => "non-negative counts",
            Self::Gamma => "positive values",
            Self::Binomial => "values in [0,1]",
        }
    }

    /// Clamps μ to valid ranges for this family after inverse-link.
    fn clamp_mu(self, mu_raw: f32) -> f32 {
        match self {
            Self::Poisson | Self::NegativeBinomial | Self::Gamma => mu_raw.max(1e-6),
            Self::Binomial => mu_raw.clamp(1e-6, 1.0 - 1e-6),
        }
    }

    /// Validates that y values are appropriate for this family.
    fn validate_response(self, y: &Vector<f32>) -> Result<()> {
        for &val in y.as_slice() {
            if !self.is_valid_response(val) {
                return Err(AprenderError::Other(format!(
                    "{} requires {}, got {val}",
                    self.name(),
                    self.constraint_description(),
                )));
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
    // Contract: glm-v1, equation = "irls_fit"
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

        // Initialize IRLS state from response mean
        let (mut beta, mut intercept, mut eta) = self.initialize_irls(y, n, p);

        // IRLS algorithm
        for _iter in 0..self.max_iter {
            let (beta_new, intercept_new, eta_new, max_change) =
                self.irls_iteration(x, y, &beta, intercept, &eta, n, p)?;

            beta = beta_new;
            intercept = intercept_new;
            eta = eta_new;

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

    /// Initializes IRLS state: coefficients (zero), intercept (from y mean), and linear predictor.
    fn initialize_irls(
        &self,
        y: &Vector<f32>,
        n: usize,
        p: usize,
    ) -> (Vec<f32>, f32, Vec<f32>) {
        let beta = vec![0.0_f32; p];
        let y_mean = y.as_slice().iter().sum::<f32>() / n as f32;
        let y_mean_safe = y_mean.clamp(0.01, 0.99); // Avoid extreme values
        let intercept = self.link.link(y_mean_safe);
        let eta = vec![intercept; n];
        (beta, intercept, eta)
    }

    /// Performs one IRLS iteration.
    ///
    /// Returns (new_beta, new_intercept, new_eta, max_change).
    #[allow(clippy::needless_range_loop)]
    fn irls_iteration(
        &self,
        x: &Matrix<f32>,
        y: &Vector<f32>,
        beta: &[f32],
        intercept: f32,
        eta: &[f32],
        n: usize,
        p: usize,
    ) -> Result<(Vec<f32>, f32, Vec<f32>, f32)> {
        // Compute μ = g⁻¹(η) and clamp to valid ranges
        let mu: Vec<f32> = eta
            .iter()
            .map(|&e| self.family.clamp_mu(self.link.inverse_link(e)))
            .collect();

        // Compute working response z and weights W
        let (z, weights) = self.compute_working_response_and_weights(y, &mu, eta, n);

        // Solve weighted least squares for new coefficients
        let beta_aug = solve_weighted_least_squares(x, &weights, &z, n, p)?;

        // Extract intercept and coefficients, apply damping
        let intercept_new = beta_aug[0];
        let beta_new = &beta_aug.as_slice()[1..];
        let step_size = self.damping_factor();

        let intercept_damped = intercept + step_size * (intercept_new - intercept);
        let beta_damped: Vec<f32> = beta
            .iter()
            .zip(beta_new)
            .map(|(old, new)| old + step_size * (new - old))
            .collect();

        // Update linear predictor with damped steps
        let eta_new = compute_linear_predictor(x, &beta_damped, intercept_damped, n, p);

        // Check convergence: max absolute change in any coefficient
        let max_change = compute_max_change(beta, &beta_damped, intercept, intercept_damped);

        Ok((beta_damped, intercept_damped, eta_new, max_change))
    }

    /// Computes working response z and IRLS weights for one iteration.
    ///
    /// z_i = η_i + (y_i - μ_i) * g'(η_i)
    /// w_i = clamp(1 / (V(μ_i) * [g'(η_i)]² + ε))
    #[allow(clippy::needless_range_loop)]
    fn compute_working_response_and_weights(
        &self,
        y: &Vector<f32>,
        mu: &[f32],
        eta: &[f32],
        n: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let mut z = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);
        for i in 0..n {
            let deriv = self.link.derivative(eta[i]);
            z.push(eta[i] + (y[i] - mu[i]) * deriv);

            let var = self.family.variance(mu[i], self.dispersion).max(1e-10);
            let weight = 1.0 / (var * deriv * deriv + 1e-10);
            weights.push(weight.clamp(1e-6, 1e6));
        }
        (z, weights)
    }

    /// Returns the step damping factor. Log link uses 0.5 to prevent divergence.
    fn damping_factor(&self) -> f32 {
        match self.link {
            Link::Log => 0.5,
            _ => 1.0,
        }
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

/// Solves the weighted least squares sub-problem: (X'WX + λI)β = X'Wz.
///
/// Builds the augmented design matrix [1|X], applies W^{1/2} weighting,
/// adds adaptive ridge regularization, and solves via Cholesky decomposition.
#[allow(clippy::needless_range_loop)]
fn solve_weighted_least_squares(
    x: &Matrix<f32>,
    weights: &[f32],
    z: &[f32],
    n: usize,
    p: usize,
) -> Result<Vector<f32>> {
    // Build augmented design matrix [1 X] to include intercept
    let x_aug = build_augmented_matrix(x, n, p)?;

    // Compute W^(1/2) * X_aug (two layouts needed for normal equations)
    let xtw = build_xtw(&x_aug, weights, n, p)?;
    let wx = build_wx(&x_aug, weights, n, p)?;

    // Compute W^(1/2) z
    let wz: Vec<f32> = z
        .iter()
        .enumerate()
        .map(|(i, &zi)| weights[i].sqrt() * zi)
        .collect();
    let wz_vec = Vector::from_vec(wz);

    // Normal equations: (W^(1/2)X)' (W^(1/2)X) β = (W^(1/2)X)' W^(1/2)z
    let mut xtwx = xtw
        .matmul(&wx)
        .map_err(|e| AprenderError::Other(format!("X'WX computation failed: {e}")))?;
    let xtwz = xtw
        .matvec(&wz_vec)
        .map_err(|e| AprenderError::Other(format!("X'Wz computation failed: {e}")))?;

    // Adaptive ridge regularization: X'WX + λI for positive definiteness
    add_ridge_regularization(&mut xtwx, p);

    xtwx.cholesky_solve(&xtwz)
        .map_err(|e| AprenderError::Other(format!("Cholesky solve failed: {e}")))
}

/// Builds the augmented design matrix [1|X] with an intercept column.
#[allow(clippy::needless_range_loop)]
fn build_augmented_matrix(x: &Matrix<f32>, n: usize, p: usize) -> Result<Matrix<f32>> {
    let mut data = Vec::with_capacity(n * (p + 1));
    for i in 0..n {
        data.push(1.0);
        for j in 0..p {
            data.push(x.get(i, j));
        }
    }
    Matrix::from_vec(n, p + 1, data)
        .map_err(|e| AprenderError::Other(format!("Augmented matrix error: {e}")))
}

/// Builds X'W^{1/2} matrix (transposed layout for left side of normal equations).
#[allow(clippy::needless_range_loop)]
fn build_xtw(x_aug: &Matrix<f32>, weights: &[f32], n: usize, p: usize) -> Result<Matrix<f32>> {
    let mut data = Vec::with_capacity((p + 1) * n);
    for j in 0..=p {
        for i in 0..n {
            data.push(x_aug.get(i, j) * weights[i].sqrt());
        }
    }
    Matrix::from_vec(p + 1, n, data)
        .map_err(|e| AprenderError::Other(format!("X'W matrix error: {e}")))
}

/// Builds W^{1/2}X matrix (weighted design matrix for right side of normal equations).
#[allow(clippy::needless_range_loop)]
fn build_wx(x_aug: &Matrix<f32>, weights: &[f32], n: usize, p: usize) -> Result<Matrix<f32>> {
    let mut data = Vec::with_capacity(n * (p + 1));
    for i in 0..n {
        for j in 0..=p {
            data.push(weights[i].sqrt() * x_aug.get(i, j));
        }
    }
    Matrix::from_vec(n, p + 1, data)
        .map_err(|e| AprenderError::Other(format!("WX matrix error: {e}")))
}

/// Adds adaptive ridge regularization λI to X'WX for numerical stability.
///
/// Ridge λ = max(max_diagonal * 1e-6, 1e-8) ensures positive definiteness.
fn add_ridge_regularization(xtwx: &mut Matrix<f32>, p: usize) {
    let max_diag = (0..=p)
        .map(|i| xtwx.get(i, i).abs())
        .fold(0.0_f32, f32::max);
    let ridge = (max_diag * 1e-6).max(1e-8);
    for i in 0..=p {
        let old_val = xtwx.get(i, i);
        xtwx.set(i, i, old_val + ridge);
    }
}

/// Computes the linear predictor η = Xβ + β₀.
#[allow(clippy::needless_range_loop)]
fn compute_linear_predictor(
    x: &Matrix<f32>,
    beta: &[f32],
    intercept: f32,
    n: usize,
    p: usize,
) -> Vec<f32> {
    let mut eta = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = intercept;
        for j in 0..p {
            val += x.get(i, j) * beta[j];
        }
        eta.push(val);
    }
    eta
}

/// Computes the maximum absolute change across all coefficients and intercept.
fn compute_max_change(
    beta_old: &[f32],
    beta_new: &[f32],
    intercept_old: f32,
    intercept_new: f32,
) -> f32 {
    let mut max_change = (intercept_new - intercept_old).abs();
    for (old, new) in beta_old.iter().zip(beta_new) {
        max_change = max_change.max((new - old).abs());
    }
    max_change
}

#[cfg(test)]
#[path = "glm_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_glm_contract.rs"]
mod tests_glm_contract;
