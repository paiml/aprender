//! Linear models for regression.
//!
//! Includes Ordinary Least Squares (OLS) and regularized regression.

use crate::error::Result;
use crate::metrics::r_squared;
use crate::primitives::{Matrix, Vector};
use crate::traits::Estimator;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Ordinary Least Squares (OLS) linear regression.
///
/// Fits a linear model by minimizing the residual sum of squares between
/// observed targets and predicted targets. The model equation is:
///
/// ```text
/// y = X β + ε
/// ```
///
/// where `β` is the coefficient vector and `ε` is random error.
///
/// # Solver
///
/// Uses normal equations: `β = (X^T X)^-1 X^T y` via Cholesky decomposition.
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// // Simple linear regression: y = 2x + 1
/// let x = Matrix::from_vec(4, 1, vec![
///     1.0,
///     2.0,
///     3.0,
///     4.0,
/// ]).expect("Valid matrix dimensions");
/// let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y).expect("Fit should succeed with valid data");
///
/// let predictions = model.predict(&x);
/// let r2 = model.score(&x, &y);
/// assert!(r2 > 0.99);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n²p + p³) where n = samples, p = features
/// - Space complexity: O(np)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl LinearRegression {
    /// Creates a new `LinearRegression` with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
        }
    }

    /// Sets whether to fit an intercept term.
    #[must_use]
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Returns the coefficients (excluding intercept).
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    #[must_use]
    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Returns the intercept term.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the model to `SafeTensors` format.
    ///
    /// `SafeTensors` format is compatible with:
    /// - `HuggingFace` ecosystem
    /// - Ollama (can convert to GGUF)
    /// - `PyTorch`, TensorFlow
    /// - realizar inference engine
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not fitted
    /// - Serialization fails
    /// - File writing fails
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Verify model is fitted
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        // Prepare tensors (BTreeMap ensures deterministic ordering)
        let mut tensors = BTreeMap::new();

        // Coefficients tensor
        let coef_data: Vec<f32> = (0..coefficients.len()).map(|i| coefficients[i]).collect();
        let coef_shape = vec![coefficients.len()];
        tensors.insert("coefficients".to_string(), (coef_data, coef_shape));

        // Intercept tensor
        let intercept_data = vec![self.intercept];
        let intercept_shape = vec![1];
        tensors.insert("intercept".to_string(), (intercept_data, intercept_shape));

        // Save to SafeTensors format
        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a model from `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - `SafeTensors` format is invalid
    /// - Required tensors are missing
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract coefficients tensor
        let coef_meta = metadata
            .get("coefficients")
            .ok_or("Missing 'coefficients' tensor in SafeTensors file")?;
        let coef_data = safetensors::extract_tensor(&raw_data, coef_meta)?;

        // Extract intercept tensor
        let intercept_meta = metadata
            .get("intercept")
            .ok_or("Missing 'intercept' tensor in SafeTensors file")?;
        let intercept_data = safetensors::extract_tensor(&raw_data, intercept_meta)?;

        // Validate intercept shape
        if intercept_data.len() != 1 {
            return Err(format!(
                "Invalid intercept tensor: expected 1 value, got {}",
                intercept_data.len()
            ));
        }

        // Construct model
        Ok(Self {
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            fit_intercept: true, // Default to true for loaded models
        })
    }

    /// Adds an intercept column of ones to the design matrix.
    fn add_intercept_column(x: &Matrix<f32>) -> Matrix<f32> {
        let (n_rows, n_cols) = x.shape();
        let mut data = Vec::with_capacity(n_rows * (n_cols + 1));

        for i in 0..n_rows {
            data.push(1.0); // Intercept column
            for j in 0..n_cols {
                data.push(x.get(i, j));
            }
        }

        Matrix::from_vec(n_rows, n_cols + 1, data)
            .expect("Internal error: failed to create design matrix")
    }
}

impl Estimator for LinearRegression {
    /// Fits the linear regression model using normal equations.
    ///
    /// Solves: β = (X^T X)^-1 X^T y
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input dimensions don't match
    /// - Not enough samples for the number of features (underdetermined system)
    /// - Matrix is singular (not positive definite)
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples must match target length".into());
        }

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Check for underdetermined system
        // When fitting intercept, we need n_samples >= n_features + 1
        // Without intercept, we need n_samples >= n_features
        let required_samples = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        if n_samples < required_samples {
            return Err(
                "Insufficient samples: LinearRegression requires at least as many samples as \
                 features (plus 1 if fitting intercept). Consider using Ridge regression or \
                 collecting more training data"
                    .into(),
            );
        }

        // Create design matrix (with or without intercept)
        let x_design = if self.fit_intercept {
            Self::add_intercept_column(x)
        } else {
            x.clone()
        };

        // Compute X^T X
        let xt = x_design.transpose();
        let xtx = xt.matmul(&x_design)?;

        // Compute X^T y
        let xty = xt.matvec(y)?;

        // Solve normal equations via Cholesky decomposition
        let beta = xtx.cholesky_solve(&xty)?;

        // Extract intercept and coefficients
        if self.fit_intercept {
            self.intercept = beta[0];
            self.coefficients = Some(beta.slice(1, n_features + 1));
        } else {
            self.intercept = 0.0;
            self.coefficients = Some(beta);
        }

        Ok(())
    }

    /// Predicts target values for input data.
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coefficients = self
            .coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        let result = x
            .matvec(coefficients)
            .expect("Matrix dimensions don't match coefficients");

        result.add_scalar(self.intercept)
    }

    /// Computes the R² score.
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}

/// Ridge regression with L2 regularization.
///
/// Fits a linear model with L2 penalty on coefficient magnitudes.
/// The optimization objective is:
///
/// ```text
/// minimize ||y - Xβ||² + α||β||²
/// ```
///
/// where `α` (alpha) controls the regularization strength.
///
/// # Solver
///
/// Uses regularized normal equations: `β = (X^T X + αI)^-1 X^T y`
///
/// # When to use Ridge
///
/// - When you have many correlated features (multicollinearity)
/// - To prevent overfitting with limited samples
/// - When all features are expected to contribute
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
/// use aprender::linear_model::Ridge;
///
/// // Data with some noise
/// let x = Matrix::from_vec(5, 2, vec![
///     1.0, 2.0,
///     2.0, 3.0,
///     3.0, 4.0,
///     4.0, 5.0,
///     5.0, 6.0,
/// ]).expect("Valid matrix dimensions");
/// let y = Vector::from_slice(&[5.0, 8.0, 11.0, 14.0, 17.0]);
///
/// let mut model = Ridge::new(1.0);  // alpha = 1.0
/// model.fit(&x, &y).expect("Fit should succeed with valid data");
///
/// let predictions = model.predict(&x);
/// let r2 = model.score(&x, &y);
/// assert!(r2 > 0.9);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ridge {
    /// Regularization strength (lambda/alpha).
    alpha: f32,
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
}

impl Ridge {
    /// Creates a new `Ridge` regression with the given regularization strength.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Regularization strength. Larger values = more regularization.
    ///   Must be non-negative. Use 0.0 for no regularization (equivalent to OLS).
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
        }
    }

    /// Sets whether to fit an intercept term.
    #[must_use]
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Returns the regularization strength (alpha).
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the coefficients (excluding intercept).
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    #[must_use]
    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Returns the intercept term.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the model to `SafeTensors` format.
    ///
    /// `SafeTensors` format is compatible with:
    /// - `HuggingFace` ecosystem
    /// - Ollama (can convert to GGUF)
    /// - `PyTorch`, TensorFlow
    /// - realizar inference engine
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not fitted
    /// - Serialization fails
    /// - File writing fails
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Verify model is fitted
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        // Prepare tensors (BTreeMap ensures deterministic ordering)
        let mut tensors = BTreeMap::new();

        // Coefficients tensor
        let coef_data: Vec<f32> = (0..coefficients.len()).map(|i| coefficients[i]).collect();
        let coef_shape = vec![coefficients.len()];
        tensors.insert("coefficients".to_string(), (coef_data, coef_shape));

        // Intercept tensor
        let intercept_data = vec![self.intercept];
        let intercept_shape = vec![1];
        tensors.insert("intercept".to_string(), (intercept_data, intercept_shape));

        // Alpha (regularization strength) as tensor
        let alpha_data = vec![self.alpha];
        let alpha_shape = vec![1];
        tensors.insert("alpha".to_string(), (alpha_data, alpha_shape));

        // Save to SafeTensors format
        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a model from `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - `SafeTensors` format is invalid
    /// - Required tensors are missing
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract coefficients tensor
        let coef_meta = metadata
            .get("coefficients")
            .ok_or("Missing 'coefficients' tensor in SafeTensors file")?;
        let coef_data = safetensors::extract_tensor(&raw_data, coef_meta)?;

        // Extract intercept tensor
        let intercept_meta = metadata
            .get("intercept")
            .ok_or("Missing 'intercept' tensor in SafeTensors file")?;
        let intercept_data = safetensors::extract_tensor(&raw_data, intercept_meta)?;

        // Extract alpha tensor
        let alpha_meta = metadata
            .get("alpha")
            .ok_or("Missing 'alpha' tensor in SafeTensors file")?;
        let alpha_data = safetensors::extract_tensor(&raw_data, alpha_meta)?;

        // Validate tensor sizes
        if intercept_data.len() != 1 {
            return Err(format!(
                "Expected intercept tensor to have 1 element, got {}",
                intercept_data.len()
            ));
        }

        if alpha_data.len() != 1 {
            return Err(format!(
                "Expected alpha tensor to have 1 element, got {}",
                alpha_data.len()
            ));
        }

        // Reconstruct model
        Ok(Self {
            alpha: alpha_data[0],
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            fit_intercept: true, // Default to true for loaded models
        })
    }
}

impl Estimator for Ridge {
    /// Fits the Ridge regression model using regularized normal equations.
    ///
    /// Solves: β = (X^T X + αI)^-1 X^T y
    ///
    /// # Errors
    ///
    /// Returns an error if input dimensions don't match or matrix is singular.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples must match target length".into());
        }

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Create design matrix (with or without intercept)
        let x_design = if self.fit_intercept {
            LinearRegression::add_intercept_column(x)
        } else {
            x.clone()
        };

        let n_params = if self.fit_intercept {
            n_features + 1
        } else {
            n_features
        };

        // Compute X^T X
        let xt = x_design.transpose();
        let mut xtx = xt.matmul(&x_design)?;

        // Add regularization: X^T X + αI
        // Note: We don't regularize the intercept term
        for i in 0..n_params {
            // Skip intercept if fitting intercept (first column)
            if self.fit_intercept && i == 0 {
                continue;
            }
            let current = xtx.get(i, i);
            xtx.set(i, i, current + self.alpha);
        }

        // Compute X^T y
        let xty = xt.matvec(y)?;

        // Solve regularized normal equations via Cholesky decomposition
        let beta = xtx.cholesky_solve(&xty)?;

        // Extract intercept and coefficients
        if self.fit_intercept {
            self.intercept = beta[0];
            self.coefficients = Some(beta.slice(1, n_features + 1));
        } else {
            self.intercept = 0.0;
            self.coefficients = Some(beta);
        }

        Ok(())
    }

    /// Predicts target values for input data.
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coefficients = self
            .coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        let result = x
            .matvec(coefficients)
            .expect("Matrix dimensions don't match coefficients");

        result.add_scalar(self.intercept)
    }

    /// Computes the R² score.
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}

/// Lasso regression with L1 regularization.
///
/// Fits a linear model with L1 penalty on coefficient magnitudes.
/// The optimization objective is:
///
/// ```text
/// minimize ||y - Xβ||² + α||β||₁
/// ```
///
/// where `α` (alpha) controls the regularization strength.
///
/// # Solver
///
/// Uses coordinate descent with soft-thresholding.
///
/// # When to use Lasso
///
/// - For automatic feature selection (produces sparse models)
/// - When you expect only a few features to be relevant
/// - When interpretability through sparsity is desired
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
/// use aprender::linear_model::Lasso;
///
/// // Data with some features
/// let x = Matrix::from_vec(5, 2, vec![
///     1.0, 2.0,
///     2.0, 3.0,
///     3.0, 4.0,
///     4.0, 5.0,
///     5.0, 6.0,
/// ]).expect("Valid matrix dimensions");
/// let y = Vector::from_slice(&[5.0, 8.0, 11.0, 14.0, 17.0]);
///
/// let mut model = Lasso::new(0.1);  // alpha = 0.1
/// model.fit(&x, &y).expect("Fit should succeed with valid data");
///
/// let predictions = model.predict(&x);
/// let r2 = model.score(&x, &y);
/// assert!(r2 > 0.9);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lasso {
    /// Regularization strength (lambda/alpha).
    alpha: f32,
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
    /// Maximum number of iterations for coordinate descent.
    max_iter: usize,
    /// Tolerance for convergence.
    tol: f32,
}

impl Lasso {
    /// Creates a new `Lasso` regression with the given regularization strength.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Regularization strength. Larger values = more sparsity.
    ///   Must be non-negative.
    #[must_use]
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Sets whether to fit an intercept term.
    #[must_use]
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Returns the regularization strength (alpha).
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the coefficients (excluding intercept).
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    #[must_use]
    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Returns the intercept term.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Soft-thresholding operator for L1 regularization.
    fn soft_threshold(x: f32, lambda: f32) -> f32 {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the model to `SafeTensors` format.
    ///
    /// `SafeTensors` format is compatible with:
    /// - `HuggingFace` ecosystem
    /// - Ollama (can convert to GGUF)
    /// - `PyTorch`, TensorFlow
    /// - realizar inference engine
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not fitted
    /// - Serialization fails
    /// - File writing fails
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Verify model is fitted
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        // Prepare tensors (BTreeMap ensures deterministic ordering)
        let mut tensors = BTreeMap::new();

        // Coefficients tensor
        let coef_data: Vec<f32> = (0..coefficients.len()).map(|i| coefficients[i]).collect();
        let coef_shape = vec![coefficients.len()];
        tensors.insert("coefficients".to_string(), (coef_data, coef_shape));

        // Intercept tensor
        let intercept_data = vec![self.intercept];
        let intercept_shape = vec![1];
        tensors.insert("intercept".to_string(), (intercept_data, intercept_shape));

        // Alpha (regularization strength) as tensor
        let alpha_data = vec![self.alpha];
        let alpha_shape = vec![1];
        tensors.insert("alpha".to_string(), (alpha_data, alpha_shape));

        // Max iterations as tensor (stored as f32 for consistency)
        let max_iter_data = vec![self.max_iter as f32];
        let max_iter_shape = vec![1];
        tensors.insert("max_iter".to_string(), (max_iter_data, max_iter_shape));

        // Tolerance as tensor
        let tol_data = vec![self.tol];
        let tol_shape = vec![1];
        tensors.insert("tol".to_string(), (tol_data, tol_shape));

        // Save to SafeTensors format
        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a model from `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - `SafeTensors` format is invalid
    /// - Required tensors are missing
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract coefficients tensor
        let coef_meta = metadata
            .get("coefficients")
            .ok_or("Missing 'coefficients' tensor in SafeTensors file")?;
        let coef_data = safetensors::extract_tensor(&raw_data, coef_meta)?;

        // Extract intercept tensor
        let intercept_meta = metadata
            .get("intercept")
            .ok_or("Missing 'intercept' tensor in SafeTensors file")?;
        let intercept_data = safetensors::extract_tensor(&raw_data, intercept_meta)?;

        // Extract alpha tensor
        let alpha_meta = metadata
            .get("alpha")
            .ok_or("Missing 'alpha' tensor in SafeTensors file")?;
        let alpha_data = safetensors::extract_tensor(&raw_data, alpha_meta)?;

        // Extract max_iter tensor
        let max_iter_meta = metadata
            .get("max_iter")
            .ok_or("Missing 'max_iter' tensor in SafeTensors file")?;
        let max_iter_data = safetensors::extract_tensor(&raw_data, max_iter_meta)?;

        // Extract tol tensor
        let tol_meta = metadata
            .get("tol")
            .ok_or("Missing 'tol' tensor in SafeTensors file")?;
        let tol_data = safetensors::extract_tensor(&raw_data, tol_meta)?;

        // Validate tensor sizes
        if intercept_data.len() != 1 {
            return Err(format!(
                "Expected intercept tensor to have 1 element, got {}",
                intercept_data.len()
            ));
        }

        if alpha_data.len() != 1 {
            return Err(format!(
                "Expected alpha tensor to have 1 element, got {}",
                alpha_data.len()
            ));
        }

        if max_iter_data.len() != 1 {
            return Err(format!(
                "Expected max_iter tensor to have 1 element, got {}",
                max_iter_data.len()
            ));
        }

        if tol_data.len() != 1 {
            return Err(format!(
                "Expected tol tensor to have 1 element, got {}",
                tol_data.len()
            ));
        }

        // Reconstruct model
        Ok(Self {
            alpha: alpha_data[0],
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            fit_intercept: true, // Default to true for loaded models
            max_iter: max_iter_data[0] as usize,
            tol: tol_data[0],
        })
    }
}

impl Estimator for Lasso {
    /// Fits the Lasso regression model using coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns an error if input dimensions don't match.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples must match target length".into());
        }

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, y_mean) = if self.fit_intercept {
            // Compute means
            let mut x_mean = vec![0.0; n_features];
            let mut y_sum = 0.0;

            for i in 0..n_samples {
                for (j, mean_j) in x_mean.iter_mut().enumerate() {
                    *mean_j += x.get(i, j);
                }
                y_sum += y[i];
            }

            for mean in &mut x_mean {
                *mean /= n_samples as f32;
            }
            let y_mean = y_sum / n_samples as f32;

            // Center data
            let mut x_data = vec![0.0; n_samples * n_features];
            let mut y_data = vec![0.0; n_samples];

            for i in 0..n_samples {
                for j in 0..n_features {
                    x_data[i * n_features + j] = x.get(i, j) - x_mean[j];
                }
                y_data[i] = y[i] - y_mean;
            }

            (
                Matrix::from_vec(n_samples, n_features, x_data)
                    .expect("Valid matrix dimensions for property test"),
                Vector::from_vec(y_data),
                y_mean,
            )
        } else {
            (x.clone(), y.clone(), 0.0)
        };

        // Initialize coefficients to zero
        let mut beta = vec![0.0; n_features];

        // Precompute X^T X diagonal (column norms squared)
        let mut col_norms_sq = vec![0.0; n_features];
        for (j, norm_sq) in col_norms_sq.iter_mut().enumerate() {
            for i in 0..n_samples {
                let val = x_centered.get(i, j);
                *norm_sq += val * val;
            }
        }

        // Coordinate descent
        for _ in 0..self.max_iter {
            let mut max_change = 0.0f32;

            for j in 0..n_features {
                if col_norms_sq[j] < 1e-10 {
                    continue; // Skip zero-variance features
                }

                // Compute residual without current feature
                let mut rho = 0.0;
                for i in 0..n_samples {
                    let mut pred = 0.0;
                    for (k, &beta_k) in beta.iter().enumerate() {
                        if k != j {
                            pred += x_centered.get(i, k) * beta_k;
                        }
                    }
                    let residual = y_centered[i] - pred;
                    rho += x_centered.get(i, j) * residual;
                }

                // Update coefficient with soft-thresholding
                let old_beta = beta[j];
                beta[j] = Self::soft_threshold(rho, self.alpha) / col_norms_sq[j];

                let change = (beta[j] - old_beta).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            // Check convergence
            if max_change < self.tol {
                break;
            }
        }

        // Set intercept
        if self.fit_intercept {
            let mut intercept = y_mean;
            let mut x_mean = vec![0.0; n_features];
            for j in 0..n_features {
                for i in 0..n_samples {
                    x_mean[j] += x.get(i, j);
                }
                x_mean[j] /= n_samples as f32;
                intercept -= beta[j] * x_mean[j];
            }
            self.intercept = intercept;
        } else {
            self.intercept = 0.0;
        }

        self.coefficients = Some(Vector::from_vec(beta));
        Ok(())
    }

    /// Predicts target values for input data.
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coefficients = self
            .coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        let result = x
            .matvec(coefficients)
            .expect("Matrix dimensions don't match coefficients");

        result.add_scalar(self.intercept)
    }

    /// Computes the R² score.
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}

/// Elastic Net regression with combined L1 and L2 regularization.
///
/// Fits a linear model with both L1 and L2 penalties:
///
/// ```text
/// minimize ||y - Xβ||² + α * l1_ratio * ||β||₁ + α * (1 - l1_ratio) * ||β||²
/// ```
///
/// # Parameters
///
/// - `alpha` - Overall regularization strength
/// - `l1_ratio` - Mix between L1 and L2 (0.0 = Ridge, 1.0 = Lasso)
///
/// # Solver
///
/// Uses coordinate descent with combined soft-thresholding and shrinkage.
///
/// # When to use Elastic Net
///
/// - When you want both sparsity (L1) and grouping effect (L2)
/// - With correlated features where Lasso may be unstable
/// - When you don't know which regularization type to use
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
/// use aprender::linear_model::ElasticNet;
///
/// let x = Matrix::from_vec(5, 2, vec![
///     1.0, 2.0,
///     2.0, 3.0,
///     3.0, 4.0,
///     4.0, 5.0,
///     5.0, 6.0,
/// ]).expect("Valid matrix dimensions");
/// let y = Vector::from_slice(&[5.0, 8.0, 11.0, 14.0, 17.0]);
///
/// // 50% L1, 50% L2
/// let mut model = ElasticNet::new(0.1, 0.5);
/// model.fit(&x, &y).expect("Fit should succeed with valid data");
///
/// let r2 = model.score(&x, &y);
/// assert!(r2 > 0.9);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticNet {
    /// Overall regularization strength.
    alpha: f32,
    /// Mix between L1 and L2 (0.0 = pure L2, 1.0 = pure L1).
    l1_ratio: f32,
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
    /// Maximum number of iterations.
    max_iter: usize,
    /// Tolerance for convergence.
    tol: f32,
}

impl ElasticNet {
    /// Creates a new `ElasticNet` with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `alpha` - Overall regularization strength
    /// * `l1_ratio` - Mix between L1 and L2 (0.0 = Ridge, 1.0 = Lasso)
    #[must_use]
    pub fn new(alpha: f32, l1_ratio: f32) -> Self {
        Self {
            alpha,
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Sets whether to fit an intercept term.
    #[must_use]
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Sets the maximum number of iterations.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Returns the regularization strength (alpha).
    #[must_use]
    pub fn alpha(&self) -> f32 {
        self.alpha
    }

    /// Returns the L1/L2 ratio.
    #[must_use]
    pub fn l1_ratio(&self) -> f32 {
        self.l1_ratio
    }

    /// Returns the coefficients (excluding intercept).
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    #[must_use]
    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.")
    }

    /// Returns the intercept term.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Returns true if the model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        self.coefficients.is_some()
    }

    /// Saves the model to a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the model to `SafeTensors` format.
    ///
    /// `SafeTensors` format is compatible with:
    /// - `HuggingFace` ecosystem
    /// - Ollama (can convert to GGUF)
    /// - `PyTorch`, TensorFlow
    /// - realizar inference engine
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not fitted
    /// - Serialization fails
    /// - File writing fails
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Verify model is fitted
        let coefficients = self
            .coefficients
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        // Prepare tensors (BTreeMap ensures deterministic ordering)
        let mut tensors = BTreeMap::new();

        // Coefficients tensor
        let coef_data: Vec<f32> = (0..coefficients.len()).map(|i| coefficients[i]).collect();
        let coef_shape = vec![coefficients.len()];
        tensors.insert("coefficients".to_string(), (coef_data, coef_shape));

        // Intercept tensor
        let intercept_data = vec![self.intercept];
        let intercept_shape = vec![1];
        tensors.insert("intercept".to_string(), (intercept_data, intercept_shape));

        // Alpha (regularization strength) as tensor
        let alpha_data = vec![self.alpha];
        let alpha_shape = vec![1];
        tensors.insert("alpha".to_string(), (alpha_data, alpha_shape));

        // L1 ratio as tensor
        let l1_ratio_data = vec![self.l1_ratio];
        let l1_ratio_shape = vec![1];
        tensors.insert("l1_ratio".to_string(), (l1_ratio_data, l1_ratio_shape));

        // Max iterations as tensor (stored as f32 for consistency)
        let max_iter_data = vec![self.max_iter as f32];
        let max_iter_shape = vec![1];
        tensors.insert("max_iter".to_string(), (max_iter_data, max_iter_shape));

        // Tolerance as tensor
        let tol_data = vec![self.tol];
        let tol_shape = vec![1];
        tensors.insert("tol".to_string(), (tol_data, tol_shape));

        // Save to SafeTensors format
        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a model from `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - `SafeTensors` format is invalid
    /// - Required tensors are missing
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract coefficients tensor
        let coef_meta = metadata
            .get("coefficients")
            .ok_or("Missing 'coefficients' tensor in SafeTensors file")?;
        let coef_data = safetensors::extract_tensor(&raw_data, coef_meta)?;

        // Extract intercept tensor
        let intercept_meta = metadata
            .get("intercept")
            .ok_or("Missing 'intercept' tensor in SafeTensors file")?;
        let intercept_data = safetensors::extract_tensor(&raw_data, intercept_meta)?;

        // Extract alpha tensor
        let alpha_meta = metadata
            .get("alpha")
            .ok_or("Missing 'alpha' tensor in SafeTensors file")?;
        let alpha_data = safetensors::extract_tensor(&raw_data, alpha_meta)?;

        // Extract l1_ratio tensor
        let l1_ratio_meta = metadata
            .get("l1_ratio")
            .ok_or("Missing 'l1_ratio' tensor in SafeTensors file")?;
        let l1_ratio_data = safetensors::extract_tensor(&raw_data, l1_ratio_meta)?;

        // Extract max_iter tensor
        let max_iter_meta = metadata
            .get("max_iter")
            .ok_or("Missing 'max_iter' tensor in SafeTensors file")?;
        let max_iter_data = safetensors::extract_tensor(&raw_data, max_iter_meta)?;

        // Extract tol tensor
        let tol_meta = metadata
            .get("tol")
            .ok_or("Missing 'tol' tensor in SafeTensors file")?;
        let tol_data = safetensors::extract_tensor(&raw_data, tol_meta)?;

        // Validate tensor sizes
        if intercept_data.len() != 1 {
            return Err(format!(
                "Expected intercept tensor to have 1 element, got {}",
                intercept_data.len()
            ));
        }

        if alpha_data.len() != 1 {
            return Err(format!(
                "Expected alpha tensor to have 1 element, got {}",
                alpha_data.len()
            ));
        }

        if l1_ratio_data.len() != 1 {
            return Err(format!(
                "Expected l1_ratio tensor to have 1 element, got {}",
                l1_ratio_data.len()
            ));
        }

        if max_iter_data.len() != 1 {
            return Err(format!(
                "Expected max_iter tensor to have 1 element, got {}",
                max_iter_data.len()
            ));
        }

        if tol_data.len() != 1 {
            return Err(format!(
                "Expected tol tensor to have 1 element, got {}",
                tol_data.len()
            ));
        }

        // Reconstruct model
        Ok(Self {
            alpha: alpha_data[0],
            l1_ratio: l1_ratio_data[0],
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            fit_intercept: true, // Default to true for loaded models
            max_iter: max_iter_data[0] as usize,
            tol: tol_data[0],
        })
    }
}

impl Estimator for ElasticNet {
    /// Fits the Elastic Net model using coordinate descent.
    ///
    /// # Errors
    ///
    /// Returns an error if input dimensions don't match.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples must match target length".into());
        }

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Center data if fitting intercept
        let (x_centered, y_centered, y_mean) = if self.fit_intercept {
            let mut x_mean = vec![0.0; n_features];
            let mut y_sum = 0.0;

            for i in 0..n_samples {
                for (j, mean_j) in x_mean.iter_mut().enumerate() {
                    *mean_j += x.get(i, j);
                }
                y_sum += y[i];
            }

            for mean in &mut x_mean {
                *mean /= n_samples as f32;
            }
            let y_mean = y_sum / n_samples as f32;

            let mut x_data = vec![0.0; n_samples * n_features];
            let mut y_data = vec![0.0; n_samples];

            for i in 0..n_samples {
                for j in 0..n_features {
                    x_data[i * n_features + j] = x.get(i, j) - x_mean[j];
                }
                y_data[i] = y[i] - y_mean;
            }

            (
                Matrix::from_vec(n_samples, n_features, x_data)
                    .expect("Valid matrix dimensions for property test"),
                Vector::from_vec(y_data),
                y_mean,
            )
        } else {
            (x.clone(), y.clone(), 0.0)
        };

        // Initialize coefficients
        let mut beta = vec![0.0; n_features];

        // Precompute column norms squared
        let mut col_norms_sq = vec![0.0; n_features];
        for (j, norm_sq) in col_norms_sq.iter_mut().enumerate() {
            for i in 0..n_samples {
                let val = x_centered.get(i, j);
                *norm_sq += val * val;
            }
        }

        // L1 and L2 penalties
        let l1_penalty = self.alpha * self.l1_ratio;
        let l2_penalty = self.alpha * (1.0 - self.l1_ratio);

        // Coordinate descent
        for _ in 0..self.max_iter {
            let mut max_change = 0.0f32;

            for j in 0..n_features {
                if col_norms_sq[j] < 1e-10 {
                    continue;
                }

                // Compute residual without current feature
                let mut rho = 0.0;
                for i in 0..n_samples {
                    let mut pred = 0.0;
                    for (k, &beta_k) in beta.iter().enumerate() {
                        if k != j {
                            pred += x_centered.get(i, k) * beta_k;
                        }
                    }
                    let residual = y_centered[i] - pred;
                    rho += x_centered.get(i, j) * residual;
                }

                // Update with soft-thresholding (L1) and shrinkage (L2)
                let old_beta = beta[j];
                let denom = col_norms_sq[j] + l2_penalty;
                beta[j] = Lasso::soft_threshold(rho, l1_penalty) / denom;

                let change = (beta[j] - old_beta).abs();
                if change > max_change {
                    max_change = change;
                }
            }

            if max_change < self.tol {
                break;
            }
        }

        // Set intercept
        if self.fit_intercept {
            let mut intercept = y_mean;
            let mut x_mean = vec![0.0; n_features];
            for j in 0..n_features {
                for i in 0..n_samples {
                    x_mean[j] += x.get(i, j);
                }
                x_mean[j] /= n_samples as f32;
                intercept -= beta[j] * x_mean[j];
            }
            self.intercept = intercept;
        } else {
            self.intercept = 0.0;
        }

        self.coefficients = Some(Vector::from_vec(beta));
        Ok(())
    }

    /// Predicts target values for input data.
    ///
    /// # Panics
    ///
    /// Panics if model is not fitted.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coefficients = self
            .coefficients
            .as_ref()
            .expect("Model not fitted. Call fit() first.");

        let result = x
            .matvec(coefficients)
            .expect("Matrix dimensions don't match coefficients");

        result.add_scalar(self.intercept)
    }

    /// Computes the R² score.
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32 {
        let y_pred = self.predict(x);
        r_squared(&y_pred, y)
    }
}


#[cfg(test)]
mod tests;
