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

// Contract: linear-models-v1, equation = "ols_fit"
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

include!("mod_part_02.rs");
include!("mod_part_03.rs");
include!("mod_part_04.rs");
include!("mod_part_05.rs");
