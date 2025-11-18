//! Linear models for regression.
//!
//! Includes Ordinary Least Squares (OLS) linear regression.

use crate::metrics::r_squared;
use crate::primitives::{Matrix, Vector};
use crate::traits::Estimator;

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
/// ]).unwrap();
/// let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y).unwrap();
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
#[derive(Debug, Clone)]
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
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<(), &'static str> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples must match target length");
        }

        if n_samples == 0 {
            return Err("Cannot fit with zero samples");
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
                 collecting more training data",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let model = LinearRegression::new();
        assert!(!model.is_fitted());
        assert!(model.fit_intercept);
    }

    #[test]
    fn test_simple_regression() {
        // y = 2x + 1
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        assert!(model.is_fitted());

        // Check coefficients
        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-4);
        assert!((model.intercept() - 1.0).abs() < 1e-4);

        // Check predictions
        let predictions = model.predict(&x);
        for i in 0..4 {
            assert!((predictions[i] - y[i]).abs() < 1e-4);
        }

        // Check R²
        let r2 = model.score(&x, &y);
        assert!((r2 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_multivariate_regression() {
        // y = 1 + 2*x1 + 3*x2
        let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0]).unwrap();
        let y = Vector::from_slice(&[6.0, 8.0, 9.0, 11.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-4);
        assert!((coef[1] - 3.0).abs() < 1e-4);
        assert!((model.intercept() - 1.0).abs() < 1e-4);

        let r2 = model.score(&x, &y);
        assert!((r2 - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_no_intercept() {
        // y = 2x (no intercept)
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

        let mut model = LinearRegression::new().with_intercept(false);
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-4);
        assert!((model.intercept() - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_predict_new_data() {
        // y = x + 1
        let x_train = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let y_train = Vector::from_slice(&[2.0, 3.0, 4.0]);

        let mut model = LinearRegression::new();
        model.fit(&x_train, &y_train).unwrap();

        let x_test = Matrix::from_vec(2, 1, vec![4.0, 5.0]).unwrap();
        let predictions = model.predict(&x_test);

        assert!((predictions[0] - 5.0).abs() < 1e-4);
        assert!((predictions[1] - 6.0).abs() < 1e-4);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let x = Matrix::from_vec(3, 2, vec![1.0; 6]).unwrap();
        let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length

        let mut model = LinearRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_data_error() {
        let x = Matrix::from_vec(0, 2, vec![]).unwrap();
        let y = Vector::from_vec(vec![]);

        let mut model = LinearRegression::new();
        let result = model.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_noise() {
        // y ≈ 2x + 1 with some noise
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Vector::from_slice(&[3.1, 4.9, 7.2, 8.8, 11.1]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Should still get approximately correct coefficients
        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 0.2);
        assert!((model.intercept() - 1.0).abs() < 0.5);

        // R² should be high but not perfect
        let r2 = model.score(&x, &y);
        assert!(r2 > 0.95);
        assert!(r2 < 1.0);
    }

    #[test]
    fn test_default() {
        let model = LinearRegression::default();
        assert!(!model.is_fitted());
    }

    #[test]
    fn test_clone() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let y = Vector::from_slice(&[2.0, 4.0, 6.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let cloned = model.clone();
        assert!(cloned.is_fitted());
        assert!((cloned.intercept() - model.intercept()).abs() < 1e-6);
    }

    #[test]
    fn test_score_range() {
        // R² should be between negative infinity and 1
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let r2 = model.score(&x, &y);
        assert!(r2 <= 1.0);
    }

    #[test]
    fn test_prediction_invariant() {
        // Property: predict(fit(X, y), X) should approximate y
        // Use non-collinear data
        let x =
            Matrix::from_vec(5, 2, vec![1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 4.0, 5.0, 5.0, 4.0]).unwrap();
        // y = 2*x1 + 3*x2 + 1
        let y = Vector::from_slice(&[6.0, 14.0, 13.0, 24.0, 23.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let predictions = model.predict(&x);

        for i in 0..y.len() {
            assert!((predictions[i] - y[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn test_coefficients_length_invariant() {
        // Property: coefficients.len() == n_features
        // Use well-conditioned data with independent columns
        let x = Matrix::from_vec(
            6,
            3,
            vec![
                1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                0.0, 1.0,
            ],
        )
        .unwrap();
        let y = Vector::from_slice(&[1.0, 2.0, 3.0, 3.0, 5.0, 4.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        assert_eq!(model.coefficients().len(), 3);
    }

    #[test]
    fn test_larger_dataset() {
        // Test with more samples
        let n = 100;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x_val = i as f32;
            x_data.push(x_val);
            y_data.push(2.0 * x_val + 3.0); // y = 2x + 3
        }

        let x = Matrix::from_vec(n, 1, x_data).unwrap();
        let y = Vector::from_vec(y_data);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-3);
        assert!((model.intercept() - 3.0).abs() < 1e-3);
    }

    #[test]
    fn test_single_sample_single_feature() {
        // Edge case: minimum viable data
        let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
        let y = Vector::from_slice(&[3.0, 5.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // y = 2x + 1
        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-4);
        assert!((model.intercept() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_negative_values() {
        // Test with negative coefficients and values
        let x = Matrix::from_vec(4, 1, vec![-2.0, -1.0, 0.0, 1.0]).unwrap();
        let y = Vector::from_slice(&[5.0, 3.0, 1.0, -1.0]); // y = -2x + 1

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        assert!((coef[0] - (-2.0)).abs() < 1e-4);
        assert!((model.intercept() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_large_values() {
        // Test numerical stability with large values
        let x = Matrix::from_vec(3, 1, vec![1000.0, 2000.0, 3000.0]).unwrap();
        let y = Vector::from_slice(&[2001.0, 4001.0, 6001.0]); // y = 2x + 1

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-2);
        assert!((model.intercept() - 1.0).abs() < 10.0); // Larger tolerance for large values
    }

    #[test]
    fn test_small_values() {
        // Test with small values
        let x = Matrix::from_vec(3, 1, vec![0.001, 0.002, 0.003]).unwrap();
        let y = Vector::from_slice(&[0.003, 0.005, 0.007]); // y = 2x + 0.001

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-2);
    }

    #[test]
    fn test_zero_intercept_data() {
        // Data that should produce zero intercept
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let y = Vector::from_slice(&[2.0, 4.0, 6.0]); // y = 2x

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        assert!((coef[0] - 2.0).abs() < 1e-4);
        assert!(model.intercept().abs() < 1e-4);
    }

    #[test]
    fn test_constant_target() {
        // All y values are the same
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let y = Vector::from_slice(&[5.0, 5.0, 5.0]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        // Coefficient should be ~0, intercept should be ~5
        let coef = model.coefficients();
        assert!(coef[0].abs() < 1e-4);
        assert!((model.intercept() - 5.0).abs() < 1e-4);
    }

    #[test]
    fn test_r2_score_bounds() {
        // R² should be in reasonable range for good fit
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let y = Vector::from_slice(&[2.1, 3.9, 6.1, 7.9, 10.1]);

        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let r2 = model.score(&x, &y);
        assert!(r2 > 0.0);
        assert!(r2 <= 1.0);
    }

    #[test]
    fn test_extrapolation() {
        // Test prediction outside training range
        let x_train = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
        let y_train = Vector::from_slice(&[2.0, 4.0, 6.0]); // y = 2x

        let mut model = LinearRegression::new();
        model.fit(&x_train, &y_train).unwrap();

        // Predict at x = 10 (extrapolation)
        let x_test = Matrix::from_vec(1, 1, vec![10.0]).unwrap();
        let predictions = model.predict(&x_test);

        assert!((predictions[0] - 20.0).abs() < 1e-4);
    }

    #[test]
    fn test_underdetermined_system_with_intercept() {
        // n_samples < n_features + 1 (underdetermined with intercept)
        // 3 samples, 5 features, fit_intercept=true means we need 6 parameters
        let x = Matrix::from_vec(
            3,
            5,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            ],
        )
        .unwrap();
        let y = Vector::from_vec(vec![10.0, 20.0, 30.0]);

        let mut model = LinearRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        // Should mention samples, features, and suggest solutions
        assert!(
            error_msg.contains("samples") || error_msg.contains("features"),
            "Error message should mention samples or features: {}",
            error_msg
        );
    }

    #[test]
    fn test_underdetermined_system_without_intercept() {
        // n_samples < n_features (underdetermined without intercept)
        // 3 samples, 5 features, fit_intercept=false
        let x = Matrix::from_vec(
            3,
            5,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 4.0, 5.0, 6.0, 3.0, 4.0, 5.0, 6.0, 7.0,
            ],
        )
        .unwrap();
        let y = Vector::from_vec(vec![10.0, 20.0, 30.0]);

        let mut model = LinearRegression::new().with_intercept(false);
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let error_msg = result.unwrap_err();
        assert!(
            error_msg.contains("samples") || error_msg.contains("features"),
            "Error message should be helpful: {}",
            error_msg
        );
    }

    #[test]
    fn test_exactly_determined_system() {
        // n_samples == n_features + 1 (exactly determined with intercept)
        // 4 samples, 3 features, fit_intercept=true means 4 parameters
        let x = Matrix::from_vec(
            4,
            3,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let y = Vector::from_vec(vec![1.0, 2.0, 3.0, 6.0]);

        let mut model = LinearRegression::new();
        let result = model.fit(&x, &y);

        // This should succeed (exactly determined)
        assert!(result.is_ok(), "Exactly determined system should work");
    }
}
