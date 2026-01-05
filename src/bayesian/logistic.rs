//! Bayesian Logistic Regression with Laplace Approximation
//!
//! This module implements Bayesian logistic regression using the Laplace approximation,
//! which provides a Gaussian approximation to the posterior distribution.
//!
//! # Theory
//!
//! Model: y ~ Bernoulli(σ(Xβ)), where σ is the sigmoid function
//! Prior: β ~ N(0, λ^(-1)I), where λ is the precision (inverse variance)
//!
//! The posterior is approximated as:
//! β|y,X ~ `N(β_MAP`, H^(-1))
//!
//! where:
//! - `β_MAP` is the maximum a posteriori estimate
//! - H is the Hessian of the negative log posterior at `β_MAP`

use crate::error::{AprenderError, Result};
use crate::primitives::{Matrix, Vector};

/// Bayesian Logistic Regression with Laplace approximation.
///
/// Uses a Gaussian approximation to the posterior distribution around
/// the MAP (Maximum A Posteriori) estimate.
///
/// # Example
///
/// ```ignore
/// use aprender::bayesian::BayesianLogisticRegression;
/// use aprender::primitives::{Matrix, Vector};
///
/// let mut model = BayesianLogisticRegression::new(1.0); // precision = 1.0
/// model.fit(&x_train, &y_train).unwrap();
///
/// let probas = model.predict_proba(&x_test).unwrap();
/// let (lower, upper) = model.predict_proba_interval(&x_test, 0.95).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct BayesianLogisticRegression {
    /// Prior precision λ (inverse variance)
    prior_precision: f32,

    /// Learning rate for gradient ascent
    learning_rate: f32,

    /// Maximum iterations for MAP estimation
    max_iter: usize,

    /// Convergence tolerance
    tol: f32,

    /// MAP estimate of coefficients (`β_MAP`)
    coefficients_map: Option<Vec<f32>>,

    /// Posterior covariance H^(-1) (inverse Hessian at MAP)
    posterior_covariance: Option<Vec<Vec<f32>>>,
}

impl BayesianLogisticRegression {
    /// Creates a new Bayesian Logistic Regression with specified prior precision.
    ///
    /// # Arguments
    ///
    /// * `prior_precision` - Precision λ of the Gaussian prior (λ = 1/σ²)
    ///   - Higher precision = stronger regularization
    ///   - Typical values: 0.1 - 10.0
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::bayesian::BayesianLogisticRegression;
    ///
    /// let model = BayesianLogisticRegression::new(1.0);
    /// ```
    #[must_use]
    pub fn new(prior_precision: f32) -> Self {
        Self {
            prior_precision,
            learning_rate: 0.1, // Increased from 0.01 for better convergence
            max_iter: 1000,
            tol: 1e-4,
            coefficients_map: None,
            posterior_covariance: None,
        }
    }

    /// Sets the learning rate for MAP estimation.
    #[must_use]
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
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
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sigmoid function: σ(z) = 1 / (1 + e^(-z))
    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Returns the MAP estimate of coefficients (available after fitting).
    #[must_use]
    pub fn coefficients_map(&self) -> Option<&[f32]> {
        self.coefficients_map.as_deref()
    }

    /// Returns the posterior covariance matrix (available after fitting).
    #[must_use]
    pub fn posterior_covariance(&self) -> Option<&[Vec<f32>]> {
        self.posterior_covariance.as_deref()
    }

    /// Fits the Bayesian Logistic Regression using Laplace approximation.
    ///
    /// Finds the MAP estimate `β_MAP` and computes the Hessian for uncertainty.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (n × p)
    /// * `y` - Binary labels (n), must be 0.0 or 1.0
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if dimensions mismatch or convergence fails
    #[allow(clippy::needless_range_loop)]
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

        // Validate binary labels
        for &label in y.as_slice() {
            if label != 0.0 && label != 1.0 {
                return Err(AprenderError::Other(format!(
                    "Labels must be 0.0 or 1.0, got {label}"
                )));
            }
        }

        // Initialize coefficients to zero
        let mut beta = vec![0.0_f32; p];

        // Gradient ascent to find MAP estimate
        for iter in 0..self.max_iter {
            // Compute predictions: p_i = σ(x_i^T β)
            let mut predictions = Vec::with_capacity(n);
            for i in 0..n {
                let mut z = 0.0_f32;
                for j in 0..p {
                    z += x.get(i, j) * beta[j];
                }
                predictions.push(Self::sigmoid(z));
            }

            // Compute gradient: ∇ℓ = X^T(y - p) - λβ
            let mut gradient = vec![0.0_f32; p];
            for j in 0..p {
                let mut grad_j = 0.0_f32;
                for i in 0..n {
                    grad_j += x.get(i, j) * (y[i] - predictions[i]);
                }
                // Average by number of samples for numerical stability
                grad_j /= n as f32;
                // Add prior gradient: -λβ_j
                grad_j -= self.prior_precision * beta[j];
                gradient[j] = grad_j;
            }

            // Update parameters: β ← β + η∇ℓ
            let mut max_update = 0.0_f32;
            for j in 0..p {
                let update = self.learning_rate * gradient[j];
                beta[j] += update;
                max_update = max_update.max(update.abs());
            }

            // Check convergence
            if max_update < self.tol {
                break;
            }

            // Prevent infinite loop
            if iter == self.max_iter - 1 {
                return Err(AprenderError::Other(format!(
                    "MAP estimation did not converge in {} iterations",
                    self.max_iter
                )));
            }
        }

        // Store MAP estimate
        self.coefficients_map = Some(beta.clone());

        // Compute Hessian H = X^T W X + λI
        // where W is diagonal with W_ii = p_i(1 - p_i)
        let mut hessian = vec![vec![0.0_f32; p]; p];

        // Compute predictions at MAP
        let mut predictions = Vec::with_capacity(n);
        for i in 0..n {
            let mut z = 0.0_f32;
            for j in 0..p {
                z += x.get(i, j) * beta[j];
            }
            predictions.push(Self::sigmoid(z));
        }

        // H = X^T W X
        for i in 0..p {
            for j in 0..p {
                let mut h_ij = 0.0_f32;
                for k in 0..n {
                    let w_k = predictions[k] * (1.0 - predictions[k]);
                    h_ij += x.get(k, i) * w_k * x.get(k, j);
                }
                hessian[i][j] = h_ij;
            }
            // Add prior precision to diagonal: H_ii += λ
            hessian[i][i] += self.prior_precision;
        }

        // Store the Hessian matrix. The posterior covariance Σ = H^(-1) is not
        // explicitly computed; instead linear systems involving H are solved
        // as needed using Cholesky decomposition during prediction. This is
        // more numerically stable and efficient for large feature dimensions.
        self.posterior_covariance = Some(hessian);

        Ok(())
    }

    /// Predicts class probabilities for test data.
    ///
    /// Returns the probability of class 1 for each sample.
    ///
    /// # Arguments
    ///
    /// * `x_test` - Test feature matrix (`n_test` × p)
    ///
    /// # Returns
    ///
    /// Vector of probabilities P(y=1 | x)
    #[allow(clippy::needless_range_loop)]
    pub fn predict_proba(&self, x_test: &Matrix<f32>) -> Result<Vector<f32>> {
        let beta = self.coefficients_map.as_ref().ok_or_else(|| {
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

        let mut probas = Vec::with_capacity(n);
        for i in 0..n {
            let mut z = 0.0_f32;
            for j in 0..p {
                z += x_test.get(i, j) * beta[j];
            }
            probas.push(Self::sigmoid(z));
        }

        Ok(Vector::from_vec(probas))
    }

    /// Predicts class labels (0 or 1) for test data.
    ///
    /// Uses threshold 0.5: predict 1 if P(y=1|x) >= 0.5, else 0.
    pub fn predict(&self, x_test: &Matrix<f32>) -> Result<Vector<f32>> {
        let probas = self.predict_proba(x_test)?;
        let labels: Vec<f32> = probas
            .as_slice()
            .iter()
            .map(|&p| if p >= 0.5 { 1.0 } else { 0.0 })
            .collect();
        Ok(Vector::from_vec(labels))
    }

    /// Predicts class probabilities with credible intervals.
    ///
    /// Uses the Laplace approximation to compute Bayesian credible intervals
    /// for predicted probabilities.
    ///
    /// # Arguments
    ///
    /// * `x_test` - Test feature matrix (`n_test` × p)
    /// * `level` - Confidence level (e.g., 0.95 for 95% credible interval)
    ///
    /// # Returns
    ///
    /// Tuple of (`lower_bounds`, `upper_bounds`) for P(y=1|x)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let (lower, upper) = model.predict_proba_interval(&x_test, 0.95)?;
    /// ```
    #[allow(clippy::needless_range_loop)]
    pub fn predict_proba_interval(
        &self,
        x_test: &Matrix<f32>,
        level: f32,
    ) -> Result<(Vector<f32>, Vector<f32>)> {
        let beta = self.coefficients_map.as_ref().ok_or_else(|| {
            AprenderError::Other("Model not fitted yet. Call fit() first.".into())
        })?;

        let hessian = self
            .posterior_covariance
            .as_ref()
            .ok_or_else(|| AprenderError::Other("Posterior covariance not available.".into()))?;

        let n = x_test.n_rows();
        let p = x_test.n_cols();

        if p != beta.len() {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{} features", beta.len()),
                actual: format!("{p} columns in x_test"),
            });
        }

        // Convert Hessian to Matrix for Cholesky operations
        let hessian_matrix = Matrix::from_vec(p, p, hessian.iter().flatten().copied().collect())
            .map_err(|e| AprenderError::Other(format!("Hessian matrix error: {e}")))?;

        // Compute z-score for the desired credible level
        // For 95%, z ≈ 1.96; for 90%, z ≈ 1.645
        let z_score = match level {
            x if (x - 0.95).abs() < 0.01 => 1.96_f32,
            x if (x - 0.90).abs() < 0.01 => 1.645_f32,
            x if (x - 0.99).abs() < 0.01 => 2.576_f32,
            _ => {
                // General formula: inverse CDF of standard normal
                // Approximation: for level α, z ≈ Φ^(-1)((1 + α)/2)
                // For simplicity, use 1.96 as default
                1.96_f32
            }
        };

        let mut lower_bounds = Vec::with_capacity(n);
        let mut upper_bounds = Vec::with_capacity(n);

        for i in 0..n {
            // Compute z_mean = x_i^T β_MAP
            let mut z_mean = 0.0_f32;
            for j in 0..p {
                z_mean += x_test.get(i, j) * beta[j];
            }

            // Compute z_var = x_i^T H^(-1) x_i
            // We solve H v = x_i, then compute v^T x_i
            let x_i = (0..p).map(|j| x_test.get(i, j)).collect::<Vec<_>>();
            let x_i_vec = Vector::from_vec(x_i.clone());

            // Solve H v = x_i using Cholesky
            let v = hessian_matrix
                .cholesky_solve(&x_i_vec)
                .map_err(|e| AprenderError::Other(format!("Cholesky solve failed: {e}")))?;

            // Compute z_var = v^T x_i = sum(v[j] * x_i[j])
            let mut z_var = 0.0_f32;
            for j in 0..p {
                z_var += v[j] * x_i[j];
            }

            // Ensure variance is non-negative
            if z_var < 0.0 {
                z_var = 0.0;
            }

            let z_std = z_var.sqrt();

            // Compute credible interval for z
            let z_lower = z_mean - z_score * z_std;
            let z_upper = z_mean + z_score * z_std;

            // Apply sigmoid to get probability bounds
            let p_lower = Self::sigmoid(z_lower);
            let p_upper = Self::sigmoid(z_upper);

            lower_bounds.push(p_lower);
            upper_bounds.push(p_upper);
        }

        Ok((
            Vector::from_vec(lower_bounds),
            Vector::from_vec(upper_bounds),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test: Constructor creates model with correct defaults
    #[test]
    fn test_new() {
        let model = BayesianLogisticRegression::new(1.0);
        assert!(model.coefficients_map.is_none());
        assert!(model.posterior_covariance.is_none());
    }

    /// Test: Builder pattern methods
    #[test]
    fn test_builder_pattern() {
        let model = BayesianLogisticRegression::new(1.0)
            .with_learning_rate(0.1)
            .with_max_iter(500)
            .with_tolerance(1e-3);

        // Model should be created successfully
        assert!(model.coefficients_map.is_none());
    }

    /// Test: Fit with simple linearly separable data
    #[test]
    fn test_fit_simple() {
        // Linearly separable data: y = 1 if x > 0, else 0
        let x =
            Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = BayesianLogisticRegression::new(0.1);
        let result = model.fit(&x, &y);

        assert!(result.is_ok(), "Fit should succeed");
        assert!(model.coefficients_map.is_some());
        assert!(model.posterior_covariance.is_some());

        // Coefficient should be positive (positive correlation)
        let beta = model
            .coefficients_map
            .as_ref()
            .expect("MAP estimate exists");
        assert!(
            beta[0] > 0.0,
            "Coefficient should be positive, got {}",
            beta[0]
        );
    }

    /// Test: Predict probabilities
    #[test]
    fn test_predict_proba() {
        // Train on simple data
        let x = Matrix::from_vec(4, 1, vec![-1.0, -0.5, 0.5, 1.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = BayesianLogisticRegression::new(0.1);
        model.fit(&x, &y).expect("Fit succeeds");

        // Predict on new data
        let x_test = Matrix::from_vec(3, 1, vec![-2.0, 0.0, 2.0]).expect("Valid test matrix");
        let probas = model.predict_proba(&x_test).expect("Prediction succeeds");

        assert_eq!(probas.len(), 3);

        // Probabilities should be in [0, 1]
        for &p in probas.as_slice() {
            assert!(
                (0.0..=1.0).contains(&p),
                "Probability should be in [0,1], got {p}"
            );
        }

        // Probabilities should be monotonically increasing
        assert!(probas[0] < probas[1], "P(y=1 | x=-2) < P(y=1 | x=0)");
        assert!(probas[1] < probas[2], "P(y=1 | x=0) < P(y=1 | x=2)");
    }

    /// Test: Predict binary labels
    #[test]
    fn test_predict() {
        let x = Matrix::from_vec(4, 1, vec![-1.0, -0.5, 0.5, 1.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = BayesianLogisticRegression::new(0.1);
        model.fit(&x, &y).expect("Fit succeeds");

        let x_test = Matrix::from_vec(2, 1, vec![-2.0, 2.0]).expect("Valid test matrix");
        let labels = model.predict(&x_test).expect("Prediction succeeds");

        assert_eq!(labels.len(), 2);

        // Labels should be 0.0 or 1.0
        for &label in labels.as_slice() {
            assert!(
                label == 0.0 || label == 1.0,
                "Label should be 0 or 1, got {label}"
            );
        }
    }

    /// Test: Dimension mismatch in fit
    #[test]
    fn test_fit_dimension_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 1.0]); // Wrong size!

        let mut model = BayesianLogisticRegression::new(1.0);
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: Invalid labels (not 0 or 1)
    #[test]
    fn test_fit_invalid_labels() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.5, 1.0]); // 0.5 is invalid!

        let mut model = BayesianLogisticRegression::new(1.0);
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }

    /// Test: Predict before fit should error
    #[test]
    fn test_predict_not_fitted() {
        let model = BayesianLogisticRegression::new(1.0);
        let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix");

        let result = model.predict_proba(&x_test);
        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }

    /// Test: Predict with dimension mismatch
    #[test]
    fn test_predict_dimension_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.0, 1.0, 1.0]);

        let mut model = BayesianLogisticRegression::new(1.0);
        model.fit(&x, &y).expect("Fit succeeds");

        // Try to predict with wrong number of features
        let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid test matrix");
        let result = model.predict_proba(&x_test);

        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::DimensionMismatch { .. }));
    }

    /// Test: MAP estimate converges
    #[test]
    fn test_map_convergence() {
        let x =
            Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = BayesianLogisticRegression::new(0.1)
            .with_max_iter(2000)
            .with_tolerance(1e-5);

        let result = model.fit(&x, &y);
        assert!(result.is_ok(), "MAP estimation should converge");
    }

    /// Test: Non-convergence with low max_iter
    #[test]
    fn test_map_non_convergence() {
        let x =
            Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = BayesianLogisticRegression::new(0.1)
            .with_max_iter(5) // Too few iterations
            .with_tolerance(1e-10); // Very strict tolerance

        let result = model.fit(&x, &y);
        assert!(result.is_err(), "Should fail to converge");
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }

    /// Test: Predict probabilities with credible intervals
    #[test]
    fn test_predict_proba_interval() {
        let x =
            Matrix::from_vec(6, 1, vec![-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]).expect("Valid matrix");
        let y = Vector::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let mut model = BayesianLogisticRegression::new(0.1);
        model.fit(&x, &y).expect("Fit succeeds");

        // Predict with 95% credible intervals
        let x_test = Matrix::from_vec(3, 1, vec![-2.0, 0.0, 2.0]).expect("Valid test matrix");
        let (lower, upper) = model
            .predict_proba_interval(&x_test, 0.95)
            .expect("Interval prediction succeeds");

        assert_eq!(lower.len(), 3);
        assert_eq!(upper.len(), 3);

        // Get point predictions
        let probas = model.predict_proba(&x_test).expect("Prediction succeeds");

        // Bounds should contain the point predictions
        for i in 0..3 {
            assert!(
                lower[i] <= probas[i],
                "Lower bound should be <= point estimate: {i}: {} <= {}",
                lower[i],
                probas[i]
            );
            assert!(
                probas[i] <= upper[i],
                "Upper bound should be >= point estimate: {i}: {} >= {}",
                probas[i],
                upper[i]
            );
            assert!(
                lower[i] >= 0.0 && lower[i] <= 1.0,
                "Lower bound should be in [0,1], got {}",
                lower[i]
            );
            assert!(
                upper[i] >= 0.0 && upper[i] <= 1.0,
                "Upper bound should be in [0,1], got {}",
                upper[i]
            );
        }

        // Intervals should have non-negative width (may be small for certain x values)
        for i in 0..3 {
            assert!(
                upper[i] >= lower[i],
                "Upper bound should be >= lower bound at {i}: {} >= {}",
                upper[i],
                lower[i]
            );
        }

        // At least some intervals should have meaningful width
        let max_width = (0..3).map(|i| upper[i] - lower[i]).fold(0.0_f32, f32::max);
        assert!(
            max_width > 0.01,
            "At least one interval should have width > 0.01, max was {max_width}"
        );
    }

    /// Test: Interval prediction before fit should error
    #[test]
    fn test_predict_interval_not_fitted() {
        let model = BayesianLogisticRegression::new(1.0);
        let x_test = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("Valid matrix");

        let result = model.predict_proba_interval(&x_test, 0.95);
        assert!(result.is_err());
        let err = result.expect_err("Should be an error");
        assert!(matches!(err, AprenderError::Other(_)));
    }
}
