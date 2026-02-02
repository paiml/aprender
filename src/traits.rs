//! Core traits for ML estimators and transformers.
//!
//! These traits define the API contracts for all ML algorithms.

use crate::error::Result;
use crate::primitives::{Matrix, Vector};

/// Primary trait for supervised learning estimators.
///
/// Estimators implement fit/predict/score following sklearn conventions.
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// // Create training data: y = 2x + 1
/// let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y_train = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
///
/// // Test data
/// let x_test = Matrix::from_vec(2, 1, vec![5.0, 6.0]).unwrap();
/// let y_test = Vector::from_slice(&[11.0, 13.0]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x_train, &y_train).unwrap();
/// let predictions = model.predict(&x_test);
/// let score = model.score(&x_test, &y_test);
/// assert!(score > 0.99);
/// ```
pub trait Estimator {
    /// Fits the model to training data.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (dimension mismatch, singular matrix, etc.).
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()>;

    /// Predicts target values for input data.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32>;

    /// Computes the score (R² for regression, accuracy for classification).
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32;
}

/// Trait for unsupervised learning models.
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// // Create data with 2 clear clusters
/// let data = Matrix::from_vec(6, 2, vec![
///     0.0, 0.0, 0.1, 0.1, 0.2, 0.0,  // Cluster 1
///     10.0, 10.0, 10.1, 10.1, 10.0, 10.2,  // Cluster 2
/// ]).unwrap();
///
/// let mut kmeans = KMeans::new(2).with_random_state(42);
/// kmeans.fit(&data).unwrap();
/// let labels = kmeans.predict(&data);
/// assert_eq!(labels.len(), 6);
/// ```
pub trait UnsupervisedEstimator {
    /// The type of labels/clusters produced.
    type Labels;

    /// Fits the model to data.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails (empty data, invalid parameters, etc.).
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()>;

    /// Predicts cluster assignments or transforms data.
    fn predict(&self, x: &Matrix<f32>) -> Self::Labels;
}

/// Trait for data transformers (scalers, encoders, etc.).
///
/// This trait defines the interface for preprocessing transformers.
/// Implementations include scalers, encoders, and feature transformers.
///
/// # Future Usage
///
/// ```text
/// let mut scaler = StandardScaler::new();
/// let x_scaled = scaler.fit_transform(&x)?;
/// let x_test_scaled = scaler.transform(&x_test)?;
/// ```
pub trait Transformer {
    /// Fits the transformer to data.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()>;

    /// Transforms data using fitted parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if transformer is not fitted.
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>>;

    /// Fits and transforms in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&mut self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::AprenderError;

    // Mock transformer to test trait default methods
    struct MockTransformer {
        fitted: bool,
        scale: f32,
    }

    impl MockTransformer {
        fn new() -> Self {
            Self {
                fitted: false,
                scale: 1.0,
            }
        }
    }

    impl Transformer for MockTransformer {
        fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
            if x.n_rows() == 0 {
                return Err(AprenderError::DimensionMismatch {
                    expected: "non-empty matrix".to_string(),
                    actual: "empty matrix (0 rows)".to_string(),
                });
            }
            // Compute mean for scaling
            let mut sum = 0.0;
            for row in 0..x.n_rows() {
                for col in 0..x.n_cols() {
                    sum += x.get(row, col);
                }
            }
            let total = x.n_rows() * x.n_cols();
            self.scale = if total > 0 { sum / total as f32 } else { 1.0 };
            if self.scale == 0.0 {
                self.scale = 1.0;
            }
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
            if !self.fitted {
                return Err(AprenderError::ValidationError {
                    message: "MockTransformer not fitted".to_string(),
                });
            }
            let mut data = Vec::with_capacity(x.n_rows() * x.n_cols());
            for row in 0..x.n_rows() {
                for col in 0..x.n_cols() {
                    data.push(x.get(row, col) / self.scale);
                }
            }
            Matrix::from_vec(x.n_rows(), x.n_cols(), data).map_err(|e| {
                AprenderError::ValidationError {
                    message: e.to_string(),
                }
            })
        }
    }

    #[test]
    fn test_transformer_fit_transform_default() {
        let mut transformer = MockTransformer::new();
        let x = Matrix::from_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]).expect("matrix");

        // fit_transform uses default implementation
        let result = transformer.fit_transform(&x);
        assert!(result.is_ok());

        let transformed = result.expect("should succeed");
        assert_eq!(transformed.n_rows(), 2);
        assert_eq!(transformed.n_cols(), 2);

        // Verify transformer was fitted
        assert!(transformer.fitted);
    }

    #[test]
    fn test_transformer_fit_then_transform() {
        let mut transformer = MockTransformer::new();
        let x = Matrix::from_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]).expect("matrix");

        // Separate fit and transform
        transformer.fit(&x).expect("fit should succeed");
        assert!(transformer.fitted);

        let transformed = transformer.transform(&x).expect("transform should succeed");
        assert_eq!(transformed.n_rows(), 2);
    }

    #[test]
    fn test_transformer_transform_without_fit() {
        let transformer = MockTransformer::new();
        let x = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix");

        let result = transformer.transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_fit_empty_matrix() {
        let mut transformer = MockTransformer::new();
        let x = Matrix::from_vec(0, 2, vec![]).expect("matrix");

        let result = transformer.fit(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_fit_transform_empty_fails() {
        let mut transformer = MockTransformer::new();
        let x = Matrix::from_vec(0, 0, vec![]).expect("matrix");

        let result = transformer.fit_transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_transformer_fit_all_zeros_resets_scale() {
        // When all input values are 0.0, sum/total = 0.0, hitting the
        // `if self.scale == 0.0 { self.scale = 1.0; }` branch (lines 154-155).
        let mut transformer = MockTransformer::new();
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 0.0, 0.0]).expect("matrix");

        transformer.fit(&x).expect("fit should succeed");
        assert!(transformer.fitted);
        // scale should have been reset from 0.0 to 1.0
        assert!((transformer.scale - 1.0).abs() < f32::EPSILON);

        // transform should work and produce the original values (divided by 1.0)
        let result = transformer.transform(&x).expect("transform should succeed");
        assert_eq!(result.n_rows(), 2);
        assert_eq!(result.n_cols(), 2);
        for row in 0..result.n_rows() {
            for col in 0..result.n_cols() {
                assert!((result.get(row, col) - 0.0).abs() < f32::EPSILON);
            }
        }
    }

    #[test]
    fn test_transformer_fit_zero_cols_total_zero_branch() {
        // A matrix with rows > 0 but cols == 0 produces total = 0,
        // hitting the `else { 1.0 }` branch in the `if total > 0` expression (line 153).
        let mut transformer = MockTransformer::new();
        let x = Matrix::from_vec(2, 0, vec![]).expect("matrix");

        transformer
            .fit(&x)
            .expect("fit should succeed with zero-col matrix");
        assert!(transformer.fitted);
        // With total == 0, scale should be set to 1.0 via the else branch,
        // then the scale == 0.0 guard doesn't fire because scale is already 1.0.
        assert!((transformer.scale - 1.0).abs() < f32::EPSILON);
    }

    // A transformer mock whose transform deliberately produces a data
    // length mismatch, triggering the map_err closure (lines 173-177).
    struct BrokenTransformer {
        fitted: bool,
    }

    impl BrokenTransformer {
        fn new() -> Self {
            Self { fitted: false }
        }
    }

    impl Transformer for BrokenTransformer {
        fn fit(&mut self, _x: &Matrix<f32>) -> Result<()> {
            self.fitted = true;
            Ok(())
        }

        fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>> {
            if !self.fitted {
                return Err(AprenderError::ValidationError {
                    message: "BrokenTransformer not fitted".to_string(),
                });
            }
            // Deliberately produce wrong number of elements to trigger
            // the Matrix::from_vec error and exercise the map_err path.
            let wrong_data = vec![0.0_f32; x.n_rows() * x.n_cols() + 1];
            Matrix::from_vec(x.n_rows(), x.n_cols(), wrong_data).map_err(|e| {
                AprenderError::ValidationError {
                    message: e.to_string(),
                }
            })
        }
    }

    #[test]
    fn test_broken_transformer_map_err_path() {
        // Exercises the map_err closure that converts a Matrix::from_vec
        // error into AprenderError::ValidationError.
        let mut transformer = BrokenTransformer::new();
        let x = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix");

        transformer.fit(&x).expect("fit should succeed");
        let result = transformer.transform(&x);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Data length must equal rows * cols"),
            "Expected dimension mismatch error, got: {err_msg}"
        );
    }

    #[test]
    fn test_broken_transformer_fit_transform_propagates_transform_error() {
        // Verifies that the default fit_transform correctly propagates
        // errors from the transform step (not just from fit).
        let mut transformer = BrokenTransformer::new();
        let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix");

        let result = transformer.fit_transform(&x);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Data length must equal rows * cols"));
    }

    #[test]
    fn test_transformer_transform_without_fit_error_message() {
        // Verify the exact error variant and message content, not just is_err().
        let transformer = MockTransformer::new();
        let x = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix");

        let result = transformer.transform(&x);
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("MockTransformer not fitted"),
            "Expected 'not fitted' message, got: {msg}"
        );
    }

    #[test]
    fn test_transformer_fit_empty_error_message() {
        // Verify the exact error variant and message for empty matrix fit.
        let mut transformer = MockTransformer::new();
        let x = Matrix::from_vec(0, 2, vec![]).expect("matrix");

        let result = transformer.fit(&x);
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("dimension mismatch"),
            "Expected dimension mismatch error, got: {msg}"
        );
        assert!(msg.contains("empty matrix"));
    }

    #[test]
    fn test_transformer_fit_transform_verifies_scaling() {
        // Verify actual numerical correctness of transformed values.
        let mut transformer = MockTransformer::new();
        // Mean of [2.0, 4.0, 6.0, 8.0] = 20.0 / 4 = 5.0
        let x = Matrix::from_vec(2, 2, vec![2.0, 4.0, 6.0, 8.0]).expect("matrix");

        let result = transformer
            .fit_transform(&x)
            .expect("fit_transform should succeed");
        // Each value divided by scale (5.0)
        assert!((result.get(0, 0) - 0.4).abs() < f32::EPSILON); // 2.0 / 5.0
        assert!((result.get(0, 1) - 0.8).abs() < f32::EPSILON); // 4.0 / 5.0
        assert!((result.get(1, 0) - 1.2).abs() < f32::EPSILON); // 6.0 / 5.0
        assert!((result.get(1, 1) - 1.6).abs() < f32::EPSILON); // 8.0 / 5.0
    }
}
