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
    // Traits are tested via their implementations
}
