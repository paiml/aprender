//! Core traits for ML estimators and transformers.
//!
//! These traits define the API contracts for all ML algorithms.

use crate::primitives::{Matrix, Vector};

/// Primary trait for supervised learning estimators.
///
/// Estimators implement fit/predict/score following sklearn conventions.
///
/// # Examples
///
/// ```ignore
/// use aprender::prelude::*;
///
/// let mut model = LinearRegression::new();
/// model.fit(&x_train, &y_train)?;
/// let predictions = model.predict(&x_test);
/// let score = model.score(&x_test, &y_test);
/// ```
pub trait Estimator {
    /// Fits the model to training data.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<(), &'static str>;

    /// Predicts target values for input data.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32>;

    /// Computes the score (R² for regression, accuracy for classification).
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32;
}

/// Trait for unsupervised learning models.
///
/// # Examples
///
/// ```ignore
/// use aprender::prelude::*;
///
/// let mut kmeans = KMeans::new(3);
/// kmeans.fit(&data)?;
/// let labels = kmeans.predict(&data);
/// ```
pub trait UnsupervisedEstimator {
    /// The type of labels/clusters produced.
    type Labels;

    /// Fits the model to data.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit(&mut self, x: &Matrix<f32>) -> Result<(), &'static str>;

    /// Predicts cluster assignments or transforms data.
    fn predict(&self, x: &Matrix<f32>) -> Self::Labels;
}

/// Trait for data transformers (scalers, encoders, etc.).
///
/// # Examples
///
/// ```ignore
/// use aprender::prelude::*;
///
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
    fn fit(&mut self, x: &Matrix<f32>) -> Result<(), &'static str>;

    /// Transforms data using fitted parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if transformer is not fitted.
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str>;

    /// Fits and transforms in one step.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    fn fit_transform(&mut self, x: &Matrix<f32>) -> Result<Matrix<f32>, &'static str> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    // Traits are tested via their implementations
}
