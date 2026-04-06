//! Classification algorithms.
//!
//! This module implements classification algorithms including:
//! - Logistic Regression for binary classification
//! - K-Nearest Neighbors (kNN) for instance-based classification
//! - Gaussian Naive Bayes for probabilistic classification
//! - Linear Support Vector Machine (SVM) for maximum-margin classification
//! - Softmax Regression for multi-class classification (planned)
//!
//! # Example
//!
//! ```
//! use aprender::classification::LogisticRegression;
//! use aprender::prelude::*;
//!
//! // Binary classification data
//! let x = Matrix::from_vec(4, 2, vec![
//!     0.0, 0.0,
//!     0.0, 1.0,
//!     1.0, 0.0,
//!     1.0, 1.0,
//! ]).expect("Matrix dimensions match data length");
//! let y = vec![0, 0, 0, 1];
//!
//! let mut model = LogisticRegression::new()
//!     .with_learning_rate(0.1)
//!     .with_max_iter(1000);
//! model.fit(&x, &y).expect("Training data is valid with 4 samples");
//! let predictions = model.predict(&x);
//!
//! assert_eq!(predictions.len(), 4);
//! for pred in predictions {
//!     assert!(pred == 0 || pred == 1);
//! }
//! ```

use crate::error::Result;
use crate::primitives::{Matrix, Vector};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Class weighting strategy for imbalanced datasets.
///
/// # Example
///
/// ```
/// use aprender::classification::{LogisticRegression, ClassWeight};
/// use aprender::prelude::*;
///
/// let mut model = LogisticRegression::new()
///     .with_class_weight(ClassWeight::Balanced);
///
/// // Imbalanced data: 90% class 0, 10% class 1
/// let x = Matrix::from_vec(10, 2, vec![
///     0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 0.0, 0.2,
///     0.1, 0.0, 0.0, 0.1, 0.2, 0.1, 0.1, 0.2,
///     5.0, 5.0, 5.1, 5.1,
/// ]).expect("10x2 matrix");
/// let y = vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
/// model.fit(&x, &y).expect("fit succeeds");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClassWeight {
    /// No class weighting (default, backward compatible).
    Uniform,
    /// Automatic sqrt-inverse weighting: `w_k = sqrt(n_total / (n_classes * n_k))`.
    ///
    /// Upweights the minority class to counteract imbalanced label distributions.
    /// Compatible with scikit-learn `class_weight='balanced'` (with sqrt dampening).
    Balanced,
    /// Manual per-class weights: `[w_0, w_1]` for binary classification.
    Manual(Vec<f32>),
}

impl Default for ClassWeight {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Logistic Regression classifier for binary classification.
///
/// Uses sigmoid activation and binary cross-entropy loss with
/// gradient descent optimization. Supports class weighting for
/// imbalanced datasets and L2 regularization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    /// Model coefficients (weights)
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term
    intercept: f32,
    /// Learning rate for gradient descent
    learning_rate: f32,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f32,
    /// Class weighting strategy
    class_weight: ClassWeight,
    /// L2 regularization strength (weight decay). 0.0 = no regularization.
    l2_penalty: f32,
}

impl LogisticRegression {
    /// Creates a new logistic regression classifier with default parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::LogisticRegression;
    ///
    /// let model = LogisticRegression::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-4,
            class_weight: ClassWeight::Uniform,
            l2_penalty: 0.0,
        }
    }

    /// Sets the learning rate.
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

    /// Sets the class weighting strategy for imbalanced datasets.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::{LogisticRegression, ClassWeight};
    ///
    /// // Automatic balanced weighting (recommended for imbalanced data)
    /// let model = LogisticRegression::new()
    ///     .with_class_weight(ClassWeight::Balanced);
    ///
    /// // Manual weights: upweight class 1 by 3x
    /// let model = LogisticRegression::new()
    ///     .with_class_weight(ClassWeight::Manual(vec![1.0, 3.0]));
    /// ```
    #[must_use]
    pub fn with_class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Sets L2 regularization strength (weight decay).
    ///
    /// Adds `l2_penalty * ||w||^2` to the loss, penalizing large coefficients.
    /// Typical values: 1e-4 to 1e-2. Default: 0.0 (no regularization).
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::LogisticRegression;
    ///
    /// let model = LogisticRegression::new()
    ///     .with_l2_penalty(1e-4);
    /// ```
    #[must_use]
    pub fn with_l2_penalty(mut self, l2_penalty: f32) -> Self {
        self.l2_penalty = l2_penalty;
        self
    }

    /// ONE PATH: Delegates to `nn::functional::sigmoid_scalar` (UCBD §4).
    fn sigmoid(z: f32) -> f32 {
        crate::nn::functional::sigmoid_scalar(z)
    }

    /// Predicts probabilities for samples.
    ///
    /// Returns probability of class 1 for each sample.
    #[must_use]
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coef = self.coefficients.as_ref().expect("Model not fitted yet");
        let (n_samples, _) = x.shape();

        let mut probas = Vec::with_capacity(n_samples);
        for row in 0..n_samples {
            let mut z = self.intercept;
            for col in 0..coef.len() {
                z += coef[col] * x.get(row, col);
            }
            probas.push(Self::sigmoid(z));
        }

        Vector::from_vec(probas)
    }

    /// Fits the logistic regression model to training data.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (`n_samples` × `n_features`)
    /// * `y` - Binary labels (`n_samples`), must be 0 or 1
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err` with message on failure
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples in X and y must match".into());
        }
        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        // Validate labels are binary (0 or 1)
        for &label in y {
            if label != 0 && label != 1 {
                return Err("Labels must be 0 or 1 for binary classification".into());
            }
        }

        // Initialize coefficients and intercept
        self.coefficients = Some(Vector::from_vec(vec![0.0; n_features]));
        self.intercept = 0.0;

        // Compute per-class sample weights
        let sample_weights = self.compute_sample_weights(y);

        // Gradient descent optimization
        for _ in 0..self.max_iter {
            // Compute predictions (probabilities)
            let probas = self.predict_proba(x);

            // Compute gradients with class weighting
            let mut coef_grad = vec![0.0; n_features];
            let mut intercept_grad = 0.0;

            for i in 0..n_samples {
                let error = sample_weights[i] * (probas[i] - y[i] as f32);
                intercept_grad += error;
                for (j, grad) in coef_grad.iter_mut().enumerate() {
                    *grad += error * x.get(i, j);
                }
            }

            // Average gradients
            let n = n_samples as f32;
            intercept_grad /= n;
            for grad in &mut coef_grad {
                *grad /= n;
            }

            // Update parameters with L2 regularization (weight decay)
            self.intercept -= self.learning_rate * intercept_grad;
            if let Some(ref mut coef) = self.coefficients {
                for j in 0..n_features {
                    coef[j] -= self.learning_rate * (coef_grad[j] + self.l2_penalty * coef[j]);
                }
            }

            // Check convergence (simplified - could check gradient norm)
            if intercept_grad.abs() < self.tol && coef_grad.iter().all(|&g| g.abs() < self.tol) {
                break;
            }
        }

        Ok(())
    }

    /// Predicts class labels for samples.
    ///
    /// Returns 0 or 1 for each sample based on probability threshold of 0.5.
    #[must_use]
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let probas = self.predict_proba(x);
        probas
            .as_slice()
            .iter()
            .map(|&p| usize::from(p >= 0.5))
            .collect()
    }

    /// Computes accuracy score on test data.
    ///
    /// Returns fraction of correctly classified samples.
    #[must_use]
    pub fn score(&self, x: &Matrix<f32>, y: &[usize]) -> f32 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();
        correct as f32 / y.len() as f32
    }

    /// Get model coefficients (weights).
    ///
    /// # Panics
    ///
    /// Panics if the model is not fitted.
    #[must_use]
    pub fn coefficients(&self) -> &Vector<f32> {
        self.coefficients.as_ref().expect("Model not fitted")
    }

    /// Get intercept (bias) term.
    #[must_use]
    pub fn intercept(&self) -> f32 {
        self.intercept
    }

    /// Compute per-sample weights from the class weighting strategy.
    fn compute_sample_weights(&self, y: &[usize]) -> Vec<f32> {
        match &self.class_weight {
            ClassWeight::Uniform => vec![1.0; y.len()],
            ClassWeight::Balanced => {
                let n = y.len() as f32;
                let n_class_0 = y.iter().filter(|&&l| l == 0).count() as f32;
                let n_class_1 = n - n_class_0;
                if n_class_0 == 0.0 || n_class_1 == 0.0 {
                    return vec![1.0; y.len()];
                }
                // sqrt-inverse weighting: w_k = sqrt(n / (2 * n_k))
                let w0 = (n / (2.0 * n_class_0)).sqrt();
                let w1 = (n / (2.0 * n_class_1)).sqrt();
                y.iter().map(|&l| if l == 0 { w0 } else { w1 }).collect()
            }
            ClassWeight::Manual(weights) => {
                if weights.len() < 2 {
                    return vec![1.0; y.len()];
                }
                y.iter()
                    .map(|&l| if l < weights.len() { weights[l] } else { 1.0 })
                    .collect()
            }
        }
    }

    /// Saves the trained model to `SafeTensors` format.
    ///
    /// `SafeTensors` is an industry-standard model serialization format
    /// compatible with `HuggingFace`, Ollama, `PyTorch`, TensorFlow, and realizar.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to save the model
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not fitted (call `fit()` first)
    /// - File writing fails
    /// - Serialization fails
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::LogisticRegression;
    /// use aprender::prelude::*;
    ///
    /// let mut model = LogisticRegression::new();
    /// let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).expect("4x2 matrix with 8 values");
    /// let y = vec![0, 0, 1, 1];
    /// model.fit(&x, &y).expect("Valid training data");
    ///
    /// model.save_safetensors("model.safetensors").expect("Model is fitted and path is writable");
    /// ```
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
    /// # Arguments
    ///
    /// * `path` - File path to load the model from
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - `SafeTensors` format is invalid
    /// - Required tensors are missing
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::LogisticRegression;
    ///
    /// # use aprender::prelude::*;
    /// # let mut model = LogisticRegression::new();
    /// # let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).expect("4x2 matrix with 8 values");
    /// # let y = vec![0, 0, 1, 1];
    /// # model.fit(&x, &y).expect("Valid training data");
    /// # model.save_safetensors("/tmp/doctest_logistic_model.safetensors").expect("Can save to /tmp");
    /// let loaded_model = LogisticRegression::load_safetensors("/tmp/doctest_logistic_model.safetensors").expect("File exists and is valid SafeTensors format");
    /// # std::fs::remove_file("/tmp/doctest_logistic_model.safetensors").ok();
    /// ```
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

        // Construct model with default hyperparameters
        // Note: Hyperparameters are not serialized as they're only needed during training
        Ok(Self {
            coefficients: Some(Vector::from_vec(coef_data)),
            intercept: intercept_data[0],
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-4,
            class_weight: ClassWeight::Uniform,
            l2_penalty: 0.0,
        })
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Distance metric for K-Nearest Neighbors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance: `sqrt(sum((x_i` - `y_i)^2`))
    Euclidean,
    /// Manhattan distance: `sum(|x_i` - `y_i`|)
    Manhattan,
    /// Minkowski distance with parameter p
    Minkowski(f32),
}

/// K-Nearest Neighbors classifier.
///
/// Instance-based learning algorithm that classifies new samples based on
/// the k closest training examples in the feature space.
///
/// # Example
///
/// ```
/// use aprender::classification::{KNearestNeighbors, DistanceMetric};
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(6, 2, vec![
///     0.0, 0.0,  // class 0
///     0.0, 1.0,  // class 0
///     1.0, 0.0,  // class 0
///     5.0, 5.0,  // class 1
///     5.0, 6.0,  // class 1
///     6.0, 5.0,  // class 1
/// ]).expect("6x2 matrix with 12 values");
/// let y = vec![0, 0, 0, 1, 1, 1];
///
/// let mut knn = KNearestNeighbors::new(3);
/// knn.fit(&x, &y).expect("Valid training data with 6 samples");
///
/// let test = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
/// let predictions = knn.predict(&test).expect("Predict should succeed");
/// assert_eq!(predictions[0], 0);  // Closer to class 0
/// ```
#[derive(Debug, Clone)]
pub struct KNearestNeighbors {
    /// Number of neighbors to use
    k: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// Whether to use weighted voting (inverse distance)
    weights: bool,
    /// Training feature matrix (stored during fit)
    x_train: Option<Matrix<f32>>,
    /// Training labels (stored during fit)
    y_train: Option<Vec<usize>>,
}

mod gaussian_nb;
pub use gaussian_nb::*;
mod linear_svm;
pub use linear_svm::*;
mod sets;

#[cfg(test)]
#[path = "tests_logreg_contract.rs"]
mod tests_logreg_contract;
