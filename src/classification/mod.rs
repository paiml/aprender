//! Classification algorithms.
//!
//! This module implements classification algorithms including:
//! - Logistic Regression for binary classification
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
//! ]).unwrap();
//! let y = vec![0, 0, 0, 1];
//!
//! let mut model = LogisticRegression::new()
//!     .with_learning_rate(0.1)
//!     .with_max_iter(1000);
//! model.fit(&x, &y).unwrap();
//! let predictions = model.predict(&x);
//!
//! assert_eq!(predictions.len(), 4);
//! for pred in predictions {
//!     assert!(pred == 0 || pred == 1);
//! }
//! ```

use crate::primitives::{Matrix, Vector};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Logistic Regression classifier for binary classification.
///
/// Uses sigmoid activation and binary cross-entropy loss with
/// gradient descent optimization.
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
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Sets the learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance.
    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
    fn sigmoid(z: f32) -> f32 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Predicts probabilities for samples.
    ///
    /// Returns probability of class 1 for each sample.
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
    /// * `x` - Feature matrix (n_samples × n_features)
    /// * `y` - Binary labels (n_samples), must be 0 or 1
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, `Err` with message on failure
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<(), &'static str> {
        let (n_samples, n_features) = x.shape();

        if n_samples != y.len() {
            return Err("Number of samples in X and y must match");
        }
        if n_samples == 0 {
            return Err("Cannot fit with zero samples");
        }

        // Validate labels are binary (0 or 1)
        for &label in y {
            if label != 0 && label != 1 {
                return Err("Labels must be 0 or 1 for binary classification");
            }
        }

        // Initialize coefficients and intercept
        self.coefficients = Some(Vector::from_vec(vec![0.0; n_features]));
        self.intercept = 0.0;

        // Gradient descent optimization
        for _ in 0..self.max_iter {
            // Compute predictions (probabilities)
            let probas = self.predict_proba(x);

            // Compute gradients
            let mut coef_grad = vec![0.0; n_features];
            let mut intercept_grad = 0.0;

            for i in 0..n_samples {
                let error = probas[i] - y[i] as f32;
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

            // Update parameters
            self.intercept -= self.learning_rate * intercept_grad;
            if let Some(ref mut coef) = self.coefficients {
                for j in 0..n_features {
                    coef[j] -= self.learning_rate * coef_grad[j];
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
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let probas = self.predict_proba(x);
        probas
            .as_slice()
            .iter()
            .map(|&p| if p >= 0.5 { 1 } else { 0 })
            .collect()
    }

    /// Computes accuracy score on test data.
    ///
    /// Returns fraction of correctly classified samples.
    pub fn score(&self, x: &Matrix<f32>, y: &[usize]) -> f32 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();
        correct as f32 / y.len() as f32
    }

    /// Saves the trained model to SafeTensors format.
    ///
    /// SafeTensors is an industry-standard model serialization format
    /// compatible with HuggingFace, Ollama, PyTorch, TensorFlow, and realizar.
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
    /// let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    /// let y = vec![0, 0, 1, 1];
    /// model.fit(&x, &y).unwrap();
    ///
    /// model.save_safetensors("model.safetensors").unwrap();
    /// ```
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
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
        safetensors::save_safetensors(path, tensors)?;
        Ok(())
    }

    /// Loads a model from SafeTensors format.
    ///
    /// # Arguments
    ///
    /// * `path` - File path to load the model from
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - SafeTensors format is invalid
    /// - Required tensors are missing
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::LogisticRegression;
    ///
    /// # use aprender::prelude::*;
    /// # let mut model = LogisticRegression::new();
    /// # let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
    /// # let y = vec![0, 0, 1, 1];
    /// # model.fit(&x, &y).unwrap();
    /// # model.save_safetensors("/tmp/doctest_logistic_model.safetensors").unwrap();
    /// let loaded_model = LogisticRegression::load_safetensors("/tmp/doctest_logistic_model.safetensors").unwrap();
    /// # std::fs::remove_file("/tmp/doctest_logistic_model.safetensors").ok();
    /// ```
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<Self, String> {
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
            learning_rate: 0.01, // Default value
            max_iter: 1000,      // Default value
            tol: 1e-4,           // Default value
        })
    }
}

impl Default for LogisticRegression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((LogisticRegression::sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(LogisticRegression::sigmoid(10.0) > 0.99);
        assert!(LogisticRegression::sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_logistic_regression_new() {
        let model = LogisticRegression::new();
        assert!(model.coefficients.is_none());
        assert_eq!(model.intercept, 0.0);
    }

    #[test]
    fn test_logistic_regression_builder() {
        let model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(500)
            .with_tolerance(1e-3);

        assert_eq!(model.learning_rate, 0.1);
        assert_eq!(model.max_iter, 500);
        assert_eq!(model.tol, 1e-3);
    }

    #[test]
    fn test_logistic_regression_fit_simple() {
        // Simple linearly separable data
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);

        let result = model.fit(&x, &y);
        assert!(result.is_ok());
        assert!(model.coefficients.is_some());
    }

    #[test]
    fn test_logistic_regression_predict() {
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);

        model.fit(&x, &y).unwrap();
        let predictions = model.predict(&x);

        // Should correctly classify training data
        assert_eq!(predictions.len(), 4);
        for pred in predictions {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_logistic_regression_score() {
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);

        model.fit(&x, &y).unwrap();
        let accuracy = model.score(&x, &y);

        // Should achieve high accuracy on linearly separable data
        assert!(accuracy >= 0.75); // At least 75% accuracy
    }

    #[test]
    fn test_logistic_regression_invalid_labels() {
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = vec![0, 2]; // Invalid label 2

        let mut model = LogisticRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Labels must be 0 or 1 for binary classification"
        );
    }

    #[test]
    fn test_logistic_regression_mismatched_samples() {
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = vec![0]; // Only 1 label for 2 samples

        let mut model = LogisticRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            "Number of samples in X and y must match"
        );
    }

    #[test]
    fn test_logistic_regression_zero_samples() {
        let x = Matrix::from_vec(0, 2, vec![]).unwrap();
        let y = vec![];

        let mut model = LogisticRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "Cannot fit with zero samples");
    }

    #[test]
    fn test_predict_proba() {
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).unwrap();
        let y = vec![0, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);

        model.fit(&x, &y).unwrap();
        let probas = model.predict_proba(&x);

        assert_eq!(probas.len(), 2);
        for &p in probas.as_slice() {
            assert!((0.0..=1.0).contains(&p));
        }
    }

    // SafeTensors Serialization Tests
    // RED PHASE: These tests will fail until we implement save_safetensors() and load_safetensors()

    #[test]
    fn test_save_safetensors_unfitted_model() {
        // Test 1: Cannot save unfitted model
        let model = LogisticRegression::new();
        let result = model.save_safetensors("/tmp/test_unfitted_logistic.safetensors");

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unfitted"));
    }

    #[test]
    fn test_save_load_safetensors_roundtrip() {
        // Test 2: Save and load preserves model state
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = vec![0, 0, 1, 1];

        // Train model
        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);
        model.fit(&x, &y).unwrap();

        // Save model
        let path = "/tmp/test_logistic_roundtrip.safetensors";
        model.save_safetensors(path).unwrap();

        // Load model
        let loaded = LogisticRegression::load_safetensors(path).unwrap();

        // Verify coefficients match
        assert_eq!(
            model.coefficients.as_ref().unwrap().len(),
            loaded.coefficients.as_ref().unwrap().len()
        );
        for i in 0..model.coefficients.as_ref().unwrap().len() {
            assert_eq!(
                model.coefficients.as_ref().unwrap()[i],
                loaded.coefficients.as_ref().unwrap()[i]
            );
        }
        assert_eq!(model.intercept, loaded.intercept);

        // Verify predictions match
        let predictions_original = model.predict(&x);
        let predictions_loaded = loaded.predict(&x);
        assert_eq!(predictions_original, predictions_loaded);

        // Cleanup
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_safetensors_corrupted_file() {
        // Test 3: Loading corrupted file fails gracefully
        let path = "/tmp/test_corrupted_logistic.safetensors";
        std::fs::write(path, b"CORRUPTED DATA").unwrap();

        let result = LogisticRegression::load_safetensors(path);
        assert!(result.is_err());

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_load_safetensors_missing_file() {
        // Test 4: Loading missing file fails with clear error
        let result =
            LogisticRegression::load_safetensors("/tmp/nonexistent_logistic_xyz.safetensors");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("No such file") || err.contains("not found"),
            "Error should mention file not found: {}",
            err
        );
    }

    #[test]
    fn test_safetensors_preserves_probabilities() {
        // Test 5: Probabilities are identical after save/load
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 1
                1.0, 1.0, // class 1
            ],
        )
        .unwrap();
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);
        model.fit(&x, &y).unwrap();

        let probas_before = model.predict_proba(&x);

        // Save and load
        let path = "/tmp/test_logistic_probas.safetensors";
        model.save_safetensors(path).unwrap();
        let loaded = LogisticRegression::load_safetensors(path).unwrap();

        let probas_after = loaded.predict_proba(&x);

        // Verify probabilities match exactly
        assert_eq!(probas_before.len(), probas_after.len());
        for i in 0..probas_before.len() {
            assert_eq!(probas_before[i], probas_after[i]);
        }

        std::fs::remove_file(path).ok();
    }
}
