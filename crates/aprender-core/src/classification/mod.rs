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

/// Gradient descent mode for LogisticRegression.
///
/// Contract: `contracts/apr-stochastic-lr-v1.yaml` — `fit_mode_enum` equation.
///
/// Controls how gradients are computed and applied:
/// - `Batch`: Full-batch gradient descent (default, backward compatible)
/// - `Stochastic`: Online SGD with per-sample updates
/// - `MiniBatch(k)`: Mini-batch SGD with updates every k samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FitMode {
    /// Full-batch gradient descent: average gradient over all samples.
    /// Default, backward compatible.
    Batch,
    /// Stochastic (online) SGD: update after each sample.
    /// Better for imbalanced data — preserves minority class signal.
    Stochastic,
    /// Mini-batch SGD: update after every k samples.
    /// Compromise between Batch (stable) and Stochastic (fast convergence).
    MiniBatch(usize),
}

impl Default for FitMode {
    fn default() -> Self {
        Self::Batch
    }
}

/// Fisher-Yates partner index for position `i` in the epoch-`seed` sample shuffle.
///
/// Contract: `contracts/apr-stochastic-lr-v1.yaml` — `stochastic_convergence`
/// ("shuffled sample order each epoch") and `minibatch_gradient` ("each sample seen
/// exactly once per epoch"). Returning a value in `[0, i]` is what makes the
/// Fisher-Yates pass a permutation.
/// The arithmetic is deliberately `u64`, not `usize`. Refs #2310. Both MMIX
/// constants exceed `u32::MAX`, so as bare `usize` literals they are a hard
/// compile error on 32-bit targets ("literal out of range for `usize`" on
/// `wasm32-unknown-unknown`), and the products overflow `u64` — `seed * MUL` from
/// `seed == 3`, `i * INC` from `i == 13` — which aborts every overflow-checked
/// build on 64-bit too. `wrapping_*` reproduces the 64-bit release-mode result
/// bit-for-bit, so no previously-trained model's epoch order shifts.
#[inline]
fn shuffle_partner(seed: usize, i: usize) -> usize {
    const LCG_MULTIPLIER: u64 = 6_364_136_223_846_793_005;
    const LCG_INCREMENT: u64 = 1_442_695_040_888_963_407;
    let mixed = (seed as u64)
        .wrapping_mul(LCG_MULTIPLIER)
        .wrapping_add((i as u64).wrapping_mul(LCG_INCREMENT));
    // `i + 1` cannot overflow: `i` indexes a live Vec, so `i < usize::MAX`.
    // The remainder is `< i + 1`, hence always representable as `usize`.
    (mixed % (i as u64 + 1)) as usize
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
    /// Gradient descent mode (GH-428: stochastic/mini-batch support).
    fit_mode: FitMode,
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
            fit_mode: FitMode::Batch,
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

    /// Sets the gradient descent mode (GH-428).
    ///
    /// Contract: `contracts/apr-stochastic-lr-v1.yaml`
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::{LogisticRegression, FitMode};
    ///
    /// // Stochastic SGD for imbalanced datasets
    /// let model = LogisticRegression::new()
    ///     .with_fit_mode(FitMode::Stochastic);
    ///
    /// // Mini-batch with 32 samples per update
    /// let model = LogisticRegression::new()
    ///     .with_fit_mode(FitMode::MiniBatch(32));
    /// ```
    #[must_use]
    pub fn with_fit_mode(mut self, mode: FitMode) -> Self {
        self.fit_mode = mode;
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

        // Contract: apr-stochastic-lr-v1.yaml — fit_mode_enum
        match self.fit_mode {
            FitMode::Batch => self.fit_batch(x, y, &sample_weights, n_samples, n_features),
            FitMode::Stochastic => {
                self.fit_stochastic(x, y, &sample_weights, n_samples, n_features)
            }
            FitMode::MiniBatch(batch_size) => {
                let bs = batch_size.max(1).min(n_samples);
                if bs == n_samples {
                    self.fit_batch(x, y, &sample_weights, n_samples, n_features)
                } else {
                    self.fit_minibatch(x, y, &sample_weights, n_samples, n_features, bs)
                }
            }
        }
    }

    /// Full-batch gradient descent (default, backward compatible).
    fn fit_batch(
        &mut self,
        x: &Matrix<f32>,
        y: &[usize],
        sample_weights: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Result<()> {
        for _ in 0..self.max_iter {
            let probas = self.predict_proba(x);
            let mut coef_grad = vec![0.0; n_features];
            let mut intercept_grad = 0.0;

            for i in 0..n_samples {
                let error = sample_weights[i] * (probas[i] - y[i] as f32);
                intercept_grad += error;
                for (j, grad) in coef_grad.iter_mut().enumerate() {
                    *grad += error * x.get(i, j);
                }
            }

            let n = n_samples as f32;
            intercept_grad /= n;
            for grad in &mut coef_grad {
                *grad /= n;
            }

            self.intercept -= self.learning_rate * intercept_grad;
            if let Some(ref mut coef) = self.coefficients {
                for j in 0..n_features {
                    coef[j] -= self.learning_rate * (coef_grad[j] + self.l2_penalty * coef[j]);
                }
            }

            if intercept_grad.abs() < self.tol && coef_grad.iter().all(|&g| g.abs() < self.tol) {
                break;
            }
        }
        Ok(())
    }

    /// Stochastic (online) SGD: update after each sample.
    /// Contract: apr-stochastic-lr-v1.yaml — stochastic_convergence
    fn fit_stochastic(
        &mut self,
        x: &Matrix<f32>,
        y: &[usize],
        sample_weights: &[f32],
        n_samples: usize,
        n_features: usize,
    ) -> Result<()> {
        // Simple deterministic shuffle via index permutation
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for epoch in 0..self.max_iter {
            // Shuffle indices each epoch (deterministic via epoch seed)
            // Contract: stochastic_convergence — "shuffled sample order each epoch"
            let seed = epoch;
            for i in (1..n_samples).rev() {
                let j = shuffle_partner(seed, i);
                indices.swap(i, j);
            }

            let mut max_grad = 0.0_f32;

            for &idx in &indices {
                // Per-sample gradient
                let prob = self.predict_proba_single(x, idx);
                let error = sample_weights[idx] * (prob - y[idx] as f32);

                // Update intercept
                self.intercept -= self.learning_rate * error;

                // Update coefficients
                if let Some(ref mut coef) = self.coefficients {
                    for j in 0..n_features {
                        let grad = error * x.get(idx, j) + self.l2_penalty * coef[j];
                        coef[j] -= self.learning_rate * grad;
                        max_grad = max_grad.max(grad.abs());
                    }
                }
            }

            if max_grad < self.tol {
                break;
            }
        }
        Ok(())
    }

    /// Mini-batch SGD: update after every batch_size samples.
    /// Contract: apr-stochastic-lr-v1.yaml — minibatch_gradient
    fn fit_minibatch(
        &mut self,
        x: &Matrix<f32>,
        y: &[usize],
        sample_weights: &[f32],
        n_samples: usize,
        n_features: usize,
        batch_size: usize,
    ) -> Result<()> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        for epoch in 0..self.max_iter {
            // Shuffle
            let seed = epoch;
            for i in (1..n_samples).rev() {
                let j = shuffle_partner(seed, i);
                indices.swap(i, j);
            }

            let mut max_grad = 0.0_f32;

            for batch in indices.chunks(batch_size) {
                let bs = batch.len() as f32;
                let mut coef_grad = vec![0.0; n_features];
                let mut intercept_grad = 0.0;

                for &idx in batch {
                    let prob = self.predict_proba_single(x, idx);
                    let error = sample_weights[idx] * (prob - y[idx] as f32);
                    intercept_grad += error;
                    for (j, grad) in coef_grad.iter_mut().enumerate() {
                        *grad += error * x.get(idx, j);
                    }
                }

                // Average over batch
                intercept_grad /= bs;
                for grad in &mut coef_grad {
                    *grad /= bs;
                }

                self.intercept -= self.learning_rate * intercept_grad;
                if let Some(ref mut coef) = self.coefficients {
                    for j in 0..n_features {
                        let grad = coef_grad[j] + self.l2_penalty * coef[j];
                        coef[j] -= self.learning_rate * grad;
                        max_grad = max_grad.max(grad.abs());
                    }
                }
            }

            if max_grad < self.tol {
                break;
            }
        }
        Ok(())
    }

    /// Predict probability for a single sample (used by stochastic/mini-batch).
    fn predict_proba_single(&self, x: &Matrix<f32>, row: usize) -> f32 {
        let coef = self
            .coefficients
            .as_ref()
            .expect("model must be initialized");
        let n_features = coef.len();
        let mut z = self.intercept;
        for col in 0..n_features {
            z += coef[col] * x.get(row, col);
        }
        Self::sigmoid(z)
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
            fit_mode: FitMode::Batch,
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

mod bernoulli_nb;
mod complement_nb;
mod discriminant_analysis;
mod gaussian_nb;
mod multinomial_nb;
pub use bernoulli_nb::BernoulliNB;
pub use complement_nb::ComplementNB;
pub use discriminant_analysis::{LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis};
pub use gaussian_nb::*;
pub use multinomial_nb::MultinomialNB;
mod linear_svm;
pub use linear_svm::*;
mod svc_rbf;
pub use svc_rbf::{Kernel, MultiClassSVC, SVCRbf};
pub mod multinomial;
pub use multinomial::*;
mod sets;
#[cfg(test)]
mod svc_rbf_sklearn_fixture;

#[cfg(test)]
#[path = "tests_logreg_contract.rs"]
mod tests_logreg_contract;

#[cfg(test)]
#[path = "tests_multinomial_contract.rs"]
mod tests_multinomial_contract;
// #2310: the SGD epoch shuffle must compile on 32-bit targets and must not
// overflow on 64-bit. Falsifiers for `shuffle_partner` and both SGD fit modes.
#[cfg(test)]
#[path = "tests_sgd_portable_shuffle.rs"]
mod tests_sgd_portable_shuffle;

// Estimator impl so LogisticRegression works with generic cross_validate /
// grid_search (Pillar 1). Labels round-trip through f32; inherent API unchanged.
impl crate::traits::Estimator for LogisticRegression {
    fn fit(&mut self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        LogisticRegression::fit(self, x, &labels)
    }
    fn predict(&self, x: &Matrix<f32>) -> crate::primitives::Vector<f32> {
        let labels: Vec<usize> = LogisticRegression::predict(self, x);
        crate::primitives::Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }
    fn score(&self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> f32 {
        let preds: Vec<usize> = LogisticRegression::predict(self, x);
        let n = y.len();
        if n == 0 {
            return 0.0;
        }
        let correct = preds
            .iter()
            .zip(y.as_slice())
            .filter(|(&p, &t)| p == t.round() as usize)
            .count();
        correct as f32 / n as f32
    }
}

// Estimator impl so KNearestNeighbors works with generic cross_validate /
// grid_search (Pillar 1). Labels round-trip through f32; inherent &[usize] API
// unchanged. KNN's inherent predict returns Result; on the (post-fit) error path
// we fall back to zeros of the right length so scoring stays well-defined.
impl crate::traits::Estimator for KNearestNeighbors {
    fn fit(&mut self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        KNearestNeighbors::fit(self, x, &labels)
    }
    fn predict(&self, x: &Matrix<f32>) -> crate::primitives::Vector<f32> {
        let labels: Vec<usize> =
            KNearestNeighbors::predict(self, x).unwrap_or_else(|_| vec![0; x.shape().0]);
        crate::primitives::Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }
    fn score(&self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> f32 {
        let preds: Vec<usize> =
            KNearestNeighbors::predict(self, x).unwrap_or_else(|_| vec![0; x.shape().0]);
        let n = y.len();
        if n == 0 {
            return 0.0;
        }
        let correct = preds
            .iter()
            .zip(y.as_slice())
            .filter(|(&p, &t)| p == t.round() as usize)
            .count();
        correct as f32 / n as f32
    }
}

// Estimator impl so GaussianNB works with generic cross_validate / grid_search
// (Pillar 1). Labels round-trip through f32; inherent &[usize] API unchanged.
// predict returns Result; the post-fit error path falls back to zeros.
impl crate::traits::Estimator for GaussianNB {
    fn fit(&mut self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> Result<()> {
        let labels: Vec<usize> = y.as_slice().iter().map(|&v| v.round() as usize).collect();
        GaussianNB::fit(self, x, &labels)
    }
    fn predict(&self, x: &Matrix<f32>) -> crate::primitives::Vector<f32> {
        let labels: Vec<usize> =
            GaussianNB::predict(self, x).unwrap_or_else(|_| vec![0; x.shape().0]);
        crate::primitives::Vector::from_vec(labels.into_iter().map(|l| l as f32).collect())
    }
    fn score(&self, x: &Matrix<f32>, y: &crate::primitives::Vector<f32>) -> f32 {
        let preds: Vec<usize> =
            GaussianNB::predict(self, x).unwrap_or_else(|_| vec![0; x.shape().0]);
        let n = y.len();
        if n == 0 {
            return 0.0;
        }
        let correct = preds
            .iter()
            .zip(y.as_slice())
            .filter(|(&p, &t)| p == t.round() as usize)
            .count();
        correct as f32 / n as f32
    }
}
