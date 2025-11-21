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

/// Distance metric for K-Nearest Neighbors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance: sqrt(sum((x_i - y_i)^2))
    Euclidean,
    /// Manhattan distance: sum(|x_i - y_i|)
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
/// let predictions = knn.predict(&test);
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

impl KNearestNeighbors {
    /// Creates a new K-Nearest Neighbors classifier.
    ///
    /// # Arguments
    ///
    /// * `k` - Number of neighbors to use for voting
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::KNearestNeighbors;
    ///
    /// let knn = KNearestNeighbors::new(5);
    /// ```
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            metric: DistanceMetric::Euclidean,
            weights: false,
            x_train: None,
            y_train: None,
        }
    }

    /// Sets the distance metric.
    #[must_use]
    pub fn with_metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Enables weighted voting (inverse distance weighting).
    #[must_use]
    pub fn with_weights(mut self, weights: bool) -> Self {
        self.weights = weights;
        self
    }

    /// Fits the model by storing the training data.
    ///
    /// kNN is a lazy learner - it simply stores the training data
    /// and defers computation until prediction time.
    ///
    /// # Errors
    ///
    /// Returns error if data dimensions are invalid.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, _n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        if y.len() != n_samples {
            return Err("Number of samples in X and y must match".into());
        }

        if self.k > n_samples {
            return Err("k cannot be larger than number of training samples".into());
        }

        // Store training data
        self.x_train = Some(x.clone());
        self.y_train = Some(y.to_vec());

        Ok(())
    }

    /// Predicts class labels for samples.
    ///
    /// For each test sample, finds the k nearest training samples
    /// and returns the majority class.
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimensions mismatch.
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let x_train = self.x_train.as_ref().ok_or("Model not fitted")?;
        let y_train = self.y_train.as_ref().ok_or("Model not fitted")?;

        let (n_samples, n_features) = x.shape();
        let (_n_train, n_train_features) = x_train.shape();

        if n_features != n_train_features {
            return Err("Feature dimension mismatch".into());
        }

        let mut predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Compute distances to all training samples
            let mut distances: Vec<(f32, usize)> = Vec::with_capacity(y_train.len());

            for (j, &label) in y_train.iter().enumerate() {
                let dist = self.compute_distance(x, i, x_train, j, n_features);
                distances.push((dist, label));
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Distance values are valid f32 (not NaN)")
            });
            let k_nearest = &distances[..self.k];

            // Vote for class
            let predicted_class = if self.weights {
                self.weighted_vote(k_nearest)
            } else {
                self.majority_vote(k_nearest)
            };

            predictions.push(predicted_class);
        }

        Ok(predictions)
    }

    /// Returns probability estimates for each class.
    ///
    /// Probabilities are computed as the proportion of neighbors belonging
    /// to each class (optionally weighted by inverse distance).
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimensions mismatch.
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        let x_train = self.x_train.as_ref().ok_or("Model not fitted")?;
        let y_train = self.y_train.as_ref().ok_or("Model not fitted")?;

        let (n_samples, n_features) = x.shape();
        let (_n_train, n_train_features) = x_train.shape();

        if n_features != n_train_features {
            return Err("Feature dimension mismatch".into());
        }

        // Find number of classes
        let n_classes = *y_train
            .iter()
            .max()
            .expect("Training labels are non-empty (verified in fit())")
            + 1;

        let mut probabilities = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Compute distances to all training samples
            let mut distances: Vec<(f32, usize)> = Vec::with_capacity(y_train.len());

            for (j, &label) in y_train.iter().enumerate() {
                let dist = self.compute_distance(x, i, x_train, j, n_features);
                distances.push((dist, label));
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .expect("Distance values are valid f32 (not NaN)")
            });
            let k_nearest = &distances[..self.k];

            // Compute class probabilities
            let mut class_counts = vec![0.0; n_classes];

            if self.weights {
                // Weighted by inverse distance
                for (dist, label) in k_nearest {
                    let weight = if *dist < 1e-10 { 1.0 } else { 1.0 / dist };
                    class_counts[*label] += weight;
                }
            } else {
                // Uniform weights
                for (_dist, label) in k_nearest {
                    class_counts[*label] += 1.0;
                }
            }

            // Normalize to probabilities
            let total: f32 = class_counts.iter().sum();
            for count in &mut class_counts {
                *count /= total;
            }

            probabilities.push(class_counts);
        }

        Ok(probabilities)
    }

    /// Computes distance between two samples.
    fn compute_distance(
        &self,
        x1: &Matrix<f32>,
        i1: usize,
        x2: &Matrix<f32>,
        i2: usize,
        n_features: usize,
    ) -> f32 {
        match self.metric {
            DistanceMetric::Euclidean => {
                let mut sum = 0.0;
                for k in 0..n_features {
                    let diff = x1.get(i1, k) - x2.get(i2, k);
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            DistanceMetric::Manhattan => {
                let mut sum = 0.0;
                for k in 0..n_features {
                    sum += (x1.get(i1, k) - x2.get(i2, k)).abs();
                }
                sum
            }
            DistanceMetric::Minkowski(p) => {
                let mut sum = 0.0;
                for k in 0..n_features {
                    sum += (x1.get(i1, k) - x2.get(i2, k)).abs().powf(p);
                }
                sum.powf(1.0 / p)
            }
        }
    }

    /// Performs majority voting among k nearest neighbors.
    fn majority_vote(&self, neighbors: &[(f32, usize)]) -> usize {
        let mut class_counts = std::collections::HashMap::new();

        for (_dist, label) in neighbors {
            *class_counts.entry(*label).or_insert(0) += 1;
        }

        *class_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(label, _)| label)
            .expect("Neighbors slice is non-empty (k >= 1)")
    }

    /// Performs weighted voting (inverse distance weighting).
    fn weighted_vote(&self, neighbors: &[(f32, usize)]) -> usize {
        let mut class_weights = std::collections::HashMap::new();

        for (dist, label) in neighbors {
            let weight = if *dist < 1e-10 { 1.0 } else { 1.0 / dist };
            *class_weights.entry(*label).or_insert(0.0) += weight;
        }

        *class_weights
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Weights are valid f32 (not NaN)"))
            .map(|(label, _)| label)
            .expect("Neighbors slice is non-empty (k >= 1)")
    }
}

/// Gaussian Naive Bayes classifier.
///
/// Assumes features follow a Gaussian (normal) distribution within each class.
/// Uses Bayes' theorem with independence assumption between features.
///
/// # Example
///
/// ```
/// use aprender::classification::GaussianNB;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(4, 2, vec![
///     0.0, 0.0,
///     0.0, 1.0,
///     1.0, 0.0,
///     1.0, 1.0,
/// ]).expect("4x2 matrix with 8 values");
/// let y = vec![0, 0, 1, 1];
///
/// let mut model = GaussianNB::new();
/// model.fit(&x, &y).expect("Valid training data");
/// let predictions = model.predict(&x).expect("Model is fitted");
/// ```
#[derive(Debug, Clone)]
pub struct GaussianNB {
    /// Class prior probabilities P(y=c)
    class_priors: Option<Vec<f32>>,
    /// Feature means per class: means[class][feature]
    means: Option<Vec<Vec<f32>>>,
    /// Feature variances per class: variances[class][feature]
    variances: Option<Vec<Vec<f32>>>,
    /// Class labels
    classes: Option<Vec<usize>>,
    /// Laplace smoothing parameter (var_smoothing)
    var_smoothing: f32,
}

impl GaussianNB {
    /// Creates a new Gaussian Naive Bayes classifier.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::GaussianNB;
    ///
    /// let model = GaussianNB::new();
    /// ```
    pub fn new() -> Self {
        Self {
            class_priors: None,
            means: None,
            variances: None,
            classes: None,
            var_smoothing: 1e-9,
        }
    }

    /// Sets the variance smoothing parameter.
    ///
    /// Adds this value to variances to avoid numerical instability.
    ///
    /// # Example
    ///
    /// ```
    /// use aprender::classification::GaussianNB;
    ///
    /// let model = GaussianNB::new().with_var_smoothing(1e-8);
    /// ```
    pub fn with_var_smoothing(mut self, var_smoothing: f32) -> Self {
        self.var_smoothing = var_smoothing;
        self
    }

    /// Trains the Gaussian Naive Bayes classifier.
    ///
    /// Computes class priors, feature means, and variances for each class.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Sample count mismatch between X and y
    /// - Empty data
    /// - Less than 2 classes
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        if n_samples == 0 {
            return Err("Cannot fit with empty data".into());
        }

        if y.len() != n_samples {
            return Err("Number of samples in X and y must match".into());
        }

        // Find unique classes
        let mut classes: Vec<usize> = y.to_vec();
        classes.sort_unstable();
        classes.dedup();

        if classes.len() < 2 {
            return Err("Need at least 2 classes".into());
        }

        let n_classes = classes.len();

        // Initialize storage
        let mut class_priors = vec![0.0; n_classes];
        let mut means = vec![vec![0.0; n_features]; n_classes];
        let mut variances = vec![vec![0.0; n_features]; n_classes];

        // Compute class priors and feature statistics
        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Find samples belonging to this class
            let class_samples: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            let n_class_samples = class_samples.len() as f32;
            class_priors[class_idx] = n_class_samples / n_samples as f32;

            // Compute mean for each feature
            for (feature_idx, mean_val) in means[class_idx].iter_mut().enumerate() {
                let sum: f32 = class_samples
                    .iter()
                    .map(|&sample_idx| x.get(sample_idx, feature_idx))
                    .sum();
                *mean_val = sum / n_class_samples;
            }

            // Compute variance for each feature
            for (feature_idx, variance_val) in variances[class_idx].iter_mut().enumerate() {
                let mean = means[class_idx][feature_idx];
                let sum_sq_diff: f32 = class_samples
                    .iter()
                    .map(|&sample_idx| {
                        let diff = x.get(sample_idx, feature_idx) - mean;
                        diff * diff
                    })
                    .sum();
                *variance_val = sum_sq_diff / n_class_samples + self.var_smoothing;
            }
        }

        self.class_priors = Some(class_priors);
        self.means = Some(means);
        self.variances = Some(variances);
        self.classes = Some(classes);

        Ok(())
    }

    /// Predicts class labels for samples.
    ///
    /// Returns the class with highest posterior probability for each sample.
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimension mismatch.
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let probabilities = self.predict_proba(x)?;
        let classes = self.classes.as_ref().ok_or("Model not fitted")?;

        let predictions: Vec<usize> = probabilities
            .iter()
            .map(|probs| {
                let max_idx = probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.partial_cmp(b)
                            .expect("Probabilities are valid f32 (not NaN)")
                    })
                    .map(|(idx, _)| idx)
                    .expect("Probabilities vector is non-empty (n_classes >= 2)");
                classes[max_idx]
            })
            .collect();

        Ok(predictions)
    }

    /// Returns probability estimates for each class.
    ///
    /// Uses Bayes' theorem with Gaussian likelihood:
    /// P(y=c|X) ∝ P(y=c) * ∏ P(x_i|y=c)
    ///
    /// # Errors
    ///
    /// Returns error if model is not fitted or dimension mismatch.
    pub fn predict_proba(&self, x: &Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        let means = self.means.as_ref().ok_or("Model not fitted")?;
        let variances = self.variances.as_ref().ok_or("Model not fitted")?;
        let class_priors = self.class_priors.as_ref().ok_or("Model not fitted")?;

        let (n_samples, n_features) = x.shape();
        let n_classes = means.len();

        if n_features != means[0].len() {
            return Err("Feature dimension mismatch".into());
        }

        let mut probabilities = Vec::with_capacity(n_samples);

        for sample_idx in 0..n_samples {
            let mut log_probs = vec![0.0; n_classes];

            // Compute log posterior for each class
            for class_idx in 0..n_classes {
                // Start with log prior
                log_probs[class_idx] = class_priors[class_idx].ln();

                // Add log likelihood for each feature (Gaussian PDF)
                for feature_idx in 0..n_features {
                    let x_val = x.get(sample_idx, feature_idx);
                    let mean = means[class_idx][feature_idx];
                    let variance = variances[class_idx][feature_idx];

                    // Log of Gaussian PDF: -0.5 * log(2π*σ²) - (x-μ)² / (2σ²)
                    let diff = x_val - mean;
                    let log_likelihood = -0.5 * (2.0 * std::f32::consts::PI * variance).ln()
                        - (diff * diff) / (2.0 * variance);

                    log_probs[class_idx] += log_likelihood;
                }
            }

            // Convert log probabilities to probabilities using log-sum-exp trick
            let max_log_prob = log_probs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp_probs: Vec<f32> = log_probs
                .iter()
                .map(|&log_p| (log_p - max_log_prob).exp())
                .collect();
            let sum: f32 = exp_probs.iter().sum();
            let normalized: Vec<f32> = exp_probs.iter().map(|p| p / sum).collect();

            probabilities.push(normalized);
        }

        Ok(probabilities)
    }
}

impl Default for GaussianNB {
    fn default() -> Self {
        Self::new()
    }
}

/// Linear Support Vector Machine (SVM) classifier.
///
/// Implements binary classification using hinge loss and subgradient descent.
/// For multi-class problems, use One-vs-Rest strategy.
///
/// # Algorithm
///
/// Minimizes the objective:
/// ```text
/// min  λ||w||² + (1/n) Σᵢ max(0, 1 - yᵢ(w·xᵢ + b))
/// ```
///
/// Where λ = 1/(2nC) controls regularization strength.
///
/// # Example
///
/// ```ignore
/// use aprender::classification::LinearSVM;
/// use aprender::primitives::Matrix;
///
/// let x = Matrix::from_vec(4, 2, vec![
///     0.0, 0.0,
///     0.0, 1.0,
///     1.0, 0.0,
///     1.0, 1.0,
/// ])?;
/// let y = vec![0, 0, 1, 1];
///
/// let mut svm = LinearSVM::new();
/// svm.fit(&x, &y)?;
/// let predictions = svm.predict(&x)?;
/// ```
#[derive(Debug, Clone)]
pub struct LinearSVM {
    /// Weights for each feature
    weights: Option<Vec<f32>>,
    /// Bias term
    bias: f32,
    /// Regularization parameter (default: 1.0)
    /// Larger C means less regularization
    c: f32,
    /// Learning rate for subgradient descent (default: 0.01)
    learning_rate: f32,
    /// Maximum iterations (default: 1000)
    max_iter: usize,
    /// Convergence tolerance (default: 1e-4)
    tol: f32,
}

impl LinearSVM {
    /// Creates a new Linear SVM with default parameters.
    ///
    /// # Default Parameters
    ///
    /// - C: 1.0 (moderate regularization)
    /// - learning_rate: 0.01
    /// - max_iter: 1000
    /// - tol: 1e-4
    pub fn new() -> Self {
        Self {
            weights: None,
            bias: 0.0,
            c: 1.0,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Sets the regularization parameter C.
    ///
    /// Larger C means less regularization (fit data more closely).
    /// Smaller C means more regularization (simpler model).
    pub fn with_c(mut self, c: f32) -> Self {
        self.c = c;
        self
    }

    /// Sets the learning rate for subgradient descent.
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
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

    /// Trains the Linear SVM on the given data.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (n_samples × n_features)
    /// - `y`: Binary labels (0 or 1)
    ///
    /// # Returns
    ///
    /// Ok(()) on success, Err with message on failure.
    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<()> {
        if x.n_rows() != y.len() {
            return Err("x and y must have the same number of samples".into());
        }

        if x.n_rows() == 0 {
            return Err("Cannot fit with 0 samples".into());
        }

        // Convert labels to {-1, +1}
        let y_signed: Vec<f32> = y
            .iter()
            .map(|&label| if label == 0 { -1.0 } else { 1.0 })
            .collect();

        let n_samples = x.n_rows();
        let n_features = x.n_cols();

        // Initialize weights and bias
        let mut w = vec![0.0; n_features];
        let mut b = 0.0;

        let lambda = 1.0 / (2.0 * n_samples as f32 * self.c);

        // Subgradient descent with learning rate decay
        for epoch in 0..self.max_iter {
            let eta = self.learning_rate / (1.0 + epoch as f32 * 0.01);
            let prev_w = w.clone();
            let prev_b = b;

            // Iterate over all samples (batch update)
            for (i, &y_i) in y_signed.iter().enumerate() {
                // Compute decision value: w·x + b
                let mut decision = b;
                for (j, &w_j) in w.iter().enumerate() {
                    decision += w_j * x.get(i, j);
                }

                // Compute margin: y * (w·x + b)
                let margin = y_i * decision;

                // Subgradient update
                if margin < 1.0 {
                    // Misclassified or within margin: update with hinge loss gradient
                    for (j, w_j) in w.iter_mut().enumerate() {
                        let gradient = 2.0 * lambda * *w_j - y_i * x.get(i, j);
                        *w_j -= eta * gradient;
                    }
                    b += eta * y_i;
                } else {
                    // Correctly classified outside margin: only regularization gradient
                    for w_j in &mut w {
                        let gradient = 2.0 * lambda * *w_j;
                        *w_j -= eta * gradient;
                    }
                }
            }

            // Check convergence (weight change between iterations)
            let mut weight_change = 0.0;
            for j in 0..n_features {
                weight_change += (w[j] - prev_w[j]).powi(2);
            }
            weight_change += (b - prev_b).powi(2);
            weight_change = weight_change.sqrt();

            if weight_change < self.tol {
                break;
            }
        }

        self.weights = Some(w);
        self.bias = b;

        Ok(())
    }

    /// Computes the decision function for the given samples.
    ///
    /// Returns w·x + b for each sample. Positive values indicate class 1,
    /// negative values indicate class 0.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (n_samples × n_features)
    ///
    /// # Returns
    ///
    /// Vector of decision values, one per sample.
    pub fn decision_function(&self, x: &Matrix<f32>) -> Result<Vec<f32>> {
        let weights = self.weights.as_ref().ok_or("Model not trained yet")?;

        if x.n_cols() != weights.len() {
            return Err("Feature dimension mismatch".into());
        }

        let mut decisions = Vec::with_capacity(x.n_rows());

        for i in 0..x.n_rows() {
            let mut decision = self.bias;
            for (j, &w_j) in weights.iter().enumerate() {
                decision += w_j * x.get(i, j);
            }
            decisions.push(decision);
        }

        Ok(decisions)
    }

    /// Predicts class labels for the given samples.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (n_samples × n_features)
    ///
    /// # Returns
    ///
    /// Vector of predicted labels (0 or 1).
    pub fn predict(&self, x: &Matrix<f32>) -> Result<Vec<usize>> {
        let decisions = self.decision_function(x)?;

        Ok(decisions
            .iter()
            .map(|&d| if d >= 0.0 { 1 } else { 0 })
            .collect())
    }
}

impl Default for LinearSVM {
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
        .expect("4x2 matrix with 8 values");
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
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);

        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");
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
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);

        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");
        let accuracy = model.score(&x, &y);

        // Should achieve high accuracy on linearly separable data
        assert!(accuracy >= 0.75); // At least 75% accuracy
    }

    #[test]
    fn test_logistic_regression_invalid_labels() {
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");
        let y = vec![0, 2]; // Invalid label 2

        let mut model = LogisticRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with invalid label value"),
            "Labels must be 0 or 1 for binary classification"
        );
    }

    #[test]
    fn test_logistic_regression_mismatched_samples() {
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");
        let y = vec![0]; // Only 1 label for 2 samples

        let mut model = LogisticRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with mismatched sample counts"),
            "Number of samples in X and y must match"
        );
    }

    #[test]
    fn test_logistic_regression_zero_samples() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
        let y = vec![];

        let mut model = LogisticRegression::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with zero samples"),
            "Cannot fit with zero samples"
        );
    }

    #[test]
    fn test_predict_proba() {
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");
        let y = vec![0, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);

        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");
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
        assert!(result
            .expect_err("Should fail when saving unfitted model")
            .contains("unfitted"));
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
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        // Train model
        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Save model
        let path = "/tmp/test_logistic_roundtrip.safetensors";
        model
            .save_safetensors(path)
            .expect("Should save fitted model to valid path");

        // Load model
        let loaded =
            LogisticRegression::load_safetensors(path).expect("Should load valid SafeTensors file");

        // Verify coefficients match
        assert_eq!(
            model
                .coefficients
                .as_ref()
                .expect("Model is fitted and has coefficients")
                .len(),
            loaded
                .coefficients
                .as_ref()
                .expect("Loaded model has coefficients")
                .len()
        );
        for i in 0..model
            .coefficients
            .as_ref()
            .expect("Model has coefficients")
            .len()
        {
            assert_eq!(
                model.coefficients.as_ref().expect("Model has coefficients")[i],
                loaded
                    .coefficients
                    .as_ref()
                    .expect("Loaded model has coefficients")[i]
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
        std::fs::write(path, b"CORRUPTED DATA").expect("Should write test file");

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
        let err = result.expect_err("Should fail when loading nonexistent file");
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
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut model = LogisticRegression::new()
            .with_learning_rate(0.1)
            .with_max_iter(1000);
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        let probas_before = model.predict_proba(&x);

        // Save and load
        let path = "/tmp/test_logistic_probas.safetensors";
        model
            .save_safetensors(path)
            .expect("Should save fitted model to valid path");
        let loaded =
            LogisticRegression::load_safetensors(path).expect("Should load valid SafeTensors file");

        let probas_after = loaded.predict_proba(&x);

        // Verify probabilities match exactly
        assert_eq!(probas_before.len(), probas_after.len());
        for i in 0..probas_before.len() {
            assert_eq!(probas_before[i], probas_after[i]);
        }

        std::fs::remove_file(path).ok();
    }

    // K-Nearest Neighbors tests
    #[test]
    fn test_knn_new() {
        let knn = KNearestNeighbors::new(3);
        assert_eq!(knn.k, 3);
        assert_eq!(knn.metric, DistanceMetric::Euclidean);
        assert!(!knn.weights);
    }

    #[test]
    fn test_knn_basic_fit_predict() {
        // Simple 2-class problem
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 0
                5.0, 5.0, // class 1
                5.0, 6.0, // class 1
                6.0, 5.0, // class 1
            ],
        )
        .expect("6x2 matrix with 12 values");
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut knn = KNearestNeighbors::new(3);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Test point close to class 0
        let test1 = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
        let pred1 = knn.predict(&test1).expect("Prediction should succeed");
        assert_eq!(pred1[0], 0);

        // Test point close to class 1
        let test2 = Matrix::from_vec(1, 2, vec![5.5, 5.5]).expect("1x2 test matrix");
        let pred2 = knn.predict(&test2).expect("Prediction should succeed");
        assert_eq!(pred2[0], 1);
    }

    #[test]
    fn test_knn_k_equals_one() {
        // With k=1, should predict nearest neighbor exactly
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                1.0, 1.0, // class 1
                2.0, 2.0, // class 0
                3.0, 3.0, // class 1
            ],
        )
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 1, 0, 1];

        let mut knn = KNearestNeighbors::new(1);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Predict on training data - should be perfect
        let predictions = knn.predict(&x).expect("Prediction should succeed");
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_knn_euclidean_distance() {
        let x = Matrix::from_vec(
            3,
            2,
            vec![
                0.0, 0.0, // class 0
                3.0, 4.0, // class 1 (distance 5.0 from origin)
                1.0, 1.0, // class 0
            ],
        )
        .expect("3x2 matrix with 6 values");
        let y = vec![0, 1, 0];

        let mut knn = KNearestNeighbors::new(1).with_metric(DistanceMetric::Euclidean);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Test point at (1.5, 2.0) - closer to (1, 1) than (3, 4)
        let test = Matrix::from_vec(1, 2, vec![1.5, 2.0]).expect("1x2 test matrix");
        let pred = knn.predict(&test).expect("Prediction should succeed");
        assert_eq!(pred[0], 0);
    }

    #[test]
    fn test_knn_manhattan_distance() {
        let x = Matrix::from_vec(
            3,
            2,
            vec![
                0.0, 0.0, // class 0
                2.0, 2.0, // class 1 (Manhattan distance 4.0 from origin)
                1.0, 0.0, // class 0
            ],
        )
        .expect("3x2 matrix with 6 values");
        let y = vec![0, 1, 0];

        let mut knn = KNearestNeighbors::new(1).with_metric(DistanceMetric::Manhattan);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        let test = Matrix::from_vec(1, 2, vec![0.5, 0.0]).expect("1x2 test matrix");
        let pred = knn.predict(&test).expect("Prediction should succeed");
        assert_eq!(pred[0], 0); // Closer to (1, 0)
    }

    #[test]
    fn test_knn_minkowski_distance() {
        let x = Matrix::from_vec(
            3,
            2,
            vec![
                0.0, 0.0, // class 0
                3.0, 4.0, // class 1
                1.0, 1.0, // class 0
            ],
        )
        .expect("3x2 matrix with 6 values");
        let y = vec![0, 1, 0];

        // Minkowski with p=3
        let mut knn = KNearestNeighbors::new(1).with_metric(DistanceMetric::Minkowski(3.0));
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        let test = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
        let pred = knn.predict(&test).expect("Prediction should succeed");
        assert_eq!(pred[0], 0);
    }

    #[test]
    fn test_knn_weighted_voting() {
        // Set up data where uniform voting gives different result than weighted
        let x = Matrix::from_vec(
            5,
            1,
            vec![
                0.0, // class 0
                0.1, // class 0
                5.0, // class 1
                5.5, // class 1
                6.0, // class 1
            ],
        )
        .expect("5x1 matrix with 5 values");
        let y = vec![0, 0, 1, 1, 1];

        let mut knn_weighted = KNearestNeighbors::new(3).with_weights(true);
        knn_weighted
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Test point at 0.05 - very close to class 0
        let test = Matrix::from_vec(1, 1, vec![0.05]).expect("1x1 test matrix");
        let pred = knn_weighted
            .predict(&test)
            .expect("Prediction should succeed");
        assert_eq!(pred[0], 0); // Should be class 0 due to proximity weighting
    }

    #[test]
    fn test_knn_predict_proba() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 0
                5.0, 5.0, // class 1
                5.0, 6.0, // class 1
                6.0, 5.0, // class 1
            ],
        )
        .expect("6x2 matrix with 12 values");
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut knn = KNearestNeighbors::new(3);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        let test = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
        let probas = knn
            .predict_proba(&test)
            .expect("Probability prediction should succeed");

        assert_eq!(probas.len(), 1);
        assert_eq!(probas[0].len(), 2); // 2 classes

        // Probabilities should sum to 1.0
        let sum: f32 = probas[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Point closer to class 0 should have higher probability for class 0
        assert!(probas[0][0] > probas[0][1]);
    }

    #[test]
    fn test_knn_multiclass() {
        // 3-class problem
        let x = Matrix::from_vec(
            9,
            2,
            vec![
                0.0, 0.0, // class 0
                0.0, 1.0, // class 0
                1.0, 0.0, // class 0
                5.0, 5.0, // class 1
                5.0, 6.0, // class 1
                6.0, 5.0, // class 1
                10.0, 10.0, // class 2
                10.0, 11.0, // class 2
                11.0, 10.0, // class 2
            ],
        )
        .expect("9x2 matrix with 18 values");
        let y = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let mut knn = KNearestNeighbors::new(3);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Test each cluster
        let test1 = Matrix::from_vec(1, 2, vec![0.5, 0.5]).expect("1x2 test matrix");
        assert_eq!(
            knn.predict(&test1).expect("Prediction should succeed")[0],
            0
        );

        let test2 = Matrix::from_vec(1, 2, vec![5.5, 5.5]).expect("1x2 test matrix");
        assert_eq!(
            knn.predict(&test2).expect("Prediction should succeed")[0],
            1
        );

        let test3 = Matrix::from_vec(1, 2, vec![10.5, 10.5]).expect("1x2 test matrix");
        assert_eq!(
            knn.predict(&test3).expect("Prediction should succeed")[0],
            2
        );
    }

    #[test]
    fn test_knn_not_fitted_error() {
        let knn = KNearestNeighbors::new(3);
        let test = Matrix::from_vec(1, 2, vec![0.0, 0.0]).expect("1x2 test matrix");

        let result = knn.predict(&test);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail when predicting with unfitted model"),
            "Model not fitted"
        );
    }

    #[test]
    fn test_knn_dimension_mismatch() {
        let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
            .expect("3x2 matrix with 6 values");
        let y = vec![0, 1, 0];

        let mut knn = KNearestNeighbors::new(1);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Test with wrong number of features
        let test = Matrix::from_vec(1, 3, vec![0.0, 0.0, 0.0]).expect("1x3 test matrix");
        let result = knn.predict(&test);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with dimension mismatch"),
            "Feature dimension mismatch"
        );
    }

    #[test]
    fn test_knn_sample_mismatch() {
        let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
            .expect("3x2 matrix with 6 values");
        let y = vec![0, 1]; // Wrong length

        let mut knn = KNearestNeighbors::new(1);
        let result = knn.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with sample mismatch"),
            "Number of samples in X and y must match"
        );
    }

    #[test]
    fn test_knn_k_too_large() {
        let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
            .expect("3x2 matrix with 6 values");
        let y = vec![0, 1, 0];

        let mut knn = KNearestNeighbors::new(5); // k > n_samples
        let result = knn.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail when k exceeds sample count"),
            "k cannot be larger than number of training samples"
        );
    }

    #[test]
    fn test_knn_empty_data() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
        let y = vec![];

        let mut knn = KNearestNeighbors::new(1);
        let result = knn.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with empty data"),
            "Cannot fit with zero samples"
        );
    }

    #[test]
    fn test_knn_builder_pattern() {
        let knn = KNearestNeighbors::new(5)
            .with_metric(DistanceMetric::Manhattan)
            .with_weights(true);

        assert_eq!(knn.k, 5);
        assert_eq!(knn.metric, DistanceMetric::Manhattan);
        assert!(knn.weights);
    }

    #[test]
    fn test_knn_distance_symmetry() {
        // Property test: distance(a, b) == distance(b, a)
        let x = Matrix::from_vec(
            2,
            2,
            vec![
                1.0, 2.0, // point a
                3.0, 4.0, // point b
            ],
        )
        .expect("2x2 matrix with 4 values");
        let y = vec![0, 1];

        let mut knn = KNearestNeighbors::new(1);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Compute both directions
        let dist_ab = knn.compute_distance(&x, 0, &x, 1, 2);
        let dist_ba = knn.compute_distance(&x, 1, &x, 0, 2);

        assert!((dist_ab - dist_ba).abs() < 1e-6);
    }

    #[test]
    fn test_knn_perfect_fit_with_k1() {
        // Property test: k=1 on training data gives perfect predictions
        let x = Matrix::from_vec(
            10,
            3,
            vec![
                1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0,
                7.0, 8.0, 7.0, 8.0, 9.0, 8.0, 9.0, 10.0, 9.0, 10.0, 11.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("10x3 matrix with 30 values");
        let y = vec![0, 0, 1, 1, 0, 1, 0, 1, 0, 1];

        let mut knn = KNearestNeighbors::new(1);
        knn.fit(&x, &y)
            .expect("Training should succeed with valid data");

        let predictions = knn.predict(&x).expect("Prediction should succeed");
        assert_eq!(predictions, y);
    }

    // ========== Gaussian Naive Bayes Tests ==========

    #[test]
    fn test_gaussian_nb_new() {
        let model = GaussianNB::new();
        assert!(model.class_priors.is_none());
        assert!(model.means.is_none());
        assert!(model.variances.is_none());
        assert_eq!(model.var_smoothing, 1e-9);
    }

    #[test]
    fn test_gaussian_nb_builder() {
        let model = GaussianNB::new().with_var_smoothing(1e-8);
        assert_eq!(model.var_smoothing, 1e-8);
    }

    #[test]
    fn test_gaussian_nb_basic_fit_predict() {
        // Simple 2-class problem: class 0 at (0,0), class 1 at (1,1)
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.1, 0.1, // class 0
                1.0, 1.0, // class 1
                0.9, 0.9, // class 1
            ],
        )
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut model = GaussianNB::new();
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        let predictions = model.predict(&x).expect("Prediction should succeed");
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_gaussian_nb_multiclass() {
        // 3-class problem
        let x = Matrix::from_vec(
            9,
            2,
            vec![
                0.0, 0.0, // class 0
                0.1, 0.1, // class 0
                0.0, 0.1, // class 0
                5.0, 5.0, // class 1
                5.1, 5.1, // class 1
                5.0, 5.1, // class 1
                -5.0, -5.0, // class 2
                -5.1, -5.1, // class 2
                -5.0, -5.1, // class 2
            ],
        )
        .expect("9x2 matrix with 18 values");
        let y = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let mut model = GaussianNB::new();
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        let predictions = model.predict(&x).expect("Prediction should succeed");
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_gaussian_nb_predict_proba() {
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.1, 0.1, // class 0
                1.0, 1.0, // class 1
                0.9, 0.9, // class 1
            ],
        )
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut model = GaussianNB::new();
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        let probabilities = model
            .predict_proba(&x)
            .expect("Probability prediction should succeed");

        // Check all samples have probabilities
        assert_eq!(probabilities.len(), 4);

        // Check probabilities sum to 1
        for probs in &probabilities {
            assert_eq!(probs.len(), 2);
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        // Check first sample (class 0) has high probability for class 0
        assert!(probabilities[0][0] > 0.5);

        // Check last sample (class 1) has high probability for class 1
        assert!(probabilities[3][1] > 0.5);
    }

    #[test]
    fn test_gaussian_nb_not_fitted_error() {
        let model = GaussianNB::new();
        let x_test = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 test matrix");

        let result = model.predict(&x_test);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail when predicting with unfitted model"),
            "Model not fitted"
        );
    }

    #[test]
    fn test_gaussian_nb_empty_data() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
        let y: Vec<usize> = vec![];

        let mut model = GaussianNB::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with empty data"),
            "Cannot fit with empty data"
        );
    }

    #[test]
    fn test_gaussian_nb_sample_mismatch() {
        let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
            .expect("3x2 matrix with 6 values");
        let y = vec![0, 1]; // Wrong length

        let mut model = GaussianNB::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with sample mismatch"),
            "Number of samples in X and y must match"
        );
    }

    #[test]
    fn test_gaussian_nb_single_class() {
        let x = Matrix::from_vec(3, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
            .expect("3x2 matrix with 6 values");
        let y = vec![0, 0, 0]; // All same class

        let mut model = GaussianNB::new();
        let result = model.fit(&x, &y);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with single class"),
            "Need at least 2 classes"
        );
    }

    #[test]
    fn test_gaussian_nb_dimension_mismatch() {
        let x_train = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.1, 0.1, 1.0, 1.0, 0.9, 0.9])
            .expect("4x2 training matrix");
        let y_train = vec![0, 0, 1, 1];

        let mut model = GaussianNB::new();
        model
            .fit(&x_train, &y_train)
            .expect("Training should succeed with valid data");

        let x_test =
            Matrix::from_vec(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("2x3 test matrix");
        let result = model.predict(&x_test);

        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with dimension mismatch"),
            "Feature dimension mismatch"
        );
    }

    #[test]
    fn test_gaussian_nb_balanced_classes() {
        // Equal number of samples per class
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, // class 0
                0.1, 0.1, // class 0
                0.2, 0.2, // class 0
                1.0, 1.0, // class 1
                1.1, 1.1, // class 1
                1.2, 1.2, // class 1
            ],
        )
        .expect("6x2 matrix with 12 values");
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut model = GaussianNB::new();
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Check class priors are equal
        let priors = model
            .class_priors
            .expect("Model is fitted and has class priors");
        assert!((priors[0] - 0.5).abs() < 1e-5);
        assert!((priors[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_gaussian_nb_imbalanced_classes() {
        // Imbalanced: 1 sample class 0, 3 samples class 1
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                1.0, 1.0, // class 1
                1.1, 1.1, // class 1
                1.2, 1.2, // class 1
            ],
        )
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 1, 1, 1];

        let mut model = GaussianNB::new();
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Check class priors reflect imbalance
        let priors = model
            .class_priors
            .expect("Model is fitted and has class priors");
        assert!((priors[0] - 0.25).abs() < 1e-5); // 1/4
        assert!((priors[1] - 0.75).abs() < 1e-5); // 3/4
    }

    #[test]
    fn test_gaussian_nb_var_smoothing() {
        // Test that variance smoothing prevents division by zero
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0 - identical points
                0.0, 0.0, // class 0 - identical points
                1.0, 1.0, // class 1 - identical points
                1.0, 1.0, // class 1 - identical points
            ],
        )
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut model = GaussianNB::new().with_var_smoothing(1e-8);
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Should not panic or produce NaN/Inf
        let predictions = model.predict(&x).expect("Prediction should succeed");
        assert_eq!(predictions, y);

        let probabilities = model
            .predict_proba(&x)
            .expect("Probability prediction should succeed");
        for probs in &probabilities {
            for &p in probs {
                assert!(p.is_finite());
                assert!((0.0..=1.0).contains(&p));
            }
        }
    }

    #[test]
    fn test_gaussian_nb_probabilities_sum_to_one() {
        // Property test: probabilities must sum to 1
        let x = Matrix::from_vec(
            10,
            3,
            vec![
                0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 1.1,
                1.1, 1.1, 1.2, 1.2, 1.2, 1.3, 1.3, 1.3, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1,
            ],
        )
        .expect("10x3 matrix with 30 values");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2];

        let mut model = GaussianNB::new();
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        let probabilities = model
            .predict_proba(&x)
            .expect("Probability prediction should succeed");

        for probs in &probabilities {
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_gaussian_nb_default() {
        let model1 = GaussianNB::new();
        let model2 = GaussianNB::default();

        assert_eq!(model1.var_smoothing, model2.var_smoothing);
    }

    #[test]
    fn test_gaussian_nb_class_separation() {
        // Well-separated classes should have high confidence
        let x = Matrix::from_vec(
            4,
            2,
            vec![
                0.0, 0.0, // class 0
                0.1, 0.1, // class 0
                10.0, 10.0, // class 1 (far away)
                10.1, 10.1, // class 1 (far away)
            ],
        )
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut model = GaussianNB::new();
        model
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        let probabilities = model
            .predict_proba(&x)
            .expect("Probability prediction should succeed");

        // First sample should have very high confidence for class 0
        assert!(probabilities[0][0] > 0.99);

        // Last sample should have very high confidence for class 1
        assert!(probabilities[3][1] > 0.99);
    }

    // ===== LinearSVM Tests =====

    #[test]
    fn test_linear_svm_new() {
        let svm = LinearSVM::new();
        assert!(svm.weights.is_none());
        assert_eq!(svm.bias, 0.0);
        assert_eq!(svm.c, 1.0);
        assert_eq!(svm.learning_rate, 0.01);
        assert_eq!(svm.max_iter, 1000);
        assert_eq!(svm.tol, 1e-4);
    }

    #[test]
    fn test_linear_svm_builder() {
        let svm = LinearSVM::new()
            .with_c(0.5)
            .with_learning_rate(0.001)
            .with_max_iter(500)
            .with_tolerance(1e-5);

        assert_eq!(svm.c, 0.5);
        assert_eq!(svm.learning_rate, 0.001);
        assert_eq!(svm.max_iter, 500);
        assert_eq!(svm.tol, 1e-5);
    }

    #[test]
    fn test_linear_svm_fit_simple() {
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
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut svm = LinearSVM::new().with_max_iter(1000).with_learning_rate(0.1);

        let result = svm.fit(&x, &y);
        assert!(result.is_ok());
        assert!(svm.weights.is_some());
    }

    #[test]
    fn test_linear_svm_predict_simple() {
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
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut svm = LinearSVM::new().with_max_iter(1000).with_learning_rate(0.1);
        svm.fit(&x, &y)
            .expect("Training should succeed with valid data");

        let predictions = svm.predict(&x).expect("Prediction should succeed");
        assert_eq!(predictions.len(), 4);

        // Should classify correctly (or close to it)
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| *pred == *true_label)
            .count();

        // Should get at least 3 out of 4 correct for simple case
        assert!(correct >= 3);
    }

    #[test]
    fn test_linear_svm_decision_function() {
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
        .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        let mut svm = LinearSVM::new().with_max_iter(1000).with_learning_rate(0.1);
        svm.fit(&x, &y)
            .expect("Training should succeed with valid data");

        let decisions = svm
            .decision_function(&x)
            .expect("Decision function should succeed");
        assert_eq!(decisions.len(), 4);

        // Class 0 samples should have negative decisions
        // Class 1 samples should have positive decisions
        // (may not be perfect for simple gradient descent)
    }

    #[test]
    fn test_linear_svm_predict_untrained() {
        let svm = LinearSVM::new();
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0]).expect("2x2 matrix with 4 values");

        let result = svm.predict(&x);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail when predicting with untrained model"),
            "Model not trained yet"
        );
    }

    #[test]
    fn test_linear_svm_dimension_mismatch() {
        let x_train = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 training matrix");
        let y = vec![0, 0, 1, 1];

        let mut svm = LinearSVM::new();
        svm.fit(&x_train, &y)
            .expect("Training should succeed with valid data");

        // Try to predict with wrong number of features
        let x_test =
            Matrix::from_vec(2, 3, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).expect("2x3 test matrix");
        let result = svm.predict(&x_test);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with dimension mismatch"),
            "Feature dimension mismatch"
        );
    }

    #[test]
    fn test_linear_svm_empty_data() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("0x2 empty matrix");
        let y = vec![];

        let mut svm = LinearSVM::new();
        let result = svm.fit(&x, &y);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with empty data"),
            "Cannot fit with 0 samples"
        );
    }

    #[test]
    fn test_linear_svm_mismatched_samples() {
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1]; // Wrong length

        let mut svm = LinearSVM::new();
        let result = svm.fit(&x, &y);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with mismatched sample counts"),
            "x and y must have the same number of samples"
        );
    }

    #[test]
    fn test_linear_svm_regularization_c() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, // class 0
                0.1, 0.1, // class 0
                0.0, 0.2, // class 0
                1.0, 1.0, // class 1
                0.9, 0.9, // class 1
                1.0, 0.8, // class 1
            ],
        )
        .expect("6x2 matrix with 12 values");
        let y = vec![0, 0, 0, 1, 1, 1];

        // High C (less regularization) - should fit data more closely
        let mut svm_high_c = LinearSVM::new()
            .with_c(10.0)
            .with_max_iter(1000)
            .with_learning_rate(0.1);
        svm_high_c
            .fit(&x, &y)
            .expect("Training should succeed with valid data");
        let pred_high_c = svm_high_c.predict(&x).expect("Prediction should succeed");

        // Low C (more regularization) - should prefer simpler model
        let mut svm_low_c = LinearSVM::new()
            .with_c(0.1)
            .with_max_iter(1000)
            .with_learning_rate(0.1);
        svm_low_c
            .fit(&x, &y)
            .expect("Training should succeed with valid data");
        let pred_low_c = svm_low_c.predict(&x).expect("Prediction should succeed");

        // Both should make predictions
        assert_eq!(pred_high_c.len(), 6);
        assert_eq!(pred_low_c.len(), 6);
    }

    #[test]
    fn test_linear_svm_binary_classification() {
        // More realistic binary classification problem
        let x = Matrix::from_vec(
            10,
            2,
            vec![
                // Class 0 (bottom-left cluster)
                0.0, 0.0, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0, 0.1,
                0.2, // Class 1 (top-right cluster)
                1.0, 1.0, 0.9, 0.9, 1.0, 0.8, 0.8, 1.0, 0.9, 1.1,
            ],
        )
        .expect("10x2 matrix with 20 values");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let mut svm = LinearSVM::new()
            .with_c(1.0)
            .with_max_iter(2000)
            .with_learning_rate(0.1);

        svm.fit(&x, &y)
            .expect("Training should succeed with valid data");
        let predictions = svm.predict(&x).expect("Prediction should succeed");

        // Should achieve reasonable accuracy
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| *pred == *true_label)
            .count();

        // Should get at least 8 out of 10 correct for well-separated clusters
        assert!(
            correct >= 8,
            "Expected at least 8/10 correct, got {}/10",
            correct
        );
    }

    #[test]
    fn test_linear_svm_convergence() {
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("4x2 matrix with 8 values");
        let y = vec![0, 0, 1, 1];

        // With very few iterations, might not converge
        let mut svm_few_iter = LinearSVM::new().with_max_iter(10).with_learning_rate(0.01);
        svm_few_iter
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        // With many iterations, should converge better
        let mut svm_many_iter = LinearSVM::new().with_max_iter(2000).with_learning_rate(0.1);
        svm_many_iter
            .fit(&x, &y)
            .expect("Training should succeed with valid data");

        // Both should train successfully
        assert!(svm_few_iter.weights.is_some());
        assert!(svm_many_iter.weights.is_some());
    }

    #[test]
    fn test_linear_svm_default() {
        let svm1 = LinearSVM::new();
        let svm2 = LinearSVM::default();

        assert_eq!(svm1.c, svm2.c);
        assert_eq!(svm1.learning_rate, svm2.learning_rate);
        assert_eq!(svm1.max_iter, svm2.max_iter);
    }
}
