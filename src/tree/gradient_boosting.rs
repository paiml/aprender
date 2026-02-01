//! Gradient Boosting Classifier implementation.
//!
//! Implements gradient boosting with decision trees as weak learners.

use super::DecisionTreeClassifier;
use crate::error::Result;

/// Gradient Boosting Classifier.
///
/// Implements gradient boosting with decision trees as weak learners.
/// Uses gradient descent in function space to iteratively improve predictions.
///
/// # Algorithm
///
/// 1. Initialize with constant prediction (log-odds)
/// 2. For each boosting iteration:
///    - Compute negative gradients (pseudo-residuals)
///    - Fit a small decision tree to residuals
///    - Update predictions with `learning_rate` * `tree_prediction`
/// 3. Final prediction = sigmoid(sum of all tree predictions)
#[derive(Debug, Clone)]
pub struct GradientBoostingClassifier {
    /// Number of boosting iterations (trees)
    n_estimators: usize,
    /// Learning rate (shrinkage parameter)
    learning_rate: f32,
    /// Maximum depth of each tree
    max_depth: usize,
    /// Initial prediction (log-odds for class 1)
    init_prediction: f32,
    /// Ensemble of decision trees
    estimators: Vec<DecisionTreeClassifier>,
}

impl GradientBoostingClassifier {
    /// Creates a new Gradient Boosting Classifier with default parameters.
    ///
    /// # Default Parameters
    ///
    /// - `n_estimators`: 100
    /// - `learning_rate`: 0.1
    /// - `max_depth`: 3
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 3,
            init_prediction: 0.0,
            estimators: Vec::new(),
        }
    }

    /// Sets the number of boosting iterations (trees).
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Sets the learning rate (shrinkage parameter).
    ///
    /// Lower values require more trees but often lead to better generalization.
    /// Typical values: 0.01 - 0.3
    #[must_use]
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum depth of each tree.
    ///
    /// Smaller depths prevent overfitting. Typical values: 3-8
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sigmoid function: σ(x) = 1 / (1 + e^(-x))
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Trains the Gradient Boosting Classifier.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (`n_samples` × `n_features`)
    /// - `y`: Binary labels (0 or 1)
    ///
    /// # Returns
    ///
    /// Ok(()) on success, Err with message on failure.
    pub fn fit(&mut self, x: &crate::primitives::Matrix<f32>, y: &[usize]) -> Result<()> {
        if x.n_rows() != y.len() {
            return Err("x and y must have the same number of samples".into());
        }

        if x.n_rows() == 0 {
            return Err("Cannot fit with 0 samples".into());
        }

        let n_samples = x.n_rows();

        // Convert labels to {0.0, 1.0}
        let y_float: Vec<f32> = y.iter().map(|&label| label as f32).collect();

        // Initialize prediction with log-odds
        let positive_count = y_float.iter().filter(|&&label| label == 1.0).count();
        let p = positive_count as f32 / n_samples as f32;
        self.init_prediction = if p > 0.0 && p < 1.0 {
            (p / (1.0 - p)).ln()
        } else if p >= 1.0 {
            5.0 // Large positive value
        } else {
            -5.0 // Large negative value
        };

        // Current raw predictions (in log-odds space)
        let mut raw_predictions = vec![self.init_prediction; n_samples];

        // Clear any existing estimators
        self.estimators = Vec::with_capacity(self.n_estimators);

        // Gradient boosting iterations
        for _ in 0..self.n_estimators {
            // Compute probabilities from raw predictions
            let probabilities: Vec<f32> =
                raw_predictions.iter().map(|&r| Self::sigmoid(r)).collect();

            // Compute negative gradients (pseudo-residuals)
            // For log-loss: residual = y - p
            let residuals: Vec<f32> = y_float
                .iter()
                .zip(probabilities.iter())
                .map(|(&yi, &pi)| yi - pi)
                .collect();

            // Convert residuals to discrete labels for tree fitting
            // Positive residual -> predict 1, Negative residual -> predict 0
            let residual_labels = self.residuals_to_labels(&residuals);

            // Fit a tree to the residuals
            let mut tree = DecisionTreeClassifier::new().with_max_depth(self.max_depth);
            tree.fit(x, &residual_labels)?;

            // Get tree predictions (these are class labels 0 or 1)
            let tree_preds = tree.predict(x);

            // Convert tree predictions back to residual estimates
            // Map 0 -> -1, 1 -> +1 for residual direction
            let tree_residuals: Vec<f32> = tree_preds
                .iter()
                .map(|&pred| if pred == 0 { -1.0 } else { 1.0 })
                .collect();

            // Update raw predictions
            for i in 0..n_samples {
                raw_predictions[i] += self.learning_rate * tree_residuals[i];
            }

            self.estimators.push(tree);
        }

        Ok(())
    }

    /// Converts residuals to class labels for tree fitting.
    ///
    /// Positive residuals -> class 1, negative residuals -> class 0
    #[allow(clippy::unused_self)]
    fn residuals_to_labels(&self, residuals: &[f32]) -> Vec<usize> {
        residuals.iter().map(|&r| usize::from(r >= 0.0)).collect()
    }

    /// Predicts class labels for the given samples.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (`n_samples` × `n_features`)
    ///
    /// # Returns
    ///
    /// Vector of predicted labels (0 or 1).
    pub fn predict(&self, x: &crate::primitives::Matrix<f32>) -> Result<Vec<usize>> {
        let probas = self.predict_proba(x)?;
        Ok(probas
            .iter()
            .map(|probs| usize::from(probs[1] >= 0.5))
            .collect())
    }

    /// Predicts class probabilities for the given samples.
    ///
    /// # Arguments
    ///
    /// - `x`: Feature matrix (`n_samples` × `n_features`)
    ///
    /// # Returns
    ///
    /// Vector of probability distributions, one per sample.
    /// Each distribution is [P(class=0), P(class=1)].
    pub fn predict_proba(&self, x: &crate::primitives::Matrix<f32>) -> Result<Vec<Vec<f32>>> {
        if self.estimators.is_empty() {
            return Err("Model not trained yet".into());
        }

        let n_samples = x.n_rows();
        let mut raw_predictions = vec![self.init_prediction; n_samples];

        // Sum predictions from all trees
        for tree in &self.estimators {
            let tree_preds = tree.predict(x);
            let tree_residuals: Vec<f32> = tree_preds
                .iter()
                .map(|&pred| if pred == 0 { -1.0 } else { 1.0 })
                .collect();

            for i in 0..n_samples {
                raw_predictions[i] += self.learning_rate * tree_residuals[i];
            }
        }

        // Convert raw predictions to probabilities
        Ok(raw_predictions
            .iter()
            .map(|&raw| {
                let prob_class1 = Self::sigmoid(raw);
                let prob_class0 = 1.0 - prob_class1;
                vec![prob_class0, prob_class1]
            })
            .collect())
    }

    /// Returns the number of estimators (trees) in the ensemble.
    #[must_use]
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Returns the learning rate.
    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the max depth.
    #[must_use]
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the number of configured estimators.
    #[must_use]
    pub fn configured_n_estimators(&self) -> usize {
        self.n_estimators
    }

    /// Returns a reference to the estimators.
    #[must_use]
    pub fn estimators(&self) -> &[DecisionTreeClassifier] {
        &self.estimators
    }
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}
