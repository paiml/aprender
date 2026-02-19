//! Random Forest implementations for classification and regression.
//!
//! Ensemble methods using bootstrap sampling and majority voting.

use super::helpers::{
    bootstrap_sample, compute_regression_tree_feature_importances,
    compute_tree_feature_importances, flatten_tree_node, reconstruct_tree_node,
};
use super::{DecisionTreeClassifier, DecisionTreeRegressor};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ============================================================================
// Random Forest Regressor
// ============================================================================

/// Random Forest Regressor.
///
/// Ensemble of decision tree regressors trained on bootstrap samples.
/// Predictions are averaged across all trees to reduce variance and overfitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestRegressor {
    pub(super) trees: Vec<DecisionTreeRegressor>,
    pub(super) n_estimators: usize,
    pub(super) max_depth: Option<usize>,
    pub(super) random_state: Option<u64>,
    /// OOB sample indices for each tree (samples NOT in bootstrap sample)
    pub(super) oob_indices: Vec<Vec<usize>>,
    /// Training features (stored for OOB evaluation)
    pub(super) x_train: Option<crate::primitives::Matrix<f32>>,
    /// Training targets (stored for OOB evaluation)
    pub(super) y_train: Option<crate::primitives::Vector<f32>>,
}

impl RandomForestRegressor {
    /// Creates a new Random Forest regressor.
    #[must_use]
    pub fn new(n_estimators: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth: None,
            random_state: None,
            oob_indices: Vec::new(),
            x_train: None,
            y_train: None,
        }
    }

    /// Sets the maximum depth for each tree.
    #[must_use]
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Sets the random state for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fits the random forest to training data.
    ///
    /// # Errors
    ///
    /// Returns an error if fitting fails.
    // Contract: random-forest-v1, equation = "bootstrap_sample"
    pub fn fit(
        &mut self,
        x: &crate::primitives::Matrix<f32>,
        y: &crate::primitives::Vector<f32>,
    ) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        // Validate input
        if n_samples != y.len() {
            return Err("Number of samples in X and y must match".into());
        }
        if n_samples == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        self.trees = Vec::with_capacity(self.n_estimators);
        self.oob_indices = Vec::with_capacity(self.n_estimators);

        // Store training data for OOB evaluation
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());

        // Train each tree on a bootstrap sample
        for i in 0..self.n_estimators {
            // Get bootstrap sample indices
            let seed = self.random_state.map(|s| s + i as u64);
            let bootstrap_indices = bootstrap_sample(n_samples, seed);

            // Compute OOB indices (samples NOT in bootstrap sample)
            let bootstrap_set: HashSet<usize> = bootstrap_indices.iter().copied().collect();
            let oob_for_tree: Vec<usize> = (0..n_samples)
                .filter(|idx| !bootstrap_set.contains(idx))
                .collect();
            self.oob_indices.push(oob_for_tree);

            // Extract bootstrap sample
            let mut bootstrap_x_data = Vec::with_capacity(n_samples * n_features);
            let mut bootstrap_y_data = Vec::with_capacity(n_samples);

            for &idx in &bootstrap_indices {
                for j in 0..n_features {
                    bootstrap_x_data.push(x.get(idx, j));
                }
                bootstrap_y_data.push(y.as_slice()[idx]);
            }

            let bootstrap_x =
                crate::primitives::Matrix::from_vec(n_samples, n_features, bootstrap_x_data)
                    .map_err(|_| "Failed to create bootstrap matrix")?;
            let bootstrap_y = crate::primitives::Vector::from_slice(&bootstrap_y_data);

            // Create and train a decision tree
            let mut tree = if let Some(max_depth) = self.max_depth {
                DecisionTreeRegressor::new().with_max_depth(max_depth)
            } else {
                DecisionTreeRegressor::new()
            };

            tree.fit(&bootstrap_x, &bootstrap_y)?;
            self.trees.push(tree);
        }

        Ok(())
    }

    /// Makes predictions for input data by averaging predictions from all trees.
    ///
    /// # Panics
    ///
    /// Panics if the model hasn't been fitted yet.
    // Contract: random-forest-v1, equation = "majority_vote"
    #[must_use]
    pub fn predict(&self, x: &crate::primitives::Matrix<f32>) -> crate::primitives::Vector<f32> {
        assert!(
            !self.trees.is_empty(),
            "Cannot predict with an unfitted Random Forest. Call fit() first."
        );

        let n_samples = x.shape().0;
        let mut predictions = vec![0.0; n_samples];

        // Get predictions from each tree and average
        for tree in &self.trees {
            let tree_preds = tree.predict(x);
            for (pred, &tree_pred) in predictions.iter_mut().zip(tree_preds.as_slice().iter()) {
                *pred += tree_pred;
            }
        }

        // Average the predictions
        let n_trees = self.trees.len() as f32;
        for pred in &mut predictions {
            *pred /= n_trees;
        }

        crate::primitives::Vector::from_slice(&predictions)
    }

    /// Calculates R² score on test data.
    #[must_use]
    pub fn score(
        &self,
        x: &crate::primitives::Matrix<f32>,
        y: &crate::primitives::Vector<f32>,
    ) -> f32 {
        let predictions = self.predict(x);
        crate::metrics::r_squared(y, &predictions)
    }

    /// Returns Out-of-Bag (OOB) predictions for training samples.
    #[must_use]
    pub fn oob_prediction(&self) -> Option<crate::primitives::Vector<f32>> {
        if self.trees.is_empty() || self.y_train.is_none() || self.x_train.is_none() {
            return None;
        }

        let x_train = self.x_train.as_ref()?;
        let y_train = self.y_train.as_ref()?;
        let n_samples = y_train.len();
        let n_features = x_train.shape().1;

        let mut oob_predictions = vec![0.0; n_samples];
        let mut oob_counts = vec![0; n_samples];

        for (tree_idx, oob_indices) in self.oob_indices.iter().enumerate() {
            let tree = &self.trees[tree_idx];

            for &sample_idx in oob_indices {
                let mut sample_data = Vec::with_capacity(n_features);
                for j in 0..n_features {
                    sample_data.push(x_train.get(sample_idx, j));
                }

                let sample_matrix =
                    crate::primitives::Matrix::from_vec(1, n_features, sample_data).ok()?;

                let tree_predictions = tree.predict(&sample_matrix);
                let predicted_value = tree_predictions.as_slice()[0];

                oob_predictions[sample_idx] += predicted_value;
                oob_counts[sample_idx] += 1;
            }
        }

        for (i, count) in oob_counts.iter().enumerate() {
            if *count > 0 {
                oob_predictions[i] /= *count as f32;
            }
        }

        Some(crate::primitives::Vector::from_slice(&oob_predictions))
    }

    /// Returns Out-of-Bag (OOB) R² score.
    #[must_use]
    pub fn oob_score(&self) -> Option<f32> {
        let oob_preds = self.oob_prediction()?;
        let y_train = self.y_train.as_ref()?;
        Some(crate::metrics::r_squared(y_train, &oob_preds))
    }

    /// Returns feature importances based on mean decrease in variance.
    #[must_use]
    pub fn feature_importances(&self) -> Option<Vec<f32>> {
        if self.trees.is_empty() || self.x_train.is_none() {
            return None;
        }

        let n_features = self.x_train.as_ref()?.shape().1;
        let mut total_importances = vec![0.0; n_features];

        for tree in &self.trees {
            if let Some(tree_node) = &tree.tree {
                let mut tree_importances = vec![0.0; n_features];
                compute_regression_tree_feature_importances(tree_node, &mut tree_importances);

                for (i, &importance) in tree_importances.iter().enumerate() {
                    total_importances[i] += importance;
                }
            }
        }

        let n_trees = self.trees.len() as f32;
        for importance in &mut total_importances {
            *importance /= n_trees;
        }

        let total_sum: f32 = total_importances.iter().sum();
        if total_sum > 0.0 {
            for importance in &mut total_importances {
                *importance /= total_sum;
            }
        }

        Some(total_importances)
    }
}

impl Default for RandomForestRegressor {
    fn default() -> Self {
        Self::new(10)
    }
}

// ============================================================================
// Random Forest Classifier
// ============================================================================

/// Random Forest classifier - an ensemble of decision trees.
///
/// Combines multiple decision trees trained on bootstrap samples
/// with random feature selection to reduce overfitting and improve accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier {
    pub(super) trees: Vec<DecisionTreeClassifier>,
    pub(super) n_estimators: usize,
    pub(super) max_depth: Option<usize>,
    pub(super) random_state: Option<u64>,
    /// OOB sample indices for each tree (samples NOT in bootstrap sample)
    pub(super) oob_indices: Vec<Vec<usize>>,
    /// Training features (stored for OOB evaluation)
    pub(super) x_train: Option<crate::primitives::Matrix<f32>>,
    /// Training labels (stored for OOB evaluation)
    pub(super) y_train: Option<Vec<usize>>,
}

include!("random_forest_part_02.rs");
include!("random_forest_part_03.rs");
