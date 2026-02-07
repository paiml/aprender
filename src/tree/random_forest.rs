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

impl RandomForestClassifier {
    /// Creates a new Random Forest classifier.
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
    pub fn fit(&mut self, x: &crate::primitives::Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_samples, n_features) = x.shape();
        self.trees = Vec::with_capacity(self.n_estimators);
        self.oob_indices = Vec::with_capacity(self.n_estimators);

        // Store training data for OOB evaluation
        self.x_train = Some(x.clone());
        self.y_train = Some(y.to_vec());

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
            let mut bootstrap_y = Vec::with_capacity(n_samples);

            for &idx in &bootstrap_indices {
                for j in 0..n_features {
                    bootstrap_x_data.push(x.get(idx, j));
                }
                bootstrap_y.push(y[idx]);
            }

            let bootstrap_x =
                crate::primitives::Matrix::from_vec(n_samples, n_features, bootstrap_x_data)
                    .map_err(|_| "Failed to create bootstrap matrix")?;

            // Create and train a decision tree
            let mut tree = if let Some(max_depth) = self.max_depth {
                DecisionTreeClassifier::new().with_max_depth(max_depth)
            } else {
                DecisionTreeClassifier::new()
            };

            tree.fit(&bootstrap_x, &bootstrap_y)?;
            self.trees.push(tree);
        }

        Ok(())
    }

    /// Makes predictions for input data.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn predict(&self, x: &crate::primitives::Matrix<f32>) -> Vec<usize> {
        let n_samples = x.shape().0;
        let mut predictions = vec![0; n_samples];

        for sample_idx in 0..n_samples {
            let mut votes: HashMap<usize, usize> = HashMap::new();

            for tree in &self.trees {
                let tree_prediction = tree.predict(x)[sample_idx];
                *votes.entry(tree_prediction).or_insert(0) += 1;
            }

            let mut max_votes = 0;
            let mut predicted_class = 0;
            for (class, count) in votes {
                if count > max_votes {
                    max_votes = count;
                    predicted_class = class;
                }
            }

            predictions[sample_idx] = predicted_class;
        }

        predictions
    }

    /// Calculates accuracy score on test data.
    #[must_use]
    pub fn score(&self, x: &crate::primitives::Matrix<f32>, y: &[usize]) -> f32 {
        let predictions = self.predict(x);
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();
        correct as f32 / y.len() as f32
    }

    /// Predict class probabilities for input features.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn predict_proba(
        &self,
        x: &crate::primitives::Matrix<f32>,
    ) -> crate::primitives::Matrix<f32> {
        let n_samples = x.shape().0;

        let n_classes = self
            .y_train
            .as_ref()
            .map_or(2, |y| y.iter().max().copied().unwrap_or(0) + 1);

        let mut proba_data = vec![0.0f32; n_samples * n_classes];
        let n_trees = self.trees.len() as f32;

        for sample_idx in 0..n_samples {
            let mut votes = vec![0usize; n_classes];

            for tree in &self.trees {
                let tree_prediction = tree.predict(x)[sample_idx];
                if tree_prediction < n_classes {
                    votes[tree_prediction] += 1;
                }
            }

            for class_idx in 0..n_classes {
                let idx = sample_idx * n_classes + class_idx;
                proba_data[idx] = votes[class_idx] as f32 / n_trees;
            }
        }

        crate::primitives::Matrix::from_vec(n_samples, n_classes, proba_data)
            .expect("Matrix creation should succeed")
    }

    /// Returns Out-of-Bag (OOB) predictions for training samples.
    #[must_use]
    pub fn oob_prediction(&self) -> Option<Vec<usize>> {
        if self.trees.is_empty() || self.y_train.is_none() || self.x_train.is_none() {
            return None;
        }

        let x_train = self.x_train.as_ref()?;
        let y_train = self.y_train.as_ref()?;
        let n_samples = y_train.len();
        let n_features = x_train.shape().1;

        let mut oob_votes: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n_samples];

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
                let predicted_class = tree_predictions[0];

                *oob_votes[sample_idx].entry(predicted_class).or_insert(0) += 1;
            }
        }

        let mut predictions = Vec::with_capacity(n_samples);
        for votes in oob_votes {
            if votes.is_empty() {
                predictions.push(0);
            } else {
                let mut max_votes = 0;
                let mut predicted_class = 0;
                for (class, count) in votes {
                    if count > max_votes {
                        max_votes = count;
                        predicted_class = class;
                    }
                }
                predictions.push(predicted_class);
            }
        }

        Some(predictions)
    }

    /// Returns Out-of-Bag (OOB) accuracy score.
    #[must_use]
    pub fn oob_score(&self) -> Option<f32> {
        let oob_preds = self.oob_prediction()?;
        let y_train = self.y_train.as_ref()?;

        let correct = oob_preds
            .iter()
            .zip(y_train.iter())
            .filter(|(pred, true_label)| pred == true_label)
            .count();

        Some(correct as f32 / y_train.len() as f32)
    }

    /// Returns feature importances based on mean decrease in impurity.
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
                compute_tree_feature_importances(tree_node, &mut tree_importances);

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

    /// Saves the Random Forest model to a `SafeTensors` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unfitted or if saving fails.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        if self.trees.is_empty() {
            return Err("Cannot save unfitted model. Call fit() first.".to_string());
        }

        let mut tensors = BTreeMap::new();

        for (tree_idx, tree) in self.trees.iter().enumerate() {
            let tree_node = tree
                .tree
                .as_ref()
                .ok_or("Tree has no root node".to_string())?;

            let mut node_features = Vec::new();
            let mut node_thresholds = Vec::new();
            let mut node_classes = Vec::new();
            let mut node_samples = Vec::new();
            let mut node_left_child = Vec::new();
            let mut node_right_child = Vec::new();

            flatten_tree_node(
                tree_node,
                &mut node_features,
                &mut node_thresholds,
                &mut node_classes,
                &mut node_samples,
                &mut node_left_child,
                &mut node_right_child,
            );

            let prefix = format!("tree_{tree_idx}_");
            tensors.insert(
                format!("{prefix}node_features"),
                (node_features.clone(), vec![node_features.len()]),
            );
            tensors.insert(
                format!("{prefix}node_thresholds"),
                (node_thresholds.clone(), vec![node_thresholds.len()]),
            );
            tensors.insert(
                format!("{prefix}node_classes"),
                (node_classes.clone(), vec![node_classes.len()]),
            );
            tensors.insert(
                format!("{prefix}node_samples"),
                (node_samples.clone(), vec![node_samples.len()]),
            );
            tensors.insert(
                format!("{prefix}node_left_child"),
                (node_left_child.clone(), vec![node_left_child.len()]),
            );
            tensors.insert(
                format!("{prefix}node_right_child"),
                (node_right_child.clone(), vec![node_right_child.len()]),
            );

            let tree_max_depth = tree.max_depth.map_or(-1.0, |d| d as f32);
            tensors.insert(
                format!("{prefix}max_depth"),
                (vec![tree_max_depth], vec![1]),
            );
        }

        tensors.insert(
            "n_estimators".to_string(),
            (vec![self.n_estimators as f32], vec![1]),
        );

        let max_depth_val = self.max_depth.map_or(-1.0, |d| d as f32);
        tensors.insert("max_depth".to_string(), (vec![max_depth_val], vec![1]));

        let random_state_val = self.random_state.map_or(-1.0, |s| s as f32);
        tensors.insert(
            "random_state".to_string(),
            (vec![random_state_val], vec![1]),
        );

        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a Random Forest model from a `SafeTensors` file.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails or if the file format is invalid.
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        let n_estimators_meta = metadata
            .get("n_estimators")
            .ok_or("Missing n_estimators tensor")?;
        let n_estimators_data = safetensors::extract_tensor(&raw_data, n_estimators_meta)?;
        let n_estimators = n_estimators_data[0] as usize;

        let max_depth_meta = metadata
            .get("max_depth")
            .ok_or("Missing max_depth tensor")?;
        let max_depth_data = safetensors::extract_tensor(&raw_data, max_depth_meta)?;
        let max_depth = if max_depth_data[0] < 0.0 {
            None
        } else {
            Some(max_depth_data[0] as usize)
        };

        let random_state_meta = metadata
            .get("random_state")
            .ok_or("Missing random_state tensor")?;
        let random_state_data = safetensors::extract_tensor(&raw_data, random_state_meta)?;
        let random_state = if random_state_data[0] < 0.0 {
            None
        } else {
            Some(random_state_data[0] as u64)
        };

        let mut trees = Vec::with_capacity(n_estimators);
        for tree_idx in 0..n_estimators {
            let prefix = format!("tree_{tree_idx}_");

            let node_features_meta = metadata
                .get(&format!("{prefix}node_features"))
                .ok_or(format!("Missing tree {tree_idx} node_features"))?;
            let node_features = safetensors::extract_tensor(&raw_data, node_features_meta)?;

            let node_thresholds_meta = metadata
                .get(&format!("{prefix}node_thresholds"))
                .ok_or(format!("Missing tree {tree_idx} node_thresholds"))?;
            let node_thresholds = safetensors::extract_tensor(&raw_data, node_thresholds_meta)?;

            let node_classes_meta = metadata
                .get(&format!("{prefix}node_classes"))
                .ok_or(format!("Missing tree {tree_idx} node_classes"))?;
            let node_classes = safetensors::extract_tensor(&raw_data, node_classes_meta)?;

            let node_samples_meta = metadata
                .get(&format!("{prefix}node_samples"))
                .ok_or(format!("Missing tree {tree_idx} node_samples"))?;
            let node_samples = safetensors::extract_tensor(&raw_data, node_samples_meta)?;

            let node_left_child_meta = metadata
                .get(&format!("{prefix}node_left_child"))
                .ok_or(format!("Missing tree {tree_idx} node_left_child"))?;
            let node_left_child = safetensors::extract_tensor(&raw_data, node_left_child_meta)?;

            let node_right_child_meta = metadata
                .get(&format!("{prefix}node_right_child"))
                .ok_or(format!("Missing tree {tree_idx} node_right_child"))?;
            let node_right_child = safetensors::extract_tensor(&raw_data, node_right_child_meta)?;

            let n_nodes = node_features.len();
            if node_thresholds.len() != n_nodes
                || node_classes.len() != n_nodes
                || node_samples.len() != n_nodes
                || node_left_child.len() != n_nodes
                || node_right_child.len() != n_nodes
            {
                return Err(format!("Mismatched array sizes for tree {tree_idx}"));
            }

            let tree_node = reconstruct_tree_node(
                0,
                &node_features,
                &node_thresholds,
                &node_classes,
                &node_samples,
                &node_left_child,
                &node_right_child,
            );

            let tree_max_depth_meta = metadata
                .get(&format!("{prefix}max_depth"))
                .ok_or(format!("Missing tree {tree_idx} max_depth"))?;
            let tree_max_depth_data = safetensors::extract_tensor(&raw_data, tree_max_depth_meta)?;
            let tree_max_depth = if tree_max_depth_data[0] < 0.0 {
                None
            } else {
                Some(tree_max_depth_data[0] as usize)
            };

            trees.push(DecisionTreeClassifier {
                tree: Some(tree_node),
                max_depth: tree_max_depth,
                n_features: None,
            });
        }

        Ok(Self {
            trees,
            n_estimators,
            max_depth,
            random_state,
            oob_indices: Vec::new(),
            x_train: None,
            y_train: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{Matrix, Vector};

    // ====================================================================
    // Helper: build a small linearly-separable classification dataset
    // ====================================================================
    fn classification_data() -> (Matrix<f32>, Vec<usize>) {
        // 8 samples, 2 features, 2 classes
        // Class 0: low feature values; Class 1: high feature values
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.1, // 0
                0.1, 0.0, // 0
                0.2, 0.2, // 0
                0.3, 0.1, // 0
                0.8, 0.9, // 1
                0.9, 0.8, // 1
                1.0, 1.0, // 1
                0.7, 0.9, // 1
            ],
        )
        .expect("classification data matrix");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];
        (x, y)
    }

    // ====================================================================
    // Helper: build a small regression dataset  y ≈ 2*x1 + 3*x2
    // ====================================================================
    fn regression_data() -> (Matrix<f32>, Vector<f32>) {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                1.0, 0.0, // 2
                0.0, 1.0, // 3
                1.0, 1.0, // 5
                2.0, 0.0, // 4
                0.0, 2.0, // 6
                2.0, 1.0, // 7
                1.0, 2.0, // 8
                3.0, 1.0, // 9
            ],
        )
        .expect("regression data matrix");
        let y = Vector::from_slice(&[2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0]);
        (x, y)
    }

    // ====================================================================
    // RandomForestClassifier — construction
    // ====================================================================

    #[test]
    fn test_classifier_new_sets_n_estimators() {
        let rf = RandomForestClassifier::new(5);
        assert_eq!(rf.n_estimators, 5);
        assert!(rf.trees.is_empty());
        assert!(rf.max_depth.is_none());
        assert!(rf.random_state.is_none());
    }

    #[test]
    fn test_classifier_with_max_depth() {
        let rf = RandomForestClassifier::new(3).with_max_depth(4);
        assert_eq!(rf.max_depth, Some(4));
    }

    #[test]
    fn test_classifier_with_random_state() {
        let rf = RandomForestClassifier::new(3).with_random_state(42);
        assert_eq!(rf.random_state, Some(42));
    }

    #[test]
    fn test_classifier_builder_chaining() {
        let rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(99);
        assert_eq!(rf.n_estimators, 10);
        assert_eq!(rf.max_depth, Some(3));
        assert_eq!(rf.random_state, Some(99));
    }

    // ====================================================================
    // RandomForestClassifier — fit / predict
    // ====================================================================

    #[test]
    fn test_classifier_fit_creates_correct_number_of_trees() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 5);
    }

    #[test]
    fn test_classifier_predict_returns_correct_length() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_classifier_predict_reasonable_accuracy() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);

        let correct = preds.iter().zip(y.iter()).filter(|(p, t)| p == t).count();
        // Should get at least 6 out of 8 correct on linearly separable data
        assert!(
            correct >= 6,
            "Expected >= 6 correct, got {correct} out of 8"
        );
    }

    #[test]
    fn test_classifier_score_returns_valid_range() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let score = rf.score(&x, &y);
        assert!(
            (0.0..=1.0).contains(&score),
            "Score {score} should be in [0.0, 1.0]"
        );
    }

    #[test]
    fn test_classifier_reproducibility_with_random_state() {
        let (x, y) = classification_data();
        let mut rf1 = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let preds1 = rf1.predict(&x);

        let mut rf2 = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let preds2 = rf2.predict(&x);

        assert_eq!(
            preds1, preds2,
            "Same random_state should yield same predictions"
        );
    }

    // ====================================================================
    // RandomForestClassifier — predict_proba
    // ====================================================================

    #[test]
    fn test_classifier_predict_proba_shape() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let proba = rf.predict_proba(&x);
        // 8 samples, 2 classes
        assert_eq!(proba.shape(), (8, 2));
    }

    #[test]
    fn test_classifier_predict_proba_sums_to_one() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let proba = rf.predict_proba(&x);

        for row in 0..proba.shape().0 {
            let sum: f32 = (0..proba.shape().1).map(|col| proba.get(row, col)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {row} probabilities sum to {sum}, expected 1.0"
            );
        }
    }

    #[test]
    fn test_classifier_predict_proba_values_in_range() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let proba = rf.predict_proba(&x);

        for row in 0..proba.shape().0 {
            for col in 0..proba.shape().1 {
                let val = proba.get(row, col);
                assert!(
                    (0.0..=1.0).contains(&val),
                    "Probability at ({row},{col}) = {val} out of range"
                );
            }
        }
    }

    // ====================================================================
    // RandomForestClassifier — OOB
    // ====================================================================

    #[test]
    fn test_classifier_oob_prediction_returns_some_after_fit() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob = rf.oob_prediction();
        assert!(oob.is_some(), "OOB prediction should be Some after fit");
        assert_eq!(oob.expect("checked above").len(), 8);
    }

    #[test]
    fn test_classifier_oob_prediction_returns_none_before_fit() {
        let rf = RandomForestClassifier::new(5);
        assert!(rf.oob_prediction().is_none());
    }

    #[test]
    fn test_classifier_oob_score_returns_some_after_fit() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
        let score_val = oob_score.expect("checked above");
        assert!(
            (0.0..=1.0).contains(&score_val),
            "OOB score {score_val} should be in [0, 1]"
        );
    }

    #[test]
    fn test_classifier_oob_score_returns_none_before_fit() {
        let rf = RandomForestClassifier::new(3);
        assert!(rf.oob_score().is_none());
    }

    // ====================================================================
    // RandomForestClassifier — feature importances
    // ====================================================================

    #[test]
    fn test_classifier_feature_importances_returns_some_after_fit() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let importances = rf.feature_importances();
        assert!(importances.is_some());
        let imp = importances.expect("checked above");
        assert_eq!(
            imp.len(),
            2,
            "Should have importance for each of 2 features"
        );
    }

    #[test]
    fn test_classifier_feature_importances_sum_to_one() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances().expect("should be Some after fit");
        let sum: f32 = imp.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Feature importances sum to {sum}, expected ~1.0"
        );
    }

    #[test]
    fn test_classifier_feature_importances_nonnegative() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances().expect("should be Some after fit");
        for (i, &val) in imp.iter().enumerate() {
            assert!(
                val >= 0.0,
                "Feature importance [{i}] = {val} should be >= 0"
            );
        }
    }

    #[test]
    fn test_classifier_feature_importances_returns_none_before_fit() {
        let rf = RandomForestClassifier::new(3);
        assert!(rf.feature_importances().is_none());
    }

    // ====================================================================
    // RandomForestClassifier — edge cases
    // ====================================================================

    #[test]
    fn test_classifier_single_tree() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(1)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 1);
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_classifier_single_feature() {
        // 1-dimensional feature space
        let x = Matrix::from_vec(6, 1, vec![0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
            .expect("single feature matrix");
        let y = vec![0, 0, 0, 1, 1, 1];
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 6);
    }

    #[test]
    fn test_classifier_no_max_depth() {
        // Fit without setting max_depth (exercises the `else` branch)
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_classifier_many_trees() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(50)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 50);
        let score = rf.score(&x, &y);
        // With 50 trees on linearly separable data, score should be high
        assert!(
            score >= 0.5,
            "Score with 50 trees should be decent, got {score}"
        );
    }

    // ====================================================================
    // RandomForestClassifier — save/load safetensors
    // ====================================================================

    #[test]
    fn test_classifier_save_unfitted_returns_error() {
        let rf = RandomForestClassifier::new(3);
        let result = rf.save_safetensors("/tmp/aprender_test_unfitted_rf.safetensors");
        assert!(result.is_err());
        assert!(result.expect_err("should be error").contains("unfitted"),);
    }

    #[test]
    fn test_classifier_save_and_load_roundtrip() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let path = "/tmp/aprender_test_rf_roundtrip.safetensors";
        rf.save_safetensors(path).expect("save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path).expect("load should succeed");
        assert_eq!(loaded.n_estimators, 3);
        assert_eq!(loaded.max_depth, Some(4));
        assert_eq!(loaded.random_state, Some(42));
        assert_eq!(loaded.trees.len(), 3);

        // Loaded model should produce same predictions
        let orig_preds = rf.predict(&x);
        let loaded_preds = loaded.predict(&x);
        assert_eq!(
            orig_preds, loaded_preds,
            "Loaded model predictions should match original"
        );

        // Cleanup
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_classifier_load_nonexistent_file_returns_error() {
        let result =
            RandomForestClassifier::load_safetensors("/tmp/aprender_no_such_file.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_classifier_save_load_no_max_depth() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(2).with_random_state(7);
        rf.fit(&x, &y).expect("fit should succeed");

        let path = "/tmp/aprender_test_rf_no_depth.safetensors";
        rf.save_safetensors(path).expect("save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path).expect("load should succeed");
        assert!(
            loaded.max_depth.is_none(),
            "max_depth should remain None after round-trip"
        );

        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn test_classifier_save_load_no_random_state() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(2).with_max_depth(2);
        rf.fit(&x, &y).expect("fit should succeed");

        let path = "/tmp/aprender_test_rf_no_rs.safetensors";
        rf.save_safetensors(path).expect("save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path).expect("load should succeed");
        assert!(
            loaded.random_state.is_none(),
            "random_state should remain None after round-trip"
        );

        let _ = std::fs::remove_file(path);
    }

    // ====================================================================
    // RandomForestRegressor — construction
    // ====================================================================

    #[test]
    fn test_regressor_new_sets_n_estimators() {
        let rf = RandomForestRegressor::new(7);
        assert_eq!(rf.n_estimators, 7);
        assert!(rf.trees.is_empty());
        assert!(rf.max_depth.is_none());
        assert!(rf.random_state.is_none());
    }

    #[test]
    fn test_regressor_default() {
        let rf = RandomForestRegressor::default();
        assert_eq!(rf.n_estimators, 10);
        assert!(rf.trees.is_empty());
    }

    #[test]
    fn test_regressor_with_max_depth() {
        let rf = RandomForestRegressor::new(3).with_max_depth(6);
        assert_eq!(rf.max_depth, Some(6));
    }

    #[test]
    fn test_regressor_with_random_state() {
        let rf = RandomForestRegressor::new(3).with_random_state(123);
        assert_eq!(rf.random_state, Some(123));
    }

    // ====================================================================
    // RandomForestRegressor — fit / predict
    // ====================================================================

    #[test]
    fn test_regressor_fit_creates_correct_number_of_trees() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 5);
    }

    #[test]
    fn test_regressor_predict_returns_correct_length() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_regressor_predictions_are_averaged() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);

        // Average predictions should be within a reasonable range of actual
        for i in 0..preds.len() {
            let pred = preds.as_slice()[i];
            let actual = y.as_slice()[i];
            assert!(
                (pred - actual).abs() < 6.0,
                "Prediction {pred} too far from actual {actual} at index {i}"
            );
        }
    }

    #[test]
    fn test_regressor_score_returns_valid_value() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let score = rf.score(&x, &y);
        // R² can be negative for very bad fits, but on training data it should be positive
        assert!(
            score > -1.0 && score <= 1.0,
            "R² score {score} seems unreasonable"
        );
    }

    #[test]
    fn test_regressor_reproducibility_with_random_state() {
        let (x, y) = regression_data();
        let mut rf1 = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let preds1 = rf1.predict(&x);

        let mut rf2 = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let preds2 = rf2.predict(&x);

        for i in 0..preds1.len() {
            assert!(
                (preds1.as_slice()[i] - preds2.as_slice()[i]).abs() < 1e-6,
                "Predictions differ at index {i}"
            );
        }
    }

    // ====================================================================
    // RandomForestRegressor — fit error paths
    // ====================================================================

    #[test]
    fn test_regressor_fit_mismatched_dimensions() {
        let x = Matrix::from_vec(4, 2, vec![1.0; 8]).expect("matrix creation");
        let y = Vector::from_slice(&[1.0, 2.0, 3.0]); // 3 != 4
        let mut rf = RandomForestRegressor::new(3);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_regressor_fit_zero_samples() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("empty matrix");
        let y = Vector::from_slice(&[]);
        let mut rf = RandomForestRegressor::new(3);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    // ====================================================================
    // RandomForestRegressor — OOB
    // ====================================================================

    #[test]
    fn test_regressor_oob_prediction_returns_some_after_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob = rf.oob_prediction();
        assert!(oob.is_some(), "OOB prediction should be Some after fit");
        assert_eq!(oob.expect("checked above").len(), 8);
    }

    #[test]
    fn test_regressor_oob_prediction_returns_none_before_fit() {
        let rf = RandomForestRegressor::new(5);
        assert!(rf.oob_prediction().is_none());
    }

    #[test]
    fn test_regressor_oob_score_returns_some_after_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
    }

    #[test]
    fn test_regressor_oob_score_returns_none_before_fit() {
        let rf = RandomForestRegressor::new(3);
        assert!(rf.oob_score().is_none());
    }

    // ====================================================================
    // RandomForestRegressor — feature importances
    // ====================================================================

    #[test]
    fn test_regressor_feature_importances_returns_some_after_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances();
        assert!(imp.is_some());
        assert_eq!(imp.expect("checked above").len(), 2);
    }

    #[test]
    fn test_regressor_feature_importances_sum_to_one() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances().expect("should be Some after fit");
        let sum: f32 = imp.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Feature importances sum to {sum}, expected ~1.0"
        );
    }

    #[test]
    fn test_regressor_feature_importances_nonnegative() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let imp = rf.feature_importances().expect("should be Some after fit");
        for (i, &val) in imp.iter().enumerate() {
            assert!(
                val >= 0.0,
                "Feature importance [{i}] = {val} should be >= 0"
            );
        }
    }

    #[test]
    fn test_regressor_feature_importances_returns_none_before_fit() {
        let rf = RandomForestRegressor::new(3);
        assert!(rf.feature_importances().is_none());
    }

    // ====================================================================
    // RandomForestRegressor — edge cases
    // ====================================================================

    #[test]
    fn test_regressor_single_tree() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(1)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 1);
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_regressor_no_max_depth() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 8);
    }

    #[test]
    fn test_regressor_single_feature() {
        let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("single feature matrix");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
        let mut rf = RandomForestRegressor::new(5)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let preds = rf.predict(&x);
        assert_eq!(preds.len(), 6);
    }

    #[test]
    #[should_panic(expected = "Cannot predict with an unfitted Random Forest")]
    fn test_regressor_predict_before_fit_panics() {
        let rf = RandomForestRegressor::new(3);
        let x = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix");
        let _ = rf.predict(&x);
    }

    #[test]
    fn test_regressor_stores_training_data() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert!(rf.x_train.is_some());
        assert!(rf.y_train.is_some());
        assert_eq!(rf.oob_indices.len(), 3);
    }

    #[test]
    fn test_classifier_stores_training_data() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert!(rf.x_train.is_some());
        assert!(rf.y_train.is_some());
        assert_eq!(rf.oob_indices.len(), 3);
    }

    #[test]
    fn test_classifier_predict_proba_without_y_train_defaults_to_2_classes() {
        // Build a classifier, fit it, then clear y_train to exercise the default path
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        rf.y_train = None;
        let proba = rf.predict_proba(&x);
        // Without y_train, n_classes defaults to 2
        assert_eq!(proba.shape().1, 2);
    }

    #[test]
    fn test_regressor_fit_overwrites_previous_fit() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        assert_eq!(rf.trees.len(), 3);

        // Fit again with different estimator count (by modifying n_estimators)
        rf.n_estimators = 5;
        rf.fit(&x, &y).expect("second fit should succeed");
        assert_eq!(rf.trees.len(), 5);
    }

    #[test]
    fn test_classifier_clone() {
        let (x, y) = classification_data();
        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let cloned = rf.clone();
        let orig_preds = rf.predict(&x);
        let cloned_preds = cloned.predict(&x);
        assert_eq!(orig_preds, cloned_preds);
    }

    #[test]
    fn test_regressor_clone() {
        let (x, y) = regression_data();
        let mut rf = RandomForestRegressor::new(3)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let cloned = rf.clone();
        let orig_preds = rf.predict(&x);
        let cloned_preds = cloned.predict(&x);
        for i in 0..orig_preds.len() {
            assert!(
                (orig_preds.as_slice()[i] - cloned_preds.as_slice()[i]).abs() < 1e-6,
                "Cloned predictions differ at index {i}"
            );
        }
    }
}
