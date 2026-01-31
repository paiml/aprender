//! Random Forest implementations for classification and regression.
//!
//! Ensemble methods using bootstrap sampling and majority voting.

use super::helpers::{
    bootstrap_sample, compute_regression_tree_feature_importances, compute_tree_feature_importances,
    flatten_tree_node, reconstruct_tree_node,
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
