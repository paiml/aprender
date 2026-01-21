//! Decision tree algorithms and ensemble methods.
//!
//! This module implements:
//! - CART (Classification and Regression Trees) using Gini impurity
//! - Random Forest ensemble classifier
//! - Gradient Boosting Machine (GBM) for sequential ensemble learning
//!
//! # Example
//!
//! ```rust,ignore
//! // Full API example (not yet implemented)
//! use aprender::prelude::*;
//! use aprender::tree::DecisionTreeClassifier;
//!
//! // Training data (simple 2D binary classification)
//! let x = Matrix::from_vec(4, 2, vec![
//!     0.0, 0.0,  // class 0
//!     0.0, 1.0,  // class 1
//!     1.0, 0.0,  // class 1
//!     1.0, 1.0,  // class 0
//! ]).expect("Matrix creation should succeed in tests");
//! let y = vec![0, 1, 1, 0];
//!
//! // Train decision tree
//! let mut tree = DecisionTreeClassifier::new()
//!     .with_max_depth(3);
//! tree.fit(&x, &y).expect("fit should succeed");
//!
//! // Make predictions
//! let predictions = tree.predict(&x);
//! ```

use crate::error::Result;
// Vector and Matrix imported via other modules
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

/// Internal node in a decision tree.
///
/// Contains a split condition (feature and threshold) and pointers to
/// left and right subtrees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Index of the feature to split on
    pub feature_idx: usize,
    /// Threshold value for the split
    pub threshold: f32,
    /// Left subtree (samples where feature <= threshold)
    pub left: Box<TreeNode>,
    /// Right subtree (samples where feature > threshold)
    pub right: Box<TreeNode>,
}

/// Leaf node in a decision tree.
///
/// Contains the predicted class label and number of training samples
/// that reached this leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Leaf {
    /// Predicted class label for this leaf
    pub class_label: usize,
    /// Number of training samples in this leaf
    pub n_samples: usize,
}

/// A node in a decision tree (either internal node or leaf).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    /// Internal decision node with split condition
    Node(Node),
    /// Leaf node with class prediction
    Leaf(Leaf),
}

impl TreeNode {
    /// Returns the depth of the tree rooted at this node.
    ///
    /// Leaf nodes have depth 0, internal nodes have depth 1 + max(left, right).
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            TreeNode::Leaf(_) => 0,
            TreeNode::Node(node) => 1 + node.left.depth().max(node.right.depth()),
        }
    }
}

// ========================================================================
// Regression Tree Structures (Issue #29)
// ========================================================================

/// Leaf node in a regression tree.
///
/// Contains the predicted value (mean of training samples) and number of
/// training samples that reached this leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionLeaf {
    /// Predicted value for this leaf (mean of y values)
    pub value: f32,
    /// Number of training samples in this leaf
    pub n_samples: usize,
}

/// Internal node in a regression tree.
///
/// Contains a split condition (feature and threshold) and pointers to
/// left and right subtrees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionNode {
    /// Index of the feature to split on
    pub feature_idx: usize,
    /// Threshold value for the split
    pub threshold: f32,
    /// Left subtree (samples where feature <= threshold)
    pub left: Box<RegressionTreeNode>,
    /// Right subtree (samples where feature > threshold)
    pub right: Box<RegressionTreeNode>,
}

/// A node in a regression tree (either internal node or leaf).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionTreeNode {
    /// Internal decision node with split condition
    Node(RegressionNode),
    /// Leaf node with value prediction
    Leaf(RegressionLeaf),
}

impl RegressionTreeNode {
    /// Returns the depth of the tree rooted at this node.
    ///
    /// Leaf nodes have depth 0, internal nodes have depth 1 + max(left, right).
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            RegressionTreeNode::Leaf(_) => 0,
            RegressionTreeNode::Node(node) => 1 + node.left.depth().max(node.right.depth()),
        }
    }
}

/// Decision tree regressor using the CART algorithm.
///
/// Uses Mean Squared Error (MSE) for splitting criterion and builds trees recursively.
/// Leaf nodes predict the mean of target values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeRegressor {
    tree: Option<RegressionTreeNode>,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
}

impl DecisionTreeRegressor {
    /// Creates a new decision tree regressor with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: None,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
        }
    }

    /// Sets the maximum depth of the tree.
    ///
    /// # Arguments
    ///
    /// * `depth` - Maximum depth (root has depth 0)
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets the minimum number of samples required to split an internal node.
    ///
    /// # Arguments
    ///
    /// * `min_samples` - Minimum samples to split (must be >= 2)
    #[must_use]
    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples.max(2);
        self
    }

    /// Sets the minimum number of samples required to be at a leaf node.
    ///
    /// # Arguments
    ///
    /// * `min_samples` - Minimum samples per leaf (must be >= 1)
    #[must_use]
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples.max(1);
        self
    }

    /// Fits the decision tree to training data.
    ///
    /// # Arguments
    ///
    /// * `x` - Training features (`n_samples` × `n_features`)
    /// * `y` - Training target values (`n_samples` continuous values)
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn fit(
        &mut self,
        x: &crate::primitives::Matrix<f32>,
        y: &crate::primitives::Vector<f32>,
    ) -> Result<()> {
        let (n_rows, _n_cols) = x.shape();
        if n_rows != y.len() {
            return Err("Number of samples in X and y must match".into());
        }
        if n_rows == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        self.tree = Some(build_regression_tree(
            x,
            y,
            0,
            self.max_depth,
            self.min_samples_split,
            self.min_samples_leaf,
        ));
        Ok(())
    }

    /// Predicts target values for samples.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (`n_samples` × `n_features`)
    ///
    /// # Returns
    ///
    /// Vector of predicted values
    ///
    /// # Panics
    ///
    /// Panics if called before `fit()`
    #[must_use]
    pub fn predict(&self, x: &crate::primitives::Matrix<f32>) -> crate::primitives::Vector<f32> {
        let (n_samples, n_features) = x.shape();
        let mut predictions = Vec::with_capacity(n_samples);

        for row in 0..n_samples {
            let mut sample = Vec::with_capacity(n_features);
            for col in 0..n_features {
                sample.push(x.get(row, col));
            }
            predictions.push(self.predict_one(&sample));
        }

        crate::primitives::Vector::from_vec(predictions)
    }

    /// Predicts the value for a single sample.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature vector for one sample
    ///
    /// # Returns
    ///
    /// Predicted value
    fn predict_one(&self, x: &[f32]) -> f32 {
        let tree = self.tree.as_ref().expect("Model not fitted");

        let mut node = tree;
        loop {
            match node {
                RegressionTreeNode::Leaf(leaf) => return leaf.value,
                RegressionTreeNode::Node(internal) => {
                    if x[internal.feature_idx] <= internal.threshold {
                        node = &internal.left;
                    } else {
                        node = &internal.right;
                    }
                }
            }
        }
    }

    /// Computes the R² score on test data.
    ///
    /// # Arguments
    ///
    /// * `x` - Test features (`n_samples` × `n_features`)
    /// * `y` - True target values (`n_samples`)
    ///
    /// # Returns
    ///
    /// R² coefficient of determination
    #[must_use]
    pub fn score(
        &self,
        x: &crate::primitives::Matrix<f32>,
        y: &crate::primitives::Vector<f32>,
    ) -> f32 {
        let predictions = self.predict(x);
        crate::metrics::r_squared(y, &predictions)
    }
}

impl Default for DecisionTreeRegressor {
    fn default() -> Self {
        Self::new()
    }
}

// ========================================================================
// End Regression Tree Structures
// ========================================================================

/// Random Forest Regressor.
///
/// Ensemble of decision tree regressors trained on bootstrap samples.
/// Predictions are averaged across all trees to reduce variance and overfitting.
///
/// # Examples
///
/// ```
/// use aprender::tree::RandomForestRegressor;
/// use aprender::primitives::{Matrix, Vector};
///
/// let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("Matrix creation should succeed in tests");
/// let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
///
/// let mut rf = RandomForestRegressor::new(10).with_max_depth(5);
/// rf.fit(&x, &y).expect("fit should succeed");
/// let predictions = rf.predict(&x);
/// let r2 = rf.score(&x, &y);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestRegressor {
    trees: Vec<DecisionTreeRegressor>,
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    /// OOB sample indices for each tree (samples NOT in bootstrap sample)
    oob_indices: Vec<Vec<usize>>,
    /// Training features (stored for OOB evaluation)
    x_train: Option<crate::primitives::Matrix<f32>>,
    /// Training targets (stored for OOB evaluation)
    y_train: Option<crate::primitives::Vector<f32>>,
}

impl RandomForestRegressor {
    /// Creates a new Random Forest regressor.
    ///
    /// # Arguments
    ///
    /// * `n_estimators` - Number of trees in the forest
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
            let bootstrap_set: std::collections::HashSet<usize> =
                bootstrap_indices.iter().copied().collect();
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
    ///
    /// # Arguments
    ///
    /// * `x` - Test features (`n_samples` × `n_features`)
    /// * `y` - True target values (`n_samples`)
    ///
    /// # Returns
    ///
    /// R² coefficient of determination
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
    ///
    /// For each training sample, predictions are made using only the trees
    /// where that sample was NOT in the bootstrap sample (out-of-bag).
    ///
    /// # Returns
    ///
    /// `Some(Vector<f32>)` if the model has been fitted, `None` otherwise.
    /// The vector has the same length as the training data.
    #[must_use]
    pub fn oob_prediction(&self) -> Option<crate::primitives::Vector<f32>> {
        // Return None if model not fitted
        if self.trees.is_empty() || self.y_train.is_none() || self.x_train.is_none() {
            return None;
        }

        let x_train = self
            .x_train
            .as_ref()
            .expect("x_train should be stored after fit");
        let y_train = self
            .y_train
            .as_ref()
            .expect("y_train should be stored after fit");
        let n_samples = y_train.len();
        let n_features = x_train.shape().1;

        // Track predictions and counts for each sample from OOB trees
        let mut oob_predictions = vec![0.0; n_samples];
        let mut oob_counts = vec![0; n_samples];

        // For each tree, make predictions on its OOB samples
        for (tree_idx, oob_indices) in self.oob_indices.iter().enumerate() {
            let tree = &self.trees[tree_idx];

            // For each OOB sample for this tree
            for &sample_idx in oob_indices {
                // Extract single sample as a 1×n_features matrix
                let mut sample_data = Vec::with_capacity(n_features);
                for j in 0..n_features {
                    sample_data.push(x_train.get(sample_idx, j));
                }

                let sample_matrix =
                    crate::primitives::Matrix::from_vec(1, n_features, sample_data).ok()?;

                // Get prediction from this tree
                let tree_predictions = tree.predict(&sample_matrix);
                let predicted_value = tree_predictions.as_slice()[0];

                // Accumulate prediction
                oob_predictions[sample_idx] += predicted_value;
                oob_counts[sample_idx] += 1;
            }
        }

        // Average predictions for each sample
        for (i, count) in oob_counts.iter().enumerate() {
            if *count > 0 {
                oob_predictions[i] /= *count as f32;
            }
            // If count is 0, sample was never OOB, keep 0.0 as default
        }

        Some(crate::primitives::Vector::from_slice(&oob_predictions))
    }

    /// Returns Out-of-Bag (OOB) R² score.
    ///
    /// Computes R² score using OOB predictions. This provides an unbiased
    /// estimate of the model's performance without needing a validation set.
    ///
    /// # Returns
    ///
    /// `Some(f32)` with R² score if model has been fitted, `None` otherwise.
    #[must_use]
    pub fn oob_score(&self) -> Option<f32> {
        let oob_preds = self.oob_prediction()?;
        let y_train = self.y_train.as_ref()?;

        Some(crate::metrics::r_squared(y_train, &oob_preds))
    }

    /// Returns feature importances based on mean decrease in variance.
    ///
    /// Feature importance is calculated as the total decrease in node variance
    /// (weighted by the number of samples) averaged over all trees in the forest.
    ///
    /// # Returns
    ///
    /// `Some(Vec<f32>)` with importance for each feature (normalized to sum to 1.0)
    /// if model has been fitted, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut rf = RandomForestRegressor::new(50);
    /// rf.fit(&x_train, &y_train).expect("fit should succeed");
    ///
    /// if let Some(importances) = rf.feature_importances() {
    ///     for (i, &importance) in importances.iter().enumerate() {
    ///         println!("Feature {}: {:.4}", i, importance);
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn feature_importances(&self) -> Option<Vec<f32>> {
        if self.trees.is_empty() || self.x_train.is_none() {
            return None;
        }

        let n_features = self
            .x_train
            .as_ref()
            .expect("x_train should be stored after fit")
            .shape()
            .1;
        let mut total_importances = vec![0.0; n_features];

        // Aggregate importances from all trees
        for tree in &self.trees {
            if let Some(tree_node) = &tree.tree {
                let mut tree_importances = vec![0.0; n_features];
                compute_regression_tree_feature_importances(tree_node, &mut tree_importances);

                // Add to total
                for (i, &importance) in tree_importances.iter().enumerate() {
                    total_importances[i] += importance;
                }
            }
        }

        // Normalize: divide by number of trees and then normalize to sum to 1.0
        let n_trees = self.trees.len() as f32;
        for importance in &mut total_importances {
            *importance /= n_trees;
        }

        // Normalize to sum to 1.0
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
        Self::new(10) // Default: 10 trees
    }
}

// ========================================================================
// End Random Forest Regression
// ========================================================================

/// Decision tree classifier using the CART algorithm.
///
/// Uses Gini impurity for splitting criterion and builds trees recursively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    tree: Option<TreeNode>,
    max_depth: Option<usize>,
    /// Number of features the model was trained on (for validation)
    #[serde(default)]
    n_features: Option<usize>,
}

impl DecisionTreeClassifier {
    /// Creates a new decision tree classifier with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tree: None,
            max_depth: None,
            n_features: None,
        }
    }

    /// Sets the maximum depth of the tree.
    ///
    /// # Arguments
    ///
    /// * `depth` - Maximum depth (root has depth 0)
    #[must_use]
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Fits the decision tree to training data.
    ///
    /// # Arguments
    ///
    /// * `x` - Training features (`n_samples` × `n_features`)
    /// * `y` - Training labels (`n_samples` class indices)
    ///
    /// # Errors
    ///
    /// Returns an error if the data is invalid.
    pub fn fit(&mut self, x: &crate::primitives::Matrix<f32>, y: &[usize]) -> Result<()> {
        let (n_rows, n_cols) = x.shape();
        if n_rows != y.len() {
            return Err("Number of samples in X and y must match".into());
        }
        if n_rows == 0 {
            return Err("Cannot fit with zero samples".into());
        }

        self.n_features = Some(n_cols);
        self.tree = Some(build_tree(x, y, 0, self.max_depth));
        Ok(())
    }

    /// Predicts class labels for samples.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature matrix (`n_samples` × `n_features`)
    ///
    /// # Returns
    ///
    /// Vector of predicted class labels
    ///
    /// # Panics
    ///
    /// Panics if called before `fit()` or if feature count doesn't match training data
    #[must_use]
    pub fn predict(&self, x: &crate::primitives::Matrix<f32>) -> Vec<usize> {
        let (n_samples, n_features) = x.shape();

        // Validate feature count matches what we trained on
        if let Some(expected) = self.n_features {
            assert!(
                n_features >= expected,
                "Feature count mismatch: model was trained with {expected} features but input has {n_features} features"
            );
        }

        let mut predictions = Vec::with_capacity(n_samples);

        for row in 0..n_samples {
            let mut sample = Vec::with_capacity(n_features);
            for col in 0..n_features {
                sample.push(x.get(row, col));
            }
            predictions.push(self.predict_one(&sample));
        }

        predictions
    }

    /// Predicts the class label for a single sample.
    ///
    /// # Arguments
    ///
    /// * `x` - Feature vector for one sample
    ///
    /// # Returns
    ///
    /// Predicted class label
    fn predict_one(&self, x: &[f32]) -> usize {
        let tree = self.tree.as_ref().expect("Model not fitted yet");

        let mut node = tree;
        loop {
            match node {
                TreeNode::Leaf(leaf) => return leaf.class_label,
                TreeNode::Node(internal) => {
                    if x[internal.feature_idx] <= internal.threshold {
                        node = &internal.left;
                    } else {
                        node = &internal.right;
                    }
                }
            }
        }
    }

    /// Computes the accuracy score on test data.
    ///
    /// # Arguments
    ///
    /// * `x` - Test features (`n_samples` × `n_features`)
    /// * `y` - True labels (`n_samples`)
    ///
    /// # Returns
    ///
    /// Accuracy (fraction of correct predictions)
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

    /// Saves the model to a binary file using bincode.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or file writing fails.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {e}"))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {e}"))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {e}"))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {e}"))?;
        Ok(model)
    }

    /// Saves the model to `SafeTensors` format.
    ///
    /// Serializes the tree structure as flat arrays using pre-order traversal.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Model is not fitted
    /// - Serialization fails
    /// - File writing fails
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Verify model is fitted
        let tree = self
            .tree
            .as_ref()
            .ok_or("Cannot save unfitted model. Call fit() first.")?;

        // Flatten tree structure into arrays via pre-order traversal
        let mut node_features = Vec::new();
        let mut node_thresholds = Vec::new();
        let mut node_classes = Vec::new();
        let mut node_samples = Vec::new();
        let mut node_left_child = Vec::new();
        let mut node_right_child = Vec::new();

        flatten_tree_node(
            tree,
            &mut node_features,
            &mut node_thresholds,
            &mut node_classes,
            &mut node_samples,
            &mut node_left_child,
            &mut node_right_child,
        );

        // Prepare tensors
        let mut tensors = BTreeMap::new();

        tensors.insert(
            "node_features".to_string(),
            (node_features.clone(), vec![node_features.len()]),
        );
        tensors.insert(
            "node_thresholds".to_string(),
            (node_thresholds.clone(), vec![node_thresholds.len()]),
        );
        tensors.insert(
            "node_classes".to_string(),
            (node_classes.clone(), vec![node_classes.len()]),
        );
        tensors.insert(
            "node_samples".to_string(),
            (node_samples.clone(), vec![node_samples.len()]),
        );
        tensors.insert(
            "node_left_child".to_string(),
            (node_left_child.clone(), vec![node_left_child.len()]),
        );
        tensors.insert(
            "node_right_child".to_string(),
            (node_right_child.clone(), vec![node_right_child.len()]),
        );

        // Max depth as tensor (-1 for None)
        let max_depth_val = self.max_depth.map_or(-1.0, |d| d as f32);
        tensors.insert("max_depth".to_string(), (vec![max_depth_val], vec![1]));

        // Save to SafeTensors format
        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a model from `SafeTensors` format.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - `SafeTensors` format is invalid
    /// - Required tensors are missing
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Extract all tensors
        let node_features = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_features")
                .ok_or("Missing 'node_features' tensor")?,
        )?;
        let node_thresholds = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_thresholds")
                .ok_or("Missing 'node_thresholds' tensor")?,
        )?;
        let node_classes = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_classes")
                .ok_or("Missing 'node_classes' tensor")?,
        )?;
        let node_samples = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_samples")
                .ok_or("Missing 'node_samples' tensor")?,
        )?;
        let node_left_child = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_left_child")
                .ok_or("Missing 'node_left_child' tensor")?,
        )?;
        let node_right_child = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("node_right_child")
                .ok_or("Missing 'node_right_child' tensor")?,
        )?;
        let max_depth_data = safetensors::extract_tensor(
            &raw_data,
            metadata
                .get("max_depth")
                .ok_or("Missing 'max_depth' tensor")?,
        )?;

        // Validate tensor sizes match
        let n_nodes = node_features.len();
        if node_thresholds.len() != n_nodes
            || node_classes.len() != n_nodes
            || node_samples.len() != n_nodes
            || node_left_child.len() != n_nodes
            || node_right_child.len() != n_nodes
        {
            return Err("Inconsistent node array sizes in SafeTensors file".to_string());
        }

        if n_nodes == 0 {
            return Err("Empty tree in SafeTensors file".to_string());
        }

        // Reconstruct tree from flat arrays
        let tree = Some(reconstruct_tree_node(
            0,
            &node_features,
            &node_thresholds,
            &node_classes,
            &node_samples,
            &node_left_child,
            &node_right_child,
        ));

        // Parse max_depth
        let max_depth = if max_depth_data[0] < 0.0 {
            None
        } else {
            Some(max_depth_data[0] as usize)
        };

        Ok(Self {
            tree,
            max_depth,
            n_features: None,
        })
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for tree serialization

/// Flattens a tree structure into parallel arrays via pre-order traversal.
///
/// Returns the index of the root node.
fn flatten_tree_node(
    node: &TreeNode,
    features: &mut Vec<f32>,
    thresholds: &mut Vec<f32>,
    classes: &mut Vec<f32>,
    samples: &mut Vec<f32>,
    left_children: &mut Vec<f32>,
    right_children: &mut Vec<f32>,
) -> usize {
    let current_idx = features.len();

    match node {
        TreeNode::Leaf(leaf) => {
            // Leaf node: feature = -1, class and samples store data
            features.push(-1.0);
            thresholds.push(0.0);
            classes.push(leaf.class_label as f32);
            samples.push(leaf.n_samples as f32);
            left_children.push(-1.0);
            right_children.push(-1.0);
        }
        TreeNode::Node(internal) => {
            // Reserve space for this node (will update child indices later)
            features.push(internal.feature_idx as f32);
            thresholds.push(internal.threshold);
            classes.push(0.0); // Not used for internal nodes
            samples.push(0.0); // Not used for internal nodes
            left_children.push(0.0); // Placeholder
            right_children.push(0.0); // Placeholder

            // Recursively flatten left subtree
            let left_idx = flatten_tree_node(
                &internal.left,
                features,
                thresholds,
                classes,
                samples,
                left_children,
                right_children,
            );

            // Recursively flatten right subtree
            let right_idx = flatten_tree_node(
                &internal.right,
                features,
                thresholds,
                classes,
                samples,
                left_children,
                right_children,
            );

            // Update child indices
            left_children[current_idx] = left_idx as f32;
            right_children[current_idx] = right_idx as f32;
        }
    }

    current_idx
}

/// Reconstructs a tree from parallel arrays.
fn reconstruct_tree_node(
    idx: usize,
    features: &[f32],
    thresholds: &[f32],
    classes: &[f32],
    samples: &[f32],
    left_children: &[f32],
    right_children: &[f32],
) -> TreeNode {
    if features[idx] < 0.0 {
        // Leaf node
        TreeNode::Leaf(Leaf {
            class_label: classes[idx] as usize,
            n_samples: samples[idx] as usize,
        })
    } else {
        // Internal node
        let left_idx = left_children[idx] as usize;
        let right_idx = right_children[idx] as usize;

        TreeNode::Node(Node {
            feature_idx: features[idx] as usize,
            threshold: thresholds[idx],
            left: Box::new(reconstruct_tree_node(
                left_idx,
                features,
                thresholds,
                classes,
                samples,
                left_children,
                right_children,
            )),
            right: Box::new(reconstruct_tree_node(
                right_idx,
                features,
                thresholds,
                classes,
                samples,
                left_children,
                right_children,
            )),
        })
    }
}

// Helper functions for tree building

#[allow(dead_code)] // Will be used in split-finding implementation
/// Calculate Gini impurity for a set of labels.
///
/// Gini impurity measures the probability of incorrectly classifying a randomly
/// chosen element if it were labeled according to the distribution of labels.
///
/// Formula: Gini = 1 - `Σ(p_i²)` where `p_i` is the proportion of class i
///
/// # Arguments
///
/// * `labels` - Slice of class labels
///
/// # Returns
///
/// Gini impurity value between 0.0 (pure) and 1.0 (maximum impurity)
fn gini_impurity(labels: &[usize]) -> f32 {
    if labels.is_empty() {
        return 0.0;
    }

    // Count occurrences of each class
    let mut counts = std::collections::HashMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }

    let n = labels.len() as f32;
    let mut gini = 1.0;

    // Gini = 1 - Σ(p_i²)
    for count in counts.values() {
        let p = *count as f32 / n;
        gini -= p * p;
    }

    gini
}

#[allow(dead_code)] // Will be used in split-finding implementation
/// Calculate weighted Gini impurity for a split.
///
/// # Arguments
///
/// * `left_labels` - Labels in left partition
/// * `right_labels` - Labels in right partition
///
/// # Returns
///
/// Weighted Gini impurity for the split
fn gini_split(left_labels: &[usize], right_labels: &[usize]) -> f32 {
    let n_left = left_labels.len() as f32;
    let n_right = right_labels.len() as f32;
    let n_total = n_left + n_right;

    if n_total == 0.0 {
        return 0.0;
    }

    let weight_left = n_left / n_total;
    let weight_right = n_right / n_total;

    weight_left * gini_impurity(left_labels) + weight_right * gini_impurity(right_labels)
}

/// Get sorted unique values from feature data.
///
/// Returns unique values in sorted order, filtering out values closer than 1e-10.
#[allow(dead_code)]
fn get_sorted_unique_values(x: &[f32]) -> Vec<f32> {
    let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .expect("f32 values should be comparable")
    });

    let mut unique_values = Vec::new();
    let mut prev_val = x[sorted_indices[0]];
    unique_values.push(prev_val);

    for &idx in &sorted_indices[1..] {
        if (x[idx] - prev_val).abs() > 1e-10 {
            unique_values.push(x[idx]);
            prev_val = x[idx];
        }
    }

    unique_values
}

/// Split labels into left and right partitions based on threshold.
///
/// # Arguments
///
/// * `x` - Feature values
/// * `y` - Labels
/// * `threshold` - Split threshold
///
/// # Returns
///
/// `(left_labels, right_labels)` tuple, or None if split is invalid
#[allow(dead_code)]
fn split_labels_by_threshold(
    x: &[f32],
    y: &[usize],
    threshold: f32,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut left_labels = Vec::new();
    let mut right_labels = Vec::new();

    for (idx, &val) in x.iter().enumerate() {
        if val <= threshold {
            left_labels.push(y[idx]);
        } else {
            right_labels.push(y[idx]);
        }
    }

    if left_labels.is_empty() || right_labels.is_empty() {
        None
    } else {
        Some((left_labels, right_labels))
    }
}

/// Calculate information gain for a potential split.
///
/// # Arguments
///
/// * `current_impurity` - Gini impurity before split
/// * `left_labels` - Labels in left partition
/// * `right_labels` - Labels in right partition
///
/// # Returns
///
/// Information gain from the split
#[allow(dead_code)]
fn calculate_information_gain(
    current_impurity: f32,
    left_labels: &[usize],
    right_labels: &[usize],
) -> f32 {
    let split_impurity = gini_split(left_labels, right_labels);
    current_impurity - split_impurity
}

/// Find the best split for a given feature.
///
/// Tries all possible threshold values (midpoints between consecutive unique values)
/// and returns the threshold with highest information gain.
///
/// # Arguments
///
/// * `x` - Feature values (`n_samples`)
/// * `y` - Labels (`n_samples`)
///
/// # Returns
///
/// `Some((threshold, gain))` if a valid split exists, `None` otherwise
#[allow(dead_code)] // Will be used in tree building implementation
fn find_best_split_for_feature(x: &[f32], y: &[usize]) -> Option<(f32, f32)> {
    if x.len() < 2 {
        return None;
    }

    let unique_values = get_sorted_unique_values(x);
    if unique_values.len() < 2 {
        return None;
    }

    let current_impurity = gini_impurity(y);
    let mut best_gain = 0.0;
    let mut best_threshold = 0.0;

    // Try each midpoint as threshold
    for i in 0..unique_values.len() - 1 {
        let threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;

        if let Some((left_labels, right_labels)) = split_labels_by_threshold(x, y, threshold) {
            let gain = calculate_information_gain(current_impurity, &left_labels, &right_labels);

            if gain > best_gain {
                best_gain = gain;
                best_threshold = threshold;
            }
        }
    }

    if best_gain > 0.0 {
        Some((best_threshold, best_gain))
    } else {
        None
    }
}

/// Find the best split across all features.
///
/// # Arguments
///
/// * `x_matrix` - Training data (`n_samples` × `n_features`)
/// * `y` - Labels (`n_samples`)
///
/// # Returns
///
/// `Some((feature_idx, threshold, gain))` if a valid split exists, `None` otherwise
#[allow(dead_code)] // Will be used in tree building implementation
fn find_best_split(
    x_matrix: &crate::primitives::Matrix<f32>,
    y: &[usize],
) -> Option<(usize, f32, f32)> {
    let (n_samples, n_features) = x_matrix.shape();

    if n_samples < 2 {
        return None;
    }

    let mut best_gain = 0.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    // Try each feature
    for feature_idx in 0..n_features {
        // Extract column for this feature
        let mut feature_values = Vec::with_capacity(n_samples);
        for row in 0..n_samples {
            feature_values.push(x_matrix.get(row, feature_idx));
        }

        // Find best split for this feature
        if let Some((threshold, gain)) = find_best_split_for_feature(&feature_values, y) {
            if gain > best_gain {
                best_gain = gain;
                best_feature = feature_idx;
                best_threshold = threshold;
            }
        }
    }

    if best_gain > 0.0 {
        Some((best_feature, best_threshold, best_gain))
    } else {
        None
    }
}

/// Find the majority class from a set of labels.
///
/// Returns the most frequent class label. Ties are broken arbitrarily.
///
/// # Arguments
///
/// * `labels` - Slice of class labels
///
/// # Returns
///
/// Most frequent class label
#[allow(dead_code)] // Will be used in tree building
fn majority_class(labels: &[usize]) -> usize {
    let mut counts = std::collections::HashMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0) += 1;
    }
    *counts
        .iter()
        .max_by_key(|(_, &count)| count)
        .expect("at least one label should exist")
        .0
}

/// Split data into subsets based on indices.
///
/// Creates a matrix and label vector containing only the rows at the given indices.
#[allow(dead_code)]
fn split_data_by_indices(
    x: &crate::primitives::Matrix<f32>,
    y: &[usize],
    indices: &[usize],
) -> (crate::primitives::Matrix<f32>, Vec<usize>) {
    let n_cols = x.shape().1;
    let mut data = Vec::with_capacity(indices.len() * n_cols);
    let mut labels = Vec::with_capacity(indices.len());

    for &idx in indices {
        for col in 0..n_cols {
            data.push(x.get(idx, col));
        }
        labels.push(y[idx]);
    }

    let matrix = crate::primitives::Matrix::from_vec(indices.len(), n_cols, data)
        .expect("matrix creation should succeed with valid indices");
    (matrix, labels)
}

/// Check if tree building should stop at this node.
///
/// Returns a leaf node if stopping criteria are met, None otherwise.
#[allow(dead_code)]
fn check_stopping_criteria(
    y: &[usize],
    depth: usize,
    max_depth: Option<usize>,
) -> Option<TreeNode> {
    let n_samples = y.len();

    // Criterion 1: All same label (pure node)
    let unique_labels: std::collections::HashSet<_> = y.iter().collect();
    if unique_labels.len() == 1 {
        return Some(TreeNode::Leaf(Leaf {
            class_label: y[0],
            n_samples,
        }));
    }

    // Criterion 2: Max depth reached
    if let Some(max_d) = max_depth {
        if depth >= max_d {
            return Some(TreeNode::Leaf(Leaf {
                class_label: majority_class(y),
                n_samples,
            }));
        }
    }

    None
}

/// Split data indices based on feature threshold.
///
/// # Returns
///
/// `Some((left_indices, right_indices))` if split is valid, None otherwise
#[allow(dead_code)]
fn split_indices_by_threshold(
    x: &crate::primitives::Matrix<f32>,
    feature_idx: usize,
    threshold: f32,
    n_samples: usize,
) -> Option<(Vec<usize>, Vec<usize>)> {
    let mut left_indices = Vec::new();
    let mut right_indices = Vec::new();

    for row in 0..n_samples {
        if x.get(row, feature_idx) <= threshold {
            left_indices.push(row);
        } else {
            right_indices.push(row);
        }
    }

    if left_indices.is_empty() || right_indices.is_empty() {
        None
    } else {
        Some((left_indices, right_indices))
    }
}

/// Build a decision tree recursively.
///
/// # Arguments
///
/// * `x` - Training data (`n_samples` × `n_features`)
/// * `y` - Labels (`n_samples`)
/// * `depth` - Current depth in tree
/// * `max_depth` - Maximum allowed depth (None = unlimited)
///
/// # Returns
///
/// Root node of the built tree
#[allow(dead_code)] // Will be used in fit() implementation
fn build_tree(
    x: &crate::primitives::Matrix<f32>,
    y: &[usize],
    depth: usize,
    max_depth: Option<usize>,
) -> TreeNode {
    let n_samples = y.len();

    // Check stopping criteria
    if let Some(leaf) = check_stopping_criteria(y, depth, max_depth) {
        return leaf;
    }

    // Try to find best split
    let Some((feature_idx, threshold, _gain)) = find_best_split(x, y) else {
        return TreeNode::Leaf(Leaf {
            class_label: majority_class(y),
            n_samples,
        });
    };

    // Split data based on threshold
    let Some((left_indices, right_indices)) =
        split_indices_by_threshold(x, feature_idx, threshold, n_samples)
    else {
        return TreeNode::Leaf(Leaf {
            class_label: majority_class(y),
            n_samples,
        });
    };

    // Create left and right datasets
    let (left_matrix, left_labels) = split_data_by_indices(x, y, &left_indices);
    let (right_matrix, right_labels) = split_data_by_indices(x, y, &right_indices);

    // Recursively build subtrees
    let left_child = build_tree(&left_matrix, &left_labels, depth + 1, max_depth);
    let right_child = build_tree(&right_matrix, &right_labels, depth + 1, max_depth);

    TreeNode::Node(Node {
        feature_idx,
        threshold,
        left: Box::new(left_child),
        right: Box::new(right_child),
    })
}

// ========================================================================
// Regression Tree Building Functions (Issue #29)
// ========================================================================

/// Compute the mean of a vector.
fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Compute the variance of target values.
fn variance_f32(y: &[f32]) -> f32 {
    if y.len() <= 1 {
        return 0.0;
    }

    let mean = mean_f32(y);
    let sum_squared_diff: f32 = y.iter().map(|&val| (val - mean).powi(2)).sum();
    sum_squared_diff / y.len() as f32
}

/// Compute Mean Squared Error for a split.
fn compute_mse(y_left: &[f32], y_right: &[f32]) -> f32 {
    let n_left = y_left.len() as f32;
    let n_right = y_right.len() as f32;
    let n_total = n_left + n_right;

    if n_total == 0.0 {
        return 0.0;
    }

    let var_left = variance_f32(y_left);
    let var_right = variance_f32(y_right);

    (n_left / n_total) * var_left + (n_right / n_total) * var_right
}

/// Get unique sorted feature values for splitting.
fn get_unique_feature_values(
    x: &crate::primitives::Matrix<f32>,
    feature_idx: usize,
    n_samples: usize,
) -> Vec<f32> {
    let mut values: Vec<f32> = (0..n_samples).map(|i| x.get(i, feature_idx)).collect();
    values.sort_by(|a, b| a.partial_cmp(b).expect("f32 values should be comparable"));
    values.dedup();
    values
}

/// Split y values by a threshold on a feature.
fn split_by_threshold(
    x: &crate::primitives::Matrix<f32>,
    y: &[f32],
    feature_idx: usize,
    threshold: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut y_left = Vec::new();
    let mut y_right = Vec::new();

    for (row, &y_val) in y.iter().enumerate() {
        if x.get(row, feature_idx) <= threshold {
            y_left.push(y_val);
        } else {
            y_right.push(y_val);
        }
    }
    (y_left, y_right)
}

/// Evaluate a single split and return gain if valid.
fn evaluate_split_gain(y_left: &[f32], y_right: &[f32], current_variance: f32) -> Option<f32> {
    if y_left.is_empty() || y_right.is_empty() {
        return None;
    }
    let split_mse = compute_mse(y_left, y_right);
    let gain = current_variance - split_mse;
    (gain > 0.0).then_some(gain)
}

/// Find the best split for a single feature.
fn find_best_regression_split_for_feature(
    x: &crate::primitives::Matrix<f32>,
    y: &[f32],
    feature_idx: usize,
    n_samples: usize,
    current_variance: f32,
) -> Option<(f32, f32)> {
    let feature_values = get_unique_feature_values(x, feature_idx, n_samples);
    let mut best_threshold = 0.0;
    let mut best_gain = 0.0;

    for i in 0..feature_values.len().saturating_sub(1) {
        let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;
        let (y_left, y_right) = split_by_threshold(x, y, feature_idx, threshold);

        if let Some(gain) = evaluate_split_gain(&y_left, &y_right, current_variance) {
            if gain > best_gain {
                best_gain = gain;
                best_threshold = threshold;
            }
        }
    }

    (best_gain > 0.0).then_some((best_threshold, best_gain))
}

/// Find the best split for regression using MSE criterion.
///
/// Returns (`feature_idx`, threshold, `mse_reduction`) if a valid split exists.
fn find_best_regression_split(
    x: &crate::primitives::Matrix<f32>,
    y: &[f32],
) -> Option<(usize, f32, f32)> {
    let (n_samples, n_features) = x.shape();

    if n_samples < 2 {
        return None;
    }

    let current_variance = variance_f32(y);
    let mut best_gain = 0.0;
    let mut best_feature = 0;
    let mut best_threshold = 0.0;

    for feature_idx in 0..n_features {
        if let Some((threshold, gain)) =
            find_best_regression_split_for_feature(x, y, feature_idx, n_samples, current_variance)
        {
            if gain > best_gain {
                best_gain = gain;
                best_feature = feature_idx;
                best_threshold = threshold;
            }
        }
    }

    (best_gain > 0.0).then_some((best_feature, best_threshold, best_gain))
}

/// Split regression data by indices.
fn split_regression_data_by_indices(
    x: &crate::primitives::Matrix<f32>,
    y: &[f32],
    indices: &[usize],
) -> (crate::primitives::Matrix<f32>, Vec<f32>) {
    let (_n_samples, n_features) = x.shape();
    let n_subset = indices.len();

    let mut subset_data = Vec::with_capacity(n_subset * n_features);
    let mut subset_labels = Vec::with_capacity(n_subset);

    for &idx in indices {
        for col in 0..n_features {
            subset_data.push(x.get(idx, col));
        }
        subset_labels.push(y[idx]);
    }

    let subset_matrix = crate::primitives::Matrix::from_vec(n_subset, n_features, subset_data)
        .unwrap_or_else(|_| {
            crate::primitives::Matrix::from_vec(0, n_features, vec![])
                .expect("empty matrix creation should succeed")
        });

    (subset_matrix, subset_labels)
}

/// Create a regression leaf node from y values.
fn make_regression_leaf(y_slice: &[f32], n_samples: usize) -> RegressionTreeNode {
    RegressionTreeNode::Leaf(RegressionLeaf {
        value: mean_f32(y_slice),
        n_samples,
    })
}

/// Check if we've reached max depth.
fn at_max_depth(depth: usize, max_depth: Option<usize>) -> bool {
    max_depth.is_some_and(|max_d| depth >= max_d)
}

/// Partition sample indices based on feature threshold.
fn partition_by_threshold(
    x: &crate::primitives::Matrix<f32>,
    n_samples: usize,
    feature_idx: usize,
    threshold: f32,
) -> (Vec<usize>, Vec<usize>) {
    let mut left = Vec::new();
    let mut right = Vec::new();
    for row in 0..n_samples {
        if x.get(row, feature_idx) <= threshold {
            left.push(row);
        } else {
            right.push(row);
        }
    }
    (left, right)
}

/// Build a regression decision tree recursively.
///
/// # Arguments
///
/// * `x` - Training data (`n_samples` × `n_features`)
/// * `y` - Target values (`n_samples`)
/// * `depth` - Current depth in tree
/// * `max_depth` - Maximum allowed depth (None = unlimited)
/// * `min_samples_split` - Minimum samples required to split
/// * `min_samples_leaf` - Minimum samples required in a leaf
///
/// # Returns
///
/// Root node of the built tree
fn build_regression_tree(
    x: &crate::primitives::Matrix<f32>,
    y: &crate::primitives::Vector<f32>,
    depth: usize,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
) -> RegressionTreeNode {
    let n_samples = y.len();
    let y_slice: Vec<f32> = y.as_slice().to_vec();

    // Early stopping checks
    if n_samples < min_samples_split
        || at_max_depth(depth, max_depth)
        || variance_f32(&y_slice) < 1e-10
    {
        return make_regression_leaf(&y_slice, n_samples);
    }

    // Try to find best split
    let Some((feature_idx, threshold, _gain)) = find_best_regression_split(x, &y_slice) else {
        return make_regression_leaf(&y_slice, n_samples);
    };

    // Partition samples
    let (left_indices, right_indices) =
        partition_by_threshold(x, n_samples, feature_idx, threshold);

    // Check min_samples_leaf constraint
    if left_indices.len() < min_samples_leaf || right_indices.len() < min_samples_leaf {
        return make_regression_leaf(&y_slice, n_samples);
    }

    // Build subtrees
    let (left_matrix, left_labels) = split_regression_data_by_indices(x, &y_slice, &left_indices);
    let (right_matrix, right_labels) =
        split_regression_data_by_indices(x, &y_slice, &right_indices);

    let left_child = build_regression_tree(
        &left_matrix,
        &crate::primitives::Vector::from_vec(left_labels),
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
    );
    let right_child = build_regression_tree(
        &right_matrix,
        &crate::primitives::Vector::from_vec(right_labels),
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
    );

    RegressionTreeNode::Node(RegressionNode {
        feature_idx,
        threshold,
        left: Box::new(left_child),
        right: Box::new(right_child),
    })
}

// ========================================================================
// End Regression Tree Building Functions
// ========================================================================

// ========================================================================
// Feature Importance Helper Functions (Issue #32)
// ========================================================================

/// Compute feature importances from a classification tree by traversing it.
///
/// Importance is based on the number of samples that pass through each split node.
/// Each split contributes to the importance of the feature it uses.
fn compute_tree_feature_importances(node: &TreeNode, importances: &mut [f32]) {
    match node {
        TreeNode::Leaf(_) => {
            // Leaf nodes don't contribute to feature importance
        }
        TreeNode::Node(n) => {
            // Add importance for this feature based on number of samples
            let n_samples = count_tree_samples(node) as f32;
            importances[n.feature_idx] += n_samples;

            // Recursively process children
            compute_tree_feature_importances(&n.left, importances);
            compute_tree_feature_importances(&n.right, importances);
        }
    }
}

/// Count total samples in a tree/subtree
fn count_tree_samples(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf(leaf) => leaf.n_samples,
        TreeNode::Node(n) => {
            // For internal nodes, sum samples from children
            count_tree_samples(&n.left) + count_tree_samples(&n.right)
        }
    }
}

/// Compute feature importances from a regression tree by traversing it.
fn compute_regression_tree_feature_importances(node: &RegressionTreeNode, importances: &mut [f32]) {
    match node {
        RegressionTreeNode::Leaf(_) => {
            // Leaf nodes don't contribute to feature importance
        }
        RegressionTreeNode::Node(n) => {
            // Add importance for this feature based on number of samples
            let n_samples = count_regression_tree_samples(node) as f32;
            importances[n.feature_idx] += n_samples;

            // Recursively process children
            compute_regression_tree_feature_importances(&n.left, importances);
            compute_regression_tree_feature_importances(&n.right, importances);
        }
    }
}

/// Count total samples in a regression tree/subtree
fn count_regression_tree_samples(node: &RegressionTreeNode) -> usize {
    match node {
        RegressionTreeNode::Leaf(leaf) => leaf.n_samples,
        RegressionTreeNode::Node(n) => {
            // For internal nodes, sum samples from children
            count_regression_tree_samples(&n.left) + count_regression_tree_samples(&n.right)
        }
    }
}

// ========================================================================
// End Feature Importance Helper Functions
// ========================================================================

/// Random Forest classifier - an ensemble of decision trees.
///
/// Combines multiple decision trees trained on bootstrap samples
/// with random feature selection to reduce overfitting and improve accuracy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier {
    trees: Vec<DecisionTreeClassifier>,
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
    /// OOB sample indices for each tree (samples NOT in bootstrap sample)
    oob_indices: Vec<Vec<usize>>,
    /// Training features (stored for OOB evaluation)
    x_train: Option<crate::primitives::Matrix<f32>>,
    /// Training labels (stored for OOB evaluation)
    y_train: Option<Vec<usize>>,
}

impl RandomForestClassifier {
    /// Creates a new Random Forest classifier.
    ///
    /// # Arguments
    ///
    /// * `n_estimators` - Number of trees in the forest
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
            let bootstrap_set: std::collections::HashSet<usize> =
                bootstrap_indices.iter().copied().collect();
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

        // Get predictions from each tree
        for sample_idx in 0..n_samples {
            let mut votes: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();

            // Collect votes from all trees
            for tree in &self.trees {
                let tree_prediction = tree.predict(x)[sample_idx];
                *votes.entry(tree_prediction).or_insert(0) += 1;
            }

            // Find class with most votes (majority voting)
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
    ///
    /// Returns probability distribution over classes based on
    /// vote proportions across trees in the forest.
    ///
    /// # Returns
    ///
    /// `Matrix<f32>` with shape `(n_samples, n_classes)` where each row
    /// sums to 1.0.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn predict_proba(
        &self,
        x: &crate::primitives::Matrix<f32>,
    ) -> crate::primitives::Matrix<f32> {
        let n_samples = x.shape().0;

        // Determine number of classes from training data
        let n_classes = self
            .y_train
            .as_ref()
            .map_or(2, |y| y.iter().max().copied().unwrap_or(0) + 1);

        let mut proba_data = vec![0.0f32; n_samples * n_classes];
        let n_trees = self.trees.len() as f32;

        // Get predictions from each tree and count votes
        for sample_idx in 0..n_samples {
            let mut votes = vec![0usize; n_classes];

            for tree in &self.trees {
                let tree_prediction = tree.predict(x)[sample_idx];
                if tree_prediction < n_classes {
                    votes[tree_prediction] += 1;
                }
            }

            // Convert votes to probabilities
            for class_idx in 0..n_classes {
                let idx = sample_idx * n_classes + class_idx;
                proba_data[idx] = votes[class_idx] as f32 / n_trees;
            }
        }

        crate::primitives::Matrix::from_vec(n_samples, n_classes, proba_data)
            .expect("Matrix creation should succeed")
    }

    /// Returns Out-of-Bag (OOB) predictions for training samples.
    ///
    /// For each training sample, predictions are made using only the trees
    /// where that sample was NOT in the bootstrap sample (out-of-bag).
    ///
    /// # Returns
    ///
    /// `Some(Vec<usize>)` if the model has been fitted, `None` otherwise.
    /// The vector has the same length as the training data.
    #[must_use]
    pub fn oob_prediction(&self) -> Option<Vec<usize>> {
        // Return None if model not fitted
        if self.trees.is_empty() || self.y_train.is_none() || self.x_train.is_none() {
            return None;
        }

        let x_train = self
            .x_train
            .as_ref()
            .expect("x_train should be stored after fit");
        let y_train = self
            .y_train
            .as_ref()
            .expect("y_train should be stored after fit");
        let n_samples = y_train.len();
        let n_features = x_train.shape().1;

        // Track votes for each sample from OOB trees
        let mut oob_votes: Vec<std::collections::HashMap<usize, usize>> =
            vec![std::collections::HashMap::new(); n_samples];

        // For each tree, make predictions on its OOB samples
        for (tree_idx, oob_indices) in self.oob_indices.iter().enumerate() {
            let tree = &self.trees[tree_idx];

            // For each OOB sample for this tree
            for &sample_idx in oob_indices {
                // Extract single sample as a 1×n_features matrix
                let mut sample_data = Vec::with_capacity(n_features);
                for j in 0..n_features {
                    sample_data.push(x_train.get(sample_idx, j));
                }

                let sample_matrix =
                    crate::primitives::Matrix::from_vec(1, n_features, sample_data).ok()?;

                // Get prediction from this tree
                let tree_predictions = tree.predict(&sample_matrix);
                let predicted_class = tree_predictions[0];

                // Record vote
                *oob_votes[sample_idx].entry(predicted_class).or_insert(0) += 1;
            }
        }

        // Convert votes to final predictions (majority voting)
        let mut predictions = Vec::with_capacity(n_samples);
        for votes in oob_votes {
            if votes.is_empty() {
                // No OOB predictions for this sample (never OOB for any tree)
                // Use 0 as default (this shouldn't happen with enough trees)
                predictions.push(0);
            } else {
                // Find class with most votes
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
    ///
    /// Computes accuracy using OOB predictions. This provides an unbiased
    /// estimate of the model's performance without needing a validation set.
    ///
    /// # Returns
    ///
    /// `Some(f32)` with accuracy in [0, 1] if model has been fitted, `None` otherwise.
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
    ///
    /// Feature importance is calculated as the total decrease in node impurity
    /// (weighted by the number of samples) averaged over all trees in the forest.
    ///
    /// # Returns
    ///
    /// `Some(Vec<f32>)` with importance for each feature (normalized to sum to 1.0)
    /// if model has been fitted, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let mut rf = RandomForestClassifier::new(50);
    /// rf.fit(&x_train, &y_train).expect("fit should succeed");
    ///
    /// if let Some(importances) = rf.feature_importances() {
    ///     for (i, &importance) in importances.iter().enumerate() {
    ///         println!("Feature {}: {:.4}", i, importance);
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn feature_importances(&self) -> Option<Vec<f32>> {
        if self.trees.is_empty() || self.x_train.is_none() {
            return None;
        }

        let n_features = self
            .x_train
            .as_ref()
            .expect("x_train should be stored after fit")
            .shape()
            .1;
        let mut total_importances = vec![0.0; n_features];

        // Aggregate importances from all trees
        for tree in &self.trees {
            if let Some(tree_node) = &tree.tree {
                let mut tree_importances = vec![0.0; n_features];
                compute_tree_feature_importances(tree_node, &mut tree_importances);

                // Add to total
                for (i, &importance) in tree_importances.iter().enumerate() {
                    total_importances[i] += importance;
                }
            }
        }

        // Normalize: divide by number of trees and then normalize to sum to 1.0
        let n_trees = self.trees.len() as f32;
        for importance in &mut total_importances {
            *importance /= n_trees;
        }

        // Normalize to sum to 1.0
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
    /// # Arguments
    ///
    /// * `path` - Path where the `SafeTensors` file will be saved
    ///
    /// # Errors
    ///
    /// Returns an error if the model is unfitted or if saving fails.
    pub fn save_safetensors<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
        use crate::serialization::safetensors;
        use std::collections::BTreeMap;

        // Check if model is fitted
        if self.trees.is_empty() {
            return Err("Cannot save unfitted model. Call fit() first.".to_string());
        }

        let mut tensors = BTreeMap::new();

        // Save each tree's structure
        for (tree_idx, tree) in self.trees.iter().enumerate() {
            // Get tree structure
            let tree_node = tree
                .tree
                .as_ref()
                .ok_or("Tree has no root node".to_string())?;

            // Flatten tree to arrays
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

            // Store each array with tree index prefix
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

            // Store tree's max_depth
            let tree_max_depth = if let Some(depth) = tree.max_depth {
                depth as f32
            } else {
                -1.0
            };
            tensors.insert(
                format!("{prefix}max_depth"),
                (vec![tree_max_depth], vec![1]),
            );
        }

        // Save hyperparameters
        tensors.insert(
            "n_estimators".to_string(),
            (vec![self.n_estimators as f32], vec![1]),
        );

        let max_depth_val = if let Some(depth) = self.max_depth {
            depth as f32
        } else {
            -1.0
        };
        tensors.insert("max_depth".to_string(), (vec![max_depth_val], vec![1]));

        let random_state_val = if let Some(state) = self.random_state {
            state as f32
        } else {
            -1.0
        };
        tensors.insert(
            "random_state".to_string(),
            (vec![random_state_val], vec![1]),
        );

        safetensors::save_safetensors(path, &tensors)?;
        Ok(())
    }

    /// Loads a Random Forest model from a `SafeTensors` file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the `SafeTensors` file
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails or if the file format is invalid.
    pub fn load_safetensors<P: AsRef<Path>>(path: P) -> std::result::Result<Self, String> {
        use crate::serialization::safetensors;

        // Load SafeTensors file
        let (metadata, raw_data) = safetensors::load_safetensors(path)?;

        // Load hyperparameters
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

        // Reconstruct each tree
        let mut trees = Vec::with_capacity(n_estimators);
        for tree_idx in 0..n_estimators {
            let prefix = format!("tree_{tree_idx}_");

            // Load tree arrays
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

            // Validate array sizes
            let n_nodes = node_features.len();
            if node_thresholds.len() != n_nodes
                || node_classes.len() != n_nodes
                || node_samples.len() != n_nodes
                || node_left_child.len() != n_nodes
                || node_right_child.len() != n_nodes
            {
                return Err(format!("Mismatched array sizes for tree {tree_idx}"));
            }

            // Reconstruct tree
            let tree_node = reconstruct_tree_node(
                0,
                &node_features,
                &node_thresholds,
                &node_classes,
                &node_samples,
                &node_left_child,
                &node_right_child,
            );

            // Load tree's max_depth
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

/// Creates a bootstrap sample (random sample with replacement).
///
/// Returns indices of samples to include in the bootstrap sample.
fn bootstrap_sample(n_samples: usize, random_state: Option<u64>) -> Vec<usize> {
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;

    let dist = Uniform::from(0..n_samples);

    let mut indices = Vec::with_capacity(n_samples);

    if let Some(seed) = random_state {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for _ in 0..n_samples {
            indices.push(dist.sample(&mut rng));
        }
    } else {
        let mut rng = rand::thread_rng();
        for _ in 0..n_samples {
            indices.push(dist.sample(&mut rng));
        }
    }

    indices
}

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
///
/// # Example
///
/// ```ignore
/// use aprender::tree::GradientBoostingClassifier;
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
/// let mut gbm = GradientBoostingClassifier::new()
///     .with_n_estimators(50)
///     .with_learning_rate(0.1)
///     .with_max_depth(3);
///
/// gbm.fit(&x, &y)?;
/// let predictions = gbm.predict(&x)?;
/// ```
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

        // Initialize predictions (raw scores, not probabilities)
        let mut raw_predictions = vec![self.init_prediction; n_samples];

        // Clear previous estimators
        self.estimators.clear();

        // Boosting iterations
        for _iteration in 0..self.n_estimators {
            // Compute negative gradients (pseudo-residuals)
            // For binary classification with log-loss:
            // gradient = y - sigmoid(raw_prediction)
            let mut residuals = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let prob = Self::sigmoid(raw_predictions[i]);
                residuals.push(y_float[i] - prob);
            }

            // Fit a decision tree to residuals
            // We'll create pseudo-labels by binning residuals into classes
            // This is a simplified approach - ideally we'd use regression trees
            let residual_labels = self.residuals_to_labels(&residuals);

            let mut tree = DecisionTreeClassifier::new().with_max_depth(self.max_depth);

            // Fit tree to residuals (as classification problem)
            if tree.fit(x, &residual_labels).is_err() {
                // If tree fitting fails (e.g., all same class), stop boosting
                break;
            }

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
}

impl Default for GradientBoostingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
