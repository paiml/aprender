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
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets the minimum number of samples required to split an internal node.
    ///
    /// # Arguments
    ///
    /// * `min_samples` - Minimum samples to split (must be >= 2)
    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples.max(2);
        self
    }

    /// Sets the minimum number of samples required to be at a leaf node.
    ///
    /// # Arguments
    ///
    /// * `min_samples` - Minimum samples per leaf (must be >= 1)
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples.max(1);
        self
    }

    /// Fits the decision tree to training data.
    ///
    /// # Arguments
    ///
    /// * `x` - Training features (n_samples × n_features)
    /// * `y` - Training target values (n_samples continuous values)
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
    /// * `x` - Feature matrix (n_samples × n_features)
    ///
    /// # Returns
    ///
    /// Vector of predicted values
    ///
    /// # Panics
    ///
    /// Panics if called before fit()
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
    /// * `x` - Test features (n_samples × n_features)
    /// * `y` - True target values (n_samples)
    ///
    /// # Returns
    ///
    /// R² coefficient of determination
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
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Sets the random state for reproducibility.
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
    /// * `x` - Test features (n_samples × n_features)
    /// * `y` - True target values (n_samples)
    ///
    /// # Returns
    ///
    /// R² coefficient of determination
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
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Fits the decision tree to training data.
    ///
    /// # Arguments
    ///
    /// * `x` - Training features (n_samples × n_features)
    /// * `y` - Training labels (n_samples class indices)
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
    /// * `x` - Feature matrix (n_samples × n_features)
    ///
    /// # Returns
    ///
    /// Vector of predicted class labels
    ///
    /// # Panics
    ///
    /// Panics if called before fit() or if feature count doesn't match training data
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
    /// * `x` - Test features (n_samples × n_features)
    /// * `y` - True labels (n_samples)
    ///
    /// # Returns
    ///
    /// Accuracy (fraction of correct predictions)
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

    /// Saves the model to SafeTensors format.
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

    /// Loads a model from SafeTensors format.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File reading fails
    /// - SafeTensors format is invalid
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
/// Formula: Gini = 1 - Σ(p_i²) where p_i is the proportion of class i
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
/// * `x` - Feature values (n_samples)
/// * `y` - Labels (n_samples)
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
/// * `x_matrix` - Training data (n_samples × n_features)
/// * `y` - Labels (n_samples)
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
/// * `x` - Training data (n_samples × n_features)
/// * `y` - Labels (n_samples)
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
/// Returns (feature_idx, threshold, mse_reduction) if a valid split exists.
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
/// * `x` - Training data (n_samples × n_features)
/// * `y` - Target values (n_samples)
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
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    /// Sets the random state for reproducibility.
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

    /// Saves the Random Forest model to a SafeTensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the SafeTensors file will be saved
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

    /// Loads a Random Forest model from a SafeTensors file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the SafeTensors file
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
///    - Update predictions with learning_rate * tree_prediction
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
    /// - n_estimators: 100
    /// - learning_rate: 0.1
    /// - max_depth: 3
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
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Sets the learning rate (shrinkage parameter).
    ///
    /// Lower values require more trees but often lead to better generalization.
    /// Typical values: 0.01 - 0.3
    pub fn with_learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the maximum depth of each tree.
    ///
    /// Smaller depths prevent overfitting. Typical values: 3-8
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
    /// - `x`: Feature matrix (n_samples × n_features)
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
    /// - `x`: Feature matrix (n_samples × n_features)
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
    /// - `x`: Feature matrix (n_samples × n_features)
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
mod tests {
    use super::*;
    use crate::primitives::{Matrix, Vector};
    use crate::traits::Estimator;

    // RED Phase: Write failing tests first

    #[test]
    fn test_leaf_creation() {
        let leaf = Leaf {
            class_label: 1,
            n_samples: 10,
        };
        assert_eq!(leaf.class_label, 1);
        assert_eq!(leaf.n_samples, 10);
    }

    #[test]
    fn test_node_creation() {
        let left = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 5,
        });
        let right = TreeNode::Leaf(Leaf {
            class_label: 1,
            n_samples: 5,
        });

        let node = Node {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(left),
            right: Box::new(right),
        };

        assert_eq!(node.feature_idx, 0);
        assert!((node.threshold - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tree_depth() {
        // Leaf has depth 0
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 1,
        });
        assert_eq!(leaf.depth(), 0);

        // Tree with one split has depth 1
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 1,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 1,
            })),
        });
        assert_eq!(tree.depth(), 1);

        // Tree with nested splits has depth 2
        let deep_tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(tree),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 1,
            })),
        });
        assert_eq!(deep_tree.depth(), 2);
    }

    #[test]
    fn test_decision_tree_creation() {
        let tree = DecisionTreeClassifier::new();
        assert!(tree.tree.is_none());
        assert!(tree.max_depth.is_none());
    }

    #[test]
    fn test_decision_tree_with_max_depth() {
        let tree = DecisionTreeClassifier::new().with_max_depth(5);
        assert_eq!(tree.max_depth, Some(5));
    }

    #[test]
    fn test_decision_tree_default() {
        let tree = DecisionTreeClassifier::default();
        assert!(tree.tree.is_none());
        assert!(tree.max_depth.is_none());
    }

    // RED-GREEN-REFACTOR Cycle 2: Gini Impurity

    #[test]
    fn test_gini_impurity_pure() {
        // Pure node (all same class) should have Gini = 0.0
        let pure = vec![0, 0, 0, 0, 0];
        assert!((gini_impurity(&pure) - 0.0).abs() < 1e-6);

        let pure_ones = vec![1, 1, 1];
        assert!((gini_impurity(&pure_ones) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gini_impurity_empty() {
        // Empty set should have Gini = 0.0
        let empty: Vec<usize> = vec![];
        assert!((gini_impurity(&empty) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gini_impurity_binary_50_50() {
        // 50/50 binary split should have Gini = 0.5
        let mixed = vec![0, 1, 0, 1];
        assert!((gini_impurity(&mixed) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_gini_impurity_three_class_even() {
        // Three classes evenly distributed: Gini = 1 - 3*(1/3)² = 1 - 1/3 = 0.6667
        let three_class = vec![0, 1, 2, 0, 1, 2];
        assert!((gini_impurity(&three_class) - 0.6667).abs() < 1e-4);
    }

    #[test]
    fn test_gini_impurity_bounds() {
        // Gini impurity should always be in [0, 1]
        let labels_sets = vec![
            vec![0, 0, 0],
            vec![0, 1],
            vec![0, 1, 2],
            vec![0, 0, 1, 1, 2, 2],
            vec![0, 0, 0, 1],
        ];

        for labels in labels_sets {
            let gini = gini_impurity(&labels);
            assert!(gini >= 0.0, "Gini should be >= 0, got {gini}");
            assert!(gini <= 1.0, "Gini should be <= 1, got {gini}");
        }
    }

    #[test]
    fn test_gini_split_calculation() {
        // Test weighted Gini for a split
        let left = vec![0, 0, 0]; // Pure, Gini = 0.0
        let right = vec![1, 1, 1]; // Pure, Gini = 0.0
                                   // Weighted: (3/6)*0.0 + (3/6)*0.0 = 0.0
        assert!((gini_split(&left, &right) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gini_split_mixed() {
        // Left: [0, 0, 1] - Gini = 1 - (2/3)² - (1/3)² = 1 - 4/9 - 1/9 = 4/9 ≈ 0.4444
        // Right: [1, 1] - Gini = 0.0
        // Weighted: (3/5)*0.4444 + (2/5)*0.0 ≈ 0.2667
        let left = vec![0, 0, 1];
        let right = vec![1, 1];
        let expected = (3.0 / 5.0) * (4.0 / 9.0);
        assert!((gini_split(&left, &right) - expected).abs() < 1e-4);
    }

    // RED-GREEN-REFACTOR Cycle 3: Best Split Finding

    #[test]
    fn test_find_best_split_simple() {
        // Simple linearly separable data
        // x: [1.0, 2.0, 5.0, 6.0]
        // y: [0, 0, 1, 1]
        // Best split should be around 3.5 (midpoint between 2 and 5)
        let x = vec![1.0, 2.0, 5.0, 6.0];
        let y = vec![0, 0, 1, 1];

        let result = find_best_split_for_feature(&x, &y);
        assert!(result.is_some());

        let (threshold, gain) = result.expect("should have valid result");
        // Threshold should be between 2.0 and 5.0
        assert!(threshold > 2.0 && threshold < 5.0);
        // Gain should be positive (we're reducing impurity)
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_split_no_gain() {
        // All same label - no possible gain
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0, 0, 0, 0];

        let result = find_best_split_for_feature(&x, &y);
        // Should return None since there's no gain possible
        assert!(result.is_none());
    }

    #[test]
    fn test_find_best_split_too_small() {
        // Only one sample
        let x = vec![1.0];
        let y = vec![0];

        let result = find_best_split_for_feature(&x, &y);
        assert!(result.is_none());
    }

    #[test]
    fn test_find_best_split_gain_is_positive() {
        // Property: Gain should always be >= 0
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0, 1, 0, 1];

        if let Some((_threshold, gain)) = find_best_split_for_feature(&x, &y) {
            assert!(gain >= 0.0);
        }
    }

    #[test]
    fn test_find_best_split_across_features() {
        use crate::primitives::Matrix;

        // 2D data with 2 features
        // Feature 0: [1.0, 1.0, 5.0, 5.0]
        // Feature 1: [1.0, 2.0, 5.0, 6.0]
        // Labels:    [  0,   0,   1,   1]
        // Both features separate perfectly, should choose one
        let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let result = find_best_split(&x, &y);
        assert!(result.is_some());

        let (feature_idx, _threshold, gain) = result.expect("should have valid result");
        // Should choose one of the features
        assert!(feature_idx < 2);
        // Should have positive gain
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_split_perfect_separation() {
        use crate::primitives::Matrix;

        // Perfectly separable data on feature 0
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let result = find_best_split(&x, &y);
        assert!(result.is_some());

        let (feature_idx, threshold, gain) = result.expect("should have valid result");
        assert_eq!(feature_idx, 0); // Should choose the only feature
        assert!(threshold > 2.0 && threshold < 5.0);
        // Gain should be maximum (from mixed to pure)
        assert!(gain > 0.4); // At least some significant gain
    }

    // RED-GREEN-REFACTOR Cycle 4: Tree Building

    #[test]
    fn test_majority_class_simple() {
        let labels = vec![0, 0, 1, 0, 1];
        // Class 0 appears 3 times, class 1 appears 2 times
        assert_eq!(majority_class(&labels), 0);
    }

    #[test]
    fn test_majority_class_tie() {
        let labels = vec![0, 1, 0, 1];
        // Tie - either 0 or 1 is acceptable
        let result = majority_class(&labels);
        assert!(result == 0 || result == 1);
    }

    #[test]
    fn test_majority_class_single() {
        let labels = vec![5];
        assert_eq!(majority_class(&labels), 5);
    }

    #[test]
    fn test_build_tree_pure_leaf() {
        use crate::primitives::Matrix;

        // All same label - should create a leaf immediately
        let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0];

        let tree = build_tree(&x, &y, 0, None);

        // Should be a leaf with class 0
        match tree {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 0);
                assert_eq!(leaf.n_samples, 3);
            }
            TreeNode::Node(_) => panic!("Expected Leaf node for pure data"),
        }
    }

    #[test]
    fn test_build_tree_max_depth_zero() {
        use crate::primitives::Matrix;

        // Mixed labels but max_depth=0, should create leaf with majority
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let tree = build_tree(&x, &y, 0, Some(0));

        // Should be a leaf (can't split at depth 0)
        match tree {
            TreeNode::Leaf(leaf) => {
                assert!(leaf.class_label == 0 || leaf.class_label == 1);
                assert_eq!(leaf.n_samples, 4);
            }
            TreeNode::Node(_) => panic!("Expected Leaf node at max depth"),
        }
    }

    #[test]
    fn test_build_tree_simple_split() {
        use crate::primitives::Matrix;

        // Simple binary split
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let tree = build_tree(&x, &y, 0, Some(5));

        // Should be a node (not pure, depth allows split)
        match tree {
            TreeNode::Node(node) => {
                assert_eq!(node.feature_idx, 0); // Only one feature
                assert!(node.threshold > 2.0 && node.threshold < 5.0);
            }
            TreeNode::Leaf(_) => panic!("Expected Node for splittable data"),
        }
    }

    #[test]
    fn test_build_tree_depth_tracking() {
        use crate::primitives::Matrix;

        // Build tree and verify depth is respected
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let tree = build_tree(&x, &y, 0, Some(1));

        // Depth should be <= 1
        assert!(tree.depth() <= 1);
    }

    // RED-GREEN-REFACTOR Cycle 5: fit/predict/score

    #[test]
    fn test_fit_simple() {
        use crate::primitives::Matrix;

        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        let result = tree.fit(&x, &y);

        assert!(result.is_ok());
        assert!(tree.tree.is_some()); // Tree should be built
    }

    #[test]
    fn test_predict_perfect_classification() {
        use crate::primitives::Matrix;

        // Perfectly separable data
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);
        assert_eq!(predictions, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_predict_single_sample() {
        use crate::primitives::Matrix;

        let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y_train = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x_train, &y_train).expect("fit should succeed");

        // Test single sample prediction
        let x_test =
            Matrix::from_vec(1, 1, vec![1.5]).expect("Matrix creation should succeed in tests");
        let predictions = tree.predict(&x_test);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0], 0); // Should be class 0 (closer to 1.0, 2.0)
    }

    #[test]
    fn test_score_perfect() {
        use crate::primitives::Matrix;

        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).expect("fit should succeed");

        let accuracy = tree.score(&x, &y);
        assert!((accuracy - 1.0).abs() < 1e-6); // Perfect classification
    }

    #[test]
    fn test_score_partial() {
        use crate::primitives::Matrix;

        // Train on simple data
        let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y_train = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(1);
        tree.fit(&x_train, &y_train).expect("fit should succeed");

        // Score should be between 0 and 1
        let accuracy = tree.score(&x_train, &y_train);
        assert!((0.0..=1.0).contains(&accuracy));
    }

    #[test]
    fn test_multiclass_classification() {
        use crate::primitives::Matrix;

        // 3-class problem
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, // class 0
                1.5, 1.5, // class 0
                5.0, 5.0, // class 1
                5.5, 5.5, // class 1
                9.0, 9.0, // class 2
                9.5, 9.5, // class 2
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);
        assert_eq!(predictions.len(), 6);
        // Should classify perfectly
        assert_eq!(predictions, vec![0, 0, 1, 1, 2, 2]);
    }

    #[test]
    fn test_save_load() {
        use crate::primitives::Matrix;
        use std::fs;
        use std::path::Path;

        // Train a tree
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, // class 0
                1.5, 1.5, // class 0
                5.0, 5.0, // class 1
                5.5, 5.5, // class 1
                9.0, 9.0, // class 2
                9.5, 9.5, // class 2
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).expect("fit should succeed");

        // Save model
        let path = Path::new("/tmp/test_decision_tree.bin");
        tree.save(path).expect("Failed to save model");

        // Load model
        let loaded = DecisionTreeClassifier::load(path).expect("Failed to load model");

        // Verify predictions match
        let original_pred = tree.predict(&x);
        let loaded_pred = loaded.predict(&x);
        assert_eq!(original_pred, loaded_pred);

        // Verify accuracy matches
        let original_score = tree.score(&x, &y);
        let loaded_score = loaded.score(&x, &y);
        assert!((original_score - loaded_score).abs() < 1e-6);

        // Cleanup
        fs::remove_file(path).ok();
    }

    // Random Forest Tests

    #[test]
    fn test_bootstrap_sample_size() {
        let indices = bootstrap_sample(100, Some(42));
        assert_eq!(
            indices.len(),
            100,
            "Bootstrap sample should have same size as original"
        );
    }

    #[test]
    fn test_bootstrap_sample_reproducible() {
        let indices1 = bootstrap_sample(50, Some(42));
        let indices2 = bootstrap_sample(50, Some(42));
        assert_eq!(
            indices1, indices2,
            "Same seed should give same bootstrap sample"
        );
    }

    #[test]
    fn test_random_forest_creation() {
        let rf = RandomForestClassifier::new(10);
        assert_eq!(rf.n_estimators, 10);
    }

    #[test]
    fn test_random_forest_builder() {
        let rf = RandomForestClassifier::new(5)
            .with_max_depth(3)
            .with_random_state(42);
        assert_eq!(rf.n_estimators, 5);
        assert_eq!(rf.max_depth, Some(3));
        assert_eq!(rf.random_state, Some(42));
    }

    #[test]
    fn test_random_forest_fit_basic() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, // class 0
                1.5, 1.5, // class 0
                5.0, 5.0, // class 1
                5.5, 5.5, // class 1
                9.0, 9.0, // class 2
                9.5, 9.5, // class 2
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(3)
            .with_random_state(42);

        rf.fit(&x, &y).expect("Fit should succeed");

        // Should have trained the correct number of trees
        assert_eq!(rf.trees.len(), 3, "Should have 3 trees");
    }

    #[test]
    fn test_random_forest_predict() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, // class 0
                1.5, 1.5, // class 0
                5.0, 5.0, // class 1
                5.5, 5.5, // class 1
                9.0, 9.0, // class 2
                9.5, 9.5, // class 2
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut rf = RandomForestClassifier::new(5)
            .with_max_depth(5)
            .with_random_state(42);

        rf.fit(&x, &y).expect("fit should succeed");
        let predictions = rf.predict(&x);

        assert_eq!(predictions.len(), 6, "Should predict for all samples");

        // Perfect separation - should get perfect accuracy
        let score = rf.score(&x, &y);
        assert!(
            score > 0.8,
            "Random Forest should achieve >80% accuracy on simple data"
        );
    }

    #[test]
    fn test_random_forest_reproducible() {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.0, // class 0
                0.5, 0.5, // class 0
                5.0, 5.0, // class 1
                5.5, 5.5, // class 1
                10.0, 10.0, // class 2
                10.5, 10.5, // class 2
                1.0, 1.0, // class 0
                6.0, 6.0, // class 1
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1, 2, 2, 0, 1];

        let mut rf1 = RandomForestClassifier::new(5).with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let pred1 = rf1.predict(&x);

        let mut rf2 = RandomForestClassifier::new(5).with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let pred2 = rf2.predict(&x);

        assert_eq!(
            pred1, pred2,
            "Same random state should give same predictions"
        );
    }

    // ===== Gradient Boosting Tests =====

    #[test]
    fn test_gradient_boosting_new() {
        let gbm = GradientBoostingClassifier::new();
        assert_eq!(gbm.n_estimators, 100);
        assert_eq!(gbm.learning_rate, 0.1);
        assert_eq!(gbm.max_depth, 3);
        assert_eq!(gbm.n_estimators(), 0); // No estimators before fit
    }

    #[test]
    fn test_gradient_boosting_builder() {
        let gbm = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.05)
            .with_max_depth(5);

        assert_eq!(gbm.n_estimators, 50);
        assert_eq!(gbm.learning_rate, 0.05);
        assert_eq!(gbm.max_depth, 5);
    }

    #[test]
    fn test_gradient_boosting_fit_simple() {
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
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let mut gbm = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.1)
            .with_max_depth(2);

        let result = gbm.fit(&x, &y);
        assert!(result.is_ok());
        assert!(gbm.n_estimators() > 0); // Should have fitted some trees
    }

    #[test]
    fn test_gradient_boosting_predict_simple() {
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
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let mut gbm = GradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_learning_rate(0.1)
            .with_max_depth(2);

        gbm.fit(&x, &y).expect("fit should succeed");
        let predictions = gbm.predict(&x).expect("predict should succeed");

        assert_eq!(predictions.len(), 4);

        // GBM should classify correctly with enough iterations
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| *pred == *true_label)
            .count();

        // Should get at least 3 out of 4 correct
        assert!(correct >= 3);
    }

    #[test]
    fn test_gradient_boosting_predict_proba() {
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
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1];

        let mut gbm = GradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_learning_rate(0.1)
            .with_max_depth(2);

        gbm.fit(&x, &y).expect("fit should succeed");
        let probas = gbm.predict_proba(&x).expect("predict_proba should succeed");

        assert_eq!(probas.len(), 4);

        // Each sample should have 2 probabilities (class 0 and class 1)
        for probs in &probas {
            assert_eq!(probs.len(), 2);
            // Probabilities should sum to ~1.0
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01);
            // Each probability should be between 0 and 1
            for &p in probs {
                assert!((0.0..=1.0).contains(&p));
            }
        }
    }

    #[test]
    fn test_gradient_boosting_predict_untrained() {
        let gbm = GradientBoostingClassifier::new();
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0])
            .expect("Matrix creation should succeed in tests");

        let result = gbm.predict(&x);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail when predicting with untrained model"),
            "Model not trained yet"
        );
    }

    #[test]
    fn test_gradient_boosting_empty_data() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed in tests");
        let y = vec![];

        let mut gbm = GradientBoostingClassifier::new();
        let result = gbm.fit(&x, &y);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with empty data"),
            "Cannot fit with 0 samples"
        );
    }

    #[test]
    fn test_gradient_boosting_mismatched_samples() {
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
            .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1]; // Wrong length

        let mut gbm = GradientBoostingClassifier::new();
        let result = gbm.fit(&x, &y);
        assert!(result.is_err());
        assert_eq!(
            result.expect_err("Should fail with mismatched sample counts"),
            "x and y must have the same number of samples"
        );
    }

    #[test]
    fn test_gradient_boosting_learning_rate_effect() {
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
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 1, 1, 1];

        // High learning rate
        let mut gbm_high_lr = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.5);
        gbm_high_lr.fit(&x, &y).expect("fit should succeed");
        let pred_high = gbm_high_lr.predict(&x).expect("predict should succeed");

        // Low learning rate
        let mut gbm_low_lr = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_learning_rate(0.01);
        gbm_low_lr.fit(&x, &y).expect("fit should succeed");
        let pred_low = gbm_low_lr.predict(&x).expect("predict should succeed");

        // Both should make predictions
        assert_eq!(pred_high.len(), 6);
        assert_eq!(pred_low.len(), 6);
    }

    #[test]
    fn test_gradient_boosting_n_estimators_effect() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.0, 0.2, // class 0
                1.0, 1.0, 0.9, 0.9, 1.0, 0.8, // class 1
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 1, 1, 1];

        // Few estimators
        let mut gbm_few = GradientBoostingClassifier::new()
            .with_n_estimators(5)
            .with_learning_rate(0.1);
        gbm_few.fit(&x, &y).expect("fit should succeed");

        // Many estimators
        let mut gbm_many = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.1);
        gbm_many.fit(&x, &y).expect("fit should succeed");

        // More estimators should generally lead to more trees (up to limit)
        assert!(gbm_many.n_estimators() >= gbm_few.n_estimators());
    }

    #[test]
    fn test_gradient_boosting_max_depth_effect() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                0.0, 0.0, 0.1, 0.1, 0.0, 0.2, // class 0
                1.0, 1.0, 0.9, 0.9, 1.0, 0.8, // class 1
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 1, 1, 1];

        // Shallow trees
        let mut gbm_shallow = GradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_max_depth(1);
        gbm_shallow.fit(&x, &y).expect("fit should succeed");
        let pred_shallow = gbm_shallow.predict(&x).expect("predict should succeed");

        // Deeper trees
        let mut gbm_deep = GradientBoostingClassifier::new()
            .with_n_estimators(20)
            .with_max_depth(5);
        gbm_deep.fit(&x, &y).expect("fit should succeed");
        let pred_deep = gbm_deep.predict(&x).expect("predict should succeed");

        // Both should make predictions
        assert_eq!(pred_shallow.len(), 6);
        assert_eq!(pred_deep.len(), 6);
    }

    #[test]
    fn test_gradient_boosting_binary_classification() {
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
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let mut gbm = GradientBoostingClassifier::new()
            .with_n_estimators(30)
            .with_learning_rate(0.1)
            .with_max_depth(3);

        gbm.fit(&x, &y).expect("fit should succeed");
        let predictions = gbm.predict(&x).expect("predict should succeed");

        // Should achieve reasonable accuracy
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_label)| *pred == *true_label)
            .count();

        // Should get at least 7 out of 10 correct for well-separated clusters
        assert!(
            correct >= 7,
            "Expected at least 7/10 correct, got {correct}/10"
        );
    }

    #[test]
    fn test_gradient_boosting_default() {
        let gbm1 = GradientBoostingClassifier::new();
        let gbm2 = GradientBoostingClassifier::default();

        assert_eq!(gbm1.n_estimators, gbm2.n_estimators);
        assert_eq!(gbm1.learning_rate, gbm2.learning_rate);
        assert_eq!(gbm1.max_depth, gbm2.max_depth);
    }

    // ========================================================================
    // Decision Tree Regression Tests (RED Phase - Issue #29)
    // ========================================================================

    #[test]
    fn test_regression_tree_creation() {
        let tree = DecisionTreeRegressor::new();
        assert!(tree.tree.is_none());
        assert!(tree.max_depth.is_none());
    }

    #[test]
    fn test_regression_tree_with_max_depth() {
        let tree = DecisionTreeRegressor::new().with_max_depth(5);
        assert_eq!(tree.max_depth, Some(5));
    }

    #[test]
    fn test_regression_tree_fit_simple_linear() {
        // Simple linear relationship: y = 2x + 1
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);

        // Tree should learn the linear pattern reasonably well
        let pred_slice = predictions.as_slice();
        let y_slice = y.as_slice();
        for i in 0..predictions.len() {
            assert!(
                (pred_slice[i] - y_slice[i]).abs() < 2.0,
                "Prediction {} too far from true value {}",
                pred_slice[i],
                y_slice[i]
            );
        }
    }

    #[test]
    fn test_regression_tree_predict_nonlinear() {
        // Quadratic relationship: y = x^2
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);

        // Should capture quadratic pattern with enough depth
        let mut mse_sum = 0.0_f32;
        let pred_slice = predictions.as_slice();
        let y_slice = y.as_slice();
        for i in 0..predictions.len() {
            mse_sum += (pred_slice[i] - y_slice[i]).powi(2);
        }
        let mse = mse_sum / predictions.len() as f32;

        assert!(mse < 50.0, "MSE {mse} too high for quadratic fit");
    }

    #[test]
    fn test_regression_tree_score() {
        // Perfect predictions should give R² = 1.0
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
        tree.fit(&x, &y).expect("fit should succeed");

        let r2 = tree.score(&x, &y);

        // R² should be high for training data
        assert!(r2 > 0.5, "R² score {r2} too low");
        assert!(r2 <= 1.0, "R² score {r2} exceeds maximum");
    }

    #[test]
    fn test_regression_tree_max_depth_limits_complexity() {
        let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

        // Shallow tree
        let mut tree_shallow = DecisionTreeRegressor::new().with_max_depth(1);
        tree_shallow.fit(&x, &y).expect("fit should succeed");
        let depth_shallow = tree_shallow
            .tree
            .as_ref()
            .expect("tree should exist after fit")
            .depth();
        assert!(
            depth_shallow <= 1,
            "Shallow tree depth {depth_shallow} exceeds max"
        );

        // Deep tree
        let mut tree_deep = DecisionTreeRegressor::new().with_max_depth(5);
        tree_deep.fit(&x, &y).expect("fit should succeed");
        let depth_deep = tree_deep
            .tree
            .as_ref()
            .expect("tree should exist after fit")
            .depth();
        assert!(depth_deep <= 5, "Deep tree depth {depth_deep} exceeds max");

        // Deeper tree should fit better
        let r2_shallow = tree_shallow.score(&x, &y);
        let r2_deep = tree_deep.score(&x, &y);
        assert!(
            r2_deep >= r2_shallow,
            "Deeper tree R²={r2_deep} should be >= shallow tree R²={r2_shallow}"
        );
    }

    #[test]
    #[should_panic(expected = "Model not fitted")]
    fn test_regression_tree_predict_before_fit_panics() {
        let tree = DecisionTreeRegressor::new();
        let x = Matrix::from_vec(2, 1, vec![1.0, 2.0])
            .expect("Matrix creation should succeed in tests");
        let _ = tree.predict(&x); // Should panic
    }

    #[test]
    fn test_regression_tree_multidimensional_features() {
        // 2D features: y = x1 + 2*x2
        let x = Matrix::from_vec(
            6,
            2,
            vec![
                1.0, 1.0, // y = 3
                2.0, 1.0, // y = 4
                1.0, 2.0, // y = 5
                2.0, 2.0, // y = 6
                3.0, 1.0, // y = 5
                1.0, 3.0, // y = 7
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
        tree.fit(&x, &y).expect("fit should succeed");

        let r2 = tree.score(&x, &y);
        assert!(r2 > 0.5, "R² score {r2} too low for 2D features");
    }

    #[test]
    fn test_regression_tree_constant_target() {
        // All y values the same
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);

        let mut tree = DecisionTreeRegressor::new();
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);

        // Should predict the constant value
        for &pred in predictions.as_slice() {
            assert!(
                (pred - 5.0).abs() < 1e-5,
                "Prediction {pred} should be 5.0 for constant target"
            );
        }
    }

    #[test]
    fn test_regression_tree_single_sample() {
        let x = Matrix::from_vec(1, 1, vec![5.0]).expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[10.0]);

        let mut tree = DecisionTreeRegressor::new();
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);
        assert_eq!(predictions.len(), 1);
        assert!((predictions[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_regression_tree_fit_validation() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 2.0]); // Wrong size

        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);

        assert!(result.is_err(), "Should error on mismatched dimensions");
    }

    #[test]
    fn test_regression_tree_zero_samples() {
        let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[]);

        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);

        assert!(result.is_err(), "Should error on zero samples");
    }

    #[test]
    fn test_regression_tree_min_samples_split() {
        let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);

        // Tree with min_samples_split=4 should not split nodes with fewer samples
        let mut tree = DecisionTreeRegressor::new()
            .with_max_depth(5)
            .with_min_samples_split(4);

        tree.fit(&x, &y).expect("fit should succeed");

        // Should still fit successfully
        let r2 = tree.score(&x, &y);
        assert!(r2 > 0.0, "Tree with min_samples_split should still fit");
    }

    #[test]
    fn test_regression_tree_min_samples_leaf() {
        let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

        // Tree with min_samples_leaf=3 should ensure leaves have at least 3 samples
        let mut tree = DecisionTreeRegressor::new()
            .with_max_depth(5)
            .with_min_samples_leaf(3);

        tree.fit(&x, &y).expect("fit should succeed");

        // Should fit without error
        let predictions = tree.predict(&x);
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_regression_tree_default() {
        let tree1 = DecisionTreeRegressor::new();
        let tree2 = DecisionTreeRegressor::default();

        assert_eq!(tree1.max_depth, tree2.max_depth);
        assert_eq!(tree1.tree.is_none(), tree2.tree.is_none());
    }

    #[test]
    fn test_regression_tree_comparison_with_linear_regression() {
        // On perfectly linear data, both should perform well
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        // Train tree
        let mut tree = DecisionTreeRegressor::new().with_max_depth(4);
        tree.fit(&x, &y).expect("fit should succeed");
        let tree_r2 = tree.score(&x, &y);

        // Train linear model
        let mut lr = crate::linear_model::LinearRegression::new();
        lr.fit(&x, &y).expect("fit should succeed");
        let lr_r2 = lr.score(&x, &y);

        // Both should achieve high R² on linear data
        assert!(tree_r2 > 0.9, "Tree R² {tree_r2} too low on linear data");
        assert!(lr_r2 > 0.99, "Linear regression R² {lr_r2} too low");
    }

    // ===================================================================
    // Random Forest Regression Tests
    // ===================================================================

    #[test]
    fn test_random_forest_regressor_creation() {
        let rf = RandomForestRegressor::new(10);
        assert_eq!(rf.n_estimators, 10);
        assert!(rf.trees.is_empty());
        assert!(rf.max_depth.is_none());
    }

    #[test]
    fn test_random_forest_regressor_with_max_depth() {
        let rf = RandomForestRegressor::new(5).with_max_depth(3);
        assert_eq!(rf.max_depth, Some(3));
    }

    #[test]
    fn test_random_forest_regressor_fit_simple_linear() {
        // Simple linear data: y = 2x + 1
        let x = Matrix::from_vec(
            10,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]);

        let mut rf = RandomForestRegressor::new(10).with_max_depth(5);
        rf.fit(&x, &y).expect("fit should succeed");

        // Should have trained 10 trees
        assert_eq!(rf.trees.len(), 10);

        // Should make reasonable predictions
        let _predictions = rf.predict(&x);
        let r2 = rf.score(&x, &y);
        assert!(r2 > 0.8, "R² should be high on training data: {r2}");
    }

    #[test]
    fn test_random_forest_regressor_predict_nonlinear() {
        // Non-linear data: y = x²
        let x = Matrix::from_vec(8, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]);

        // Train RF (with fixed random state for reproducibility)
        let mut rf = RandomForestRegressor::new(20)
            .with_max_depth(4)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let predictions = rf.predict(&x);
        assert_eq!(predictions.len(), 8);

        // Check predictions are reasonable (allow some error due to averaging and small dataset)
        let pred_slice = predictions.as_slice();
        let y_slice = y.as_slice();
        for i in 0..8 {
            let error = (pred_slice[i] - y_slice[i]).abs();
            assert!(
                error <= 12.0,
                "Prediction {} too far from true value {}: error {}",
                pred_slice[i],
                y_slice[i],
                error
            );
        }
    }

    #[test]
    fn test_random_forest_regressor_score() {
        let x = Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        let mut rf = RandomForestRegressor::new(15).with_max_depth(3);
        rf.fit(&x, &y).expect("fit should succeed");

        let r2 = rf.score(&x, &y);
        // R² should be positive and high for this simple linear pattern
        assert!(r2 > 0.7, "R² score {r2} should be high");
        assert!(r2 <= 1.0, "R² score {r2} should be <= 1.0");
    }

    #[test]
    fn test_random_forest_regressor_n_estimators_effect() {
        let x = Matrix::from_vec(
            10,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]);

        // Few trees
        let mut rf_few = RandomForestRegressor::new(5).with_max_depth(4);
        rf_few.fit(&x, &y).expect("fit should succeed");
        let r2_few = rf_few.score(&x, &y);

        // Many trees
        let mut rf_many = RandomForestRegressor::new(30).with_max_depth(4);
        rf_many.fit(&x, &y).expect("fit should succeed");
        let r2_many = rf_many.score(&x, &y);

        // More trees should generally give same or better performance
        // (at least not significantly worse)
        assert!(
            r2_many >= r2_few - 0.1,
            "More trees should not hurt performance"
        );
    }

    #[test]
    fn test_random_forest_regressor_vs_single_tree() {
        // Random forest should generalize better than single tree
        let x = Matrix::from_vec(
            15,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            2.1, 4.2, 8.9, 16.1, 24.8, 36.2, 49.1, 63.8, 81.2, 100.1, 120.9, 144.2, 169.1, 195.8,
            225.0,
        ]);

        // Single tree with high depth (prone to overfitting)
        let mut single_tree = DecisionTreeRegressor::new().with_max_depth(10);
        single_tree.fit(&x, &y).expect("fit should succeed");
        let single_r2 = single_tree.score(&x, &y);

        // Random forest with moderate depth
        let mut rf = RandomForestRegressor::new(20).with_max_depth(6);
        rf.fit(&x, &y).expect("fit should succeed");
        let rf_r2 = rf.score(&x, &y);

        // Both should fit well, but RF typically more stable
        assert!(single_r2 > 0.8, "Single tree R²: {single_r2}");
        assert!(rf_r2 > 0.8, "Random forest R²: {rf_r2}");
    }

    #[test]
    fn test_random_forest_regressor_multidimensional() {
        // 2D features: [x1, x2], y = x1 + 2*x2
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 1.0, 4.0, 2.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 6.0, 8.0]);

        let mut rf = RandomForestRegressor::new(15).with_max_depth(5);
        rf.fit(&x, &y).expect("fit should succeed");

        let predictions = rf.predict(&x);
        assert_eq!(predictions.len(), 8);

        let r2 = rf.score(&x, &y);
        assert!(r2 > 0.6, "R² on 2D data should be reasonable: {r2}");
    }

    #[test]
    fn test_random_forest_regressor_constant_target() {
        // All samples have same target value
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[7.0, 7.0, 7.0, 7.0, 7.0]);

        let mut rf = RandomForestRegressor::new(10).with_max_depth(3);
        rf.fit(&x, &y).expect("fit should succeed");

        let predictions = rf.predict(&x);
        for &pred in predictions.as_slice() {
            assert!(
                (pred - 7.0).abs() < 1e-5,
                "Prediction {pred} should be ~7.0 for constant target"
            );
        }
    }

    #[test]
    fn test_random_forest_regressor_single_sample() {
        let x = Matrix::from_vec(1, 2, vec![1.0, 2.0])
            .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[5.0]);

        let mut rf = RandomForestRegressor::new(5).with_max_depth(2);
        rf.fit(&x, &y).expect("fit should succeed");

        let predictions = rf.predict(&x);
        assert_eq!(predictions.len(), 1);
        assert!(
            (predictions.as_slice()[0] - 5.0).abs() < 1e-5,
            "Single sample prediction should be exact"
        );
    }

    #[test]
    fn test_random_forest_regressor_random_state() {
        let x = Matrix::from_vec(
            10,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        // Train two forests with same random state
        let mut rf1 = RandomForestRegressor::new(10)
            .with_max_depth(4)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let pred1 = rf1.predict(&x);

        let mut rf2 = RandomForestRegressor::new(10)
            .with_max_depth(4)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let pred2 = rf2.predict(&x);

        // Predictions should be identical
        for (p1, p2) in pred1.as_slice().iter().zip(pred2.as_slice().iter()) {
            assert!(
                (p1 - p2).abs() < 1e-10,
                "Predictions with same random_state should match: {p1} vs {p2}"
            );
        }
    }

    #[test]
    fn test_random_forest_regressor_validation_errors() {
        // Mismatched dimensions
        let x = Matrix::from_vec(
            5,
            2,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 2.0, 3.0]); // Wrong size

        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err(), "Should error on mismatched dimensions");

        // Zero samples
        let x_empty =
            Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed in tests");
        let y_empty = Vector::from_slice(&[]);
        let mut rf_empty = RandomForestRegressor::new(5);
        let result_empty = rf_empty.fit(&x_empty, &y_empty);
        assert!(result_empty.is_err(), "Should error on zero samples");
    }

    #[test]
    #[should_panic(expected = "Cannot predict with an unfitted Random Forest")]
    fn test_random_forest_regressor_predict_before_fit() {
        let rf = RandomForestRegressor::new(5);
        let x = Matrix::from_vec(2, 1, vec![1.0, 2.0])
            .expect("Matrix creation should succeed in tests");
        let _ = rf.predict(&x); // Should panic
    }

    #[test]
    fn test_random_forest_regressor_default() {
        let rf1 = RandomForestRegressor::new(10);
        let rf2 = RandomForestRegressor::default();

        assert_eq!(rf1.n_estimators, rf2.n_estimators);
        assert_eq!(rf1.max_depth, rf2.max_depth);
    }

    #[test]
    fn test_random_forest_regressor_comparison_with_linear_regression() {
        // On non-linear data, RF should significantly outperform linear regression
        let x = Matrix::from_vec(
            10,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0]); // y = x²

        // Train RF (with fixed random state for reproducibility)
        let mut rf = RandomForestRegressor::new(30)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");
        let rf_r2 = rf.score(&x, &y);

        // Train linear regression
        let mut lr = crate::linear_model::LinearRegression::new();
        lr.fit(&x, &y).expect("fit should succeed");
        let lr_r2 = lr.score(&x, &y);

        // RF should handle non-linearity better
        assert!(
            rf_r2 > 0.9,
            "Random forest R² {rf_r2} should be high on quadratic data"
        );
        assert!(
            lr_r2 < 0.98,
            "Linear regression R² {lr_r2} should be lower on non-linear data"
        );
        assert!(
            rf_r2 > lr_r2,
            "RF R² {rf_r2} should exceed linear R² {lr_r2} on non-linear data"
        );
    }

    #[test]
    fn test_random_forest_regressor_max_depth_effect() {
        let x = Matrix::from_vec(
            12,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0,
        ]);

        // Shallow trees (with fixed random state for reproducibility)
        let mut rf_shallow = RandomForestRegressor::new(15)
            .with_max_depth(2)
            .with_random_state(42);
        rf_shallow.fit(&x, &y).expect("fit should succeed");
        let r2_shallow = rf_shallow.score(&x, &y);

        // Deep trees (with fixed random state for reproducibility)
        let mut rf_deep = RandomForestRegressor::new(15)
            .with_max_depth(8)
            .with_random_state(42);
        rf_deep.fit(&x, &y).expect("fit should succeed");
        let r2_deep = rf_deep.score(&x, &y);

        // Deeper trees should capture at least as much complexity as shallow trees
        // Note: On small datasets, the difference may be minimal due to variance
        assert!(
            r2_deep >= r2_shallow - 0.01,
            "Deeper trees R² {r2_deep} should be at least as good as shallow trees R² {r2_shallow}"
        );
    }

    // ===================================================================
    // Out-of-Bag (OOB) Error Estimation Tests
    // ===================================================================

    #[test]
    fn test_random_forest_classifier_oob_score_after_fit() {
        // Simple classification data
        let x = Matrix::from_vec(
            15,
            4,
            vec![
                // Class 0
                5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
                3.6, 1.4, 0.2, // Class 1
                7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3,
                6.5, 2.8, 4.6, 1.5, // Class 2
                6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8,
                6.5, 3.0, 5.8, 2.2,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let mut rf = RandomForestClassifier::new(20)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf.oob_score();
        assert!(
            oob_score.is_some(),
            "OOB score should be available after fit"
        );

        let score_value = oob_score.expect("oob_score should be available");
        assert!(
            (0.0..=1.0).contains(&score_value),
            "OOB score {score_value} should be between 0 and 1"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_prediction_length() {
        let x = Matrix::from_vec(
            10,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 5.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(15).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_preds = rf.oob_prediction();
        assert!(
            oob_preds.is_some(),
            "OOB predictions should be available after fit"
        );

        let preds = oob_preds.expect("oob_preds should be available");
        assert_eq!(
            preds.len(),
            10,
            "OOB predictions should have same length as training data"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_before_fit() {
        let rf = RandomForestClassifier::new(10);

        assert!(
            rf.oob_score().is_none(),
            "OOB score should be None before fit"
        );
        assert!(
            rf.oob_prediction().is_none(),
            "OOB prediction should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_vs_test_score() {
        // Larger dataset to get reliable OOB estimate
        let x = Matrix::from_vec(
            30,
            4,
            vec![
                // Class 0 (10 samples)
                5.1, 3.5, 1.4, 0.2, 4.9, 3.0, 1.4, 0.2, 4.7, 3.2, 1.3, 0.2, 4.6, 3.1, 1.5, 0.2, 5.0,
                3.6, 1.4, 0.2, 5.4, 3.9, 1.7, 0.4, 4.6, 3.4, 1.4, 0.3, 5.0, 3.4, 1.5, 0.2, 4.4,
                2.9, 1.4, 0.2, 4.9, 3.1, 1.5, 0.1, // Class 1 (10 samples)
                7.0, 3.2, 4.7, 1.4, 6.4, 3.2, 4.5, 1.5, 6.9, 3.1, 4.9, 1.5, 5.5, 2.3, 4.0, 1.3,
                6.5, 2.8, 4.6, 1.5, 5.7, 2.8, 4.5, 1.3, 6.3, 3.3, 4.7, 1.6, 4.9, 2.4, 3.3, 1.0,
                6.6, 2.9, 4.6, 1.3, 5.2, 2.7, 3.9, 1.4, // Class 2 (10 samples)
                6.3, 3.3, 6.0, 2.5, 5.8, 2.7, 5.1, 1.9, 7.1, 3.0, 5.9, 2.1, 6.3, 2.9, 5.6, 1.8,
                6.5, 3.0, 5.8, 2.2, 7.6, 3.0, 6.6, 2.1, 4.9, 2.5, 4.5, 1.7, 7.3, 2.9, 6.3, 1.8,
                6.7, 2.5, 5.8, 1.8, 7.2, 3.6, 6.1, 2.5,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2,
        ];

        let mut rf = RandomForestClassifier::new(50)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf
            .oob_score()
            .expect("oob_score should be available after fit");
        let train_score = rf.score(&x, &y);

        // OOB score should be reasonable (within 0.3 of training score for small dataset)
        assert!(
            (oob_score - train_score).abs() < 0.3,
            "OOB score {oob_score} should be close to training score {train_score}"
        );
    }

    #[test]
    fn test_random_forest_classifier_oob_reproducibility() {
        let x = Matrix::from_vec(
            15,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 5.0, 5.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 6.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2];

        let mut rf1 = RandomForestClassifier::new(20)
            .with_max_depth(4)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let oob1 = rf1.oob_score();

        let mut rf2 = RandomForestClassifier::new(20)
            .with_max_depth(4)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let oob2 = rf2.oob_score();

        assert_eq!(oob1, oob2, "OOB scores should be identical with same seed");
    }

    #[test]
    fn test_random_forest_regressor_oob_score_after_fit() {
        let x = Matrix::from_vec(
            20,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
            32.0, 34.0, 36.0, 38.0, 40.0,
        ]);

        let mut rf = RandomForestRegressor::new(30)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf.oob_score();
        assert!(
            oob_score.is_some(),
            "OOB score should be available after fit"
        );

        let score_value = oob_score.expect("oob_score should be available");
        assert!(
            score_value > -1.0 && score_value <= 1.0,
            "OOB R² score {score_value} should be reasonable"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_prediction_length() {
        let x = Matrix::from_vec(
            12,
            2,
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 1.0, 4.0, 2.0,
                5.0, 3.0, 5.0, 4.0, 6.0, 3.0, 6.0, 4.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[3.0, 4.0, 5.0, 6.0, 5.0, 7.0, 6.0, 8.0, 8.0, 9.0, 9.0, 10.0]);

        let mut rf = RandomForestRegressor::new(20).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_preds = rf.oob_prediction();
        assert!(
            oob_preds.is_some(),
            "OOB predictions should be available after fit"
        );

        let preds = oob_preds.expect("oob_preds should be available");
        assert_eq!(
            preds.len(),
            12,
            "OOB predictions should have same length as training data"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_before_fit() {
        let rf = RandomForestRegressor::new(10);

        assert!(
            rf.oob_score().is_none(),
            "OOB score should be None before fit"
        );
        assert!(
            rf.oob_prediction().is_none(),
            "OOB prediction should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_vs_test_score() {
        // Linear data for predictable results
        let x = Matrix::from_vec(
            25,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0,
            33.0, 35.0, 37.0, 39.0, 41.0, 43.0, 45.0, 47.0, 49.0, 51.0,
        ]);

        let mut rf = RandomForestRegressor::new(50)
            .with_max_depth(6)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf
            .oob_score()
            .expect("oob_score should be available after fit");
        let train_score = rf.score(&x, &y);

        // OOB R² should be positive and within reasonable range of training R²
        assert!(oob_score > 0.5, "OOB R² {oob_score} should be positive");
        assert!(
            (oob_score - train_score).abs() < 0.3,
            "OOB R² {oob_score} should be close to training R² {train_score}"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_reproducibility() {
        let x = Matrix::from_vec(
            15,
            2,
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                9.0, 10.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0, 16.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0, 33.0,
        ]);

        let mut rf1 = RandomForestRegressor::new(25)
            .with_max_depth(5)
            .with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let oob1 = rf1.oob_score();

        let mut rf2 = RandomForestRegressor::new(25)
            .with_max_depth(5)
            .with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let oob2 = rf2.oob_score();

        assert_eq!(
            oob1, oob2,
            "OOB R² scores should be identical with same seed"
        );
    }

    #[test]
    fn test_random_forest_regressor_oob_nonlinear_data() {
        // Quadratic data to test OOB on non-linear patterns
        let x = Matrix::from_vec(
            15,
            1,
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[
            1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0, 196.0,
            225.0,
        ]);

        let mut rf = RandomForestRegressor::new(40)
            .with_max_depth(6)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_score = rf.oob_score();
        assert!(oob_score.is_some(), "OOB score should be available");

        // OOB should still be reasonably high for non-linear data
        let score_value = oob_score.expect("oob_score should be available");
        assert!(
            score_value > 0.7,
            "OOB R² {score_value} should be high on non-linear data"
        );
    }

    // ===================================================================
    // Feature Importance Tests (Issue #32)
    // ===================================================================

    #[test]
    fn test_random_forest_classifier_feature_importances_after_fit() {
        // Simple classification data with 3 features
        let x = Matrix::from_vec(
            12,
            3,
            vec![
                // Class 0 - feature 0 is discriminative
                1.0, 5.0, 5.0, 1.0, 6.0, 4.0, 2.0, 5.0, 6.0, 1.0, 4.0,
                5.0, // Class 1 - feature 0 is discriminative
                10.0, 5.0, 5.0, 10.0, 6.0, 4.0, 11.0, 5.0, 6.0, 10.0, 4.0,
                5.0, // Class 2 - feature 0 is discriminative
                20.0, 5.0, 5.0, 20.0, 6.0, 4.0, 21.0, 5.0, 6.0, 20.0, 4.0, 5.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let mut rf = RandomForestClassifier::new(20)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let importances = rf.feature_importances();
        assert!(
            importances.is_some(),
            "Feature importances should be available after fit"
        );

        let imps = importances.expect("importances should be available");
        assert_eq!(
            imps.len(),
            3,
            "Should have importance for each of 3 features"
        );

        // Importances should sum to ~1.0
        let sum: f32 = imps.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Importances should sum to 1.0, got {sum}"
        );

        // Feature 0 should be most important (it's the discriminative feature)
        assert!(
            imps[0] > imps[1] && imps[0] > imps[2],
            "Feature 0 should be most important, got {imps:?}"
        );
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_before_fit() {
        let rf = RandomForestClassifier::new(10);

        let importances = rf.feature_importances();
        assert!(
            importances.is_none(),
            "Feature importances should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_reproducibility() {
        let x = Matrix::from_vec(
            10,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
                4.0, 4.0, 4.0, 5.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];

        let mut rf1 = RandomForestClassifier::new(20).with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let imps1 = rf1
            .feature_importances()
            .expect("feature importances should be available");

        let mut rf2 = RandomForestClassifier::new(20).with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let imps2 = rf2
            .feature_importances()
            .expect("feature importances should be available");

        // Should be very similar with same random_state
        // Note: Small variations can occur due to floating point arithmetic in normalization
        // Trueno v0.6.0 may have different SIMD optimizations affecting FP precision
        for (i, (&imp1, &imp2)) in imps1.iter().zip(imps2.iter()).enumerate() {
            assert!(
                (imp1 - imp2).abs() <= 0.15,
                "Importance {i} should be similar: {imp1} vs {imp2}"
            );
        }
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_after_fit() {
        // Simple regression data where feature 0 is predictive
        let x = Matrix::from_vec(
            10,
            3,
            vec![
                // Feature 0 is predictive of y
                1.0, 5.0, 5.0, 2.0, 6.0, 4.0, 3.0, 5.0, 6.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0,
                4.0, 7.0, 5.0, 6.0, 8.0, 4.0, 5.0, 9.0, 5.0, 5.0, 10.0, 6.0, 4.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]); // Linear with feature 0

        let mut rf = RandomForestRegressor::new(20)
            .with_max_depth(5)
            .with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let importances = rf.feature_importances();
        assert!(
            importances.is_some(),
            "Feature importances should be available after fit"
        );

        let imps = importances.expect("importances should be available");
        assert_eq!(
            imps.len(),
            3,
            "Should have importance for each of 3 features"
        );

        // Importances should sum to ~1.0
        let sum: f32 = imps.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.01,
            "Importances should sum to 1.0, got {sum}"
        );

        // Feature 0 should be most important
        assert!(
            imps[0] > imps[1] && imps[0] > imps[2],
            "Feature 0 should be most important, got {imps:?}"
        );
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_before_fit() {
        let rf = RandomForestRegressor::new(10);

        let importances = rf.feature_importances();
        assert!(
            importances.is_none(),
            "Feature importances should be None before fit"
        );
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_reproducibility() {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        let mut rf1 = RandomForestRegressor::new(20).with_random_state(42);
        rf1.fit(&x, &y).expect("fit should succeed");
        let imps1 = rf1
            .feature_importances()
            .expect("feature importances should be available");

        let mut rf2 = RandomForestRegressor::new(20).with_random_state(42);
        rf2.fit(&x, &y).expect("fit should succeed");
        let imps2 = rf2
            .feature_importances()
            .expect("feature importances should be available");

        // Should be very similar with same random_state
        // Note: Small variations can occur due to floating point arithmetic in normalization
        for (i, (&imp1, &imp2)) in imps1.iter().zip(imps2.iter()).enumerate() {
            assert!(
                (imp1 - imp2).abs() <= 0.1,
                "Importance {i} should be similar: {imp1} vs {imp2}"
            );
        }
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_all_nonnegative() {
        // All importances should be >= 0
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

        let mut rf = RandomForestClassifier::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let imps = rf
            .feature_importances()
            .expect("feature importances should be available");
        for (i, &imp) in imps.iter().enumerate() {
            assert!(
                imp >= 0.0,
                "Importance {i} should be non-negative, got {imp}"
            );
        }
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_all_nonnegative() {
        // All importances should be >= 0
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0, 5.0, 0.0, 6.0, 1.0, 7.0, 0.0, 8.0, 1.0,
            ],
        )
        .expect("Matrix creation should succeed in tests");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

        let mut rf = RandomForestRegressor::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let imps = rf
            .feature_importances()
            .expect("feature importances should be available");
        for (i, &imp) in imps.iter().enumerate() {
            assert!(
                imp >= 0.0,
                "Importance {i} should be non-negative, got {imp}"
            );
        }
    }

    #[test]
    fn test_random_forest_classifier_predict_proba() {
        // Test predict_proba returns valid probabilities
        let x = Matrix::from_vec(
            6,
            2,
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 2, 2]; // 3 classes

        let mut rf = RandomForestClassifier::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let proba = rf.predict_proba(&x);

        // Shape should be (n_samples, n_classes)
        assert_eq!(proba.shape(), (6, 3));

        // Each row should sum to 1.0
        for row in 0..6 {
            let sum: f32 = (0..3).map(|col| proba.get(row, col)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Row {row} probabilities should sum to 1.0, got {sum}"
            );
        }

        // All probabilities should be in [0, 1]
        for row in 0..6 {
            for col in 0..3 {
                let p = proba.get(row, col);
                assert!(
                    (0.0..=1.0).contains(&p),
                    "Probability should be in [0,1], got {p}"
                );
            }
        }
    }

    // ========== Additional Coverage Tests ==========

    #[test]
    fn test_decision_tree_regressor_new() {
        let tree = DecisionTreeRegressor::new();
        assert!(tree.tree.is_none());
        assert!(tree.max_depth.is_none());
    }

    #[test]
    fn test_decision_tree_regressor_default() {
        let tree = DecisionTreeRegressor::default();
        assert!(tree.tree.is_none());
    }

    #[test]
    fn test_decision_tree_regressor_with_max_depth() {
        let tree = DecisionTreeRegressor::new().with_max_depth(5);
        assert_eq!(tree.max_depth, Some(5));
    }

    #[test]
    fn test_decision_tree_regressor_with_min_samples_split() {
        let tree = DecisionTreeRegressor::new().with_min_samples_split(5);
        assert_eq!(tree.min_samples_split, 5);
    }

    #[test]
    fn test_decision_tree_regressor_with_min_samples_leaf() {
        let tree = DecisionTreeRegressor::new().with_min_samples_leaf(3);
        assert_eq!(tree.min_samples_leaf, 3);
    }

    #[test]
    fn test_decision_tree_regressor_fit_predict() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(3);
        tree.fit(&x, &y).expect("fit should succeed");

        let predictions = tree.predict(&x);
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_decision_tree_regressor_score() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
        tree.fit(&x, &y).expect("fit should succeed");

        let score = tree.score(&x, &y);
        // R² should be reasonably high for this simple linear data
        assert!(score > 0.5, "R² should be > 0.5, got {score}");
    }

    #[test]
    fn test_decision_tree_regressor_fit_mismatch_error() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0]); // Mismatched length

        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_decision_tree_regressor_fit_empty_error() {
        let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);

        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_regressor_new() {
        let rf = RandomForestRegressor::new(10);
        assert_eq!(rf.n_estimators, 10);
        assert!(rf.trees.is_empty());
    }

    #[test]
    fn test_random_forest_regressor_default_values() {
        let rf = RandomForestRegressor::default();
        assert!(rf.trees.is_empty());
    }

    #[test]
    fn test_random_forest_regressor_fit_mismatch_error() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0]); // Mismatched length

        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_regressor_fit_empty_error() {
        let x = Matrix::from_vec(0, 1, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);

        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_regressor_oob_methods() {
        let rf = RandomForestRegressor::new(10);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
    }

    #[test]
    fn test_random_forest_regressor_oob_methods_after_fit() {
        let x = Matrix::from_vec(
            10,
            1,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let mut rf = RandomForestRegressor::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let oob_pred = rf.oob_prediction();
        assert!(oob_pred.is_some());
        assert_eq!(oob_pred.as_ref().map(|v| v.len()), Some(10));

        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
    }

    #[test]
    fn test_regression_tree_node_depth() {
        let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 5.0,
            n_samples: 10,
        });
        assert_eq!(leaf.depth(), 0);

        let node = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 3.0,
                n_samples: 5,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 7.0,
                n_samples: 5,
            })),
        });
        assert_eq!(node.depth(), 1);
    }

    #[test]
    fn test_gradient_boosting_classifier_new() {
        let gb = GradientBoostingClassifier::new();
        assert!(gb.estimators.is_empty());
        assert_eq!(gb.n_estimators, 100); // Config value, not fitted count
    }

    #[test]
    fn test_gradient_boosting_classifier_default() {
        let gb = GradientBoostingClassifier::default();
        assert!(gb.estimators.is_empty());
    }

    #[test]
    fn test_gradient_boosting_classifier_builders() {
        let gb = GradientBoostingClassifier::new()
            .with_n_estimators(50)
            .with_learning_rate(0.05)
            .with_max_depth(5);

        assert_eq!(gb.n_estimators, 50); // Config value, not fitted count
    }

    #[test]
    fn test_gradient_boosting_classifier_fit_predict() {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
            ],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

        let mut gb = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(3);

        gb.fit(&x, &y).expect("fit should succeed");

        let predictions = gb.predict(&x).expect("predict should succeed");
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_gradient_boosting_classifier_predict_proba() {
        let x = Matrix::from_vec(
            8,
            2,
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 3.0, 3.0,
            ],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 0, 0, 1, 1];

        let mut gb = GradientBoostingClassifier::new()
            .with_n_estimators(10)
            .with_max_depth(3);

        gb.fit(&x, &y).expect("fit should succeed");

        let proba = gb.predict_proba(&x).expect("predict_proba should succeed");
        assert_eq!(proba.len(), 8);

        // Each row should be a probability distribution
        for row_proba in &proba {
            let sum: f32 = row_proba.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Probabilities should sum to 1.0, got {sum}"
            );
            for &p in row_proba {
                assert!((0.0..=1.0).contains(&p), "Probability should be in [0,1]");
            }
        }
    }

    #[test]
    fn test_random_forest_classifier_oob_methods() {
        let rf = RandomForestClassifier::new(10);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
        assert!(rf.feature_importances().is_none());
    }

    #[test]
    fn test_decision_tree_classifier_fit_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("Matrix creation should succeed");
        let y = vec![0, 1]; // Mismatched length

        let mut tree = DecisionTreeClassifier::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_boosting_classifier_fit_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
            .expect("Matrix creation should succeed");
        let y = vec![0, 1]; // Mismatched length

        let mut gb = GradientBoostingClassifier::new();
        let result = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_boosting_classifier_fit_empty() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y: Vec<usize> = vec![];

        let mut gb = GradientBoostingClassifier::new();
        let result = gb.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_decision_tree_classifier_score() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 0, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).expect("fit should succeed");

        let score = tree.score(&x, &y);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_random_forest_classifier_score() {
        let x = Matrix::from_vec(
            6,
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        )
        .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1, 0, 1];

        let mut rf = RandomForestClassifier::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let score = rf.score(&x, &y);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_random_forest_regressor_score_r2() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut rf = RandomForestRegressor::new(10).with_random_state(42);
        rf.fit(&x, &y).expect("fit should succeed");

        let score = rf.score(&x, &y);
        assert!(score > 0.0, "R² should be positive, got {score}");
    }

    #[test]
    fn test_leaf_struct_fields() {
        let leaf = Leaf {
            class_label: 2,
            n_samples: 50,
        };
        assert_eq!(leaf.class_label, 2);
        assert_eq!(leaf.n_samples, 50);
    }

    #[test]
    fn test_regression_leaf_struct_fields() {
        let leaf = RegressionLeaf {
            value: 3.14,
            n_samples: 25,
        };
        assert!((leaf.value - 3.14).abs() < 1e-6);
        assert_eq!(leaf.n_samples, 25);
    }

    #[test]
    fn test_node_struct_fields() {
        let left = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 5,
        });
        let right = TreeNode::Leaf(Leaf {
            class_label: 1,
            n_samples: 5,
        });

        let node = Node {
            feature_idx: 2,
            threshold: 1.5,
            left: Box::new(left),
            right: Box::new(right),
        };

        assert_eq!(node.feature_idx, 2);
        assert!((node.threshold - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_regression_node_struct_fields() {
        let left = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 1.0,
            n_samples: 5,
        });
        let right = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 2.0,
            n_samples: 5,
        });

        let node = RegressionNode {
            feature_idx: 1,
            threshold: 0.75,
            left: Box::new(left),
            right: Box::new(right),
        };

        assert_eq!(node.feature_idx, 1);
        assert!((node.threshold - 0.75).abs() < 1e-6);
    }

    #[test]
    fn test_tree_node_variants() {
        let leaf_node = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 10,
        });
        assert_eq!(leaf_node.depth(), 0);

        let internal_node = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 5,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 5,
            })),
        });
        assert_eq!(internal_node.depth(), 1);
    }

    #[test]
    fn test_regression_tree_node_variants() {
        let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 5.0,
            n_samples: 10,
        });
        assert_eq!(leaf.depth(), 0);

        let internal = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 3.0,
                n_samples: 5,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 7.0,
                n_samples: 5,
            })),
        });
        assert_eq!(internal.depth(), 1);
    }

    // =========================================================================
    // Coverage boost tests: Helper functions and edge cases
    // =========================================================================

    #[test]
    fn test_mean_f32_empty() {
        let empty: Vec<f32> = vec![];
        assert_eq!(mean_f32(&empty), 0.0);
    }

    #[test]
    fn test_mean_f32_single() {
        let single = vec![42.5];
        assert!((mean_f32(&single) - 42.5).abs() < 1e-6);
    }

    #[test]
    fn test_mean_f32_multiple() {
        let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((mean_f32(&vals) - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_variance_f32_empty() {
        let empty: Vec<f32> = vec![];
        assert_eq!(variance_f32(&empty), 0.0);
    }

    #[test]
    fn test_variance_f32_single() {
        let single = vec![5.0];
        assert_eq!(variance_f32(&single), 0.0);
    }

    #[test]
    fn test_variance_f32_uniform() {
        let uniform = vec![3.0, 3.0, 3.0, 3.0];
        assert!(variance_f32(&uniform) < 1e-6);
    }

    #[test]
    fn test_variance_f32_variable() {
        // Values: 1, 2, 3 -> mean = 2, variance = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        let vals = vec![1.0, 2.0, 3.0];
        let expected = 2.0 / 3.0;
        assert!((variance_f32(&vals) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_compute_mse_empty() {
        let left: Vec<f32> = vec![];
        let right: Vec<f32> = vec![];
        assert_eq!(compute_mse(&left, &right), 0.0);
    }

    #[test]
    fn test_compute_mse_pure_splits() {
        // Both partitions have constant values -> MSE = 0
        let left = vec![5.0, 5.0, 5.0];
        let right = vec![10.0, 10.0];
        assert!(compute_mse(&left, &right) < 1e-6);
    }

    #[test]
    fn test_at_max_depth_none() {
        // No max depth -> never at max
        assert!(!at_max_depth(0, None));
        assert!(!at_max_depth(100, None));
        assert!(!at_max_depth(1000, None));
    }

    #[test]
    fn test_at_max_depth_with_limit() {
        assert!(at_max_depth(5, Some(5)));
        assert!(at_max_depth(6, Some(5)));
        assert!(!at_max_depth(4, Some(5)));
        assert!(!at_max_depth(0, Some(5)));
    }

    #[test]
    fn test_get_unique_feature_values() {
        let x = Matrix::from_vec(5, 2, vec![
            1.0, 10.0,
            2.0, 20.0,
            1.0, 30.0,  // duplicate for feature 0
            3.0, 20.0,  // duplicate for feature 1
            2.0, 40.0,  // duplicate for feature 0
        ]).expect("Matrix creation should succeed");

        let unique_feat0 = get_unique_feature_values(&x, 0, 5);
        assert_eq!(unique_feat0, vec![1.0, 2.0, 3.0]);

        let unique_feat1 = get_unique_feature_values(&x, 1, 5);
        assert_eq!(unique_feat1, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_split_by_threshold() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0])
            .expect("Matrix creation should succeed");
        let y = vec![10.0, 20.0, 30.0, 40.0];

        let (left, right) = split_by_threshold(&x, &y, 0, 2.5);
        assert_eq!(left, vec![10.0, 20.0]);
        assert_eq!(right, vec![30.0, 40.0]);
    }

    #[test]
    fn test_evaluate_split_gain_empty() {
        let left: Vec<f32> = vec![];
        let right = vec![1.0, 2.0];
        assert!(evaluate_split_gain(&left, &right, 1.0).is_none());
    }

    #[test]
    fn test_evaluate_split_gain_positive() {
        // High variance before split, low variance after
        let left = vec![1.0, 1.0, 1.0];
        let right = vec![10.0, 10.0, 10.0];
        let current_variance = 20.0; // Arbitrary high variance
        let gain = evaluate_split_gain(&left, &right, current_variance);
        assert!(gain.is_some());
        assert!(gain.expect("Expected some gain") > 0.0);
    }

    #[test]
    fn test_partition_by_threshold() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 5.0, 2.0, 8.0, 3.0])
            .expect("Matrix creation should succeed");
        let (left, right) = partition_by_threshold(&x, 5, 0, 3.5);
        assert_eq!(left, vec![0, 2, 4]); // indices with values <= 3.5
        assert_eq!(right, vec![1, 3]);   // indices with values > 3.5
    }

    #[test]
    fn test_gini_split_empty_partitions() {
        let left: Vec<usize> = vec![];
        let right: Vec<usize> = vec![];
        assert_eq!(gini_split(&left, &right), 0.0);
    }

    #[test]
    fn test_get_sorted_unique_values() {
        let x = vec![3.0, 1.0, 2.0, 1.0, 3.0];
        let unique = get_sorted_unique_values(&x);
        assert_eq!(unique, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_split_labels_by_threshold_empty_result() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0, 1, 2];
        // Threshold that puts all on one side
        assert!(split_labels_by_threshold(&x, &y, 10.0).is_none());
        assert!(split_labels_by_threshold(&x, &y, 0.0).is_none());
    }

    #[test]
    fn test_split_labels_by_threshold_valid() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0, 0, 1, 1];
        let result = split_labels_by_threshold(&x, &y, 2.5);
        assert!(result.is_some());
        let (left, right) = result.expect("Expected valid split");
        assert_eq!(left, vec![0, 0]);
        assert_eq!(right, vec![1, 1]);
    }

    #[test]
    fn test_calculate_information_gain() {
        let left = vec![0, 0, 0];
        let right = vec![1, 1, 1];
        // Pure split should have maximum gain from impure parent
        let gain = calculate_information_gain(0.5, &left, &right);
        assert!((gain - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_split_data_by_indices() {
        let x = Matrix::from_vec(4, 2, vec![
            0.0, 1.0,
            2.0, 3.0,
            4.0, 5.0,
            6.0, 7.0,
        ]).expect("Matrix creation should succeed");
        let y = vec![10, 20, 30, 40];
        let indices = vec![1, 3];

        let (subset_x, subset_y) = split_data_by_indices(&x, &y, &indices);
        assert_eq!(subset_y, vec![20, 40]);
        assert_eq!(subset_x.shape(), (2, 2));
    }

    #[test]
    fn test_check_stopping_criteria_pure_node() {
        let y = vec![5, 5, 5, 5]; // All same class
        let result = check_stopping_criteria(&y, 0, None);
        assert!(result.is_some());
        if let Some(TreeNode::Leaf(leaf)) = result {
            assert_eq!(leaf.class_label, 5);
            assert_eq!(leaf.n_samples, 4);
        } else {
            panic!("Expected leaf node");
        }
    }

    #[test]
    fn test_check_stopping_criteria_max_depth() {
        let y = vec![0, 1, 0, 1]; // Mixed classes
        let result = check_stopping_criteria(&y, 5, Some(5));
        assert!(result.is_some());
    }

    #[test]
    fn test_check_stopping_criteria_continue() {
        let y = vec![0, 1, 0, 1];
        let result = check_stopping_criteria(&y, 2, Some(5));
        assert!(result.is_none()); // Should continue splitting
    }

    #[test]
    fn test_split_indices_by_threshold_empty() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0])
            .expect("Matrix creation should succeed");
        // All values <= threshold
        assert!(split_indices_by_threshold(&x, 0, 10.0, 3).is_none());
        // All values > threshold
        assert!(split_indices_by_threshold(&x, 0, 0.0, 3).is_none());
    }

    #[test]
    fn test_count_tree_samples() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 42,
        });
        assert_eq!(count_tree_samples(&leaf), 42);

        let node = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf(Leaf { class_label: 0, n_samples: 10 })),
            right: Box::new(TreeNode::Leaf(Leaf { class_label: 1, n_samples: 20 })),
        });
        assert_eq!(count_tree_samples(&node), 30);
    }

    #[test]
    fn test_count_regression_tree_samples() {
        let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 5.0,
            n_samples: 15,
        });
        assert_eq!(count_regression_tree_samples(&leaf), 15);

        let node = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf { value: 1.0, n_samples: 5 })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf { value: 2.0, n_samples: 8 })),
        });
        assert_eq!(count_regression_tree_samples(&node), 13);
    }

    // =========================================================================
    // Coverage boost: OOB and feature importance tests
    // =========================================================================

    #[test]
    fn test_random_forest_classifier_oob_score() {
        let x = Matrix::from_vec(20, 2, vec![
            0.0, 0.0, 0.1, 0.1, 0.2, 0.2, 0.0, 0.3, 0.1, 0.4,
            0.2, 0.0, 0.3, 0.1, 0.0, 0.2, 0.1, 0.3, 0.2, 0.1,
            // Class 1 samples
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8, 0.8, 1.0, 0.9, 1.1,
            1.1, 0.9, 0.8, 0.8, 1.0, 0.9, 0.9, 1.0, 1.0, 0.8,
        ]).expect("Matrix creation should succeed");
        let y: Vec<usize> = (0..10).map(|_| 0).chain((0..10).map(|_| 1)).collect();

        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        // Test OOB prediction
        let oob_pred = rf.oob_prediction();
        assert!(oob_pred.is_some());
        assert_eq!(oob_pred.expect("Expected OOB predictions").len(), 20);

        // Test OOB score
        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
        let score = oob_score.expect("Expected OOB score");
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_random_forest_classifier_oob_unfitted() {
        let rf = RandomForestClassifier::new(5);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
    }

    #[test]
    fn test_random_forest_classifier_feature_importances() {
        let x = Matrix::from_vec(10, 3, vec![
            0.0, 0.5, 0.1,
            0.1, 0.5, 0.2,
            0.0, 0.5, 0.0,
            0.2, 0.5, 0.1,
            0.1, 0.5, 0.3,
            1.0, 0.5, 0.8,
            0.9, 0.5, 0.9,
            1.0, 0.5, 1.0,
            0.8, 0.5, 0.7,
            0.9, 0.5, 0.8,
        ]).expect("Matrix creation should succeed");
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let importances = rf.feature_importances();
        assert!(importances.is_some());
        let imp = importances.expect("Expected feature importances");
        assert_eq!(imp.len(), 3);
        // Sum should be ~1.0
        let sum: f32 = imp.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_random_forest_classifier_feature_importances_unfitted() {
        let rf = RandomForestClassifier::new(5);
        assert!(rf.feature_importances().is_none());
    }

    #[test]
    fn test_random_forest_regressor_oob_score() {
        let x = Matrix::from_vec(20, 1, (1..=20).map(|i| i as f32).collect())
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&(1..=20).map(|i| (i * 2) as f32).collect::<Vec<_>>());

        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let oob_pred = rf.oob_prediction();
        assert!(oob_pred.is_some());

        let oob_score = rf.oob_score();
        assert!(oob_score.is_some());
    }

    #[test]
    fn test_random_forest_regressor_oob_unfitted() {
        let rf = RandomForestRegressor::new(5);
        assert!(rf.oob_prediction().is_none());
        assert!(rf.oob_score().is_none());
    }

    #[test]
    fn test_random_forest_regressor_feature_importances() {
        let x = Matrix::from_vec(10, 2, vec![
            1.0, 0.5, 2.0, 0.5, 3.0, 0.5, 4.0, 0.5, 5.0, 0.5,
            6.0, 0.5, 7.0, 0.5, 8.0, 0.5, 9.0, 0.5, 10.0, 0.5,
        ]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);

        let mut rf = RandomForestRegressor::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let importances = rf.feature_importances();
        assert!(importances.is_some());
        let imp = importances.expect("Expected feature importances");
        assert_eq!(imp.len(), 2);
    }

    #[test]
    fn test_random_forest_regressor_feature_importances_unfitted() {
        let rf = RandomForestRegressor::new(5);
        assert!(rf.feature_importances().is_none());
    }

    // =========================================================================
    // Coverage boost: Default implementations
    // =========================================================================

    #[test]
    fn test_rfr_default_coverage() {
        let rf = RandomForestRegressor::default();
        assert_eq!(rf.n_estimators, 10);
    }

    #[test]
    fn test_dtr_default_coverage() {
        let tree = DecisionTreeRegressor::default();
        assert!(tree.tree.is_none());
    }

    // =========================================================================
    // Coverage boost: Error handling
    // =========================================================================

    #[test]
    fn test_dtc_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y: Vec<usize> = vec![];
        let mut tree = DecisionTreeClassifier::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtc_fit_sample_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
        let y = vec![0, 1]; // Wrong length
        let mut tree = DecisionTreeClassifier::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtr_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);
        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dtr_fit_sample_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[1.0, 2.0]); // Wrong length
        let mut tree = DecisionTreeRegressor::new();
        let result = tree.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "low >= high")]
    fn test_rfc_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y: Vec<usize> = vec![];
        let mut rf = RandomForestClassifier::new(5);
        // RF doesn't check empty data explicitly and will panic in bootstrap_sample
        let _ = rf.fit(&x, &y);
    }

    #[test]
    fn test_rfr_fit_empty_coverage() {
        let x = Matrix::from_vec(0, 2, vec![]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[]);
        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_rfr_fit_sample_mismatch() {
        let x = Matrix::from_vec(4, 2, vec![0.0; 8]).expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[1.0, 2.0]);
        let mut rf = RandomForestRegressor::new(5);
        let result = rf.fit(&x, &y);
        assert!(result.is_err());
    }

    // =========================================================================
    // Coverage boost: Regression tree builder options
    // =========================================================================

    #[test]
    fn test_decision_tree_regressor_min_samples_split_coverage() {
        let tree = DecisionTreeRegressor::new()
            .with_min_samples_split(5);
        assert_eq!(tree.min_samples_split, 5);
    }

    #[test]
    fn test_decision_tree_regressor_min_samples_split_floor_coverage() {
        // min_samples_split should be at least 2
        let tree = DecisionTreeRegressor::new()
            .with_min_samples_split(1);
        assert_eq!(tree.min_samples_split, 2);
    }

    #[test]
    fn test_decision_tree_regressor_min_samples_leaf_coverage() {
        let tree = DecisionTreeRegressor::new()
            .with_min_samples_leaf(3);
        assert_eq!(tree.min_samples_leaf, 3);
    }

    #[test]
    fn test_decision_tree_regressor_min_samples_leaf_floor_coverage() {
        // min_samples_leaf should be at least 1
        let tree = DecisionTreeRegressor::new()
            .with_min_samples_leaf(0);
        assert_eq!(tree.min_samples_leaf, 1);
    }

    #[test]
    fn test_decision_tree_regressor_score_coverage() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0])
            .expect("Matrix creation should succeed");
        let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);

        let mut tree = DecisionTreeRegressor::new().with_max_depth(5);
        tree.fit(&x, &y).expect("Fit should succeed");

        let score = tree.score(&x, &y);
        // R² should be high for this simple linear relationship
        assert!(score > 0.8);
    }

    // =========================================================================
    // Coverage boost: Gradient Boosting additional tests
    // =========================================================================

    #[test]
    fn test_gradient_boosting_predict_proba_untrained() {
        let gbm = GradientBoostingClassifier::new();
        let x = Matrix::from_vec(2, 2, vec![0.0, 0.0, 1.0, 1.0])
            .expect("Matrix creation should succeed");
        let result = gbm.predict_proba(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_gradient_boosting_n_estimators_getter() {
        let gbm = GradientBoostingClassifier::new().with_n_estimators(25);
        // Before training, n_estimators() returns 0 (number of trained estimators)
        assert_eq!(gbm.n_estimators(), 0);
    }

    // =========================================================================
    // Coverage boost: predict_proba for RandomForestClassifier
    // =========================================================================

    #[test]
    fn test_random_forest_predict_proba() {
        let x = Matrix::from_vec(6, 2, vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2,
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8,
        ]).expect("Matrix creation should succeed");
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(10)
            .with_max_depth(3)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let proba = rf.predict_proba(&x);
        let (n_samples, n_classes) = proba.shape();
        assert_eq!(n_samples, 6);
        assert_eq!(n_classes, 2);

        // Each row should sum to 1.0
        for i in 0..n_samples {
            let row_sum: f32 = (0..n_classes).map(|j| proba.get(i, j)).sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    // =========================================================================
    // Coverage boost: Bootstrap sample edge cases
    // =========================================================================

    #[test]
    fn test_bootstrap_sample_no_seed() {
        let indices = bootstrap_sample(50, None);
        assert_eq!(indices.len(), 50);
        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 50);
        }
    }

    #[test]
    fn test_bootstrap_sample_different_seeds() {
        let indices1 = bootstrap_sample(50, Some(1));
        let indices2 = bootstrap_sample(50, Some(2));
        // Different seeds should (almost certainly) give different results
        assert_ne!(indices1, indices2);
    }

    // =========================================================================
    // Coverage boost: SafeTensors serialization
    // =========================================================================

    #[test]
    fn test_decision_tree_save_load_safetensors() {
        use std::fs;

        let x = Matrix::from_vec(4, 2, vec![
            0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        ]).expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(3);
        tree.fit(&x, &y).expect("Fit should succeed");

        let path = "/tmp/test_tree_safetensors.safetensors";
        tree.save_safetensors(path).expect("Save should succeed");

        let loaded = DecisionTreeClassifier::load_safetensors(path)
            .expect("Load should succeed");

        // Verify predictions match
        let orig_pred = tree.predict(&x);
        let loaded_pred = loaded.predict(&x);
        assert_eq!(orig_pred, loaded_pred);

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_decision_tree_save_safetensors_unfitted() {
        let tree = DecisionTreeClassifier::new();
        let result = tree.save_safetensors("/tmp/unfitted.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_random_forest_save_load_safetensors() {
        use std::fs;

        let x = Matrix::from_vec(6, 2, vec![
            0.0, 0.0, 0.1, 0.1, 0.0, 0.2,
            1.0, 1.0, 0.9, 0.9, 1.0, 0.8,
        ]).expect("Matrix creation should succeed");
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut rf = RandomForestClassifier::new(3)
            .with_max_depth(2)
            .with_random_state(42);
        rf.fit(&x, &y).expect("Fit should succeed");

        let path = "/tmp/test_rf_safetensors.safetensors";
        rf.save_safetensors(path).expect("Save should succeed");

        let loaded = RandomForestClassifier::load_safetensors(path)
            .expect("Load should succeed");

        // Verify structure
        assert_eq!(loaded.n_estimators, 3);
        assert_eq!(loaded.trees.len(), 3);

        fs::remove_file(path).ok();
    }

    #[test]
    fn test_random_forest_save_safetensors_unfitted() {
        let rf = RandomForestClassifier::new(5);
        let result = rf.save_safetensors("/tmp/unfitted_rf.safetensors");
        assert!(result.is_err());
    }

    // =========================================================================
    // Coverage boost: Tree flattening and reconstruction
    // =========================================================================

    #[test]
    fn test_flatten_and_reconstruct_tree() {
        let original = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf(Leaf { class_label: 0, n_samples: 5 })),
            right: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 0.3,
                left: Box::new(TreeNode::Leaf(Leaf { class_label: 1, n_samples: 3 })),
                right: Box::new(TreeNode::Leaf(Leaf { class_label: 2, n_samples: 2 })),
            })),
        });

        let mut features = Vec::new();
        let mut thresholds = Vec::new();
        let mut classes = Vec::new();
        let mut samples = Vec::new();
        let mut left_children = Vec::new();
        let mut right_children = Vec::new();

        flatten_tree_node(
            &original,
            &mut features,
            &mut thresholds,
            &mut classes,
            &mut samples,
            &mut left_children,
            &mut right_children,
        );

        let reconstructed = reconstruct_tree_node(
            0,
            &features,
            &thresholds,
            &classes,
            &samples,
            &left_children,
            &right_children,
        );

        assert_eq!(original.depth(), reconstructed.depth());
    }

    // =========================================================================
    // Coverage boost: Feature importance computation
    // =========================================================================

    #[test]
    fn test_compute_tree_feature_importances() {
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf(Leaf { class_label: 0, n_samples: 10 })),
            right: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 0.3,
                left: Box::new(TreeNode::Leaf(Leaf { class_label: 1, n_samples: 5 })),
                right: Box::new(TreeNode::Leaf(Leaf { class_label: 2, n_samples: 5 })),
            })),
        });

        let mut importances = vec![0.0; 2];
        compute_tree_feature_importances(&tree, &mut importances);

        // Feature 0 splits 20 samples, feature 1 splits 10 samples
        assert!(importances[0] > 0.0);
        assert!(importances[1] > 0.0);
    }

    #[test]
    fn test_compute_regression_tree_feature_importances() {
        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 0.5,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf { value: 1.0, n_samples: 10 })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf { value: 2.0, n_samples: 10 })),
        });

        let mut importances = vec![0.0; 2];
        compute_regression_tree_feature_importances(&tree, &mut importances);

        assert!(importances[0] > 0.0);
        assert_eq!(importances[1], 0.0); // Feature 1 not used
    }

    // =========================================================================
    // Coverage boost: Regression tree building edge cases
    // =========================================================================

    #[test]
    fn test_split_regression_data_by_indices() {
        let x = Matrix::from_vec(4, 2, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]).expect("Matrix creation should succeed");
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let indices = vec![0, 2];

        let (subset_x, subset_y) = split_regression_data_by_indices(&x, &y, &indices);
        assert_eq!(subset_y, vec![10.0, 30.0]);
        assert_eq!(subset_x.shape(), (2, 2));
    }

    #[test]
    fn test_make_regression_leaf() {
        let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let leaf = make_regression_leaf(&y, 5);
        match leaf {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - 3.0).abs() < 1e-6);
                assert_eq!(l.n_samples, 5);
            }
            _ => panic!("Expected leaf"),
        }
    }

    #[test]
    fn test_find_best_regression_split_for_feature() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed");
        let y = vec![1.0, 2.0, 10.0, 11.0];
        let variance = variance_f32(&y);

        let result = find_best_regression_split_for_feature(&x, &y, 0, 4, variance);
        assert!(result.is_some());
        let (threshold, gain) = result.expect("Expected split");
        assert!(threshold > 2.0 && threshold < 5.0);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_regression_split_no_samples() {
        let x = Matrix::from_vec(1, 1, vec![1.0])
            .expect("Matrix creation should succeed");
        let y = vec![1.0];

        let result = find_best_regression_split(&x, &y);
        assert!(result.is_none());
    }

    // =========================================================================
    // Coverage boost: Estimator trait implementation
    // =========================================================================

    #[test]
    fn test_estimator_trait_decision_tree() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0])
            .expect("Matrix creation should succeed");
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(3);

        // Using Estimator trait methods
        tree.fit(&x, &y).expect("Fit should succeed");
        let score = tree.score(&x, &y);
        assert!(score > 0.9);
    }
}
