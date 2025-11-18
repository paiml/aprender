//! Decision tree algorithms for classification.
//!
//! This module implements the CART (Classification and Regression Trees) algorithm
//! for building decision trees using Gini impurity as the split criterion.
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
//! ]).unwrap();
//! let y = vec![0, 1, 1, 0];
//!
//! // Train decision tree
//! let mut tree = DecisionTreeClassifier::new()
//!     .with_max_depth(3);
//! tree.fit(&x, &y).unwrap();
//!
//! // Make predictions
//! let predictions = tree.predict(&x);
//! ```

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

/// Decision tree classifier using the CART algorithm.
///
/// Uses Gini impurity for splitting criterion and builds trees recursively.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeClassifier {
    tree: Option<TreeNode>,
    max_depth: Option<usize>,
}

impl DecisionTreeClassifier {
    /// Creates a new decision tree classifier with default parameters.
    pub fn new() -> Self {
        Self {
            tree: None,
            max_depth: None,
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
    pub fn fit(
        &mut self,
        x: &crate::primitives::Matrix<f32>,
        y: &[usize],
    ) -> Result<(), &'static str> {
        let (n_rows, _n_cols) = x.shape();
        if n_rows != y.len() {
            return Err("Number of samples in X and y must match");
        }
        if n_rows == 0 {
            return Err("Cannot fit with zero samples");
        }

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
    /// Panics if called before fit()
    pub fn predict(&self, x: &crate::primitives::Matrix<f32>) -> Vec<usize> {
        let (n_samples, n_features) = x.shape();
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
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {}", e))?;
        fs::write(path, bytes).map_err(|e| format!("File write failed: {}", e))?;
        Ok(())
    }

    /// Loads a model from a binary file.
    ///
    /// # Errors
    ///
    /// Returns an error if file reading or deserialization fails.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("File read failed: {}", e))?;
        let model =
            bincode::deserialize(&bytes).map_err(|e| format!("Deserialization failed: {}", e))?;
        Ok(model)
    }
}

impl Default for DecisionTreeClassifier {
    fn default() -> Self {
        Self::new()
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

    // Get sorted unique values
    let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
    sorted_indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

    let mut unique_values = Vec::new();
    let mut prev_val = x[sorted_indices[0]];
    unique_values.push(prev_val);

    for &idx in &sorted_indices[1..] {
        if (x[idx] - prev_val).abs() > 1e-10 {
            unique_values.push(x[idx]);
            prev_val = x[idx];
        }
    }

    if unique_values.len() < 2 {
        return None;
    }

    // Calculate current impurity
    let current_impurity = gini_impurity(y);

    let mut best_gain = 0.0;
    let mut best_threshold = 0.0;

    // Try each midpoint as threshold
    for i in 0..unique_values.len() - 1 {
        let threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;

        // Split labels based on threshold
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
            continue;
        }

        // Calculate weighted Gini for this split
        let split_impurity = gini_split(&left_labels, &right_labels);
        let gain = current_impurity - split_impurity;

        if gain > best_gain {
            best_gain = gain;
            best_threshold = threshold;
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
    *counts.iter().max_by_key(|(_, &count)| count).unwrap().0
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

    // Criterion 1: All same label (pure node)
    let unique_labels: std::collections::HashSet<_> = y.iter().collect();
    if unique_labels.len() == 1 {
        return TreeNode::Leaf(Leaf {
            class_label: y[0],
            n_samples,
        });
    }

    // Criterion 2: Max depth reached
    if let Some(max_d) = max_depth {
        if depth >= max_d {
            return TreeNode::Leaf(Leaf {
                class_label: majority_class(y),
                n_samples,
            });
        }
    }

    // Try to find best split
    if let Some((feature_idx, threshold, _gain)) = find_best_split(x, y) {
        // Split data based on threshold
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for row in 0..n_samples {
            if x.get(row, feature_idx) <= threshold {
                left_indices.push(row);
            } else {
                right_indices.push(row);
            }
        }

        // Safety check - ensure split is valid
        if left_indices.is_empty() || right_indices.is_empty() {
            return TreeNode::Leaf(Leaf {
                class_label: majority_class(y),
                n_samples,
            });
        }

        // Create left and right datasets
        let (_n_rows, n_cols) = x.shape();
        let mut left_data = Vec::with_capacity(left_indices.len() * n_cols);
        let mut left_labels = Vec::with_capacity(left_indices.len());

        for &idx in &left_indices {
            for col in 0..n_cols {
                left_data.push(x.get(idx, col));
            }
            left_labels.push(y[idx]);
        }

        let mut right_data = Vec::with_capacity(right_indices.len() * n_cols);
        let mut right_labels = Vec::with_capacity(right_indices.len());

        for &idx in &right_indices {
            for col in 0..n_cols {
                right_data.push(x.get(idx, col));
            }
            right_labels.push(y[idx]);
        }

        let left_matrix =
            crate::primitives::Matrix::from_vec(left_indices.len(), n_cols, left_data).unwrap();
        let right_matrix =
            crate::primitives::Matrix::from_vec(right_indices.len(), n_cols, right_data).unwrap();

        // Recursively build subtrees
        let left_child = build_tree(&left_matrix, &left_labels, depth + 1, max_depth);
        let right_child = build_tree(&right_matrix, &right_labels, depth + 1, max_depth);

        TreeNode::Node(Node {
            feature_idx,
            threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        })
    } else {
        // No valid split found - create leaf
        TreeNode::Leaf(Leaf {
            class_label: majority_class(y),
            n_samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
            assert!(gini >= 0.0, "Gini should be >= 0, got {}", gini);
            assert!(gini <= 1.0, "Gini should be <= 1, got {}", gini);
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

        let (threshold, gain) = result.unwrap();
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
        let x = Matrix::from_vec(4, 2, vec![1.0, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 0, 1, 1];

        let result = find_best_split(&x, &y);
        assert!(result.is_some());

        let (feature_idx, _threshold, gain) = result.unwrap();
        // Should choose one of the features
        assert!(feature_idx < 2);
        // Should have positive gain
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_split_perfect_separation() {
        use crate::primitives::Matrix;

        // Perfectly separable data on feature 0
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 0, 1, 1];

        let result = find_best_split(&x, &y);
        assert!(result.is_some());

        let (feature_idx, threshold, gain) = result.unwrap();
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
        let x = Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 0, 0];

        let tree = build_tree(&x, &y, 0, None);

        // Should be a leaf with class 0
        match tree {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 0);
                assert_eq!(leaf.n_samples, 3);
            }
            _ => panic!("Expected Leaf node for pure data"),
        }
    }

    #[test]
    fn test_build_tree_max_depth_zero() {
        use crate::primitives::Matrix;

        // Mixed labels but max_depth=0, should create leaf with majority
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = vec![0, 0, 1, 1];

        let tree = build_tree(&x, &y, 0, Some(0));

        // Should be a leaf (can't split at depth 0)
        match tree {
            TreeNode::Leaf(leaf) => {
                assert!(leaf.class_label == 0 || leaf.class_label == 1);
                assert_eq!(leaf.n_samples, 4);
            }
            _ => panic!("Expected Leaf node at max depth"),
        }
    }

    #[test]
    fn test_build_tree_simple_split() {
        use crate::primitives::Matrix;

        // Simple binary split
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 0, 1, 1];

        let tree = build_tree(&x, &y, 0, Some(5));

        // Should be a node (not pure, depth allows split)
        match tree {
            TreeNode::Node(node) => {
                assert_eq!(node.feature_idx, 0); // Only one feature
                assert!(node.threshold > 2.0 && node.threshold < 5.0);
            }
            _ => panic!("Expected Node for splittable data"),
        }
    }

    #[test]
    fn test_build_tree_depth_tracking() {
        use crate::primitives::Matrix;

        // Build tree and verify depth is respected
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 0, 1, 1];

        let tree = build_tree(&x, &y, 0, Some(1));

        // Depth should be <= 1
        assert!(tree.depth() <= 1);
    }

    // RED-GREEN-REFACTOR Cycle 5: fit/predict/score

    #[test]
    fn test_fit_simple() {
        use crate::primitives::Matrix;

        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
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
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).unwrap();

        let predictions = tree.predict(&x);
        assert_eq!(predictions, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_predict_single_sample() {
        use crate::primitives::Matrix;

        let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y_train = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x_train, &y_train).unwrap();

        // Test single sample prediction
        let x_test = Matrix::from_vec(1, 1, vec![1.5]).unwrap();
        let predictions = tree.predict(&x_test);
        assert_eq!(predictions.len(), 1);
        assert_eq!(predictions[0], 0); // Should be class 0 (closer to 1.0, 2.0)
    }

    #[test]
    fn test_score_perfect() {
        use crate::primitives::Matrix;

        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).unwrap();

        let accuracy = tree.score(&x, &y);
        assert!((accuracy - 1.0).abs() < 1e-6); // Perfect classification
    }

    #[test]
    fn test_score_partial() {
        use crate::primitives::Matrix;

        // Train on simple data
        let x_train = Matrix::from_vec(4, 1, vec![1.0, 2.0, 5.0, 6.0]).unwrap();
        let y_train = vec![0, 0, 1, 1];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(1);
        tree.fit(&x_train, &y_train).unwrap();

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
        .unwrap();
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).unwrap();

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
        .unwrap();
        let y = vec![0, 0, 1, 1, 2, 2];

        let mut tree = DecisionTreeClassifier::new().with_max_depth(5);
        tree.fit(&x, &y).unwrap();

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
}
