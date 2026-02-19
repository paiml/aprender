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

// Submodules
mod classifier;
mod gradient_boosting;
mod helpers;
mod random_forest;
mod regressor;

// Re-exports
pub use classifier::DecisionTreeClassifier;
pub use gradient_boosting::GradientBoostingClassifier;
pub use helpers::{gini_impurity, gini_split};
pub use random_forest::{RandomForestClassifier, RandomForestRegressor};
pub use regressor::DecisionTreeRegressor;

use serde::{Deserialize, Serialize};

// ============================================================================
// Core Types
// ============================================================================

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

// ============================================================================
// Regression Tree Structures
// ============================================================================

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

#[cfg(test)]
mod tests;
