//! Decision Tree Regressor implementation.
//!
//! Uses the CART algorithm with MSE for splitting.

use super::helpers::build_regression_tree;
use super::RegressionTreeNode;
use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Decision tree regressor using the CART algorithm.
///
/// Uses Mean Squared Error (MSE) for splitting criterion and builds trees recursively.
/// Leaf nodes predict the mean of target values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTreeRegressor {
    pub(super) tree: Option<RegressionTreeNode>,
    pub(super) max_depth: Option<usize>,
    pub(super) min_samples_split: usize,
    pub(super) min_samples_leaf: usize,
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
