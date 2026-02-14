//! Helper functions for tree building algorithms.
//!
//! This module contains internal helper functions used by decision tree
//! and ensemble methods.

use super::{Leaf, Node, RegressionLeaf, RegressionNode, RegressionTreeNode, TreeNode};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Classification Tree Helpers
// ============================================================================

/// Flattens a tree structure into parallel arrays via pre-order traversal.
///
/// Returns the index of the root node.
pub(super) fn flatten_tree_node(
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
pub(super) fn reconstruct_tree_node(
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

/// Calculate Gini impurity for a set of labels.
///
/// Gini impurity measures the probability of incorrectly classifying a randomly
/// chosen element if it were labeled according to the distribution of labels.
///
/// Formula: Gini = 1 - `Σ(p_i²)` where `p_i` is the proportion of class i
#[allow(dead_code)]
pub(super) fn gini_impurity(labels: &[usize]) -> f32 {
    if labels.is_empty() {
        return 0.0;
    }

    // Count occurrences of each class
    let mut counts = HashMap::new();
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

/// Calculate weighted Gini impurity for a split.
#[allow(dead_code)]
pub(super) fn gini_split(left_labels: &[usize], right_labels: &[usize]) -> f32 {
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
#[allow(dead_code)]
pub(super) fn get_sorted_unique_values(x: &[f32]) -> Vec<f32> {
    let mut sorted_indices: Vec<usize> = (0..x.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .expect("f32 values should be comparable")
    });

    let mut unique_values = Vec::new();
    let mut prev_val = x[sorted_indices[0]];
    unique_values.push(prev_val);

    for &idx in sorted_indices.get(1..).unwrap_or(&[]) {
        if (x[idx] - prev_val).abs() > 1e-10 {
            unique_values.push(x[idx]);
            prev_val = x[idx];
        }
    }

    unique_values
}

/// Split labels into left and right partitions based on threshold.
#[allow(dead_code)]
pub(super) fn split_labels_by_threshold(
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
#[allow(dead_code)]
pub(super) fn calculate_information_gain(
    current_impurity: f32,
    left_labels: &[usize],
    right_labels: &[usize],
) -> f32 {
    let split_impurity = gini_split(left_labels, right_labels);
    current_impurity - split_impurity
}

/// Find the best split for a given feature.
#[allow(dead_code)]
pub(super) fn find_best_split_for_feature(x: &[f32], y: &[usize]) -> Option<(f32, f32)> {
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
#[allow(dead_code)]
pub(super) fn find_best_split(
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
#[allow(dead_code)]
pub(super) fn majority_class(labels: &[usize]) -> usize {
    let mut counts = HashMap::new();
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
#[allow(dead_code)]
pub(super) fn split_data_by_indices(
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
#[allow(dead_code)]
pub(super) fn check_stopping_criteria(
    y: &[usize],
    depth: usize,
    max_depth: Option<usize>,
) -> Option<TreeNode> {
    let n_samples = y.len();

    // Criterion 1: All same label (pure node)
    let unique_labels: HashSet<_> = y.iter().collect();
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
#[allow(dead_code)]
pub(super) fn split_indices_by_threshold(
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
#[allow(dead_code)]
pub(super) fn build_tree(
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

// ============================================================================
// Regression Tree Helpers
// ============================================================================

/// Compute the mean of a vector.
pub(super) fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f32>() / values.len() as f32
    }
}

/// Compute the variance of target values.
pub(super) fn variance_f32(y: &[f32]) -> f32 {
    if y.len() <= 1 {
        return 0.0;
    }

    let mean = mean_f32(y);
    let sum_squared_diff: f32 = y.iter().map(|&val| (val - mean).powi(2)).sum();
    sum_squared_diff / y.len() as f32
}

/// Compute Mean Squared Error for a split.
pub(super) fn compute_mse(y_left: &[f32], y_right: &[f32]) -> f32 {
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
pub(super) fn get_unique_feature_values(
    x: &crate::primitives::Matrix<f32>,
    feature_idx: usize,
    n_samples: usize,
) -> Vec<f32> {
    let mut values: Vec<f32> = (0..n_samples).map(|i| x.get(i, feature_idx)).collect();
    values.sort_by(|a, b| a.total_cmp(b));
    values.dedup();
    values
}

/// Split y values by a threshold on a feature.
pub(super) fn split_by_threshold(
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
pub(super) fn evaluate_split_gain(
    y_left: &[f32],
    y_right: &[f32],
    current_variance: f32,
) -> Option<f32> {
    if y_left.is_empty() || y_right.is_empty() {
        return None;
    }
    let split_mse = compute_mse(y_left, y_right);
    let gain = current_variance - split_mse;
    (gain > 0.0).then_some(gain)
}

/// Find the best split for a single feature.
pub(super) fn find_best_regression_split_for_feature(
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
pub(super) fn find_best_regression_split(
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
pub(super) fn split_regression_data_by_indices(
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
        .expect("subset data length matches n_subset * n_features");

    (subset_matrix, subset_labels)
}

/// Create a regression leaf node from y values.
pub(super) fn make_regression_leaf(y_slice: &[f32], n_samples: usize) -> RegressionTreeNode {
    RegressionTreeNode::Leaf(RegressionLeaf {
        value: mean_f32(y_slice),
        n_samples,
    })
}

/// Check if we've reached max depth.
pub(super) fn at_max_depth(depth: usize, max_depth: Option<usize>) -> bool {
    max_depth.is_some_and(|max_d| depth >= max_d)
}

/// Partition sample indices based on feature threshold.
pub(super) fn partition_by_threshold(
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
pub(super) fn build_regression_tree(
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

// ============================================================================
// Feature Importance Helpers
// ============================================================================

/// Compute feature importances from a classification tree by traversing it.
pub(super) fn compute_tree_feature_importances(node: &TreeNode, importances: &mut [f32]) {
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
pub(super) fn count_tree_samples(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf(leaf) => leaf.n_samples,
        TreeNode::Node(n) => {
            // For internal nodes, sum samples from children
            count_tree_samples(&n.left) + count_tree_samples(&n.right)
        }
    }
}

/// Compute feature importances from a regression tree by traversing it.
pub(super) fn compute_regression_tree_feature_importances(
    node: &RegressionTreeNode,
    importances: &mut [f32],
) {
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
pub(super) fn count_regression_tree_samples(node: &RegressionTreeNode) -> usize {
    match node {
        RegressionTreeNode::Leaf(leaf) => leaf.n_samples,
        RegressionTreeNode::Node(n) => {
            // For internal nodes, sum samples from children
            count_regression_tree_samples(&n.left) + count_regression_tree_samples(&n.right)
        }
    }
}

// ============================================================================
// Bootstrap Sampling
// ============================================================================

/// Creates a bootstrap sample (random sample with replacement).
///
/// Returns indices of samples to include in the bootstrap sample.
pub(super) fn bootstrap_sample(n_samples: usize, random_state: Option<u64>) -> Vec<usize> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{Matrix, Vector};

    // ========================================================================
    // Gini Impurity Tests
    // ========================================================================

    #[test]
    fn test_gini_impurity_empty() {
        assert!((gini_impurity(&[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_pure_single_class() {
        // All same class -> Gini = 0
        let labels = vec![0, 0, 0, 0];
        assert!((gini_impurity(&labels) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_two_classes_balanced() {
        // 50/50 split -> Gini = 0.5
        let labels = vec![0, 1, 0, 1];
        assert!((gini_impurity(&labels) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_two_classes_unbalanced() {
        // 3/4 class 0, 1/4 class 1 -> Gini = 1 - (0.75^2 + 0.25^2) = 1 - 0.625 = 0.375
        let labels = vec![0, 0, 0, 1];
        assert!((gini_impurity(&labels) - 0.375).abs() < 1e-7);
    }

    #[test]
    fn test_gini_impurity_three_classes_uniform() {
        // Three classes equally distributed -> Gini = 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
        let labels = vec![0, 1, 2, 0, 1, 2];
        let expected = 1.0 - 3.0 * (1.0_f32 / 3.0).powi(2);
        assert!((gini_impurity(&labels) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_gini_impurity_single_sample() {
        let labels = vec![5];
        assert!((gini_impurity(&labels) - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Gini Split Tests
    // ========================================================================

    #[test]
    fn test_gini_split_empty_both() {
        assert!((gini_split(&[], &[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_split_perfect_split() {
        // Perfect split: all 0s on left, all 1s on right -> weighted Gini = 0
        let left = vec![0, 0, 0];
        let right = vec![1, 1, 1];
        assert!((gini_split(&left, &right) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_gini_split_worst_case() {
        // Both sides have same distribution -> split doesn't help
        let left = vec![0, 1];
        let right = vec![0, 1];
        assert!((gini_split(&left, &right) - 0.5).abs() < 1e-7);
    }

    // ========================================================================
    // Sorted Unique Values Tests
    // ========================================================================

    #[test]
    fn test_get_sorted_unique_values_basic() {
        let values = vec![3.0, 1.0, 2.0, 1.0, 3.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_get_sorted_unique_values_single() {
        let values = vec![5.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![5.0]);
    }

    #[test]
    fn test_get_sorted_unique_values_all_same() {
        let values = vec![2.0, 2.0, 2.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![2.0]);
    }

    #[test]
    fn test_get_sorted_unique_values_already_sorted() {
        let values = vec![1.0, 2.0, 3.0, 4.0];
        let unique = get_sorted_unique_values(&values);
        assert_eq!(unique, vec![1.0, 2.0, 3.0, 4.0]);
    }

    // ========================================================================
    // Split Labels by Threshold Tests
    // ========================================================================

    #[test]
    fn test_split_labels_by_threshold_normal() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0, 0, 1, 1];
        let result = split_labels_by_threshold(&x, &y, 2.5);
        let (left, right) = result.expect("split should produce non-empty partitions");
        assert_eq!(left, vec![0, 0]);
        assert_eq!(right, vec![1, 1]);
    }

    #[test]
    fn test_split_labels_by_threshold_all_left() {
        // All values <= threshold -> right is empty -> None
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0, 1, 0];
        assert!(split_labels_by_threshold(&x, &y, 10.0).is_none());
    }

    #[test]
    fn test_split_labels_by_threshold_all_right() {
        // All values > threshold -> left is empty -> None
        let x = vec![5.0, 6.0, 7.0];
        let y = vec![0, 1, 0];
        assert!(split_labels_by_threshold(&x, &y, 0.0).is_none());
    }

    #[test]
    fn test_split_labels_by_threshold_boundary() {
        // Exact threshold value goes to left (<=)
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0, 1, 2];
        let (left, right) = split_labels_by_threshold(&x, &y, 2.0).expect("split should succeed");
        assert_eq!(left, vec![0, 1]);
        assert_eq!(right, vec![2]);
    }

    // ========================================================================
    // Calculate Information Gain Tests
    // ========================================================================

    #[test]
    fn test_calculate_information_gain_perfect_split() {
        let current = gini_impurity(&[0, 0, 1, 1]);
        let gain = calculate_information_gain(current, &[0, 0], &[1, 1]);
        // Perfect split: gain equals the full impurity
        assert!((gain - current).abs() < 1e-7);
    }

    #[test]
    fn test_calculate_information_gain_no_improvement() {
        let current = gini_impurity(&[0, 1, 0, 1]);
        let gain = calculate_information_gain(current, &[0, 1], &[0, 1]);
        // Same distribution on both sides -> zero gain
        assert!(gain.abs() < 1e-7);
    }

    // ========================================================================
    // Find Best Split for Feature Tests
    // ========================================================================

    #[test]
    fn test_find_best_split_for_feature_too_few_samples() {
        let x = vec![1.0];
        let y = vec![0];
        assert!(find_best_split_for_feature(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_split_for_feature_all_same_value() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![0, 1, 0];
        assert!(find_best_split_for_feature(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_split_for_feature_clear_split() {
        // Feature perfectly separates classes
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![0, 0, 1, 1];
        let result = find_best_split_for_feature(&x, &y);
        let (threshold, gain) = result.expect("should find a split");
        assert!((threshold - 2.5).abs() < 1e-7);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_split_for_feature_pure_labels() {
        // All same label -> no gain possible
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![0, 0, 0];
        assert!(find_best_split_for_feature(&x, &y).is_none());
    }

    // ========================================================================
    // Find Best Split (all features) Tests
    // ========================================================================

    #[test]
    fn test_find_best_split_single_sample() {
        let x = Matrix::from_vec(1, 2, vec![1.0, 2.0]).expect("matrix creation");
        let y = vec![0];
        assert!(find_best_split(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_split_separable() {
        // Feature 0 separates perfectly, feature 1 does not
        let x = Matrix::from_vec(4, 2, vec![1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0])
            .expect("matrix creation");
        let y = vec![0, 0, 1, 1];
        let (feat, threshold, gain) = find_best_split(&x, &y).expect("should find split");
        assert_eq!(feat, 0);
        assert!((threshold - 2.5).abs() < 1e-7);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_split_pure_labels() {
        let x =
            Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix creation");
        let y = vec![0, 0, 0];
        assert!(find_best_split(&x, &y).is_none());
    }

    // ========================================================================
    // Majority Class Tests
    // ========================================================================

    #[test]
    fn test_majority_class_single() {
        assert_eq!(majority_class(&[7]), 7);
    }

    #[test]
    fn test_majority_class_clear_winner() {
        assert_eq!(majority_class(&[0, 1, 1, 1, 0]), 1);
    }

    #[test]
    fn test_majority_class_all_same() {
        assert_eq!(majority_class(&[3, 3, 3]), 3);
    }

    // ========================================================================
    // Split Data by Indices Tests
    // ========================================================================

    #[test]
    fn test_split_data_by_indices_basic() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("matrix creation");
        let y = vec![0, 1, 2, 3];
        let indices = vec![0, 2];
        let (subset_x, subset_y) = split_data_by_indices(&x, &y, &indices);
        assert_eq!(subset_x.shape(), (2, 2));
        assert_eq!(subset_y, vec![0, 2]);
        assert!((subset_x.get(0, 0) - 1.0).abs() < 1e-7);
        assert!((subset_x.get(0, 1) - 2.0).abs() < 1e-7);
        assert!((subset_x.get(1, 0) - 5.0).abs() < 1e-7);
        assert!((subset_x.get(1, 1) - 6.0).abs() < 1e-7);
    }

    // ========================================================================
    // Check Stopping Criteria Tests
    // ========================================================================

    #[test]
    fn test_check_stopping_criteria_pure_node() {
        let y = vec![1, 1, 1];
        let result = check_stopping_criteria(&y, 0, None);
        let node = result.expect("pure node should trigger stop");
        match node {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 1);
                assert_eq!(leaf.n_samples, 3);
            }
            TreeNode::Node(_) => panic!("expected leaf, got internal node"),
        }
    }

    #[test]
    fn test_check_stopping_criteria_max_depth() {
        let y = vec![0, 1, 1];
        let result = check_stopping_criteria(&y, 5, Some(5));
        let node = result.expect("max depth should trigger stop");
        match node {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 1); // majority
                assert_eq!(leaf.n_samples, 3);
            }
            TreeNode::Node(_) => panic!("expected leaf, got internal node"),
        }
    }

    #[test]
    fn test_check_stopping_criteria_no_stop() {
        let y = vec![0, 1, 1];
        let result = check_stopping_criteria(&y, 2, Some(5));
        assert!(result.is_none());
    }

    #[test]
    fn test_check_stopping_criteria_no_max_depth() {
        let y = vec![0, 1, 1];
        let result = check_stopping_criteria(&y, 100, None);
        assert!(result.is_none());
    }

    // ========================================================================
    // Split Indices by Threshold Tests
    // ========================================================================

    #[test]
    fn test_split_indices_by_threshold_normal() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 3.0, 2.0, 4.0]).expect("matrix creation");
        let result = split_indices_by_threshold(&x, 0, 2.5, 4);
        let (left, right) = result.expect("should split successfully");
        assert_eq!(left, vec![0, 2]);
        assert_eq!(right, vec![1, 3]);
    }

    #[test]
    fn test_split_indices_by_threshold_all_left() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        assert!(split_indices_by_threshold(&x, 0, 10.0, 3).is_none());
    }

    #[test]
    fn test_split_indices_by_threshold_all_right() {
        let x = Matrix::from_vec(3, 1, vec![5.0, 6.0, 7.0]).expect("matrix creation");
        assert!(split_indices_by_threshold(&x, 0, 0.0, 3).is_none());
    }

    // ========================================================================
    // Flatten / Reconstruct Tree Tests
    // ========================================================================

    #[test]
    fn test_flatten_and_reconstruct_leaf() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 2,
            n_samples: 10,
        });

        let mut features = Vec::new();
        let mut thresholds = Vec::new();
        let mut classes = Vec::new();
        let mut samples = Vec::new();
        let mut left_children = Vec::new();
        let mut right_children = Vec::new();

        let root_idx = flatten_tree_node(
            &leaf,
            &mut features,
            &mut thresholds,
            &mut classes,
            &mut samples,
            &mut left_children,
            &mut right_children,
        );

        assert_eq!(root_idx, 0);
        assert_eq!(features.len(), 1);
        assert!((features[0] - (-1.0)).abs() < 1e-7);
        assert!((classes[0] - 2.0).abs() < 1e-7);
        assert!((samples[0] - 10.0).abs() < 1e-7);

        // Reconstruct
        let reconstructed = reconstruct_tree_node(
            root_idx,
            &features,
            &thresholds,
            &classes,
            &samples,
            &left_children,
            &right_children,
        );
        match reconstructed {
            TreeNode::Leaf(l) => {
                assert_eq!(l.class_label, 2);
                assert_eq!(l.n_samples, 10);
            }
            TreeNode::Node(_) => panic!("expected leaf"),
        }
    }

    #[test]
    fn test_flatten_and_reconstruct_tree() {
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 2.5,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 3,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 2,
            })),
        });

        let mut features = Vec::new();
        let mut thresholds = Vec::new();
        let mut classes = Vec::new();
        let mut samples = Vec::new();
        let mut left_children = Vec::new();
        let mut right_children = Vec::new();

        let root_idx = flatten_tree_node(
            &tree,
            &mut features,
            &mut thresholds,
            &mut classes,
            &mut samples,
            &mut left_children,
            &mut right_children,
        );

        assert_eq!(root_idx, 0);
        assert_eq!(features.len(), 3); // root + 2 leaves

        let reconstructed = reconstruct_tree_node(
            root_idx,
            &features,
            &thresholds,
            &classes,
            &samples,
            &left_children,
            &right_children,
        );
        match reconstructed {
            TreeNode::Node(n) => {
                assert_eq!(n.feature_idx, 0);
                assert!((n.threshold - 2.5).abs() < 1e-7);
                match n.left.as_ref() {
                    TreeNode::Leaf(l) => assert_eq!(l.class_label, 0),
                    _ => panic!("expected left leaf"),
                }
                match n.right.as_ref() {
                    TreeNode::Leaf(l) => assert_eq!(l.class_label, 1),
                    _ => panic!("expected right leaf"),
                }
            }
            TreeNode::Leaf(_) => panic!("expected node, got leaf"),
        }
    }

    // ========================================================================
    // Build Tree Tests
    // ========================================================================

    #[test]
    fn test_build_tree_pure_data() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let y = vec![0, 0, 0];
        let tree = build_tree(&x, &y, 0, None);
        match tree {
            TreeNode::Leaf(leaf) => {
                assert_eq!(leaf.class_label, 0);
                assert_eq!(leaf.n_samples, 3);
            }
            TreeNode::Node(_) => panic!("pure data should produce a leaf"),
        }
    }

    #[test]
    fn test_build_tree_max_depth_zero() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![0, 0, 1, 1];
        let tree = build_tree(&x, &y, 0, Some(0));
        match tree {
            TreeNode::Leaf(leaf) => {
                // Majority class with depth=0
                assert_eq!(leaf.n_samples, 4);
            }
            TreeNode::Node(_) => panic!("max_depth=0 should produce a leaf"),
        }
    }

    #[test]
    fn test_build_tree_separable() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![0, 0, 1, 1];
        let tree = build_tree(&x, &y, 0, Some(5));
        match &tree {
            TreeNode::Node(n) => {
                assert_eq!(n.feature_idx, 0);
                // Threshold should be 2.5 (midpoint between 2 and 3)
                assert!((n.threshold - 2.5).abs() < 1e-7);
            }
            TreeNode::Leaf(_) => panic!("separable data should produce a split"),
        }
    }

    #[test]
    fn test_build_tree_depth_limited() {
        // Three classes in feature space, max_depth=1 forces a single split
        let x =
            Matrix::from_vec(6, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("matrix creation");
        let y = vec![0, 0, 1, 1, 2, 2];
        let tree = build_tree(&x, &y, 0, Some(1));
        match &tree {
            TreeNode::Node(n) => {
                // Both children must be leaves since depth=1
                assert!(matches!(n.left.as_ref(), TreeNode::Leaf(_)));
                assert!(matches!(n.right.as_ref(), TreeNode::Leaf(_)));
            }
            TreeNode::Leaf(_) => panic!("mixed labels should produce a split"),
        }
    }

    // ========================================================================
    // Mean and Variance Tests
    // ========================================================================

    #[test]
    fn test_mean_f32_empty() {
        assert!((mean_f32(&[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_mean_f32_single() {
        assert!((mean_f32(&[5.0]) - 5.0).abs() < 1e-7);
    }

    #[test]
    fn test_mean_f32_multiple() {
        assert!((mean_f32(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-7);
    }

    #[test]
    fn test_variance_f32_empty() {
        assert!((variance_f32(&[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_variance_f32_single() {
        assert!((variance_f32(&[42.0]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_variance_f32_uniform() {
        // Var([1,2,3,4]) = mean of squared deviations from mean(2.5)
        // = ((1.5^2 + 0.5^2 + 0.5^2 + 1.5^2)/4) = (2.25+0.25+0.25+2.25)/4 = 5.0/4 = 1.25
        assert!((variance_f32(&[1.0, 2.0, 3.0, 4.0]) - 1.25).abs() < 1e-6);
    }

    #[test]
    fn test_variance_f32_all_same() {
        assert!((variance_f32(&[3.0, 3.0, 3.0]) - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Compute MSE Tests
    // ========================================================================

    #[test]
    fn test_compute_mse_empty_both() {
        assert!((compute_mse(&[], &[]) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_compute_mse_perfect_split() {
        // Left: all same, right: all same -> MSE = 0
        let left = vec![1.0, 1.0, 1.0];
        let right = vec![5.0, 5.0, 5.0];
        assert!((compute_mse(&left, &right) - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_compute_mse_mixed() {
        let left = vec![1.0, 3.0]; // variance = 1.0
        let right = vec![5.0, 7.0]; // variance = 1.0
                                    // MSE = (2/4)*1.0 + (2/4)*1.0 = 1.0
        assert!((compute_mse(&left, &right) - 1.0).abs() < 1e-6);
    }

    // ========================================================================
    // Get Unique Feature Values Tests
    // ========================================================================

    #[test]
    fn test_get_unique_feature_values_basic() {
        let x = Matrix::from_vec(4, 2, vec![3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0])
            .expect("matrix creation");
        let values = get_unique_feature_values(&x, 0, 4);
        assert_eq!(values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_get_unique_feature_values_second_column() {
        let x =
            Matrix::from_vec(3, 2, vec![0.0, 5.0, 0.0, 3.0, 0.0, 5.0]).expect("matrix creation");
        let values = get_unique_feature_values(&x, 1, 3);
        assert_eq!(values, vec![3.0, 5.0]);
    }

    // ========================================================================
    // Split by Threshold (Regression) Tests
    // ========================================================================

    #[test]
    fn test_split_by_threshold_regression() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let (left, right) = split_by_threshold(&x, &y, 0, 2.5);
        assert_eq!(left, vec![10.0, 20.0]);
        assert_eq!(right, vec![30.0, 40.0]);
    }

    #[test]
    fn test_split_by_threshold_all_left() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let y = vec![10.0, 20.0, 30.0];
        let (left, right) = split_by_threshold(&x, &y, 0, 100.0);
        assert_eq!(left.len(), 3);
        assert!(right.is_empty());
    }

    // ========================================================================
    // Evaluate Split Gain Tests
    // ========================================================================

    #[test]
    fn test_evaluate_split_gain_empty_left() {
        assert!(evaluate_split_gain(&[], &[1.0, 2.0], 1.0).is_none());
    }

    #[test]
    fn test_evaluate_split_gain_empty_right() {
        assert!(evaluate_split_gain(&[1.0, 2.0], &[], 1.0).is_none());
    }

    #[test]
    fn test_evaluate_split_gain_positive() {
        let current_variance = variance_f32(&[1.0, 1.0, 5.0, 5.0]);
        let gain = evaluate_split_gain(&[1.0, 1.0], &[5.0, 5.0], current_variance);
        let g = gain.expect("should have positive gain");
        assert!(g > 0.0);
        // Perfect split removes all variance
        assert!((g - current_variance).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_split_gain_no_gain() {
        // Same distribution on both sides
        let current_variance = variance_f32(&[1.0, 5.0, 1.0, 5.0]);
        let result = evaluate_split_gain(&[1.0, 5.0], &[1.0, 5.0], current_variance);
        assert!(result.is_none());
    }

    // ========================================================================
    // Find Best Regression Split Tests
    // ========================================================================

    #[test]
    fn test_find_best_regression_split_single_sample() {
        let x = Matrix::from_vec(1, 1, vec![1.0]).expect("matrix creation");
        let y = vec![5.0];
        assert!(find_best_regression_split(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_regression_split_separable() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = vec![1.0, 1.0, 10.0, 10.0];
        let result = find_best_regression_split(&x, &y);
        let (feat, threshold, gain) = result.expect("should find a split");
        assert_eq!(feat, 0);
        assert!((threshold - 2.5).abs() < 1e-7);
        assert!(gain > 0.0);
    }

    #[test]
    fn test_find_best_regression_split_constant_y() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let y = vec![5.0, 5.0, 5.0];
        assert!(find_best_regression_split(&x, &y).is_none());
    }

    #[test]
    fn test_find_best_regression_split_for_feature_single_unique_value() {
        let x = Matrix::from_vec(3, 1, vec![2.0, 2.0, 2.0]).expect("matrix creation");
        let y = vec![1.0, 2.0, 3.0];
        let current_var = variance_f32(&y);
        let result = find_best_regression_split_for_feature(&x, &y, 0, 3, current_var);
        assert!(result.is_none());
    }

    // ========================================================================
    // Split Regression Data by Indices Tests
    // ========================================================================

    #[test]
    fn test_split_regression_data_by_indices() {
        let x = Matrix::from_vec(4, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("matrix creation");
        let y = vec![10.0, 20.0, 30.0, 40.0];
        let indices = vec![1, 3];
        let (subset_x, subset_y) = split_regression_data_by_indices(&x, &y, &indices);
        assert_eq!(subset_x.shape(), (2, 2));
        assert_eq!(subset_y, vec![20.0, 40.0]);
        assert!((subset_x.get(0, 0) - 3.0).abs() < 1e-7);
        assert!((subset_x.get(0, 1) - 4.0).abs() < 1e-7);
        assert!((subset_x.get(1, 0) - 7.0).abs() < 1e-7);
        assert!((subset_x.get(1, 1) - 8.0).abs() < 1e-7);
    }

    // ========================================================================
    // Make Regression Leaf Tests
    // ========================================================================

    #[test]
    fn test_make_regression_leaf() {
        let y = vec![2.0, 4.0, 6.0];
        let leaf = make_regression_leaf(&y, 3);
        match leaf {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - 4.0).abs() < 1e-7);
                assert_eq!(l.n_samples, 3);
            }
            RegressionTreeNode::Node(_) => panic!("expected leaf"),
        }
    }

    #[test]
    fn test_make_regression_leaf_empty() {
        let leaf = make_regression_leaf(&[], 0);
        match leaf {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - 0.0).abs() < 1e-7);
                assert_eq!(l.n_samples, 0);
            }
            RegressionTreeNode::Node(_) => panic!("expected leaf"),
        }
    }

    // ========================================================================
    // At Max Depth Tests
    // ========================================================================

    #[test]
    fn test_at_max_depth_none() {
        assert!(!at_max_depth(100, None));
    }

    #[test]
    fn test_at_max_depth_not_reached() {
        assert!(!at_max_depth(3, Some(5)));
    }

    #[test]
    fn test_at_max_depth_exactly_reached() {
        assert!(at_max_depth(5, Some(5)));
    }

    #[test]
    fn test_at_max_depth_exceeded() {
        assert!(at_max_depth(10, Some(5)));
    }

    // ========================================================================
    // Partition by Threshold Tests
    // ========================================================================

    #[test]
    fn test_partition_by_threshold() {
        let x = Matrix::from_vec(5, 1, vec![1.0, 5.0, 2.0, 4.0, 3.0]).expect("matrix creation");
        let (left, right) = partition_by_threshold(&x, 5, 0, 3.0);
        assert_eq!(left, vec![0, 2, 4]);
        assert_eq!(right, vec![1, 3]);
    }

    #[test]
    fn test_partition_by_threshold_all_left() {
        let x = Matrix::from_vec(3, 1, vec![1.0, 2.0, 3.0]).expect("matrix creation");
        let (left, right) = partition_by_threshold(&x, 3, 0, 10.0);
        assert_eq!(left.len(), 3);
        assert!(right.is_empty());
    }

    // ========================================================================
    // Build Regression Tree Tests
    // ========================================================================

    #[test]
    fn test_build_regression_tree_constant_target() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![5.0, 5.0, 5.0, 5.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 2, 1);
        match tree {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - 5.0).abs() < 1e-7);
            }
            RegressionTreeNode::Node(_) => panic!("constant target should produce leaf"),
        }
    }

    #[test]
    fn test_build_regression_tree_max_depth_zero() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(0), 2, 1);
        match tree {
            RegressionTreeNode::Leaf(l) => {
                assert!((l.value - mean_f32(&[1.0, 1.0, 10.0, 10.0])).abs() < 1e-6);
            }
            RegressionTreeNode::Node(_) => panic!("max_depth=0 should produce leaf"),
        }
    }

    #[test]
    fn test_build_regression_tree_separable() {
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 2, 1);
        match &tree {
            RegressionTreeNode::Node(n) => {
                assert_eq!(n.feature_idx, 0);
                assert!((n.threshold - 2.5).abs() < 1e-7);
            }
            RegressionTreeNode::Leaf(_) => panic!("separable data should produce split"),
        }
    }

    #[test]
    fn test_build_regression_tree_min_samples_split() {
        // min_samples_split=10 means we never split with 4 samples
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 10, 1);
        match tree {
            RegressionTreeNode::Leaf(_) => {} // expected
            RegressionTreeNode::Node(_) => {
                panic!("min_samples_split=10 should prevent splitting 4 samples")
            }
        }
    }

    #[test]
    fn test_build_regression_tree_min_samples_leaf() {
        // min_samples_leaf=3 with 4 samples: 2/2 split disallowed, must be leaf
        let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).expect("matrix creation");
        let y = Vector::from_vec(vec![1.0, 1.0, 10.0, 10.0]);
        let tree = build_regression_tree(&x, &y, 0, Some(5), 2, 3);
        match tree {
            RegressionTreeNode::Leaf(_) => {} // expected
            RegressionTreeNode::Node(_) => {
                panic!("min_samples_leaf=3 should prevent 2/2 split")
            }
        }
    }

    // ========================================================================
    // Feature Importance Tests
    // ========================================================================

    #[test]
    fn test_count_tree_samples_leaf() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 42,
        });
        assert_eq!(count_tree_samples(&leaf), 42);
    }

    #[test]
    fn test_count_tree_samples_tree() {
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 1.0,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 10,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 20,
            })),
        });
        assert_eq!(count_tree_samples(&tree), 30);
    }

    #[test]
    fn test_compute_tree_feature_importances_leaf_only() {
        let leaf = TreeNode::Leaf(Leaf {
            class_label: 0,
            n_samples: 10,
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&leaf, &mut importances);
        assert_eq!(importances, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_tree_feature_importances_single_split() {
        let tree = TreeNode::Node(Node {
            feature_idx: 1,
            threshold: 2.0,
            left: Box::new(TreeNode::Leaf(Leaf {
                class_label: 0,
                n_samples: 5,
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 1,
                n_samples: 3,
            })),
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&tree, &mut importances);
        // Feature 1 used at root with 8 total samples
        assert!((importances[0] - 0.0).abs() < 1e-7);
        assert!((importances[1] - 8.0).abs() < 1e-7);
        assert!((importances[2] - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Regression Feature Importance Tests
    // ========================================================================

    #[test]
    fn test_count_regression_tree_samples_leaf() {
        let leaf = RegressionTreeNode::Leaf(RegressionLeaf {
            value: 3.5,
            n_samples: 15,
        });
        assert_eq!(count_regression_tree_samples(&leaf), 15);
    }

    #[test]
    fn test_count_regression_tree_samples_tree() {
        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 0,
            threshold: 1.0,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 7,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 5.0,
                n_samples: 13,
            })),
        });
        assert_eq!(count_regression_tree_samples(&tree), 20);
    }

    #[test]
    fn test_compute_regression_tree_feature_importances() {
        let tree = RegressionTreeNode::Node(RegressionNode {
            feature_idx: 2,
            threshold: 3.0,
            left: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 1.0,
                n_samples: 4,
            })),
            right: Box::new(RegressionTreeNode::Leaf(RegressionLeaf {
                value: 9.0,
                n_samples: 6,
            })),
        });
        let mut importances = vec![0.0; 4];
        compute_regression_tree_feature_importances(&tree, &mut importances);
        assert!((importances[2] - 10.0).abs() < 1e-7);
        assert!((importances[0] - 0.0).abs() < 1e-7);
        assert!((importances[1] - 0.0).abs() < 1e-7);
        assert!((importances[3] - 0.0).abs() < 1e-7);
    }

    // ========================================================================
    // Bootstrap Sample Tests
    // ========================================================================

    #[test]
    fn test_bootstrap_sample_deterministic() {
        let sample1 = bootstrap_sample(10, Some(42));
        let sample2 = bootstrap_sample(10, Some(42));
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_bootstrap_sample_length() {
        let sample = bootstrap_sample(20, Some(1));
        assert_eq!(sample.len(), 20);
    }

    #[test]
    fn test_bootstrap_sample_range() {
        let n = 10;
        let sample = bootstrap_sample(n, Some(99));
        for &idx in &sample {
            assert!(idx < n, "index {idx} should be less than {n}");
        }
    }

    #[test]
    fn test_bootstrap_sample_without_seed() {
        let sample = bootstrap_sample(10, None);
        assert_eq!(sample.len(), 10);
        for &idx in &sample {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_bootstrap_sample_different_seeds() {
        let sample1 = bootstrap_sample(100, Some(1));
        let sample2 = bootstrap_sample(100, Some(2));
        // Very unlikely to be identical with different seeds
        assert_ne!(sample1, sample2);
    }

    // ========================================================================
    // Multi-Feature Regression Split Tests
    // ========================================================================

    #[test]
    fn test_find_best_regression_split_multi_feature() {
        // Feature 1 is the informative one, feature 0 is noise
        #[rustfmt::skip]
        let x = Matrix::from_vec(6, 2, vec![
            5.0, 1.0,
            5.0, 2.0,
            5.0, 3.0,
            5.0, 4.0,
            5.0, 5.0,
            5.0, 6.0,
        ]).expect("matrix creation");
        let y = vec![1.0, 1.0, 1.0, 10.0, 10.0, 10.0];
        let (feat, _threshold, gain) =
            find_best_regression_split(&x, &y).expect("should find split");
        assert_eq!(feat, 1); // should pick the informative feature
        assert!(gain > 0.0);
    }

    // ========================================================================
    // Deep Tree Flatten/Reconstruct Test
    // ========================================================================

    #[test]
    fn test_flatten_reconstruct_deep_tree() {
        // Three levels deep
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 5.0,
            left: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 2.0,
                left: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 0,
                    n_samples: 2,
                })),
                right: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 1,
                    n_samples: 3,
                })),
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 5,
            })),
        });

        let mut features = Vec::new();
        let mut thresholds = Vec::new();
        let mut classes = Vec::new();
        let mut samples = Vec::new();
        let mut left_children = Vec::new();
        let mut right_children = Vec::new();

        let root_idx = flatten_tree_node(
            &tree,
            &mut features,
            &mut thresholds,
            &mut classes,
            &mut samples,
            &mut left_children,
            &mut right_children,
        );

        // 5 nodes total: 2 internal + 3 leaves
        assert_eq!(features.len(), 5);

        let reconstructed = reconstruct_tree_node(
            root_idx,
            &features,
            &thresholds,
            &classes,
            &samples,
            &left_children,
            &right_children,
        );

        // Verify structure
        match &reconstructed {
            TreeNode::Node(root) => {
                assert_eq!(root.feature_idx, 0);
                assert!((root.threshold - 5.0).abs() < 1e-7);
                match root.left.as_ref() {
                    TreeNode::Node(left) => {
                        assert_eq!(left.feature_idx, 1);
                        assert!((left.threshold - 2.0).abs() < 1e-7);
                        match left.left.as_ref() {
                            TreeNode::Leaf(ll) => assert_eq!(ll.class_label, 0),
                            _ => panic!("expected leaf"),
                        }
                        match left.right.as_ref() {
                            TreeNode::Leaf(lr) => assert_eq!(lr.class_label, 1),
                            _ => panic!("expected leaf"),
                        }
                    }
                    _ => panic!("expected node"),
                }
                match root.right.as_ref() {
                    TreeNode::Leaf(r) => assert_eq!(r.class_label, 2),
                    _ => panic!("expected leaf"),
                }
            }
            _ => panic!("expected node at root"),
        }
    }

    // ========================================================================
    // Feature Importance with Nested Tree
    // ========================================================================

    #[test]
    fn test_compute_tree_feature_importances_nested() {
        // Root splits on feature 0, left child splits on feature 1
        let tree = TreeNode::Node(Node {
            feature_idx: 0,
            threshold: 5.0,
            left: Box::new(TreeNode::Node(Node {
                feature_idx: 1,
                threshold: 2.0,
                left: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 0,
                    n_samples: 2,
                })),
                right: Box::new(TreeNode::Leaf(Leaf {
                    class_label: 1,
                    n_samples: 3,
                })),
            })),
            right: Box::new(TreeNode::Leaf(Leaf {
                class_label: 2,
                n_samples: 5,
            })),
        });
        let mut importances = vec![0.0; 3];
        compute_tree_feature_importances(&tree, &mut importances);
        // feature 0 at root: total samples = 2+3+5 = 10
        assert!((importances[0] - 10.0).abs() < 1e-7);
        // feature 1 at left subtree: total samples = 2+3 = 5
        assert!((importances[1] - 5.0).abs() < 1e-7);
        assert!((importances[2] - 0.0).abs() < 1e-7);
    }
}
