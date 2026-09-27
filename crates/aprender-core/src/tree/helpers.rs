//! Helper functions for tree building algorithms.
//!
//! This module contains internal helper functions used by decision tree
//! and ensemble methods.

use super::{Leaf, Node, RegressionLeaf, RegressionNode, RegressionTreeNode, TreeNode};
use std::collections::HashSet;

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
        // Leaf node. The flat SafeTensors schema does not persist per-node
        // impurity / n_node_samples, so they default to 0 on reconstruct. These
        // fields only feed MDI feature-importance, which is computed on the live
        // fitted tree, not a deserialized one.
        TreeNode::Leaf(Leaf {
            class_label: classes[idx] as usize,
            n_samples: samples[idx] as usize,
            impurity: 0.0,
        })
    } else {
        // Internal node
        let left_idx = left_children[idx] as usize;
        let right_idx = right_children[idx] as usize;

        TreeNode::Node(Node {
            feature_idx: features[idx] as usize,
            threshold: thresholds[idx],
            impurity: 0.0,
            n_node_samples: 0,
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
// Contract: decision-tree-v1, equation = "gini_impurity"

pub fn gini_impurity(labels: &[usize]) -> f32 {
    contract_pre_gini_impurity!();
    if labels.is_empty() {
        return 0.0;
    }

    // Count occurrences of each class (BTreeMap for deterministic iteration order)
    let mut counts = std::collections::BTreeMap::new();
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
// Contract: decision-tree-v1, equation = "gini_split"

pub fn gini_split(left_labels: &[usize], right_labels: &[usize]) -> f32 {
    contract_pre_gini_split!();
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
/// Oracle-only since #3815. The rescan split search this fed was replaced by
/// the single-pass one in [`find_best_split_for_feature`]; it is kept intact,
/// verbatim, under `cfg(test)` so the fast path is checked against the
/// algorithm it replaced (`FALSIFY-DT-008`) rather than against a second
/// derivation of the fast path's own premise.
#[cfg(test)]
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
///
/// Oracle-only since #3815 — see [`get_sorted_unique_values`]. This full
/// rescan of all `n` samples, once per candidate threshold, was the quadratic
/// term.
#[cfg(test)]
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
///
/// Oracle-only since #3815 — see [`get_sorted_unique_values`].
#[cfg(test)]
pub(super) fn calculate_information_gain(
    current_impurity: f32,
    left_labels: &[usize],
    right_labels: &[usize],
) -> f32 {
    let split_impurity = gini_split(left_labels, right_labels);
    current_impurity - split_impurity
}

/// The pre-#3815 split search, kept verbatim as the oracle for
/// `FALSIFY-DT-008`.
///
/// `O(n)` candidate thresholds x `O(n)` rescan per candidate = `O(n^2)` per
/// feature per node. [`find_best_split_for_feature`] must return exactly this,
/// bit for bit, for every input.
#[cfg(test)]
pub(super) fn find_best_split_for_feature_rescan(x: &[f32], y: &[usize]) -> Option<(f32, f32)> {
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
        let threshold = f32::midpoint(unique_values[i], unique_values[i + 1]);

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

/// Gini impurity of a class-count histogram.
///
/// The counting form of [`gini_impurity`] (`decision-tree-v1`, equation
/// `gini_impurity`): same `1 - sum p_k^2`, same ascending-class order, same
/// f32 operation order — so the two agree bit for bit. That is what lets
/// #3815's single-pass search keep the rescan's exact thresholds. Zero counts
/// are skipped because subtracting `0.0` is exact, so they cannot change the
/// result either way.
fn gini_from_counts(counts: impl Iterator<Item = usize>, n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }

    let n_f = n as f32;
    let mut gini = 1.0;

    for count in counts {
        if count == 0 {
            continue;
        }
        let p = count as f32 / n_f;
        gini -= p * p;
    }

    gini
}

/// Find the best split for a given feature.
///
/// Sorts `(x_i, y_i)` by `x_i` once, then walks the candidate thresholds in
/// ascending order carrying running per-class counts, so each candidate is
/// scored from those counts in `O(K)` instead of re-partitioning all `n`
/// samples. `O(n log n + n*K)` per feature per node, against the `O(n^2)` of
/// the rescan it replaces (#3815) — which is why a 4-feature depth-4 tree took
/// 52.8s at 40k rows and blocked fitting past ~100k entirely.
///
/// **The result is bit-identical to the rescan, not merely equivalent.** The
/// candidate set (midpoints between values distinct under the same chained
/// `1e-10` tolerance), the arithmetic ([`gini_from_counts`] reproduces
/// [`gini_impurity`]'s exact f32 operation order) and the strict `>`
/// tie-breaking are all unchanged, and `FALSIFY-DT-008` asserts equality of
/// raw bits against [`find_best_split_for_feature_rescan`].
pub(super) fn find_best_split_for_feature(x: &[f32], y: &[usize]) -> Option<(f32, f32)> {
    let n = x.len();
    if n < 2 {
        return None;
    }

    // Same comparator as the rescan's unique-value scan: a NaN feature is a
    // programmer error here and still panics rather than sorting arbitrarily.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        x[a].partial_cmp(&x[b])
            .expect("f32 values should be comparable")
    });

    // Candidate thresholds are midpoints between adjacent DISTINCT values,
    // distinct under the rescan's chained 1e-10 tolerance (compared against the
    // last value kept, not the immediate predecessor).
    let mut unique_values = Vec::new();
    let mut prev_val = x[order[0]];
    unique_values.push(prev_val);
    for &idx in order.get(1..).unwrap_or(&[]) {
        if (x[idx] - prev_val).abs() > 1e-10 {
            unique_values.push(x[idx]);
            prev_val = x[idx];
        }
    }
    if unique_values.len() < 2 {
        return None;
    }

    // Dense class ids in ascending label order, so the running counts can be
    // summed in the order `gini_impurity`'s BTreeMap yields.
    let class_ids: Vec<usize> = {
        let labels: std::collections::BTreeSet<usize> = y.iter().copied().collect();
        let index: std::collections::BTreeMap<usize, usize> = labels
            .into_iter()
            .enumerate()
            .map(|(i, l)| (l, i))
            .collect();
        y.iter().map(|label| index[label]).collect()
    };
    let n_classes = class_ids.iter().copied().max().map_or(0, |m| m + 1);

    let mut total_counts = vec![0usize; n_classes];
    for &class_id in &class_ids {
        total_counts[class_id] += 1;
    }

    let current_impurity = gini_impurity(y);
    let mut best_gain = 0.0;
    let mut best_threshold = 0.0;

    let mut left_counts = vec![0usize; n_classes];
    let mut n_left = 0usize;
    let mut cursor = 0usize;

    for i in 0..unique_values.len() - 1 {
        let threshold = f32::midpoint(unique_values[i], unique_values[i + 1]);

        // Sorted order means every sample with `value <= threshold` is a prefix,
        // and thresholds are non-decreasing, so the cursor only moves forward:
        // O(n) in total across every candidate, not O(n) per candidate.
        while cursor < n && x[order[cursor]] <= threshold {
            left_counts[class_ids[order[cursor]]] += 1;
            n_left += 1;
            cursor += 1;
        }

        // The rescan skipped a threshold whose partition left either side empty.
        if n_left == 0 || n_left == n {
            continue;
        }

        let n_left_f = n_left as f32;
        let n_right_f = (n - n_left) as f32;
        let n_total = n_left_f + n_right_f;
        let gini_left = gini_from_counts(left_counts.iter().copied(), n_left);
        let gini_right = gini_from_counts(
            total_counts
                .iter()
                .zip(left_counts.iter())
                .map(|(total, left)| total - left),
            n - n_left,
        );
        let split_impurity = (n_left_f / n_total) * gini_left + (n_right_f / n_total) * gini_right;
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

pub(super) fn majority_class(labels: &[usize]) -> usize {
    let mut counts = std::collections::BTreeMap::new();
    for &label in labels {
        *counts.entry(label).or_insert(0usize) += 1;
    }
    // BTreeMap iterates in key order → deterministic tie-breaking (lowest class wins)
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .expect("at least one label should exist")
        .0
}

/// Split data into subsets based on indices.

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
            impurity: gini_impurity(y),
        }));
    }

    // Criterion 2: Max depth reached
    if let Some(max_d) = max_depth {
        if depth >= max_d {
            return Some(TreeNode::Leaf(Leaf {
                class_label: majority_class(y),
                n_samples,
                impurity: gini_impurity(y),
            }));
        }
    }

    None
}

/// Split data indices based on feature threshold.

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

pub(super) fn build_tree(
    x: &crate::primitives::Matrix<f32>,
    y: &[usize],
    depth: usize,
    max_depth: Option<usize>,
) -> TreeNode {
    let n_samples = y.len();

    // Impurity (gini) of the samples reaching this node, computed BEFORE the
    // split. Drives MDI feature-importance (tree-feature-importances-mdi-v1).
    let node_impurity = gini_impurity(y);

    // Check stopping criteria
    if let Some(leaf) = check_stopping_criteria(y, depth, max_depth) {
        return leaf;
    }

    // Try to find best split
    let Some((feature_idx, threshold, _gain)) = find_best_split(x, y) else {
        return TreeNode::Leaf(Leaf {
            class_label: majority_class(y),
            n_samples,
            impurity: node_impurity,
        });
    };

    // Split data based on threshold
    let Some((left_indices, right_indices)) =
        split_indices_by_threshold(x, feature_idx, threshold, n_samples)
    else {
        return TreeNode::Leaf(Leaf {
            class_label: majority_class(y),
            n_samples,
            impurity: node_impurity,
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
        impurity: node_impurity,
        n_node_samples: n_samples,
        left: Box::new(left_child),
        right: Box::new(right_child),
    })
}

include!("regression_helpers.rs");
include!("helpers_tests.rs");
