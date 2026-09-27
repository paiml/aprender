
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

/// Running count, mean and sum of squared deviations (`M2`) of a sequence of
/// targets, updated one value at a time (Welford), in `f64`.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct Moments {
    n: usize,
    mean: f64,
    m2: f64,
}

impl Moments {
    /// The moments with one more target.
    fn push(self, y: f32) -> Self {
        let n = self.n + 1;
        let y = f64::from(y);
        let delta = y - self.mean;
        let mean = self.mean + delta / n as f64;
        Self {
            n,
            mean,
            m2: self.m2 + delta * (y - mean),
        }
    }
}

/// Weighted within-child variance of a split, from each child's moments:
/// `(n_l/n)·(M2_l/n_l) + (n_r/n)·(M2_r/n_r) = (M2_l + M2_r)/n`.
pub(super) fn split_mse_from_moments(left: Moments, right: Moments) -> f32 {
    contract_pre_mse_split!();
    let n_total = (left.n + right.n) as f64;
    if n_total == 0.0 {
        return 0.0;
    }
    ((left.m2 + right.m2) / n_total) as f32
}

/// Rows `0..n_samples` in split order: every row with a non-NaN value on
/// `feature_idx`, ascending, then the NaN rows. Returns the order and the
/// number of non-NaN rows. A NaN row goes right at every threshold
/// (`NaN <= t` is false), so it sits after every boundary.
fn regression_split_order(
    x: &crate::primitives::Matrix<f32>,
    feature_idx: usize,
    n_samples: usize,
) -> (Vec<usize>, usize) {
    let (mut order, nan_rows): (Vec<usize>, Vec<usize>) =
        (0..n_samples).partition(|&row| !x.get(row, feature_idx).is_nan());
    order.sort_by(|&a, &b| x.get(a, feature_idx).total_cmp(&x.get(b, feature_idx)));
    let n_ordered = order.len();
    order.extend(nan_rows);
    (order, n_ordered)
}

/// `suffix[k]` = moments of the targets of `order[k..]`.
fn suffix_moments(order: &[usize], y: &[f32]) -> Vec<Moments> {
    let mut suffix = vec![Moments::default(); order.len() + 1];
    for k in (0..order.len()).rev() {
        suffix[k] = suffix[k + 1].push(y[order[k]]);
    }
    suffix
}

/// Find the best split for a single feature.
///
/// #3859: sort the rows once, then walk the candidate thresholds in ascending
/// order carrying the left child's moments forward and reading the right
/// child's from a precomputed suffix — `O(n log n)` per feature per node,
/// where the rescan it replaced was `O(n^2)`.
///
/// The candidates, the `<=` partition and the tie-breaking (first strictly
/// greater gain wins) are the rescan's. The variance is not: the rescan takes
/// an `f32` two-pass variance of each child in row order, this takes `f64`
/// running moments in sorted order. That reassociates the sums, so the gains
/// differ in the low bits and a near-tie can resolve to the other candidate.
/// `FALSIFY-DT-010` states and asserts the bound.
pub(super) fn find_best_regression_split_for_feature(
    x: &crate::primitives::Matrix<f32>,
    y: &[f32],
    feature_idx: usize,
    n_samples: usize,
    current_variance: f32,
) -> Option<(f32, f32)> {
    let (order, n_ordered) = regression_split_order(x, feature_idx, n_samples);
    let suffix = suffix_moments(&order, y);
    let values: Vec<f32> = order[..n_ordered]
        .iter()
        .map(|&row| x.get(row, feature_idx))
        .collect();

    let mut left = Moments::default();
    let mut boundary = 0;
    let mut best_threshold = 0.0;
    let mut best_gain = 0.0;

    for pair in values.windows(2).filter(|p| p[0] != p[1]) {
        let threshold = f32::midpoint(pair[0], pair[1]);
        while boundary < n_ordered && values[boundary] <= threshold {
            left = left.push(y[order[boundary]]);
            boundary += 1;
        }
        let right = suffix[boundary];
        if left.n == 0 || right.n == 0 {
            continue;
        }
        let gain = current_variance - split_mse_from_moments(left, right);
        if gain > best_gain {
            best_gain = gain;
            best_threshold = threshold;
        }
    }

    (best_gain > 0.0).then_some((best_threshold, best_gain))
}

/// Compute Mean Squared Error for a split.
///
/// Oracle-only since #3859. The rescan split search this fed was replaced by
/// the sort-once search in [`find_best_regression_split_for_feature`]; it is
/// kept verbatim under `cfg(test)` so the fast path is checked against the
/// algorithm it replaced (`FALSIFY-DT-010`), not against a second derivation
/// of the fast path's own premise.
#[cfg(test)]
pub(super) fn compute_mse(y_left: &[f32], y_right: &[f32]) -> f32 {
    contract_pre_mse_split!();
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
///
/// Oracle-only since #3859. The rescan split search this fed was replaced by
/// the sort-once search in [`find_best_regression_split_for_feature`]; it is
/// kept verbatim under `cfg(test)` so the fast path is checked against the
/// algorithm it replaced (`FALSIFY-DT-010`), not against a second derivation
/// of the fast path's own premise.
#[cfg(test)]
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
///
/// Oracle-only since #3859. The rescan split search this fed was replaced by
/// the sort-once search in [`find_best_regression_split_for_feature`]; it is
/// kept verbatim under `cfg(test)` so the fast path is checked against the
/// algorithm it replaced (`FALSIFY-DT-010`), not against a second derivation
/// of the fast path's own premise.
#[cfg(test)]
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
///
/// Oracle-only since #3859. The rescan split search this fed was replaced by
/// the sort-once search in [`find_best_regression_split_for_feature`]; it is
/// kept verbatim under `cfg(test)` so the fast path is checked against the
/// algorithm it replaced (`FALSIFY-DT-010`), not against a second derivation
/// of the fast path's own premise.
#[cfg(test)]
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

/// The pre-#3859 split search, kept verbatim as the oracle for
/// `FALSIFY-DT-010`.
///
/// `O(n)` candidate thresholds x `O(n)` rescan per candidate = `O(n^2)` per
/// feature per node.
#[cfg(test)]
pub(super) fn find_best_regression_split_for_feature_rescan(
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
        let threshold = f32::midpoint(feature_values[i], feature_values[i + 1]);
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
///
/// Stores the leaf's variance (impurity) so parent splits can use it in the MDI
/// feature-importance formula (see `tree-feature-importances-mdi-v1`).
pub(super) fn make_regression_leaf(y_slice: &[f32], n_samples: usize) -> RegressionTreeNode {
    RegressionTreeNode::Leaf(RegressionLeaf {
        value: mean_f32(y_slice),
        n_samples,
        impurity: variance_f32(y_slice),
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

    // Impurity (variance) of the samples reaching this node, computed BEFORE the
    // split. Drives MDI feature-importance (tree-feature-importances-mdi-v1).
    let node_impurity = variance_f32(&y_slice);

    // Early stopping checks
    if n_samples < min_samples_split || at_max_depth(depth, max_depth) || node_impurity < 1e-10 {
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
        impurity: node_impurity,
        n_node_samples: n_samples,
        left: Box::new(left_child),
        right: Box::new(right_child),
    })
}

// ============================================================================
// Feature Importance Helpers
// ============================================================================

/// Returns `(n_node_samples, impurity)` for any classification tree node.
///
/// Leaves carry their own sample count and gini impurity; internal nodes carry
/// the values recorded at fit time for the samples reaching them.
fn tree_node_n_and_impurity(node: &TreeNode) -> (f32, f32) {
    match node {
        TreeNode::Leaf(leaf) => (leaf.n_samples as f32, leaf.impurity),
        TreeNode::Node(n) => (n.n_node_samples as f32, n.impurity),
    }
}

/// Compute feature importances from a classification tree by traversing it.
///
/// Uses the sklearn Mean Decrease in Impurity (MDI) formula
/// (`tree/_tree.pyx::compute_feature_importances`): for each split node,
/// `importance[feature] += n_node·impurity − n_left·left_impurity − n_right·right_impurity`.
/// Consumers normalize the accumulated importances to sum 1.
pub(super) fn compute_tree_feature_importances(node: &TreeNode, importances: &mut [f32]) {
    match node {
        TreeNode::Leaf(_) => {
            // Leaf nodes don't make a split → no impurity decrease to attribute.
        }
        TreeNode::Node(n) => {
            let n_node = n.n_node_samples as f32;
            let (n_left, left_impurity) = tree_node_n_and_impurity(&n.left);
            let (n_right, right_impurity) = tree_node_n_and_impurity(&n.right);

            // MDI weighted impurity decrease for the split made at this node.
            importances[n.feature_idx] +=
                n_node * n.impurity - n_left * left_impurity - n_right * right_impurity;

            // Recursively process children
            compute_tree_feature_importances(&n.left, importances);
            compute_tree_feature_importances(&n.right, importances);
        }
    }
}

/// Count total samples in a tree/subtree.
///
/// Retained for test coverage of tree-sample accounting; MDI feature-importance
/// (the production consumer) now reads per-node `n_node_samples` directly.
#[cfg(test)]
pub(super) fn count_tree_samples(node: &TreeNode) -> usize {
    match node {
        TreeNode::Leaf(leaf) => leaf.n_samples,
        TreeNode::Node(n) => {
            // For internal nodes, sum samples from children
            count_tree_samples(&n.left) + count_tree_samples(&n.right)
        }
    }
}

/// Returns `(n_node_samples, impurity)` for any regression tree node.
///
/// Leaves carry their own sample count and variance; internal nodes carry the
/// values recorded at fit time for the samples reaching them.
fn regression_node_n_and_impurity(node: &RegressionTreeNode) -> (f32, f32) {
    match node {
        RegressionTreeNode::Leaf(leaf) => (leaf.n_samples as f32, leaf.impurity),
        RegressionTreeNode::Node(n) => (n.n_node_samples as f32, n.impurity),
    }
}

/// Compute feature importances from a regression tree by traversing it.
///
/// Uses the sklearn Mean Decrease in Impurity (MDI) formula
/// (`tree/_tree.pyx::compute_feature_importances`): for each split node,
/// `importance[feature] += n_node·variance − n_left·left_variance − n_right·right_variance`.
/// Consumers normalize the accumulated importances to sum 1.
pub(super) fn compute_regression_tree_feature_importances(
    node: &RegressionTreeNode,
    importances: &mut [f32],
) {
    match node {
        RegressionTreeNode::Leaf(_) => {
            // Leaf nodes don't make a split → no variance decrease to attribute.
        }
        RegressionTreeNode::Node(n) => {
            let n_node = n.n_node_samples as f32;
            let (n_left, left_impurity) = regression_node_n_and_impurity(&n.left);
            let (n_right, right_impurity) = regression_node_n_and_impurity(&n.right);

            // MDI weighted variance decrease for the split made at this node.
            importances[n.feature_idx] +=
                n_node * n.impurity - n_left * left_impurity - n_right * right_impurity;

            // Recursively process children
            compute_regression_tree_feature_importances(&n.left, importances);
            compute_regression_tree_feature_importances(&n.right, importances);
        }
    }
}

/// Count total samples in a regression tree/subtree.
///
/// Retained for test coverage of tree-sample accounting; MDI feature-importance
/// (the production consumer) now reads per-node `n_node_samples` directly.
#[cfg(test)]
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
    use rand::distr::{Distribution, Uniform};
    use rand::SeedableRng;

    let dist = Uniform::new(0, n_samples).expect("valid uniform range: 0..n_samples");

    let mut indices = Vec::with_capacity(n_samples);

    if let Some(seed) = random_state {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for _ in 0..n_samples {
            indices.push(dist.sample(&mut rng));
        }
    } else {
        let mut rng = rand::rng();
        for _ in 0..n_samples {
            indices.push(dist.sample(&mut rng));
        }
    }

    indices
}
