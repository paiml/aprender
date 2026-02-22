//! Isolation Forest for anomaly detection.
//!
//! Uses an ensemble of isolation trees to detect outliers based on path length.
//! Anomalies are easier to isolate (shorter paths) than normal points.

use crate::error::Result;
use crate::primitives::Matrix;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Internal node structure for Isolation Tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsolationNode {
    /// Feature index to split on (None for leaf)
    split_feature: Option<usize>,
    /// Split value (None for leaf)
    split_value: Option<f32>,
    /// Left child (samples with feature < `split_value`)
    left: Option<Box<IsolationNode>>,
    /// Right child (samples with feature >= `split_value`)
    right: Option<Box<IsolationNode>>,
    /// Size of node (for path length calculation)
    size: usize,
}

/// Single Isolation Tree for anomaly detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct IsolationTree {
    root: Option<IsolationNode>,
    max_depth: usize,
}

impl IsolationTree {
    fn new(max_depth: usize) -> Self {
        Self {
            root: None,
            max_depth,
        }
    }

    fn fit(&mut self, x: &Matrix<f32>, rng: &mut impl rand::Rng) {
        let indices: Vec<usize> = (0..x.shape().0).collect();
        self.root = Some(self.build_tree(x, &indices, 0, rng));
    }

    fn build_tree(
        &self,
        x: &Matrix<f32>,
        indices: &[usize],
        depth: usize,
        rng: &mut impl rand::Rng,
    ) -> IsolationNode {
        let n_samples = indices.len();

        // Terminal conditions
        if depth >= self.max_depth || n_samples <= 1 {
            return IsolationNode {
                split_feature: None,
                split_value: None,
                left: None,
                right: None,
                size: n_samples,
            };
        }

        // Random feature selection
        let n_features = x.shape().1;
        let feature_idx = rng.random_range(0..n_features);

        // Find min/max for this feature in current samples
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        for &idx in indices {
            let val = x.get(idx, feature_idx);
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // If all values are the same, make leaf
        if (max_val - min_val).abs() < 1e-10 {
            return IsolationNode {
                split_feature: None,
                split_value: None,
                left: None,
                right: None,
                size: n_samples,
            };
        }

        // Random split value between min and max
        let split_val = rng.random_range(min_val..max_val);

        // Partition samples
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        for &idx in indices {
            if x.get(idx, feature_idx) < split_val {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // If split doesn't separate, make leaf
        if left_indices.is_empty() || right_indices.is_empty() {
            return IsolationNode {
                split_feature: None,
                split_value: None,
                left: None,
                right: None,
                size: n_samples,
            };
        }

        // Recursively build children
        let left = self.build_tree(x, &left_indices, depth + 1, rng);
        let right = self.build_tree(x, &right_indices, depth + 1, rng);

        IsolationNode {
            split_feature: Some(feature_idx),
            split_value: Some(split_val),
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            size: n_samples,
        }
    }

    fn path_length(&self, sample: &[f32]) -> f32 {
        if let Some(ref root) = self.root {
            self.path_length_recursive(sample, root, 0.0)
        } else {
            0.0
        }
    }

    #[allow(clippy::self_only_used_in_recursion)]
    fn path_length_recursive(&self, sample: &[f32], node: &IsolationNode, depth: f32) -> f32 {
        // Leaf node - add average path length for remaining samples
        if node.split_feature.is_none() {
            return depth + Self::c(node.size);
        }

        let feature_idx = node
            .split_feature
            .expect("Split feature must exist for non-leaf nodes");
        let split_val = node
            .split_value
            .expect("Split value must exist for non-leaf nodes");

        if sample[feature_idx] < split_val {
            if let Some(ref left) = node.left {
                self.path_length_recursive(sample, left, depth + 1.0)
            } else {
                depth + Self::c(node.size)
            }
        } else if let Some(ref right) = node.right {
            self.path_length_recursive(sample, right, depth + 1.0)
        } else {
            depth + Self::c(node.size)
        }
    }

    /// Average path length of unsuccessful search in BST (for normalization)
    fn c(n: usize) -> f32 {
        if n <= 1 {
            0.0
        } else if n == 2 {
            1.0
        } else {
            let n_f32 = n as f32;
            2.0 * ((n_f32 - 1.0).ln() + 0.577_215_7) - 2.0 * (n_f32 - 1.0) / n_f32
        }
    }
}

/// Isolation Forest for anomaly detection.
///
/// Uses an ensemble of isolation trees to detect outliers based on path length.
/// Anomalies are easier to isolate (shorter paths) than normal points.
///
/// # Algorithm
///
/// 1. Build N isolation trees on random subsamples
/// 2. Each tree recursively splits data by random feature + random threshold
/// 3. Compute average path length across all trees
/// 4. Convert to anomaly score (shorter path = more anomalous)
/// 5. Use contamination parameter to set classification threshold
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(
///     6,
///     2,
///     vec![
///         2.0, 2.0, 2.1, 2.0, 1.9, 2.1, 2.0, 1.9,  // Normal
///         10.0, 10.0, -10.0, -10.0,                 // Outliers
///     ],
/// )
/// .expect("Valid matrix dimensions and data length");
///
/// let mut iforest = IsolationForest::new()
///     .with_contamination(0.3)
///     .with_random_state(42);
/// iforest.fit(&data).expect("Fit succeeds with valid data");
///
/// // Predict returns 1 for normal, -1 for anomaly
/// let predictions = iforest.predict(&data);
///
/// // score_samples returns anomaly scores (lower = more anomalous)
/// let scores = iforest.score_samples(&data);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n log m) where n=samples, `m=max_samples`
/// - Space complexity: O(t * m) where `t=n_estimators`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest {
    /// Number of trees in the ensemble
    n_estimators: usize,
    /// Number of samples to draw for each tree
    max_samples: Option<usize>,
    /// Expected proportion of anomalies in the dataset
    contamination: f32,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Maximum tree depth
    max_depth: usize,
    /// Ensemble of isolation trees
    trees: Vec<IsolationTree>,
    /// Threshold for anomaly classification (computed from contamination)
    threshold: Option<f32>,
    /// Average path length normalization constant
    c_norm: f32,
}

impl IsolationForest {
    /// Create a new Isolation Forest with default parameters.
    ///
    /// Default: 100 trees, auto `max_samples`, 0.1 contamination
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_samples: None,
            contamination: 0.1,
            random_state: None,
            max_depth: 10,
            trees: Vec::new(),
            threshold: None,
            c_norm: 1.0,
        }
    }

    /// Set the number of trees in the ensemble.
    #[must_use]
    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the number of samples to draw for each tree.
    #[must_use]
    pub fn with_max_samples(mut self, max_samples: usize) -> Self {
        self.max_samples = Some(max_samples);
        self
    }

    /// Set the expected proportion of anomalies (0 to 0.5).
    #[must_use]
    pub fn with_contamination(mut self, contamination: f32) -> Self {
        self.contamination = contamination.clamp(0.0, 0.5);
        self
    }

    /// Set random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Check if model has been fitted.
    #[must_use]
    pub fn is_fitted(&self) -> bool {
        !self.trees.is_empty()
    }

    /// Fit the Isolation Forest on training data.
    pub fn fit(&mut self, x: &Matrix<f32>) -> Result<()> {
        let (n_samples, _n_features) = x.shape();

        // Determine max_samples (default: min(256, n_samples))
        let max_samples = self.max_samples.unwrap_or_else(|| n_samples.min(256));

        // Compute normalization constant
        self.c_norm = IsolationTree::c(max_samples);

        // Compute max tree depth
        self.max_depth = (max_samples as f32).log2().ceil() as usize;

        // Initialize RNG
        let mut rng: Box<dyn rand::RngCore> = if let Some(seed) = self.random_state {
            Box::new(rand::rngs::StdRng::seed_from_u64(seed))
        } else {
            Box::new(rand::rngs::StdRng::from_os_rng())
        };

        // Build ensemble of trees
        self.trees.clear();
        for _ in 0..self.n_estimators {
            // Sample random subset
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(max_samples);

            // Extract subsample
            let subsample = self.extract_subsample(x, &indices);

            // Build tree
            let mut tree = IsolationTree::new(self.max_depth);
            tree.fit(&subsample, &mut rng);
            self.trees.push(tree);
        }

        // Compute anomaly scores on training data to determine threshold
        let scores = self.score_samples(x);
        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("Anomaly scores must be valid floats for comparison")
        });

        // Threshold at contamination quantile
        let threshold_idx = (self.contamination * n_samples as f32) as usize;
        self.threshold = Some(sorted_scores[threshold_idx.min(n_samples - 1)]);

        Ok(())
    }

    /// Extract subsample from data.
    #[allow(clippy::unused_self)]
    fn extract_subsample(&self, x: &Matrix<f32>, indices: &[usize]) -> Matrix<f32> {
        let (_, n_features) = x.shape();
        let n_samples = indices.len();
        let mut data = vec![0.0; n_samples * n_features];

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..n_features {
                data[i * n_features + j] = x.get(idx, j);
            }
        }

        Matrix::from_vec(n_samples, n_features, data)
            .expect("Subsampled matrix dimensions match collected data length")
    }

    /// Compute anomaly scores for samples.
    ///
    /// Returns a vector of scores where lower scores indicate higher anomaly likelihood.
    #[allow(clippy::needless_range_loop)]
    #[must_use]
    pub fn score_samples(&self, x: &Matrix<f32>) -> Vec<f32> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");

        let (n_samples, n_features) = x.shape();
        let mut scores = vec![0.0; n_samples];

        for i in 0..n_samples {
            // Extract sample
            let sample: Vec<f32> = (0..n_features).map(|j| x.get(i, j)).collect();

            // Average path length across all trees
            let avg_path_length: f32 = self
                .trees
                .iter()
                .map(|tree| tree.path_length(&sample))
                .sum::<f32>()
                / self.n_estimators as f32;

            // Anomaly score: 2^(-avg_path / c_norm)
            // Scores close to 1 = anomaly, close to 0 = normal
            let score = 2f32.powf(-avg_path_length / self.c_norm);

            // Invert so lower = more anomalous (for consistency with decision_function)
            scores[i] = -score;
        }

        scores
    }

    /// Predict anomaly labels for samples.
    ///
    /// Returns 1 for normal points and -1 for anomalies.
    #[must_use]
    pub fn predict(&self, x: &Matrix<f32>) -> Vec<i32> {
        assert!(self.is_fitted(), "Model not fitted. Call fit() first.");

        let threshold = self
            .threshold
            .expect("Threshold must be set during fit phase");
        let scores = self.score_samples(x);

        scores
            .iter()
            .map(|&score| if score < threshold { -1 } else { 1 })
            .collect()
    }
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}
