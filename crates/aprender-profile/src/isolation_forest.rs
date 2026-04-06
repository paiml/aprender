//! Isolation Forest for ML-based outlier detection (Sprint 22)
//!
//! Implements the Isolation Forest algorithm for unsupervised anomaly detection
//! with explainability features (XAI).
//!
//! # Algorithm Overview
//!
//! Isolation Forest isolates anomalies by randomly partitioning the feature space.
//! Anomalies are easier to isolate (shorter paths in trees) compared to normal points.
//!
//! # References
//!
//! Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest.
//! In 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422).

use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashMap;

/// Default sub-sampling size (following original paper)
const DEFAULT_SUBSAMPLE_SIZE: usize = 256;

/// A node in an Isolation Tree
#[derive(Debug, Clone)]
enum IsolationNode {
    /// Internal node with split feature and threshold
    Internal {
        feature_idx: usize,
        threshold: f64,
        left: Box<IsolationNode>,
        right: Box<IsolationNode>,
    },
    /// Leaf node with sample count (for path length calculation)
    Leaf { size: usize },
}

impl IsolationNode {
    /// Calculate path length from root to this node for a given sample
    fn path_length(&self, sample: &[f64], current_depth: usize) -> f64 {
        match self {
            IsolationNode::Internal { feature_idx, threshold, left, right } => {
                if sample[*feature_idx] < *threshold {
                    left.path_length(sample, current_depth + 1)
                } else {
                    right.path_length(sample, current_depth + 1)
                }
            }
            IsolationNode::Leaf { size } => {
                // Add average path length for unresolved instances
                current_depth as f64 + Self::average_path_length(*size)
            }
        }
    }

    /// Calculate average path length for BST with n samples (expected value)
    fn average_path_length(n: usize) -> f64 {
        if n <= 1 {
            return 0.0;
        }
        // Harmonic number approximation: H(n-1) ≈ ln(n-1) + γ
        const EULER_GAMMA: f64 = 0.5772156649;
        2.0 * ((n - 1) as f64).ln() + EULER_GAMMA - 2.0 * (n - 1) as f64 / n as f64
    }
}

/// Single Isolation Tree
#[derive(Debug, Clone)]
pub struct IsolationTree {
    root: IsolationNode,
}

impl IsolationTree {
    /// Build a tree from samples
    fn build(samples: &[Vec<f64>], max_depth: usize) -> Self {
        let root = Self::build_node(samples, 0, max_depth);
        IsolationTree { root }
    }

    /// Recursively build tree nodes
    fn build_node(samples: &[Vec<f64>], depth: usize, max_depth: usize) -> IsolationNode {
        // Base cases: stop splitting
        if samples.is_empty() {
            return IsolationNode::Leaf { size: 0 };
        }

        if depth >= max_depth || samples.len() <= 1 {
            return IsolationNode::Leaf { size: samples.len() };
        }

        // All samples are identical - create leaf
        if samples.windows(2).all(|w| w[0] == w[1]) {
            return IsolationNode::Leaf { size: samples.len() };
        }

        let num_features = samples[0].len();
        let mut rng = rand::thread_rng();

        // Randomly select a feature to split on
        let feature_idx = rng.gen_range(0..num_features);

        // Get min/max for this feature
        let mut min_val = f64::MAX;
        let mut max_val = f64::MIN;
        for sample in samples {
            let val = sample[feature_idx];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // If all values are the same for this feature, create leaf
        if (max_val - min_val).abs() < f64::EPSILON {
            return IsolationNode::Leaf { size: samples.len() };
        }

        // Random split threshold between min and max
        let threshold = rng.gen_range(min_val..max_val);

        // Partition samples
        let (left_samples, right_samples): (Vec<Vec<f64>>, Vec<Vec<f64>>) =
            samples.iter().cloned().partition(|sample| sample[feature_idx] < threshold);

        // If partition is empty on one side, create leaf
        if left_samples.is_empty() || right_samples.is_empty() {
            return IsolationNode::Leaf { size: samples.len() };
        }

        // Recursively build children
        let left = Box::new(Self::build_node(&left_samples, depth + 1, max_depth));
        let right = Box::new(Self::build_node(&right_samples, depth + 1, max_depth));

        IsolationNode::Internal { feature_idx, threshold, left, right }
    }

    /// Calculate path length for a sample
    fn path_length(&self, sample: &[f64]) -> f64 {
        self.root.path_length(sample, 0)
    }
}

/// Isolation Forest - ensemble of Isolation Trees
pub struct IsolationForest {
    trees: Vec<IsolationTree>,
    num_trees: usize,
    subsample_size: usize,
}

impl IsolationForest {
    /// Create a new Isolation Forest
    pub fn new(num_trees: usize, subsample_size: Option<usize>) -> Self {
        IsolationForest {
            trees: Vec::new(),
            num_trees,
            subsample_size: subsample_size.unwrap_or(DEFAULT_SUBSAMPLE_SIZE),
        }
    }

    /// Fit the model on training data
    pub fn fit(&mut self, samples: &[Vec<f64>]) {
        let mut rng = rand::thread_rng();
        let max_depth = (self.subsample_size as f64).log2().ceil() as usize;

        for _ in 0..self.num_trees {
            // Sub-sample data
            let sample_size = self.subsample_size.min(samples.len());
            let mut indices: Vec<_> = (0..samples.len()).collect();
            indices.shuffle(&mut rng);
            let subsamples: Vec<_> =
                indices[..sample_size].iter().map(|&i| samples[i].clone()).collect();

            // Build tree
            let tree = IsolationTree::build(&subsamples, max_depth);
            self.trees.push(tree);
        }
    }

    /// Calculate anomaly score for a sample (higher = more anomalous)
    /// Returns score in range [0, 1]
    pub fn anomaly_score(&self, sample: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }

        if self.trees.is_empty() {
            return 0.5; // No trees → indeterminate score
        }

        // Average path length across all trees
        let avg_path_length: f64 =
            self.trees.iter().map(|tree| tree.path_length(sample)).sum::<f64>()
                / self.trees.len() as f64;

        // Normalize by expected path length
        let c = IsolationNode::average_path_length(self.subsample_size);
        if c == 0.0 {
            return 0.5; // subsample_size <= 1 → indeterminate
        }
        2_f64.powf(-avg_path_length / c)
    }

    /// Predict outliers based on contamination threshold
    /// Returns true if sample is an outlier
    pub fn predict(&self, sample: &[f64], contamination: f32) -> bool {
        let score = self.anomaly_score(sample);
        // Scores close to 1.0 are anomalies, close to 0.5 are normal
        score > 0.5 + (f64::from(contamination) / 2.0)
    }
}

/// Feature extracted from syscall data
#[derive(Debug, Clone)]
pub struct SyscallFeature {
    pub syscall_name: String,
    pub avg_duration_us: f64,
    pub call_count: u64,
    pub total_duration_us: f64,
}

/// Extract features from syscall statistics
pub fn extract_features(
    syscall_data: &HashMap<String, (u64, u64)>,
) -> (Vec<String>, Vec<Vec<f64>>) {
    let mut syscall_names = Vec::new();
    let mut features = Vec::new();

    for (name, (count, total_time_ns)) in syscall_data {
        if *count == 0 {
            continue;
        }

        let total_time_us = *total_time_ns as f64 / 1000.0;
        let avg_time_us = total_time_us / *count as f64;

        syscall_names.push(name.clone());

        // Feature vector: [avg_duration, call_count, total_duration]
        features.push(vec![
            avg_time_us,
            (*count as f64).ln().max(0.0), // Log scale for count
            total_time_us.ln().max(0.0),   // Log scale for total time
        ]);
    }

    (syscall_names, features)
}

/// Outlier detected by Isolation Forest
#[derive(Debug, Clone)]
pub struct Outlier {
    pub syscall: String,
    pub anomaly_score: f64,
    pub avg_duration_us: f64,
    pub call_count: u64,
    pub feature_importance: Vec<(String, f64)>,
}

/// Result of Isolation Forest analysis
#[derive(Debug, Clone)]
pub struct OutlierReport {
    pub outliers: Vec<Outlier>,
    pub total_samples: usize,
    pub contamination: f32,
    pub num_trees: usize,
}

/// Analyze syscall data for outliers using Isolation Forest
pub fn analyze_outliers(
    syscall_data: &HashMap<String, (u64, u64)>,
    num_trees: usize,
    contamination: f32,
    explain: bool,
) -> OutlierReport {
    // Extract features
    let (syscall_names, features) = extract_features(syscall_data);

    if features.len() < 2 {
        // Insufficient data
        return OutlierReport {
            outliers: Vec::new(),
            total_samples: features.len(),
            contamination,
            num_trees,
        };
    }

    // Train Isolation Forest
    let mut forest = IsolationForest::new(num_trees, None);
    forest.fit(&features);

    // Detect outliers
    let mut outliers = Vec::new();

    for (name, feature_vec) in syscall_names.iter().zip(features.iter()) {
        let score = forest.anomaly_score(feature_vec);
        let is_outlier = forest.predict(feature_vec, contamination);

        if is_outlier {
            let (count, total_time_ns) = syscall_data[name];
            let avg_duration_us = total_time_ns as f64 / 1000.0 / count as f64;

            let feature_importance =
                if explain { calculate_feature_importance(feature_vec) } else { Vec::new() };

            outliers.push(Outlier {
                syscall: name.clone(),
                anomaly_score: score,
                avg_duration_us,
                call_count: count,
                feature_importance,
            });
        }
    }

    // Sort by anomaly score (highest first, handle NaN gracefully)
    outliers.sort_by(|a, b| {
        b.anomaly_score.partial_cmp(&a.anomaly_score).unwrap_or(std::cmp::Ordering::Equal)
    });

    OutlierReport { outliers, total_samples: features.len(), contamination, num_trees }
}

/// Calculate feature importance for explainability (XAI)
fn calculate_feature_importance(features: &[f64]) -> Vec<(String, f64)> {
    let feature_names = ["avg_duration", "call_frequency", "total_duration"];

    // Simple feature importance: normalized absolute values
    let total: f64 = features.iter().map(|&f| f.abs()).sum();

    feature_names
        .iter()
        .zip(features.iter())
        .map(|(name, &value)| {
            let importance = if total > 0.0 { (value.abs() / total) * 100.0 } else { 0.0 };
            ((*name).to_string(), importance)
        })
        .collect()
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(IsolationTree: Send, Sync);
static_assertions::assert_impl_all!(IsolationForest: Send, Sync);
static_assertions::assert_impl_all!(SyscallFeature: Send, Sync);
static_assertions::assert_impl_all!(Outlier: Send, Sync);
static_assertions::assert_impl_all!(OutlierReport: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_tree_creation() {
        let samples = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![10.0, 20.0], // Outlier
        ];

        // Use multiple trees to average out randomness
        let mut outlier_total = 0.0;
        let mut normal_total = 0.0;
        for _ in 0..10 {
            let tree = IsolationTree::build(&samples, 10);
            outlier_total += tree.path_length(&[10.0, 20.0]);
            normal_total += tree.path_length(&[1.0, 2.0]);
        }
        let outlier_avg = outlier_total / 10.0;
        let normal_avg = normal_total / 10.0;

        // Outlier should have shorter average path (more isolated)
        assert!(
            outlier_avg < normal_avg,
            "Outlier avg ({}) should be < normal avg ({})",
            outlier_avg,
            normal_avg
        );
    }

    #[test]
    fn test_isolation_forest_detects_outliers() {
        let samples = vec![
            vec![1.0, 2.0],
            vec![1.1, 2.1],
            vec![0.9, 1.9],
            vec![1.2, 2.2],
            vec![10.0, 20.0], // Clear outlier
        ];

        let mut forest = IsolationForest::new(100, Some(4));
        forest.fit(&samples);

        let outlier_score = forest.anomaly_score(&[10.0, 20.0]);
        let normal_score = forest.anomaly_score(&[1.0, 2.0]);

        // Outlier should have higher score
        assert!(
            outlier_score > normal_score,
            "Outlier score ({}) should be > normal score ({})",
            outlier_score,
            normal_score
        );
        assert!(outlier_score > 0.50, "Outlier score ({}) should be > 0.50", outlier_score);
        // Should be anomalous (>0.5 is anomaly baseline)
    }

    #[test]
    fn test_feature_extraction() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (100, 1_000_000)); // 1ms total, 10us avg
        data.insert("read".to_string(), (10, 10_000_000)); // 10ms total, 1000us avg (outlier)

        let (names, features) = extract_features(&data);

        assert_eq!(names.len(), 2);
        assert_eq!(features.len(), 2);
        assert_eq!(features[0].len(), 3); // 3 features per syscall
    }

    #[test]
    fn test_analyze_outliers() {
        let mut data = HashMap::new();
        // Normal syscalls
        data.insert("write".to_string(), (100, 1_000_000));
        data.insert("read".to_string(), (100, 1_000_000));
        // Outlier - very slow
        data.insert("slow_syscall".to_string(), (10, 100_000_000));

        let report = analyze_outliers(&data, 100, 0.1, false);

        assert!(!report.outliers.is_empty());
        assert_eq!(report.total_samples, 3);
    }

    #[test]
    fn test_feature_importance() {
        let features = vec![10.0, 5.0, 2.0];
        let importance = calculate_feature_importance(&features);

        assert_eq!(importance.len(), 3);
        // Sum should be ~100%
        let total: f64 = importance.iter().map(|(_, v)| v).sum();
        assert!((total - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_average_path_length() {
        // Test known values from paper
        assert_eq!(IsolationNode::average_path_length(1), 0.0);
        let apl_10 = IsolationNode::average_path_length(10);
        assert!(apl_10 > 2.0 && apl_10 < 4.0); // Reasonable range
    }

    #[test]
    fn test_insufficient_data() {
        let mut data = HashMap::new();
        data.insert("write".to_string(), (1, 1000));

        let report = analyze_outliers(&data, 10, 0.1, false);
        assert_eq!(report.outliers.len(), 0);
        assert_eq!(report.total_samples, 1);
    }
}
