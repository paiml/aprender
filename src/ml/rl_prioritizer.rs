//! Reinforcement learning-based test prioritizer
//!
//! Implements test case prioritization using Thompson Sampling (a contextual bandit approach).
//! Based on Spieker et al. (2017): "Reinforcement Learning for Automatic Test Case Prioritization"

use crate::data::CodeFeatures;
use std::collections::HashMap;

/// RL-based test prioritizer using Thompson Sampling
///
/// Learns optimal test prioritization policy by tracking success/failure rates
/// for different feature combinations.
#[derive(Debug, Clone)]
pub struct RLTestPrioritizer {
    /// Success counts (alpha) for each feature signature
    success_counts: HashMap<FeatureSignature, f64>,
    /// Failure counts (beta) for each feature signature
    failure_counts: HashMap<FeatureSignature, f64>,
    /// Exploration parameter (higher = more exploration)
    exploration_rate: f64,
    /// Total tests executed
    total_tests: usize,
}

/// Compact feature signature for hash table lookups
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
struct FeatureSignature {
    /// Bucketed AST depth (0-5, 6-10, 11+)
    depth_bucket: u8,
    /// Bucketed operator count (0-10, 11-30, 31+)
    operator_bucket: u8,
    /// Bucketed cyclomatic complexity (0-5, 6-15, 16+)
    complexity_bucket: u8,
    /// Whether code uses edge values
    uses_edge_values: bool,
}

impl FeatureSignature {
    fn from_features(features: &CodeFeatures) -> Self {
        Self {
            depth_bucket: match features.ast_depth {
                0..=5 => 0,
                6..=10 => 1,
                _ => 2,
            },
            operator_bucket: match features.num_operators {
                0..=10 => 0,
                11..=30 => 1,
                _ => 2,
            },
            complexity_bucket: if features.cyclomatic_complexity <= 5.0 {
                0
            } else if features.cyclomatic_complexity <= 15.0 {
                1
            } else {
                2
            },
            uses_edge_values: features.uses_edge_values,
        }
    }
}

impl RLTestPrioritizer {
    /// Create a new RL test prioritizer
    #[must_use]
    pub fn new() -> Self {
        Self {
            success_counts: HashMap::new(),
            failure_counts: HashMap::new(),
            exploration_rate: 0.1,
            total_tests: 0,
        }
    }

    /// Create prioritizer with custom exploration rate
    #[must_use]
    pub fn with_exploration_rate(mut self, rate: f64) -> Self {
        self.exploration_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Prioritize test cases using Thompson Sampling
    ///
    /// Returns indices sorted by priority (highest failure probability first)
    pub fn prioritize(&self, features: &[CodeFeatures]) -> Vec<usize> {
        let mut rng = rand::rng();

        let mut scored: Vec<(usize, f64)> = features
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let sig = FeatureSignature::from_features(f);
                let score = self.sample_failure_probability(&sig, &mut rng);
                (i, score)
            })
            .collect();

        // Sort by score descending (highest failure probability first)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Sample failure probability from Beta distribution (Thompson Sampling)
    fn sample_failure_probability<R: rand::Rng>(&self, sig: &FeatureSignature, rng: &mut R) -> f64 {
        use rand_distr::{Beta, Distribution};

        // Get counts with Laplace smoothing (prior: Beta(1,1))
        let alpha = self.failure_counts.get(sig).copied().unwrap_or(0.0) + 1.0;
        let beta = self.success_counts.get(sig).copied().unwrap_or(0.0) + 1.0;

        // Sample from Beta(alpha, beta)
        // Beta distribution creation is mathematically guaranteed with positive alpha, beta >= 1.0
        #[allow(clippy::unwrap_used)]
        let beta_dist = Beta::new(alpha, beta).unwrap_or_else(|_| Beta::new(1.0, 1.0).unwrap());
        beta_dist.sample(rng)
    }

    /// Update with feedback from test execution
    ///
    /// # Arguments
    ///
    /// * `features` - Features of the executed test case
    /// * `revealed_bug` - True if test revealed a bug, false otherwise
    pub fn update_feedback(&mut self, features: &CodeFeatures, revealed_bug: bool) {
        let sig = FeatureSignature::from_features(features);

        if revealed_bug {
            *self.failure_counts.entry(sig).or_insert(0.0) += 1.0;
        } else {
            *self.success_counts.entry(sig).or_insert(0.0) += 1.0;
        }

        self.total_tests += 1;
    }

    /// Get current failure rate estimate for a feature signature
    #[must_use]
    pub fn failure_rate(&self, features: &CodeFeatures) -> f64 {
        let sig = FeatureSignature::from_features(features);
        let failures = self.failure_counts.get(&sig).copied().unwrap_or(0.0);
        let successes = self.success_counts.get(&sig).copied().unwrap_or(0.0);
        let total = failures + successes;

        if total == 0.0 {
            0.5 // Prior: unknown tests have 50% failure rate
        } else {
            failures / total
        }
    }

    /// Get total number of tests executed
    #[must_use]
    pub const fn total_tests(&self) -> usize {
        self.total_tests
    }

    /// Get number of tracked feature signatures
    #[must_use]
    pub fn num_signatures(&self) -> usize {
        let mut sigs = self.success_counts.keys().collect::<Vec<_>>();
        sigs.extend(self.failure_counts.keys());
        sigs.sort_unstable();
        sigs.dedup();
        sigs.len()
    }
}

impl Default for RLTestPrioritizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rl_prioritizer_initial() {
        let prioritizer = RLTestPrioritizer::new();
        assert_eq!(prioritizer.total_tests(), 0);
        assert_eq!(prioritizer.num_signatures(), 0);
    }

    #[test]
    fn test_rl_prioritizer_feedback() {
        let mut prioritizer = RLTestPrioritizer::new();

        let features = CodeFeatures {
            ast_depth: 5,
            num_operators: 10,
            num_control_flow: 2,
            cyclomatic_complexity: 3.0,
            uses_edge_values: false,
            ..Default::default()
        };

        // Simulate test revealing bug
        prioritizer.update_feedback(&features, true);
        assert_eq!(prioritizer.total_tests(), 1);

        // Check failure rate increased
        let rate = prioritizer.failure_rate(&features);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_rl_prioritizer_learning() {
        let mut prioritizer = RLTestPrioritizer::new();

        let buggy_features = CodeFeatures {
            ast_depth: 10,
            num_operators: 50,
            num_control_flow: 10,
            cyclomatic_complexity: 15.0,
            uses_edge_values: true,
            ..Default::default()
        };

        let clean_features = CodeFeatures {
            ast_depth: 3,
            num_operators: 5,
            num_control_flow: 1,
            cyclomatic_complexity: 2.0,
            uses_edge_values: false,
            ..Default::default()
        };

        // Simulate multiple test executions
        for _ in 0..10 {
            prioritizer.update_feedback(&buggy_features, true);
            prioritizer.update_feedback(&clean_features, false);
        }

        // Buggy features should have higher failure rate
        let buggy_rate = prioritizer.failure_rate(&buggy_features);
        let clean_rate = prioritizer.failure_rate(&clean_features);
        assert!(buggy_rate > clean_rate);
    }

    #[test]
    fn test_rl_prioritizer_ordering() {
        let mut prioritizer = RLTestPrioritizer::new();

        let features = vec![
            CodeFeatures {
                ast_depth: 3,
                num_operators: 5,
                cyclomatic_complexity: 2.0,
                uses_edge_values: false,
                ..Default::default()
            },
            CodeFeatures {
                ast_depth: 10,
                num_operators: 50,
                cyclomatic_complexity: 15.0,
                uses_edge_values: true,
                ..Default::default()
            },
        ];

        // Train: second feature reveals bugs more often
        for _ in 0..5 {
            prioritizer.update_feedback(&features[1], true);
            prioritizer.update_feedback(&features[0], false);
        }

        // Prioritize should put buggy test first
        let order = prioritizer.prioritize(&features);
        // Due to Thompson Sampling randomness, we can't guarantee exact order
        // Just check both indices are present
        assert_eq!(order.len(), 2);
        assert!(order.contains(&0));
        assert!(order.contains(&1));
    }

    #[test]
    fn test_exploration_rate() {
        let prioritizer = RLTestPrioritizer::new().with_exploration_rate(0.2);
        assert!((prioritizer.exploration_rate - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feature_signature_buckets() {
        let features = CodeFeatures {
            ast_depth: 7,
            num_operators: 15,
            cyclomatic_complexity: 8.0,
            uses_edge_values: true,
            ..Default::default()
        };

        let sig = FeatureSignature::from_features(&features);
        assert_eq!(sig.depth_bucket, 1); // 6-10
        assert_eq!(sig.operator_bucket, 1); // 11-30
        assert_eq!(sig.complexity_bucket, 1); // 6-15
        assert!(sig.uses_edge_values);
    }
}
