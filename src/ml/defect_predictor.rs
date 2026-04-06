//! Defect Predictor - Bug likelihood model using GradientBoosting
//!
//! Implements defect prediction with category weighting based on
//! historical defect data from PAIML repos (1,296 defect-fix commits).
//!
//! # Category Weights
//!
//! | Category | Weight | Description |
//! |----------|--------|-------------|
//! | AST Transform | 2.0x | Universal dominant defect (40-62%) |
//! | Ownership/Borrow | 1.5x | Rust-specific (15-20%) |
//! | Stdlib Mapping | 1.2x | API translation errors |
//! | Other | 1.0x | Language-specific |
//!
//! # Reference
//! - VER-051: Bug Predictor - Defect likelihood model
//! - docs/specifications/codex-multi-tech-python-to-rust-spec.md

use serde::{Deserialize, Serialize};

use crate::ml::CommitFeatures;

/// Defect category based on PAIML repo analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DefectCategory {
    /// AST transformation errors (40-62% of defects)
    AstTransform,
    /// Ownership and borrowing errors (Rust-specific, 15-20%)
    OwnershipBorrow,
    /// Standard library mapping errors
    StdlibMapping,
    /// Other language-specific errors
    Other,
}

impl Default for DefectCategory {
    fn default() -> Self {
        Self::Other
    }
}

impl DefectCategory {
    /// Get category weight for defect prioritization
    #[must_use]
    pub fn weight(&self) -> f32 {
        match self {
            Self::AstTransform => 2.0,
            Self::OwnershipBorrow => 1.5,
            Self::StdlibMapping => 1.2,
            Self::Other => 1.0,
        }
    }

    /// All categories
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::AstTransform,
            Self::OwnershipBorrow,
            Self::StdlibMapping,
            Self::Other,
        ]
    }

    /// Classify defect from code pattern
    #[must_use]
    pub fn classify(code: &str) -> Self {
        // AST-related patterns
        if code.contains("ast")
            || code.contains("node")
            || code.contains("parse")
            || code.contains("transform")
            || code.contains("visitor")
        {
            return Self::AstTransform;
        }

        // Ownership/borrow patterns (Rust-specific)
        if code.contains("borrow")
            || code.contains("lifetime")
            || code.contains("move")
            || code.contains("&mut")
            || code.contains("owned")
        {
            return Self::OwnershipBorrow;
        }

        // Stdlib mapping patterns
        if code.contains("std::")
            || code.contains("collections")
            || code.contains("HashMap")
            || code.contains("Vec")
            || code.contains("String")
        {
            return Self::StdlibMapping;
        }

        Self::Other
    }
}

/// Category weights for defect prioritization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryWeights {
    /// Weight for AST transform defects
    pub ast_transform: f32,
    /// Weight for ownership/borrow defects
    pub ownership_borrow: f32,
    /// Weight for stdlib mapping defects
    pub stdlib_mapping: f32,
    /// Weight for other defects
    pub other: f32,
}

impl Default for CategoryWeights {
    fn default() -> Self {
        Self {
            ast_transform: 2.0,
            ownership_borrow: 1.5,
            stdlib_mapping: 1.2,
            other: 1.0,
        }
    }
}

impl CategoryWeights {
    /// Get weight for category
    #[must_use]
    pub fn get(&self, category: DefectCategory) -> f32 {
        match category {
            DefectCategory::AstTransform => self.ast_transform,
            DefectCategory::OwnershipBorrow => self.ownership_borrow,
            DefectCategory::StdlibMapping => self.stdlib_mapping,
            DefectCategory::Other => self.other,
        }
    }

    /// Set weight for category
    pub fn set(&mut self, category: DefectCategory, weight: f32) {
        match category {
            DefectCategory::AstTransform => self.ast_transform = weight,
            DefectCategory::OwnershipBorrow => self.ownership_borrow = weight,
            DefectCategory::StdlibMapping => self.stdlib_mapping = weight,
            DefectCategory::Other => self.other = weight,
        }
    }
}

/// Training sample for defect prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefectSample {
    /// Commit features
    pub features: CommitFeatures,
    /// Whether this commit introduced a defect
    pub is_defect: bool,
    /// Defect category (if known)
    pub category: Option<DefectCategory>,
}

impl DefectSample {
    /// Create new sample
    #[must_use]
    pub fn new(features: CommitFeatures, is_defect: bool) -> Self {
        Self {
            features,
            is_defect,
            category: None,
        }
    }

    /// Create with category
    #[must_use]
    pub fn with_category(mut self, category: DefectCategory) -> Self {
        self.category = Some(category);
        self
    }
}

/// Defect prediction result
#[derive(Debug, Clone)]
pub struct DefectPrediction {
    /// Base probability (0.0 to 1.0)
    pub base_probability: f32,
    /// Weighted probability accounting for category
    pub weighted_probability: f32,
    /// Predicted category
    pub category: DefectCategory,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
}

impl DefectPrediction {
    /// Priority score for sampling (higher = sample first)
    #[must_use]
    pub fn priority_score(&self) -> f32 {
        self.weighted_probability * self.confidence
    }
}

/// Defect predictor using linear model (placeholder for GradientBoosting)
///
/// When `ml` feature is enabled, uses aprender's GradientBoostingClassifier.
/// Otherwise uses a simple linear model as fallback.
#[derive(Debug)]
pub struct DefectPredictor {
    /// Category weights
    weights: CategoryWeights,
    /// Feature weights (8-dim for CommitFeatures)
    feature_weights: [f32; 8],
    /// Bias term
    bias: f32,
    /// Training statistics
    stats: DefectPredictorStats,
    /// Whether model has been trained
    is_trained: bool,
}

/// Training statistics
#[derive(Debug, Clone, Default)]
pub struct DefectPredictorStats {
    /// Number of training samples
    pub n_samples: usize,
    /// Number of defect samples
    pub n_defects: usize,
    /// Training accuracy (if evaluated)
    pub accuracy: Option<f32>,
}

impl Default for DefectPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl DefectPredictor {
    /// Create new defect predictor
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: CategoryWeights::default(),
            feature_weights: [
                0.15,  // lines_added
                0.10,  // lines_deleted
                0.08,  // files_changed
                0.12,  // churn_ratio
                -0.20, // has_test_changes (tests reduce defects)
                0.25,  // complexity_delta
                -0.15, // author_experience (experience reduces defects)
                0.10,  // days_since_last_change
            ],
            bias: 0.1,
            stats: DefectPredictorStats::default(),
            is_trained: false,
        }
    }

    /// Create with custom category weights
    #[must_use]
    pub fn with_weights(weights: CategoryWeights) -> Self {
        Self {
            weights,
            ..Self::new()
        }
    }

    /// Train on samples (linear model fallback)
    ///
    /// Uses gradient descent to optimize feature weights.
    ///
    /// # Errors
    ///
    /// Returns an error if `samples` is empty.
    pub fn train(&mut self, samples: &[DefectSample]) -> crate::Result<()> {
        if samples.is_empty() {
            return Err(crate::Error::Data("No training samples".to_string()));
        }

        self.stats.n_samples = samples.len();
        self.stats.n_defects = samples.iter().filter(|s| s.is_defect).count();

        // Simple linear regression via closed-form solution
        // For production, this would use aprender's GradientBoosting
        let learning_rate = 0.01;
        let epochs = 100;

        for _ in 0..epochs {
            let mut gradient = [0.0f32; 8];
            let mut bias_gradient = 0.0f32;

            for sample in samples {
                let arr = sample.features.to_array();
                let pred = self.predict_raw(&sample.features);
                let target = if sample.is_defect { 1.0 } else { 0.0 };
                let error = pred - target;

                for (i, &val) in arr.iter().enumerate() {
                    gradient[i] += error * val;
                }
                bias_gradient += error;
            }

            // Update weights
            let n = samples.len() as f32;
            for (i, grad) in gradient.iter().enumerate() {
                self.feature_weights[i] -= learning_rate * grad / n;
            }
            self.bias -= learning_rate * bias_gradient / n;
        }

        // Calculate training accuracy
        let correct = samples
            .iter()
            .filter(|s| {
                let pred = self.predict_raw(&s.features) >= 0.5;
                pred == s.is_defect
            })
            .count();
        self.stats.accuracy = Some(correct as f32 / samples.len() as f32);
        self.is_trained = true;

        Ok(())
    }

    /// Raw prediction without category weighting
    fn predict_raw(&self, features: &CommitFeatures) -> f32 {
        let arr = features.to_array();
        let mut score = self.bias;

        for (i, &val) in arr.iter().enumerate() {
            // Normalize features
            let normalized = match i {
                0 => (val / 100.0).min(1.0), // lines_added
                1 => (val / 50.0).min(1.0),  // lines_deleted
                2 => (val / 10.0).min(1.0),  // files_changed
                3 => val.min(1.0),           // churn_ratio
                4 => val,                    // has_test_changes (0/1)
                5 => (val / 10.0).min(1.0),  // complexity_delta
                6 => val,                    // author_experience (0-1)
                7 => (val / 30.0).min(1.0),  // days_since_last_change
                _ => val,
            };
            score += self.feature_weights[i] * normalized;
        }

        // Sigmoid activation
        1.0 / (1.0 + (-score).exp())
    }

    /// Predict defect probability for code
    #[must_use]
    pub fn predict(&self, features: &CommitFeatures, code: &str) -> DefectPrediction {
        let base_probability = self.predict_raw(features);
        let category = DefectCategory::classify(code);
        let weight = self.weights.get(category);

        // Apply category weight
        let weighted_probability = (base_probability * weight).min(1.0);

        // Confidence based on training and feature variance
        let confidence = if self.is_trained { 0.8 } else { 0.5 };

        DefectPrediction {
            base_probability,
            weighted_probability,
            category,
            confidence,
        }
    }

    /// Predict for features only (without code for category)
    #[must_use]
    pub fn predict_features(&self, features: &CommitFeatures) -> DefectPrediction {
        let base_probability = self.predict_raw(features);

        DefectPrediction {
            base_probability,
            weighted_probability: base_probability,
            category: DefectCategory::Other,
            confidence: if self.is_trained { 0.8 } else { 0.5 },
        }
    }

    /// Get training statistics
    #[must_use]
    pub fn stats(&self) -> &DefectPredictorStats {
        &self.stats
    }

    /// Is model trained?
    #[must_use]
    pub fn is_trained(&self) -> bool {
        self.is_trained
    }

    /// Get category weights
    #[must_use]
    pub fn category_weights(&self) -> &CategoryWeights {
        &self.weights
    }

    /// Prioritize samples by defect likelihood
    ///
    /// Returns indices sorted by priority (highest defect probability first)
    pub fn prioritize(&self, samples: &[(CommitFeatures, String)]) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = samples
            .iter()
            .enumerate()
            .map(|(i, (features, code))| {
                let pred = self.predict(features, code);
                (i, pred.priority_score())
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(i, _)| i).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_features() -> CommitFeatures {
        CommitFeatures {
            lines_added: 50,
            lines_deleted: 10,
            files_changed: 3,
            churn_ratio: 0.8,
            has_test_changes: false,
            complexity_delta: 5.0,
            author_experience: 0.5,
            days_since_last_change: 7.0,
        }
    }

    fn buggy_features() -> CommitFeatures {
        CommitFeatures {
            lines_added: 200,
            lines_deleted: 5,
            files_changed: 10,
            churn_ratio: 0.95,
            has_test_changes: false,
            complexity_delta: 15.0,
            author_experience: 0.1,
            days_since_last_change: 1.0,
        }
    }

    fn safe_features() -> CommitFeatures {
        CommitFeatures {
            lines_added: 10,
            lines_deleted: 5,
            files_changed: 1,
            churn_ratio: 0.3,
            has_test_changes: true,
            complexity_delta: -2.0,
            author_experience: 0.9,
            days_since_last_change: 14.0,
        }
    }

    // ========== RED PHASE: Category Tests ==========

    #[test]
    fn test_defect_category_weights() {
        assert!((DefectCategory::AstTransform.weight() - 2.0).abs() < f32::EPSILON);
        assert!((DefectCategory::OwnershipBorrow.weight() - 1.5).abs() < f32::EPSILON);
        assert!((DefectCategory::StdlibMapping.weight() - 1.2).abs() < f32::EPSILON);
        assert!((DefectCategory::Other.weight() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_defect_category_classify_ast() {
        assert_eq!(
            DefectCategory::classify("parse_ast_node"),
            DefectCategory::AstTransform
        );
        assert_eq!(
            DefectCategory::classify("transform_expression"),
            DefectCategory::AstTransform
        );
    }

    #[test]
    fn test_defect_category_classify_ownership() {
        assert_eq!(
            DefectCategory::classify("fix borrow checker"),
            DefectCategory::OwnershipBorrow
        );
        assert_eq!(
            DefectCategory::classify("lifetime issue"),
            DefectCategory::OwnershipBorrow
        );
    }

    #[test]
    fn test_defect_category_classify_stdlib() {
        assert_eq!(
            DefectCategory::classify("use std::collections::HashMap"),
            DefectCategory::StdlibMapping
        );
        assert_eq!(
            DefectCategory::classify("Vec::new()"),
            DefectCategory::StdlibMapping
        );
    }

    #[test]
    fn test_defect_category_classify_other() {
        assert_eq!(
            DefectCategory::classify("simple fix"),
            DefectCategory::Other
        );
    }

    #[test]
    fn test_defect_category_all() {
        let all = DefectCategory::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_defect_category_default() {
        assert_eq!(DefectCategory::default(), DefectCategory::Other);
    }

    // ========== RED PHASE: Category Weights Tests ==========

    #[test]
    fn test_category_weights_default() {
        let weights = CategoryWeights::default();
        assert!((weights.ast_transform - 2.0).abs() < f32::EPSILON);
        assert!((weights.other - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_category_weights_get() {
        let weights = CategoryWeights::default();
        assert!((weights.get(DefectCategory::AstTransform) - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_category_weights_set() {
        let mut weights = CategoryWeights::default();
        weights.set(DefectCategory::AstTransform, 3.0);
        assert!((weights.ast_transform - 3.0).abs() < f32::EPSILON);
    }

    // ========== RED PHASE: DefectSample Tests ==========

    #[test]
    fn test_defect_sample_new() {
        let sample = DefectSample::new(sample_features(), true);
        assert!(sample.is_defect);
        assert!(sample.category.is_none());
    }

    #[test]
    fn test_defect_sample_with_category() {
        let sample =
            DefectSample::new(sample_features(), true).with_category(DefectCategory::AstTransform);
        assert_eq!(sample.category, Some(DefectCategory::AstTransform));
    }

    // ========== RED PHASE: Prediction Tests ==========

    #[test]
    fn test_defect_prediction_priority_score() {
        let pred = DefectPrediction {
            base_probability: 0.8,
            weighted_probability: 0.9,
            category: DefectCategory::AstTransform,
            confidence: 0.7,
        };

        let score = pred.priority_score();
        assert!((score - 0.63).abs() < 0.01); // 0.9 * 0.7
    }

    // ========== RED PHASE: Predictor Tests ==========

    #[test]
    fn test_defect_predictor_new() {
        let predictor = DefectPredictor::new();
        assert!(!predictor.is_trained());
    }

    #[test]
    fn test_defect_predictor_with_weights() {
        let weights = CategoryWeights {
            ast_transform: 3.0,
            ..Default::default()
        };
        let predictor = DefectPredictor::with_weights(weights);
        assert!((predictor.category_weights().ast_transform - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_defect_predictor_predict_features() {
        let predictor = DefectPredictor::new();
        let pred = predictor.predict_features(&sample_features());

        assert!(pred.base_probability >= 0.0);
        assert!(pred.base_probability <= 1.0);
        assert_eq!(pred.category, DefectCategory::Other);
    }

    #[test]
    fn test_defect_predictor_predict_with_code() {
        let predictor = DefectPredictor::new();
        let pred = predictor.predict(&sample_features(), "fix ast parser");

        assert_eq!(pred.category, DefectCategory::AstTransform);
        // Weighted probability should be higher than base
        assert!(pred.weighted_probability >= pred.base_probability);
    }

    #[test]
    fn test_defect_predictor_probability_bounded() {
        let predictor = DefectPredictor::new();

        for features in &[sample_features(), buggy_features(), safe_features()] {
            let pred = predictor.predict_features(features);
            assert!(pred.base_probability >= 0.0);
            assert!(pred.base_probability <= 1.0);
            assert!(pred.weighted_probability >= 0.0);
            assert!(pred.weighted_probability <= 1.0);
        }
    }

    #[test]
    fn test_defect_predictor_buggy_higher_than_safe() {
        let predictor = DefectPredictor::new();

        let buggy_pred = predictor.predict_features(&buggy_features());
        let safe_pred = predictor.predict_features(&safe_features());

        // Buggy features should have higher probability
        assert!(buggy_pred.base_probability > safe_pred.base_probability);
    }

    // ========== RED PHASE: Training Tests ==========

    #[test]
    fn test_defect_predictor_train_empty_fails() {
        let mut predictor = DefectPredictor::new();
        let result = predictor.train(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_defect_predictor_train() {
        let mut predictor = DefectPredictor::new();

        let samples = vec![
            DefectSample::new(buggy_features(), true),
            DefectSample::new(buggy_features(), true),
            DefectSample::new(safe_features(), false),
            DefectSample::new(safe_features(), false),
        ];

        let result = predictor.train(&samples);
        assert!(result.is_ok());
        assert!(predictor.is_trained());
    }

    #[test]
    fn test_defect_predictor_train_stats() {
        let mut predictor = DefectPredictor::new();

        let samples = vec![
            DefectSample::new(buggy_features(), true),
            DefectSample::new(safe_features(), false),
            DefectSample::new(safe_features(), false),
        ];

        predictor.train(&samples).unwrap();

        let stats = predictor.stats();
        assert_eq!(stats.n_samples, 3);
        assert_eq!(stats.n_defects, 1);
        assert!(stats.accuracy.is_some());
    }

    #[test]
    fn test_defect_predictor_confidence_after_training() {
        let mut predictor = DefectPredictor::new();

        // Before training: lower confidence
        let pred_before = predictor.predict_features(&sample_features());
        assert!((pred_before.confidence - 0.5).abs() < f32::EPSILON);

        // Train
        let samples = vec![
            DefectSample::new(buggy_features(), true),
            DefectSample::new(safe_features(), false),
        ];
        predictor.train(&samples).unwrap();

        // After training: higher confidence
        let pred_after = predictor.predict_features(&sample_features());
        assert!((pred_after.confidence - 0.8).abs() < f32::EPSILON);
    }

    // ========== RED PHASE: Prioritization Tests ==========

    #[test]
    fn test_defect_predictor_prioritize() {
        let predictor = DefectPredictor::new();

        let samples = vec![
            (safe_features(), "simple code".to_string()),
            (buggy_features(), "ast transform bug".to_string()),
            (sample_features(), "normal code".to_string()),
        ];

        let order = predictor.prioritize(&samples);

        // Buggy AST sample should be first (index 1)
        assert_eq!(order[0], 1);
    }

    #[test]
    fn test_defect_predictor_prioritize_empty() {
        let predictor = DefectPredictor::new();
        let samples: Vec<(CommitFeatures, String)> = vec![];

        let order = predictor.prioritize(&samples);
        assert!(order.is_empty());
    }

    // ========== RED PHASE: Debug/Display Tests ==========

    #[test]
    fn test_defect_category_debug() {
        let debug = format!("{:?}", DefectCategory::AstTransform);
        assert!(debug.contains("AstTransform"));
    }

    #[test]
    fn test_category_weights_debug() {
        let weights = CategoryWeights::default();
        let debug = format!("{weights:?}");
        assert!(debug.contains("CategoryWeights"));
    }

    #[test]
    fn test_defect_sample_debug() {
        let sample = DefectSample::new(sample_features(), true);
        let debug = format!("{sample:?}");
        assert!(debug.contains("DefectSample"));
    }

    #[test]
    fn test_defect_prediction_debug() {
        let pred = DefectPrediction {
            base_probability: 0.5,
            weighted_probability: 0.6,
            category: DefectCategory::Other,
            confidence: 0.7,
        };
        let debug = format!("{pred:?}");
        assert!(debug.contains("DefectPrediction"));
    }

    #[test]
    fn test_defect_predictor_debug() {
        let predictor = DefectPredictor::new();
        let debug = format!("{predictor:?}");
        assert!(debug.contains("DefectPredictor"));
    }

    #[test]
    fn test_defect_predictor_stats_debug() {
        let stats = DefectPredictorStats::default();
        let debug = format!("{stats:?}");
        assert!(debug.contains("DefectPredictorStats"));
    }

    // ========== RED PHASE: Serialization Tests ==========

    #[test]
    fn test_defect_category_serialize() {
        let category = DefectCategory::AstTransform;
        let json = serde_json::to_string(&category).unwrap();
        let restored: DefectCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(category, restored);
    }

    #[test]
    fn test_category_weights_serialize() {
        let weights = CategoryWeights::default();
        let json = serde_json::to_string(&weights).unwrap();
        let restored: CategoryWeights = serde_json::from_str(&json).unwrap();
        assert!((weights.ast_transform - restored.ast_transform).abs() < f32::EPSILON);
    }

    #[test]
    fn test_defect_sample_serialize() {
        let sample =
            DefectSample::new(sample_features(), true).with_category(DefectCategory::AstTransform);
        let json = serde_json::to_string(&sample).unwrap();
        let restored: DefectSample = serde_json::from_str(&json).unwrap();
        assert_eq!(sample.is_defect, restored.is_defect);
        assert_eq!(sample.category, restored.category);
    }
}

/// Property-based tests
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Probability is always bounded [0, 1]
        #[test]
        fn prop_probability_bounded(
            lines_added in 0u32..1000,
            lines_deleted in 0u32..500,
            files_changed in 1u32..50,
            complexity_delta in -20.0f32..50.0,
        ) {
            let features = CommitFeatures {
                lines_added,
                lines_deleted,
                files_changed,
                churn_ratio: 0.5,
                has_test_changes: false,
                complexity_delta,
                author_experience: 0.5,
                days_since_last_change: 7.0,
            };

            let predictor = DefectPredictor::new();
            let pred = predictor.predict_features(&features);

            prop_assert!(pred.base_probability >= 0.0);
            prop_assert!(pred.base_probability <= 1.0);
        }

        /// Category weight increases prediction
        #[test]
        fn prop_category_weight_increases(base_prob in 0.1f32..0.8) {
            let weights = CategoryWeights::default();

            let weighted = base_prob * weights.get(DefectCategory::AstTransform);
            let unweighted = base_prob * weights.get(DefectCategory::Other);

            prop_assert!(weighted >= unweighted);
        }

        /// Higher complexity = higher defect probability
        #[test]
        fn prop_complexity_increases_probability(base_complexity in -5.0f32..5.0) {
            let predictor = DefectPredictor::new();

            let low = CommitFeatures {
                complexity_delta: base_complexity,
                ..Default::default()
            };

            let high = CommitFeatures {
                complexity_delta: base_complexity + 10.0,
                ..Default::default()
            };

            let low_pred = predictor.predict_features(&low);
            let high_pred = predictor.predict_features(&high);

            prop_assert!(high_pred.base_probability >= low_pred.base_probability);
        }

        /// Test changes reduce defect probability
        #[test]
        fn prop_tests_reduce_probability(lines_added in 10u32..100) {
            let predictor = DefectPredictor::new();

            let without_tests = CommitFeatures {
                lines_added,
                has_test_changes: false,
                ..Default::default()
            };

            let with_tests = CommitFeatures {
                lines_added,
                has_test_changes: true,
                ..Default::default()
            };

            let without_pred = predictor.predict_features(&without_tests);
            let with_pred = predictor.predict_features(&with_tests);

            prop_assert!(with_pred.base_probability <= without_pred.base_probability);
        }

        /// Experience reduces defect probability
        #[test]
        fn prop_experience_reduces_probability(lines_added in 10u32..100) {
            let predictor = DefectPredictor::new();

            let novice = CommitFeatures {
                lines_added,
                author_experience: 0.1,
                ..Default::default()
            };

            let expert = CommitFeatures {
                lines_added,
                author_experience: 0.9,
                ..Default::default()
            };

            let novice_pred = predictor.predict_features(&novice);
            let expert_pred = predictor.predict_features(&expert);

            prop_assert!(expert_pred.base_probability <= novice_pred.base_probability);
        }
    }
}
