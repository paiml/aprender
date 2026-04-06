//! ML model training pipeline
//!
//! This module provides the ML training pipeline for:
//! - Bug prediction model (aprender)
//! - Code embedding model (entrenar + trueno)
//! - Transpilation suggestion model (entrenar)
//!
//! # Architecture
//!
//! From spec Section 5:
//! - **aprender**: Classical ML for fast inference (RandomForest, GradientBoosting)
//! - **entrenar**: LoRA fine-tuning for code models
//! - **trueno**: SIMD-accelerated tensor operations

mod active_learner;
mod aprender;
mod augmentation;
mod backend;
mod codex_pipeline;
mod commit_features;
mod defect_predictor;
mod entrenar;
mod evaluator;
mod experiment;
mod quality_gate;
mod rich_labeling;
mod rl_prioritizer;
mod trainer;
mod training;

pub use self::active_learner::{
    ActiveLearner, Cluster, ClusterStats, ClusteringResult, CodeEmbedder, CodeEmbedding,
    KMeansClustering,
};
pub use self::aprender::AprenderBugPredictor;
pub use self::augmentation::{AugmentationResult, BatchAugmenter, CodeEDA, CodeEDAConfig};
pub use self::backend::{Backend, BackendSelector, BatchConfig, OpComplexity, SelectionStats};
pub use self::codex_pipeline::{
    CodexPipeline, DataQualityMetrics, PipelineConfig, PipelineResult, PipelineStats,
    PreparedSample, StageResult,
};
pub use self::commit_features::{CommitFeatureExtractor, CommitFeatures, FeatureStats};
pub use self::defect_predictor::{
    CategoryWeights, DefectCategory, DefectPrediction, DefectPredictor, DefectPredictorStats,
    DefectSample,
};
pub use self::entrenar::{
    generate_entrenar_config, CodeTranslationExample, DistillTrainingConfig, DistillationConfig,
    DistillationResult, EntrenarExporter, ExportConfig, ExportFormat, ExportStats, PromptTemplate,
    StudentConfig,
};
pub use self::evaluator::{
    benchmark_inference, calculate_feature_importance, BenchmarkResult, ComparisonMetrics,
    ConfusionMatrix, FeatureImportance, ModelComparison, RocCurve, RocPoint,
};
pub use self::experiment::{
    AppleChip, ComputeDevice, CostMetrics, CpuArchitecture, EnergyMetrics, ExperimentMetrics,
    GenerationExperiment, GpuVendor, TpuVersion,
};
pub use self::quality_gate::{
    CodeQualityFeatures, FeatureExtractor as QualityFeatureExtractor, QualityGate,
    QualityGateStats, QualityVerdict,
};
pub use self::rich_labeling::{
    AstDiff, ErrorCategory, ExecutionMetrics, LabelExtractor, RichLabel, SoftLabels,
    SoftLabelsBuilder,
};
pub use self::rl_prioritizer::RLTestPrioritizer;
pub use self::trainer::{
    ModelMetrics, ModelTrainer as LegacyModelTrainer, SerializedModel,
    TrainingConfig as LegacyTrainingConfig, TrainingResult,
};
pub use self::training::{
    train_test_split, verdict_to_label, CrossValidationResults, ModelTrainer, TrainedModel,
    TrainingConfig, TrainingError, TrainingExample, TrainingMetrics,
};
use crate::data::CodeFeatures;

/// Heuristic bug prediction model (legacy)
///
/// Simple heuristic-based bug predictor. For production use, prefer
/// [`AprenderBugPredictor`] which uses trained RandomForest models.
///
/// This implementation uses hand-crafted rules and doesn't require training.
#[derive(Debug, Default)]
pub struct BugPredictor {
    /// Model weights (not used in heuristic version)
    _weights: Vec<f32>,
}

impl BugPredictor {
    /// Create a new bug predictor
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Predict probability of a bug using heuristics
    ///
    /// Returns probability in range [0, 1]
    ///
    /// For ML-based prediction, use [`AprenderBugPredictor::train`] instead.
    #[must_use]
    pub fn predict(&self, features: &CodeFeatures) -> f32 {
        let mut score = 0.0_f32;

        score += features.ast_depth as f32 * 0.05;
        score += features.num_operators as f32 * 0.02;

        if features.uses_edge_values {
            score += 0.3;
        }

        score += features.cyclomatic_complexity * 0.01;

        score.clamp(0.0, 1.0)
    }

    /// Load a trained model from file
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails
    ///
    /// Note: Model persistence not implemented for heuristic predictor.
    /// Use [`AprenderBugPredictor`] for trained model persistence.
    pub fn load(_path: &str) -> crate::Result<Self> {
        Ok(Self::default())
    }
}

/// Heuristic test case prioritizer (legacy)
///
/// Simple prioritizer using heuristic bug prediction. For production use with
/// reinforcement learning, prefer [`RLTestPrioritizer`] which implements
/// Thompson Sampling (Spieker et al. 2017).
#[derive(Debug, Default)]
pub struct TestPrioritizer {
    /// Historical failure rates by feature (not used in heuristic version)
    #[allow(dead_code)]
    feature_failure_rates: Vec<(String, f32)>,
}

impl TestPrioritizer {
    /// Create a new test prioritizer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Prioritize test cases by predicted failure probability
    ///
    /// Uses heuristic [`BugPredictor`]. For RL-based prioritization,
    /// use [`RLTestPrioritizer`] instead.
    ///
    /// Returns indices sorted by priority (highest first)
    pub fn prioritize(&self, features: &[CodeFeatures]) -> Vec<usize> {
        let predictor = BugPredictor::new();

        let mut scored: Vec<(usize, f32)> = features
            .iter()
            .enumerate()
            .map(|(i, f)| (i, predictor.predict(f)))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Update with feedback from test results
    ///
    /// No-op for heuristic prioritizer. Use [`RLTestPrioritizer::update_feedback`]
    /// for learning-based prioritization.
    pub fn update_feedback(&mut self, _feature: &str, _failed: bool) {}
}

/// Feature extractor for code analysis
#[derive(Debug, Default)]
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Create a new feature extractor
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Extract features from source code using heuristics
    ///
    /// Uses simple pattern matching. For AST-based feature extraction,
    /// parse code with language-specific grammars first.
    #[must_use]
    pub fn extract(&self, code: &str) -> CodeFeatures {
        let lines: Vec<&str> = code.lines().collect();
        let operators = code
            .chars()
            .filter(|c| ['+', '-', '*', '/', '%', '<', '>', '='].contains(c))
            .count();

        CodeFeatures {
            ast_depth: lines.len().min(10) as u32,
            num_operators: operators as u32,
            num_control_flow: count_keywords(code, &["if", "for", "while", "return"]),
            cyclomatic_complexity: 1.0
                + count_keywords(code, &["if", "elif", "for", "while"]) as f32,
            num_type_coercions: 0,
            uses_edge_values: code.contains(" 0")
                || code.contains("-1")
                || code.contains("[]")
                || code.contains("None"),
        }
    }
}

fn count_keywords(code: &str, keywords: &[&str]) -> u32 {
    keywords
        .iter()
        .map(|kw| code.matches(kw).count() as u32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bug_predictor_basic() {
        let predictor = BugPredictor::new();
        let features = CodeFeatures::default();
        let prob = predictor.predict(&features);
        assert!((0.0..=1.0).contains(&prob));
    }

    #[test]
    fn test_bug_predictor_edge_values() {
        let predictor = BugPredictor::new();
        let features = CodeFeatures {
            uses_edge_values: true,
            ..Default::default()
        };
        let prob = predictor.predict(&features);
        assert!(prob >= 0.3); // Edge values add 0.3 to score
    }

    #[test]
    fn test_prioritizer() {
        let prioritizer = TestPrioritizer::new();
        let features = vec![
            CodeFeatures::default(),
            CodeFeatures {
                uses_edge_values: true,
                ..Default::default()
            },
            CodeFeatures {
                ast_depth: 10,
                ..Default::default()
            },
        ];

        let order = prioritizer.prioritize(&features);
        // ast_depth=10 gives score 0.5, edge_values gives 0.3
        // So index 2 (ast_depth=10) should be first, then index 1 (edge_values)
        assert_eq!(order[0], 2);
        assert_eq!(order[1], 1);
    }

    #[test]
    fn test_feature_extractor() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract("x = 0\nif x < 1:\n    y = -1");

        assert!(features.num_operators > 0);
        assert!(features.num_control_flow > 0);
        assert!(features.uses_edge_values);
    }

    #[test]
    fn test_bug_predictor_load() {
        let predictor = BugPredictor::load("/nonexistent/path");
        assert!(predictor.is_ok());
    }

    #[test]
    fn test_prioritizer_update_feedback() {
        let mut prioritizer = TestPrioritizer::new();
        // Just verify it doesn't panic
        prioritizer.update_feedback("test_feature", true);
        prioritizer.update_feedback("test_feature", false);
    }

    #[test]
    fn test_bug_predictor_debug() {
        let predictor = BugPredictor::new();
        let debug = format!("{:?}", predictor);
        assert!(debug.contains("BugPredictor"));
    }

    #[test]
    fn test_prioritizer_debug() {
        let prioritizer = TestPrioritizer::new();
        let debug = format!("{:?}", prioritizer);
        assert!(debug.contains("TestPrioritizer"));
    }

    #[test]
    fn test_feature_extractor_debug() {
        let extractor = FeatureExtractor::new();
        let debug = format!("{:?}", extractor);
        assert!(debug.contains("FeatureExtractor"));
    }

    #[test]
    fn test_bug_predictor_high_complexity() {
        let predictor = BugPredictor::new();
        let features = CodeFeatures {
            ast_depth: 20,
            num_operators: 50,
            cyclomatic_complexity: 50.0,
            uses_edge_values: true,
            ..Default::default()
        };
        let prob = predictor.predict(&features);
        // Should be clamped to 1.0
        assert!((prob - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_extractor_empty_list() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract("x = []");
        assert!(features.uses_edge_values);
    }

    #[test]
    fn test_feature_extractor_none() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract("x = None");
        assert!(features.uses_edge_values);
    }
}
