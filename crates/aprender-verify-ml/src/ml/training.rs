//! Bug prediction model training pipeline
//!
//! Trains ML models on verification data. See VERIFICAR-051.

use crate::transpiler::{CodeFeatures, TranspilerVerdict};
use std::path::Path;

/// Training example: features + label (bug or not)
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Extracted code features
    pub features: CodeFeatures,
    /// True if this example exposed a bug (non-Pass verdict)
    pub is_bug: bool,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Train/test split ratio (0.0 to 1.0)
    pub train_ratio: f64,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Minimum examples required for training
    pub min_examples: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.8,
            cv_folds: 5,
            seed: 42,
            min_examples: 100,
        }
    }
}

/// Training metrics
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Accuracy (correct / total)
    pub accuracy: f64,
    /// Precision (TP / (TP + FP))
    pub precision: f64,
    /// Recall (TP / (TP + FN))
    pub recall: f64,
    /// F1 score (harmonic mean of precision and recall)
    pub f1_score: f64,
    /// Area under ROC curve
    pub auc_roc: f64,
    /// Number of training examples
    pub train_size: usize,
    /// Number of test examples
    pub test_size: usize,
}

impl TrainingMetrics {
    /// Calculate F1 from precision and recall
    #[must_use]
    pub fn calculate_f1(precision: f64, recall: f64) -> f64 {
        if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        }
    }
}

/// Cross-validation results
#[derive(Debug, Clone, Default)]
pub struct CrossValidationResults {
    /// Metrics for each fold
    pub fold_metrics: Vec<TrainingMetrics>,
    /// Mean accuracy across folds
    pub mean_accuracy: f64,
    /// Std deviation of accuracy
    pub std_accuracy: f64,
    /// Mean F1 across folds
    pub mean_f1: f64,
}

impl CrossValidationResults {
    /// Calculate summary statistics from fold metrics
    #[must_use]
    pub fn summarize(fold_metrics: Vec<TrainingMetrics>) -> Self {
        if fold_metrics.is_empty() {
            return Self::default();
        }

        let n = fold_metrics.len() as f64;
        let mean_accuracy = fold_metrics.iter().map(|m| m.accuracy).sum::<f64>() / n;
        let mean_f1 = fold_metrics.iter().map(|m| m.f1_score).sum::<f64>() / n;

        let variance = fold_metrics
            .iter()
            .map(|m| (m.accuracy - mean_accuracy).powi(2))
            .sum::<f64>()
            / n;
        let std_accuracy = variance.sqrt();

        Self {
            fold_metrics,
            mean_accuracy,
            std_accuracy,
            mean_f1,
        }
    }
}

/// Trained model that can be saved/loaded
pub trait TrainedModel: Send + Sync {
    /// Predict bug probability for features
    fn predict(&self, features: &CodeFeatures) -> f64;

    /// Save model to file
    ///
    /// # Errors
    ///
    /// Returns IO error if save fails
    fn save(&self, path: &Path) -> std::io::Result<()>;

    /// Model version/metadata
    fn metadata(&self) -> ModelMetadata;
}

/// Model metadata for serialization
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model type name
    pub model_type: String,
    /// Training timestamp
    pub trained_at: String,
    /// Number of training examples
    pub train_examples: usize,
    /// Training metrics
    pub metrics: TrainingMetrics,
}

/// Model trainer trait
pub trait ModelTrainer {
    /// Train model on examples
    ///
    /// # Errors
    ///
    /// Returns error if training fails or insufficient data
    fn train(
        &self,
        examples: &[TrainingExample],
        config: &TrainingConfig,
    ) -> Result<Box<dyn TrainedModel>, TrainingError>;

    /// Run cross-validation
    ///
    /// # Errors
    ///
    /// Returns error if cross-validation fails
    fn cross_validate(
        &self,
        examples: &[TrainingExample],
        config: &TrainingConfig,
    ) -> Result<CrossValidationResults, TrainingError>;
}

/// Training errors
#[derive(Debug, Clone)]
pub enum TrainingError {
    /// Not enough training examples
    InsufficientData {
        /// Minimum required examples
        required: usize,
        /// Actually provided examples
        provided: usize,
    },
    /// Invalid configuration
    InvalidConfig(String),
    /// Model training failed
    TrainingFailed(String),
    /// IO error during save/load
    IoError(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientData { required, provided } => {
                write!(f, "Insufficient data: need {required}, got {provided}")
            }
            Self::InvalidConfig(msg) => write!(f, "Invalid config: {msg}"),
            Self::TrainingFailed(msg) => write!(f, "Training failed: {msg}"),
            Self::IoError(msg) => write!(f, "IO error: {msg}"),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Convert verdict to bug label
#[must_use]
pub fn verdict_to_label(verdict: &TranspilerVerdict) -> bool {
    !matches!(verdict, TranspilerVerdict::Pass)
}

/// Split examples into train/test sets
#[must_use]
pub fn train_test_split(
    examples: &[TrainingExample],
    train_ratio: f64,
    seed: u64,
) -> (Vec<TrainingExample>, Vec<TrainingExample>) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut train = Vec::new();
    let mut test = Vec::new();

    for (i, example) in examples.iter().enumerate() {
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let hash = hasher.finish();
        #[allow(clippy::cast_sign_loss)]
        let threshold = (train_ratio * u64::MAX as f64) as u64;

        if hash < threshold {
            train.push(example.clone());
        } else {
            test.push(example.clone());
        }
    }

    (train, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_examples(n: usize) -> Vec<TrainingExample> {
        (0..n)
            .map(|i| TrainingExample {
                features: CodeFeatures {
                    ast_depth: i % 5,
                    cyclomatic_complexity: i % 10,
                    ..Default::default()
                },
                is_bug: i % 3 == 0,
            })
            .collect()
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.train_ratio, 0.8);
        assert_eq!(config.cv_folds, 5);
        assert_eq!(config.min_examples, 100);
    }

    #[test]
    fn test_training_metrics_f1() {
        assert_eq!(TrainingMetrics::calculate_f1(0.8, 0.6), 0.6857142857142857);
        assert_eq!(TrainingMetrics::calculate_f1(0.0, 0.0), 0.0);
        assert_eq!(TrainingMetrics::calculate_f1(1.0, 1.0), 1.0);
    }

    #[test]
    fn test_cross_validation_summarize() {
        let folds = vec![
            TrainingMetrics {
                accuracy: 0.8,
                f1_score: 0.75,
                ..Default::default()
            },
            TrainingMetrics {
                accuracy: 0.85,
                f1_score: 0.80,
                ..Default::default()
            },
            TrainingMetrics {
                accuracy: 0.9,
                f1_score: 0.85,
                ..Default::default()
            },
        ];

        let cv = CrossValidationResults::summarize(folds);
        assert!((cv.mean_accuracy - 0.85).abs() < 0.001);
        assert!((cv.mean_f1 - 0.8).abs() < 0.001);
        assert!(cv.std_accuracy > 0.0);
    }

    #[test]
    fn test_cross_validation_empty() {
        let cv = CrossValidationResults::summarize(vec![]);
        assert_eq!(cv.mean_accuracy, 0.0);
        assert_eq!(cv.fold_metrics.len(), 0);
    }

    #[test]
    fn test_verdict_to_label() {
        assert!(!verdict_to_label(&TranspilerVerdict::Pass));
        assert!(verdict_to_label(&TranspilerVerdict::OutputMismatch));
        assert!(verdict_to_label(&TranspilerVerdict::TranspileError(
            "err".into()
        )));
        assert!(verdict_to_label(&TranspilerVerdict::Timeout));
    }

    #[test]
    fn test_train_test_split_ratio() {
        let examples = sample_examples(1000);
        let (train, test) = train_test_split(&examples, 0.8, 42);

        // Should be approximately 80/20 split
        let train_ratio = train.len() as f64 / examples.len() as f64;
        assert!(train_ratio > 0.7 && train_ratio < 0.9);
        assert_eq!(train.len() + test.len(), examples.len());
    }

    #[test]
    fn test_train_test_split_deterministic() {
        let examples = sample_examples(100);
        let (train1, _) = train_test_split(&examples, 0.8, 42);
        let (train2, _) = train_test_split(&examples, 0.8, 42);

        assert_eq!(train1.len(), train2.len());
    }

    #[test]
    fn test_training_error_display() {
        let err = TrainingError::InsufficientData {
            required: 100,
            provided: 50,
        };
        assert!(err.to_string().contains("100"));
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn test_model_metadata_clone() {
        let meta = ModelMetadata {
            model_type: "RandomForest".into(),
            trained_at: "2025-01-01".into(),
            train_examples: 1000,
            metrics: TrainingMetrics::default(),
        };
        let cloned = meta.clone();
        assert_eq!(cloned.model_type, meta.model_type);
    }

    // RED PHASE: Tests that require aprender integration

    #[test]
    #[ignore = "requires aprender ml feature"]
    fn test_random_forest_training() {
        // TODO: Implement with aprender RandomForestClassifier
        // let trainer = RandomForestTrainer::new();
        // let examples = sample_examples(1000);
        // let model = trainer.train(&examples, &TrainingConfig::default()).unwrap();
        // assert!(model.predict(&CodeFeatures::default()) >= 0.0);
        unimplemented!("RandomForest training not yet implemented")
    }

    #[test]
    #[ignore = "requires aprender ml feature"]
    fn test_cross_validation_with_model() {
        // TODO: Implement CV with actual model
        // let trainer = RandomForestTrainer::new();
        // let examples = sample_examples(500);
        // let cv = trainer.cross_validate(&examples, &TrainingConfig::default()).unwrap();
        // assert_eq!(cv.fold_metrics.len(), 5);
        // assert!(cv.mean_accuracy > 0.5);
        unimplemented!("Cross-validation not yet implemented")
    }

    #[test]
    #[ignore = "requires aprender ml feature"]
    fn test_model_save_load() {
        // TODO: Implement model persistence
        // let trainer = RandomForestTrainer::new();
        // let model = trainer.train(&examples, &config).unwrap();
        // model.save(Path::new("/tmp/model.bin")).unwrap();
        // let loaded = RandomForestTrainer::load(Path::new("/tmp/model.bin")).unwrap();
        // assert_eq!(model.predict(&features), loaded.predict(&features));
        unimplemented!("Model save/load not yet implemented")
    }

    #[test]
    #[ignore = "requires aprender ml feature"]
    fn test_stratified_split() {
        // TODO: Implement stratified sampling
        // let examples = sample_examples(1000);
        // let (train, test) = stratified_split(&examples, 0.8, 42);
        // let train_bug_ratio = train.iter().filter(|e| e.is_bug).count() as f64 / train.len() as f64;
        // let test_bug_ratio = test.iter().filter(|e| e.is_bug).count() as f64 / test.len() as f64;
        // assert!((train_bug_ratio - test_bug_ratio).abs() < 0.1);
        unimplemented!("Stratified split not yet implemented")
    }
}
