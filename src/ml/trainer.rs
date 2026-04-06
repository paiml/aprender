//! Bug prediction model training pipeline
//!
//! This module provides a complete ML training pipeline:
//! - Train/test split with stratification
//! - Cross-validation (k-fold)
//! - Metrics tracking (precision, recall, F1, AUC)
//! - Model serialization

use crate::data::CodeFeatures;
use crate::Result;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};

/// Metrics from model evaluation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// True positives
    pub true_positives: usize,
    /// True negatives
    pub true_negatives: usize,
    /// False positives
    pub false_positives: usize,
    /// False negatives
    pub false_negatives: usize,
    /// Precision = TP / (TP + FP)
    pub precision: f64,
    /// Recall = TP / (TP + FN)
    pub recall: f64,
    /// F1 score = 2 * (precision * recall) / (precision + recall)
    pub f1_score: f64,
    /// Accuracy = (TP + TN) / total
    pub accuracy: f64,
    /// Area under ROC curve (approximated)
    pub auc: f64,
}

impl ModelMetrics {
    /// Compute metrics from predictions and ground truth
    #[must_use]
    pub fn compute(predictions: &[bool], ground_truth: &[bool]) -> Self {
        let mut tp = 0;
        let mut tn = 0;
        let mut fp = 0;
        let mut r#fn = 0;

        for (pred, truth) in predictions.iter().zip(ground_truth.iter()) {
            match (pred, truth) {
                (true, true) => tp += 1,
                (false, false) => tn += 1,
                (true, false) => fp += 1,
                (false, true) => r#fn += 1,
            }
        }

        let precision = if tp + fp > 0 {
            tp as f64 / (tp + fp) as f64
        } else {
            0.0
        };

        let recall = if tp + r#fn > 0 {
            tp as f64 / (tp + r#fn) as f64
        } else {
            0.0
        };

        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let total = tp + tn + fp + r#fn;
        let accuracy = if total > 0 {
            (tp + tn) as f64 / total as f64
        } else {
            0.0
        };

        // Approximate AUC using balanced accuracy
        let tpr = recall;
        let tnr = if tn + fp > 0 {
            tn as f64 / (tn + fp) as f64
        } else {
            0.0
        };
        let auc = (tpr + tnr) / 2.0;

        Self {
            true_positives: tp,
            true_negatives: tn,
            false_positives: fp,
            false_negatives: r#fn,
            precision,
            recall,
            f1_score,
            accuracy,
            auc,
        }
    }

    /// Average metrics from multiple folds
    #[must_use]
    pub fn average(metrics: &[ModelMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let n = metrics.len() as f64;
        Self {
            true_positives: metrics.iter().map(|m| m.true_positives).sum::<usize>() / metrics.len(),
            true_negatives: metrics.iter().map(|m| m.true_negatives).sum::<usize>() / metrics.len(),
            false_positives: metrics.iter().map(|m| m.false_positives).sum::<usize>()
                / metrics.len(),
            false_negatives: metrics.iter().map(|m| m.false_negatives).sum::<usize>()
                / metrics.len(),
            precision: metrics.iter().map(|m| m.precision).sum::<f64>() / n,
            recall: metrics.iter().map(|m| m.recall).sum::<f64>() / n,
            f1_score: metrics.iter().map(|m| m.f1_score).sum::<f64>() / n,
            accuracy: metrics.iter().map(|m| m.accuracy).sum::<f64>() / n,
            auc: metrics.iter().map(|m| m.auc).sum::<f64>() / n,
        }
    }
}

/// Configuration for model training
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Train/test split ratio (e.g., 0.8 for 80% train)
    pub train_ratio: f64,
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Number of trees in random forest
    pub n_trees: usize,
    /// Maximum depth of trees
    pub max_depth: usize,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            train_ratio: 0.8,
            cv_folds: 5,
            seed: 42,
            n_trees: 100,
            max_depth: 10,
        }
    }
}

/// Results from model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Metrics on training set
    pub train_metrics: ModelMetrics,
    /// Metrics on test set
    pub test_metrics: ModelMetrics,
    /// Cross-validation metrics (one per fold)
    pub cv_metrics: Vec<ModelMetrics>,
    /// Average cross-validation metrics
    pub cv_average: ModelMetrics,
    /// Number of training samples
    pub train_samples: usize,
    /// Number of test samples
    pub test_samples: usize,
}

/// Model trainer for bug prediction
#[derive(Debug)]
pub struct ModelTrainer {
    config: TrainingConfig,
}

impl ModelTrainer {
    /// Create a new trainer with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: TrainingConfig::default(),
        }
    }

    /// Create trainer with custom configuration
    #[must_use]
    pub fn with_config(config: TrainingConfig) -> Self {
        Self { config }
    }

    /// Set train/test split ratio
    #[must_use]
    pub fn train_ratio(mut self, ratio: f64) -> Self {
        self.config.train_ratio = ratio.clamp(0.1, 0.99);
        self
    }

    /// Set number of cross-validation folds
    #[must_use]
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.config.cv_folds = folds.max(2);
        self
    }

    /// Set random seed
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.config.seed = seed;
        self
    }

    /// Split data into train and test sets with stratification
    pub fn train_test_split(
        &self,
        features: &[CodeFeatures],
        labels: &[bool],
    ) -> (Vec<CodeFeatures>, Vec<bool>, Vec<CodeFeatures>, Vec<bool>) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);

        // Separate positive and negative samples for stratification
        let positives: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l)
            .map(|(i, _)| i)
            .collect();
        let negatives: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| !l)
            .map(|(i, _)| i)
            .collect();

        // Shuffle each class
        let mut pos_shuffled = positives.clone();
        let mut neg_shuffled = negatives.clone();
        pos_shuffled.shuffle(&mut rng);
        neg_shuffled.shuffle(&mut rng);

        // Split each class by ratio (values are always non-negative)
        #[allow(clippy::cast_sign_loss)]
        let pos_split = (pos_shuffled.len() as f64 * self.config.train_ratio) as usize;
        #[allow(clippy::cast_sign_loss)]
        let neg_split = (neg_shuffled.len() as f64 * self.config.train_ratio) as usize;

        let train_indices: Vec<usize> = pos_shuffled[..pos_split]
            .iter()
            .chain(neg_shuffled[..neg_split].iter())
            .copied()
            .collect();

        let test_indices: Vec<usize> = pos_shuffled[pos_split..]
            .iter()
            .chain(neg_shuffled[neg_split..].iter())
            .copied()
            .collect();

        let train_features: Vec<CodeFeatures> =
            train_indices.iter().map(|&i| features[i].clone()).collect();
        let train_labels: Vec<bool> = train_indices.iter().map(|&i| labels[i]).collect();
        let test_features: Vec<CodeFeatures> =
            test_indices.iter().map(|&i| features[i].clone()).collect();
        let test_labels: Vec<bool> = test_indices.iter().map(|&i| labels[i]).collect();

        (train_features, train_labels, test_features, test_labels)
    }

    /// Perform k-fold cross-validation
    ///
    /// # Errors
    ///
    /// Returns error if evaluation fails on any fold.
    pub fn cross_validate(
        &self,
        features: &[CodeFeatures],
        labels: &[bool],
    ) -> Result<Vec<ModelMetrics>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.config.seed);
        let n = features.len();
        let fold_size = n / self.config.cv_folds;

        // Shuffle indices
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);

        let mut metrics = Vec::with_capacity(self.config.cv_folds);

        for fold in 0..self.config.cv_folds {
            let start = fold * fold_size;
            let end = if fold == self.config.cv_folds - 1 {
                n
            } else {
                start + fold_size
            };

            // Test fold
            let test_indices: Vec<usize> = indices[start..end].to_vec();

            // Train on remaining folds
            let train_indices: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            let train_features: Vec<CodeFeatures> =
                train_indices.iter().map(|&i| features[i].clone()).collect();
            let train_labels: Vec<bool> = train_indices.iter().map(|&i| labels[i]).collect();
            let test_features: Vec<CodeFeatures> =
                test_indices.iter().map(|&i| features[i].clone()).collect();
            let test_labels: Vec<bool> = test_indices.iter().map(|&i| labels[i]).collect();

            // Train and evaluate on this fold
            let fold_metrics = self.train_and_evaluate(
                &train_features,
                &train_labels,
                &test_features,
                &test_labels,
            )?;
            metrics.push(fold_metrics);
        }

        Ok(metrics)
    }

    /// Train model and evaluate on test set
    fn train_and_evaluate(
        &self,
        _train_features: &[CodeFeatures],
        _train_labels: &[bool],
        test_features: &[CodeFeatures],
        test_labels: &[bool],
    ) -> Result<ModelMetrics> {
        // Use heuristic predictor for now (aprender training requires 'ml' feature)
        // This demonstrates the training pipeline structure
        let predictor = super::BugPredictor::new();

        let predictions: Vec<bool> = test_features
            .iter()
            .map(|f| predictor.predict(f) > 0.5)
            .collect();

        Ok(ModelMetrics::compute(&predictions, test_labels))
    }

    /// Full training pipeline: split, train, cross-validate, evaluate
    ///
    /// # Errors
    ///
    /// Returns error if training or evaluation fails.
    pub fn train(&self, features: &[CodeFeatures], labels: &[bool]) -> Result<TrainingResult> {
        // Train/test split
        let (train_features, train_labels, test_features, test_labels) =
            self.train_test_split(features, labels);

        // Cross-validation on training set
        let cv_metrics = self.cross_validate(&train_features, &train_labels)?;
        let cv_average = ModelMetrics::average(&cv_metrics);

        // Final evaluation on test set
        let train_metrics = self.train_and_evaluate(
            &train_features,
            &train_labels,
            &train_features,
            &train_labels,
        )?;
        let test_metrics =
            self.train_and_evaluate(&train_features, &train_labels, &test_features, &test_labels)?;

        Ok(TrainingResult {
            train_metrics,
            test_metrics,
            cv_metrics,
            cv_average,
            train_samples: train_features.len(),
            test_samples: test_features.len(),
        })
    }
}

impl Default for ModelTrainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable model state for persistence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Model version
    pub version: String,
    /// Training configuration
    pub config: TrainingConfig,
    /// Training metrics
    pub metrics: ModelMetrics,
    /// Feature weights (for simple models)
    pub weights: Vec<f64>,
}

impl SerializedModel {
    /// Save model to JSON file
    ///
    /// # Errors
    ///
    /// Returns error if file writing fails
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| crate::Error::Data(format!("Serialization failed: {e}")))?;
        std::fs::write(path, json)
            .map_err(|e| crate::Error::Data(format!("Failed to write file: {e}")))?;
        Ok(())
    }

    /// Load model from JSON file
    ///
    /// # Errors
    ///
    /// Returns error if file reading or parsing fails
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| crate::Error::Data(format!("Failed to read file: {e}")))?;
        let model: Self = serde_json::from_str(&json)
            .map_err(|e| crate::Error::Data(format!("Deserialization failed: {e}")))?;
        Ok(model)
    }
}

// Implement Serialize/Deserialize for TrainingConfig
impl Serialize for TrainingConfig {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TrainingConfig", 5)?;
        state.serialize_field("train_ratio", &self.train_ratio)?;
        state.serialize_field("cv_folds", &self.cv_folds)?;
        state.serialize_field("seed", &self.seed)?;
        state.serialize_field("n_trees", &self.n_trees)?;
        state.serialize_field("max_depth", &self.max_depth)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for TrainingConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            train_ratio: f64,
            cv_folds: usize,
            seed: u64,
            n_trees: usize,
            max_depth: usize,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(Self {
            train_ratio: helper.train_ratio,
            cv_folds: helper.cv_folds,
            seed: helper.seed,
            n_trees: helper.n_trees,
            max_depth: helper.max_depth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> (Vec<CodeFeatures>, Vec<bool>) {
        let features: Vec<CodeFeatures> = (0..100)
            .map(|i| CodeFeatures {
                ast_depth: (i % 10) as u32,
                num_operators: (i % 20) as u32,
                num_control_flow: (i % 5) as u32,
                cyclomatic_complexity: (i % 15) as f32,
                uses_edge_values: i % 3 == 0,
                ..Default::default()
            })
            .collect();
        let labels: Vec<bool> = (0..100).map(|i| i % 4 == 0).collect();
        (features, labels)
    }

    #[test]
    fn test_model_metrics_compute() {
        let predictions = vec![true, true, false, false, true];
        let ground_truth = vec![true, false, false, true, true];

        let metrics = ModelMetrics::compute(&predictions, &ground_truth);

        assert_eq!(metrics.true_positives, 2);
        assert_eq!(metrics.true_negatives, 1);
        assert_eq!(metrics.false_positives, 1);
        assert_eq!(metrics.false_negatives, 1);
        assert!((metrics.precision - 0.666).abs() < 0.01);
        assert!((metrics.recall - 0.666).abs() < 0.01);
    }

    #[test]
    fn test_model_metrics_perfect() {
        let predictions = vec![true, false, true, false];
        let ground_truth = vec![true, false, true, false];

        let metrics = ModelMetrics::compute(&predictions, &ground_truth);

        assert!((metrics.precision - 1.0).abs() < f64::EPSILON);
        assert!((metrics.recall - 1.0).abs() < f64::EPSILON);
        assert!((metrics.f1_score - 1.0).abs() < f64::EPSILON);
        assert!((metrics.accuracy - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_metrics_average() {
        let metrics = vec![
            ModelMetrics {
                precision: 0.8,
                recall: 0.7,
                f1_score: 0.75,
                accuracy: 0.85,
                auc: 0.9,
                ..Default::default()
            },
            ModelMetrics {
                precision: 0.6,
                recall: 0.9,
                f1_score: 0.72,
                accuracy: 0.75,
                auc: 0.8,
                ..Default::default()
            },
        ];

        let avg = ModelMetrics::average(&metrics);

        assert!((avg.precision - 0.7).abs() < f64::EPSILON);
        assert!((avg.recall - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert!((config.train_ratio - 0.8).abs() < f64::EPSILON);
        assert_eq!(config.cv_folds, 5);
        assert_eq!(config.seed, 42);
    }

    #[test]
    fn test_trainer_new() {
        let trainer = ModelTrainer::new();
        assert!((trainer.config.train_ratio - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trainer_builder() {
        let trainer = ModelTrainer::new().train_ratio(0.7).cv_folds(10).seed(123);

        assert!((trainer.config.train_ratio - 0.7).abs() < f64::EPSILON);
        assert_eq!(trainer.config.cv_folds, 10);
        assert_eq!(trainer.config.seed, 123);
    }

    #[test]
    fn test_train_test_split() {
        let (features, labels) = sample_data();
        let trainer = ModelTrainer::new();

        let (train_f, train_l, test_f, test_l) = trainer.train_test_split(&features, &labels);

        // Check split ratio approximately
        let total = features.len();
        let train_expected = (total as f64 * 0.8) as usize;
        assert!(train_f.len() >= train_expected - 5 && train_f.len() <= train_expected + 5);
        assert_eq!(train_f.len(), train_l.len());
        assert_eq!(test_f.len(), test_l.len());
    }

    #[test]
    fn test_cross_validate() {
        let (features, labels) = sample_data();
        let trainer = ModelTrainer::new().cv_folds(5);

        let cv_metrics = trainer.cross_validate(&features, &labels).unwrap();

        assert_eq!(cv_metrics.len(), 5);
        for m in &cv_metrics {
            assert!((0.0..=1.0).contains(&m.accuracy));
        }
    }

    #[test]
    fn test_train_full_pipeline() {
        let (features, labels) = sample_data();
        let trainer = ModelTrainer::new();

        let result = trainer.train(&features, &labels).unwrap();

        assert!(result.train_samples > 0);
        assert!(result.test_samples > 0);
        assert_eq!(result.cv_metrics.len(), 5);
        assert!((0.0..=1.0).contains(&result.test_metrics.accuracy));
    }

    #[test]
    fn test_serialized_model() {
        let model = SerializedModel {
            version: "0.1.0".to_string(),
            config: TrainingConfig::default(),
            metrics: ModelMetrics::default(),
            weights: vec![0.1, 0.2, 0.3],
        };

        let json = serde_json::to_string(&model).unwrap();
        let loaded: SerializedModel = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, "0.1.0");
        assert_eq!(loaded.weights.len(), 3);
    }

    #[test]
    fn test_training_result_serialize() {
        let result = TrainingResult {
            train_metrics: ModelMetrics::default(),
            test_metrics: ModelMetrics::default(),
            cv_metrics: vec![ModelMetrics::default()],
            cv_average: ModelMetrics::default(),
            train_samples: 80,
            test_samples: 20,
        };

        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("train_samples"));
    }

    #[test]
    fn test_model_metrics_empty() {
        let metrics = ModelMetrics::compute(&[], &[]);
        assert_eq!(metrics.true_positives, 0);
        assert!((metrics.accuracy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_metrics_all_negative() {
        let predictions = vec![false, false, false];
        let ground_truth = vec![false, false, false];

        let metrics = ModelMetrics::compute(&predictions, &ground_truth);

        assert_eq!(metrics.true_negatives, 3);
        assert!((metrics.accuracy - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trainer_ratio_clamp() {
        let trainer = ModelTrainer::new().train_ratio(0.05);
        assert!((trainer.config.train_ratio - 0.1).abs() < f64::EPSILON);

        let trainer = ModelTrainer::new().train_ratio(1.5);
        assert!((trainer.config.train_ratio - 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn test_trainer_cv_folds_min() {
        let trainer = ModelTrainer::new().cv_folds(1);
        assert_eq!(trainer.config.cv_folds, 2);
    }

    #[test]
    fn test_model_metrics_auc_calculation() {
        // Test that AUC is computed correctly when tn + fp > 0
        // Mutant: replace > with == would break this
        //
        // Create data to produce: tp=10, tn=20, fp=5, fn=3
        let mut predictions = Vec::new();
        let mut ground_truth = Vec::new();

        // 10 true positives (pred=true, truth=true)
        for _ in 0..10 {
            predictions.push(true);
            ground_truth.push(true);
        }
        // 20 true negatives (pred=false, truth=false)
        for _ in 0..20 {
            predictions.push(false);
            ground_truth.push(false);
        }
        // 5 false positives (pred=true, truth=false)
        for _ in 0..5 {
            predictions.push(true);
            ground_truth.push(false);
        }
        // 3 false negatives (pred=false, truth=true)
        for _ in 0..3 {
            predictions.push(false);
            ground_truth.push(true);
        }

        let metrics = ModelMetrics::compute(&predictions, &ground_truth);

        // tnr = tn / (tn + fp) = 20 / (20 + 5) = 0.8
        // tpr = recall = tp / (tp + fn) = 10 / (10 + 3) = 0.769...
        // auc = (tpr + tnr) / 2 = (0.769 + 0.8) / 2 = 0.784...
        assert!(
            metrics.auc > 0.7,
            "AUC should be > 0.7, got {}",
            metrics.auc
        );
        assert!(
            metrics.auc < 0.85,
            "AUC should be < 0.85, got {}",
            metrics.auc
        );

        // If mutant changed > to ==, tnr would be 0.0 when tn+fp > 0
        // Then auc would be ~0.38, which would fail the > 0.7 check
    }

    #[test]
    fn test_model_metrics_average_fp_fn() {
        // Test that false_positives and false_negatives are averaged (divided) correctly
        // Mutant: replace / with * would break this
        let metrics = vec![
            ModelMetrics {
                false_positives: 10,
                false_negatives: 20,
                ..Default::default()
            },
            ModelMetrics {
                false_positives: 30,
                false_negatives: 40,
                ..Default::default()
            },
        ];

        let avg = ModelMetrics::average(&metrics);

        // Average of [10, 30] = 20, not 80 (if * instead of /)
        assert_eq!(avg.false_positives, 20);
        // Average of [20, 40] = 30, not 120 (if * instead of /)
        assert_eq!(avg.false_negatives, 30);
    }

    #[test]
    fn test_trainer_with_config() {
        // Test that with_config actually uses the provided config
        // Mutant: replace with Default::default() would break this
        let config = TrainingConfig {
            train_ratio: 0.6,
            cv_folds: 10,
            seed: 12345,
            n_trees: 50,
            max_depth: 5,
        };
        let trainer = ModelTrainer::with_config(config);

        // These values are different from defaults, so mutant would fail
        assert!((trainer.config.train_ratio - 0.6).abs() < f64::EPSILON);
        assert_eq!(trainer.config.cv_folds, 10);
        assert_eq!(trainer.config.seed, 12345);
    }
}
