//! Model Evaluation Framework (APR-073)
//!
//! Comprehensive evaluation module implementing the Model Evaluation Framework Specification.
//! Provides standardized metrics, model comparison, and drift detection with Jidoka principles.
//!
//! ## Architecture
//!
//! - `classification`: Multi-class classification metrics, confusion matrix, reports
//! - `evaluator`: ModelEvaluator for running evaluations and comparisons
//! - `drift`: Statistical drift detection (KS, Chi-sq, PSI)
//! - `retrain`: Auto-retraining with Andon pattern
//!
//! ## Example
//!
//! ```ignore
//! use entrenar::eval::{ModelEvaluator, EvalConfig, Metric, Average};
//!
//! let evaluator = ModelEvaluator::new(EvalConfig {
//!     metrics: vec![Metric::Accuracy, Metric::F1(Average::Weighted)],
//!     cv_folds: 5,
//!     ..Default::default()
//! });
//!
//! let result = evaluator.evaluate(&model, &x_test, &y_test)?;
//! println!("Accuracy: {:.2}%", result.get_score(Metric::Accuracy) * 100.0);
//! ```

pub mod classification;
pub mod drift;
pub mod evaluator;
pub mod generative;
pub mod retrain;

// Re-export basic drift types from monitor for backward compatibility
pub use crate::monitor::drift::{AnomalySeverity, DriftStatus, SlidingWindowBaseline};

// Re-export advanced drift types
pub use drift::{DriftCallback, DriftDetector, DriftResult, DriftSummary, DriftTest, Severity};

// Re-export retrain types
pub use retrain::{
    Action, AutoRetrainer, RetrainCallback, RetrainConfig, RetrainPolicy, RetrainerStats,
};

// Re-export main types
pub use classification::{
    classification_report, confusion_matrix, Average, ConfusionMatrix, MultiClassMetrics,
};
pub use evaluator::{
    EvalConfig, EvalResult, KFold, Leaderboard, Metric, ModelEvaluator, RougeVariant,
};

// Re-export generative metric functions
pub use generative::{
    bleu_score, ndcg_at_k, pass_at_k, perplexity, real_time_factor_inverse, rouge_l, rouge_n,
    word_error_rate,
};
