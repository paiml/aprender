//! Classification metrics for model evaluation
//!
//! Provides multi-class classification metrics including:
//! - Confusion matrix computation
//! - Per-class precision, recall, F1
//! - Macro, micro, and weighted averaging
//! - sklearn-style classification reports

mod average;
mod confusion;
mod metrics;
mod report;

#[cfg(test)]
mod basic_tests;
#[cfg(test)]
mod sklearn_parity_tests;

// Re-export all public types and functions
pub use average::Average;
pub use confusion::ConfusionMatrix;
pub use metrics::MultiClassMetrics;
pub use report::{classification_report, confusion_matrix};
