//! Evaluation metrics for training and validation
//!
//! This module provides common metrics for evaluating model performance:
//! - Classification: Accuracy, Precision, Recall, F1
//! - Regression: MSE, MAE, R²

mod classification;
mod regression;
mod trait_def;

#[cfg(test)]
mod tests;

// Re-export the Metric trait
pub use trait_def::Metric;

// Re-export classification metrics
pub use classification::{Accuracy, F1Score, Precision, Recall};

// Re-export regression metrics
pub use regression::{R2Score, MAE, RMSE};
