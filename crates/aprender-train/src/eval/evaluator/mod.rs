//! Model Evaluator for standardized evaluation and comparison
//!
//! Provides ModelEvaluator for running comprehensive evaluations,
//! comparing multiple models, and generating leaderboards.

mod config;
mod kfold;
mod leaderboard;
mod metric;
mod model_evaluator;
mod result;

#[cfg(test)]
mod tests;

// Re-export all public types for API compatibility
pub use config::EvalConfig;
pub use kfold::KFold;
pub use leaderboard::Leaderboard;
pub use metric::{Metric, RougeVariant};
pub use model_evaluator::ModelEvaluator;
pub use result::EvalResult;
