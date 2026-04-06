//! Context-Aware Regression Predictor (PMAT-039)
//!
//! Context-aware regression thresholds accounting for system state and historical trends.
//!
//! # Features
//!
//! - Context capture (temperature, memory, frequency)
//! - Adaptive threshold computation based on context
//! - Trend detection from historical data
//! - False positive reduction through learned patterns
//!
//! # Falsification Criteria (F1311-F1320)
//!
//! See `tests/context_regression_f1311.rs` for falsification tests.

mod predictor;
mod types;

pub use predictor::ContextRegressionPredictor;
pub use types::{
    BaselineEntry, RegressionCheck, RegressionThreshold, SystemContext, Trend,
    DEFAULT_COLD_START_MARGIN, DEFAULT_STALENESS_SEC, MIN_SAMPLES_FOR_CONTEXT,
};

#[cfg(test)]
mod tests;
