//! Performance Prediction Model (PMAT-033)
//!
//! Predict performance for untested workload sizes using historical baselines.
//!
//! # Features
//!
//! - Curve fitting (polynomial, exponential, roofline)
//! - Performance prediction for arbitrary sizes
//! - Confidence bounds estimation
//! - Model selection and comparison
//!
//! # Falsification Criteria (F1251-F1260)
//!
//! See `tests/performance_prediction_f1251.rs` for falsification tests.

mod predictor;
mod types;

pub use predictor::PerformancePredictor;
pub use types::{DataPoint, FittedModel, ModelType, Prediction, MIN_SAMPLES_FOR_FIT};

#[cfg(test)]
mod tests;
