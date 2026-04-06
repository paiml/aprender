#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! 100-Point Popperian Falsification Test Suite for ML Tuner
//!
//! Implements SHOWCASE-BRICK-001 Section 12.7 falsification protocol.
//! GitHub Issue: https://github.com/paiml/trueno/issues/84
//!
//! Categories (20 points each):
//! - F001-F020: Model Accuracy
//! - F021-F040: Feature Engineering
//! - F041-F060: Training Data Quality
//! - F061-F080: Integration Correctness
//! - F081-F100: Generalization & Robustness
//! - F280-F295: Phase 14 ML-Tuner Evolution

mod feature_engineering;
mod generalization_robustness;
mod integration_correctness;
mod model_accuracy;
mod phase14_evolution;
mod training_data_quality;
