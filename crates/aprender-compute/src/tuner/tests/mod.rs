//! Tuner module tests.
//!
//! Organized into focused test groups:
//! - `model_accuracy`: Tests F001-F020 for model prediction accuracy
//! - `feature_engineering`: Tests F021-F040 for feature normalization
//! - `training_data`: Tests F041-F060 for training data validation
//! - `integration`: Tests F061-F080 for component integration
//! - `generalization`: Tests F081-F100 for model generalization
//! - `classifiers`: Additional classifier and bottleneck tests
//! - `coverage`: Additional coverage for builder, error, and config tests

mod classifiers;
mod coverage;
mod data_collector_coverage;
mod evolution;
mod feature_engineering;
mod generalization;
mod integration;
mod model_accuracy;
mod training_data;
