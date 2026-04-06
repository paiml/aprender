//! Calibration data collection for pruning
//!
//! Provides utilities for collecting activation statistics needed by
//! activation-weighted pruning methods like Wanda and SparseGPT.
//!
//! # Toyota Way: Genchi Genbutsu (Go and See)
//! Uses real activation data from calibration samples, not theoretical estimates.

mod collector;
mod config;
mod stats;

#[cfg(test)]
mod tests;

pub use collector::CalibrationCollector;
pub use config::CalibrationConfig;
// LayerActivationStats is used internally by CalibrationCollector and in tests
#[allow(unused_imports)]
pub(crate) use stats::LayerActivationStats;
