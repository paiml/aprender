//! PTQ (Post-Training Quantization) Calibration
//!
//! Calibration methods for determining quantization parameters (scale, zero_point)
//! from representative data:
//! - Min-Max: Uses the full range of observed values
//! - Percentile: Uses percentiles to be robust to outliers
//! - Moving Average: Smooths calibration over multiple batches

mod calibrator;
mod helpers;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types and functions
pub use calibrator::Calibrator;
pub use helpers::{calibrate_min_max, calibrate_percentile};
pub use types::{CalibrationMethod, CalibrationResult};
