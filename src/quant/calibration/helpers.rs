//! Helper functions for calibration
//!
//! Contains utility functions used by the calibration module.

use super::calibrator::Calibrator;
use super::types::CalibrationResult;

/// Simple deterministic pseudo-random for reservoir sampling
pub(crate) fn rand_simple(seed: usize) -> usize {
    // Simple LCG-based PRNG
    let a: usize = 1103515245;
    let c: usize = 12345;
    let m: usize = 1 << 31;
    (a.wrapping_mul(seed).wrapping_add(c)) % m
}

/// Convenience function for min-max calibration
pub fn calibrate_min_max(data: &[f32], bits: usize, symmetric: bool) -> CalibrationResult {
    let mut calibrator = Calibrator::min_max(bits, symmetric);
    calibrator.observe(data);
    calibrator.compute()
}

/// Convenience function for percentile calibration
pub fn calibrate_percentile(
    data: &[f32],
    bits: usize,
    symmetric: bool,
    lower: f32,
    upper: f32,
) -> CalibrationResult {
    let mut calibrator = Calibrator::percentile(bits, symmetric, lower, upper, data.len());
    calibrator.observe(data);
    calibrator.compute()
}
