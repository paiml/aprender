//! Type definitions for PTQ calibration
//!
//! Contains the core types used throughout the calibration module.

/// Calibration method for PTQ
#[derive(Clone, Debug, PartialEq, Default)]
pub enum CalibrationMethod {
    /// Min-max calibration: scale from actual min/max values
    #[default]
    MinMax,
    /// Percentile calibration: scale from percentile values (more robust to outliers)
    Percentile {
        /// Lower percentile (e.g., 0.01 for 0.01%)
        lower: f32,
        /// Upper percentile (e.g., 99.99 for 99.99%)
        upper: f32,
    },
    /// Moving average: smoothed min/max over multiple batches
    MovingAverage {
        /// Smoothing factor (0 = no smoothing, 1 = fully use new value)
        momentum: f32,
    },
}

/// Calibration result containing scale and zero_point
#[derive(Clone, Debug)]
pub struct CalibrationResult {
    /// Scale factor for quantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i32,
    /// Observed minimum value
    pub observed_min: f32,
    /// Observed maximum value
    pub observed_max: f32,
    /// Method used for calibration
    pub method: CalibrationMethod,
}
