//! Thermal Trend Prediction (PMAT-030)
//!
//! Enhanced thermal analysis with trend prediction, throttle forecasting,
//! and cooldown recommendations.
//!
//! # Features
//!
//! - Temperature trend prediction
//! - Throttle risk calculation
//! - Cooldown time recommendations
//! - Thermal-latency correlation analysis
//!
//! # References
//!
//! - Brooks 2000: "Dynamic Thermal Management for High-Performance Microprocessors" HPCA
//! - Rotem et al. 2012: "Power-Management Architecture of Intel Microarchitectures" IEEE Micro
//!
//! # Falsification Criteria (F1221-F1230)
//!
//! See `tests/thermal_prediction_f1221.rs` for falsification tests.

mod analysis;
mod analyzer;
mod regression;
mod types;

pub use analyzer::ThermalAnalyzer;
pub use types::{
    CooldownRecommendation, RiskCategory, ThermalCorrelation, ThermalPrediction, ThermalSample,
    ThermalVariance, ThrottleRisk,
};

/// Default throttle threshold temperature in Celsius
pub const DEFAULT_THROTTLE_THRESHOLD_C: f64 = 85.0;

/// Minimum samples required for analysis
pub const MIN_SAMPLES_FOR_ANALYSIS: usize = 3;

/// Convenience function to analyze thermal data
pub fn analyze_thermal(
    samples: &[(f64, f64)], // (temp, timestamp) pairs
    horizon_sec: f64,
) -> Option<ThermalPrediction> {
    let mut analyzer = ThermalAnalyzer::new(samples.len());
    for &(temp, time) in samples {
        analyzer.add(temp, time);
    }
    analyzer.predict_trend(horizon_sec)
}

/// Convenience function to assess throttle risk
pub fn assess_throttle_risk(current_temp: f64, threshold: f64, trend_slope: f64) -> ThrottleRisk {
    ThrottleRisk::assess(current_temp, threshold, trend_slope, 10.0)
}

#[cfg(test)]
mod tests;
