//! Anomaly Detection Engine (PMAT-034)
//!
//! Automated anomaly detection and outlier classification for performance data.
//!
//! # Features
//!
//! - Z-score outlier detection (>3σ)
//! - IQR-based robust outlier detection
//! - Change point detection for performance cliffs
//! - Anomaly classification and root cause identification
//!
//! # Falsification Criteria (F1261-F1270)
//!
//! See `tests/anomaly_detection_f1261.rs` for falsification tests.

mod detector;
mod types;

pub use detector::AnomalyDetector;
pub use types::{
    Anomaly, AnomalyReport, AnomalySeverity, AnomalyType, ChangePoint, DEFAULT_IQR_MULTIPLIER,
    DEFAULT_ZSCORE_THRESHOLD, MIN_SAMPLES_FOR_DETECTION,
};

#[cfg(test)]
mod tests;
