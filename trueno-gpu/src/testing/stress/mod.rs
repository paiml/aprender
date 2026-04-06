//! Stress Testing Framework with Randomized Inputs
//!
//! Frame-by-frame stress testing with:
//! - Randomized inputs via simular (deterministic RNG)
//! - Performance profiling via renacer
//! - Anomaly detection for regression identification
//!
//! # Sovereign Stack
//!
//! - `simular` v0.2.0: Deterministic RNG (SimRng)
//! - `renacer` v0.7.0: Profiling and anomaly detection

mod rng;
mod runner;
mod types;

#[cfg(test)]
mod tests_core;
#[cfg(test)]
mod tests_extended;

pub use rng::StressRng;
pub use runner::{StressConfig, StressTestRunner};
pub use types::{
    verify_performance, Anomaly, AnomalyKind, FrameProfile, PerformanceResult,
    PerformanceThresholds, StressReport,
};
