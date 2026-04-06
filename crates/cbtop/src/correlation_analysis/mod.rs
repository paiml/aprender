//! Multi-Metric Correlation Analysis (PMAT-032)
//!
//! Correlate performance variance with system events (interrupts, I/O, processes).
//!
//! # Features
//!
//! - Correlate CV spikes with system events
//! - Detect "noisy neighbor" interference
//! - Recommend isolation strategies
//! - Capture system state snapshots
//!
//! # Falsification Criteria (F1241-F1250)
//!
//! See `tests/correlation_analysis_f1241.rs` for falsification tests.

mod analyzer;
mod types;

pub use analyzer::CorrelationAnalyzer;
pub use types::{
    CorrelationResult, EventSample, EventType, InterferenceCategory, InterferenceResult,
    IsolationAction, IsolationRecommendation, PerformanceSample, SystemSnapshot,
};

#[cfg(test)]
mod tests;
