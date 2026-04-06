//! Hansei (反省) Post-Training Report Generator
//!
//! Toyota Way principle: Reflection and continuous improvement through
//! systematic analysis of training outcomes.
//!
//! Reference: Liker, J.K. (2004). The Toyota Way: 14 Management Principles.

mod analyzer;
mod output;
mod types;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use analyzer::HanseiAnalyzer;
pub use output::PostTrainingReport;
pub use types::{IssueSeverity, MetricSummary, TrainingIssue, Trend};
