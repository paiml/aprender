//! Type definitions for the Hansei report module.
//!
//! Contains severity levels, issue types, metric summaries, and trends.

/// Severity level for identified issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl std::fmt::Display for IssueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IssueSeverity::Info => write!(f, "INFO"),
            IssueSeverity::Warning => write!(f, "WARNING"),
            IssueSeverity::Error => write!(f, "ERROR"),
            IssueSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// An identified issue from post-training analysis
#[derive(Debug, Clone)]
pub struct TrainingIssue {
    pub severity: IssueSeverity,
    pub category: String,
    pub description: String,
    pub recommendation: String,
}

/// Summary statistics for a metric over the training run
#[derive(Debug, Clone)]
pub struct MetricSummary {
    pub initial: f64,
    pub final_value: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub trend: Trend,
}

/// Trend direction for a metric
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trend {
    Improving,
    Degrading,
    Stable,
    Oscillating,
}

impl std::fmt::Display for Trend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Trend::Improving => write!(f, "↑ Improving"),
            Trend::Degrading => write!(f, "↓ Degrading"),
            Trend::Stable => write!(f, "→ Stable"),
            Trend::Oscillating => write!(f, "~ Oscillating"),
        }
    }
}
