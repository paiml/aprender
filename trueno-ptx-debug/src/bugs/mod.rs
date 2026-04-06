//! Bug class definitions and registry

mod registry;

pub use registry::{BugClass, BugPattern, BugRegistry};

/// Severity levels for bugs
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Low severity - performance or style issue
    Low,
    /// Medium severity - potential correctness issue
    Medium,
    /// High severity - likely causes incorrect behavior
    High,
    /// Critical severity - causes crashes or data corruption
    Critical,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Low => write!(f, "LOW"),
            Severity::Medium => write!(f, "MEDIUM"),
            Severity::High => write!(f, "HIGH"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}
