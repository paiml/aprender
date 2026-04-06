//! Violation count tracking
//!
//! Provides structured counts of violations by severity level.

/// Counts of violations by severity
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViolationCounts {
    /// Critical violations (severity >= 0.8)
    pub critical: u32,
    /// Warning violations (0.5 <= severity < 0.8)
    pub warnings: u32,
    /// Minor violations (severity < 0.5)
    pub minor: u32,
    /// Total violations
    pub total: u32,
}
