//! Audit status for dependencies.

use serde::{Deserialize, Serialize};

/// Audit status for a dependency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditStatus {
    /// No issues found
    Clean,
    /// Warnings present (license issues, yanked versions)
    Warning,
    /// Vulnerabilities found
    Vulnerable,
}

impl AuditStatus {
    /// Returns true if the status indicates a failure condition
    pub fn is_failure(&self) -> bool {
        matches!(self, Self::Vulnerable)
    }

    /// Returns true if the status indicates any issues
    pub fn has_issues(&self) -> bool {
        !matches!(self, Self::Clean)
    }
}
