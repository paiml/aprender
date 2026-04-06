//! Audit summary for dependency scanning.

use serde::{Deserialize, Serialize};

use super::{AuditStatus, DependencyAudit};

/// Summary of a full dependency audit
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuditSummary {
    /// Total dependencies scanned
    pub total_dependencies: u32,

    /// Clean dependencies
    pub clean_count: u32,

    /// Dependencies with warnings
    pub warning_count: u32,

    /// Vulnerable dependencies
    pub vulnerable_count: u32,

    /// Individual audit results
    pub audits: Vec<DependencyAudit>,
}

impl AuditSummary {
    /// Create a summary from a list of audits
    pub fn from_audits(audits: Vec<DependencyAudit>) -> Self {
        let total_dependencies = audits.len() as u32;
        let clean_count =
            audits.iter().filter(|a| a.audit_status == AuditStatus::Clean).count() as u32;
        let warning_count =
            audits.iter().filter(|a| a.audit_status == AuditStatus::Warning).count() as u32;
        let vulnerable_count =
            audits.iter().filter(|a| a.audit_status == AuditStatus::Vulnerable).count() as u32;

        Self { total_dependencies, clean_count, warning_count, vulnerable_count, audits }
    }

    /// Returns true if any vulnerabilities were found
    pub fn has_vulnerabilities(&self) -> bool {
        self.vulnerable_count > 0
    }

    /// Returns true if any issues (warnings or vulnerabilities) were found
    pub fn has_issues(&self) -> bool {
        self.warning_count > 0 || self.vulnerable_count > 0
    }

    /// Get all vulnerable dependencies
    pub fn vulnerable_deps(&self) -> Vec<&DependencyAudit> {
        self.audits.iter().filter(|a| a.is_vulnerable()).collect()
    }
}
