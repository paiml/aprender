//! Supply Chain Auditing (ENT-006)
//!
//! Provides cargo-deny integration for dependency vulnerability scanning
//! and license compliance checking.

mod advisory;
mod audit_status;
mod audit_summary;
mod dependency_audit;
mod error;
mod severity;

#[cfg(test)]
mod tests;

pub use advisory::Advisory;
pub use audit_status::AuditStatus;
pub use audit_summary::AuditSummary;
pub use dependency_audit::DependencyAudit;
pub use error::{Result, SupplyChainError};
pub use severity::Severity;
