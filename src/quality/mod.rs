//! Quality Gates Module (ENT-005, ENT-006, ENT-007)
//!
//! Provides code quality metrics, supply chain auditing, and failure diagnostics
//! for training runs following Jidoka (自働化) principles.
//!
//! # Components
//!
//! - [`pmat`] - PMAT code quality metrics (coverage, mutation score, clippy)
//! - [`supply_chain`] - cargo-deny dependency auditing
//! - [`failure`] - Structured failure diagnostics with Pareto analysis

pub mod failure;
pub mod pmat;
pub mod supply_chain;

pub use failure::{FailureCategory, FailureContext};
pub use pmat::{CodeQualityMetrics, PmatGrade};
pub use supply_chain::{Advisory, AuditStatus, DependencyAudit, Severity};
