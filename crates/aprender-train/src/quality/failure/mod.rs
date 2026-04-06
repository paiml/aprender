//! Failure Context Structured Diagnostics (ENT-007)
//!
//! Provides structured failure diagnostics with categorization,
//! suggested fixes, and Pareto analysis for training runs.

mod analysis;
mod types;

pub use analysis::{top_failure_categories, ParetoAnalysis};
pub use types::{FailureCategory, FailureContext};

#[cfg(test)]
mod tests;
