//! Release Orchestrator Module
//!
//! Implements the `batuta stack release` command functionality.
//! Coordinates releases across multiple crates in topological order,
//! ensuring all quality gates pass before publishing.
//!
//! ## Module Structure
//!
//! - `orchestrator`: Core `ReleaseOrchestrator` struct and methods
//! - `tests`: Unit tests for release orchestration
//! - `proptests`: Property-based tests using proptest

mod orchestrator;

#[cfg(test)]
mod proptests;

#[cfg(test)]
mod tests;

// Re-export the main orchestrator
pub use orchestrator::ReleaseOrchestrator;

// Re-export types from releaser_types module for convenience
pub use super::releaser_types::{
    format_plan_text, BumpType, ReleaseConfig, ReleaseResult, ReleasedCrate,
};
