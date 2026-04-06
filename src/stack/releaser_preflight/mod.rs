//! Release Preflight Checks
//!
//! Preflight check methods for ReleaseOrchestrator extracted from releaser.rs.
//! Contains all check_* methods for various quality gates.
//!
//! The common command-execution pattern is factored into [`helpers::run_check_command`],
//! which handles argument parsing, spawning, UTF-8 decoding, and the
//! not-found / general-error branches that every check shares.

mod checks;
pub mod helpers;

#[cfg(test)]
mod tests;

// Re-export helper functions for use by other modules
pub use helpers::{
    parse_count_from_json_multi, parse_value_from_json, run_check_command, score_check_result,
};
