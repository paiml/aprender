//! Tests for `batuta code` CLI module.
//!
//! Tests the library-level `agent::code` module which contains all logic.

use batuta::agent::code::*;
use batuta::serve::backends::PrivacyTier;

#[test]
fn test_cmd_code_is_accessible() {
    // Verify the public API entry point exists and is callable.
    // We can't run cmd_code without a model, but we can verify it compiles.
    // Verify cmd_code exists and is callable (signature tested in agent/code_tests.rs)
    let _ = cmd_code as fn(_, _, _, _, _, _, _) -> _;
}

#[test]
fn test_exit_codes_match_spec() {
    assert_eq!(exit_code::SUCCESS, 0);
    assert_eq!(exit_code::AGENT_ERROR, 1);
    assert_eq!(exit_code::BUDGET_EXHAUSTED, 2);
    assert_eq!(exit_code::MAX_TURNS, 3);
    assert_eq!(exit_code::SANDBOX_VIOLATION, 4);
    assert_eq!(exit_code::NO_MODEL, 5);
}
