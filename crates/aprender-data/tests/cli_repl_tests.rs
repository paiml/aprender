#![allow(clippy::unwrap_used)] // Tests can use unwrap() for simplicity
#![allow(clippy::expect_used)] // Tests can use expect() for simplicity
#![allow(deprecated)] // assert_cmd::Command::cargo_bin deprecation
//! CLI REPL Integration Tests (ALIM-REPL-001 through ALIM-REPL-007)
//!
//! Test Approach: CLI integration tests with `assert_cmd`
//! Pattern: Matches bashrs REPL test coverage
//!
//! Quality targets:
//! - Integration tests: 10+ scenarios
//! - CLI interaction validated
//! - User experience verified

#![allow(non_snake_case)] // Test naming convention: test_ALIM_REPL_<req>_<scenario>

use assert_cmd::Command;
use predicates::prelude::*;

/// Helper function to create alimentar REPL command
fn alimentar_repl() -> Command {
    let mut cmd = Command::cargo_bin("alimentar").expect("Failed to find alimentar binary");
    cmd.arg("repl");
    cmd
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALIM-REPL-001: Stateful Session Tests
// ═══════════════════════════════════════════════════════════════════════════════

/// Test: ALIM-REPL-001-001 - REPL starts and accepts quit command
#[test]
fn test_ALIM_REPL_001_repl_starts_and_quits() {
    alimentar_repl()
        .write_stdin("quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("alimentar"))
        .stdout(predicate::str::contains("Goodbye!"));
}

/// Test: ALIM-REPL-001-002 - REPL handles empty input gracefully
#[test]
fn test_ALIM_REPL_001_repl_handles_empty_input() {
    alimentar_repl()
        .write_stdin("\n\nexit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("alimentar"));
}

/// Test: ALIM-REPL-001-003 - REPL handles EOF (Ctrl-D) gracefully
#[test]
fn test_ALIM_REPL_001_repl_handles_eof() {
    alimentar_repl()
        .write_stdin("")
        .assert()
        .success()
        .stdout(predicate::str::contains("alimentar"))
        .stdout(predicate::str::contains("Goodbye!"));
}

/// Test: ALIM-REPL-001-004 - REPL accepts exit command as alternative to quit
#[test]
fn test_ALIM_REPL_001_repl_accepts_exit() {
    alimentar_repl()
        .write_stdin("exit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("Goodbye!"));
}

/// Test: ALIM-REPL-001-005 - REPL accepts q as shorthand for quit
#[test]
fn test_ALIM_REPL_001_repl_accepts_q_shorthand() {
    alimentar_repl()
        .write_stdin("q\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("Goodbye!"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALIM-REPL-004: Contextual Help System Tests
// ═══════════════════════════════════════════════════════════════════════════════

/// Test: ALIM-REPL-004-001 - REPL shows help command
#[test]
fn test_ALIM_REPL_004_repl_shows_help() {
    alimentar_repl()
        .write_stdin("help\nquit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("help"))
        .stdout(predicate::str::contains("quit"));
}

/// Test: ALIM-REPL-004-002 - REPL shows help with ? shorthand
#[test]
fn test_ALIM_REPL_004_repl_shows_help_shorthand() {
    alimentar_repl()
        .write_stdin("?\nquit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("help"));
}

/// Test: ALIM-REPL-004-003 - REPL shows topic-specific help
#[test]
fn test_ALIM_REPL_004_repl_topic_help() {
    alimentar_repl()
        .write_stdin("help quality\nquit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("quality"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALIM-REPL-006: Reproducible Session Export Tests
// ═══════════════════════════════════════════════════════════════════════════════

/// Test: ALIM-REPL-006-001 - REPL shows history
#[test]
fn test_ALIM_REPL_006_repl_shows_history() {
    alimentar_repl()
        .write_stdin("info\nhistory\nquit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("info"));
}

/// Test: ALIM-REPL-006-002 - REPL exports history as script
#[test]
fn test_ALIM_REPL_006_repl_exports_history() {
    alimentar_repl()
        .write_stdin("info\nhistory --export\nquit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("#!/usr/bin/env bash"))
        .stdout(predicate::str::contains("alimentar session export"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// ALIM-REPL-007: Progressive Disclosure Commands Tests
// ═══════════════════════════════════════════════════════════════════════════════

/// Test: ALIM-REPL-007-001 - REPL lists datasets command
#[test]
fn test_ALIM_REPL_007_repl_datasets_command() {
    alimentar_repl()
        .write_stdin("datasets\nquit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("No datasets loaded"));
}

/// Test: ALIM-REPL-007-002 - REPL info without dataset shows error
#[test]
fn test_ALIM_REPL_007_repl_info_no_dataset() {
    alimentar_repl()
        .write_stdin("info\nquit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("No active dataset"));
}

/// Test: ALIM-REPL-007-003 - REPL schema without dataset shows error
#[test]
fn test_ALIM_REPL_007_repl_schema_no_dataset() {
    alimentar_repl()
        .write_stdin("schema\nquit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("No active dataset"));
}

/// Test: ALIM-REPL-007-004 - REPL head without dataset shows error
#[test]
fn test_ALIM_REPL_007_repl_head_no_dataset() {
    alimentar_repl()
        .write_stdin("head\nquit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("No active dataset"));
}

// ═══════════════════════════════════════════════════════════════════════════════
// Error Handling Tests
// ═══════════════════════════════════════════════════════════════════════════════

/// Test: Unknown command shows error
#[test]
fn test_ALIM_REPL_error_unknown_command() {
    alimentar_repl()
        .write_stdin("foobar\nquit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("Unknown command"));
}

/// Test: Load with missing path shows error
#[test]
fn test_ALIM_REPL_error_load_missing_path() {
    alimentar_repl()
        .write_stdin("load\nquit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("requires"));
}

/// Test: Quality subcommand required
#[test]
fn test_ALIM_REPL_error_quality_missing_subcommand() {
    alimentar_repl()
        .write_stdin("quality\nquit\n")
        .assert()
        .success()
        .stderr(predicate::str::contains("requires subcommand"));
}
