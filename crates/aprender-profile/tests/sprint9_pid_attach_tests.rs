//! Integration tests for -p PID attach (Sprint 9-10)
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests

use predicates::prelude::*;

#[test]
fn test_pid_flag_exists() {
    // Test that -p flag is recognized
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--help");

    cmd.assert().success().stdout(predicate::str::contains("-p, --pid <PID>"));
}

#[test]
fn test_pid_and_command_mutual_exclusion() {
    // Test that -p and command cannot both be specified
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-p").arg("1234").arg("--").arg("echo").arg("test");

    cmd.assert().failure().stderr(predicate::str::contains("Cannot specify both"));
}

#[test]
fn test_invalid_pid() {
    // Test that invalid PID format is rejected
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-p").arg("not_a_number");

    cmd.assert().failure().stderr(predicate::str::contains("invalid digit found in string"));
}

#[test]
fn test_nonexistent_pid() {
    // Test that non-existent PID is handled gracefully
    // Use PID 99999999 which is very unlikely to exist
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-p").arg("99999999");

    cmd.assert().failure().stderr(predicate::str::contains("Failed to attach"));
}

#[test]
fn test_no_command_no_pid() {
    // Test that either command or PID must be specified
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Must specify either -p PID or command"));
}

// Note: Testing actual PID attachment requires special permissions
// (CAP_SYS_PTRACE or ptrace_scope=0) and a cooperating process.
// The above tests verify the CLI interface and error handling.
// Manual testing with a helper program is needed for full validation.
