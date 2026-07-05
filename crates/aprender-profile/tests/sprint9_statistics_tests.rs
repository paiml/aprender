//! Sprint 9-10: Statistics mode tests
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
//!
//! Test -c flag for syscall statistics summary

use predicates::prelude::*;

#[test]
fn test_statistics_mode_shows_summary() {
    // Test that -c shows statistics table
    // Statistics output goes to stderr (matching strace behavior)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stderr(predicate::str::contains("% time"))
        .stderr(predicate::str::contains("syscall"))
        .stderr(predicate::str::contains("total"));
}

#[test]
fn test_statistics_mode_suppresses_individual_calls() {
    // Test that -c does not show individual syscall lines
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stdout(predicate::str::contains("write(").not()); // Should NOT show individual calls
}

#[test]
fn test_statistics_with_filter() {
    // Test that -c works with -e trace= filtering
    // Statistics output goes to stderr (matching strace behavior)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c")
        .arg("-e")
        .arg("trace=write,brk")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stderr(predicate::str::contains("write"))
        .stderr(predicate::str::contains("brk"));
}

#[test]
fn test_statistics_shows_call_counts() {
    // Test that statistics show call counts
    // Statistics output goes to stderr (matching strace behavior)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stderr(predicate::str::is_match(r"\d+\s+write").expect("test")); // Number before "write"
}
