//! Sprint 1-2 MVP Tests - GREEN Phase Complete!
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
//!
//! Goal: renacer -- COMMAND works and traces write syscall only

use predicates::prelude::*;

#[test]
fn test_cli_requires_command() {
    // Test that running without -- COMMAND or -p PID fails with helpful error
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Must specify either -p PID or command"));
}

#[test]
fn test_cli_help() {
    // Test that --help works
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--help").assert().success().stdout(predicate::str::contains("Usage"));
}

#[test]
fn test_trace_simple_echo() {
    // Test tracing echo command (should show write syscall)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("echo")
        .arg("Hello")
        .assert()
        .success()
        .stdout(predicate::str::contains("write("));
}

#[test]
fn test_trace_shows_write_syscall() {
    // Test that write syscall details are shown
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("echo")
        .arg("Test")
        .assert()
        .success()
        .stdout(predicate::str::contains("write(")) // Sprint 3-4: syscall name shown
        .stdout(predicate::str::contains("5")); // count or return value
}

#[test]
fn test_trace_exit_code_preserved() {
    // Test that traced program's exit code is preserved
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--").arg("sh").arg("-c").arg("exit 42").assert().code(42);
}
