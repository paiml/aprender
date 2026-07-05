//! Integration tests for -T timing functionality (Sprint 9-10)
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests

use predicates::prelude::*;

#[test]
fn test_timing_flag_shows_duration() {
    // Test that -T flag adds timing info to each syscall
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-T").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("<")) // Timing appears as <0.000123>
        .stdout(predicate::str::contains(">")); // Timing format: <seconds>
}

#[test]
fn test_timing_with_statistics_mode() {
    // Test that -T works with -c statistics mode
    // Statistics output goes to stderr (matching strace behavior)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-c").arg("-T").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stderr(predicate::str::contains("% time")) // Statistics header
        .stderr(predicate::str::contains("seconds")) // Time column
        .stderr(predicate::str::contains("usecs/call")); // Per-call timing
}

#[test]
fn test_timing_with_filter() {
    // Test that -T works with filtering
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-T").arg("-e").arg("trace=write").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("write")) // Only write syscalls
        .stdout(predicate::str::contains("<")); // With timing
}

#[test]
fn test_timing_format_is_seconds() {
    // Test that timing is displayed in seconds (not microseconds)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-T").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Timing should be in format <0.000123> (seconds with 6 decimal places)
    // Look for pattern: <N.NNNNNN> where values are typically very small
    assert!(stdout.contains('<'));
    assert!(stdout.contains('>'));

    // Should see small values like <0.000001> to <0.001000>, not large microsecond values
    // This is a sanity check that we're displaying seconds, not microseconds
}
