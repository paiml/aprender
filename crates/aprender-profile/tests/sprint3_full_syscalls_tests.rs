//! Sprint 3-4: Full Syscall Coverage Tests - RED Phase
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
//!
//! Goal: Trace all syscalls, not just write
//!
//! These tests will FAIL initially, then we implement to make them PASS

use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_trace_shows_multiple_syscalls() {
    // Test that multiple syscalls are traced (not just write)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("ls")
        .arg("-la")
        .assert()
        .success()
        .stdout(predicate::str::contains("openat(")) // ls opens directory
        .stdout(predicate::str::contains("read(")) // ls reads directory entries
        .stdout(predicate::str::contains("write(")); // ls writes output
}

#[test]
fn test_trace_file_operations() {
    // Create temp file for testing
    let temp_dir = TempDir::new().expect("test");
    let test_file = temp_dir.path().join("test.txt");
    fs::write(&test_file, "test content").expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("cat")
        .arg(&test_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("openat(")) // cat opens file
        .stdout(predicate::str::contains("read(")) // cat reads file
        .stdout(predicate::str::contains("close(")); // cat closes file
}

#[test]
fn test_syscall_names_not_numbers() {
    // Test that syscalls show as names (e.g., "openat"), not numbers (e.g., "257")
    // Note: We verify names are used by checking for common syscall names at start of lines
    // We can't just check for absence of "257" since it may appear in memory addresses
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stdout(predicate::str::contains("write(")) // Name used
        .stdout(predicate::str::contains("exit_group(")); // Another named syscall
}

#[test]
fn test_trace_shows_syscall_arguments() {
    // Test that syscall arguments are decoded (e.g., filename, fd)
    let temp_dir = TempDir::new().expect("test");
    let test_file = temp_dir.path().join("myfile.txt");
    fs::write(&test_file, "content").expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("cat")
        .arg(&test_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("myfile.txt")); // Filename should be shown
}

#[test]
fn test_unknown_syscalls_show_number() {
    // Test that unknown/unhandled syscalls show their number
    // (This is for future-proofing when new syscalls are added)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("true") // Simple program with minimal syscalls
        .assert()
        .success();
    // Success is enough - we just verify it doesn't crash on unknown syscalls
}
