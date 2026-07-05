//! Sprint 9-10: Syscall filtering tests
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
//!
//! Test -e trace= expression filtering

use predicates::prelude::*;

#[test]
fn test_filter_single_syscall() {
    // Test filtering to a single syscall
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e")
        .arg("trace=write")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stdout(predicate::str::contains("write("))
        .stdout(predicate::str::contains("brk(").not()); // brk should be filtered out
}

#[test]
fn test_filter_multiple_syscalls() {
    // Test filtering to multiple specific syscalls
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e")
        .arg("trace=write,brk")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stdout(predicate::str::contains("write("))
        .stdout(predicate::str::contains("brk("));
}

#[test]
fn test_filter_file_class() {
    // Test filtering to file-related syscalls
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e")
        .arg("trace=file")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stdout(predicate::str::contains("openat("))
        .stdout(predicate::str::contains("brk(").not()); // brk should be filtered out
}

#[test]
fn test_no_filter_shows_all() {
    // Test that without filter, all syscalls are shown
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stdout(predicate::str::contains("write("))
        .stdout(predicate::str::contains("brk("))
        .stdout(predicate::str::contains("mmap("));
}

#[test]
fn test_filter_network_class() {
    // Test network filter class (won't match echo, but should parse correctly)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=network").arg("--").arg("echo").arg("test").assert().success();
    // echo doesn't make network calls, so output should be minimal/empty for syscalls
}

#[test]
fn test_filter_mixed_class_and_syscall() {
    // Test combining a class with a specific syscall
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e")
        .arg("trace=file,brk")
        .arg("--")
        .arg("echo")
        .arg("test")
        .assert()
        .success()
        .stdout(predicate::str::contains("openat(")) // from file class
        .stdout(predicate::str::contains("brk(")); // specific syscall
}
