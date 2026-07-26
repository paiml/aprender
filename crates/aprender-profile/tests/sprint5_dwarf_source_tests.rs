//! Sprint 5-6: DWARF Source Correlation Tests - RED Phase
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
//!
//! Goal: Map instruction pointers to source file:line using DWARF debug info
//!
//! These tests will FAIL initially, then we implement to make them PASS

use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command as StdCommand;
use tempfile::TempDir;

/// Helper: Compile a simple Rust program with debug info
fn compile_test_program(code: &str, opt_level: u8) -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().expect("test");
    let src_file = temp_dir.path().join("test.rs");
    let bin_file = temp_dir.path().join("test_bin");

    fs::write(&src_file, code).expect("test");

    StdCommand::new("rustc")
        .arg(&src_file)
        .arg("-o")
        .arg(&bin_file)
        .arg("-g") // Include debug info
        .arg("-C")
        .arg(format!("opt-level={}", opt_level))
        .status()
        .expect("Failed to compile test program");

    (temp_dir, bin_file)
}

#[test]
#[ignore] // TODO(v0.2.0): Requires stack unwinding to attribute syscalls from libc back to user code
fn test_dwarf_shows_source_location() {
    // Test that syscalls are annotated with source file:line
    // NOTE: This requires walking the call stack to find user code frames
    // Current implementation only looks at IP which points to libc during syscalls
    let code = r#"
fn main() {
    println!("test");
}
"#;

    let (_temp_dir, bin_file) = compile_test_program(code, 0);

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--source") // New flag to enable source correlation
        .arg("--")
        .arg(&bin_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("test.rs:3")); // Line 3: println!
}

#[test]
#[ignore] // TODO(v0.2.0): Requires stack unwinding - see test_dwarf_shows_source_location
fn test_dwarf_opt_level_1_accuracy() {
    // Test DWARF accuracy with -C opt-level=1
    let code = r#"
fn main() {
    let msg = String::from("hello");
    println!("{}", msg);
}
"#;

    let (_temp_dir, bin_file) = compile_test_program(code, 1);

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--source")
        .arg("--")
        .arg(&bin_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("test.rs:4")); // Line 4: println!
}

#[test]
#[ignore] // TODO(v0.2.0): Requires stack unwinding - see test_dwarf_shows_source_location
fn test_dwarf_shows_function_name() {
    // Test that function names are shown from DWARF
    let code = r#"
fn do_work() {
    println!("working");
}

fn main() {
    do_work();
}
"#;

    let (_temp_dir, bin_file) = compile_test_program(code, 0);

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--source")
        .arg("--")
        .arg(&bin_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("do_work")) // Function name
        .stdout(predicate::str::contains("test.rs:3")); // Line in do_work
}

#[test]
fn test_source_flag_disabled_by_default() {
    // Test that source correlation is off by default (no performance impact)
    let code = r#"
fn main() {
    println!("test");
}
"#;

    let (_temp_dir, bin_file) = compile_test_program(code, 0);

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--")
        .arg(&bin_file)
        .assert()
        .success()
        .stdout(predicate::str::contains("test.rs:3").not()); // NO source info without --source
}

#[test]
fn test_dwarf_no_debug_info_graceful_fallback() {
    // Test graceful handling when no debug info available
    let temp_dir = TempDir::new().expect("test");
    let src_file = temp_dir.path().join("test.rs");
    let bin_file = temp_dir.path().join("test_bin");

    fs::write(&src_file, "fn main() { println!(\"test\"); }").expect("test");

    // Compile WITHOUT debug info (-g)
    StdCommand::new("rustc")
        .arg(&src_file)
        .arg("-o")
        .arg(&bin_file)
        .arg("-C")
        .arg("strip=symbols") // Strip debug info
        .status()
        .expect("test");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--source").arg("--").arg(&bin_file).assert().success(); // Should not crash
                                                                     // Output should just be normal (no source info)
}
