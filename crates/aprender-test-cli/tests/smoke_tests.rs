//! Smoke tests for probador CLI
//!
//! These tests verify basic CLI functionality works correctly.
//! Critical for a crate that replaces Playwright in Rust.

#![allow(deprecated)] // Allow deprecated Command::cargo_bin until assert_cmd is updated
#![allow(clippy::expect_used, clippy::unwrap_used)]

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

/// Get a command for the probador binary
fn probador() -> Command {
    Command::cargo_bin("probador").expect("probador binary should exist")
}

// ============================================================================
// Basic CLI Tests
// ============================================================================

#[test]
fn test_version_flag() {
    probador()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("1.0.0"));
}

#[test]
fn test_help_flag() {
    probador()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("WASM"))
        .stdout(predicate::str::contains("test"))
        .stdout(predicate::str::contains("playbook"));
}

#[test]
fn test_no_args_shows_help() {
    // Running with no args should show help or error gracefully
    probador().assert().failure(); // Requires a subcommand
}

// ============================================================================
// Subcommand Help Tests
// ============================================================================

#[test]
fn test_test_subcommand_help() {
    probador()
        .args(["test", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Run tests"));
}

#[test]
fn test_playbook_subcommand_help() {
    probador()
        .args(["playbook", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("playbook"))
        .stdout(predicate::str::contains("validate"));
}

#[test]
fn test_coverage_subcommand_help() {
    probador()
        .args(["coverage", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("coverage"));
}

#[test]
fn test_record_subcommand_help() {
    probador()
        .args(["record", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("record"));
}

#[test]
fn test_report_subcommand_help() {
    probador()
        .args(["report", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("report"));
}

#[test]
fn test_serve_subcommand_help() {
    probador()
        .args(["serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("serve"));
}

#[test]
fn test_watch_subcommand_help() {
    probador()
        .args(["watch", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("watch"));
}

#[test]
fn test_init_subcommand_help() {
    probador()
        .args(["init", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("init"));
}

#[test]
fn test_config_subcommand_help() {
    probador()
        .args(["config", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("config"));
}

// ============================================================================
// Playbook Validation Tests
// ============================================================================

#[test]
fn test_playbook_validate_valid_yaml() {
    let temp = TempDir::new().expect("create temp dir");
    let playbook_path = temp.path().join("test.yaml");

    let yaml = r#"
version: "1.0"
name: "Smoke Test"
machine:
  id: "smoke_test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "go"
      from: "start"
      to: "end"
      event: "proceed"
"#;

    fs::write(&playbook_path, yaml).expect("write playbook");

    probador()
        .args(["playbook", playbook_path.to_str().unwrap(), "--validate"])
        .assert()
        .success()
        .stdout(predicate::str::contains("smoke_test"));
}

#[test]
fn test_playbook_validate_invalid_yaml() {
    let temp = TempDir::new().expect("create temp dir");
    let playbook_path = temp.path().join("invalid.yaml");

    fs::write(&playbook_path, "not: valid: yaml: content").expect("write");

    probador()
        .args(["playbook", playbook_path.to_str().unwrap(), "--validate"])
        .assert()
        .failure();
}

#[test]
fn test_playbook_validate_missing_file() {
    probador()
        .args(["playbook", "/nonexistent/path.yaml", "--validate"])
        .assert()
        .failure();
}

#[test]
fn test_playbook_export_svg() {
    let temp = TempDir::new().expect("create temp dir");
    let playbook_path = temp.path().join("test.yaml");
    let output_path = temp.path().join("output.svg");

    let yaml = r#"
version: "1.0"
name: "SVG Export Test"
machine:
  id: "svg_test"
  initial: "a"
  states:
    a:
      id: "a"
    b:
      id: "b"
      final_state: true
  transitions:
    - id: "t1"
      from: "a"
      to: "b"
      event: "go"
"#;

    fs::write(&playbook_path, yaml).expect("write playbook");

    probador()
        .args([
            "playbook",
            playbook_path.to_str().unwrap(),
            "--export",
            "svg",
            "--export-output",
            output_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(output_path.exists(), "SVG file should be created");
    let content = fs::read_to_string(&output_path).expect("read svg");
    assert!(content.contains("<svg"), "Should contain SVG markup");
}

#[test]
fn test_playbook_export_dot() {
    let temp = TempDir::new().expect("create temp dir");
    let playbook_path = temp.path().join("test.yaml");
    let output_path = temp.path().join("output.dot");

    let yaml = r#"
version: "1.0"
name: "DOT Export Test"
machine:
  id: "dot_test"
  initial: "start"
  states:
    start:
      id: "start"
    end:
      id: "end"
      final_state: true
  transitions:
    - id: "t1"
      from: "start"
      to: "end"
      event: "finish"
"#;

    fs::write(&playbook_path, yaml).expect("write playbook");

    probador()
        .args([
            "playbook",
            playbook_path.to_str().unwrap(),
            "--export",
            "dot",
            "--export-output",
            output_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(output_path.exists(), "DOT file should be created");
    let content = fs::read_to_string(&output_path).expect("read dot");
    assert!(content.contains("digraph"), "Should contain DOT syntax");
}

#[test]
fn test_playbook_text_output() {
    let temp = TempDir::new().expect("create temp dir");
    let playbook_path = temp.path().join("test.yaml");

    let yaml = r#"
version: "1.0"
name: "Text Output Test"
machine:
  id: "text_test"
  initial: "s1"
  states:
    s1:
      id: "s1"
    s2:
      id: "s2"
      final_state: true
  transitions:
    - id: "t1"
      from: "s1"
      to: "s2"
      event: "next"
"#;

    fs::write(&playbook_path, yaml).expect("write playbook");

    probador()
        .args(["playbook", playbook_path.to_str().unwrap(), "--validate"])
        .assert()
        .success()
        .stdout(predicate::str::contains("text_test"))
        .stdout(predicate::str::contains("Valid: yes"));
}

// ============================================================================
// Mutation Testing Smoke Test
// ============================================================================

#[test]
fn test_playbook_mutate() {
    let temp = TempDir::new().expect("create temp dir");
    let playbook_path = temp.path().join("test.yaml");

    let yaml = r#"
version: "1.0"
name: "Mutation Test"
machine:
  id: "mutation_test"
  initial: "idle"
  states:
    idle:
      id: "idle"
    active:
      id: "active"
    done:
      id: "done"
      final_state: true
  transitions:
    - id: "start"
      from: "idle"
      to: "active"
      event: "begin"
    - id: "finish"
      from: "active"
      to: "done"
      event: "complete"
"#;

    fs::write(&playbook_path, yaml).expect("write playbook");

    probador()
        .args(["playbook", playbook_path.to_str().unwrap(), "--mutate"])
        .assert()
        .success()
        .stdout(predicate::str::contains("mutant"));
}

// ============================================================================
// Init Command Smoke Test
// ============================================================================

#[test]
fn test_init_runs_successfully() {
    let temp = TempDir::new().expect("create temp dir");

    // Init command should run without error
    probador()
        .current_dir(temp.path())
        .args(["init"])
        .assert()
        .success();
}

// ============================================================================
// Config Command Smoke Test
// ============================================================================

#[test]
fn test_config_runs_successfully() {
    // Config command should run without error
    probador().args(["config"]).assert().success();
}

// ============================================================================
// Verbosity Flags
// ============================================================================

#[test]
fn test_verbose_flag() {
    probador().args(["-v", "--help"]).assert().success();
}

#[test]
fn test_quiet_flag() {
    probador().args(["-q", "--help"]).assert().success();
}

// ============================================================================
// Error Handling
// ============================================================================

#[test]
fn test_invalid_subcommand() {
    probador()
        .arg("notacommand")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error"));
}

#[test]
fn test_invalid_flag() {
    probador().arg("--notaflag").assert().failure();
}
