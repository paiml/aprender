//! Integration tests for --format json output (Sprint 9-10)
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests

use predicates::prelude::*;

#[test]
fn test_json_output_valid_format() {
    // Test that --format json produces valid JSON
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("--").arg("echo").arg("test");

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("\"version\":"))
        .stdout(predicate::str::contains("\"format\": \"renacer-json-v1\""))
        .stdout(predicate::str::contains("\"syscalls\":"))
        .stdout(predicate::str::contains("\"summary\":"));
}

#[test]
fn test_json_output_parses() {
    // Test that JSON output is actually valid JSON
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    let stdout = String::from_utf8_lossy(&output.stdout);

    // Find JSON output (skip the "test" line from echo)
    let json_start = stdout.find('{').expect("test");
    let json_str = &stdout[json_start..];

    // Should parse as valid JSON
    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("test");
    assert_eq!(parsed["format"], "renacer-json-v1");
    assert!(parsed["syscalls"].is_array());
    assert!(parsed["summary"].is_object());
}

#[test]
fn test_json_with_timing() {
    // Test that -T works with JSON output
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("-T").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_start = stdout.find('{').expect("test");
    let json_str = &stdout[json_start..];

    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("test");

    // Should have total_time_us in summary
    assert!(parsed["summary"]["total_time_us"].is_number());

    // At least some syscalls should have duration_us
    let syscalls = parsed["syscalls"].as_array().expect("test");
    let has_duration = syscalls.iter().any(|s| s["duration_us"].is_number());
    assert!(has_duration, "Expected at least one syscall with duration_us");
}

#[test]
fn test_json_with_filtering() {
    // Test that filtering works with JSON output
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("-e").arg("trace=write").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_start = stdout.find('{').expect("test");
    let json_str = &stdout[json_start..];

    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("test");
    let syscalls = parsed["syscalls"].as_array().expect("test");

    // All syscalls should be "write"
    for syscall in syscalls {
        assert_eq!(syscall["name"], "write");
    }
}

#[test]
fn test_json_summary_fields() {
    // Test that summary contains expected fields
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--format").arg("json").arg("--").arg("true"); // Simple command

    let output = cmd.output().expect("test");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json_start = stdout.find('{').expect("test");
    let json_str = &stdout[json_start..];

    let parsed: serde_json::Value = serde_json::from_str(json_str).expect("test");
    let summary = &parsed["summary"];

    assert!(summary["total_syscalls"].is_number());
    assert!(summary["exit_code"].is_number());
    assert_eq!(summary["exit_code"], 0);
}
