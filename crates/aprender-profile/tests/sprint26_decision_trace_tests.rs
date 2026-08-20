//! Sprint 26: Transpiler Decision-Time Tracing Integration Tests
//!
//! Tests for capturing and parsing transpiler decision traces via stderr

#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests

use assert_cmd::Command;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_decision_trace_capture_basic() {
    // RED phase: This test will fail until we implement stderr capture
    //
    // Test that renacer captures decision traces written to stderr
    // Expected: DecisionTracer should parse `[DECISION]` lines from stderr

    let temp_dir = TempDir::new().expect("test");
    let test_bin = temp_dir.path().join("decision_writer");

    // Create a simple test program that writes decision traces to stderr
    let test_program = r#"
fn main() {
    eprintln!("[DECISION] test_category::test_decision input={{\"test_key\":\"test_value\"}}");
    eprintln!("[RESULT] test_category::test_decision result={{\"output\":42}}");
    println!("Normal stdout output");
}
"#;

    // Compile the test program
    let src_file = temp_dir.path().join("decision_writer.rs");
    fs::write(&src_file, test_program).expect("test");

    let compile_status = std::process::Command::new("rustc")
        .arg(&src_file)
        .arg("-o")
        .arg(&test_bin)
        .status()
        .expect("Failed to compile test program");

    assert!(compile_status.success(), "Failed to compile test program");

    // Run renacer with --trace-transpiler-decisions flag
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--trace-transpiler-decisions")
        .arg("-c") // Use statistics mode to suppress noise
        .arg("--")
        .arg(&test_bin)
        .output()
        .expect("Failed to execute renacer");

    // Verify renacer runs successfully
    assert!(
        output.status.success(),
        "renacer should run successfully: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Sprint 26: Verify decision traces are captured and displayed
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("test_category::test_decision"),
        "Output should contain decision name: {}",
        stdout
    );
    assert!(stdout.contains("test_key"), "Output should contain input key: {}", stdout);
}

#[test]
fn test_decision_trace_disabled_by_default() {
    // Verify that decision tracing is disabled by default (no overhead)
    let temp_dir = TempDir::new().expect("test");
    let test_bin = temp_dir.path().join("decision_writer");

    let test_program = r#"
fn main() {
    eprintln!("[DECISION] should_not_be_captured input={{}}");
    println!("stdout");
}
"#;

    let src_file = temp_dir.path().join("decision_writer.rs");
    fs::write(&src_file, test_program).expect("test");

    std::process::Command::new("rustc")
        .arg(&src_file)
        .arg("-o")
        .arg(&test_bin)
        .status()
        .expect("Failed to compile");

    // Run without --trace-transpiler-decisions flag
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd.arg("-c").arg("--").arg(&test_bin).output().expect("Failed to execute");

    assert!(output.status.success());

    // Decision traces should NOT appear in output when flag is disabled
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("should_not_be_captured"));
}

#[test]
fn test_decision_trace_multiple_decisions() {
    // Test capturing multiple decision traces in sequence
    let temp_dir = TempDir::new().expect("test");
    let test_bin = temp_dir.path().join("multi_decision");

    let test_program = r#"
fn main() {
    eprintln!("[DECISION] flow::branch1 input={{\"condition\":true}}");
    eprintln!("[RESULT] flow::branch1 result={{\"taken\":\"left\"}}");
    eprintln!("[DECISION] flow::branch2 input={{\"condition\":false}}");
    eprintln!("[RESULT] flow::branch2 result={{\"taken\":\"right\"}}");
}
"#;

    let src_file = temp_dir.path().join("multi_decision.rs");
    fs::write(&src_file, test_program).expect("test");

    std::process::Command::new("rustc")
        .arg(&src_file)
        .arg("-o")
        .arg(&test_bin)
        .status()
        .expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--trace-transpiler-decisions")
        .arg("-c")
        .arg("--")
        .arg(&test_bin)
        .output()
        .expect("test");

    assert!(output.status.success());

    // Sprint 26: Verify all decision traces are captured
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("flow::branch1"), "Output should contain first decision: {}", stdout);
    assert!(stdout.contains("flow::branch2"), "Output should contain second decision: {}", stdout);
    assert!(stdout.contains("condition"), "Output should contain input parameters: {}", stdout);
}

#[test]
fn test_decision_trace_ignores_non_decision_stderr() {
    // Verify that normal stderr output is not parsed as decisions
    let temp_dir = TempDir::new().expect("test");
    let test_bin = temp_dir.path().join("mixed_stderr");

    let test_program = r#"
fn main() {
    eprintln!("Normal error message");
    eprintln!("[DECISION] valid::decision input={{}}");
    eprintln!("Another normal message");
}
"#;

    let src_file = temp_dir.path().join("mixed_stderr.rs");
    fs::write(&src_file, test_program).expect("test");

    std::process::Command::new("rustc")
        .arg(&src_file)
        .arg("-o")
        .arg(&test_bin)
        .status()
        .expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    let output = cmd
        .arg("--trace-transpiler-decisions")
        .arg("-c")
        .arg("--")
        .arg(&test_bin)
        .output()
        .expect("test");

    assert!(output.status.success());

    // Sprint 26: Verify only valid decision traces are captured
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("valid::decision"),
        "Output should contain valid decision trace: {}",
        stdout
    );
    assert!(
        !stdout.contains("Normal error message"),
        "Output should not contain non-decision stderr: {}",
        stdout
    );
    assert!(
        !stdout.contains("Another normal message"),
        "Output should not contain non-decision stderr: {}",
        stdout
    );
}
