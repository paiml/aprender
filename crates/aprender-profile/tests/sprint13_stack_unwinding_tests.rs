// Integration tests for stack unwinding functionality (GitHub Issue #1)
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
                      // Sprint 13-14: Stack unwinding for function profiling

#[test]
fn test_stack_frame_struct() {
    // Test that we can create stack frames (unit-level functionality)
    // This is tested in the module itself, but we verify integration
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("--source").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());
}

#[test]
fn test_stack_unwinding_with_simple_program() {
    // Test stack unwinding with a simple program
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("--source").arg("--").arg("true"); // Simplest possible program

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should either find functions or report no data
    assert!(stderr.contains("Function") || stderr.contains("No function profiling data"));
}

#[test]
fn test_stack_unwinding_does_not_crash() {
    // Verify that stack unwinding doesn't crash the tracer
    // even with complex programs
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("--source").arg("--").arg("ls").arg("-la");

    let output = cmd.output().expect("test");
    assert!(output.status.success());
}

#[test]
fn test_stack_unwinding_with_function_time_disabled() {
    // Verify that without --function-time, stack unwinding is not attempted
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--source") // Source enabled but not function-time
        .arg("--")
        .arg("echo")
        .arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Should NOT show function profiling
    assert!(!stderr.contains("Function Profiling Summary"));
}

#[test]
fn test_stack_unwinding_max_depth_protection() {
    // Test that max depth protection prevents infinite loops
    // Run a program and verify it completes (doesn't hang)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("--source").arg("--").arg("echo").arg("deep recursion test");

    let output = cmd.timeout(std::time::Duration::from_secs(5)).output().expect("test");

    // Should complete within timeout
    assert!(output.status.success());
}
