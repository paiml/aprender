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
    // even with complex programs.
    // #5018: `ls -la` lists a directory this test owns. It used to list the
    // working directory (the crate dir), which CI shares with everything else
    // on the checkout, so its contents were an input from outside the test.
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join("file.txt"), b"fixed contents\n").expect("write file");
    std::fs::create_dir(dir.path().join("subdir")).expect("create subdir");

    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("--source").arg("--").arg("ls").arg("-la").arg(dir.path());

    let output = cmd.output().expect("test");
    // On failure, say which exit it was: the tracee's own status, the tracer's
    // "Error: ..." (exit 1), a panic (101) or a signal. The bare assert that
    // failed in #5018 said none of these.
    let stderr = String::from_utf8_lossy(&output.stderr);
    let lines: Vec<&str> = stderr.lines().collect();
    let tail = lines[lines.len().saturating_sub(20)..].join("\n");
    assert!(output.status.success(), "{}; last 20 stderr lines:\n{tail}", output.status);
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
