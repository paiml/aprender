//! Tests for tracing instrumentation feature
//!
//! EXTREME TDD: RED phase tests for --debug flag and tracing output

use std::process::Command;

fn get_binary_path() -> String {
    env!("CARGO_BIN_EXE_renacer").to_string()
}

#[test]
fn test_debug_flag_accepted() {
    // --debug flag should be accepted without "unknown argument" error
    let output = Command::new(get_binary_path())
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should not contain "unknown argument" or similar error
    assert!(
        !stderr.contains("unknown") && !stderr.contains("unexpected"),
        "Debug flag was not accepted: {}",
        stderr
    );
}

#[test]
fn test_debug_output_to_stderr() {
    // Debug mode should produce tracing output on stderr
    let output = Command::new(get_binary_path())
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should contain tracing level markers
    assert!(
        stderr.contains("TRACE") || stderr.contains("DEBUG") || stderr.contains("INFO"),
        "No tracing output found in stderr: {}",
        stderr
    );
}

#[test]
fn test_tracing_shows_ptrace_operations() {
    // Debug output should show ptrace/waitpid operations
    let output = Command::new(get_binary_path())
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should show key operations
    assert!(
        stderr.contains("waitpid") || stderr.contains("ptrace") || stderr.contains("syscall"),
        "No ptrace operation tracing found: {}",
        stderr
    );
}

#[test]
fn test_tracing_shows_child_pid() {
    // Debug output should show child process PID
    let output = Command::new(get_binary_path())
        .args(["--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should show PID information
    assert!(
        stderr.contains("pid=") || stderr.contains("child") || stderr.contains("Pid("),
        "No PID information in tracing output: {}",
        stderr
    );
}

#[test]
fn test_normal_mode_no_debug_output() {
    // Normal mode (without --debug) should NOT produce tracing output
    let output = Command::new(get_binary_path())
        .args(["--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT contain tracing markers
    assert!(
        !stderr.contains("TRACE") && !stderr.contains("DEBUG"),
        "Normal mode should not have debug output: {}",
        stderr
    );
}

#[test]
fn test_debug_with_timeout_shows_last_operation() {
    // Even if tracer hangs, we should see the last operation before hang
    // This test uses timeout to kill a potentially hanging process
    let output = Command::new("timeout")
        .args(["2", &get_binary_path(), "--debug", "--", "/bin/true"])
        .output()
        .expect("Failed to execute");

    let stderr = String::from_utf8_lossy(&output.stderr);

    // If we get output, it should show operations
    // If process completed normally, check for tracing
    // If process timed out, we should still see some tracing
    if !stderr.is_empty() {
        assert!(
            stderr.contains("TRACE")
                || stderr.contains("DEBUG")
                || stderr.contains("INFO")
                || stderr.contains("starting"),
            "Should have some tracing even with timeout: {}",
            stderr
        );
    }
}
