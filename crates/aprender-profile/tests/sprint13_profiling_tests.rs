//! Integration tests for self-profiling feature (Sprint 13-14)
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
//!
//! Tests for --profile-self flag and ProfilingContext

#[test]
fn test_profile_self_flag_outputs_summary() {
    // Test that --profile-self produces profiling summary to stderr
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.args(["--profile-self", "--", "echo", "test"]);

    let output = cmd.output().expect("Failed to execute command");

    // Check that command succeeded
    assert!(output.status.success(), "Command failed with status: {}", output.status);

    // Convert stderr to string
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should contain profiling summary header
    assert!(
        stderr.contains("Renacer Self-Profiling Results"),
        "Missing profiling header. stderr:\n{}",
        stderr
    );

    // Should contain syscall count
    assert!(
        stderr.contains("Total syscalls traced:"),
        "Missing syscall count. stderr:\n{}",
        stderr
    );

    // Should contain wall time
    assert!(stderr.contains("Total wall time:"), "Missing wall time. stderr:\n{}", stderr);

    // Should contain user/kernel breakdown
    assert!(stderr.contains("Kernel time (ptrace):"), "Missing kernel time. stderr:\n{}", stderr);
    assert!(stderr.contains("User time (renacer):"), "Missing user time. stderr:\n{}", stderr);
}

#[test]
fn test_profile_self_without_flag_no_output() {
    // Test that without --profile-self, no profiling summary appears
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.args(["--", "echo", "test"]);

    let output = cmd.output().expect("Failed to execute command");

    // Check that command succeeded
    assert!(output.status.success(), "Command failed with status: {}", output.status);

    // Convert stderr to string
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT contain profiling summary
    assert!(
        !stderr.contains("Renacer Self-Profiling Results"),
        "Unexpected profiling output without flag. stderr:\n{}",
        stderr
    );
}

#[test]
fn test_profile_self_with_statistics_mode() {
    // Test that --profile-self works with -c (statistics mode)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.args(["--profile-self", "-c", "--", "echo", "test"]);

    let output = cmd.output().expect("Failed to execute command");

    // Check that command succeeded
    assert!(output.status.success(), "Command failed with status: {}", output.status);

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Statistics summary goes to stderr (matching strace behavior)
    assert!(
        stderr.contains("syscall") && stderr.contains("total"),
        "Missing statistics summary. stderr:\n{}",
        stderr
    );
    // Profiling summary also goes to stderr
    assert!(
        stderr.contains("Renacer Self-Profiling Results"),
        "Missing profiling summary. stderr:\n{}",
        stderr
    );
    // stdout should contain the command output
    let _ = stdout; // suppress unused variable warning
}

#[test]
fn test_profile_self_reports_nonzero_syscalls() {
    // Test that profiling reports > 0 syscalls for a real command
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.args(["--profile-self", "--", "ls", "/tmp"]);

    let output = cmd.output().expect("Failed to execute command");

    // Check that command succeeded
    assert!(output.status.success(), "Command failed with status: {}", output.status);

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should report at least 1 syscall (ls makes many syscalls)
    assert!(
        stderr.contains("Total syscalls traced:"),
        "Missing syscall count. stderr:\n{}",
        stderr
    );

    // Extract the syscall count (crude parsing, but works for tests)
    // Look for line like "Total syscalls traced:     42"
    if let Some(line) = stderr.lines().find(|l| l.contains("Total syscalls traced:")) {
        // Count should be > 0
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() >= 2 {
            let count_str = parts[1].trim();
            let count: u64 = count_str.parse().expect("Failed to parse syscall count");
            assert!(count > 0, "Syscall count should be > 0, got {}", count);
        }
    } else {
        panic!("Could not find syscall count line in stderr");
    }
}

#[test]
fn test_profile_self_with_filtering() {
    // Test that --profile-self works with syscall filtering
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.args(["--profile-self", "-e", "trace=open", "--", "echo", "test"]);

    let output = cmd.output().expect("Failed to execute command");

    // Check that command succeeded
    assert!(output.status.success(), "Command failed with status: {}", output.status);

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should contain profiling summary
    assert!(
        stderr.contains("Renacer Self-Profiling Results"),
        "Missing profiling summary. stderr:\n{}",
        stderr
    );
}
