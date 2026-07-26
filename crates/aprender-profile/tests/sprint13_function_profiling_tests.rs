// Integration tests for --function-time flag (GitHub Issue #1)
#![allow(deprecated)] // suppress assert_cmd::Command::cargo_bin deprecation in tests
                      // Sprint 13-14: Function-level profiling with DWARF correlation

#[test]
fn test_function_time_flag_accepted() {
    // Test that --function-time flag is accepted by CLI
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    // Should execute echo successfully and show profiling message
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("exit_group") || stderr.contains("No function profiling data"));
}

#[test]
fn test_function_time_output_format() {
    // Test that function profiling output appears when --function-time is used
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("--").arg("echo").arg("hello");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should show either the profiling table header or "No function profiling data"
    // (depends on whether DWARF info is available for echo)
    assert!(
        stderr.contains("Function Profiling Summary")
            || stderr.contains("No function profiling data collected")
            || stderr.contains("═══")
    );
}

#[test]
fn test_function_time_with_filter() {
    // Test that --function-time works with filtering
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--function-time").arg("-e").arg("trace=write").arg("--").arg("echo").arg("hello");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Syscall traces go to stdout, profiling summary to stderr
    assert!(stdout.contains("write("));
    assert!(
        stderr.contains("No function profiling data")
            || stderr.contains("Function Profiling Summary")
    );
}

#[test]
fn test_function_time_without_flag_no_profiling() {
    // Test that without --function-time, no profiling summary appears
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Should NOT show function profiling messages
    assert!(!stderr.contains("Function Profiling Summary"));
    assert!(!stderr.contains("No function profiling data collected"));
}
