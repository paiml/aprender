// Sprint 16: Advanced Filtering - Regex Pattern Tests
// RED Phase: These tests should fail until we implement regex support

/// Test basic regex pattern matching syscalls starting with "open"
#[test]
fn test_regex_prefix_pattern() {
    // Test that trace=/^open.*/ matches openat but not close or write
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/^open.*/").arg("--").arg("cat").arg("/dev/null");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show openat (matches /^open.*/)
    assert!(stdout.contains("openat("), "Should show openat syscall when using regex /^open.*/");

    // Should NOT show close or write (don't match pattern)
    assert!(!stdout.contains("close("), "Should not show close syscall with /^open.*/ pattern");
    assert!(!stdout.contains("write("), "Should not show write syscall with /^open.*/ pattern");
}

/// Test regex pattern matching syscalls ending with "at"
#[test]
fn test_regex_suffix_pattern() {
    // Test that trace=/.*at$/ matches openat, newfstatat but not open, close
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/.*at$/").arg("--").arg("cat").arg("/dev/null");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show openat (ends with "at")
    assert!(stdout.contains("openat("), "Should show openat syscall when using regex /.*at$/");

    // Should NOT show close (doesn't end with "at")
    assert!(!stdout.contains("close("), "Should not show close syscall with /.*at$/ pattern");
}

/// Test regex OR pattern
#[test]
fn test_regex_or_pattern() {
    // Test that trace=/read|write/ matches both read and write
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/read|write/").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show write (matches /read|write/)
    assert!(stdout.contains("write("), "Should show write syscall when using regex /read|write/");

    // Should NOT show other syscalls like openat
    assert!(
        !stdout.contains("openat("),
        "Should not show openat syscall with /read|write/ pattern"
    );
}

/// Test invalid regex returns error
#[test]
fn test_invalid_regex_error() {
    // Test that trace=/[invalid/ returns an error
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/[invalid/").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    // Should fail with invalid regex
    assert!(!output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("regex") || stderr.contains("invalid"),
        "Error message should mention regex or invalid syntax"
    );
}

/// Test mixed regex and literal syscalls
#[test]
fn test_mixed_regex_and_literal() {
    // Test that trace=/^open.*/,close works (regex + literal)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/^open.*/,close").arg("--").arg("cat").arg("/dev/null");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show openat (matches regex)
    assert!(stdout.contains("openat("), "Should show openat from regex pattern");

    // Should show close (literal match)
    assert!(stdout.contains("close("), "Should show close from literal match");

    // Should NOT show read (not in filter)
    assert!(!stdout.contains("read("), "Should not show read when not in filter");
}

/// Test regex with negation operator (Sprint 15 + Sprint 16)
#[test]
fn test_regex_with_negation() {
    // Test that trace=/^open.*/,!/openat/ shows open* except openat
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/^open.*/,!/openat/").arg("--").arg("cat").arg("/dev/null");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should NOT show openat (excluded by negation)
    assert!(!stdout.contains("openat("), "Should not show openat when explicitly excluded");

    // Note: cat /dev/null primarily uses openat, so we may not see other open* syscalls
    // This test validates that exclusion works even if nothing else matches
}

/// Test regex with statistics mode
#[test]
fn test_regex_with_statistics() {
    // Verify regex works with -c flag
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/^open.*/").arg("-c").arg("--").arg("cat").arg("/dev/null");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stderr = String::from_utf8_lossy(&output.stderr);

    // Statistics output goes to stderr (matching strace behavior)
    assert!(
        stderr.contains("openat") || stderr.contains("% time"),
        "Statistics should show matched syscalls"
    );
}

/// Test case-insensitive regex
#[test]
fn test_regex_case_insensitive() {
    // Test that trace=/(?i)OPEN.*/ matches openat (case-insensitive)
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/(?i)OPEN.*/").arg("--").arg("cat").arg("/dev/null");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show openat (case-insensitive match)
    assert!(stdout.contains("openat("), "Should show openat with case-insensitive regex");
}

/// Test empty regex pattern
#[test]
fn test_empty_regex_pattern() {
    // Test that trace=/()/ is handled gracefully
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=/()/").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    // Should either succeed with no matches or provide clear error
    // We'll validate behavior based on implementation decision
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Valid to either show nothing or error - test ensures no panic
    assert!(output.status.success() || !stderr.is_empty(), "Should handle empty regex gracefully");
}
