// Sprint 15: Advanced Filtering - Negation Operator Tests
// RED Phase: These tests should fail until we implement negation support

/// Test basic negation of a single syscall
#[test]
fn test_negation_single_syscall() {
    // Test that trace=!close excludes only the close syscall
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=!close").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show other syscalls like write
    assert!(stdout.contains("write("), "Should show write syscall when excluding close");

    // Should NOT show close syscall
    assert!(!stdout.contains("close("), "Should not show close syscall when using trace=!close");
}

/// Test negation of multiple syscalls
#[test]
fn test_negation_multiple_syscalls() {
    // Test that trace=!open,!close excludes both syscalls
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=!open,!close").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show write
    assert!(stdout.contains("write("));

    // Should NOT show open or close
    assert!(!stdout.contains("open("));
    assert!(!stdout.contains("close("));
}

/// Test negation of a syscall class
#[test]
fn test_negation_syscall_class() {
    // Test that trace=!file excludes all file operations
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=!file").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should NOT show file syscalls
    assert!(!stdout.contains("openat("));
    assert!(!stdout.contains("read("));
    assert!(!stdout.contains("write("));
    assert!(!stdout.contains("close("));

    // Should show non-file syscalls (if any executed)
    // Note: echo might not execute many non-file syscalls, so we just verify no file ops
}

/// Test mixed positive and negative filters
#[test]
fn test_mixed_positive_negative() {
    // Test that trace=file,!close shows file operations except close
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=file,!close").arg("--").arg("cat").arg("/dev/null");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Should show openat (file operation)
    assert!(stdout.contains("openat("), "Should show openat when filtering file operations");

    // Should NOT show close even though it's a file operation
    assert!(!stdout.contains("close("), "Should not show close when explicitly excluded");
}

/// Test negation with statistics mode
#[test]
fn test_negation_with_statistics() {
    // Verify negation works with -c flag
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=!close").arg("-c").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Statistics output should not include close
    assert!(!stdout.contains("close"), "Statistics should not include excluded syscalls");
}

/// Test invalid negation syntax
#[test]
fn test_invalid_negation_syntax() {
    // Test that trace=! (empty negation) returns an error
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=!").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    // Should fail with invalid syntax
    assert!(!output.status.success());
}

/// Test negation of non-existent syscall
#[test]
fn test_negation_nonexistent_syscall() {
    // Test that trace=!nonexistent doesn't cause errors
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("aprender-profile");
    cmd.arg("-e").arg("trace=!nonexistent_syscall").arg("--").arg("echo").arg("test");

    let output = cmd.output().expect("test");
    // Should succeed and show all syscalls (nothing excluded)
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("write("));
}
