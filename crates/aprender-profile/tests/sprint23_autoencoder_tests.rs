// Sprint 23: Deep Learning - Autoencoder Anomaly Detection
// EXTREME TDD: RED phase - Integration tests for Autoencoder-based anomaly detection
//
// Goal: Implement simple linear Autoencoder for unsupervised anomaly detection

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Test 1: Verify --dl-anomaly flag is accepted
// ============================================================================

#[test]
fn test_dl_anomaly_flag_accepted() {
    // Test that --dl-anomaly flag is accepted
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--dl-anomaly").arg("--").arg("echo").arg("test");

    // Should not error on flag parsing
    cmd.assert().success();
}

// ============================================================================
// Test 2: --dl-anomaly works with statistics mode
// ============================================================================

#[test]
fn test_dl_anomaly_with_statistics() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--dl-anomaly").arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 3: --dl-anomaly detects anomalous syscalls
// ============================================================================

#[test]
fn test_dl_anomaly_detects_anomalies() {
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("dl_anomaly_test");

    // Create program with normal and anomalous syscalls
    let source = r#"
#include <unistd.h>
#include <time.h>
int main() {
    // Normal fast writes (establish baseline)
    for (int i = 0; i < 50; i++) {
        write(1, "x", 1);
    }

    // Anomalous slow write
    struct timespec ts = {0, 100000000};  // 100ms
    nanosleep(&ts, NULL);
    write(1, "slow", 4);

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("dl_anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--dl-anomaly").arg("--").arg(&test_program);

    // Should detect anomalies in output
    cmd.assert().success().stderr(
        predicate::str::contains("Autoencoder Anomaly Detection")
            .or(predicate::str::contains("Reconstruction Error")),
    );
}

// ============================================================================
// Test 4: Reconstruction error threshold configuration
// ============================================================================

#[test]
fn test_dl_threshold_configuration() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c")
        .arg("--dl-anomaly")
        .arg("--dl-threshold")
        .arg("2.0")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 5: Autoencoder works with JSON output
// ============================================================================

#[test]
fn test_dl_anomaly_json_export() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--format").arg("json").arg("--dl-anomaly").arg("--").arg("echo").arg("test");

    cmd.assert().success().stdout(
        predicate::str::contains("autoencoder").or(predicate::str::contains("dl_analysis")),
    );
}

// ============================================================================
// Test 6: Autoencoder with filtering
// ============================================================================

#[test]
fn test_dl_anomaly_with_filtering() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--dl-anomaly").arg("-e").arg("trace=file").arg("--").arg("ls").arg("-la");

    cmd.assert().success();
}

// ============================================================================
// Test 7: Minimum samples requirement
// ============================================================================

#[test]
fn test_dl_anomaly_minimum_samples() {
    // Test that Autoencoder requires minimum samples
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("min_samples_test");

    let source = r#"
#include <unistd.h>
int main() {
    // Only 2 syscalls - insufficient for meaningful autoencoder
    write(1, "x", 1);
    write(1, "y", 1);
    return 0;
}
"#;
    let source_file = tmp_dir.path().join("min_samples.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("test");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--dl-anomaly").arg("--").arg(&test_program);

    // Should handle gracefully (no panic)
    cmd.assert().success();
}

// ============================================================================
// Test 8: Backward compatibility without --dl-anomaly
// ============================================================================

#[test]
fn test_backward_compatibility_without_dl_anomaly() {
    // Ensure existing functionality works without --dl-anomaly flag
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 9: Autoencoder with timing mode
// ============================================================================

#[test]
fn test_dl_anomaly_with_timing() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("-T").arg("--dl-anomaly").arg("--").arg("echo").arg("hello");

    cmd.assert().success();
}

// ============================================================================
// Test 10: Compare Autoencoder with other ML methods
// ============================================================================

#[test]
fn test_dl_anomaly_with_other_ml() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c")
        .arg("--dl-anomaly")
        .arg("--ml-anomaly") // Also enable KMeans
        .arg("--ml-outliers") // Also enable Isolation Forest
        .arg("--")
        .arg("echo")
        .arg("test");

    // Should work with all methods enabled
    cmd.assert().success();
}

// ============================================================================
// Test 11: Autoencoder with explainability
// ============================================================================

#[test]
fn test_dl_anomaly_with_explainability() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--dl-anomaly").arg("--explain").arg("--").arg("echo").arg("test");

    // Should provide explanations
    cmd.assert().success();
}

// ============================================================================
// Test 12: Hidden layer size configuration
// ============================================================================

#[test]
fn test_dl_hidden_size_configuration() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c")
        .arg("--dl-anomaly")
        .arg("--dl-hidden-size")
        .arg("4")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 13: Training epochs configuration
// ============================================================================

#[test]
fn test_dl_epochs_configuration() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c")
        .arg("--dl-anomaly")
        .arg("--dl-epochs")
        .arg("50")
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}
