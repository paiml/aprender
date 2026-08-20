// Sprint 22: ML Outlier Detection with Isolation Forest + XAI
// EXTREME TDD: RED phase - Integration tests for Isolation Forest-based anomaly detection
//
// Goal: Implement Isolation Forest for unsupervised anomaly detection with explainability

#![allow(deprecated)] // Command::cargo_bin is deprecated but still functional

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

// ============================================================================
// Test 1: Verify --ml-outliers flag is accepted
// ============================================================================

#[test]
fn test_ml_outliers_flag_accepted() {
    // Test that --ml-outliers flag is accepted
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--ml-outliers").arg("--").arg("echo").arg("test");

    // Should not error on flag parsing
    cmd.assert().success();
}

// ============================================================================
// Test 2: --ml-outliers works with statistics mode
// ============================================================================

#[test]
fn test_ml_outliers_with_statistics() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--ml-outliers").arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 3: --ml-outliers detects anomalous syscalls
// ============================================================================

#[test]
fn test_ml_outliers_detects_anomalies() {
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("anomaly_test");

    // Create program with normal and anomalous syscalls
    let source = r#"
#include <unistd.h>
#include <time.h>
int main() {
    // Normal fast writes (establish baseline)
    for (int i = 0; i < 100; i++) {
        write(1, "x", 1);
    }

    // Anomalous slow write
    struct timespec ts = {0, 100000000};  // 100ms
    nanosleep(&ts, NULL);
    write(1, "slow", 4);

    return 0;
}
"#;
    let source_file = tmp_dir.path().join("anomaly_test.c");
    fs::write(&source_file, source).expect("test");

    std::process::Command::new("gcc")
        .arg(&source_file)
        .arg("-o")
        .arg(&test_program)
        .output()
        .expect("Failed to compile test program");

    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--ml-outliers").arg("--").arg(&test_program);

    // Should detect anomalies in output
    cmd.assert().success().stderr(predicate::str::contains("Isolation Forest Anomaly Detection"));
}

// ============================================================================
// Test 4: --explain flag provides explainability
// ============================================================================

#[test]
fn test_explain_flag_provides_explainability() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--ml-outliers").arg("--explain").arg("--").arg("echo").arg("test");

    cmd.assert().success().stderr(
        predicate::str::contains("Feature Importance").or(predicate::str::contains("Explanation")),
    );
}

// ============================================================================
// Test 5: Isolation Forest works with JSON output
// ============================================================================

#[test]
fn test_ml_outliers_json_export() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--format").arg("json").arg("--ml-outliers").arg("--").arg("echo").arg("test");

    cmd.assert().success().stdout(
        predicate::str::contains("isolation_forest").or(predicate::str::contains("ml_outliers")),
    );
}

// ============================================================================
// Test 6: Isolation Forest with filtering
// ============================================================================

#[test]
fn test_ml_outliers_with_filtering() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--ml-outliers").arg("-e").arg("trace=file").arg("--").arg("ls").arg("-la");

    cmd.assert().success();
}

// ============================================================================
// Test 7: Minimum samples requirement
// ============================================================================

#[test]
fn test_ml_outliers_minimum_samples() {
    // Test that Isolation Forest requires minimum samples
    let tmp_dir = TempDir::new().expect("test");
    let test_program = tmp_dir.path().join("min_samples_test");

    let source = r#"
#include <unistd.h>
int main() {
    // Only 2 syscalls - insufficient for meaningful Isolation Forest
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
    cmd.arg("-c").arg("--ml-outliers").arg("--").arg(&test_program);

    // Should handle gracefully (no panic)
    cmd.assert().success();
}

// ============================================================================
// Test 8: Backward compatibility without --ml-outliers
// ============================================================================

#[test]
fn test_backward_compatibility_without_ml_outliers() {
    // Ensure existing functionality works without --ml-outliers flag
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("--").arg("echo").arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 9: Isolation Forest with timing mode
// ============================================================================

#[test]
fn test_ml_outliers_with_timing() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c").arg("-T").arg("--ml-outliers").arg("--").arg("echo").arg("hello");

    cmd.assert().success();
}

// ============================================================================
// Test 10: Compare with existing ML anomaly detection
// ============================================================================

#[test]
fn test_ml_outliers_compare_with_kmeans() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c")
        .arg("--ml-outliers")
        .arg("--ml-anomaly") // Also enable KMeans for comparison
        .arg("--")
        .arg("echo")
        .arg("test");

    // Should work with both methods enabled
    cmd.assert().success();
}

// ============================================================================
// Test 11: Isolation Forest with source correlation
// ============================================================================

#[test]
fn test_ml_outliers_with_source_correlation() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("--source").arg("--ml-outliers").arg("--").arg("echo").arg("test");

    // Should correlate anomalies with source locations
    cmd.assert().success();
}

// ============================================================================
// Test 12: Anomaly score threshold configuration
// ============================================================================

#[test]
fn test_ml_outliers_threshold_configuration() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c")
        .arg("--ml-outliers")
        .arg("--ml-outlier-threshold")
        .arg("0.6") // Custom contamination threshold
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}

// ============================================================================
// Test 13: Number of trees configuration
// ============================================================================

#[test]
fn test_ml_outliers_num_trees_configuration() {
    let mut cmd = Command::cargo_bin("aprender-profile").expect("test");
    cmd.arg("-c")
        .arg("--ml-outliers")
        .arg("--ml-outlier-trees")
        .arg("150") // Custom number of trees
        .arg("--")
        .arg("echo")
        .arg("test");

    cmd.assert().success();
}
