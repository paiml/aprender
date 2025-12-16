//! Real-World Integration Tests for aprender-shell
//!
//! These tests use realistic shell history fixtures (same as bashrs benchmarks)
//! to validate production-like scenarios with assert_cmd.

#![allow(clippy::unwrap_used)] // Tests can use unwrap for simplicity
#![allow(deprecated)] // cargo_bin still works, just deprecated for custom build-dir

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

// Load benchmark fixtures
const SMALL_HISTORY: &str = include_str!("../benches/fixtures/small_history.txt");
const MEDIUM_HISTORY: &str = include_str!("../benches/fixtures/medium_history.txt");
const LARGE_HISTORY: &str = include_str!("../benches/fixtures/large_history.txt");

/// Create an aprender-shell command
fn aprender_shell() -> Command {
    Command::cargo_bin("aprender-shell").expect("Failed to find aprender-shell binary")
}

/// Create a temporary history file from fixture content
fn create_fixture_history(content: &str) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    // Filter out comments for realistic history
    for line in content.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() && !trimmed.starts_with('#') {
            writeln!(file, "{}", trimmed).expect("Failed to write command");
        }
    }
    file
}

// ============================================================================
// Test: REAL_001 - Small History (Developer Basics)
// ============================================================================

#[test]
fn test_real_001_train_small_history() {
    let history = create_fixture_history(SMALL_HISTORY);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Training"))
        .stdout(predicate::str::contains("Model saved"));
}

#[test]
fn test_real_001_suggest_git_commands() {
    let history = create_fixture_history(SMALL_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Suggest git commands
    aprender_shell()
        .args(["suggest", "git ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("git")); // Should suggest git commands
}

#[test]
fn test_real_001_suggest_cargo_commands() {
    let history = create_fixture_history(SMALL_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Suggest cargo commands
    aprender_shell()
        .args(["suggest", "cargo ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("cargo")); // Should suggest cargo commands
}

// ============================================================================
// Test: REAL_002 - Medium History (Full Developer Workflow)
// ============================================================================

#[test]
fn test_real_002_train_medium_history() {
    let history = create_fixture_history(MEDIUM_HISTORY);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands loaded"));
}

#[test]
fn test_real_002_stats_medium_history() {
    let history = create_fixture_history(MEDIUM_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Get stats
    aprender_shell()
        .args(["stats", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("N-gram size"))
        .stdout(predicate::str::contains("Vocabulary size"));
}

#[test]
fn test_real_002_docker_suggestions() {
    let history = create_fixture_history(MEDIUM_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Suggest docker commands
    aprender_shell()
        .args(["suggest", "docker ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("docker")); // Should have docker suggestions
}

#[test]
fn test_real_002_kubectl_suggestions() {
    let history = create_fixture_history(MEDIUM_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Suggest kubectl commands
    aprender_shell()
        .args(["suggest", "kubectl ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("kubectl")); // Should have kubectl suggestions
}

// ============================================================================
// Test: REAL_003 - Large History (Production Scale)
// ============================================================================

#[test]
fn test_real_003_train_large_history() {
    let history = create_fixture_history(LARGE_HISTORY);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Model saved"));
}

#[test]
#[ignore = "Flaky latency test - fails under CI/coverage load"]
fn test_real_003_suggest_latency_acceptable() {
    use std::time::Instant;

    let history = create_fixture_history(LARGE_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Warmup run to exclude binary startup time from measurement
    aprender_shell()
        .args(["suggest", "git ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success();

    // Measure suggestion latency (excluding binary startup)
    let start = Instant::now();
    aprender_shell()
        .args(["suggest", "git ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success();
    let elapsed = start.elapsed();

    // Should be under 200ms even for large models (warmup excludes startup overhead)
    assert!(
        elapsed.as_millis() < 200,
        "Large model suggestion took {}ms, should be <200ms",
        elapsed.as_millis()
    );
}

#[test]
fn test_real_003_partial_token_completion() {
    let history = create_fixture_history(LARGE_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Partial token "git co" should suggest commit/checkout
    aprender_shell()
        .args(["suggest", "git co", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("git co")); // Should complete partial token
}

// ============================================================================
// Test: REAL_004 - Validation and Cross-Validation
// ============================================================================

#[test]
fn test_real_004_validate_medium_history() {
    let history = create_fixture_history(MEDIUM_HISTORY);

    aprender_shell()
        .args(["validate", "-f", history.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("VALIDATION RESULTS"))
        .stdout(predicate::str::contains("Hit@"));
}

// ============================================================================
// Test: REAL_005 - Data Augmentation
// ============================================================================

#[test]
fn test_real_005_augment_small_history() {
    let history = create_fixture_history(SMALL_HISTORY);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "augment",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
            "-a",
            "0.5",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Data Augmentation"));
}

#[test]
fn test_real_005_augment_with_code_eda() {
    let history = create_fixture_history(MEDIUM_HISTORY);
    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "augment",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
            "--use-code-eda",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("CodeEDA"));
}

// ============================================================================
// Test: REAL_006 - Analysis Command
// ============================================================================

#[test]
fn test_real_006_analyze_medium_history() {
    let history = create_fixture_history(MEDIUM_HISTORY);

    aprender_shell()
        .args(["analyze", "-f", history.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Command Analysis"))
        .stdout(predicate::str::contains("git"))
        .stdout(predicate::str::contains("cargo"))
        .stdout(predicate::str::contains("docker"));
}

#[test]
fn test_real_006_analyze_large_history() {
    let history = create_fixture_history(LARGE_HISTORY);

    aprender_shell()
        .args([
            "analyze",
            "-f",
            history.path().to_str().unwrap(),
            "--top",
            "5",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Top 5 Base Commands"));
}

// ============================================================================
// Test: REAL_007 - Export/Import with Large Data
// ============================================================================

#[test]
fn test_real_007_export_import_roundtrip() {
    let history = create_fixture_history(MEDIUM_HISTORY);
    let model = NamedTempFile::new().unwrap();
    let export_file = NamedTempFile::new().unwrap();
    let reimported_model = NamedTempFile::new().unwrap();

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Export
    aprender_shell()
        .args([
            "export",
            export_file.path().to_str().unwrap(),
            "-m",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Import
    aprender_shell()
        .args([
            "import",
            export_file.path().to_str().unwrap(),
            "-o",
            reimported_model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Verify reimported model works
    aprender_shell()
        .args([
            "suggest",
            "git ",
            "-m",
            reimported_model.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("git"));
}

// ============================================================================
// Test: REAL_008 - Paged Model for Very Large History
// ============================================================================

#[test]
fn test_real_008_paged_model_training() {
    let history = create_fixture_history(LARGE_HISTORY);
    let model_dir = tempfile::tempdir().unwrap();
    let model_path = model_dir.path().join("paged.model");

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model_path.to_str().unwrap(),
            "--memory-limit",
            "1", // 1MB limit to force paging
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Paged model saved"));
}

// ============================================================================
// Test: REAL_009 - Incremental Updates
// ============================================================================

#[test]
fn test_real_009_incremental_update() {
    let history1 = create_fixture_history(SMALL_HISTORY);
    // Create extended history that includes the original commands plus new ones
    let mut extended_content = String::from(SMALL_HISTORY);
    extended_content.push_str("\nnew-special-command arg1\nnew-special-command arg2\n");
    let history2 = create_fixture_history(&extended_content);
    let model = NamedTempFile::new().unwrap();

    // Initial training
    aprender_shell()
        .args([
            "train",
            "-f",
            history1.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Incremental update - should report either "updated" or "up to date"
    aprender_shell()
        .args([
            "update",
            "-f",
            history2.path().to_str().unwrap(),
            "-m",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();
}

// ============================================================================
// Test: REAL_010 - End-to-End User Workflow
// ============================================================================

#[test]
fn test_real_010_complete_user_workflow() {
    let history = create_fixture_history(MEDIUM_HISTORY);
    let model = NamedTempFile::new().unwrap();

    // Step 1: Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Step 2: Get stats
    aprender_shell()
        .args(["stats", "-m", model.path().to_str().unwrap()])
        .assert()
        .success();

    // Step 3: Use suggestions for common patterns
    let prefixes = ["git ", "cargo ", "docker ", "npm "];
    for prefix in &prefixes {
        aprender_shell()
            .args(["suggest", prefix, "-m", model.path().to_str().unwrap()])
            .assert()
            .success();
    }

    // Step 4: Validate quality
    aprender_shell()
        .args(["validate", "-f", history.path().to_str().unwrap()])
        .assert()
        .success();
}
