//! CLI Integration Tests for aprender-shell
//!
//! Uses assert_cmd (MANDATORY) for end-to-end CLI testing.
//! Tests actual binary execution with real inputs/outputs.

#![allow(clippy::unwrap_used)] // Tests can use unwrap for simplicity
#![allow(deprecated)] // cargo_bin still works, just deprecated for custom build-dir

use assert_cmd::Command;
use predicates::prelude::*;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Helper Functions
// ============================================================================

/// Create an aprender-shell command (MANDATORY pattern)
fn aprender_shell() -> Command {
    Command::cargo_bin("aprender-shell").expect("Failed to find aprender-shell binary")
}

/// Create a temporary history file with given commands
fn create_temp_history(commands: &[&str]) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    for cmd in commands {
        writeln!(file, "{}", cmd).expect("Failed to write command");
    }
    file
}

/// Create a ZSH-style history file with timestamps
fn create_zsh_history(commands: &[&str]) -> NamedTempFile {
    let mut file = NamedTempFile::new().expect("Failed to create temp file");
    for (i, cmd) in commands.iter().enumerate() {
        writeln!(file, ": {}:0;{}", 1700000000 + i, cmd).expect("Failed to write command");
    }
    file
}

// ============================================================================
// Test: CLI_001 - Help and Version
// ============================================================================

#[test]
fn test_cli_001_help_flag() {
    aprender_shell()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("aprender-shell"))
        .stdout(predicate::str::contains("AI-powered shell completion"));
}

#[test]
fn test_cli_001_version_flag() {
    aprender_shell()
        .arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("aprender-shell"));
}

#[test]
fn test_cli_001_subcommand_help() {
    aprender_shell()
        .args(["train", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Train a model"));
}

// ============================================================================
// Test: CLI_002 - Train Command
// ============================================================================

#[test]
fn test_cli_002_train_basic() {
    let history = create_temp_history(&[
        "git status",
        "git commit -m test",
        "git push",
        "cargo build",
        "cargo test",
    ]);

    let output = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            output.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Training"))
        .stdout(predicate::str::contains("Model saved"));
}

#[test]
fn test_cli_002_train_filters_corrupted() {
    // Train with corrupted commands - they should be filtered
    let history = create_temp_history(&[
        "git status",
        "git commit-m test", // corrupted - should be filtered
        "git push",
        "cargo build-r", // corrupted - should be filtered
        "cargo test",
    ]);

    let output = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            output.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands loaded: 3")); // Only 3 valid
}

#[test]
fn test_cli_002_train_zsh_format() {
    let history = create_zsh_history(&["git status", "git commit -m test", "ls -la"]);

    let output = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            output.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Commands loaded: 3"));
}

// ============================================================================
// Test: CLI_003 - Suggest Command
// ============================================================================

#[test]
fn test_cli_003_suggest_basic() {
    // First train a model
    let history = create_temp_history(&[
        "git status",
        "git status",
        "git commit -m test",
        "git push origin main",
    ]);

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
        .success();

    // Now test suggestions
    aprender_shell()
        .args(["suggest", "git ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("git status")); // Most frequent
}

#[test]
fn test_cli_003_suggest_partial_token() {
    let history = create_temp_history(&[
        "git commit -m test",
        "git checkout main",
        "git clone url",
        "git status",
    ]);

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
        .success();

    // Partial token "git c" should suggest commit/checkout/clone
    aprender_shell()
        .args(["suggest", "git c", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("git c")); // All should start with "git c"
}

#[test]
fn test_cli_003_suggest_no_corrupted() {
    let history = create_temp_history(&[
        "git commit -m test",
        "git commit-m broken", // corrupted
        "git checkout main",
    ]);

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
        .success();

    // Should NOT suggest corrupted "commit-m"
    aprender_shell()
        .args(["suggest", "git co", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("commit-m").not());
}

// ============================================================================
// Test: CLI_004 - Stats Command
// ============================================================================

#[test]
fn test_cli_004_stats() {
    let history = create_temp_history(&["git status", "git commit -m test", "cargo build"]);

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
        .success();

    aprender_shell()
        .args(["stats", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("N-gram size"))
        .stdout(predicate::str::contains("Vocabulary size"));
}

// ============================================================================
// Test: CLI_005 - Validate Command
// ============================================================================

#[test]
fn test_cli_005_validate() {
    // Validate trains its own model internally using train/test split
    let history = create_temp_history(&[
        "git status",
        "git status",
        "git commit -m test",
        "git push",
        "cargo build",
        "cargo test",
        "cargo run",
        "ls -la",
        "cd src",
        "cat file.txt",
    ]);

    aprender_shell()
        .args(["validate", "-f", history.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("VALIDATION RESULTS"))
        .stdout(predicate::str::contains("Hit@"));
}

// ============================================================================
// Test: CLI_006 - Augment Command (Synthetic Data)
// ============================================================================

#[test]
fn test_cli_006_augment_basic() {
    let history = create_temp_history(&[
        "git status",
        "git commit -m test",
        "git push origin main",
        "cargo build --release",
        "cargo test --all",
    ]);

    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "augment",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
            "-a",
            "0.5", // 50% augmentation
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Data Augmentation"))
        .stdout(predicate::str::contains("Coverage"));
}

#[test]
fn test_cli_006_augment_with_diversity() {
    let history = create_temp_history(&[
        "git status",
        "git commit -m test",
        "git push",
        "cargo build",
        "cargo test",
    ]);

    let model = NamedTempFile::new().unwrap();

    aprender_shell()
        .args([
            "augment",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
            "--monitor-diversity",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Diversity"));
}

// ============================================================================
// Test: CLI_007 - Error Handling
// ============================================================================

#[test]
fn test_cli_007_missing_history_file() {
    aprender_shell()
        .args(["train", "-f", "/nonexistent/path/history"])
        .assert()
        .failure();
}

#[test]
fn test_cli_007_invalid_ngram_size() {
    let history = create_temp_history(&["git status"]);

    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-n",
            "99", // Invalid - should be 2-5
        ])
        .assert()
        .failure() // Rejects invalid n-gram sizes
        .stderr(predicate::str::contains("N-gram size must be between 2 and 5"));
}

// ============================================================================
// Test: CLI_008 - ZSH Widget Generation
// ============================================================================

#[test]
fn test_cli_008_zsh_widget() {
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("aprender-shell ZSH widget"))
        .stdout(predicate::str::contains("_aprender_suggest"))
        .stdout(predicate::str::contains("bindkey"));
}

// ============================================================================
// Test: CLI_009 - Export/Import
// ============================================================================

#[test]
fn test_cli_009_export_import_roundtrip() {
    let history = create_temp_history(&["git status", "git commit -m test", "cargo build"]);

    let model = NamedTempFile::new().unwrap();
    let export_file = NamedTempFile::new().unwrap();
    let imported_model = NamedTempFile::new().unwrap();

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
        .success()
        .stdout(predicate::str::contains("exported"));

    // Import
    aprender_shell()
        .args([
            "import",
            export_file.path().to_str().unwrap(),
            "-o",
            imported_model.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("imported"));
}

include!("parts/cli_integration_010.rs");
include!("parts/cli_integration_017.rs");
include!("parts/cli_integration_021.rs");
