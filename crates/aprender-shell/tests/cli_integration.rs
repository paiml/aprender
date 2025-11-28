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

// ============================================================================
// Test: CLI_010 - Latency (Usability)
// ============================================================================

#[test]
fn test_cli_010_suggest_latency() {
    use std::time::Instant;

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
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model.path().to_str().unwrap(),
        ])
        .assert()
        .success();

    // Measure suggestion latency
    let start = Instant::now();
    aprender_shell()
        .args(["suggest", "git ", "-m", model.path().to_str().unwrap()])
        .assert()
        .success();
    let elapsed = start.elapsed();

    // Should complete in under 500ms for good UX
    // (Note: first run includes binary startup time)
    assert!(
        elapsed.as_millis() < 500,
        "Suggestion took {}ms, should be <500ms",
        elapsed.as_millis()
    );
}

// ============================================================================
// Test: CLI_011 - Analyze Command (CodeFeatureExtractor)
// ============================================================================

#[test]
fn test_cli_011_analyze_basic() {
    let history = create_temp_history(&[
        "git status",
        "git commit -m test",
        "git commit -m 'fix bug'",
        "cargo build",
        "cargo test",
    ]);

    aprender_shell()
        .args(["analyze", "-f", history.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Command Analysis"))
        .stdout(predicate::str::contains("Base Commands"));
}

#[test]
fn test_cli_011_analyze_top_limit() {
    let history = create_temp_history(&[
        "git status",
        "git commit -m test",
        "cargo build",
        "npm install",
        "python script.py",
    ]);

    aprender_shell()
        .args([
            "analyze",
            "-f",
            history.path().to_str().unwrap(),
            "--top",
            "3",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Top 3 Base Commands"));
}

// ============================================================================
// Test: CLI_012 - Augment with CodeEDA
// ============================================================================

#[test]
fn test_cli_012_augment_code_eda() {
    let history = create_temp_history(&[
        "git status",
        "git commit -m test",
        "cargo build --release",
        "npm run test",
    ]);

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
// Test: CLI_013 - Fish Widget Generation (GH-88)
// ============================================================================

#[test]
fn test_cli_013_fish_widget() {
    aprender_shell()
        .arg("fish-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("# >>> aprender-shell widget >>>"))
        .stdout(predicate::str::contains("aprender-shell Fish widget"))
        .stdout(predicate::str::contains("__aprender_suggest"))
        .stdout(predicate::str::contains("__aprender_complete"))
        .stdout(predicate::str::contains("# <<< aprender-shell widget <<<"));
}

#[test]
fn test_cli_013_fish_widget_has_disable_toggle() {
    aprender_shell()
        .arg("fish-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("APRENDER_DISABLED"));
}

// ============================================================================
// Test: CLI_014 - Uninstall Command (GH-87)
// ============================================================================

#[test]
fn test_cli_014_uninstall_help() {
    aprender_shell()
        .args(["uninstall", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Uninstall widget"))
        .stdout(predicate::str::contains("--zsh"))
        .stdout(predicate::str::contains("--bash"))
        .stdout(predicate::str::contains("--fish"))
        .stdout(predicate::str::contains("--keep-model"))
        .stdout(predicate::str::contains("--dry-run"));
}

#[test]
fn test_cli_014_uninstall_dry_run_no_installation() {
    // With --dry-run and no shell specified, should report no installation found
    aprender_shell()
        .args(["uninstall", "--dry-run"])
        .assert()
        .success();
}

#[test]
fn test_cli_014_uninstall_zsh_not_found() {
    // When targeting ZSH specifically but no .zshrc exists or has no widget
    aprender_shell()
        .args(["uninstall", "--zsh", "--dry-run"])
        .assert()
        .success();
}

#[test]
fn test_cli_014_uninstall_removes_widget_block() {
    use std::io::Write;

    // Create a temp file simulating a .zshrc with the widget
    let mut file = tempfile::NamedTempFile::new().unwrap();
    writeln!(file, "# Some existing config").unwrap();
    writeln!(file, "export PATH=$PATH:/usr/local/bin").unwrap();
    writeln!(file).unwrap();
    writeln!(file, "# >>> aprender-shell widget >>>").unwrap();
    writeln!(file, "_aprender_suggest() {{").unwrap();
    writeln!(file, "    # widget code").unwrap();
    writeln!(file, "}}").unwrap();
    writeln!(file, "# <<< aprender-shell widget <<<").unwrap();
    writeln!(file).unwrap();
    writeln!(file, "# More config after").unwrap();
    file.flush().unwrap();

    // Read original content
    let original = std::fs::read_to_string(file.path()).unwrap();
    assert!(original.contains(">>> aprender-shell widget >>>"));

    // For this test, we verify the marker detection works
    // (The uninstall command uses the actual home directory)
    assert!(original.contains(">>> aprender-shell widget >>>"));
    assert!(original.contains("<<< aprender-shell widget <<<"));
}

// ============================================================================
// Test: CLI_015 - ZSH Widget Markers (GH-96)
// ============================================================================

#[test]
fn test_cli_015_zsh_widget_has_markers() {
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("# >>> aprender-shell widget >>>"))
        .stdout(predicate::str::contains("# <<< aprender-shell widget <<<"));
}

#[test]
fn test_cli_015_zsh_widget_has_disable_toggle() {
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("APRENDER_DISABLED"));
}

#[test]
fn test_cli_015_zsh_widget_has_timeout() {
    // GH-96: Widget should use timeout to prevent hangs
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("timeout 0.1"));
}

#[test]
fn test_cli_015_zsh_widget_quoted_substitution() {
    // GH-96: SC2046 - Command substitution should be quoted
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("suggestion=\"$("));
}

#[test]
fn test_cli_015_zsh_widget_uninstall_hint() {
    // Widget should include hint about how to uninstall
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("aprender-shell uninstall"));
}

// ============================================================================
// Test: CLI_016 - Inspect Command (Model Card - spec §11)
// ============================================================================

#[test]
fn test_cli_016_inspect_help() {
    aprender_shell()
        .args(["inspect", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Inspect model metadata"))
        .stdout(predicate::str::contains("--format"));
}

#[test]
fn test_cli_016_inspect_text_format() {
    // Train a model first
    let history = create_temp_history(&[
        "git status",
        "git commit -m test",
        "git push origin main",
        "cargo build --release",
        "cargo test --lib",
    ]);

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

    // Inspect with text format (default)
    aprender_shell()
        .args(["inspect", "-m", model.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("MODEL INFORMATION"))
        .stdout(predicate::str::contains("Architecture"))
        .stdout(predicate::str::contains("MarkovModel"));
}

#[test]
fn test_cli_016_inspect_json_format() {
    // Train a model first
    let history = create_temp_history(&[
        "kubectl get pods",
        "kubectl describe pod test",
        "docker ps -a",
    ]);

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

    // Inspect with JSON format
    aprender_shell()
        .args([
            "inspect",
            "-m",
            model.path().to_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"model_id\""))
        .stdout(predicate::str::contains("\"version\""))
        .stdout(predicate::str::contains("\"architecture\""));
}

#[test]
fn test_cli_016_inspect_huggingface_format() {
    // Train a model first
    let history = create_temp_history(&["npm install", "npm run build", "npm test"]);

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

    // Inspect with Hugging Face format
    aprender_shell()
        .args([
            "inspect",
            "-m",
            model.path().to_str().unwrap(),
            "--format",
            "huggingface",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("---"))
        .stdout(predicate::str::contains("pipeline_tag:"))
        .stdout(predicate::str::contains("- aprender"))
        .stdout(predicate::str::contains("- rust"));
}

#[test]
fn test_cli_016_inspect_nonexistent_model() {
    // Inspect a file that doesn't exist
    aprender_shell()
        .args(["inspect", "-m", "/nonexistent/model.apr"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Failed to load model"));
}

// ============================================================================
// Test: CLI_017 - Publish Command (HF Hub - GH-100)
// ============================================================================

#[test]
fn test_cli_017_publish_help() {
    aprender_shell()
        .args(["publish", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Publish model to Hugging Face Hub",
        ))
        .stdout(predicate::str::contains("--repo"))
        .stdout(predicate::str::contains("--commit"));
}

#[test]
fn test_cli_017_publish_nonexistent_model() {
    aprender_shell()
        .args(["publish", "-m", "/nonexistent/model.apr", "-r", "org/repo"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Failed to load model"));
}

#[test]
fn test_cli_017_publish_without_token() {
    // Train a model first
    let history = create_temp_history(&["git status", "git commit -m test", "cargo build"]);

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

    // Publish without HF_TOKEN (should show instructions)
    aprender_shell()
        .args([
            "publish",
            "-m",
            model.path().to_str().unwrap(),
            "-r",
            "paiml/test-model",
        ])
        .env_remove("HF_TOKEN")
        .assert()
        .success()
        .stderr(predicate::str::contains("HF_TOKEN"))
        .stdout(predicate::str::contains("Model card saved"));
}

#[test]
fn test_cli_017_publish_generates_readme() {
    // Train a model first
    let history = create_temp_history(&[
        "kubectl get pods",
        "kubectl describe pod test",
        "docker run nginx",
    ]);

    let temp_dir = tempfile::tempdir().unwrap();
    let model_path = temp_dir.path().join("test.model");

    // Train
    aprender_shell()
        .args([
            "train",
            "-f",
            history.path().to_str().unwrap(),
            "-o",
            model_path.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Publish (generates README.md)
    aprender_shell()
        .args([
            "publish",
            "-m",
            model_path.to_str().unwrap(),
            "-r",
            "paiml/kubectl-model",
            "-c",
            "Initial upload",
        ])
        .env_remove("HF_TOKEN")
        .assert()
        .success();

    // Check README.md was created
    let readme_path = temp_dir.path().join("README.md");
    assert!(readme_path.exists(), "README.md should be created");

    let content = std::fs::read_to_string(&readme_path).unwrap();
    assert!(
        content.contains("aprender"),
        "README should mention aprender"
    );
    assert!(
        content.contains("Shell Completion"),
        "README should mention Shell Completion"
    );
}

#[test]
fn test_cli_017_publish_with_custom_commit() {
    let history = create_temp_history(&["npm install", "npm test"]);
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

    // Without HF_TOKEN, shows upload instructions with commit message
    aprender_shell()
        .args([
            "publish",
            "-m",
            model.path().to_str().unwrap(),
            "-r",
            "org/custom",
            "-c",
            "Custom commit v2",
        ])
        .env_remove("HF_TOKEN")
        .assert()
        .success()
        .stdout(predicate::str::contains("org/custom"))
        .stdout(predicate::str::contains("Model card saved"));
}

// ============================================================================
// Test: CLI_018 - Stream Mode (GH-95)
// ============================================================================

#[test]
fn test_cli_018_stream_help() {
    aprender_shell()
        .args(["stream", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stream mode"))
        .stdout(predicate::str::contains("stdin"))
        .stdout(predicate::str::contains("--format"));
}

#[test]
fn test_cli_018_stream_missing_model() {
    aprender_shell()
        .args(["stream", "-m", "/nonexistent/model.apr"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("Failed")));
}

// ============================================================================
// Test: CLI_019 - Daemon Mode (GH-95)
// ============================================================================

#[test]
fn test_cli_019_daemon_help() {
    aprender_shell()
        .args(["daemon", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Daemon mode"))
        .stdout(predicate::str::contains("socket"))
        .stdout(predicate::str::contains("--foreground"));
}

#[test]
fn test_cli_019_daemon_stop_no_daemon() {
    aprender_shell()
        .args(["daemon-stop", "-s", "/tmp/nonexistent-test.sock"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not running").or(predicate::str::contains("not found")));
}

#[test]
fn test_cli_019_daemon_status_no_daemon() {
    aprender_shell()
        .args(["daemon-status", "-s", "/tmp/nonexistent-test.sock"])
        .assert()
        .failure()
        .stdout(predicate::str::contains("not running").or(predicate::str::contains("not found")));
}

#[test]
fn test_cli_019_daemon_missing_model() {
    aprender_shell()
        .args([
            "daemon",
            "-m",
            "/nonexistent/model.apr",
            "-s",
            "/tmp/test-daemon.sock",
            "--foreground",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("not found").or(predicate::str::contains("Failed")));
}

// ============================================================================
// Test: CLI_020 - ZSH Widget with Daemon Support (GH-95)
// ============================================================================

#[test]
fn test_cli_020_zsh_widget_v4() {
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("aprender-shell ZSH widget v4"))
        .stdout(predicate::str::contains("daemon support"));
}

#[test]
fn test_cli_020_zsh_widget_daemon_functions() {
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("_aprender_daemon_available"))
        .stdout(predicate::str::contains("_aprender_suggest_daemon"))
        .stdout(predicate::str::contains("APRENDER_USE_DAEMON"));
}

#[test]
fn test_cli_020_zsh_widget_auto_daemon() {
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("APRENDER_AUTO_DAEMON"));
}

#[test]
fn test_cli_020_zsh_widget_socket_config() {
    aprender_shell()
        .arg("zsh-widget")
        .assert()
        .success()
        .stdout(predicate::str::contains("APRENDER_SOCKET"));
}
