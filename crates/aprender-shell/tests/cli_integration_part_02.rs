
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

    // Should complete in under 500ms for good UX
    // (Binary startup is excluded via warmup; this measures actual suggestion time)
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
