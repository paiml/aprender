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
