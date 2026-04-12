#[test]
fn test_executor_corroborated_with_stderr() {
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("The answer is 4.", "Warning: some benign warning");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: stderr-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let _result = executor.execute(&playbook).expect("Execution failed");

    let evidence = executor.evidence().all();
    assert!(!evidence.is_empty());
    // Corroborated scenario evidence (not G0-VALIDATE) should have stderr
    let ev = evidence
        .iter()
        .find(|e| e.stderr.is_some())
        .expect("should have evidence with stderr");
    assert!(ev.stderr.as_ref().unwrap().contains("Warning"));
}

// =========================================================================
// Falsified with stderr
// =========================================================================

#[test]
fn test_executor_falsified_with_stderr() {
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("", "Error: model failed")
        .with_exit_code(1);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        failure_policy: FailurePolicy::CollectAll,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: falsified-stderr
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 1
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");
    assert!(result.failed >= 1);

    let evidence = executor.evidence().all();
    let ev = evidence
        .iter()
        .find(|e| e.stderr.is_some())
        .expect("should have evidence with stderr");
    assert!(ev.stderr.is_some());
}

// =========================================================================
// execute_profile_flamegraph / execute_profile_focus /
// execute_backend_equivalence / execute_serve_lifecycle
// These use Command::new("apr") directly and will fail since apr isn't
// installed, but we cover the error paths.
// =========================================================================

#[test]
fn test_execute_profile_flamegraph_no_apr() {
    let executor = ToolExecutor::new("test-model.gguf".to_string(), true, 5000);
    let temp_dir = tempfile::tempdir().unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    // apr binary not found => stderr contains error
    assert!(!result.passed);
    assert_eq!(result.tool, "profile-flamegraph");
    assert_eq!(result.gate_id, "F-PROFILE-002");
}

#[test]
fn test_execute_profile_flamegraph_with_mock_success() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    // Mock returns success but no SVG file is created
    assert_eq!(result.tool, "profile-flamegraph");
    assert_eq!(result.gate_id, "F-PROFILE-002");
    assert!(!result.passed); // No SVG file generated
}

#[test]
fn test_execute_profile_flamegraph_with_svg_file() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        false,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    // Pre-create a valid SVG file
    let svg_path = temp_dir.path().join("profile_flamegraph.svg");
    std::fs::write(&svg_path, "<svg><rect/></svg>").unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    assert!(result.passed);
    assert!(result.stdout.contains("valid: true"));
}

#[test]
fn test_execute_profile_flamegraph_with_invalid_svg() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "test-model.gguf".to_string(),
        true,
        5000,
        Arc::new(mock_runner),
    );
    let temp_dir = tempfile::tempdir().unwrap();
    // Pre-create an invalid SVG file
    let svg_path = temp_dir.path().join("profile_flamegraph.svg");
    std::fs::write(&svg_path, "not a valid svg at all").unwrap();
    let result = executor.execute_profile_flamegraph(temp_dir.path());
    assert!(!result.passed);
    assert!(result.stdout.contains("valid: false"));
}
