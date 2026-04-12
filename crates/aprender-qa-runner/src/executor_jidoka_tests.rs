#[test]
fn test_g0_pull_fail_stops_execution() {
    // Jidoka: If G0-PULL fails, skip all subsequent tests
    // Bug 204: model_path must be None so G0-PULL actually runs
    let mock_runner = MockCommandRunner::new().with_pull_failure();

    let config = ExecutionConfig {
        model_path: None,
        run_conversion_tests: true,
        run_golden_rule_test: true,
        run_contract_tests: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: pull-fail-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Gateway should be failed
    assert!(result.gateway_failed.is_some());
    assert!(
        result
            .gateway_failed
            .as_ref()
            .unwrap()
            .contains("G0-PULL-001")
    );

    // No scenarios passed
    assert_eq!(result.passed, 0);
    // 3 scenarios + 1 pull failure = 4 total failed
    assert_eq!(result.failed, 4);
}

#[test]
fn test_g0_pull_sets_model_path() {
    // When model_path is None, G0-PULL should set it from pulled path
    let mock_runner = MockCommandRunner::new().with_pull_model_path("/pulled/model.safetensors");

    let config = ExecutionConfig {
        model_path: None, // No model path initially
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: pull-set-path-test
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

    // Should not fail on gateway
    assert!(result.gateway_failed.is_none());
    // G0-PULL should pass
    assert!(result.passed >= 1);
}

/// Helper: create a temp model directory with a safetensors file
fn make_temp_model_dir() -> tempfile::TempDir {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).expect("mkdir safetensors");
    std::fs::write(st_dir.join("model.safetensors"), b"fake").expect("write");
    dir
}

#[test]
fn test_g0_validate_pass() {
    let mock_runner = MockCommandRunner::new(); // validate_strict_success defaults to true
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    assert_eq!(passed, 1);
    assert_eq!(failed, 0);

    let evidence = executor.evidence().all();
    let validate_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-VALIDATE-001")
        .expect("should have G0-VALIDATE evidence");
    assert!(validate_ev.outcome.is_pass());
    assert!(validate_ev.output.contains("G0 PASS"));
}

#[test]
fn test_g0_validate_fail_corrupt_model() {
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert_eq!(failed, 1);

    let evidence = executor.evidence().all();
    let validate_ev = evidence
        .iter()
        .find(|e| e.gate_id == "G0-VALIDATE-001")
        .expect("should have G0-VALIDATE evidence");
    assert!(!validate_ev.outcome.is_pass());
    assert!(validate_ev.reason.contains("G0 FAIL"));
}

#[test]
fn test_g0_validate_fail_stops_execution() {
    // Jidoka: If G0-VALIDATE fails, skip all subsequent tests
    let mock_runner = MockCommandRunner::new().with_validate_strict_failure();
    let dir = make_temp_model_dir();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        run_conversion_tests: true,
        run_golden_rule_test: true,
        run_contract_tests: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: validate-fail-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 3
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Gateway should be failed
    assert!(result.gateway_failed.is_some());
    assert!(
        result
            .gateway_failed
            .as_ref()
            .unwrap()
            .contains("G0-VALIDATE-001")
    );

    // Bug 204: G0-PULL skipped (model_path is set), then G0-VALIDATE fails
    assert_eq!(result.passed, 0);
    // 3 scenarios + 1 validate failure = 4 total failed
    assert_eq!(result.failed, 4);
}

#[test]
fn test_g0_all_subgates_na_continues_execution() {
    // When all G0 sub-gates return (0, 0) — not applicable — execution continues.
    // With Jidoka early returns, ANY G0 sub-gate failure stops the line.
    // model_path = None means all sub-gates (FORMAT/VALIDATE/TENSOR/INTEGRITY/LAYOUT)
    // are skipped (no model to check), so scenarios proceed unblocked.
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: all-gates-na-test
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

    // No gateway failure — all sub-gates returned N/A, Jidoka didn't trigger
    assert!(result.gateway_failed.is_none());
    // Scenarios executed
    assert!(result.passed >= 1);
}

#[test]
fn test_g0_validate_no_model_path() {
    // When no model_path is set, G0-VALIDATE should be skipped (0, 0)
    let mock_runner = MockCommandRunner::new();

    let config = ExecutionConfig {
        model_path: None, // No model path
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: no-model-path-test
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

    // No gateway failure
    assert!(result.gateway_failed.is_none());
    // 1 scenario + 1 G0-PULL (no validate — mock path has no safetensors)
    assert_eq!(result.total_scenarios, 2);
}

#[test]
fn test_g0_validate_no_safetensors_files() {
    // When model dir has no safetensors files, G0-VALIDATE auto-passes (0, 0)
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let mock_runner = MockCommandRunner::new();

    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_g0_validate_multiple_shards() {
    // Multi-file sharded models: validate each shard
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let st_dir = dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).expect("mkdir");
    std::fs::write(st_dir.join("model-00001-of-00002.safetensors"), b"shard1").expect("write");
    std::fs::write(st_dir.join("model-00002-of-00002.safetensors"), b"shard2").expect("write");

    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some(dir.path().to_string_lossy().to_string()),
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_validate_check(dir.path(), &model_id);

    // Both shards should be validated
    assert_eq!(passed, 2);
    assert_eq!(failed, 0);
}

#[test]
fn test_find_safetensors_files_single_file() {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let file = dir.path().join("model.safetensors");
    std::fs::write(&file, b"test").expect("write");

    let files = Executor::find_safetensors_files(&file);
    assert_eq!(files.len(), 1);
    assert_eq!(files[0], file);
}

#[test]
fn test_find_safetensors_files_non_safetensors() {
    let dir = tempfile::TempDir::new().expect("create temp dir");
    let file = dir.path().join("model.gguf");
    std::fs::write(&file, b"test").expect("write");

    let files = Executor::find_safetensors_files(&file);
    assert!(files.is_empty());
}

#[test]
fn test_find_safetensors_files_directory() {
    let dir = make_temp_model_dir();
    let files = Executor::find_safetensors_files(dir.path());
    assert_eq!(files.len(), 1);
}

#[test]
fn test_integrity_scenario_creation() {
    let model_id = ModelId::new("test", "model");
    let scenario = Executor::integrity_scenario(&model_id);

    assert_eq!(scenario.model.org, "test");
    assert_eq!(scenario.model.name, "model");
    assert_eq!(scenario.format, Format::SafeTensors);
    assert!(scenario.prompt.contains("G0"));
}

#[test]
fn test_run_g0_integrity_check_no_safetensors() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");
    // No safetensors files

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    // No safetensors = auto-pass (0, 0)
    assert_eq!(passed, 0);
    assert_eq!(failed, 0);
}

#[test]
fn test_run_g0_integrity_check_missing_config() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Create safetensors but no config.json
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    // Should fail due to missing config
    assert_eq!(passed, 0);
    assert!(failed > 0);

    // Evidence should contain G0-INTEGRITY failure
    let evidence = executor.evidence();
    assert!(
        evidence
            .all()
            .iter()
            .any(|e| e.gate_id.starts_with("G0-INTEGRITY"))
    );
}

#[test]
fn test_run_g0_integrity_check_pass() {
    use tempfile::TempDir;
    let dir = TempDir::new().expect("create temp dir");

    // Create matching config and safetensors
    create_test_config_for_executor(dir.path(), 24, 896, 151_936);
    create_mock_safetensors_for_test(dir.path(), 24, 896, 151_936);

    let mut executor = Executor::new();
    let model_id = ModelId::new("test", "model");
    let (passed, failed) = executor.run_g0_integrity_check(dir.path(), &model_id);

    assert_eq!(passed, 1);
    assert_eq!(failed, 0);

    // Evidence should show corroborated
    let evidence = executor.evidence();
    assert!(
        evidence
            .all()
            .iter()
            .any(|e| { e.gate_id.starts_with("G0-INTEGRITY") && e.outcome.is_pass() })
    );
}
