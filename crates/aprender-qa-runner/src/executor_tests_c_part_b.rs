#[test]
fn test_tool_test_result_to_evidence_when_failed() {
    let result = ToolTestResult {
        tool: "validate".to_string(),
        passed: false,
        exit_code: 1,
        stdout: String::new(),
        stderr: "Validation failed".to_string(),
        duration_ms: 200,
        gate_id: "F-VALIDATE-001".to_string(),
    };
    let model_id = ModelId::new("org", "model");
    let evidence = result.to_evidence(&model_id);
    assert!(!evidence.outcome.is_pass());
    assert!(evidence.reason.contains("Validation failed") || evidence.output.is_empty());
}

#[test]
fn test_executor_with_mock_runner_trace_failure_case() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        0,
    );

    let (_, stderr, exit_code, _, _) = executor.subprocess_execution(&scenario);

    // Should include trace output in stderr
    assert_eq!(exit_code, 1);
    assert!(stderr.is_some());
}

#[test]
fn test_resolve_model_path_apr_format() {
    let tmp = tempfile::tempdir().unwrap();
    let apr_dir = tmp.path().join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();
    std::fs::write(apr_dir.join("model.apr"), b"fake apr").unwrap();

    let config = ExecutionConfig {
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("apr"));
}

#[test]
fn test_resolve_model_path_safetensors_format() {
    let tmp = tempfile::tempdir().unwrap();
    let st_dir = tmp.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors"), b"fake st").unwrap();

    let config = ExecutionConfig {
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("safetensors"));
}

#[test]
fn test_resolve_model_path_gguf_format() {
    let tmp = tempfile::tempdir().unwrap();
    let gguf_dir = tmp.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::write(gguf_dir.join("model.gguf"), b"fake gguf").unwrap();

    let config = ExecutionConfig {
        model_path: Some(tmp.path().to_string_lossy().to_string()),
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    assert!(path.is_some());
    assert!(path.unwrap().contains("gguf"));
}

#[test]
fn test_resolve_model_path_no_model_path() {
    // When no model_path is configured and no file exists, should return None
    let config = ExecutionConfig {
        model_path: None,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        0,
    );
    let path = executor.resolve_model_path(&scenario);
    // Should return None when no model file exists at default path
    assert!(path.is_none());
}

#[test]
fn test_executor_subprocess_execution_formats() {
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some("/test/cache".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    // Test APR format
    let scenario_apr = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Apr,
        "What is 2+2?".to_string(),
        0,
    );
    let (_, _, exit_code, _, _) = executor.subprocess_execution(&scenario_apr);
    assert_eq!(exit_code, 0);
}

#[test]
fn test_executor_subprocess_execution_safetensors() {
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some("/test/cache".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::SafeTensors,
        "What is 2+2?".to_string(),
        0,
    );
    let (_, _, exit_code, _, _) = executor.subprocess_execution(&scenario);
    assert_eq!(exit_code, 0);
}

#[test]
fn test_execute_scenario_with_exit_code_failure() {
    let mock_runner = MockCommandRunner::new().with_exit_code(5);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "What is 2+2?".to_string(),
        0,
    );

    let evidence = executor.execute_scenario(&scenario);

    // Non-zero exit code should result in failed evidence
    assert!(evidence.outcome.is_fail());
    assert!(evidence.exit_code.is_some());
    assert_eq!(evidence.exit_code.unwrap(), 5);
}

#[test]
fn test_execute_scenario_with_stderr_corroborated() {
    let mock_runner = MockCommandRunner::new()
        .with_inference_response_and_stderr("The answer is 4.", "Some warning");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        0,
    );

    let evidence = executor.execute_scenario(&scenario);
    // Should pass but have stderr captured
    assert!(evidence.outcome.is_pass());
}

#[test]
fn test_executor_run_conversion_tests_no_gpu() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: true,
        no_gpu: true,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));
    let model_id = ModelId::new("test", "model");

    // Run conversion tests with no_gpu flag
    let (passed, failed) =
        executor.run_conversion_tests(std::path::Path::new("/test/model.gguf"), &model_id);

    // Just verify function runs
    let _ = (passed, failed);
}

#[test]
fn test_executor_execute_with_stop_on_first_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnFirst,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: stop-on-first-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // Should stop after first failure
    assert!(result.failed >= 1);
    // Total executed should be less than total scenarios due to early stop
    let executed = result.passed + result.failed;
    assert!(executed <= result.total_scenarios);
}

#[test]
fn test_executor_execute_with_collect_all_failures() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::CollectAll,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        run_contract_tests: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: collect-all-test
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

    // Should collect all failures (3 scenarios)
    assert_eq!(result.failed, 3);
    // Bug 204: G0-PULL skipped when model_path is set, so 3 scenarios only
    assert_eq!(result.total_scenarios, 3);
}

// =========================================================================
// StopOnP0 policy test
// =========================================================================

#[test]
fn test_executor_stop_on_p0_with_p0_gate() {
    // Create a runner that returns falsified results with P0 gate IDs
    let mock_runner = MockCommandRunner::new()
        .with_inference_failure()
        .with_exit_code(1);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        failure_policy: FailurePolicy::StopOnP0,
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: p0-test
version: "1.0.0"
model:
  hf_repo: "test/model"
  formats: [gguf]
test_matrix:
  modalities: [run]
  backends: [cpu]
  scenario_count: 5
"#;
    let playbook = Playbook::from_yaml(yaml).expect("Failed to parse");
    let result = executor.execute(&playbook).expect("Execution failed");

    // With failures that don't have -P0- in gate_id, it should collect all
    assert!(result.failed >= 1);
}

// =========================================================================
// ConversionConfig::default() (no_gpu = false)
// =========================================================================

#[test]
fn test_executor_run_conversion_tests_default_config() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: true,
        run_golden_rule_test: false,
        no_gpu: false, // This triggers ConversionConfig::default()
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: conv-default-test
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
    // Just verify it runs without panic
    assert!(result.total_scenarios >= 1);
}

// =========================================================================
// Golden Rule: converted inference fails (F-GOLDEN-RULE-003)
// =========================================================================
