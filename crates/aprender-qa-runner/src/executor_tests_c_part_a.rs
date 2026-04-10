#[test]
fn test_tool_executor_validate_failure() {
    let mock_runner = MockCommandRunner::new().with_validate_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_validate();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_bench_failure() {
    let mock_runner = MockCommandRunner::new().with_bench_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_bench();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_check_failure() {
    let mock_runner = MockCommandRunner::new().with_check_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_check();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_profile_failure() {
    let mock_runner = MockCommandRunner::new().with_profile_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_profile();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_trace_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_trace("layer");

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}

#[test]
fn test_tool_executor_profile_ci_passes_with_metrics() {
    // Test that profile CI passes when output contains metrics
    let mock_runner = MockCommandRunner::new().with_tps(100.0);
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_profile_ci();

    assert!(result.passed);
    assert!(result.stdout.contains("throughput"));
}

#[test]
fn test_tool_executor_with_no_gpu_true() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        true, // no_gpu = true
        30_000,
        Arc::new(mock_runner),
    );

    // Just verify executor is created correctly
    let debug_str = format!("{executor:?}");
    assert!(debug_str.contains("no_gpu: true"));
}

#[test]
fn test_tool_executor_execute_trace_levels() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result_layer = executor.execute_trace("layer");
    assert!(result_layer.tool.contains("trace-layer"));

    let result_op = executor.execute_trace("op");
    assert!(result_op.tool.contains("trace-op"));

    let result_tensor = executor.execute_trace("tensor");
    assert!(result_tensor.tool.contains("trace-tensor"));
}

#[test]
fn test_resolve_model_path_gguf() {
    let temp_dir = tempfile::tempdir().unwrap();
    let gguf_dir = temp_dir.path().join("gguf");
    std::fs::create_dir_all(&gguf_dir).unwrap();
    std::fs::write(gguf_dir.join("model.gguf"), b"fake").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
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
    assert!(path.unwrap().contains("gguf"));
}

#[test]
fn test_resolve_model_path_apr() {
    let temp_dir = tempfile::tempdir().unwrap();
    let apr_dir = temp_dir.path().join("apr");
    std::fs::create_dir_all(&apr_dir).unwrap();
    std::fs::write(apr_dir.join("model.apr"), b"fake").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
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
    assert!(path.unwrap().contains("apr"));
}

#[test]
fn test_resolve_model_path_safetensors() {
    let temp_dir = tempfile::tempdir().unwrap();
    let st_dir = temp_dir.path().join("safetensors");
    std::fs::create_dir_all(&st_dir).unwrap();
    std::fs::write(st_dir.join("model.safetensors"), b"fake").unwrap();

    let config = ExecutionConfig {
        model_path: Some(temp_dir.path().to_string_lossy().to_string()),
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
    assert!(path.unwrap().contains("safetensors"));
}

#[test]
fn test_resolve_model_path_no_cache() {
    // No model_path and no files - should return None
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
    // With no model path and no files, should return None
    assert!(path.is_none());
}

#[test]
fn test_executor_execute_dry_run() {
    let mock_runner = MockCommandRunner::new();
    let config = ExecutionConfig {
        dry_run: true,
        ..Default::default()
    };
    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: dry-run-test
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

    // In dry run mode, all scenarios should be skipped
    assert_eq!(result.skipped, 3);
    // G0-PULL passes
    assert!(result.passed >= 1);
}

#[test]
fn test_executor_execute_with_stop_on_first_policy() {
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

    // With StopOnFirst policy, should stop after first failure
    assert_eq!(result.failed, 1);
}

#[test]
fn test_executor_execute_with_collect_all_policy() {
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

    // With CollectAll policy, should collect all failures
    assert_eq!(result.failed, 3);
}

#[test]
fn test_executor_default_impl() {
    let executor = Executor::default();
    assert_eq!(executor.config().max_workers, 4);
    assert!(!executor.config().dry_run);
}

#[test]
fn test_parse_tps_from_output_with_tps() {
    let output = "Info: Loading model\ntok/s: 42.5\nDone";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 42.5).abs() < 0.01);
}

#[test]
fn test_parse_tps_from_output_no_tps() {
    let output = "Some random output without tok/s";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_none());
}

#[test]
fn test_extract_generated_text() {
    let output = "=== Model Info ===\nThis is generated text\ntok/s: 30.0";
    let text = Executor::extract_generated_text(output);
    assert!(text.contains("This is generated text"));
    assert!(!text.contains("tok/s"));
    assert!(!text.contains("==="));
}

#[test]
fn test_extract_output_text_multiline_detailed() {
    let output = "Some prefix\nOutput:\nLine 1\nLine 2\nLine 3\nCompleted in 1s";
    let text = Executor::extract_output_text(output);
    assert!(text.contains("Line 1"));
    assert!(text.contains("Line 2"));
    assert!(text.contains("Line 3"));
}

#[test]
fn test_extract_output_text_with_empty_lines() {
    let output = "Output:\nActual output here\n\nCompleted";
    let text = Executor::extract_output_text(output);
    assert!(text.contains("Actual output here"));
}

#[test]
fn test_failure_policy_default_is_stop_on_p0() {
    let policy = FailurePolicy::default();
    assert_eq!(policy, FailurePolicy::StopOnP0);
}

#[test]
fn test_execution_config_debug_display() {
    let config = ExecutionConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("ExecutionConfig"));
    assert!(debug_str.contains("failure_policy"));
}

#[test]
fn test_tool_test_result_all_fields() {
    let result = ToolTestResult {
        tool: "test-tool".to_string(),
        passed: true,
        exit_code: 0,
        stdout: "stdout".to_string(),
        stderr: String::new(),
        duration_ms: 100,
        gate_id: "F-TEST-001".to_string(),
    };
    assert_eq!(result.tool, "test-tool");
    assert!(result.passed);
    assert_eq!(result.gate_id, "F-TEST-001");
}

#[test]
fn test_executor_evidence_accessor() {
    let executor = Executor::new();
    let evidence = executor.evidence();
    assert_eq!(evidence.total(), 0);
}

#[test]
fn test_execution_result_is_success_false_due_to_failed() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 9,
        failed: 1,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert!(!result.is_success());
}

#[test]
fn test_execution_result_is_success_when_all_pass() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 10,
        failed: 0,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert!(result.is_success());
}
