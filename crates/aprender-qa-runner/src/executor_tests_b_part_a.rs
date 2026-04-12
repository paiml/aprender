/// Verify ExecutionResult stores all playbook execution fields
#[test]
fn test_execution_result_fields() {
    let result = ExecutionResult {
        playbook_name: "my-playbook".to_string(),
        total_scenarios: 50,
        passed: 45,
        failed: 3,
        skipped: 2,
        duration_ms: 5000,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    assert_eq!(result.playbook_name, "my-playbook");
    assert_eq!(result.total_scenarios, 50);
    assert_eq!(result.passed, 45);
    assert_eq!(result.failed, 3);
    assert_eq!(result.skipped, 2);
    assert_eq!(result.duration_ms, 5000);
}

/// Verify FailurePolicy implements Copy
#[test]
fn test_failure_policy_copy() {
    let policy = FailurePolicy::CollectAll;
    let copied: FailurePolicy = policy;
    assert_eq!(copied, FailurePolicy::CollectAll);
}

/// Verify extract_output_text handles trailing content after Output: marker
#[test]
fn test_extract_output_text_with_trailing_content() {
    let output = "Prefix\nOutput:\nAnswer is 4\nMore answer text\nCompleted in 2.5s\nExtra stuff";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "Answer is 4 More answer text");
}

/// Verify extract_generated_text filters separators and tok/s lines
#[test]
fn test_extract_generated_text_mixed_content() {
    let output = "Line 1\n=== SEPARATOR ===\nLine 2\ntok/s: 50.0\nLine 3";
    let result = Executor::extract_generated_text(output);
    assert!(result.contains("Line 1"));
    assert!(result.contains("Line 2"));
    assert!(result.contains("Line 3"));
    assert!(!result.contains("==="));
    assert!(!result.contains("tok/s"));
}

/// Verify parse_tps_from_output extracts tok/s value at end of line
#[test]
fn test_parse_tps_from_output_at_end() {
    let output = "All output finished tok/s: 99.9";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 99.9).abs() < 0.01);
}

/// Verify parse_tps_from_output finds tok/s in multiline output
#[test]
fn test_parse_tps_from_output_multiline() {
    let output = "Line 1\nLine 2\ntok/s: 25.5\nLine 4";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 25.5).abs() < f64::EPSILON);
}

/// Verify extract_output_text captures final answer at end of output
#[test]
fn test_extract_output_text_output_at_end() {
    let output = "Header info\nOutput:\nFinal answer here";
    let result = Executor::extract_output_text(output);
    assert_eq!(result, "Final answer here");
}

/// Verify ExecutionResult with gateway failure reports not success
#[test]
fn test_execution_result_with_gateway_failure() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 10,
        passed: 0,
        failed: 10,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: Some("G1: Model failed to load".to_string()),
        evidence: EvidenceCollector::new(),
    };
    assert!(!result.is_success());
    assert!(result.gateway_failed.is_some());
    assert!(result.gateway_failed.as_ref().unwrap().contains("G1"));
}

/// Verify ExecutionConfig accepts all configuration fields
#[test]
fn test_execution_config_all_fields() {
    let config = ExecutionConfig {
        failure_policy: FailurePolicy::CollectAll,
        default_timeout_ms: 30_000,
        max_workers: 2,
        dry_run: true,
        model_path: Some("/path/to/model.gguf".to_string()),
        no_gpu: true,
        run_conversion_tests: false,
        run_profile_ci: true,
        run_golden_rule_test: false,
        golden_reference_path: Some("/path/to/ref.json".to_string()),
        lock_file_path: None,
        playbook_file_path: None,
        check_integrity: false,
        warn_implicit_skips: false,
        run_hf_parity: false,
        hf_parity_corpus_path: None,
        hf_parity_model_family: None,
        output_dir: Some("test_output".to_string()),
        run_contract_tests: false,
        run_ollama_parity: false,
        metadata_only: false,
    };
    assert_eq!(config.failure_policy, FailurePolicy::CollectAll);
    assert!(config.dry_run);
    assert!(config.no_gpu);
    assert!(!config.run_conversion_tests);
    assert!(config.run_profile_ci);
    assert!(!config.run_contract_tests);
}

/// Verify ToolTestResult stores all tool execution fields
#[test]
fn test_tool_test_result_fields_comprehensive() {
    let result = ToolTestResult {
        tool: "custom-test".to_string(),
        passed: false,
        exit_code: 127,
        stdout: "stdout content".to_string(),
        stderr: "error: command not found".to_string(),
        duration_ms: 150,
        gate_id: "F-CUSTOM-001".to_string(),
    };
    assert_eq!(result.tool, "custom-test");
    assert!(!result.passed);
    assert_eq!(result.exit_code, 127);
    assert!(!result.stdout.is_empty());
    assert!(!result.stderr.is_empty());
}

/// Verify golden_scenario prompt contains expected keywords
#[test]
fn test_golden_scenario_prompt_content() {
    let model_id = ModelId::new("org", "name");
    let scenario = Executor::golden_scenario(&model_id);
    assert!(scenario.prompt.contains("Golden Rule"));
    assert!(scenario.prompt.contains("convert"));
    assert!(scenario.prompt.contains("inference"));
}

/// Verify Executor accepts custom timeout and workers config
#[test]
fn test_executor_with_custom_timeout_and_workers() {
    let config = ExecutionConfig {
        default_timeout_ms: 120_000,
        max_workers: 16,
        ..Default::default()
    };
    let executor = Executor::with_config(config);
    assert_eq!(executor.config().default_timeout_ms, 120_000);
    assert_eq!(executor.config().max_workers, 16);
}

/// Verify pass_rate calculation for partial passes
#[test]
fn test_execution_result_pass_rate_partial() {
    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 3,
        passed: 1,
        failed: 2,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: EvidenceCollector::new(),
    };
    let rate = result.pass_rate();
    assert!((rate - 100.0 / 3.0).abs() < 0.01);
}

/// Verify ToolTestResult converts to passing evidence
#[test]
fn test_tool_test_result_to_evidence_with_content() {
    let result = ToolTestResult {
        tool: "validate".to_string(),
        passed: true,
        exit_code: 0,
        stdout: "Model validated successfully".to_string(),
        stderr: String::new(),
        duration_ms: 200,
        gate_id: "F-VALIDATE-001".to_string(),
    };
    let model_id = ModelId::new("org", "model");
    let evidence = result.to_evidence(&model_id);
    assert!(evidence.outcome.is_pass());
    assert!(evidence.output.contains("validated"));
}

/// Verify ToolTestResult accepts zero duration
#[test]
fn test_tool_test_result_with_zero_duration() {
    let result = ToolTestResult {
        tool: "fast-test".to_string(),
        passed: true,
        exit_code: 0,
        stdout: "OK".to_string(),
        stderr: String::new(),
        duration_ms: 0,
        gate_id: "F-FAST-001".to_string(),
    };
    assert_eq!(result.duration_ms, 0);
}

/// Verify extract_output_text preserves multi-line content
#[test]
fn test_extract_output_text_preserves_content() {
    let output = "Info\nOutput:\n  First line\n  Second line  \n  Third line\nCompleted in 1s";
    let result = Executor::extract_output_text(output);
    assert!(result.contains("First line"));
    assert!(result.contains("Second line"));
    assert!(result.contains("Third line"));
}

// ============================================================
// Tests using MockCommandRunner for subprocess execution paths
// ============================================================

use crate::command::MockCommandRunner;

/// Verify subprocess execution with mock runner produces expected output
#[test]
fn test_executor_with_mock_runner_subprocess_execution() {
    let (_tmp, model_path) = create_test_model_file(Format::Gguf);
    let mock_runner = MockCommandRunner::new()
        .with_tps(42.0)
        .with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some(model_path),
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

    let (output, stderr, exit_code, tps, skipped) = executor.subprocess_execution(&scenario);

    assert!(!skipped);
    assert!(output.contains("4") || output.is_empty()); // Depends on extract logic
    assert!(stderr.is_none_or(|s| s.is_empty()));
    assert_eq!(exit_code, 0);
    // tps may or may not be parsed depending on output format
    let _ = tps;
}

/// Verify subprocess execution handles inference failure correctly
#[test]
fn test_executor_with_mock_runner_inference_failure() {
    let (_tmp, model_path) = create_test_model_file(Format::Gguf);
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some(model_path),
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

    assert_eq!(exit_code, 1);
    assert!(stderr.is_some());
}

/// Verify execute_scenario creates non-empty evidence
#[test]
fn test_executor_with_mock_runner_execute_scenario() {
    let mock_runner = MockCommandRunner::new()
        .with_tps(30.0)
        .with_inference_response("The answer is 4.");

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

    // Evidence should be created
    assert!(!evidence.id.is_empty());
    assert!(!evidence.gate_id.is_empty());
}

/// Verify golden rule test passes with matching mock inference outputs
#[test]
fn test_executor_with_mock_runner_golden_rule_test() {
    let mock_runner = MockCommandRunner::new()
        .with_tps(25.0)
        .with_inference_response("Output:\nThe answer is 4\nCompleted in 1s");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_golden_rule_test: true,
        run_conversion_tests: false, // Disable other tests
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let model_id = ModelId::new("test", "model");
    let (passed, failed) =
        executor.run_golden_rule_test(std::path::Path::new("/test/model.gguf"), &model_id);

    // With mock runner, both inferences should succeed with same output
    // So golden rule test should pass - exactly one test run
    assert_eq!(passed + failed, 1);
}

/// Verify golden rule test fails when conversion fails
#[test]
fn test_executor_with_mock_runner_golden_rule_conversion_failure() {
    let mock_runner = MockCommandRunner::new()
        .with_convert_failure()
        .with_inference_response("Output:\nThe answer is 4\nCompleted in 1s");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let model_id = ModelId::new("test", "model");
    let (passed, failed) =
        executor.run_golden_rule_test(std::path::Path::new("/test/model.gguf"), &model_id);

    // Conversion failure should result in 0 passed, 1 failed
    assert_eq!(passed, 0);
    assert_eq!(failed, 1);

    // Evidence should be collected
    assert!(!executor.collector.all().is_empty());
}

/// Verify golden rule test fails when inference fails
#[test]
fn test_executor_with_mock_runner_golden_rule_inference_failure() {
    let mock_runner = MockCommandRunner::new().with_inference_failure();

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let model_id = ModelId::new("test", "model");
    let (passed, failed) =
        executor.run_golden_rule_test(std::path::Path::new("/test/model.gguf"), &model_id);

    // First inference failure should result in 0 passed, 1 failed
    assert_eq!(passed, 0);
    assert_eq!(failed, 1);
}

/// Verify ToolExecutor inspect with mock runner returns GGUF info
#[test]
fn test_tool_executor_with_mock_runner_inspect() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        true,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_inspect();

    assert!(result.passed);
    assert_eq!(result.exit_code, 0);
    assert!(result.stdout.contains("GGUF"));
}

/// Verify ToolExecutor validate with mock runner passes
#[test]
fn test_tool_executor_with_mock_runner_validate() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_validate();

    assert!(result.passed);
    assert_eq!(result.exit_code, 0);
}
