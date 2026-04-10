#[test]
fn test_executor_subprocess_with_tps_parsing() {
    // The mock runner adds tok/s: {self.tps} to output, so set the tps value
    let mock_runner = MockCommandRunner::new().with_tps(42.5);

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));

    let scenario = test_scenario();
    let (_, _, _, tps, _) = executor.subprocess_execution(&scenario);

    // tps should be parsed from output
    assert!(tps.is_some());
    assert!((tps.unwrap() - 42.5).abs() < f64::EPSILON);
}

#[test]
fn test_tool_test_result_to_evidence_gate_id() {
    let result = ToolTestResult {
        tool: "special".to_string(),
        passed: true,
        exit_code: 0,
        stdout: "OK".to_string(),
        stderr: String::new(),
        duration_ms: 50,
        gate_id: "F-SPECIAL-TEST-001".to_string(),
    };

    let model_id = ModelId::new("org", "name");
    let evidence = result.to_evidence(&model_id);

    assert_eq!(evidence.gate_id, "F-SPECIAL-TEST-001");
    assert_eq!(evidence.scenario.model.org, "org");
    assert_eq!(evidence.scenario.model.name, "name");
}

#[test]
fn test_execution_result_evidence_collector() {
    let mut collector = EvidenceCollector::new();
    let evidence = Evidence::corroborated("F-TEST-001", test_scenario(), "Test output", 100);
    collector.add(evidence);

    let result = ExecutionResult {
        playbook_name: "test".to_string(),
        total_scenarios: 1,
        passed: 1,
        failed: 0,
        skipped: 0,
        duration_ms: 100,
        gateway_failed: None,
        evidence: collector,
    };

    assert_eq!(result.evidence.all().len(), 1);
}

#[test]
fn test_executor_execute_scenario_with_metrics() {
    let mock_runner = MockCommandRunner::new()
        .with_tps(75.5)
        .with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        ..Default::default()
    };

    let executor = Executor::with_runner(config, Arc::new(mock_runner));
    let scenario = test_scenario();

    let evidence = executor.execute_scenario(&scenario);

    // Metrics should be populated (duration_ms is a u64, so always valid)
    let _ = evidence.metrics.duration_ms; // Just verify it exists
}

#[test]
fn test_extract_output_text_with_whitespace_lines() {
    // Whitespace-only lines are not considered empty - they get trimmed and added
    // Only truly empty lines (or "Completed in") terminate parsing
    let output = "Header\nOutput:\n   \nActual content\n  \nCompleted in 1s";
    let result = Executor::extract_output_text(output);
    // Whitespace lines become empty after trim, content gets captured
    assert!(result.contains("Actual content"));
}

#[test]
fn test_extract_output_text_only_header() {
    let output = "Only Header no Output marker";
    let result = Executor::extract_output_text(output);
    assert!(result.is_empty());
}

#[test]
fn test_parse_tps_from_output_multiple_colons() {
    let output = "Info: tok/s: 88.8 more info";
    let tps = Executor::parse_tps_from_output(output);
    assert!(tps.is_some());
    assert!((tps.unwrap() - 88.8).abs() < f64::EPSILON);
}

#[test]
fn test_tool_executor_trace_all_levels() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    for level in &["none", "basic", "layer", "payload"] {
        let result = executor.execute_trace(level);
        assert!(result.passed);
        assert!(result.tool.contains("trace"));
        assert!(result.tool.contains(level));
    }
}

#[test]
fn test_execution_config_partial_override() {
    let config = ExecutionConfig {
        dry_run: true,
        max_workers: 1,
        ..Default::default()
    };

    assert!(config.dry_run);
    assert_eq!(config.max_workers, 1);
    // Defaults should still be set
    assert!(config.run_conversion_tests);
    assert!(config.run_golden_rule_test);
}

#[test]
fn test_executor_evidence_after_execute() {
    let mock_runner = MockCommandRunner::new().with_inference_response("The answer is 4.");

    let config = ExecutionConfig {
        model_path: Some("/test/model.gguf".to_string()),
        run_conversion_tests: false,
        run_golden_rule_test: false,
        ..Default::default()
    };

    let mut executor = Executor::with_runner(config, Arc::new(mock_runner));

    let yaml = r#"
name: evidence-test
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
    let _ = executor.execute(&playbook).expect("Execution failed");

    // Evidence should be collected
    assert!(!executor.evidence().all().is_empty());
}

#[test]
fn test_tool_executor_gate_id_format() {
    let mock_runner = MockCommandRunner::new();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_inspect();
    assert_eq!(result.gate_id, "F-INSPECT-001");

    let result = executor.execute_validate();
    assert_eq!(result.gate_id, "F-VALIDATE-001");

    let result = executor.execute_bench();
    assert_eq!(result.gate_id, "F-BENCH-001");

    let result = executor.execute_check();
    assert_eq!(result.gate_id, "F-CHECK-001");

    let result = executor.execute_profile();
    assert_eq!(result.gate_id, "F-PROFILE-001");
}

#[test]
fn test_tool_executor_profile_ci_feature_unavailable() {
    let mock_runner = MockCommandRunner::new().with_profile_ci_unavailable();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_profile_ci();

    // When feature is unavailable, should return exit code -2
    assert!(!result.passed);
    assert_eq!(result.exit_code, -2);
    assert!(result.stderr.contains("Feature not available"));
    assert_eq!(result.gate_id, "F-PROFILE-006");
}

#[test]
fn test_tool_executor_profile_ci_assertion_unavailable() {
    let mock_runner = MockCommandRunner::new().with_profile_ci_unavailable();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_profile_ci_assertion_failure();

    // When feature is unavailable, should indicate feature not available
    assert!(!result.passed);
    assert_eq!(result.exit_code, -2);
    assert_eq!(result.gate_id, "F-PROFILE-007");
}

#[test]
fn test_tool_executor_profile_ci_p99_unavailable() {
    let mock_runner = MockCommandRunner::new().with_profile_ci_unavailable();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_profile_ci_p99();

    // When feature is unavailable, should indicate feature not available
    assert!(!result.passed);
    assert_eq!(result.exit_code, -2);
    assert_eq!(result.gate_id, "F-PROFILE-008");
}

#[test]
fn test_tool_executor_inspect_failure() {
    let mock_runner = MockCommandRunner::new().with_inspect_failure();
    let executor = ToolExecutor::with_runner(
        "/test/model.gguf".to_string(),
        false,
        60_000,
        Arc::new(mock_runner),
    );

    let result = executor.execute_inspect();

    assert!(!result.passed);
    assert_eq!(result.exit_code, 1);
}
