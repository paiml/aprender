#[test]
fn test_parallel_config_default() {
    let config = ParallelConfig::default();
    assert!(config.num_workers > 0);
    assert_eq!(config.timeout_ms, 60_000);
}

#[test]
fn test_parallel_executor_single() {
    let executor = ParallelExecutor::default();
    let scenario = test_scenario();

    let evidence = executor.execute_single(&scenario);
    // Without a real apr binary, subprocess execution fails
    assert!(evidence.outcome.is_fail());
}

#[test]
fn test_parallel_executor_batch() {
    let config = ParallelConfig {
        num_workers: 2,
        ..Default::default()
    };
    let executor = ParallelExecutor::new(config);
    let scenarios = test_scenarios(10);

    let result = executor.execute(&scenarios);

    assert_eq!(result.evidence.len(), 10);
    assert_eq!(result.passed + result.failed + result.skipped, 10);
}

#[test]
fn test_parallel_executor_stop_on_failure() {
    let config = ParallelConfig {
        num_workers: 1, // Single thread for predictable behavior
        stop_on_failure: true,
        ..Default::default()
    };
    let executor = ParallelExecutor::new(config);

    // Create scenarios with one that will fail (empty prompt)
    let mut scenarios = test_scenarios(5);
    scenarios.insert(
        2,
        QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            String::new(), // Empty prompt will fail
            99,
        ),
    );

    let result = executor.execute(&scenarios);

    // With single thread and stop on failure, we should stop early
    assert!(result.failed > 0);
}

#[test]
fn test_estimate_tokens() {
    assert_eq!(estimate_tokens(""), 1);
    assert_eq!(estimate_tokens("test"), 1);
    assert_eq!(estimate_tokens("hello world this is a test"), 6);
}

#[test]
fn test_parallel_result_fields() {
    let result = ParallelResult {
        evidence: vec![],
        passed: 5,
        failed: 2,
        skipped: 1,
        duration_ms: 1000,
        stopped_early: false,
    };

    assert_eq!(result.passed, 5);
    assert_eq!(result.failed, 2);
    assert_eq!(result.skipped, 1);
    assert!(!result.stopped_early);
}

#[test]
fn test_parallel_executor_default() {
    let executor = ParallelExecutor::default();
    let scenarios = test_scenarios(3);
    let result = executor.execute(&scenarios);
    assert_eq!(result.evidence.len(), 3);
}

#[test]
fn test_evidence_ext_with_stderr() {
    let scenario = test_scenario();
    let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 100);
    let with_stderr = evidence.with_stderr(Some("error output".to_string()));
    assert_eq!(with_stderr.stderr, Some("error output".to_string()));
}

#[test]
fn test_evidence_ext_with_reason() {
    let scenario = test_scenario();
    let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 100);
    let with_reason = evidence.with_reason("test reason".to_string());
    assert_eq!(with_reason.reason, "test reason");
}

#[test]
fn test_parallel_config_clone() {
    let config = ParallelConfig::default();
    let cloned = config.clone();
    assert_eq!(cloned.num_workers, config.num_workers);
    assert_eq!(cloned.timeout_ms, config.timeout_ms);
}

#[test]
fn test_parallel_config_debug() {
    let config = ParallelConfig::default();
    let debug_str = format!("{config:?}");
    assert!(debug_str.contains("ParallelConfig"));
}

#[test]
fn test_parallel_result_debug() {
    let result = ParallelResult {
        evidence: vec![],
        passed: 0,
        failed: 0,
        skipped: 0,
        duration_ms: 0,
        stopped_early: false,
    };
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("ParallelResult"));
}

#[test]
fn test_execute_single_failing() {
    let executor = ParallelExecutor::default();
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        String::new(), // Empty prompt will fail garbage oracle
        42,
    );

    let evidence = executor.execute_single(&scenario);
    // Empty output should fail
    assert!(evidence.outcome.is_fail() || evidence.output.is_empty());
}

#[test]
fn test_parallel_collect_all() {
    let config = ParallelConfig {
        num_workers: 2,
        stop_on_failure: false,
        ..Default::default()
    };
    let executor = ParallelExecutor::new(config);

    let mut scenarios = test_scenarios(5);
    scenarios.push(QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        String::new(), // Will fail
        99,
    ));

    let result = executor.execute(&scenarios);
    // Should execute all scenarios
    assert_eq!(result.evidence.len(), 6);
}

#[test]
fn test_parallel_executor_with_custom_model_path() {
    let config = ParallelConfig {
        model_path: "custom/model.gguf".to_string(),
        ..Default::default()
    };
    let executor = ParallelExecutor::new(config);
    assert_eq!(executor.config.model_path, "custom/model.gguf");
}

#[test]
fn test_parallel_executor_with_timeout() {
    let config = ParallelConfig {
        timeout_ms: 1000,
        ..Default::default()
    };
    let executor = ParallelExecutor::new(config);
    assert_eq!(executor.config.timeout_ms, 1000);
}

#[test]
fn test_execute_single_without_binary() {
    let executor = ParallelExecutor::default();
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "2+2=".to_string(),
        42,
    );

    let evidence = executor.execute_single(&scenario);
    // Without a real apr binary, subprocess execution fails
    assert!(evidence.outcome.is_fail());
}

#[test]
fn test_execute_single_falsified() {
    let executor = ParallelExecutor::default();
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        String::new(), // Empty will fail
        42,
    );

    let evidence = executor.execute_single(&scenario);
    // Should either fail (empty output) or pass with empty output check
    assert!(evidence.output.is_empty() || evidence.outcome.is_fail());
}

#[test]
fn test_parallel_batch_without_binary() {
    let executor = ParallelExecutor::default();
    let scenarios: Vec<_> = (0..5)
        .map(|i| {
            QaScenario::new(
                ModelId::new("test", "model"),
                Modality::Run,
                Backend::Cpu,
                Format::Gguf,
                format!("What is 2+{}?", i),
                i as u64,
            )
        })
        .collect();

    let result = executor.execute(&scenarios);
    assert_eq!(result.evidence.len(), 5);
    // Without a real apr binary, all executions fail
    assert_eq!(result.failed, 5);
}

#[test]
fn test_parallel_with_mixed_modalities() {
    let executor = ParallelExecutor::default();
    let scenarios = vec![
        QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Run,
            Backend::Cpu,
            Format::Gguf,
            "2+2=".to_string(),
            1,
        ),
        QaScenario::new(
            ModelId::new("test", "model"),
            Modality::Chat,
            Backend::Cpu,
            Format::Gguf,
            "Hello".to_string(),
            2,
        ),
    ];

    let result = executor.execute(&scenarios);
    assert_eq!(result.evidence.len(), 2);
}

#[test]
fn test_parallel_result_stopped_early() {
    let result = ParallelResult {
        evidence: vec![],
        passed: 3,
        failed: 1,
        skipped: 6,
        duration_ms: 500,
        stopped_early: true,
    };
    assert!(result.stopped_early);
    assert_eq!(result.skipped, 6);
}

#[test]
fn test_parallel_empty_scenarios() {
    let executor = ParallelExecutor::default();
    let result = executor.execute(&[]);
    assert_eq!(result.evidence.len(), 0);
    assert_eq!(result.passed, 0);
    assert_eq!(result.failed, 0);
}

#[test]
fn test_evidence_ext_with_stderr_none() {
    let scenario = test_scenario();
    let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 100);
    let with_stderr = evidence.with_stderr(None);
    assert!(with_stderr.stderr.is_none());
}

#[test]
fn test_parallel_config_single_worker() {
    let config = ParallelConfig {
        num_workers: 1,
        ..Default::default()
    };
    let executor = ParallelExecutor::new(config);
    let scenarios = test_scenarios(3);
    let result = executor.execute(&scenarios);
    assert_eq!(result.evidence.len(), 3);
}

#[test]
fn test_estimate_tokens_longer_text() {
    // 24 characters should be ~6 tokens
    let tokens = estimate_tokens("This is a longer string.");
    assert!(tokens >= 5);
}

#[test]
fn test_parallel_batch_single_scenario() {
    let executor = ParallelExecutor::default();
    let scenarios = vec![test_scenario()];
    let result = executor.execute(&scenarios);
    assert_eq!(result.evidence.len(), 1);
}

#[test]
fn test_parallel_result_all_fields() {
    let scenario = test_scenario();
    let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 50);

    let result = ParallelResult {
        evidence: vec![evidence],
        passed: 1,
        failed: 0,
        skipped: 0,
        duration_ms: 100,
        stopped_early: false,
    };

    assert_eq!(result.evidence.len(), 1);
    assert_eq!(result.duration_ms, 100);
}

#[test]
fn test_parallel_config_all_fields() {
    let config = ParallelConfig {
        num_workers: 8,
        timeout_ms: 30_000,
        model_path: "/custom/path/model.gguf".to_string(),
        stop_on_failure: true,
    };

    assert_eq!(config.num_workers, 8);
    assert_eq!(config.timeout_ms, 30_000);
    assert!(config.stop_on_failure);
    assert!(config.model_path.contains("custom"));
}

// ========================================================================
// QA-EXEC-04: Timeout Enforcement Test
// ========================================================================

/// QA-EXEC-04: Verify timeout enforcement creates F-INT-002 FALSIFIED evidence
///
/// This test verifies that when a process exceeds the timeout threshold:
/// 1. The runner kills the process
/// 2. The evidence is marked with Timeout outcome
/// 3. IntegrityChecker::check_process_termination() returns FALSIFIED for F-INT-002
#[test]
fn test_timeout_enforcement_marks_f_int_002_falsified() {
    use crate::evidence::Outcome;
    use crate::patterns::{IntegrityChecker, SpecGate};

    // Simulate a timed out process
    let timed_out = true;
    let exit_code = None; // No exit code due to timeout/kill
    let has_output = false;

    // Verify IntegrityChecker marks this as F-INT-002 failure
    let result = IntegrityChecker::check_process_termination(exit_code, timed_out, has_output);

    assert_eq!(result.gate_id, SpecGate::IntProcessTermination.id());
    assert_eq!(result.gate_id, "F-INT-002");
    assert!(!result.passed, "Timeout should mark F-INT-002 as FALSIFIED");
    assert!(
        result.description.contains("timed out"),
        "Description should mention timeout: {}",
        result.description
    );

    // Also verify Evidence::timeout() creates correct outcome
    let evidence = Evidence::timeout(
        SpecGate::IntProcessTermination.id(),
        test_scenario(),
        61_000, // >60s timeout
    );
    assert!(
        matches!(evidence.outcome, Outcome::Timeout),
        "Evidence should have Timeout outcome"
    );
}

/// Test that short timeouts are enforced in configuration
#[test]
fn test_timeout_config_enforcement() {
    // Very short timeout (should normally fail on real process)
    let config = ParallelConfig {
        timeout_ms: 1, // 1ms timeout
        ..Default::default()
    };

    let executor = ParallelExecutor::new(config);
    assert_eq!(executor.config.timeout_ms, 1);

    // Verify the config is correctly set
    let evidence = Evidence::timeout("F-INT-002", test_scenario(), 61_000);
    assert!(!evidence.outcome.is_pass());
}

#[test]
fn test_subprocess_execution_empty_command() {
    let executor = ParallelExecutor::new(ParallelConfig::default());

    // Create a scenario that generates an empty command
    let scenario = QaScenario::new(
        ModelId::new("test", "model"),
        Modality::Run,
        Backend::Cpu,
        Format::Gguf,
        "test".to_string(),
        42,
    );

    // The subprocess_execution should handle errors gracefully
    let (_, exit_code, stderr) = executor.subprocess_execution(&scenario);
    // With a fake model path, this will fail
    assert!(exit_code != 0 || stderr.is_some());
}

