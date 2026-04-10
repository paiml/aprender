#[test]
fn test_evidence_with_metrics() {
    let scenario = test_scenario();
    let evidence = Evidence::corroborated("F-TEST-001", scenario, "output", 100);
    let with_metrics = evidence.with_metrics(PerformanceMetrics {
        duration_ms: 500,
        tokens_per_second: Some(10.0),
        ..Default::default()
    });
    assert_eq!(with_metrics.metrics.duration_ms, 500);
}

#[test]
fn test_parallel_result_with_stopped_early() {
    let result = ParallelResult {
        evidence: vec![],
        passed: 2,
        failed: 1,
        skipped: 7,
        duration_ms: 100,
        stopped_early: true,
    };
    assert!(result.stopped_early);
    assert_eq!(result.skipped, 7);
}

#[test]
fn test_parallel_executor_execute_with_subprocess_mode() {
    // This test verifies subprocess configuration is accepted
    let config = ParallelConfig {
        num_workers: 1,
        model_path: "/nonexistent/path.gguf".to_string(),
        stop_on_failure: true,
        ..Default::default()
    };
    let executor = ParallelExecutor::new(config);

    // Execute with empty scenarios should return quickly
    let result = executor.execute(&[]);
    assert_eq!(result.evidence.len(), 0);
}
