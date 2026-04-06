use super::*;

#[test]
fn test_benchmark_metric_statistics() {
    let metric = BenchmarkMetric::new("latency", vec![100.0, 102.0, 98.0, 101.0, 99.0], "μs");

    assert_eq!(metric.mean(), 100.0);
    assert!((metric.std_dev() - 1.58).abs() < 0.1);
    assert!(metric.cv() < 2.0); // Low variance
}

#[test]
fn test_benchmark_metric_empty() {
    let metric = BenchmarkMetric::new("empty", vec![], "ms");

    assert_eq!(metric.mean(), 0.0);
    assert_eq!(metric.std_dev(), 0.0);
    assert_eq!(metric.cv(), 0.0);
}

#[test]
fn test_benchmark_results_creation() {
    let mut results = BenchmarkResults::new("abc123", "main");

    results.add_metric(BenchmarkMetric::new("latency", vec![100.0], "μs"));

    assert_eq!(results.commit, "abc123");
    assert_eq!(results.branch, "main");
    assert!(results.get_metric("latency").is_some());
    assert!(results.get_metric("throughput").is_none());
}

#[test]
fn test_metric_regression_latency() {
    let baseline = BenchmarkMetric::new("latency_p50", vec![100.0; 5], "μs");
    let current = BenchmarkMetric::new("latency_p50", vec![110.0; 5], "μs"); // 10% worse

    let regression = MetricRegression::from_metrics(&baseline, &current, 5.0, 2.0);

    assert_eq!(regression.percent_change, 10.0);
    assert!(regression.is_regression);
    assert!(!regression.is_warning);
}

#[test]
fn test_metric_regression_throughput() {
    let baseline = BenchmarkMetric::new("throughput", vec![1000.0; 5], "ops/s");
    let current = BenchmarkMetric::new("throughput", vec![900.0; 5], "ops/s"); // 10% worse

    let regression = MetricRegression::from_metrics(&baseline, &current, 5.0, 2.0);

    assert_eq!(regression.percent_change, -10.0);
    assert!(regression.is_regression); // Decrease in throughput is regression
}

#[test]
fn test_metric_regression_warning() {
    let baseline = BenchmarkMetric::new("latency_p50", vec![100.0; 5], "μs");
    let current = BenchmarkMetric::new("latency_p50", vec![103.0; 5], "μs"); // 3% worse

    let regression = MetricRegression::from_metrics(&baseline, &current, 5.0, 2.0);

    assert!(!regression.is_regression);
    assert!(regression.is_warning);
}

#[test]
fn test_pipeline_config_default() {
    let config = PipelineConfig::default();

    assert_eq!(config.base_branch, "main");
    assert_eq!(config.timeout_sec, 600);
    assert_eq!(config.regression_threshold_percent, 5.0);
    assert_eq!(config.iterations, 10);
}

#[test]
fn test_pipeline_status_terminal() {
    assert!(!PipelineStatus::Pending.is_terminal());
    assert!(!PipelineStatus::Running.is_terminal());
    assert!(PipelineStatus::Passed.is_terminal());
    assert!(PipelineStatus::Failed.is_terminal());
    assert!(PipelineStatus::Cancelled.is_terminal());
}

#[test]
fn test_pipeline_status_github() {
    assert_eq!(PipelineStatus::Passed.github_state(), "success");
    assert_eq!(PipelineStatus::Failed.github_state(), "failure");
    assert_eq!(PipelineStatus::Running.github_state(), "pending");
}

#[test]
fn test_git_ref_as_str() {
    assert_eq!(GitRef::Branch("main".to_string()).as_ref_str(), "main");
    assert_eq!(GitRef::Commit("abc123".to_string()).as_ref_str(), "abc123");
    assert_eq!(
        GitRef::Tag("v1.0.0".to_string()).as_ref_str(),
        "refs/tags/v1.0.0"
    );
    assert_eq!(GitRef::PullRequest(123).as_ref_str(), "refs/pull/123/head");
}

#[test]
fn test_pipeline_run() {
    let config = PipelineConfig::default();
    let mut pipeline = RegressionPipeline::new(config);

    let result = pipeline.run(&GitRef::Branch("feature".to_string()));

    assert!(result.is_ok());
    let analysis = result.unwrap();
    assert!(!analysis.regressions.is_empty());
}

#[test]
fn test_pipeline_baseline_caching() {
    let config = PipelineConfig::default();
    let mut pipeline = RegressionPipeline::new(config);

    // First run creates baseline
    let _ = pipeline.run(&GitRef::Branch("feature".to_string()));
    assert!(!pipeline.baseline_cache.is_empty());

    // Clear cache
    pipeline.clear_baseline_cache();
    assert!(pipeline.baseline_cache.is_empty());
}

#[test]
fn test_regression_analysis_counts() {
    let config = PipelineConfig::default();
    let mut pipeline = RegressionPipeline::new(config);

    let result = pipeline
        .run(&GitRef::Branch("feature".to_string()))
        .unwrap();

    // All simulated metrics should be stable
    assert_eq!(result.regression_count(), 0);
    assert_eq!(result.warning_count(), 0);
}

#[test]
fn test_generate_report() {
    let config = PipelineConfig::default();
    let mut pipeline = RegressionPipeline::new(config);

    let analysis = pipeline
        .run(&GitRef::Branch("feature".to_string()))
        .unwrap();
    let report = pipeline.generate_report(&analysis);

    assert!(report.contains("# Performance Regression Report"));
    assert!(report.contains("| Metric |"));
    assert!(report.contains("latency_p50"));
}

#[test]
fn test_error_display() {
    let err = PipelineError::GitError {
        reason: "not found".to_string(),
    };
    assert!(err.to_string().contains("not found"));

    let err = PipelineError::Timeout { timeout_sec: 60 };
    assert!(err.to_string().contains("60"));
}

#[test]
fn test_store_artifact() {
    let config = PipelineConfig::default();
    let mut pipeline = RegressionPipeline::new(config);

    let analysis = pipeline
        .run(&GitRef::Branch("feature".to_string()))
        .unwrap();
    let artifact_id = pipeline.store_artifact(&analysis).unwrap();

    assert!(!artifact_id.is_empty());
    assert!(artifact_id.contains(&analysis.current.commit));
}

// FKR-048: Pipeline detects regression within 60 seconds
#[test]
fn test_fkr_048_regression_detection_timing() {
    let config = PipelineConfig {
        timeout_sec: 60,
        ..Default::default()
    };
    let mut pipeline = RegressionPipeline::new(config);

    let start = Instant::now();

    // Run pipeline
    let result = pipeline.run(&GitRef::PullRequest(123));

    let elapsed = start.elapsed();

    // Must complete within 60 seconds
    assert!(
        elapsed.as_secs() < 60,
        "Pipeline took too long: {:?}",
        elapsed
    );

    // Must produce valid result
    assert!(result.is_ok());
    let analysis = result.unwrap();

    // Must have analyzed metrics
    assert!(!analysis.regressions.is_empty());

    // Must have determined status
    assert!(analysis.status.is_terminal() || analysis.status == PipelineStatus::Passed);
}

#[test]
fn test_history_tracking() {
    let config = PipelineConfig::default();
    let mut pipeline = RegressionPipeline::new(config);

    assert!(pipeline.history().is_empty());

    let _ = pipeline.run(&GitRef::Branch("feature1".to_string()));
    assert_eq!(pipeline.history().len(), 1);

    let _ = pipeline.run(&GitRef::Branch("feature2".to_string()));
    assert_eq!(pipeline.history().len(), 2);
}

#[test]
fn test_worst_regression() {
    let baseline = BenchmarkResults::new("base", "main");
    let current = BenchmarkResults::new("curr", "feature");

    let regressions = vec![
        MetricRegression {
            name: "metric1".to_string(),
            baseline_mean: 100.0,
            current_mean: 105.0,
            percent_change: 5.0,
            is_regression: true,
            is_warning: false,
            unit: "ms".to_string(),
        },
        MetricRegression {
            name: "metric2".to_string(),
            baseline_mean: 100.0,
            current_mean: 115.0,
            percent_change: 15.0,
            is_regression: true,
            is_warning: false,
            unit: "ms".to_string(),
        },
    ];

    let analysis = RegressionAnalysis {
        baseline,
        current,
        regressions,
        status: PipelineStatus::Failed,
        analysis_duration_ms: 100,
        summary: "Test".to_string(),
    };

    let worst = analysis.worst_regression().unwrap();
    assert_eq!(worst.name, "metric2");
    assert_eq!(worst.percent_change, 15.0);
}
