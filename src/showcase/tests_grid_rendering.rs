use super::*;

#[test]
fn test_showcase_runner_to_grid_native_without_gguf_with_ollama() {
    // Cover lines 346-360: apr_native_stats Some but apr_gguf_stats None, with ollama stats Some
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("TestModel", "0.5B", "Q4_K_M")
        .with_gpu_info("Test GPU", 8.0);

    // Add only APR native (no GGUF) - covers default GGUF fallback at lines 346-350
    let native_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(180),
        Duration::from_millis(5),
    )];
    runner.record_apr_native(BenchmarkStats::from_results(native_results));

    // Add Ollama stats - covers line 360 (Some(s) branch for baseline)
    let ollama_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(400),
        Duration::from_millis(50),
    )];
    runner.record_ollama(BenchmarkStats::from_results(ollama_results));

    let grid = runner.to_grid();
    let output = grid.render();
    assert!(!output.is_empty());
}

#[test]
fn test_stats_to_measurement_without_gpu_metrics() {
    // Cover line 397: stats_to_measurement where first result has no GPU metrics
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config);

    // Results without GPU metrics
    let results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(256),
        Duration::from_millis(7),
    )];
    let stats = BenchmarkStats::from_results(results);

    // Access stats_to_measurement indirectly through to_grid
    let mut runner_with_stats = ShowcaseRunner::new(ShowcaseConfig::default());
    runner_with_stats.record_apr_gguf(stats.clone());

    let grid = runner_with_stats.to_grid();
    let output = grid.render();
    assert!(!output.is_empty());

    // Verify measurement doesn't crash without GPU metrics
    drop(runner);
}

#[test]
fn test_pmat_verification_to_report_point_41_fail() {
    // Cover line 675: point_41_pass = false branch
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // Slow APR at 50 tok/s (fails point 41: 50 < 200 * 1.25 = 250)
    let apr_results: Vec<BenchmarkResult> = (0..3)
        .map(|_| BenchmarkResult::new(128, Duration::from_millis(2560), Duration::from_millis(50)))
        .collect();
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let verification = PmatVerification::verify(&runner);
    assert!(!verification.point_41_pass);

    let report = verification.to_report();
    assert!(report.contains("FAIL"));
}

#[test]
fn test_pmat_verification_to_report_point_42_fail() {
    // Cover line 684: point_42_pass = false branch (< 60 tok/s)
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // APR at 30 tok/s (fails point 42: 30 < 60)
    let apr_results: Vec<BenchmarkResult> = (0..3)
        .map(|_| BenchmarkResult::new(128, Duration::from_millis(4267), Duration::from_millis(100)))
        .collect();
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let verification = PmatVerification::verify(&runner);
    assert!(!verification.point_42_pass);

    let report = verification.to_report();
    assert!(report.contains("FAIL"));
}

#[test]
fn test_pmat_verification_to_report_point_49_fail() {
    // Cover line 693: point_49_pass = false branch (CV >= 5%)
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // High variance results (different throughputs)
    let apr_results = vec![
        BenchmarkResult::new(128, Duration::from_millis(100), Duration::from_millis(5)),
        BenchmarkResult::new(128, Duration::from_millis(500), Duration::from_millis(50)),
        BenchmarkResult::new(128, Duration::from_millis(200), Duration::from_millis(20)),
    ];
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let verification = PmatVerification::verify(&runner);
    assert!(!verification.point_49_pass); // CV should be high

    let report = verification.to_report();
    assert!(report.contains("FAIL") || report.contains("Point 49"));
}

#[test]
fn test_pmat_verification_to_report_ollama_2x_pending() {
    // Cover line 702: ollama_2x_pass = false branch ("PENDING")
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // APR at 400 tok/s (fails 2x Ollama: 400 < 318 * 2 = 636)
    let apr_results: Vec<BenchmarkResult> = (0..3)
        .map(|_| BenchmarkResult::new(128, Duration::from_millis(320), Duration::from_millis(10)))
        .collect();
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let verification = PmatVerification::verify(&runner);
    assert!(!verification.ollama_2x_pass);

    let report = verification.to_report();
    assert!(report.contains("PENDING"));
}

#[test]
fn test_pmat_verification_to_report_overall_needs_work() {
    // Cover line 712: all_pass = false branch ("NEEDS WORK")
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config);

    let verification = PmatVerification::verify(&runner);
    assert!(!verification.all_pass);

    let report = verification.to_report();
    assert!(report.contains("NEEDS WORK"));
}

#[test]
fn test_stats_to_measurement_with_gpu_metrics_coverage() {
    // Ensure GPU metrics path in stats_to_measurement is fully exercised
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("Model", "1B", "Q4_K")
        .with_gpu_info("GPU", 8.0);

    // Results with GPU metrics
    let results = vec![
        BenchmarkResult::new(100, Duration::from_millis(200), Duration::from_millis(5))
            .with_gpu_metrics(95.0, 8192.0),
        BenchmarkResult::new(100, Duration::from_millis(195), Duration::from_millis(6))
            .with_gpu_metrics(92.0, 8100.0),
    ];
    runner.record_apr_gguf(BenchmarkStats::from_results(results));

    // This should hit line 395-396 in stats_to_measurement
    let grid = runner.to_grid();
    let output = grid.render();
    assert!(!output.is_empty());
}

#[test]
fn test_showcase_runner_native_only_no_baseline_stats() {
    // Cover lines 346-350: native stats with no GGUF, use default apr_gguf
    // Cover lines 354-359: native stats with no ollama, use default baseline
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("Model", "1B", "Q4_K")
        .with_gpu_info("GPU", 8.0);

    // Only native stats, no GGUF, no Ollama
    let native_results = vec![BenchmarkResult::new(
        100,
        Duration::from_millis(150),
        Duration::from_millis(4),
    )];
    runner.record_apr_native(BenchmarkStats::from_results(native_results));

    // This should hit the default branches in to_grid for apr_gguf and ollama
    let grid = runner.to_grid();
    let output = grid.render();
    assert!(!output.is_empty());
}
