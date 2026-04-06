
#[test]
fn test_apr_benchmark_result_compare_to_baseline_zero_baseline() {
    let result = AprBenchmarkResult {
        tokens_per_second: 50.0,
        peak_memory_mb: 100.0,
        ..Default::default()
    };
    let baseline = AprBenchmarkResult {
        tokens_per_second: 0.0,
        peak_memory_mb: 0.0,
        ..Default::default()
    };
    let comparison = result.compare_to_baseline(&baseline);
    // Division by zero should give 1.0
    assert_eq!(comparison.throughput_ratio, 1.0);
    assert_eq!(comparison.memory_ratio, 1.0);
}

#[test]
fn test_apr_prefill_result_default() {
    let result = AprPrefillResult::default();
    assert_eq!(result.prompt_tokens, 0);
    assert_eq!(result.prefill_time_ms, 0.0);
    assert_eq!(result.prefill_tok_s, 0.0);
}

#[test]
fn test_apr_load_result_default() {
    let result = AprLoadResult::default();
    assert_eq!(result.load_time_ms, 0.0);
}

#[test]
fn test_apr_parity_comparison_is_parity_true() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.96,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(comparison.is_parity());
}

#[test]
fn test_apr_parity_comparison_is_parity_false() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.90,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(!comparison.is_parity());
}

#[test]
fn test_apr_parity_comparison_is_parity_exact() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.95,
        memory_ratio: 1.0,
        parity_threshold_pct: APR_PARITY_THRESHOLD_PCT,
    };
    assert!(comparison.is_parity());
}

#[test]
fn test_apr_benchmark_constants() {
    assert_eq!(APR_CPU_DECODE_THRESHOLD_TOK_S, 50.0);
    assert_eq!(APR_PREFILL_THRESHOLD_TOK_S, 100.0);
    assert_eq!(APR_PARITY_THRESHOLD_PCT, 95.0);
}

#[test]
fn test_apr_benchmark_result_clone() {
    let result = AprBenchmarkResult {
        tokens_generated: 100,
        tokens_per_second: 50.0,
        throughput_p50: 48.0,
        throughput_p99: 52.0,
        throughput_std_dev: 2.0,
        peak_memory_mb: 200.0,
        model_memory_mb: 150.0,
        total_time_ms: 2000.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.tokens_generated, 100);
    assert_eq!(cloned.tokens_per_second, 50.0);
}

#[test]
fn test_apr_benchmark_result_debug() {
    let result = AprBenchmarkResult::default();
    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("AprBenchmarkResult"));
}

#[test]
fn test_apr_prefill_result_clone_debug() {
    let result = AprPrefillResult {
        prompt_tokens: 128,
        prefill_time_ms: 10.0,
        prefill_tok_s: 12800.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.prompt_tokens, 128);

    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("AprPrefillResult"));
}

#[test]
fn test_apr_load_result_clone_debug() {
    let result = AprLoadResult {
        load_time_ms: 500.0,
    };
    let cloned = result.clone();
    assert_eq!(cloned.load_time_ms, 500.0);

    let debug_str = format!("{result:?}");
    assert!(debug_str.contains("AprLoadResult"));
}

#[test]
fn test_apr_parity_comparison_clone_debug() {
    let comparison = AprParityComparison {
        throughput_ratio: 0.95,
        memory_ratio: 1.1,
        parity_threshold_pct: 95.0,
    };
    let cloned = comparison.clone();
    assert!((cloned.throughput_ratio - 0.95).abs() < 1e-6);

    let debug_str = format!("{comparison:?}");
    assert!(debug_str.contains("AprParityComparison"));
}
