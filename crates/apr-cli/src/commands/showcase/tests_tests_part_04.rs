
#[test]
fn test_calculate_stddev_many_values() {
    // All same value should give 0 stddev
    let values: Vec<f64> = vec![5.0; 100];
    let stddev = calculate_stddev(&values);
    assert_eq!(stddev, 0.0);
}

// ========================================================================
// generate_jitter Tests
// ========================================================================

#[test]
fn test_generate_jitter_bounded() {
    // Run many times to check bounds
    for _ in 0..1000 {
        let j = generate_jitter();
        assert!(j >= -1.0, "jitter {j} below -1.0");
        assert!(j <= 1.0, "jitter {j} above 1.0");
    }
}

// ========================================================================
// extract_json_field Tests
// ========================================================================

#[test]
fn test_extract_json_field_basic() {
    let json = r#"{"eval_count":42,"eval_duration":1000000000}"#;
    let count = super::benchmark::extract_json_field(json, "eval_count");
    assert_eq!(count, Some(42.0));

    let duration = super::benchmark::extract_json_field(json, "eval_duration");
    assert_eq!(duration, Some(1_000_000_000.0));
}

#[test]
fn test_extract_json_field_with_spaces() {
    let json = r#"{"eval_count": 42, "eval_duration": 1000000000}"#;
    let count = super::benchmark::extract_json_field(json, "eval_count");
    assert_eq!(count, Some(42.0));
}

#[test]
fn test_extract_json_field_float() {
    let json = r#"{"value": 3.14}"#;
    let val = super::benchmark::extract_json_field(json, "value");
    assert!((val.unwrap() - 3.14).abs() < 0.001);
}

#[test]
fn test_extract_json_field_missing() {
    let json = r#"{"other": 42}"#;
    let val = super::benchmark::extract_json_field(json, "missing");
    assert!(val.is_none());
}

#[test]
fn test_extract_json_field_empty_json() {
    let json = r#"{}"#;
    let val = super::benchmark::extract_json_field(json, "field");
    assert!(val.is_none());
}

#[test]
fn test_extract_json_field_nested() {
    // Should find the first occurrence at top level
    let json = r#"{"eval_count":100,"nested":{"eval_count":200}}"#;
    let val = super::benchmark::extract_json_field(json, "eval_count");
    assert_eq!(val, Some(100.0));
}

// ========================================================================
// format_benchmark_csv Comprehensive Tests
// ========================================================================

#[test]
fn test_format_benchmark_csv_header_structure() {
    let bench = BenchmarkComparison {
        apr_tps: 1.0,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 1.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 0.1,
        runs: 1,
    };
    let csv = format_benchmark_csv(&bench);
    let header = csv.lines().next().unwrap();
    assert_eq!(
        header,
        "system,tokens_per_sec,ttft_ms,speedup_pct,stddev,runs"
    );
}

#[test]
fn test_format_benchmark_csv_apr_row_format() {
    let bench = BenchmarkComparison {
        apr_tps: 123.45,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 67.89,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 2.34,
        runs: 50,
    };
    let csv = format_benchmark_csv(&bench);
    assert!(csv.contains("APR,123.45,67.89,,2.34,50"));
}

#[test]
fn test_format_benchmark_csv_ollama_only() {
    let bench = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: None,
        ollama_tps: Some(32.0),
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: Some(150.0),
        speedup_vs_llama: None,
        speedup_vs_ollama: Some(37.5),
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let csv = format_benchmark_csv(&bench);

    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 3); // Header + APR + Ollama
    assert!(!csv.contains("llama.cpp"));
    assert!(csv.contains("Ollama"));
}

// ========================================================================
// ShowcaseResults Default Tests
// ========================================================================

#[test]
fn test_showcase_results_default_all_false() {
    let r = ShowcaseResults::default();
    assert!(!r.import);
    assert!(!r.gguf_inference);
    assert!(!r.convert);
    assert!(!r.apr_inference);
    assert!(!r.visualize);
    assert!(!r.chat);
}

#[test]
fn test_showcase_results_default_all_none() {
    let r = ShowcaseResults::default();
    assert!(r.benchmark.is_none());
    assert!(r.zram_demo.is_none());
    assert!(r.cuda_demo.is_none());
    assert!(r.brick_demo.is_none());
}

// ========================================================================
// BrickDemoResult Tests
// ========================================================================

#[test]
fn test_brick_demo_result_default() {
    let r = BrickDemoResult::default();
    assert_eq!(r.layers_measured, 0);
    assert!(r.layer_timings_us.is_empty());
    assert!(r.bottleneck.is_none());
    assert_eq!(r.total_us, 0.0);
    assert_eq!(r.tokens_per_sec, 0.0);
    assert!(!r.assertions_passed);
}

#[test]
fn test_brick_demo_result_with_data() {
    let r = BrickDemoResult {
        layers_measured: 28,
        layer_timings_us: vec![100.0, 120.0, 110.0],
        bottleneck: Some(("FfnBrick".to_string(), 120.0)),
        total_us: 3080.0,
        tokens_per_sec: 324.7,
        assertions_passed: true,
    };
    assert_eq!(r.layers_measured, 28);
    assert_eq!(r.layer_timings_us.len(), 3);
    assert!(r.bottleneck.is_some());
    let (name, time) = r.bottleneck.unwrap();
    assert_eq!(name, "FfnBrick");
    assert!((time - 120.0).abs() < 0.001);
    assert!(r.assertions_passed);
}

#[test]
fn test_brick_demo_result_clone() {
    let original = BrickDemoResult {
        layers_measured: 10,
        layer_timings_us: vec![50.0, 55.0],
        bottleneck: Some(("Attn".to_string(), 55.0)),
        total_us: 525.0,
        tokens_per_sec: 1904.8,
        assertions_passed: true,
    };
    let cloned = original.clone();
    assert_eq!(cloned.layers_measured, 10);
    assert_eq!(cloned.layer_timings_us.len(), 2);
}

// ========================================================================
// ZramDemoResult Tests
// ========================================================================

#[test]
fn test_zram_demo_result_clone() {
    let original = ZramDemoResult {
        lz4_ratio: 2.5,
        zstd_ratio: 3.2,
        zero_page_gbps: 175.0,
        lz4_gbps: 3.5,
        simd_backend: "Avx2".to_string(),
        context_extension: 2.5,
    };
    let cloned = original.clone();
    assert!((cloned.lz4_ratio - 2.5).abs() < 0.001);
    assert_eq!(cloned.simd_backend, "Avx2");
}

#[test]
fn test_zram_demo_result_context_extension_calculation() {
    // Verify context extension represents multiplier
    let result = ZramDemoResult {
        lz4_ratio: 3.0,
        zstd_ratio: 4.0,
        zero_page_gbps: 200.0,
        lz4_gbps: 5.0,
        simd_backend: "Neon".to_string(),
        context_extension: 2.5,
    };
    // 16K * 2.5 = 40K tokens
    let extended_tokens = 16_000.0 * result.context_extension;
    assert!((extended_tokens - 40_000.0).abs() < 0.1);
}

// ========================================================================
// CudaDemoResult Tests
// ========================================================================

#[test]
fn test_cuda_demo_result_clone() {
    let original = CudaDemoResult {
        device_count: 2,
        device_name: "Tesla V100".to_string(),
        total_vram_gb: 32.0,
        free_vram_gb: 28.0,
        cuda_available: true,
        graph_capture_available: true,
        graph_speedup: 50.0,
        dp4a_available: true,
        dp4a_arithmetic_intensity: 2.0,
    };
    let cloned = original.clone();
    assert_eq!(cloned.device_count, 2);
    assert_eq!(cloned.device_name, "Tesla V100");
    assert!(cloned.cuda_available);
}

#[test]
fn test_cuda_demo_result_no_gpu() {
    let result = CudaDemoResult {
        device_count: 0,
        device_name: String::new(),
        total_vram_gb: 0.0,
        free_vram_gb: 0.0,
        cuda_available: false,
        graph_capture_available: false,
        graph_speedup: 0.0,
        dp4a_available: false,
        dp4a_arithmetic_intensity: 0.0,
    };
    assert!(!result.cuda_available);
    assert_eq!(result.device_count, 0);
    assert!(!result.graph_capture_available);
    assert!(!result.dp4a_available);
}

// ========================================================================
// Validation: Single Step Mode Tests
// ========================================================================

#[test]
fn test_falsification_standalone_cuda_demo_passes() {
    let results = ShowcaseResults::default();
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::CudaDemo),
        ..Default::default()
    };
    // Standalone demos should pass without full pipeline
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_standalone_zram_demo_passes() {
    let results = ShowcaseResults::default();
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::ZramDemo),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_standalone_brick_demo_passes() {
    let results = ShowcaseResults::default();
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::BrickDemo),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_single_import_step_passes() {
    let results = ShowcaseResults {
        import: true,
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::Import),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_single_import_step_fails() {
    let results = ShowcaseResults {
        import: false,
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::Import),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_err());
}

#[test]
fn test_falsification_single_convert_step_passes() {
    let results = ShowcaseResults {
        convert: true,
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::Convert),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_single_gguf_step_fails() {
    let results = ShowcaseResults {
        gguf_inference: false,
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::GgufInference),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_err());
}

#[test]
fn test_falsification_single_apr_inference_step_fails() {
    let results = ShowcaseResults {
        apr_inference: false,
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::AprInference),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_err());
}

// ========================================================================
// Validation: Coefficient of Variation Edge Cases
// ========================================================================

#[test]
fn test_falsification_zero_tps_high_cv() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: true,
        benchmark: Some(BenchmarkComparison {
            apr_tps: 0.0,
            llama_cpp_tps: None,
            ollama_tps: None,
            apr_ttft_ms: 0.0,
            llama_cpp_ttft_ms: None,
            ollama_ttft_ms: None,
            speedup_vs_llama: None,
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.0,
            runs: 30,
        }),
        ..Default::default()
    };
    // CV = 1.0/0.0 * 100 = 100% (capped) > 5%
    assert!(super::validation::validate_falsification(&results, &full_run_config()).is_err());
}

// ========================================================================
// Validation: Multiple Failures
// ========================================================================

#[test]
fn test_falsification_multiple_step_failures() {
    let results = ShowcaseResults {
        import: false,
        gguf_inference: false,
        convert: false,
        apr_inference: false,
        benchmark: None,
        ..Default::default()
    };
    let err = super::validation::validate_falsification(&results, &full_run_config());
    assert!(err.is_err());
}
