
// ========================================================================
// Export: Default Path Tests
// ========================================================================

#[test]
fn test_export_json_default_path() {
    let bench = BenchmarkComparison {
        apr_tps: 50.0,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 60.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 1.0,
        runs: 30,
    };

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        export_format: ExportFormat::Json,
        export_path: None, // Should use default
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    export_benchmark_results(&bench, &config).unwrap();

    // Default path should be model_dir/benchmark-results.json
    let expected_path = temp_dir.path().join("benchmark-results.json");
    assert!(expected_path.exists());
}

#[test]
fn test_export_csv_default_path() {
    let bench = BenchmarkComparison {
        apr_tps: 50.0,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 60.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 1.0,
        runs: 30,
    };

    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        export_format: ExportFormat::Csv,
        export_path: None,
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };

    export_benchmark_results(&bench, &config).unwrap();

    let expected_path = temp_dir.path().join("benchmark-results.csv");
    assert!(expected_path.exists());
}

// ========================================================================
// NEW: print_benchmark_results coverage (benchmark.rs lines 592-646)
// ========================================================================

#[test]
fn test_print_benchmark_results_apr_only() {
    // Exercises the APR-only path (no baselines, no speedup annotations)
    let comparison = BenchmarkComparison {
        apr_tps: 50.0,
        apr_ttft_ms: 70.0,
        apr_tps_stddev: 2.0,
        runs: 30,
        llama_cpp_tps: None,
        llama_cpp_ttft_ms: None,
        ollama_tps: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
    };
    // Should not panic
    super::benchmark::print_benchmark_results(&comparison);
}

#[test]
fn test_print_benchmark_results_with_llama_cpp() {
    // Exercises llama.cpp row + speedup annotation
    let comparison = BenchmarkComparison {
        apr_tps: 50.0,
        apr_ttft_ms: 70.0,
        apr_tps_stddev: 2.0,
        runs: 30,
        llama_cpp_tps: Some(35.0),
        llama_cpp_ttft_ms: Some(120.0),
        ollama_tps: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: Some(42.9),
        speedup_vs_ollama: None,
    };
    super::benchmark::print_benchmark_results(&comparison);
}

#[test]
fn test_print_benchmark_results_with_ollama() {
    // Exercises Ollama row + speedup annotation
    let comparison = BenchmarkComparison {
        apr_tps: 50.0,
        apr_ttft_ms: 70.0,
        apr_tps_stddev: 2.0,
        runs: 30,
        llama_cpp_tps: None,
        llama_cpp_ttft_ms: None,
        ollama_tps: Some(32.0),
        ollama_ttft_ms: Some(150.0),
        speedup_vs_llama: None,
        speedup_vs_ollama: Some(56.25),
    };
    super::benchmark::print_benchmark_results(&comparison);
}

#[test]
fn test_print_benchmark_results_all_baselines() {
    // Exercises both baseline rows and both speedup annotations
    let comparison = BenchmarkComparison {
        apr_tps: 50.0,
        apr_ttft_ms: 70.0,
        apr_tps_stddev: 2.0,
        runs: 30,
        llama_cpp_tps: Some(35.0),
        llama_cpp_ttft_ms: Some(120.0),
        ollama_tps: Some(32.0),
        ollama_ttft_ms: Some(150.0),
        speedup_vs_llama: Some(42.9),
        speedup_vs_ollama: Some(56.25),
    };
    super::benchmark::print_benchmark_results(&comparison);
}

#[test]
fn test_print_benchmark_results_speedup_below_target() {
    // Exercises the FAIL branch for speedup < 25%
    let comparison = BenchmarkComparison {
        apr_tps: 38.0,
        apr_ttft_ms: 90.0,
        apr_tps_stddev: 3.0,
        runs: 30,
        llama_cpp_tps: Some(35.0),
        llama_cpp_ttft_ms: Some(100.0),
        ollama_tps: Some(36.0),
        ollama_ttft_ms: Some(110.0),
        speedup_vs_llama: Some(8.6),  // Below 25%
        speedup_vs_ollama: Some(5.6), // Below 25%
    };
    super::benchmark::print_benchmark_results(&comparison);
}

#[test]
fn test_print_benchmark_results_llama_no_ttft() {
    // Exercises llama.cpp with None ttft (defaults to 0.0)
    let comparison = BenchmarkComparison {
        apr_tps: 50.0,
        apr_ttft_ms: 70.0,
        apr_tps_stddev: 2.0,
        runs: 30,
        llama_cpp_tps: Some(35.0),
        llama_cpp_ttft_ms: None,
        ollama_tps: Some(32.0),
        ollama_ttft_ms: None,
        speedup_vs_llama: Some(42.9),
        speedup_vs_ollama: Some(56.25),
    };
    super::benchmark::print_benchmark_results(&comparison);
}

// ========================================================================
// NEW: extract_json_field edge cases (benchmark.rs lines 469-482)
// ========================================================================

#[test]
fn test_extract_json_field_zero_value() {
    let json = r#"{"count":0}"#;
    let val = super::benchmark::extract_json_field(json, "count");
    assert_eq!(val, Some(0.0));
}

#[test]
fn test_extract_json_field_large_value() {
    let json = r#"{"duration":999999999999}"#;
    let val = super::benchmark::extract_json_field(json, "duration");
    assert_eq!(val, Some(999_999_999_999.0));
}

#[test]
fn test_extract_json_field_trailing_comma() {
    let json = r#"{"value":42,"other":99}"#;
    let val = super::benchmark::extract_json_field(json, "value");
    assert_eq!(val, Some(42.0));
}

#[test]
fn test_extract_json_field_string_value_returns_none() {
    // Non-numeric value after field name
    let json = r#"{"name":"hello"}"#;
    let val = super::benchmark::extract_json_field(json, "name");
    // "hello" starts with '"' which is not digit/dot, so empty parse -> None
    assert!(val.is_none());
}

#[test]
fn test_extract_json_field_boolean_returns_none() {
    let json = r#"{"enabled":true}"#;
    let val = super::benchmark::extract_json_field(json, "enabled");
    // 't' is not digit/dot
    assert!(val.is_none());
}

#[test]
fn test_extract_json_field_null_returns_none() {
    let json = r#"{"value":null}"#;
    let val = super::benchmark::extract_json_field(json, "value");
    // 'n' is not digit/dot
    assert!(val.is_none());
}

#[test]
fn test_extract_json_field_decimal_leading_dot() {
    // ".5" is a valid float parse
    let json = r#"{"value":.5}"#;
    let val = super::benchmark::extract_json_field(json, "value");
    assert_eq!(val, Some(0.5));
}

#[test]
fn test_extract_json_field_real_ollama_response() {
    // Realistic Ollama API response format
    let json = r#"{"model":"qwen2.5-coder:1.5b","created_at":"2025-01-09T12:00:00Z","response":"def hello()","done":true,"total_duration":1234567890,"load_duration":100000000,"prompt_eval_count":10,"prompt_eval_duration":200000000,"eval_count":25,"eval_duration":900000000}"#;
    let eval_count = super::benchmark::extract_json_field(json, "eval_count");
    assert_eq!(eval_count, Some(25.0));
    let eval_duration = super::benchmark::extract_json_field(json, "eval_duration");
    assert_eq!(eval_duration, Some(900_000_000.0));
    let prompt_eval_duration = super::benchmark::extract_json_field(json, "prompt_eval_duration");
    assert_eq!(prompt_eval_duration, Some(200_000_000.0));
}

// ========================================================================
// NEW: format_benchmark_csv edge cases (benchmark.rs lines 69-103)
// ========================================================================

#[test]
fn test_format_benchmark_csv_llama_no_speedup() {
    // llama.cpp present but no speedup calculated
    let bench = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: Some(35.0),
        ollama_tps: None,
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: Some(120.0),
        ollama_ttft_ms: None,
        speedup_vs_llama: None, // No speedup
        speedup_vs_ollama: None,
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let csv = format_benchmark_csv(&bench);
    // llama row should have empty speedup field
    assert!(csv.contains("llama.cpp,35.00,120.00,,N/A,N/A"));
}

#[test]
fn test_format_benchmark_csv_llama_zero_ttft() {
    // llama.cpp with None ttft defaults to 0.0
    let bench = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: Some(35.0),
        ollama_tps: None,
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: None, // defaults to 0.0
        ollama_ttft_ms: None,
        speedup_vs_llama: Some(25.7),
        speedup_vs_ollama: None,
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let csv = format_benchmark_csv(&bench);
    assert!(csv.contains("llama.cpp,35.00,0.00,25.70,N/A,N/A"));
}

#[test]
fn test_format_benchmark_csv_ollama_zero_ttft() {
    // Ollama with None ttft defaults to 0.0
    let bench = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: None,
        ollama_tps: Some(32.0),
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None, // defaults to 0.0
        speedup_vs_llama: None,
        speedup_vs_ollama: Some(37.5),
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let csv = format_benchmark_csv(&bench);
    assert!(csv.contains("Ollama,32.00,0.00,37.50,N/A,N/A"));
}

#[test]
fn test_format_benchmark_csv_ollama_no_speedup() {
    let bench = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: None,
        ollama_tps: Some(32.0),
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: Some(150.0),
        speedup_vs_llama: None,
        speedup_vs_ollama: None, // No speedup
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let csv = format_benchmark_csv(&bench);
    // Ollama row should have empty speedup
    assert!(csv.contains("Ollama,32.00,150.00,,N/A,N/A"));
}

#[test]
fn test_format_benchmark_csv_zero_values() {
    let bench = BenchmarkComparison {
        apr_tps: 0.0,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 0.0,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 0.0,
        runs: 0,
    };
    let csv = format_benchmark_csv(&bench);
    assert!(csv.contains("APR,0.00,0.00,,0.00,0"));
}

#[test]
fn test_format_benchmark_csv_high_precision_values() {
    let bench = BenchmarkComparison {
        apr_tps: 123.456789,
        llama_cpp_tps: None,
        ollama_tps: None,
        apr_ttft_ms: 0.001,
        llama_cpp_ttft_ms: None,
        ollama_ttft_ms: None,
        speedup_vs_llama: None,
        speedup_vs_ollama: None,
        apr_tps_stddev: 0.12345,
        runs: 1,
    };
    let csv = format_benchmark_csv(&bench);
    // Should be formatted to 2 decimal places
    assert!(csv.contains("APR,123.46,0.00,,0.12,1"));
}

// ========================================================================
// NEW: run_chat coverage (demo.rs line 372-382)
// ========================================================================

#[test]
fn test_run_chat_returns_ok_true() {
    let config = ShowcaseConfig::default();
    let result = super::demo::run_chat(&config);
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_run_chat_with_all_tiers() {
    for tier in &[
        ModelTier::Tiny,
        ModelTier::Small,
        ModelTier::Medium,
        ModelTier::Large,
    ] {
        let config = ShowcaseConfig::with_tier(*tier);
        let result = super::demo::run_chat(&config);
        assert!(result.is_ok());
    }
}

// ========================================================================
// NEW: run_brick_demo non-inference path (demo.rs lines 1159-1167)
// ========================================================================

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_brick_demo_no_inference_feature() {
    let config = ShowcaseConfig::default();
    let result = super::demo::run_brick_demo(&config);
    assert!(result.is_ok());
    let demo_result = result.unwrap();
    assert_eq!(demo_result.layers_measured, 0);
    assert!(demo_result.layer_timings_us.is_empty());
    assert!(demo_result.bottleneck.is_none());
    assert_eq!(demo_result.total_us, 0.0);
    assert_eq!(demo_result.tokens_per_sec, 0.0);
    assert!(!demo_result.assertions_passed);
}

// ========================================================================
// NEW: run_cuda_demo non-cuda path (demo.rs lines 789-805)
// ========================================================================

#[test]
#[cfg(not(feature = "cuda"))]
fn test_run_cuda_demo_no_cuda_feature() {
    let config = ShowcaseConfig::default();
    let result = run_cuda_demo(&config);
    assert!(result.is_ok());
    let cuda_result = result.unwrap();
    assert_eq!(cuda_result.device_count, 0);
    assert_eq!(cuda_result.device_name, "disabled");
    assert_eq!(cuda_result.total_vram_gb, 0.0);
    assert_eq!(cuda_result.free_vram_gb, 0.0);
    assert!(!cuda_result.cuda_available);
    assert!(!cuda_result.graph_capture_available);
    assert!((cuda_result.graph_speedup - 1.0).abs() < f64::EPSILON);
    assert!(!cuda_result.dp4a_available);
    assert_eq!(cuda_result.dp4a_arithmetic_intensity, 0.0);
}

// ========================================================================
// NEW: run_zram_demo non-zram path (demo.rs lines 571-584)
// ========================================================================

#[test]
#[cfg(not(feature = "zram"))]
fn test_run_zram_demo_no_zram_feature() {
    let config = ShowcaseConfig::default();
    let result = run_zram_demo(&config);
    assert!(result.is_ok());
    let zram_result = result.unwrap();
    assert_eq!(zram_result.lz4_ratio, 0.0);
    assert_eq!(zram_result.zstd_ratio, 0.0);
    assert_eq!(zram_result.zero_page_gbps, 0.0);
    assert_eq!(zram_result.lz4_gbps, 0.0);
    assert_eq!(zram_result.simd_backend, "disabled");
    assert_eq!(zram_result.context_extension, 0.0);
}
