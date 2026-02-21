
// ========================================================================
// NEW: print_header coverage (validation.rs lines 8-37)
// ========================================================================

#[test]
fn test_print_header_tiny() {
    super::validation::print_header(ModelTier::Tiny);
}

#[test]
fn test_print_header_small() {
    super::validation::print_header(ModelTier::Small);
}

#[test]
fn test_print_header_medium() {
    super::validation::print_header(ModelTier::Medium);
}

#[test]
fn test_print_header_large() {
    super::validation::print_header(ModelTier::Large);
}

// ========================================================================
// NEW: print_summary coverage (validation.rs lines 39-96)
// ========================================================================

#[test]
fn test_print_summary_all_pass() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: true,
        benchmark: Some(BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: Some(32.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: Some(150.0),
            speedup_vs_llama: Some(25.7),
            speedup_vs_ollama: Some(37.5),
            apr_tps_stddev: 1.5,
            runs: 30,
        }),
        visualize: true,
        chat: true,
        zram_demo: Some(ZramDemoResult {
            lz4_ratio: 2.5,
            zstd_ratio: 3.2,
            zero_page_gbps: 175.0,
            lz4_gbps: 3.5,
            simd_backend: "Avx2".to_string(),
            context_extension: 2.5,
        }),
        cuda_demo: Some(CudaDemoResult {
            device_count: 1,
            device_name: "RTX 4090".to_string(),
            total_vram_gb: 24.0,
            free_vram_gb: 20.0,
            cuda_available: true,
            graph_capture_available: true,
            graph_speedup: 70.0,
            dp4a_available: true,
            dp4a_arithmetic_intensity: 1.78,
        }),
        brick_demo: None,
    };
    let config = ShowcaseConfig::default();
    // Should not panic
    super::validation::print_summary(&results, &config);
}

#[test]
fn test_print_summary_all_fail() {
    let results = ShowcaseResults::default();
    let config = ShowcaseConfig::default();
    super::validation::print_summary(&results, &config);
}

#[test]
fn test_print_summary_no_benchmark() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: false,
        convert: true,
        apr_inference: false,
        benchmark: None,
        visualize: false,
        chat: false,
        zram_demo: None,
        cuda_demo: None,
        brick_demo: None,
    };
    let config = ShowcaseConfig::default();
    super::validation::print_summary(&results, &config);
}

#[test]
fn test_print_summary_benchmark_below_threshold() {
    // Exercises the speedup check branches where pass = false
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: true,
        benchmark: Some(BenchmarkComparison {
            apr_tps: 36.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: Some(32.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: Some(150.0),
            speedup_vs_llama: Some(2.9),   // Below 25%
            speedup_vs_ollama: Some(12.5), // Below 25%
            apr_tps_stddev: 1.5,
            runs: 30,
        }),
        visualize: true,
        chat: true,
        zram_demo: None,
        cuda_demo: None,
        brick_demo: None,
    };
    let config = ShowcaseConfig::default();
    super::validation::print_summary(&results, &config);
}

#[test]
fn test_print_summary_zram_below_thresholds() {
    // Exercises ZRAM summary with failing checks
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: true,
        benchmark: None,
        visualize: false,
        chat: false,
        zram_demo: Some(ZramDemoResult {
            lz4_ratio: 1.5, // Below 2.0 threshold
            zstd_ratio: 1.8,
            zero_page_gbps: 100.0, // Below 150.0 threshold
            lz4_gbps: 2.0,
            simd_backend: "Scalar".to_string(),
            context_extension: 1.5, // Below 2.0 threshold
        }),
        cuda_demo: None,
        brick_demo: None,
    };
    let config = ShowcaseConfig::default();
    super::validation::print_summary(&results, &config);
}

#[test]
fn test_print_summary_cuda_no_device() {
    let results = ShowcaseResults {
        import: false,
        gguf_inference: false,
        convert: false,
        apr_inference: false,
        benchmark: None,
        visualize: false,
        chat: false,
        zram_demo: None,
        cuda_demo: Some(CudaDemoResult {
            device_count: 0,
            device_name: "N/A".to_string(),
            total_vram_gb: 0.0,
            free_vram_gb: 0.0,
            cuda_available: false,
            graph_capture_available: false,
            graph_speedup: 1.0,
            dp4a_available: false,
            dp4a_arithmetic_intensity: 0.0,
        }),
        brick_demo: None,
    };
    let config = ShowcaseConfig::default();
    super::validation::print_summary(&results, &config);
}

// ========================================================================
// NEW: mod.rs run() coverage - no step path (mod.rs lines 49-67)
// ========================================================================

#[test]
fn test_run_no_step_no_auto_verify() {
    // This path prints available steps and returns Ok(())
    let config = ShowcaseConfig {
        step: None,
        auto_verify: false,
        ..Default::default()
    };
    let result = super::run(&config);
    assert!(result.is_ok());
}

// ========================================================================
// NEW: validate_falsification single Benchmark step (validation.rs)
// ========================================================================

#[test]
fn test_falsification_single_benchmark_step_no_data() {
    // Benchmark step explicitly requested but no data collected
    let results = ShowcaseResults::default();
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::Benchmark),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_err());
}

#[test]
fn test_falsification_single_benchmark_step_valid() {
    let results = ShowcaseResults {
        benchmark: Some(BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: None,
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: None,
            speedup_vs_llama: Some(25.7),
            speedup_vs_ollama: None,
            apr_tps_stddev: 1.5,
            runs: 30,
        }),
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::Benchmark),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_single_visualize_step_passes() {
    // Visualize step is not validated strictly - should pass even with defaults
    let results = ShowcaseResults::default();
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::Visualize),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_single_chat_step_passes() {
    let results = ShowcaseResults::default();
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::Chat),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

// ========================================================================
// NEW: validate_falsification --step all (treated like full run)
// ========================================================================

#[test]
fn test_falsification_step_all_passes() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: true,
        benchmark: Some(BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: Some(32.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: Some(150.0),
            speedup_vs_llama: Some(25.7),
            speedup_vs_ollama: Some(37.5),
            apr_tps_stddev: 1.5,
            runs: 30,
        }),
        visualize: true,
        chat: true,
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::All),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_ok());
}

#[test]
fn test_falsification_step_all_fails_with_failures() {
    let results = ShowcaseResults {
        import: false,
        ..Default::default()
    };
    let config = ShowcaseConfig {
        step: Some(ShowcaseStep::All),
        ..Default::default()
    };
    assert!(super::validation::validate_falsification(&results, &config).is_err());
}

// ========================================================================
// NEW: validate_falsification both baselines fail (Points 41+42)
// ========================================================================

#[test]
fn test_falsification_both_speedups_below_25() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: true,
        benchmark: Some(BenchmarkComparison {
            apr_tps: 37.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: Some(36.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(80.0),
            ollama_ttft_ms: Some(82.0),
            speedup_vs_llama: Some(5.7),  // Below 25%
            speedup_vs_ollama: Some(2.8), // Below 25%
            apr_tps_stddev: 1.0,
            runs: 30,
        }),
        ..Default::default()
    };
    assert!(validate_falsification(&results, &full_run_config()).is_err());
}

#[test]
fn test_falsification_ollama_speedup_below_25_llama_passes() {
    let results = ShowcaseResults {
        import: true,
        gguf_inference: true,
        convert: true,
        apr_inference: true,
        benchmark: Some(BenchmarkComparison {
            apr_tps: 44.0,
            llama_cpp_tps: Some(35.0),
            ollama_tps: Some(42.0),
            apr_ttft_ms: 78.0,
            llama_cpp_ttft_ms: Some(120.0),
            ollama_ttft_ms: Some(80.0),
            speedup_vs_llama: Some(25.7), // Passes
            speedup_vs_ollama: Some(4.8), // Fails
            apr_tps_stddev: 1.5,
            runs: 30,
        }),
        ..Default::default()
    };
    // Ollama below threshold means failure
    assert!(validate_falsification(&results, &full_run_config()).is_err());
}

// ========================================================================
// NEW: run_benchmark non-inference path (benchmark.rs lines 402-445)
// ========================================================================

#[test]
#[cfg(not(feature = "inference"))]
fn test_run_benchmark_no_inference_no_baselines() {
    let config = ShowcaseConfig {
        baselines: vec![], // No baselines to avoid system calls
        ..Default::default()
    };
    let result = super::benchmark::run_benchmark(&config);
    assert!(result.is_ok());
    let bench = result.unwrap();
    // Non-inference path uses simulated values around 44.0 tok/s
    assert!(bench.apr_tps > 0.0);
    assert!(bench.apr_ttft_ms > 0.0);
    assert_eq!(bench.apr_tps_stddev, 2.0);
    assert_eq!(bench.runs, 30);
    assert!(bench.llama_cpp_tps.is_none());
    assert!(bench.ollama_tps.is_none());
}

// ========================================================================
// NEW: run_visualize non-visualization path (demo.rs lines 67-87)
// ========================================================================

#[test]
#[cfg(not(feature = "visualization"))]
fn test_run_visualize_no_visualization_feature() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let result = super::demo::run_visualize(&config, None);
    assert!(result.is_ok());
    assert!(result.unwrap());
    // Should have created an SVG file
    let svg_path = temp_dir.path().join("showcase-performance.svg");
    assert!(svg_path.exists());
    let content = std::fs::read_to_string(&svg_path).unwrap();
    assert!(content.contains("<svg"));
    assert!(content.contains("APR Inference"));
}

#[test]
#[cfg(not(feature = "visualization"))]
fn test_run_visualize_with_benchmark_data() {
    let temp_dir = tempfile::tempdir().unwrap();
    let config = ShowcaseConfig {
        model_dir: temp_dir.path().to_path_buf(),
        ..Default::default()
    };
    let bench = BenchmarkComparison {
        apr_tps: 44.0,
        llama_cpp_tps: Some(35.0),
        ollama_tps: None,
        apr_ttft_ms: 78.0,
        llama_cpp_ttft_ms: Some(120.0),
        ollama_ttft_ms: None,
        speedup_vs_llama: Some(25.7),
        speedup_vs_ollama: None,
        apr_tps_stddev: 1.5,
        runs: 30,
    };
    let result = super::demo::run_visualize(&config, Some(&bench));
    assert!(result.is_ok());
}
