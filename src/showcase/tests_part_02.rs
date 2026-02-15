
#[test]
fn test_showcase_runner_to_grid_no_gguf_stats() {
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config).with_model_info("TestModel", "0.5B", "Q4_K_M");

    let grid = runner.to_grid();
    let output = grid.render();
    // Should still render without GGUF stats
    assert!(!output.is_empty());
}

#[test]
fn test_component_timing_clone() {
    let timing = ComponentTiming {
        name: "MatMul".to_string(),
        duration: Duration::from_millis(100),
        calls: 1000,
    };
    let cloned = timing.clone();
    assert_eq!(cloned.name, "MatMul");
    assert_eq!(cloned.calls, 1000);
}

#[test]
fn test_profiling_collector_default() {
    let collector = ProfilingCollector::default();
    assert!(collector.start.is_none());
}

#[test]
fn test_profiling_collector_empty() {
    let collector = ProfilingCollector::new();
    let hotspots = collector.into_hotspots();
    assert!(hotspots.is_empty());
}

#[test]
fn test_profiling_collector_zero_total() {
    let mut collector = ProfilingCollector::new();
    collector.record("Component", Duration::ZERO, 0);
    let hotspots = collector.into_hotspots();
    assert!(hotspots.is_empty());
}

#[test]
fn test_profiling_collector_zero_calls() {
    let mut collector = ProfilingCollector::new();
    collector.record("Component", Duration::from_millis(100), 0);
    let hotspots = collector.into_hotspots();
    // Should still create hotspot but with zero avg_per_call
    assert_eq!(hotspots.len(), 1);
    assert_eq!(hotspots[0].avg_per_call, Duration::ZERO);
}

#[test]
fn test_explain_component_all_branches() {
    // MatMul variants
    let (exp, is_exp) = explain_component("MatMul", 40.0);
    assert!(exp.contains("Matrix ops"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("GEMM", 30.0);
    assert!(exp.contains("Matrix ops"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("TensorCore", 25.0);
    assert!(exp.contains("Matrix ops"));
    assert!(is_exp);

    // Attention variants
    let (exp, is_exp) = explain_component("Attention", 20.0);
    assert!(exp.contains("Attention"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("FlashAttention", 15.0);
    assert!(exp.contains("Attention"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("IncrementalAttention", 10.0);
    assert!(exp.contains("Attention"));
    assert!(is_exp);

    // Normalization - normal range
    let (exp, is_exp) = explain_component("RMSNorm", 10.0);
    assert!(exp.contains("normal range"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("LayerNorm", 8.0);
    assert!(exp.contains("normal range"));
    assert!(is_exp);

    // Normalization - high
    let (exp, is_exp) = explain_component("RMSNorm", 20.0);
    assert!(exp.contains("megakernel"));
    assert!(!is_exp);

    // Memory transfer
    let (exp, is_exp) = explain_component("MemcpyH2D", 10.0);
    assert!(exp.contains("persistent buffers"));
    assert!(!is_exp);

    let (exp, is_exp) = explain_component("MemcpyD2H", 10.0);
    assert!(exp.contains("persistent buffers"));
    assert!(!is_exp);

    let (exp, is_exp) = explain_component("Transfer", 10.0);
    assert!(exp.contains("persistent buffers"));
    assert!(!is_exp);

    // KV Cache - normal
    let (exp, is_exp) = explain_component("KVCache", 15.0);
    assert!(exp.contains("normal range"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("KV_Cache", 10.0);
    assert!(exp.contains("normal range"));
    assert!(is_exp);

    // KV Cache - high
    let (exp, is_exp) = explain_component("KVCache", 25.0);
    assert!(exp.contains("ZRAM"));
    assert!(!is_exp);

    // FFN variants
    let (exp, is_exp) = explain_component("SwiGLU", 15.0);
    assert!(exp.contains("FFN"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("FFN", 12.0);
    assert!(exp.contains("FFN"));
    assert!(is_exp);

    // Embedding
    let (exp, is_exp) = explain_component("Embedding", 5.0);
    assert!(exp.contains("Embedding"));
    assert!(is_exp);

    // Sampling
    let (exp, is_exp) = explain_component("Sampling", 3.0);
    assert!(exp.contains("Sampling"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("TopK", 2.0);
    assert!(exp.contains("Sampling"));
    assert!(is_exp);

    let (exp, is_exp) = explain_component("TopP", 2.0);
    assert!(exp.contains("Sampling"));
    assert!(is_exp);

    // Unknown - low percentage
    let (exp, is_exp) = explain_component("UnknownComponent", 10.0);
    assert!(exp.is_empty());
    assert!(is_exp);

    // Unknown - high percentage
    let (exp, is_exp) = explain_component("UnknownComponent", 25.0);
    assert!(exp.contains("investigate"));
    assert!(!is_exp);
}

#[test]
fn test_pmat_verification_all_fail() {
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config);

    let verification = PmatVerification::verify(&runner);

    assert!(!verification.point_41_pass); // 0 < 200 * 1.25
    assert!(!verification.point_42_pass); // 0 < 60
    assert!(!verification.point_49_pass); // cv = 1.0 >= 0.05
    assert!(!verification.ollama_2x_pass); // 0 < 318 * 2
    assert!(!verification.all_pass);
}

#[test]
fn test_pmat_verification_2x_ollama_pass() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // APR at 700 tok/s
    let apr_results = vec![
        BenchmarkResult::new(128, Duration::from_millis(183), Duration::from_millis(5)),
        BenchmarkResult::new(128, Duration::from_millis(182), Duration::from_millis(5)),
        BenchmarkResult::new(128, Duration::from_millis(184), Duration::from_millis(5)),
    ];
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let verification = PmatVerification::verify(&runner);
    assert!(verification.ollama_2x_pass);
}

#[test]
fn test_pmat_verification_to_report() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    let apr_results = vec![
        BenchmarkResult::new(128, Duration::from_millis(200), Duration::from_millis(5)),
        BenchmarkResult::new(128, Duration::from_millis(200), Duration::from_millis(5)),
    ];
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let verification = PmatVerification::verify(&runner);
    let report = verification.to_report();

    assert!(report.contains("PMAT Verification"));
    assert!(report.contains("Point 41"));
    assert!(report.contains("Point 42"));
    assert!(report.contains("Point 49"));
    assert!(report.contains("2x Ollama"));
    assert!(report.contains("Overall"));
}

#[test]
fn test_pmat_verification_to_report_all_pass() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // Very high throughput with low variance
    let apr_results: Vec<BenchmarkResult> = (0..10)
        .map(|_| BenchmarkResult::new(128, Duration::from_millis(100), Duration::from_millis(5)))
        .collect();
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let verification = PmatVerification::verify(&runner);
    let report = verification.to_report();

    assert!(report.contains("PASS"));
}

#[test]
fn test_zram_config_default() {
    let config = zram::ZramConfig::default();
    assert!(config.adaptive);
    assert!((config.min_savings - 0.1).abs() < 0.01);
}

#[test]
#[cfg(not(feature = "showcase-zram"))]
fn test_zram_config_custom_stub() {
    let config = zram::ZramConfig {
        algorithm: "zstd".to_string(),
        adaptive: false,
        min_savings: 0.2,
    };
    assert_eq!(config.algorithm, "zstd");
    assert!(!config.adaptive);
}

#[test]
#[cfg(feature = "showcase-zram")]
fn test_zram_config_custom_real() {
    use zram::ZramAlgorithm;
    let config = zram::ZramConfig {
        algorithm: ZramAlgorithm::Zstd { level: 3 },
        adaptive: false,
        min_savings: 0.2,
    };
    assert!(!config.adaptive);
    assert!((config.min_savings - 0.2).abs() < 0.01);
}

#[test]
#[cfg(not(feature = "showcase-zram"))]
fn test_zram_result_fields_stub() {
    let result = zram::ZramResult {
        original_size: 4096,
        compressed_size: 2048,
        ratio: 2.0,
        zero_page: false,
    };
    assert_eq!(result.original_size, 4096);
    assert_eq!(result.compressed_size, 2048);
    assert_eq!(result.ratio, 2.0);
    assert!(!result.zero_page);
}

#[test]
#[cfg(feature = "showcase-zram")]
fn test_zram_result_fields_real() {
    let result = zram::ZramResult {
        original_size: 4096,
        compressed_size: 2048,
        ratio: 2.0,
        algorithm: "zstd".to_string(),
        zero_page: false,
    };
    assert_eq!(result.original_size, 4096);
    assert_eq!(result.compressed_size, 2048);
    assert_eq!(result.ratio, 2.0);
    assert_eq!(result.algorithm, "zstd");
    assert!(!result.zero_page);
}

#[test]
fn test_profiler_config_default() {
    let config = profiler::RenacerProfilerConfig::default();
    // Stub module has zeroed defaults
    assert_eq!(config.device_id, 0);
}

#[test]
fn test_profiler_config_custom() {
    let config = profiler::RenacerProfilerConfig {
        threshold_us: 50,
        trace_all: true,
        device_id: 1,
    };
    assert_eq!(config.threshold_us, 50);
    assert!(config.trace_all);
    assert_eq!(config.device_id, 1);
}

#[test]
fn test_benchmark_result_debug() {
    let result = BenchmarkResult::new(128, Duration::from_millis(250), Duration::from_millis(7));
    let debug_str = format!("{:?}", result);
    assert!(debug_str.contains("128"));
}

#[test]
fn test_benchmark_stats_debug() {
    let results = vec![BenchmarkResult::new(
        100,
        Duration::from_millis(200),
        Duration::from_millis(10),
    )];
    let stats = BenchmarkStats::from_results(results);
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("mean_throughput"));
}

#[test]
fn test_showcase_config_debug() {
    let config = ShowcaseConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("iterations"));
}

#[test]
fn test_showcase_runner_debug() {
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config);
    let debug_str = format!("{:?}", runner);
    assert!(debug_str.contains("ShowcaseRunner"));
}

#[test]
fn test_pmat_verification_debug() {
    let verification = PmatVerification {
        point_41_pass: true,
        point_42_pass: true,
        point_49_pass: true,
        ollama_2x_pass: false,
        all_pass: true,
    };
    let debug_str = format!("{:?}", verification);
    assert!(debug_str.contains("point_41_pass"));
}

#[test]
fn test_pmat_verification_clone() {
    let verification = PmatVerification {
        point_41_pass: true,
        point_42_pass: false,
        point_49_pass: true,
        ollama_2x_pass: true,
        all_pass: false,
    };
    let cloned = verification.clone();
    assert_eq!(cloned.point_41_pass, verification.point_41_pass);
    assert_eq!(cloned.all_pass, verification.all_pass);
}

#[test]
fn test_benchmark_result_clone() {
    let result = BenchmarkResult::new(128, Duration::from_millis(250), Duration::from_millis(7))
        .with_gpu_metrics(80.0, 2048.0);
    let cloned = result.clone();
    assert_eq!(cloned.tokens, 128);
    assert_eq!(cloned.gpu_util, Some(80.0));
}

#[test]
fn test_benchmark_stats_clone() {
    let results = vec![BenchmarkResult::new(
        100,
        Duration::from_millis(200),
        Duration::from_millis(10),
    )];
    let stats = BenchmarkStats::from_results(results);
    let cloned = stats.clone();
    assert_eq!(cloned.results.len(), stats.results.len());
}

#[test]
fn test_showcase_config_clone() {
    let config = ShowcaseConfig {
        model_path: "/tmp/test.gguf".to_string(),
        iterations: 5,
        ..Default::default()
    };
    let cloned = config.clone();
    assert_eq!(cloned.model_path, "/tmp/test.gguf");
    assert_eq!(cloned.iterations, 5);
}

// ========================================================================
// Coverage Tests for to_grid() branches - Issue: uncovered closure branches
// ========================================================================

#[test]
fn test_showcase_runner_to_grid_with_ollama_and_llamacpp_stats() {
    // Cover lines 327, 336: closure branches for Some(ollama_stats) and Some(llamacpp_stats)
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("TestModel", "0.5B", "Q4_K_M")
        .with_gpu_info("Test GPU", 8.0);

    // Add APR GGUF with GPU metrics
    let gguf_results =
        vec![
            BenchmarkResult::new(128, Duration::from_millis(256), Duration::from_millis(7))
                .with_gpu_metrics(80.0, 2048.0),
        ];
    runner.record_apr_gguf(BenchmarkStats::from_results(gguf_results));

    // Add Ollama stats (covers line 327 - Some(s) branch)
    let ollama_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(400),
        Duration::from_millis(50),
    )];
    runner.record_ollama(BenchmarkStats::from_results(ollama_results));

    // Add llama.cpp stats (covers line 336 - Some(s) branch)
    let llamacpp_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(640),
        Duration::from_millis(30),
    )];
    runner.record_llamacpp(BenchmarkStats::from_results(llamacpp_results));

    let grid = runner.to_grid();
    let output = grid.render();
    assert!(!output.is_empty());
}
