//\! Showcase Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

use super::*;

#[test]
fn test_benchmark_stats_computation() {
    let results = vec![
        BenchmarkResult::new(128, Duration::from_millis(250), Duration::from_millis(7)),
        BenchmarkResult::new(128, Duration::from_millis(248), Duration::from_millis(6)),
        BenchmarkResult::new(128, Duration::from_millis(252), Duration::from_millis(8)),
    ];

    let stats = BenchmarkStats::from_results(results);

    assert!(stats.mean_throughput > 500.0); // ~512 tok/s
    assert!(stats.cv < 0.1); // Low variance
}

#[test]
fn test_showcase_runner_grid_generation() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
        .with_gpu_info("RTX 4090", 24.0);

    // Simulate benchmark results
    let results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(250),
        Duration::from_millis(7),
    )];
    runner.record_apr_gguf(BenchmarkStats::from_results(results));

    let grid = runner.to_grid();
    let output = grid.render();

    assert!(output.contains("APR serve GGUF"));
    assert!(output.contains("Qwen2.5-Coder"));
}

#[test]
fn test_pmat_verification() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // 500 tok/s APR vs 200 tok/s llama.cpp = 2.5x > 1.25x threshold
    let apr_results = vec![
        BenchmarkResult::new(128, Duration::from_millis(256), Duration::from_millis(7)),
        BenchmarkResult::new(128, Duration::from_millis(254), Duration::from_millis(7)),
        BenchmarkResult::new(128, Duration::from_millis(258), Duration::from_millis(7)),
    ];
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    let llamacpp_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(640),
        Duration::from_millis(30),
    )];
    runner.record_llamacpp(BenchmarkStats::from_results(llamacpp_results));

    let verification = PmatVerification::verify(&runner);

    assert!(verification.point_41_pass); // 500 > 200 * 1.25
    assert!(verification.point_42_pass); // 500 > 60
    assert!(verification.point_49_pass); // CV very low
}

#[test]
fn test_profiling_hotspots() {
    let mut collector = ProfilingCollector::new();
    collector.start();

    collector.record("Q4K_GEMV", Duration::from_millis(150), 3584);
    collector.record("Attention", Duration::from_millis(80), 3584);
    collector.record("RMSNorm", Duration::from_millis(30), 7168);
    collector.record("Other", Duration::from_millis(5), 100);

    let hotspots = collector.into_hotspots();

    assert_eq!(hotspots.len(), 3); // Q4K_GEMV, Attention, RMSNorm (>5%)
    assert_eq!(hotspots[0].component, "Q4K_GEMV"); // Sorted by percentage
}

#[test]
fn test_component_explanation() {
    let (exp, expected) = explain_component("Q4K_GEMV", 45.0);
    assert!(exp.contains("Matrix ops"));
    assert!(expected);

    let (exp, expected) = explain_component("KernelLaunch", 10.0);
    assert!(exp.contains("CUDA graphs"));
    assert!(!expected);
}

// ========================================================================
// Additional Coverage Tests for showcase.rs
// ========================================================================

#[test]
fn test_showcase_config_default_values() {
    let config = ShowcaseConfig::default();
    assert!(config.model_path.is_empty());
    assert_eq!(config.iterations, 10);
    assert_eq!(config.warmup_iterations, 3);
    assert_eq!(config.gen_tokens, 128);
    assert!(config.prompt.contains("fibonacci"));
    assert!(config.colors);
    assert!(!config.profile);
    assert!(!config.zram);
    assert_eq!(config.gpu_device, 0);
}

#[test]
fn test_showcase_config_custom_values() {
    let config = ShowcaseConfig {
        model_path: "/tmp/model.gguf".to_string(),
        iterations: 20,
        warmup_iterations: 5,
        gen_tokens: 256,
        prompt: "Hello world".to_string(),
        colors: false,
        profile: true,
        zram: true,
        gpu_device: 1,
    };
    assert_eq!(config.model_path, "/tmp/model.gguf");
    assert_eq!(config.iterations, 20);
    assert!(config.profile);
    assert!(config.zram);
}

#[test]
fn test_benchmark_result_new_zero_duration() {
    let result = BenchmarkResult::new(100, Duration::ZERO, Duration::from_millis(5));
    assert_eq!(result.tokens, 100);
    assert_eq!(result.throughput, 0.0); // Zero duration => zero throughput
    assert_eq!(result.gpu_util, None);
    assert_eq!(result.gpu_mem_mb, None);
}

#[test]
fn test_benchmark_result_with_gpu_metrics() {
    let result = BenchmarkResult::new(128, Duration::from_millis(250), Duration::from_millis(7))
        .with_gpu_metrics(85.5, 4096.0);

    assert_eq!(result.gpu_util, Some(85.5));
    assert_eq!(result.gpu_mem_mb, Some(4096.0));
}

#[test]
fn test_benchmark_stats_empty_results() {
    let stats = BenchmarkStats::from_results(vec![]);
    assert_eq!(stats.mean_throughput, 0.0);
    assert_eq!(stats.std_throughput, 0.0);
    assert_eq!(stats.mean_ttft_ms, 0.0);
    assert_eq!(stats.ci_95, (0.0, 0.0));
    assert_eq!(stats.cv, 0.0);
    assert!(stats.results.is_empty());
}

#[test]
fn test_benchmark_stats_single_result() {
    let results = vec![BenchmarkResult::new(
        100,
        Duration::from_millis(200),
        Duration::from_millis(10),
    )];
    let stats = BenchmarkStats::from_results(results);

    assert!(stats.mean_throughput > 0.0);
    assert_eq!(stats.std_throughput, 0.0); // Single result = zero variance
    assert!(stats.mean_ttft_ms > 0.0);
}

#[test]
fn test_benchmark_stats_large_sample() {
    let results: Vec<BenchmarkResult> = (0..35)
        .map(|i| {
            BenchmarkResult::new(
                100,
                Duration::from_millis(200 + i as u64),
                Duration::from_millis(5),
            )
        })
        .collect();

    let stats = BenchmarkStats::from_results(results);

    // Large sample uses 1.96 t-value
    assert!(stats.mean_throughput > 0.0);
    assert!(stats.results.len() >= 30);
}

#[test]
fn test_benchmark_stats_zero_mean_throughput() {
    let results = vec![BenchmarkResult::new(0, Duration::ZERO, Duration::ZERO)];
    let stats = BenchmarkStats::from_results(results);
    assert_eq!(stats.cv, 0.0); // No division by zero
}

#[test]
fn test_showcase_runner_new() {
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config);

    assert!(runner.apr_gguf_stats.is_none());
    assert!(runner.apr_native_stats.is_none());
    assert!(runner.ollama_stats.is_none());
    assert!(runner.llamacpp_stats.is_none());
    assert!(runner.hotspots.is_empty());
    assert!(runner.model_name.is_empty());
}

#[test]
fn test_showcase_runner_with_model_info() {
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config).with_model_info("TestModel", "7B", "Q4_K_M");

    assert_eq!(runner.model_name, "TestModel");
    assert_eq!(runner.model_params, "7B");
    assert_eq!(runner.quantization, "Q4_K_M");
}

#[test]
fn test_showcase_runner_with_gpu_info() {
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config).with_gpu_info("RTX 4090", 24.0);

    assert_eq!(runner.gpu_name, "RTX 4090");
    assert_eq!(runner.gpu_vram_gb, 24.0);
}

#[test]
fn test_showcase_runner_run_benchmark() {
    let config = ShowcaseConfig {
        iterations: 3,
        warmup_iterations: 1,
        ..Default::default()
    };
    let runner = ShowcaseRunner::new(config);

    let mut call_count = 0u32;
    let stats = runner.run_benchmark("test", || {
        call_count += 1;
        (100, Duration::from_millis(200), Duration::from_millis(10))
    });

    // Warmup (1) + iterations (3) = 4 calls
    assert_eq!(call_count, 4);
    assert_eq!(stats.results.len(), 3);
}

#[test]
fn test_showcase_runner_record_apr_native() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    let results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(200),
        Duration::from_millis(5),
    )];
    runner.record_apr_native(BenchmarkStats::from_results(results));

    assert!(runner.apr_native_stats.is_some());
}

#[test]
fn test_showcase_runner_record_ollama() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    let results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(400),
        Duration::from_millis(50),
    )];
    runner.record_ollama(BenchmarkStats::from_results(results));

    assert!(runner.ollama_stats.is_some());
}

#[test]
fn test_showcase_runner_add_hotspot() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    let hotspot = ProfilingHotspot {
        component: "MatMul".to_string(),
        time: Duration::from_millis(100),
        percentage: 45.0,
        call_count: 1000,
        avg_per_call: Duration::from_nanos(100000),
        explanation: "Matrix ops".to_string(),
        is_expected: true,
    };

    runner.add_hotspot(hotspot);
    assert_eq!(runner.hotspots.len(), 1);
}

#[test]
fn test_showcase_runner_check_point_41_pass() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // APR at 500 tok/s
    let apr_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(256),
        Duration::from_millis(7),
    )];
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    // llama.cpp at 200 tok/s
    let llamacpp_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(640),
        Duration::from_millis(30),
    )];
    runner.record_llamacpp(BenchmarkStats::from_results(llamacpp_results));

    assert!(runner.check_point_41()); // 500 >= 200 * 1.25 = 250
}

#[test]
fn test_showcase_runner_check_point_41_fail() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // APR at 100 tok/s (slower than baseline)
    let apr_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(1280),
        Duration::from_millis(50),
    )];
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    assert!(!runner.check_point_41()); // 100 < 200 * 1.25
}

#[test]
fn test_showcase_runner_check_point_41_no_stats() {
    let config = ShowcaseConfig::default();
    let runner = ShowcaseRunner::new(config);

    assert!(!runner.check_point_41()); // 0 < 200 * 1.25
}

#[test]
fn test_showcase_runner_check_2x_ollama_pass() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // APR native at 700 tok/s
    let apr_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(183),
        Duration::from_millis(5),
    )];
    runner.record_apr_native(BenchmarkStats::from_results(apr_results));

    // Ollama at 318 tok/s
    let ollama_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(402),
        Duration::from_millis(50),
    )];
    runner.record_ollama(BenchmarkStats::from_results(ollama_results));

    assert!(runner.check_2x_ollama()); // 700 >= 318 * 2 = 636
}

#[test]
fn test_showcase_runner_check_2x_ollama_with_gguf_fallback() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config);

    // Only APR GGUF at 700 tok/s (no native)
    let apr_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(183),
        Duration::from_millis(5),
    )];
    runner.record_apr_gguf(BenchmarkStats::from_results(apr_results));

    assert!(runner.check_2x_ollama()); // Uses GGUF as fallback, 700 >= 318 * 2
}

#[test]
fn test_showcase_runner_render_report() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("TestModel", "0.5B", "Q4_K_M")
        .with_gpu_info("Test GPU", 8.0);

    let results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(250),
        Duration::from_millis(7),
    )];
    runner.record_apr_gguf(BenchmarkStats::from_results(results));

    let report = runner.render_report();
    assert!(!report.is_empty());
}

#[test]
fn test_showcase_runner_to_grid_with_both_stats() {
    let config = ShowcaseConfig::default();
    let mut runner = ShowcaseRunner::new(config)
        .with_model_info("TestModel", "0.5B", "Q4_K_M")
        .with_gpu_info("Test GPU", 8.0);

    // Add APR GGUF
    let gguf_results =
        vec![
            BenchmarkResult::new(128, Duration::from_millis(256), Duration::from_millis(7))
                .with_gpu_metrics(80.0, 2048.0),
        ];
    runner.record_apr_gguf(BenchmarkStats::from_results(gguf_results));

    // Add APR native
    let native_results = vec![BenchmarkResult::new(
        128,
        Duration::from_millis(180),
        Duration::from_millis(5),
    )];
    runner.record_apr_native(BenchmarkStats::from_results(native_results));

    // Add hotspot
    let hotspot = ProfilingHotspot {
        component: "MatMul".to_string(),
        time: Duration::from_millis(100),
        percentage: 45.0,
        call_count: 1000,
        avg_per_call: Duration::from_nanos(100000),
        explanation: "Matrix ops".to_string(),
        is_expected: true,
    };
    runner.add_hotspot(hotspot);

    let grid = runner.to_grid();
    let output = grid.render();
    assert!(!output.is_empty());
}

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
