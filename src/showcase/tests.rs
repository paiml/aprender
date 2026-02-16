//\! Showcase Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

pub(crate) use super::*;

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

#[path = "tests_part_02.rs"]

mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
