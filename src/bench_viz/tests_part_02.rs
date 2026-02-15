
#[test]
fn test_truncate_exact_length() {
    let s = "hello";
    assert_eq!(truncate(s, 5), "hello");
}

#[test]
fn test_explain_inference_hotspot_matmul() {
    let (explanation, is_expected) = explain_inference_hotspot("MatMul", 50.0);
    assert!(explanation.contains("Matrix ops"));
    assert!(is_expected);
}

#[test]
fn test_explain_inference_hotspot_attention() {
    let (explanation, is_expected) = explain_inference_hotspot("Attention", 30.0);
    assert!(explanation.contains("Attention"));
    assert!(is_expected);
}

#[test]
fn test_explain_inference_hotspot_kv_cache_normal() {
    let (explanation, is_expected) = explain_inference_hotspot("KV_Cache", 15.0);
    assert!(explanation.contains("normal"));
    assert!(is_expected);
}

#[test]
fn test_explain_inference_hotspot_kv_cache_high() {
    let (explanation, is_expected) = explain_inference_hotspot("KVCache", 25.0);
    assert!(explanation.contains("high"));
    assert!(!is_expected);
}

#[test]
fn test_explain_inference_hotspot_softmax_normal() {
    let (explanation, is_expected) = explain_inference_hotspot("Softmax", 5.0);
    assert!(explanation.contains("normal"));
    assert!(is_expected);
}

#[test]
fn test_explain_inference_hotspot_softmax_high() {
    let (explanation, is_expected) = explain_inference_hotspot("Softmax", 15.0);
    assert!(explanation.contains("high"));
    assert!(!is_expected);
}

#[test]
fn test_explain_inference_hotspot_rmsnorm_normal() {
    let (explanation, is_expected) = explain_inference_hotspot("RMSNorm", 10.0);
    assert!(explanation.contains("normal"));
    assert!(is_expected);
}

#[test]
fn test_explain_inference_hotspot_rmsnorm_high() {
    let (explanation, is_expected) = explain_inference_hotspot("LayerNorm", 20.0);
    assert!(explanation.contains("high"));
    assert!(!is_expected);
}

#[test]
fn test_explain_inference_hotspot_memcpy() {
    let (explanation, is_expected) = explain_inference_hotspot("MemcpyH2D", 10.0);
    assert!(explanation.contains("Memory transfer"));
    assert!(!is_expected);
}

#[test]
fn test_explain_inference_hotspot_kernel_launch() {
    let (explanation, is_expected) = explain_inference_hotspot("KernelLaunch", 10.0);
    assert!(explanation.contains("Kernel launch"));
    assert!(!is_expected);
}

#[test]
fn test_explain_inference_hotspot_embedding() {
    let (explanation, is_expected) = explain_inference_hotspot("Embedding", 10.0);
    assert!(explanation.contains("Embedding"));
    assert!(is_expected);
}

#[test]
fn test_explain_inference_hotspot_sampling() {
    let (explanation, is_expected) = explain_inference_hotspot("Sampling", 5.0);
    assert!(explanation.contains("Sampling"));
    assert!(is_expected);
}

#[test]
fn test_explain_inference_hotspot_unknown_low() {
    let (explanation, is_expected) = explain_inference_hotspot("SomeOther", 10.0);
    assert!(is_expected);
    assert!(explanation.is_empty());
}

#[test]
fn test_explain_inference_hotspot_unknown_high() {
    let (explanation, is_expected) = explain_inference_hotspot("SomeOther", 30.0);
    assert!(!is_expected);
    assert!(explanation.contains("Unknown"));
}

#[test]
fn test_benchmark_grid_render_with_colors() {
    let mut grid = BenchmarkGrid::new()
        .with_model("Test", "0.5B", "Q4_K")
        .with_gpu("RTX 4090", 24.0)
        .with_config(BenchConfig {
            colors: true,
            ..Default::default()
        });

    grid.set_gguf_row(
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput_samples(vec![500.0, 502.0])
            .with_ttft(7.0),
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput(318.0)
            .with_ttft(50.0),
        BenchMeasurement::new("llama.cpp", "GGUF")
            .with_throughput(200.0)
            .with_ttft(30.0),
    );

    let output = grid.render();
    assert!(output.contains("\x1b[")); // Has ANSI codes
}

#[test]
fn test_benchmark_grid_render_scientific_with_ttft() {
    let mut grid = BenchmarkGrid::new()
        .with_model("Test", "0.5B", "Q4_K")
        .with_gpu("RTX 4090", 24.0)
        .with_config(BenchConfig::default());

    let mut apr = BenchMeasurement::new("APR", "GGUF")
        .with_throughput_samples(vec![500.0, 502.0])
        .with_ttft_samples(vec![7.0, 7.2]);
    apr.compute_stats(2.0);
    grid.gguf_apr = Some(apr);

    let mut ollama = BenchMeasurement::new("Ollama", "GGUF")
        .with_throughput_samples(vec![318.0])
        .with_ttft_samples(vec![50.0]);
    ollama.compute_stats(2.0);
    grid.gguf_ollama = Some(ollama);

    let output = grid.render_scientific();
    assert!(output.contains("Time to First Token"));
}

#[test]
fn test_benchmark_grid_render_profiling_log() {
    let mut grid = BenchmarkGrid::new()
        .with_model("Test Model", "0.5B", "Q4_K")
        .with_gpu("RTX 4090", 24.0);

    grid.gguf_apr = Some(
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput(500.0)
            .with_ttft(7.0)
            .with_gpu(85.0, 2048.0),
    );
    grid.gguf_ollama = Some(
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput(318.0)
            .with_ttft(50.0),
    );
    grid.gguf_llamacpp = Some(
        BenchMeasurement::new("llama.cpp", "GGUF")
            .with_throughput(200.0)
            .with_ttft(30.0),
    );
    grid.apr_native = Some(
        BenchMeasurement::new("APR", ".apr")
            .with_throughput(600.0)
            .with_ttft(5.0)
            .with_gpu(90.0, 2048.0),
    );

    grid.add_hotspot(ProfilingHotspot {
        component: "Q4K_GEMV".to_string(),
        time: Duration::from_millis(150),
        percentage: 45.0,
        call_count: 1000,
        avg_per_call: Duration::from_micros(150),
        explanation: "Matrix ops dominate - expected".to_string(),
        is_expected: true,
    });

    grid.add_hotspot(ProfilingHotspot {
        component: "Slow".to_string(),
        time: Duration::from_millis(50),
        percentage: 15.0,
        call_count: 100,
        avg_per_call: Duration::from_micros(500),
        explanation: "Unexpected slowdown".to_string(),
        is_expected: false,
    });

    let output = grid.render_profiling_log();
    assert!(output.contains("INFERENCE PROFILING REPORT"));
    assert!(output.contains("Test Model"));
    assert!(output.contains("Q4K_GEMV"));
    assert!(output.contains("OPTIMIZATION RECOMMENDATIONS"));
}

#[test]
fn test_benchmark_grid_render_profiling_log_no_hotspots() {
    let grid = BenchmarkGrid::new()
        .with_model("Test", "0.5B", "Q4_K")
        .with_gpu("RTX 4090", 24.0);

    let output = grid.render_profiling_log();
    assert!(output.contains("No unexpected hotspots"));
}

#[test]
fn test_benchmark_grid_add_hotspot() {
    let mut grid = BenchmarkGrid::new();
    let hotspot = ProfilingHotspot {
        component: "Test".to_string(),
        time: Duration::from_millis(100),
        percentage: 50.0,
        call_count: 100,
        avg_per_call: Duration::from_millis(1),
        explanation: String::new(),
        is_expected: true,
    };
    grid.add_hotspot(hotspot);
    assert_eq!(grid.hotspots.len(), 1);
}

#[test]
fn test_bench_measurement_mean_ttft_no_stats() {
    let m = BenchMeasurement::new("APR", "GGUF").with_ttft(10.0);
    assert_eq!(m.mean_ttft(), 10.0);
}

#[test]
fn test_bench_measurement_mean_ttft_empty() {
    let m = BenchMeasurement::new("APR", "GGUF");
    assert_eq!(m.mean_ttft(), 0.0);
}

#[test]
fn test_bench_measurement_mean_throughput_empty() {
    let m = BenchMeasurement::new("APR", "GGUF");
    assert_eq!(m.mean_throughput(), 0.0);
}

#[test]
fn test_bench_stats_even_samples_median() {
    let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0, 104.0], 2.0);
    // Sorted: 98, 100, 102, 104 -> median = (100+102)/2 = 101
    assert!((stats.median - 101.0).abs() < 0.1);
}

#[test]
fn test_bench_stats_large_samples_t_value() {
    // 30+ samples use t_value = 1.96
    let samples: Vec<f64> = (0..35).map(|i| 100.0 + i as f64 * 0.1).collect();
    let stats = BenchStats::from_samples(samples, 2.0);
    assert!(stats.ci_95.0 < stats.mean);
    assert!(stats.ci_95.1 > stats.mean);
}

#[test]
fn test_explain_flash_attention() {
    let (explanation, is_expected) = explain_inference_hotspot("FlashAttention", 40.0);
    assert!(explanation.contains("Attention"));
    assert!(is_expected);
}

#[test]
fn test_explain_gemm() {
    let (explanation, is_expected) = explain_inference_hotspot("GEMM", 60.0);
    assert!(explanation.contains("Matrix ops"));
    assert!(is_expected);
}

#[test]
fn test_explain_topk() {
    let (explanation, is_expected) = explain_inference_hotspot("TopK", 5.0);
    assert!(explanation.contains("Sampling"));
    assert!(is_expected);
}

#[test]
fn test_explain_topp() {
    let (explanation, is_expected) = explain_inference_hotspot("TopP", 5.0);
    assert!(explanation.contains("Sampling"));
    assert!(is_expected);
}

#[test]
fn test_explain_memcpy_d2h() {
    let (explanation, is_expected) = explain_inference_hotspot("MemcpyD2H", 10.0);
    assert!(explanation.contains("Memory transfer"));
    assert!(!is_expected);
}

#[test]
fn test_explain_transfer() {
    let (explanation, is_expected) = explain_inference_hotspot("Transfer", 10.0);
    assert!(explanation.contains("Memory transfer"));
    assert!(!is_expected);
}

#[test]
fn test_benchmark_grid_speedup_zero_baseline() {
    let mut grid = BenchmarkGrid::new().with_config(BenchConfig {
        colors: false,
        ..Default::default()
    });

    grid.set_apr_row(
        BenchMeasurement::new("APR", ".apr").with_throughput(600.0),
        BenchMeasurement::new("APR", "GGUF").with_throughput(500.0),
        BenchMeasurement::new("Ollama", "GGUF").with_throughput(0.0), // Zero baseline
    );

    let output = grid.render();
    // Should handle zero baseline without panic
    assert!(output.contains("APR serve .apr"));
}

#[test]
fn test_bench_config_clone() {
    let config = BenchConfig {
        iterations: 15,
        warmup_iterations: 5,
        outlier_threshold: 2.5,
        colors: false,
        confidence_level: 0.99,
    };
    let cloned = config.clone();
    assert_eq!(cloned.iterations, 15);
    assert_eq!(cloned.warmup_iterations, 5);
}

#[test]
fn test_bench_stats_clone() {
    let stats = BenchStats::from_samples(vec![100.0, 102.0], 2.0);
    let cloned = stats.clone();
    assert_eq!(cloned.mean, stats.mean);
    assert_eq!(cloned.samples.len(), stats.samples.len());
}

#[test]
fn test_bench_measurement_clone() {
    let m = BenchMeasurement::new("APR", "GGUF").with_throughput(500.0);
    let cloned = m.clone();
    assert_eq!(cloned.engine, "APR");
    assert_eq!(cloned.throughput_samples.len(), 1);
}

#[test]
fn test_benchmark_grid_clone() {
    let grid = BenchmarkGrid::new()
        .with_model("Test", "0.5B", "Q4_K")
        .with_gpu("RTX 4090", 24.0);
    let cloned = grid.clone();
    assert_eq!(cloned.model_name, "Test");
}

#[test]
fn test_profiling_hotspot_clone() {
    let hotspot = ProfilingHotspot {
        component: "Test".to_string(),
        time: Duration::from_millis(100),
        percentage: 50.0,
        call_count: 100,
        avg_per_call: Duration::from_millis(1),
        explanation: "Test".to_string(),
        is_expected: true,
    };
    let cloned = hotspot.clone();
    assert_eq!(cloned.component, "Test");
}
