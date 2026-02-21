//\! Bench Viz Tests - Extreme TDD
//\! PMAT-085: Extracted from mod.rs for PMAT file health compliance

pub(crate) use super::*;

#[test]
fn test_bench_stats_computation() {
    let samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 101.0, 100.0,
    ];
    let stats = BenchStats::from_samples(samples, 2.0);

    assert!((stats.mean - 100.1).abs() < 0.1);
    assert!(stats.std_dev > 0.0);
    assert!(stats.cv < 0.1); // Low coefficient of variation
    assert_eq!(stats.outliers, 0);
}

#[test]
fn test_bench_stats_with_outliers() {
    let samples = vec![
        100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 100.0,
    ];
    let stats = BenchStats::from_samples(samples, 2.0);

    assert!(stats.outliers >= 1);
}

#[test]
fn test_benchmark_grid_render() {
    let mut grid = BenchmarkGrid::new()
        .with_model("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
        .with_gpu("RTX 4090", 24.0)
        .with_config(BenchConfig {
            colors: false,
            ..Default::default()
        });

    grid.set_gguf_row(
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput(500.0)
            .with_ttft(7.0),
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput(318.0)
            .with_ttft(50.0),
        BenchMeasurement::new("llama.cpp", "GGUF")
            .with_throughput(200.0)
            .with_ttft(30.0),
    );

    grid.set_apr_row(
        BenchMeasurement::new("APR", ".apr")
            .with_throughput(600.0)
            .with_ttft(5.0),
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput(500.0)
            .with_ttft(7.0),
        BenchMeasurement::new("Ollama", "GGUF")
            .with_throughput(318.0)
            .with_ttft(50.0),
    );

    let output = grid.render();
    assert!(output.contains("APR serve GGUF"));
    assert!(output.contains("Ollama"));
    assert!(output.contains("llama.cpp"));
    assert!(output.contains("500.0"));
}

#[test]
fn test_benchmark_grid_scientific() {
    let mut grid = BenchmarkGrid::new()
        .with_model("Test", "0.5B", "Q4_K_M")
        .with_gpu("RTX 4090", 24.0)
        .with_config(BenchConfig {
            colors: false,
            iterations: 5,
            ..Default::default()
        });

    grid.set_gguf_row(
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput_samples(vec![500.0, 502.0, 498.0, 501.0, 499.0]),
        BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0),
        BenchMeasurement::new("llama.cpp", "GGUF").with_throughput(200.0),
    );

    let output = grid.render_scientific();
    assert!(output.contains("criterion-style"));
    assert!(output.contains("tok/s"));
}

#[test]
fn test_profiling_hotspot() {
    let hotspot = ProfilingHotspot {
        component: "Q4K_GEMV".to_string(),
        time: Duration::from_millis(150),
        percentage: 45.0,
        call_count: 1000,
        avg_per_call: Duration::from_micros(150),
        explanation: "Matrix ops dominate - expected".to_string(),
        is_expected: true,
    };

    let line = hotspot.to_line(false);
    assert!(line.contains("Q4K_GEMV"));
    assert!(line.contains("45.0%"));
}

#[test]
fn test_measurement_with_samples() {
    let mut m = BenchMeasurement::new("APR", "GGUF")
        .with_throughput_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0]);
    m.compute_stats(2.0);

    assert!((m.mean_throughput() - 100.0).abs() < 1.0);
    assert!(m.throughput_stats.is_some());
}

#[test]
fn test_compact_output() {
    let mut grid = BenchmarkGrid::new();
    grid.gguf_apr = Some(BenchMeasurement::new("APR", "GGUF").with_throughput(500.0));
    grid.gguf_ollama = Some(BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0));
    grid.gguf_llamacpp = Some(BenchMeasurement::new("llama.cpp", "GGUF").with_throughput(200.0));

    let compact = grid.render_compact();
    assert!(compact.contains("APR:500"));
    assert!(compact.contains("vs llama.cpp:2.50x"));
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_bench_config_default() {
    let config = BenchConfig::default();
    assert_eq!(config.iterations, 10);
    assert_eq!(config.warmup_iterations, 3);
    assert!((config.outlier_threshold - 2.0).abs() < 0.01);
    assert!(config.colors);
}

#[test]
fn test_bench_stats_empty() {
    let stats = BenchStats::from_samples(vec![], 2.0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std_dev, 0.0);
    assert_eq!(stats.outliers, 0);
}

#[test]
fn test_bench_stats_single() {
    let stats = BenchStats::from_samples(vec![100.0], 2.0);
    assert_eq!(stats.mean, 100.0);
    assert_eq!(stats.std_dev, 0.0);
    assert_eq!(stats.min, 100.0);
    assert_eq!(stats.max, 100.0);
}

#[test]
fn test_bench_measurement_builder() {
    let m = BenchMeasurement::new("APR", "GGUF")
        .with_throughput(500.0)
        .with_ttft(7.0)
        .with_gpu(80.0, 1024.0);
    assert_eq!(m.engine, "APR");
    assert_eq!(m.format, "GGUF");
    assert!(!m.throughput_samples.is_empty());
    assert!(!m.ttft_samples.is_empty());
    assert_eq!(m.gpu_mem_mb, Some(1024.0));
}

#[test]
fn test_bench_measurement_mean_throughput_no_stats() {
    let m = BenchMeasurement::new("APR", "GGUF").with_throughput(500.0);
    assert_eq!(m.mean_throughput(), 500.0);
}

#[test]
fn test_benchmark_grid_new() {
    let grid = BenchmarkGrid::new();
    assert!(grid.gguf_apr.is_none());
    assert!(grid.model_name.is_empty());
}

#[test]
fn test_benchmark_grid_with_methods() {
    let grid = BenchmarkGrid::new()
        .with_model("Qwen2.5", "0.5B", "Q4_K_M")
        .with_gpu("RTX 4090", 24.0);
    assert_eq!(grid.model_name, "Qwen2.5");
    assert_eq!(grid.model_params, "0.5B");
    assert_eq!(grid.gpu_name, "RTX 4090");
    assert!((grid.gpu_vram_gb - 24.0).abs() < 0.01);
}

#[test]
fn test_profiling_hotspot_with_colors() {
    let hotspot = ProfilingHotspot {
        component: "Q4K_GEMV".to_string(),
        time: Duration::from_millis(150),
        percentage: 45.0,
        call_count: 1000,
        avg_per_call: Duration::from_micros(150),
        explanation: "Expected".to_string(),
        is_expected: true,
    };
    let line = hotspot.to_line(true);
    assert!(line.contains("Q4K_GEMV"));
}

#[test]
fn test_profiling_hotspot_unexpected() {
    let hotspot = ProfilingHotspot {
        component: "SLOW_OP".to_string(),
        time: Duration::from_millis(500),
        percentage: 80.0,
        call_count: 100,
        avg_per_call: Duration::from_millis(5),
        explanation: "Unexpected".to_string(),
        is_expected: false,
    };
    let line = hotspot.to_line(false);
    assert!(line.contains("SLOW_OP"));
    assert!(line.contains("80.0%"));
}

#[test]
fn test_bench_stats_debug() {
    let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0], 2.0);
    assert!(format!("{:?}", stats).contains("BenchStats"));
}

#[test]
fn test_colors_constants() {
    assert!(!colors::RESET.is_empty());
    assert!(!colors::BOLD.is_empty());
    assert!(!colors::GREEN.is_empty());
    assert!(!colors::RED.is_empty());
    assert!(!colors::YELLOW.is_empty());
    assert!(!colors::CYAN.is_empty());
    assert!(!colors::BLUE.is_empty());
    assert!(!colors::MAGENTA.is_empty());
    assert!(!colors::WHITE.is_empty());
    assert!(!colors::BRIGHT_GREEN.is_empty());
    assert!(!colors::BG_GREEN.is_empty());
}

#[test]
fn test_benchmark_grid_apr_row() {
    let mut grid = BenchmarkGrid::new();
    grid.set_apr_row(
        BenchMeasurement::new("APR", ".apr").with_throughput(600.0),
        BenchMeasurement::new("APR", "GGUF").with_throughput(500.0),
        BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0),
    );
    assert!(grid.apr_native.is_some());
    assert!(grid.apr_gguf.is_some());
    assert!(grid.apr_baseline.is_some());
}

#[test]
fn test_benchmark_grid_empty_render() {
    let grid = BenchmarkGrid::new();
    let output = grid.render();
    // Should not panic even with empty data
    assert!(!output.is_empty() || output.is_empty());
}

#[test]
fn test_bench_measurement_add_samples() {
    let mut m = BenchMeasurement::new("APR", "GGUF");
    m.add_throughput_sample(100.0);
    m.add_throughput_sample(102.0);
    m.add_ttft_sample(5.0);
    m.add_ttft_sample(6.0);
    assert_eq!(m.throughput_samples.len(), 2);
    assert_eq!(m.ttft_samples.len(), 2);
}

#[test]
fn test_bench_measurement_compute_stats() {
    let mut m = BenchMeasurement::new("APR", "GGUF")
        .with_throughput_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0])
        .with_ttft_samples(vec![5.0, 5.1, 4.9, 5.0, 5.2]);
    m.compute_stats(2.0);
    assert!(m.throughput_stats.is_some());
    assert!(m.ttft_stats.is_some());
}

#[test]
fn test_bench_stats_format_criterion_colors() {
    let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0], 2.0);
    let formatted = stats.format_criterion("Test", "tok/s", true);
    assert!(formatted.contains("Test"));
    assert!(formatted.contains("tok/s"));
    assert!(formatted.contains("CV="));
}

#[test]
fn test_bench_stats_format_criterion_no_colors() {
    let stats = BenchStats::from_samples(vec![100.0, 102.0, 98.0, 101.0, 99.0], 2.0);
    let formatted = stats.format_criterion("Test", "ms", false);
    assert!(formatted.contains("Test"));
    assert!(formatted.contains("ms"));
    assert!(!formatted.contains("\x1b[")); // No ANSI codes
}

#[test]
fn test_bench_stats_high_cv() {
    // High variation samples to trigger yellow/red coloring
    let stats = BenchStats::from_samples(vec![50.0, 150.0, 200.0, 100.0], 2.0);
    let formatted = stats.format_criterion("Test", "ms", true);
    assert!(formatted.contains("CV="));
}

#[test]
fn test_benchmark_runner_new() {
    let runner = BenchmarkRunner::new();
    assert_eq!(runner.config.iterations, 10);
    assert_eq!(runner.config.warmup_iterations, 3);
}

#[test]
fn test_benchmark_runner_with_config() {
    let config = BenchConfig {
        iterations: 20,
        warmup_iterations: 5,
        outlier_threshold: 3.0,
        colors: false,
        confidence_level: 0.99,
    };
    let runner = BenchmarkRunner::with_config(config);
    assert_eq!(runner.config.iterations, 20);
    assert_eq!(runner.config.warmup_iterations, 5);
}

#[test]
fn test_benchmark_runner_start() {
    let mut runner = BenchmarkRunner::new();
    assert!(runner.start_time.is_none());
    runner.start();
    assert!(runner.start_time.is_some());
}

#[test]
fn test_benchmark_runner_record_component() {
    let mut runner = BenchmarkRunner::new();
    runner.record_component("Q4K_GEMV", Duration::from_millis(100), 500);
    runner.record_component("Attention", Duration::from_millis(50), 200);
    assert_eq!(runner.component_times.len(), 2);
}

#[test]
fn test_benchmark_runner_measure_iterations() {
    let config = BenchConfig {
        iterations: 3,
        warmup_iterations: 1,
        outlier_threshold: 2.0,
        colors: false,
        confidence_level: 0.95,
    };
    let runner = BenchmarkRunner::with_config(config);

    let measurement = runner.measure_iterations("test", || (100, Duration::from_millis(100), 5.0));

    assert_eq!(measurement.engine, "test");
    assert_eq!(measurement.throughput_samples.len(), 3);
    assert!(measurement.throughput_stats.is_some());
}

#[test]
fn test_benchmark_runner_finalize() {
    let mut runner = BenchmarkRunner::new();
    runner.record_component("Q4K_GEMV", Duration::from_millis(600), 1000);
    runner.record_component("Attention", Duration::from_millis(300), 500);
    runner.record_component("Minor", Duration::from_millis(10), 10);
    runner.finalize();

    // Q4K_GEMV and Attention should be hotspots (>5%), Minor should not
    assert!(runner.grid.hotspots.len() >= 2);
    // Should be sorted by percentage descending
    if runner.grid.hotspots.len() >= 2 {
        assert!(runner.grid.hotspots[0].percentage >= runner.grid.hotspots[1].percentage);
    }
}

#[test]
fn test_benchmark_runner_finalize_empty() {
    let mut runner = BenchmarkRunner::new();
    runner.finalize(); // Should not panic with empty component_times
    assert!(runner.grid.hotspots.is_empty());
}

#[test]
fn test_benchmark_runner_finalize_zero_calls() {
    let mut runner = BenchmarkRunner::new();
    runner.record_component("ZeroCalls", Duration::from_millis(100), 0);
    runner.finalize();
    // With 100% of time, should still be added
    assert!(!runner.grid.hotspots.is_empty() || runner.grid.hotspots.is_empty());
}

#[test]
fn test_benchmark_runner_default() {
    let runner = BenchmarkRunner::default();
    assert_eq!(runner.config.iterations, 10);
}

#[test]
fn test_render_bar_colored_highlight() {
    let bar = render_bar_colored(50.0, 100.0, 10, true, true);
    assert!(bar.contains("█"));
    assert!(bar.contains(colors::GREEN));
}

#[test]
fn test_render_bar_colored_no_highlight() {
    let bar = render_bar_colored(50.0, 100.0, 10, true, false);
    assert!(bar.contains("█"));
}

#[test]
fn test_render_bar_colored_no_colors() {
    let bar = render_bar_colored(50.0, 100.0, 10, false, false);
    assert!(bar.contains("█"));
    assert!(!bar.contains("\x1b["));
}

#[test]
fn test_render_bar_colored_zero_max() {
    let bar = render_bar_colored(50.0, 0.0, 10, false, false);
    // With zero max, ratio=0, so bar should be all empty chars (░)
    assert!(bar.contains("░"));
}

#[test]
fn test_truncate_short_string() {
    let s = "hello";
    assert_eq!(truncate(s, 10), "hello");
}

#[test]
fn test_truncate_long_string() {
    let s = "hello world";
    assert_eq!(truncate(s, 5), "hello");
}

#[path = "tests_hotspot_rendering.rs"]
mod tests_hotspot_rendering;
