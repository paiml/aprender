
// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_grid_ascii() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
            .with_gpu("RTX 4090", 24.0);

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

        let ascii = grid.render_ascii();
        assert!(ascii.contains("APR serve GGUF"));
        assert!(ascii.contains("Ollama"));
        assert!(ascii.contains("llama.cpp"));
        assert!(ascii.contains("500.0 tok/s"));
    }

    #[test]
    fn test_profiling_log() {
        let mut grid = BenchmarkGrid::new()
            .with_model("Qwen2.5-Coder-0.5B", "0.5B", "Q4_K_M")
            .with_gpu("RTX 4090", 24.0);

        grid.gguf_apr = Some(
            BenchMeasurement::new("APR", "GGUF")
                .with_throughput(500.0)
                .with_ttft(7.0)
                .with_gpu(95.0, 2048.0),
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

        let log = grid.render_profiling_log();
        assert!(log.contains("PROFILING REPORT"));
        assert!(log.contains("Q4K_GEMV"));
        assert!(log.contains("45.0%"));
    }

    #[test]
    fn test_compact_output() {
        let mut grid = BenchmarkGrid::new();
        grid.gguf_apr = Some(BenchMeasurement::new("APR", "GGUF").with_throughput(500.0));
        grid.gguf_ollama = Some(BenchMeasurement::new("Ollama", "GGUF").with_throughput(318.0));
        grid.gguf_llamacpp =
            Some(BenchMeasurement::new("llama.cpp", "GGUF").with_throughput(200.0));

        let compact = grid.render_compact();
        assert!(compact.contains("APR:500"));
        assert!(compact.contains("vs llama.cpp:2.50x"));
    }

    #[test]
    fn test_runner_profiling() {
        let mut runner = BenchmarkRunner::new();
        runner.start();

        runner.record_component("Q4K_GEMV", Duration::from_millis(100), 500);
        runner.record_component("Attention", Duration::from_millis(50), 500);
        runner.record_component("Other", Duration::from_millis(10), 100);

        runner.finalize();

        assert!(!runner.grid.hotspots.is_empty());
        assert_eq!(runner.grid.hotspots[0].component, "Q4K_GEMV");
    }

    #[test]
    fn test_render_bar() {
        let bar = render_bar(50.0, 100.0, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 5);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 5);
    }

    // =========================================================================
    // Coverage Tests: BenchMeasurement
    // =========================================================================

    #[test]
    fn test_bench_measurement_new() {
        let m = BenchMeasurement::new("TestEngine", "TestFormat");
        assert_eq!(m.engine, "TestEngine");
        assert_eq!(m.format, "TestFormat");
        assert_eq!(m.tokens_per_sec, 0.0);
        assert_eq!(m.ttft_ms, 0.0);
        assert_eq!(m.tokens_generated, 0);
        assert!(m.gpu_util.is_none());
        assert!(m.gpu_mem_mb.is_none());
    }

    #[test]
    fn test_bench_measurement_with_throughput() {
        let m = BenchMeasurement::new("APR", "GGUF").with_throughput(100.0);
        assert_eq!(m.tokens_per_sec, 100.0);
    }

    #[test]
    fn test_bench_measurement_with_ttft() {
        let m = BenchMeasurement::new("APR", "GGUF").with_ttft(25.5);
        assert_eq!(m.ttft_ms, 25.5);
    }

    #[test]
    fn test_bench_measurement_with_tokens() {
        let duration = Duration::from_secs(2);
        let m = BenchMeasurement::new("APR", "GGUF").with_tokens(200, duration);
        assert_eq!(m.tokens_generated, 200);
        assert_eq!(m.duration, duration);
        assert!((m.tokens_per_sec - 100.0).abs() < 0.1);
    }

    #[test]
    fn test_bench_measurement_with_tokens_zero_duration() {
        let m = BenchMeasurement::new("APR", "GGUF").with_tokens(100, Duration::ZERO);
        assert_eq!(m.tokens_generated, 100);
        // Zero duration means no TPS calculation
    }

    #[test]
    fn test_bench_measurement_with_gpu() {
        let m = BenchMeasurement::new("APR", "GGUF").with_gpu(95.0, 4096.0);
        assert_eq!(m.gpu_util, Some(95.0));
        assert_eq!(m.gpu_mem_mb, Some(4096.0));
    }

    #[test]
    fn test_bench_measurement_debug() {
        let m = BenchMeasurement::new("APR", "GGUF").with_throughput(100.0);
        let debug_str = format!("{:?}", m);
        assert!(debug_str.contains("BenchMeasurement"));
        assert!(debug_str.contains("APR"));
    }

    #[test]
    fn test_bench_measurement_clone() {
        let m = BenchMeasurement::new("APR", "GGUF")
            .with_throughput(100.0)
            .with_gpu(90.0, 1024.0);
        let cloned = m.clone();
        assert_eq!(cloned.engine, m.engine);
        assert_eq!(cloned.tokens_per_sec, m.tokens_per_sec);
        assert_eq!(cloned.gpu_util, m.gpu_util);
    }

    // =========================================================================
    // Coverage Tests: ProfilingHotspot
    // =========================================================================

    #[test]
    fn test_profiling_hotspot_debug() {
        let hotspot = ProfilingHotspot {
            component: "Attention".to_string(),
            time: Duration::from_millis(100),
            percentage: 50.0,
            call_count: 1000,
            avg_per_call: Duration::from_micros(100),
            explanation: "Expected".to_string(),
            is_expected: true,
        };
        let debug_str = format!("{:?}", hotspot);
        assert!(debug_str.contains("ProfilingHotspot"));
        assert!(debug_str.contains("Attention"));
    }

    #[test]
    fn test_profiling_hotspot_clone() {
        let hotspot = ProfilingHotspot {
            component: "GEMM".to_string(),
            time: Duration::from_millis(200),
            percentage: 75.0,
            call_count: 500,
            avg_per_call: Duration::from_micros(400),
            explanation: "Matrix multiplication".to_string(),
            is_expected: true,
        };
        let cloned = hotspot.clone();
        assert_eq!(cloned.component, hotspot.component);
        assert_eq!(cloned.percentage, hotspot.percentage);
    }

    // =========================================================================
    // Coverage Tests: BenchmarkGrid
    // =========================================================================

    #[test]
    fn test_benchmark_grid_new() {
        let grid = BenchmarkGrid::new();
        assert!(grid.gguf_apr.is_none());
        assert!(grid.gguf_ollama.is_none());
        assert!(grid.gguf_llamacpp.is_none());
        assert!(grid.hotspots.is_empty());
    }

    #[test]
    fn test_benchmark_grid_with_model() {
        let grid = BenchmarkGrid::new().with_model("Llama-7B", "7B", "Q4_K_M");
        assert_eq!(grid.model_name, "Llama-7B");
        assert_eq!(grid.model_params, "7B");
        assert_eq!(grid.quantization, "Q4_K_M");
    }

    #[test]
    fn test_benchmark_grid_with_gpu() {
        let grid = BenchmarkGrid::new().with_gpu("RTX 3090", 24.0);
        assert_eq!(grid.gpu_name, "RTX 3090");
        assert_eq!(grid.gpu_vram_gb, 24.0);
    }

    #[test]
    fn test_benchmark_grid_add_hotspot() {
        let mut grid = BenchmarkGrid::new();
        grid.add_hotspot(ProfilingHotspot {
            component: "Test".to_string(),
            time: Duration::from_millis(50),
            percentage: 25.0,
            call_count: 100,
            avg_per_call: Duration::from_micros(500),
            explanation: "Test hotspot".to_string(),
            is_expected: true,
        });
        assert_eq!(grid.hotspots.len(), 1);
        assert_eq!(grid.hotspots[0].component, "Test");
    }

    // =========================================================================
    // Coverage Tests: render_bar edge cases
    // =========================================================================

    #[test]
    fn test_render_bar_zero() {
        let bar = render_bar(0.0, 100.0, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 0);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 10);
    }

    #[test]
    fn test_render_bar_full() {
        let bar = render_bar(100.0, 100.0, 10);
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 10);
        assert_eq!(bar.chars().filter(|c| *c == '░').count(), 0);
    }

    #[test]
    fn test_render_bar_over_max() {
        let bar = render_bar(150.0, 100.0, 10);
        // Should clamp to max
        assert_eq!(bar.chars().filter(|c| *c == '█').count(), 10);
    }

    // =========================================================================
    // Coverage Tests: truncate
    // =========================================================================

    #[test]
    fn test_truncate_short_string() {
        let result = truncate("short", 10);
        assert_eq!(result, "short");
    }

    #[test]
    fn test_truncate_exact_length() {
        let result = truncate("exactly10c", 10);
        assert_eq!(result, "exactly10c");
    }

    #[test]
    fn test_truncate_long_string() {
        let result = truncate("this is a very long string", 10);
        assert_eq!(result.len(), 10);
    }

    // =========================================================================
    // Coverage Tests: explain_inference_hotspot
    // =========================================================================

    #[test]
    fn test_explain_inference_hotspot_gemv() {
        let (explanation, is_expected) = explain_inference_hotspot("Q4K_GEMV", 50.0);
        assert!(is_expected);
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_explain_inference_hotspot_attention() {
        let (explanation, is_expected) = explain_inference_hotspot("Attention", 30.0);
        assert!(is_expected);
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_explain_inference_hotspot_unknown() {
        let (explanation, is_expected) = explain_inference_hotspot("UnknownComponent", 60.0);
        // High percentage for unknown component is unexpected
        assert!(!is_expected);
        assert!(!explanation.is_empty());
    }

    #[test]
    fn test_explain_inference_hotspot_low_percentage() {
        let (explanation, is_expected) = explain_inference_hotspot("SomeComponent", 5.0);
        // Low percentage unknown component returns empty string and is expected
        assert!(is_expected);
        // Note: The function returns empty string for low percentage unknown components
        // which means "nothing to report" - this is valid behavior
        let _ = explanation;
    }
}
