//! T-COV-95 Deep Coverage: bench_viz.rs pure functions (Part 02)
//!
//! Tests BenchmarkRunner::measure, BenchmarkGrid rendering edge cases,
//! explain_inference_hotspot coverage for all match arms,
//! render_bar edge cases, and ProfilingHotspot::to_line formatting.

use crate::bench_viz::*;
use std::time::Duration;

// ============================================================================
// BenchmarkRunner tests
// ============================================================================

#[test]
fn test_benchmark_runner_default() {
    let runner = BenchmarkRunner::default();
    assert!(runner.grid.gguf_apr.is_none());
    assert!(runner.grid.hotspots.is_empty());
}

#[test]
fn test_benchmark_runner_measure() {
    let mut runner = BenchmarkRunner::new();
    runner.start();

    let result = runner.measure("test_op", || 42);
    assert_eq!(result, 42);
    assert_eq!(runner.component_times.len(), 1);
    assert_eq!(runner.component_times[0].0, "test_op");
}

#[test]
fn test_benchmark_runner_measure_preserves_return_value() {
    let mut runner = BenchmarkRunner::new();
    let result = runner.measure("compute", || vec![1, 2, 3]);
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_benchmark_runner_record_component() {
    let mut runner = BenchmarkRunner::new();
    runner.record_component("Q4K_GEMV", Duration::from_millis(100), 500);
    runner.record_component("Attention", Duration::from_millis(50), 500);

    assert_eq!(runner.component_times.len(), 2);
    assert_eq!(runner.component_times[0].0, "Q4K_GEMV");
    assert_eq!(runner.component_times[1].0, "Attention");
}

#[test]
fn test_benchmark_runner_finalize_empty() {
    let mut runner = BenchmarkRunner::new();
    runner.finalize();
    assert!(runner.grid.hotspots.is_empty());
}

#[test]
fn test_benchmark_runner_finalize_filters_small_components() {
    let mut runner = BenchmarkRunner::new();
    // One large component and one tiny one
    runner.record_component("BigOp", Duration::from_millis(100), 1);
    runner.record_component("TinyOp", Duration::from_millis(1), 1);

    runner.finalize();
    // Only BigOp should appear (TinyOp < 5%)
    assert!(!runner.grid.hotspots.is_empty());
    assert!(runner
        .grid
        .hotspots
        .iter()
        .any(|h| h.component == "BigOp"));
}

#[test]
fn test_benchmark_runner_finalize_sorts_by_percentage() {
    let mut runner = BenchmarkRunner::new();
    runner.record_component("Small", Duration::from_millis(50), 100);
    runner.record_component("Large", Duration::from_millis(150), 100);

    runner.finalize();
    if runner.grid.hotspots.len() >= 2 {
        assert!(runner.grid.hotspots[0].percentage >= runner.grid.hotspots[1].percentage);
    }
}

#[test]
fn test_benchmark_runner_finalize_zero_calls() {
    let mut runner = BenchmarkRunner::new();
    runner.record_component("ZeroCalls", Duration::from_millis(100), 0);

    runner.finalize();
    // Should handle zero calls gracefully
    for hotspot in &runner.grid.hotspots {
        if hotspot.component == "ZeroCalls" {
            assert_eq!(hotspot.avg_per_call, Duration::ZERO);
        }
    }
}

// ============================================================================
// explain_inference_hotspot - all match arms
// ============================================================================

#[test]
fn test_explain_hotspot_matmul() {
    let (explanation, is_expected) = explain_inference_hotspot("MatMul", 60.0);
    assert!(is_expected);
    assert!(explanation.contains("Matrix ops"));
}

#[test]
fn test_explain_hotspot_gemm() {
    let (explanation, is_expected) = explain_inference_hotspot("GEMM", 40.0);
    assert!(is_expected);
    assert!(explanation.contains("Matrix ops"));
}

#[test]
fn test_explain_hotspot_flash_attention() {
    let (explanation, is_expected) = explain_inference_hotspot("FlashAttention", 25.0);
    assert!(is_expected);
    assert!(explanation.contains("Attention"));
}

#[test]
fn test_explain_hotspot_kv_cache_normal() {
    let (explanation, is_expected) = explain_inference_hotspot("KV_Cache", 10.0);
    assert!(is_expected);
    assert!(explanation.contains("normal range"));
}

#[test]
fn test_explain_hotspot_kv_cache_high() {
    let (explanation, is_expected) = explain_inference_hotspot("KVCache", 25.0);
    assert!(!is_expected);
    assert!(explanation.contains("overhead high"));
}

#[test]
fn test_explain_hotspot_softmax_normal() {
    let (explanation, is_expected) = explain_inference_hotspot("Softmax", 5.0);
    assert!(is_expected);
    assert!(explanation.contains("normal range"));
}

#[test]
fn test_explain_hotspot_softmax_high() {
    let (explanation, is_expected) = explain_inference_hotspot("Softmax", 15.0);
    assert!(!is_expected);
    assert!(explanation.contains("unusually high"));
}

#[test]
fn test_explain_hotspot_rmsnorm_normal() {
    let (explanation, is_expected) = explain_inference_hotspot("RMSNorm", 8.0);
    assert!(is_expected);
    assert!(explanation.contains("normal range"));
}

#[test]
fn test_explain_hotspot_layernorm_high() {
    let (explanation, is_expected) = explain_inference_hotspot("LayerNorm", 20.0);
    assert!(!is_expected);
    assert!(explanation.contains("overhead high"));
}

#[test]
fn test_explain_hotspot_memcpy_h2d() {
    let (explanation, is_expected) = explain_inference_hotspot("MemcpyH2D", 10.0);
    assert!(!is_expected);
    assert!(explanation.contains("Memory transfer"));
}

#[test]
fn test_explain_hotspot_memcpy_d2h() {
    let (explanation, is_expected) = explain_inference_hotspot("MemcpyD2H", 15.0);
    assert!(!is_expected);
    assert!(explanation.contains("Memory transfer"));
}

#[test]
fn test_explain_hotspot_transfer() {
    let (explanation, is_expected) = explain_inference_hotspot("Transfer", 12.0);
    assert!(!is_expected);
    assert!(explanation.contains("Memory transfer"));
}

#[test]
fn test_explain_hotspot_kernel_launch() {
    let (explanation, is_expected) = explain_inference_hotspot("KernelLaunch", 10.0);
    assert!(!is_expected);
    assert!(explanation.contains("Kernel launch"));
}

#[test]
fn test_explain_hotspot_embedding() {
    let (explanation, is_expected) = explain_inference_hotspot("Embedding", 6.0);
    assert!(is_expected);
    assert!(explanation.contains("Embedding"));
}

#[test]
fn test_explain_hotspot_sampling() {
    let (explanation, is_expected) = explain_inference_hotspot("Sampling", 7.0);
    assert!(is_expected);
    assert!(explanation.contains("Sampling"));
}

#[test]
fn test_explain_hotspot_topk() {
    let (explanation, is_expected) = explain_inference_hotspot("TopK", 6.0);
    assert!(is_expected);
    assert!(explanation.contains("Sampling"));
}

#[test]
fn test_explain_hotspot_topp() {
    let (explanation, is_expected) = explain_inference_hotspot("TopP", 8.0);
    assert!(is_expected);
    assert!(explanation.contains("Sampling"));
}

#[test]
fn test_explain_hotspot_unknown_low() {
    let (explanation, is_expected) = explain_inference_hotspot("CustomOp", 10.0);
    assert!(is_expected);
    assert!(explanation.is_empty());
}

#[test]
fn test_explain_hotspot_unknown_high() {
    let (explanation, is_expected) = explain_inference_hotspot("CustomOp", 25.0);
    assert!(!is_expected);
    assert!(explanation.contains("investigate"));
}

// ============================================================================
// ProfilingHotspot::to_line tests
// ============================================================================

#[test]
fn test_profiling_hotspot_to_line_expected() {
    let hotspot = ProfilingHotspot {
        component: "Attention".to_string(),
        time: Duration::from_millis(100),
        percentage: 50.0,
        call_count: 1000,
        avg_per_call: Duration::from_micros(100),
        explanation: "Expected behavior".to_string(),
        is_expected: true,
    };
    let line = hotspot.to_line();
    assert!(line.contains("Attention"));
    assert!(line.contains("50.0%"));
    assert!(line.contains("1000"));
}

#[test]
fn test_profiling_hotspot_to_line_unexpected() {
    let hotspot = ProfilingHotspot {
        component: "MemcpyH2D".to_string(),
        time: Duration::from_millis(200),
        percentage: 30.0,
        call_count: 500,
        avg_per_call: Duration::from_micros(400),
        explanation: "Consider persistent buffers".to_string(),
        is_expected: false,
    };
    let line = hotspot.to_line();
    assert!(line.contains("MemcpyH2D"));
    assert!(line.contains("30.0%"));
}

// ============================================================================
// BenchmarkGrid rendering edge cases
// ============================================================================

#[test]
fn test_benchmark_grid_render_ascii_empty() {
    let grid = BenchmarkGrid::new();
    let ascii = grid.render_ascii();
    // Should still render frame without crashing
    assert!(ascii.contains("INFERENCE BENCHMARK COMPARISON"));
    assert!(ascii.contains("0.0 tok/s"));
}

#[test]
fn test_benchmark_grid_render_profiling_log_no_hotspots() {
    let grid = BenchmarkGrid::new()
        .with_model("TestModel", "1B", "Q4_K")
        .with_gpu("RTX 4090", 24.0);
    let log = grid.render_profiling_log();
    assert!(log.contains("PROFILING REPORT"));
    assert!(log.contains("TestModel"));
    assert!(log.contains("No unexpected hotspots"));
}

#[test]
fn test_benchmark_grid_render_profiling_log_with_unexpected_hotspot() {
    let mut grid = BenchmarkGrid::new()
        .with_model("TestModel", "1B", "Q4_K")
        .with_gpu("RTX 4090", 24.0);

    grid.add_hotspot(ProfilingHotspot {
        component: "MemcpyH2D".to_string(),
        time: Duration::from_millis(100),
        percentage: 30.0,
        call_count: 200,
        avg_per_call: Duration::from_micros(500),
        explanation: "Memory transfer overhead".to_string(),
        is_expected: false,
    });

    let log = grid.render_profiling_log();
    assert!(log.contains("MemcpyH2D"));
    assert!(log.contains("Memory transfer overhead"));
}

#[test]
fn test_benchmark_grid_render_profiling_log_with_measurements() {
    let mut grid = BenchmarkGrid::new()
        .with_model("Qwen2.5", "0.5B", "Q4_K_M")
        .with_gpu("RTX 4090", 24.0);

    grid.gguf_apr = Some(
        BenchMeasurement::new("APR", "GGUF")
            .with_throughput(250.0)
            .with_ttft(10.0)
            .with_gpu(85.0, 1024.0),
    );
    grid.apr_native = Some(
        BenchMeasurement::new("APR", ".apr")
            .with_throughput(300.0)
            .with_ttft(8.0)
            .with_gpu(90.0, 800.0),
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

    let log = grid.render_profiling_log();
    assert!(log.contains("APR GGUF"));
    assert!(log.contains("APR .apr"));
    assert!(log.contains("Ollama"));
    assert!(log.contains("llama.cpp"));
    assert!(log.contains("GPU Util"));
}

#[test]
fn test_benchmark_grid_render_compact_zeros() {
    let grid = BenchmarkGrid::new();
    let compact = grid.render_compact();
    assert!(compact.contains("APR:0"));
}

#[test]
fn test_benchmark_grid_render_profiling_log_low_tps() {
    let mut grid = BenchmarkGrid::new()
        .with_model("TestModel", "1B", "Q4_K")
        .with_gpu("RTX 3090", 24.0);

    grid.gguf_apr = Some(BenchMeasurement::new("APR", "GGUF").with_throughput(100.0));
    let log = grid.render_profiling_log();
    // Low TPS should trigger Phase 2 optimization suggestions
    assert!(log.contains("Phase 2 Optimizations"));
}

#[test]
fn test_benchmark_grid_set_gguf_row() {
    let mut grid = BenchmarkGrid::new();
    grid.set_gguf_row(
        BenchMeasurement::new("APR", "GGUF"),
        BenchMeasurement::new("Ollama", "GGUF"),
        BenchMeasurement::new("llama.cpp", "GGUF"),
    );
    assert!(grid.gguf_apr.is_some());
    assert!(grid.gguf_ollama.is_some());
    assert!(grid.gguf_llamacpp.is_some());
}

#[test]
fn test_benchmark_grid_set_apr_row() {
    let mut grid = BenchmarkGrid::new();
    grid.set_apr_row(
        BenchMeasurement::new("APR", ".apr"),
        BenchMeasurement::new("APR", "GGUF"),
        BenchMeasurement::new("Ollama", "GGUF"),
    );
    assert!(grid.apr_native.is_some());
    assert!(grid.apr_gguf.is_some());
    assert!(grid.apr_baseline.is_some());
}

// ============================================================================
// BenchMeasurement chaining tests
// ============================================================================

#[test]
fn test_bench_measurement_full_chain() {
    let m = BenchMeasurement::new("APR", "GGUF")
        .with_throughput(500.0)
        .with_ttft(7.0)
        .with_tokens(100, Duration::from_secs(2))
        .with_gpu(95.0, 2048.0);

    assert_eq!(m.engine, "APR");
    assert_eq!(m.format, "GGUF");
    assert_eq!(m.ttft_ms, 7.0);
    assert_eq!(m.tokens_generated, 100);
    // tokens_per_sec was overwritten by with_tokens
    assert!((m.tokens_per_sec - 50.0).abs() < 0.1);
    assert_eq!(m.gpu_util, Some(95.0));
    assert_eq!(m.gpu_mem_mb, Some(2048.0));
}

// ============================================================================
// render_bar edge cases
// ============================================================================

#[test]
fn test_render_bar_zero_max() {
    let bar = render_bar(50.0, 0.0, 10);
    // Zero max should produce all empty
    assert_eq!(bar.chars().filter(|c| *c == '█').count(), 0);
}

#[test]
fn test_render_bar_width_one() {
    let bar = render_bar(100.0, 100.0, 1);
    assert_eq!(bar.chars().filter(|c| *c == '█').count(), 1);
}

#[test]
fn test_render_bar_width_zero() {
    let bar = render_bar(100.0, 100.0, 0);
    assert!(bar.is_empty());
}

// ============================================================================
// truncate edge cases
// ============================================================================

#[test]
fn test_truncate_empty_string() {
    let result = truncate("", 10);
    assert_eq!(result, "");
}

#[test]
fn test_truncate_zero_max() {
    let result = truncate("hello", 0);
    assert_eq!(result.len(), 0);
}
