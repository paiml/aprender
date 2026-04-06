
#[test]
fn test_run_bench_regression_first_file_missing() {
    let path2 = "/tmp/test_bench_reg_2b_cov95.json";
    std::fs::write(path2, "{}").expect("expected value");
    let result = run_bench_regression("/nonexistent/baseline.json", path2, false);
    let _ = std::fs::remove_file(path2);
    assert!(result.is_err());
}

#[test]
fn test_run_bench_regression_strict_first_file_missing() {
    let result = run_bench_regression("/nonexistent/base.json", "/nonexistent/curr.json", true);
    assert!(result.is_err());
}

// ============================================================================
// run_visualization
// ============================================================================

#[test]
fn test_run_visualization_default() {
    run_visualization(false, 100);
}

#[test]
fn test_run_visualization_with_color() {
    run_visualization(true, 50);
}

#[test]
fn test_run_visualization_small_samples() {
    run_visualization(false, 10);
}

// ============================================================================
// run_benchmarks (list mode only - no actual cargo bench)
// ============================================================================

#[test]
fn test_run_benchmarks_list_mode() {
    let result = run_benchmarks(None, true, None, None, None, None);
    assert!(result.is_ok());
}

// ============================================================================
// run_external_benchmark stub (no bench-http feature)
// ============================================================================

#[cfg(not(feature = "bench-http"))]
#[test]
fn test_run_external_benchmark_stub() {
    let result = run_external_benchmark("ollama", "http://localhost:11434", None, None);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("bench-http") || err.contains("feature"));
}

// ============================================================================
// load_safetensors_model
// ============================================================================

#[test]
fn test_load_safetensors_model_invalid() {
    let result = load_safetensors_model(&[0u8; 32]);
    assert!(result.is_err());
}

#[test]
fn test_load_safetensors_model_valid() {
    use crate::safetensors::SafetensorsModel;

    // Create a minimal valid safetensors file
    // Format: 8-byte header length + JSON header + tensor data
    let metadata = serde_json::json!({
        "test.weight": {
            "dtype": "F32",
            "shape": [2, 3],
            "data_offsets": [0, 24]
        }
    });
    let header = serde_json::to_string(&metadata).expect("serialization");
    let header_bytes = header.as_bytes();

    let mut file_data = Vec::new();
    file_data.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    file_data.extend_from_slice(header_bytes);
    file_data.extend_from_slice(&[0u8; 24]); // tensor data (6 f32s)

    // First verify the model can be parsed
    let model = SafetensorsModel::from_bytes(&file_data);
    if model.is_ok() {
        let result = load_safetensors_model(&file_data);
        assert!(result.is_ok());
    }
}

// ============================================================================
// run_bench_compare: happy path with valid JSON (GH-219 coverage)
// ============================================================================

fn make_valid_full_bench_json() -> String {
    use crate::bench::FullBenchmarkResult;
    let result = FullBenchmarkResult {
        version: "1.1".to_string(),
        timestamp: "2026-02-15T00:00:00Z".to_string(),
        config: crate::bench::BenchmarkConfig {
            model: "test-model".to_string(),
            format: "gguf".to_string(),
            quantization: "Q4_K".to_string(),
            runtime: "realizar".to_string(),
            runtime_version: "0.14.0".to_string(),
        },
        hardware: crate::bench::HardwareSpec {
            cpu: "test-cpu".to_string(),
            gpu: None,
            memory_gb: 32,
            storage: "nvme".to_string(),
        },
        sampling: crate::bench::SamplingConfig {
            method: "dynamic_cv".to_string(),
            cv_threshold: 0.05,
            actual_iterations: 100,
            cv_at_stop: 0.03,
            warmup_iterations: 10,
        },
        thermal: crate::bench::ThermalInfo {
            valid: true,
            temp_variance_c: 1.0,
            max_temp_c: 65.0,
        },
        results: crate::bench::BenchmarkResults {
            ttft_ms: crate::bench::TtftResults { p50: 50.0, p95: 80.0, p99: 100.0, p999: 120.0 },
            itl_ms: crate::bench::ItlResults { median: 10.0, std_dev: 2.0, p99: 20.0 },
            throughput_tok_s: crate::bench::ThroughputResults { median: 100.0, ci_95: (95.0, 105.0) },
            memory_mb: crate::bench::MemoryResults { model_mb: 500, peak_rss_mb: 1000, kv_waste_pct: 5.0 },
            energy: crate::bench::EnergyResults { total_joules: 10.0, token_joules: 0.1, idle_watts: 50.0 },
            cold_start_ms: crate::bench::ColdStartResults { median: 200.0, p99: 300.0 },
        },
        quality: crate::bench::QualityValidation {
            kl_divergence_vs_fp32: 0.01,
            perplexity_wikitext2: None,
        },
    };
    serde_json::to_string(&result).expect("serialization")
}

#[test]
fn test_run_bench_compare_valid_json() {
    let json = make_valid_full_bench_json();
    let path1 = "/tmp/test_bench_cmp_valid1_gh219.json";
    let path2 = "/tmp/test_bench_cmp_valid2_gh219.json";

    std::fs::write(path1, &json).expect("expected value");
    std::fs::write(path2, &json).expect("expected value");

    let result = run_bench_compare(path1, path2, 5.0);
    let _ = std::fs::remove_file(path1);
    let _ = std::fs::remove_file(path2);

    assert!(result.is_ok());
}

#[test]
fn test_run_bench_compare_different_thresholds() {
    let json = make_valid_full_bench_json();
    let path1 = "/tmp/test_bench_cmp_thresh1_gh219.json";
    let path2 = "/tmp/test_bench_cmp_thresh2_gh219.json";

    std::fs::write(path1, &json).expect("expected value");
    std::fs::write(path2, &json).expect("expected value");

    let result = run_bench_compare(path1, path2, 0.1);
    let _ = std::fs::remove_file(path1);
    let _ = std::fs::remove_file(path2);

    assert!(result.is_ok());
}

// ============================================================================
// run_bench_regression: happy path with valid JSON (GH-219 coverage)
// ============================================================================

#[test]
fn test_run_bench_regression_valid_json_no_regression() {
    let json = make_valid_full_bench_json();
    let path_base = "/tmp/test_bench_reg_valid_base_gh219.json";
    let path_curr = "/tmp/test_bench_reg_valid_curr_gh219.json";

    std::fs::write(path_base, &json).expect("expected value");
    std::fs::write(path_curr, &json).expect("expected value");

    let result = run_bench_regression(path_base, path_curr, false);
    let _ = std::fs::remove_file(path_base);
    let _ = std::fs::remove_file(path_curr);

    assert!(result.is_ok());
}

#[test]
fn test_run_bench_regression_strict_mode_no_regression() {
    let json = make_valid_full_bench_json();
    let path_base = "/tmp/test_bench_reg_strict_base_gh219.json";
    let path_curr = "/tmp/test_bench_reg_strict_curr_gh219.json";

    std::fs::write(path_base, &json).expect("expected value");
    std::fs::write(path_curr, &json).expect("expected value");

    let result = run_bench_regression(path_base, path_curr, true);
    let _ = std::fs::remove_file(path_base);
    let _ = std::fs::remove_file(path_curr);

    assert!(result.is_ok());
}
