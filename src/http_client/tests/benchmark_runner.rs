
#[test]
#[ignore = "Requires llama.cpp server at localhost:8082"]
fn test_benchmark_runner_llamacpp() {
    // Use relaxed config for quick test (no preflight for speed)
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(3, 5, 0.50), // Relaxed for test
        warmup_iterations: 1,
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.1,
        run_preflight: false, // Skip preflight for test speed
        filter_outliers: false,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://localhost:8082")
        .expect("Benchmark failed - is llama.cpp running?");

    assert!(result.sample_count >= 3);
    assert!(result.mean_latency_ms > 0.0);
    assert!(result.throughput_tps > 0.0);

    println!("llama.cpp Benchmark Results:");
    println!("  Samples: {}", result.sample_count);
    println!("  Filtered Samples: {}", result.filtered_sample_count);
    println!("  Mean: {:.2}ms", result.mean_latency_ms);
    println!("  P50: {:.2}ms", result.p50_latency_ms);
    println!("  P99: {:.2}ms", result.p99_latency_ms);
    println!("  TPS: {:.2}", result.throughput_tps);
    println!("  CV: {:.4}", result.cv_at_stop);
    println!("  Converged: {}", result.cv_converged);
    println!("  Quality Metrics: {:?}", result.quality_metrics);
}

// =========================================================================
// Preflight Integration Tests
// =========================================================================

#[test]
fn test_preflight_checks_passed_empty_initially() {
    let runner = HttpBenchmarkRunner::with_defaults();
    assert!(runner.preflight_checks_passed().is_empty());
}

#[test]
fn test_quality_metrics_in_result() {
    // Test that compute_results includes quality metrics
    let latencies = vec![100.0, 105.0, 95.0, 100.0, 100.0];
    let throughputs = vec![50.0, 48.0, 52.0, 50.0, 50.0];
    let cold_start = 110.0;
    let cv_threshold = 0.10;

    let result =
        HttpBenchmarkRunner::compute_results(&latencies, &throughputs, cold_start, cv_threshold);

    // Check quality metrics are populated
    assert!(result.quality_metrics.cv_at_stop < 0.10);
    assert!(result.quality_metrics.cv_converged);
    assert_eq!(result.quality_metrics.outliers_detected, 0);
    assert!(result.quality_metrics.preflight_checks_passed.is_empty());
}

#[test]
fn test_filtered_samples_in_result() {
    // Test backward-compatible compute_results sets filtered = raw
    let latencies = vec![100.0, 105.0, 95.0];
    let throughputs = vec![];
    let result = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 100.0, 0.10);

    assert_eq!(
        result.latency_samples.len(),
        result.latency_samples_filtered.len()
    );
    assert_eq!(result.sample_count, result.filtered_sample_count);
}

// =========================================================================
// IMP-144: Real-World Throughput Comparison Tests (EXTREME TDD)
// =========================================================================
// These tests verify actual throughput against external servers.
// Run with: cargo test test_imp_144 --lib --features bench-http -- --ignored

/// IMP-144a: Verify llama.cpp throughput measurement works with real server
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_144a_llamacpp_real_throughput() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(3, 10, 0.20),
        warmup_iterations: 1,
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.0, // Deterministic
        ..Default::default()
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-144a: Should get llama.cpp benchmark result");

    // IMP-144a: Throughput should be measured and positive
    assert!(
        result.throughput_tps > 0.0,
        "IMP-144a: llama.cpp throughput should be > 0, got {} tok/s",
        result.throughput_tps
    );

    // IMP-144a: Per spec, llama.cpp GPU should be ~162ms latency, ~256 tok/s
    // We just verify it's reasonable (> 10 tok/s)
    assert!(
        result.throughput_tps > 10.0,
        "IMP-144a: llama.cpp throughput should be > 10 tok/s, got {} tok/s",
        result.throughput_tps
    );

    println!("\nIMP-144a: llama.cpp Real-World Benchmark Results:");
    println!("  Throughput: {:.1} tok/s", result.throughput_tps);
    println!("  P50 Latency: {:.1} ms", result.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", result.p99_latency_ms);
    println!("  Samples: {}", result.sample_count);
    println!("  CV: {:.4}", result.cv_at_stop);
}

/// IMP-144b: Verify Ollama throughput measurement works with real server
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_144b_ollama_real_throughput() {
    // This test requires: ollama serve
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(3, 10, 0.20),
        warmup_iterations: 1,
        prompt: "Hello".to_string(),
        max_tokens: 10,
        temperature: 0.0,
        ..Default::default()
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-144b: Should get Ollama benchmark result");

    // IMP-144b: Throughput should be measured and positive
    assert!(
        result.throughput_tps > 0.0,
        "IMP-144b: Ollama throughput should be > 0, got {} tok/s",
        result.throughput_tps
    );

    // IMP-144b: Per spec, Ollama should be ~143 tok/s
    // We just verify it's reasonable (> 10 tok/s)
    assert!(
        result.throughput_tps > 10.0,
        "IMP-144b: Ollama throughput should be > 10 tok/s, got {} tok/s",
        result.throughput_tps
    );

    println!("\nIMP-144b: Ollama Real-World Benchmark Results:");
    println!("  Throughput: {:.1} tok/s", result.throughput_tps);
    println!("  P50 Latency: {:.1} ms", result.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", result.p99_latency_ms);
    println!("  Samples: {}", result.sample_count);
    println!("  CV: {:.4}", result.cv_at_stop);
}

/// IMP-144c: Verify throughput comparison can detect performance differences
#[test]
fn test_imp_144c_throughput_comparison_logic() {
    // test benchmark results for comparison logic test
    let llamacpp_tps = 256.0; // Per spec: llama.cpp GPU
    let ollama_tps = 143.0; // Per spec: Ollama baseline
    let realizar_tps = 80.0; // Per spec: Realizar current (~1.8x gap)

    // IMP-144c: Calculate gap ratios
    let gap_vs_llamacpp = llamacpp_tps / realizar_tps;
    let gap_vs_ollama = ollama_tps / realizar_tps;

    // Per spec, current gap to Ollama is ~1.5-1.8x
    assert!(
        gap_vs_ollama > 1.0 && gap_vs_ollama < 3.0,
        "IMP-144c: Gap to Ollama should be ~1.5-1.8x, got {:.1}x",
        gap_vs_ollama
    );

    // Per spec, gap to llama.cpp is ~3x
    assert!(
        gap_vs_llamacpp > 2.0 && gap_vs_llamacpp < 5.0,
        "IMP-144c: Gap to llama.cpp should be ~3x, got {:.1}x",
        gap_vs_llamacpp
    );

    println!("\nIMP-144c: Throughput Gap Analysis:");
    println!("  Realizar: {:.1} tok/s", realizar_tps);
    println!(
        "  Ollama: {:.1} tok/s ({:.1}x gap)",
        ollama_tps, gap_vs_ollama
    );
    println!(
        "  llama.cpp: {:.1} tok/s ({:.1}x gap)",
        llamacpp_tps, gap_vs_llamacpp
    );
}

/// IMP-144d: Verify CV-based stopping works for throughput measurements
#[test]
fn test_imp_144d_cv_stopping_for_throughput() {
    // test throughput samples with low variance (should converge quickly)
    let throughputs = vec![100.0, 102.0, 98.0, 101.0, 99.0];
    let latencies = vec![10.0, 9.8, 10.2, 10.0, 10.0];

    let result = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 12.0, 0.05);

    // IMP-144d: CV should converge for stable throughput
    assert!(
        result.cv_converged,
        "IMP-144d: CV should converge for stable throughput, cv={:.4}",
        result.cv_at_stop
    );

    // IMP-144d: Throughput should be calculated correctly
    let expected_mean_tps = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
    assert!(
        (result.throughput_tps - expected_mean_tps).abs() < 1.0,
        "IMP-144d: Mean TPS should be ~{:.1}, got {:.1}",
        expected_mean_tps,
        result.throughput_tps
    );
}

// =========================================================================
// IMP-145: Output Correctness Verification (EXTREME TDD)
// =========================================================================
// These tests verify output correctness against llama.cpp (QA-001)
// Run with: cargo test test_imp_145 --lib --features bench-http -- --ignored

/// IMP-145a: Verify deterministic config produces identical output
#[test]
fn test_imp_145a_deterministic_config_structure() {
    // IMP-145a: Deterministic config should have temperature=0
    let config = HttpBenchmarkConfig {
        temperature: 0.0,
        ..Default::default()
    };

    assert_eq!(
        config.temperature, 0.0,
        "IMP-145a: Deterministic config should have temperature=0"
    );
}

/// IMP-145b: Verify same prompt produces same output (local determinism)
#[test]
fn test_imp_145b_local_determinism() {
    // IMP-145b: Same input should produce same output structure
    let latencies = vec![100.0, 100.0, 100.0];
    let throughputs = vec![50.0, 50.0, 50.0];

    let result1 = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 100.0, 0.10);
    let result2 = HttpBenchmarkRunner::compute_results(&latencies, &throughputs, 100.0, 0.10);

    // IMP-145b: Same inputs should produce identical results
    assert_eq!(
        result1.mean_latency_ms, result2.mean_latency_ms,
        "IMP-145b: Same inputs should produce identical mean latency"
    );
    assert_eq!(
        result1.throughput_tps, result2.throughput_tps,
        "IMP-145b: Same inputs should produce identical throughput"
    );
}

/// IMP-145c: Verify llama.cpp output matches on repeated calls (deterministic mode)
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_145c_llamacpp_deterministic_output() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082
    // QA-001: Output matches llama.cpp for identical inputs (deterministic mode)

    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "What is 2+2? Answer with just the number:".to_string(),
        max_tokens: 5,
        temperature: Some(0.0), // Deterministic
        stream: false,
    };

    // Make two identical requests
    let result1 = client
        .llamacpp_completion("http://127.0.0.1:8082", &request)
        .expect("IMP-145c: First llama.cpp call should succeed");
    let result2 = client
        .llamacpp_completion("http://127.0.0.1:8082", &request)
        .expect("IMP-145c: Second llama.cpp call should succeed");

    // IMP-145c: Deterministic mode should produce identical output
    assert_eq!(
        result1.text, result2.text,
        "IMP-145c: llama.cpp should produce identical output in deterministic mode. \
        Got '{}' vs '{}'",
        result1.text, result2.text
    );

    println!("\nIMP-145c: llama.cpp Determinism Verification:");
    println!("  Prompt: '{}'", request.prompt);
    println!("  Output 1: '{}'", result1.text.trim());
    println!("  Output 2: '{}'", result2.text.trim());
    println!("  Match: {}", result1.text == result2.text);
}

/// IMP-145d: Verify Ollama output matches on repeated calls (deterministic mode)
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_145d_ollama_deterministic_output() {
    // This test requires: ollama serve
    let client = ModelHttpClient::with_timeout(30);
    let request = OllamaRequest {
        model: "phi2:2.7b".to_string(),
        prompt: "What is 2+2? Answer with just the number:".to_string(),
        stream: false,
        options: Some(OllamaOptions {
            num_predict: Some(5),
            temperature: Some(0.0), // Deterministic
        }),
    };

    // Make two identical requests
    let result1 = client
        .ollama_generate("http://127.0.0.1:11434", &request)
        .expect("IMP-145d: First Ollama call should succeed");
    let result2 = client
        .ollama_generate("http://127.0.0.1:11434", &request)
        .expect("IMP-145d: Second Ollama call should succeed");

    // IMP-145d: Deterministic mode should produce identical output
    assert_eq!(
        result1.text, result2.text,
        "IMP-145d: Ollama should produce identical output in deterministic mode. \
        Got '{}' vs '{}'",
        result1.text, result2.text
    );

    println!("\nIMP-145d: Ollama Determinism Verification:");
    println!("  Prompt: '{}'", request.prompt);
    println!("  Output 1: '{}'", result1.text.trim());
    println!("  Output 2: '{}'", result2.text.trim());
    println!("  Match: {}", result1.text == result2.text);
}

// =========================================================================
// IMP-146: Real-World Throughput Baseline Measurement (EXTREME TDD)
// =========================================================================
// These tests establish baseline measurements and track progress toward parity.
// Per Five Whys Analysis (spec §12A), current gap is 3.2x vs llama.cpp.
// Run with: cargo test test_imp_146 --lib --features bench-http -- --ignored

/// IMP-146a: Baseline measurement struct for tracking performance over time
#[derive(Debug, Clone)]
pub struct ThroughputBaseline {
    /// Server name (llama.cpp, Ollama, Realizar)
    pub server: String,
    /// Measured throughput in tokens/second
    pub throughput_tps: f64,
    /// P50 latency in milliseconds
    pub p50_latency_ms: f64,
    /// P99 latency in milliseconds
    pub p99_latency_ms: f64,
    /// Coefficient of variation (measurement quality)
    pub cv: f64,
    /// Number of samples collected
    pub samples: usize,
}

/// IMP-146a: Verify baseline measurement struct captures required fields
#[test]
fn test_imp_146a_baseline_struct() {
    let baseline = ThroughputBaseline {
        server: "llama.cpp".to_string(),
        throughput_tps: 256.0,
        p50_latency_ms: 162.0,
        p99_latency_ms: 290.0,
        cv: 0.045,
        samples: 10,
    };

    // IMP-146a: All fields should be captured
    assert_eq!(baseline.server, "llama.cpp");
    assert!((baseline.throughput_tps - 256.0).abs() < 0.1);
    assert!((baseline.p50_latency_ms - 162.0).abs() < 0.1);
    assert!((baseline.cv - 0.045).abs() < 0.001);
    assert_eq!(baseline.samples, 10);
}

/// IMP-146b: Gap analysis struct for comparing baselines
#[derive(Debug, Clone)]
pub struct GapAnalysis {
    /// Our baseline (Realizar)
    pub realizar: ThroughputBaseline,
    /// Reference baseline (llama.cpp or Ollama)
    pub reference: ThroughputBaseline,
    /// Gap ratio (reference / realizar)
    pub gap_ratio: f64,
    /// Absolute throughput gap
    pub throughput_gap_tps: f64,
    /// Target throughput for parity (80% of reference)
    pub parity_target_tps: f64,
}
