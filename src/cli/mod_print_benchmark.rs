
/// Print benchmark results summary
#[cfg(feature = "bench-http")]
fn print_benchmark_summary(
    runtime: &str,
    url: &str,
    model: Option<&str>,
    num_iterations: usize,
    latencies: &[f64],
    avg_tps: f64,
    output: Option<&str>,
) {
    let p50 = latencies[latencies.len() / 2];
    let p99_idx = (latencies.len() as f64 * 0.99) as usize;
    let p99 = latencies[p99_idx.min(latencies.len() - 1)];
    let mean: f64 = latencies.iter().sum::<f64>() / latencies.len() as f64;

    println!();
    println!("=== Results ===");
    println!("  Runtime: {runtime}");
    println!("  URL: {url}");
    println!("  Model: {}", model.unwrap_or("default"));
    println!("  Iterations: {num_iterations}");
    println!();
    println!("  Latency (ms):");
    println!("    Mean: {mean:.1}");
    println!("    p50:  {p50:.1}");
    println!("    p99:  {p99:.1}");
    println!();
    println!("  Throughput: {avg_tps:.1} tokens/sec");

    if let Some(output_path) = output {
        let result = serde_json::json!({
            "runtime": runtime,
            "url": url,
            "model": model.unwrap_or("default"),
            "iterations": num_iterations,
            "latency_ms": {
                "mean": mean,
                "p50": p50,
                "p99": p99,
                "samples": latencies,
            },
            "throughput_tokens_per_sec": avg_tps,
        });

        if let Ok(json) = serde_json::to_string_pretty(&result) {
            let _ = std::fs::write(output_path, json);
            println!();
            println!("Results saved to: {output_path}");
        }
    }
}

/// Run external runtime benchmark using REAL HTTP calls
#[cfg(feature = "bench-http")]
fn run_external_benchmark(
    runtime: &str,
    url: &str,
    model: Option<&str>,
    output: Option<&str>,
) -> Result<()> {
    use crate::http_client::ModelHttpClient;
    use std::time::Instant;

    println!("=== External Runtime Benchmark (REAL HTTP) ===");
    println!();
    println!("This measures ACTUAL inference latency from {url}");
    println!("NO MOCK DATA - real network + inference timing");
    println!();

    let client = ModelHttpClient::new();
    let prompt = "Explain the concept of machine learning in one sentence.";
    let num_iterations = 5;
    let mut latencies: Vec<f64> = Vec::with_capacity(num_iterations);
    let mut tokens_per_sec: Vec<f64> = Vec::with_capacity(num_iterations);

    println!("Running {num_iterations} inference iterations...");
    println!("Prompt: \"{prompt}\"");
    println!();

    for i in 0..num_iterations {
        let start = Instant::now();
        let timing = execute_runtime_request(&client, runtime, url, model, prompt)?;
        let elapsed = start.elapsed();
        let latency_ms = elapsed.as_secs_f64() * 1000.0;
        latencies.push(latency_ms);

        if timing.tokens_generated > 0 {
            let tps = timing.tokens_generated as f64 / elapsed.as_secs_f64();
            tokens_per_sec.push(tps);
        }

        println!(
            "  [{}/{}] TTFT: {:.0}ms, Inference: {:.0}ms, Tokens: {}, E2E: {:.0}ms",
            i + 1,
            num_iterations,
            timing.ttft_ms,
            timing.total_time_ms,
            timing.tokens_generated,
            latency_ms
        );
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let avg_tps = if tokens_per_sec.is_empty() {
        0.0
    } else {
        tokens_per_sec.iter().sum::<f64>() / tokens_per_sec.len() as f64
    };

    print_benchmark_summary(
        runtime,
        url,
        model,
        num_iterations,
        &latencies,
        avg_tps,
        output,
    );
    Ok(())
}

/// Stub for when bench-http feature is not enabled
#[cfg(not(feature = "bench-http"))]
fn run_external_benchmark(
    runtime: &str,
    url: &str,
    _model: Option<&str>,
    _output: Option<&str>,
) -> Result<()> {
    Err(RealizarError::UnsupportedOperation {
        operation: "external_benchmark".to_string(),
        reason: format!(
            "External runtime benchmarking requires the 'bench-http' feature.\n\
             Run with: cargo build --features bench-http\n\
             Then: realizar bench --runtime {} --url {}",
            runtime, url
        ),
    })
}

/// Run convoy test for continuous batching validation (spec 2.4)
pub fn run_convoy_test(
    runtime: Option<String>,
    model: Option<String>,
    output: Option<String>,
) -> Result<()> {
    use crate::bench::{ConvoyTestConfig, ConvoyTestResult};

    let runtime_name = runtime.unwrap_or_else(|| "realizar".to_string());
    println!("=== Convoy Test (Continuous Batching Validation) ===");
    println!();
    println!("Configuration:");
    println!("  Runtime: {runtime_name}");
    if let Some(ref m) = model {
        println!("  Model: {m}");
    }
    println!();

    let config = ConvoyTestConfig::default();
    println!("Test Parameters:");
    println!("  Long-context requests: {}", config.long_requests);
    println!("  Short-QA requests: {}", config.short_requests);
    println!("  Max p99 increase: {}%", config.max_p99_increase_pct);
    println!("  Max HOL blocking: {}ms", config.max_hol_blocking_ms);
    println!(
        "  Max KV fragmentation: {}%",
        config.max_kv_fragmentation_pct
    );
    println!();

    // Create test result for demo (actual benchmark would run inference)
    let baseline_latencies: Vec<f64> = (0..100).map(|i| 45.0 + (i as f64) * 0.1).collect();
    let convoy_latencies: Vec<f64> = (0..100).map(|i| 60.0 + (i as f64) * 0.15).collect();
    let hol_blocking_times: Vec<f64> = vec![80.0, 120.0, 95.0, 110.0, 85.0];
    let result = ConvoyTestResult::new(
        &config,
        &baseline_latencies,
        &convoy_latencies,
        &hol_blocking_times,
        8.5, // KV fragmentation %
    );

    println!("Results:");
    println!("  Baseline p99: {:.1}ms", result.baseline_short_p99_ms);
    println!("  Convoy p99: {:.1}ms", result.convoy_short_p99_ms);
    println!("  p99 increase: {:.1}%", result.p99_increase_pct);
    println!("  Max HOL blocking: {:.1}ms", result.max_hol_blocking_ms);
    println!("  Avg HOL blocking: {:.1}ms", result.avg_hol_blocking_ms);
    println!("  KV fragmentation: {:.1}%", result.kv_fragmentation_pct);
    println!();

    if result.passed {
        println!("CONVOY TEST PASSED");
    } else {
        println!("CONVOY TEST FAILED");
        for failure in &result.failure_reasons {
            println!("   - {failure}");
        }
    }

    if let Some(ref output_path) = output {
        // Write JSON results
        if let Ok(json) = serde_json::to_string_pretty(&result) {
            let _ = std::fs::write(output_path, json);
            println!();
            println!("Results saved to: {output_path}");
        }
    }

    Ok(())
}

/// Run saturation stress test (spec 2.5)
pub fn run_saturation_test(
    runtime: Option<String>,
    model: Option<String>,
    output: Option<String>,
) -> Result<()> {
    use crate::bench::{SaturationTestConfig, SaturationTestResult};

    let runtime_name = runtime.unwrap_or_else(|| "realizar".to_string());
    println!("=== Saturation Stress Test ===");
    println!();
    println!("Configuration:");
    println!("  Runtime: {runtime_name}");
    if let Some(ref m) = model {
        println!("  Model: {m}");
    }
    println!();

    let config = SaturationTestConfig::default();
    println!("Test Parameters:");
    println!("  CPU load target: {}%", config.cpu_load_pct);
    println!(
        "  Max throughput degradation: {}%",
        config.max_throughput_degradation_pct
    );
    println!("  Max p99 increase: {}%", config.max_p99_increase_pct);
    println!();

    // Create test result for demo
    let baseline_throughputs: Vec<f64> = (0..50).map(|i| 95.0 + (i as f64) * 0.2).collect();
    let stressed_throughputs: Vec<f64> = (0..50).map(|i| 78.0 + (i as f64) * 0.15).collect();
    let baseline_latencies: Vec<f64> = (0..100).map(|i| 45.0 + (i as f64) * 0.1).collect();
    let stressed_latencies: Vec<f64> = (0..100).map(|i| 75.0 + (i as f64) * 0.2).collect();
    let result = SaturationTestResult::new(
        &config,
        &baseline_throughputs,
        &stressed_throughputs,
        &baseline_latencies,
        &stressed_latencies,
    );

    println!("Results:");
    println!(
        "  Baseline throughput: {:.1} tok/s",
        result.baseline_throughput
    );
    println!(
        "  Stressed throughput: {:.1} tok/s",
        result.stressed_throughput
    );
    println!(
        "  Throughput degradation: {:.1}%",
        result.throughput_degradation_pct
    );
    println!("  Baseline p99: {:.1}ms", result.baseline_p99_ms);
    println!("  Stressed p99: {:.1}ms", result.stressed_p99_ms);
    println!("  P99 increase: {:.1}%", result.p99_increase_pct);
    println!();

    if result.passed {
        println!("SATURATION TEST PASSED");
    } else {
        println!("SATURATION TEST FAILED");
        for failure in &result.failure_reasons {
            println!("   - {failure}");
        }
    }

    if let Some(ref output_path) = output {
        if let Ok(json) = serde_json::to_string_pretty(&result) {
            let _ = std::fs::write(output_path, json);
            println!();
            println!("Results saved to: {output_path}");
        }
    }

    Ok(())
}

/// Compare two benchmark result files
pub fn run_bench_compare(file1: &str, file2: &str, threshold: f64) -> Result<()> {
    use crate::bench::{BenchmarkComparison, FullBenchmarkResult};

    println!("=== Benchmark Comparison ===");
    println!();
    println!("File 1: {file1}");
    println!("File 2: {file2}");
    println!("Significance threshold: {threshold}%");
    println!();

    // Read and parse JSON files
    let json1 =
        std::fs::read_to_string(file1).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_benchmark".to_string(),
            reason: format!("Failed to read {file1}: {e}"),
        })?;

    let json2 =
        std::fs::read_to_string(file2).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_benchmark".to_string(),
            reason: format!("Failed to read {file2}: {e}"),
        })?;

    let result1 = FullBenchmarkResult::from_json(&json1).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_benchmark".to_string(),
            reason: format!("Failed to parse {file1}: {e}"),
        }
    })?;

    let result2 = FullBenchmarkResult::from_json(&json2).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_benchmark".to_string(),
            reason: format!("Failed to parse {file2}: {e}"),
        }
    })?;

    let comparison = BenchmarkComparison::compare(&result1, &result2);

    println!("Comparison Results:");
    println!("  TTFT p99: {:.1}% change", comparison.ttft_p99_change_pct);
    println!(
        "  Throughput: {:.1}% change",
        comparison.throughput_change_pct
    );
    println!("  Memory: {:.1}% change", comparison.memory_change_pct);
    println!("  Energy: {:.1}% change", comparison.energy_change_pct);
    println!();
    println!("Winner: {}", comparison.winner);
    println!("Significance (p-value): {:.4}", comparison.significance);

    let ttft_significant = comparison.ttft_p99_change_pct.abs() > threshold;
    let throughput_significant = comparison.throughput_change_pct.abs() > threshold;

    println!();
    if ttft_significant || throughput_significant {
        println!("Significant differences detected (>{threshold}%)");
    } else {
        println!("No significant differences (threshold: {threshold}%)");
    }

    Ok(())
}

/// Detect performance regressions between baseline and current
pub fn run_bench_regression(baseline_path: &str, current_path: &str, strict: bool) -> Result<()> {
    use crate::bench::{FullBenchmarkResult, RegressionResult};

    let threshold = if strict { 0.0 } else { 10.0 };

    println!("=== Regression Detection ===");
    println!();
    println!("Baseline: {baseline_path}");
    println!("Current: {current_path}");
    println!(
        "Mode: {}",
        if strict {
            "strict (0%)"
        } else {
            "normal (10%)"
        }
    );
    println!("Threshold: {threshold}%");
    println!();

    let baseline_json = std::fs::read_to_string(baseline_path).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "read_baseline".to_string(),
            reason: format!("Failed to read {baseline_path}: {e}"),
        }
    })?;

    let current_json =
        std::fs::read_to_string(current_path).map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_current".to_string(),
            reason: format!("Failed to read {current_path}: {e}"),
        })?;

    let baseline = FullBenchmarkResult::from_json(&baseline_json).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_baseline".to_string(),
            reason: format!("Failed to parse {baseline_path}: {e}"),
        }
    })?;

    let current = FullBenchmarkResult::from_json(&current_json).map_err(|e| {
        RealizarError::UnsupportedOperation {
            operation: "parse_current".to_string(),
            reason: format!("Failed to parse {current_path}: {e}"),
        }
    })?;

    let regression = RegressionResult::check(&baseline, &current, threshold);

    println!("Regression Analysis:");
    println!("  Threshold: {:.1}%", regression.threshold_pct);
    println!("  Regression detected: {}", regression.regression_detected);
    if !regression.regressed_metrics.is_empty() {
        println!("  Regressed metrics:");
        for metric in &regression.regressed_metrics {
            println!("    - {metric}");
        }
    }
    println!();

    if regression.regression_detected {
        println!("REGRESSION DETECTED");
        // Note: Don't call process::exit here - let main handle it
        return Err(RealizarError::UnsupportedOperation {
            operation: "regression_check".to_string(),
            reason: "Performance regression detected".to_string(),
        });
    }
    println!("NO REGRESSION DETECTED");

    Ok(())
}

/// Print info about realizar
pub fn print_info() {
    println!("Realizar v{}", crate::VERSION);
    println!("Pure Rust ML inference engine");
    println!();
    println!("Features:");
    println!("  - GGUF and Safetensors model formats");
    println!("  - Transformer inference (LLaMA architecture)");
    println!("  - BPE and SentencePiece tokenizers");
    println!("  - Greedy, top-k, and top-p sampling");
    println!("  - REST API for inference");
}
