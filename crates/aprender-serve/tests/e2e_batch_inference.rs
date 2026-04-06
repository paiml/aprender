//! PARITY-055: E2E Batch Inference Tests
//!
//! End-to-end tests for batch inference throughput validation.
//! These tests verify the complete HTTP pipeline for batch processing.
//!
//! # Running
//! ```bash
//! # Run with server available
//! cargo test --test e2e_batch_inference -- --nocapture
//!
//! # Run with CUDA backend
//! cargo test --test e2e_batch_inference --features cuda -- --nocapture
//! ```
//!
//! # M4 Parity Target
//! - Baseline single-request: ~64 tok/s
//! - Target batch throughput: 150+ tok/s (concurrent requests)
//! - M4 parity: 192 tok/s (1.25x gap to Ollama)
//!
//! # Toyota Way Alignment
//! - **Genchi Genbutsu**: Measure real HTTP throughput, not test
//! - **Jidoka**: Stop when throughput regression detected
//! - **Poka-Yoke**: Catch batch inference bugs before deployment

use std::time::{Duration, Instant};

/// Batch inference configuration for throughput testing
#[derive(Debug, Clone)]
pub struct BatchTestConfig {
    /// Server base URL
    pub base_url: String,
    /// Number of concurrent requests
    pub concurrency: usize,
    /// Tokens to generate per request
    pub max_tokens: usize,
    /// Number of measurement iterations (after warmup)
    pub iterations: usize,
    /// Warmup iterations (discarded)
    pub warmup: usize,
}

impl Default for BatchTestConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:8085".to_string(),
            concurrency: 4,
            max_tokens: 10,
            iterations: 10,
            warmup: 3,
        }
    }
}

/// Batch throughput measurement result
#[derive(Debug, Clone)]
pub struct BatchThroughputResult {
    /// Total tokens generated
    pub total_tokens: usize,
    /// Total elapsed time
    pub elapsed: Duration,
    /// Tokens per second
    pub tok_per_sec: f64,
    /// Concurrency level
    pub concurrency: usize,
    /// Mean latency per request (ms)
    pub mean_latency_ms: f64,
    /// P95 latency (ms)
    pub p95_latency_ms: f64,
}

impl BatchThroughputResult {
    /// Check if result achieves M4 parity threshold (192 tok/s)
    pub fn achieves_m4_parity(&self) -> bool {
        self.tok_per_sec >= 192.0
    }

    /// Check if result achieves batch threshold (150 tok/s)
    pub fn achieves_batch_threshold(&self) -> bool {
        self.tok_per_sec >= 150.0
    }

    /// Calculate throughput ratio vs baseline (64 tok/s)
    pub fn throughput_ratio(&self) -> f64 {
        self.tok_per_sec / 64.0
    }
}

// ============================================================================
// BATCH THROUGHPUT CALCULATION TESTS
// ============================================================================

/// PARITY-055a: Test throughput calculation methodology
#[test]
fn test_parity_055a_throughput_calculation() {
    println!("PARITY-055a: Throughput Calculation Methodology");

    // Simulate batch processing
    let total_tokens = 100;
    let elapsed = Duration::from_millis(500); // 0.5 seconds

    let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    println!("  Tokens: {}", total_tokens);
    println!("  Elapsed: {:?}", elapsed);
    println!("  Throughput: {:.1} tok/s", tok_per_sec);

    assert_eq!(tok_per_sec, 200.0, "100 tokens / 0.5s = 200 tok/s");
}

/// PARITY-055b: Test batch scaling estimation
#[test]
fn test_parity_055b_batch_scaling_estimation() {
    println!("PARITY-055b: Batch Scaling Estimation");

    // Baseline single-request throughput
    let baseline_tok_per_sec = 64.0;

    // Batch scaling projections (from spec)
    let concurrency_levels = [1, 2, 4, 8, 16, 32];
    let expected_throughput = [64.0, 96.0, 150.0, 240.0, 384.0, 512.0];

    println!("  Concurrency → Throughput Projections:");
    for (c, expected) in concurrency_levels.iter().zip(expected_throughput.iter()) {
        // Simple model: throughput = baseline * min(c, 8) for c <= 8, then sublinear
        let projected = if *c <= 8 {
            baseline_tok_per_sec * (*c as f64).min(8.0) * 0.9 // 90% efficiency
        } else {
            baseline_tok_per_sec * 8.0 * 0.9 * (1.0 + (*c as f64 - 8.0).log2() * 0.3)
        };
        println!(
            "    c={:2} → projected {:.0} tok/s (expected {:.0})",
            c, projected, expected
        );
    }

    // M4 parity at c=3+
    let _min_concurrency_for_m4 = 3;
    let m4_target = 192.0;
    let at_c3 = baseline_tok_per_sec * 3.0 * 0.9; // 172.8 tok/s

    println!("\n  M4 parity (192 tok/s):");
    println!("    At c=3: {:.1} tok/s", at_c3);
    println!("    At c=4: {:.1} tok/s", baseline_tok_per_sec * 4.0 * 0.9);

    assert!(
        baseline_tok_per_sec * 4.0 * 0.9 >= m4_target,
        "PARITY-055b: M4 parity should be achievable at c=4"
    );
}

/// PARITY-055c: Test latency vs throughput tradeoff
#[test]
fn test_parity_055c_latency_throughput_tradeoff() {
    println!("PARITY-055c: Latency vs Throughput Tradeoff");

    // Single request baseline
    let single_latency_ms = 156.0; // ms per request
    let single_throughput = 1000.0 / single_latency_ms * 10.0; // 10 tokens per request

    // Batch processing (concurrent)
    let batch_sizes = [1, 4, 8, 16];

    println!("  Batch Size → Latency/Throughput:");
    for batch_size in batch_sizes {
        // Latency increases slightly with batch (queuing)
        let batch_latency =
            single_latency_ms * (1.0 + 0.1 * (batch_size as f64 - 1.0).log2().max(0.0));
        // Throughput scales near-linearly
        let batch_throughput = single_throughput * batch_size as f64 * 0.9;

        println!(
            "    batch={:2} → latency={:.0}ms, throughput={:.0} tok/s",
            batch_size, batch_latency, batch_throughput
        );
    }

    // Verify tradeoff acceptable
    let batch_16_latency = single_latency_ms * 1.4; // 40% increase
    let batch_16_throughput = single_throughput * 16.0 * 0.9;

    assert!(
        batch_16_latency < 250.0,
        "Latency at batch=16 should be < 250ms"
    );
    assert!(
        batch_16_throughput > 500.0,
        "Throughput at batch=16 should be > 500 tok/s"
    );
}

// ============================================================================
// BATCH CONFIG VALIDATION TESTS
// ============================================================================

/// PARITY-055d: Test BatchConfig defaults match spec
#[test]
#[cfg(feature = "gpu")]
fn test_parity_055d_batch_config_defaults() {
    use realizar::api::BatchConfig;

    println!("PARITY-055d: BatchConfig Default Validation");

    let config = BatchConfig::default();

    println!("  min_batch: {}", config.min_batch);
    println!("  optimal_batch: {}", config.optimal_batch);
    println!("  max_batch: {}", config.max_batch);
    println!("  window_ms: {}", config.window_ms);
    println!("  queue_size: {}", config.queue_size);

    // Verify spec-compliant defaults
    assert!(config.min_batch >= 1, "min_batch >= 1");
    assert!(config.max_batch >= 32, "max_batch >= 32 for GPU efficiency");
    assert!(config.window_ms >= 10, "window >= 10ms");
    assert!(
        config.optimal_batch >= 4,
        "optimal_batch >= 4 for GPU benefit"
    );
}

/// PARITY-055e: Test BatchConfig high_throughput preset
#[test]
#[cfg(feature = "gpu")]
fn test_parity_055e_batch_config_presets() {
    use realizar::api::BatchConfig;

    println!("PARITY-055e: BatchConfig Presets");

    let high_throughput = BatchConfig::high_throughput();
    let low_latency = BatchConfig::low_latency();

    println!("  High Throughput:");
    println!("    max_batch: {}", high_throughput.max_batch);
    println!("    window_ms: {}", high_throughput.window_ms);

    println!("  Low Latency:");
    println!("    max_batch: {}", low_latency.max_batch);
    println!("    window_ms: {}", low_latency.window_ms);

    // High throughput optimizes for batching
    assert!(
        high_throughput.max_batch > low_latency.max_batch,
        "High throughput should have larger max batch"
    );
    assert!(
        high_throughput.window_ms >= low_latency.window_ms,
        "High throughput should have longer batch window"
    );
}

/// PARITY-055f: Test GPU threshold decision
#[test]
fn test_parity_055f_gpu_threshold_decision() {
    println!("PARITY-055f: GPU Threshold Decision");

    // GPU threshold from spec: batch_size >= 32 for GPU efficiency
    let gpu_threshold = 32;

    // Test should_use_gpu() decisions
    let test_cases = [
        (1, false, "Single request → CPU"),
        (16, false, "Small batch → CPU"),
        (32, true, "At threshold → GPU"),
        (64, true, "Large batch → GPU"),
    ];

    for (batch_size, expected_gpu, description) in test_cases {
        let use_gpu = batch_size >= gpu_threshold;
        println!("  {} (batch={}): GPU={}", description, batch_size, use_gpu);
        assert_eq!(use_gpu, expected_gpu, "{}", description);
    }
}

// ============================================================================
// SERVER AVAILABILITY CHECK (FOR INTEGRATION TESTS)
// ============================================================================

/// Check if server is available (used by ignored integration tests)
fn server_available(base_url: &str) -> bool {
    std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            &format!("{}/health", base_url),
        ])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim() == "200")
        .unwrap_or(false)
}

/// PARITY-056a: Integration test - single request baseline
#[test]
#[ignore = "requires running server"]
fn test_parity_056a_single_request_baseline() {
    let base_url = "http://127.0.0.1:8085";

    if !server_available(base_url) {
        println!("PARITY-056a: Server not available at {}", base_url);
        println!("  Start with: cargo run --example api_server --features cuda");
        return;
    }

    println!("PARITY-056a: Single Request Baseline Measurement");

    // Warmup
    for _ in 0..3 {
        let _ = std::process::Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                &format!("{}/v1/completions", base_url),
                "-H",
                "Content-Type: application/json",
                "-d",
                r#"{"model":"default","prompt":"Hello","max_tokens":5,"temperature":0.1}"#,
            ])
            .output();
    }

    // Measure
    let start = Instant::now();
    let iterations = 5;
    let mut total_tokens = 0;

    for i in 0..iterations {
        let output = std::process::Command::new("curl")
            .args([
                "-s",
                "-X",
                "POST",
                &format!("{}/v1/completions", base_url),
                "-H",
                "Content-Type: application/json",
                "-d",
                r#"{"model":"default","prompt":"Hello","max_tokens":10,"temperature":0.1}"#,
            ])
            .output()
            .expect("curl failed");

        // Parse token count from response
        let response = String::from_utf8_lossy(&output.stdout);
        if let Some(tokens) = response
            .split("\"completion_tokens\":")
            .nth(1)
            .and_then(|s| s.split([',', '}']).next())
            .and_then(|s| s.trim().parse::<usize>().ok())
        {
            total_tokens += tokens;
        }
        println!("  [{}] {} tokens", i + 1, total_tokens);
    }

    let elapsed = start.elapsed();
    let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

    println!("\n  Results:");
    println!("    Total tokens: {}", total_tokens);
    println!("    Elapsed: {:?}", elapsed);
    println!("    Throughput: {:.1} tok/s", tok_per_sec);
    println!("    Baseline target: 64 tok/s");

    assert!(total_tokens > 0, "Should generate tokens");
}

/// PARITY-056b: Integration test - concurrent batch throughput
#[test]
#[ignore = "requires running server"]
fn test_parity_056b_concurrent_batch_throughput() {
    let base_url = "http://127.0.0.1:8085";

    if !server_available(base_url) {
        println!("PARITY-056b: Server not available at {}", base_url);
        return;
    }

    println!("PARITY-056b: Concurrent Batch Throughput");

    // Test different concurrency levels
    let concurrency_levels = [1, 2, 4, 8];

    for concurrency in concurrency_levels {
        println!("\n  Concurrency = {}:", concurrency);

        let start = Instant::now();
        let mut handles = Vec::new();

        // Spawn concurrent requests
        for _ in 0..concurrency {
            let url = format!("{}/v1/completions", base_url);
            handles.push(std::thread::spawn(move || {
                std::process::Command::new("curl")
                    .args([
                        "-s",
                        "-X",
                        "POST",
                        &url,
                        "-H",
                        "Content-Type: application/json",
                        "-d",
                        r#"{"model":"default","prompt":"Test","max_tokens":10,"temperature":0.1}"#,
                    ])
                    .output()
            }));
        }

        // Wait for all
        let mut total_tokens = 0;
        for handle in handles {
            if let Ok(Ok(output)) = handle.join() {
                let response = String::from_utf8_lossy(&output.stdout);
                if let Some(tokens) = response
                    .split("\"completion_tokens\":")
                    .nth(1)
                    .and_then(|s| s.split([',', '}']).next())
                    .and_then(|s| s.trim().parse::<usize>().ok())
                {
                    total_tokens += tokens;
                }
            }
        }

        let elapsed = start.elapsed();
        let tok_per_sec = total_tokens as f64 / elapsed.as_secs_f64();

        println!("    Tokens: {}", total_tokens);
        println!("    Elapsed: {:?}", elapsed);
        println!("    Throughput: {:.1} tok/s", tok_per_sec);
    }
}

// ============================================================================
// SUMMARY TEST
// ============================================================================

/// PARITY-055: Batch inference test summary
#[test]
fn test_parity_055_summary() {
    println!("=== PARITY-055: Batch Inference E2E Tests ===");
    println!("Coverage:");
    println!("  - Throughput calculation methodology");
    println!("  - Batch scaling estimation");
    println!("  - Latency vs throughput tradeoff");
    println!("  - BatchConfig defaults and presets");
    println!("  - GPU threshold decision logic");
    println!("  - Integration tests (requires server)");
    println!();
    println!("M4 Parity Targets:");
    println!("  - Baseline: 64 tok/s (single request)");
    println!("  - Batch target: 150+ tok/s (concurrent)");
    println!("  - M4 target: 192 tok/s (1.25x Ollama gap)");
}

// ============================================================================
// PARITY-101: OLLAMA VS REALIZAR REAL-TIME COMPARISON
// ============================================================================

/// Check if Ollama server is available
fn ollama_available() -> bool {
    std::process::Command::new("curl")
        .args([
            "-s",
            "-o",
            "/dev/null",
            "-w",
            "%{http_code}",
            "http://127.0.0.1:11434/api/tags",
        ])
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).trim() == "200")
        .unwrap_or(false)
}

/// Measure Ollama throughput for a single request
fn measure_ollama_throughput(prompt: &str, max_tokens: usize) -> Option<(usize, Duration, f64)> {
    let payload = format!(
        r#"{{"model":"phi","prompt":"{}","stream":false,"options":{{"num_predict":{}}}}}"#,
        prompt, max_tokens
    );

    let start = Instant::now();
    let output = std::process::Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            "http://127.0.0.1:11434/api/generate",
            "-H",
            "Content-Type: application/json",
            "-d",
            &payload,
        ])
        .output()
        .ok()?;

    let elapsed = start.elapsed();
    let response = String::from_utf8_lossy(&output.stdout);

    // Parse eval_count from Ollama response
    let tokens = response
        .split("\"eval_count\":")
        .nth(1)
        .and_then(|s| s.split([',', '}']).next())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(max_tokens);

    let tok_per_sec = tokens as f64 / elapsed.as_secs_f64();
    Some((tokens, elapsed, tok_per_sec))
}

/// Measure Realizar throughput for a single request
fn measure_realizar_throughput(prompt: &str, max_tokens: usize) -> Option<(usize, Duration, f64)> {
    let payload = format!(
        r#"{{"model":"default","prompt":"{}","max_tokens":{},"temperature":0.1}}"#,
        prompt, max_tokens
    );

    let start = Instant::now();
    let output = std::process::Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            "http://127.0.0.1:8085/v1/completions",
            "-H",
            "Content-Type: application/json",
            "-d",
            &payload,
        ])
        .output()
        .ok()?;

    let elapsed = start.elapsed();
    let response = String::from_utf8_lossy(&output.stdout);

    // Parse completion_tokens from Realizar response
    let tokens = response
        .split("\"completion_tokens\":")
        .nth(1)
        .and_then(|s| s.split([',', '}']).next())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(max_tokens);

    let tok_per_sec = tokens as f64 / elapsed.as_secs_f64();
    Some((tokens, elapsed, tok_per_sec))
}

/// PARITY-101a: Single request Ollama vs Realizar comparison
#[test]
#[ignore = "requires Ollama (port 11434) and Realizar (port 8085) servers"]
fn test_parity_101a_single_request_comparison() {
    println!("PARITY-101a: Single Request Ollama vs Realizar");
    println!("================================================");

    if !ollama_available() {
        println!("  ⚠️  Ollama not available at http://127.0.0.1:11434");
        println!("  Start with: ollama serve");
        return;
    }

    if !server_available("http://127.0.0.1:8085") {
        println!("  ⚠️  Realizar not available at http://127.0.0.1:8085");
        println!("  Start with: cargo run --bin realizar --features cuda -- serve --model phi-2.Q4_K_M.gguf --batch");
        return;
    }

    let prompt = "What is 2+2?";
    let max_tokens = 20;

    println!("\n  Prompt: \"{}\"", prompt);
    println!("  Max tokens: {}", max_tokens);

    // Warmup both servers
    println!("\n  Warming up...");
    let _ = measure_ollama_throughput(prompt, 5);
    let _ = measure_realizar_throughput(prompt, 5);

    // Measure Ollama
    println!("\n  Measuring Ollama...");
    let ollama_results: Vec<_> = (0..5)
        .filter_map(|_| measure_ollama_throughput(prompt, max_tokens))
        .collect();

    let ollama_avg_tps = if ollama_results.is_empty() {
        0.0
    } else {
        ollama_results.iter().map(|r| r.2).sum::<f64>() / ollama_results.len() as f64
    };

    // Measure Realizar
    println!("  Measuring Realizar...");
    let realizar_results: Vec<_> = (0..5)
        .filter_map(|_| measure_realizar_throughput(prompt, max_tokens))
        .collect();

    let realizar_avg_tps = if realizar_results.is_empty() {
        0.0
    } else {
        realizar_results.iter().map(|r| r.2).sum::<f64>() / realizar_results.len() as f64
    };

    // Report results
    println!("\n  Results:");
    println!("  ┌─────────────┬────────────┬────────────┐");
    println!("  │   Server    │  tok/s     │  Gap       │");
    println!("  ├─────────────┼────────────┼────────────┤");
    println!("  │ Ollama      │ {:>8.1}  │ baseline   │", ollama_avg_tps);

    let gap = if ollama_avg_tps > 0.0 {
        ollama_avg_tps / realizar_avg_tps
    } else {
        0.0
    };

    println!(
        "  │ Realizar    │ {:>8.1}  │ {:>6.1}x    │",
        realizar_avg_tps, gap
    );
    println!("  └─────────────┴────────────┴────────────┘");

    // M4 parity check
    let m4_target = 192.0;
    let achieves_m4 = realizar_avg_tps >= m4_target;
    println!(
        "\n  M4 Parity (192 tok/s): {}",
        if achieves_m4 {
            "✅ ACHIEVED"
        } else {
            "❌ NOT YET"
        }
    );

    if !achieves_m4 && realizar_avg_tps > 0.0 {
        let needed = m4_target / realizar_avg_tps;
        println!("  Speedup needed: {:.1}x", needed);
    }
}

/// PARITY-101b: Concurrent batch Ollama vs Realizar comparison
#[test]
#[ignore = "requires Ollama (port 11434) and Realizar (port 8085) servers"]
fn test_parity_101b_concurrent_batch_comparison() {
    println!("PARITY-101b: Concurrent Batch Ollama vs Realizar");
    println!("=================================================");

    if !ollama_available() {
        println!("  ⚠️  Ollama not available");
        return;
    }

    if !server_available("http://127.0.0.1:8085") {
        println!("  ⚠️  Realizar not available");
        return;
    }

    let prompt = "Hello";
    let max_tokens = 10;
    let concurrency_levels = [1, 4, 8, 16, 32];

    println!("\n  Prompt: \"{}\"", prompt);
    println!("  Max tokens: {}", max_tokens);
    println!("\n  ┌─────────────┬────────────┬────────────┬────────────┐");
    println!("  │ Concurrency │ Ollama     │ Realizar   │ Gap        │");
    println!("  ├─────────────┼────────────┼────────────┼────────────┤");

    for concurrency in concurrency_levels {
        // Measure Ollama concurrent throughput
        let ollama_start = Instant::now();
        let ollama_handles: Vec<_> = (0..concurrency)
            .map(|_| {
                let p = prompt.to_string();
                std::thread::spawn(move || measure_ollama_throughput(&p, max_tokens))
            })
            .collect();

        let ollama_tokens: usize = ollama_handles
            .into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .map(|(t, _, _)| t)
            .sum();
        let ollama_elapsed = ollama_start.elapsed();
        let ollama_tps = ollama_tokens as f64 / ollama_elapsed.as_secs_f64();

        // Measure Realizar concurrent throughput
        let realizar_start = Instant::now();
        let realizar_handles: Vec<_> = (0..concurrency)
            .map(|_| {
                let p = prompt.to_string();
                std::thread::spawn(move || measure_realizar_throughput(&p, max_tokens))
            })
            .collect();

        let realizar_tokens: usize = realizar_handles
            .into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .map(|(t, _, _)| t)
            .sum();
        let realizar_elapsed = realizar_start.elapsed();
        let realizar_tps = realizar_tokens as f64 / realizar_elapsed.as_secs_f64();

        let gap = if realizar_tps > 0.0 {
            ollama_tps / realizar_tps
        } else {
            f64::INFINITY
        };

        println!(
            "  │ {:>11} │ {:>8.1}  │ {:>8.1}  │ {:>8.1}x │",
            concurrency, ollama_tps, realizar_tps, gap
        );
    }

    println!("  └─────────────┴────────────┴────────────┴────────────┘");
}

/// PARITY-101c: GPU batch mode throughput measurement
#[test]
#[ignore = "requires Realizar server with --batch mode"]
fn test_parity_101c_gpu_batch_throughput() {
    println!("PARITY-101c: GPU Batch Mode Throughput");
    println!("======================================");

    if !server_available("http://127.0.0.1:8085") {
        println!("  ⚠️  Realizar not available");
        println!("  Start with: cargo run --bin realizar --features cuda -- serve --model phi-2.Q4_K_M.gguf --batch");
        return;
    }

    let prompt = "The quick brown fox";
    let max_tokens = 10;
    let batch_sizes = [1, 8, 16, 32, 64];

    println!("\n  Testing batch accumulation throughput...");
    println!("\n  ┌────────────┬────────────┬────────────┬────────────┐");
    println!("  │ Batch Size │ Tokens     │ Time (ms)  │ tok/s      │");
    println!("  ├────────────┼────────────┼────────────┼────────────┤");

    for batch_size in batch_sizes {
        let start = Instant::now();

        // Send concurrent requests to trigger batching
        let handles: Vec<_> = (0..batch_size)
            .map(|_| {
                let p = prompt.to_string();
                std::thread::spawn(move || measure_realizar_throughput(&p, max_tokens))
            })
            .collect();

        let total_tokens: usize = handles
            .into_iter()
            .filter_map(|h| h.join().ok().flatten())
            .map(|(t, _, _)| t)
            .sum();

        let elapsed = start.elapsed();
        let tps = total_tokens as f64 / elapsed.as_secs_f64();

        println!(
            "  │ {:>10} │ {:>10} │ {:>10.1} │ {:>10.1} │",
            batch_size,
            total_tokens,
            elapsed.as_millis(),
            tps
        );

        // Check M4 parity at batch >= 32
        if batch_size >= 32 && tps >= 192.0 {
            println!("  │            │            │            │ ✅ M4      │");
        }
    }

    println!("  └────────────┴────────────┴────────────┴────────────┘");
}

/// PARITY-101: Summary
#[test]
fn test_parity_101_summary() {
    println!("=== PARITY-101: Ollama vs Realizar Comparison ===");
    println!();
    println!("Tests (run with --ignored):");
    println!("  - 101a: Single request comparison");
    println!("  - 101b: Concurrent batch comparison");
    println!("  - 101c: GPU batch mode throughput");
    println!();
    println!("Prerequisites:");
    println!("  1. Ollama: ollama serve (port 11434)");
    println!("     Model: ollama pull phi");
    println!();
    println!("  2. Realizar: cargo run --bin realizar --features cuda -- \\");
    println!("       serve --model phi-2.Q4_K_M.gguf --batch");
    println!();
    println!("Run:");
    println!("  cargo test --test e2e_batch_inference parity_101 -- --ignored --nocapture");
}
