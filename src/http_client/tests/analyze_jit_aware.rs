
impl JitAwareBenchmark {
    pub fn analyze(samples: &[f64]) -> Self {
        let detector = WarmupDetector::default_detector();
        let raw_stats = ThroughputWithVariance::from_samples(samples);
        let warmup_result = detector.detect_warmup(samples);
        let filtered_stats = warmup_result.steady_state_stats();

        let improvement = if raw_stats.mean_tps > 0.0 {
            ((filtered_stats.mean_tps - raw_stats.mean_tps) / raw_stats.mean_tps) * 100.0
        } else {
            0.0
        };

        Self {
            detector,
            raw_stats,
            filtered_stats,
            warmup_result,
            improvement_percent: improvement,
        }
    }

    /// Check if JIT filtering made a significant difference
    pub fn filtering_significant(&self) -> bool {
        self.warmup_result.warmup_detected && self.improvement_percent.abs() > 5.0
    }
}

/// IMP-161b: Test JIT-aware benchmark analysis
#[test]
fn test_imp_161b_jit_aware_benchmark() {
    // Simulate JIT warmup scenario
    let samples = vec![
        40.0, 60.0, 80.0, 90.0, 95.0, // JIT warming up
        100.0, 98.0, 102.0, 99.0, 101.0, 100.0, 99.0, 101.0, 100.0, 98.0, // JIT hot
    ];

    let analysis = JitAwareBenchmark::analyze(&samples);

    // IMP-161b: Filtered mean should be higher than raw
    assert!(
        analysis.filtered_stats.mean_tps > analysis.raw_stats.mean_tps,
        "IMP-161b: Filtered mean should be higher after removing warmup"
    );

    // IMP-161b: Should show significant improvement
    assert!(
        analysis.improvement_percent > 5.0,
        "IMP-161b: Should show >5% improvement, got {:.2}%",
        analysis.improvement_percent
    );

    // IMP-161b: Filtering should be significant
    assert!(
        analysis.filtering_significant(),
        "IMP-161b: JIT filtering should be significant"
    );

    println!("\nIMP-161b: JIT-Aware Benchmark:");
    println!(
        "  Raw mean: {:.2} tok/s (n={})",
        analysis.raw_stats.mean_tps, analysis.raw_stats.sample_count
    );
    println!(
        "  Filtered mean: {:.2} tok/s (n={})",
        analysis.filtered_stats.mean_tps, analysis.filtered_stats.sample_count
    );
    println!("  Improvement: {:.2}%", analysis.improvement_percent);
    println!(
        "  Warmup removed: {} iterations",
        analysis.warmup_result.warmup_iterations
    );
    println!(
        "  Filtering significant: {}",
        analysis.filtering_significant()
    );
}

/// IMP-161c: Cold vs warm start detection
#[derive(Debug, Clone)]
pub struct ColdWarmComparison {
    /// Cold start measurement (first request)
    pub cold_latency_ms: f64,
    /// Warm start measurement (subsequent average)
    pub warm_latency_ms: f64,
    /// Cold start penalty ratio
    pub cold_penalty_ratio: f64,
    /// Whether cold start penalty is significant (>2x)
    pub significant_cold_penalty: bool,
}

impl ColdWarmComparison {
    pub fn analyze(latencies: &[f64]) -> Self {
        if latencies.is_empty() {
            return Self {
                cold_latency_ms: 0.0,
                warm_latency_ms: 0.0,
                cold_penalty_ratio: 1.0,
                significant_cold_penalty: false,
            };
        }

        let cold_latency = latencies[0];
        let warm_latency = if latencies.len() > 1 {
            latencies[1..].iter().sum::<f64>() / (latencies.len() - 1) as f64
        } else {
            cold_latency
        };

        let penalty_ratio = if warm_latency > 0.0 {
            cold_latency / warm_latency
        } else {
            1.0
        };

        Self {
            cold_latency_ms: cold_latency,
            warm_latency_ms: warm_latency,
            cold_penalty_ratio: penalty_ratio,
            significant_cold_penalty: penalty_ratio > 2.0,
        }
    }
}

/// IMP-161c: Test cold/warm start detection
#[test]
fn test_imp_161c_cold_warm_detection() {
    // Simulate cold start: first request is slow
    let latencies = vec![
        500.0, // Cold start (model loading, JIT compilation)
        100.0, 105.0, 98.0, 102.0, 99.0, 101.0, 100.0, 103.0, 97.0, // Warm
    ];

    let analysis = ColdWarmComparison::analyze(&latencies);

    // IMP-161c: Cold start should be ~500ms
    assert!(
        (analysis.cold_latency_ms - 500.0).abs() < 1.0,
        "IMP-161c: Cold latency should be 500ms, got {:.2}",
        analysis.cold_latency_ms
    );

    // IMP-161c: Warm latency should be ~100ms
    assert!(
        (analysis.warm_latency_ms - 100.5).abs() < 5.0,
        "IMP-161c: Warm latency should be ~100ms, got {:.2}",
        analysis.warm_latency_ms
    );

    // IMP-161c: Cold penalty should be significant (~5x)
    assert!(
        analysis.significant_cold_penalty,
        "IMP-161c: Cold start penalty should be significant"
    );

    assert!(
        analysis.cold_penalty_ratio > 4.0 && analysis.cold_penalty_ratio < 6.0,
        "IMP-161c: Cold penalty ratio should be ~5x, got {:.2}x",
        analysis.cold_penalty_ratio
    );

    println!("\nIMP-161c: Cold/Warm Start Detection:");
    println!("  Cold start latency: {:.2} ms", analysis.cold_latency_ms);
    println!("  Warm average latency: {:.2} ms", analysis.warm_latency_ms);
    println!("  Cold penalty ratio: {:.2}x", analysis.cold_penalty_ratio);
    println!(
        "  Significant penalty: {}",
        analysis.significant_cold_penalty
    );
}

/// IMP-161d: Real-world warmup detection with llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_161d_realworld_warmup_detection() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Say hello:".to_string(),
        max_tokens: 10,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect 15 samples to detect warmup
    let mut latencies_ms = Vec::new();
    for _ in 0..15 {
        let start = std::time::Instant::now();
        if client
            .llamacpp_completion("http://127.0.0.1:8082", &request)
            .is_ok()
        {
            latencies_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if latencies_ms.len() < 10 {
        println!("IMP-161d: Not enough samples collected");
        return;
    }

    // Convert to throughput (approximated from latency)
    let throughputs: Vec<f64> = latencies_ms
        .iter()
        .map(|lat| if *lat > 0.0 { 10000.0 / lat } else { 0.0 }) // ~10 tokens
        .collect();

    let analysis = JitAwareBenchmark::analyze(&throughputs);
    let cold_warm = ColdWarmComparison::analyze(&latencies_ms);

    println!("\nIMP-161d: Real-World Warmup Detection (llama.cpp):");
    println!("  Samples collected: {}", latencies_ms.len());
    println!("  Raw mean: {:.2} tok/s", analysis.raw_stats.mean_tps);
    println!(
        "  Filtered mean: {:.2} tok/s",
        analysis.filtered_stats.mean_tps
    );
    println!(
        "  Warmup iterations: {}",
        analysis.warmup_result.warmup_iterations
    );
    println!(
        "  Filtering improvement: {:.2}%",
        analysis.improvement_percent
    );
    println!("  Cold start latency: {:.2} ms", cold_warm.cold_latency_ms);
    println!("  Warm latency: {:.2} ms", cold_warm.warm_latency_ms);
    println!("  Cold penalty: {:.2}x", cold_warm.cold_penalty_ratio);
}

// =========================================================================
// IMP-162: MAD Outlier Detection Verification (QA-034, EXTREME TDD)
