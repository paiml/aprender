
/// IMP-164c: Test CI gate evaluation
#[test]
fn test_imp_164c_ci_gate_evaluation() {
    let baseline = ThroughputWithVariance::from_samples(&[
        80.0, 82.0, 78.0, 81.0, 79.0, 80.0, 83.0, 77.0, 80.0, 81.0,
    ]);

    // Good: slight improvement
    let good_current = ThroughputWithVariance::from_samples(&[
        85.0, 87.0, 83.0, 86.0, 84.0, 85.0, 88.0, 82.0, 85.0, 86.0,
    ]);
    let good_result =
        CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &good_current);
    assert!(
        good_result.passed && !good_result.warning,
        "IMP-164c: Improvement should pass without warning"
    );

    // Warning: small regression
    let warn_current = ThroughputWithVariance::from_samples(&[
        78.0, 80.0, 76.0, 79.0, 77.0, 78.0, 81.0, 75.0, 78.0, 79.0,
    ]);
    let warn_result =
        CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &warn_current);
    assert!(
        warn_result.passed && warn_result.warning,
        "IMP-164c: 2-5% regression should warn"
    );

    // Blocked: large regression
    let bad_current = ThroughputWithVariance::from_samples(&[
        70.0, 72.0, 68.0, 71.0, 69.0, 70.0, 73.0, 67.0, 70.0, 71.0,
    ]);
    let bad_result =
        CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &bad_current);
    assert!(!bad_result.passed, "IMP-164c: >5% regression should block");

    println!("\nIMP-164c: CI Gate Evaluation:");
    println!(
        "  Baseline: {:.1} tok/s (CV={:.4})",
        baseline.mean_tps, baseline.cv
    );
    println!(
        "  Good: {} (quality: {})",
        good_result.message, good_result.measurement_quality
    );
    println!(
        "  Warning: {} (quality: {})",
        warn_result.message, warn_result.measurement_quality
    );
    println!(
        "  Blocked: {} (quality: {})",
        bad_result.message, bad_result.measurement_quality
    );
}

/// IMP-164d: Real-world regression check against llama.cpp baseline
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_164d_realworld_regression_check() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Count to 5:".to_string(),
        max_tokens: 15,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect throughput samples
    let mut throughputs = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens = result.text.split_whitespace().count().max(1);
            throughputs.push(tokens as f64 / elapsed);
        }
    }

    if throughputs.len() < 5 {
        println!("IMP-164d: Not enough samples");
        return;
    }

    let current = ThroughputWithVariance::from_samples(&throughputs);

    // Use spec baseline: 256 tok/s for llama.cpp
    let baseline = ThroughputWithVariance::from_samples(&[256.0; 10]);

    let tracker = ThroughputRegressionTracker::check(baseline.mean_tps, current.mean_tps, 5.0);
    let ci_result = CIGateResult::evaluate(CIGateConfig::default_config(), &baseline, &current);

    println!("\nIMP-164d: Real-World Regression Check (llama.cpp):");
    println!("  Spec baseline: {:.1} tok/s", baseline.mean_tps);
    println!(
        "  Current measured: {:.1} tok/s (CV={:.4})",
        current.mean_tps, current.cv
    );
    println!("  {}", tracker.ci_message());
    println!(
        "  CI Gate: {} (quality: {})",
        ci_result.message, ci_result.measurement_quality
    );
}

// =========================================================================
// IMP-165: Real-World Memory Efficiency Comparison (QA-013, EXTREME TDD)
// =========================================================================
// Per spec QA-013: Memory usage < 1.5x model size
// Compares memory efficiency between inference engines.
// Run with: cargo test test_imp_165 --lib --features bench-http

/// IMP-165a: Memory efficiency measurement
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyMeasurement {
    /// Server name
    pub server_name: String,
    /// Model file size in MB
    pub model_size_mb: f64,
    /// Peak memory usage during inference in MB
    pub peak_memory_mb: f64,
    /// Memory overhead ratio (peak / model)
    pub overhead_ratio: f64,
    /// Whether it meets QA-013 (< 1.5x)
    pub meets_qa013: bool,
    /// Memory efficiency score (0-100, higher is better)
    pub efficiency_score: f64,
}

impl MemoryEfficiencyMeasurement {
    pub fn new(server_name: &str, model_size_mb: f64, peak_memory_mb: f64) -> Self {
        let overhead_ratio = if model_size_mb > 0.0 {
            peak_memory_mb / model_size_mb
        } else {
            1.0
        };

        let meets_qa013 = overhead_ratio < 1.5;

        // Efficiency score: 100 at 1.0x, 0 at 2.0x
        let efficiency_score = ((2.0 - overhead_ratio) / 1.0 * 100.0).clamp(0.0, 100.0);

        Self {
            server_name: server_name.to_string(),
            model_size_mb,
            peak_memory_mb,
            overhead_ratio,
            meets_qa013,
            efficiency_score,
        }
    }

    /// Calculate wasted memory in MB
    pub fn wasted_memory_mb(&self) -> f64 {
        (self.peak_memory_mb - self.model_size_mb).max(0.0)
    }
}

/// IMP-165a: Test memory efficiency measurement
#[test]
fn test_imp_165a_memory_efficiency_measurement() {
    // Efficient server: peak close to model size
    let efficient = MemoryEfficiencyMeasurement::new("Efficient", 1000.0, 1100.0);
    assert!(
        efficient.meets_qa013,
        "IMP-165a: 1.1x overhead should meet QA-013"
    );
    assert!(
        efficient.efficiency_score > 80.0,
        "IMP-165a: Efficient server should have high score, got {:.1}",
        efficient.efficiency_score
    );

    // Borderline server: just under 1.5x
    let borderline = MemoryEfficiencyMeasurement::new("Borderline", 1000.0, 1400.0);
    assert!(
        borderline.meets_qa013,
        "IMP-165a: 1.4x overhead should meet QA-013"
    );

    // Inefficient server: exceeds 1.5x
    let inefficient = MemoryEfficiencyMeasurement::new("Inefficient", 1000.0, 1800.0);
    assert!(
        !inefficient.meets_qa013,
        "IMP-165a: 1.8x overhead should fail QA-013"
    );
    assert!(
        inefficient.efficiency_score < 30.0,
        "IMP-165a: Inefficient server should have low score, got {:.1}",
        inefficient.efficiency_score
    );

    println!("\nIMP-165a: Memory Efficiency Measurement:");
    println!(
        "  Efficient: {:.1}x overhead, score={:.1}, QA-013={}",
        efficient.overhead_ratio, efficient.efficiency_score, efficient.meets_qa013
    );
    println!(
        "  Borderline: {:.1}x overhead, score={:.1}, QA-013={}",
        borderline.overhead_ratio, borderline.efficiency_score, borderline.meets_qa013
    );
    println!(
        "  Inefficient: {:.1}x overhead, score={:.1}, QA-013={}",
        inefficient.overhead_ratio, inefficient.efficiency_score, inefficient.meets_qa013
    );
}

/// IMP-165b: Multi-server memory comparison
#[derive(Debug, Clone)]
pub struct MemoryEfficiencyComparison {
    /// All server measurements
    pub measurements: Vec<MemoryEfficiencyMeasurement>,
    /// Server with best efficiency
    pub most_efficient: String,
    /// Server with worst efficiency
    pub least_efficient: String,
    /// Average overhead ratio across all servers
    pub avg_overhead_ratio: f64,
}

impl MemoryEfficiencyComparison {
    pub fn compare(measurements: Vec<MemoryEfficiencyMeasurement>) -> Self {
        if measurements.is_empty() {
            return Self {
                measurements: Vec::new(),
                most_efficient: "none".to_string(),
                least_efficient: "none".to_string(),
                avg_overhead_ratio: 1.0,
            };
        }

        let avg =
            measurements.iter().map(|m| m.overhead_ratio).sum::<f64>() / measurements.len() as f64;

        let most = measurements
            .iter()
            .min_by(|a, b| {
                a.overhead_ratio
                    .partial_cmp(&b.overhead_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(|| "none".to_string(), |m| m.server_name.clone());

        let least = measurements
            .iter()
            .max_by(|a, b| {
                a.overhead_ratio
                    .partial_cmp(&b.overhead_ratio)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or_else(|| "none".to_string(), |m| m.server_name.clone());

        Self {
            measurements,
            most_efficient: most,
            least_efficient: least,
            avg_overhead_ratio: avg,
        }
    }
}

/// IMP-165b: Test multi-server memory comparison
#[test]
fn test_imp_165b_memory_comparison() {
    let model_size = 4000.0; // 4GB model

    let measurements = vec![
        MemoryEfficiencyMeasurement::new("llama.cpp", model_size, 4200.0), // 1.05x
        MemoryEfficiencyMeasurement::new("Ollama", model_size, 4800.0),    // 1.2x
        MemoryEfficiencyMeasurement::new("Realizar", model_size, 5200.0),  // 1.3x
    ];

    let comparison = MemoryEfficiencyComparison::compare(measurements);

    // IMP-165b: llama.cpp should be most efficient
    assert_eq!(
        comparison.most_efficient, "llama.cpp",
        "IMP-165b: llama.cpp should be most efficient"
    );

    // IMP-165b: Realizar should be least efficient
    assert_eq!(
        comparison.least_efficient, "Realizar",
        "IMP-165b: Realizar should be least efficient"
    );

    // IMP-165b: All should meet QA-013
    let all_meet = comparison.measurements.iter().all(|m| m.meets_qa013);
    assert!(
        all_meet,
        "IMP-165b: All servers should meet QA-013 (< 1.5x)"
    );

    println!("\nIMP-165b: Memory Efficiency Comparison:");
    println!("  Model size: {:.0} MB", model_size);
    for m in &comparison.measurements {
        println!(
            "  {}: {:.0} MB peak ({:.2}x), score={:.1}, QA-013={}",
            m.server_name, m.peak_memory_mb, m.overhead_ratio, m.efficiency_score, m.meets_qa013
        );
    }
    println!("  Most efficient: {}", comparison.most_efficient);
    println!("  Least efficient: {}", comparison.least_efficient);
    println!("  Average overhead: {:.2}x", comparison.avg_overhead_ratio);
}

/// IMP-165c: Memory per token efficiency
#[derive(Debug, Clone)]
pub struct MemoryPerTokenEfficiency {
    /// Server name
    pub server_name: String,
    /// Memory per token in KB
    pub memory_per_token_kb: f64,
    /// Context length tested
    pub context_length: usize,
    /// Whether scaling is linear (expected) or super-linear (bad)
    pub linear_scaling: bool,
}

impl MemoryPerTokenEfficiency {
    pub fn analyze(server_name: &str, context_memory_pairs: &[(usize, f64)]) -> Self {
        if context_memory_pairs.len() < 2 {
            return Self {
                server_name: server_name.to_string(),
                memory_per_token_kb: 0.0,
                context_length: 0,
                linear_scaling: true,
            };
        }

        // Calculate memory per token for last measurement
        let (last_ctx, last_mem) = context_memory_pairs.last().expect("test");
        let (first_ctx, first_mem) = context_memory_pairs.first().expect("test");

        let delta_mem = last_mem - first_mem;
        let delta_ctx = (*last_ctx - *first_ctx) as f64;
        let mem_per_token = if delta_ctx > 0.0 {
            (delta_mem * 1024.0) / delta_ctx // Convert MB to KB
        } else {
            0.0
        };

        // Check for linear scaling: memory growth should be proportional to context
        // If growth rate increases significantly, scaling is super-linear (bad)
        let linear = if context_memory_pairs.len() >= 3 {
            let mid = context_memory_pairs.len() / 2;
            let (mid_ctx, mid_mem) = &context_memory_pairs[mid];

            let rate1 = (mid_mem - first_mem) / (*mid_ctx - *first_ctx) as f64;
            let rate2 = (last_mem - mid_mem) / (*last_ctx - *mid_ctx) as f64;

            // Linear if rates are within 20% of each other
            (rate2 / rate1 - 1.0).abs() < 0.20
        } else {
            true
        };

        Self {
            server_name: server_name.to_string(),
            memory_per_token_kb: mem_per_token,
            context_length: *last_ctx,
            linear_scaling: linear,
        }
    }
}

/// IMP-165c: Test memory per token efficiency
#[test]
fn test_imp_165c_memory_per_token() {
    // Linear scaling: good behavior
    let linear_data = vec![
        (512, 5000.0),  // 512 tokens, 5GB
        (1024, 5500.0), // 1024 tokens, 5.5GB
        (2048, 6500.0), // 2048 tokens, 6.5GB
    ];
    let linear = MemoryPerTokenEfficiency::analyze("LinearServer", &linear_data);

    assert!(
        linear.linear_scaling,
        "IMP-165c: Linear memory growth should be detected"
    );

    // Super-linear scaling: bad behavior (memory explodes)
    let superlinear_data = vec![
        (512, 5000.0),   // 512 tokens, 5GB
        (1024, 6000.0),  // 1024 tokens, 6GB (+2MB/tok)
        (2048, 10000.0), // 2048 tokens, 10GB (+4MB/tok - rate doubled!)
    ];
    let superlinear = MemoryPerTokenEfficiency::analyze("SuperLinearServer", &superlinear_data);

    assert!(
        !superlinear.linear_scaling,
        "IMP-165c: Super-linear growth should be detected"
    );

    println!("\nIMP-165c: Memory Per Token Efficiency:");
    println!(
        "  Linear server: {:.2} KB/token, linear={}",
        linear.memory_per_token_kb, linear.linear_scaling
    );
    println!(
        "  Super-linear server: {:.2} KB/token, linear={}",
        superlinear.memory_per_token_kb, superlinear.linear_scaling
    );
}

/// IMP-165d: Real-world memory efficiency (placeholder)
#[test]
#[ignore = "Requires running llama.cpp server with memory monitoring"]
fn test_imp_165d_realworld_memory_efficiency() {
    // This test would require:
    // 1. Running llama.cpp server
    // 2. Monitoring memory usage via /proc or similar
    // 3. Known model size for comparison

    // test values based on typical observations
    let model_size_mb = 4000.0; // 4GB Q4_K model

    let measurements = vec![
        MemoryEfficiencyMeasurement::new("llama.cpp", model_size_mb, 4200.0),
        MemoryEfficiencyMeasurement::new("Ollama", model_size_mb, 4600.0),
    ];

    let comparison = MemoryEfficiencyComparison::compare(measurements);

    println!("\nIMP-165d: Real-World Memory Efficiency:");
    println!("  Model size: {:.0} MB", model_size_mb);
    for m in &comparison.measurements {
        println!(
            "  {}: {:.2}x overhead, wasted={:.0} MB, QA-013={}",
            m.server_name,
            m.overhead_ratio,
            m.wasted_memory_mb(),
            m.meets_qa013
        );
    }
}

// =========================================================================
// IMP-166: Real-World Cold Start Verification (QA-016, EXTREME TDD)
