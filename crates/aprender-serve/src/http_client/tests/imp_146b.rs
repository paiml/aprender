
/// IMP-146b: Verify gap analysis calculates ratios correctly
#[test]
fn test_imp_146b_gap_analysis() {
    let realizar = ThroughputBaseline {
        server: "Realizar".to_string(),
        throughput_tps: 80.0, // Per spec: current ~80 tok/s
        p50_latency_ms: 520.0,
        p99_latency_ms: 800.0,
        cv: 0.08,
        samples: 10,
    };

    let llamacpp = ThroughputBaseline {
        server: "llama.cpp".to_string(),
        throughput_tps: 256.0, // Per spec: ~256 tok/s GPU
        p50_latency_ms: 162.0,
        p99_latency_ms: 290.0,
        cv: 0.045,
        samples: 10,
    };

    let gap = GapAnalysis {
        gap_ratio: llamacpp.throughput_tps / realizar.throughput_tps,
        throughput_gap_tps: llamacpp.throughput_tps - realizar.throughput_tps,
        parity_target_tps: llamacpp.throughput_tps * 0.8, // 80% is parity
        realizar,
        reference: llamacpp,
    };

    // IMP-146b: Gap should be ~3.2x per Five Whys analysis
    assert!(
        gap.gap_ratio > 2.5 && gap.gap_ratio < 4.0,
        "IMP-146b: Gap to llama.cpp should be ~3.2x, got {:.1}x",
        gap.gap_ratio
    );

    // IMP-146b: Parity target should be 80% of reference
    assert!(
        (gap.parity_target_tps - 204.8).abs() < 1.0,
        "IMP-146b: Parity target should be ~205 tok/s, got {:.1}",
        gap.parity_target_tps
    );

    println!("\nIMP-146b: Gap Analysis:");
    println!("  Realizar: {:.1} tok/s", gap.realizar.throughput_tps);
    println!("  llama.cpp: {:.1} tok/s", gap.reference.throughput_tps);
    println!("  Gap: {:.1}x", gap.gap_ratio);
    println!("  Target for parity: {:.1} tok/s", gap.parity_target_tps);
}

/// IMP-146c: Real-world baseline measurement against llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_146c_llamacpp_baseline_measurement() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10), // Scientific rigor
        warmup_iterations: 2,
        prompt: "Explain what machine learning is in one paragraph:".to_string(),
        max_tokens: 50,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-146c: llama.cpp baseline measurement should succeed");

    // IMP-146c: Build baseline from result
    let baseline = ThroughputBaseline {
        server: "llama.cpp".to_string(),
        throughput_tps: result.throughput_tps,
        p50_latency_ms: result.p50_latency_ms,
        p99_latency_ms: result.p99_latency_ms,
        cv: result.cv_at_stop,
        samples: result.sample_count,
    };

    // IMP-146c: Baseline should have reasonable values
    assert!(
        baseline.throughput_tps > 50.0,
        "IMP-146c: llama.cpp should achieve > 50 tok/s, got {:.1}",
        baseline.throughput_tps
    );
    assert!(
        baseline.cv < 0.20,
        "IMP-146c: CV should be < 20% for reliable measurement, got {:.2}",
        baseline.cv
    );

    println!("\nIMP-146c: llama.cpp Baseline Measurement:");
    println!("  Throughput: {:.1} tok/s", baseline.throughput_tps);
    println!("  P50 Latency: {:.1} ms", baseline.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", baseline.p99_latency_ms);
    println!(
        "  CV: {:.4} ({})",
        baseline.cv,
        if baseline.cv < 0.05 {
            "excellent"
        } else if baseline.cv < 0.10 {
            "good"
        } else {
            "acceptable"
        }
    );
    println!("  Samples: {}", baseline.samples);
}

/// IMP-146d: Real-world baseline measurement against Ollama
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_146d_ollama_baseline_measurement() {
    // This test requires: ollama serve
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "Explain what machine learning is in one paragraph:".to_string(),
        max_tokens: 50,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-146d: Ollama baseline measurement should succeed");

    // IMP-146d: Build baseline from result
    let baseline = ThroughputBaseline {
        server: "Ollama".to_string(),
        throughput_tps: result.throughput_tps,
        p50_latency_ms: result.p50_latency_ms,
        p99_latency_ms: result.p99_latency_ms,
        cv: result.cv_at_stop,
        samples: result.sample_count,
    };

    // IMP-146d: Baseline should have reasonable values
    assert!(
        baseline.throughput_tps > 30.0,
        "IMP-146d: Ollama should achieve > 30 tok/s, got {:.1}",
        baseline.throughput_tps
    );
    assert!(
        baseline.cv < 0.20,
        "IMP-146d: CV should be < 20% for reliable measurement, got {:.2}",
        baseline.cv
    );

    println!("\nIMP-146d: Ollama Baseline Measurement:");
    println!("  Throughput: {:.1} tok/s", baseline.throughput_tps);
    println!("  P50 Latency: {:.1} ms", baseline.p50_latency_ms);
    println!("  P99 Latency: {:.1} ms", baseline.p99_latency_ms);
    println!(
        "  CV: {:.4} ({})",
        baseline.cv,
        if baseline.cv < 0.05 {
            "excellent"
        } else if baseline.cv < 0.10 {
            "good"
        } else {
            "acceptable"
        }
    );
    println!("  Samples: {}", baseline.samples);
}

// =========================================================================
// IMP-151: Real-World Throughput Regression Tests (EXTREME TDD)
// =========================================================================
// These tests track performance progress and detect regressions.
// Per Five Whys Analysis, target: 80 tok/s → 120 tok/s (P1) → 200 tok/s (P2)

/// IMP-151a: Performance milestone tracking struct
#[derive(Debug, Clone)]
pub struct PerformanceMilestone {
    /// Milestone name (e.g., "P1", "P2", "Parity")
    pub name: String,
    /// Target throughput in tokens/second
    pub target_tps: f64,
    /// Current achieved throughput
    pub achieved_tps: f64,
    /// Gap to target as percentage
    pub gap_percent: f64,
    /// Whether milestone is achieved
    pub achieved: bool,
}

impl PerformanceMilestone {
    pub fn new(name: &str, target_tps: f64, achieved_tps: f64) -> Self {
        let gap_percent = if target_tps > 0.0 {
            ((target_tps - achieved_tps) / target_tps) * 100.0
        } else {
            0.0
        };
        Self {
            name: name.to_string(),
            target_tps,
            achieved_tps,
            gap_percent,
            achieved: achieved_tps >= target_tps,
        }
    }
}

/// IMP-151a: Verify milestone tracking struct works correctly
#[test]
fn test_imp_151a_milestone_tracking() {
    // Current baseline: 80 tok/s
    let current_tps = 80.0;

    // Define milestones per Five Whys roadmap
    let p1_milestone = PerformanceMilestone::new("P1", 120.0, current_tps);
    let p2_milestone = PerformanceMilestone::new("P2", 200.0, current_tps);
    let parity_milestone = PerformanceMilestone::new("Parity", 205.0, current_tps);

    // IMP-151a: Verify milestone calculations
    assert!(
        !p1_milestone.achieved,
        "IMP-151a: P1 not yet achieved at 80 tok/s"
    );
    assert!(
        (p1_milestone.gap_percent - 33.3).abs() < 1.0,
        "IMP-151a: Gap to P1 should be ~33%, got {:.1}%",
        p1_milestone.gap_percent
    );

    assert!(!p2_milestone.achieved, "IMP-151a: P2 not yet achieved");
    assert!(
        (p2_milestone.gap_percent - 60.0).abs() < 1.0,
        "IMP-151a: Gap to P2 should be ~60%, got {:.1}%",
        p2_milestone.gap_percent
    );

    println!("\nIMP-151a: Performance Milestone Tracking:");
    println!("  Current: {:.1} tok/s", current_tps);
    println!(
        "  P1 (120 tok/s): {:.1}% gap, achieved={}",
        p1_milestone.gap_percent, p1_milestone.achieved
    );
    println!(
        "  P2 (200 tok/s): {:.1}% gap, achieved={}",
        p2_milestone.gap_percent, p2_milestone.achieved
    );
    println!(
        "  Parity (205 tok/s): {:.1}% gap, achieved={}",
        parity_milestone.gap_percent, parity_milestone.achieved
    );
}

/// IMP-151b: Regression detection struct
#[derive(Debug, Clone)]
pub struct RegressionCheck {
    /// Test name
    pub test_name: String,
    /// Baseline throughput (previous best)
    pub baseline_tps: f64,
    /// Current throughput
    pub current_tps: f64,
    /// Regression threshold percentage (e.g., 5% = flag if >5% slower)
    pub threshold_percent: f64,
    /// Whether regression detected
    pub regression_detected: bool,
    /// Improvement percentage (negative = regression)
    pub improvement_percent: f64,
}

impl RegressionCheck {
    pub fn new(
        test_name: &str,
        baseline_tps: f64,
        current_tps: f64,
        threshold_percent: f64,
    ) -> Self {
        let improvement_percent = if baseline_tps > 0.0 {
            ((current_tps - baseline_tps) / baseline_tps) * 100.0
        } else {
            0.0
        };
        let regression_detected = improvement_percent < -threshold_percent;
        Self {
            test_name: test_name.to_string(),
            baseline_tps,
            current_tps,
            threshold_percent,
            regression_detected,
            improvement_percent,
        }
    }
}

/// IMP-151b: Verify regression detection works correctly
#[test]
fn test_imp_151b_regression_detection() {
    // Scenario 1: No regression (improvement)
    let check1 = RegressionCheck::new("dequant_q4k", 80.0, 85.0, 5.0);
    assert!(
        !check1.regression_detected,
        "IMP-151b: 85 vs 80 should not be regression"
    );
    assert!(
        (check1.improvement_percent - 6.25).abs() < 0.1,
        "IMP-151b: Should show ~6.25% improvement"
    );

    // Scenario 2: Minor regression within threshold
    let check2 = RegressionCheck::new("fused_matvec", 100.0, 97.0, 5.0);
    assert!(
        !check2.regression_detected,
        "IMP-151b: 3% drop within 5% threshold"
    );

    // Scenario 3: Significant regression exceeds threshold
    let check3 = RegressionCheck::new("simd_extract", 100.0, 90.0, 5.0);
    assert!(
        check3.regression_detected,
        "IMP-151b: 10% drop should trigger regression"
    );

    println!("\nIMP-151b: Regression Detection:");
    println!(
        "  Test 1 (85 vs 80): {:.1}% change, regression={}",
        check1.improvement_percent, check1.regression_detected
    );
    println!(
        "  Test 2 (97 vs 100): {:.1}% change, regression={}",
        check2.improvement_percent, check2.regression_detected
    );
    println!(
        "  Test 3 (90 vs 100): {:.1}% change, regression={}",
        check3.improvement_percent, check3.regression_detected
    );
}

/// IMP-151c: Real-world regression test against llama.cpp baseline
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_151c_llamacpp_regression_check() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "What is 2+2? Answer briefly:".to_string(),
        max_tokens: 20,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-151c: llama.cpp benchmark should succeed");

    // llama.cpp baseline: ~256 tok/s (per spec)
    let expected_baseline = 256.0;
    let tolerance_percent = 30.0; // Allow 30% variance for different hardware

    let check = RegressionCheck::new(
        "llamacpp_throughput",
        expected_baseline,
        result.throughput_tps,
        tolerance_percent,
    );

    println!("\nIMP-151c: llama.cpp Regression Check:");
    println!("  Expected baseline: {:.1} tok/s", expected_baseline);
    println!("  Measured: {:.1} tok/s", result.throughput_tps);
    println!("  Difference: {:.1}%", check.improvement_percent);
    println!("  Regression: {}", check.regression_detected);

    // Note: Not asserting regression here since hardware varies
    // This is for tracking, not blocking
}

/// IMP-151d: Real-world regression test against Ollama baseline
#[test]
#[ignore = "Requires running Ollama server on port 11434"]
fn test_imp_151d_ollama_regression_check() {
    // This test requires: ollama serve
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "What is 2+2? Answer briefly:".to_string(),
        max_tokens: 20,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);
    let result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-151d: Ollama benchmark should succeed");

    // Ollama baseline: ~143 tok/s (per spec)
    let expected_baseline = 143.0;
    let tolerance_percent = 30.0;

    let check = RegressionCheck::new(
        "ollama_throughput",
        expected_baseline,
        result.throughput_tps,
        tolerance_percent,
    );

    println!("\nIMP-151d: Ollama Regression Check:");
    println!("  Expected baseline: {:.1} tok/s", expected_baseline);
    println!("  Measured: {:.1} tok/s", result.throughput_tps);
    println!("  Difference: {:.1}%", check.improvement_percent);
    println!("  Regression: {}", check.regression_detected);
}

// =========================================================================
// IMP-152: End-to-End Performance Comparison Benchmark (EXTREME TDD)
// Per spec §8.3: Side-by-side comparison of Realizar vs Ollama vs llama.cpp
// =========================================================================

/// IMP-152a: End-to-end comparison result tracking
#[derive(Debug, Clone)]
pub struct E2EComparisonResult {
    /// Realizar throughput (tok/s)
    pub realizar_tps: f64,
    /// Ollama throughput (tok/s)
    pub ollama_tps: f64,
    /// llama.cpp throughput (tok/s)
    pub llamacpp_tps: f64,
    /// Gap vs Ollama (positive = Realizar is faster)
    pub gap_vs_ollama_percent: f64,
    /// Gap vs llama.cpp (positive = Realizar is faster)
    pub gap_vs_llamacpp_percent: f64,
    /// Parity achieved (within 10% of llama.cpp)
    pub parity_achieved: bool,
    /// Timestamp of comparison
    pub timestamp: String,
}
