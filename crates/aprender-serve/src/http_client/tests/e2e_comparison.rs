
impl E2EComparisonResult {
    pub fn new(realizar_tps: f64, ollama_tps: f64, llamacpp_tps: f64) -> Self {
        let gap_vs_ollama = if ollama_tps > 0.0 {
            ((realizar_tps - ollama_tps) / ollama_tps) * 100.0
        } else {
            0.0
        };
        let gap_vs_llamacpp = if llamacpp_tps > 0.0 {
            ((realizar_tps - llamacpp_tps) / llamacpp_tps) * 100.0
        } else {
            0.0
        };
        // Parity = within 10% of llama.cpp (per spec)
        let parity_achieved = gap_vs_llamacpp >= -10.0;

        Self {
            realizar_tps,
            ollama_tps,
            llamacpp_tps,
            gap_vs_ollama_percent: gap_vs_ollama,
            gap_vs_llamacpp_percent: gap_vs_llamacpp,
            parity_achieved,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// IMP-152a: Test E2E comparison result struct
#[test]
fn test_imp_152a_e2e_comparison_struct() {
    // Scenario: Realizar at 200 tok/s, Ollama at 143, llama.cpp at 256
    let result = E2EComparisonResult::new(200.0, 143.0, 256.0);

    // Verify gap calculations
    let expected_ollama_gap: f64 = ((200.0 - 143.0) / 143.0) * 100.0; // +39.9%
    let expected_llamacpp_gap: f64 = ((200.0 - 256.0) / 256.0) * 100.0; // -21.9%

    assert!(
        (result.gap_vs_ollama_percent - expected_ollama_gap).abs() < 0.1,
        "IMP-152a: Ollama gap should be ~39.9%"
    );
    assert!(
        (result.gap_vs_llamacpp_percent - expected_llamacpp_gap).abs() < 0.1,
        "IMP-152a: llama.cpp gap should be ~-21.9%"
    );
    assert!(
        !result.parity_achieved,
        "IMP-152a: -21.9% gap should not be parity"
    );

    println!("\nIMP-152a: E2E Comparison Result:");
    println!("  Realizar: {:.1} tok/s", result.realizar_tps);
    println!("  Ollama:   {:.1} tok/s", result.ollama_tps);
    println!("  llama.cpp: {:.1} tok/s", result.llamacpp_tps);
    println!("  Gap vs Ollama: {:+.1}%", result.gap_vs_ollama_percent);
    println!(
        "  Gap vs llama.cpp: {:+.1}%",
        result.gap_vs_llamacpp_percent
    );
    println!("  Parity achieved: {}", result.parity_achieved);
}

/// IMP-152b: Test parity threshold detection
#[test]
fn test_imp_152b_parity_detection() {
    // Scenario 1: Just within parity (232 tok/s vs 256 = -9.4% gap)
    // 232/256 = 0.906, so gap = -9.4% which is > -10%
    let at_parity = E2EComparisonResult::new(232.0, 143.0, 256.0);
    assert!(
        at_parity.parity_achieved,
        "IMP-152b: 232 vs 256 should be parity (-9.4%)"
    );

    // Scenario 2: Beyond parity (260 tok/s = +1.5% faster)
    let beyond_parity = E2EComparisonResult::new(260.0, 143.0, 256.0);
    assert!(
        beyond_parity.parity_achieved,
        "IMP-152b: 260 vs 256 should definitely be parity"
    );
    assert!(
        beyond_parity.gap_vs_llamacpp_percent > 0.0,
        "IMP-152b: 260 vs 256 should show positive gap"
    );

    // Scenario 3: Below parity (200 tok/s = -21.9% gap)
    let below_parity = E2EComparisonResult::new(200.0, 143.0, 256.0);
    assert!(
        !below_parity.parity_achieved,
        "IMP-152b: 200 vs 256 should NOT be parity"
    );

    // Scenario 4: Exactly at threshold (231 tok/s = -9.8% gap)
    let exact_threshold = E2EComparisonResult::new(231.0, 143.0, 256.0);
    assert!(
        exact_threshold.parity_achieved,
        "IMP-152b: 231 vs 256 should be parity (-9.8%)"
    );

    println!("\nIMP-152b: Parity Detection:");
    println!(
        "  232 vs 256 = {:.1}% gap, parity={}",
        at_parity.gap_vs_llamacpp_percent, at_parity.parity_achieved
    );
    println!(
        "  260 vs 256 = {:+.1}% gap, parity={}",
        beyond_parity.gap_vs_llamacpp_percent, beyond_parity.parity_achieved
    );
    println!(
        "  200 vs 256 = {:.1}% gap, parity={}",
        below_parity.gap_vs_llamacpp_percent, below_parity.parity_achieved
    );
    println!(
        "  231 vs 256 = {:.1}% gap, parity={}",
        exact_threshold.gap_vs_llamacpp_percent, exact_threshold.parity_achieved
    );
}

/// IMP-152c: Real-world E2E comparison (requires both servers)
#[test]
#[ignore = "Requires running Ollama (11434) and llama.cpp (8082) servers"]
fn test_imp_152c_real_e2e_comparison() {
    // This test requires:
    // 1. ollama serve
    // 2. llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let config = HttpBenchmarkConfig {
        cv_criterion: CvStoppingCriterion::new(5, 20, 0.10),
        warmup_iterations: 2,
        prompt: "What is the capital of France? Answer in one word:".to_string(),
        max_tokens: 20,
        temperature: 0.0,
        run_preflight: true,
        filter_outliers: true,
        outlier_k_factor: 3.0,
    };

    let mut runner = HttpBenchmarkRunner::new(config);

    // Benchmark both external servers
    let llamacpp_result = runner
        .benchmark_llamacpp("http://127.0.0.1:8082")
        .expect("IMP-152c: llama.cpp benchmark failed");
    let ollama_result = runner
        .benchmark_ollama("http://127.0.0.1:11434", "phi2:2.7b")
        .expect("IMP-152c: Ollama benchmark failed");

    // test Realizar result based on IMP-900 benchmark projections
    // IMP-900 shows 61 tok/s projected, targeting 80+ with further optimizations
    let realizar_tps: f64 = 61.0; // IMP-900 projected throughput

    let comparison = E2EComparisonResult::new(
        realizar_tps,
        ollama_result.throughput_tps,
        llamacpp_result.throughput_tps,
    );

    println!("\nIMP-152c: Real-World E2E Comparison:");
    println!("  Realizar:  {:.1} tok/s (test)", comparison.realizar_tps);
    println!("  Ollama:    {:.1} tok/s (measured)", comparison.ollama_tps);
    println!(
        "  llama.cpp: {:.1} tok/s (measured)",
        comparison.llamacpp_tps
    );
    println!(
        "  Gap vs Ollama:    {:+.1}%",
        comparison.gap_vs_ollama_percent
    );
    println!(
        "  Gap vs llama.cpp: {:+.1}%",
        comparison.gap_vs_llamacpp_percent
    );
    println!("  Parity achieved:  {}", comparison.parity_achieved);
    println!("  Timestamp: {}", comparison.timestamp);
}

/// IMP-152d: Progress delta tracking across milestones
#[derive(Debug, Clone)]
pub struct ProgressDelta {
    /// Previous comparison result
    pub previous_tps: f64,
    /// Current comparison result
    pub current_tps: f64,
    /// Absolute improvement (tok/s)
    pub delta_tps: f64,
    /// Relative improvement percentage
    pub delta_percent: f64,
    /// Target for next milestone
    pub next_milestone_tps: f64,
    /// Percentage progress toward next milestone
    pub progress_to_next: f64,
}

impl ProgressDelta {
    pub fn new(previous_tps: f64, current_tps: f64, next_milestone_tps: f64) -> Self {
        let delta_tps = current_tps - previous_tps;
        let delta_percent = if previous_tps > 0.0 {
            (delta_tps / previous_tps) * 100.0
        } else {
            0.0
        };
        let progress_to_next = if next_milestone_tps > current_tps {
            ((current_tps - previous_tps) / (next_milestone_tps - previous_tps)) * 100.0
        } else {
            100.0 // Already at or beyond milestone
        };
        Self {
            previous_tps,
            current_tps,
            delta_tps,
            delta_percent,
            next_milestone_tps,
            progress_to_next,
        }
    }
}

/// IMP-152d: Test progress delta tracking
#[test]
fn test_imp_152d_progress_delta_tracking() {
    // Scenario: Improved from 80 tok/s to 100 tok/s, targeting P1 = 120 tok/s
    let delta = ProgressDelta::new(80.0, 100.0, 120.0);

    assert!(
        (delta.delta_tps - 20.0).abs() < 0.01,
        "IMP-152d: Delta should be 20 tok/s"
    );
    assert!(
        (delta.delta_percent - 25.0).abs() < 0.1,
        "IMP-152d: Delta should be 25%"
    );
    assert!(
        (delta.progress_to_next - 50.0).abs() < 0.1,
        "IMP-152d: Progress should be 50% (20 of 40 tok/s needed)"
    );

    // Scenario: At milestone (120 tok/s achieved, targeting P2 = 200)
    let delta2 = ProgressDelta::new(100.0, 120.0, 200.0);
    assert!(
        (delta2.delta_percent - 20.0).abs() < 0.1,
        "IMP-152d: Delta should be 20%"
    );

    // Scenario: Beyond milestone
    let delta3 = ProgressDelta::new(180.0, 210.0, 200.0);
    assert!(
        (delta3.progress_to_next - 100.0).abs() < 0.01,
        "IMP-152d: Should be 100% when beyond milestone"
    );

    println!("\nIMP-152d: Progress Delta Tracking:");
    println!(
        "  80 → 100 (target 120): {:+.0} tok/s ({:+.1}%), {:.0}% to milestone",
        delta.delta_tps, delta.delta_percent, delta.progress_to_next
    );
    println!(
        "  100 → 120 (target 200): {:+.0} tok/s ({:+.1}%), {:.0}% to milestone",
        delta2.delta_tps, delta2.delta_percent, delta2.progress_to_next
    );
    println!(
        "  180 → 210 (target 200): {:+.0} tok/s ({:+.1}%), {:.0}% to milestone",
        delta3.delta_tps, delta3.delta_percent, delta3.progress_to_next
    );
}

// =========================================================================
// IMP-153: Performance Progress Tracking Metrics (EXTREME TDD)
// Per spec §9.1: Historical tracking and trend analysis for performance
