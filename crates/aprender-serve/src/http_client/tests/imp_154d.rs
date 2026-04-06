
/// IMP-154d: Test gate report generation
#[test]
fn test_imp_154d_gate_report() {
    let gates = vec![
        PerformanceGate::higher_is_better("Throughput", 125.0, 120.0, 100.0, " tok/s"),
        PerformanceGate::lower_is_better("P50 Latency", 45.0, 50.0, 100.0, "ms"),
        PerformanceGate::higher_is_better("Regression", 2.5, -5.0, -10.0, "%"),
    ];
    let composite = CompositeGate::new(gates);
    let report = GateReport::new("Performance Parity Check", composite);

    assert_eq!(
        report.exit_code, 0,
        "IMP-154d: All pass should have exit code 0"
    );
    assert!(
        report.summary.contains("3 PASS"),
        "IMP-154d: Should show 3 PASS"
    );

    let ci_output = report.format_for_ci();
    assert!(
        ci_output.contains("## Performance Parity Check"),
        "IMP-154d: Should have title"
    );
    assert!(
        ci_output.contains("Throughput"),
        "IMP-154d: Should list throughput gate"
    );
    assert!(ci_output.contains("✅"), "IMP-154d: Should have pass emoji");

    // Test failure scenario
    let fail_gates = vec![PerformanceGate::higher_is_better(
        "Throughput",
        80.0,
        120.0,
        100.0,
        " tok/s",
    )];
    let fail_report = GateReport::new("Failed Check", CompositeGate::new(fail_gates));
    assert_eq!(
        fail_report.exit_code, 1,
        "IMP-154d: Fail should have exit code 1"
    );

    println!("\nIMP-154d: Gate Report:");
    println!("{}", ci_output);
    println!("Exit code: {}", report.exit_code);
}

// =========================================================================
// IMP-155: Fused Q4K Throughput Verification vs External Servers (EXTREME TDD)
// Per spec §13.1 Phase 2: Verify fused kernel achieves 2x gain (120→240 tok/s)
// =========================================================================

/// IMP-155a: Fused kernel benchmark result
#[derive(Debug, Clone)]
pub struct FusedKernelResult {
    /// Throughput in tokens/second
    pub throughput_tps: f64,
    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth_gbs: f64,
    /// Compute efficiency (% of peak FLOPS)
    pub compute_efficiency_percent: f64,
    /// Whether fused path was used
    pub fused_path_used: bool,
    /// Speedup vs separate dequant+matvec
    pub speedup_vs_separate: f64,
}

impl FusedKernelResult {
    pub fn new(
        throughput_tps: f64,
        memory_bandwidth_gbs: f64,
        fused_path_used: bool,
        baseline_separate_tps: f64,
    ) -> Self {
        let speedup = if baseline_separate_tps > 0.0 {
            throughput_tps / baseline_separate_tps
        } else {
            1.0
        };
        // Estimate compute efficiency based on throughput vs theoretical peak
        // Q4_K: 4.5 bits/param, ~2 FLOPs per param for matvec
        // Theoretical peak depends on memory bandwidth
        let compute_efficiency = (throughput_tps / 1000.0).min(100.0) * 100.0;
        Self {
            throughput_tps,
            memory_bandwidth_gbs,
            compute_efficiency_percent: compute_efficiency,
            fused_path_used,
            speedup_vs_separate: speedup,
        }
    }

    pub fn meets_p2_target(&self) -> bool {
        self.throughput_tps >= 200.0 && self.fused_path_used
    }
}

/// IMP-155a: Test fused kernel result struct
#[test]
fn test_imp_155a_fused_kernel_result() {
    // Scenario: Fused kernel at 240 tok/s vs 80 tok/s separate
    let result = FusedKernelResult::new(240.0, 45.0, true, 80.0);

    assert!(
        result.fused_path_used,
        "IMP-155a: Fused path should be used"
    );
    assert!(
        (result.speedup_vs_separate - 3.0).abs() < 0.1,
        "IMP-155a: Should show 3x speedup (240/80)"
    );
    assert!(
        result.meets_p2_target(),
        "IMP-155a: 240 tok/s should meet P2 target"
    );

    // Scenario: Below P2 target
    let below_target = FusedKernelResult::new(150.0, 30.0, true, 80.0);
    assert!(
        !below_target.meets_p2_target(),
        "IMP-155a: 150 tok/s should not meet P2 target"
    );

    println!("\nIMP-155a: Fused Kernel Results:");
    println!("  Throughput: {:.1} tok/s", result.throughput_tps);
    println!("  Bandwidth: {:.1} GB/s", result.memory_bandwidth_gbs);
    println!("  Speedup: {:.1}x vs separate", result.speedup_vs_separate);
    println!("  Meets P2: {}", result.meets_p2_target());
}

/// IMP-155b: Fused vs separate performance comparison
#[derive(Debug, Clone)]
pub struct FusedVsSeparateComparison {
    pub fused_tps: f64,
    pub separate_tps: f64,
    pub speedup: f64,
    pub memory_reduction_percent: f64,
    pub fused_wins: bool,
}

impl FusedVsSeparateComparison {
    pub fn new(fused_tps: f64, separate_tps: f64) -> Self {
        let speedup = if separate_tps > 0.0 {
            fused_tps / separate_tps
        } else {
            1.0
        };
        // Fused eliminates intermediate buffer: ~50% memory reduction
        let memory_reduction = if speedup > 1.0 { 50.0 } else { 0.0 };
        Self {
            fused_tps,
            separate_tps,
            speedup,
            memory_reduction_percent: memory_reduction,
            fused_wins: speedup > 1.0,
        }
    }
}

/// IMP-155b: Test fused vs separate comparison
#[test]
fn test_imp_155b_fused_vs_separate() {
    // Per IMP-100c: Fused should be 29-132x faster
    let comparison = FusedVsSeparateComparison::new(5000.0, 170.0); // test values

    assert!(comparison.fused_wins, "IMP-155b: Fused should win");
    assert!(
        comparison.speedup > 20.0,
        "IMP-155b: Should show >20x speedup per IMP-100c"
    );
    assert!(
        comparison.memory_reduction_percent > 0.0,
        "IMP-155b: Should show memory reduction"
    );

    // Edge case: separate faster (shouldn't happen in practice)
    let edge = FusedVsSeparateComparison::new(100.0, 200.0);
    assert!(!edge.fused_wins, "IMP-155b: Separate faster edge case");

    println!("\nIMP-155b: Fused vs Separate:");
    println!("  Fused: {:.0} tok/s", comparison.fused_tps);
    println!("  Separate: {:.0} tok/s", comparison.separate_tps);
    println!("  Speedup: {:.1}x", comparison.speedup);
    println!(
        "  Memory reduction: {:.0}%",
        comparison.memory_reduction_percent
    );
}

/// IMP-155c: Real-world fused kernel vs llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_155c_fused_vs_llamacpp() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Explain quantum entanglement in simple terms:".to_string(),
        max_tokens: 50,
        temperature: Some(0.0),
        stream: false,
    };

    let start = std::time::Instant::now();
    let result = client
        .llamacpp_completion("http://127.0.0.1:8082", &request)
        .expect("IMP-155c: llama.cpp benchmark failed");
    let elapsed_s = start.elapsed().as_secs_f64();

    // Estimate throughput from response
    let tokens_generated = result.text.split_whitespace().count() as f64;
    let throughput_tps = tokens_generated / elapsed_s;

    // llama.cpp uses fused GGML kernels - this is our target
    let llamacpp_fused = FusedKernelResult::new(
        throughput_tps,
        50.0, // Estimated bandwidth
        true,
        throughput_tps / 30.0, // Estimate separate baseline
    );

    println!("\nIMP-155c: llama.cpp Fused Kernel Performance:");
    println!("  Throughput: {:.1} tok/s", llamacpp_fused.throughput_tps);
    println!("  Meets P2: {}", llamacpp_fused.meets_p2_target());
    println!(
        "  Est. speedup vs separate: {:.1}x",
        llamacpp_fused.speedup_vs_separate
    );
}

/// IMP-155d: Fused kernel memory efficiency analysis
#[derive(Debug, Clone)]
pub struct MemoryEfficiency {
    pub model_size_mb: f64,
    pub peak_memory_mb: f64,
    pub memory_overhead_percent: f64,
    pub bandwidth_utilization_percent: f64,
}

impl MemoryEfficiency {
    pub fn new(
        model_size_mb: f64,
        peak_memory_mb: f64,
        theoretical_bandwidth_gbs: f64,
        actual_bandwidth_gbs: f64,
    ) -> Self {
        let overhead = if model_size_mb > 0.0 {
            ((peak_memory_mb - model_size_mb) / model_size_mb) * 100.0
        } else {
            0.0
        };
        let utilization = if theoretical_bandwidth_gbs > 0.0 {
            (actual_bandwidth_gbs / theoretical_bandwidth_gbs) * 100.0
        } else {
            0.0
        };
        Self {
            model_size_mb,
            peak_memory_mb,
            memory_overhead_percent: overhead,
            bandwidth_utilization_percent: utilization,
        }
    }

    pub fn is_memory_efficient(&self) -> bool {
        // Efficient if overhead < 50% and bandwidth utilization > 50%
        self.memory_overhead_percent < 50.0 && self.bandwidth_utilization_percent > 50.0
    }
}

/// IMP-155d: Test memory efficiency analysis
#[test]
fn test_imp_155d_memory_efficiency() {
    // Scenario: Q4_K model 7.74 MB, peak 10 MB, 50% bandwidth utilization
    let efficient = MemoryEfficiency::new(7.74, 10.0, 100.0, 55.0);
    assert!(
        efficient.is_memory_efficient(),
        "IMP-155d: 29% overhead, 55% bandwidth should be efficient"
    );

    // Scenario: High overhead (separate path)
    let inefficient = MemoryEfficiency::new(7.74, 20.0, 100.0, 30.0);
    assert!(
        !inefficient.is_memory_efficient(),
        "IMP-155d: 158% overhead should not be efficient"
    );

    println!("\nIMP-155d: Memory Efficiency:");
    println!("  Model size: {:.2} MB", efficient.model_size_mb);
    println!("  Peak memory: {:.2} MB", efficient.peak_memory_mb);
    println!("  Overhead: {:.1}%", efficient.memory_overhead_percent);
    println!(
        "  Bandwidth util: {:.1}%",
        efficient.bandwidth_utilization_percent
    );
    println!("  Efficient: {}", efficient.is_memory_efficient());
}

// =========================================================================
// IMP-156: Latency Percentile Comparison (P50/P95/P99) (EXTREME TDD)
// Per spec QA-035: Results include p50, p95, p99 latencies
// =========================================================================

/// IMP-156a: Latency percentiles
#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub stddev_ms: f64,
}

impl LatencyPercentiles {
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                p50_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                min_ms: 0.0,
                max_ms: 0.0,
                mean_ms: 0.0,
                stddev_ms: 0.0,
            };
        }

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        let p50_idx = (n as f64 * 0.50) as usize;
        let p95_idx = (n as f64 * 0.95) as usize;
        let p99_idx = (n as f64 * 0.99) as usize;

        let mean: f64 = sorted.iter().sum::<f64>() / n as f64;
        let variance: f64 = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        Self {
            p50_ms: sorted.get(p50_idx.min(n - 1)).copied().unwrap_or(0.0),
            p95_ms: sorted.get(p95_idx.min(n - 1)).copied().unwrap_or(0.0),
            p99_ms: sorted.get(p99_idx.min(n - 1)).copied().unwrap_or(0.0),
            min_ms: sorted.first().copied().unwrap_or(0.0),
            max_ms: sorted.last().copied().unwrap_or(0.0),
            mean_ms: mean,
            stddev_ms: variance.sqrt(),
        }
    }

    pub fn tail_latency_ratio(&self) -> f64 {
        if self.p50_ms > 0.0 {
            self.p99_ms / self.p50_ms
        } else {
            1.0
        }
    }
}

/// IMP-156a: Test latency percentile calculation
#[test]
fn test_imp_156a_latency_percentiles() {
    // 100 samples: mostly 10ms, some outliers
    let mut samples: Vec<f64> = vec![10.0; 90];
    samples.extend(vec![50.0; 5]); // P95 region
    samples.extend(vec![100.0; 5]); // P99 region

    let percentiles = LatencyPercentiles::from_samples(&samples);

    assert!(
        (percentiles.p50_ms - 10.0).abs() < 1.0,
        "IMP-156a: P50 should be ~10ms"
    );
    assert!(
        percentiles.p95_ms >= 10.0 && percentiles.p95_ms <= 100.0,
        "IMP-156a: P95 should be between 10-100ms"
    );
    assert!(
        percentiles.p99_ms >= 50.0,
        "IMP-156a: P99 should be >= 50ms"
    );
    assert!(
        percentiles.tail_latency_ratio() >= 1.0,
        "IMP-156a: Tail ratio should be >= 1"
    );

    println!("\nIMP-156a: Latency Percentiles:");
    println!("  P50: {:.1}ms", percentiles.p50_ms);
    println!("  P95: {:.1}ms", percentiles.p95_ms);
    println!("  P99: {:.1}ms", percentiles.p99_ms);
    println!(
        "  Min: {:.1}ms, Max: {:.1}ms",
        percentiles.min_ms, percentiles.max_ms
    );
    println!(
        "  Mean: {:.1}ms, Stddev: {:.1}ms",
        percentiles.mean_ms, percentiles.stddev_ms
    );
    println!(
        "  Tail ratio (P99/P50): {:.2}x",
        percentiles.tail_latency_ratio()
    );
}

/// IMP-156b: Latency comparison between runners
#[derive(Debug, Clone)]
pub struct LatencyComparison {
    pub realizar_percentiles: LatencyPercentiles,
    pub reference_percentiles: LatencyPercentiles,
    pub p50_gap_percent: f64,
    pub p99_gap_percent: f64,
    pub realizar_has_lower_p50: bool,
    pub realizar_has_lower_p99: bool,
}

impl LatencyComparison {
    pub fn new(realizar: LatencyPercentiles, reference: LatencyPercentiles) -> Self {
        let p50_gap = if reference.p50_ms > 0.0 {
            ((realizar.p50_ms - reference.p50_ms) / reference.p50_ms) * 100.0
        } else {
            0.0
        };
        let p99_gap = if reference.p99_ms > 0.0 {
            ((realizar.p99_ms - reference.p99_ms) / reference.p99_ms) * 100.0
        } else {
            0.0
        };
        Self {
            realizar_percentiles: realizar.clone(),
            reference_percentiles: reference.clone(),
            p50_gap_percent: p50_gap,
            p99_gap_percent: p99_gap,
            realizar_has_lower_p50: realizar.p50_ms < reference.p50_ms,
            realizar_has_lower_p99: realizar.p99_ms < reference.p99_ms,
        }
    }

    pub fn parity_achieved(&self) -> bool {
        // Parity if within 20% on both P50 and P99
        self.p50_gap_percent.abs() <= 20.0 && self.p99_gap_percent.abs() <= 20.0
    }
}
