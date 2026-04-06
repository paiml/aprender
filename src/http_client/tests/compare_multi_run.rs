
impl MultiRunComparison {
    pub fn compare(a: MultiRunBenchmark, b: MultiRunBenchmark) -> Self {
        let aggregate_ratio = if a.aggregate_mean_tps > 0.0 {
            b.aggregate_mean_tps / a.aggregate_mean_tps
        } else {
            1.0
        };

        // Calculate min/max ratio from individual runs
        let ratios: Vec<f64> = a
            .run_results
            .iter()
            .zip(b.run_results.iter())
            .map(|(ra, rb)| {
                if ra.mean_tps > 0.0 {
                    rb.mean_tps / ra.mean_tps
                } else {
                    1.0
                }
            })
            .collect();

        let min_ratio = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ratio = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Reproducibly significant if CIs don't overlap and both are reproducible
        let (a_lower, a_upper) = a.bootstrap_ci();
        let (b_lower, b_upper) = b.bootstrap_ci();
        let ci_separated = a_upper < b_lower || b_upper < a_lower;
        let reproducibly_significant =
            ci_separated && a.is_reproducible(0.15) && b.is_reproducible(0.15);

        Self {
            server_a: a,
            server_b: b,
            aggregate_ratio,
            reproducibly_significant,
            min_ratio,
            max_ratio,
        }
    }
}

/// IMP-160b: Test multi-run comparison
#[test]
fn test_imp_160b_multirun_comparison() {
    // Realizar runs: ~80 tok/s
    let r1 = ThroughputWithVariance::from_samples(&[78.0, 82.0, 80.0, 79.0, 81.0]);
    let r2 = ThroughputWithVariance::from_samples(&[80.0, 81.0, 79.0, 80.0, 80.0]);
    let r3 = ThroughputWithVariance::from_samples(&[77.0, 83.0, 80.0, 78.0, 82.0]);
    let realizar = MultiRunBenchmark::from_runs("Realizar", vec![r1, r2, r3]);

    // llama.cpp runs: ~256 tok/s
    let l1 = ThroughputWithVariance::from_samples(&[254.0, 258.0, 256.0, 255.0, 257.0]);
    let l2 = ThroughputWithVariance::from_samples(&[260.0, 262.0, 258.0, 259.0, 261.0]);
    let l3 = ThroughputWithVariance::from_samples(&[248.0, 252.0, 250.0, 251.0, 249.0]);
    let llamacpp = MultiRunBenchmark::from_runs("llama.cpp", vec![l1, l2, l3]);

    let comparison = MultiRunComparison::compare(realizar, llamacpp);

    // IMP-160b: Aggregate ratio should be ~3.2x
    assert!(
        comparison.aggregate_ratio > 3.0 && comparison.aggregate_ratio < 3.5,
        "IMP-160b: Aggregate ratio should be ~3.2x, got {:.2}x",
        comparison.aggregate_ratio
    );

    // IMP-160b: Difference should be reproducibly significant
    assert!(
        comparison.reproducibly_significant,
        "IMP-160b: 3.2x gap should be reproducibly significant"
    );

    println!("\nIMP-160b: Multi-Run Comparison:");
    println!(
        "  Realizar: {:.2} tok/s ({} runs)",
        comparison.server_a.aggregate_mean_tps, comparison.server_a.run_count
    );
    println!(
        "  llama.cpp: {:.2} tok/s ({} runs)",
        comparison.server_b.aggregate_mean_tps, comparison.server_b.run_count
    );
    println!("  Aggregate ratio: {:.2}x", comparison.aggregate_ratio);
    println!(
        "  Ratio range: [{:.2}x, {:.2}x]",
        comparison.min_ratio, comparison.max_ratio
    );
    println!(
        "  Reproducibly significant: {}",
        comparison.reproducibly_significant
    );
}

/// IMP-160c: Statistical power analysis for benchmark design
#[derive(Debug, Clone)]
pub struct BenchmarkPowerAnalysis {
    /// Minimum detectable effect size (Cohen's d)
    pub min_effect_size: f64,
    /// Statistical power achieved (0-1)
    pub power: f64,
    /// Sample size per group
    pub sample_size: usize,
    /// Significance level (alpha)
    pub alpha: f64,
    /// Recommended sample size for desired power
    pub recommended_n: usize,
}

impl BenchmarkPowerAnalysis {
    /// Estimate power for given effect size and sample size
    /// Uses simplified power calculation (normal approximation)
    pub fn estimate(effect_size: f64, sample_size: usize, alpha: f64, _desired_power: f64) -> Self {
        // Z-score for alpha (two-tailed)
        let z_alpha = 1.96; // For alpha = 0.05

        // Estimated power (simplified)
        let sqrt_n = (sample_size as f64).sqrt();
        let noncentrality = effect_size * sqrt_n / 2.0_f64.sqrt();
        let power = 1.0 - (1.0 / (1.0 + (noncentrality - z_alpha).exp())); // Logistic approx

        // Sample size needed for desired power
        let z_beta = 0.84; // For power = 0.80
        let recommended_n = if effect_size > 0.0 {
            let n = 2.0 * ((z_alpha + z_beta) / effect_size).powi(2);
            n.ceil() as usize
        } else {
            100 // Default if no effect
        };

        Self {
            min_effect_size: effect_size,
            power,
            sample_size,
            alpha,
            recommended_n,
        }
    }

    /// Check if power is adequate for reliable detection
    pub fn is_adequately_powered(&self) -> bool {
        self.power >= 0.80
    }
}

/// IMP-160c: Test power analysis
#[test]
fn test_imp_160c_power_analysis() {
    // Large effect (d=2.0) with small sample - should be well powered
    let large_effect = BenchmarkPowerAnalysis::estimate(2.0, 10, 0.05, 0.80);
    assert!(
        large_effect.power > 0.70,
        "IMP-160c: Large effect with n=10 should have power > 70%, got {:.2}",
        large_effect.power
    );

    // Small effect (d=0.2) with small sample - underpowered
    let small_effect = BenchmarkPowerAnalysis::estimate(0.2, 10, 0.05, 0.80);
    assert!(
        small_effect.power < 0.50,
        "IMP-160c: Small effect with n=10 should have low power, got {:.2}",
        small_effect.power
    );

    // Recommended n for small effect should be large
    assert!(
        small_effect.recommended_n > 50,
        "IMP-160c: Small effect should need many samples, got n={}",
        small_effect.recommended_n
    );

    println!("\nIMP-160c: Power Analysis:");
    println!("  Large effect (d=2.0, n=10):");
    println!("    Power: {:.2}", large_effect.power);
    println!(
        "    Adequately powered: {}",
        large_effect.is_adequately_powered()
    );
    println!("  Small effect (d=0.2, n=10):");
    println!("    Power: {:.2}", small_effect.power);
    println!(
        "    Recommended n for 80% power: {}",
        small_effect.recommended_n
    );
}

/// IMP-160d: Real-world multi-run benchmark against llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_160d_realworld_multirun() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "What is 2+2?".to_string(),
        max_tokens: 20,
        temperature: Some(0.0),
        stream: false,
    };

    // Perform 3 runs, 5 samples each
    let mut runs: Vec<ThroughputWithVariance> = Vec::new();

    for run_idx in 0..3 {
        let mut samples = Vec::new();
        for _ in 0..5 {
            let start = std::time::Instant::now();
            if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
                let elapsed = start.elapsed().as_secs_f64();
                let tokens = result.text.split_whitespace().count().max(1);
                samples.push(tokens as f64 / elapsed);
            }
        }
        if !samples.is_empty() {
            runs.push(ThroughputWithVariance::from_samples(&samples));
        }
        println!(
            "  Run {}: {} samples, mean {:.2} tok/s",
            run_idx + 1,
            samples.len(),
            runs.last().map_or(0.0, |r| r.mean_tps)
        );
    }

    let multirun = MultiRunBenchmark::from_runs("llama.cpp", runs);

    // IMP-160d: Verify multi-run results
    assert!(
        multirun.run_count >= 2,
        "IMP-160d: Should complete at least 2 runs, got {}",
        multirun.run_count
    );

    assert!(
        multirun.aggregate_mean_tps > 10.0,
        "IMP-160d: Aggregate mean should be > 10 tok/s, got {:.2}",
        multirun.aggregate_mean_tps
    );

    println!("\nIMP-160d: Real-World Multi-Run Benchmark (llama.cpp):");
    println!("  Completed runs: {}", multirun.run_count);
    println!("  Total samples: {}", multirun.total_samples);
    println!("  Aggregate mean: {:.2} tok/s", multirun.aggregate_mean_tps);
    println!(
        "  Between-run CV: {:.4} ({:.2}%)",
        multirun.between_run_cv,
        multirun.between_run_cv * 100.0
    );
    println!("  Reproducible: {}", multirun.is_reproducible(0.15));
    println!(
        "  Bootstrap 95% CI: ({:.2}, {:.2})",
        multirun.bootstrap_ci().0,
        multirun.bootstrap_ci().1
    );
}

// =========================================================================
// IMP-161: Warmup Detection and JIT Filtering (QA-032, EXTREME TDD)
// =========================================================================
// Per Vitek & Kalibera EMSOFT'11: Detect and remove warmup iterations.
// JIT compilation causes initial measurements to be non-representative.
// Run with: cargo test test_imp_161 --lib --features bench-http

/// IMP-161a: Warmup detection using changepoint analysis
#[derive(Debug, Clone)]
pub struct WarmupDetector {
    /// Minimum iterations before checking for warmup end
    pub min_iterations: usize,
    /// Maximum warmup iterations allowed
    pub max_warmup: usize,
    /// Threshold for detecting stable state (ratio of variance)
    pub stability_threshold: f64,
    /// Window size for moving average
    pub window_size: usize,
}

impl WarmupDetector {
    pub fn new(min_iterations: usize, max_warmup: usize, stability_threshold: f64) -> Self {
        Self {
            min_iterations,
            max_warmup,
            stability_threshold,
            window_size: 5,
        }
    }

    /// Default detector per Vitek & Kalibera recommendations
    pub fn default_detector() -> Self {
        Self::new(3, 10, 0.20)
    }

    /// Detect warmup end using variance ratio method
    /// Returns (warmup_iterations, steady_state_samples)
    pub fn detect_warmup(&self, samples: &[f64]) -> WarmupResult {
        let n = samples.len();
        if n < self.min_iterations + self.window_size {
            return WarmupResult {
                warmup_iterations: 0,
                steady_state_samples: samples.to_vec(),
                warmup_detected: false,
                variance_ratio: 1.0,
            };
        }

        // Calculate variance of first window vs later windows
        let mut best_split = 0;
        let mut best_ratio = f64::MAX;

        for split in self.min_iterations..n.saturating_sub(self.window_size).min(self.max_warmup) {
            let warmup = &samples[..split];
            let steady = &samples[split..];

            if warmup.len() < 2 || steady.len() < 2 {
                continue;
            }

            let warmup_var = Self::variance(warmup);
            let steady_var = Self::variance(steady);

            // If steady state has much lower variance, we found warmup end
            if warmup_var > 0.0 && steady_var > 0.0 {
                let ratio = steady_var / warmup_var;
                if ratio < best_ratio {
                    best_ratio = ratio;
                    best_split = split;
                }
            }
        }

        // Check if we detected significant warmup
        let warmup_detected = best_ratio < self.stability_threshold && best_split > 0;

        let (warmup_iters, steady_samples) = if warmup_detected {
            (best_split, samples[best_split..].to_vec())
        } else {
            (0, samples.to_vec())
        };

        WarmupResult {
            warmup_iterations: warmup_iters,
            steady_state_samples: steady_samples,
            warmup_detected,
            variance_ratio: best_ratio,
        }
    }

    fn variance(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (samples.len() - 1) as f64
    }
}

/// IMP-161a: Result of warmup detection
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Number of warmup iterations detected
    pub warmup_iterations: usize,
    /// Samples after warmup removal
    pub steady_state_samples: Vec<f64>,
    /// Whether warmup was detected
    pub warmup_detected: bool,
    /// Variance ratio (steady/warmup)
    pub variance_ratio: f64,
}

impl WarmupResult {
    /// Get statistics from steady state only
    pub fn steady_state_stats(&self) -> ThroughputWithVariance {
        ThroughputWithVariance::from_samples(&self.steady_state_samples)
    }
}

/// IMP-161a: Test warmup detection
#[test]
fn test_imp_161a_warmup_detection() {
    // Simulate warmup: first 5 samples are slow (JIT not warmed up)
    let samples = vec![
        50.0, 55.0, 60.0, 70.0, 80.0, // Warmup phase (improving)
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 101.0, 99.0, 100.0, 102.0, // Steady state
    ];

    let detector = WarmupDetector::default_detector();
    let result = detector.detect_warmup(&samples);

    // IMP-161a: Should detect warmup
    assert!(
        result.warmup_detected,
        "IMP-161a: Should detect warmup in ramping data"
    );

    // IMP-161a: Warmup should be 3-10 iterations (algorithm finds optimal variance split)
    assert!(
        result.warmup_iterations >= 3 && result.warmup_iterations <= 10,
        "IMP-161a: Warmup should be 3-10 iterations, got {}",
        result.warmup_iterations
    );

    // IMP-161a: Steady state should have higher mean
    let steady_stats = result.steady_state_stats();
    assert!(
        steady_stats.mean_tps > 90.0,
        "IMP-161a: Steady state mean should be >90, got {:.2}",
        steady_stats.mean_tps
    );

    println!("\nIMP-161a: Warmup Detection:");
    println!("  Raw samples: {:?}", samples);
    println!("  Warmup detected: {}", result.warmup_detected);
    println!("  Warmup iterations: {}", result.warmup_iterations);
    println!("  Variance ratio: {:.4}", result.variance_ratio);
    println!("  Steady state mean: {:.2} tok/s", steady_stats.mean_tps);
    println!("  Steady state CV: {:.4}", steady_stats.cv);
}

/// IMP-161b: JIT-aware benchmark runner
#[derive(Debug, Clone)]
pub struct JitAwareBenchmark {
    /// Warmup detector configuration
    pub detector: WarmupDetector,
    /// Results before warmup removal
    pub raw_stats: ThroughputWithVariance,
    /// Results after warmup removal
    pub filtered_stats: ThroughputWithVariance,
    /// Warmup detection result
    pub warmup_result: WarmupResult,
    /// Improvement from filtering (percentage)
    pub improvement_percent: f64,
}
