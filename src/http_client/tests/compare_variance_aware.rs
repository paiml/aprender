
/// IMP-159b: Variance-aware throughput comparison
#[derive(Debug, Clone)]
pub struct VarianceAwareComparison {
    /// First measurement (e.g., Realizar)
    pub measurement_a: ThroughputWithVariance,
    /// Second measurement (e.g., llama.cpp)
    pub measurement_b: ThroughputWithVariance,
    /// Ratio of means (B/A)
    pub mean_ratio: f64,
    /// Whether difference is statistically significant
    pub statistically_significant: bool,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
}

impl VarianceAwareComparison {
    /// Compare two measurements with statistical analysis
    pub fn compare(a: &ThroughputWithVariance, b: &ThroughputWithVariance) -> Self {
        let mean_ratio = if a.mean_tps > 0.0 {
            b.mean_tps / a.mean_tps
        } else {
            1.0
        };

        // Cohen's d effect size
        let pooled_stddev = f64::midpoint(a.stddev_tps.powi(2), b.stddev_tps.powi(2)).sqrt();
        let effect_size = if pooled_stddev > 0.0 {
            (b.mean_tps - a.mean_tps).abs() / pooled_stddev
        } else {
            0.0
        };

        // Statistical significance: CI don't overlap
        let (a_lower, a_upper) = a.confidence_interval();
        let (b_lower, b_upper) = b.confidence_interval();
        let statistically_significant = a_upper < b_lower || b_upper < a_lower;

        Self {
            measurement_a: a.clone(),
            measurement_b: b.clone(),
            mean_ratio,
            statistically_significant,
            effect_size,
        }
    }

    /// Check if B is significantly faster than A
    pub fn b_significantly_faster(&self) -> bool {
        self.statistically_significant && self.mean_ratio > 1.0
    }

    /// Get effect size interpretation (small/medium/large per Cohen)
    pub fn effect_interpretation(&self) -> &'static str {
        if self.effect_size < 0.2 {
            "negligible"
        } else if self.effect_size < 0.5 {
            "small"
        } else if self.effect_size < 0.8 {
            "medium"
        } else {
            "large"
        }
    }
}

/// IMP-159b: Test variance-aware comparison
#[test]
fn test_imp_159b_variance_aware_comparison() {
    // Realizar samples: ~80 tok/s
    let realizar_samples = vec![78.0, 82.0, 80.0, 79.0, 81.0, 80.0, 77.0, 83.0, 80.0, 79.0];
    let realizar = ThroughputWithVariance::from_samples(&realizar_samples);

    // llama.cpp samples: ~256 tok/s
    let llamacpp_samples = vec![
        250.0, 260.0, 255.0, 252.0, 258.0, 256.0, 248.0, 262.0, 254.0, 257.0,
    ];
    let llamacpp = ThroughputWithVariance::from_samples(&llamacpp_samples);

    let comparison = VarianceAwareComparison::compare(&realizar, &llamacpp);

    // IMP-159b: Ratio should be ~3.2x
    assert!(
        comparison.mean_ratio > 3.0 && comparison.mean_ratio < 3.5,
        "IMP-159b: Ratio should be ~3.2x, got {:.2}x",
        comparison.mean_ratio
    );

    // IMP-159b: Difference should be statistically significant
    assert!(
        comparison.statistically_significant,
        "IMP-159b: 3.2x difference should be statistically significant"
    );

    // IMP-159b: Effect size should be large
    assert!(
        comparison.effect_size > 0.8,
        "IMP-159b: Effect size should be large (>0.8), got {:.2}",
        comparison.effect_size
    );

    println!("\nIMP-159b: Variance-Aware Comparison:");
    println!(
        "  Realizar: {:.2} ± {:.2} tok/s (CV={:.4})",
        realizar.mean_tps, realizar.ci_95_margin, realizar.cv
    );
    println!(
        "  llama.cpp: {:.2} ± {:.2} tok/s (CV={:.4})",
        llamacpp.mean_tps, llamacpp.ci_95_margin, llamacpp.cv
    );
    println!("  Ratio: {:.2}x", comparison.mean_ratio);
    println!("  Significant: {}", comparison.statistically_significant);
    println!(
        "  Effect size: {:.2} ({})",
        comparison.effect_size,
        comparison.effect_interpretation()
    );
}

/// IMP-159c: CV-based stopping criterion per Hoefler & Belli
#[derive(Debug, Clone)]
pub struct AdaptiveSampler {
    /// Target CV threshold
    pub target_cv: f64,
    /// Minimum samples before checking CV
    pub min_samples: usize,
    /// Maximum samples (hard limit)
    pub max_samples: usize,
    /// Current samples
    samples: Vec<f64>,
}

impl AdaptiveSampler {
    pub fn new(target_cv: f64, min_samples: usize, max_samples: usize) -> Self {
        Self {
            target_cv,
            min_samples,
            max_samples,
            samples: Vec::new(),
        }
    }

    /// Add a sample and check if we should stop
    pub fn add_sample(&mut self, value: f64) -> bool {
        self.samples.push(value);

        // Check stopping criterion
        if self.samples.len() < self.min_samples {
            return false; // Need more samples
        }

        if self.samples.len() >= self.max_samples {
            return true; // Hit max limit
        }

        // Check CV
        let stats = ThroughputWithVariance::from_samples(&self.samples);
        stats.cv <= self.target_cv
    }

    /// Get current statistics
    pub fn current_stats(&self) -> ThroughputWithVariance {
        ThroughputWithVariance::from_samples(&self.samples)
    }

    /// Get sample count
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

/// IMP-159c: Test adaptive sampling with CV stopping
#[test]
fn test_imp_159c_adaptive_cv_stopping() {
    // Scenario 1: Stable measurements should stop early
    let mut sampler = AdaptiveSampler::new(0.05, 5, 20);
    let stable_values = [100.0, 101.0, 99.0, 100.0, 100.0, 101.0, 99.0, 100.0];

    let mut stopped_at = 0;
    for (i, &value) in stable_values.iter().enumerate() {
        if sampler.add_sample(value) {
            stopped_at = i + 1;
            break;
        }
    }

    // IMP-159c: Should stop early with stable values
    assert!(
        stopped_at >= 5 && stopped_at <= 8,
        "IMP-159c: Stable values should stop at 5-8 samples, stopped at {}",
        stopped_at
    );

    let final_stats = sampler.current_stats();
    assert!(
        final_stats.cv <= 0.05,
        "IMP-159c: Final CV should be <= 5%, got {:.4}",
        final_stats.cv
    );

    println!("\nIMP-159c: Adaptive CV Stopping:");
    println!("  Target CV: {:.2}%", sampler.target_cv * 100.0);
    println!("  Stopped at: {} samples", stopped_at);
    println!(
        "  Final CV: {:.4} ({:.2}%)",
        final_stats.cv,
        final_stats.cv * 100.0
    );
    println!("  Mean: {:.2} tok/s", final_stats.mean_tps);
}

/// IMP-159d: Real-world variance tracking with llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_159d_realworld_variance_tracking() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(30);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Count from 1 to 10:".to_string(),
        max_tokens: 30,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect samples with adaptive stopping
    let mut sampler = AdaptiveSampler::new(0.10, 5, 15);
    let mut iteration = 0;

    while !sampler.add_sample(0.0) && iteration < 15 {
        let start = std::time::Instant::now();
        if let Ok(result) = client.llamacpp_completion("http://127.0.0.1:8082", &request) {
            let elapsed = start.elapsed().as_secs_f64();
            let tokens = result.text.split_whitespace().count();
            let throughput = tokens as f64 / elapsed;

            // Replace the dummy 0.0 with actual throughput
            sampler.samples.pop();
            sampler.samples.push(throughput);
        }
        iteration += 1;
    }

    let stats = sampler.current_stats();

    // IMP-159d: Verify we got meaningful measurements
    assert!(
        stats.sample_count >= 5,
        "IMP-159d: Should collect at least 5 samples, got {}",
        stats.sample_count
    );

    assert!(
        stats.mean_tps > 10.0,
        "IMP-159d: Mean throughput should be > 10 tok/s, got {:.2}",
        stats.mean_tps
    );

    println!("\nIMP-159d: Real-World Variance Tracking (llama.cpp):");
    println!("  Samples collected: {}", stats.sample_count);
    println!("  Mean throughput: {:.2} tok/s", stats.mean_tps);
    println!("  Stddev: {:.2} tok/s", stats.stddev_tps);
    println!("  CV: {:.4} ({:.2}%)", stats.cv, stats.cv * 100.0);
    println!(
        "  95% CI: ({:.2}, {:.2})",
        stats.confidence_interval().0,
        stats.confidence_interval().1
    );
    println!(
        "  Meets 10% CV threshold: {}",
        stats.meets_cv_threshold(0.10)
    );
}

// =========================================================================
// IMP-160: Multi-Run Statistical Benchmark Analysis (EXTREME TDD)
// =========================================================================
// Per spec: Scientific benchmarking requires multiple independent runs.
// Implements bootstrap confidence intervals and effect size analysis.
// Run with: cargo test test_imp_160 --lib --features bench-http

/// IMP-160a: Multi-run benchmark result aggregation
#[derive(Debug, Clone)]
pub struct MultiRunBenchmark {
    /// Server name being benchmarked
    pub server_name: String,
    /// Number of complete benchmark runs
    pub run_count: usize,
    /// Results from each run (each run has its own stats)
    pub run_results: Vec<ThroughputWithVariance>,
    /// Aggregated mean across all runs
    pub aggregate_mean_tps: f64,
    /// Standard deviation of run means
    pub run_mean_stddev: f64,
    /// CV of run means (variability between runs)
    pub between_run_cv: f64,
    /// Overall sample count (sum of all runs)
    pub total_samples: usize,
}

impl MultiRunBenchmark {
    /// Create from multiple benchmark runs
    pub fn from_runs(server_name: &str, runs: Vec<ThroughputWithVariance>) -> Self {
        let run_count = runs.len();
        if run_count == 0 {
            return Self {
                server_name: server_name.to_string(),
                run_count: 0,
                run_results: Vec::new(),
                aggregate_mean_tps: 0.0,
                run_mean_stddev: 0.0,
                between_run_cv: 0.0,
                total_samples: 0,
            };
        }

        // Collect run means for aggregation
        let run_means: Vec<f64> = runs.iter().map(|r| r.mean_tps).collect();
        let total_samples: usize = runs.iter().map(|r| r.sample_count).sum();

        // Aggregate statistics
        let aggregate_mean = run_means.iter().sum::<f64>() / run_count as f64;
        let variance = if run_count > 1 {
            run_means
                .iter()
                .map(|x| (x - aggregate_mean).powi(2))
                .sum::<f64>()
                / (run_count - 1) as f64
        } else {
            0.0
        };
        let run_stddev = variance.sqrt();
        let cv = if aggregate_mean > 0.0 {
            run_stddev / aggregate_mean
        } else {
            0.0
        };

        Self {
            server_name: server_name.to_string(),
            run_count,
            run_results: runs,
            aggregate_mean_tps: aggregate_mean,
            run_mean_stddev: run_stddev,
            between_run_cv: cv,
            total_samples,
        }
    }

    /// Check if results are reproducible (low between-run variance)
    pub fn is_reproducible(&self, cv_threshold: f64) -> bool {
        self.run_count >= 3 && self.between_run_cv <= cv_threshold
    }

    /// Get bootstrap 95% CI from run means
    pub fn bootstrap_ci(&self) -> (f64, f64) {
        if self.run_count < 3 {
            return (self.aggregate_mean_tps, self.aggregate_mean_tps);
        }
        // Simple percentile bootstrap approximation
        let t_value = 2.0; // Approximate for small samples
        let margin = t_value * self.run_mean_stddev / (self.run_count as f64).sqrt();
        (
            self.aggregate_mean_tps - margin,
            self.aggregate_mean_tps + margin,
        )
    }
}

/// IMP-160a: Test multi-run aggregation
#[test]
fn test_imp_160a_multirun_aggregation() {
    // Simulate 5 benchmark runs for llama.cpp
    let run1 = ThroughputWithVariance::from_samples(&[254.0, 258.0, 252.0, 256.0, 255.0]);
    let run2 = ThroughputWithVariance::from_samples(&[260.0, 262.0, 258.0, 261.0, 259.0]);
    let run3 = ThroughputWithVariance::from_samples(&[248.0, 252.0, 250.0, 249.0, 251.0]);
    let run4 = ThroughputWithVariance::from_samples(&[255.0, 257.0, 254.0, 256.0, 256.0]);
    let run5 = ThroughputWithVariance::from_samples(&[250.0, 252.0, 251.0, 253.0, 249.0]);

    let multirun = MultiRunBenchmark::from_runs("llama.cpp", vec![run1, run2, run3, run4, run5]);

    // IMP-160a: Should have 5 runs
    assert_eq!(multirun.run_count, 5, "IMP-160a: Should have 5 runs");

    // IMP-160a: Aggregate mean should be ~255
    assert!(
        (multirun.aggregate_mean_tps - 254.0).abs() < 3.0,
        "IMP-160a: Aggregate mean should be ~254, got {:.2}",
        multirun.aggregate_mean_tps
    );

    // IMP-160a: Between-run CV should be low (reproducible)
    assert!(
        multirun.between_run_cv < 0.05,
        "IMP-160a: Between-run CV should be < 5%, got {:.4}",
        multirun.between_run_cv
    );

    // IMP-160a: Should be reproducible
    assert!(
        multirun.is_reproducible(0.10),
        "IMP-160a: Results should be reproducible"
    );

    println!("\nIMP-160a: Multi-Run Aggregation:");
    println!("  Server: {}", multirun.server_name);
    println!("  Runs: {}", multirun.run_count);
    println!("  Total samples: {}", multirun.total_samples);
    println!("  Aggregate mean: {:.2} tok/s", multirun.aggregate_mean_tps);
    println!(
        "  Between-run stddev: {:.2} tok/s",
        multirun.run_mean_stddev
    );
    println!(
        "  Between-run CV: {:.4} ({:.2}%)",
        multirun.between_run_cv,
        multirun.between_run_cv * 100.0
    );
    println!(
        "  Bootstrap 95% CI: ({:.2}, {:.2})",
        multirun.bootstrap_ci().0,
        multirun.bootstrap_ci().1
    );
}

/// IMP-160b: Multi-run comparison between servers
#[derive(Debug, Clone)]
pub struct MultiRunComparison {
    /// Server A (e.g., Realizar)
    pub server_a: MultiRunBenchmark,
    /// Server B (e.g., llama.cpp)
    pub server_b: MultiRunBenchmark,
    /// Ratio of aggregate means (B/A)
    pub aggregate_ratio: f64,
    /// Whether difference is reproducibly significant
    pub reproducibly_significant: bool,
    /// Minimum observed ratio across runs
    pub min_ratio: f64,
    /// Maximum observed ratio across runs
    pub max_ratio: f64,
}
