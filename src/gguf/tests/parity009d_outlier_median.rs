
/// Test PARITY-009d: QA-034 Outlier detection using MAD
#[test]
fn test_parity009d_outlier_detection_mad() {
    /// Median Absolute Deviation (MAD) outlier detection
    /// Per Fleming & Wallace: MAD is robust to outliers
    fn median(values: &mut [f64]) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = values.len() / 2;
        if values.len().is_multiple_of(2) {
            f64::midpoint(values[mid - 1], values[mid])
        } else {
            values[mid]
        }
    }

    fn mad(values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        let med = median(&mut sorted);
        let mut deviations: Vec<f64> = values.iter().map(|v| (v - med).abs()).collect();
        median(&mut deviations)
    }

    fn detect_outliers(values: &[f64], threshold: f64) -> Vec<usize> {
        let mut sorted = values.to_vec();
        let med = median(&mut sorted);
        let mad_value = mad(values);
        let k = 1.4826; // Scale factor for normal distribution

        values
            .iter()
            .enumerate()
            .filter(|(_, &v)| {
                if mad_value == 0.0 {
                    false
                } else {
                    ((v - med).abs() / (k * mad_value)) > threshold
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    // Test data with outliers
    let values = vec![100.0, 101.0, 99.0, 102.0, 98.0, 500.0, 100.5, 99.5];
    let outliers = detect_outliers(&values, 3.0); // 3 MAD threshold

    println!("\nPARITY-009d: MAD outlier detection");
    println!("  Values: {:?}", values);
    println!("  MAD: {:.2}", mad(&values));
    println!("  Outliers at indices: {:?}", outliers);

    assert!(
        outliers.contains(&5),
        "QA-034: Should detect 500.0 as outlier"
    );
    assert!(
        !outliers.contains(&0),
        "QA-034: 100.0 should not be outlier"
    );
}

/// Test PARITY-009e: QA-035 p50, p95, p99 latencies
#[test]
fn test_parity009e_latency_percentiles() {
    /// Latency percentile calculator per Georges et al.
    #[derive(Debug, Clone)]
    struct LatencyStats {
        p50: f64,
        p95: f64,
        p99: f64,
        min: f64,
        max: f64,
        mean: f64,
    }

    impl LatencyStats {
        fn from_latencies(latencies: &[f64]) -> Self {
            let mut sorted = latencies.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let percentile = |p: f64| -> f64 {
                let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
                sorted[idx.min(sorted.len() - 1)]
            };

            Self {
                p50: percentile(0.50),
                p95: percentile(0.95),
                p99: percentile(0.99),
                min: sorted[0],
                max: sorted[sorted.len() - 1],
                mean: latencies.iter().sum::<f64>() / latencies.len() as f64,
            }
        }
    }

    // Simulate latency distribution
    let latencies: Vec<f64> = (0..100)
        .map(|i| 10.0 + (i as f64 * 0.5) + if i > 95 { 50.0 } else { 0.0 })
        .collect();

    let stats = LatencyStats::from_latencies(&latencies);

    println!("\nPARITY-009e: Latency percentiles");
    println!("  p50: {:.2}ms", stats.p50);
    println!("  p95: {:.2}ms", stats.p95);
    println!("  p99: {:.2}ms", stats.p99);
    println!("  min: {:.2}ms, max: {:.2}ms", stats.min, stats.max);

    assert!(stats.p50 < stats.p95, "QA-035: p50 should be less than p95");
    assert!(stats.p95 < stats.p99, "QA-035: p95 should be less than p99");
    assert!(stats.min <= stats.p50, "QA-035: min should be <= p50");
    assert!(stats.p99 <= stats.max, "QA-035: p99 should be <= max");
}

/// Test PARITY-009f: QA-036 Throughput with variance
#[test]
fn test_parity009f_throughput_variance() {
    /// Throughput measurement with variance tracking
    #[derive(Debug, Clone)]
    struct ThroughputStats {
        mean_tps: f64,
        variance: f64,
        stddev: f64,
        cv: f64,
        samples: usize,
    }

    impl ThroughputStats {
        fn from_samples(tps_samples: &[f64]) -> Self {
            let n = tps_samples.len() as f64;
            let mean = tps_samples.iter().sum::<f64>() / n;
            let variance = tps_samples.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / n;
            let stddev = variance.sqrt();
            let cv = if mean > 0.0 { stddev / mean } else { 0.0 };

            Self {
                mean_tps: mean,
                variance,
                stddev,
                cv,
                samples: tps_samples.len(),
            }
        }

        fn is_stable(&self) -> bool {
            self.cv < 0.05 // 5% CV threshold
        }

        fn confidence_interval_95(&self) -> (f64, f64) {
            let margin = 1.96 * self.stddev / (self.samples as f64).sqrt();
            (self.mean_tps - margin, self.mean_tps + margin)
        }
    }

    // Simulate throughput measurements
    let tps_samples = vec![200.0, 205.0, 198.0, 202.0, 201.0, 199.0, 203.0, 200.5];
    let stats = ThroughputStats::from_samples(&tps_samples);
    let (ci_low, ci_high) = stats.confidence_interval_95();

    println!("\nPARITY-009f: Throughput with variance");
    println!("  Mean: {:.2} tok/s", stats.mean_tps);
    println!("  StdDev: {:.2}", stats.stddev);
    println!("  CV: {:.4}", stats.cv);
    println!("  95% CI: [{:.2}, {:.2}]", ci_low, ci_high);

    assert!(
        stats.is_stable(),
        "QA-036: Measurements should be stable (CV < 0.05)"
    );
    assert!(stats.variance > 0.0, "QA-036: Variance should be positive");
    assert!(
        ci_low < stats.mean_tps && stats.mean_tps < ci_high,
        "QA-036: Mean should be in CI"
    );
}

/// Test PARITY-009g: QA-037 Versioned benchmark results
#[test]
fn test_parity009g_versioned_results() {
    /// Versioned benchmark result for reproducibility
    #[derive(Debug, Clone)]
    struct VersionedBenchmarkResult {
        // Version info
        schema_version: String,
        benchmark_version: String,
        realizar_version: String,

        // Metadata
        timestamp: String,
        git_commit: String,
        environment_hash: String,

        // Results
        throughput_tps: f64,
        latency_p50_ms: f64,
        latency_p99_ms: f64,
        cv: f64,
        iterations: usize,
    }

    impl VersionedBenchmarkResult {
        fn new(tps: f64, p50: f64, p99: f64, cv: f64, iterations: usize) -> Self {
            Self {
                schema_version: "1.0.0".to_string(),
                benchmark_version: "PARITY-009".to_string(),
                realizar_version: env!("CARGO_PKG_VERSION").to_string(),
                timestamp: "2025-12-13T22:00:00Z".to_string(),
                git_commit: "abc123def".to_string(),
                environment_hash: "sha256:...".to_string(),
                throughput_tps: tps,
                latency_p50_ms: p50,
                latency_p99_ms: p99,
                cv,
                iterations,
            }
        }

        fn is_valid(&self) -> bool {
            !self.schema_version.is_empty()
                && !self.benchmark_version.is_empty()
                && !self.realizar_version.is_empty()
                && self.throughput_tps > 0.0
                && self.cv >= 0.0
                && self.iterations > 0
        }

        fn is_reproducible(&self) -> bool {
            !self.git_commit.is_empty()
                && !self.timestamp.is_empty()
                && !self.environment_hash.is_empty()
        }
    }

    let result = VersionedBenchmarkResult::new(
        200.5, // tps
        5.2,   // p50
        12.8,  // p99
        0.025, // cv
        50,    // iterations
    );

    println!("\nPARITY-009g: Versioned results");
    println!("  Schema: {}", result.schema_version);
    println!("  Benchmark: {}", result.benchmark_version);
    println!("  Realizar: {}", result.realizar_version);
    println!("  Throughput: {:.2} tok/s", result.throughput_tps);

    assert!(result.is_valid(), "QA-037: Result must be valid");
    assert!(
        result.is_reproducible(),
        "QA-037: Result must be reproducible"
    );
    assert_eq!(
        result.schema_version, "1.0.0",
        "QA-037: Schema version must be set"
    );
}

// ========================================================================
// PARITY-010: Benchmark Infrastructure QA-038 to QA-040
// ========================================================================

/// Test PARITY-010a: QA-038 Preflight checks validate server availability
#[test]
fn test_parity010a_preflight_server_checks() {
    /// Preflight check result
    #[derive(Debug, Clone)]
    enum PreflightStatus {
        Pass,
        Fail(String),
        Skip(String),
    }

    /// Server availability check
    #[derive(Debug)]
    struct ServerPreflightCheck {
        name: String,
        endpoint: String,
        timeout_ms: u64,
        required: bool,
    }

    impl ServerPreflightCheck {
        fn new(name: &str, endpoint: &str, required: bool) -> Self {
            Self {
                name: name.to_string(),
                endpoint: endpoint.to_string(),
                timeout_ms: 5000,
                required,
            }
        }

        /// Simulate server check (real impl would use HTTP client)
        fn check(&self, server_available: bool) -> PreflightStatus {
            if server_available {
                PreflightStatus::Pass
            } else if self.required {
                PreflightStatus::Fail(format!("{} not available at {}", self.name, self.endpoint))
            } else {
                PreflightStatus::Skip(format!("{} optional, skipping", self.name))
            }
        }
    }

    /// Preflight suite for benchmark servers
    #[derive(Debug)]
    struct PreflightSuite {
        checks: Vec<ServerPreflightCheck>,
    }

    impl PreflightSuite {
        fn new() -> Self {
            Self {
                checks: vec![
                    ServerPreflightCheck::new("Ollama", "http://localhost:11434", true),
                    ServerPreflightCheck::new("llama.cpp", "http://localhost:8080", false),
                    ServerPreflightCheck::new("vLLM", "http://localhost:8000", false),
                ],
            }
        }

        fn run(&self, availability: &[bool]) -> (usize, usize, usize) {
            let mut passed = 0;
            let mut failed = 0;
            let mut skipped = 0;

            for (check, &available) in self.checks.iter().zip(availability.iter()) {
                match check.check(available) {
                    PreflightStatus::Pass => passed += 1,
                    PreflightStatus::Fail(_) => failed += 1,
                    PreflightStatus::Skip(_) => skipped += 1,
                }
            }

            (passed, failed, skipped)
        }

        fn all_required_pass(&self, availability: &[bool]) -> bool {
            for (check, &available) in self.checks.iter().zip(availability.iter()) {
                if check.required && !available {
                    return false;
                }
            }
            true
        }
    }

    let suite = PreflightSuite::new();

    // Test: All servers available
    let (passed, failed, _skipped) = suite.run(&[true, true, true]);
    assert_eq!(passed, 3, "QA-038: All 3 servers should pass");
    assert_eq!(failed, 0, "QA-038: No failures");

    // Test: Only required (Ollama) available
    let (passed, _failed, skipped) = suite.run(&[true, false, false]);
    assert_eq!(passed, 1, "QA-038: Ollama passes");
    assert_eq!(skipped, 2, "QA-038: Optional servers skipped");
    assert!(
        suite.all_required_pass(&[true, false, false]),
        "QA-038: Required servers pass"
    );

    // Test: Required server unavailable
    assert!(
        !suite.all_required_pass(&[false, true, true]),
        "QA-038: Should fail if Ollama down"
    );

    println!("\nPARITY-010a: Preflight server checks");
    println!("  Checks defined: {}", suite.checks.len());
    println!("  Required: Ollama");
    println!("  Optional: llama.cpp, vLLM");
}
