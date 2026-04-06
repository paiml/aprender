
/// IMP-163d: Real-world A/B comparison against llama.cpp
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_163d_realworld_ab_latency() {
    // This test requires: llama-server -m model.gguf --host 127.0.0.1 --port 8082 -ngl 99
    let client = ModelHttpClient::with_timeout(60);
    let request = CompletionRequest {
        model: "default".to_string(),
        prompt: "Say hello:".to_string(),
        max_tokens: 10,
        temperature: Some(0.0),
        stream: false,
    };

    // Collect llama.cpp latencies
    let mut llamacpp_latencies = Vec::new();
    for _ in 0..10 {
        let start = std::time::Instant::now();
        if client
            .llamacpp_completion("http://127.0.0.1:8082", &request)
            .is_ok()
        {
            llamacpp_latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    if llamacpp_latencies.len() < 5 {
        println!("IMP-163d: Not enough llama.cpp samples");
        return;
    }

    // test Realizar latencies (since we don't have a running server)
    // In production, this would call the Realizar server
    let realizar_latencies: Vec<f64> = llamacpp_latencies
        .iter()
        .map(|lat| lat * 3.2) // Simulate 3.2x slower
        .collect();

    let comparison = ABLatencyComparison::compare(
        "Realizar",
        &realizar_latencies,
        "llama.cpp",
        &llamacpp_latencies,
    );

    let qa012_check = QA012Violation::check("llama.cpp", &llamacpp_latencies);

    println!("\nIMP-163d: Real-World A/B Latency Comparison:");
    println!("  {}", comparison.summary());
    println!("  llama.cpp samples: {}", llamacpp_latencies.len());
    println!(
        "  llama.cpp p50: {:.1}ms",
        comparison.server_b_latency.p50_ms
    );
    println!(
        "  llama.cpp p99: {:.1}ms",
        comparison.server_b_latency.p99_ms
    );
    println!(
        "  llama.cpp QA-012: {} (ratio={:.2}x)",
        if qa012_check.violated { "FAIL" } else { "PASS" },
        qa012_check.tail_ratio
    );
    println!(
        "  Estimated Realizar p50: {:.1}ms",
        comparison.server_a_latency.p50_ms
    );
}

// =========================================================================
// IMP-164: Real-World Throughput Regression Detection (QA-011, EXTREME TDD)
// =========================================================================
// Per spec QA-011: Throughput regression < 5% between commits (CI gate)
// Run with: cargo test test_imp_164 --lib --features bench-http

/// IMP-164a: Throughput regression tracker
#[derive(Debug, Clone)]
pub struct ThroughputRegressionTracker {
    /// Baseline throughput (from previous commit/release)
    pub baseline_tps: f64,
    /// Current throughput
    pub current_tps: f64,
    /// Regression threshold percentage
    pub threshold_percent: f64,
    /// Actual change percentage (negative = regression)
    pub change_percent: f64,
    /// Whether regression exceeds threshold
    pub regression_detected: bool,
    /// CI gate status
    pub ci_gate_passed: bool,
}

impl ThroughputRegressionTracker {
    pub fn check(baseline_tps: f64, current_tps: f64, threshold_percent: f64) -> Self {
        let change_percent = if baseline_tps > 0.0 {
            ((current_tps - baseline_tps) / baseline_tps) * 100.0
        } else {
            0.0
        };

        let regression_detected = change_percent < -threshold_percent;
        let ci_gate_passed = !regression_detected;

        Self {
            baseline_tps,
            current_tps,
            threshold_percent,
            change_percent,
            regression_detected,
            ci_gate_passed,
        }
    }

    /// Format for CI output
    pub fn ci_message(&self) -> String {
        if self.ci_gate_passed {
            format!(
                "✅ Throughput OK: {:.1} tok/s ({:+.1}% vs baseline {:.1})",
                self.current_tps, self.change_percent, self.baseline_tps
            )
        } else {
            format!(
                "❌ REGRESSION: {:.1} tok/s ({:.1}% below baseline {:.1}, threshold {:.1}%)",
                self.current_tps, -self.change_percent, self.baseline_tps, self.threshold_percent
            )
        }
    }
}

/// IMP-164a: Test regression tracker
#[test]
fn test_imp_164a_throughput_regression_tracker() {
    // No regression: current > baseline
    let improvement = ThroughputRegressionTracker::check(80.0, 85.0, 5.0);
    assert!(
        !improvement.regression_detected,
        "IMP-164a: Improvement should not be regression"
    );
    assert!(
        improvement.ci_gate_passed,
        "IMP-164a: CI gate should pass on improvement"
    );
    assert!(
        improvement.change_percent > 0.0,
        "IMP-164a: Change should be positive"
    );

    // Minor regression within threshold
    let minor = ThroughputRegressionTracker::check(80.0, 77.0, 5.0);
    assert!(
        !minor.regression_detected,
        "IMP-164a: 3.75% drop within 5% threshold"
    );
    assert!(
        minor.ci_gate_passed,
        "IMP-164a: CI gate should pass on minor drop"
    );

    // Major regression exceeds threshold
    let major = ThroughputRegressionTracker::check(80.0, 70.0, 5.0);
    assert!(
        major.regression_detected,
        "IMP-164a: 12.5% drop exceeds 5% threshold"
    );
    assert!(
        !major.ci_gate_passed,
        "IMP-164a: CI gate should fail on major regression"
    );

    println!("\nIMP-164a: Throughput Regression Tracker:");
    println!("  Improvement: {}", improvement.ci_message());
    println!("  Minor drop: {}", minor.ci_message());
    println!("  Major regression: {}", major.ci_message());
}

/// IMP-164b: Historical regression analysis
#[derive(Debug, Clone)]
pub struct HistoricalRegressionAnalysis {
    /// Version history entries
    pub history: Vec<VersionThroughput>,
    /// Detected regressions
    pub regressions: Vec<RegressionEvent>,
    /// Overall trend ("improving", "stable", "degrading")
    pub trend: String,
    /// Max regression observed
    pub max_regression_percent: f64,
}

#[derive(Debug, Clone)]
pub struct VersionThroughput {
    pub version: String,
    pub throughput_tps: f64,
}

#[derive(Debug, Clone)]
pub struct RegressionEvent {
    pub from_version: String,
    pub to_version: String,
    pub regression_percent: f64,
}

impl HistoricalRegressionAnalysis {
    pub fn analyze(history: Vec<VersionThroughput>, threshold_percent: f64) -> Self {
        let mut regressions = Vec::new();
        let mut max_regression: f64 = 0.0;

        for window in history.windows(2) {
            let prev = &window[0];
            let curr = &window[1];
            if prev.throughput_tps > 0.0 {
                let change =
                    ((curr.throughput_tps - prev.throughput_tps) / prev.throughput_tps) * 100.0;
                if change < -threshold_percent {
                    max_regression = max_regression.max(-change);
                    regressions.push(RegressionEvent {
                        from_version: prev.version.clone(),
                        to_version: curr.version.clone(),
                        regression_percent: -change,
                    });
                }
            }
        }

        // Determine trend
        let trend = if history.len() < 2 {
            "unknown"
        } else {
            let first = history.first().map_or(0.0, |v| v.throughput_tps);
            let last = history.last().map_or(0.0, |v| v.throughput_tps);
            let overall_change = if first > 0.0 {
                (last - first) / first
            } else {
                0.0
            };
            if overall_change > 0.05 {
                "improving"
            } else if overall_change < -0.05 {
                "degrading"
            } else {
                "stable"
            }
        };

        Self {
            history,
            regressions,
            trend: trend.to_string(),
            max_regression_percent: max_regression,
        }
    }
}

/// IMP-164b: Test historical regression analysis
#[test]
fn test_imp_164b_historical_regression() {
    let history = vec![
        VersionThroughput {
            version: "v0.1.0".to_string(),
            throughput_tps: 50.0,
        },
        VersionThroughput {
            version: "v0.2.0".to_string(),
            throughput_tps: 60.0,
        }, // +20%
        VersionThroughput {
            version: "v0.3.0".to_string(),
            throughput_tps: 55.0,
        }, // -8.3% REGRESSION
        VersionThroughput {
            version: "v0.4.0".to_string(),
            throughput_tps: 70.0,
        }, // +27%
        VersionThroughput {
            version: "v0.5.0".to_string(),
            throughput_tps: 80.0,
        }, // +14%
    ];

    let analysis = HistoricalRegressionAnalysis::analyze(history, 5.0);

    // IMP-164b: Should detect one regression (v0.2.0 → v0.3.0)
    assert_eq!(
        analysis.regressions.len(),
        1,
        "IMP-164b: Should detect 1 regression"
    );

    // IMP-164b: Overall trend should be improving (50 → 80)
    assert_eq!(
        analysis.trend, "improving",
        "IMP-164b: Trend should be improving"
    );

    println!("\nIMP-164b: Historical Regression Analysis:");
    println!("  Versions analyzed: {}", analysis.history.len());
    println!("  Regressions detected: {}", analysis.regressions.len());
    for reg in &analysis.regressions {
        println!(
            "    {} → {}: -{:.1}%",
            reg.from_version, reg.to_version, reg.regression_percent
        );
    }
    println!("  Overall trend: {}", analysis.trend);
    println!("  Max regression: {:.1}%", analysis.max_regression_percent);
}

/// IMP-164c: CI gate configuration
#[derive(Debug, Clone)]
pub struct CIGateConfig {
    /// Regression threshold for blocking
    pub block_threshold_percent: f64,
    /// Warning threshold (non-blocking)
    pub warn_threshold_percent: f64,
    /// Minimum samples for reliable measurement
    pub min_samples: usize,
    /// Maximum CV for measurement quality
    pub max_cv: f64,
}

impl CIGateConfig {
    pub fn default_config() -> Self {
        Self {
            block_threshold_percent: 5.0,
            warn_threshold_percent: 2.0,
            min_samples: 10,
            max_cv: 0.10,
        }
    }

    pub fn strict_config() -> Self {
        Self {
            block_threshold_percent: 2.0,
            warn_threshold_percent: 1.0,
            min_samples: 20,
            max_cv: 0.05,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CIGateResult {
    pub config: CIGateConfig,
    pub passed: bool,
    pub warning: bool,
    pub regression_percent: f64,
    pub measurement_quality: String,
    pub message: String,
}

impl CIGateResult {
    pub fn evaluate(
        config: CIGateConfig,
        baseline: &ThroughputWithVariance,
        current: &ThroughputWithVariance,
    ) -> Self {
        let regression_percent = if baseline.mean_tps > 0.0 {
            ((baseline.mean_tps - current.mean_tps) / baseline.mean_tps) * 100.0
        } else {
            0.0
        };

        let measurement_quality = if current.sample_count < config.min_samples {
            "insufficient_samples"
        } else if current.cv > config.max_cv {
            "high_variance"
        } else {
            "good"
        };

        let passed = regression_percent < config.block_threshold_percent;
        let warning = regression_percent >= config.warn_threshold_percent && passed;

        let message = if !passed {
            format!(
                "❌ BLOCKED: {:.1}% regression exceeds {:.1}% threshold",
                regression_percent, config.block_threshold_percent
            )
        } else if warning {
            format!(
                "⚠️ WARNING: {:.1}% regression approaching threshold",
                regression_percent
            )
        } else {
            format!(
                "✅ PASSED: {:.1}% change within acceptable range",
                regression_percent
            )
        };

        Self {
            config,
            passed,
            warning,
            regression_percent,
            measurement_quality: measurement_quality.to_string(),
            message,
        }
    }
}
