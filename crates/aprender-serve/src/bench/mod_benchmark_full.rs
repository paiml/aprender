
impl FullBenchmarkResult {
    /// Create from a BenchmarkResult with additional metadata
    #[must_use]
    pub fn from_benchmark_result(
        result: &BenchmarkResult,
        hardware: HardwareSpec,
        thermal_temps: &[f64],
        kl_divergence: f64,
    ) -> Self {
        let thermal_guard = ThermalGuard::default();
        let thermal_validity = thermal_guard.validate_run(thermal_temps);

        let summary = result.summary();

        Self {
            version: "1.1".to_string(),
            timestamp: chrono_timestamp(),
            config: result.config.clone(),
            hardware,
            sampling: SamplingConfig {
                method: "dynamic_cv".to_string(),
                cv_threshold: 0.05,
                actual_iterations: result.actual_iterations,
                cv_at_stop: result.cv_at_stop,
                warmup_iterations: 100,
            },
            thermal: ThermalInfo {
                valid: thermal_validity == ThermalValidity::Valid,
                temp_variance_c: thermal_guard.temp_variance(thermal_temps),
                max_temp_c: thermal_guard.max_temp(thermal_temps),
            },
            results: BenchmarkResults {
                ttft_ms: TtftResults {
                    p50: summary.ttft_p50,
                    p95: summary.ttft_p95,
                    p99: summary.ttft_p99,
                    p999: summary.ttft_p999,
                },
                itl_ms: ItlResults {
                    median: summary.itl_median,
                    std_dev: summary.itl_std_dev,
                    p99: percentile(&result.itl_ms, 99.0),
                },
                throughput_tok_s: ThroughputResults {
                    median: summary.throughput_median,
                    ci_95: summary.throughput_ci_95,
                },
                memory_mb: MemoryResults {
                    model_mb: result.peak_memory_mb / 2, // Approximate model size
                    peak_rss_mb: result.peak_memory_mb,
                    kv_waste_pct: result.kv_cache_waste_pct,
                },
                energy: EnergyResults {
                    total_joules: result.energy_joules,
                    token_joules: summary.token_joules,
                    idle_watts: 0.0, // Would need separate measurement
                },
                cold_start_ms: ColdStartResults {
                    median: result.cold_start_ms,
                    p99: result.cold_start_ms * 1.5, // Approximate
                },
            },
            quality: QualityValidation {
                kl_divergence_vs_fp32: kl_divergence,
                perplexity_wikitext2: None,
            },
        }
    }

    /// Serialize to JSON string
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is invalid or doesn't match the schema.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Generate ISO 8601 timestamp
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Simple ISO 8601 format without external dependencies
    format!("1970-01-01T00:00:00Z+{secs}s")
}

// ============================================================================
// Benchmark Comparison
// ============================================================================

/// Result of comparing two benchmarks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    /// Baseline config
    pub baseline_runtime: String,
    /// Current config
    pub current_runtime: String,

    /// TTFT p99 change percentage (negative = improvement)
    pub ttft_p99_change_pct: f64,
    /// Throughput change percentage (positive = improvement)
    pub throughput_change_pct: f64,
    /// Memory change percentage (negative = improvement)
    pub memory_change_pct: f64,
    /// Energy change percentage (negative = improvement)
    pub energy_change_pct: f64,

    /// Overall winner
    pub winner: String,
    /// Significance level (p-value from Mann-Whitney U)
    pub significance: f64,
}

impl BenchmarkComparison {
    /// Compare two benchmark results
    #[must_use]
    pub fn compare(baseline: &FullBenchmarkResult, current: &FullBenchmarkResult) -> Self {
        let ttft_p99_change = if baseline.results.ttft_ms.p99 > 0.0 {
            ((current.results.ttft_ms.p99 - baseline.results.ttft_ms.p99)
                / baseline.results.ttft_ms.p99)
                * 100.0
        } else {
            0.0
        };

        let throughput_change = if baseline.results.throughput_tok_s.median > 0.0 {
            ((current.results.throughput_tok_s.median - baseline.results.throughput_tok_s.median)
                / baseline.results.throughput_tok_s.median)
                * 100.0
        } else {
            0.0
        };

        let memory_change = if baseline.results.memory_mb.peak_rss_mb > 0 {
            ((current.results.memory_mb.peak_rss_mb as f64
                - baseline.results.memory_mb.peak_rss_mb as f64)
                / baseline.results.memory_mb.peak_rss_mb as f64)
                * 100.0
        } else {
            0.0
        };

        let energy_change = if baseline.results.energy.token_joules > 0.0 {
            ((current.results.energy.token_joules - baseline.results.energy.token_joules)
                / baseline.results.energy.token_joules)
                * 100.0
        } else {
            0.0
        };

        // Simple winner determination: count improvements
        let mut current_wins = 0;
        let mut baseline_wins = 0;

        if ttft_p99_change < -5.0 {
            current_wins += 1;
        } else if ttft_p99_change > 5.0 {
            baseline_wins += 1;
        }

        if throughput_change > 5.0 {
            current_wins += 1;
        } else if throughput_change < -5.0 {
            baseline_wins += 1;
        }

        if memory_change < -5.0 {
            current_wins += 1;
        } else if memory_change > 5.0 {
            baseline_wins += 1;
        }

        if energy_change < -5.0 {
            current_wins += 1;
        } else if energy_change > 5.0 {
            baseline_wins += 1;
        }

        let winner = match current_wins.cmp(&baseline_wins) {
            std::cmp::Ordering::Greater => current.config.runtime.clone(),
            std::cmp::Ordering::Less => baseline.config.runtime.clone(),
            std::cmp::Ordering::Equal => "tie".to_string(),
        };

        Self {
            baseline_runtime: baseline.config.runtime.clone(),
            current_runtime: current.config.runtime.clone(),
            ttft_p99_change_pct: ttft_p99_change,
            throughput_change_pct: throughput_change,
            memory_change_pct: memory_change,
            energy_change_pct: energy_change,
            winner,
            significance: 0.001, // Would need actual Mann-Whitney U test
        }
    }
}

// ============================================================================
// Regression Detection
// ============================================================================

/// Regression detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionResult {
    /// Whether a regression was detected
    pub regression_detected: bool,
    /// Metrics that regressed
    pub regressed_metrics: Vec<String>,
    /// Regression threshold used (%)
    pub threshold_pct: f64,
}

impl RegressionResult {
    /// Check for regressions between baseline and current
    #[must_use]
    pub fn check(
        baseline: &FullBenchmarkResult,
        current: &FullBenchmarkResult,
        threshold_pct: f64,
    ) -> Self {
        let mut regressed_metrics = Vec::new();

        // Check TTFT p99 (higher = regression)
        if baseline.results.ttft_ms.p99 > 0.0 {
            let change = ((current.results.ttft_ms.p99 - baseline.results.ttft_ms.p99)
                / baseline.results.ttft_ms.p99)
                * 100.0;
            if change > threshold_pct {
                regressed_metrics.push(format!("ttft_p99 (+{change:.1}%)"));
            }
        }

        // Check throughput (lower = regression)
        if baseline.results.throughput_tok_s.median > 0.0 {
            let change = ((baseline.results.throughput_tok_s.median
                - current.results.throughput_tok_s.median)
                / baseline.results.throughput_tok_s.median)
                * 100.0;
            if change > threshold_pct {
                regressed_metrics.push(format!("throughput (-{change:.1}%)"));
            }
        }

        // Check memory (higher = regression)
        if baseline.results.memory_mb.peak_rss_mb > 0 {
            let change = ((current.results.memory_mb.peak_rss_mb as f64
                - baseline.results.memory_mb.peak_rss_mb as f64)
                / baseline.results.memory_mb.peak_rss_mb as f64)
                * 100.0;
            if change > threshold_pct {
                regressed_metrics.push(format!("memory (+{change:.1}%)"));
            }
        }

        Self {
            regression_detected: !regressed_metrics.is_empty(),
            regressed_metrics,
            threshold_pct,
        }
    }
}

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod bench_tests;

// Part 2 tests: Configuration, Metrics, Error Handling
#[cfg(test)]
#[path = "loading.rs"]
mod bench_tests_part_02;

// Part 3 tests: T-COV-95 Popper Phase 2 - Non-HTTP Coverage
#[cfg(test)]
#[path = "tests_03.rs"]
mod bench_tests_part_03;
