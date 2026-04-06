//! Statistical measurement and analysis for benchmarking
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - BENCH-004: MeasurementProtocol
//! - BENCH-005: LatencyStatistics
//! - BENCH-006: Outlier Detection (MAD-based)
//! - BENCH-007: Regression Detection
//! - BENCH-008: Welch's t-test for Statistical Significance

#![allow(clippy::cast_precision_loss)]

use std::time::Duration;

// ============================================================================
// BENCH-004: MeasurementProtocol (following SPEC-BENCH-001)
// ============================================================================

/// Complete measurement protocol for benchmarking
///
/// Follows MLPerf™ Inference benchmarking principles for scientific rigor.
#[derive(Debug, Clone)]
pub struct MeasurementProtocol {
    /// Number of latency samples to collect
    pub latency_samples: usize,
    /// Percentiles to compute (e.g., 50, 90, 95, 99, 99.9)
    pub latency_percentiles: Vec<f64>,
    /// Duration for throughput measurement
    pub throughput_duration: Duration,
    /// Ramp-up time before throughput measurement
    pub throughput_ramp_up: Duration,
    /// Number of memory samples to collect
    pub memory_samples: usize,
    /// Interval between memory samples
    pub memory_interval: Duration,
}

impl Default for MeasurementProtocol {
    fn default() -> Self {
        Self {
            latency_samples: 100,
            latency_percentiles: vec![50.0, 90.0, 95.0, 99.0, 99.9],
            throughput_duration: Duration::from_secs(60),
            throughput_ramp_up: Duration::from_secs(10),
            memory_samples: 10,
            memory_interval: Duration::from_secs(1),
        }
    }
}

impl MeasurementProtocol {
    /// Create a new measurement protocol with default values
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of latency samples
    #[must_use]
    pub fn with_latency_samples(mut self, samples: usize) -> Self {
        self.latency_samples = samples;
        self
    }

    /// Set the percentiles to compute
    #[must_use]
    pub fn with_percentiles(mut self, percentiles: Vec<f64>) -> Self {
        self.latency_percentiles = percentiles;
        self
    }

    /// Set the throughput measurement duration
    #[must_use]
    pub fn with_throughput_duration(mut self, duration: Duration) -> Self {
        self.throughput_duration = duration;
        self
    }

    /// Set the number of memory samples
    #[must_use]
    pub fn with_memory_samples(mut self, samples: usize) -> Self {
        self.memory_samples = samples;
        self
    }
}

// ============================================================================
// BENCH-005: LatencyStatistics (following SPEC-BENCH-001 Section 7.1)
// ============================================================================

/// Comprehensive latency statistics following MLPerf™ reporting standards
#[derive(Debug, Clone)]
pub struct LatencyStatistics {
    /// Mean latency
    pub mean: Duration,
    /// Standard deviation
    pub std_dev: Duration,
    /// Minimum latency
    pub min: Duration,
    /// Maximum latency
    pub max: Duration,
    /// 50th percentile (median)
    pub p50: Duration,
    /// 90th percentile
    pub p90: Duration,
    /// 95th percentile
    pub p95: Duration,
    /// 99th percentile
    pub p99: Duration,
    /// 99.9th percentile (tail latency)
    pub p999: Duration,
    /// Number of samples
    pub samples: usize,
    /// 95% confidence interval (lower, upper)
    pub confidence_interval_95: (Duration, Duration),
}

impl LatencyStatistics {
    /// Compute statistics from a slice of duration samples
    ///
    /// # Panics
    /// Panics if samples is empty
    #[must_use]
    pub fn from_samples(samples: &[Duration]) -> Self {
        assert!(!samples.is_empty(), "samples must not be empty");

        let n = samples.len();
        let n_f64 = n as f64;

        // Compute mean
        let sum_nanos: u128 = samples.iter().map(Duration::as_nanos).sum();
        let mean_nanos = sum_nanos / n as u128;
        let mean = Duration::from_nanos(mean_nanos as u64);

        // Compute standard deviation
        let variance: f64 = samples
            .iter()
            .map(|s| {
                let diff = s.as_nanos() as f64 - mean_nanos as f64;
                diff * diff
            })
            .sum::<f64>()
            / (n_f64 - 1.0).max(1.0);
        let std_dev_nanos = variance.sqrt();
        let std_dev = Duration::from_nanos(std_dev_nanos as u64);

        // Sort for percentile computation
        let mut sorted: Vec<Duration> = samples.to_vec();
        sorted.sort();

        // Min/max
        let min = sorted[0];
        let max = sorted[n - 1];

        // Percentiles using nearest-rank method
        let percentile = |p: f64| -> Duration {
            let idx = ((p / 100.0) * n_f64).ceil() as usize;
            sorted[idx.saturating_sub(1).min(n - 1)]
        };

        let p50 = percentile(50.0);
        let p90 = percentile(90.0);
        let p95 = percentile(95.0);
        let p99 = percentile(99.0);
        let p999 = percentile(99.9);

        // 95% confidence interval using t-distribution approximation
        // For large n, t ≈ 1.96
        let t_value = if n >= 30 { 1.96 } else { 2.0 + 4.0 / n_f64 };
        let margin = std_dev_nanos * t_value / n_f64.sqrt();
        let lower = Duration::from_nanos((mean_nanos as f64 - margin).max(0.0) as u64);
        let upper = Duration::from_nanos((mean_nanos as f64 + margin) as u64);

        Self {
            mean,
            std_dev,
            min,
            max,
            p50,
            p90,
            p95,
            p99,
            p999,
            samples: n,
            confidence_interval_95: (lower, upper),
        }
    }
}

// ============================================================================
// BENCH-006: Outlier Detection (MAD-based)
// ============================================================================

/// Detect outliers using Median Absolute Deviation (MAD) method
///
/// More robust than standard deviation for non-normal distributions.
/// Uses the modified Z-score method with configurable threshold.
///
/// # Arguments
/// * `samples` - Slice of f64 samples
/// * `threshold` - Modified Z-score threshold (typically 3.5 for strict, 2.0 for lenient)
///
/// # Returns
/// Vector of indices that are considered outliers
pub fn detect_outliers(samples: &[f64], threshold: f64) -> Vec<usize> {
    if samples.len() < 3 {
        return Vec::new();
    }

    // Calculate median
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.len().is_multiple_of(2) {
        f64::midpoint(sorted[sorted.len() / 2 - 1], sorted[sorted.len() / 2])
    } else {
        sorted[sorted.len() / 2]
    };

    // Calculate MAD (Median Absolute Deviation)
    let mut deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if deviations.len().is_multiple_of(2) {
        f64::midpoint(
            deviations[deviations.len() / 2 - 1],
            deviations[deviations.len() / 2],
        )
    } else {
        deviations[deviations.len() / 2]
    };

    // Avoid division by zero
    if mad < f64::EPSILON {
        return Vec::new();
    }

    // Constant for normal distribution approximation
    let k = 1.4826;

    // Find outliers using modified Z-score
    samples
        .iter()
        .enumerate()
        .filter(|(_, &x)| {
            let modified_z = (x - median) / (k * mad);
            modified_z.abs() > threshold
        })
        .map(|(i, _)| i)
        .collect()
}

// ============================================================================
// BENCH-007: Regression Detection
// ============================================================================

/// Single benchmark metric for comparison
#[derive(Debug, Clone)]
pub struct BenchmarkMetrics {
    /// Metric name
    pub name: String,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Number of samples
    pub samples: usize,
}

/// Individual regression item
#[derive(Debug, Clone)]
pub struct Regression {
    /// Metric that regressed
    pub metric: String,
    /// Baseline value
    pub baseline: f64,
    /// Current value
    pub current: f64,
    /// Percentage change
    pub change_percent: f64,
}

/// Report from regression analysis
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// Metrics that exceeded failure threshold
    pub regressions: Vec<Regression>,
    /// Metrics that exceeded warning threshold
    pub warnings: Vec<Regression>,
    /// Metrics that improved significantly
    pub improvements: Vec<Regression>,
    /// Overall pass/fail (no regressions)
    pub passed: bool,
}

/// Performance regression detector
///
/// Compares baseline and current benchmark results to detect
/// performance regressions, warnings, and improvements.
#[derive(Debug, Clone)]
pub struct RegressionDetector {
    /// Warning threshold (default: 2%)
    pub warning_threshold: f64,
    /// Failure threshold (default: 5%)
    pub failure_threshold: f64,
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self {
            warning_threshold: 0.02, // 2%
            failure_threshold: 0.05, // 5%
        }
    }
}

impl RegressionDetector {
    /// Compare baseline and current metrics
    pub fn compare(
        &self,
        baseline: &BenchmarkMetrics,
        current: &BenchmarkMetrics,
    ) -> RegressionReport {
        let mut regressions = Vec::new();
        let mut warnings = Vec::new();
        let mut improvements = Vec::new();

        // Calculate percentage change (positive = regression for latency-like metrics)
        let change = (current.mean - baseline.mean) / baseline.mean;

        let item = Regression {
            metric: baseline.name.clone(),
            baseline: baseline.mean,
            current: current.mean,
            change_percent: change * 100.0,
        };

        if change > self.failure_threshold {
            regressions.push(item);
        } else if change > self.warning_threshold {
            warnings.push(item);
        } else if change < -self.warning_threshold {
            improvements.push(item);
        }

        RegressionReport {
            passed: regressions.is_empty(),
            regressions,
            warnings,
            improvements,
        }
    }
}

// ============================================================================
// BENCH-008: Welch's t-test for Statistical Significance
// Per Hoefler & Belli [17], statistical testing is required for valid comparisons
// ============================================================================

/// Result of Welch's t-test for comparing two sample means
#[derive(Debug, Clone)]
pub struct WelchTTestResult {
    /// Calculated t-statistic
    pub t_statistic: f64,
    /// Welch-Satterthwaite degrees of freedom
    pub degrees_of_freedom: f64,
    /// Two-tailed p-value
    pub p_value: f64,
    /// Whether the difference is statistically significant at given alpha
    pub significant: bool,
}

include!("welch_t_test.rs");
include!("statistics_measurement_protocol.rs");
