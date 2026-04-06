//! Latency Distribution Analysis Module (PMAT-026)
//!
//! Enhanced latency distribution analysis with tail latency detection,
//! jitter calculation, and histogram statistics for identifying performance anomalies.
//!
//! # Components
//!
//! | Component | Formula | Use Case |
//! |-----------|---------|----------|
//! | Jitter (IPDV) | std_dev(\|latency[i] - latency[i-1]\|) | Connection stability |
//! | Tail Ratio | P99/P50 | Tail latency severity |
//! | Bimodality Coefficient | (skewness² + 1) / kurtosis | Distribution shape |
//! | Histogram Entropy | -Σ(p × log(p)) | Distribution uniformity |

mod classification;
mod histogram;
mod moments;

pub use classification::{DistributionShape, TailSeverity};
pub use histogram::{HistogramBucket, LatencyHistogram};

use crate::statistics::percentile;
use moments::{calculate_jitter, calculate_moments};

/// Latency distribution analysis result
#[derive(Debug, Clone)]
pub struct LatencyDistribution {
    /// Median latency (50th percentile)
    pub p50: f64,
    /// 90th percentile latency
    pub p90: f64,
    /// 99th percentile latency
    pub p99: f64,
    /// 99.9th percentile latency
    pub p999: f64,
    /// Jitter (inter-packet delay variation)
    pub jitter: f64,
    /// Tail ratio (P99/P50) - higher means worse tail latency
    pub tail_ratio: f64,
    /// Bimodality coefficient - >0.555 suggests bimodal distribution
    pub bimodality_coefficient: f64,
    /// Latency histogram
    pub histogram: LatencyHistogram,
    /// Number of samples
    pub sample_count: usize,
    /// Minimum latency
    pub min: f64,
    /// Maximum latency
    pub max: f64,
    /// Mean latency
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Skewness (asymmetry)
    pub skewness: f64,
    /// Kurtosis (tail heaviness)
    pub kurtosis: f64,
    /// Outlier ratio (% beyond 3σ)
    pub outlier_ratio: f64,
}

impl LatencyDistribution {
    /// Analyze latency samples to produce distribution statistics
    pub fn analyze(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();

        // Basic statistics
        let mean = samples.iter().sum::<f64>() / n as f64;
        let min = samples.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Standard deviation
        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        // Percentiles
        let p50 = percentile(samples, 0.50);
        let p90 = percentile(samples, 0.90);
        let p99 = percentile(samples, 0.99);
        let p999 = percentile(samples, 0.999);

        // Jitter (inter-sample delay variation)
        let jitter = calculate_jitter(samples);

        // Tail ratio
        let tail_ratio = if p50 > 0.0 { p99 / p50 } else { 1.0 };

        // Higher moments (skewness, kurtosis)
        let (skewness, kurtosis) = calculate_moments(samples, mean, std_dev);

        // Bimodality coefficient: (skewness² + 1) / kurtosis
        let bimodality_coefficient = if kurtosis > 0.0 {
            (skewness.powi(2) + 1.0) / kurtosis
        } else {
            0.0
        };

        // Outlier ratio (beyond 3σ)
        let outlier_count = samples
            .iter()
            .filter(|&&x| (x - mean).abs() > 3.0 * std_dev)
            .count();
        let outlier_ratio = outlier_count as f64 / n as f64 * 100.0;

        // Build histogram
        let histogram = LatencyHistogram::build(samples, 20);

        Some(Self {
            p50,
            p90,
            p99,
            p999,
            jitter,
            tail_ratio,
            bimodality_coefficient,
            histogram,
            sample_count: n,
            min,
            max,
            mean,
            std_dev,
            skewness,
            kurtosis,
            outlier_ratio,
        })
    }

    /// Classify tail latency severity
    pub fn tail_severity(&self) -> TailSeverity {
        TailSeverity::from_ratio(self.tail_ratio)
    }

    /// Classify distribution shape
    pub fn distribution_shape(&self) -> DistributionShape {
        DistributionShape::classify(self.bimodality_coefficient, self.histogram.entropy)
    }

    /// Check if tail latency is problematic (P99/P50 > 3)
    pub fn has_tail_problem(&self) -> bool {
        self.tail_ratio > 3.0
    }

    /// Check if distribution appears bimodal
    pub fn is_bimodal(&self) -> bool {
        self.bimodality_coefficient > 0.555
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "n={} p50={:.2}µs p99={:.2}µs tail_ratio={:.2} jitter={:.2}µs shape={}",
            self.sample_count,
            self.p50,
            self.p99,
            self.tail_ratio,
            self.jitter,
            self.distribution_shape().name()
        )
    }
}

#[cfg(test)]
mod tests;
