//! Core latency distribution analysis logic.

use crate::statistics::percentile;

use super::histogram::LatencyHistogram;
use super::moments;
use super::{min_max, LatencyDistribution};

impl LatencyDistribution {
    /// Analyze latency samples to produce distribution statistics
    pub fn analyze(samples: &[f64]) -> Option<Self> {
        if samples.is_empty() {
            return None;
        }

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let (min, max) = min_max(samples);

        let variance = if n > 1 {
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();

        let p50 = percentile(samples, 0.50);
        let p90 = percentile(samples, 0.90);
        let p99 = percentile(samples, 0.99);
        let p999 = percentile(samples, 0.999);

        let jitter = moments::calculate_jitter(samples);
        let tail_ratio = if p50 > 0.0 { p99 / p50 } else { 1.0 };
        let (skewness, kurtosis) = moments::calculate_moments(samples, mean, std_dev);

        let bimodality_coefficient = if kurtosis > 0.0 {
            (skewness.powi(2) + 1.0) / kurtosis
        } else {
            0.0
        };

        let outlier_count = samples
            .iter()
            .filter(|&&x| (x - mean).abs() > 3.0 * std_dev)
            .count();
        let outlier_ratio = outlier_count as f64 / n as f64 * 100.0;

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
}
