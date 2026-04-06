use crate::http_client::tests::part_02::LatencyPercentiles;
use crate::http_client::tests::part_03::ThroughputWithVariance;
use crate::http_client::*;
// =========================================================================
// Per spec QA-034: Use Median Absolute Deviation for robust outlier detection.
// MAD is more resistant to outliers than standard deviation.
// Outlier threshold: |x - median| > k * MAD, typically k = 3.0
// Run with: cargo test test_imp_162 --lib --features bench-http

/// IMP-162a: MAD-based outlier detection
#[derive(Debug, Clone)]
pub struct MadOutlierDetector {
    /// K factor for outlier threshold (default: 3.0)
    pub k_factor: f64,
    /// Consistency constant for normal distribution (1.4826)
    pub consistency_constant: f64,
}

impl MadOutlierDetector {
    pub fn new(k_factor: f64) -> Self {
        Self {
            k_factor,
            // 1.4826 makes MAD consistent with std dev for normal distribution
            consistency_constant: 1.4826,
        }
    }

    /// Default detector with k=3.0 (standard outlier threshold)
    pub fn default_detector() -> Self {
        Self::new(3.0)
    }

    /// Calculate median of samples
    fn median(samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            f64::midpoint(sorted[mid - 1], sorted[mid])
        } else {
            sorted[mid]
        }
    }

    /// Calculate MAD (Median Absolute Deviation)
    pub fn calculate_mad(&self, samples: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let median = Self::median(samples);
        let absolute_deviations: Vec<f64> = samples.iter().map(|x| (x - median).abs()).collect();
        Self::median(&absolute_deviations) * self.consistency_constant
    }

    /// Detect outliers in samples
    pub fn detect_outliers(&self, samples: &[f64]) -> MadOutlierResult {
        if samples.is_empty() {
            return MadOutlierResult {
                median: 0.0,
                mad: 0.0,
                scaled_mad: 0.0,
                threshold: 0.0,
                outlier_indices: Vec::new(),
                outlier_values: Vec::new(),
                clean_samples: Vec::new(),
                outlier_count: 0,
                outlier_percent: 0.0,
            };
        }

        let median = Self::median(samples);
        let mad = self.calculate_mad(samples);
        let scaled_mad = mad; // Already scaled by consistency constant
        let threshold = self.k_factor * scaled_mad;

        let mut outlier_indices = Vec::new();
        let mut outlier_values = Vec::new();
        let mut clean_samples = Vec::new();

        for (i, &value) in samples.iter().enumerate() {
            if (value - median).abs() > threshold {
                outlier_indices.push(i);
                outlier_values.push(value);
            } else {
                clean_samples.push(value);
            }
        }

        let outlier_count = outlier_indices.len();
        let outlier_percent = if !samples.is_empty() {
            (outlier_count as f64 / samples.len() as f64) * 100.0
        } else {
            0.0
        };

        MadOutlierResult {
            median,
            mad,
            scaled_mad,
            threshold,
            outlier_indices,
            outlier_values,
            clean_samples,
            outlier_count,
            outlier_percent,
        }
    }
}

/// IMP-162a: Result of MAD outlier detection
#[derive(Debug, Clone)]
pub struct MadOutlierResult {
    /// Median of samples
    pub median: f64,
    /// MAD (Median Absolute Deviation)
    pub mad: f64,
    /// Scaled MAD (for normal distribution consistency)
    pub scaled_mad: f64,
    /// Outlier threshold (k * MAD)
    pub threshold: f64,
    /// Indices of detected outliers
    pub outlier_indices: Vec<usize>,
    /// Values of detected outliers
    pub outlier_values: Vec<f64>,
    /// Samples with outliers removed
    pub clean_samples: Vec<f64>,
    /// Number of outliers detected
    pub outlier_count: usize,
    /// Percentage of outliers
    pub outlier_percent: f64,
}

impl MadOutlierResult {
    /// Get statistics from clean samples
    pub fn clean_stats(&self) -> ThroughputWithVariance {
        ThroughputWithVariance::from_samples(&self.clean_samples)
    }

    /// Check if outlier filtering was significant
    pub fn filtering_significant(&self) -> bool {
        self.outlier_count > 0 && self.outlier_percent > 1.0
    }
}

/// IMP-162a: Test MAD outlier detection
#[test]
fn test_imp_162a_mad_outlier_detection() {
    // Data with clear outliers
    let samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0, // Normal
        500.0, // High outlier
        10.0,  // Low outlier
    ];

    let detector = MadOutlierDetector::default_detector();
    let result = detector.detect_outliers(&samples);

    // IMP-162a: Median should be ~100
    assert!(
        (result.median - 100.0).abs() < 5.0,
        "IMP-162a: Median should be ~100, got {:.2}",
        result.median
    );

    // IMP-162a: Should detect 2 outliers
    assert_eq!(
        result.outlier_count, 2,
        "IMP-162a: Should detect 2 outliers, got {}",
        result.outlier_count
    );

    // IMP-162a: Outliers should include 500 and 10
    assert!(
        result.outlier_values.contains(&500.0) && result.outlier_values.contains(&10.0),
        "IMP-162a: Outliers should include 500 and 10"
    );

    // IMP-162a: Clean samples should have ~10 values
    assert_eq!(
        result.clean_samples.len(),
        10,
        "IMP-162a: Clean samples should have 10 values"
    );

    println!("\nIMP-162a: MAD Outlier Detection:");
    println!("  Samples: {:?}", samples);
    println!("  Median: {:.2}", result.median);
    println!("  MAD: {:.2}", result.mad);
    println!("  Threshold: {:.2}", result.threshold);
    println!("  Outliers: {:?}", result.outlier_values);
    println!("  Outlier %: {:.2}%", result.outlier_percent);
    println!("  Clean sample count: {}", result.clean_samples.len());
}

/// IMP-162b: MAD vs Standard Deviation comparison
#[derive(Debug, Clone)]
pub struct MadVsStdComparison {
    /// Standard deviation of samples
    pub stddev: f64,
    /// MAD of samples
    pub mad: f64,
    /// Outliers detected by stddev method (k * stddev from mean)
    pub stddev_outliers: usize,
    /// Outliers detected by MAD method
    pub mad_outliers: usize,
    /// Robustness ratio (stddev / MAD) - higher means more outlier influence
    pub robustness_ratio: f64,
}

impl MadVsStdComparison {
    pub fn compare(samples: &[f64], k_factor: f64) -> Self {
        if samples.is_empty() {
            return Self {
                stddev: 0.0,
                mad: 0.0,
                stddev_outliers: 0,
                mad_outliers: 0,
                robustness_ratio: 1.0,
            };
        }

        // Standard deviation method
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let stddev = variance.sqrt();

        // Count outliers by stddev method
        let stddev_threshold = k_factor * stddev;
        let stddev_outliers = samples
            .iter()
            .filter(|&&x| (x - mean).abs() > stddev_threshold)
            .count();

        // MAD method
        let detector = MadOutlierDetector::new(k_factor);
        let mad_result = detector.detect_outliers(samples);

        let robustness_ratio = if mad_result.mad > 0.0 {
            stddev / mad_result.mad
        } else {
            1.0
        };

        Self {
            stddev,
            mad: mad_result.mad,
            stddev_outliers,
            mad_outliers: mad_result.outlier_count,
            robustness_ratio,
        }
    }
}

/// IMP-162b: Test MAD vs stddev comparison
#[test]
fn test_imp_162b_mad_vs_stddev() {
    // Data with extreme outlier - stddev is heavily influenced
    let samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0,
        1000.0, // Extreme outlier
    ];

    let comparison = MadVsStdComparison::compare(&samples, 3.0);

    // IMP-162b: Stddev should be inflated by outlier
    assert!(
        comparison.stddev > 100.0,
        "IMP-162b: Stddev should be heavily inflated, got {:.2}",
        comparison.stddev
    );

    // IMP-162b: MAD should be robust
    assert!(
        comparison.mad < 10.0,
        "IMP-162b: MAD should be robust to outlier, got {:.2}",
        comparison.mad
    );

    // IMP-162b: Robustness ratio should be high (stddev >> MAD)
    assert!(
        comparison.robustness_ratio > 10.0,
        "IMP-162b: Robustness ratio should be high, got {:.2}",
        comparison.robustness_ratio
    );

    // IMP-162b: MAD should detect the outlier
    assert!(
        comparison.mad_outliers >= 1,
        "IMP-162b: MAD should detect outlier"
    );

    println!("\nIMP-162b: MAD vs Standard Deviation:");
    println!("  Stddev: {:.2} (inflated by outlier)", comparison.stddev);
    println!("  MAD: {:.2} (robust)", comparison.mad);
    println!("  Robustness ratio: {:.2}x", comparison.robustness_ratio);
    println!("  Stddev outliers: {}", comparison.stddev_outliers);
    println!("  MAD outliers: {}", comparison.mad_outliers);
}

/// IMP-162c: Benchmark result cleaning with MAD
#[derive(Debug, Clone)]
pub struct CleanedBenchmarkResult {
    /// Raw statistics (before cleaning)
    pub raw_stats: ThroughputWithVariance,
    /// Cleaned statistics (after outlier removal)
    pub cleaned_stats: ThroughputWithVariance,
    /// MAD outlier detection result
    pub outlier_result: MadOutlierResult,
    /// Improvement in CV after cleaning
    pub cv_improvement_percent: f64,
    /// Change in mean after cleaning
    pub mean_change_percent: f64,
}

impl CleanedBenchmarkResult {
    pub fn clean(samples: &[f64]) -> Self {
        let raw_stats = ThroughputWithVariance::from_samples(samples);
        let detector = MadOutlierDetector::default_detector();
        let outlier_result = detector.detect_outliers(samples);
        let cleaned_stats = outlier_result.clean_stats();

        let cv_improvement = if raw_stats.cv > 0.0 {
            ((raw_stats.cv - cleaned_stats.cv) / raw_stats.cv) * 100.0
        } else {
            0.0
        };

        let mean_change = if raw_stats.mean_tps > 0.0 {
            ((cleaned_stats.mean_tps - raw_stats.mean_tps) / raw_stats.mean_tps) * 100.0
        } else {
            0.0
        };

        Self {
            raw_stats,
            cleaned_stats,
            outlier_result,
            cv_improvement_percent: cv_improvement,
            mean_change_percent: mean_change,
        }
    }

    /// Check if cleaning made a significant improvement
    pub fn cleaning_beneficial(&self) -> bool {
        self.cv_improvement_percent > 10.0 && self.outlier_result.outlier_count > 0
    }
}

/// IMP-162c: Test benchmark cleaning with MAD
#[test]
fn test_imp_162c_benchmark_cleaning() {
    // Realistic benchmark data with outliers
    let samples = vec![
        100.0, 102.0, 98.0, 101.0, 99.0, 100.0, 103.0, 97.0, 100.0, 101.0, // Normal
        200.0, // GC pause / thermal throttle
        50.0,  // Cold cache / context switch
    ];

    let result = CleanedBenchmarkResult::clean(&samples);

    // IMP-162c: Cleaned CV should be lower than raw
    assert!(
        result.cleaned_stats.cv < result.raw_stats.cv,
        "IMP-162c: Cleaned CV should be lower"
    );

    // IMP-162c: Should show significant CV improvement
    assert!(
        result.cv_improvement_percent > 50.0,
        "IMP-162c: CV should improve significantly, got {:.2}%",
        result.cv_improvement_percent
    );

    // IMP-162c: Cleaning should be beneficial
    assert!(
        result.cleaning_beneficial(),
        "IMP-162c: Cleaning should be beneficial"
    );

    println!("\nIMP-162c: Benchmark Cleaning with MAD:");
    println!(
        "  Raw mean: {:.2} tok/s (CV={:.4})",
        result.raw_stats.mean_tps, result.raw_stats.cv
    );
    println!(
        "  Cleaned mean: {:.2} tok/s (CV={:.4})",
        result.cleaned_stats.mean_tps, result.cleaned_stats.cv
    );
    println!(
        "  Outliers removed: {}",
        result.outlier_result.outlier_count
    );
    println!("  CV improvement: {:.2}%", result.cv_improvement_percent);
    println!("  Mean change: {:.2}%", result.mean_change_percent);
    println!("  Cleaning beneficial: {}", result.cleaning_beneficial());
}

include!("imp_162d.rs");
include!("imp_163d.rs");
include!("imp_164c.rs");
