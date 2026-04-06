#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-026: Latency Distribution Analysis
//!
//! F1141-F1155: Latency distribution falsification tests
//!
//! These tests verify the latency distribution analysis module for:
//! - Jitter calculation
//! - Tail ratio detection
//! - Bimodality detection
//! - Histogram entropy

use cbtop::{DistributionShape, LatencyDistribution, LatencyHistogram, TailSeverity};

// =============================================================================
// F1141: Jitter Calculation Tests
// =============================================================================

/// F1141.1: Jitter is zero for constant latency
#[test]
fn f1141_jitter_constant() {
    let samples = vec![10.0; 100];
    let dist = LatencyDistribution::analyze(&samples).unwrap();
    assert_eq!(dist.jitter, 0.0);
}

/// F1141.2: Jitter increases with variance
#[test]
fn f1141_jitter_increases() {
    let stable = vec![10.0, 10.0, 10.0, 10.0, 10.0];
    let unstable = vec![10.0, 50.0, 10.0, 50.0, 10.0, 50.0];

    let dist_stable = LatencyDistribution::analyze(&stable).unwrap();
    let dist_unstable = LatencyDistribution::analyze(&unstable).unwrap();

    // Unstable should have higher jitter (inter-sample variation)
    assert!(dist_unstable.jitter >= dist_stable.jitter);
}

/// F1141.3: Single sample has zero jitter
#[test]
fn f1141_single_sample_jitter() {
    let samples = vec![42.0];
    let dist = LatencyDistribution::analyze(&samples).unwrap();
    assert_eq!(dist.jitter, 0.0);
}

// =============================================================================
// F1142: Tail Ratio Tests
// =============================================================================

/// F1142.1: Tail ratio > 3 flagged as problem
#[test]
fn f1142_tail_ratio_problem() {
    // Create samples with heavy tail
    let mut samples: Vec<f64> = (1..=90).map(|x| x as f64).collect();
    samples.extend(vec![500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0]);

    let dist = LatencyDistribution::analyze(&samples).unwrap();
    assert!(dist.tail_ratio > 3.0);
    assert!(dist.has_tail_problem());
}

/// F1142.2: Normal distribution has low tail ratio
#[test]
fn f1142_normal_tail_ratio() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    // Uniform distribution should have tail ratio < 3
    assert!(dist.tail_ratio < 3.0);
    assert!(!dist.has_tail_problem());
}

/// F1142.3: Tail severity classification
#[test]
fn f1142_tail_severity() {
    assert_eq!(TailSeverity::from_ratio(1.5), TailSeverity::Excellent);
    assert_eq!(TailSeverity::from_ratio(2.5), TailSeverity::Good);
    assert_eq!(TailSeverity::from_ratio(4.0), TailSeverity::Warning);
    assert_eq!(TailSeverity::from_ratio(10.0), TailSeverity::Critical);
}

// =============================================================================
// F1143: Bimodality Detection Tests
// =============================================================================

/// F1143.1: Bimodal data detected
#[test]
fn f1143_bimodal_detection() {
    // Create bimodal distribution (two distinct clusters)
    let mut samples: Vec<f64> = vec![10.0; 50];
    samples.extend(vec![100.0; 50]);

    let dist = LatencyDistribution::analyze(&samples).unwrap();

    // Bimodality coefficient should be high
    // Note: exact threshold depends on implementation
    assert!(dist.bimodality_coefficient > 0.3);
}

/// F1143.2: Unimodal data has lower bimodality coefficient
#[test]
fn f1143_unimodal_lower_bc() {
    // Unimodal: all values close together
    let unimodal: Vec<f64> = vec![50.0; 100];

    // Bimodal: two distinct clusters
    let mut bimodal: Vec<f64> = vec![10.0; 50];
    bimodal.extend(vec![100.0; 50]);

    let dist_uni = LatencyDistribution::analyze(&unimodal).unwrap();
    let dist_bi = LatencyDistribution::analyze(&bimodal).unwrap();

    // Bimodal should have higher bimodality coefficient
    assert!(dist_bi.bimodality_coefficient >= dist_uni.bimodality_coefficient);
}

// =============================================================================
// F1144: Histogram Entropy Tests
// =============================================================================

/// F1144.1: Entropy is normalized 0-1
#[test]
fn f1144_entropy_range() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 10);

    assert!(hist.entropy >= 0.0);
    assert!(hist.entropy <= 1.0);
}

/// F1144.2: Concentrated data has low entropy
#[test]
fn f1144_concentrated_entropy() {
    let samples = vec![50.0; 100];
    let hist = LatencyHistogram::build(&samples, 10);

    // All samples in one bucket = low entropy
    assert!(hist.entropy < 0.1);
}

/// F1144.3: Uniform data has high entropy
#[test]
fn f1144_uniform_entropy() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 10);

    // Uniform distribution should have high entropy
    assert!(hist.entropy > 0.9);
}

// =============================================================================
// F1145: Empty Input Handling
// =============================================================================

/// F1145.1: Empty samples returns None
#[test]
fn f1145_empty_input() {
    let samples: Vec<f64> = vec![];
    assert!(LatencyDistribution::analyze(&samples).is_none());
}

/// F1145.2: Empty histogram
#[test]
fn f1145_empty_histogram() {
    let samples: Vec<f64> = vec![];
    let hist = LatencyHistogram::build(&samples, 10);

    assert_eq!(hist.total_samples, 0);
    assert_eq!(hist.entropy, 0.0);
}

// =============================================================================
// F1146: Single Element Handling
// =============================================================================

/// F1146.1: Single sample has jitter = 0
#[test]
fn f1146_single_element() {
    let samples = vec![42.0];
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert_eq!(dist.jitter, 0.0);
    assert_eq!(dist.sample_count, 1);
    assert_eq!(dist.p50, 42.0);
}

// =============================================================================
// F1147: Percentile Ordering
// =============================================================================

/// F1147.1: P50 <= P90 <= P99 <= P999
#[test]
fn f1147_percentile_ordering() {
    let samples: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert!(dist.p50 <= dist.p90);
    assert!(dist.p90 <= dist.p99);
    assert!(dist.p99 <= dist.p999);
}

/// F1147.2: Large dataset maintains ordering
#[test]
fn f1147_large_dataset_ordering() {
    // Large dataset should maintain percentile ordering
    let samples: Vec<f64> = (1..=10000).map(|x| x as f64).collect();

    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert!(dist.p50 <= dist.p90);
    assert!(dist.p90 <= dist.p99);
    assert!(dist.p99 <= dist.p999);
}

// =============================================================================
// F1148: Bucket Counts Sum
// =============================================================================

/// F1148.1: Bucket counts sum to total samples
#[test]
fn f1148_bucket_sum() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 10);

    assert!(hist.verify_counts());
    assert_eq!(hist.total_samples, 100);
}

/// F1148.2: No samples lost with many buckets
#[test]
fn f1148_many_buckets() {
    let samples: Vec<f64> = (1..=1000).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 100);

    assert!(hist.verify_counts());
}

// =============================================================================
// F1149: Mode Detection
// =============================================================================

/// F1149.1: Mode bucket identified correctly
#[test]
fn f1149_mode_detection() {
    // Create samples concentrated in middle
    let mut samples: Vec<f64> = vec![50.0; 80];
    samples.extend((1..=20).map(|x| x as f64));

    let hist = LatencyHistogram::build(&samples, 10);

    // Mode bucket should contain the 50.0 values
    let mode = hist.mode().unwrap();
    assert!(mode.count >= 80);
}

/// F1149.2: Mode is most frequent bucket
#[test]
fn f1149_mode_most_frequent() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 10);

    let mode = hist.mode().unwrap();
    for bucket in &hist.buckets {
        assert!(bucket.count <= mode.count);
    }
}

// =============================================================================
// F1150: Outlier Ratio
// =============================================================================

/// F1150.1: Outlier ratio calculated correctly
#[test]
fn f1150_outlier_ratio() {
    // Normal samples with a few outliers
    let mut samples: Vec<f64> = vec![10.0; 97];
    samples.extend(vec![1000.0, 2000.0, 3000.0]); // 3 outliers

    let dist = LatencyDistribution::analyze(&samples).unwrap();

    // Should detect ~3% outliers
    assert!(dist.outlier_ratio >= 2.0);
    assert!(dist.outlier_ratio <= 5.0);
}

/// F1150.2: No outliers in tight distribution
#[test]
fn f1150_no_outliers() {
    let samples: Vec<f64> = vec![10.0; 100];
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert_eq!(dist.outlier_ratio, 0.0);
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test distribution shape classification
#[test]
fn test_distribution_shape() {
    assert_eq!(DistributionShape::classify(0.3, 0.8).name(), "unimodal");
    assert_eq!(DistributionShape::classify(0.6, 0.8).name(), "bimodal");
    assert_eq!(DistributionShape::classify(0.3, 0.98).name(), "uniform");
}

/// Test tail severity names
#[test]
fn test_tail_severity_names() {
    assert_eq!(TailSeverity::Excellent.name(), "excellent");
    assert_eq!(TailSeverity::Good.name(), "good");
    assert_eq!(TailSeverity::Warning.name(), "warning");
    assert_eq!(TailSeverity::Critical.name(), "critical");
}

/// Test summary output
#[test]
fn test_summary_output() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    let summary = dist.summary();
    assert!(summary.contains("n=100"));
    assert!(summary.contains("p50="));
    assert!(summary.contains("p99="));
}

/// Test min/max tracking
#[test]
fn test_min_max() {
    let samples: Vec<f64> = (10..=100).map(|x| x as f64).collect();
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    assert_eq!(dist.min, 10.0);
    assert_eq!(dist.max, 100.0);
}

/// Test standard deviation
#[test]
fn test_std_dev() {
    // Known case: samples with std_dev = 0
    let samples = vec![42.0; 10];
    let dist = LatencyDistribution::analyze(&samples).unwrap();
    assert_eq!(dist.std_dev, 0.0);

    // Samples with variance
    let samples2: Vec<f64> = (1..=10).map(|x| x as f64).collect();
    let dist2 = LatencyDistribution::analyze(&samples2).unwrap();
    assert!(dist2.std_dev > 0.0);
}

/// Test skewness for symmetric distribution
#[test]
fn test_skewness_symmetric() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let dist = LatencyDistribution::analyze(&samples).unwrap();

    // Uniform distribution is symmetric, skewness near 0
    assert!(dist.skewness.abs() < 0.5);
}

/// Test histogram bucket percentages
#[test]
fn test_bucket_percentages() {
    let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
    let hist = LatencyHistogram::build(&samples, 10);

    // Percentages should sum to ~100%
    let total_percent: f64 = hist.buckets.iter().map(|b| b.percentage).sum();
    assert!((total_percent - 100.0).abs() < 0.1);
}
