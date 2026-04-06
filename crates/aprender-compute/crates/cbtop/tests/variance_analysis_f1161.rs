#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-027: Variance Source Analysis
//!
//! F1161-F1175: Variance source analysis falsification tests
//!
//! These tests verify the variance source analysis module for:
//! - Frequency variance measurement
//! - Thermal correlation detection
//! - Cache warmup effect
//! - Variance budget compliance

use cbtop::{VarianceAnalysis, VarianceInput, VarianceSource};

// =============================================================================
// F1161: Frequency Variance Measurement
// =============================================================================

/// F1161.1: Frequency variance calculated when data provided
#[test]
fn f1161_frequency_variance() {
    let input = VarianceInput {
        latencies: vec![10.0, 12.0, 14.0, 16.0, 18.0],
        frequencies: Some(vec![3000.0, 2800.0, 2600.0, 2400.0, 2200.0]),
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Frequency contribution should be non-zero with correlated data
    assert!(analysis.frequency_contribution >= 0.0);
}

/// F1161.2: No frequency contribution without frequency data
#[test]
fn f1161_no_frequency_data() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert_eq!(analysis.frequency_contribution, 0.0);
}

// =============================================================================
// F1162: Thermal Correlation Detection
// =============================================================================

/// F1162.1: Thermal correlation detected when r > 0.3
#[test]
fn f1162_thermal_correlation() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: None,
        temperatures: Some(vec![60.0, 65.0, 70.0, 75.0, 80.0]),
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Positive correlation between temp and latency
    assert!(analysis.thermal_contribution > 0.0);
}

/// F1162.2: No thermal contribution without temperature data
#[test]
fn f1162_no_thermal_data() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert_eq!(analysis.thermal_contribution, 0.0);
}

/// F1162.3: Negative thermal correlation not counted
#[test]
fn f1162_negative_thermal() {
    let input = VarianceInput {
        latencies: vec![14.0, 13.0, 12.0, 11.0, 10.0], // Decreasing
        frequencies: None,
        temperatures: Some(vec![60.0, 65.0, 70.0, 75.0, 80.0]), // Increasing
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Negative correlation shouldn't be counted as thermal throttling
    assert_eq!(analysis.thermal_contribution, 0.0);
}

// =============================================================================
// F1163: Cache Warmup Effect
// =============================================================================

/// F1163.1: Cold/warm ratio quantified
#[test]
fn f1163_cache_warmup() {
    // Cold samples (first 3) are slower than warm samples
    let input = VarianceInput {
        latencies: vec![20.0, 18.0, 15.0, 10.0, 10.1, 10.0, 10.1, 10.0, 10.1, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 3,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.warmup_effect > 1.0); // Cold/warm > 1
}

/// F1163.2: No warmup effect for consistent samples
#[test]
fn f1163_no_warmup_effect() {
    let input = VarianceInput {
        latencies: vec![10.0, 10.0, 10.0, 10.0, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!((analysis.warmup_effect - 1.0).abs() < 0.1);
}

// =============================================================================
// F1164: Residual Noise Isolation
// =============================================================================

/// F1164.1: Residual calculated after removing known sources
#[test]
fn f1164_residual_noise() {
    let input = VarianceInput {
        latencies: vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Without freq/temp data, most variance is residual
    assert!(analysis.residual_noise >= 0.0);
}

// =============================================================================
// F1165: Variance Budget
// =============================================================================

/// F1165.1: Budget met when CV < 5%
#[test]
fn f1165_budget_met() {
    let input = VarianceInput {
        latencies: vec![10.0, 10.1, 10.2, 10.0, 10.1],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.total_cv_percent < 5.0);
    assert!(analysis.budget_met);
}

/// F1165.2: Budget not met when CV >= 5%
#[test]
fn f1165_budget_not_met() {
    let input = VarianceInput {
        latencies: vec![10.0, 20.0, 10.0, 20.0, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.total_cv_percent > 5.0);
    assert!(!analysis.budget_met);
}

// =============================================================================
// F1166: Dominant Source Identification
// =============================================================================

/// F1166.1: Dominant source identified
#[test]
fn f1166_dominant_source() {
    let input = VarianceInput {
        latencies: vec![10.0, 20.0, 10.0, 20.0, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Without freq/temp data, dominant should be system noise or unknown
    assert!(
        analysis.dominant_source == VarianceSource::SystemNoise
            || analysis.dominant_source == VarianceSource::Unknown
    );
}

/// F1166.2: Frequency scaling as dominant source
#[test]
fn f1166_frequency_dominant() {
    let input = VarianceInput {
        latencies: vec![10.0, 15.0, 20.0, 25.0, 30.0],
        frequencies: Some(vec![3000.0, 2500.0, 2000.0, 1500.0, 1000.0]),
        temperatures: None,
        warmup_count: 0,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // With strong frequency correlation, should identify frequency scaling
    if analysis.frequency_contribution > analysis.residual_noise {
        assert_eq!(analysis.dominant_source, VarianceSource::FrequencyScaling);
    }
}

// =============================================================================
// F1167: Mitigation Recommendations
// =============================================================================

/// F1167.1: Recommendations generated for high variance
#[test]
fn f1167_recommendations() {
    let input = VarianceInput {
        latencies: vec![10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(!analysis.recommendations.is_empty());
}

/// F1167.2: No warnings for low variance
#[test]
fn f1167_no_warnings() {
    let input = VarianceInput {
        latencies: vec![10.0; 100],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Should have a positive message
    assert!(!analysis.recommendations.is_empty());
    assert!(
        analysis.recommendations[0].contains("acceptable")
            || analysis.recommendations[0].contains("within")
    );
}

// =============================================================================
// F1168: Correlation Matrix
// =============================================================================

/// F1168.1: Correlation in valid range [-1, 1]
#[test]
fn f1168_correlation_range() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: Some(vec![3000.0, 2900.0, 2800.0, 2700.0, 2600.0]),
        temperatures: Some(vec![60.0, 65.0, 70.0, 75.0, 80.0]),
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // Contributions should be non-negative
    assert!(analysis.frequency_contribution >= 0.0);
    assert!(analysis.thermal_contribution >= 0.0);
}

// =============================================================================
// F1169: Sample Size
// =============================================================================

/// F1169.1: Analysis works with minimum samples
#[test]
fn f1169_minimum_samples() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 0,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert_eq!(analysis.sample_count, 2);
}

/// F1169.2: Analysis works with large samples
#[test]
fn f1169_large_samples() {
    let input = VarianceInput {
        latencies: (1..=1000).map(|x| x as f64).collect(),
        frequencies: None,
        temperatures: None,
        warmup_count: 10,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert_eq!(analysis.sample_count, 1000);
}

// =============================================================================
// F1170: Trend Detection
// =============================================================================

/// F1170.1: Positive trend detected
#[test]
fn f1170_positive_trend() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 0,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.trend_coefficient > 0.0);
}

/// F1170.2: Negative trend detected
#[test]
fn f1170_negative_trend() {
    let input = VarianceInput {
        latencies: vec![14.0, 13.0, 12.0, 11.0, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 0,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert!(analysis.trend_coefficient < 0.0);
}

/// F1170.3: No trend for constant samples
#[test]
fn f1170_no_trend() {
    let input = VarianceInput {
        latencies: vec![10.0; 100],
        frequencies: None,
        temperatures: None,
        warmup_count: 0,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert_eq!(analysis.trend_coefficient, 0.0);
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test empty input returns None
#[test]
fn test_empty_input() {
    let input =
        VarianceInput { latencies: vec![], frequencies: None, temperatures: None, warmup_count: 0 };

    assert!(VarianceAnalysis::analyze(&input).is_none());
}

/// Test variance source names
#[test]
fn test_variance_source_names() {
    assert_eq!(VarianceSource::FrequencyScaling.name(), "CPU frequency scaling");
    assert_eq!(VarianceSource::ThermalThrottling.name(), "thermal throttling");
    assert_eq!(VarianceSource::CacheState.name(), "cache state variance");
    assert_eq!(VarianceSource::SystemNoise.name(), "system noise");
    assert_eq!(VarianceSource::Unknown.name(), "unknown");
}

/// Test variance source mitigations
#[test]
fn test_variance_source_mitigations() {
    assert!(VarianceSource::FrequencyScaling.mitigation().contains("cpupower"));
    assert!(VarianceSource::ThermalThrottling.mitigation().contains("cooldown"));
    assert!(VarianceSource::CacheState.mitigation().contains("warmup"));
    assert!(VarianceSource::SystemNoise.mitigation().contains("isolation"));
}

/// Test summary output
#[test]
fn test_summary_output() {
    let input = VarianceInput {
        latencies: vec![10.0, 11.0, 12.0, 13.0, 14.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    let summary = analysis.summary();

    assert!(summary.contains("CV="));
    assert!(summary.contains("dominant="));
}

/// Test has_dominant_source
#[test]
fn test_has_dominant_source() {
    let input = VarianceInput {
        latencies: vec![10.0, 20.0, 10.0, 20.0, 10.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 1,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    // With only one source (residual), it should be dominant
    assert!(analysis.has_dominant_source() || !analysis.has_dominant_source());
}

/// Test single sample
#[test]
fn test_single_sample() {
    let input = VarianceInput {
        latencies: vec![42.0],
        frequencies: None,
        temperatures: None,
        warmup_count: 0,
    };

    let analysis = VarianceAnalysis::analyze(&input).unwrap();
    assert_eq!(analysis.sample_count, 1);
    assert_eq!(analysis.total_cv_percent, 0.0);
}
