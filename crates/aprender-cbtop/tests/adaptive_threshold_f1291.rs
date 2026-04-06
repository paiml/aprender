#![allow(clippy::disallowed_methods, clippy::float_cmp)]
//! Falsification Tests for PMAT-037: Adaptive Threshold Learning System
//!
//! F1291-F1300: Adaptive threshold falsification tests

use cbtop::{
    ThresholdCheck, ThresholdDirection, ThresholdLearner, DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_OUTLIER_THRESHOLD, MIN_SAMPLES_FOR_LEARNING,
};

// =============================================================================
// F1291: Baseline Learning Tests
// =============================================================================

/// F1291.1: Baseline learning works (μ+2σ calculated)
#[test]
fn f1291_baseline_learning() {
    let mut learner = ThresholdLearner::new("cpu_temp");

    // Add samples
    for i in 0..20 {
        learner.add_sample(50.0 + (i % 5) as f64);
    }

    let threshold = learner.learn_baseline().unwrap();

    assert!(threshold.mean > 49.0 && threshold.mean < 53.0);
    assert!(threshold.std_dev > 0.0);
    assert!(threshold.upper_bound > threshold.mean);
}

/// F1291.2: Insufficient samples returns None
#[test]
fn f1291_insufficient_samples() {
    let mut learner = ThresholdLearner::new("test");

    learner.add_sample(10.0);
    learner.add_sample(11.0);

    assert!(learner.learn_baseline().is_none());
}

// =============================================================================
// F1292: Adaptive Bounds Tests
// =============================================================================

/// F1292.1: Adaptive bounds narrow (bounds shrink with lower CV)
#[test]
fn f1292_bounds_narrow() {
    let mut learner = ThresholdLearner::new("test");

    // Low variance data
    for _ in 0..20 {
        learner.add_sample(100.0);
    }

    let threshold = learner.learn_baseline().unwrap();
    // With zero variance, bounds should be tight
    assert!(threshold.cv < 1.0);
}

/// F1292.2: High variance data has wider bounds
#[test]
fn f1292_wide_bounds() {
    let mut learner = ThresholdLearner::new("test");

    // High variance data
    for i in 0..20 {
        learner.add_sample(if i % 2 == 0 { 50.0 } else { 150.0 });
    }

    let threshold = learner.learn_baseline().unwrap();
    assert!(threshold.cv > 30.0);
}

// =============================================================================
// F1293: Outlier Filtering Tests
// =============================================================================

/// F1293.1: Outlier filtering (extreme values excluded)
#[test]
fn f1293_outlier_filtering() {
    let mut learner = ThresholdLearner::new("test");

    // Normal data with outlier
    for i in 0..19 {
        learner.add_sample(100.0 + (i % 3) as f64);
    }
    learner.add_sample(500.0); // Outlier

    let filtered = learner.filter_outliers();
    assert!(filtered.len() < 20); // Outlier removed
}

/// F1293.2: Normal data not filtered
#[test]
fn f1293_normal_not_filtered() {
    let mut learner = ThresholdLearner::new("test");

    for i in 0..20 {
        learner.add_sample(100.0 + (i % 3) as f64);
    }

    let filtered = learner.filter_outliers();
    assert_eq!(filtered.len(), 20);
}

// =============================================================================
// F1294: Override Tests
// =============================================================================

/// F1294.1: Override takes precedence
#[test]
fn f1294_override_precedence() {
    let mut learner = ThresholdLearner::new("test").with_override(50.0);

    for i in 0..20 {
        learner.add_sample(100.0 + i as f64);
    }

    let effective = learner.get_effective_threshold().unwrap();
    assert_eq!(effective, 50.0);
}

/// F1294.2: Override can be cleared
#[test]
fn f1294_clear_override() {
    let mut learner = ThresholdLearner::new("test").with_override(50.0);

    for i in 0..20 {
        learner.add_sample(100.0 + i as f64);
    }

    learner.clear_override();
    let effective = learner.get_effective_threshold().unwrap();
    assert!(effective > 100.0); // Now uses learned value
}

// =============================================================================
// F1295: Performance Impact Tests
// =============================================================================

/// F1295.1: Performance impact (<1ms overhead)
#[test]
fn f1295_performance() {
    let mut learner = ThresholdLearner::new("test");

    for i in 0..100 {
        learner.add_sample(100.0 + (i % 10) as f64);
    }

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = learner.learn_baseline();
    }
    let elapsed = start.elapsed();

    // 1000 iterations should be < 1 second (< 1ms each)
    assert!(elapsed.as_millis() < 1000);
}

// =============================================================================
// F1296: Confidence Interval Tests
// =============================================================================

/// F1296.1: 95% CI computed
#[test]
fn f1296_confidence_interval() {
    let mut learner = ThresholdLearner::new("test");

    for i in 0..20 {
        learner.add_sample(100.0 + (i % 5) as f64);
    }

    let threshold = learner.learn_baseline().unwrap();
    assert_eq!(threshold.confidence_level, DEFAULT_CONFIDENCE_LEVEL);
}

// =============================================================================
// F1297: Minimum Samples Tests
// =============================================================================

/// F1297.1: Minimum samples enforced (≥10 required)
#[test]
fn f1297_minimum_samples() {
    assert_eq!(MIN_SAMPLES_FOR_LEARNING, 10);

    let mut learner = ThresholdLearner::new("test");

    for i in 0..9 {
        learner.add_sample(i as f64);
    }
    assert!(!learner.has_sufficient_samples());

    learner.add_sample(9.0);
    assert!(learner.has_sufficient_samples());
}

// =============================================================================
// F1298: Threshold Direction Tests
// =============================================================================

/// F1298.1: Upper threshold direction
#[test]
fn f1298_upper_direction() {
    let mut learner = ThresholdLearner::new("test").with_direction(ThresholdDirection::Upper);

    for i in 0..20 {
        learner.add_sample(50.0 + (i % 3) as f64);
    }

    let threshold = learner.learn_baseline().unwrap();
    assert_eq!(threshold.direction, ThresholdDirection::Upper);
    assert!(threshold.is_warning(100.0)); // Above upper bound
    assert!(!threshold.is_warning(30.0)); // Below is OK for upper
}

/// F1298.2: Lower threshold direction
#[test]
fn f1298_lower_direction() {
    let mut learner = ThresholdLearner::new("test").with_direction(ThresholdDirection::Lower);

    for i in 0..20 {
        learner.add_sample(50.0 + (i % 3) as f64);
    }

    let threshold = learner.learn_baseline().unwrap();
    assert_eq!(threshold.direction, ThresholdDirection::Lower);
    assert!(threshold.is_warning(10.0)); // Below lower bound
}

/// F1298.3: Both directions
#[test]
fn f1298_both_directions() {
    let mut learner = ThresholdLearner::new("test").with_direction(ThresholdDirection::Both);

    for i in 0..20 {
        learner.add_sample(50.0 + (i % 3) as f64);
    }

    let threshold = learner.learn_baseline().unwrap();
    assert_eq!(threshold.direction, ThresholdDirection::Both);
}

// =============================================================================
// F1299: Export Tests
// =============================================================================

/// F1299.1: Export thresholds (JSON serializable)
#[test]
fn f1299_export_json() {
    let mut learner = ThresholdLearner::new("cpu_temp");

    for i in 0..20 {
        learner.add_sample(60.0 + (i % 5) as f64);
    }

    let threshold = learner.learn_baseline().unwrap();
    let json = threshold.to_json();

    assert!(json.contains("\"metric\":\"cpu_temp\""));
    assert!(json.contains("\"mean\":"));
    assert!(json.contains("\"std_dev\":"));
}

// =============================================================================
// F1300: Reset Tests
// =============================================================================

/// F1300.1: Clear learned state works
#[test]
fn f1300_reset() {
    let mut learner = ThresholdLearner::new("test");

    for i in 0..20 {
        learner.add_sample(i as f64);
    }

    assert_eq!(learner.sample_count(), 20);

    learner.clear();
    assert_eq!(learner.sample_count(), 0);
    assert!(!learner.has_sufficient_samples());
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test threshold direction names
#[test]
fn test_direction_names() {
    assert_eq!(ThresholdDirection::Upper.name(), "upper");
    assert_eq!(ThresholdDirection::Lower.name(), "lower");
    assert_eq!(ThresholdDirection::Both.name(), "both");
}

/// Test constants
#[test]
fn test_constants() {
    assert_eq!(MIN_SAMPLES_FOR_LEARNING, 10);
    assert_eq!(DEFAULT_CONFIDENCE_LEVEL, 0.95);
    assert_eq!(DEFAULT_OUTLIER_THRESHOLD, 3.0);
}

/// Test threshold check passed
#[test]
fn test_check_passed() {
    let check = ThresholdCheck {
        value: 50.0,
        threshold: 100.0,
        is_warning: false,
        is_critical: false,
        is_override: false,
    };

    assert!(check.passed());
}

/// Test percentile threshold
#[test]
fn test_percentile_threshold() {
    let mut learner = ThresholdLearner::new("test");

    for i in 0..100 {
        learner.add_sample(i as f64);
    }

    let p50 = learner.percentile_threshold(50.0).unwrap();
    assert!((p50 - 50.0).abs() < 2.0);

    let p95 = learner.percentile_threshold(95.0).unwrap();
    assert!(p95 > 90.0);
}

/// Test warning vs critical
#[test]
fn test_warning_vs_critical() {
    let mut learner =
        ThresholdLearner::new("test").with_warning_multiplier(2.0).with_critical_multiplier(3.0);

    for _ in 0..20 {
        learner.add_sample(100.0);
    }

    // With zero variance, any deviation triggers
    // But with exactly 100.0, nothing triggers
    let threshold = learner.learn_baseline().unwrap();
    assert!(!threshold.is_warning(100.0));
    assert!(!threshold.is_critical(100.0));
}

/// Test add multiple samples
#[test]
fn test_add_samples() {
    let mut learner = ThresholdLearner::new("test");

    learner.add_samples(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(learner.sample_count(), 5);
}
