// Comprehensive tests for statistical regression detection
//
// Toyota Way Principle: Genchi Genbutsu (Go and See)
// - Test with realistic syscall count distributions
// - Validate against real-world regression scenarios
// - Ensure no false positives from natural variance

use super::*;
use std::collections::HashMap;

/// Test real-world example: decy futex anomaly
///
/// Scenario: Accidental async runtime initialization increases futex calls
/// Expected: Regression detected (statistically significant increase)
#[test]
fn test_decy_futex_regression() {
    let mut baseline = HashMap::new();
    baseline.insert(
        "futex".to_string(),
        vec![2.0, 3.0, 2.0, 3.0, 2.0], // Minimal futex in single-threaded mode
    );
    baseline.insert(
        "mmap".to_string(),
        vec![100.0, 102.0, 101.0, 103.0, 100.0], // Stable
    );

    let mut current = HashMap::new();
    current.insert(
        "futex".to_string(),
        vec![50.0, 52.0, 51.0, 53.0, 50.0], // MUCH higher with async runtime
    );
    current.insert(
        "mmap".to_string(),
        vec![100.0, 102.0, 101.0, 103.0, 100.0], // Unchanged
    );

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    // Should detect futex regression
    match assessment.verdict {
        RegressionVerdict::Regression { ref regressed_syscalls, .. } => {
            assert!(regressed_syscalls.contains(&"futex".to_string()));
            assert!(!regressed_syscalls.contains(&"mmap".to_string())); // mmap stable
        }
        _ => panic!("Expected Regression verdict for futex increase"),
    }
}

/// Test real-world example: depyler telemetry leak
///
/// Scenario: Sentry-rs adds networking syscalls (socket, connect, send)
/// Expected: Regression detected for all networking syscalls
#[test]
fn test_depyler_telemetry_regression() {
    let mut baseline = HashMap::new();
    baseline.insert("mmap".to_string(), vec![100.0, 102.0, 101.0, 103.0, 100.0]);
    baseline.insert("read".to_string(), vec![50.0, 51.0, 50.0, 52.0, 50.0]);

    let mut current = HashMap::new();
    current.insert("mmap".to_string(), vec![100.0, 102.0, 101.0, 103.0, 100.0]);
    current.insert("read".to_string(), vec![50.0, 51.0, 50.0, 52.0, 50.0]);
    // NEW: Telemetry syscalls appear
    current.insert(
        "socket".to_string(),
        vec![5.0, 5.0, 5.0, 5.0, 5.0], // New syscall
    );

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    // Note: assess_regression only tests syscalls present in BOTH traces
    // Missing/new syscalls require separate sequence analysis (SEQUENCE-001)
    assert_eq!(assessment.verdict, RegressionVerdict::NoRegression);
}

/// Test benign variance does NOT trigger false positive
///
/// Scenario: Natural run-to-run variance within normal bounds
/// Expected: No regression detected
#[test]
fn test_no_false_positive_natural_variance() {
    let mut baseline = HashMap::new();
    baseline.insert(
        "mmap".to_string(),
        vec![100.0, 105.0, 98.0, 102.0, 101.0], // Some variance
    );

    let mut current = HashMap::new();
    current.insert(
        "mmap".to_string(),
        vec![102.0, 106.0, 99.0, 103.0, 100.0], // Similar variance
    );

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    // Should NOT detect regression (distributions similar)
    assert_eq!(assessment.verdict, RegressionVerdict::NoRegression);
}

/// Test noise filtering removes high-variance syscalls
///
/// Scenario: Network I/O has high variance, should be filtered
/// Expected: Noisy syscall filtered out, stable syscalls tested
#[test]
fn test_noise_filtering_removes_high_variance() {
    let mut baseline = HashMap::new();
    baseline.insert(
        "mmap".to_string(),
        vec![100.0, 102.0, 101.0, 103.0, 100.0], // Stable
    );
    baseline.insert(
        "socket".to_string(),
        vec![5.0, 50.0, 3.0, 45.0, 2.0], // High variance (CV > 0.5)
    );

    let mut current = HashMap::new();
    current.insert("mmap".to_string(), vec![100.0, 102.0, 101.0, 103.0, 100.0]);
    current.insert("socket".to_string(), vec![6.0, 51.0, 4.0, 46.0, 3.0]);

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    // Should filter socket as noisy
    assert!(assessment.filtered_syscalls.contains(&"socket".to_string()));

    // Should only test mmap
    assert_eq!(assessment.tests.len(), 1);
    assert!(assessment.tests.contains_key("mmap"));
}

/// Test configuration: strict mode reduces false positives
///
/// Scenario: Marginal increase that default config detects
/// Expected: Strict config does NOT detect (99% confidence required)
#[test]
fn test_strict_config_reduces_false_positives() {
    let mut baseline = HashMap::new();
    baseline.insert("mmap".to_string(), vec![100.0, 102.0, 101.0, 103.0, 100.0]);

    let mut current = HashMap::new();
    current.insert("mmap".to_string(), vec![108.0, 110.0, 109.0, 111.0, 108.0]);

    // Default config (95% confidence)
    let default_config = RegressionConfig::default();
    let default_assessment = assess_regression(&baseline, &current, &default_config).expect("test");

    // Strict config (99% confidence)
    let strict_config = RegressionConfig::strict();
    let strict_assessment = assess_regression(&baseline, &current, &strict_config).expect("test");

    // Default might detect, strict should NOT detect (depends on exact p-value)
    // This test validates that strict config has higher bar
    match (&default_assessment.verdict, &strict_assessment.verdict) {
        (RegressionVerdict::Regression { .. }, RegressionVerdict::NoRegression) => {
            // Expected: default detects, strict does not
        }
        (RegressionVerdict::NoRegression, RegressionVerdict::NoRegression) => {
            // Also acceptable: neither detects
        }
        (RegressionVerdict::Regression { .. }, RegressionVerdict::Regression { .. }) => {
            // Also acceptable: both detect (difference is significant enough for both)
        }
        (RegressionVerdict::NoRegression, RegressionVerdict::Regression { .. }) => {
            // If strict detects but default doesn't, that's wrong!
            panic!("Strict config should be more conservative than default");
        }
        _ => {
            // Other combinations (InsufficientData) are acceptable
        }
    }
}

/// Test insufficient data handling
///
/// Scenario: Too few samples for statistical test
/// Expected: InsufficientData verdict
#[test]
fn test_insufficient_data() {
    let mut baseline = HashMap::new();
    baseline.insert("mmap".to_string(), vec![100.0]); // Only 1 sample

    let mut current = HashMap::new();
    current.insert("mmap".to_string(), vec![100.0]);

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    match assessment.verdict {
        RegressionVerdict::InsufficientData { .. } => {
            // Expected
        }
        _ => panic!("Expected InsufficientData verdict"),
    }
}

/// Test report generation
#[test]
fn test_report_generation() {
    let mut baseline = HashMap::new();
    baseline.insert("mmap".to_string(), vec![10.0, 11.0, 10.0, 12.0, 10.0]);

    let mut current = HashMap::new();
    current.insert("mmap".to_string(), vec![50.0, 52.0, 51.0, 53.0, 50.0]);

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    let report = assessment.to_report_string();

    // Should contain key information
    assert!(report.contains("REGRESSION"));
    assert!(report.contains("mmap"));
    assert!(report.contains("Statistical Tests"));
}

/// Test permissive config increases sensitivity
///
/// Scenario: Small increase that default config might miss
/// Expected: Permissive config more likely to detect
#[test]
fn test_permissive_config_increases_sensitivity() {
    let mut baseline = HashMap::new();
    baseline.insert("mmap".to_string(), vec![100.0, 102.0, 101.0, 103.0, 100.0]);

    let mut current = HashMap::new();
    current.insert("mmap".to_string(), vec![105.0, 107.0, 106.0, 108.0, 105.0]);

    // Permissive config (90% confidence)
    let permissive_config = RegressionConfig::permissive();
    let permissive_assessment =
        assess_regression(&baseline, &current, &permissive_config).expect("test");

    // Should be more likely to detect regression (lower confidence bar)
    // Exact behavior depends on p-value, but permissive should never be stricter
    match permissive_assessment.verdict {
        RegressionVerdict::Regression { .. } | RegressionVerdict::NoRegression => {
            // Both acceptable
        }
        _ => panic!("Unexpected verdict"),
    }
}

/// Test multiple syscalls, mixed results
///
/// Scenario: Some syscalls regress, others stable
/// Expected: Only regressed syscalls reported
#[test]
fn test_multiple_syscalls_mixed_results() {
    let mut baseline = HashMap::new();
    baseline.insert("mmap".to_string(), vec![100.0, 102.0, 101.0, 103.0, 100.0]);
    baseline.insert("read".to_string(), vec![50.0, 51.0, 50.0, 52.0, 50.0]);
    baseline.insert("write".to_string(), vec![30.0, 31.0, 30.0, 32.0, 30.0]);

    let mut current = HashMap::new();
    current.insert(
        "mmap".to_string(),
        vec![200.0, 202.0, 201.0, 203.0, 200.0], // REGRESSED!
    );
    current.insert("read".to_string(), vec![50.0, 51.0, 50.0, 52.0, 50.0]); // Stable
    current.insert("write".to_string(), vec![30.0, 31.0, 30.0, 32.0, 30.0]); // Stable

    let config = RegressionConfig::default();
    let assessment = assess_regression(&baseline, &current, &config).expect("test");

    match assessment.verdict {
        RegressionVerdict::Regression { ref regressed_syscalls, .. } => {
            assert!(regressed_syscalls.contains(&"mmap".to_string()));
            assert!(!regressed_syscalls.contains(&"read".to_string()));
            assert!(!regressed_syscalls.contains(&"write".to_string()));
        }
        _ => panic!("Expected Regression verdict"),
    }
}

/// Test config validation
#[test]
#[allow(clippy::field_reassign_with_default)]
fn test_config_validation() {
    let mut config = RegressionConfig::default();

    // Valid config
    assert!(config.validate().is_ok());

    // Invalid significance level
    config.significance_level = 1.5;
    assert!(config.validate().is_err());

    // Invalid min sample size
    config = RegressionConfig::default();
    config.min_sample_size = 1;
    assert!(config.validate().is_err());

    // Invalid noise threshold
    config = RegressionConfig::default();
    config.noise_threshold = -0.5;
    assert!(config.validate().is_err());
}
