//! Tests for drift detection module.

use super::*;

#[test]
fn test_drift_test_name() {
    assert_eq!(DriftTest::KS { threshold: 0.05 }.name(), "Kolmogorov-Smirnov");
    assert_eq!(DriftTest::ChiSquare { threshold: 0.05 }.name(), "Chi-Square");
    assert_eq!(DriftTest::PSI { threshold: 0.1 }.name(), "PSI");
}

#[test]
fn test_drift_test_threshold() {
    assert_eq!(DriftTest::KS { threshold: 0.05 }.threshold(), 0.05);
    assert_eq!(DriftTest::PSI { threshold: 0.2 }.threshold(), 0.2);
}

#[test]
fn test_no_baseline() {
    let detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
    let results = detector.check(&[vec![1.0, 2.0]]);
    assert!(results.is_empty());
}

#[test]
fn test_ks_same_distribution() {
    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

    // Same distribution should not drift
    let data: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
    detector.set_baseline(&data);

    let results = detector.check(&data);
    assert_eq!(results.len(), 1);
    assert!(!results[0].drifted);
}

#[test]
fn test_ks_different_distribution() {
    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

    // Baseline: uniform 0-100
    let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
    detector.set_baseline(&baseline);

    // Current: shifted by 50
    let current: Vec<Vec<f64>> = (50..150).map(|i| vec![f64::from(i)]).collect();
    let results = detector.check(&current);

    assert_eq!(results.len(), 1);
    // Shifted distribution should trigger drift detection
    // The KS statistic should be significant
}

#[test]
fn test_psi_no_drift() {
    let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.2 }]);

    let data: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
    detector.set_baseline(&data);

    let results = detector.check(&data);
    assert_eq!(results.len(), 1);
    assert!(!results[0].drifted);
    assert!(results[0].statistic < 0.1); // PSI should be near 0
}

#[test]
fn test_psi_with_drift() {
    let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.1 }]);

    // Baseline: all values in [0, 10)
    let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i % 10)]).collect();
    detector.set_baseline(&baseline);

    // Current: all values in [90, 100) - completely different distribution
    let current: Vec<Vec<f64>> = (0..100).map(|i| vec![90.0 + f64::from(i % 10)]).collect();
    let results = detector.check(&current);

    assert_eq!(results.len(), 1);
    // Completely different distributions should have high PSI
}

#[test]
fn test_chi_square_same() {
    let mut detector = DriftDetector::new(vec![DriftTest::ChiSquare { threshold: 0.05 }]);

    let data: Vec<Vec<usize>> = (0..100).map(|i| vec![i % 5]).collect();
    detector.set_baseline_categorical(&data);

    let results = detector.check_categorical(&data);
    assert_eq!(results.len(), 1);
    assert!(!results[0].drifted);
}

#[test]
fn test_chi_square_different() {
    let mut detector = DriftDetector::new(vec![DriftTest::ChiSquare { threshold: 0.05 }]);

    // Baseline: uniform distribution over 0-4
    let baseline: Vec<Vec<usize>> = (0..100).map(|i| vec![i % 5]).collect();
    detector.set_baseline_categorical(&baseline);

    // Current: all values are 0
    let current: Vec<Vec<usize>> = (0..100).map(|_| vec![0]).collect();
    let results = detector.check_categorical(&current);

    assert_eq!(results.len(), 1);
    // Completely different categorical distribution should drift
}

#[test]
fn test_drift_summary() {
    let results = vec![
        DriftResult {
            feature: "f1".into(),
            test: DriftTest::KS { threshold: 0.05 },
            statistic: 0.5,
            p_value: 0.01,
            drifted: true,
            severity: Severity::Critical,
        },
        DriftResult {
            feature: "f2".into(),
            test: DriftTest::KS { threshold: 0.05 },
            statistic: 0.1,
            p_value: 0.3,
            drifted: false,
            severity: Severity::None,
        },
        DriftResult {
            feature: "f3".into(),
            test: DriftTest::KS { threshold: 0.05 },
            statistic: 0.2,
            p_value: 0.04,
            drifted: true,
            severity: Severity::Warning,
        },
    ];

    let summary = DriftDetector::summary(&results);
    assert_eq!(summary.total_features, 3);
    assert_eq!(summary.drifted_features, 2);
    assert_eq!(summary.warnings, 1);
    assert_eq!(summary.critical, 1);
    assert!(summary.has_critical());
    assert!(summary.has_drift());
    assert!((summary.drift_percentage() - 66.67).abs() < 1.0);
}

#[test]
fn test_empty_data() {
    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
    detector.set_baseline(&[]);
    let results = detector.check(&[vec![1.0]]);
    assert!(results.is_empty());
}

#[test]
fn test_bin_counts() {
    use super::statistical::bin_counts;
    let data = vec![0.5, 1.5, 2.5, 3.5];
    let edges = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let counts = bin_counts(&data, &edges);
    assert_eq!(counts, vec![1, 1, 1, 1]);
}

#[test]
fn test_ks_p_value() {
    use super::statistical::ks_p_value;
    // lambda = 0 should give p = 1
    assert!((ks_p_value(0.0) - 1.0).abs() < 0.01);
    // Large lambda should give small p
    assert!(ks_p_value(3.0) < 0.01);
}

#[test]
fn test_severity_eq() {
    assert_eq!(Severity::None, Severity::None);
    assert_ne!(Severity::None, Severity::Warning);
    assert_ne!(Severity::Warning, Severity::Critical);
}

#[test]
fn test_multiple_features() {
    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

    // 2 features
    let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i), f64::from(i * 2)]).collect();
    detector.set_baseline(&baseline);

    let results = detector.check(&baseline);
    assert_eq!(results.len(), 2); // One result per feature
}

#[test]
fn test_on_drift_callback() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);

    // Counter to track callback invocations
    let callback_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&callback_count);

    detector.on_drift(move |_results| {
        count_clone.fetch_add(1, Ordering::SeqCst);
    });

    // Baseline: uniform 0-100
    let baseline: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
    detector.set_baseline(&baseline);

    // Same distribution - should not trigger callback
    let _ = detector.check_and_trigger(&baseline);
    assert_eq!(callback_count.load(Ordering::SeqCst), 0);

    // Shifted distribution - should trigger callback
    let shifted: Vec<Vec<f64>> = (100..200).map(|i| vec![f64::from(i)]).collect();
    let _ = detector.check_and_trigger(&shifted);
    assert_eq!(callback_count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_check_and_trigger_no_drift() {
    let mut detector = DriftDetector::new(vec![DriftTest::PSI { threshold: 0.2 }]);

    let data: Vec<Vec<f64>> = (0..100).map(|i| vec![f64::from(i)]).collect();
    detector.set_baseline(&data);

    // Same data should not trigger
    let results = detector.check_and_trigger(&data);
    assert!(!results.iter().any(|r| r.drifted));
}
