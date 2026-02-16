pub(crate) use super::*;

#[test]
fn test_drift_status_needs_retraining() {
    assert!(!DriftStatus::NoDrift.needs_retraining());
    assert!(!DriftStatus::Warning { score: 0.15 }.needs_retraining());
    assert!(DriftStatus::Drift { score: 0.25 }.needs_retraining());
}

#[test]
fn test_drift_status_score() {
    assert_eq!(DriftStatus::NoDrift.score(), None);
    assert_eq!(DriftStatus::Warning { score: 0.15 }.score(), Some(0.15));
    assert_eq!(DriftStatus::Drift { score: 0.25 }.score(), Some(0.25));
}

#[test]
fn test_drift_config_default() {
    let config = DriftConfig::default();
    assert!((config.warning_threshold - 0.1).abs() < 1e-6);
    assert!((config.drift_threshold - 0.2).abs() < 1e-6);
    assert_eq!(config.min_samples, 30);
}

#[test]
fn test_drift_config_builder() {
    let config = DriftConfig::new(0.15, 0.3)
        .with_min_samples(50)
        .with_window_size(200);

    assert!((config.warning_threshold - 0.15).abs() < 1e-6);
    assert!((config.drift_threshold - 0.3).abs() < 1e-6);
    assert_eq!(config.min_samples, 50);
    assert_eq!(config.window_size, 200);
}

#[test]
fn test_detector_no_drift() {
    let reference = Vector::from_slice(&(0..100).map(|i| i as f32).collect::<Vec<_>>());
    let current = Vector::from_slice(&(0..100).map(|i| (i as f32) + 0.1).collect::<Vec<_>>());

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let status = detector.detect_univariate(&reference, &current);

    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_detector_significant_drift() {
    let reference = Vector::from_slice(&(0..100).map(|i| i as f32).collect::<Vec<_>>());
    // Shifted by 50 (large drift)
    let current = Vector::from_slice(&(0..100).map(|i| (i + 50) as f32).collect::<Vec<_>>());

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let status = detector.detect_univariate(&reference, &current);

    assert!(matches!(status, DriftStatus::Drift { .. }));
}

#[test]
fn test_detector_insufficient_samples() {
    let reference = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let current = Vector::from_slice(&[10.0, 20.0, 30.0]);

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(30));
    let status = detector.detect_univariate(&reference, &current);

    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_detector_multivariate() {
    let reference =
        Matrix::from_vec(50, 2, (0..100).map(|i| i as f32).collect()).expect("valid dimensions");
    let current =
        Matrix::from_vec(50, 2, (0..100).map(|i| i as f32).collect()).expect("valid dimensions");

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let (overall, feature_statuses) = detector.detect_multivariate(&reference, &current);

    assert_eq!(feature_statuses.len(), 2);
    assert!(matches!(overall, DriftStatus::NoDrift));
}

#[test]
fn test_performance_drift_degradation() {
    let baseline = vec![0.95, 0.94, 0.96, 0.95, 0.94];
    let current = vec![0.75, 0.74, 0.73, 0.74, 0.75]; // Significant drop

    let detector = DriftDetector::new(DriftConfig::default());
    let status = detector.detect_performance_drift(&baseline, &current);

    assert!(matches!(
        status,
        DriftStatus::Drift { .. } | DriftStatus::Warning { .. }
    ));
}

#[test]
fn test_performance_drift_improvement() {
    let baseline = vec![0.75, 0.74, 0.73, 0.74, 0.75];
    let current = vec![0.95, 0.94, 0.96, 0.95, 0.94]; // Improvement

    let detector = DriftDetector::new(DriftConfig::default());
    let status = detector.detect_performance_drift(&baseline, &current);

    // Improvement should not trigger drift
    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_rolling_monitor() {
    let config = DriftConfig::default()
        .with_min_samples(5)
        .with_window_size(10);
    let mut monitor = RollingDriftMonitor::new(config);

    monitor.set_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Observe same values - no drift
    for i in 0..10 {
        let _ = monitor.observe((i + 1) as f32);
    }
    // After filling window with same data, check no drift detected
    let status = monitor.check_drift();
    assert!(!status.needs_retraining());
}

#[test]
fn test_rolling_monitor_drift() {
    let config = DriftConfig::default()
        .with_min_samples(5)
        .with_window_size(10);
    let mut monitor = RollingDriftMonitor::new(config);

    monitor.set_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    // Observe very different values - should trigger drift
    for _ in 0..10 {
        monitor.observe(1000.0);
    }

    let status = monitor.check_drift();
    assert!(status.needs_retraining());
}

#[test]
fn test_rolling_monitor_reset() {
    let config = DriftConfig::default().with_min_samples(5);
    let mut monitor = RollingDriftMonitor::new(config);

    monitor.set_reference(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    for _ in 0..5 {
        monitor.observe(100.0);
    }

    monitor.reset_current();
    let status = monitor.check_drift();
    // After reset, not enough samples
    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_retraining_trigger() {
    let config = DriftConfig::default().with_min_samples(3);
    let mut trigger = RetrainingTrigger::new(2, config).with_consecutive_required(2);

    trigger.set_baseline_performance(&[0.95, 0.94, 0.96, 0.95, 0.94]);

    // Good performance - no trigger
    assert!(!trigger.observe_performance(0.94));
    assert!(!trigger.observe_performance(0.95));

    // Reset and test trigger
    trigger.reset();
    assert!(!trigger.is_triggered());
}

#[test]
fn test_retraining_trigger_activation() {
    let config = DriftConfig::new(0.01, 0.02).with_min_samples(3);
    let mut trigger = RetrainingTrigger::new(1, config).with_consecutive_required(2);

    trigger.set_baseline_performance(&[0.95, 0.94, 0.96]);

    // Observe significant performance drop repeatedly
    trigger.observe_performance(0.50);
    trigger.observe_performance(0.51);
    trigger.observe_performance(0.49);

    // Should eventually trigger
    assert!(trigger.observe_performance(0.48) || trigger.is_triggered());
}

#[test]
fn test_helper_mean() {
    assert!((mean(&[]) - 0.0).abs() < 1e-6);
    assert!((mean(&[5.0]) - 5.0).abs() < 1e-6);
    assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-6);
}

#[test]
fn test_helper_std_dev() {
    assert!((std_dev(&[], 0.0) - 0.0).abs() < 1e-6);
    assert!((std_dev(&[5.0], 5.0) - 0.0).abs() < 1e-6);
    // std([1,2,3,4,5]) ≈ 1.5811
    let data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let std = std_dev(&data, mean(&data));
    assert!((std - 1.5811).abs() < 0.001);
}

// ================================================================
// Additional coverage tests for missed branches
// ================================================================

#[test]
fn test_performance_drift_empty_baseline() {
    let detector = DriftDetector::new(DriftConfig::default());
    let status = detector.detect_performance_drift(&[], &[0.9, 0.8]);
    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_performance_drift_empty_current() {
    let detector = DriftDetector::new(DriftConfig::default());
    let status = detector.detect_performance_drift(&[0.9, 0.8], &[]);
    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_performance_drift_zero_std_relative_drop() {
    // When baseline_std < 1e-10, uses relative drop instead
    // All baseline scores identical => std = 0
    let baseline = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let current = vec![0.3, 0.3, 0.3, 0.3, 0.3]; // Significant drop

    let detector = DriftDetector::new(DriftConfig::new(0.1, 0.2));
    let status = detector.detect_performance_drift(&baseline, &current);

    // relative_drop = (0.5 - 0.3) / 0.5 = 0.4, exceeds drift threshold
    assert!(
        matches!(status, DriftStatus::Drift { .. }),
        "Expected Drift, got {:?}",
        status
    );
}

#[test]
fn test_performance_drift_zero_std_no_drop() {
    // All identical and current also identical => no drop
    let baseline = vec![0.9, 0.9, 0.9, 0.9, 0.9];
    let current = vec![0.9, 0.9, 0.9, 0.9, 0.9];

    let detector = DriftDetector::new(DriftConfig::new(0.1, 0.2));
    let status = detector.detect_performance_drift(&baseline, &current);

    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_performance_drift_zero_std_improvement() {
    // Current is better than baseline => relative_drop is negative => clamp to 0
    let baseline = vec![0.5, 0.5, 0.5, 0.5, 0.5];
    let current = vec![0.8, 0.8, 0.8, 0.8, 0.8];

    let detector = DriftDetector::new(DriftConfig::new(0.1, 0.2));
    let status = detector.detect_performance_drift(&baseline, &current);

    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_univariate_zero_std_reference() {
    // All reference values identical => ref_std < 1e-10 => NoDrift
    let reference = Vector::from_slice(&vec![5.0; 50]);
    let current = Vector::from_slice(&vec![100.0; 50]);

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let status = detector.detect_univariate(&reference, &current);

    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_multivariate_drift_on_one_feature() {
    // Feature 0: same distribution, Feature 1: drifted
    let mut ref_data = Vec::with_capacity(100);
    let mut cur_data = Vec::with_capacity(100);

    for i in 0..50 {
        ref_data.push(i as f32); // Feature 0
        ref_data.push(i as f32); // Feature 1
        cur_data.push(i as f32); // Feature 0 (same)
        cur_data.push((i + 200) as f32); // Feature 1 (drifted)
    }

    let reference = Matrix::from_vec(50, 2, ref_data).expect("valid dimensions");
    let current = Matrix::from_vec(50, 2, cur_data).expect("valid dimensions");

    let detector = DriftDetector::new(DriftConfig::default().with_min_samples(10));
    let (overall, feature_statuses) = detector.detect_multivariate(&reference, &current);

    assert_eq!(feature_statuses.len(), 2);
    // Feature 0 should be no drift; feature 1 should show drift
    assert!(matches!(feature_statuses[0], DriftStatus::NoDrift));
    assert!(matches!(feature_statuses[1], DriftStatus::Drift { .. }));
    // Overall should reflect the drifted feature
    assert!(matches!(overall, DriftStatus::Drift { .. }));
}

#[test]
fn test_rolling_monitor_update_reference() {
    let config = DriftConfig::default()
        .with_min_samples(3)
        .with_window_size(10);
    let mut monitor = RollingDriftMonitor::new(config);

    monitor.set_reference(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    for _ in 0..5 {
        monitor.observe(10.0);
    }

    // Update reference to current window
    monitor.update_reference();

    // After update_reference, current_window should be cleared
    // and reference_window should contain the old current data
    let status = monitor.check_drift();
    // No data in current => NoDrift (insufficient samples)
    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_rolling_monitor_window_overflow() {
    let config = DriftConfig::default()
        .with_min_samples(3)
        .with_window_size(5);
    let mut monitor = RollingDriftMonitor::new(config);

    monitor.set_reference(&[1.0, 2.0, 3.0, 4.0, 5.0]);

    // Add more values than window size to trigger overflow removal
    for i in 0..10 {
        monitor.observe(i as f32);
    }

    // Window should be trimmed to max_window (5)
    let status = monitor.check_drift();
    // Just verify no panic and returns a valid status
    assert!(matches!(
        status,
        DriftStatus::NoDrift | DriftStatus::Warning { .. } | DriftStatus::Drift { .. }
    ));
}

#[test]
fn test_rolling_monitor_set_reference_overflow() {
    let config = DriftConfig::default().with_window_size(3);
    let mut monitor = RollingDriftMonitor::new(config);

    // Provide more data than window size
    monitor.set_reference(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    // Should only keep last 3 values
    let status = monitor.check_drift();
    assert!(matches!(status, DriftStatus::NoDrift));
}

#[test]
fn test_retraining_trigger_set_baseline_features() {
    let config = DriftConfig::default().with_min_samples(3);
    let mut trigger = RetrainingTrigger::new(2, config);

    let features = Matrix::from_vec(
        5,
        2,
        vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0],
    )
    .expect("valid dimensions");

    trigger.set_baseline_features(&features);
    // Should not panic; baseline is now set
}

#[test]
fn test_retraining_trigger_set_baseline_features_more_monitors() {
    // More monitors than feature columns
    let config = DriftConfig::default().with_min_samples(3);
    let mut trigger = RetrainingTrigger::new(5, config);

    let features =
        Matrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid dimensions");

    // Only first 2 monitors get baselines, the rest skip due to i < features.n_cols()
    trigger.set_baseline_features(&features);
}

#[test]
fn test_retraining_trigger_consecutive_reset() {
    let config = DriftConfig::new(0.01, 0.02).with_min_samples(3);
    let mut trigger = RetrainingTrigger::new(1, config).with_consecutive_required(3);

    trigger.set_baseline_performance(&[0.95, 0.94, 0.96]);

    // Observe 2 drift signals, then a non-drift to reset counter
    trigger.observe_performance(0.1);
    trigger.observe_performance(0.1);
    assert!(!trigger.is_triggered());

    // This good observation resets consecutive count
    trigger.observe_performance(0.95);
    assert!(!trigger.is_triggered());
}

#[test]
fn test_classify_drift_warning() {
    let detector = DriftDetector::new(DriftConfig::new(0.1, 0.5));

    // Score between warning and drift thresholds
    let status = detector.classify_drift(0.3);
    assert!(matches!(status, DriftStatus::Warning { score } if (score - 0.3).abs() < 1e-6));
}

#[test]
fn test_classify_drift_exact_threshold() {
    let detector = DriftDetector::new(DriftConfig::new(0.1, 0.5));

    // Exactly at drift threshold
    let status = detector.classify_drift(0.5);
    assert!(matches!(status, DriftStatus::Drift { .. }));

    // Exactly at warning threshold
    let status = detector.classify_drift(0.1);
    assert!(matches!(status, DriftStatus::Warning { .. }));
}

#[test]
fn test_drift_status_clone() {
    let status = DriftStatus::Warning { score: 0.15 };
    let cloned = status.clone();
    assert_eq!(status, cloned);
}

#[test]
fn test_drift_config_clone() {
    let config = DriftConfig::new(0.1, 0.3);
    let cloned = config.clone();
    assert!((cloned.warning_threshold - 0.1).abs() < 1e-6);
    assert!((cloned.drift_threshold - 0.3).abs() < 1e-6);
}
