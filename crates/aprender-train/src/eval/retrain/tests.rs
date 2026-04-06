//! Tests for the auto-retraining module.

use super::*;
use crate::eval::drift::{DriftDetector, DriftTest};

fn create_detector() -> DriftDetector {
    DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }])
}

/// Build a single-feature batch of sequential f64 values: `[start, start+1, ..., end-1]`.
fn make_batch(range: std::ops::Range<usize>) -> Vec<Vec<f64>> {
    range.map(|i| vec![f64::from(i as u32)]).collect()
}

/// Build a two-feature batch: `[[i, i*2.0] for i in range]`.
fn make_batch_two_features(range: std::ops::Range<usize>) -> Vec<Vec<f64>> {
    range.map(|i| vec![f64::from(i as u32), f64::from(i as u32) * 2.0]).collect()
}

#[test]
fn test_retrain_policy_default() {
    let policy = RetrainPolicy::default();
    assert!(matches!(policy, RetrainPolicy::AnyCritical));
}

#[test]
fn test_retrain_config_default() {
    let config = RetrainConfig::default();
    assert_eq!(config.cooldown_batches, 100);
    assert_eq!(config.max_retrains, 0);
    assert!(config.log_warnings);
}

#[test]
fn test_auto_retrainer_no_baseline() {
    let detector = create_detector();
    let config = RetrainConfig::default();
    let mut retrainer = AutoRetrainer::new(detector, config);

    let batch = make_batch(0..10);
    let action = retrainer.process_batch(&batch).expect("operation should succeed");

    assert_eq!(action, Action::None);
}

#[test]
fn test_auto_retrainer_no_drift() {
    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig { cooldown_batches: 0, ..Default::default() };
    let mut retrainer = AutoRetrainer::new(detector, config);

    // Same distribution should not trigger
    let action = retrainer.process_batch(&baseline).expect("operation should succeed");
    assert_eq!(action, Action::None);
}

#[test]
fn test_auto_retrainer_with_drift() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig { cooldown_batches: 0, ..Default::default() };
    let mut retrainer = AutoRetrainer::new(detector, config);

    let retrain_count = Arc::new(AtomicUsize::new(0));
    let count_clone = Arc::clone(&retrain_count);

    retrainer.on_retrain(move |_results| {
        count_clone.fetch_add(1, Ordering::SeqCst);
        Ok("job-123".to_string())
    });

    // Shifted distribution should trigger
    let shifted = make_batch(100..200);
    let action = retrainer.process_batch(&shifted).expect("operation should succeed");

    assert!(matches!(action, Action::RetrainTriggered(_)));
    assert_eq!(retrain_count.load(Ordering::SeqCst), 1);
}

#[test]
fn test_cooldown_prevents_retrain() {
    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig {
        cooldown_batches: 10, // Require 10 batches between retrains
        ..Default::default()
    };
    let mut retrainer = AutoRetrainer::new(detector, config);

    retrainer.on_retrain(|_| Ok("job".to_string()));

    // Reset cooldown so first batch can trigger
    retrainer.reset_cooldown();

    // First batch with drift should trigger
    let shifted = make_batch(100..200);
    let action1 = retrainer.process_batch(&shifted).expect("operation should succeed");
    assert!(matches!(action1, Action::RetrainTriggered(_)));

    // Immediate second batch should be blocked by cooldown
    let action2 = retrainer.process_batch(&shifted).expect("operation should succeed");
    assert_eq!(action2, Action::WarningLogged);
}

#[test]
fn test_max_retrains_limit() {
    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig { cooldown_batches: 0, max_retrains: 2, ..Default::default() };
    let mut retrainer = AutoRetrainer::new(detector, config);

    retrainer.on_retrain(|_| Ok("job".to_string()));

    let shifted = make_batch(100..200);

    // First two should trigger
    assert!(matches!(
        retrainer.process_batch(&shifted).expect("operation should succeed"),
        Action::RetrainTriggered(_)
    ));
    assert!(matches!(
        retrainer.process_batch(&shifted).expect("operation should succeed"),
        Action::RetrainTriggered(_)
    ));

    // Third should be blocked
    assert_eq!(
        retrainer.process_batch(&shifted).expect("operation should succeed"),
        Action::WarningLogged
    );
}

#[test]
fn test_feature_count_policy() {
    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
    let baseline = make_batch_two_features(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig {
        policy: RetrainPolicy::FeatureCount { count: 2 },
        cooldown_batches: 0,
        ..Default::default()
    };
    let mut retrainer = AutoRetrainer::new(detector, config);

    retrainer.on_retrain(|_| Ok("job".to_string()));

    // Both features shifted - should trigger
    let shifted = make_batch_two_features(100..200);
    let action = retrainer.process_batch(&shifted).expect("operation should succeed");
    assert!(matches!(action, Action::RetrainTriggered(_)));
}

#[test]
fn test_stats() {
    let detector = create_detector();
    let config = RetrainConfig::default();
    let retrainer = AutoRetrainer::new(detector, config);

    let stats = retrainer.stats();
    assert_eq!(stats.total_retrains, 0);
    assert_eq!(stats.batches_since_retrain, 0);
}

#[test]
fn test_action_eq() {
    assert_eq!(Action::None, Action::None);
    assert_eq!(Action::WarningLogged, Action::WarningLogged);
    assert_ne!(Action::None, Action::WarningLogged);
    assert_eq!(
        Action::RetrainTriggered("a".to_string()),
        Action::RetrainTriggered("a".to_string())
    );
}

/// APR-073 Section 10.4: Callback must trigger synchronously on drift detection
#[test]
fn test_callback_latency() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig { cooldown_batches: 0, ..Default::default() };
    let mut retrainer = AutoRetrainer::new(detector, config);

    let callback_fired = Arc::new(AtomicBool::new(false));
    let fired_clone = Arc::clone(&callback_fired);

    retrainer.on_retrain(move |_results| {
        fired_clone.store(true, Ordering::SeqCst);
        Ok("job-latency-test".to_string())
    });

    let shifted = make_batch(100..200);
    let action = retrainer.process_batch(&shifted).expect("operation should succeed");

    assert!(matches!(action, Action::RetrainTriggered(_)));
    assert!(
        callback_fired.load(Ordering::SeqCst),
        "Callback must fire synchronously during process_batch"
    );
}

#[test]
fn test_critical_feature_policy() {
    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig {
        policy: RetrainPolicy::CriticalFeature { names: vec!["feature_0".to_string()] },
        cooldown_batches: 0,
        ..Default::default()
    };
    let mut retrainer = AutoRetrainer::new(detector, config);

    retrainer.on_retrain(|_| Ok("job".to_string()));

    // Shifted distribution should trigger for critical feature
    let shifted = make_batch(100..200);
    let action = retrainer.process_batch(&shifted).expect("operation should succeed");
    assert!(matches!(action, Action::RetrainTriggered(_)));
}

#[test]
fn test_drift_percentage_policy() {
    let mut detector = DriftDetector::new(vec![DriftTest::KS { threshold: 0.05 }]);
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig {
        policy: RetrainPolicy::DriftPercentage { threshold: 0.5 },
        cooldown_batches: 0,
        ..Default::default()
    };
    let mut retrainer = AutoRetrainer::new(detector, config);

    retrainer.on_retrain(|_| Ok("job".to_string()));

    // Shifted distribution - 100% drift (1 of 1 feature)
    let shifted = make_batch(100..200);
    let action = retrainer.process_batch(&shifted).expect("operation should succeed");
    assert!(matches!(action, Action::RetrainTriggered(_)));
}

#[test]
fn test_action_clone() {
    let action1 = Action::None;
    let cloned = action1.clone();
    assert_eq!(action1, cloned);

    let action2 = Action::RetrainTriggered("job-123".to_string());
    let cloned2 = action2.clone();
    assert_eq!(action2, cloned2);
}

#[test]
fn test_retrain_config_clone() {
    let config = RetrainConfig {
        policy: RetrainPolicy::FeatureCount { count: 3 },
        cooldown_batches: 50,
        max_retrains: 5,
        log_warnings: false,
    };
    let cloned = config.clone();
    assert_eq!(cloned.cooldown_batches, 50);
    assert_eq!(cloned.max_retrains, 5);
    assert!(!cloned.log_warnings);
}

#[test]
fn test_retrainer_stats_clone() {
    let stats = RetrainerStats { total_retrains: 3, batches_since_retrain: 42 };
    let cloned = stats.clone();
    assert_eq!(cloned.total_retrains, 3);
    assert_eq!(cloned.batches_since_retrain, 42);
}

#[test]
fn test_detector_access() {
    let detector = create_detector();
    let config = RetrainConfig::default();
    let mut retrainer = AutoRetrainer::new(detector, config);

    // Test immutable access
    let _detector = retrainer.detector();

    // Test mutable access and modify baseline
    let baseline = make_batch(0..10);
    retrainer.detector_mut().set_baseline(&baseline);
}

#[test]
fn test_no_callback_set_with_drift() {
    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig { cooldown_batches: 0, ..Default::default() };
    let mut retrainer = AutoRetrainer::new(detector, config);
    // No callback set

    // Shifted distribution - should want to retrain but has no callback
    let shifted = make_batch(100..200);
    let action = retrainer.process_batch(&shifted).expect("operation should succeed");
    // Returns WarningLogged since there's no callback
    assert_eq!(action, Action::WarningLogged);
}

#[test]
fn test_no_drift_during_cooldown() {
    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig { cooldown_batches: 10, log_warnings: true, ..Default::default() };
    let mut retrainer = AutoRetrainer::new(detector, config);

    // Same distribution - no drift during cooldown
    let action = retrainer.process_batch(&baseline).expect("operation should succeed");
    assert_eq!(action, Action::None);
}

#[test]
fn test_max_retrains_with_warnings_disabled() {
    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig {
        cooldown_batches: 0,
        max_retrains: 1,
        log_warnings: false,
        ..Default::default()
    };
    let mut retrainer = AutoRetrainer::new(detector, config);

    retrainer.on_retrain(|_| Ok("job".to_string()));

    let shifted = make_batch(100..200);

    // First should trigger
    assert!(matches!(
        retrainer.process_batch(&shifted).expect("operation should succeed"),
        Action::RetrainTriggered(_)
    ));

    // Second should be blocked, and since log_warnings is false, returns None
    assert_eq!(retrainer.process_batch(&shifted).expect("operation should succeed"), Action::None);
}

#[test]
fn test_retrain_policy_clone() {
    let policy1 = RetrainPolicy::FeatureCount { count: 5 };
    let cloned = policy1.clone();
    assert!(matches!(cloned, RetrainPolicy::FeatureCount { count: 5 }));

    let policy2 = RetrainPolicy::CriticalFeature { names: vec!["a".to_string()] };
    let cloned2 = policy2.clone();
    if let RetrainPolicy::CriticalFeature { names } = cloned2 {
        assert_eq!(names, vec!["a".to_string()]);
    } else {
        panic!("Wrong variant");
    }

    let policy3 = RetrainPolicy::DriftPercentage { threshold: 0.75 };
    let cloned3 = policy3.clone();
    if let RetrainPolicy::DriftPercentage { threshold } = cloned3 {
        assert!((threshold - 0.75).abs() < f64::EPSILON);
    } else {
        panic!("Wrong variant");
    }

    let policy4 = RetrainPolicy::AnyCritical;
    let cloned4 = policy4.clone();
    assert!(matches!(cloned4, RetrainPolicy::AnyCritical));
}

#[test]
fn test_warnings_with_no_drift_but_in_cooldown() {
    let mut detector = create_detector();
    let baseline = make_batch(0..100);
    detector.set_baseline(&baseline);

    let config = RetrainConfig { cooldown_batches: 10, log_warnings: true, ..Default::default() };
    let mut retrainer = AutoRetrainer::new(detector, config);

    retrainer.on_retrain(|_| Ok("job".to_string()));
    retrainer.reset_cooldown();

    // First retrain
    let shifted = make_batch(100..200);
    let action1 = retrainer.process_batch(&shifted).expect("operation should succeed");
    assert!(matches!(action1, Action::RetrainTriggered(_)));

    // In cooldown now - should return WarningLogged for drift
    let action2 = retrainer.process_batch(&shifted).expect("operation should succeed");
    assert_eq!(action2, Action::WarningLogged);
}
