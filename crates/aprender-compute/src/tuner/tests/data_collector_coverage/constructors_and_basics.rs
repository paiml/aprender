//! Constructor, online learning toggle, record, samples, len, is_empty, merge tests

use super::*;

// ============================================================================
// Constructor tests
// ============================================================================

#[test]
fn new_collector_has_correct_defaults() {
    let c = TunerDataCollector::new();
    assert!(c.is_empty());
    assert_eq!(c.len(), 0);
    assert!(!c.is_online_learning_enabled());
    assert_eq!(c.retrain_threshold, 100);
    assert_eq!(c.samples_at_last_train, 0);
    assert!(c.feedback.is_empty());
    assert!(c.error_window.is_empty());
}

#[test]
fn with_online_learning_enables_flag() {
    let c = TunerDataCollector::with_online_learning();
    assert!(c.is_online_learning_enabled());
    assert!(c.is_empty());
}

// ============================================================================
// Online learning toggle
// ============================================================================

#[test]
fn enable_then_disable_online_learning() {
    let mut c = TunerDataCollector::new();
    assert!(!c.is_online_learning_enabled());

    c.enable_online_learning();
    assert!(c.is_online_learning_enabled());

    c.disable_online_learning();
    assert!(!c.is_online_learning_enabled());
}

// ============================================================================
// record() with BrickProfiler
// ============================================================================

#[test]
fn record_returns_none_when_profiler_has_no_tokens() {
    let mut collector = TunerDataCollector::new();
    let profiler = BrickProfiler::new();
    let config = RunConfig::default();
    // BrickProfiler with no timing data -> tokens_per_sec() returns None
    let result = collector.record(&profiler, &config, KernelType::TiledQ4K);
    assert!(result.is_none());
    assert!(collector.is_empty());
}

// ============================================================================
// samples(), len(), is_empty()
// ============================================================================

#[test]
fn samples_returns_all_pushed_samples() {
    let mut c = TunerDataCollector::new();
    assert!(c.samples().is_empty());
    c.samples.push(make_sample(42.0));
    c.samples.push(make_sample(99.0));
    assert_eq!(c.samples().len(), 2);
    assert_eq!(c.len(), 2);
    assert!(!c.is_empty());
}

// ============================================================================
// merge()
// ============================================================================

#[test]
fn merge_combines_samples_from_two_collectors() {
    let mut a = make_collector_with_samples(3);
    let b = make_collector_with_samples(2);
    a.merge(&b);
    assert_eq!(a.len(), 5);
}

#[test]
fn merge_with_empty_collector_is_noop() {
    let mut a = make_collector_with_samples(3);
    let b = TunerDataCollector::new();
    a.merge(&b);
    assert_eq!(a.len(), 3);
}

#[test]
fn merge_into_empty_collector() {
    let mut a = TunerDataCollector::new();
    let b = make_collector_with_samples(5);
    a.merge(&b);
    assert_eq!(a.len(), 5);
}
