// ============================================================================
// F3: Context Lifecycle Chaos -- GPU Context Management
// ============================================================================

use trueno_cuda_edge::lifecycle_chaos::{
    ChaosScenario, ContextLeakDetector, DestructionOrdering, LifecycleChaosConfig,
};

/// Test all 8 chaos scenarios for GPU context lifecycle.
#[test]
fn chaos_scenario_coverage() {
    let scenarios = ChaosScenario::all();
    assert_eq!(scenarios.len(), 8);

    // Verify critical scenarios are present
    assert!(scenarios.contains(&ChaosScenario::DoubleDestroy));
    assert!(scenarios.contains(&ChaosScenario::UseAfterDestroy));
    assert!(scenarios.contains(&ChaosScenario::LeakedContext));
    assert!(scenarios.contains(&ChaosScenario::ContextExhaustion));
}

/// Test default chaos configuration.
#[test]
fn default_chaos_config() {
    let config = LifecycleChaosConfig::default();

    assert_eq!(config.scenarios.len(), 8);
    assert_eq!(config.max_contexts, 64);
    assert!(config.capture_memory_snapshots);
}

/// Test destruction ordering validation.
#[test]
fn destruction_ordering_patterns() {
    // LIFO (reverse) -- correct for CUDA
    let lifo = DestructionOrdering::new(vec![2, 1, 0]);
    assert!(lifo.is_reverse());
    assert!(!lifo.is_forward());

    // FIFO (forward) -- may cause issues
    let fifo = DestructionOrdering::new(vec![0, 1, 2]);
    assert!(fifo.is_forward());
    assert!(!fifo.is_reverse());

    // Random -- neither
    let random = DestructionOrdering::new(vec![1, 0, 2]);
    assert!(!random.is_reverse());
    assert!(!random.is_forward());
}

/// Test memory leak detection with tolerance.
#[test]
fn leak_detection_with_tolerance() {
    let detector = ContextLeakDetector::new();

    // Within 1 MB tolerance: no leak
    let report = detector.analyze(100_000_000, 100_500_000);
    assert!(!report.has_leaks());

    // Above 1 MB tolerance: leak detected
    let report = detector.analyze(100_000_000, 102_000_000);
    assert!(report.has_leaks());
    assert!(report.total_leaked_bytes() > 0);
}

/// Test custom tolerance for strict leak detection.
#[test]
fn custom_leak_tolerance() {
    let strict = ContextLeakDetector::with_tolerance(1024); // 1 KB

    let report = strict.analyze(1000, 3000);
    assert!(report.has_leaks()); // 2000 > 1024
}
