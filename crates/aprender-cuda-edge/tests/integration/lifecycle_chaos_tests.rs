//! Integration tests for the Context Lifecycle Chaos framework (TCE-LIFECYCLE).
//!
//! Falsification tests from Section 8.2, F3 (claims 21-30).

#![allow(clippy::unwrap_used)]

use trueno_cuda_edge::lifecycle_chaos::{
    generate_destruction_orderings, validate_ordering, ChaosScenario, ContextLeakDetector,
    DestructionOrdering, LifecycleChaosConfig, OrderingValidation, LEAK_TOLERANCE_BYTES,
};

/// Claim 21: All 8 chaos scenarios are enumerated
#[test]
fn claim_21_all_chaos_scenarios_enumerated() {
    let scenarios = ChaosScenario::all();
    assert_eq!(scenarios.len(), 8, "Must enumerate exactly 8 chaos scenarios");
}

/// Claim 22: Destruction orderings are valid permutations
#[test]
fn claim_22_orderings_are_valid_permutations() {
    for n in 1..=5 {
        let orderings = generate_destruction_orderings(n);
        for ordering in &orderings {
            assert_eq!(
                validate_ordering(ordering, n),
                OrderingValidation::Valid,
                "All generated orderings must be valid permutations"
            );
        }
    }
}

/// Claim 23: Leak detector respects 1MB tolerance
#[test]
fn claim_23_leak_detector_tolerance() {
    let detector = ContextLeakDetector::new();
    assert_eq!(detector.tolerance(), LEAK_TOLERANCE_BYTES, "Default tolerance must be 1MB");

    // Just under tolerance: no leak
    let before = 100_000_000;
    let after = before + LEAK_TOLERANCE_BYTES - 1;
    let report = detector.analyze(before, after);
    assert!(!report.has_leaks(), "Memory within tolerance should not report leak");

    // Just over tolerance: leak
    let after_over = before + LEAK_TOLERANCE_BYTES + 1;
    let report_over = detector.analyze(before, after_over);
    assert!(report_over.has_leaks(), "Memory over tolerance must report leak");
}

/// Claim 24: Context leaks are detected
#[test]
fn claim_24_context_leaks_detected() {
    let detector = ContextLeakDetector::new();
    let report = detector.analyze_with_contexts(
        100_000_000,
        100_000_000,
        &[1, 2, 3],
        &[1, 2, 3, 4], // Context 4 is new → leaked
    );
    assert!(report.has_leaks(), "New context after test must be detected as leak");
}

/// Claim 25: N contexts produce N! orderings
#[test]
fn claim_25_factorial_orderings() {
    fn factorial(n: usize) -> usize {
        (1..=n).product()
    }

    for n in 0..=5 {
        let orderings = generate_destruction_orderings(n);
        let expected = factorial(n.max(1)); // 0! = 1
        assert_eq!(orderings.len(), expected, "{n} contexts must produce {expected} orderings");
    }
}

/// Claim 26: Reverse ordering is LIFO
#[test]
fn claim_26_reverse_ordering_is_lifo() {
    let ordering = DestructionOrdering::new(vec![3, 2, 1, 0]);
    assert!(ordering.is_reverse(), "Reverse ordering must be detected as LIFO");
    assert!(!ordering.is_forward(), "Reverse ordering is not FIFO");
}

/// Claim 27: Memory decrease is not a leak
#[test]
fn claim_27_memory_decrease_not_leak() {
    let detector = ContextLeakDetector::new();
    let report = detector.analyze(200_000_000, 100_000_000);
    assert!(!report.has_leaks(), "Memory decrease must not be reported as leak");
}

/// Claim 28: Default config includes all scenarios
#[test]
fn claim_28_default_config_all_scenarios() {
    let config = LifecycleChaosConfig::default();
    assert_eq!(config.scenarios.len(), 8, "Default config must include all 8 scenarios");
}

/// Test that each scenario has a unique description
#[test]
fn scenarios_have_unique_descriptions() {
    let scenarios = ChaosScenario::all();
    let descriptions: Vec<_> = scenarios.iter().map(ChaosScenario::description).collect();
    for (i, a) in descriptions.iter().enumerate() {
        for (j, b) in descriptions.iter().enumerate() {
            if i != j {
                assert_ne!(a, b, "Scenarios at indices {i} and {j} have duplicate descriptions");
            }
        }
    }
}

/// Test forward ordering detection
#[test]
fn forward_ordering_is_fifo() {
    let ordering = DestructionOrdering::new(vec![0, 1, 2, 3]);
    assert!(ordering.is_forward(), "Forward ordering must be detected as FIFO");
    assert!(!ordering.is_reverse(), "Forward ordering is not LIFO");
}

/// Test invalid ordering detection
#[test]
fn invalid_ordering_detected() {
    let ordering = DestructionOrdering::new(vec![0, 1, 1]); // Duplicate
    let result = validate_ordering(&ordering, 3);
    assert!(
        matches!(result, OrderingValidation::Invalid { .. }),
        "Duplicate indices must be invalid"
    );
}

/// Test leak report total bytes calculation
#[test]
fn leak_report_total_bytes() {
    let detector = ContextLeakDetector::new();
    let before = 100_000_000;
    let after = before + 2 * LEAK_TOLERANCE_BYTES;
    let report = detector.analyze(before, after);
    assert_eq!(
        report.total_leaked_bytes(),
        2 * LEAK_TOLERANCE_BYTES,
        "Total leaked bytes must match actual delta"
    );
}
