//! Integration tests for the Null Pointer Sentinel Fuzzer (TCE-NULL).
//!
//! Falsification tests from Section 8.2, F1 (claims 1-10).

#![allow(clippy::unwrap_used, clippy::disallowed_methods)]

use trueno_cuda_edge::null_fuzzer::{
    InjectionStrategy, NonNullDevicePtr, NullFuzzerConfig, NullSentinelFuzzer, PropagationTracker,
};

/// Claim 1: `NonNullDevicePtr::new(0)` returns `Err`
#[test]
fn claim_01_non_null_ptr_rejects_zero() {
    let result = NonNullDevicePtr::<u8>::new(0);
    assert!(result.is_err(), "NonNullDevicePtr::new(0) must return Err");
}

/// Claim 2: `NonNullDevicePtr::new(valid)` returns `Ok`
#[test]
fn claim_02_non_null_ptr_accepts_valid() {
    let valid_addr = 0x1000_u64;
    let result = NonNullDevicePtr::<u8>::new(valid_addr);
    assert!(result.is_ok(), "NonNullDevicePtr::new(valid) must return Ok");
    assert_eq!(result.unwrap().addr(), valid_addr);
}

/// Claim 6: Periodic injection strategy produces deterministic results
#[test]
fn claim_06_periodic_injection_deterministic() {
    let config = NullFuzzerConfig {
        strategy: InjectionStrategy::Periodic { interval: 5 },
        total_calls: 100,
        fail_fast: false,
    };

    // Run twice with same config
    let mut fuzzer1 = NullSentinelFuzzer::new(config.clone());
    let mut fuzzer2 = NullSentinelFuzzer::new(config);

    let mut injections1 = Vec::new();
    let mut injections2 = Vec::new();

    for _ in 0..20 {
        injections1.push(fuzzer1.next_call());
        injections2.push(fuzzer2.next_call());
    }

    assert_eq!(injections1, injections2, "Periodic injection must be deterministic");
}

/// Claim 7: Size-threshold strategy only injects above threshold
#[test]
fn claim_07_size_threshold_below_no_injection() {
    let strategy = InjectionStrategy::SizeThreshold {
        threshold_bytes: 1_048_576, // 1MB
    };

    // SizeThreshold requires allocation context, so should_inject returns false
    // without that context
    assert!(!strategy.should_inject(0), "SizeThreshold without context must not inject");
}

/// Claim 8: Fuzzer report counts match actual injections
#[test]
fn claim_08_report_counts_match() {
    let config = NullFuzzerConfig {
        strategy: InjectionStrategy::Periodic { interval: 3 },
        total_calls: 30,
        fail_fast: false,
    };

    let mut fuzzer = NullSentinelFuzzer::new(config);
    let mut injection_count = 0;

    for _ in 0..30 {
        if fuzzer.next_call() {
            injection_count += 1;
        }
    }

    // Periodic with interval 3 over 30 calls: indices 0, 3, 6, 9, 12, 15, 18, 21, 24, 27 = 10
    assert_eq!(injection_count, 10, "Expected 10 injections");
}

/// Claim 10: No false positives on valid code
#[test]
fn claim_10_no_false_positives() {
    // Valid non-zero addresses should never produce errors
    for addr in [0x1000_u64, 0x2000, 0xFFFF_FFFF, u64::MAX] {
        let ptr = NonNullDevicePtr::<f32>::new(addr);
        assert!(ptr.is_ok(), "Valid address 0x{:x} must not produce false positive", addr);
    }
}

/// Test propagation tracker records call chains
#[test]
fn propagation_tracker_records_chain() {
    use trueno_cuda_edge::null_fuzzer::PropagationOutcome;

    let mut tracker = PropagationTracker::new();

    // Simulate a call chain: kernel_a -> kernel_b -> kernel_c
    tracker.enter("kernel_a".into(), 0);
    tracker.enter("kernel_b".into(), 1);
    tracker.enter("kernel_c".into(), 2);

    assert_eq!(tracker.current_depth(), 3);

    // Record outcome
    tracker.record(PropagationOutcome::Uncaught);

    let completed = tracker.completed();
    assert_eq!(completed.len(), 1);

    let (path, _outcome) = &completed[0];
    assert_eq!(path.depth(), 3);
    assert_eq!(path.injection_point().unwrap().function, "kernel_a");
    assert_eq!(path.final_use().unwrap().function, "kernel_c");
}
