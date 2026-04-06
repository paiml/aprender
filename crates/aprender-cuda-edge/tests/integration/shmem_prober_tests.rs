//! Integration tests for the Shared Memory Boundary Prober (TCE-SHMEM).
//!
//! Falsification tests from Section 8.2, F2 (claims 11-20).

#![allow(clippy::unwrap_used)]

use trueno_cuda_edge::shmem_prober::{
    check_allocation, check_sentinels, compute_sentinel_offsets, AccessPattern,
    BankConflictInjector, ComputeCapability, SharedMemoryRegion, SENTINEL_AFTER, SENTINEL_BEFORE,
};

/// Claim 11: Sentinel value is not overwritten by correct code
#[test]
fn claim_11_sentinel_unchanged_by_correct_code() {
    // Simulate correct behavior: sentinel values remain unchanged
    let violations = check_sentinels(SENTINEL_BEFORE, SENTINEL_AFTER);
    assert!(
        violations.is_empty(),
        "Correct code must not corrupt sentinels"
    );
}

/// Claim 12: Off-by-one write overwrites sentinel
#[test]
fn claim_12_off_by_one_detected() {
    // Simulate off-by-one: underflow sentinel corrupted
    let corrupted_before = 0xBAAD_F00D_u32;
    let violations = check_sentinels(corrupted_before, SENTINEL_AFTER);
    assert_eq!(violations.len(), 1, "Off-by-one must be detected");
}

/// Claim 13: 32-way bank conflict detected
#[test]
fn claim_13_full_bank_conflict_detected() {
    let injector = BankConflictInjector::default();
    let serialization = injector.expected_serialization(AccessPattern::FullConflict);
    assert_eq!(
        serialization, 32,
        "32-way bank conflict must have serialization factor 32"
    );
}

/// Claim 14: Padded access eliminates bank conflicts
#[test]
fn claim_14_padded_access_no_conflicts() {
    let injector = BankConflictInjector::default();
    let serialization = injector.expected_serialization(AccessPattern::Padded);
    assert_eq!(
        serialization, 1,
        "Padded access must have serialization factor 1"
    );
}

/// Claim 15: Shared memory overflow caught
#[test]
fn claim_15_shmem_overflow_caught() {
    let cc = ComputeCapability::new(7, 0); // Volta: 96KB limit
    let requested = 256 * 1024; // 256KB

    let result = check_allocation(cc, requested);
    assert!(result.is_err(), "256KB allocation on 96KB SM must fail");
}

/// Claim 17: Bank conflict levels distinguished
#[test]
fn claim_17_bank_conflict_levels_distinguished() {
    let injector = BankConflictInjector::default();

    let levels = [
        (AccessPattern::Sequential, 1),
        (AccessPattern::Stride2, 2),
        (AccessPattern::FullConflict, 32),
        (AccessPattern::Stride32, 1), // broadcast
        (AccessPattern::Padded, 1),
    ];

    for (pattern, expected) in levels {
        let actual = injector.expected_serialization(pattern);
        assert_eq!(
            actual, expected,
            "Pattern {:?} should have serialization {}",
            pattern, expected
        );
    }
}

/// Claim 18: Prober works across compute capabilities
#[test]
fn claim_18_multiple_compute_capabilities() {
    let capabilities = [
        (ComputeCapability::new(7, 0), 96 * 1024),  // Volta
        (ComputeCapability::new(8, 0), 164 * 1024), // Ampere
        (ComputeCapability::new(9, 0), 228 * 1024), // Hopper
    ];

    for (cc, expected_limit) in capabilities {
        let limit = trueno_cuda_edge::shmem_prober::shared_memory_limit(cc);
        assert_eq!(
            limit, expected_limit,
            "Compute capability {} should have limit {}",
            cc, expected_limit
        );
    }
}

/// Claim 19: Zero false positives on correct kernels
#[test]
fn claim_19_no_false_positives() {
    // All correct sentinel checks must pass
    for _ in 0..10 {
        let violations = check_sentinels(SENTINEL_BEFORE, SENTINEL_AFTER);
        assert!(violations.is_empty(), "No false positives allowed");
    }
}

/// Test sentinel offset computation
#[test]
fn sentinel_offsets_computed_correctly() {
    let regions = vec![
        SharedMemoryRegion::new(0, 1024),
        SharedMemoryRegion::new(1032, 2048), // After first region + sentinels
    ];

    let offsets = compute_sentinel_offsets(&regions);
    assert_eq!(offsets.len(), 2);

    // First region: before at 0, after at 4 + 1024 = 1028
    assert_eq!(offsets[0].0, 0);
    assert_eq!(offsets[0].1, 4 + 1024);
}

/// Test bank index cycling
#[test]
fn bank_index_cycles_every_32_words() {
    let injector = BankConflictInjector::default();

    // Bank 0 at offset 0
    assert_eq!(injector.bank_for_offset(0), 0);
    // Bank 1 at offset 4
    assert_eq!(injector.bank_for_offset(4), 1);
    // Bank 31 at offset 124
    assert_eq!(injector.bank_for_offset(124), 31);
    // Bank 0 again at offset 128 (wraps)
    assert_eq!(injector.bank_for_offset(128), 0);
}
