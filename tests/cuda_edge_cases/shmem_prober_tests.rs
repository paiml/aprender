// ============================================================================
// F2: Shared Memory Boundary Prober -- GPU Memory Constraints
// ============================================================================

use trueno_cuda_edge::shmem_prober::{
    check_allocation, compute_sentinel_offsets, shared_memory_limit, AccessPattern,
    BankConflictInjector, ComputeCapability, SharedMemoryRegion,
};

/// Test compute capability to shared memory mapping.
/// Trueno targets Volta (7.x), Ampere (8.x), Hopper (9.x).
#[test]
fn compute_capability_shmem_limits() {
    // Volta/Turing (SM 7.x): 96 KB
    let volta = ComputeCapability::new(7, 0);
    assert_eq!(shared_memory_limit(volta), 96 * 1024);

    // Ampere (SM 8.x): 164 KB
    let ampere = ComputeCapability::new(8, 0);
    assert_eq!(shared_memory_limit(ampere), 164 * 1024);

    // Hopper (SM 9.x): 228 KB
    let hopper = ComputeCapability::new(9, 0);
    assert_eq!(shared_memory_limit(hopper), 228 * 1024);

    // Older (fallback): 48 KB
    let kepler = ComputeCapability::new(3, 5);
    assert_eq!(shared_memory_limit(kepler), 48 * 1024);
}

/// Test allocation validation for GPU kernels.
#[test]
fn allocation_bounds_checking() {
    let ampere = ComputeCapability::new(8, 0);
    let limit = shared_memory_limit(ampere);

    // At limit: OK
    assert!(check_allocation(ampere, limit).is_ok());

    // Below limit: OK
    assert!(check_allocation(ampere, limit - 1).is_ok());

    // Above limit: Error
    assert!(check_allocation(ampere, limit + 1).is_err());
}

/// Test bank conflict analysis for optimized memory access.
#[test]
fn bank_conflict_patterns() {
    let injector = BankConflictInjector::new();

    // Sequential: no conflicts (ideal)
    assert_eq!(injector.expected_serialization(AccessPattern::Sequential), 1);

    // Stride-2: 2-way conflicts
    assert_eq!(injector.expected_serialization(AccessPattern::Stride2), 2);

    // Full conflict: 32-way serialization (worst case)
    assert_eq!(injector.expected_serialization(AccessPattern::FullConflict), 32);

    // Padded: no conflicts (bank conflict avoidance)
    assert_eq!(injector.expected_serialization(AccessPattern::Padded), 1);

    // Stride-32 (broadcast): no conflicts
    assert_eq!(injector.expected_serialization(AccessPattern::Stride32), 1);
}

/// Test sentinel placement for boundary detection.
#[test]
fn sentinel_boundary_detection() {
    let regions = vec![
        SharedMemoryRegion::new(0, 1024),    // 1 KB region
        SharedMemoryRegion::new(1024, 2048), // 2 KB region
    ];

    let offsets = compute_sentinel_offsets(&regions);
    assert_eq!(offsets.len(), 2);

    // Region 0: before=0, after=0+4+1024=1028
    assert_eq!(offsets[0], (0, 1028));

    // Region 1: before=1024, after=1024+4+2048=3076
    assert_eq!(offsets[1], (1024, 3076));
}

/// Test bank index calculation for memory layout.
#[test]
fn bank_index_calculation() {
    let injector = BankConflictInjector::new();

    // 32 banks, 4-byte words
    // Bank = (offset / 4) % 32
    assert_eq!(injector.bank_for_offset(0), 0);
    assert_eq!(injector.bank_for_offset(4), 1);
    assert_eq!(injector.bank_for_offset(124), 31);
    assert_eq!(injector.bank_for_offset(128), 0); // wraps
}
