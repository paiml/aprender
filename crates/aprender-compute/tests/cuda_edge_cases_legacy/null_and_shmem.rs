//! F1: Null Pointer Sentinel Fuzzer + F2: Shared Memory Boundary Prober

use trueno_cuda_edge::{
    null_fuzzer::{InjectionStrategy, NonNullDevicePtr, NullFuzzerConfig, NullSentinelFuzzer},
    shmem_prober::{
        check_allocation, compute_sentinel_offsets, shared_memory_limit, AccessPattern,
        BankConflictInjector, ComputeCapability, SharedMemoryRegion,
    },
};

// ============================================================================
// F1: Null Pointer Sentinel Fuzzer — Device Memory Safety
// ============================================================================

mod null_fuzzer_tests {
    use super::*;

    /// Verify NonNullDevicePtr enforces non-null constraint at construction.
    /// Critical for trueno's GPU memory allocation wrappers.
    #[test]
    fn device_ptr_construction_safety() {
        // Null address must be rejected
        let null_result = NonNullDevicePtr::<f32>::new(0);
        assert!(null_result.is_err());

        // Valid GPU address should succeed
        let valid_result = NonNullDevicePtr::<f32>::new(0x7f00_0000_0000);
        assert!(valid_result.is_ok());

        // Verify address is preserved
        let ptr = valid_result.unwrap();
        assert_eq!(ptr.addr(), 0x7f00_0000_0000);
    }

    /// Test type-level safety for different element types.
    #[test]
    fn device_ptr_type_safety() {
        // f32 tensor buffer
        let f32_ptr = NonNullDevicePtr::<f32>::new(0x1000).unwrap();
        assert_eq!(f32_ptr.addr(), 0x1000);

        // f16 tensor buffer (different type parameter)
        let f16_ptr = NonNullDevicePtr::<u16>::new(0x2000).unwrap();
        assert_eq!(f16_ptr.addr(), 0x2000);

        // u8 quantized buffer
        let u8_ptr = NonNullDevicePtr::<u8>::new(0x3000).unwrap();
        assert_eq!(u8_ptr.addr(), 0x3000);
    }

    /// Test injection strategies for GPU kernel fuzzing.
    #[test]
    fn injection_strategies_for_kernel_calls() {
        // Periodic: inject every N kernel launches
        let periodic = InjectionStrategy::Periodic { interval: 100 };
        assert!(periodic.should_inject(0));
        assert!(periodic.should_inject(100));
        assert!(!periodic.should_inject(50));

        // Size threshold: inject for large allocations
        let size_based = InjectionStrategy::SizeThreshold {
            threshold_bytes: 1024 * 1024 * 1024, // 1 GB
        };
        // Requires context, returns false without it
        assert!(!size_based.should_inject(0));

        // Probabilistic: deterministic for reproducibility
        let prob = InjectionStrategy::Probabilistic { probability: 0.25 };
        // 25% = inject when (idx % 100) < 25
        assert!(prob.should_inject(0));
        assert!(prob.should_inject(24));
        assert!(!prob.should_inject(25));
    }

    /// Test fuzzer state machine for GPU call sequences.
    #[test]
    fn fuzzer_call_sequence() {
        let config = NullFuzzerConfig {
            strategy: InjectionStrategy::Periodic { interval: 5 },
            total_calls: 100,
            fail_fast: false,
        };

        let mut fuzzer = NullSentinelFuzzer::new(config);

        // Track injection pattern
        let mut injections = Vec::new();
        for _ in 0..20 {
            injections.push(fuzzer.next_call());
        }

        // Should inject at 0, 5, 10, 15
        assert!(injections[0]);
        assert!(!injections[1]);
        assert!(injections[5]);
        assert!(injections[10]);
        assert!(injections[15]);
    }
}

// ============================================================================
// F2: Shared Memory Boundary Prober — GPU Memory Constraints
// ============================================================================

mod shmem_prober_tests {
    use super::*;

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
}
