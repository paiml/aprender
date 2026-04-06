//! GPU Edge-Case Testing with trueno-cuda-edge
//!
//! This test suite applies trueno-cuda-edge's falsification frameworks to
//! realizar's GPU code, verifying:
//!
//! - Null pointer handling in CUDA operations
//! - Quantization parity between CPU and GPU
//! - PTX verification for generated kernels
//! - Shared memory boundary safety
//!
//! These tests run without actual GPU hardware by testing the pure-Rust
//! type system guarantees and configuration validation.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use trueno_cuda_edge::{
    falsification::{all_claims, ClaimStatus, FalsificationReport, Framework},
    lifecycle_chaos::{
        ChaosScenario, ContextLeakDetector, DestructionOrdering, LifecycleChaosConfig,
    },
    null_fuzzer::{InjectionStrategy, NonNullDevicePtr, NullFuzzerConfig, NullSentinelFuzzer},
    ptx_poison::{default_mutators, PtxMutator, PtxVerifier, MINIMAL_VALID_PTX},
    quant_oracle::{check_values_parity, BoundaryValueGenerator, ParityConfig, QuantFormat},
    shmem_prober::{
        check_allocation, compute_sentinel_offsets, shared_memory_limit, AccessPattern,
        BankConflictInjector, ComputeCapability, SharedMemoryRegion,
    },
    supervisor::{
        GpuHealthMonitor, HealthAction, HeartbeatStatus, SupervisionStrategy, SupervisionTree,
    },
};

// ============================================================================
// F1: Null Pointer Sentinel Fuzzer Tests
// ============================================================================

mod null_fuzzer_integration {
    use super::*;

    /// Verify NonNullDevicePtr correctly rejects null addresses.
    /// This is critical for realizar's CUDA memory management.
    #[test]
    fn device_ptr_null_rejection() {
        // Simulate what realizar would do when allocating GPU memory
        let result = NonNullDevicePtr::<f32>::new(0);
        assert!(result.is_err(), "NonNullDevicePtr must reject address 0");

        // Valid addresses should succeed
        let valid = NonNullDevicePtr::<f32>::new(0x7f00_0000_0000);
        assert!(valid.is_ok());
        assert_eq!(valid.unwrap().addr(), 0x7f00_0000_0000);
    }

    /// Test injection strategy configuration for GPU memory testing.
    #[test]
    fn injection_strategy_for_batch_testing() {
        // Configure periodic injection to test every 10th CUDA call
        let config = NullFuzzerConfig {
            strategy: InjectionStrategy::Periodic { interval: 10 },
            total_calls: 1000,
            fail_fast: false,
        };

        let mut fuzzer = NullSentinelFuzzer::new(config);

        // First call (index 0) should inject
        assert!(fuzzer.next_call());
        // Calls 1-9 should not inject
        for _ in 1..10 {
            assert!(!fuzzer.next_call());
        }
        // Call 10 should inject again
        assert!(fuzzer.next_call());
    }

    /// Test size-threshold injection for large tensor allocations.
    #[test]
    fn size_threshold_for_large_allocations() {
        let strategy = InjectionStrategy::SizeThreshold {
            threshold_bytes: 1024 * 1024 * 1024, // 1 GB
        };

        // Size threshold requires context, so should_inject returns false
        assert!(!strategy.should_inject(0));
    }

    /// Test probabilistic injection for stress testing.
    #[test]
    fn probabilistic_injection_deterministic() {
        let strategy = InjectionStrategy::Probabilistic { probability: 0.1 };

        // 10% probability: inject when (call_index % 100) < 10
        assert!(strategy.should_inject(0)); // 0 < 10
        assert!(strategy.should_inject(5)); // 5 < 10
        assert!(!strategy.should_inject(10)); // 10 >= 10
        assert!(!strategy.should_inject(99)); // 99 >= 10
    }
}

// ============================================================================
// F2: Shared Memory Boundary Prober Tests
// ============================================================================

mod shmem_prober_integration {
    use super::*;

    /// Verify compute capability detection for realizar's GPU targeting.
    #[test]
    fn compute_capability_mapping() {
        // Common GPU architectures realizar targets
        let ampere = ComputeCapability::new(8, 0); // A100
        let hopper = ComputeCapability::new(9, 0); // H100

        // Shared memory limits via function
        assert_eq!(shared_memory_limit(ampere), 164 * 1024);
        assert_eq!(shared_memory_limit(hopper), 228 * 1024);
    }

    /// Test allocation validation for realizar's kernel launches.
    #[test]
    fn allocation_validation() {
        let ampere = ComputeCapability::new(8, 0);

        // 128 KB allocation should succeed on Ampere
        assert!(check_allocation(ampere, 128 * 1024).is_ok());

        // 200 KB should fail (exceeds 164 KB limit)
        assert!(check_allocation(ampere, 200 * 1024).is_err());
    }

    /// Test sentinel offset computation for boundary checking.
    #[test]
    fn sentinel_offsets_for_tensor_buffers() {
        let regions = vec![
            SharedMemoryRegion::new(0, 4096),    // 4 KB tensor buffer
            SharedMemoryRegion::new(4096, 2048), // 2 KB scratch space
        ];

        let offsets = compute_sentinel_offsets(&regions);

        // Each region produces a (before, after) tuple
        assert_eq!(offsets.len(), 2);

        // First region: sentinel before at 0, sentinel after at 0 + 4 + 4096 = 4100
        assert_eq!(offsets[0].0, 0);
        assert_eq!(offsets[0].1, 4100);

        // Second region: sentinel before at 4096, sentinel after at 4096 + 4 + 2048 = 6148
        assert_eq!(offsets[1].0, 4096);
        assert_eq!(offsets[1].1, 6148);
    }

    /// Test bank conflict detection for optimized memory access.
    #[test]
    fn bank_conflict_analysis() {
        let injector = BankConflictInjector::new();

        // Sequential access has no conflicts
        assert_eq!(
            injector.expected_serialization(AccessPattern::Sequential),
            1
        );

        // Full conflict has 32x serialization
        assert_eq!(
            injector.expected_serialization(AccessPattern::FullConflict),
            32
        );

        // Stride-2 access causes 2-way bank conflicts
        assert_eq!(injector.expected_serialization(AccessPattern::Stride2), 2);
    }

    /// Verify bank index calculation for memory layout optimization.
    #[test]
    fn bank_index_calculation() {
        let injector = BankConflictInjector::new();

        // Word at offset 0 → bank 0
        assert_eq!(injector.bank_for_offset(0), 0);
        // Word at offset 4 → bank 1
        assert_eq!(injector.bank_for_offset(4), 1);
        // Word at offset 128 → bank 0 (wraps)
        assert_eq!(injector.bank_for_offset(128), 0);
    }
}

// ============================================================================
// F3: Context Lifecycle Chaos Tests
// ============================================================================

mod lifecycle_chaos_integration {
    use super::*;

    /// Test all chaos scenarios for realizar's CUDA context management.
    #[test]
    fn chaos_scenarios_coverage() {
        let config = LifecycleChaosConfig::default();

        assert_eq!(config.scenarios.len(), 8);
        assert!(config.scenarios.contains(&ChaosScenario::DoubleDestroy));
        assert!(config.scenarios.contains(&ChaosScenario::UseAfterDestroy));
        assert!(config.scenarios.contains(&ChaosScenario::LeakedContext));
    }

    /// Test destruction ordering validation.
    #[test]
    fn destruction_ordering_validation() {
        // LIFO ordering (correct for CUDA contexts)
        let lifo = DestructionOrdering::new(vec![2, 1, 0]);
        assert!(lifo.is_reverse());

        // FIFO ordering (may cause issues)
        let fifo = DestructionOrdering::new(vec![0, 1, 2]);
        assert!(fifo.is_forward());
    }

    /// Test leak detection for realizar's GPU memory management.
    #[test]
    fn leak_detection_tolerance() {
        let detector = ContextLeakDetector::new();

        // No leak within 1 MB tolerance
        let report = detector.analyze(100_000_000, 100_500_000);
        assert!(!report.has_leaks());

        // Leak above tolerance
        let report = detector.analyze(100_000_000, 102_000_000);
        assert!(report.has_leaks());
    }
}

// ============================================================================
// F4: Quantization Parity Oracle Tests
// ============================================================================

mod quant_oracle_integration {
    use super::*;

    /// Test quantization format tolerances for realizar's Q4K/Q6K kernels.
    #[test]
    fn quantization_tolerances() {
        // Q4_K tolerance for 4-bit quantization
        assert!((QuantFormat::Q4K.tolerance() - 0.05).abs() < f64::EPSILON);

        // Q6_K tolerance for 6-bit quantization (tighter)
        assert!((QuantFormat::Q6K.tolerance() - 0.01).abs() < f64::EPSILON);

        // Q8_0 tolerance for 8-bit quantization
        assert!((QuantFormat::Q8_0.tolerance() - 0.005).abs() < f64::EPSILON);
    }

    /// Test boundary value generation for quantization testing.
    #[test]
    fn boundary_values_for_q4k() {
        let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);
        let universal = gen.universal_boundaries();

        // Must include critical values
        assert!(universal.iter().any(|v| v.is_nan()));
        assert!(universal.iter().any(|v| v.is_infinite()));
        assert!(universal.contains(&0.0));
    }

    /// Test parity configuration for CPU/GPU comparison.
    #[test]
    fn parity_config_for_inference() {
        let config = ParityConfig::new(QuantFormat::Q4K);

        assert_eq!(config.format, QuantFormat::Q4K);
        assert!((config.tolerance() - 0.05).abs() < f64::EPSILON);
    }

    /// Test quantization level counts for different formats.
    #[test]
    fn quantization_levels() {
        assert_eq!(QuantFormat::Q4K.levels(), 16);
        assert_eq!(QuantFormat::Q5K.levels(), 32);
        assert_eq!(QuantFormat::Q6K.levels(), 64);
        assert_eq!(QuantFormat::Q8_0.levels(), 256);
    }

    /// Test parity checking with real values.
    #[test]
    fn parity_check_identical_values() {
        let cpu = vec![1.0, 2.0, 3.0, 4.0];
        let gpu = vec![1.0, 2.0, 3.0, 4.0];
        let config = ParityConfig::new(QuantFormat::Q4K);

        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(report.passed());
    }

    /// Test parity checking with tolerance violations.
    #[test]
    fn parity_check_with_violations() {
        let cpu = vec![1.0, 2.0, 3.0, 4.0];
        let gpu = vec![1.0, 2.5, 3.0, 4.0]; // 0.5 diff at index 1

        let config = ParityConfig::new(QuantFormat::Q4K); // 0.05 tolerance

        let report = check_values_parity(&cpu, &gpu, &config);
        assert!(!report.passed());
        assert_eq!(report.violations.len(), 1);
    }
}

// ============================================================================
// F5: PTX Compilation Poison Trap Tests
// ============================================================================

mod ptx_poison_integration {
    use super::*;

    /// Verify PTX verifier catches invalid kernels.
    #[test]
    fn ptx_verification_for_generated_kernels() {
        let verifier = PtxVerifier::new();

        // Valid PTX should pass
        let result = verifier.verify(MINIMAL_VALID_PTX);
        assert!(result.is_ok());

        // Empty PTX should fail
        let result = verifier.verify("");
        assert!(result.is_err());
    }

    /// Test PTX structural checks.
    #[test]
    fn ptx_structural_requirements() {
        let verifier = PtxVerifier::new();

        // Missing .version
        let no_version = ".target sm_80\n.address_size 64\n.entry k() { ret; }";
        let errors = verifier.check_all(no_version);
        assert!(!errors.is_empty());

        // Missing .entry or .func
        let no_entry = ".version 7.0\n.target sm_80\n.address_size 64\n";
        let errors = verifier.check_all(no_entry);
        assert!(!errors.is_empty());
    }

    /// Test mutation operators for kernel testing.
    #[test]
    fn mutation_operators_available() {
        let mutators = default_mutators();
        assert_eq!(mutators.len(), 8);

        // Check key mutators are present
        assert!(mutators.contains(&PtxMutator::FlipAddSub));
        assert!(mutators.contains(&PtxMutator::FlipMulDiv));
        assert!(mutators.contains(&PtxMutator::InvertPredicate));
        assert!(mutators.contains(&PtxMutator::RemoveBarrier));
    }

    /// Test mutation application to PTX source.
    #[test]
    fn mutation_application() {
        let ptx = "add.f32 %f1, %f2, %f3;";

        // FlipAddSub should change add to sub
        let mutated = PtxMutator::FlipAddSub.apply(ptx);
        assert!(mutated.is_some());
        assert!(mutated.unwrap().contains("sub.f32"));
    }
}

// ============================================================================
// Supervisor Integration Tests
// ============================================================================

mod supervisor_integration {
    use super::*;

    /// Test supervision strategies for realizar's GPU worker management.
    #[test]
    fn supervision_strategies() {
        // One-for-one: only restart crashed worker
        assert!(SupervisionStrategy::OneForOne.is_isolated());

        // One-for-all: restart all workers on crash
        assert!(!SupervisionStrategy::OneForAll.is_isolated());

        // Rest-for-one: restart crashed + dependent workers
        assert!(!SupervisionStrategy::RestForOne.is_isolated());
    }

    /// Test supervision tree crash handling.
    #[test]
    fn supervision_tree_crash_recovery() {
        let mut tree = SupervisionTree::new(
            SupervisionStrategy::OneForOne,
            3, // 3 workers
        );

        // Simulate crash of worker 1 at time 0
        let action = tree.handle_crash(1, 0);

        // Should restart only worker 1
        match action {
            trueno_cuda_edge::supervisor::SupervisorAction::Restart(indices) => {
                assert_eq!(indices, vec![1]);
            },
            _ => panic!("Expected Restart action"),
        }
    }

    /// Test health monitoring thresholds.
    #[test]
    fn health_monitoring_thresholds() {
        let monitor = GpuHealthMonitor::builder()
            .max_missed(3)
            .throttle_temp(85)
            .shutdown_temp(95)
            .build();

        // Healthy status
        let action = monitor.check_status(HeartbeatStatus::Alive);
        assert_eq!(action, HealthAction::Healthy);

        // Dead status triggers shutdown
        let action = monitor.check_status(HeartbeatStatus::Dead);
        assert_eq!(action, HealthAction::Shutdown);

        // Thermal throttle
        let action = monitor.check_temperature(90);
        assert_eq!(action, HealthAction::Throttle);

        // Thermal shutdown
        let action = monitor.check_temperature(96);
        assert_eq!(action, HealthAction::Shutdown);
    }
}

// ============================================================================
// Falsification Protocol Integration
// ============================================================================

mod falsification_protocol {
    use super::*;

    /// Verify all 50 claims are registered.
    #[test]
    fn protocol_completeness() {
        let claims = all_claims();
        assert_eq!(claims.len(), 50);
    }

    /// Test framework coverage tracking.
    #[test]
    fn framework_coverage_tracking() {
        let mut report = FalsificationReport::new();

        // Mark some claims as verified
        report.mark_verified("NF-001");
        report.mark_verified("NF-002");
        report.mark_verified("SP-001");

        // Check coverage
        let coverage = report.coverage();
        assert!(coverage > 0.0);
        assert!(coverage < 1.0);
    }

    /// Test claim grouping by framework.
    #[test]
    fn claims_by_framework() {
        let report = FalsificationReport::new();
        let grouped = report.by_framework();

        // Each framework should have claims
        assert!(grouped.contains_key(&Framework::NullFuzzer));
        assert!(grouped.contains_key(&Framework::ShmemProber));
        assert!(grouped.contains_key(&Framework::LifecycleChaos));
        assert!(grouped.contains_key(&Framework::QuantOracle));
        assert!(grouped.contains_key(&Framework::PtxPoison));
    }

    /// Test status transitions.
    #[test]
    fn status_transitions() {
        let mut report = FalsificationReport::new();

        // Initially pending
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Pending));

        // Mark verified
        report.mark_verified("NF-001");
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Verified));

        // Mark another as violated
        report.mark_violated("NF-002");
        assert_eq!(report.status("NF-002"), Some(ClaimStatus::Violated));
    }
}
