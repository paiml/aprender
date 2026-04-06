#![allow(
    clippy::unwrap_used,
    clippy::float_cmp,
    clippy::manual_range_contains,
    clippy::needless_borrows_for_generic_args
)]
//! Integration tests for trueno-cuda-edge GPU edge-case testing framework
//!
//! These tests validate that batuta can leverage trueno-cuda-edge for:
//! - Orchestration pipeline GPU validation
//! - Backend selection verification
//! - Quantization parity in ML converters
//! - PTX kernel validation for trueno integration

use trueno_cuda_edge::falsification::{
    all_claims, claims_for_framework, ClaimStatus, FalsificationReport, Framework,
};
use trueno_cuda_edge::lifecycle_chaos::{ChaosScenario, ContextLeakDetector, Leak};
use trueno_cuda_edge::null_fuzzer::{
    InjectionStrategy, NonNullDevicePtr, NullFuzzerConfig, NullSentinelFuzzer,
};
use trueno_cuda_edge::ptx_poison::{default_mutators, PtxMutator, PtxVerifier, MINIMAL_VALID_PTX};
use trueno_cuda_edge::quant_oracle::{
    check_values_parity, BoundaryValueGenerator, ParityConfig, QuantFormat,
};
use trueno_cuda_edge::shmem_prober::{
    check_allocation, compute_sentinel_offsets, shared_memory_limit, AccessPattern,
    ComputeCapability, SharedMemoryRegion,
};
use trueno_cuda_edge::supervisor::{
    GpuHealthMonitor, HealthAction, HeartbeatStatus, SupervisionStrategy, SupervisionTree,
    SupervisorAction,
};

// ============================================================================
// Orchestration Pipeline GPU Validation Tests
// ============================================================================

#[test]
fn test_pipeline_null_pointer_safety() {
    // Batuta orchestrates GPU workloads - verify null safety
    assert!(NonNullDevicePtr::<f32>::new(0).is_err());
    assert!(NonNullDevicePtr::<f32>::new(0x7f00_0000_0000).is_ok());
}

#[test]
fn test_pipeline_injection_strategies() {
    // Test injection strategies for fault injection during orchestration
    let strategies = [
        InjectionStrategy::Periodic { interval: 10 },
        InjectionStrategy::SizeThreshold { threshold_bytes: 1024 },
        InjectionStrategy::Probabilistic { probability: 0.1 },
        InjectionStrategy::Targeted { arg_indices: vec![0, 2] },
    ];

    for strategy in strategies {
        let config = NullFuzzerConfig { strategy, total_calls: 100, fail_fast: false };
        let fuzzer = NullSentinelFuzzer::new(config);
        assert!(fuzzer.config().total_calls == 100);
    }
}

#[test]
fn test_pipeline_fuzzer_periodic_injection() {
    let config = NullFuzzerConfig {
        strategy: InjectionStrategy::Periodic { interval: 5 },
        total_calls: 20,
        fail_fast: false,
    };

    let fuzzer = NullSentinelFuzzer::new(config);

    // Use strategy's should_inject directly for stateless checking
    let strategy = &fuzzer.config().strategy;
    assert!(strategy.should_inject(0)); // 0
    assert!(!strategy.should_inject(1)); // 1
    assert!(!strategy.should_inject(2)); // 2
    assert!(!strategy.should_inject(3)); // 3
    assert!(!strategy.should_inject(4)); // 4
    assert!(strategy.should_inject(5)); // 5
}

// ============================================================================
// Backend Selection Verification Tests
// ============================================================================

#[test]
fn test_backend_shared_memory_limits() {
    // Batuta's backend selection considers GPU capabilities
    let pascal = ComputeCapability::new(6, 0);
    let volta = ComputeCapability::new(7, 0);
    let ampere = ComputeCapability::new(8, 0);
    let hopper = ComputeCapability::new(9, 0);

    assert_eq!(shared_memory_limit(pascal), 48 * 1024);
    assert_eq!(shared_memory_limit(volta), 96 * 1024);
    assert_eq!(shared_memory_limit(ampere), 164 * 1024);
    assert_eq!(shared_memory_limit(hopper), 228 * 1024);
}

#[test]
fn test_backend_allocation_validation() {
    let ampere = ComputeCapability::new(8, 0);

    // Valid allocation
    assert!(check_allocation(ampere, 128 * 1024).is_ok());

    // At limit
    assert!(check_allocation(ampere, 164 * 1024).is_ok());

    // Over limit
    assert!(check_allocation(ampere, 165 * 1024).is_err());
}

#[test]
fn test_backend_bank_conflict_patterns() {
    // Different access patterns affect backend performance
    let patterns = [
        AccessPattern::Sequential,
        AccessPattern::Stride2,
        AccessPattern::Stride32,
        AccessPattern::FullConflict,
        AccessPattern::Padded,
    ];

    for pattern in patterns {
        // Each pattern has a serialization factor
        let factor = pattern.serialization_factor();
        assert!(factor >= 1 && factor <= 32);
    }
}

#[test]
fn test_backend_sentinel_offsets() {
    let regions = vec![SharedMemoryRegion::new(0, 1024)];
    let offsets = compute_sentinel_offsets(&regions);

    assert_eq!(offsets.len(), 1);
    // Offsets are (before_offset, after_offset) tuples
    let (before, after) = offsets[0];
    assert_eq!(before, 0);
    assert!(after > before);
}

// ============================================================================
// ML Converter Quantization Parity Tests
// ============================================================================

#[test]
fn test_converter_format_tolerances() {
    // Batuta's ML converters need format-specific tolerances
    assert_eq!(QuantFormat::Q4K.tolerance(), 0.05);
    assert_eq!(QuantFormat::Q5K.tolerance(), 0.02);
    assert_eq!(QuantFormat::Q6K.tolerance(), 0.01);
    assert_eq!(QuantFormat::Q8_0.tolerance(), 0.005);
    assert_eq!(QuantFormat::F16.tolerance(), 0.001);
    assert_eq!(QuantFormat::F32.tolerance(), f64::EPSILON);
}

#[test]
fn test_converter_boundary_values() {
    // Test boundary values for quantization edge cases
    let gen = BoundaryValueGenerator::new(QuantFormat::Q4K);

    let universal = gen.universal_boundaries();
    assert!(universal.contains(&0.0));
    assert!(universal.iter().any(|v| v.is_nan()));
    assert!(universal.iter().any(|v| v.is_infinite() && v.is_sign_positive()));
    assert!(universal.iter().any(|v| v.is_infinite() && v.is_sign_negative()));

    let format_bounds = gen.format_boundaries();
    // Q4K has 16 levels × 2 signs = 32 values
    assert_eq!(format_bounds.len(), 32);
}

#[test]
fn test_converter_parity_check_pass() {
    let config = ParityConfig::new(QuantFormat::Q4K);

    let cpu = vec![1.0, 2.0, 3.0, 4.0];
    let gpu = vec![1.01, 2.01, 3.01, 4.01]; // Within 5% tolerance

    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(report.passed());
}

#[test]
fn test_converter_parity_check_fail() {
    let config = ParityConfig::new(QuantFormat::Q4K);

    let cpu = vec![1.0, 2.0, 3.0, 4.0];
    let gpu = vec![1.0, 2.5, 3.0, 4.0]; // 25% difference at index 1

    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(!report.passed());
    assert_eq!(report.violations.len(), 1);
    assert_eq!(report.violations[0].index, 1);
}

#[test]
fn test_converter_nan_handling() {
    let config = ParityConfig::new(QuantFormat::Q4K);

    let cpu = vec![f64::NAN, 1.0];
    let gpu = vec![f64::NAN, 1.0];

    let report = check_values_parity(&cpu, &gpu, &config);
    assert!(report.passed()); // NaN == NaN for parity purposes
}

// ============================================================================
// Trueno PTX Kernel Validation Tests
// ============================================================================

#[test]
fn test_trueno_ptx_verification() {
    let verifier = PtxVerifier::new();

    // Valid PTX passes
    assert!(verifier.verify(MINIMAL_VALID_PTX).is_ok());

    // Invalid PTX fails
    assert!(verifier.verify("").is_err());
    assert!(verifier.verify("invalid").is_err());
}

#[test]
fn test_trueno_ptx_structural_checks() {
    let verifier = PtxVerifier::new();

    let missing_version = ".target sm_80\n.address_size 64\n.entry k() { ret; }";
    assert!(!verifier.check_all(missing_version).is_empty());

    let missing_target = ".version 7.0\n.address_size 64\n.entry k() { ret; }";
    assert!(!verifier.check_all(missing_target).is_empty());

    let unbalanced = ".version 7.0\n.target sm_80\n.address_size 64\n.entry k() {";
    assert!(!verifier.check_all(unbalanced).is_empty());
}

#[test]
fn test_trueno_ptx_mutation_operators() {
    let mutators = default_mutators();
    assert_eq!(mutators.len(), 8);

    // Test specific mutations
    let add_src = "add.f32 %f1, %f2, %f3;";
    let mutated = PtxMutator::FlipAddSub.apply(add_src);
    assert!(mutated.is_some());
    assert!(mutated.unwrap().contains("sub.f32"));
}

#[test]
fn test_trueno_ptx_barrier_removal() {
    let kernel =
        ".version 7.0\n.target sm_80\n.address_size 64\n.entry k() {\n    bar.sync 0;\n    ret;\n}";

    let mutated = PtxMutator::RemoveBarrier.apply(kernel);
    assert!(mutated.is_some());
    assert!(!mutated.unwrap().contains("bar.sync"));
}

// ============================================================================
// GPU Context Lifecycle Tests
// ============================================================================

#[test]
fn test_lifecycle_chaos_scenarios() {
    let scenarios = ChaosScenario::all();
    assert_eq!(scenarios.len(), 8);

    assert!(scenarios.contains(&ChaosScenario::DoubleDestroy));
    assert!(scenarios.contains(&ChaosScenario::UseAfterDestroy));
    assert!(scenarios.contains(&ChaosScenario::LeakedContext));
}

#[test]
fn test_lifecycle_leak_detection() {
    let detector = ContextLeakDetector::new();

    // No leak within tolerance
    let report = detector.analyze(100 * 1024 * 1024, 100 * 1024 * 1024 + 500 * 1024);
    assert!(report.leaks.is_empty());

    // Leak detected above tolerance
    let report = detector.analyze(100 * 1024 * 1024, 102 * 1024 * 1024);
    assert!(!report.leaks.is_empty());
}

#[test]
fn test_lifecycle_leak_types() {
    let leaks = [
        Leak::Context { context_id: 1 },
        Leak::Stream { stream_id: 2 },
        Leak::Memory { bytes: 1024 },
    ];

    for leak in &leaks {
        match leak {
            Leak::Context { context_id } => assert!(*context_id > 0),
            Leak::Stream { stream_id } => assert!(*stream_id > 0),
            Leak::Memory { bytes } => assert!(*bytes > 0),
        }
    }
}

// ============================================================================
// Supervision and Health Monitoring Tests
// ============================================================================

#[test]
fn test_supervision_strategies() {
    // OneForOne: isolated restarts
    let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);
    let action = tree.handle_crash(1, 0);
    assert!(matches!(action, SupervisorAction::Restart(ref indices) if indices == &[1]));

    // OneForAll: restart all
    let mut tree = SupervisionTree::new(SupervisionStrategy::OneForAll, 4);
    let action = tree.handle_crash(1, 0);
    assert!(matches!(action, SupervisorAction::Restart(ref indices) if indices.len() == 4));

    // RestForOne: restart crashed + later workers
    let mut tree = SupervisionTree::new(SupervisionStrategy::RestForOne, 4);
    let action = tree.handle_crash(1, 0);
    assert!(matches!(action, SupervisorAction::Restart(ref indices) if indices.len() >= 3));
}

#[test]
fn test_health_monitor_temperature() {
    let monitor =
        GpuHealthMonitor::builder().max_missed(3).throttle_temp(85).shutdown_temp(95).build();

    // Normal operation
    assert_eq!(monitor.check_temperature(70), HealthAction::Healthy);

    // Approaching throttle
    assert_eq!(monitor.check_temperature(85), HealthAction::Throttle);

    // Critical temperature
    assert_eq!(monitor.check_temperature(95), HealthAction::Shutdown);
}

#[test]
fn test_health_monitor_heartbeat_status() {
    let monitor =
        GpuHealthMonitor::builder().max_missed(3).throttle_temp(85).shutdown_temp(95).build();

    // Alive is healthy
    assert_eq!(monitor.check_status(HeartbeatStatus::Alive), HealthAction::Healthy);

    // Within tolerance
    assert_eq!(monitor.check_status(HeartbeatStatus::MissedBeats(2)), HealthAction::Healthy);

    // At threshold
    assert_eq!(monitor.check_status(HeartbeatStatus::MissedBeats(3)), HealthAction::RestartWorker);

    // Above threshold
    assert_eq!(monitor.check_status(HeartbeatStatus::MissedBeats(5)), HealthAction::RestartWorker);

    // Dead worker
    assert_eq!(monitor.check_status(HeartbeatStatus::Dead), HealthAction::Shutdown);
}

// ============================================================================
// Falsification Protocol Tests
// ============================================================================

#[test]
fn test_falsification_claim_count() {
    let claims = all_claims();
    assert_eq!(claims.len(), 50);
}

#[test]
fn test_falsification_framework_claims() {
    let nf = claims_for_framework(Framework::NullFuzzer);
    let sp = claims_for_framework(Framework::ShmemProber);
    let lc = claims_for_framework(Framework::LifecycleChaos);
    let qo = claims_for_framework(Framework::QuantOracle);
    let pp = claims_for_framework(Framework::PtxPoison);
    let sv = claims_for_framework(Framework::Supervisor);

    assert_eq!(nf.len(), 10);
    assert_eq!(sp.len(), 10);
    assert_eq!(lc.len(), 8);
    assert_eq!(qo.len(), 8);
    assert_eq!(pp.len(), 8);
    assert_eq!(sv.len(), 6);
}

#[test]
fn test_falsification_report_tracking() {
    let mut report = FalsificationReport::new();

    // Initial state
    assert_eq!(report.coverage(), 0.0);
    assert!(!report.is_complete());

    // Mark some verified
    report.mark_verified("NF-001");
    report.mark_verified("NF-002");
    assert!(report.coverage() > 0.0);

    // Mark one violated
    report.mark_violated("SP-001");
    let violated = report.violated_claims();
    assert_eq!(violated.len(), 1);
    assert!(violated.contains(&"SP-001".to_string()));
}

#[test]
fn test_falsification_complete_coverage() {
    let mut report = FalsificationReport::new();

    // Mark all claims
    for claim in all_claims() {
        report.mark_verified(claim.id);
    }

    assert!(report.is_complete());
    assert_eq!(report.coverage(), 1.0);
}

#[test]
fn test_falsification_framework_grouping() {
    let mut report = FalsificationReport::new();

    // Mark all NullFuzzer claims
    for claim in claims_for_framework(Framework::NullFuzzer) {
        report.mark_verified(claim.id);
    }

    let grouped = report.by_framework();
    let nf_claims = grouped.get(&Framework::NullFuzzer).unwrap();

    assert_eq!(nf_claims.len(), 10);
    assert!(nf_claims.iter().all(|(_, status)| *status == ClaimStatus::Verified));
}
