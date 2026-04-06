//! F5: PTX Compilation Poison Trap + Supervisor Integration + Falsification Protocol

use trueno_cuda_edge::{
    falsification::{all_claims, ClaimStatus, FalsificationReport, Framework},
    ptx_poison::{default_mutators, PtxMutator, PtxVerifier, MINIMAL_VALID_PTX},
    supervisor::{
        GpuHealthMonitor, HealthAction, HeartbeatStatus, SupervisionStrategy, SupervisionTree,
    },
};

// ============================================================================
// F5: PTX Compilation Poison Trap — Kernel Verification
// ============================================================================

mod ptx_poison_tests {
    use super::*;

    /// Test PTX structural verification.
    #[test]
    fn ptx_structural_verification() {
        let verifier = PtxVerifier::new();

        // Valid PTX passes
        let result = verifier.verify(MINIMAL_VALID_PTX);
        assert!(result.is_ok());

        // Empty PTX fails
        let result = verifier.verify("");
        assert!(result.is_err());

        // Missing .version fails
        let no_version = ".target sm_80\n.address_size 64\n.entry k() { ret; }";
        let errors = verifier.check_all(no_version);
        assert!(!errors.is_empty());
    }

    /// Test mutation operators for kernel testing.
    #[test]
    fn mutation_operators() {
        let mutators = default_mutators();
        assert_eq!(mutators.len(), 8);

        // Arithmetic mutations
        assert!(mutators.contains(&PtxMutator::FlipAddSub));
        assert!(mutators.contains(&PtxMutator::FlipMulDiv));

        // Control flow mutations
        assert!(mutators.contains(&PtxMutator::InvertPredicate));
        assert!(mutators.contains(&PtxMutator::RemoveBarrier));

        // Precision mutations
        assert!(mutators.contains(&PtxMutator::WidenPrecision));
    }

    /// Test mutation application to PTX source.
    #[test]
    fn mutation_application() {
        // FlipAddSub: add → sub
        let ptx = "add.f32 %f1, %f2, %f3;";
        let mutated = PtxMutator::FlipAddSub.apply(ptx);
        assert!(mutated.is_some());
        assert!(mutated.unwrap().contains("sub.f32"));

        // FlipMulDiv: mul → div
        let ptx = "mul.f32 %f1, %f2, %f3;";
        let mutated = PtxMutator::FlipMulDiv.apply(ptx);
        assert!(mutated.is_some());
        assert!(mutated.unwrap().contains("div.f32"));

        // InvertPredicate: setp.lt → setp.ge
        let ptx = "setp.lt.f32 %p1, %f1, %f2;";
        let mutated = PtxMutator::InvertPredicate.apply(ptx);
        assert!(mutated.is_some());
        assert!(mutated.unwrap().contains("setp.ge"));
    }

    /// Test PTX verification catches common errors.
    #[test]
    fn ptx_common_errors() {
        let verifier = PtxVerifier::new();

        // Missing .target
        let no_target = ".version 7.0\n.address_size 64\n.entry k() { ret; }";
        let errors = verifier.check_all(no_target);
        assert!(!errors.is_empty());

        // Missing .address_size
        let no_addr = ".version 7.0\n.target sm_80\n.entry k() { ret; }";
        let errors = verifier.check_all(no_addr);
        assert!(!errors.is_empty());

        // Missing entry point
        let no_entry = ".version 7.0\n.target sm_80\n.address_size 64\n";
        let errors = verifier.check_all(no_entry);
        assert!(!errors.is_empty());
    }
}

// ============================================================================
// Supervisor Integration — GPU Worker Management
// ============================================================================

mod supervisor_tests {
    use super::*;

    /// Test supervision strategies for GPU workers.
    #[test]
    fn supervision_strategies() {
        // One-for-one: isolated restarts
        assert!(SupervisionStrategy::OneForOne.is_isolated());

        // One-for-all: restart all on any failure
        assert!(!SupervisionStrategy::OneForAll.is_isolated());

        // Rest-for-one: restart crashed + dependents
        assert!(!SupervisionStrategy::RestForOne.is_isolated());
    }

    /// Test supervision tree crash handling.
    #[test]
    fn supervision_tree_operations() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::OneForOne, 4);

        // Crash worker 2 at time 0
        let action = tree.handle_crash(2, 0);
        match action {
            trueno_cuda_edge::supervisor::SupervisorAction::Restart(indices) => {
                assert_eq!(indices, vec![2]);
            }
            _ => panic!("Expected Restart action"),
        }
    }

    /// Test one-for-all strategy.
    #[test]
    fn one_for_all_restarts() {
        let mut tree = SupervisionTree::new(SupervisionStrategy::OneForAll, 3);

        let action = tree.handle_crash(1, 0);
        match action {
            trueno_cuda_edge::supervisor::SupervisorAction::Restart(indices) => {
                assert_eq!(indices, vec![0, 1, 2]);
            }
            _ => panic!("Expected Restart action"),
        }
    }

    /// Test health monitoring for GPU workers.
    #[test]
    fn health_monitoring() {
        let monitor =
            GpuHealthMonitor::builder().max_missed(3).throttle_temp(85).shutdown_temp(95).build();

        // Alive: healthy
        assert_eq!(monitor.check_status(HeartbeatStatus::Alive), HealthAction::Healthy);

        // Missed beats below threshold: healthy
        assert_eq!(monitor.check_status(HeartbeatStatus::MissedBeats(2)), HealthAction::Healthy);

        // Missed beats at threshold: restart
        assert_eq!(
            monitor.check_status(HeartbeatStatus::MissedBeats(3)),
            HealthAction::RestartWorker
        );

        // Dead: shutdown
        assert_eq!(monitor.check_status(HeartbeatStatus::Dead), HealthAction::Shutdown);
    }

    /// Test thermal monitoring thresholds.
    #[test]
    fn thermal_monitoring() {
        let monitor = GpuHealthMonitor::new(3, 85, 95);

        // Below throttle: healthy
        assert_eq!(monitor.check_temperature(70), HealthAction::Healthy);

        // At throttle threshold: throttle
        assert_eq!(monitor.check_temperature(85), HealthAction::Throttle);

        // Between throttle and shutdown: throttle
        assert_eq!(monitor.check_temperature(90), HealthAction::Throttle);

        // At shutdown threshold: shutdown
        assert_eq!(monitor.check_temperature(95), HealthAction::Shutdown);
    }
}

// ============================================================================
// Falsification Protocol — Coverage Tracking
// ============================================================================

mod falsification_tests {
    use super::*;

    /// Verify 50-point protocol completeness.
    #[test]
    fn protocol_completeness() {
        let claims = all_claims();
        assert_eq!(claims.len(), 50);
    }

    /// Test claim framework distribution.
    #[test]
    fn framework_distribution() {
        let claims = all_claims();

        let null_fuzzer = claims.iter().filter(|c| c.framework == Framework::NullFuzzer).count();
        let shmem = claims.iter().filter(|c| c.framework == Framework::ShmemProber).count();
        let lifecycle = claims.iter().filter(|c| c.framework == Framework::LifecycleChaos).count();
        let quant = claims.iter().filter(|c| c.framework == Framework::QuantOracle).count();
        let ptx = claims.iter().filter(|c| c.framework == Framework::PtxPoison).count();
        let supervisor = claims.iter().filter(|c| c.framework == Framework::Supervisor).count();

        assert_eq!(null_fuzzer, 10);
        assert_eq!(shmem, 10);
        assert_eq!(lifecycle, 8);
        assert_eq!(quant, 8);
        assert_eq!(ptx, 8);
        assert_eq!(supervisor, 6);
    }

    /// Test report status tracking.
    #[test]
    fn report_status_tracking() {
        let mut report = FalsificationReport::new();

        // All start pending
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Pending));

        // Mark verified
        report.mark_verified("NF-001");
        assert_eq!(report.status("NF-001"), Some(ClaimStatus::Verified));

        // Mark violated
        report.mark_violated("NF-002");
        assert_eq!(report.status("NF-002"), Some(ClaimStatus::Violated));

        // Coverage increases
        assert!(report.coverage() > 0.0);
    }

    /// Test framework grouping.
    #[test]
    fn framework_grouping() {
        let report = FalsificationReport::new();
        let grouped = report.by_framework();

        assert!(grouped.contains_key(&Framework::NullFuzzer));
        assert!(grouped.contains_key(&Framework::ShmemProber));
        assert!(grouped.contains_key(&Framework::LifecycleChaos));
        assert!(grouped.contains_key(&Framework::QuantOracle));
        assert!(grouped.contains_key(&Framework::PtxPoison));
        assert!(grouped.contains_key(&Framework::Supervisor));
    }
}
