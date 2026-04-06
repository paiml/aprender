//! OneShot pruning schedule methods.

use super::PruningSchedule;

impl PruningSchedule {
    /// Compute the target sparsity at a given training step for OneShot schedule.
    pub(super) fn oneshot_sparsity_at_step(prune_step: usize, step: usize) -> f32 {
        if step >= prune_step {
            1.0
        } else {
            0.0
        }
    }

    /// Check if pruning should be applied at this step for OneShot schedule.
    pub(super) fn oneshot_should_prune_at_step(prune_step: usize, step: usize) -> bool {
        step == prune_step
    }

    /// Get the total number of pruning operations for OneShot schedule.
    pub(super) fn oneshot_num_pruning_steps() -> usize {
        1
    }

    /// Validate OneShot schedule (always valid).
    pub(super) fn oneshot_validate() -> Result<(), String> {
        Ok(())
    }

    /// Check if OneShot pruning has completed.
    pub(super) fn oneshot_is_complete(prune_step: usize, step: usize) -> bool {
        step > prune_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // OneShot Schedule Tests
    // =========================================================================

    #[test]
    fn test_oneshot_before_step_returns_zero() {
        // TEST_ID: SCHED-001
        // FALSIFIES: OneShot returns non-zero before prune step
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.sparsity_at_step(0),
            0.0,
            "SCHED-001 FALSIFIED: OneShot should return 0.0 before prune step"
        );
        assert_eq!(
            schedule.sparsity_at_step(999),
            0.0,
            "SCHED-001 FALSIFIED: OneShot should return 0.0 at step before prune"
        );
    }

    #[test]
    fn test_oneshot_at_step_returns_one() {
        // TEST_ID: SCHED-002
        // FALSIFIES: OneShot returns wrong value at prune step
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.sparsity_at_step(1000),
            1.0,
            "SCHED-002 FALSIFIED: OneShot should return 1.0 at prune step"
        );
    }

    #[test]
    fn test_oneshot_after_step_returns_one() {
        // TEST_ID: SCHED-003
        // FALSIFIES: OneShot returns wrong value after prune step
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.sparsity_at_step(1001),
            1.0,
            "SCHED-003 FALSIFIED: OneShot should return 1.0 after prune step"
        );
        assert_eq!(
            schedule.sparsity_at_step(10000),
            1.0,
            "SCHED-003 FALSIFIED: OneShot should return 1.0 long after prune step"
        );
    }

    #[test]
    fn test_oneshot_step_zero() {
        // TEST_ID: SCHED-004
        // Edge case: prune at step 0
        let schedule = PruningSchedule::OneShot { step: 0 };
        assert_eq!(
            schedule.sparsity_at_step(0),
            1.0,
            "SCHED-004 FALSIFIED: OneShot at step 0 should return 1.0 immediately"
        );
    }

    #[test]
    fn test_oneshot_should_prune_only_at_step() {
        // TEST_ID: SCHED-005
        let schedule = PruningSchedule::OneShot { step: 500 };
        assert!(
            !schedule.should_prune_at_step(499),
            "SCHED-005 FALSIFIED: should_prune should be false before step"
        );
        assert!(
            schedule.should_prune_at_step(500),
            "SCHED-005 FALSIFIED: should_prune should be true at step"
        );
        assert!(
            !schedule.should_prune_at_step(501),
            "SCHED-005 FALSIFIED: should_prune should be false after step"
        );
    }

    #[test]
    fn test_validate_oneshot_always_valid() {
        // TEST_ID: SCHED-030
        let schedule = PruningSchedule::OneShot { step: 0 };
        assert!(schedule.validate().is_ok(), "SCHED-030 FALSIFIED: OneShot should always be valid");
    }

    #[test]
    fn test_num_pruning_steps_oneshot() {
        // TEST_ID: SCHED-040
        let schedule = PruningSchedule::OneShot { step: 1000 };
        assert_eq!(
            schedule.num_pruning_steps(),
            1,
            "SCHED-040 FALSIFIED: OneShot should have exactly 1 pruning step"
        );
    }

    #[test]
    fn test_is_complete_oneshot() {
        // TEST_ID: SCHED-043
        let schedule = PruningSchedule::OneShot { step: 100 };
        assert!(
            !schedule.is_complete(100),
            "SCHED-043 FALSIFIED: OneShot should not be complete at prune step"
        );
        assert!(
            schedule.is_complete(101),
            "SCHED-043 FALSIFIED: OneShot should be complete after prune step"
        );
    }

    #[test]
    fn test_oneshot_num_pruning_steps() {
        // TEST_ID: SCHED-072
        let schedule = PruningSchedule::OneShot { step: 0 };
        assert_eq!(schedule.num_pruning_steps(), 1);
    }

    #[test]
    fn test_is_complete_oneshot_at_zero() {
        // TEST_ID: SCHED-074
        let schedule = PruningSchedule::OneShot { step: 0 };
        assert!(!schedule.is_complete(0));
        assert!(schedule.is_complete(1));
    }

    #[test]
    fn test_debug_format() {
        // TEST_ID: SCHED-064
        let schedule = PruningSchedule::OneShot { step: 100 };
        let debug = format!("{schedule:?}");
        assert!(
            debug.contains("OneShot"),
            "SCHED-064 FALSIFIED: Debug should contain variant name"
        );
        assert!(debug.contains("100"), "SCHED-064 FALSIFIED: Debug should contain step value");
    }

    #[test]
    fn test_serialize_oneshot() {
        // TEST_ID: SCHED-050
        let schedule = PruningSchedule::OneShot { step: 1000 };
        let json = serde_json::to_string(&schedule).expect("JSON serialization should succeed");
        assert!(
            json.contains("one_shot"),
            "SCHED-050 FALSIFIED: OneShot should serialize with type=one_shot"
        );
        let deserialized: PruningSchedule =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(
            schedule, deserialized,
            "SCHED-050 FALSIFIED: Deserialized should match original"
        );
    }

    #[test]
    fn test_deserialize_oneshot_from_yaml() {
        // TEST_ID: SCHED-084
        let yaml = "type: one_shot\nstep: 500\n";
        let schedule: PruningSchedule =
            serde_yaml::from_str(yaml).expect("operation should succeed");
        match schedule {
            PruningSchedule::OneShot { step } => assert_eq!(step, 500),
            _ => panic!("Should deserialize to OneShot"),
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// OneShot is idempotent after prune step
        #[test]
        fn oneshot_idempotent(
            prune_step in 0usize..1000,
            test_step in 0usize..2000,
        ) {
            let schedule = PruningSchedule::OneShot { step: prune_step };
            let sparsity = schedule.sparsity_at_step(test_step);

            if test_step >= prune_step {
                prop_assert_eq!(sparsity, 1.0);
            } else {
                prop_assert_eq!(sparsity, 0.0);
            }
        }
    }
}
