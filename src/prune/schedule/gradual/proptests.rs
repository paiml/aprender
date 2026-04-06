//! Property-based tests for gradual pruning schedule.

use crate::prune::schedule::PruningSchedule;
use proptest::prelude::*;

proptest! {
    /// Gradual sparsity is always monotonically increasing
    #[test]
    fn gradual_monotonic(
        start in 0usize..1000,
        duration in 1usize..1000,
        initial in 0.0f32..0.5,
        final_val in 0.5f32..1.0,
    ) {
        let schedule = PruningSchedule::Gradual {
            start_step: start,
            end_step: start + duration,
            initial_sparsity: initial,
            final_sparsity: final_val,
            frequency: 1,
        };

        let mut prev = initial;
        for step in start..=(start + duration) {
            let sparsity = schedule.sparsity_at_step(step);
            prop_assert!(sparsity >= prev - 1e-5);
            prev = sparsity;
        }
    }

    /// Sparsity is always bounded by initial and final
    #[test]
    fn sparsity_bounded(
        start in 0usize..100,
        duration in 1usize..100,
        initial in 0.0f32..0.5,
        final_val in 0.5f32..1.0,
        test_step in 0usize..500,
    ) {
        let schedule = PruningSchedule::Gradual {
            start_step: start,
            end_step: start + duration,
            initial_sparsity: initial,
            final_sparsity: final_val,
            frequency: 1,
        };

        let sparsity = schedule.sparsity_at_step(test_step);
        prop_assert!(sparsity >= initial - 1e-6);
        prop_assert!(sparsity <= final_val + 1e-6);
    }

    /// Serialize/deserialize roundtrip
    #[test]
    fn serde_roundtrip(
        start in 0usize..1000,
        duration in 1usize..1000,
        initial in 0.0f32..0.5,
        final_val in 0.5f32..1.0,
    ) {
        let schedule = PruningSchedule::Gradual {
            start_step: start,
            end_step: start + duration,
            initial_sparsity: initial,
            final_sparsity: final_val,
            frequency: 10,
        };

        let json = serde_json::to_string(&schedule);
        prop_assert!(json.is_ok(), "serialize failed: {:?}", json.err());
        let json = json.expect("operation should succeed");
        let deserialized: Result<PruningSchedule, _> = serde_json::from_str(&json);
        prop_assert!(deserialized.is_ok(), "deserialize failed: {:?}", deserialized.err());
        prop_assert_eq!(schedule, deserialized.expect("deserialization should succeed"));
    }
}
