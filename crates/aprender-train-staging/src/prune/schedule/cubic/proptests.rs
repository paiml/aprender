//! Property-based tests for cubic pruning schedule.

use crate::prune::schedule::PruningSchedule;
use proptest::prelude::*;

proptest! {
    /// Cubic sparsity is always monotonically increasing
    #[test]
    fn cubic_monotonic(
        start in 0usize..1000,
        duration in 1usize..1000,
        final_val in 0.1f32..1.0,
    ) {
        let schedule = PruningSchedule::Cubic {
            start_step: start,
            end_step: start + duration,
            final_sparsity: final_val,
        };

        let mut prev = 0.0;
        for step in start..=(start + duration) {
            let sparsity = schedule.sparsity_at_step(step);
            prop_assert!(sparsity >= prev - 1e-5);
            prev = sparsity;
        }
    }
}
