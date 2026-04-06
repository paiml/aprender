//! Core cubic pruning schedule implementation.

use crate::prune::schedule::PruningSchedule;

impl PruningSchedule {
    /// Compute the target sparsity at a given training step for Cubic schedule.
    /// Formula: s_t = s_f * (1 - (1 - t/T)^3)
    pub(in crate::prune::schedule) fn cubic_sparsity_at_step(
        start_step: usize,
        end_step: usize,
        final_sparsity: f32,
        step: usize,
    ) -> f32 {
        if step < start_step {
            0.0
        } else if step >= end_step {
            final_sparsity
        } else {
            let t = (step - start_step) as f32;
            let total = (end_step - start_step) as f32;
            let ratio = 1.0 - t / total;
            final_sparsity * (1.0 - ratio.powi(3))
        }
    }

    /// Check if pruning should be applied at this step for Cubic schedule.
    /// Cubic prunes at every step in the window.
    pub(in crate::prune::schedule) fn cubic_should_prune_at_step(
        start_step: usize,
        end_step: usize,
        step: usize,
    ) -> bool {
        step >= start_step && step <= end_step
    }

    /// Get the total number of pruning operations for Cubic schedule.
    pub(in crate::prune::schedule) fn cubic_num_pruning_steps(
        start_step: usize,
        end_step: usize,
    ) -> usize {
        end_step - start_step + 1
    }

    /// Validate Cubic schedule.
    pub(in crate::prune::schedule) fn cubic_validate(
        start_step: usize,
        end_step: usize,
        final_sparsity: f32,
    ) -> Result<(), String> {
        if end_step <= start_step {
            return Err(format!(
                "end_step ({end_step}) must be greater than start_step ({start_step})"
            ));
        }
        if !(0.0..=1.0).contains(&final_sparsity) {
            return Err(format!("final_sparsity ({final_sparsity}) must be between 0.0 and 1.0"));
        }
        Ok(())
    }

    /// Check if Cubic pruning has completed.
    pub(in crate::prune::schedule) fn cubic_is_complete(end_step: usize, step: usize) -> bool {
        step > end_step
    }
}
