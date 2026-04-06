//! Gradual pruning schedule methods.

#[cfg(test)]
mod proptests;
#[cfg(test)]
mod tests;

use super::PruningSchedule;

impl PruningSchedule {
    /// Compute the target sparsity at a given training step for Gradual schedule.
    pub(super) fn gradual_sparsity_at_step(
        start_step: usize,
        end_step: usize,
        initial_sparsity: f32,
        final_sparsity: f32,
        step: usize,
    ) -> f32 {
        if step < start_step {
            initial_sparsity
        } else if step >= end_step {
            final_sparsity
        } else {
            let progress = (step - start_step) as f32 / (end_step - start_step) as f32;
            initial_sparsity + progress * (final_sparsity - initial_sparsity)
        }
    }

    /// Check if pruning should be applied at this step for Gradual schedule.
    pub(super) fn gradual_should_prune_at_step(
        start_step: usize,
        end_step: usize,
        frequency: usize,
        step: usize,
    ) -> bool {
        if step < start_step || step > end_step {
            return false;
        }
        if frequency == 0 {
            return step == start_step;
        }
        (step - start_step).is_multiple_of(frequency)
    }

    /// Get the total number of pruning operations for Gradual schedule.
    pub(super) fn gradual_num_pruning_steps(
        start_step: usize,
        end_step: usize,
        frequency: usize,
    ) -> usize {
        if frequency == 0 {
            1
        } else {
            (end_step - start_step) / frequency + 1
        }
    }

    /// Validate Gradual schedule.
    pub(super) fn gradual_validate(
        start_step: usize,
        end_step: usize,
        initial_sparsity: f32,
        final_sparsity: f32,
    ) -> Result<(), String> {
        if end_step <= start_step {
            return Err(format!(
                "end_step ({end_step}) must be greater than start_step ({start_step})"
            ));
        }
        if !(0.0..=1.0).contains(&initial_sparsity) {
            return Err(format!(
                "initial_sparsity ({initial_sparsity}) must be between 0.0 and 1.0"
            ));
        }
        if !(0.0..=1.0).contains(&final_sparsity) {
            return Err(format!("final_sparsity ({final_sparsity}) must be between 0.0 and 1.0"));
        }
        Ok(())
    }

    /// Check if Gradual pruning has completed.
    pub(super) fn gradual_is_complete(end_step: usize, step: usize) -> bool {
        step > end_step
    }
}
