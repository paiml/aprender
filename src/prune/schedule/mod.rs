//! Pruning schedule definitions
//!
//! Defines when and how sparsity increases during training:
//! - OneShot: Prune all at once at a specific step
//! - Gradual: Linear interpolation from initial to final sparsity
//! - Cubic: Cubic schedule (Zhu & Gupta, 2017) for smoother transitions
//!
//! # Toyota Way: Kaizen (Continuous Improvement)
//! Gradual and cubic schedules allow incremental model adaptation.
//!
//! # References
//! - Zhu, M., & Gupta, S. (2017). To prune, or not to prune: exploring the
//!   efficacy of pruning for model compression. arXiv:1710.01878.

mod cubic;
mod gradual;
mod oneshot;
mod types;

pub use types::PruningSchedule;

impl PruningSchedule {
    /// Compute the target sparsity at a given training step.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    ///
    /// # Returns
    ///
    /// Target sparsity as a value between 0.0 and 1.0.
    ///
    /// # Panics
    ///
    /// Does not panic. Returns bounded values for all inputs.
    pub fn sparsity_at_step(&self, step: usize) -> f32 {
        match self {
            PruningSchedule::OneShot { step: prune_step } => {
                Self::oneshot_sparsity_at_step(*prune_step, step)
            }
            PruningSchedule::Gradual {
                start_step,
                end_step,
                initial_sparsity,
                final_sparsity,
                ..
            } => Self::gradual_sparsity_at_step(
                *start_step,
                *end_step,
                *initial_sparsity,
                *final_sparsity,
                step,
            ),
            PruningSchedule::Cubic { start_step, end_step, final_sparsity } => {
                Self::cubic_sparsity_at_step(*start_step, *end_step, *final_sparsity, step)
            }
        }
    }

    /// Check if pruning should be applied at this step based on frequency.
    ///
    /// For `OneShot`, returns true only at the prune step.
    /// For `Gradual`, returns true at steps matching the frequency.
    /// For `Cubic`, returns true at every step in the pruning window.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    pub fn should_prune_at_step(&self, step: usize) -> bool {
        match self {
            PruningSchedule::OneShot { step: prune_step } => {
                Self::oneshot_should_prune_at_step(*prune_step, step)
            }
            PruningSchedule::Gradual { start_step, end_step, frequency, .. } => {
                Self::gradual_should_prune_at_step(*start_step, *end_step, *frequency, step)
            }
            PruningSchedule::Cubic { start_step, end_step, .. } => {
                Self::cubic_should_prune_at_step(*start_step, *end_step, step)
            }
        }
    }

    /// Get the total number of pruning operations for this schedule.
    ///
    /// # Returns
    ///
    /// Expected number of times pruning will be applied.
    pub fn num_pruning_steps(&self) -> usize {
        match self {
            PruningSchedule::OneShot { .. } => Self::oneshot_num_pruning_steps(),
            PruningSchedule::Gradual { start_step, end_step, frequency, .. } => {
                Self::gradual_num_pruning_steps(*start_step, *end_step, *frequency)
            }
            PruningSchedule::Cubic { start_step, end_step, .. } => {
                Self::cubic_num_pruning_steps(*start_step, *end_step)
            }
        }
    }

    /// Check if the schedule is valid.
    ///
    /// # Errors
    ///
    /// Returns an error message if the schedule is invalid.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            PruningSchedule::OneShot { .. } => Self::oneshot_validate(),
            PruningSchedule::Gradual {
                start_step,
                end_step,
                initial_sparsity,
                final_sparsity,
                ..
            } => Self::gradual_validate(*start_step, *end_step, *initial_sparsity, *final_sparsity),
            PruningSchedule::Cubic { start_step, end_step, final_sparsity } => {
                Self::cubic_validate(*start_step, *end_step, *final_sparsity)
            }
        }
    }

    /// Check if pruning has completed (current step is past the schedule).
    pub fn is_complete(&self, step: usize) -> bool {
        match self {
            PruningSchedule::OneShot { step: prune_step } => {
                Self::oneshot_is_complete(*prune_step, step)
            }
            PruningSchedule::Gradual { end_step, .. } => Self::gradual_is_complete(*end_step, step),
            PruningSchedule::Cubic { end_step, .. } => Self::cubic_is_complete(*end_step, step),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_schedule() {
        // TEST_ID: SCHED-045
        let schedule = PruningSchedule::default();
        match schedule {
            PruningSchedule::OneShot { step } => {
                assert_eq!(step, 0, "SCHED-045 FALSIFIED: Default should be OneShot at step 0");
            }
            _ => panic!("SCHED-045 FALSIFIED: Default should be OneShot variant"),
        }
    }
}
