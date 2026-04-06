//! Type definitions for pruning schedules.

use serde::{Deserialize, Serialize};

/// Pruning schedule defining when sparsity increases during training.
///
/// # Variants
///
/// - `OneShot`: All pruning happens at a single step
/// - `Gradual`: Linear interpolation between initial and final sparsity
/// - `Cubic`: Cubic polynomial schedule for smoother transitions
///
/// # Example
///
/// ```
/// use entrenar::prune::PruningSchedule;
///
/// // One-shot pruning at step 1000
/// let oneshot = PruningSchedule::OneShot { step: 1000 };
/// assert_eq!(oneshot.sparsity_at_step(500), 0.0);
/// assert_eq!(oneshot.sparsity_at_step(1000), 1.0);
///
/// // Gradual pruning from steps 100-1000
/// let gradual = PruningSchedule::Gradual {
///     start_step: 100,
///     end_step: 1000,
///     initial_sparsity: 0.0,
///     final_sparsity: 0.5,
///     frequency: 10,
/// };
/// assert_eq!(gradual.sparsity_at_step(50), 0.0);
/// assert_eq!(gradual.sparsity_at_step(1000), 0.5);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PruningSchedule {
    /// Prune once at specified step.
    OneShot {
        /// Step at which to apply pruning.
        step: usize,
    },

    /// Gradually increase sparsity over steps with linear interpolation.
    Gradual {
        /// Step to begin pruning.
        start_step: usize,
        /// Step at which final sparsity is reached.
        end_step: usize,
        /// Initial sparsity (typically 0.0).
        initial_sparsity: f32,
        /// Target final sparsity.
        final_sparsity: f32,
        /// Prune every N steps.
        frequency: usize,
    },

    /// Cubic sparsity schedule (Zhu & Gupta, 2017).
    ///
    /// Formula: s_t = s_f * (1 - (1 - t/T)^3)
    ///
    /// This provides faster initial pruning that slows as it approaches
    /// the target, giving the model more time to adapt.
    Cubic {
        /// Step to begin pruning.
        start_step: usize,
        /// Step at which final sparsity is reached.
        end_step: usize,
        /// Target final sparsity.
        final_sparsity: f32,
    },
}

impl Default for PruningSchedule {
    fn default() -> Self {
        PruningSchedule::OneShot { step: 0 }
    }
}
