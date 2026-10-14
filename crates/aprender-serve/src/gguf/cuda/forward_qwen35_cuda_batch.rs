//! aprender#5279 P1 (#5289): one greedy decode step for m sequences at once.
//!
//! [`Qwen35CudaModel::forward_batch_greedy`] is the seam the serve batch
//! scheduler drives. Today it loops [`Qwen35CudaModel::forward_single_greedy`]
//! over the slots, so each slot's tokens are exactly the ones it would get
//! decoded alone. Later phases replace the loop with batched projections and
//! slot-dimensioned `DeltaNet`/attention kernels behind the same signature,
//! and are held bitwise to this one.

use super::{Qwen35CudaModel, Qwen35CudaState, Result};
use crate::error::RealizarError;

/// Refuse a batch whose three per-slot inputs disagree in length, or that has
/// no slots, before any device work.
pub(crate) fn check_batch_shape(tokens: usize, states: usize, positions: usize) -> Result<()> {
    if tokens == 0 || tokens != states || tokens != positions {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "qwen35_cuda_batch: {tokens} tokens, {states} states and {positions} positions; \
                 the three must be equal and non-zero"
            ),
        });
    }
    Ok(())
}

impl Qwen35CudaModel<'_> {
    /// The token temperature 0 picks for each of m slots: slot `i` runs
    /// `tokens[i]` at `positions[i]` against `states[i]`, which is updated in
    /// place exactly as [`Self::forward_single_greedy`] updates it.
    ///
    /// # Errors
    /// A shape mismatch (refused before any device work), or the first slot's
    /// [`Self::forward_single_greedy`] error. Slots before the failing one have
    /// already advanced.
    pub fn forward_batch_greedy(
        &mut self,
        tokens: &[u32],
        states: &mut [Qwen35CudaState],
        positions: &[usize],
    ) -> Result<Vec<u32>> {
        check_batch_shape(tokens.len(), states.len(), positions.len())?;
        tokens
            .iter()
            .zip(states.iter_mut())
            .zip(positions)
            .map(|((&token, state), &position)| self.forward_single_greedy(token, state, position))
            .collect()
    }
}

#[cfg(test)]
#[path = "forward_qwen35_cuda_batch_tests.rs"]
mod tests;
