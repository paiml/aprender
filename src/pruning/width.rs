//! Minitron Width Pruning: Channel removal based on activation importance.
//!
//! # Toyota Way: Heijunka (Level Loading)
//! Balanced channel importance ensures no single dimension is over-pruned.
//!
//! # Algorithm
//! For each hidden dimension d:
//!   importance(d) = mean(|activations[:, d]|^2)
//!
//! Channels with lowest activation magnitude are pruned across all layers.
//!
//! # Constraint
//! Attention heads must be pruned as complete units to maintain validity.
//! Target hidden dimension must be divisible by number of attention heads.
//!
//! # References
//! - Muralidharan, S., et al. (2024). Compact language models via pruning
//!   and knowledge distillation. arXiv:2407.14679.

use super::error::PruningError;
use crate::autograd::Tensor;

/// Result of width (channel) pruning operation.
#[derive(Debug, Clone)]
pub struct WidthPruningResult {
    /// Original hidden dimension.
    pub original_hidden_dim: usize,
    /// Final hidden dimension after pruning.
    pub final_hidden_dim: usize,
    /// Original intermediate (FFN) dimension.
    pub original_intermediate_dim: usize,
    /// Final intermediate dimension after pruning.
    pub final_intermediate_dim: usize,
    /// Indices of hidden channels kept (sorted).
    pub hidden_channels_kept: Vec<usize>,
    /// Indices of intermediate channels kept (sorted).
    pub intermediate_channels_kept: Vec<usize>,
}

impl WidthPruningResult {
    /// Create a new width pruning result.
    #[must_use]
    pub fn new(
        original_hidden_dim: usize,
        final_hidden_dim: usize,
        original_intermediate_dim: usize,
        final_intermediate_dim: usize,
        hidden_channels_kept: Vec<usize>,
        intermediate_channels_kept: Vec<usize>,
    ) -> Self {
        Self {
            original_hidden_dim,
            final_hidden_dim,
            original_intermediate_dim,
            final_intermediate_dim,
            hidden_channels_kept,
            intermediate_channels_kept,
        }
    }

    /// Get hidden dimension compression ratio.
    #[must_use]
    pub fn hidden_compression_ratio(&self) -> f32 {
        if self.final_hidden_dim == 0 {
            f32::INFINITY
        } else {
            self.original_hidden_dim as f32 / self.final_hidden_dim as f32
        }
    }

    /// Get intermediate dimension compression ratio.
    #[must_use]
    pub fn intermediate_compression_ratio(&self) -> f32 {
        if self.final_intermediate_dim == 0 {
            f32::INFINITY
        } else {
            self.original_intermediate_dim as f32 / self.final_intermediate_dim as f32
        }
    }

    /// Get percentage of hidden channels removed.
    #[must_use]
    pub fn hidden_removal_percentage(&self) -> f32 {
        if self.original_hidden_dim == 0 {
            0.0
        } else {
            let removed = self.original_hidden_dim - self.final_hidden_dim;
            removed as f32 / self.original_hidden_dim as f32 * 100.0
        }
    }

    /// Get percentage of intermediate channels removed.
    #[must_use]
    pub fn intermediate_removal_percentage(&self) -> f32 {
        if self.original_intermediate_dim == 0 {
            0.0
        } else {
            let removed = self.original_intermediate_dim - self.final_intermediate_dim;
            removed as f32 / self.original_intermediate_dim as f32 * 100.0
        }
    }
}

/// Channel importance scores.
#[derive(Debug, Clone)]
pub struct ChannelImportance {
    /// Importance scores for hidden dimension channels.
    pub hidden: Tensor,
    /// Importance scores for intermediate (FFN) dimension channels.
    pub intermediate: Tensor,
    /// Number of calibration samples used.
    pub num_samples: usize,
}

impl ChannelImportance {
    /// Create new channel importance scores.
    #[must_use]
    pub fn new(hidden: Tensor, intermediate: Tensor, num_samples: usize) -> Self {
        Self {
            hidden,
            intermediate,
            num_samples,
        }
    }

    /// Get hidden dimension size.
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.hidden.data().len()
    }

    /// Get intermediate dimension size.
    #[must_use]
    pub fn intermediate_dim(&self) -> usize {
        self.intermediate.data().len()
    }

    /// Get top-k hidden channel indices by importance.
    #[must_use]
    pub fn top_hidden_channels(&self, k: usize) -> Vec<usize> {
        top_k_indices(self.hidden.data(), k)
    }

    /// Get top-k intermediate channel indices by importance.
    #[must_use]
    pub fn top_intermediate_channels(&self, k: usize) -> Vec<usize> {
        top_k_indices(self.intermediate.data(), k)
    }
}

/// Get indices of top-k elements by value.
fn top_k_indices(data: &[f32], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f32)> = data.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut result: Vec<usize> = indexed.into_iter().take(k).map(|(idx, _)| idx).collect();
    result.sort_unstable(); // Return sorted indices
    result
}

/// Minitron Width Pruner: Channel removal based on activation importance.
///
/// # Algorithm
/// 1. Compute channel importance from calibration activations
/// 2. Select top-k channels to keep for each dimension type
/// 3. Ensure head divisibility constraint for attention
///
/// # Toyota Way: Poka-Yoke
/// Validates head divisibility to prevent invalid model configurations.
#[derive(Debug, Clone)]
pub struct WidthPruner {
    /// Target hidden dimension (must be divisible by `num_heads`).
    target_hidden_dim: usize,
    /// Target intermediate dimension for FFN.
    target_intermediate_dim: usize,
    /// Number of attention heads (for divisibility constraint).
    num_attention_heads: usize,
}

impl WidthPruner {
    /// Create a new width pruner.
    ///
    /// # Arguments
    /// * `target_hidden_dim` - Target hidden dimension (must be divisible by `num_heads`)
    /// * `target_intermediate_dim` - Target intermediate (FFN) dimension
    /// * `num_attention_heads` - Number of attention heads
    #[must_use]
    pub fn new(
        target_hidden_dim: usize,
        target_intermediate_dim: usize,
        num_attention_heads: usize,
    ) -> Self {
        Self {
            target_hidden_dim,
            target_intermediate_dim,
            num_attention_heads,
        }
    }

    /// Get target hidden dimension.
    #[must_use]
    pub fn target_hidden_dim(&self) -> usize {
        self.target_hidden_dim
    }

    /// Get target intermediate dimension.
    #[must_use]
    pub fn target_intermediate_dim(&self) -> usize {
        self.target_intermediate_dim
    }

    /// Get number of attention heads.
    #[must_use]
    pub fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }

    /// Validate configuration.
    ///
    /// # Errors
    /// * If `target_hidden_dim` is not divisible by `num_attention_heads`
    /// * If target dimensions exceed original dimensions
    pub fn validate(
        &self,
        original_hidden_dim: usize,
        original_intermediate_dim: usize,
    ) -> Result<(), PruningError> {
        // Check head divisibility
        if self.target_hidden_dim % self.num_attention_heads != 0 {
            return Err(PruningError::InvalidPattern {
                message: format!(
                    "target_hidden_dim ({}) must be divisible by num_attention_heads ({})",
                    self.target_hidden_dim, self.num_attention_heads
                ),
            });
        }

        // Check target doesn't exceed original
        if self.target_hidden_dim > original_hidden_dim {
            return Err(PruningError::InvalidSparsity {
                value: self.target_hidden_dim as f32,
                constraint: format!(
                    "target_hidden_dim ({}) exceeds original ({})",
                    self.target_hidden_dim, original_hidden_dim
                ),
            });
        }

        if self.target_intermediate_dim > original_intermediate_dim {
            return Err(PruningError::InvalidSparsity {
                value: self.target_intermediate_dim as f32,
                constraint: format!(
                    "target_intermediate_dim ({}) exceeds original ({})",
                    self.target_intermediate_dim, original_intermediate_dim
                ),
            });
        }

        Ok(())
    }

    /// Compute channel importance from activation data.
    ///
    /// importance(d) = mean(|activations[:, d]|^2)
    ///
    /// # Arguments
    /// * `hidden_activations` - Hidden state activations [`num_samples`, `hidden_dim`]
    /// * `intermediate_activations` - FFN intermediate activations [`num_samples`, `intermediate_dim`]
    ///
    /// # Returns
    /// Channel importance scores
    pub fn compute_channel_importance(
        &self,
        hidden_activations: &Tensor,
        intermediate_activations: &Tensor,
    ) -> Result<ChannelImportance, PruningError> {
        let h_shape = hidden_activations.shape();
        let i_shape = intermediate_activations.shape();

        // Validate shapes
        if h_shape.len() != 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0], // 2D expected
                got: h_shape.to_vec(),
            });
        }

        if i_shape.len() != 2 {
            return Err(PruningError::ShapeMismatch {
                expected: vec![0, 0],
                got: i_shape.to_vec(),
            });
        }

        // Validate consistent sample counts
        if h_shape[0] != i_shape[0] {
            return Err(PruningError::ShapeMismatch {
                expected: vec![h_shape[0], i_shape[1]],
                got: vec![i_shape[0], i_shape[1]],
            });
        }

        let num_samples = h_shape[0];
        let hidden_dim = h_shape[1];
        let intermediate_dim = i_shape[1];

        // Compute hidden importance: mean(x^2) per channel
        let h_data = hidden_activations.data();
        let mut hidden_importance = vec![0.0f32; hidden_dim];

        for sample in 0..num_samples {
            for d in 0..hidden_dim {
                let val = h_data[sample * hidden_dim + d];
                hidden_importance[d] += val * val;
            }
        }

        if num_samples > 0 {
            for imp in &mut hidden_importance {
                *imp /= num_samples as f32;
            }
        }

        // Compute intermediate importance
        let i_data = intermediate_activations.data();
        let mut intermediate_importance = vec![0.0f32; intermediate_dim];

        for sample in 0..num_samples {
            for d in 0..intermediate_dim {
                let val = i_data[sample * intermediate_dim + d];
                intermediate_importance[d] += val * val;
            }
        }

        if num_samples > 0 {
            for imp in &mut intermediate_importance {
                *imp /= num_samples as f32;
            }
        }

        Ok(ChannelImportance::new(
            Tensor::new(&hidden_importance, &[hidden_dim]),
            Tensor::new(&intermediate_importance, &[intermediate_dim]),
            num_samples,
        ))
    }

    /// Select channels to keep based on importance.
    ///
    /// # Arguments
    /// * `importance` - Channel importance scores
    ///
    /// # Returns
    /// Indices of channels to keep for hidden and intermediate dimensions
    pub fn select_channels_to_keep(
        &self,
        importance: &ChannelImportance,
    ) -> Result<(Vec<usize>, Vec<usize>), PruningError> {
        // Validate configuration
        self.validate(importance.hidden_dim(), importance.intermediate_dim())?;

        // Select top-k channels
        let hidden_keep = importance.top_hidden_channels(self.target_hidden_dim);
        let intermediate_keep = importance.top_intermediate_channels(self.target_intermediate_dim);

        Ok((hidden_keep, intermediate_keep))
    }

    /// Generate pruning mask for hidden dimension.
    ///
    /// # Arguments
    /// * `original_dim` - Original hidden dimension
    /// * `channels_to_keep` - Indices of channels to keep
    ///
    /// # Returns
    /// Binary mask tensor \[`original_dim`\]
    #[must_use]
    pub fn generate_hidden_mask(&self, original_dim: usize, channels_to_keep: &[usize]) -> Tensor {
        let mut mask = vec![0.0f32; original_dim];
        for &idx in channels_to_keep {
            if idx < original_dim {
                mask[idx] = 1.0;
            }
        }
        Tensor::new(&mask, &[original_dim])
    }

    /// Compute head dimension after pruning.
    #[must_use]
    pub fn head_dim_after_pruning(&self) -> usize {
        if self.num_attention_heads == 0 {
            0
        } else {
            self.target_hidden_dim / self.num_attention_heads
        }
    }
}

impl Default for WidthPruner {
    fn default() -> Self {
        Self::new(0, 0, 1)
    }
}

#[cfg(test)]
#[path = "width_tests.rs"]
mod tests;
