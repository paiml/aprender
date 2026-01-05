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
    /// Binary mask tensor [`original_dim`]
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
mod tests {
    use super::*;

    // ==========================================================================
    // FALSIFICATION: WidthPruner configuration
    // ==========================================================================
    #[test]
    fn test_width_pruner_new() {
        let pruner = WidthPruner::new(512, 2048, 8);

        assert_eq!(pruner.target_hidden_dim(), 512);
        assert_eq!(pruner.target_intermediate_dim(), 2048);
        assert_eq!(pruner.num_attention_heads(), 8);
    }

    #[test]
    fn test_width_pruner_default() {
        let pruner = WidthPruner::default();
        assert_eq!(pruner.target_hidden_dim(), 0);
        assert_eq!(pruner.num_attention_heads(), 1);
    }

    // ==========================================================================
    // FALSIFICATION: Validation
    // ==========================================================================
    #[test]
    fn test_validate_success() {
        let pruner = WidthPruner::new(512, 2048, 8); // 512/8 = 64
        assert!(pruner.validate(1024, 4096).is_ok());
    }

    #[test]
    fn test_validate_head_divisibility() {
        let pruner = WidthPruner::new(500, 2048, 8); // 500/8 = 62.5 - not divisible
        let result = pruner.validate(1024, 4096);

        assert!(
            result.is_err(),
            "WID-01 FALSIFIED: Should error on non-divisible"
        );
        match result.unwrap_err() {
            PruningError::InvalidPattern { message } => {
                assert!(message.contains("divisible"));
            }
            _ => panic!("WID-01 FALSIFIED: Expected InvalidPattern error"),
        }
    }

    #[test]
    fn test_validate_target_exceeds_original_hidden() {
        let pruner = WidthPruner::new(2048, 2048, 8);
        let result = pruner.validate(1024, 4096); // 2048 > 1024

        assert!(
            result.is_err(),
            "WID-02 FALSIFIED: Should error when target > original"
        );
    }

    #[test]
    fn test_validate_target_exceeds_original_intermediate() {
        let pruner = WidthPruner::new(512, 8192, 8);
        let result = pruner.validate(1024, 4096); // 8192 > 4096

        assert!(
            result.is_err(),
            "WID-03 FALSIFIED: Should error when target > original"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Channel importance computation
    // ==========================================================================
    #[test]
    fn test_compute_channel_importance_basic() {
        let pruner = WidthPruner::new(2, 2, 1);

        // 2 samples, 4 hidden channels
        let hidden = Tensor::new(
            &[
                1.0, 2.0, 3.0, 4.0, // sample 0
                1.0, 2.0, 3.0, 4.0, // sample 1
            ],
            &[2, 4],
        );

        // 2 samples, 3 intermediate channels
        let intermediate = Tensor::new(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);

        let importance = pruner
            .compute_channel_importance(&hidden, &intermediate)
            .unwrap();

        assert_eq!(importance.hidden_dim(), 4);
        assert_eq!(importance.intermediate_dim(), 3);
        assert_eq!(importance.num_samples, 2);

        // importance[d] = mean(x^2) = x^2 for constant values
        let h_data = importance.hidden.data();
        assert!(
            (h_data[0] - 1.0).abs() < 1e-5,
            "WID-04 FALSIFIED: Channel 0 importance should be 1.0"
        );
        assert!(
            (h_data[1] - 4.0).abs() < 1e-5,
            "WID-04 FALSIFIED: Channel 1 importance should be 4.0"
        );
        assert!(
            (h_data[2] - 9.0).abs() < 1e-5,
            "WID-04 FALSIFIED: Channel 2 importance should be 9.0"
        );
        assert!(
            (h_data[3] - 16.0).abs() < 1e-5,
            "WID-04 FALSIFIED: Channel 3 importance should be 16.0"
        );
    }

    #[test]
    fn test_compute_channel_importance_varying() {
        let pruner = WidthPruner::new(2, 2, 1);

        // importance = mean(x^2) across samples
        let hidden = Tensor::new(
            &[
                1.0, 3.0, // sample 0: [1, 9]
                3.0, 1.0, // sample 1: [9, 1]
            ],
            &[2, 2],
        );

        // Intermediate must have same number of samples
        let intermediate = Tensor::new(
            &[
                0.0, 0.0, // sample 0
                0.0, 0.0, // sample 1
            ],
            &[2, 2],
        );

        let importance = pruner
            .compute_channel_importance(&hidden, &intermediate)
            .unwrap();

        let h_data = importance.hidden.data();
        // Channel 0: mean(1, 9) = 5
        // Channel 1: mean(9, 1) = 5
        assert!(
            (h_data[0] - 5.0).abs() < 1e-5,
            "WID-05 FALSIFIED: Channel 0 importance should be 5.0, got {}",
            h_data[0]
        );
        assert!(
            (h_data[1] - 5.0).abs() < 1e-5,
            "WID-05 FALSIFIED: Channel 1 importance should be 5.0, got {}",
            h_data[1]
        );
    }

    #[test]
    fn test_compute_channel_importance_invalid_shape() {
        let pruner = WidthPruner::new(2, 2, 1);

        // 1D tensor (invalid)
        let hidden = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
        let intermediate = Tensor::new(&[1.0, 2.0], &[1, 2]);

        let result = pruner.compute_channel_importance(&hidden, &intermediate);
        assert!(
            result.is_err(),
            "WID-06 FALSIFIED: Should error on 1D tensor"
        );
    }

    #[test]
    fn test_compute_channel_importance_mismatched_samples() {
        let pruner = WidthPruner::new(2, 2, 1);

        // Different number of samples
        let hidden = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let intermediate = Tensor::new(&[1.0, 2.0], &[1, 2]); // Only 1 sample

        let result = pruner.compute_channel_importance(&hidden, &intermediate);
        assert!(
            result.is_err(),
            "WID-06b FALSIFIED: Should error on mismatched sample counts"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Top-k channel selection
    // ==========================================================================
    #[test]
    fn test_top_hidden_channels() {
        let importance = ChannelImportance::new(
            Tensor::new(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5]),
            Tensor::new(&[], &[0]),
            1,
        );

        let top3 = importance.top_hidden_channels(3);

        assert_eq!(top3.len(), 3, "WID-07 FALSIFIED: Should return 3 channels");
        // Top 3 by importance: index 4 (5.0), index 2 (4.0), index 0 (3.0)
        // Sorted: [0, 2, 4]
        assert!(
            top3.contains(&4),
            "WID-07 FALSIFIED: Should contain index 4 (highest)"
        );
        assert!(
            top3.contains(&2),
            "WID-07 FALSIFIED: Should contain index 2"
        );
        assert!(
            top3.contains(&0),
            "WID-07 FALSIFIED: Should contain index 0"
        );
    }

    #[test]
    fn test_top_intermediate_channels() {
        let importance = ChannelImportance::new(
            Tensor::new(&[], &[0]),
            Tensor::new(&[10.0, 20.0, 5.0, 15.0], &[4]),
            1,
        );

        let top2 = importance.top_intermediate_channels(2);

        assert_eq!(top2.len(), 2);
        // Top 2: index 1 (20.0), index 3 (15.0)
        assert!(top2.contains(&1));
        assert!(top2.contains(&3));
    }

    // ==========================================================================
    // FALSIFICATION: Select channels to keep
    // ==========================================================================
    #[test]
    fn test_select_channels_to_keep() {
        let pruner = WidthPruner::new(2, 2, 1);
        let importance = ChannelImportance::new(
            Tensor::new(&[1.0, 5.0, 3.0, 4.0], &[4]), // top-2: indices 1, 3
            Tensor::new(&[10.0, 5.0, 15.0], &[3]),    // top-2: indices 0, 2
            1,
        );

        let (hidden_keep, intermediate_keep) = pruner.select_channels_to_keep(&importance).unwrap();

        assert_eq!(
            hidden_keep.len(),
            2,
            "WID-08 FALSIFIED: Should keep 2 hidden"
        );
        assert_eq!(
            intermediate_keep.len(),
            2,
            "WID-08 FALSIFIED: Should keep 2 intermediate"
        );

        // Verify correct channels selected
        assert!(hidden_keep.contains(&1)); // 5.0
        assert!(hidden_keep.contains(&3)); // 4.0
        assert!(intermediate_keep.contains(&0)); // 10.0
        assert!(intermediate_keep.contains(&2)); // 15.0
    }

    #[test]
    fn test_select_channels_validates() {
        let pruner = WidthPruner::new(100, 100, 8); // 100/8 = 12.5 - invalid
        let importance = ChannelImportance::new(
            Tensor::new(&[1.0; 50], &[50]),
            Tensor::new(&[1.0; 200], &[200]),
            1,
        );

        let result = pruner.select_channels_to_keep(&importance);
        assert!(
            result.is_err(),
            "WID-09 FALSIFIED: Should validate before selecting"
        );
    }

    // ==========================================================================
    // FALSIFICATION: Generate hidden mask
    // ==========================================================================
    #[test]
    fn test_generate_hidden_mask() {
        let pruner = WidthPruner::new(3, 3, 1);
        let mask = pruner.generate_hidden_mask(5, &[0, 2, 4]);

        let data = mask.data();
        assert_eq!(data.len(), 5);
        assert_eq!(data[0], 1.0, "WID-10 FALSIFIED: Channel 0 should be kept");
        assert_eq!(data[1], 0.0, "WID-10 FALSIFIED: Channel 1 should be pruned");
        assert_eq!(data[2], 1.0, "WID-10 FALSIFIED: Channel 2 should be kept");
        assert_eq!(data[3], 0.0, "WID-10 FALSIFIED: Channel 3 should be pruned");
        assert_eq!(data[4], 1.0, "WID-10 FALSIFIED: Channel 4 should be kept");
    }

    #[test]
    fn test_generate_hidden_mask_out_of_bounds() {
        let pruner = WidthPruner::new(3, 3, 1);
        let mask = pruner.generate_hidden_mask(3, &[0, 1, 100]); // 100 is out of bounds

        let data = mask.data();
        assert_eq!(data.len(), 3);
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], 1.0);
        assert_eq!(data[2], 0.0); // index 100 ignored, index 2 not set
    }

    // ==========================================================================
    // FALSIFICATION: Head dimension calculation
    // ==========================================================================
    #[test]
    fn test_head_dim_after_pruning() {
        let pruner = WidthPruner::new(512, 2048, 8);
        assert_eq!(pruner.head_dim_after_pruning(), 64); // 512/8
    }

    #[test]
    fn test_head_dim_zero_heads() {
        let pruner = WidthPruner::new(512, 2048, 0);
        assert_eq!(pruner.head_dim_after_pruning(), 0);
    }

    // ==========================================================================
    // FALSIFICATION: WidthPruningResult
    // ==========================================================================
    #[test]
    fn test_width_pruning_result_compression() {
        let result = WidthPruningResult::new(
            1024,
            512, // hidden: 1024 -> 512
            4096,
            2048, // intermediate: 4096 -> 2048
            vec![0, 1, 2, 3],
            vec![0, 1],
        );

        assert!(
            (result.hidden_compression_ratio() - 2.0).abs() < 1e-5,
            "WID-11 FALSIFIED: Hidden compression should be 2x"
        );
        assert!(
            (result.intermediate_compression_ratio() - 2.0).abs() < 1e-5,
            "WID-11 FALSIFIED: Intermediate compression should be 2x"
        );
    }

    #[test]
    fn test_width_pruning_result_removal_percentage() {
        let result = WidthPruningResult::new(1000, 750, 2000, 1500, vec![], vec![]);

        assert!(
            (result.hidden_removal_percentage() - 25.0).abs() < 1e-5,
            "WID-12 FALSIFIED: Hidden removal should be 25%"
        );
        assert!(
            (result.intermediate_removal_percentage() - 25.0).abs() < 1e-5,
            "WID-12 FALSIFIED: Intermediate removal should be 25%"
        );
    }

    #[test]
    fn test_width_pruning_result_edge_cases() {
        let result = WidthPruningResult::new(0, 0, 0, 0, vec![], vec![]);

        assert_eq!(result.hidden_compression_ratio(), f32::INFINITY);
        assert_eq!(result.hidden_removal_percentage(), 0.0);
    }

    // ==========================================================================
    // FALSIFICATION: top_k_indices helper
    // ==========================================================================
    #[test]
    fn test_top_k_indices_basic() {
        let data = &[5.0, 1.0, 3.0, 2.0, 4.0];
        let top3 = top_k_indices(data, 3);

        // Top 3 values: 5.0 (idx 0), 4.0 (idx 4), 3.0 (idx 2)
        // Sorted: [0, 2, 4]
        assert_eq!(top3, vec![0, 2, 4]);
    }

    #[test]
    fn test_top_k_indices_all() {
        let data = &[1.0, 2.0, 3.0];
        let all = top_k_indices(data, 5); // k > len

        assert_eq!(all, vec![0, 1, 2]);
    }

    #[test]
    fn test_top_k_indices_empty() {
        let data: &[f32] = &[];
        let result = top_k_indices(data, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_top_k_indices_ties() {
        let data = &[1.0, 2.0, 2.0, 1.0];
        let top2 = top_k_indices(data, 2);

        // Two 2.0 values at indices 1 and 2
        assert_eq!(top2.len(), 2);
        assert!(top2.contains(&1) || top2.contains(&2)); // At least one of them
    }

    // ==========================================================================
    // FALSIFICATION: Clone and Debug
    // ==========================================================================
    #[test]
    fn test_width_pruner_clone() {
        let orig = WidthPruner::new(512, 2048, 8);
        let cloned = orig.clone();

        assert_eq!(orig.target_hidden_dim(), cloned.target_hidden_dim());
        assert_eq!(
            orig.target_intermediate_dim(),
            cloned.target_intermediate_dim()
        );
        assert_eq!(orig.num_attention_heads(), cloned.num_attention_heads());
    }

    #[test]
    fn test_width_pruner_debug() {
        let pruner = WidthPruner::new(512, 2048, 8);
        let debug = format!("{:?}", pruner);
        assert!(debug.contains("WidthPruner"));
    }

    #[test]
    fn test_channel_importance_debug() {
        let imp = ChannelImportance::new(Tensor::new(&[1.0], &[1]), Tensor::new(&[1.0], &[1]), 1);
        let debug = format!("{:?}", imp);
        assert!(debug.contains("ChannelImportance"));
    }

    #[test]
    fn test_width_pruning_result_debug() {
        let result = WidthPruningResult::new(100, 50, 200, 100, vec![], vec![]);
        let debug = format!("{:?}", result);
        assert!(debug.contains("WidthPruningResult"));
    }
}
