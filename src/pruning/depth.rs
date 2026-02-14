//! Minitron Depth Pruning: Layer removal based on Block Importance.
//!
//! # Toyota Way: Muda (Waste Elimination)
//! Removes entire transformer layers that contribute minimally to output
//! transformation, providing direct wall-clock speedup.
//!
//! # Algorithm
//! For each transformer layer l:
//!   BI(l) = 1 - `cosine_similarity(input_l`, `output_l`)
//!
//! Layers with lowest BI contribute least to output transformation.
//!
//! # Key Insight
//! If a layer's output is very similar to its input, the layer
//! is performing minimal transformation and can be removed.
//!
//! # References
//! - Muralidharan, S., et al. (2024). Compact language models via pruning
//!   and knowledge distillation. arXiv:2407.14679.
//! - Men, X., et al. (2024). `ShortGPT`: Layers in large language models
//!   are more redundant than you expect. arXiv:2403.03853.

use super::error::PruningError;
use crate::autograd::Tensor;

/// Result of depth (layer) pruning operation.
#[derive(Debug, Clone)]
pub struct DepthPruningResult {
    /// List of (`layer_index`, `block_importance_score`) for removed layers.
    pub removed_layers: Vec<(usize, f32)>,
    /// Original number of layers.
    pub original_depth: usize,
    /// Final number of layers after pruning.
    pub final_depth: usize,
}

impl DepthPruningResult {
    /// Create a new depth pruning result.
    #[must_use]
    pub fn new(removed_layers: Vec<(usize, f32)>, original_depth: usize) -> Self {
        let final_depth = original_depth.saturating_sub(removed_layers.len());
        Self {
            removed_layers,
            original_depth,
            final_depth,
        }
    }

    /// Get compression ratio (original / final).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.final_depth == 0 {
            f32::INFINITY
        } else {
            self.original_depth as f32 / self.final_depth as f32
        }
    }

    /// Get percentage of layers removed.
    #[must_use]
    pub fn removal_percentage(&self) -> f32 {
        if self.original_depth == 0 {
            0.0
        } else {
            self.removed_layers.len() as f32 / self.original_depth as f32 * 100.0
        }
    }
}

/// Block Importance (BI) scores for all layers.
#[derive(Debug, Clone)]
pub struct BlockImportanceScores {
    /// Scores per layer: (`layer_index`, `block_importance`)
    pub scores: Vec<(usize, f32)>,
    /// Number of calibration samples used.
    pub num_samples: usize,
}

impl BlockImportanceScores {
    /// Create new block importance scores.
    #[must_use]
    pub fn new(scores: Vec<(usize, f32)>, num_samples: usize) -> Self {
        Self {
            scores,
            num_samples,
        }
    }

    /// Get score for a specific layer.
    #[must_use]
    pub fn get(&self, layer_idx: usize) -> Option<f32> {
        self.scores
            .iter()
            .find(|(idx, _)| *idx == layer_idx)
            .map(|(_, score)| *score)
    }

    /// Get layers sorted by importance (ascending - least important first).
    #[must_use]
    pub fn sorted_by_importance(&self) -> Vec<(usize, f32)> {
        let mut sorted = self.scores.clone();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted
    }

    /// Get the N least important layers.
    #[must_use]
    pub fn least_important(&self, n: usize) -> Vec<(usize, f32)> {
        self.sorted_by_importance().into_iter().take(n).collect()
    }
}

/// Minitron Depth Pruner: Layer removal based on Block Importance.
///
/// # Algorithm
/// 1. Compute Block Importance (BI) for each layer using calibration data
/// 2. BI(l) = 1 - `cosine_similarity(input_l`, `output_l`)
/// 3. Remove layers with lowest BI (least contribution to output)
///
/// # Toyota Way: Genchi Genbutsu
/// Uses real calibration data to measure actual layer contributions.
#[derive(Debug, Clone)]
pub struct DepthPruner {
    /// Number of layers to remove.
    num_layers_to_remove: usize,
    /// Whether to use iterative removal (recompute BI after each removal).
    iterative: bool,
    /// Minimum layers to keep (safety constraint).
    min_layers: usize,
}

impl DepthPruner {
    /// Create a new depth pruner.
    ///
    /// # Arguments
    /// * `num_layers_to_remove` - Number of layers to remove
    #[must_use]
    pub fn new(num_layers_to_remove: usize) -> Self {
        Self {
            num_layers_to_remove,
            iterative: true,
            min_layers: 1,
        }
    }

    /// Set whether to use iterative removal.
    ///
    /// Iterative: Recompute BI after each layer removal (more accurate, slower).
    /// One-shot: Compute BI once and remove all at once (faster, may be less optimal).
    #[must_use]
    pub fn with_iterative(mut self, iterative: bool) -> Self {
        self.iterative = iterative;
        self
    }

    /// Set minimum number of layers to keep.
    #[must_use]
    pub fn with_min_layers(mut self, min_layers: usize) -> Self {
        self.min_layers = min_layers;
        self
    }

    /// Get number of layers to remove.
    #[must_use]
    pub fn num_layers_to_remove(&self) -> usize {
        self.num_layers_to_remove
    }

    /// Check if iterative mode is enabled.
    #[must_use]
    pub fn is_iterative(&self) -> bool {
        self.iterative
    }

    /// Compute cosine similarity between two tensors.
    ///
    /// `cos_sim` = (a · b) / (||a|| * ||b||)
    pub fn cosine_similarity(a: &Tensor, b: &Tensor) -> Result<f32, PruningError> {
        let a_data = a.data();
        let b_data = b.data();

        if a_data.len() != b_data.len() {
            return Err(PruningError::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        if a_data.is_empty() {
            return Ok(1.0); // Empty tensors are identical
        }

        let mut dot_product = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;

        for i in 0..a_data.len() {
            dot_product += a_data[i] * b_data[i];
            norm_a += a_data[i] * a_data[i];
            norm_b += b_data[i] * b_data[i];
        }

        norm_a = norm_a.sqrt();
        norm_b = norm_b.sqrt();

        // Handle zero vectors
        if norm_a < 1e-10 || norm_b < 1e-10 {
            if norm_a < 1e-10 && norm_b < 1e-10 {
                return Ok(1.0); // Both zero - consider identical
            }
            return Ok(0.0); // One zero - orthogonal
        }

        let cos_sim = dot_product / (norm_a * norm_b);

        // Clamp to [-1, 1] for numerical stability
        Ok(cos_sim.clamp(-1.0, 1.0))
    }

    /// Compute Block Importance for a single layer.
    ///
    /// BI = 1 - `cosine_similarity(input`, output)
    ///
    /// # Returns
    /// Block importance score in range [0, 2]:
    /// - 0: Output identical to input (layer does nothing)
    /// - 1: Output orthogonal to input
    /// - 2: Output opposite to input
    pub fn compute_layer_importance(input: &Tensor, output: &Tensor) -> Result<f32, PruningError> {
        let cos_sim = Self::cosine_similarity(input, output)?;
        Ok(1.0 - cos_sim)
    }

    /// Compute Block Importance scores for all layers.
    ///
    /// # Arguments
    /// * `layer_inputs` - Input activations for each layer
    /// * `layer_outputs` - Output activations for each layer
    ///
    /// # Returns
    /// Block importance scores for all layers
    pub fn compute_block_importance(
        &self,
        layer_inputs: &[Tensor],
        layer_outputs: &[Tensor],
    ) -> Result<BlockImportanceScores, PruningError> {
        if layer_inputs.len() != layer_outputs.len() {
            return Err(PruningError::ShapeMismatch {
                expected: vec![layer_inputs.len()],
                got: vec![layer_outputs.len()],
            });
        }

        if layer_inputs.is_empty() {
            return Ok(BlockImportanceScores::new(vec![], 0));
        }

        let mut scores = Vec::with_capacity(layer_inputs.len());

        for (idx, (input, output)) in layer_inputs.iter().zip(layer_outputs.iter()).enumerate() {
            let bi = Self::compute_layer_importance(input, output)?;
            scores.push((idx, bi));
        }

        Ok(BlockImportanceScores::new(scores, 1))
    }

    /// Select layers to remove based on Block Importance scores.
    ///
    /// # Arguments
    /// * `scores` - Block importance scores for all layers
    /// * `num_layers` - Total number of layers
    ///
    /// # Returns
    /// Indices of layers to remove (sorted descending for safe removal)
    pub fn select_layers_to_remove(
        &self,
        scores: &BlockImportanceScores,
        num_layers: usize,
    ) -> Result<Vec<usize>, PruningError> {
        // Validate we can remove requested layers
        let max_removable = num_layers.saturating_sub(self.min_layers);

        if self.num_layers_to_remove > max_removable {
            return Err(PruningError::InvalidSparsity {
                value: self.num_layers_to_remove as f32,
                constraint: format!(
                    "Cannot remove {} layers from {} total (min {} required, max removable: {})",
                    self.num_layers_to_remove, num_layers, self.min_layers, max_removable
                ),
            });
        }

        let actual_remove = self.num_layers_to_remove;

        // Get least important layers
        let to_remove: Vec<usize> = scores
            .least_important(actual_remove)
            .into_iter()
            .map(|(idx, _)| idx)
            .collect();

        // Sort descending so we can remove from highest index first
        let mut sorted = to_remove;
        sorted.sort_by(|a, b| b.cmp(a));

        Ok(sorted)
    }

    /// Validate layer removal configuration.
    pub fn validate(&self, num_layers: usize) -> Result<(), PruningError> {
        if num_layers < self.min_layers {
            return Err(PruningError::InvalidSparsity {
                value: num_layers as f32,
                constraint: format!(
                    "Model has {} layers but minimum is {}",
                    num_layers, self.min_layers
                ),
            });
        }

        let max_removable = num_layers.saturating_sub(self.min_layers);
        if self.num_layers_to_remove > max_removable {
            return Err(PruningError::InvalidSparsity {
                value: self.num_layers_to_remove as f32,
                constraint: format!(
                    "Cannot remove {} layers from {} (max removable: {})",
                    self.num_layers_to_remove, num_layers, max_removable
                ),
            });
        }

        Ok(())
    }
}

impl Default for DepthPruner {
    fn default() -> Self {
        Self::new(0)
    }
}

#[cfg(test)]
#[path = "depth_tests.rs"]
mod tests;
