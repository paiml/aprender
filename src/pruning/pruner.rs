//! High-level pruning interface.
//!
//! # Toyota Way: Genchi Genbutsu
//! Pruners operate on actual model weights, not abstractions.
//!
//! # References
//! - Han, S., et al. (2015). Learning both weights and connections. `NeurIPS`.

use super::calibration::CalibrationContext;
use super::error::PruningError;
use super::importance::{Importance, ImportanceScores};
use super::mask::{SparsityMask, SparsityPattern};
use crate::nn::Module;
use std::collections::HashMap;

/// Result of a pruning operation with diagnostics.
///
/// Contains statistics about the pruning operation including
/// achieved sparsity, parameter counts, and per-layer breakdown.
#[derive(Debug, Clone)]
pub struct PruningResult {
    /// Actual achieved sparsity (may differ from target for structured pruning).
    pub achieved_sparsity: f32,
    /// Number of parameters pruned (set to zero).
    pub parameters_pruned: usize,
    /// Total parameters in module.
    pub total_parameters: usize,
    /// Per-layer sparsity breakdown.
    pub layer_sparsity: HashMap<String, f32>,
    /// Estimated memory savings in bytes (assumes FP32).
    pub memory_savings_bytes: usize,
}

impl PruningResult {
    /// Create a new pruning result.
    #[must_use]
    pub fn new(achieved_sparsity: f32, parameters_pruned: usize, total_parameters: usize) -> Self {
        Self {
            achieved_sparsity,
            parameters_pruned,
            total_parameters,
            layer_sparsity: HashMap::new(),
            memory_savings_bytes: parameters_pruned * 4, // FP32 = 4 bytes
        }
    }

    /// Add layer sparsity information.
    #[must_use]
    pub fn with_layer_sparsity(mut self, layer_name: String, sparsity: f32) -> Self {
        self.layer_sparsity.insert(layer_name, sparsity);
        self
    }

    /// Get compression ratio (original / pruned size).
    #[must_use]
    pub fn compression_ratio(&self) -> f32 {
        if self.total_parameters == 0 || self.achieved_sparsity >= 1.0 {
            return f32::INFINITY;
        }
        1.0 / (1.0 - self.achieved_sparsity)
    }
}

impl Default for PruningResult {
    fn default() -> Self {
        Self::new(0.0, 0, 0)
    }
}

/// High-level pruning interface.
///
/// # Toyota Way: Genchi Genbutsu
/// Pruners must operate on actual model weights, not abstractions.
///
/// # Object Safety
/// This trait is object-safe and can be used with `dyn Pruner`.
pub trait Pruner: Send + Sync {
    /// Generate a sparsity mask based on importance scores.
    ///
    /// # Arguments
    /// * `scores` - Pre-computed importance scores
    /// * `target_sparsity` - Desired fraction of weights to prune (0.0 to 1.0)
    /// * `pattern` - Sparsity pattern constraint (unstructured, N:M, block)
    ///
    /// # Returns
    /// * `Ok(SparsityMask)` - Generated mask
    /// * `Err(PruningError)` - If mask generation fails
    fn generate_mask(
        &self,
        scores: &ImportanceScores,
        target_sparsity: f32,
        pattern: SparsityPattern,
    ) -> Result<SparsityMask, PruningError>;

    /// Apply a sparsity mask to a module, zeroing pruned weights.
    ///
    /// # Arguments
    /// * `module` - The module to prune (modified in-place)
    /// * `mask` - The sparsity mask to apply
    ///
    /// # Returns
    /// * `Ok(PruningResult)` - Statistics about the pruning operation
    /// * `Err(PruningError)` - If mask application fails
    ///
    /// # Safety
    /// This operation modifies weights in-place. The mask must match
    /// the module's parameter shapes exactly.
    fn apply_mask(
        &self,
        module: &mut dyn Module,
        mask: &SparsityMask,
    ) -> Result<PruningResult, PruningError>;

    /// Get the importance estimator used by this pruner.
    fn importance(&self) -> &dyn Importance;

    /// Name of this pruner for logging.
    fn name(&self) -> &'static str;
}

/// Simple magnitude-based pruner.
///
/// Uses weight magnitude as importance and generates masks to achieve
/// the target sparsity by pruning the smallest weights.
#[derive(Debug, Clone)]
pub struct MagnitudePruner {
    importance: super::magnitude::MagnitudeImportance,
}

impl MagnitudePruner {
    /// Create a new magnitude pruner with L2 norm.
    #[must_use]
    pub fn new() -> Self {
        Self {
            importance: super::magnitude::MagnitudeImportance::l2(),
        }
    }

    /// Create a magnitude pruner with L1 norm.
    #[must_use]
    pub fn l1() -> Self {
        Self {
            importance: super::magnitude::MagnitudeImportance::l1(),
        }
    }

    /// Create a magnitude pruner with L2 norm.
    #[must_use]
    pub fn l2() -> Self {
        Self {
            importance: super::magnitude::MagnitudeImportance::l2(),
        }
    }
}

impl Default for MagnitudePruner {
    fn default() -> Self {
        Self::new()
    }
}

impl Pruner for MagnitudePruner {
    fn generate_mask(
        &self,
        scores: &ImportanceScores,
        target_sparsity: f32,
        pattern: SparsityPattern,
    ) -> Result<SparsityMask, PruningError> {
        match pattern {
            SparsityPattern::Unstructured => {
                super::mask::generate_unstructured_mask(&scores.values, target_sparsity)
            }
            SparsityPattern::NM { n, m } => super::mask::generate_nm_mask(&scores.values, n, m),
            SparsityPattern::Block { height, width } => {
                super::mask::generate_block_mask(&scores.values, height, width, target_sparsity)
            }
            SparsityPattern::Row => super::mask::generate_row_mask(&scores.values, target_sparsity),
            SparsityPattern::Column => {
                super::mask::generate_column_mask(&scores.values, target_sparsity)
            }
        }
    }

    fn apply_mask(
        &self,
        module: &mut dyn Module,
        mask: &SparsityMask,
    ) -> Result<PruningResult, PruningError> {
        let mut params = module.parameters_mut();
        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: "unknown".to_string(),
            });
        }

        // Apply mask to first parameter (weight matrix)
        let weights = &mut *params[0];
        let total = weights.data().len();

        mask.apply(weights)?;

        let zeros = weights.data().iter().filter(|&&v| v == 0.0).count();
        let achieved_sparsity = zeros as f32 / total as f32;

        Ok(PruningResult::new(achieved_sparsity, zeros, total))
    }

    fn importance(&self) -> &dyn Importance {
        &self.importance
    }

    fn name(&self) -> &'static str {
        "magnitude_pruner"
    }
}

/// Wanda-based pruner.
///
/// Uses activation-weighted importance (Wanda) and generates masks
/// to achieve the target sparsity. Requires calibration data.
#[derive(Debug, Clone)]
pub struct WandaPruner {
    importance: super::wanda::WandaImportance,
}

impl WandaPruner {
    /// Create a new Wanda pruner for a specific layer.
    ///
    /// # Arguments
    /// * `layer_name` - Layer identifier to look up in `CalibrationContext`
    pub fn new(layer_name: impl Into<String>) -> Self {
        Self {
            importance: super::wanda::WandaImportance::new(layer_name),
        }
    }
}

impl Pruner for WandaPruner {
    fn generate_mask(
        &self,
        scores: &ImportanceScores,
        target_sparsity: f32,
        pattern: SparsityPattern,
    ) -> Result<SparsityMask, PruningError> {
        match pattern {
            SparsityPattern::Unstructured => {
                super::mask::generate_unstructured_mask(&scores.values, target_sparsity)
            }
            SparsityPattern::NM { n, m } => super::mask::generate_nm_mask(&scores.values, n, m),
            SparsityPattern::Block { height, width } => {
                super::mask::generate_block_mask(&scores.values, height, width, target_sparsity)
            }
            SparsityPattern::Row => super::mask::generate_row_mask(&scores.values, target_sparsity),
            SparsityPattern::Column => {
                super::mask::generate_column_mask(&scores.values, target_sparsity)
            }
        }
    }

    fn apply_mask(
        &self,
        module: &mut dyn Module,
        mask: &SparsityMask,
    ) -> Result<PruningResult, PruningError> {
        let mut params = module.parameters_mut();
        if params.is_empty() {
            return Err(PruningError::NoParameters {
                module: "unknown".to_string(),
            });
        }

        let weights = &mut *params[0];
        let total = weights.data().len();

        mask.apply(weights)?;

        let zeros = weights.data().iter().filter(|&&v| v == 0.0).count();
        let achieved_sparsity = zeros as f32 / total as f32;

        Ok(PruningResult::new(achieved_sparsity, zeros, total))
    }

    fn importance(&self) -> &dyn Importance {
        &self.importance
    }

    fn name(&self) -> &'static str {
        "wanda_pruner"
    }
}

/// Convenience function to prune a module with a single call.
///
/// # Arguments
/// * `module` - Module to prune
/// * `pruner` - Pruner to use
/// * `target_sparsity` - Desired sparsity ratio
/// * `pattern` - Sparsity pattern
/// * `context` - Optional calibration context
///
/// # Returns
/// Pruning result with statistics.
pub fn prune_module(
    module: &mut dyn Module,
    pruner: &dyn Pruner,
    target_sparsity: f32,
    pattern: SparsityPattern,
    context: Option<&CalibrationContext>,
) -> Result<PruningResult, PruningError> {
    // Compute importance scores
    let scores = pruner.importance().compute(module, context)?;

    // Generate mask
    let mask = pruner.generate_mask(&scores, target_sparsity, pattern)?;

    // Apply mask
    pruner.apply_mask(module, &mask)
}

#[cfg(test)]
#[path = "pruner_tests.rs"]
mod tests;
