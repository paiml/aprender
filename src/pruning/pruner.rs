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
mod tests {
    use super::*;
    use crate::autograd::Tensor;

    // Mock module for testing
    struct MockModule {
        weights: Tensor,
    }

    impl MockModule {
        fn new(data: &[f32], shape: &[usize]) -> Self {
            Self {
                weights: Tensor::new(data, shape),
            }
        }
    }

    impl Module for MockModule {
        fn forward(&self, input: &Tensor) -> Tensor {
            input.clone()
        }

        fn parameters(&self) -> Vec<&Tensor> {
            vec![&self.weights]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
            vec![&mut self.weights]
        }
    }

    // ==========================================================================
    // FALSIFICATION: PruningResult construction
    // ==========================================================================
    #[test]
    fn test_pruning_result_new() {
        let result = PruningResult::new(0.5, 50, 100);

        assert!((result.achieved_sparsity - 0.5).abs() < 1e-6);
        assert_eq!(result.parameters_pruned, 50);
        assert_eq!(result.total_parameters, 100);
        assert_eq!(result.memory_savings_bytes, 200); // 50 * 4 bytes
    }

    #[test]
    fn test_pruning_result_compression_ratio() {
        let result = PruningResult::new(0.5, 50, 100);
        // 50% sparsity = 2x compression
        assert!((result.compression_ratio() - 2.0).abs() < 1e-6);

        let result = PruningResult::new(0.0, 0, 100);
        // 0% sparsity = 1x compression (no compression)
        assert!((result.compression_ratio() - 1.0).abs() < 1e-6);

        let result = PruningResult::new(0.75, 75, 100);
        // 75% sparsity = 4x compression
        assert!((result.compression_ratio() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_pruning_result_with_layer_sparsity() {
        let result = PruningResult::new(0.5, 50, 100)
            .with_layer_sparsity("layer0".to_string(), 0.4)
            .with_layer_sparsity("layer1".to_string(), 0.6);

        assert_eq!(result.layer_sparsity.len(), 2);
        assert!((result.layer_sparsity["layer0"] - 0.4).abs() < 1e-6);
        assert!((result.layer_sparsity["layer1"] - 0.6).abs() < 1e-6);
    }

    // ==========================================================================
    // FALSIFICATION: Pruner trait is object-safe
    // ==========================================================================
    #[test]
    fn test_pruner_trait_object_safe() {
        fn accept_dyn(_: &dyn Pruner) {}
        let pruner = MagnitudePruner::new();
        accept_dyn(&pruner);
    }

    // ==========================================================================
    // FALSIFICATION: MagnitudePruner construction
    // ==========================================================================
    #[test]
    fn test_magnitude_pruner_new() {
        let pruner = MagnitudePruner::new();
        assert_eq!(pruner.name(), "magnitude_pruner");
        assert!(!pruner.importance().requires_calibration());
    }

    #[test]
    fn test_magnitude_pruner_l1() {
        let pruner = MagnitudePruner::l1();
        assert_eq!(pruner.importance().name(), "magnitude_l1");
    }

    #[test]
    fn test_magnitude_pruner_l2() {
        let pruner = MagnitudePruner::l2();
        assert_eq!(pruner.importance().name(), "magnitude_l2");
    }

    // ==========================================================================
    // FALSIFICATION: MagnitudePruner generates correct masks
    // ==========================================================================
    #[test]
    fn test_magnitude_pruner_generate_mask_unstructured() {
        let pruner = MagnitudePruner::new();
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
            .unwrap();

        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude_pruner_generate_mask_nm() {
        let pruner = MagnitudePruner::new();
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::NM { n: 2, m: 4 })
            .unwrap();

        // 2:4 = 50% sparsity
        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    // ==========================================================================
    // FALSIFICATION: MagnitudePruner applies masks correctly
    // ==========================================================================
    #[test]
    fn test_magnitude_pruner_apply_mask() {
        let pruner = MagnitudePruner::new();
        let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
            .unwrap();
        let result = pruner.apply_mask(&mut module, &mask).unwrap();

        assert!((result.achieved_sparsity - 0.5).abs() < 1e-6);
        assert_eq!(result.parameters_pruned, 2);
        assert_eq!(result.total_parameters, 4);
    }

    // ==========================================================================
    // FALSIFICATION: prune_module convenience function
    // ==========================================================================
    #[test]
    fn test_prune_module_convenience() {
        let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let pruner = MagnitudePruner::new();

        let result = prune_module(
            &mut module,
            &pruner,
            0.5,
            SparsityPattern::Unstructured,
            None,
        )
        .unwrap();

        assert!((result.achieved_sparsity - 0.5).abs() < 1e-6);
    }

    // ==========================================================================
    // FALSIFICATION: WandaPruner construction
    // ==========================================================================
    #[test]
    fn test_wanda_pruner_new() {
        let pruner = WandaPruner::new("layer0");
        assert_eq!(pruner.name(), "wanda_pruner");
        assert!(pruner.importance().requires_calibration());
    }

    // ==========================================================================
    // FALSIFICATION: Default implementations
    // ==========================================================================
    #[test]
    fn test_pruning_result_default() {
        let result = PruningResult::default();
        assert_eq!(result.achieved_sparsity, 0.0);
        assert_eq!(result.parameters_pruned, 0);
        assert_eq!(result.total_parameters, 0);
    }

    #[test]
    fn test_magnitude_pruner_default() {
        let pruner = MagnitudePruner::default();
        assert_eq!(pruner.name(), "magnitude_pruner");
    }

    // ==========================================================================
    // FALSIFICATION: Edge case - 100% sparsity
    // ==========================================================================
    #[test]
    fn test_magnitude_pruner_full_sparsity() {
        let pruner = MagnitudePruner::new();
        let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let mask = pruner
            .generate_mask(&scores, 1.0, SparsityPattern::Unstructured)
            .unwrap();
        let result = pruner.apply_mask(&mut module, &mask).unwrap();

        assert!((result.achieved_sparsity - 1.0).abs() < 1e-6);
    }

    // ==========================================================================
    // FALSIFICATION: Edge case - 0% sparsity
    // ==========================================================================
    #[test]
    fn test_magnitude_pruner_zero_sparsity() {
        let pruner = MagnitudePruner::new();
        let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let mask = pruner
            .generate_mask(&scores, 0.0, SparsityPattern::Unstructured)
            .unwrap();
        let result = pruner.apply_mask(&mut module, &mask).unwrap();

        // Should keep all weights
        assert!((result.achieved_sparsity - 0.0).abs() < 1e-6);
    }

    // ==========================================================================
    // FALSIFICATION: Compression ratio edge cases
    // ==========================================================================
    #[test]
    fn test_compression_ratio_empty() {
        let result = PruningResult::new(0.0, 0, 0);
        // 0 total parameters is a degenerate case - return INFINITY
        assert!(result.compression_ratio().is_infinite());
    }

    #[test]
    fn test_compression_ratio_full_sparsity() {
        let result = PruningResult::new(1.0, 100, 100);
        assert!(result.compression_ratio().is_infinite());
    }

    // ==========================================================================
    // FALSIFICATION: WandaPruner generate_mask patterns
    // ==========================================================================
    #[test]
    fn test_wanda_pruner_generate_mask_row() {
        let pruner = WandaPruner::new("layer0");
        let scores = ImportanceScores::new(
            Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            "test".to_string(),
        );

        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Row)
            .unwrap();
        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wanda_pruner_generate_mask_column() {
        let pruner = WandaPruner::new("layer0");
        let scores = ImportanceScores::new(
            Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]),
            "test".to_string(),
        );

        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Column)
            .unwrap();
        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wanda_pruner_generate_mask_block() {
        let pruner = WandaPruner::new("layer0");
        let scores = ImportanceScores::new(
            Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]),
            "test".to_string(),
        );

        let result = pruner.generate_mask(
            &scores,
            0.5,
            SparsityPattern::Block {
                height: 1,
                width: 1,
            },
        );
        // Block mask on 3x3 with 1x1 blocks should work
        assert!(result.is_ok());
    }

    #[test]
    fn test_wanda_pruner_generate_mask_nm() {
        let pruner = WandaPruner::new("layer0");
        let scores = ImportanceScores::new(
            Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]),
            "test".to_string(),
        );

        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::NM { n: 2, m: 4 })
            .unwrap();
        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_wanda_pruner_generate_mask_unstructured() {
        let pruner = WandaPruner::new("layer0");
        let scores =
            ImportanceScores::new(Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]), "test".to_string());

        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
            .unwrap();
        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    // ==========================================================================
    // FALSIFICATION: WandaPruner apply_mask
    // ==========================================================================
    #[test]
    fn test_wanda_pruner_apply_mask() {
        let pruner = WandaPruner::new("layer0");
        let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

        let mask_tensor = Tensor::new(&[1.0, 0.0, 1.0, 0.0], &[4]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        let result = pruner.apply_mask(&mut module, &mask).unwrap();
        assert_eq!(result.parameters_pruned, 2);
        assert_eq!(result.total_parameters, 4);
    }

    #[test]
    fn test_wanda_pruner_apply_mask_empty_module() {
        struct EmptyModule;
        impl Module for EmptyModule {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.clone()
            }
            fn parameters(&self) -> Vec<&Tensor> {
                vec![]
            }
            fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
                vec![]
            }
        }

        let pruner = WandaPruner::new("layer0");
        let mut module = EmptyModule;
        let mask_tensor = Tensor::new(&[1.0], &[1]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        let result = pruner.apply_mask(&mut module, &mask);
        assert!(result.is_err());
    }

    // ==========================================================================
    // FALSIFICATION: MagnitudePruner Row/Column patterns
    // ==========================================================================
    #[test]
    fn test_magnitude_pruner_generate_mask_row() {
        let pruner = MagnitudePruner::new();
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Row)
            .unwrap();
        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude_pruner_generate_mask_column() {
        let pruner = MagnitudePruner::new();
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let mask = pruner
            .generate_mask(&scores, 0.5, SparsityPattern::Column)
            .unwrap();
        assert!((mask.sparsity() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_magnitude_pruner_generate_mask_block() {
        let pruner = MagnitudePruner::new();
        let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let scores = pruner.importance().compute(&module, None).unwrap();
        let result = pruner.generate_mask(
            &scores,
            0.5,
            SparsityPattern::Block {
                height: 1,
                width: 1,
            },
        );
        assert!(result.is_ok());
    }

    // ==========================================================================
    // FALSIFICATION: MagnitudePruner apply_mask empty module
    // ==========================================================================
    #[test]
    fn test_magnitude_pruner_apply_mask_empty_module() {
        struct EmptyModule;
        impl Module for EmptyModule {
            fn forward(&self, input: &Tensor) -> Tensor {
                input.clone()
            }
            fn parameters(&self) -> Vec<&Tensor> {
                vec![]
            }
            fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
                vec![]
            }
        }

        let pruner = MagnitudePruner::new();
        let mut module = EmptyModule;
        let mask_tensor = Tensor::new(&[1.0], &[1]);
        let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured).unwrap();

        let result = pruner.apply_mask(&mut module, &mask);
        assert!(result.is_err());
    }
}
