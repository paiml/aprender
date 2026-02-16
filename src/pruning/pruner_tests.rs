pub(crate) use super::*;
pub(crate) use crate::autograd::Tensor;

// Mock module for testing
pub(super) struct MockModule {
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

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed for valid module");
    let mask = pruner
        .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
        .expect("unstructured mask generation should succeed");

    assert!((mask.sparsity() - 0.5).abs() < 1e-6);
}

#[test]
fn test_magnitude_pruner_generate_mask_nm() {
    let pruner = MagnitudePruner::new();
    let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[8]);

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed for valid module");
    let mask = pruner
        .generate_mask(&scores, 0.5, SparsityPattern::NM { n: 2, m: 4 })
        .expect("N:M mask generation should succeed");

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

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed");
    let mask = pruner
        .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
        .expect("unstructured mask generation should succeed");
    let result = pruner
        .apply_mask(&mut module, &mask)
        .expect("mask application should succeed");

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
    .expect("prune_module should succeed for valid inputs");

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

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed");
    let mask = pruner
        .generate_mask(&scores, 1.0, SparsityPattern::Unstructured)
        .expect("full sparsity mask generation should succeed");
    let result = pruner
        .apply_mask(&mut module, &mask)
        .expect("mask application should succeed");

    assert!((result.achieved_sparsity - 1.0).abs() < 1e-6);
}

// ==========================================================================
// FALSIFICATION: Edge case - 0% sparsity
// ==========================================================================
#[test]
fn test_magnitude_pruner_zero_sparsity() {
    let pruner = MagnitudePruner::new();
    let mut module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[4]);

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed");
    let mask = pruner
        .generate_mask(&scores, 0.0, SparsityPattern::Unstructured)
        .expect("zero sparsity mask generation should succeed");
    let result = pruner
        .apply_mask(&mut module, &mask)
        .expect("mask application should succeed");

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
        .expect("row mask generation should succeed");
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
        .expect("column mask generation should succeed");
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
        .expect("N:M mask generation should succeed");
    assert!((mask.sparsity() - 0.5).abs() < 1e-6);
}

#[test]
fn test_wanda_pruner_generate_mask_unstructured() {
    let pruner = WandaPruner::new("layer0");
    let scores =
        ImportanceScores::new(Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]), "test".to_string());

    let mask = pruner
        .generate_mask(&scores, 0.5, SparsityPattern::Unstructured)
        .expect("unstructured mask generation should succeed");
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
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured)
        .expect("mask creation should succeed for valid tensor");

    let result = pruner
        .apply_mask(&mut module, &mask)
        .expect("mask application should succeed");
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
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured)
        .expect("mask creation should succeed for valid tensor");

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

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed");
    let mask = pruner
        .generate_mask(&scores, 0.5, SparsityPattern::Row)
        .expect("row mask generation should succeed");
    assert!((mask.sparsity() - 0.5).abs() < 1e-6);
}

#[test]
fn test_magnitude_pruner_generate_mask_column() {
    let pruner = MagnitudePruner::new();
    let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed");
    let mask = pruner
        .generate_mask(&scores, 0.5, SparsityPattern::Column)
        .expect("column mask generation should succeed");
    assert!((mask.sparsity() - 0.5).abs() < 1e-6);
}

#[test]
fn test_magnitude_pruner_generate_mask_block() {
    let pruner = MagnitudePruner::new();
    let module = MockModule::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

    let scores = pruner
        .importance()
        .compute(&module, None)
        .expect("importance computation should succeed");
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
    let mask = SparsityMask::new(mask_tensor, SparsityPattern::Unstructured)
        .expect("mask creation should succeed for valid tensor");

    let result = pruner.apply_mask(&mut module, &mask);
    assert!(result.is_err());
}
