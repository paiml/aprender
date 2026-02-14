use super::*;

// ==========================================================================
// FALSIFICATION: Cosine similarity computation
// ==========================================================================
#[test]
fn test_cosine_similarity_identical() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

    let sim = DepthPruner::cosine_similarity(&a, &b).unwrap();
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "DEP-01 FALSIFIED: Identical vectors should have similarity 1.0, got {}",
        sim
    );
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = Tensor::new(&[1.0, 0.0, 0.0], &[3]);
    let b = Tensor::new(&[0.0, 1.0, 0.0], &[3]);

    let sim = DepthPruner::cosine_similarity(&a, &b).unwrap();
    assert!(
        sim.abs() < 1e-5,
        "DEP-02 FALSIFIED: Orthogonal vectors should have similarity 0.0, got {}",
        sim
    );
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[-1.0, -2.0, -3.0], &[3]);

    let sim = DepthPruner::cosine_similarity(&a, &b).unwrap();
    assert!(
        (sim - (-1.0)).abs() < 1e-5,
        "DEP-03 FALSIFIED: Opposite vectors should have similarity -1.0, got {}",
        sim
    );
}

#[test]
fn test_cosine_similarity_zero_vectors() {
    let a = Tensor::new(&[0.0, 0.0, 0.0], &[3]);
    let b = Tensor::new(&[0.0, 0.0, 0.0], &[3]);

    let sim = DepthPruner::cosine_similarity(&a, &b).unwrap();
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "DEP-04 FALSIFIED: Zero vectors should be treated as identical, got {}",
        sim
    );
}

#[test]
fn test_cosine_similarity_one_zero() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[0.0, 0.0, 0.0], &[3]);

    let sim = DepthPruner::cosine_similarity(&a, &b).unwrap();
    assert!(
        sim.abs() < 1e-5,
        "DEP-05 FALSIFIED: One zero vector should give similarity 0.0, got {}",
        sim
    );
}

#[test]
fn test_cosine_similarity_shape_mismatch() {
    let a = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let b = Tensor::new(&[1.0, 2.0], &[2]);

    let result = DepthPruner::cosine_similarity(&a, &b);
    assert!(
        result.is_err(),
        "DEP-06 FALSIFIED: Should error on shape mismatch"
    );
}

#[test]
fn test_cosine_similarity_empty() {
    let a = Tensor::new(&[], &[0]);
    let b = Tensor::new(&[], &[0]);

    let sim = DepthPruner::cosine_similarity(&a, &b).unwrap();
    assert!(
        (sim - 1.0).abs() < 1e-5,
        "DEP-07 FALSIFIED: Empty vectors should be identical"
    );
}

// ==========================================================================
// FALSIFICATION: Block Importance computation
// ==========================================================================
#[test]
fn test_block_importance_identical_io() {
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = Tensor::new(&[1.0, 2.0, 3.0], &[3]);

    let bi = DepthPruner::compute_layer_importance(&input, &output).unwrap();
    assert!(
        bi.abs() < 1e-5,
        "DEP-08 FALSIFIED: Identical I/O should give BI=0, got {}",
        bi
    );
}

#[test]
fn test_block_importance_orthogonal_io() {
    let input = Tensor::new(&[1.0, 0.0, 0.0], &[3]);
    let output = Tensor::new(&[0.0, 1.0, 0.0], &[3]);

    let bi = DepthPruner::compute_layer_importance(&input, &output).unwrap();
    assert!(
        (bi - 1.0).abs() < 1e-5,
        "DEP-09 FALSIFIED: Orthogonal I/O should give BI=1, got {}",
        bi
    );
}

#[test]
fn test_block_importance_opposite_io() {
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = Tensor::new(&[-1.0, -2.0, -3.0], &[3]);

    let bi = DepthPruner::compute_layer_importance(&input, &output).unwrap();
    assert!(
        (bi - 2.0).abs() < 1e-5,
        "DEP-10 FALSIFIED: Opposite I/O should give BI=2, got {}",
        bi
    );
}

// ==========================================================================
// FALSIFICATION: Batch block importance computation
// ==========================================================================
#[test]
fn test_compute_block_importance_multiple_layers() {
    let pruner = DepthPruner::new(1);

    let inputs = vec![
        Tensor::new(&[1.0, 0.0], &[2]),
        Tensor::new(&[1.0, 0.0], &[2]),
        Tensor::new(&[1.0, 0.0], &[2]),
    ];
    let outputs = vec![
        Tensor::new(&[1.0, 0.0], &[2]), // BI = 0 (identical)
        Tensor::new(&[0.9, 0.1], &[2]), // BI > 0 (slightly different)
        Tensor::new(&[0.0, 1.0], &[2]), // BI = 1 (orthogonal)
    ];

    let scores = pruner.compute_block_importance(&inputs, &outputs).unwrap();

    assert_eq!(
        scores.scores.len(),
        3,
        "DEP-11 FALSIFIED: Should have 3 scores"
    );

    // Layer 0 should have lowest BI
    let layer0_score = scores.get(0).unwrap();
    assert!(
        layer0_score.abs() < 0.01,
        "DEP-11 FALSIFIED: Layer 0 should have BI≈0, got {}",
        layer0_score
    );

    // Layer 2 should have highest BI
    let layer2_score = scores.get(2).unwrap();
    assert!(
        (layer2_score - 1.0).abs() < 0.01,
        "DEP-11 FALSIFIED: Layer 2 should have BI≈1, got {}",
        layer2_score
    );
}

#[test]
fn test_compute_block_importance_empty() {
    let pruner = DepthPruner::new(0);

    let scores = pruner.compute_block_importance(&[], &[]).unwrap();
    assert!(
        scores.scores.is_empty(),
        "DEP-12 FALSIFIED: Empty input should give empty scores"
    );
}

#[test]
fn test_compute_block_importance_mismatched_lengths() {
    let pruner = DepthPruner::new(1);

    let inputs = vec![Tensor::new(&[1.0], &[1])];
    let outputs = vec![Tensor::new(&[1.0], &[1]), Tensor::new(&[1.0], &[1])];

    let result = pruner.compute_block_importance(&inputs, &outputs);
    assert!(
        result.is_err(),
        "DEP-13 FALSIFIED: Should error on mismatched input/output lengths"
    );
}

// ==========================================================================
// FALSIFICATION: BlockImportanceScores helpers
// ==========================================================================
#[test]
fn test_block_importance_scores_sorted() {
    let scores = BlockImportanceScores::new(vec![(0, 0.5), (1, 0.1), (2, 0.9)], 1);

    let sorted = scores.sorted_by_importance();
    assert_eq!(
        sorted[0].0, 1,
        "DEP-14 FALSIFIED: Layer 1 should be first (lowest BI)"
    );
    assert_eq!(sorted[1].0, 0, "DEP-14 FALSIFIED: Layer 0 should be second");
    assert_eq!(
        sorted[2].0, 2,
        "DEP-14 FALSIFIED: Layer 2 should be last (highest BI)"
    );
}

#[test]
fn test_block_importance_scores_least_important() {
    let scores = BlockImportanceScores::new(vec![(0, 0.5), (1, 0.1), (2, 0.9), (3, 0.3)], 1);

    let least = scores.least_important(2);
    assert_eq!(least.len(), 2, "DEP-15 FALSIFIED: Should return 2 layers");
    assert_eq!(least[0].0, 1, "DEP-15 FALSIFIED: Layer 1 should be first");
    assert_eq!(least[1].0, 3, "DEP-15 FALSIFIED: Layer 3 should be second");
}

// ==========================================================================
// FALSIFICATION: Layer selection
// ==========================================================================
#[test]
fn test_select_layers_to_remove_basic() {
    let pruner = DepthPruner::new(2).with_min_layers(1);
    let scores = BlockImportanceScores::new(vec![(0, 0.5), (1, 0.1), (2, 0.9), (3, 0.3)], 1);

    let to_remove = pruner.select_layers_to_remove(&scores, 4).unwrap();

    assert_eq!(
        to_remove.len(),
        2,
        "DEP-16 FALSIFIED: Should select 2 layers"
    );
    // Should be sorted descending
    assert!(
        to_remove[0] > to_remove[1],
        "DEP-16 FALSIFIED: Should be sorted descending"
    );
}

#[test]
fn test_select_layers_respects_min_layers() {
    let pruner = DepthPruner::new(5).with_min_layers(2);
    let scores = BlockImportanceScores::new(vec![(0, 0.1), (1, 0.2), (2, 0.3)], 1);

    let result = pruner.select_layers_to_remove(&scores, 3);
    assert!(
        result.is_err(),
        "DEP-17 FALSIFIED: Should error when removal violates min_layers"
    );
}

#[test]
fn test_select_layers_errors_on_excessive_removal() {
    let pruner = DepthPruner::new(10).with_min_layers(1);
    let scores = BlockImportanceScores::new(vec![(0, 0.1), (1, 0.2), (2, 0.3)], 1);

    // 3 layers with min_layers=1 means max_removable=2
    // Requesting 10 should error, not silently clamp
    let result = pruner.select_layers_to_remove(&scores, 3);
    assert!(
        result.is_err(),
        "DEP-18 FALSIFIED: Should error when requesting more removal than allowed"
    );
}

#[test]
fn test_select_layers_exact_max() {
    let pruner = DepthPruner::new(2).with_min_layers(1);
    let scores = BlockImportanceScores::new(vec![(0, 0.1), (1, 0.2), (2, 0.3)], 1);

    // 3 layers with min_layers=1 means max_removable=2 (exactly what we request)
    let to_remove = pruner.select_layers_to_remove(&scores, 3).unwrap();
    assert_eq!(
        to_remove.len(),
        2,
        "DEP-18b FALSIFIED: Should allow exact max removal"
    );
}

// ==========================================================================
// FALSIFICATION: DepthPruningResult
// ==========================================================================
#[test]
fn test_depth_pruning_result_compression_ratio() {
    let result = DepthPruningResult::new(vec![(0, 0.1), (1, 0.2)], 10);

    assert_eq!(result.final_depth, 8);
    assert!(
        (result.compression_ratio() - 1.25).abs() < 1e-5,
        "DEP-19 FALSIFIED: Compression ratio should be 1.25"
    );
}

#[test]
fn test_depth_pruning_result_removal_percentage() {
    let result = DepthPruningResult::new(vec![(0, 0.1), (1, 0.2)], 10);

    assert!(
        (result.removal_percentage() - 20.0).abs() < 1e-5,
        "DEP-20 FALSIFIED: Removal percentage should be 20%"
    );
}

#[test]
fn test_depth_pruning_result_empty() {
    let result = DepthPruningResult::new(vec![], 0);

    assert_eq!(result.final_depth, 0);
    assert_eq!(result.removal_percentage(), 0.0);
}

#[test]
fn test_depth_pruning_result_all_removed() {
    let result = DepthPruningResult::new(vec![(0, 0.1)], 1);

    assert_eq!(result.final_depth, 0);
    assert_eq!(result.compression_ratio(), f32::INFINITY);
}

// ==========================================================================
// FALSIFICATION: DepthPruner configuration
// ==========================================================================
#[test]
fn test_depth_pruner_builder() {
    let pruner = DepthPruner::new(3).with_iterative(false).with_min_layers(2);

    assert_eq!(pruner.num_layers_to_remove(), 3);
    assert!(!pruner.is_iterative());
    assert_eq!(pruner.min_layers, 2);
}

#[test]
fn test_depth_pruner_default() {
    let pruner = DepthPruner::default();
    assert_eq!(pruner.num_layers_to_remove(), 0);
    assert!(pruner.is_iterative());
}

#[test]
fn test_depth_pruner_validate_success() {
    let pruner = DepthPruner::new(3).with_min_layers(2);
    assert!(pruner.validate(10).is_ok());
}

#[test]
fn test_depth_pruner_validate_too_few_layers() {
    let pruner = DepthPruner::new(3).with_min_layers(5);
    assert!(pruner.validate(3).is_err());
}

#[test]
fn test_depth_pruner_validate_too_many_to_remove() {
    let pruner = DepthPruner::new(8).with_min_layers(2);
    let result = pruner.validate(5);
    assert!(
        result.is_err(),
        "DEP-21 FALSIFIED: Should error when trying to remove 8 from 5 (min 2)"
    );
}

// ==========================================================================
// FALSIFICATION: Clone and Debug
// ==========================================================================
#[test]
fn test_depth_pruner_clone() {
    let orig = DepthPruner::new(5).with_iterative(false);
    let cloned = orig.clone();

    assert_eq!(orig.num_layers_to_remove(), cloned.num_layers_to_remove());
    assert_eq!(orig.is_iterative(), cloned.is_iterative());
}

#[test]
fn test_depth_pruner_debug() {
    let pruner = DepthPruner::new(3);
    let debug = format!("{:?}", pruner);
    assert!(debug.contains("DepthPruner"));
}

#[test]
fn test_block_importance_scores_debug() {
    let scores = BlockImportanceScores::new(vec![(0, 0.5)], 1);
    let debug = format!("{:?}", scores);
    assert!(debug.contains("BlockImportanceScores"));
}

#[test]
fn test_depth_pruning_result_debug() {
    let result = DepthPruningResult::new(vec![], 5);
    let debug = format!("{:?}", result);
    assert!(debug.contains("DepthPruningResult"));
}
