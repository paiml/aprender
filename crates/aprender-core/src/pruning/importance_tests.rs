pub(crate) use super::*;

// ==========================================================================
// FALSIFICATION: ImportanceStats computes correct statistics
// ==========================================================================
#[test]
fn test_importance_stats_from_tensor_known_values() {
    // Known values: [1.0, 2.0, 3.0, 4.0]
    // min=1, max=4, mean=2.5, std=sqrt(1.25)=1.118033988749895
    let tensor = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let stats = ImportanceStats::from_tensor(&tensor);

    assert!(
        (stats.min - 1.0).abs() < 1e-6,
        "IMP-01 FALSIFIED: min should be 1.0, got {}",
        stats.min
    );
    assert!(
        (stats.max - 4.0).abs() < 1e-6,
        "IMP-01 FALSIFIED: max should be 4.0, got {}",
        stats.max
    );
    assert!(
        (stats.mean - 2.5).abs() < 1e-6,
        "IMP-01 FALSIFIED: mean should be 2.5, got {}",
        stats.mean
    );
    // Population std = sqrt((1+0.25+0.25+2.25)/4) = sqrt(1.25) = 1.118...
    assert!(
        (stats.std - 1.118033988749895).abs() < 1e-4,
        "IMP-01 FALSIFIED: std should be ~1.118, got {}",
        stats.std
    );
}

#[test]
fn test_importance_stats_all_same_values() {
    // All same: std should be 0
    let tensor = Tensor::new(&[5.0, 5.0, 5.0, 5.0], &[4]);
    let stats = ImportanceStats::from_tensor(&tensor);

    assert!(
        (stats.min - 5.0).abs() < 1e-6,
        "IMP-02 FALSIFIED: min should be 5.0"
    );
    assert!(
        (stats.max - 5.0).abs() < 1e-6,
        "IMP-02 FALSIFIED: max should be 5.0"
    );
    assert!(
        (stats.mean - 5.0).abs() < 1e-6,
        "IMP-02 FALSIFIED: mean should be 5.0"
    );
    assert!(
        (stats.std - 0.0).abs() < 1e-6,
        "IMP-02 FALSIFIED: std should be 0.0, got {}",
        stats.std
    );
}

#[test]
fn test_importance_stats_single_element() {
    let tensor = Tensor::new(&[42.0], &[1]);
    let stats = ImportanceStats::from_tensor(&tensor);

    assert!(
        (stats.min - 42.0).abs() < 1e-6,
        "IMP-03 FALSIFIED: min should be 42.0"
    );
    assert!(
        (stats.max - 42.0).abs() < 1e-6,
        "IMP-03 FALSIFIED: max should be 42.0"
    );
    assert!(
        (stats.mean - 42.0).abs() < 1e-6,
        "IMP-03 FALSIFIED: mean should be 42.0"
    );
    assert!(
        (stats.std - 0.0).abs() < 1e-6,
        "IMP-03 FALSIFIED: std should be 0.0"
    );
}

// ==========================================================================
// FALSIFICATION: ImportanceStats handles edge cases (Jidoka)
// ==========================================================================
#[test]
fn test_importance_stats_empty_tensor_returns_defaults() {
    // Empty tensor should return safe defaults, not panic
    let tensor = Tensor::new(&[], &[0]);
    let stats = ImportanceStats::from_tensor(&tensor);

    // Should have safe default values (all zeros)
    assert_eq!(
        stats.min, 0.0,
        "IMP-04 FALSIFIED: empty tensor min should be 0"
    );
    assert_eq!(
        stats.max, 0.0,
        "IMP-04 FALSIFIED: empty tensor max should be 0"
    );
    assert_eq!(
        stats.mean, 0.0,
        "IMP-04 FALSIFIED: empty tensor mean should be 0"
    );
    assert_eq!(
        stats.std, 0.0,
        "IMP-04 FALSIFIED: empty tensor std should be 0"
    );
}

#[test]
fn test_importance_stats_negative_values() {
    let tensor = Tensor::new(&[-3.0, -1.0, 0.0, 2.0], &[4]);
    let stats = ImportanceStats::from_tensor(&tensor);

    assert!(
        (stats.min - (-3.0)).abs() < 1e-6,
        "IMP-05 FALSIFIED: min should be -3.0, got {}",
        stats.min
    );
    assert!(
        (stats.max - 2.0).abs() < 1e-6,
        "IMP-05 FALSIFIED: max should be 2.0, got {}",
        stats.max
    );
    assert!(
        (stats.mean - (-0.5)).abs() < 1e-6,
        "IMP-05 FALSIFIED: mean should be -0.5, got {}",
        stats.mean
    );
}

#[test]
fn test_importance_stats_large_range() {
    let tensor = Tensor::new(&[0.0001, 1000000.0], &[2]);
    let stats = ImportanceStats::from_tensor(&tensor);

    assert!(
        (stats.min - 0.0001).abs() < 1e-6,
        "IMP-06 FALSIFIED: min should be 0.0001"
    );
    assert!(
        (stats.max - 1000000.0).abs() < 1.0,
        "IMP-06 FALSIFIED: max should be 1000000.0"
    );
}

// ==========================================================================
// FALSIFICATION: ImportanceScores construction and validation
// ==========================================================================
#[test]
fn test_importance_scores_new_computes_stats() {
    let values = Tensor::new(&[0.1, 0.5, 0.3, 0.2], &[4]);
    let scores = ImportanceScores::new(values.clone(), "magnitude_l2".to_string());

    assert_eq!(
        scores.method, "magnitude_l2",
        "IMP-07 FALSIFIED: method name mismatch"
    );
    assert!(
        (scores.stats.min - 0.1).abs() < 1e-6,
        "IMP-07 FALSIFIED: stats.min should be 0.1"
    );
    assert!(
        (scores.stats.max - 0.5).abs() < 1e-6,
        "IMP-07 FALSIFIED: stats.max should be 0.5"
    );
}

#[test]
fn test_importance_scores_shape_preserved() {
    let values = Tensor::new(&[1.0; 12], &[3, 4]);
    let scores = ImportanceScores::new(values.clone(), "test".to_string());

    assert_eq!(
        scores.shape(),
        &[3, 4],
        "IMP-08 FALSIFIED: shape should be preserved"
    );
}

#[test]
fn test_importance_scores_len() {
    let values = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let scores = ImportanceScores::new(values, "test".to_string());

    assert_eq!(scores.len(), 3, "IMP-09 FALSIFIED: len should be 3");
    assert!(!scores.is_empty(), "IMP-09 FALSIFIED: should not be empty");
}

#[test]
fn test_importance_scores_empty() {
    let values = Tensor::new(&[], &[0]);
    let scores = ImportanceScores::new(values, "test".to_string());

    assert_eq!(scores.len(), 0, "IMP-10 FALSIFIED: len should be 0");
    assert!(scores.is_empty(), "IMP-10 FALSIFIED: should be empty");
}

// ==========================================================================
// FALSIFICATION: Importance trait requirements
// ==========================================================================
#[test]
fn test_importance_trait_is_object_safe() {
    // This compiles only if Importance is object-safe
    fn accept_dyn(_: &dyn Importance) {}

    // The function exists, which proves object safety
    // Actual usage will come with concrete implementations
    let _ = accept_dyn;
}

// ==========================================================================
// FALSIFICATION: Sparsity computation
// ==========================================================================
#[test]
fn test_sparsity_at_threshold() {
    let values = Tensor::new(&[0.1, 0.2, 0.3, 0.4, 0.5], &[5]);
    let stats = ImportanceStats::from_tensor(&values);

    // Threshold 0.3: values [0.1, 0.2] are below = 2/5 = 0.4
    let sparsity = stats.sparsity_at(&values, 0.3);
    assert!(
        (sparsity - 0.4).abs() < 1e-6,
        "IMP-11 FALSIFIED: sparsity at 0.3 should be 0.4, got {}",
        sparsity
    );

    // Threshold 0.6: all values below = 5/5 = 1.0
    let sparsity = stats.sparsity_at(&values, 0.6);
    assert!(
        (sparsity - 1.0).abs() < 1e-6,
        "IMP-11 FALSIFIED: sparsity at 0.6 should be 1.0"
    );

    // Threshold 0.0: no values below = 0/5 = 0.0
    let sparsity = stats.sparsity_at(&values, 0.0);
    assert!(
        (sparsity - 0.0).abs() < 1e-6,
        "IMP-11 FALSIFIED: sparsity at 0.0 should be 0.0"
    );
}

#[test]
fn test_sparsity_at_empty_tensor() {
    let values = Tensor::new(&[], &[0]);
    let stats = ImportanceStats::from_tensor(&values);

    let sparsity = stats.sparsity_at(&values, 0.5);
    assert_eq!(
        sparsity, 0.0,
        "IMP-12 FALSIFIED: sparsity of empty tensor should be 0.0"
    );
}

// ==========================================================================
// FALSIFICATION: Stats default
// ==========================================================================
#[test]
fn test_importance_stats_default() {
    let stats = ImportanceStats::default();
    assert_eq!(stats.min, 0.0);
    assert_eq!(stats.max, 0.0);
    assert_eq!(stats.mean, 0.0);
    assert_eq!(stats.std, 0.0);
    assert!(stats.sparsity_at_threshold.is_empty());
}

// ==========================================================================
// FALSIFICATION: Clone and Debug
// ==========================================================================
#[test]
fn test_importance_stats_clone() {
    let tensor = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let stats = ImportanceStats::from_tensor(&tensor);
    let cloned = stats.clone();

    assert_eq!(stats.min, cloned.min);
    assert_eq!(stats.max, cloned.max);
    assert_eq!(stats.mean, cloned.mean);
    assert_eq!(stats.std, cloned.std);
}

#[test]
fn test_importance_stats_debug() {
    let tensor = Tensor::new(&[1.0, 2.0], &[2]);
    let stats = ImportanceStats::from_tensor(&tensor);
    let debug = format!("{:?}", stats);
    assert!(debug.contains("ImportanceStats"));
}

#[test]
fn test_importance_scores_clone() {
    let values = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let scores = ImportanceScores::new(values, "test".to_string());
    let cloned = scores.clone();

    assert_eq!(scores.method, cloned.method);
    assert_eq!(scores.len(), cloned.len());
}

#[test]
fn test_importance_scores_debug() {
    let values = Tensor::new(&[1.0], &[1]);
    let scores = ImportanceScores::new(values, "test".to_string());
    let debug = format!("{:?}", scores);
    assert!(debug.contains("ImportanceScores"));
}

// ==========================================================================
// FALSIFICATION: Additional sparsity tests
// ==========================================================================
#[test]
fn test_sparsity_at_exact_value() {
    let values = Tensor::new(&[0.1, 0.2, 0.2, 0.3], &[4]);
    let stats = ImportanceStats::from_tensor(&values);

    // Threshold exactly at 0.2 - values strictly less than 0.2 = 1/4 = 0.25
    let sparsity = stats.sparsity_at(&values, 0.2);
    assert!(
        (sparsity - 0.25).abs() < 1e-6,
        "IMP-13 FALSIFIED: sparsity at 0.2 should be 0.25"
    );
}

#[test]
fn test_importance_scores_with_2d_tensor() {
    let values = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let scores = ImportanceScores::new(values, "test".to_string());

    assert_eq!(scores.shape(), &[2, 3]);
    assert_eq!(scores.len(), 6);
    assert!((scores.stats.min - 1.0).abs() < 1e-6);
    assert!((scores.stats.max - 6.0).abs() < 1e-6);
}
