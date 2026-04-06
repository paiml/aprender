//! Integration tests for the decision module.

use super::citl::{CitlTrainer, ErrorFixPair};
use super::pattern_store::{DecisionPattern, PatternStore};

// ─── PatternStore + CitlTrainer integration ────────────────────────────

#[test]
fn test_store_and_search_round_trip() {
    let mut store = PatternStore::new();

    store.add_pattern(DecisionPattern::new(
        "type_fix",
        "Fix type mismatch by changing variable type",
        vec![1.0, 0.0, 0.0, 0.5],
        0.95,
        "type_error",
    ));
    store.add_pattern(DecisionPattern::new(
        "borrow_fix",
        "Fix borrow checker by adding lifetime",
        vec![0.0, 1.0, 0.0, 0.3],
        0.88,
        "borrow_error",
    ));
    store.add_pattern(DecisionPattern::new(
        "move_fix",
        "Fix use-after-move by cloning",
        vec![0.0, 0.0, 1.0, 0.7],
        0.72,
        "move_error",
    ));

    // Search for something similar to type_fix
    let results = store.search(&[0.9, 0.1, 0.0, 0.5], 2);
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].pattern_id, "type_fix");
}

#[test]
fn test_citl_trainer_learns_permutation() {
    // Train CITL to learn a swap mapping: [a, b] -> [b, a]
    let pairs: Vec<ErrorFixPair> = (0..20)
        .map(|i| {
            let a = (i as f32) * 0.1;
            let b = 1.0 - a;
            ErrorFixPair::new(vec![a, b], vec![b, a], 0.9)
        })
        .collect();

    let trainer = CitlTrainer::train(&pairs).expect("operation should succeed");

    // Test prediction
    let pred = trainer.predict_fix(&[0.3, 0.7]);
    // Should predict approximately [0.7, 0.3]
    assert!((pred[0] - 0.7).abs() < 0.15, "Expected ~0.7, got {}", pred[0]);
    assert!((pred[1] - 0.3).abs() < 0.15, "Expected ~0.3, got {}", pred[1]);
}

#[test]
fn test_end_to_end_store_train_predict() {
    // 1. Populate pattern store with known patterns
    let mut store = PatternStore::new();

    let patterns = vec![
        DecisionPattern::new("p1", "null pointer fix", vec![1.0, 0.0, 0.0], 0.9, "null"),
        DecisionPattern::new("p2", "overflow fix", vec![0.0, 1.0, 0.0], 0.85, "overflow"),
        DecisionPattern::new("p3", "bounds fix", vec![0.0, 0.0, 1.0], 0.8, "bounds"),
    ];

    for p in &patterns {
        store.add_pattern(p.clone());
    }

    // 2. Create training pairs from patterns
    let pairs: Vec<ErrorFixPair> = patterns
        .iter()
        .map(|p| {
            // Error is the pattern weights, fix is a shifted version
            let fix: Vec<f32> = p.feature_weights.iter().map(|w| 1.0 - w).collect();
            ErrorFixPair::new(p.feature_weights.clone(), fix, p.confidence)
        })
        .collect();

    // 3. Train CITL model
    let trainer = CitlTrainer::train(&pairs).expect("operation should succeed");

    // 4. For a new error, search for similar pattern and predict fix
    let error_features = vec![0.9, 0.1, 0.0];
    let similar = store.search(&error_features, 1);
    assert_eq!(similar[0].pattern_id, "p1");

    let predicted_fix = trainer.predict_fix(&error_features);
    assert_eq!(predicted_fix.len(), 3);
    // For error close to [1,0,0], fix should be close to [0,1,1]
    assert!(predicted_fix[0] < 0.5, "fix[0] should be low: {}", predicted_fix[0]);
}

#[test]
fn test_pattern_store_crud_operations() {
    let mut store = PatternStore::new();

    // Create
    store.add_pattern(DecisionPattern::new("a", "first", vec![1.0], 0.5, "cat"));
    store.add_pattern(DecisionPattern::new("b", "second", vec![2.0], 0.6, "cat"));
    store.add_pattern(DecisionPattern::new("c", "third", vec![3.0], 0.7, "cat"));
    assert_eq!(store.len(), 3);

    // Read
    let a = store.get_pattern("a").expect("operation should succeed");
    assert_eq!(a.description, "first");

    // Update (replace)
    store.add_pattern(DecisionPattern::new("a", "updated", vec![1.5], 0.9, "cat"));
    assert_eq!(store.get_pattern("a").expect("operation should succeed").description, "updated");
    assert_eq!(store.len(), 3); // Still 3, not 4

    // Delete
    let removed = store.remove_pattern("b").expect("operation should succeed");
    assert_eq!(removed.description, "second");
    assert_eq!(store.len(), 2);
    assert!(store.get_pattern("b").is_none());

    // List
    let all = store.list_patterns();
    assert_eq!(all.len(), 2);
}
