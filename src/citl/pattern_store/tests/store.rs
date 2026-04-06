//! DecisionPatternStore tests.

use super::*;

#[test]
fn test_pattern_store_new() {
    let store = DecisionPatternStore::new().expect("operation should succeed");
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
}

#[test]
fn test_pattern_store_with_config() {
    let config = PatternStoreConfig { chunk_size: 512, embedding_dim: 768, rrf_k: 30.0 };
    let store = DecisionPatternStore::with_config(config.clone()).expect("config should be valid");
    assert_eq!(store.config().chunk_size, 512);
    assert_eq!(store.config().embedding_dim, 768);
}

#[test]
fn test_pattern_store_index_fix() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let pattern = FixPattern::new("E0308", "- i32\n+ &str").with_decision("type_mismatch");

    store.index_fix(pattern).expect("operation should succeed");

    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());
}

#[test]
fn test_pattern_store_index_multiple() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    store.index_fix(FixPattern::new("E0308", "fix1")).expect("operation should succeed");
    store.index_fix(FixPattern::new("E0308", "fix2")).expect("operation should succeed");
    store.index_fix(FixPattern::new("E0382", "fix3")).expect("operation should succeed");

    assert_eq!(store.len(), 3);
}

#[test]
fn test_pattern_store_get() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let pattern = FixPattern::new("E0308", "diff");
    let id = pattern.id;

    store.index_fix(pattern).expect("operation should succeed");

    let retrieved = store.get(&id);
    assert!(retrieved.is_some());
    assert_eq!(retrieved.expect("operation should succeed").error_code, "E0308");
}

#[test]
fn test_pattern_store_get_mut() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let pattern = FixPattern::new("E0308", "diff");
    let id = pattern.id;

    store.index_fix(pattern).expect("operation should succeed");

    if let Some(p) = store.get_mut(&id) {
        p.record_success();
    }

    assert_eq!(store.get(&id).expect("key should exist").success_count, 1);
}

#[test]
fn test_pattern_store_record_outcome() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let pattern = FixPattern::new("E0308", "diff");
    let id = pattern.id;

    store.index_fix(pattern).expect("operation should succeed");
    store.record_outcome(&id, true);
    store.record_outcome(&id, false);

    let p = store.get(&id).expect("key should exist");
    assert_eq!(p.success_count, 1);
    assert_eq!(p.attempt_count, 2);
}

#[test]
fn test_pattern_store_patterns_for_error() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    store.index_fix(FixPattern::new("E0308", "fix1")).expect("operation should succeed");
    store.index_fix(FixPattern::new("E0308", "fix2")).expect("operation should succeed");
    store.index_fix(FixPattern::new("E0382", "fix3")).expect("operation should succeed");

    let e0308_patterns = store.patterns_for_error("E0308");
    assert_eq!(e0308_patterns.len(), 2);

    let e0382_patterns = store.patterns_for_error("E0382");
    assert_eq!(e0382_patterns.len(), 1);

    let e0000_patterns = store.patterns_for_error("E0000");
    assert!(e0000_patterns.is_empty());
}

#[test]
fn test_pattern_store_suggest_fix() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let pattern = FixPattern::new("E0308", "- let x: i32 = \"hello\";\n+ let x: &str = \"hello\";")
        .with_decision("type_mismatch_detected")
        .with_decision("infer_correct_type");

    store.index_fix(pattern).expect("operation should succeed");

    let context = vec!["type_mismatch".to_string()];
    let suggestions = store.suggest_fix("E0308", &context, 5).expect("operation should succeed");

    assert!(!suggestions.is_empty());
    assert_eq!(suggestions[0].pattern.error_code, "E0308");
}

#[test]
fn test_pattern_store_suggest_fix_empty() {
    let store = DecisionPatternStore::new().expect("operation should succeed");

    let suggestions = store.suggest_fix("E0308", &[], 5).expect("operation should succeed");
    assert!(suggestions.is_empty());
}

#[test]
fn test_pattern_store_suggest_fix_ranking() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    // Pattern with high success rate
    let mut pattern1 = FixPattern::new("E0308", "fix1 high success");
    pattern1.record_success();
    pattern1.record_success();
    store.index_fix(pattern1).expect("operation should succeed");

    // Pattern with low success rate
    let mut pattern2 = FixPattern::new("E0308", "fix2 low success");
    pattern2.record_failure();
    pattern2.record_failure();
    store.index_fix(pattern2).expect("operation should succeed");

    let suggestions = store.suggest_fix("E0308", &[], 5).expect("operation should succeed");

    // Higher success rate should be ranked first
    assert!(!suggestions.is_empty());
    // The first suggestion should have higher weighted score
    if suggestions.len() >= 2 {
        assert!(suggestions[0].weighted_score() >= suggestions[1].weighted_score());
    }
}

#[test]
fn test_pattern_store_export_import_json() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    store.index_fix(FixPattern::new("E0308", "fix1")).expect("operation should succeed");
    store.index_fix(FixPattern::new("E0382", "fix2")).expect("operation should succeed");

    let json = store.export_json().expect("operation should succeed");

    let mut new_store = DecisionPatternStore::new().expect("operation should succeed");
    let count = new_store.import_json(&json).expect("operation should succeed");

    assert_eq!(count, 2);
    assert_eq!(new_store.len(), 2);
}

#[test]
fn test_pattern_store_debug() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");
    store.index_fix(FixPattern::new("E0308", "fix")).expect("operation should succeed");

    let debug = format!("{store:?}");
    assert!(debug.contains("DecisionPatternStore"));
    assert!(debug.contains("pattern_count"));
}
