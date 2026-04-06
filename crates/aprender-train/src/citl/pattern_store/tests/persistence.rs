//! APR persistence tests.

use super::*;

#[test]
fn test_pattern_store_save_load_apr() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    // Add patterns with various states
    let mut pattern1 = FixPattern::new("E0308", "- i32\n+ &str")
        .with_decision("type_mismatch")
        .with_decision("infer_string");
    pattern1.record_success();
    pattern1.record_success();
    store.index_fix(pattern1).expect("operation should succeed");

    let pattern2 = FixPattern::new("E0382", "- x\n+ x.clone()").with_decision("borrow_after_move");
    store.index_fix(pattern2).expect("operation should succeed");

    // Save to temp file
    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("patterns.apr");

    store.save_apr(&path).expect("save should succeed");
    assert!(path.exists());

    // Load and verify
    let loaded = DecisionPatternStore::load_apr(&path).expect("load should succeed");
    assert_eq!(loaded.len(), 2);

    // Verify patterns for each error code
    let e0308_patterns = loaded.patterns_for_error("E0308");
    assert_eq!(e0308_patterns.len(), 1);
    assert_eq!(e0308_patterns[0].success_count, 2);
    assert_eq!(e0308_patterns[0].decision_sequence.len(), 2);

    let e0382_patterns = loaded.patterns_for_error("E0382");
    assert_eq!(e0382_patterns.len(), 1);
}

#[test]
fn test_pattern_store_apr_config_preserved() {
    let config = PatternStoreConfig { chunk_size: 512, embedding_dim: 768, rrf_k: 30.0 };
    let mut store = DecisionPatternStore::with_config(config).expect("config should be valid");
    store.index_fix(FixPattern::new("E0308", "fix")).expect("operation should succeed");

    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("patterns.apr");

    store.save_apr(&path).expect("save should succeed");
    let loaded = DecisionPatternStore::load_apr(&path).expect("load should succeed");

    assert_eq!(loaded.config().chunk_size, 512);
    assert_eq!(loaded.config().embedding_dim, 768);
    assert!((loaded.config().rrf_k - 30.0).abs() < 0.01);
}

#[test]
fn test_pattern_store_apr_empty() {
    let store = DecisionPatternStore::new().expect("operation should succeed");

    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("empty.apr");

    store.save_apr(&path).expect("save should succeed");
    let loaded = DecisionPatternStore::load_apr(&path).expect("load should succeed");

    assert!(loaded.is_empty());
}

#[test]
fn test_pattern_store_apr_suggest_after_load() {
    let mut store = DecisionPatternStore::new().expect("operation should succeed");

    let pattern = FixPattern::new("E0308", "type fix").with_decision("detect_mismatch");
    store.index_fix(pattern).expect("operation should succeed");

    let temp_dir = tempfile::tempdir().expect("temp file creation should succeed");
    let path = temp_dir.path().join("patterns.apr");

    store.save_apr(&path).expect("save should succeed");
    let loaded = DecisionPatternStore::load_apr(&path).expect("load should succeed");

    // RAG index should be rebuilt and queryable
    let suggestions =
        loaded.suggest_fix("E0308", &["detect_mismatch".into()], 5).expect("load should succeed");
    assert!(!suggestions.is_empty());
    assert_eq!(suggestions[0].pattern.error_code, "E0308");
}
