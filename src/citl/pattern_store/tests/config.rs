//! PatternStoreConfig tests.

use super::*;

#[test]
fn test_pattern_store_config_default() {
    let config = PatternStoreConfig::default();
    assert_eq!(config.chunk_size, 256);
    assert_eq!(config.embedding_dim, 384);
    assert!((config.rrf_k - 60.0).abs() < 0.01);
}
