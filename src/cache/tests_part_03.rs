
#[test]
fn test_model_registry_l1_fixed_policy_overflow() {
    // Test the eviction break path when Fixed policy + cache full
    let config = CacheConfig::new()
        .with_l1_size(1024)
        .with_eviction_policy(EvictionPolicy::Fixed);
    let mut registry = ModelRegistry::new(config);

    // Fill the cache
    let entry1 = CacheEntry::new(
        [1u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 800]),
    );
    registry.insert_l1("model1".to_string(), entry1);

    // Try to insert another entry that would overflow
    let entry2 = CacheEntry::new(
        [2u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 500]),
    );
    registry.insert_l1("model2".to_string(), entry2);

    // With Fixed policy, should NOT evict model1, but model2 is still inserted
    // (cache will be over capacity, but Fixed policy doesn't evict)
    assert!(registry.contains("model1"));
    assert!(registry.contains("model2"));
}

#[test]
fn test_model_registry_l2_fixed_policy_overflow() {
    // Test the L2 eviction break path when Fixed policy + cache full
    let config = CacheConfig::new()
        .with_l2_size(1024)
        .with_eviction_policy(EvictionPolicy::Fixed);
    let mut registry = ModelRegistry::new(config);

    // Fill the cache
    let entry1 = CacheEntry::new(
        [1u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 800]),
    );
    registry.insert_l2("model1".to_string(), entry1);

    // Try to insert another entry that would overflow
    let entry2 = CacheEntry::new(
        [2u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 500]),
    );
    registry.insert_l2("model2".to_string(), entry2);

    // With Fixed policy, should NOT evict model1, but model2 is still inserted
    assert!(registry.contains("model1"));
    assert!(registry.contains("model2"));
}

#[test]
fn test_cache_metadata_is_expired_success_path() {
    // Test is_expired when TTL exists and elapsed succeeds
    // (This path is hard to test without sleep, but we can verify the logic)
    let meta = CacheMetadata::new(1024).with_ttl(Duration::from_secs(3600));

    // Entry with TTL of 1 hour should not be expired immediately
    assert!(!meta.is_expired());
}

#[test]
fn test_model_registry_clear_both_caches() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry1 = CacheEntry::new(
        [1u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 1024]),
    );
    let entry2 = CacheEntry::new(
        [2u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 512]),
    );

    registry.insert_l1("model1".to_string(), entry1);
    registry.insert_l2("model2".to_string(), entry2);

    registry.clear();

    assert!(!registry.contains("model1"));
    assert!(!registry.contains("model2"));
    assert_eq!(registry.l1_current_bytes, 0);
    assert_eq!(registry.l2_current_bytes, 0);
}

#[test]
fn test_model_info_fields() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [1u8; 32],
        ModelType::new(42),
        CacheData::Decompressed(vec![0u8; 1024]),
    );

    registry.insert_l1("test_model".to_string(), entry);

    let list = registry.list();
    assert_eq!(list.len(), 1);

    let info = &list[0];
    assert_eq!(info.name, "test_model");
    assert_eq!(info.model_type.0, 42);
    assert_eq!(info.size_bytes, 1024);
    assert!(!info.is_bundled);
    assert!(info.is_cached);
    assert_eq!(info.cache_tier, Some(CacheTier::L1Hot));
}
