use super::*;

#[test]
fn test_model_registry_l2_eviction() {
    let config = CacheConfig::new().with_l2_size(2048);
    let mut registry = ModelRegistry::new(config);

    for i in 0..5 {
        let entry = CacheEntry::new(
            [i as u8; 32],
            ModelType::new(1),
            CacheData::Compressed(vec![0u8; 1024]),
        );
        registry.insert_l2(format!("model{}", i), entry);
    }

    assert!(registry.l2_current_bytes <= 2048);
}

#[test]
fn test_model_registry_remove_l2() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 1024]),
    );

    registry.insert_l2("model1".to_string(), entry);
    registry.remove("model1");
    assert!(!registry.contains("model1"));
}

#[test]
fn test_model_registry_lfu_eviction() {
    let config = CacheConfig::new()
        .with_l1_size(2048)
        .with_eviction_policy(EvictionPolicy::LFU);
    let mut registry = ModelRegistry::new(config);

    for i in 0..5 {
        let entry = CacheEntry::new(
            [i as u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );
        registry.insert_l1(format!("model{}", i), entry);
    }

    assert!(registry.l1_current_bytes <= 2048);
}

#[test]
fn test_model_registry_arc_eviction() {
    let config = CacheConfig::new()
        .with_l1_size(2048)
        .with_eviction_policy(EvictionPolicy::ARC);
    let mut registry = ModelRegistry::new(config);

    for i in 0..5 {
        let entry = CacheEntry::new(
            [i as u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );
        registry.insert_l1(format!("model{}", i), entry);
    }

    assert!(registry.l1_current_bytes <= 2048);
}

#[test]
fn test_model_registry_fixed_no_eviction() {
    let config = CacheConfig::new()
        .with_l1_size(1024)
        .with_eviction_policy(EvictionPolicy::Fixed);
    let mut registry = ModelRegistry::new(config);

    // First entry fits
    let entry1 = CacheEntry::new(
        [1u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 512]),
    );
    registry.insert_l1("model1".to_string(), entry1);

    // Second entry also fits
    let entry2 = CacheEntry::new(
        [2u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 256]),
    );
    registry.insert_l1("model2".to_string(), entry2);

    assert!(registry.contains("model1"));
    assert!(registry.contains("model2"));
}

#[test]
fn test_model_registry_get_tier_none() {
    let config = CacheConfig::default();
    let registry = ModelRegistry::new(config);
    assert_eq!(registry.get_tier("nonexistent"), None);
}

#[test]
fn test_model_registry_list_both_caches() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry1 = CacheEntry::new(
        [1u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 1024]),
    );
    let entry2 = CacheEntry::new(
        [2u8; 32],
        ModelType::new(2),
        CacheData::Compressed(vec![0u8; 512]),
    );

    registry.insert_l1("model1".to_string(), entry1);
    registry.insert_l2("model2".to_string(), entry2);

    let list = registry.list();
    assert_eq!(list.len(), 2);
}

#[test]
fn test_cache_stats_hit_rate_zero() {
    let stats = CacheStats {
        l1_entries: 0,
        l1_bytes: 0,
        l1_hits: 0,
        l1_misses: 0,
        l2_entries: 0,
        l2_bytes: 0,
        l2_hits: 0,
        l2_misses: 0,
        uptime: Duration::from_secs(1),
    };
    assert_eq!(stats.hit_rate(), 0.0);
}

#[test]
fn test_cache_stats_total_bytes() {
    let stats = CacheStats {
        l1_entries: 1,
        l1_bytes: 1024,
        l1_hits: 10,
        l1_misses: 5,
        l2_entries: 2,
        l2_bytes: 2048,
        l2_hits: 5,
        l2_misses: 3,
        uptime: Duration::from_secs(1),
    };
    assert_eq!(stats.total_bytes(), 3072);
    assert_eq!(stats.total_entries(), 3);
}

#[test]
fn test_model_type_new() {
    let mt = ModelType::new(42);
    assert_eq!(mt.0, 42);
}

#[test]
fn test_model_registry_insert_replaces() {
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
        CacheData::Decompressed(vec![0u8; 2048]),
    );

    registry.insert_l1("model".to_string(), entry1);
    assert_eq!(registry.l1_current_bytes, 1024);

    registry.insert_l1("model".to_string(), entry2);
    assert_eq!(registry.l1_current_bytes, 2048);
}

#[test]
fn test_clone_debug_traits() {
    let policy = EvictionPolicy::LRU;
    let _ = policy.clone();
    let _ = format!("{policy:?}");

    let budget = MemoryBudget::new(100);
    let _ = budget.clone();
    let _ = format!("{budget:?}");

    let stats = AccessStats::new();
    let _ = stats.clone();
    let _ = format!("{stats:?}");

    let meta = CacheMetadata::new(1024);
    let _ = meta.clone();
    let _ = format!("{meta:?}");

    let config = CacheConfig::default();
    let _ = config.clone();
    let _ = format!("{config:?}");
}

// ==================== Eviction Policy Tests ====================

#[test]
fn test_eviction_policy_all_descriptions() {
    // Test that all policies have descriptions
    assert!(!EvictionPolicy::LRU.description().is_empty());
    assert!(!EvictionPolicy::LFU.description().is_empty());
    assert!(!EvictionPolicy::ARC.description().is_empty());
    assert!(!EvictionPolicy::Clock.description().is_empty());
    assert!(!EvictionPolicy::Fixed.description().is_empty());
}

#[test]
fn test_eviction_policy_all_use_cases() {
    // Test that all policies have recommended use cases
    assert!(!EvictionPolicy::LRU.recommended_use_case().is_empty());
    assert!(!EvictionPolicy::LFU.recommended_use_case().is_empty());
    assert!(!EvictionPolicy::ARC.recommended_use_case().is_empty());
    assert!(!EvictionPolicy::Clock.recommended_use_case().is_empty());
    assert!(!EvictionPolicy::Fixed.recommended_use_case().is_empty());
}

#[test]
fn test_cache_tier_latencies() {
    // Test that all tiers have defined latencies
    let l1_latency = CacheTier::L1Hot.typical_latency();
    let l2_latency = CacheTier::L2Warm.typical_latency();
    let l3_latency = CacheTier::L3Cold.typical_latency();

    // L1 should be fastest, L3 slowest
    assert!(l1_latency < l2_latency);
    assert!(l2_latency < l3_latency);
}

#[test]
fn test_cache_tier_all_names() {
    assert!(!CacheTier::L1Hot.name().is_empty());
    assert!(!CacheTier::L2Warm.name().is_empty());
    assert!(!CacheTier::L3Cold.name().is_empty());
}

#[test]
fn test_memory_budget_can_evict_reserved() {
    let mut budget = MemoryBudget::new(100);
    budget.reserve_page(1);
    budget.reserve_page(2);

    // Reserved pages cannot be evicted
    assert!(!budget.can_evict(1));
    assert!(!budget.can_evict(2));

    // Unreserved pages can be evicted
    assert!(budget.can_evict(3));

    // After release, can evict
    budget.release_page(1);
    assert!(budget.can_evict(1));
}

#[test]
fn test_cache_metadata_all_builders() {
    let path = PathBuf::from("/test/model.gguf");
    let mtime = SystemTime::now();

    let meta = CacheMetadata::new(1024)
        .with_source(path.clone(), mtime)
        .with_ttl(Duration::from_secs(300))
        .with_compression_ratio(0.75);

    assert_eq!(meta.size_bytes, 1024);
    assert!((meta.compression_ratio - 0.75).abs() < f32::EPSILON);
    assert_eq!(meta.source_path, Some(path));
}

#[test]
fn test_cache_data_size_and_compression() {
    let compressed = CacheData::Compressed(vec![1u8; 100]);
    let decompressed = CacheData::Decompressed(vec![1u8; 200]);

    assert_eq!(compressed.size(), 100);
    assert_eq!(decompressed.size(), 200);

    assert!(compressed.is_compressed());
    assert!(!decompressed.is_compressed());
}

#[test]
fn test_access_stats_prefetch_effectiveness() {
    let mut stats = AccessStats::new();

    // No prefetches should give 0 effectiveness
    stats.record_hit(100, 1);
    stats.record_hit(100, 2);
    assert!((stats.prefetch_effectiveness() - 0.0).abs() < f64::EPSILON);

    // Prefetch hits should increase effectiveness
    stats.record_prefetch_hit();
    stats.record_prefetch_hit();
    stats.record_hit(100, 3);
    assert!(stats.prefetch_effectiveness() > 0.0);
}

// ==================== Additional Coverage Tests ====================

#[test]
fn test_cache_metadata_age_instant() {
    // Test age() without sleep (non-ignored version)
    let meta = CacheMetadata::new(1024);
    // Age should be very small (close to 0), but not panic
    let age = meta.age();
    assert!(age.as_secs() < 1);
}

#[test]
fn test_cache_stats_hit_rate_with_hits() {
    let stats = CacheStats {
        l1_entries: 2,
        l1_bytes: 2048,
        l1_hits: 8,
        l1_misses: 2,
        l2_entries: 1,
        l2_bytes: 1024,
        l2_hits: 4,
        l2_misses: 1,
        uptime: Duration::from_secs(10),
    };

    // Total: 12 hits, 3 misses = 12/15 = 0.8 hit rate
    let hit_rate = stats.hit_rate();
    assert!((hit_rate - 0.8).abs() < 0.01);
}

#[test]
fn test_model_registry_l2_insert_replaces() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry1 = CacheEntry::new(
        [1u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 1024]),
    );
    let entry2 = CacheEntry::new(
        [2u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 2048]),
    );

    registry.insert_l2("model".to_string(), entry1);
    assert_eq!(registry.l2_current_bytes, 1024);

    // Insert with same name should replace
    registry.insert_l2("model".to_string(), entry2);
    assert_eq!(registry.l2_current_bytes, 2048);
}

#[test]
fn test_model_registry_stats_with_l2() {
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

    // Access L2 to record hits
    registry.get("model2");

    let stats = registry.stats();
    assert_eq!(stats.l1_entries, 1);
    assert_eq!(stats.l2_entries, 1);
    assert_eq!(stats.l2_bytes, 512);
    // L2 should have recorded a hit
    assert!(stats.l2_hits >= 1);
}

#[test]
fn test_model_registry_fixed_policy_l2_no_eviction() {
    let config = CacheConfig::new()
        .with_l2_size(1024)
        .with_eviction_policy(EvictionPolicy::Fixed);
    let mut registry = ModelRegistry::new(config);

    // Insert entries
    let entry1 = CacheEntry::new(
        [1u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 512]),
    );
    registry.insert_l2("model1".to_string(), entry1);

    let entry2 = CacheEntry::new(
        [2u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 256]),
    );
    registry.insert_l2("model2".to_string(), entry2);

    // With Fixed policy, entries should stay (no eviction)
    assert!(registry.contains("model1"));
    assert!(registry.contains("model2"));
}
