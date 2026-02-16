pub(crate) use super::*;
#[test]
fn test_eviction_policy_default() {
    assert_eq!(EvictionPolicy::default(), EvictionPolicy::LRU);
}

#[test]
fn test_eviction_policy_supports_eviction() {
    assert!(EvictionPolicy::LRU.supports_eviction());
    assert!(EvictionPolicy::LFU.supports_eviction());
    assert!(EvictionPolicy::ARC.supports_eviction());
    assert!(EvictionPolicy::Clock.supports_eviction());
    assert!(!EvictionPolicy::Fixed.supports_eviction());
}

#[test]
fn test_memory_budget() {
    let budget = MemoryBudget::new(100);
    assert_eq!(budget.max_pages, 100);
    assert_eq!(budget.high_watermark, 90);
    assert_eq!(budget.low_watermark, 70);
}

#[test]
fn test_memory_budget_eviction_needed() {
    let budget = MemoryBudget::new(100);
    assert!(!budget.needs_eviction(50));
    assert!(!budget.needs_eviction(89));
    assert!(budget.needs_eviction(90));
    assert!(budget.needs_eviction(100));
}

#[test]
fn test_memory_budget_reserved_pages() {
    let mut budget = MemoryBudget::new(100);
    budget.reserve_page(1);
    budget.reserve_page(2);

    assert!(!budget.can_evict(1));
    assert!(!budget.can_evict(2));
    assert!(budget.can_evict(3));

    budget.release_page(1);
    assert!(budget.can_evict(1));
}

#[test]
fn test_access_stats() {
    let mut stats = AccessStats::new();
    assert_eq!(stats.hit_rate(), 0.0);

    stats.record_hit(100, 1);
    stats.record_hit(200, 2);
    stats.record_miss(3);

    assert!((stats.hit_rate() - 0.666).abs() < 0.01);
    assert!((stats.avg_access_time_ns() - 150.0).abs() < 0.01);
}

#[test]
#[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
fn test_cache_metadata_expiration() {
    let meta = CacheMetadata::new(1024).with_ttl(Duration::from_millis(1));
    std::thread::sleep(Duration::from_millis(5));
    assert!(meta.is_expired());
}

#[test]
fn test_cache_metadata_no_expiration() {
    let meta = CacheMetadata::new(1024);
    assert!(!meta.is_expired());
}

#[test]
fn test_cache_data_size() {
    let compressed = CacheData::Compressed(vec![0u8; 100]);
    assert_eq!(compressed.size(), 100);
    assert!(compressed.is_compressed());

    let decompressed = CacheData::Decompressed(vec![0u8; 200]);
    assert_eq!(decompressed.size(), 200);
    assert!(!decompressed.is_compressed());
}

#[test]
fn test_cache_tier_latency() {
    assert!(CacheTier::L1Hot.typical_latency() < CacheTier::L2Warm.typical_latency());
    assert!(CacheTier::L2Warm.typical_latency() < CacheTier::L3Cold.typical_latency());
}

#[test]
fn test_cache_config_default() {
    let config = CacheConfig::default();
    assert_eq!(config.l1_max_bytes, 64 * 1024 * 1024);
    assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
    assert!(config.prefetch_enabled);
}

#[test]
fn test_cache_config_embedded() {
    let config = CacheConfig::embedded(1024 * 1024);
    assert_eq!(config.l1_max_bytes, 1024 * 1024);
    assert_eq!(config.l2_max_bytes, 0);
    assert_eq!(config.eviction_policy, EvictionPolicy::Fixed);
    assert!(!config.prefetch_enabled);
}

#[test]
fn test_model_registry_basic() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 1024]),
    );

    registry.insert_l1("model1".to_string(), entry);
    assert!(registry.contains("model1"));
    assert_eq!(registry.get_tier("model1"), Some(CacheTier::L1Hot));
}

#[test]
fn test_model_registry_get() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 1024]),
    );

    registry.insert_l1("model1".to_string(), entry);
    assert!(registry.get("model1").is_some());
    assert!(registry.get("nonexistent").is_none());
}

#[test]
fn test_model_registry_remove() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 1024]),
    );

    registry.insert_l1("model1".to_string(), entry);
    assert!(registry.contains("model1"));

    registry.remove("model1");
    assert!(!registry.contains("model1"));
}

#[test]
fn test_model_registry_eviction() {
    let config = CacheConfig::new().with_l1_size(2048);
    let mut registry = ModelRegistry::new(config);

    // Insert entries that will require eviction
    for i in 0..5 {
        let entry = CacheEntry::new(
            [i as u8; 32],
            ModelType::new(1),
            CacheData::Decompressed(vec![0u8; 1024]),
        );
        registry.insert_l1(format!("model{}", i), entry);
    }

    // Should have evicted some entries
    assert!(registry.l1_current_bytes <= 2048);
}

#[test]
fn test_model_registry_stats() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 1024]),
    );

    registry.insert_l1("model1".to_string(), entry);
    let stats = registry.stats();

    assert_eq!(stats.l1_entries, 1);
    assert_eq!(stats.l1_bytes, 1024);
    assert_eq!(stats.total_entries(), 1);
}

#[test]
fn test_model_registry_list() {
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
        CacheData::Decompressed(vec![0u8; 2048]),
    );

    registry.insert_l1("model1".to_string(), entry1);
    registry.insert_l1("model2".to_string(), entry2);

    let list = registry.list();
    assert_eq!(list.len(), 2);
}

#[test]
fn test_model_registry_clear() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 1024]),
    );

    registry.insert_l1("model1".to_string(), entry);
    registry.clear();

    assert!(!registry.contains("model1"));
    assert_eq!(registry.l1_current_bytes, 0);
}

// Additional tests for coverage

#[test]
fn test_eviction_policy_description() {
    assert!(EvictionPolicy::LRU.description().contains("Recently"));
    assert!(EvictionPolicy::LFU.description().contains("Frequently"));
    assert!(EvictionPolicy::ARC.description().contains("Adaptive"));
    assert!(EvictionPolicy::Clock.description().contains("Clock"));
    assert!(EvictionPolicy::Fixed
        .description()
        .contains("deterministic"));
}

#[test]
fn test_eviction_policy_recommended_use_case() {
    assert!(EvictionPolicy::LRU
        .recommended_use_case()
        .contains("Sequential"));
    assert!(EvictionPolicy::LFU
        .recommended_use_case()
        .contains("Random"));
    assert!(EvictionPolicy::ARC.recommended_use_case().contains("Mixed"));
    assert!(EvictionPolicy::Clock
        .recommended_use_case()
        .contains("Embedded"));
    assert!(EvictionPolicy::Fixed
        .recommended_use_case()
        .contains("NASA"));
}

#[test]
fn test_memory_budget_with_watermarks() {
    let budget = MemoryBudget::with_watermarks(100, 0.8, 0.6);
    assert_eq!(budget.max_pages, 100);
    assert_eq!(budget.high_watermark, 80);
    assert_eq!(budget.low_watermark, 60);
}

#[test]
fn test_memory_budget_can_stop_eviction() {
    let budget = MemoryBudget::new(100);
    assert!(budget.can_stop_eviction(70));
    assert!(budget.can_stop_eviction(50));
    assert!(!budget.can_stop_eviction(71));
    assert!(!budget.can_stop_eviction(100));
}

#[test]
fn test_access_stats_prefetch() {
    let mut stats = AccessStats::new();
    stats.record_hit(100, 1);
    stats.record_prefetch_hit();
    stats.record_hit(100, 2);
    stats.record_prefetch_hit();

    assert_eq!(stats.prefetch_hits, 2);
    assert!((stats.prefetch_effectiveness() - 1.0).abs() < 0.01);
}

#[test]
fn test_access_stats_zero_hits() {
    let stats = AccessStats::new();
    assert_eq!(stats.avg_access_time_ns(), 0.0);
    assert_eq!(stats.prefetch_effectiveness(), 0.0);
}

#[test]
fn test_cache_metadata_with_source() {
    let path = PathBuf::from("/tmp/model.apr");
    let mtime = SystemTime::now();
    let meta = CacheMetadata::new(1024).with_source(path.clone(), mtime);

    assert_eq!(meta.source_path, Some(path));
    assert_eq!(meta.source_mtime, Some(mtime));
}

#[test]
fn test_cache_metadata_is_stale() {
    let old_mtime = SystemTime::UNIX_EPOCH;
    let path = PathBuf::from("/tmp/model.apr");
    let meta = CacheMetadata::new(1024).with_source(path, old_mtime);

    let new_mtime = SystemTime::now();
    assert!(meta.is_stale(new_mtime));
}

#[test]
fn test_cache_metadata_not_stale() {
    let meta = CacheMetadata::new(1024);
    // No source, so not stale
    assert!(!meta.is_stale(SystemTime::now()));
}

#[test]
#[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
fn test_cache_metadata_age() {
    let meta = CacheMetadata::new(1024);
    std::thread::sleep(Duration::from_millis(5));
    assert!(meta.age().as_millis() >= 5);
}

#[test]
fn test_cache_metadata_with_compression_ratio() {
    let meta = CacheMetadata::new(1024).with_compression_ratio(0.5);
    assert!((meta.compression_ratio - 0.5).abs() < 0.01);
}

#[test]
fn test_cache_data_mapped() {
    let mapped = CacheData::Mapped {
        path: PathBuf::from("/tmp/model.apr"),
        offset: 1024,
        length: 4096,
    };
    assert_eq!(mapped.size(), 4096);
    assert!(mapped.is_mapped());
    assert!(!mapped.is_compressed());
}

#[test]
fn test_cache_entry_tier() {
    let entry_decompressed = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 100]),
    );
    assert_eq!(entry_decompressed.tier(), CacheTier::L1Hot);

    let entry_compressed = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 100]),
    );
    assert_eq!(entry_compressed.tier(), CacheTier::L2Warm);

    let entry_mapped = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Mapped {
            path: PathBuf::from("/tmp/x"),
            offset: 0,
            length: 100,
        },
    );
    assert_eq!(entry_mapped.tier(), CacheTier::L2Warm);
}

#[test]
fn test_cache_entry_is_valid() {
    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 100]),
    );
    assert!(entry.is_valid());
}

#[test]
#[ignore = "Uses thread::sleep - run with cargo test -- --ignored"]
fn test_cache_entry_is_valid_expired() {
    // Entry with expired TTL
    let mut entry_expired = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Decompressed(vec![0u8; 100]),
    );
    entry_expired.metadata = CacheMetadata::new(100).with_ttl(Duration::from_millis(1));
    std::thread::sleep(Duration::from_millis(5));
    assert!(!entry_expired.is_valid());
}

#[test]
fn test_cache_tier_name() {
    assert_eq!(CacheTier::L1Hot.name(), "L1 Hot Cache");
    assert_eq!(CacheTier::L2Warm.name(), "L2 Warm Cache");
    assert_eq!(CacheTier::L3Cold.name(), "L3 Cold Storage");
}

#[test]
fn test_cache_config_builders() {
    let config = CacheConfig::new()
        .with_l1_size(1024)
        .with_l2_size(2048)
        .with_eviction_policy(EvictionPolicy::LFU)
        .with_ttl(Duration::from_secs(60))
        .with_prefetch(false);

    assert_eq!(config.l1_max_bytes, 1024);
    assert_eq!(config.l2_max_bytes, 2048);
    assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
    assert_eq!(config.default_ttl, Some(Duration::from_secs(60)));
    assert!(!config.prefetch_enabled);
}

#[test]
fn test_model_registry_l2_operations() {
    let config = CacheConfig::default();
    let mut registry = ModelRegistry::new(config);

    let entry = CacheEntry::new(
        [0u8; 32],
        ModelType::new(1),
        CacheData::Compressed(vec![0u8; 1024]),
    );

    registry.insert_l2("model1".to_string(), entry);
    assert!(registry.contains("model1"));
    assert_eq!(registry.get_tier("model1"), Some(CacheTier::L2Warm));

    assert!(registry.get("model1").is_some());
}

#[path = "tests_part_02.rs"]

mod tests_part_02;
#[path = "tests_part_03.rs"]
mod tests_part_03;
