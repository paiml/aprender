use super::*;
#[test]
fn test_shard_index_empty() {
    let index = ShardIndex::new();
    assert_eq!(index.shard_count(), 0);
    assert_eq!(index.tensor_count(), 0);
    assert!(!index.is_valid());
}

#[test]
fn test_shard_index_add_mapping() {
    let mut index = ShardIndex::new();
    index.add_mapping("layer1.weight", "model-00001.safetensors");
    index.add_mapping("layer2.weight", "model-00002.safetensors");

    assert_eq!(index.shard_count(), 2);
    assert_eq!(index.tensor_count(), 2);
    assert!(index.is_valid());

    assert_eq!(
        index.shard_for_tensor("layer1.weight"),
        Some("model-00001.safetensors")
    );
    assert_eq!(index.shard_for_tensor("nonexistent"), None);
}

#[test]
fn test_shard_index_from_json_empty() {
    let index = ShardIndex::from_json("");
    assert!(index.is_ok());
    let idx = index.expect("parse failed");
    assert!(!idx.is_valid());
}

#[test]
fn test_shard_index_from_json_basic() {
    let json = r#"{"weight_map": {"layer1.weight": "shard1.safetensors"}}"#;
    let index = ShardIndex::from_json(json);
    assert!(index.is_ok());
    let idx = index.expect("parse failed");
    assert_eq!(idx.tensor_count(), 1);
}

#[test]
fn test_shard_index_tensors_by_shard() {
    let mut index = ShardIndex::new();
    index.add_mapping("a.weight", "shard1.safetensors");
    index.add_mapping("b.weight", "shard1.safetensors");
    index.add_mapping("c.weight", "shard2.safetensors");

    let by_shard = index.tensors_by_shard();
    assert_eq!(by_shard.len(), 2);
    assert_eq!(by_shard.get("shard1.safetensors").map(|v| v.len()), Some(2));
}

#[test]
fn test_shard_index_tensor_names_sorted() {
    let mut index = ShardIndex::new();
    index.add_mapping("z.weight", "shard1.safetensors");
    index.add_mapping("a.weight", "shard1.safetensors");
    index.add_mapping("m.weight", "shard2.safetensors");

    let names = index.tensor_names();
    assert_eq!(names, vec!["a.weight", "m.weight", "z.weight"]);
}

#[test]
fn test_cached_shard() {
    let mut shard = CachedShard::new("test.safetensors".to_string());
    shard.add_tensor("tensor1".to_string(), vec![1, 2, 3, 4]);

    assert!(shard.has_tensor("tensor1"));
    assert!(!shard.has_tensor("tensor2"));
    assert_eq!(shard.get_tensor("tensor1"), Some(&[1u8, 2, 3, 4][..]));
    assert_eq!(shard.size, 4);
}

#[test]
fn test_shard_cache_basic() {
    let mut cache = ShardCache::new(2, 1024);

    let shard1 = CachedShard::new("shard1.safetensors".to_string());
    cache.insert(shard1);

    assert_eq!(cache.stats().cached_shards, 1);

    // Access should hit
    assert!(cache.get("shard1.safetensors").is_some());
    assert_eq!(cache.stats().hits, 1);

    // Miss
    assert!(cache.get("nonexistent").is_none());
    assert_eq!(cache.stats().misses, 1);
}

#[test]
fn test_shard_cache_eviction() {
    let mut cache = ShardCache::new(2, 1024 * 1024);

    let shard1 = CachedShard::new("shard1.safetensors".to_string());
    let shard2 = CachedShard::new("shard2.safetensors".to_string());
    let shard3 = CachedShard::new("shard3.safetensors".to_string());

    cache.insert(shard1);
    cache.insert(shard2);
    assert_eq!(cache.stats().cached_shards, 2);

    // Insert third should evict first (LRU)
    cache.insert(shard3);
    assert_eq!(cache.stats().cached_shards, 2);
    assert!(cache.get("shard1.safetensors").is_none()); // Evicted
    assert!(cache.get("shard3.safetensors").is_some()); // Present
}

#[test]
fn test_shard_cache_lru_order() {
    let mut cache = ShardCache::new(2, 1024 * 1024);

    let shard1 = CachedShard::new("shard1.safetensors".to_string());
    let shard2 = CachedShard::new("shard2.safetensors".to_string());

    cache.insert(shard1);
    cache.insert(shard2);

    // Access shard1 to make it most recent
    let _ = cache.get("shard1.safetensors");

    // Insert shard3 should evict shard2 (least recently used)
    let shard3 = CachedShard::new("shard3.safetensors".to_string());
    cache.insert(shard3);

    assert!(cache.get("shard1.safetensors").is_some()); // Still present
    assert!(cache.get("shard2.safetensors").is_none()); // Evicted
}

#[test]
fn test_shard_cache_hit_rate() {
    let mut cache = ShardCache::new(2, 1024);

    let shard = CachedShard::new("test.safetensors".to_string());
    cache.insert(shard);

    // 2 hits, 1 miss
    let _ = cache.get("test.safetensors");
    let _ = cache.get("test.safetensors");
    let _ = cache.get("nonexistent");

    let rate = cache.hit_rate();
    assert!((rate - 0.666).abs() < 0.01);
}

#[test]
fn test_import_config_default() {
    let config = ShardedImportConfig::default();
    assert_eq!(config.max_cached_shards, 2);
    assert!(config.sort_tensors);
    assert!(config.validate().is_ok());
}

#[test]
fn test_import_config_validate() {
    let mut config = ShardedImportConfig::default();
    config.max_cached_shards = 0;
    assert!(config.validate().is_err());
}

#[test]
fn test_sharded_importer_new() {
    let importer = ShardedImporter::default();
    assert_eq!(importer.cache_stats().cached_shards, 0);
}

#[test]
fn test_import_phase() {
    assert_eq!(ImportPhase::Parsing, ImportPhase::Parsing);
    assert_ne!(ImportPhase::Loading, ImportPhase::Complete);
}

#[test]
fn test_estimate_shard_memory() {
    let estimate = estimate_shard_memory(1_000_000);
    assert!(estimate > 1_000_000); // Should include overhead
    assert!(estimate < 1_100_000); // But not too much
}

#[test]
fn test_is_sharded_model_nonexistent() {
    let result = is_sharded_model(Path::new("/nonexistent/path"));
    assert!(!result);
}

#[test]
fn test_get_shard_files_nonexistent() {
    let result = get_shard_files(Path::new("/nonexistent/path"));
    assert!(result.is_err());
}

#[test]
fn test_import_report() {
    let report = ImportReport {
        tensor_count: 100,
        shard_count: 4,
        bytes_written: 1024 * 1024,
        peak_memory_bytes: 2 * 1024 * 1024 * 1024,
        cache_hit_rate: 0.75,
        duration_ms: 5000,
        warnings: vec![],
    };

    assert_eq!(report.tensor_count, 100);
    assert!(report.warnings.is_empty());
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_shard_index_metadata() {
    let mut index = ShardIndex::new();
    index.set_metadata("total_size", "1000000");
    index.set_metadata("format_version", "1.0");

    assert_eq!(index.get_metadata("total_size"), Some("1000000"));
    assert_eq!(index.get_metadata("format_version"), Some("1.0"));
    assert_eq!(index.get_metadata("nonexistent"), None);
}

#[test]
fn test_shard_index_shard_files() {
    let mut index = ShardIndex::new();
    index.add_mapping("a.weight", "shard1.safetensors");
    index.add_mapping("b.weight", "shard2.safetensors");

    let files = index.shard_files();
    assert_eq!(files.len(), 2);
    assert!(files.contains(&"shard1.safetensors".to_string()));
    assert!(files.contains(&"shard2.safetensors".to_string()));
}

#[test]
fn test_shard_index_shard_index() {
    let mut index = ShardIndex::new();
    index.add_mapping("a.weight", "shard1.safetensors");
    index.add_mapping("b.weight", "shard2.safetensors");

    assert_eq!(index.shard_index("shard1.safetensors"), Some(0));
    assert_eq!(index.shard_index("shard2.safetensors"), Some(1));
    assert_eq!(index.shard_index("nonexistent"), None);
}

#[test]
fn test_shard_index_from_json_multiple() {
    let json = r#"{
            "weight_map": {
                "layer1.weight": "shard1.safetensors",
                "layer2.weight": "shard1.safetensors",
                "layer3.weight": "shard2.safetensors"
            }
        }"#;
    let index = ShardIndex::from_json(json).expect("parse should succeed");
    assert_eq!(index.tensor_count(), 3);
    assert_eq!(index.shard_count(), 2);
    assert!(index.is_valid());
}

#[test]
fn test_shard_index_duplicate_shard() {
    let mut index = ShardIndex::new();
    index.add_mapping("a.weight", "shard1.safetensors");
    index.add_mapping("b.weight", "shard1.safetensors");
    index.add_mapping("c.weight", "shard1.safetensors");

    // Same shard for all tensors
    assert_eq!(index.shard_count(), 1);
    assert_eq!(index.tensor_count(), 3);
}

#[test]
fn test_cached_shard_empty() {
    let shard = CachedShard::new("empty.safetensors".to_string());
    assert_eq!(shard.size, 0);
    assert!(!shard.has_tensor("any"));
    assert_eq!(shard.get_tensor("any"), None);
}

#[test]
fn test_cached_shard_multiple_tensors() {
    let mut shard = CachedShard::new("multi.safetensors".to_string());
    shard.add_tensor("t1".to_string(), vec![1, 2, 3]);
    shard.add_tensor("t2".to_string(), vec![4, 5, 6, 7]);
    shard.add_tensor("t3".to_string(), vec![8]);

    assert_eq!(shard.size, 8); // 3 + 4 + 1
    assert!(shard.has_tensor("t1"));
    assert!(shard.has_tensor("t2"));
    assert!(shard.has_tensor("t3"));
    assert_eq!(shard.get_tensor("t2"), Some(&[4u8, 5, 6, 7][..]));
}

#[test]
fn test_shard_cache_clear() {
    let mut cache = ShardCache::new(3, 1024 * 1024);
    cache.insert(CachedShard::new("s1.safetensors".to_string()));
    cache.insert(CachedShard::new("s2.safetensors".to_string()));

    assert_eq!(cache.stats().cached_shards, 2);

    cache.clear();
    assert_eq!(cache.stats().cached_shards, 0);
}

#[test]
fn test_shard_cache_hit_rate_empty() {
    let cache = ShardCache::new(2, 1024);
    // No hits or misses yet
    let rate = cache.hit_rate();
    // 0 / 0 should return 0.0
    assert!((rate - 0.0).abs() < 0.01);
}

#[test]
fn test_shard_cache_max_bytes_eviction() {
    // Very small cache - only 100 bytes
    let mut cache = ShardCache::new(10, 100);

    let mut shard1 = CachedShard::new("s1.safetensors".to_string());
    shard1.add_tensor("t1".to_string(), vec![0u8; 50]);
    cache.insert(shard1);

    let mut shard2 = CachedShard::new("s2.safetensors".to_string());
    shard2.add_tensor("t2".to_string(), vec![0u8; 50]);
    cache.insert(shard2);

    // Cache can hold both (50 + 50 = 100)
    assert_eq!(cache.stats().cached_shards, 2);

    // Third shard should evict first
    let mut shard3 = CachedShard::new("s3.safetensors".to_string());
    shard3.add_tensor("t3".to_string(), vec![0u8; 50]);
    cache.insert(shard3);

    // Should have 2 shards (s2 and s3)
    assert_eq!(cache.stats().cached_shards, 2);
}

#[test]
fn test_import_config_builder() {
    let config = ShardedImportConfig {
        max_cached_shards: 4,
        max_cache_bytes: 10 * 1024 * 1024 * 1024,
        sort_tensors: false,
        verify_checksums: true,
        buffer_size: 8192,
    };

    assert_eq!(config.max_cached_shards, 4);
    assert!(!config.sort_tensors);
    assert!(config.validate().is_ok());
}

#[test]
fn test_import_config_verify_checksums() {
    let config = ShardedImportConfig {
        max_cached_shards: 2,
        max_cache_bytes: 5 * 1024 * 1024 * 1024,
        sort_tensors: true,
        verify_checksums: true,
        buffer_size: 4096,
    };

    assert!(config.verify_checksums);
    assert!(config.validate().is_ok());
}

#[test]
fn test_import_phase_debug() {
    let phase = ImportPhase::Loading;
    let debug_str = format!("{:?}", phase);
    assert!(debug_str.contains("Loading"));
}

#[test]
fn test_sharded_importer_with_custom_config() {
    let config = ShardedImportConfig {
        max_cached_shards: 3,
        max_cache_bytes: 8 * 1024 * 1024 * 1024,
        sort_tensors: true,
        verify_checksums: false,
        buffer_size: 8192,
    };

    let importer = ShardedImporter::new(config, PathBuf::from("/tmp"));
    assert_eq!(importer.cache_stats().cached_shards, 0);
}

#[test]
fn test_estimate_shard_memory_small() {
    let estimate = estimate_shard_memory(1000);
    assert!(estimate >= 1000);
}

#[test]
fn test_estimate_shard_memory_large() {
    let estimate = estimate_shard_memory(1_000_000_000); // 1GB
    assert!(estimate > 1_000_000_000);
}

#[test]
fn test_import_report_with_warnings() {
    let report = ImportReport {
        tensor_count: 50,
        shard_count: 2,
        bytes_written: 512 * 1024,
        peak_memory_bytes: 1024 * 1024 * 1024,
        cache_hit_rate: 0.5,
        duration_ms: 1000,
        warnings: vec!["Some warning".to_string(), "Another warning".to_string()],
    };

    assert_eq!(report.warnings.len(), 2);
    assert_eq!(report.cache_hit_rate, 0.5);
}

#[test]
fn test_shard_index_tensors_by_shard_empty() {
    let index = ShardIndex::new();
    let by_shard = index.tensors_by_shard();
    assert!(by_shard.is_empty());
}

#[test]
fn test_shard_index_default() {
    let index = ShardIndex::default();
    assert_eq!(index.shard_count(), 0);
    assert!(!index.is_valid());
}

include!("tests_part_02.rs");
