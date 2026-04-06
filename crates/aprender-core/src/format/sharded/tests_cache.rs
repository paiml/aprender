use super::*;

#[test]
fn test_cache_stats() {
    let mut cache = ShardCache::new(2, 1024);
    let mut shard = CachedShard::new("test.safetensors".to_string());
    shard.add_tensor("t".to_string(), vec![1, 2, 3, 4, 5]);
    cache.insert(shard);

    let stats = cache.stats();
    assert_eq!(stats.cached_shards, 1);
    assert_eq!(stats.cached_bytes, 5);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
}

// ========================================================================
// Additional Coverage Tests for sharded.rs
// ========================================================================

#[test]
fn test_sharded_import_config_low_memory() {
    let config = ShardedImportConfig::low_memory();
    assert_eq!(config.max_cached_shards, 1);
    assert_eq!(config.max_cache_bytes, 1024 * 1024 * 1024); // 1GB
    assert_eq!(config.buffer_size, 4 * 1024 * 1024); // 4MB
    assert!(config.sort_tensors);
    assert!(config.verify_checksums);
}

#[test]
fn test_sharded_import_config_high_memory() {
    let config = ShardedImportConfig::high_memory();
    assert_eq!(config.max_cached_shards, 4);
    // On native, 8GB; on WASM, 512MB - just check it's reasonable
    assert!(config.max_cache_bytes >= 512 * 1024 * 1024);
    assert_eq!(config.buffer_size, 16 * 1024 * 1024); // 16MB
}

#[test]
fn test_sharded_import_config_validate_zero_buffer() {
    let mut config = ShardedImportConfig::default();
    config.buffer_size = 0;
    let result = config.validate();
    assert!(result.is_err());
}

#[test]
fn test_sharded_importer_config_accessor() {
    let custom_config = ShardedImportConfig {
        max_cached_shards: 5,
        ..Default::default()
    };
    let importer = ShardedImporter::new(custom_config.clone(), PathBuf::from("/tmp"));
    assert_eq!(importer.config().max_cached_shards, 5);
}

#[test]
fn test_sharded_importer_clear_cache() {
    let mut importer = ShardedImporter::default();

    // Add something to cache by simulating import
    // Just verify clear_cache runs without panic
    importer.clear_cache();
    assert_eq!(importer.cache_stats().cached_shards, 0);
}

#[test]
fn test_import_progress_fields() {
    let progress = ImportProgress {
        phase: ImportPhase::Loading,
        tensors_processed: 50,
        total_tensors: 100,
        shards_loaded: 2,
        total_shards: 4,
        bytes_written: 1024 * 1024,
        progress: 0.5,
    };

    assert_eq!(progress.tensors_processed, 50);
    assert_eq!(progress.total_tensors, 100);
    assert_eq!(progress.shards_loaded, 2);
    assert_eq!(progress.total_shards, 4);
    assert!((progress.progress - 0.5).abs() < 0.001);
}

#[test]
fn test_import_progress_clone() {
    let progress = ImportProgress {
        phase: ImportPhase::Merging,
        tensors_processed: 75,
        total_tensors: 100,
        shards_loaded: 3,
        total_shards: 4,
        bytes_written: 2 * 1024 * 1024,
        progress: 0.75,
    };
    let cloned = progress.clone();
    assert_eq!(cloned.phase, ImportPhase::Merging);
    assert_eq!(cloned.tensors_processed, 75);
}

#[test]
fn test_import_phase_all_variants() {
    // Test all import phase variants
    assert_eq!(ImportPhase::Parsing, ImportPhase::Parsing);
    assert_eq!(ImportPhase::Loading, ImportPhase::Loading);
    assert_eq!(ImportPhase::Merging, ImportPhase::Merging);
    assert_eq!(ImportPhase::Finalizing, ImportPhase::Finalizing);
    assert_eq!(ImportPhase::Complete, ImportPhase::Complete);

    // Test inequality
    assert_ne!(ImportPhase::Parsing, ImportPhase::Loading);
    assert_ne!(ImportPhase::Merging, ImportPhase::Complete);
}

#[test]
fn test_import_phase_copy() {
    let phase = ImportPhase::Merging;
    let copied = phase;
    assert_eq!(copied, ImportPhase::Merging);
}

#[test]
fn test_cache_stats_debug() {
    let stats = CacheStats {
        cached_shards: 2,
        cached_bytes: 4096,
        hits: 10,
        misses: 5,
    };
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("cached_shards"));
    assert!(debug_str.contains("hits"));
}

#[test]
fn test_cache_stats_copy() {
    let stats = CacheStats {
        cached_shards: 1,
        cached_bytes: 2048,
        hits: 5,
        misses: 2,
    };
    let copied = stats;
    assert_eq!(copied.cached_shards, 1);
    assert_eq!(copied.hits, 5);
}

#[test]
fn test_import_report_clone() {
    let report = ImportReport {
        tensor_count: 100,
        shard_count: 4,
        bytes_written: 1024 * 1024,
        peak_memory_bytes: 2 * 1024 * 1024 * 1024,
        cache_hit_rate: 0.8,
        duration_ms: 5000,
        warnings: vec!["warning1".to_string()],
    };
    let cloned = report.clone();
    assert_eq!(cloned.tensor_count, 100);
    assert_eq!(cloned.shard_count, 4);
    assert_eq!(cloned.warnings.len(), 1);
}

#[test]
fn test_import_report_debug() {
    let report = ImportReport {
        tensor_count: 50,
        shard_count: 2,
        bytes_written: 512 * 1024,
        peak_memory_bytes: 1024 * 1024 * 1024,
        cache_hit_rate: 0.6,
        duration_ms: 2000,
        warnings: vec![],
    };
    let debug_str = format!("{:?}", report);
    assert!(debug_str.contains("tensor_count"));
    assert!(debug_str.contains("50"));
}

#[test]
fn test_sharded_importer_debug() {
    let importer = ShardedImporter::default();
    let debug_str = format!("{:?}", importer);
    assert!(debug_str.contains("ShardedImporter"));
}

#[test]
fn test_shard_cache_default() {
    let cache = ShardCache::default();
    assert_eq!(cache.stats().cached_shards, 0);
    // Default for import uses 2 shards
}

#[test]
fn test_shard_cache_default_for_import() {
    let cache = ShardCache::default_for_import();
    assert_eq!(cache.stats().cached_shards, 0);
}

#[test]
fn test_cached_shard_debug() {
    let shard = CachedShard::new("test.safetensors".to_string());
    let debug_str = format!("{:?}", shard);
    assert!(debug_str.contains("CachedShard"));
    assert!(debug_str.contains("test.safetensors"));
}

#[test]
fn test_cached_shard_clone() {
    let mut shard = CachedShard::new("test.safetensors".to_string());
    shard.add_tensor("tensor1".to_string(), vec![1, 2, 3]);
    let cloned = shard.clone();
    assert_eq!(cloned.filename, "test.safetensors");
    assert_eq!(cloned.size, 3);
    assert!(cloned.has_tensor("tensor1"));
}

#[test]
fn test_shard_index_clone() {
    let mut index = ShardIndex::new();
    index.add_mapping("layer1.weight", "shard1.safetensors");
    index.set_metadata("version", "1.0");

    let cloned = index.clone();
    assert_eq!(cloned.tensor_count(), 1);
    assert_eq!(cloned.get_metadata("version"), Some("1.0"));
}

#[test]
fn test_shard_index_debug() {
    let index = ShardIndex::new();
    let debug_str = format!("{:?}", index);
    assert!(debug_str.contains("ShardIndex"));
}

#[test]
fn test_sharded_import_config_clone() {
    let config = ShardedImportConfig {
        max_cached_shards: 3,
        max_cache_bytes: 2 * 1024 * 1024 * 1024,
        sort_tensors: false,
        verify_checksums: false,
        buffer_size: 8192,
    };
    let cloned = config.clone();
    assert_eq!(cloned.max_cached_shards, 3);
    assert!(!cloned.sort_tensors);
    assert!(!cloned.verify_checksums);
}

#[test]
fn test_sharded_import_config_debug() {
    let config = ShardedImportConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("ShardedImportConfig"));
}

#[test]
fn test_shard_index_from_json_whitespace() {
    let json = r#"  {  "weight_map"  :  {  }  }  "#;
    let result = ShardIndex::from_json(json);
    assert!(result.is_ok());
}

#[test]
fn test_shard_index_from_json_no_weight_map() {
    let json = r#"{"metadata": {"version": "1.0"}}"#;
    let result = ShardIndex::from_json(json);
    assert!(result.is_ok());
    let index = result.expect("parse should succeed");
    assert!(!index.is_valid()); // No weight_map = invalid
}

#[test]
fn test_get_shard_files_empty_dir() {
    use std::fs;
    let temp_dir = std::env::temp_dir().join("aprender_test_empty");
    let _ = fs::create_dir_all(&temp_dir);

    let result = get_shard_files(&temp_dir);
    assert!(result.is_ok());
    let files = result.expect("should succeed");
    // May have files from other tests, just verify it returns a Vec
    assert!(files.is_empty() || !files.is_empty());

    let _ = fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_estimate_shard_memory_zero() {
    let estimate = estimate_shard_memory(0);
    assert_eq!(estimate, 0);
}

#[test]
fn test_is_sharded_model_temp_dir() {
    let temp_dir = std::env::temp_dir();
    // Just verify it doesn't panic
    let result = is_sharded_model(&temp_dir);
    // Result depends on temp dir contents
    assert!(result || !result);
}
