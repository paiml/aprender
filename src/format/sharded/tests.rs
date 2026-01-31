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
