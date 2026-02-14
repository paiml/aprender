use super::*;
use crate::bundle::{BundleManifest, BundleWriter, ModelEntry};
use std::collections::HashMap;
use tempfile::NamedTempFile;

fn create_test_bundle(models: &[(&str, Vec<u8>)]) -> NamedTempFile {
    let temp = NamedTempFile::new().expect("Failed to create temp file");

    let mut manifest = BundleManifest::new();
    let mut model_map = HashMap::new();

    for (name, data) in models {
        manifest.add_model(ModelEntry::new(*name, data.len()));
        model_map.insert((*name).to_string(), data.clone());
    }

    let writer = BundleWriter::create(temp.path()).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &model_map)
        .expect("Failed to write bundle");

    temp
}

#[test]
fn test_paging_config_default() {
    let config = PagingConfig::default();
    assert_eq!(config.max_memory, DEFAULT_MAX_MEMORY);
    assert_eq!(config.page_size, DEFAULT_PAGE_SIZE);
    assert!(config.prefetch);
    assert_eq!(config.eviction, EvictionStrategy::LRU);
}

#[test]
fn test_paging_config_builder() {
    let config = PagingConfig::new()
        .with_max_memory(50_000)
        .with_page_size(8192)
        .with_prefetch(false)
        .with_eviction(EvictionStrategy::LFU);

    assert_eq!(config.max_memory, 50_000);
    assert_eq!(config.page_size, 8192);
    assert!(!config.prefetch);
    assert_eq!(config.eviction, EvictionStrategy::LFU);
}

#[test]
fn test_paging_stats() {
    let mut stats = PagingStats::default();
    assert_eq!(stats.hit_rate(), 0.0);

    stats.hits = 3;
    stats.misses = 1;
    assert!((stats.hit_rate() - 0.75).abs() < f32::EPSILON);
}

#[test]
fn test_paged_bundle_open() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3]), ("model2", vec![4, 5, 6, 7])]);

    let bundle =
        PagedBundle::open(temp.path(), PagingConfig::default()).expect("Failed to open bundle");

    assert_eq!(bundle.model_names().len(), 2);
    assert_eq!(bundle.cached_count(), 0);
    assert_eq!(bundle.memory_used(), 0);
}

#[test]
fn test_paged_bundle_get_model() {
    let temp = create_test_bundle(&[("weights", vec![10, 20, 30, 40, 50])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
        .expect("Failed to open");

    let data = bundle.get_model("weights").expect("Failed to get model");
    assert_eq!(data, &[10, 20, 30, 40, 50]);
    assert_eq!(bundle.cached_count(), 1);
    assert_eq!(bundle.stats().misses, 1);
    assert_eq!(bundle.stats().hits, 0);

    // Second access should be a hit
    let _ = bundle.get_model("weights").expect("Failed to get model");
    assert_eq!(bundle.stats().hits, 1);
}

#[test]
fn test_paged_bundle_eviction() {
    // Use 1000-byte models with 1500-byte max memory
    // This forces eviction after the first model since 2000 > 1500
    let temp = create_test_bundle(&[
        ("model1", vec![1; 1000]),
        ("model2", vec![2; 1000]),
        ("model3", vec![3; 1000]),
    ]);

    // Small max memory to force eviction (1500 = 1.5 models worth)
    let mut bundle = PagedBundle::open(
        temp.path(),
        PagingConfig::new()
            .with_max_memory(1500)
            .with_prefetch(false),
    )
    .expect("Failed to open");

    // Load first model - fits in memory
    let _ = bundle.get_model("model1").expect("Failed");
    assert_eq!(bundle.cached_count(), 1);
    assert_eq!(bundle.memory_used(), 1000);

    // Load second model - should trigger eviction of model1
    // 1000 + 1000 = 2000 > 1500, must evict
    let _ = bundle.get_model("model2").expect("Failed");
    assert!(
        bundle.stats().evictions > 0,
        "Expected evictions > 0, got {}",
        bundle.stats().evictions
    );
    assert!(bundle.memory_used() <= 1500);

    // Load third model - should trigger another eviction
    let _ = bundle.get_model("model3").expect("Failed");
    assert!(bundle.stats().evictions >= 2);
    assert!(bundle.memory_used() <= 1500);
}

#[test]
fn test_paged_bundle_explicit_evict() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
        .expect("Failed to open");

    // Load model
    let _ = bundle.get_model("model1").expect("Failed");
    assert!(bundle.is_cached("model1"));

    // Explicitly evict
    let evicted = bundle.evict("model1");
    assert!(evicted);
    assert!(!bundle.is_cached("model1"));
}

#[test]
fn test_paged_bundle_clear_cache() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3]), ("model2", vec![4, 5, 6])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
        .expect("Failed to open");

    let _ = bundle.get_model("model1").expect("Failed");
    let _ = bundle.get_model("model2").expect("Failed");
    assert_eq!(bundle.cached_count(), 2);

    bundle.clear_cache();
    assert_eq!(bundle.cached_count(), 0);
    assert_eq!(bundle.memory_used(), 0);
}

#[test]
fn test_paged_bundle_prefetch_hint() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new()).expect("Failed to open");

    // Pre-fetch
    bundle.prefetch_hint("model1").expect("Prefetch failed");
    assert!(bundle.is_cached("model1"));

    // Access should now be a hit
    let _ = bundle.get_model("model1").expect("Failed");
    assert_eq!(bundle.stats().hits, 1);
    assert_eq!(bundle.stats().misses, 0);
}

#[test]
fn test_paged_bundle_nonexistent_model() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
        .expect("Failed to open");

    let result = bundle.get_model("nonexistent");
    assert!(result.is_err());
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_paging_config_with_prefetch_count() {
    let config = PagingConfig::new().with_prefetch_count(5);
    assert_eq!(config.prefetch_count, 5);
}

#[test]
fn test_paging_config_min_values_enforced() {
    // max_memory has minimum of 1024
    let config = PagingConfig::new().with_max_memory(100);
    assert_eq!(config.max_memory, 1024);

    // page_size has minimum of 512
    let config2 = PagingConfig::new().with_page_size(100);
    assert_eq!(config2.page_size, 512);
}

#[test]
fn test_paging_stats_reset() {
    let mut stats = PagingStats {
        hits: 10,
        misses: 5,
        evictions: 3,
        bytes_loaded: 1000,
        memory_used: 500,
        prefetches: 2,
    };

    stats.reset();

    assert_eq!(stats.hits, 0);
    assert_eq!(stats.misses, 0);
    assert_eq!(stats.evictions, 0);
    assert_eq!(stats.bytes_loaded, 0);
    assert_eq!(stats.memory_used, 0);
    assert_eq!(stats.prefetches, 0);
}

#[test]
fn test_eviction_strategy_default() {
    let strategy = EvictionStrategy::default();
    assert_eq!(strategy, EvictionStrategy::LRU);
}

#[test]
fn test_eviction_strategy_lfu() {
    // Create bundle with LFU eviction
    let temp = create_test_bundle(&[
        ("model1", vec![1; 1000]),
        ("model2", vec![2; 1000]),
        ("model3", vec![3; 1000]),
    ]);

    let mut bundle = PagedBundle::open(
        temp.path(),
        PagingConfig::new()
            .with_max_memory(1500)
            .with_prefetch(false)
            .with_eviction(EvictionStrategy::LFU),
    )
    .expect("Failed to open");

    // Load and access models
    let _ = bundle.get_model("model1").expect("Failed");
    let _ = bundle.get_model("model2").expect("Failed");

    // model1 was evicted to load model2
    assert!(bundle.stats().evictions > 0);
}

#[test]
fn test_paged_bundle_get_metadata() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let bundle = PagedBundle::open(temp.path(), PagingConfig::default()).expect("Failed to open");

    let metadata = bundle.get_metadata("model1");
    assert!(metadata.is_some());
    let entry = metadata.unwrap();
    assert_eq!(entry.name, "model1");
    assert_eq!(entry.size, 3);

    // Non-existent model
    assert!(bundle.get_metadata("nonexistent").is_none());
}

#[test]
fn test_paged_bundle_config() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let config = PagingConfig::new()
        .with_max_memory(100_000)
        .with_page_size(4096);

    let bundle = PagedBundle::open(temp.path(), config).expect("Failed to open");

    assert_eq!(bundle.config().max_memory, 100_000);
    assert_eq!(bundle.config().page_size, 4096);
}

#[test]
fn test_paged_bundle_debug_impl() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let bundle = PagedBundle::open(temp.path(), PagingConfig::default()).expect("Failed to open");

    let debug_str = format!("{:?}", bundle);
    assert!(debug_str.contains("PagedBundle"));
    assert!(debug_str.contains("models"));
    assert!(debug_str.contains("cached"));
    assert!(debug_str.contains("memory_used"));
    assert!(debug_str.contains("hit_rate"));
}

#[test]
fn test_evict_nonexistent_model() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
        .expect("Failed to open");

    // Try to evict a model that's not cached
    let evicted = bundle.evict("model1");
    assert!(!evicted);
}

#[test]
fn test_prefetch_hint_nonexistent_model() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new()).expect("Failed to open");

    // Prefetch hint for non-existent model should succeed (no-op)
    let result = bundle.prefetch_hint("nonexistent");
    assert!(result.is_ok());
    assert!(!bundle.is_cached("nonexistent"));
}

#[test]
fn test_prefetch_hint_already_cached() {
    let temp = create_test_bundle(&[("model1", vec![1, 2, 3])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new()).expect("Failed to open");

    // Load model first
    let _ = bundle.get_model("model1").expect("Failed");
    let prefetches_before = bundle.stats().prefetches;

    // Prefetch hint for already cached model should be no-op
    bundle.prefetch_hint("model1").expect("Prefetch failed");
    assert_eq!(bundle.stats().prefetches, prefetches_before);
}

#[test]
fn test_paging_stats_clone() {
    let stats = PagingStats {
        hits: 5,
        misses: 3,
        evictions: 1,
        bytes_loaded: 100,
        memory_used: 50,
        prefetches: 2,
    };

    let cloned = stats.clone();
    assert_eq!(cloned.hits, stats.hits);
    assert_eq!(cloned.misses, stats.misses);
    assert_eq!(cloned.evictions, stats.evictions);
}

#[test]
fn test_paging_config_clone() {
    let config = PagingConfig::new()
        .with_max_memory(50_000)
        .with_eviction(EvictionStrategy::LFU);

    let cloned = config.clone();
    assert_eq!(cloned.max_memory, config.max_memory);
    assert_eq!(cloned.eviction, EvictionStrategy::LFU);
}

#[test]
fn test_eviction_strategy_copy() {
    let strategy = EvictionStrategy::LFU;
    let copied = strategy;
    assert_eq!(copied, EvictionStrategy::LFU);
}

#[test]
fn test_cache_hit_updates_lru() {
    let temp = create_test_bundle(&[
        ("model1", vec![1; 500]),
        ("model2", vec![2; 500]),
        ("model3", vec![3; 500]),
    ]);

    // Max memory can hold 2 models
    let mut bundle = PagedBundle::open(
        temp.path(),
        PagingConfig::new()
            .with_max_memory(1200)
            .with_prefetch(false),
    )
    .expect("Failed to open");

    // Load model1 and model2
    let _ = bundle.get_model("model1").expect("Failed");
    let _ = bundle.get_model("model2").expect("Failed");

    // Access model1 again (should update LRU order)
    let _ = bundle.get_model("model1").expect("Failed");
    assert_eq!(bundle.stats().hits, 1);

    // Now load model3 - model2 should be evicted (oldest in LRU)
    let _ = bundle.get_model("model3").expect("Failed");

    // model1 should still be cached, model2 should be evicted
    assert!(bundle.is_cached("model1"));
    assert!(bundle.is_cached("model3"));
}

#[test]
fn test_model_names() {
    let temp = create_test_bundle(&[("alpha", vec![1]), ("beta", vec![2]), ("gamma", vec![3])]);

    let bundle = PagedBundle::open(temp.path(), PagingConfig::default()).expect("Failed to open");

    let names = bundle.model_names();
    assert_eq!(names.len(), 3);
    assert!(names.contains(&"alpha"));
    assert!(names.contains(&"beta"));
    assert!(names.contains(&"gamma"));
}

#[test]
fn test_prefetch_with_access_pattern() {
    let temp = create_test_bundle(&[
        ("model1", vec![1; 100]),
        ("model2", vec![2; 100]),
        ("model3", vec![3; 100]),
    ]);

    let mut bundle = PagedBundle::open(
        temp.path(),
        PagingConfig::new()
            .with_max_memory(1_000_000) // Large enough to not evict
            .with_prefetch(true)
            .with_prefetch_count(2),
    )
    .expect("Failed to open");

    // Build access pattern: model1 -> model2
    let _ = bundle.get_model("model1").expect("Failed");
    let _ = bundle.get_model("model2").expect("Failed");

    // Access model1 again - should trigger prefetch based on pattern
    let _ = bundle.get_model("model1").expect("Failed");

    // Verify prefetch system was exercised
    assert!(bundle.stats().hits > 0);
}

#[test]
fn test_bytes_loaded_tracking() {
    let temp = create_test_bundle(&[("model1", vec![1; 100]), ("model2", vec![2; 200])]);

    let mut bundle = PagedBundle::open(temp.path(), PagingConfig::new().with_prefetch(false))
        .expect("Failed to open");

    let _ = bundle.get_model("model1").expect("Failed");
    assert_eq!(bundle.stats().bytes_loaded, 100);

    let _ = bundle.get_model("model2").expect("Failed");
    assert_eq!(bundle.stats().bytes_loaded, 300);
}
