use super::*;

// ========================================================================
// HF-QUERY-002-001: HubAssetType Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_001_asset_type_display() {
    assert_eq!(format!("{}", HubAssetType::Model), "model");
    assert_eq!(format!("{}", HubAssetType::Dataset), "dataset");
    assert_eq!(format!("{}", HubAssetType::Space), "space");
}

#[test]
fn test_HF_QUERY_002_002_asset_type_serialize() {
    let json = serde_json::to_string(&HubAssetType::Model).unwrap();
    assert_eq!(json, "\"model\"");
}

// ========================================================================
// HF-QUERY-002-010: HubAsset Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_010_hub_asset_new() {
    let asset = HubAsset::new("org/model", HubAssetType::Model);
    assert_eq!(asset.id, "org/model");
    assert_eq!(asset.author, "org");
    assert_eq!(asset.asset_type, HubAssetType::Model);
}

#[test]
fn test_HF_QUERY_002_011_hub_asset_with_downloads() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_downloads(1000);
    assert_eq!(asset.downloads, 1000);
}

#[test]
fn test_HF_QUERY_002_012_hub_asset_with_likes() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_likes(100);
    assert_eq!(asset.likes, 100);
}

#[test]
fn test_HF_QUERY_002_013_hub_asset_with_tags() {
    let asset = HubAsset::new("org/model", HubAssetType::Model)
        .with_tags(vec!["nlp".into(), "bert".into()]);
    assert_eq!(asset.tags.len(), 2);
}

#[test]
fn test_HF_QUERY_002_014_hub_asset_with_pipeline_tag() {
    let asset =
        HubAsset::new("org/model", HubAssetType::Model).with_pipeline_tag("text-generation");
    assert_eq!(asset.pipeline_tag, Some("text-generation".to_string()));
}

#[test]
fn test_HF_QUERY_002_015_hub_asset_with_library() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_library("transformers");
    assert_eq!(asset.library, Some("transformers".to_string()));
}

#[test]
fn test_HF_QUERY_002_016_hub_asset_with_license() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_license("apache-2.0");
    assert_eq!(asset.license, Some("apache-2.0".to_string()));
}

#[test]
fn test_HF_QUERY_002_017_hub_asset_serialize() {
    let asset = HubAsset::new("org/model", HubAssetType::Model).with_downloads(100);
    let json = serde_json::to_string(&asset).unwrap();
    assert!(json.contains("\"id\":\"org/model\""));
    assert!(json.contains("\"downloads\":100"));
}

// ========================================================================
// HF-QUERY-002-020: SearchFilters Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_020_filters_new() {
    let filters = SearchFilters::new();
    assert_eq!(filters.limit, 20);
    assert!(filters.task.is_none());
}

#[test]
fn test_HF_QUERY_002_021_filters_with_task() {
    let filters = SearchFilters::new().with_task("text-generation");
    assert_eq!(filters.task, Some("text-generation".to_string()));
}

#[test]
fn test_HF_QUERY_002_022_filters_with_library() {
    let filters = SearchFilters::new().with_library("transformers");
    assert_eq!(filters.library, Some("transformers".to_string()));
}

#[test]
fn test_HF_QUERY_002_023_filters_with_author() {
    let filters = SearchFilters::new().with_author("meta-llama");
    assert_eq!(filters.author, Some("meta-llama".to_string()));
}

#[test]
fn test_HF_QUERY_002_024_filters_with_min_downloads() {
    let filters = SearchFilters::new().with_min_downloads(10000);
    assert_eq!(filters.min_downloads, Some(10000));
}

#[test]
fn test_HF_QUERY_002_025_filters_with_min_likes() {
    let filters = SearchFilters::new().with_min_likes(100);
    assert_eq!(filters.min_likes, Some(100));
}

#[test]
fn test_HF_QUERY_002_026_filters_with_limit() {
    let filters = SearchFilters::new().with_limit(10);
    assert_eq!(filters.limit, 10);
}

#[test]
fn test_HF_QUERY_002_027_filters_with_sort() {
    let filters = SearchFilters::new().with_sort("downloads", "desc");
    assert_eq!(filters.sort, Some("downloads".to_string()));
    assert_eq!(filters.sort_direction, Some("desc".to_string()));
}

#[test]
fn test_HF_QUERY_002_028_filters_chain() {
    let filters = SearchFilters::new()
        .with_task("text-generation")
        .with_library("transformers")
        .with_min_downloads(1000)
        .with_limit(5);
    assert_eq!(filters.task, Some("text-generation".to_string()));
    assert_eq!(filters.library, Some("transformers".to_string()));
    assert_eq!(filters.min_downloads, Some(1000));
    assert_eq!(filters.limit, 5);
}

// ========================================================================
// HF-QUERY-002-030: ResponseCache Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_030_cache_new() {
    let cache = ResponseCache::new(Duration::from_secs(60));
    let stats = cache.stats();
    assert_eq!(stats.search_entries, 0);
    assert_eq!(stats.asset_entries, 0);
}

#[test]
fn test_HF_QUERY_002_031_cache_default_ttl() {
    let cache = ResponseCache::default_ttl();
    let stats = cache.stats();
    assert_eq!(stats.ttl_secs, 15 * 60);
}

#[test]
fn test_HF_QUERY_002_032_cache_search() {
    let mut cache = ResponseCache::default_ttl();
    let results = vec![HubAsset::new("test", HubAssetType::Model)];
    cache.cache_search("key", results.clone());

    let cached = cache.get_search("key");
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().len(), 1);
}

#[test]
fn test_HF_QUERY_002_033_cache_search_miss() {
    let cache = ResponseCache::default_ttl();
    assert!(cache.get_search("nonexistent").is_none());
}

#[test]
fn test_HF_QUERY_002_034_cache_asset() {
    let mut cache = ResponseCache::default_ttl();
    let asset = HubAsset::new("org/model", HubAssetType::Model);
    cache.cache_asset("org/model", asset);

    let cached = cache.get_asset("org/model");
    assert!(cached.is_some());
}

#[test]
fn test_HF_QUERY_002_035_cache_clear() {
    let mut cache = ResponseCache::default_ttl();
    cache.cache_search("key", vec![]);
    cache.cache_asset("id", HubAsset::new("id", HubAssetType::Model));

    cache.clear();
    assert!(cache.get_search("key").is_none());
    assert!(cache.get_asset("id").is_none());
}

#[test]
fn test_HF_QUERY_002_036_cache_expired() {
    let mut cache = ResponseCache::new(Duration::from_millis(1));
    cache.cache_search("key", vec![]);

    // Wait for expiration
    std::thread::sleep(Duration::from_millis(10));

    assert!(cache.get_search("key").is_none());
}

// ========================================================================
// HF-QUERY-002-040: HubClient Search Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_040_client_new() {
    let client = HubClient::new();
    assert!(!client.offline_mode);
}

#[test]
fn test_HF_QUERY_002_041_client_search_models() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let results = client.search_models(&filters).unwrap();
    assert!(!results.is_empty());
}

#[test]
fn test_HF_QUERY_002_042_client_search_models_with_task_filter() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_task("text-generation");
    let results = client.search_models(&filters).unwrap();
    assert!(results.iter().all(|m| m.pipeline_tag == Some("text-generation".to_string())));
}

#[test]
fn test_HF_QUERY_002_043_client_search_models_with_library_filter() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_library("diffusers");
    let results = client.search_models(&filters).unwrap();
    assert!(results.iter().all(|m| m.library == Some("diffusers".to_string())));
}

#[test]
fn test_HF_QUERY_002_044_client_search_models_with_min_downloads() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_min_downloads(5_000_000);
    let results = client.search_models(&filters).unwrap();
    assert!(results.iter().all(|m| m.downloads >= 5_000_000));
}

#[test]
fn test_HF_QUERY_002_045_client_search_datasets() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let results = client.search_datasets(&filters).unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|d| d.asset_type == HubAssetType::Dataset));
}

#[test]
fn test_HF_QUERY_002_046_client_search_spaces() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let results = client.search_spaces(&filters).unwrap();
    assert!(!results.is_empty());
    assert!(results.iter().all(|s| s.asset_type == HubAssetType::Space));
}

#[test]
fn test_HF_QUERY_002_047_client_search_caching() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();

    // First call
    let results1 = client.search_models(&filters).unwrap();
    // Second call should hit cache
    let results2 = client.search_models(&filters).unwrap();

    assert_eq!(results1.len(), results2.len());
}

#[test]
fn test_HF_QUERY_002_048_client_offline_mode() {
    let mut client = HubClient::new().offline();
    let filters = SearchFilters::new();
    let result = client.search_models(&filters);
    assert!(matches!(result, Err(HubError::OfflineMode)));
}

// ========================================================================
// HF-QUERY-003-001: Asset Metadata Tests
// ========================================================================

#[test]
fn test_HF_QUERY_003_001_get_model() {
    let mut client = HubClient::new();
    let result = client.get_model("meta-llama/Llama-2-7b-hf");
    assert!(result.is_ok());
    let model = result.unwrap();
    assert_eq!(model.id, "meta-llama/Llama-2-7b-hf");
    assert_eq!(model.asset_type, HubAssetType::Model);
}

#[test]
fn test_HF_QUERY_003_002_get_model_not_found() {
    let mut client = HubClient::new();
    let result = client.get_model("nonexistent/model");
    assert!(matches!(result, Err(HubError::NotFound(_))));
}

#[test]
fn test_HF_QUERY_003_003_get_dataset() {
    let mut client = HubClient::new();
    let result = client.get_dataset("squad");
    assert!(result.is_ok());
    let dataset = result.unwrap();
    assert_eq!(dataset.id, "squad");
    assert_eq!(dataset.asset_type, HubAssetType::Dataset);
}

#[test]
fn test_HF_QUERY_003_004_get_space() {
    let mut client = HubClient::new();
    let result = client.get_space("gradio/chatbot");
    assert!(result.is_ok());
    let space = result.unwrap();
    assert_eq!(space.id, "gradio/chatbot");
    assert_eq!(space.asset_type, HubAssetType::Space);
}

#[test]
fn test_HF_QUERY_003_005_get_model_caching() {
    let mut client = HubClient::new();

    // First call
    let model1 = client.get_model("bert-base-uncased").unwrap();
    // Second call hits cache
    let model2 = client.get_model("bert-base-uncased").unwrap();

    assert_eq!(model1.id, model2.id);
}

#[test]
fn test_HF_QUERY_003_006_clear_cache() {
    let mut client = HubClient::new();
    client.get_model("bert-base-uncased").unwrap();

    let stats_before = client.cache_stats();
    assert!(stats_before.asset_entries > 0);

    client.clear_cache();
    let stats_after = client.cache_stats();
    assert_eq!(stats_after.asset_entries, 0);
}

// ========================================================================
// HF-QUERY-002/003-050: Error Tests
// ========================================================================

#[test]
fn test_HF_QUERY_002_050_error_display_not_found() {
    let err = HubError::NotFound("test".to_string());
    assert_eq!(format!("{}", err), "Asset not found: test");
}

#[test]
fn test_HF_QUERY_002_051_error_display_rate_limited() {
    let err = HubError::RateLimited { retry_after: Some(60) };
    assert!(format!("{}", err).contains("60 seconds"));
}

#[test]
fn test_HF_QUERY_002_052_error_display_offline() {
    let err = HubError::OfflineMode;
    assert!(format!("{}", err).contains("Offline"));
}

#[test]
fn test_HF_QUERY_002_053_error_display_network() {
    let err = HubError::NetworkError("connection refused".to_string());
    assert!(format!("{}", err).contains("connection refused"));
}

// ========================================================================
// Coverage gap tests: Builder chaining, mock search filters, cache expiry,
// offline mode for all methods, error variants, asset author extraction
// ========================================================================

#[test]
fn test_HF_QUERY_002_054_hub_asset_no_slash_author() {
    let asset = HubAsset::new("single-name", HubAssetType::Model);
    assert_eq!(asset.author, "single-name");
}

#[test]
fn test_HF_QUERY_002_055_hub_asset_builder_chain() {
    let asset = HubAsset::new("org/model", HubAssetType::Model)
        .with_downloads(1000)
        .with_likes(50)
        .with_tags(vec!["nlp".into()])
        .with_pipeline_tag("text-generation")
        .with_library("transformers")
        .with_license("mit");

    assert_eq!(asset.downloads, 1000);
    assert_eq!(asset.likes, 50);
    assert_eq!(asset.tags.len(), 1);
    assert_eq!(asset.pipeline_tag, Some("text-generation".to_string()));
    assert_eq!(asset.library, Some("transformers".to_string()));
    assert_eq!(asset.license, Some("mit".to_string()));
}

#[test]
fn test_HF_QUERY_002_056_hub_asset_default_fields() {
    let asset = HubAsset::new("org/model", HubAssetType::Dataset);
    assert_eq!(asset.downloads, 0);
    assert_eq!(asset.likes, 0);
    assert!(asset.tags.is_empty());
    assert!(asset.pipeline_tag.is_none());
    assert!(asset.library.is_none());
    assert!(asset.license.is_none());
    assert!(asset.last_modified.is_empty());
    assert!(asset.card_content.is_none());
}

#[test]
fn test_HF_QUERY_002_057_search_models_min_likes_filter() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_min_likes(10_000);
    let results = client.search_models(&filters).unwrap();
    assert!(results.iter().all(|m| m.likes >= 10_000));
}

#[test]
fn test_HF_QUERY_002_058_search_models_limit() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_limit(2);
    let results = client.search_models(&filters).unwrap();
    assert!(results.len() <= 2);
}

#[test]
fn test_HF_QUERY_002_059_search_datasets_min_downloads_filter() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_min_downloads(5_000_000);
    let results = client.search_datasets(&filters).unwrap();
    assert!(results.iter().all(|d| d.downloads >= 5_000_000));
}

#[test]
fn test_HF_QUERY_002_060_search_datasets_limit() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_limit(1);
    let results = client.search_datasets(&filters).unwrap();
    assert!(results.len() <= 1);
}

#[test]
fn test_HF_QUERY_002_061_search_spaces_min_downloads_filter() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_min_downloads(200_000);
    let results = client.search_spaces(&filters).unwrap();
    assert!(results.iter().all(|s| s.downloads >= 200_000));
}

#[test]
fn test_HF_QUERY_002_062_search_spaces_limit() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new().with_limit(1);
    let results = client.search_spaces(&filters).unwrap();
    assert!(results.len() <= 1);
}

#[test]
fn test_HF_QUERY_002_063_offline_datasets() {
    let mut client = HubClient::new().offline();
    let filters = SearchFilters::new();
    let result = client.search_datasets(&filters);
    assert!(matches!(result, Err(HubError::OfflineMode)));
}

#[test]
fn test_HF_QUERY_002_064_offline_spaces() {
    let mut client = HubClient::new().offline();
    let filters = SearchFilters::new();
    let result = client.search_spaces(&filters);
    assert!(matches!(result, Err(HubError::OfflineMode)));
}

#[test]
fn test_HF_QUERY_003_007_offline_get_model() {
    let mut client = HubClient::new().offline();
    let result = client.get_model("meta-llama/Llama-2-7b-hf");
    assert!(matches!(result, Err(HubError::OfflineMode)));
}

#[test]
fn test_HF_QUERY_003_008_offline_get_dataset() {
    let mut client = HubClient::new().offline();
    let result = client.get_dataset("squad");
    assert!(matches!(result, Err(HubError::OfflineMode)));
}

#[test]
fn test_HF_QUERY_003_009_offline_get_space() {
    let mut client = HubClient::new().offline();
    let result = client.get_space("gradio/chatbot");
    assert!(matches!(result, Err(HubError::OfflineMode)));
}

#[test]
fn test_HF_QUERY_003_010_get_dataset_not_found() {
    let mut client = HubClient::new();
    let result = client.get_dataset("nonexistent");
    assert!(matches!(result, Err(HubError::NotFound(_))));
}

#[test]
fn test_HF_QUERY_003_011_get_space_not_found() {
    let mut client = HubClient::new();
    let result = client.get_space("nonexistent");
    assert!(matches!(result, Err(HubError::NotFound(_))));
}

#[test]
fn test_HF_QUERY_003_012_get_dataset_caching() {
    let mut client = HubClient::new();
    let d1 = client.get_dataset("squad").unwrap();
    let d2 = client.get_dataset("squad").unwrap();
    assert_eq!(d1.id, d2.id);
}

#[test]
fn test_HF_QUERY_003_013_get_space_caching() {
    let mut client = HubClient::new();
    let s1 = client.get_space("gradio/chatbot").unwrap();
    let s2 = client.get_space("gradio/chatbot").unwrap();
    assert_eq!(s1.id, s2.id);
}

#[test]
fn test_HF_QUERY_003_014_get_model_bert() {
    let mut client = HubClient::new();
    let model = client.get_model("bert-base-uncased").unwrap();
    assert_eq!(model.id, "bert-base-uncased");
    assert_eq!(model.pipeline_tag, Some("fill-mask".to_string()));
    assert_eq!(model.downloads, 50_000_000);
}

#[test]
fn test_HF_QUERY_002_065_search_datasets_caching() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let r1 = client.search_datasets(&filters).unwrap();
    let r2 = client.search_datasets(&filters).unwrap();
    assert_eq!(r1.len(), r2.len());
}

#[test]
fn test_HF_QUERY_002_066_search_spaces_caching() {
    let mut client = HubClient::new();
    let filters = SearchFilters::new();
    let r1 = client.search_spaces(&filters).unwrap();
    let r2 = client.search_spaces(&filters).unwrap();
    assert_eq!(r1.len(), r2.len());
}

#[test]
fn test_HF_QUERY_002_067_cache_expired_search() {
    let mut cache = ResponseCache::new(Duration::from_millis(1));
    let asset = HubAsset::new("test", HubAssetType::Model);
    cache.cache_asset("key", asset);

    std::thread::sleep(Duration::from_millis(10));

    assert!(cache.get_asset("key").is_none());
}

#[test]
fn test_HF_QUERY_002_068_cache_clear_expired() {
    let mut cache = ResponseCache::new(Duration::from_millis(1));
    cache.cache_search("expired_key", vec![HubAsset::new("test", HubAssetType::Model)]);
    cache.cache_asset("expired_asset", HubAsset::new("test", HubAssetType::Model));

    std::thread::sleep(Duration::from_millis(10));

    cache.clear_expired();

    let stats = cache.stats();
    assert_eq!(stats.search_entries, 0, "Expired search entries should be cleared");
    assert_eq!(stats.asset_entries, 0, "Expired asset entries should be cleared");
}

#[test]
fn test_HF_QUERY_002_069_cache_stats() {
    let mut cache = ResponseCache::new(Duration::from_secs(300));
    cache.cache_search("s1", vec![]);
    cache.cache_search("s2", vec![]);
    cache.cache_asset("a1", HubAsset::new("a1", HubAssetType::Model));

    let stats = cache.stats();
    assert_eq!(stats.search_entries, 2);
    assert_eq!(stats.asset_entries, 1);
    assert_eq!(stats.ttl_secs, 300);
}

#[test]
fn test_HF_QUERY_002_070_error_display_rate_limited_no_retry() {
    let err = HubError::RateLimited { retry_after: None };
    let display = format!("{}", err);
    assert_eq!(display, "Rate limited");
}

#[test]
fn test_HF_QUERY_002_071_error_display_invalid_response() {
    let err = HubError::InvalidResponse("bad data".to_string());
    let display = format!("{}", err);
    assert!(display.contains("Invalid response"));
    assert!(display.contains("bad data"));
}

#[test]
fn test_HF_QUERY_002_072_error_is_error_trait() {
    let err = HubError::NotFound("test".to_string());
    // Verify it implements std::error::Error
    let error: &dyn std::error::Error = &err;
    assert!(!error.to_string().is_empty());
}

#[test]
fn test_HF_QUERY_002_073_hub_error_equality() {
    assert_eq!(HubError::OfflineMode, HubError::OfflineMode);
    assert_ne!(HubError::NotFound("a".to_string()), HubError::NotFound("b".to_string()));
    assert_eq!(HubError::NotFound("same".to_string()), HubError::NotFound("same".to_string()));
}

#[test]
fn test_HF_QUERY_002_074_hub_error_clone() {
    let err = HubError::RateLimited { retry_after: Some(30) };
    let cloned = err.clone();
    assert_eq!(err, cloned);
}

#[test]
fn test_HF_QUERY_002_075_client_with_base_url() {
    let client = HubClient::with_base_url("http://localhost:8080");
    assert_eq!(client.base_url, "http://localhost:8080");
    assert!(!client.offline_mode);
}

#[test]
fn test_HF_QUERY_002_076_client_default() {
    let client = HubClient::default();
    assert!(!client.offline_mode);
}

#[test]
fn test_HF_QUERY_002_077_filters_with_query() {
    let filters = SearchFilters::new().with_query("llama");
    assert_eq!(filters.query, Some("llama".to_string()));
}

#[test]
fn test_HF_QUERY_002_078_filters_with_license() {
    let filters = SearchFilters::new().with_license("apache-2.0");
    assert_eq!(filters.license, Some("apache-2.0".to_string()));
}

#[test]
fn test_HF_QUERY_002_079_hub_asset_type_serde_roundtrip() {
    for asset_type in &[HubAssetType::Model, HubAssetType::Dataset, HubAssetType::Space] {
        let json = serde_json::to_string(asset_type).unwrap();
        let deserialized: HubAssetType = serde_json::from_str(&json).unwrap();
        assert_eq!(asset_type, &deserialized);
    }
}

#[test]
fn test_HF_QUERY_002_080_hub_asset_deserialize() {
    let json = r#"{
        "id": "org/model",
        "asset_type": "model",
        "author": "org",
        "downloads": 100,
        "likes": 10,
        "tags": ["nlp"],
        "pipeline_tag": "text-generation",
        "library": "transformers",
        "license": "mit",
        "last_modified": "2024-01-01",
        "card_content": null
    }"#;
    let asset: HubAsset = serde_json::from_str(json).unwrap();
    assert_eq!(asset.id, "org/model");
    assert_eq!(asset.downloads, 100);
    assert_eq!(asset.tags, vec!["nlp"]);
}

#[test]
fn test_HF_QUERY_002_081_cache_entry_not_expired() {
    let entry = CacheEntry::new(42, Duration::from_secs(3600));
    assert!(!entry.is_expired());
}

#[test]
fn test_HF_QUERY_002_082_cache_entry_expired() {
    let entry = CacheEntry::new(42, Duration::from_millis(1));
    std::thread::sleep(Duration::from_millis(10));
    assert!(entry.is_expired());
}

#[test]
fn test_HF_QUERY_002_083_cache_overwrite() {
    let mut cache = ResponseCache::default_ttl();
    let results1 = vec![HubAsset::new("first", HubAssetType::Model)];
    let results2 = vec![
        HubAsset::new("second", HubAssetType::Model),
        HubAsset::new("third", HubAssetType::Model),
    ];

    cache.cache_search("key", results1);
    assert_eq!(cache.get_search("key").unwrap().len(), 1);

    cache.cache_search("key", results2);
    assert_eq!(cache.get_search("key").unwrap().len(), 2);
}

#[test]
fn test_HF_QUERY_002_084_cache_stats_serialize() {
    let stats = CacheStats { search_entries: 5, asset_entries: 3, ttl_secs: 900 };
    let json = serde_json::to_string(&stats).unwrap();
    assert!(json.contains("\"search_entries\":5"));
    assert!(json.contains("\"asset_entries\":3"));
    assert!(json.contains("\"ttl_secs\":900"));
}
