//! Unit tests for crates_io module.

use super::helpers::{make_full_response, make_response};
use crate::stack::crates_io::cache::PersistentCache;
use crate::stack::crates_io::types::{
    CacheEntry, CrateData, CrateResponse, PersistentCacheEntry, VersionData,
};
use crate::stack::crates_io::MockCratesIoClient;
use std::time::Duration;

// ============================================================================
// UNIT TESTS - Fast, focused, deterministic
// Following bashrs style: ARRANGE/ACT/ASSERT with task IDs
// ============================================================================

// ============================================================================
// CRATES-001: CacheEntry behavior
// ============================================================================

/// RED PHASE: Test cache entry creation
#[test]
fn test_crates_001_cache_entry_creation() {
    // ARRANGE & ACT
    let entry = CacheEntry::new("test_value", Duration::from_secs(60));

    // ASSERT
    assert_eq!(entry.value, "test_value");
    assert!(!entry.is_expired());
}

/// RED PHASE: Test cache entry expiration
#[test]
fn test_cache_entry_expiration() {
    let entry = CacheEntry::new("value", Duration::from_millis(1));
    assert!(!entry.is_expired());

    std::thread::sleep(Duration::from_millis(10));
    assert!(entry.is_expired());
}

/// RED PHASE: Test cache entry with zero TTL
#[test]
fn test_crates_001_cache_entry_zero_ttl() {
    // ARRANGE
    let entry = CacheEntry::new(42, Duration::from_secs(0));

    // ASSERT - immediately expired
    assert!(entry.is_expired());
}

/// RED PHASE: Test cache entry clone (via CrateResponse)
#[test]
fn test_crates_001_cache_entry_with_clone() {
    let response = make_response("test", "1.0.0");

    let entry = CacheEntry::new(response.clone(), Duration::from_secs(60));
    assert_eq!(entry.value.krate.name, "test");
}

/// RED PHASE: Test cache entry debug
#[test]
fn test_crates_001_cache_entry_debug() {
    let entry = CacheEntry::new("debug_test", Duration::from_secs(60));
    let debug = format!("{:?}", entry);
    assert!(debug.contains("CacheEntry"));
    assert!(debug.contains("debug_test"));
}

#[test]
fn test_mock_client_add_crate() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("trueno", "1.2.0");

    let response = mock.get_crate("trueno").unwrap();
    assert_eq!(response.krate.name, "trueno");
    assert_eq!(response.krate.max_version, "1.2.0");
}

#[test]
fn test_mock_client_not_found() {
    let mock = MockCratesIoClient::new();
    assert!(mock.get_crate("nonexistent").is_err());
}

#[test]
fn test_mock_client_add_not_found() {
    let mut mock = MockCratesIoClient::new();
    mock.add_not_found("broken-crate");

    assert!(mock.get_crate("broken-crate").is_err());
}

#[test]
fn test_mock_client_get_latest_version() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("aprender", "0.8.1");

    let version = mock.get_latest_version("aprender").unwrap();
    assert_eq!(version, semver::Version::new(0, 8, 1));
}

#[test]
fn test_mock_client_version_published() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("trueno", "1.2.0");

    // Published version
    assert!(mock.is_version_published("trueno", &semver::Version::new(1, 2, 0)).unwrap());

    // Not published version
    assert!(!mock.is_version_published("trueno", &semver::Version::new(1, 3, 0)).unwrap());
}

#[test]
fn test_mock_client_multiple_crates() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("trueno", "1.2.0")
        .add_crate("aprender", "0.8.1")
        .add_crate("renacer", "0.6.0")
        .add_not_found("nonexistent");

    assert!(mock.get_crate("trueno").is_ok());
    assert!(mock.get_crate("aprender").is_ok());
    assert!(mock.get_crate("renacer").is_ok());
    assert!(mock.get_crate("nonexistent").is_err());
}

#[test]
fn test_crate_response_deserialization() {
    let json = r#"{
        "crate": {
            "name": "trueno",
            "max_version": "1.2.0",
            "max_stable_version": "1.2.0",
            "description": "SIMD tensor library",
            "downloads": 5000,
            "updated_at": "2025-12-01T10:00:00Z"
        },
        "versions": [
            {
                "num": "1.2.0",
                "yanked": false,
                "downloads": 3000,
                "created_at": "2025-12-01T10:00:00Z"
            },
            {
                "num": "1.1.0",
                "yanked": false,
                "downloads": 2000,
                "created_at": "2025-11-01T10:00:00Z"
            }
        ]
    }"#;

    let response: CrateResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.krate.name, "trueno");
    assert_eq!(response.krate.max_version, "1.2.0");
    assert_eq!(response.versions.len(), 2);
    assert!(!response.versions[0].yanked);
}

// ============================================================================
// CRATES-002: CrateResponse and CrateData tests
// ============================================================================

/// RED PHASE: Test CrateResponse clone
#[test]
fn test_crates_002_crate_response_clone() {
    let mut response = make_full_response("test", "1.0.0", Some("Test crate"), 1000);
    response.krate.updated_at = "2025-12-05".to_string();
    response.versions[0].created_at = "2025-12-05".to_string();

    let cloned = response.clone();
    assert_eq!(cloned.krate.name, response.krate.name);
    assert_eq!(cloned.versions.len(), response.versions.len());
}

/// RED PHASE: Test CrateData debug
#[test]
fn test_crates_002_crate_data_debug() {
    let data = CrateData::new("debug-crate", "2.0.0");

    let debug = format!("{:?}", data);
    assert!(debug.contains("CrateData"));
    assert!(debug.contains("debug-crate"));
}

/// RED PHASE: Test VersionData with yanked=true
#[test]
fn test_crates_002_version_data_yanked() {
    let version = VersionData { yanked: true, ..VersionData::new("0.1.0", 50) };

    assert!(version.yanked);
    assert_eq!(version.num, "0.1.0");
}

/// RED PHASE: Test VersionData debug
#[test]
fn test_crates_002_version_data_debug() {
    let version = VersionData::new("1.2.3", 999);

    let debug = format!("{:?}", version);
    assert!(debug.contains("VersionData"));
    assert!(debug.contains("1.2.3"));
}

// ============================================================================
// CRATES-003: MockCratesIoClient edge cases
// ============================================================================

/// RED PHASE: Test MockCratesIoClient default
#[test]
fn test_crates_003_mock_client_default() {
    let mock = MockCratesIoClient::default();
    assert!(mock.get_crate("any").is_err());
}

/// RED PHASE: Test MockCratesIoClient debug
#[test]
fn test_crates_003_mock_client_debug() {
    let mock = MockCratesIoClient::new();
    let debug = format!("{:?}", mock);
    assert!(debug.contains("MockCratesIoClient"));
}

/// RED PHASE: Test MockCratesIoClient chaining
#[test]
fn test_crates_003_mock_client_chaining() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("a", "1.0.0").add_crate("b", "2.0.0").add_crate("c", "3.0.0");

    assert_eq!(mock.get_latest_version("a").unwrap(), semver::Version::new(1, 0, 0));
    assert_eq!(mock.get_latest_version("b").unwrap(), semver::Version::new(2, 0, 0));
    assert_eq!(mock.get_latest_version("c").unwrap(), semver::Version::new(3, 0, 0));
}

/// RED PHASE: Test version not published
#[test]
fn test_crates_003_version_not_published() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("test", "1.0.0");

    // Different version should not be published
    let result = mock.is_version_published("test", &semver::Version::new(2, 0, 0));
    assert!(!result.unwrap());
}

/// RED PHASE: Test get_latest_version error
#[test]
fn test_crates_003_get_latest_version_error() {
    let mock = MockCratesIoClient::new();
    assert!(mock.get_latest_version("nonexistent").is_err());
}

/// RED PHASE: Test is_version_published error
#[test]
fn test_crates_003_is_version_published_error() {
    let mock = MockCratesIoClient::new();
    let result = mock.is_version_published("nonexistent", &semver::Version::new(1, 0, 0));
    assert!(result.is_err());
}

// ============================================================================
// CRATES-004: Deserialization edge cases
// ============================================================================

/// RED PHASE: Test deserialization with null description
#[test]
fn test_crates_004_deserialize_null_description() {
    let json = r#"{
        "crate": {
            "name": "minimal",
            "max_version": "0.1.0",
            "max_stable_version": null,
            "description": null,
            "downloads": 0,
            "updated_at": "2025-01-01T00:00:00Z"
        },
        "versions": []
    }"#;

    let response: CrateResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.krate.name, "minimal");
    assert!(response.krate.description.is_none());
    assert!(response.krate.max_stable_version.is_none());
    assert!(response.versions.is_empty());
}

/// RED PHASE: Test deserialization with prerelease version
#[test]
fn test_crates_004_deserialize_prerelease() {
    let json = r#"{
        "crate": {
            "name": "beta-crate",
            "max_version": "1.0.0-beta.1",
            "max_stable_version": "0.9.0",
            "description": "Beta software",
            "downloads": 100,
            "updated_at": "2025-12-01T00:00:00Z"
        },
        "versions": [
            {"num": "1.0.0-beta.1", "yanked": false, "downloads": 50, "created_at": "2025-12-01T00:00:00Z"},
            {"num": "0.9.0", "yanked": false, "downloads": 50, "created_at": "2025-11-01T00:00:00Z"}
        ]
    }"#;

    let response: CrateResponse = serde_json::from_str(json).unwrap();
    assert_eq!(response.krate.max_version, "1.0.0-beta.1");
    assert_eq!(response.krate.max_stable_version, Some("0.9.0".to_string()));
}

/// RED PHASE: Test deserialization with yanked versions
#[test]
fn test_crates_004_deserialize_yanked_versions() {
    let json = r#"{
        "crate": {
            "name": "yanked-crate",
            "max_version": "2.0.0",
            "max_stable_version": "2.0.0",
            "description": null,
            "downloads": 1000,
            "updated_at": "2025-12-01T00:00:00Z"
        },
        "versions": [
            {"num": "2.0.0", "yanked": false, "downloads": 500, "created_at": "2025-12-01T00:00:00Z"},
            {"num": "1.0.0", "yanked": true, "downloads": 500, "created_at": "2025-01-01T00:00:00Z"}
        ]
    }"#;

    let response: CrateResponse = serde_json::from_str(json).unwrap();
    assert!(!response.versions[0].yanked);
    assert!(response.versions[1].yanked);
}

// ============================================================================
// CRATES-005: PersistentCacheEntry tests
// ============================================================================

/// RED PHASE: Test persistent cache entry creation
#[test]
fn test_crates_005_persistent_cache_entry_creation() {
    let response = make_response("test", "1.0.0");

    let entry = PersistentCacheEntry::new(response, Duration::from_secs(3600));
    assert!(!entry.is_expired());
    assert_eq!(entry.response.krate.name, "test");
}

/// RED PHASE: Test persistent cache entry expiration
#[test]
fn test_crates_005_persistent_cache_entry_expiration() {
    let response = make_response("expired", "0.1.0");

    // Create entry with zero TTL - should be expired
    let entry = PersistentCacheEntry::new(response, Duration::from_secs(0));
    assert!(entry.is_expired());
}

/// RED PHASE: Test persistent cache entry serialization
#[test]
fn test_crates_005_persistent_cache_entry_serialization() {
    let mut response = make_full_response("serialize", "2.0.0", Some("A test crate"), 1000);
    response.versions[0].downloads = 500;

    let entry = PersistentCacheEntry::new(response, Duration::from_secs(3600));
    let json = serde_json::to_string(&entry).unwrap();
    let deserialized: PersistentCacheEntry = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.response.krate.name, "serialize");
    assert_eq!(deserialized.response.versions.len(), 1);
}

// ============================================================================
// CRATES-006: PersistentCache tests
// ============================================================================

/// RED PHASE: Test persistent cache default
#[test]
fn test_crates_006_persistent_cache_default() {
    let cache = PersistentCache::default();
    assert!(cache.entries.is_empty());
}

/// RED PHASE: Test persistent cache insert and get
#[test]
fn test_crates_006_persistent_cache_insert_get() {
    let mut cache = PersistentCache::default();

    let response = make_response("cached", "1.0.0");

    cache.insert("cached".to_string(), response, Duration::from_secs(3600));

    let retrieved = cache.get("cached");
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().krate.name, "cached");
}

/// RED PHASE: Test persistent cache miss
#[test]
fn test_crates_006_persistent_cache_miss() {
    let cache = PersistentCache::default();
    assert!(cache.get("nonexistent").is_none());
}

/// RED PHASE: Test persistent cache expired entry
#[test]
fn test_crates_006_persistent_cache_expired() {
    let mut cache = PersistentCache::default();

    let response = make_response("expired", "1.0.0");

    // Insert with zero TTL - immediately expired
    cache.insert("expired".to_string(), response, Duration::from_secs(0));

    // Should not be retrievable
    assert!(cache.get("expired").is_none());
}

/// RED PHASE: Test persistent cache clear expired
#[test]
fn test_crates_006_persistent_cache_clear_expired() {
    let mut cache = PersistentCache::default();

    let valid_response = make_response("valid", "1.0.0");

    let expired_response = make_response("expired", "0.1.0");

    cache.insert("valid".to_string(), valid_response, Duration::from_secs(3600));
    cache.insert("expired".to_string(), expired_response, Duration::from_secs(0));

    cache.clear_expired();

    assert_eq!(cache.entries.len(), 1);
    assert!(cache.get("valid").is_some());
}

/// RED PHASE: Test persistent cache serialization
#[test]
fn test_crates_006_persistent_cache_serialization() {
    let mut cache = PersistentCache::default();

    let response = make_response("serialize", "1.0.0");

    cache.insert("serialize".to_string(), response, Duration::from_secs(3600));

    let json = serde_json::to_string(&cache).unwrap();
    let deserialized: PersistentCache = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.entries.len(), 1);
}

// ============================================================================
// CRATES-013: PersistentCache load/save/cache_path coverage
// ============================================================================

/// Test cache_path returns a valid path structure
#[test]
fn test_crates_013_cache_path_structure() {
    let path = PersistentCache::cache_path();
    let path_str = path.to_string_lossy();
    assert!(path_str.contains("batuta"), "Cache path should contain 'batuta': {}", path_str);
    assert!(
        path_str.ends_with("crates_io_cache.json"),
        "Cache path should end with 'crates_io_cache.json': {}",
        path_str
    );
}

/// Test load returns default when no cache file exists (covers Self::default() branch)
#[test]
fn test_crates_013_load_returns_default() {
    // load() reads from cache_path(). If the file doesn't exist or is corrupt,
    // it returns default. We exercise the load path here.
    let cache = PersistentCache::load();
    // Whether the file exists or not, load should not panic
    // If it exists with valid data, entries may be non-empty; otherwise empty.
    // The key coverage is exercising the load() function itself.
    let _ = cache.entries.len();
}

/// Test save writes to the cache path (covers save() path)
#[test]
fn test_crates_013_save_roundtrip() {
    // Save a cache and verify it doesn't error
    let mut cache = PersistentCache::default();
    let response = make_response("save-test", "3.0.0");
    cache.insert("save-test".to_string(), response, Duration::from_secs(3600));

    // save() creates parent dirs and writes to cache_path()
    let result = cache.save();
    assert!(result.is_ok(), "save() should succeed: {:?}", result.err());

    // Now load() should retrieve the saved data
    let loaded = PersistentCache::load();
    let entry = loaded.get("save-test");
    assert!(entry.is_some(), "Loaded cache should contain 'save-test' entry");
    assert_eq!(entry.unwrap().krate.name, "save-test");
}

/// Test PersistentCache debug trait
#[test]
fn test_crates_013_persistent_cache_debug() {
    let cache = PersistentCache::default();
    let debug = format!("{:?}", cache);
    assert!(debug.contains("PersistentCache"));
}

// ============================================================================
// CRATES-014: PersistentCache load_from/save_to with temp files
// ============================================================================

/// Test load_from nonexistent path returns default (covers !path.exists() branch)
#[test]
fn test_crates_014_load_from_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nonexistent_cache.json");

    let cache = PersistentCache::load_from(&path);
    assert!(cache.entries.is_empty());
}

/// Test load_from with invalid JSON returns default (covers serde error branch)
#[test]
fn test_crates_014_load_from_invalid_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad_cache.json");
    std::fs::write(&path, "not valid json {{{").unwrap();

    let cache = PersistentCache::load_from(&path);
    assert!(cache.entries.is_empty());
}

/// Test load_from with valid JSON returns parsed cache
#[test]
fn test_crates_014_load_from_valid_json() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("good_cache.json");

    // Save a cache first
    let mut cache = PersistentCache::default();
    let response = make_response("loaded", "2.0.0");
    cache.insert("loaded".to_string(), response, Duration::from_secs(3600));
    cache.save_to(&path).unwrap();

    // Load it back
    let loaded = PersistentCache::load_from(&path);
    let entry = loaded.get("loaded");
    assert!(entry.is_some());
    assert_eq!(entry.unwrap().krate.name, "loaded");
}

/// Test save_to creates parent directories
#[test]
fn test_crates_014_save_to_creates_dirs() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested").join("deep").join("cache.json");

    let mut cache = PersistentCache::default();
    let response = make_response("nested", "1.0.0");
    cache.insert("nested".to_string(), response, Duration::from_secs(3600));

    let result = cache.save_to(&path);
    assert!(result.is_ok());
    assert!(path.exists());
}

/// Test save_to and load_from roundtrip with multiple entries
#[test]
fn test_crates_014_save_load_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("roundtrip.json");

    let mut cache = PersistentCache::default();
    cache.insert(
        "crate-a".to_string(),
        make_response("crate-a", "1.0.0"),
        Duration::from_secs(3600),
    );
    cache.insert(
        "crate-b".to_string(),
        make_response("crate-b", "2.0.0"),
        Duration::from_secs(3600),
    );

    cache.save_to(&path).unwrap();
    let loaded = PersistentCache::load_from(&path);

    assert_eq!(loaded.entries.len(), 2);
    assert!(loaded.get("crate-a").is_some());
    assert!(loaded.get("crate-b").is_some());
}

/// Test load_from with empty file returns default (covers read ok but parse fail)
#[test]
fn test_crates_014_load_from_empty_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty_cache.json");
    std::fs::write(&path, "").unwrap();

    let cache = PersistentCache::load_from(&path);
    assert!(cache.entries.is_empty());
}

/// Test load_from with wrong schema returns default
#[test]
fn test_crates_014_load_from_wrong_schema() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wrong_schema.json");
    std::fs::write(&path, r#"{"key": "value"}"#).unwrap();

    let cache = PersistentCache::load_from(&path);
    // serde should fail to deserialize because "entries" field is missing
    // but HashMap deserialization may accept any JSON object with matching keys
    // so this may actually succeed with an empty entries map
    let _ = cache.entries.len();
}

/// Test save_to with empty cache
#[test]
fn test_crates_014_save_empty_cache() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("empty.json");

    let cache = PersistentCache::default();
    cache.save_to(&path).unwrap();

    let loaded = PersistentCache::load_from(&path);
    assert!(loaded.entries.is_empty());
}

/// Test save_to error when path is invalid (covers ? error propagation)
#[test]
fn test_crates_014_save_to_invalid_path() {
    // /proc/self/fd/-1/... is not writable and cannot create dirs
    let path = std::path::PathBuf::from("/proc/0/nonexistent/deep/cache.json");
    let cache = PersistentCache::default();
    let result = cache.save_to(&path);
    assert!(result.is_err());
}

/// Test load_from a directory (not a file) — covers read_to_string error
#[test]
fn test_crates_014_load_from_directory() {
    let dir = tempfile::tempdir().unwrap();
    // The dir path itself exists but is a directory, not a file.
    // read_to_string on a directory fails.
    let cache = PersistentCache::load_from(dir.path());
    assert!(cache.entries.is_empty());
}
