//! Tests for CratesIoClient.

use super::helpers::make_response;
use crate::stack::crates_io::types::CacheEntry;
use crate::stack::crates_io::MockCratesIoClient;
use std::time::Duration;

// ============================================================================
// CRATES-007: CratesIoClient tests
// ============================================================================

#[test]
#[cfg(feature = "native")]
fn test_crates_007_client_default() {
    use crate::stack::crates_io::client::CratesIoClient;
    let client = CratesIoClient::default();
    assert!(client.cache.is_empty());
}

#[test]
#[cfg(feature = "native")]
fn test_crates_007_client_clear_cache() {
    use crate::stack::crates_io::client::CratesIoClient;
    let mut client = CratesIoClient::new();

    // Insert something into cache
    let response = make_response("test", "1.0.0");
    client.cache.insert("test".to_string(), CacheEntry::new(response, Duration::from_secs(3600)));

    assert!(!client.cache.is_empty());
    client.clear_cache();
    assert!(client.cache.is_empty());
}

#[test]
#[cfg(feature = "native")]
fn test_crates_007_client_clear_expired() {
    use crate::stack::crates_io::client::CratesIoClient;
    let mut client = CratesIoClient::new();

    // Insert valid entry
    let valid_response = make_response("valid", "1.0.0");
    client
        .cache
        .insert("valid".to_string(), CacheEntry::new(valid_response, Duration::from_secs(3600)));

    // Insert expired entry
    let expired_response = make_response("expired", "0.1.0");
    client
        .cache
        .insert("expired".to_string(), CacheEntry::new(expired_response, Duration::from_secs(0)));

    assert_eq!(client.cache.len(), 2);
    client.clear_expired();
    assert_eq!(client.cache.len(), 1);
    assert!(client.cache.contains_key("valid"));
}

// ============================================================================
// CRATES-008: MockCratesIoClient tests
// ============================================================================

#[test]
fn test_crates_008_mock_client_default() {
    let mock = MockCratesIoClient::default();
    assert!(mock.responses.is_empty());
}

#[test]
fn test_crates_008_mock_client_new() {
    let mock = MockCratesIoClient::new();
    assert!(mock.responses.is_empty());
}

#[test]
fn test_crates_008_mock_client_add_crate() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("test", "1.0.0");

    assert!(mock.responses.contains_key("test"));
}

#[test]
fn test_crates_008_mock_client_add_not_found() {
    let mut mock = MockCratesIoClient::new();
    mock.add_not_found("broken");

    assert!(mock.responses.contains_key("broken"));
    let response = mock.responses.get("broken").unwrap();
    assert!(response.is_err());
}

#[test]
fn test_crates_008_mock_client_get_crate_success() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("trueno", "0.8.0");

    let result = mock.get_crate("trueno");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().krate.max_version, "0.8.0");
}

#[test]
fn test_crates_008_mock_client_get_crate_error() {
    let mut mock = MockCratesIoClient::new();
    mock.add_not_found("notfound");

    let result = mock.get_crate("notfound");
    assert!(result.is_err());
}

#[test]
fn test_crates_008_mock_client_get_crate_not_configured() {
    let mock = MockCratesIoClient::new();
    let result = mock.get_crate("unknown");
    assert!(result.is_err());
}

#[test]
fn test_crates_008_mock_client_get_latest_version() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("aprender", "0.17.0");

    let version = mock.get_latest_version("aprender").unwrap();
    assert_eq!(version, semver::Version::new(0, 17, 0));
}

#[test]
fn test_crates_008_mock_client_is_version_published() {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("realizar", "0.2.3");

    let published = mock.is_version_published("realizar", &semver::Version::new(0, 2, 3)).unwrap();
    assert!(published);

    let not_published =
        mock.is_version_published("realizar", &semver::Version::new(0, 3, 0)).unwrap();
    assert!(!not_published);
}

// ============================================================================
// CRATES-009: VersionData tests
// ============================================================================

#[test]
fn test_crates_009_version_data_debug() {
    use crate::stack::crates_io::types::VersionData;
    let version = VersionData::new("1.0.0", 1000);
    let debug = format!("{:?}", version);
    assert!(debug.contains("1.0.0"));
}

#[test]
fn test_crates_009_version_data_clone() {
    use crate::stack::crates_io::types::VersionData;
    let version = VersionData { yanked: true, ..VersionData::new("1.0.0", 500) };
    let cloned = version.clone();
    assert_eq!(cloned.num, version.num);
    assert_eq!(cloned.yanked, version.yanked);
}

// ============================================================================
// CRATES-010: CrateData tests
// ============================================================================

#[test]
fn test_crates_010_crate_data_with_all_fields() {
    use crate::stack::crates_io::types::CrateData;
    let data = CrateData {
        max_stable_version: Some("2.0.0".to_string()),
        description: Some("A test crate".to_string()),
        downloads: 10000,
        ..CrateData::new("test", "2.0.0")
    };
    let debug = format!("{:?}", data);
    assert!(debug.contains("test"));
    assert!(debug.contains("2.0.0"));
}
