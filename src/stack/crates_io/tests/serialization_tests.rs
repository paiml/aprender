//! Serialization tests for crates_io module.

use super::helpers::{make_full_response, make_response};
use crate::stack::crates_io::types::{CrateData, CrateResponse, VersionData};
use crate::stack::crates_io::MockCratesIoClient;

// ============================================================================
// CRATES-011: Serialization roundtrip tests
// ============================================================================

#[test]
fn test_crates_011_crate_data_serialization() {
    let data = CrateData {
        max_stable_version: Some("1.2.3".to_string()),
        description: Some("A test crate for serialization".to_string()),
        downloads: 12345,
        ..CrateData::new("serialize-test", "1.2.3")
    };

    let json = serde_json::to_string(&data).unwrap();
    let deserialized: CrateData = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.name, data.name);
    assert_eq!(deserialized.max_version, data.max_version);
    assert_eq!(deserialized.max_stable_version, data.max_stable_version);
    assert_eq!(deserialized.description, data.description);
    assert_eq!(deserialized.downloads, data.downloads);
}

#[test]
fn test_crates_011_version_data_serialization() {
    let version = VersionData { yanked: true, ..VersionData::new("2.0.0-alpha.1", 5000) };

    let json = serde_json::to_string(&version).unwrap();
    let deserialized: VersionData = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.num, version.num);
    assert_eq!(deserialized.yanked, version.yanked);
    assert_eq!(deserialized.downloads, version.downloads);
}

#[test]
fn test_crates_011_full_response_serialization() {
    let mut response = make_full_response("full-test", "3.0.0", Some("Complete test"), 50000);
    response.versions = vec![
        VersionData::new("3.0.0", 30000),
        VersionData::new("2.0.0", 15000),
        VersionData { yanked: true, ..VersionData::new("1.0.0", 5000) },
    ];

    let json = serde_json::to_string_pretty(&response).unwrap();
    let deserialized: CrateResponse = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.krate.name, response.krate.name);
    assert_eq!(deserialized.versions.len(), 3);
    assert!(deserialized.versions[2].yanked);
}

// ============================================================================
// CRATES-012: Mock client edge cases
// ============================================================================

#[test]
fn test_crates_012_mock_invalid_version_parse() {
    let mut mock = MockCratesIoClient::new();
    // Add a crate response directly with an invalid version string
    mock.responses.insert(
        "invalid-version".to_string(),
        Ok(make_response("invalid-version", "not-a-version")),
    );

    let result = mock.get_latest_version("invalid-version");
    assert!(result.is_err());
}

#[test]
fn test_crates_012_mock_is_version_published_with_yanked() {
    let mut mock = MockCratesIoClient::new();
    let mut yanked_response = make_response("yanked-test", "2.0.0");
    yanked_response.versions = vec![
        VersionData::new("2.0.0", 0),
        VersionData { yanked: true, ..VersionData::new("1.0.0", 0) },
    ];
    mock.responses.insert("yanked-test".to_string(), Ok(yanked_response));

    // Yanked version should not be considered published
    let yanked = mock.is_version_published("yanked-test", &semver::Version::new(1, 0, 0)).unwrap();
    assert!(!yanked);

    // Non-yanked version should be published
    let published =
        mock.is_version_published("yanked-test", &semver::Version::new(2, 0, 0)).unwrap();
    assert!(published);
}

#[test]
fn test_crates_012_mock_empty_versions() {
    let mut mock = MockCratesIoClient::new();
    mock.responses.insert("no-versions".to_string(), Ok(make_response("no-versions", "1.0.0")));

    // No versions means version is not published
    let result = mock.is_version_published("no-versions", &semver::Version::new(1, 0, 0)).unwrap();
    assert!(!result);
}
