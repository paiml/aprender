//! Type definitions for crates.io API responses and caching.

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Cache entry with TTL
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub expires_at: Instant,
}

impl<T> CacheEntry<T> {
    pub fn new(value: T, ttl: Duration) -> Self {
        Self { value, expires_at: Instant::now() + ttl }
    }

    pub fn is_expired(&self) -> bool {
        Instant::now() > self.expires_at
    }
}

/// Persistent cache entry (stored on disk)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistentCacheEntry {
    /// Cached response data
    pub response: CrateResponse,
    /// Expiration timestamp (Unix epoch seconds)
    pub expires_at: u64,
}

impl PersistentCacheEntry {
    pub fn new(response: CrateResponse, ttl: Duration) -> Self {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        Self { response, expires_at: now + ttl.as_secs() }
    }

    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        now >= self.expires_at
    }
}

/// Response from crates.io API for a single crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateResponse {
    #[serde(rename = "crate")]
    pub krate: CrateData,
    pub versions: Vec<VersionData>,
}

/// Core crate data from crates.io
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateData {
    pub name: String,
    pub max_version: String,
    pub max_stable_version: Option<String>,
    pub description: Option<String>,
    pub downloads: u64,
    pub updated_at: String,
}

impl CrateData {
    /// Create a new CrateData with sensible defaults.
    ///
    /// Sets `max_stable_version` to None, `description` to None,
    /// `downloads` to 0, and `updated_at` to a standard timestamp.
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            max_version: version.into(),
            max_stable_version: None,
            description: None,
            downloads: 0,
            updated_at: "2025-12-05T00:00:00Z".to_string(),
        }
    }
}

/// Version data from crates.io
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionData {
    pub num: String,
    pub yanked: bool,
    pub downloads: u64,
    pub created_at: String,
}

impl VersionData {
    /// Create a new VersionData with sensible defaults (yanked=false, standard timestamp).
    pub fn new(num: impl Into<String>, downloads: u64) -> Self {
        Self {
            num: num.into(),
            yanked: false,
            downloads,
            created_at: "2025-12-05T00:00:00Z".to_string(),
        }
    }
}

/// Response from crates.io dependencies endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyResponse {
    pub dependencies: Vec<DependencyData>,
}

/// Individual dependency data from crates.io
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyData {
    pub crate_id: String,
    #[serde(rename = "req")]
    pub version_req: String,
    pub kind: String, // "normal", "dev", "build"
    pub optional: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_entry_new() {
        let entry = CacheEntry::new("test", Duration::from_secs(60));
        assert_eq!(entry.value, "test");
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_cache_entry_expired() {
        let entry = CacheEntry::new("test", Duration::from_secs(0));
        std::thread::sleep(Duration::from_millis(10));
        assert!(entry.is_expired());
    }

    #[test]
    fn test_persistent_cache_entry_new() {
        let response = CrateResponse { krate: CrateData::new("test", "1.0.0"), versions: vec![] };
        let entry = PersistentCacheEntry::new(response, Duration::from_secs(3600));
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_persistent_cache_entry_expired() {
        let response = CrateResponse { krate: CrateData::new("test", "1.0.0"), versions: vec![] };
        let mut entry = PersistentCacheEntry::new(response, Duration::from_secs(1));
        // Set expiration to the past
        entry.expires_at = 0;
        assert!(entry.is_expired());
    }

    #[test]
    fn test_crate_data_new() {
        let data = CrateData::new("trueno", "0.14.0");
        assert_eq!(data.name, "trueno");
        assert_eq!(data.max_version, "0.14.0");
        assert!(data.max_stable_version.is_none());
        assert!(data.description.is_none());
        assert_eq!(data.downloads, 0);
    }

    #[test]
    fn test_version_data_new() {
        let data = VersionData::new("1.0.0", 1000);
        assert_eq!(data.num, "1.0.0");
        assert_eq!(data.downloads, 1000);
        assert!(!data.yanked);
    }

    #[test]
    fn test_crate_response_serialization() {
        let response = CrateResponse {
            krate: CrateData::new("test", "1.0.0"),
            versions: vec![VersionData::new("1.0.0", 100)],
        };
        let json = serde_json::to_string(&response).expect("json serialize failed");
        let deserialized: CrateResponse =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized.krate.name, "test");
        assert_eq!(deserialized.versions.len(), 1);
    }

    #[test]
    fn test_dependency_response_serialization() {
        let response = DependencyResponse {
            dependencies: vec![DependencyData {
                crate_id: "serde".to_string(),
                version_req: "^1.0".to_string(),
                kind: "normal".to_string(),
                optional: false,
            }],
        };
        let json = serde_json::to_string(&response).expect("json serialize failed");
        let deserialized: DependencyResponse =
            serde_json::from_str(&json).expect("json deserialize failed");
        assert_eq!(deserialized.dependencies.len(), 1);
        assert_eq!(deserialized.dependencies[0].crate_id, "serde");
    }

    #[test]
    fn test_crate_data_with_description() {
        let mut data = CrateData::new("trueno", "0.14.0");
        data.description = Some("SIMD/GPU compute".to_string());
        assert_eq!(data.description.expect("unexpected failure"), "SIMD/GPU compute");
    }

    #[test]
    fn test_version_data_yanked() {
        let mut data = VersionData::new("0.9.0", 50);
        data.yanked = true;
        assert!(data.yanked);
    }

    #[test]
    fn test_dependency_data_optional() {
        let dep = DependencyData {
            crate_id: "feature".to_string(),
            version_req: "^2.0".to_string(),
            kind: "normal".to_string(),
            optional: true,
        };
        assert!(dep.optional);
        assert_eq!(dep.kind, "normal");
    }
}
