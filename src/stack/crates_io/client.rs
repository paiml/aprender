//! Crates.io API client implementation.

use super::cache::PersistentCache;
use super::types::{CacheEntry, CrateResponse, DependencyData, DependencyResponse};
use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::time::Duration;

/// Client for interacting with crates.io API
#[derive(Debug)]
pub struct CratesIoClient {
    /// HTTP client
    #[cfg(feature = "native")]
    client: reqwest::Client,

    /// In-memory cache for crate info (15 minute TTL)
    pub cache: HashMap<String, CacheEntry<CrateResponse>>,

    /// Persistent cache for offline mode
    pub persistent_cache: Option<PersistentCache>,

    /// Cache TTL
    pub cache_ttl: Duration,

    /// Offline mode - only use cached data
    offline: bool,
}

impl CratesIoClient {
    /// Create a new crates.io client
    #[cfg(feature = "native")]
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .user_agent("batuta/0.1 (https://github.com/paiml/batuta)")
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");

        Self {
            client,
            cache: HashMap::new(),
            persistent_cache: None,
            cache_ttl: Duration::from_secs(15 * 60), // 15 minutes
            offline: false,
        }
    }

    /// Create a client with custom TTL
    #[cfg(feature = "native")]
    pub fn with_cache_ttl(mut self, ttl: Duration) -> Self {
        self.cache_ttl = ttl;
        self
    }

    /// Enable persistent file-based cache
    #[cfg(feature = "native")]
    pub fn with_persistent_cache(mut self) -> Self {
        self.persistent_cache = Some(PersistentCache::load());
        self
    }

    /// Set offline mode (only use cached data)
    #[cfg(feature = "native")]
    pub fn set_offline(&mut self, offline: bool) {
        self.offline = offline;
    }

    /// Check if client is in offline mode
    #[cfg(feature = "native")]
    pub fn is_offline(&self) -> bool {
        self.offline
    }

    /// HTTP GET + status check + JSON parse helper.
    ///
    /// Performs a GET request to `url`, checks for 404 / non-success status,
    /// and deserialises the response body into `T`.
    #[cfg(feature = "native")]
    async fn fetch_and_parse<T: DeserializeOwned>(&self, url: &str, context: &str) -> Result<T> {
        let response = self
            .client
            .get(url)
            .send()
            .await
            .map_err(|e| anyhow!("Failed to fetch {}: {}", context, e))?;

        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(anyhow!("'{}' not found on crates.io", context));
        }

        if !response.status().is_success() {
            return Err(anyhow!("Failed to fetch {}: HTTP {}", context, response.status()));
        }

        response.json().await.map_err(|e| anyhow!("Failed to parse {} response: {}", context, e))
    }

    /// Get crate info from crates.io (cached)
    #[cfg(feature = "native")]
    pub async fn get_crate(&mut self, name: &str) -> Result<CrateResponse> {
        // Check in-memory cache first
        if let Some(entry) = self.cache.get(name) {
            if !entry.is_expired() {
                return Ok(entry.value.clone());
            }
        }

        // Check persistent cache
        if let Some(ref persistent) = self.persistent_cache {
            if let Some(response) = persistent.get(name) {
                // Also add to in-memory cache for faster subsequent access
                self.cache
                    .insert(name.to_string(), CacheEntry::new(response.clone(), self.cache_ttl));
                return Ok(response.clone());
            }
        }

        // In offline mode, return error if not in cache
        if self.offline {
            return Err(anyhow!("Crate '{}' not found in cache (offline mode)", name));
        }

        // Fetch from API
        let url = format!("https://crates.io/api/v1/crates/{}", name);
        let crate_response: CrateResponse =
            self.fetch_and_parse(&url, &format!("crate {}", name)).await?;

        // Cache the result in memory
        self.cache
            .insert(name.to_string(), CacheEntry::new(crate_response.clone(), self.cache_ttl));

        // Also save to persistent cache
        if let Some(ref mut persistent) = self.persistent_cache {
            persistent.insert(name.to_string(), crate_response.clone(), self.cache_ttl);
            let _ = persistent.save(); // Ignore save errors
        }

        Ok(crate_response)
    }

    /// Get the latest version of a crate
    #[cfg(feature = "native")]
    pub async fn get_latest_version(&mut self, name: &str) -> Result<semver::Version> {
        let response = self.get_crate(name).await?;
        response.krate.max_version.parse().map_err(|e| anyhow!("Failed to parse version: {}", e))
    }

    /// Check if a specific version is published
    #[cfg(feature = "native")]
    pub async fn is_version_published(
        &mut self,
        name: &str,
        version: &semver::Version,
    ) -> Result<bool> {
        let response = self.get_crate(name).await?;
        let version_str = version.to_string();

        Ok(response.versions.iter().any(|v| v.num == version_str && !v.yanked))
    }

    /// Check if a crate exists on crates.io
    #[cfg(feature = "native")]
    pub async fn crate_exists(&mut self, name: &str) -> bool {
        self.get_crate(name).await.is_ok()
    }

    /// Get all published versions of a crate (non-yanked)
    #[cfg(feature = "native")]
    pub async fn get_versions(&mut self, name: &str) -> Result<Vec<semver::Version>> {
        let response = self.get_crate(name).await?;

        let mut versions: Vec<semver::Version> = response
            .versions
            .iter()
            .filter(|v| !v.yanked)
            .filter_map(|v| v.num.parse().ok())
            .collect();

        versions.sort();
        versions.reverse(); // Newest first

        Ok(versions)
    }

    /// Get dependencies for a specific crate version from crates.io
    ///
    /// Fetches from `/api/v1/crates/{name}/{version}/dependencies`
    #[cfg(feature = "native")]
    pub async fn get_dependencies(
        &mut self,
        name: &str,
        version: &str,
    ) -> Result<Vec<DependencyData>> {
        // In offline mode, return error
        if self.offline {
            return Err(anyhow!(
                "Cannot fetch dependencies for {}@{} (offline mode)",
                name,
                version
            ));
        }

        let url = format!("https://crates.io/api/v1/crates/{}/{}/dependencies", name, version);
        let context = format!("dependencies for {}@{}", name, version);

        let dep_response: DependencyResponse = self.fetch_and_parse(&url, &context).await?;

        Ok(dep_response.dependencies)
    }

    /// Verify a crate is available on crates.io (post-publish check)
    #[cfg(feature = "native")]
    pub async fn verify_available(&mut self, name: &str, version: &semver::Version) -> Result<()> {
        // Clear cache to get fresh data
        self.cache.remove(name);

        let max_attempts = 10;
        let delay = Duration::from_secs(3);

        for attempt in 1..=max_attempts {
            if self.is_version_published(name, version).await? {
                return Ok(());
            }

            if attempt < max_attempts {
                tokio::time::sleep(delay).await;
            }
        }

        Err(anyhow!(
            "Crate {}@{} not available on crates.io after {} attempts",
            name,
            version,
            max_attempts
        ))
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Clear expired cache entries
    pub fn clear_expired(&mut self) {
        self.cache.retain(|_, entry| !entry.is_expired());
    }
}

#[cfg(feature = "native")]
impl Default for CratesIoClient {
    fn default() -> Self {
        Self::new()
    }
}
