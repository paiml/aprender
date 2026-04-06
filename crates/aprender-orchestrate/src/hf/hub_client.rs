//! HuggingFace Hub API Client
//!
//! Implements HF-QUERY-002 (Hub Search) and HF-QUERY-003 (Asset Metadata)
//!
//! Provides live queries to HuggingFace Hub API:
//! - Model search with filters
//! - Dataset search with filters
//! - Space search with filters
//! - Asset metadata retrieval
//!
//! ## Observability (HF-OBS-001, HF-OBS-002)
//!
//! All Hub operations are instrumented with tracing spans:
//! - `hf.search.models` - Model search operations
//! - `hf.search.datasets` - Dataset search operations
//! - `hf.search.spaces` - Space search operations
//! - `hf.get.model` - Model metadata retrieval
//! - `hf.get.dataset` - Dataset metadata retrieval
//! - `hf.get.space` - Space metadata retrieval

// Allow dead_code for now - these types are tested and will be used
// once live Hub API integration is implemented (HUB-API milestone)
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tracing::{debug, info, instrument, warn};

// ============================================================================
// HF-QUERY-002: Hub Asset Types
// ============================================================================

/// Type of Hub asset
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HubAssetType {
    Model,
    Dataset,
    Space,
}

impl std::fmt::Display for HubAssetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model => write!(f, "model"),
            Self::Dataset => write!(f, "dataset"),
            Self::Space => write!(f, "space"),
        }
    }
}

/// Hub asset metadata (model, dataset, or space)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubAsset {
    /// Asset ID (e.g., "meta-llama/Llama-2-7b-hf")
    pub id: String,
    /// Asset type
    pub asset_type: HubAssetType,
    /// Author/organization
    pub author: String,
    /// Downloads count
    pub downloads: u64,
    /// Likes count
    pub likes: u64,
    /// Tags
    pub tags: Vec<String>,
    /// Pipeline tag (task) - for models
    pub pipeline_tag: Option<String>,
    /// Library (transformers, diffusers, etc.) - for models
    pub library: Option<String>,
    /// License
    pub license: Option<String>,
    /// Last modified timestamp
    pub last_modified: String,
    /// Model card/README content (optional, fetched separately)
    pub card_content: Option<String>,
}

impl HubAsset {
    pub fn new(id: impl Into<String>, asset_type: HubAssetType) -> Self {
        let id_str = id.into();
        let author = id_str.split('/').next().unwrap_or("unknown").to_string();
        Self {
            id: id_str,
            asset_type,
            author,
            downloads: 0,
            likes: 0,
            tags: Vec::new(),
            pipeline_tag: None,
            library: None,
            license: None,
            last_modified: String::new(),
            card_content: None,
        }
    }

    pub fn with_downloads(mut self, downloads: u64) -> Self {
        self.downloads = downloads;
        self
    }

    pub fn with_likes(mut self, likes: u64) -> Self {
        self.likes = likes;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_pipeline_tag(mut self, tag: impl Into<String>) -> Self {
        self.pipeline_tag = Some(tag.into());
        self
    }

    pub fn with_library(mut self, library: impl Into<String>) -> Self {
        self.library = Some(library.into());
        self
    }

    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }
}

// ============================================================================
// HF-QUERY-002: Search Filters
// ============================================================================

/// Search filters for Hub queries
#[derive(Debug, Clone, Default)]
pub struct SearchFilters {
    /// Filter by task (pipeline_tag)
    pub task: Option<String>,
    /// Filter by library
    pub library: Option<String>,
    /// Filter by author/organization
    pub author: Option<String>,
    /// Filter by license
    pub license: Option<String>,
    /// Minimum downloads threshold
    pub min_downloads: Option<u64>,
    /// Minimum likes threshold
    pub min_likes: Option<u64>,
    /// Search query text
    pub query: Option<String>,
    /// Maximum results to return
    pub limit: usize,
    /// Sort field
    pub sort: Option<String>,
    /// Sort direction (asc/desc)
    pub sort_direction: Option<String>,
}

impl SearchFilters {
    pub fn new() -> Self {
        Self { limit: 20, ..Default::default() }
    }

    pub fn with_task(mut self, task: impl Into<String>) -> Self {
        self.task = Some(task.into());
        self
    }

    pub fn with_library(mut self, library: impl Into<String>) -> Self {
        self.library = Some(library.into());
        self
    }

    pub fn with_author(mut self, author: impl Into<String>) -> Self {
        self.author = Some(author.into());
        self
    }

    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    pub fn with_min_downloads(mut self, min: u64) -> Self {
        self.min_downloads = Some(min);
        self
    }

    pub fn with_min_likes(mut self, min: u64) -> Self {
        self.min_likes = Some(min);
        self
    }

    pub fn with_query(mut self, query: impl Into<String>) -> Self {
        self.query = Some(query.into());
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    pub fn with_sort(mut self, field: impl Into<String>, direction: impl Into<String>) -> Self {
        self.sort = Some(field.into());
        self.sort_direction = Some(direction.into());
        self
    }
}

// ============================================================================
// HF-QUERY-002/003: Response Cache
// ============================================================================

/// Cache entry with TTL
#[derive(Debug, Clone)]
struct CacheEntry<T> {
    data: T,
    created: Instant,
    ttl: Duration,
}

impl<T> CacheEntry<T> {
    fn new(data: T, ttl: Duration) -> Self {
        Self { data, created: Instant::now(), ttl }
    }

    fn is_expired(&self) -> bool {
        self.created.elapsed() > self.ttl
    }
}

/// Response cache for Hub queries
#[derive(Debug, Default)]
pub struct ResponseCache {
    search_cache: HashMap<String, CacheEntry<Vec<HubAsset>>>,
    asset_cache: HashMap<String, CacheEntry<HubAsset>>,
    ttl: Duration,
}

impl ResponseCache {
    pub fn new(ttl: Duration) -> Self {
        Self { search_cache: HashMap::new(), asset_cache: HashMap::new(), ttl }
    }

    /// Default cache with 15 minute TTL
    pub fn default_ttl() -> Self {
        Self::new(Duration::from_secs(15 * 60))
    }

    /// Cache a search result
    pub fn cache_search(&mut self, key: &str, results: Vec<HubAsset>) {
        self.search_cache.insert(key.to_string(), CacheEntry::new(results, self.ttl));
    }

    /// Get cached search result
    pub fn get_search(&self, key: &str) -> Option<&Vec<HubAsset>> {
        self.search_cache.get(key).and_then(|entry| {
            if entry.is_expired() {
                None
            } else {
                Some(&entry.data)
            }
        })
    }

    /// Cache an asset
    pub fn cache_asset(&mut self, id: &str, asset: HubAsset) {
        self.asset_cache.insert(id.to_string(), CacheEntry::new(asset, self.ttl));
    }

    /// Get cached asset
    pub fn get_asset(&self, id: &str) -> Option<&HubAsset> {
        self.asset_cache.get(id).and_then(
            |entry| {
                if entry.is_expired() {
                    None
                } else {
                    Some(&entry.data)
                }
            },
        )
    }

    /// Clear expired entries
    pub fn clear_expired(&mut self) {
        self.search_cache.retain(|_, entry| !entry.is_expired());
        self.asset_cache.retain(|_, entry| !entry.is_expired());
    }

    /// Clear all cache
    pub fn clear(&mut self) {
        self.search_cache.clear();
        self.asset_cache.clear();
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            search_entries: self.search_cache.len(),
            asset_entries: self.asset_cache.len(),
            ttl_secs: self.ttl.as_secs(),
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize)]
pub struct CacheStats {
    pub search_entries: usize,
    pub asset_entries: usize,
    pub ttl_secs: u64,
}

// ============================================================================
// HF-QUERY-002/003: Hub Client
// ============================================================================

/// HuggingFace Hub API client
#[derive(Debug)]
pub struct HubClient {
    base_url: String,
    cache: ResponseCache,
    offline_mode: bool,
}

impl HubClient {
    /// Create new client with default settings
    pub fn new() -> Self {
        Self {
            base_url: "https://huggingface.co/api".to_string(),
            cache: ResponseCache::default_ttl(),
            offline_mode: false,
        }
    }

    /// Create client with custom base URL (for testing)
    pub fn with_base_url(base_url: impl Into<String>) -> Self {
        Self { base_url: base_url.into(), cache: ResponseCache::default_ttl(), offline_mode: false }
    }

    /// Enable offline mode (only return cached data)
    pub fn offline(mut self) -> Self {
        self.offline_mode = true;
        self
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    // ========================================================================
    // HF-QUERY-002: Search Methods (HF-OBS-001: Instrumented with tracing)
    // ========================================================================

    /// Search models on HuggingFace Hub
    #[instrument(name = "hf.search.models", skip(self), fields(
        task = filters.task.as_deref(),
        limit = filters.limit,
        cache_hit = tracing::field::Empty,
        result_count = tracing::field::Empty
    ))]
    pub fn search_models(&mut self, filters: &SearchFilters) -> Result<Vec<HubAsset>, HubError> {
        let cache_key = format!("models:{:?}", filters);

        // Check cache first
        if let Some(cached) = self.cache.get_search(&cache_key) {
            debug!(cache_hit = true, "Model search cache hit");
            tracing::Span::current().record("cache_hit", true);
            tracing::Span::current().record("result_count", cached.len());
            return Ok(cached.clone());
        }

        if self.offline_mode {
            warn!("Model search attempted in offline mode");
            return Err(HubError::OfflineMode);
        }

        // In a real implementation, this would make an HTTP request
        // For now, return mock data for testing
        let results = self.mock_model_search(filters);
        self.cache.cache_search(&cache_key, results.clone());
        info!(result_count = results.len(), "Model search completed");
        tracing::Span::current().record("cache_hit", false);
        tracing::Span::current().record("result_count", results.len());
        Ok(results)
    }

    /// Search datasets on HuggingFace Hub
    #[instrument(name = "hf.search.datasets", skip(self), fields(
        limit = filters.limit,
        cache_hit = tracing::field::Empty,
        result_count = tracing::field::Empty
    ))]
    pub fn search_datasets(&mut self, filters: &SearchFilters) -> Result<Vec<HubAsset>, HubError> {
        let cache_key = format!("datasets:{:?}", filters);

        if let Some(cached) = self.cache.get_search(&cache_key) {
            debug!(cache_hit = true, "Dataset search cache hit");
            tracing::Span::current().record("cache_hit", true);
            tracing::Span::current().record("result_count", cached.len());
            return Ok(cached.clone());
        }

        if self.offline_mode {
            warn!("Dataset search attempted in offline mode");
            return Err(HubError::OfflineMode);
        }

        let results = self.mock_dataset_search(filters);
        self.cache.cache_search(&cache_key, results.clone());
        info!(result_count = results.len(), "Dataset search completed");
        tracing::Span::current().record("cache_hit", false);
        tracing::Span::current().record("result_count", results.len());
        Ok(results)
    }

    /// Search spaces on HuggingFace Hub
    #[instrument(name = "hf.search.spaces", skip(self), fields(
        limit = filters.limit,
        cache_hit = tracing::field::Empty,
        result_count = tracing::field::Empty
    ))]
    pub fn search_spaces(&mut self, filters: &SearchFilters) -> Result<Vec<HubAsset>, HubError> {
        let cache_key = format!("spaces:{:?}", filters);

        if let Some(cached) = self.cache.get_search(&cache_key) {
            debug!(cache_hit = true, "Space search cache hit");
            tracing::Span::current().record("cache_hit", true);
            tracing::Span::current().record("result_count", cached.len());
            return Ok(cached.clone());
        }

        if self.offline_mode {
            warn!("Space search attempted in offline mode");
            return Err(HubError::OfflineMode);
        }

        let results = self.mock_space_search(filters);
        self.cache.cache_search(&cache_key, results.clone());
        info!(result_count = results.len(), "Space search completed");
        tracing::Span::current().record("cache_hit", false);
        tracing::Span::current().record("result_count", results.len());
        Ok(results)
    }

    // ========================================================================
    // HF-QUERY-003: Asset Metadata Methods (HF-OBS-002: Instrumented with tracing)
    // ========================================================================

    /// Get model metadata
    #[instrument(name = "hf.get.model", skip(self), fields(
        asset_id = id,
        cache_hit = tracing::field::Empty
    ))]
    pub fn get_model(&mut self, id: &str) -> Result<HubAsset, HubError> {
        let cache_key = format!("model:{}", id);

        if let Some(cached) = self.cache.get_asset(&cache_key) {
            debug!(cache_hit = true, "Model metadata cache hit");
            tracing::Span::current().record("cache_hit", true);
            return Ok(cached.clone());
        }

        if self.offline_mode {
            warn!(asset_id = id, "Model get attempted in offline mode");
            return Err(HubError::OfflineMode);
        }

        let asset = self.mock_get_model(id)?;
        self.cache.cache_asset(&cache_key, asset.clone());
        info!(asset_id = id, "Model metadata retrieved");
        tracing::Span::current().record("cache_hit", false);
        Ok(asset)
    }

    /// Get dataset metadata
    #[instrument(name = "hf.get.dataset", skip(self), fields(
        asset_id = id,
        cache_hit = tracing::field::Empty
    ))]
    pub fn get_dataset(&mut self, id: &str) -> Result<HubAsset, HubError> {
        let cache_key = format!("dataset:{}", id);

        if let Some(cached) = self.cache.get_asset(&cache_key) {
            debug!(cache_hit = true, "Dataset metadata cache hit");
            tracing::Span::current().record("cache_hit", true);
            return Ok(cached.clone());
        }

        if self.offline_mode {
            warn!(asset_id = id, "Dataset get attempted in offline mode");
            return Err(HubError::OfflineMode);
        }

        let asset = self.mock_get_dataset(id)?;
        self.cache.cache_asset(&cache_key, asset.clone());
        info!(asset_id = id, "Dataset metadata retrieved");
        tracing::Span::current().record("cache_hit", false);
        Ok(asset)
    }

    /// Get space metadata
    #[instrument(name = "hf.get.space", skip(self), fields(
        asset_id = id,
        cache_hit = tracing::field::Empty
    ))]
    pub fn get_space(&mut self, id: &str) -> Result<HubAsset, HubError> {
        let cache_key = format!("space:{}", id);

        if let Some(cached) = self.cache.get_asset(&cache_key) {
            debug!(cache_hit = true, "Space metadata cache hit");
            tracing::Span::current().record("cache_hit", true);
            return Ok(cached.clone());
        }

        if self.offline_mode {
            warn!(asset_id = id, "Space get attempted in offline mode");
            return Err(HubError::OfflineMode);
        }

        let asset = self.mock_get_space(id)?;
        self.cache.cache_asset(&cache_key, asset.clone());
        info!(asset_id = id, "Space metadata retrieved");
        tracing::Span::current().record("cache_hit", false);
        Ok(asset)
    }

    // ========================================================================
    // Mock implementations (replace with real API calls)
    // ========================================================================

    fn mock_model_search(&self, filters: &SearchFilters) -> Vec<HubAsset> {
        let mut results = vec![
            HubAsset::new("meta-llama/Llama-2-7b-hf", HubAssetType::Model)
                .with_downloads(5_000_000)
                .with_likes(10_000)
                .with_pipeline_tag("text-generation")
                .with_library("transformers")
                .with_license("llama2"),
            HubAsset::new("openai/whisper-large-v3", HubAssetType::Model)
                .with_downloads(2_000_000)
                .with_likes(5_000)
                .with_pipeline_tag("automatic-speech-recognition")
                .with_library("transformers")
                .with_license("apache-2.0"),
            HubAsset::new("stabilityai/stable-diffusion-xl-base-1.0", HubAssetType::Model)
                .with_downloads(3_000_000)
                .with_likes(8_000)
                .with_pipeline_tag("text-to-image")
                .with_library("diffusers")
                .with_license("openrail++"),
            HubAsset::new("sentence-transformers/all-MiniLM-L6-v2", HubAssetType::Model)
                .with_downloads(10_000_000)
                .with_likes(2_000)
                .with_pipeline_tag("sentence-similarity")
                .with_library("sentence-transformers")
                .with_license("apache-2.0"),
            HubAsset::new("bert-base-uncased", HubAssetType::Model)
                .with_downloads(50_000_000)
                .with_likes(15_000)
                .with_pipeline_tag("fill-mask")
                .with_library("transformers")
                .with_license("apache-2.0"),
        ];

        // Apply filters
        if let Some(ref task) = filters.task {
            results.retain(|m| m.pipeline_tag.as_ref().is_some_and(|t| t == task));
        }
        if let Some(ref library) = filters.library {
            results.retain(|m| m.library.as_ref().is_some_and(|l| l == library));
        }
        if let Some(min) = filters.min_downloads {
            results.retain(|m| m.downloads >= min);
        }
        if let Some(min) = filters.min_likes {
            results.retain(|m| m.likes >= min);
        }

        results.truncate(filters.limit);
        results
    }

    fn mock_dataset_search(&self, filters: &SearchFilters) -> Vec<HubAsset> {
        let mut results = vec![
            HubAsset::new("squad", HubAssetType::Dataset)
                .with_downloads(5_000_000)
                .with_likes(1_000)
                .with_tags(vec!["question-answering".into(), "english".into()]),
            HubAsset::new("imdb", HubAssetType::Dataset)
                .with_downloads(3_000_000)
                .with_likes(500)
                .with_tags(vec!["text-classification".into(), "sentiment".into()]),
            HubAsset::new("wikipedia", HubAssetType::Dataset)
                .with_downloads(10_000_000)
                .with_likes(2_000)
                .with_tags(vec!["text".into(), "multilingual".into()]),
        ];

        if let Some(min) = filters.min_downloads {
            results.retain(|d| d.downloads >= min);
        }

        results.truncate(filters.limit);
        results
    }

    fn mock_space_search(&self, filters: &SearchFilters) -> Vec<HubAsset> {
        let mut results = vec![
            HubAsset::new("gradio/chatbot", HubAssetType::Space)
                .with_downloads(100_000)
                .with_likes(500)
                .with_tags(vec!["gradio".into(), "chat".into()]),
            HubAsset::new("stabilityai/stable-diffusion", HubAssetType::Space)
                .with_downloads(500_000)
                .with_likes(2_000)
                .with_tags(vec!["gradio".into(), "image-generation".into()]),
        ];

        if let Some(min) = filters.min_downloads {
            results.retain(|s| s.downloads >= min);
        }

        results.truncate(filters.limit);
        results
    }

    fn mock_get_model(&self, id: &str) -> Result<HubAsset, HubError> {
        // Return mock data for known models
        match id {
            "meta-llama/Llama-2-7b-hf" => Ok(HubAsset::new(id, HubAssetType::Model)
                .with_downloads(5_000_000)
                .with_likes(10_000)
                .with_pipeline_tag("text-generation")
                .with_library("transformers")
                .with_license("llama2")),
            "bert-base-uncased" => Ok(HubAsset::new(id, HubAssetType::Model)
                .with_downloads(50_000_000)
                .with_likes(15_000)
                .with_pipeline_tag("fill-mask")
                .with_library("transformers")
                .with_license("apache-2.0")),
            _ => Err(HubError::NotFound(id.to_string())),
        }
    }

    fn mock_get_dataset(&self, id: &str) -> Result<HubAsset, HubError> {
        match id {
            "squad" => Ok(HubAsset::new(id, HubAssetType::Dataset)
                .with_downloads(5_000_000)
                .with_likes(1_000)
                .with_tags(vec!["question-answering".into()])),
            _ => Err(HubError::NotFound(id.to_string())),
        }
    }

    fn mock_get_space(&self, id: &str) -> Result<HubAsset, HubError> {
        match id {
            "gradio/chatbot" => Ok(HubAsset::new(id, HubAssetType::Space)
                .with_downloads(100_000)
                .with_likes(500)
                .with_tags(vec!["gradio".into(), "chat".into()])),
            _ => Err(HubError::NotFound(id.to_string())),
        }
    }
}

impl Default for HubClient {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Error Types
// ============================================================================

/// Hub API error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HubError {
    /// Asset not found
    NotFound(String),
    /// Rate limited
    RateLimited { retry_after: Option<u64> },
    /// Network error
    NetworkError(String),
    /// Offline mode - no cached data available
    OfflineMode,
    /// Invalid response from API
    InvalidResponse(String),
}

impl std::fmt::Display for HubError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "Asset not found: {}", id),
            Self::RateLimited { retry_after } => {
                if let Some(secs) = retry_after {
                    write!(f, "Rate limited, retry after {} seconds", secs)
                } else {
                    write!(f, "Rate limited")
                }
            }
            Self::NetworkError(msg) => write!(f, "Network error: {}", msg),
            Self::OfflineMode => write!(f, "Offline mode: no cached data available"),
            Self::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
        }
    }
}

impl std::error::Error for HubError {}

// ============================================================================
// Tests - Extreme TDD
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
#[path = "hub_client_tests.rs"]
mod tests;
