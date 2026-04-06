//! Unified Model Fetcher
//!
//! Provides an "ollama-like" experience for model management:
//! - Short aliases (llama3, mistral) resolve to full model references
//! - Automatic caching with progress reporting
//! - Format detection and quantization selection
//!
//! ## Example
//!
//! ```rust,ignore
//! use pacha::fetcher::{ModelFetcher, FetchConfig};
//!
//! let fetcher = ModelFetcher::new()?;
//!
//! // Pull with short name
//! let model = fetcher.pull("llama3:8b-q4_k_m", |progress| {
//!     println!("{}", progress.progress_bar(40));
//! })?;
//!
//! // List cached models
//! for entry in fetcher.list()? {
//!     println!("{}: {}", entry.name, entry.size_human());
//! }
//! ```

use crate::aliases::{AliasEntry, AliasRegistry, ResolvedAlias};
use crate::cache::{
    format_bytes, CacheConfig, CacheEntry, CacheManager, CacheStats, DownloadProgress,
    EvictionPolicy,
};
use crate::error::{PachaError, Result};
use crate::format::{detect_format, ModelFormat, QuantType};
use crate::resolver::ModelResolver;
use crate::uri::ModelUri;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ============================================================================
// FETCH-001: Configuration
// ============================================================================

/// Fetcher configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchConfig {
    /// Cache configuration
    pub cache: CacheConfig,
    /// Preferred quantization (used when not specified in model ref)
    pub default_quant: Option<QuantType>,
    /// Whether to auto-pull missing models
    pub auto_pull: bool,
    /// Maximum concurrent downloads
    pub max_concurrent: usize,
    /// Verify model integrity after download
    pub verify_integrity: bool,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
}

impl Default for FetchConfig {
    fn default() -> Self {
        Self {
            cache: CacheConfig::default(),
            default_quant: Some(QuantType::Q4_K_M),
            auto_pull: true,
            max_concurrent: 2,
            verify_integrity: true,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

impl FetchConfig {
    /// Create new configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set cache configuration
    #[must_use]
    pub fn with_cache(mut self, cache: CacheConfig) -> Self {
        self.cache = cache;
        self
    }

    /// Set default quantization
    #[must_use]
    pub fn with_default_quant(mut self, quant: QuantType) -> Self {
        self.default_quant = Some(quant);
        self
    }

    /// Enable/disable auto-pull
    #[must_use]
    pub fn with_auto_pull(mut self, enabled: bool) -> Self {
        contract_pre_configuration!();
        self.auto_pull = enabled;
        self
    }

    /// Set eviction policy
    #[must_use]
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }
}

// ============================================================================
// FETCH-002: Fetch Result
// ============================================================================

/// Result of a model fetch operation
#[derive(Debug)]
pub struct FetchResult {
    /// Path to the cached model file
    pub path: PathBuf,
    /// Detected model format
    pub format: ModelFormat,
    /// Model size in bytes
    pub size_bytes: u64,
    /// Whether this was a cache hit
    pub cache_hit: bool,
    /// Original reference used
    pub reference: String,
    /// Resolved URI
    pub resolved_uri: String,
    /// Content hash
    pub hash: String,
}

impl FetchResult {
    /// Get human-readable size
    #[must_use]
    pub fn size_human(&self) -> String {
        format_bytes(self.size_bytes)
    }

    /// Check if model is quantized
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        match &self.format {
            ModelFormat::Gguf(info) => info.quantization.is_some(),
            _ => false,
        }
    }

    /// Get quantization type if available
    #[must_use]
    pub fn quant_type(&self) -> Option<QuantType> {
        match &self.format {
            ModelFormat::Gguf(info) => {
                info.quantization.as_ref().and_then(|q| QuantType::from_str(q))
            }
            _ => None,
        }
    }
}

// ============================================================================
// FETCH-003: Model Fetcher
// ============================================================================

/// Unified model fetcher with caching and alias support
pub struct ModelFetcher {
    /// Configuration
    config: FetchConfig,
    /// Alias registry
    aliases: AliasRegistry,
    /// Cache manager
    cache: CacheManager,
    /// Model resolver
    resolver: Option<ModelResolver>,
    /// Cache directory
    cache_dir: PathBuf,
}

impl ModelFetcher {
    /// Create a new fetcher with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(FetchConfig::default())
    }

    /// Create a fetcher with custom configuration
    pub fn with_config(config: FetchConfig) -> Result<Self> {
        let cache_dir = get_default_cache_dir();

        // Ensure cache directory exists
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            PachaError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to create cache dir: {}", cache_dir.display()),
            ))
        })?;

        // GH-162 FIX: Load existing cache entries from manifest
        let mut cache = CacheManager::new(config.cache.clone()).with_policy(config.eviction_policy);
        Self::load_manifest(&cache_dir, &mut cache);

        let resolver = ModelResolver::new_default().ok();

        Ok(Self { config, aliases: AliasRegistry::with_defaults(), cache, resolver, cache_dir })
    }

    /// Load cache manifest from disk (GH-162 fix)
    fn load_manifest(cache_dir: &Path, cache: &mut CacheManager) {
        let manifest_path = cache_dir.join("manifest.json");
        if let Ok(data) = std::fs::read_to_string(&manifest_path) {
            if let Ok(entries) = serde_json::from_str::<Vec<CacheEntry>>(&data) {
                for entry in entries {
                    // Verify file still exists before adding to cache
                    if entry.path.exists() {
                        cache.add(entry);
                    }
                }
            }
        }
    }

    /// Save cache manifest to disk (GH-162 fix).
    ///
    /// Uses atomic write (temp file + rename) to prevent corruption if
    /// the process crashes mid-write.
    fn save_manifest(&self) {
        let manifest_path = self.cache_dir.join("manifest.json");
        let tmp_path = self.cache_dir.join("manifest.json.tmp");
        let entries: Vec<&CacheEntry> = self.cache.list();
        if let Ok(data) = serde_json::to_string_pretty(&entries) {
            if std::fs::write(&tmp_path, &data).is_ok() {
                let _ = std::fs::rename(&tmp_path, &manifest_path);
            }
        }
    }

    /// Create a fetcher with a specific cache directory
    pub fn with_cache_dir(cache_dir: PathBuf, config: FetchConfig) -> Result<Self> {
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            PachaError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to create cache dir: {}", cache_dir.display()),
            ))
        })?;

        let cache = CacheManager::new(config.cache.clone()).with_policy(config.eviction_policy);

        let resolver = ModelResolver::new_default().ok();

        Ok(Self { config, aliases: AliasRegistry::with_defaults(), cache, resolver, cache_dir })
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &FetchConfig {
        &self.config
    }

    /// Get alias registry
    #[must_use]
    pub fn aliases(&self) -> &AliasRegistry {
        &self.aliases
    }

    /// Add or update an alias
    pub fn add_alias(&mut self, alias: &str, uri: &str) -> Result<()> {
        self.aliases.add(AliasEntry::new(alias, uri));
        Ok(())
    }

    /// Resolve a model reference to full URI
    pub fn resolve_ref(&self, model_ref: &str) -> Result<ResolvedAlias> {
        contract_pre_configuration!(model_ref.as_bytes());
        let resolved = self.aliases.resolve(model_ref);
        // Check if it was actually resolved from an alias or just passthrough
        if resolved.is_alias || model_ref.contains("://") {
            Ok(resolved)
        } else {
            // Not found as an alias and not a full URI
            Err(PachaError::NotFound {
                kind: "alias".to_string(),
                name: model_ref.to_string(),
                version: "N/A".to_string(),
            })
        }
    }

    /// Pull a model (fetch if not cached)
    pub fn pull<F>(&mut self, model_ref: &str, progress_fn: F) -> Result<FetchResult>
    where
        F: Fn(&DownloadProgress),
    {
        contract_pre_configuration!(model_ref);
        // Resolve the reference (always returns a result)
        let resolved = self.aliases.resolve(model_ref);

        // Determine the URI to fetch
        let uri_str = resolved.uri;

        // Check if already cached
        let cache_key = Self::cache_key(&uri_str);
        if let Some(entry) = self.cache.get(&cache_key, "1.0") {
            let format = format_from_path(&entry.path);
            return Ok(FetchResult {
                path: entry.path.clone(),
                format,
                size_bytes: entry.size_bytes,
                cache_hit: true,
                reference: model_ref.to_string(),
                resolved_uri: uri_str,
                hash: entry.hash.clone(),
            });
        }

        // Need to fetch
        let uri = ModelUri::parse(&uri_str)?;
        let resolver = self
            .resolver
            .as_ref()
            .ok_or_else(|| PachaError::NotInitialized(PathBuf::from("~/.pacha")))?;

        // Start progress tracking
        let mut progress = DownloadProgress::new(0); // Unknown size initially
        progress_fn(&progress);

        // Resolve the model
        let resolved_model = resolver.resolve(&uri)?;

        // Update progress with final size
        progress = DownloadProgress::new(resolved_model.data.len() as u64);
        progress.update(resolved_model.data.len() as u64);
        progress_fn(&progress);

        // Detect format
        let format = detect_format(&resolved_model.data);

        // Compute hash
        let hash = blake3::hash(&resolved_model.data).to_hex().to_string();

        // Determine file extension
        let extension = match &format {
            ModelFormat::Gguf(_) => "gguf",
            ModelFormat::SafeTensors(_) => "safetensors",
            ModelFormat::Apr(_) => "apr",
            ModelFormat::Onnx(_) => "onnx",
            ModelFormat::PyTorch => "pt",
            ModelFormat::Unknown => "bin",
        };

        // Save to cache
        let filename = format!("{}.{}", &hash[..16], extension);
        let cache_path = self.cache_dir.join(&filename);

        std::fs::write(&cache_path, &resolved_model.data).map_err(|e| {
            PachaError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to write to cache: {}", cache_path.display()),
            ))
        })?;

        // Add to cache manager
        let entry = CacheEntry::new(
            &cache_key,
            "1.0",
            resolved_model.data.len() as u64,
            &hash,
            cache_path.clone(),
        );
        self.cache.add(entry);

        // GH-162 FIX: Persist cache manifest after adding entry
        self.save_manifest();

        Ok(FetchResult {
            path: cache_path,
            format,
            size_bytes: resolved_model.data.len() as u64,
            cache_hit: false,
            reference: model_ref.to_string(),
            resolved_uri: uri_str,
            hash,
        })
    }

    /// Pull without progress callback
    pub fn pull_quiet(&mut self, model_ref: &str) -> Result<FetchResult> {
        contract_pre_configuration!(model_ref);
        self.pull(model_ref, |_| {})
    }

    /// Check if a model is cached
    #[must_use]
    pub fn is_cached(&self, model_ref: &str) -> bool {
        let resolved = self.aliases.resolve(model_ref);
        let key = Self::cache_key(&resolved.uri);
        self.cache.contains(&key, "1.0")
    }

    /// Remove a model from cache
    pub fn remove(&mut self, model_ref: &str) -> Result<bool> {
        let resolved = self.aliases.resolve(model_ref);
        let uri = resolved.uri;

        let key = Self::cache_key(&uri);
        if let Some(entry) = self.cache.remove(&key, "1.0") {
            // Remove file
            if entry.path.exists() {
                std::fs::remove_file(&entry.path).ok();
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// List all cached models
    #[must_use]
    pub fn list(&self) -> Vec<CachedModel> {
        self.cache
            .list()
            .iter()
            .map(|e| {
                let format = format_from_path(&e.path);
                CachedModel {
                    name: e.name.clone(),
                    version: e.version.clone(),
                    size_bytes: e.size_bytes,
                    format,
                    path: e.path.clone(),
                    last_accessed: e.last_accessed,
                    access_count: e.access_count,
                    pinned: e.pinned,
                }
            })
            .collect()
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Cleanup old/unused models
    pub fn cleanup(&mut self) -> u64 {
        self.cache.cleanup_to_target()
    }

    /// Cleanup models older than specified days
    pub fn cleanup_old(&mut self) -> u64 {
        self.cache.cleanup_old_entries()
    }

    /// Clear entire cache
    pub fn clear(&mut self) -> u64 {
        // Also remove files
        for entry in self.cache.list() {
            if entry.path.exists() {
                std::fs::remove_file(&entry.path).ok();
            }
        }
        self.cache.clear()
    }

    /// Pin a model (prevent eviction)
    pub fn pin(&mut self, model_ref: &str) -> bool {
        let key = Self::cache_key(model_ref);
        self.cache.pin(&key, "1.0")
    }

    /// Unpin a model
    pub fn unpin(&mut self, model_ref: &str) -> bool {
        let key = Self::cache_key(model_ref);
        self.cache.unpin(&key, "1.0")
    }

    /// Get cache directory path
    #[must_use]
    pub fn cache_dir(&self) -> &PathBuf {
        &self.cache_dir
    }

    // Internal: Generate cache key from URI
    fn cache_key(uri: &str) -> String {
        // Normalize the URI for caching
        uri.replace("://", "_").replace('/', "_").replace(':', "_")
    }
}

// ============================================================================
// FETCH-004: Cached Model Info
// ============================================================================

/// Information about a cached model
#[derive(Debug, Clone)]
pub struct CachedModel {
    /// Model name/reference
    pub name: String,
    /// Version
    pub version: String,
    /// Size in bytes
    pub size_bytes: u64,
    /// Detected format
    pub format: ModelFormat,
    /// Path to cached file
    pub path: PathBuf,
    /// Last access time
    pub last_accessed: std::time::SystemTime,
    /// Access count
    pub access_count: u64,
    /// Whether pinned
    pub pinned: bool,
}

impl CachedModel {
    /// Get human-readable size
    #[must_use]
    pub fn size_human(&self) -> String {
        format_bytes(self.size_bytes)
    }

    /// Get quantization type if GGUF
    #[must_use]
    pub fn quant_type(&self) -> Option<QuantType> {
        match &self.format {
            ModelFormat::Gguf(info) => {
                info.quantization.as_ref().and_then(|q| QuantType::from_str(q))
            }
            _ => None,
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Determine ModelFormat from file path extension
fn format_from_path(path: &Path) -> ModelFormat {
    let ext = path.extension().and_then(|e| e.to_str()).map(|e| e.to_lowercase());

    match ext.as_deref() {
        Some("gguf") => ModelFormat::Gguf(Default::default()),
        Some("safetensors") => ModelFormat::SafeTensors(Default::default()),
        Some("apr") => ModelFormat::Apr(Default::default()),
        Some("onnx") => ModelFormat::Onnx(Default::default()),
        Some("pt") | Some("pth") => ModelFormat::PyTorch,
        _ => ModelFormat::Unknown,
    }
}

/// Get the default cache directory
fn get_default_cache_dir() -> PathBuf {
    // Try XDG_CACHE_HOME first (Linux/BSD standard)
    if let Ok(cache_home) = std::env::var("XDG_CACHE_HOME") {
        return PathBuf::from(cache_home).join("pacha").join("models");
    }

    // Fall back to HOME/.cache (Linux/macOS)
    if let Ok(home) = std::env::var("HOME") {
        return PathBuf::from(home).join(".cache").join("pacha").join("models");
    }

    // Windows: try LOCALAPPDATA
    if let Ok(local_app_data) = std::env::var("LOCALAPPDATA") {
        return PathBuf::from(local_app_data).join("pacha").join("cache").join("models");
    }

    // Final fallback
    PathBuf::from(".cache").join("pacha").join("models")
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ========================================================================
    // Config Tests
    // ========================================================================

    #[test]
    fn test_fetch_config_default() {
        let config = FetchConfig::default();
        assert!(config.auto_pull);
        assert_eq!(config.max_concurrent, 2);
        assert!(config.verify_integrity);
    }

    #[test]
    fn test_fetch_config_builder() {
        let config = FetchConfig::new()
            .with_default_quant(QuantType::Q8_0)
            .with_auto_pull(false)
            .with_eviction_policy(EvictionPolicy::LFU);

        assert_eq!(config.default_quant, Some(QuantType::Q8_0));
        assert!(!config.auto_pull);
        assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
    }

    #[test]
    fn test_fetch_config_with_cache() {
        let cache_config = CacheConfig::new().with_max_size_gb(100.0);
        let config = FetchConfig::new().with_cache(cache_config.clone());

        assert_eq!(config.cache.max_size_bytes, cache_config.max_size_bytes);
    }

    // ========================================================================
    // Fetcher Creation Tests
    // ========================================================================

    #[test]
    fn test_fetcher_with_cache_dir() {
        let dir = TempDir::new().unwrap();
        let result = ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default());
        assert!(result.is_ok());
    }

    #[test]
    fn test_fetcher_cache_dir_created() {
        let dir = TempDir::new().unwrap();
        let cache_dir = dir.path().join("models");

        let _ = ModelFetcher::with_cache_dir(cache_dir.clone(), FetchConfig::default()).unwrap();

        assert!(cache_dir.exists());
    }

    #[test]
    fn test_fetcher_config_access() {
        let dir = TempDir::new().unwrap();
        let config = FetchConfig::new().with_auto_pull(false);
        let fetcher = ModelFetcher::with_cache_dir(dir.path().to_path_buf(), config).unwrap();

        assert!(!fetcher.config().auto_pull);
    }

    // ========================================================================
    // Alias Tests
    // ========================================================================

    #[test]
    fn test_fetcher_has_default_aliases() {
        let dir = TempDir::new().unwrap();
        let fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        let aliases = fetcher.aliases();
        assert!(aliases.get("llama3").is_some());
        assert!(aliases.get("mistral").is_some());
    }

    #[test]
    fn test_fetcher_add_alias() {
        let dir = TempDir::new().unwrap();
        let mut fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        fetcher.add_alias("mymodel", "hf://my-org/my-model").unwrap();

        assert!(fetcher.aliases().get("mymodel").is_some());
    }

    #[test]
    fn test_fetcher_resolve_ref() {
        let dir = TempDir::new().unwrap();
        let fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        let resolved = fetcher.resolve_ref("llama3");
        assert!(resolved.is_ok());
        let uri = resolved.unwrap().uri;
        // Aliases resolve to hf:// scheme
        assert!(uri.starts_with("hf://"), "Expected hf:// URI, got: {}", uri);
    }

    #[test]
    fn test_fetcher_resolve_ref_not_found() {
        let dir = TempDir::new().unwrap();
        let fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        let resolved = fetcher.resolve_ref("nonexistent-model-xyz");
        assert!(resolved.is_err());
    }

    // ========================================================================
    // Cache Tests
    // ========================================================================

    #[test]
    fn test_fetcher_is_cached_empty() {
        let dir = TempDir::new().unwrap();
        let fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        assert!(!fetcher.is_cached("llama3"));
    }

    #[test]
    fn test_fetcher_stats_empty() {
        let dir = TempDir::new().unwrap();
        let fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        let stats = fetcher.stats();
        assert_eq!(stats.model_count, 0);
        assert_eq!(stats.total_size_bytes, 0);
    }

    #[test]
    fn test_fetcher_list_empty() {
        let dir = TempDir::new().unwrap();
        let fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        assert!(fetcher.list().is_empty());
    }

    #[test]
    fn test_fetcher_clear() {
        let dir = TempDir::new().unwrap();
        let mut fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        let freed = fetcher.clear();
        assert_eq!(freed, 0); // Nothing to clear
    }

    #[test]
    fn test_fetcher_cleanup() {
        let dir = TempDir::new().unwrap();
        let mut fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        let freed = fetcher.cleanup();
        assert_eq!(freed, 0);
    }

    // ========================================================================
    // Cache Key Tests
    // ========================================================================

    #[test]
    fn test_cache_key_generation() {
        let key1 = ModelFetcher::cache_key("hf://meta-llama/Llama-3-8B");
        let key2 = ModelFetcher::cache_key("pacha://model:1.0.0");

        assert!(!key1.contains("://"));
        assert!(!key2.contains("://"));
    }

    #[test]
    fn test_cache_key_unique() {
        let key1 = ModelFetcher::cache_key("hf://org/model1");
        let key2 = ModelFetcher::cache_key("hf://org/model2");

        assert_ne!(key1, key2);
    }

    // ========================================================================
    // Fetch Result Tests
    // ========================================================================

    #[test]
    fn test_fetch_result_size_human() {
        let result = FetchResult {
            path: PathBuf::from("/cache/model.gguf"),
            format: ModelFormat::Unknown,
            size_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            cache_hit: true,
            reference: "llama3".to_string(),
            resolved_uri: "hf://meta-llama/Llama-3-8B".to_string(),
            hash: "abc123".to_string(),
        };

        assert!(result.size_human().contains("GB"));
    }

    #[test]
    fn test_fetch_result_not_quantized() {
        let result = FetchResult {
            path: PathBuf::from("/cache/model.safetensors"),
            format: ModelFormat::SafeTensors(Default::default()),
            size_bytes: 1000,
            cache_hit: false,
            reference: "test".to_string(),
            resolved_uri: "test".to_string(),
            hash: "hash".to_string(),
        };

        assert!(!result.is_quantized());
        assert!(result.quant_type().is_none());
    }

    #[test]
    fn test_fetch_result_quantized_gguf() {
        use crate::format::GgufInfo;

        let result = FetchResult {
            path: PathBuf::from("/cache/model.gguf"),
            format: ModelFormat::Gguf(GgufInfo {
                version: 3,
                tensor_count: 100,
                metadata_count: 10,
                quantization: Some("Q4_K_M".to_string()),
                ..Default::default()
            }),
            size_bytes: 4_000_000_000,
            cache_hit: true,
            reference: "llama3:8b-q4_k_m".to_string(),
            resolved_uri: "hf://...".to_string(),
            hash: "hash".to_string(),
        };

        assert!(result.is_quantized());
        assert_eq!(result.quant_type(), Some(QuantType::Q4_K_M));
    }

    // ========================================================================
    // Cached Model Tests
    // ========================================================================

    #[test]
    fn test_cached_model_size_human() {
        let model = CachedModel {
            name: "llama3".to_string(),
            version: "8b".to_string(),
            size_bytes: 4 * 1024 * 1024 * 1024,
            format: ModelFormat::Unknown,
            path: PathBuf::from("/cache"),
            last_accessed: std::time::SystemTime::now(),
            access_count: 5,
            pinned: false,
        };

        assert!(model.size_human().contains("GB"));
    }

    #[test]
    fn test_cached_model_quant_type() {
        use crate::format::GgufInfo;

        let model = CachedModel {
            name: "llama3".to_string(),
            version: "8b".to_string(),
            size_bytes: 4_000_000_000,
            format: ModelFormat::Gguf(GgufInfo {
                version: 3,
                tensor_count: 100,
                metadata_count: 10,
                quantization: Some("Q8_0".to_string()),
                ..Default::default()
            }),
            path: PathBuf::from("/cache/model.gguf"),
            last_accessed: std::time::SystemTime::now(),
            access_count: 1,
            pinned: true,
        };

        assert_eq!(model.quant_type(), Some(QuantType::Q8_0));
    }

    // ========================================================================
    // Pin/Unpin Tests
    // ========================================================================

    #[test]
    fn test_fetcher_pin_nonexistent() {
        let dir = TempDir::new().unwrap();
        let mut fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        assert!(!fetcher.pin("nonexistent"));
    }

    #[test]
    fn test_fetcher_unpin_nonexistent() {
        let dir = TempDir::new().unwrap();
        let mut fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        assert!(!fetcher.unpin("nonexistent"));
    }

    // ========================================================================
    // Remove Tests
    // ========================================================================

    #[test]
    fn test_fetcher_remove_nonexistent() {
        let dir = TempDir::new().unwrap();
        let mut fetcher =
            ModelFetcher::with_cache_dir(dir.path().to_path_buf(), FetchConfig::default()).unwrap();

        let result = fetcher.remove("nonexistent");
        assert!(result.is_ok());
        assert!(!result.unwrap());
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_fetch_config_serialization() {
        let config = FetchConfig::new().with_default_quant(QuantType::Q4_K_M).with_auto_pull(false);

        let json = serde_json::to_string(&config).unwrap();
        let parsed: FetchConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.default_quant, config.default_quant);
        assert_eq!(parsed.auto_pull, config.auto_pull);
    }
}
