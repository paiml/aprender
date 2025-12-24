//! Sharded model import module (GH-127).
//!
//! Handles multi-tensor/multi-shard model imports from HuggingFace and other sources.
//! Optimizes memory usage for 10B+ parameter models using streaming and LRU caching.
//!
//! # Problem
//!
//! Large models (>10B parameters) are often split across multiple safetensors files:
//! - `model-00001-of-00006.safetensors`
//! - `model-00002-of-00006.safetensors`
//! - etc.
//!
//! Loading all shards simultaneously causes OOM. This module implements:
//! - Parse `model.safetensors.index.json` for tensor→shard mapping
//! - Stream tensors in alphabetical order
//! - LRU cache for 1-2 shards at a time (2-5GB memory instead of 100GB)
//!
//! # Architecture
//!
//! ```text
//! index.json → ShardIndex → StreamingMerger → Output APR
//!                  ↓              ↓
//!           TensorMapping    LRU Cache (2 shards)
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::format::sharded::{ShardIndex, ShardedImporter};
//!
//! // Parse shard index
//! let index = ShardIndex::from_json("{}").unwrap_or_default();
//! assert_eq!(index.shard_count(), 0);
//! ```
//!
//! # References
//!
//! - HuggingFace safetensors: https://huggingface.co/docs/safetensors
//! - Memory-efficient model loading patterns
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>` where fallible

use crate::error::{AprenderError, Result};
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

// ============================================================================
// Shard Index (from model.safetensors.index.json)
// ============================================================================

/// Parsed shard index from HuggingFace format.
///
/// Represents the `model.safetensors.index.json` file structure:
/// ```json
/// {
///   "metadata": {...},
///   "weight_map": {
///     "layer.0.weight": "model-00001-of-00006.safetensors",
///     "layer.1.weight": "model-00002-of-00006.safetensors"
///   }
/// }
/// ```
#[derive(Debug, Clone, Default)]
pub struct ShardIndex {
    /// Tensor name → shard filename mapping
    weight_map: HashMap<String, String>,
    /// Unique shard filenames in order
    shard_files: Vec<String>,
    /// Shard filename → index mapping
    shard_indices: HashMap<String, usize>,
    /// Metadata from index file
    metadata: HashMap<String, String>,
}

impl ShardIndex {
    /// Create empty shard index
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse from HuggingFace index.json format.
    ///
    /// # Errors
    /// Returns error if JSON parsing fails.
    pub fn from_json(json: &str) -> Result<Self> {
        if json.is_empty() {
            return Ok(Self::default());
        }

        // Simplified JSON parsing (would use serde_json in production)
        let mut index = Self::new();

        // Parse weight_map section
        if let Some(weight_map_start) = json.find("\"weight_map\"") {
            let after_key = &json[weight_map_start..];
            if let Some(brace_start) = after_key.find('{') {
                let content = &after_key[brace_start + 1..];
                if let Some(brace_end) = content.find('}') {
                    let entries = &content[..brace_end];

                    // Parse individual entries
                    for entry in entries.split(',') {
                        let parts: Vec<&str> = entry.split(':').collect();
                        if parts.len() >= 2 {
                            let tensor_name = parts[0]
                                .trim()
                                .trim_matches('"')
                                .trim_matches('\\')
                                .to_string();
                            let shard_file = parts[1]
                                .trim()
                                .trim_matches('"')
                                .trim_matches('\\')
                                .to_string();

                            if !tensor_name.is_empty() && !shard_file.is_empty() {
                                index.add_mapping(&tensor_name, &shard_file);
                            }
                        }
                    }
                }
            }
        }

        Ok(index)
    }

    /// Add tensor to shard mapping
    pub fn add_mapping(&mut self, tensor_name: &str, shard_file: &str) {
        self.weight_map
            .insert(tensor_name.to_string(), shard_file.to_string());

        if !self.shard_indices.contains_key(shard_file) {
            let idx = self.shard_files.len();
            self.shard_files.push(shard_file.to_string());
            self.shard_indices.insert(shard_file.to_string(), idx);
        }
    }

    /// Get shard file for a tensor
    #[must_use]
    pub fn shard_for_tensor(&self, tensor_name: &str) -> Option<&str> {
        self.weight_map.get(tensor_name).map(String::as_str)
    }

    /// Get shard index (0-based) for a shard file
    #[must_use]
    pub fn shard_index(&self, shard_file: &str) -> Option<usize> {
        self.shard_indices.get(shard_file).copied()
    }

    /// Get total number of shards
    #[must_use]
    pub fn shard_count(&self) -> usize {
        self.shard_files.len()
    }

    /// Get total number of tensors
    #[must_use]
    pub fn tensor_count(&self) -> usize {
        self.weight_map.len()
    }

    /// Get all shard filenames in order
    #[must_use]
    pub fn shard_files(&self) -> &[String] {
        &self.shard_files
    }

    /// Get all tensor names (sorted for consistent processing)
    #[must_use]
    pub fn tensor_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.weight_map.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Group tensors by shard for efficient loading
    #[must_use]
    pub fn tensors_by_shard(&self) -> HashMap<&str, Vec<&str>> {
        let mut by_shard: HashMap<&str, Vec<&str>> = HashMap::new();

        for (tensor, shard) in &self.weight_map {
            by_shard
                .entry(shard.as_str())
                .or_default()
                .push(tensor.as_str());
        }

        // Sort tensors within each shard
        for tensors in by_shard.values_mut() {
            tensors.sort_unstable();
        }

        by_shard
    }

    /// Check if index is valid
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.weight_map.is_empty() && !self.shard_files.is_empty()
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }

    /// Get metadata
    #[must_use]
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

// ============================================================================
// LRU Shard Cache
// ============================================================================

/// Cached shard data
#[derive(Debug, Clone)]
pub struct CachedShard {
    /// Shard filename
    pub filename: String,
    /// Tensor data (name → bytes)
    pub tensors: HashMap<String, Vec<u8>>,
    /// Size in bytes
    pub size: usize,
}

impl CachedShard {
    /// Create new cached shard
    #[must_use]
    pub fn new(filename: String) -> Self {
        Self {
            filename,
            tensors: HashMap::new(),
            size: 0,
        }
    }

    /// Add tensor to cache
    pub fn add_tensor(&mut self, name: String, data: Vec<u8>) {
        self.size += data.len();
        self.tensors.insert(name, data);
    }

    /// Get tensor data
    #[must_use]
    pub fn get_tensor(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(Vec::as_slice)
    }

    /// Check if tensor exists
    #[must_use]
    pub fn has_tensor(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }
}

/// LRU cache for shards to minimize memory usage.
///
/// Keeps only the most recently accessed shards in memory.
#[derive(Debug)]
pub struct ShardCache {
    /// Maximum number of shards to keep
    max_shards: usize,
    /// Maximum total size in bytes
    max_bytes: usize,
    /// Cached shards (LRU order: front = oldest, back = newest)
    cache: VecDeque<CachedShard>,
    /// Current total size
    current_size: usize,
    /// Cache hit count (for metrics)
    hits: usize,
    /// Cache miss count
    misses: usize,
}

impl ShardCache {
    /// Create new LRU cache with limits.
    ///
    /// # Arguments
    /// * `max_shards` - Maximum number of shards to keep (typically 2-3)
    /// * `max_bytes` - Maximum total bytes (typically 2-5 GB)
    #[must_use]
    pub fn new(max_shards: usize, max_bytes: usize) -> Self {
        Self {
            max_shards: max_shards.max(1),
            max_bytes,
            cache: VecDeque::new(),
            current_size: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Default cache for large model import (2 shards, platform-appropriate max)
    #[must_use]
    pub fn default_for_import() -> Self {
        // WASM32 has 32-bit address space, use 256MB; native uses 4GB
        #[cfg(target_arch = "wasm32")]
        let max_bytes = 256 * 1024 * 1024; // 256MB for WASM
        #[cfg(not(target_arch = "wasm32"))]
        let max_bytes = 4_usize * 1024 * 1024 * 1024; // 4GB for native
        Self::new(2, max_bytes)
    }

    /// Get cached shard, returning None if not cached
    #[must_use]
    pub fn get(&mut self, filename: &str) -> Option<&CachedShard> {
        // Find shard in cache
        let pos = self.cache.iter().position(|s| s.filename == filename);

        if let Some(idx) = pos {
            // Move to back (most recently used)
            if idx < self.cache.len() - 1 {
                let shard = self.cache.remove(idx);
                if let Some(s) = shard {
                    self.cache.push_back(s);
                }
            }
            self.hits += 1;
            self.cache.back()
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert shard into cache, evicting old entries if needed
    pub fn insert(&mut self, shard: CachedShard) {
        // Evict if at capacity
        while self.cache.len() >= self.max_shards
            || (self.current_size + shard.size > self.max_bytes && !self.cache.is_empty())
        {
            if let Some(evicted) = self.cache.pop_front() {
                self.current_size = self.current_size.saturating_sub(evicted.size);
            }
        }

        self.current_size += shard.size;
        self.cache.push_back(shard);
    }

    /// Clear cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_size = 0;
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            cached_shards: self.cache.len(),
            cached_bytes: self.current_size,
            hits: self.hits,
            misses: self.misses,
        }
    }

    /// Get hit rate (0.0 to 1.0)
    #[must_use]
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total > 0 {
            self.hits as f32 / total as f32
        } else {
            0.0
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Number of shards currently cached
    pub cached_shards: usize,
    /// Total bytes cached
    pub cached_bytes: usize,
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
}

impl Default for ShardCache {
    fn default() -> Self {
        Self::default_for_import()
    }
}

// ============================================================================
// Sharded Importer
// ============================================================================

/// Import progress callback
pub type ProgressCallback = Box<dyn Fn(ImportProgress) + Send + Sync>;

/// Import progress information
#[derive(Debug, Clone)]
pub struct ImportProgress {
    /// Current phase (parsing, loading, merging)
    pub phase: ImportPhase,
    /// Tensors processed
    pub tensors_processed: usize,
    /// Total tensors
    pub total_tensors: usize,
    /// Shards loaded
    pub shards_loaded: usize,
    /// Total shards
    pub total_shards: usize,
    /// Bytes written
    pub bytes_written: u64,
    /// Estimated progress (0.0 to 1.0)
    pub progress: f32,
}

/// Import phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImportPhase {
    /// Parsing index file
    Parsing,
    /// Loading shard data
    Loading,
    /// Merging into output
    Merging,
    /// Finalizing output file
    Finalizing,
    /// Complete
    Complete,
}

/// Configuration for sharded import
#[derive(Debug, Clone)]
pub struct ShardedImportConfig {
    /// Maximum shards in cache
    pub max_cached_shards: usize,
    /// Maximum cache size in bytes
    pub max_cache_bytes: usize,
    /// Process tensors in alphabetical order (for determinism)
    pub sort_tensors: bool,
    /// Verify tensor checksums during import
    pub verify_checksums: bool,
    /// Buffer size for file I/O
    pub buffer_size: usize,
}

impl Default for ShardedImportConfig {
    fn default() -> Self {
        // WASM32 has 32-bit address space, use smaller defaults
        #[cfg(target_arch = "wasm32")]
        let max_cache_bytes = 256 * 1024 * 1024; // 256MB for WASM
        #[cfg(not(target_arch = "wasm32"))]
        let max_cache_bytes = 4_usize * 1024 * 1024 * 1024; // 4GB for native

        Self {
            max_cached_shards: 2,
            max_cache_bytes,
            sort_tensors: true,
            verify_checksums: true,
            buffer_size: 8 * 1024 * 1024, // 8MB
        }
    }
}

impl ShardedImportConfig {
    /// Config for low-memory systems (1GB cache)
    #[must_use]
    pub fn low_memory() -> Self {
        Self {
            max_cached_shards: 1,
            max_cache_bytes: 1024 * 1024 * 1024, // 1GB
            buffer_size: 4 * 1024 * 1024,        // 4MB
            ..Self::default()
        }
    }

    /// Config for high-memory systems (8GB cache on native, 512MB on WASM)
    #[must_use]
    pub fn high_memory() -> Self {
        // WASM32 has 32-bit address space
        #[cfg(target_arch = "wasm32")]
        let max_cache_bytes = 512 * 1024 * 1024; // 512MB for WASM
        #[cfg(not(target_arch = "wasm32"))]
        let max_cache_bytes = 8_usize * 1024 * 1024 * 1024; // 8GB for native

        Self {
            max_cached_shards: 4,
            max_cache_bytes,
            buffer_size: 16 * 1024 * 1024, // 16MB
            ..Self::default()
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.max_cached_shards == 0 {
            return Err(AprenderError::FormatError {
                message: "max_cached_shards must be > 0".to_string(),
            });
        }
        if self.buffer_size == 0 {
            return Err(AprenderError::FormatError {
                message: "buffer_size must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

/// Result of sharded import operation
#[derive(Debug, Clone)]
pub struct ImportReport {
    /// Total tensors imported
    pub tensor_count: usize,
    /// Total shards processed
    pub shard_count: usize,
    /// Total bytes written
    pub bytes_written: u64,
    /// Peak memory usage estimate
    pub peak_memory_bytes: u64,
    /// Cache hit rate
    pub cache_hit_rate: f32,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// Warnings encountered
    pub warnings: Vec<String>,
}

/// Sharded model importer.
///
/// Handles memory-efficient import of multi-shard models.
#[derive(Debug)]
pub struct ShardedImporter {
    /// Configuration
    config: ShardedImportConfig,
    /// Shard cache
    cache: ShardCache,
    /// Base directory for shard files
    base_dir: PathBuf,
}

impl ShardedImporter {
    /// Create new importer with configuration
    #[must_use]
    pub fn new(config: ShardedImportConfig, base_dir: PathBuf) -> Self {
        let cache = ShardCache::new(config.max_cached_shards, config.max_cache_bytes);
        Self {
            config,
            cache,
            base_dir,
        }
    }

    /// Create with default configuration
    #[must_use]
    pub fn with_defaults(base_dir: PathBuf) -> Self {
        Self::new(ShardedImportConfig::default(), base_dir)
    }

    /// Parse shard index file.
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
    pub fn parse_index(&self, index_path: &Path) -> Result<ShardIndex> {
        let content = std::fs::read_to_string(index_path).map_err(AprenderError::Io)?;

        ShardIndex::from_json(&content)
    }

    /// Load a single shard file.
    ///
    /// # Errors
    /// Returns error if shard cannot be loaded.
    pub fn load_shard(&mut self, shard_file: &str) -> Result<&CachedShard> {
        // Check cache first
        if self.cache.get(shard_file).is_some() {
            return self
                .cache
                .get(shard_file)
                .ok_or_else(|| AprenderError::FormatError {
                    message: "Cache inconsistency".to_string(),
                });
        }

        // Load from disk
        let shard_path = self.base_dir.join(shard_file);
        let mut shard = CachedShard::new(shard_file.to_string());

        // Simplified loading (real impl would parse safetensors format)
        if shard_path.exists() {
            let metadata = std::fs::metadata(&shard_path).map_err(AprenderError::Io)?;
            shard.size = metadata.len() as usize;
        }

        self.cache.insert(shard);

        self.cache
            .get(shard_file)
            .ok_or_else(|| AprenderError::FormatError {
                message: "Failed to retrieve cached shard".to_string(),
            })
    }

    /// Stream merge multiple shards into single output.
    ///
    /// # Arguments
    /// * `index` - Parsed shard index
    /// * `output_path` - Output APR file path
    ///
    /// # Errors
    /// Returns error if merge fails.
    pub fn stream_merge(
        &mut self,
        index: &ShardIndex,
        _output_path: &Path,
    ) -> Result<ImportReport> {
        self.config.validate()?;

        let start_time = std::time::Instant::now();
        let mut warnings = Vec::new();

        let tensor_names = if self.config.sort_tensors {
            index.tensor_names()
        } else {
            index.weight_map.keys().map(String::as_str).collect()
        };

        let mut bytes_written = 0u64;
        let mut peak_memory = 0u64;

        // Process tensors in order
        for tensor_name in &tensor_names {
            if let Some(shard_file) = index.shard_for_tensor(tensor_name) {
                // Load shard (may hit cache)
                match self.load_shard(shard_file) {
                    Ok(_shard) => {
                        // Would write tensor to output here
                        bytes_written += 1024; // Placeholder
                    }
                    Err(e) => {
                        warnings.push(format!("Failed to load shard {shard_file}: {e}"));
                    }
                }
            }

            // Track peak memory
            let current_memory = self.cache.stats().cached_bytes as u64;
            peak_memory = peak_memory.max(current_memory);
        }

        let duration_ms = start_time.elapsed().as_millis() as u64;

        Ok(ImportReport {
            tensor_count: tensor_names.len(),
            shard_count: index.shard_count(),
            bytes_written,
            peak_memory_bytes: peak_memory,
            cache_hit_rate: self.cache.hit_rate(),
            duration_ms,
            warnings,
        })
    }

    /// Get cache statistics
    #[must_use]
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear shard cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &ShardedImportConfig {
        &self.config
    }
}

impl Default for ShardedImporter {
    fn default() -> Self {
        Self::with_defaults(PathBuf::from("."))
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Detect if a directory contains sharded model files.
///
/// Looks for `model.safetensors.index.json` or multiple `*.safetensors` files.
#[must_use]
pub fn is_sharded_model(dir: &Path) -> bool {
    // Check for index file
    if dir.join("model.safetensors.index.json").exists() {
        return true;
    }

    // Check for multiple safetensors files
    let safetensors_count = std::fs::read_dir(dir).map_or(0, |entries| {
        entries
            .filter_map(std::result::Result::ok)
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
            .count()
    });

    safetensors_count > 1
}

/// Get shard file paths from directory.
///
/// Returns files in sorted order for deterministic processing.
pub fn get_shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let entries = std::fs::read_dir(dir).map_err(AprenderError::Io)?;

    let mut files: Vec<PathBuf> = entries
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "safetensors"))
        .map(|e| e.path())
        .collect();

    files.sort();
    Ok(files)
}

/// Estimate memory required for loading a shard.
///
/// Based on file size and typical SafeTensors overhead.
#[must_use]
pub fn estimate_shard_memory(file_size: u64) -> u64 {
    // SafeTensors has ~8 bytes header overhead per tensor
    // Plus some allocation overhead
    file_size + (file_size / 100) // ~1% overhead
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
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
}
