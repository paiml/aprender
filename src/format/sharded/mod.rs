//! Sharded model import module (GH-127).
//!
//! Handles multi-tensor/multi-shard model imports from `HuggingFace` and other sources.
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
//! - `HuggingFace` safetensors: <https://huggingface.co/docs/safetensors>
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

/// Parsed shard index from `HuggingFace` format.
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

    /// Parse from `HuggingFace` index.json format.
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

include!("config.rs");
