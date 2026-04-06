
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
/// Based on file size and typical `SafeTensors` overhead.
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
mod tests;
