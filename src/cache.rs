//! Model caching and warming for reduced latency
//!
//! Provides LRU cache for model instances to reduce cold start latency and
//! improve throughput for repeated model usage.
//!
//! ## Features
//!
//! - **LRU Eviction**: Least Recently Used models are evicted when cache is full
//! - **Cache Warming**: Pre-load models on startup for zero cold starts
//! - **Metrics**: Track cache hits, misses, and evictions
//! - **Thread-Safe**: Concurrent access via Arc<RwLock>
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::cache::ModelCache;
//!
//! let cache = ModelCache::new(10); // capacity: 10 models
//! cache.warm(&["model1", "model2"])?;
//!
//! let model = cache.get_or_load("model1", || load_model("model1"))?;
//! ```

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use crate::{
    error::RealizarError,
    layers::{Model, ModelConfig},
    tokenizer::BPETokenizer,
};

/// Type alias for model and tokenizer pair
pub type ModelPair = (Arc<Model>, Arc<BPETokenizer>);

/// Type alias for the internal cache storage
type CacheStorage = Arc<RwLock<HashMap<CacheKey, CacheEntry>>>;

/// Cache key for identifying cached models
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    /// Model identifier (name, path, or config hash)
    pub id: String,
}

impl CacheKey {
    /// Create a new cache key
    #[must_use]
    pub fn new(id: String) -> Self {
        Self { id }
    }

    /// Create cache key from model config
    #[must_use]
    pub fn from_config(config: &ModelConfig) -> Self {
        // Simple hash based on config parameters
        let id = format!(
            "v{}_h{}_n{}_l{}_i{}",
            config.vocab_size,
            config.hidden_dim,
            config.num_heads,
            config.num_layers,
            config.intermediate_dim
        );
        Self::new(id)
    }
}

/// Cached model entry with metadata
#[derive(Clone)]
pub struct CacheEntry {
    /// The cached model
    pub model: Arc<Model>,
    /// The tokenizer for this model
    pub tokenizer: Arc<BPETokenizer>,
    /// Access count for LRU tracking
    access_count: u64,
    /// Last access timestamp
    last_access: std::time::Instant,
}

impl CacheEntry {
    /// Create a new cache entry
    #[must_use]
    pub fn new(model: Model, tokenizer: BPETokenizer) -> Self {
        Self {
            model: Arc::new(model),
            tokenizer: Arc::new(tokenizer),
            access_count: 0,
            last_access: std::time::Instant::now(),
        }
    }

    /// Record an access to this entry
    fn record_access(&mut self) {
        self.access_count += 1;
        self.last_access = std::time::Instant::now();
    }
}

/// Cache metrics for monitoring
#[derive(Debug, Default, Clone)]
pub struct CacheMetrics {
    /// Total cache hits
    pub hits: u64,
    /// Total cache misses
    pub misses: u64,
    /// Total evictions
    pub evictions: u64,
    /// Current cache size
    pub size: usize,
}

impl CacheMetrics {
    /// Calculate hit rate as percentage
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f64 / total as f64) * 100.0
        }
    }
}

/// LRU model cache
pub struct ModelCache {
    /// Cached entries
    cache: CacheStorage,
    /// Maximum cache capacity
    capacity: usize,
    /// Cache metrics
    metrics: Arc<RwLock<CacheMetrics>>,
}

impl ModelCache {
    /// Create a new model cache with the specified capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of models to cache
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            capacity,
            metrics: Arc::new(RwLock::new(CacheMetrics::default())),
        }
    }

    /// Get a model from cache or load it using the provided function
    ///
    /// # Arguments
    ///
    /// * `key` - Cache key for the model
    /// * `loader` - Function to load the model if not cached
    ///
    /// # Errors
    ///
    /// Returns error if model loading fails
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned (extremely rare, indicates thread panic while holding lock)
    pub fn get_or_load<F>(&self, key: &CacheKey, loader: F) -> Result<ModelPair, RealizarError>
    where
        F: FnOnce() -> Result<(Model, BPETokenizer), RealizarError>,
    {
        // Try to get from cache first (read lock)
        {
            let mut cache = self
                .cache
                .write()
                .expect("RwLock poisoned: thread panicked while holding cache write lock");
            if let Some(entry) = cache.get_mut(key) {
                entry.record_access();
                let mut metrics = self
                    .metrics
                    .write()
                    .expect("RwLock poisoned: thread panicked while holding metrics write lock");
                metrics.hits += 1;
                return Ok((entry.model.clone(), entry.tokenizer.clone()));
            }
        }

        // Cache miss - load the model
        let (model, tokenizer) = loader()?;
        let entry = CacheEntry::new(model, tokenizer);

        // Insert into cache (write lock)
        {
            let mut cache = self
                .cache
                .write()
                .expect("RwLock poisoned: thread panicked while holding cache write lock");
            let mut metrics = self
                .metrics
                .write()
                .expect("RwLock poisoned: thread panicked while holding metrics write lock");

            metrics.misses += 1;

            // Check if we need to evict
            if cache.len() >= self.capacity && !cache.contains_key(key) {
                Self::evict_lru(&mut cache, &mut metrics);
            }

            cache.insert(key.clone(), entry.clone());
            metrics.size = cache.len();
        }

        Ok((entry.model, entry.tokenizer))
    }

    /// Evict the least recently used entry from the cache
    fn evict_lru(cache: &mut HashMap<CacheKey, CacheEntry>, metrics: &mut CacheMetrics) {
        if let Some((lru_key, _)) = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_access)
            .map(|(k, e)| (k.clone(), e.clone()))
        {
            cache.remove(&lru_key);
            metrics.evictions += 1;
        }
    }

    /// Get current cache metrics
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    #[must_use]
    pub fn metrics(&self) -> CacheMetrics {
        self.metrics
            .read()
            .expect("RwLock poisoned: thread panicked while holding metrics read lock")
            .clone()
    }

    /// Clear all cached models
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    pub fn clear(&self) {
        let mut cache = self
            .cache
            .write()
            .expect("RwLock poisoned: thread panicked while holding cache write lock");
        cache.clear();
        let mut metrics = self
            .metrics
            .write()
            .expect("RwLock poisoned: thread panicked while holding metrics write lock");
        metrics.size = 0;
    }

    /// Get the number of cached models
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    #[must_use]
    pub fn len(&self) -> usize {
        self.cache
            .read()
            .expect("RwLock poisoned: thread panicked while holding cache read lock")
            .len()
    }

    /// Check if the cache is empty
    ///
    /// # Panics
    ///
    /// Panics if the internal `RwLock` is poisoned
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

include!("cache_tests.rs");
