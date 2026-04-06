
/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum L1 cache size in bytes
    pub l1_max_bytes: usize,
    /// Maximum L2 cache size in bytes
    pub l2_max_bytes: usize,
    /// Eviction policy
    pub eviction_policy: EvictionPolicy,
    /// Default TTL for entries
    pub default_ttl: Option<Duration>,
    /// Enable prefetching
    pub prefetch_enabled: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_bytes: 64 * 1024 * 1024,   // 64 MB
            l2_max_bytes: 1024 * 1024 * 1024, // 1 GB
            eviction_policy: EvictionPolicy::LRU,
            default_ttl: None,
            prefetch_enabled: true,
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create configuration for embedded deployment
    #[must_use]
    pub fn embedded(l1_bytes: usize) -> Self {
        Self {
            l1_max_bytes: l1_bytes,
            l2_max_bytes: 0, // No L2 on embedded
            eviction_policy: EvictionPolicy::Fixed,
            default_ttl: None,
            prefetch_enabled: false,
        }
    }

    /// Set L1 cache size
    #[must_use]
    pub fn with_l1_size(mut self, bytes: usize) -> Self {
        self.l1_max_bytes = bytes;
        self
    }

    /// Set L2 cache size
    #[must_use]
    pub fn with_l2_size(mut self, bytes: usize) -> Self {
        self.l2_max_bytes = bytes;
        self
    }

    /// Set eviction policy
    #[must_use]
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }

    /// Set default TTL
    #[must_use]
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = Some(ttl);
        self
    }

    /// Enable or disable prefetching
    #[must_use]
    pub fn with_prefetch(mut self, enabled: bool) -> Self {
        self.prefetch_enabled = enabled;
        self
    }
}

/// Model information for registry
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Size in bytes
    pub size_bytes: usize,
    /// Whether model is bundled (compile-time)
    pub is_bundled: bool,
    /// Whether model is currently cached
    pub is_cached: bool,
    /// Cache tier if cached
    pub cache_tier: Option<CacheTier>,
}

/// Model registry for bundled and dynamic models
#[derive(Debug)]
pub struct ModelRegistry {
    /// L1 cache entries
    l1_cache: HashMap<String, CacheEntry>,
    /// L2 cache entries
    l2_cache: HashMap<String, CacheEntry>,
    /// Cache configuration
    config: CacheConfig,
    /// Current L1 cache size
    l1_current_bytes: usize,
    /// Current L2 cache size
    l2_current_bytes: usize,
    /// Global access counter for LRU
    access_counter: u64,
    /// Cache creation time
    created_at: Instant,
}

impl ModelRegistry {
    /// Create a new model registry
    #[must_use]
    pub fn new(config: CacheConfig) -> Self {
        Self {
            l1_cache: HashMap::new(),
            l2_cache: HashMap::new(),
            config,
            l1_current_bytes: 0,
            l2_current_bytes: 0,
            access_counter: 0,
            created_at: Instant::now(),
        }
    }

    /// Insert a model into L1 cache
    pub fn insert_l1(&mut self, name: String, entry: CacheEntry) {
        let size = entry.data.size();

        // Evict if necessary
        while self.l1_current_bytes + size > self.config.l1_max_bytes && !self.l1_cache.is_empty() {
            if let Some(evict_key) = self.find_eviction_candidate_l1() {
                self.evict_l1(&evict_key);
            } else {
                break;
            }
        }

        if let Some(old) = self.l1_cache.insert(name, entry) {
            self.l1_current_bytes -= old.data.size();
        }
        self.l1_current_bytes += size;
    }

    /// Insert a model into L2 cache
    pub fn insert_l2(&mut self, name: String, entry: CacheEntry) {
        let size = entry.data.size();

        // Evict if necessary
        while self.l2_current_bytes + size > self.config.l2_max_bytes && !self.l2_cache.is_empty() {
            if let Some(evict_key) = self.find_eviction_candidate_l2() {
                self.evict_l2(&evict_key);
            } else {
                break;
            }
        }

        if let Some(old) = self.l2_cache.insert(name, entry) {
            self.l2_current_bytes -= old.data.size();
        }
        self.l2_current_bytes += size;
    }

    /// Get a model by name (checks L1, then L2)
    pub fn get(&mut self, name: &str) -> Option<&CacheEntry> {
        self.access_counter += 1;
        let timestamp = self.access_counter;

        // Check L1 first
        if let Some(entry) = self.l1_cache.get_mut(name) {
            entry.stats.record_hit(100, timestamp);
            return self.l1_cache.get(name);
        }

        // Check L2
        if let Some(entry) = self.l2_cache.get_mut(name) {
            entry.stats.record_hit(1000, timestamp);
            return self.l2_cache.get(name);
        }

        None
    }

    /// Check if a model exists in cache
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.l1_cache.contains_key(name) || self.l2_cache.contains_key(name)
    }

    /// Get cache tier for a model
    #[must_use]
    pub fn get_tier(&self, name: &str) -> Option<CacheTier> {
        if self.l1_cache.contains_key(name) {
            Some(CacheTier::L1Hot)
        } else if self.l2_cache.contains_key(name) {
            Some(CacheTier::L2Warm)
        } else {
            None
        }
    }

    /// Remove a model from cache
    pub fn remove(&mut self, name: &str) {
        if let Some(entry) = self.l1_cache.remove(name) {
            self.l1_current_bytes -= entry.data.size();
        }
        if let Some(entry) = self.l2_cache.remove(name) {
            self.l2_current_bytes -= entry.data.size();
        }
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.l1_cache.clear();
        self.l2_cache.clear();
        self.l1_current_bytes = 0;
        self.l2_current_bytes = 0;
    }

    /// Get cache statistics
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        let l1_entries = self.l1_cache.len();
        let l2_entries = self.l2_cache.len();

        let (l1_hits, l1_misses) = self.l1_cache.values().fold((0, 0), |(h, m), e| {
            (h + e.stats.hit_count, m + e.stats.miss_count)
        });

        let (l2_hits, l2_misses) = self.l2_cache.values().fold((0, 0), |(h, m), e| {
            (h + e.stats.hit_count, m + e.stats.miss_count)
        });

        CacheStats {
            l1_entries,
            l1_bytes: self.l1_current_bytes,
            l1_hits,
            l1_misses,
            l2_entries,
            l2_bytes: self.l2_current_bytes,
            l2_hits,
            l2_misses,
            uptime: self.created_at.elapsed(),
        }
    }

    /// List all cached models
    #[must_use]
    pub fn list(&self) -> Vec<ModelInfo> {
        let mut models = Vec::new();

        for (name, entry) in &self.l1_cache {
            models.push(ModelInfo {
                name: name.clone(),
                model_type: entry.model_type,
                size_bytes: entry.data.size(),
                is_bundled: false,
                is_cached: true,
                cache_tier: Some(CacheTier::L1Hot),
            });
        }

        for (name, entry) in &self.l2_cache {
            if !self.l1_cache.contains_key(name) {
                models.push(ModelInfo {
                    name: name.clone(),
                    model_type: entry.model_type,
                    size_bytes: entry.data.size(),
                    is_bundled: false,
                    is_cached: true,
                    cache_tier: Some(CacheTier::L2Warm),
                });
            }
        }

        models
    }

    // Find eviction candidate in L1 based on policy
    fn find_eviction_candidate_l1(&self) -> Option<String> {
        match self.config.eviction_policy {
            EvictionPolicy::Fixed => None,
            EvictionPolicy::LRU | EvictionPolicy::Clock => self
                .l1_cache
                .iter()
                .min_by_key(|(_, e)| e.stats.last_access)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::LFU => self
                .l1_cache
                .iter()
                .min_by_key(|(_, e)| e.stats.hit_count)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::ARC => {
                // Simplified ARC: balance recency and frequency
                self.l1_cache
                    .iter()
                    .min_by_key(|(_, e)| {
                        e.stats.last_access.saturating_add(e.stats.hit_count * 100)
                    })
                    .map(|(k, _)| k.clone())
            }
        }
    }

    // Find eviction candidate in L2
    fn find_eviction_candidate_l2(&self) -> Option<String> {
        match self.config.eviction_policy {
            EvictionPolicy::Fixed => None,
            _ => self
                .l2_cache
                .iter()
                .min_by_key(|(_, e)| e.stats.last_access)
                .map(|(k, _)| k.clone()),
        }
    }

    // Evict from L1
    fn evict_l1(&mut self, key: &str) {
        if let Some(entry) = self.l1_cache.remove(key) {
            self.l1_current_bytes -= entry.data.size();
        }
    }

    // Evict from L2
    fn evict_l2(&mut self, key: &str) {
        if let Some(entry) = self.l2_cache.remove(key) {
            self.l2_current_bytes -= entry.data.size();
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of L1 entries
    pub l1_entries: usize,
    /// L1 cache size in bytes
    pub l1_bytes: usize,
    /// L1 hit count
    pub l1_hits: u64,
    /// L1 miss count
    pub l1_misses: u64,
    /// Number of L2 entries
    pub l2_entries: usize,
    /// L2 cache size in bytes
    pub l2_bytes: usize,
    /// L2 hit count
    pub l2_hits: u64,
    /// L2 miss count
    pub l2_misses: u64,
    /// Cache uptime
    pub uptime: Duration,
}

impl CacheStats {
    /// Get overall hit rate
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.l1_hits + self.l2_hits;
        let total_misses = self.l1_misses + self.l2_misses;
        let total = total_hits + total_misses;
        if total == 0 {
            0.0
        } else {
            total_hits as f64 / total as f64
        }
    }

    /// Get total cache size
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.l1_bytes + self.l2_bytes
    }

    /// Get total entries
    #[must_use]
    pub fn total_entries(&self) -> usize {
        self.l1_entries + self.l2_entries
    }
}

#[cfg(test)]
mod tests;
