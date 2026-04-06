
// ============================================================================
// PREFIX CACHING (per llama.cpp)
// ============================================================================
//
// Prefix caching allows reusing KV cache values for common prompt prefixes.
// When multiple requests share the same prefix tokens (e.g., system prompts),
// the KV cache for those tokens is computed once and reused.
//
// Benefits:
// - Reduces time-to-first-token for common prompts
// - Saves computation for repeated system instructions
// - Enables efficient multi-turn conversation handling
// ============================================================================

/// Hash type for prefix cache lookup
pub type PrefixHash = u64;

/// Compute hash for a token sequence (used for prefix lookup)
pub fn compute_prefix_hash(tokens: &[u32]) -> PrefixHash {
    // Simple FNV-1a hash for token sequences
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for &token in tokens {
        hash ^= token as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3); // FNV prime
    }
    hash
}

/// Cached prefix entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedPrefix {
    /// Hash of the prefix tokens
    pub hash: PrefixHash,
    /// Number of tokens in prefix
    pub num_tokens: usize,
    /// Page IDs containing the cached KV values
    pub page_ids: Vec<PageId>,
    /// Reference count (number of sequences using this prefix)
    pub ref_count: usize,
    /// Last access timestamp (for LRU eviction)
    pub last_access: u64,
}

impl CachedPrefix {
    /// Create new cached prefix
    pub fn new(hash: PrefixHash, num_tokens: usize, page_ids: Vec<PageId>) -> Self {
        Self {
            hash,
            num_tokens,
            page_ids,
            ref_count: 1,
            last_access: 0,
        }
    }

    /// Increment reference count
    pub fn add_ref(&mut self) {
        self.ref_count += 1;
    }

    /// Decrement reference count
    pub fn remove_ref(&mut self) -> bool {
        self.ref_count = self.ref_count.saturating_sub(1);
        self.ref_count == 0
    }
}

/// Prefix cache for KV cache reuse
///
/// Per llama.cpp's prompt cache: stores computed KV values for common
/// prompt prefixes, enabling fast cache hits for repeated system prompts.
pub struct PrefixCache {
    /// Cached prefixes by hash
    cache: HashMap<PrefixHash, CachedPrefix>,
    /// Maximum number of cached prefixes
    max_entries: usize,
    /// Access counter for LRU
    access_counter: u64,
    /// Statistics
    stats: PrefixCacheStats,
}

/// Statistics for prefix cache
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrefixCacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Total prefixes cached
    pub prefixes_cached: u64,
    /// Prefixes evicted
    pub prefixes_evicted: u64,
    /// Tokens saved (not recomputed)
    pub tokens_saved: u64,
}

impl PrefixCacheStats {
    /// Hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl PrefixCache {
    /// Create new prefix cache
    pub fn new(max_entries: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_entries),
            max_entries,
            access_counter: 0,
            stats: PrefixCacheStats::default(),
        }
    }

    /// Look up cached prefix by hash
    pub fn lookup(&mut self, hash: PrefixHash) -> Option<&CachedPrefix> {
        if let Some(entry) = self.cache.get_mut(&hash) {
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            self.stats.hits += 1;
            // Return immutable reference
            self.cache.get(&hash)
        } else {
            self.stats.misses += 1;
            None
        }
    }

    /// Look up cached prefix by tokens
    pub fn lookup_tokens(&mut self, tokens: &[u32]) -> Option<&CachedPrefix> {
        let hash = compute_prefix_hash(tokens);
        self.lookup(hash)
    }

    /// Check if prefix is cached (without updating stats)
    pub fn contains(&self, hash: PrefixHash) -> bool {
        self.cache.contains_key(&hash)
    }

    /// Insert cached prefix
    pub fn insert(&mut self, prefix: CachedPrefix) -> bool {
        let hash = prefix.hash;

        // Evict if at capacity
        if self.cache.len() >= self.max_entries && !self.cache.contains_key(&hash) {
            self.evict_lru();
        }

        if self.cache.len() < self.max_entries {
            self.stats.prefixes_cached += 1;
            self.stats.tokens_saved += prefix.num_tokens as u64;
            self.cache.insert(hash, prefix);
            true
        } else {
            false
        }
    }

    /// Add reference to cached prefix
    pub fn add_ref(&mut self, hash: PrefixHash) -> bool {
        if let Some(entry) = self.cache.get_mut(&hash) {
            entry.add_ref();
            self.access_counter += 1;
            entry.last_access = self.access_counter;
            true
        } else {
            false
        }
    }

    /// Remove reference from cached prefix
    /// Returns true if prefix was removed (no more references)
    pub fn remove_ref(&mut self, hash: PrefixHash) -> bool {
        if let Some(entry) = self.cache.get_mut(&hash) {
            if entry.remove_ref() {
                // No more references, remove from cache
                self.cache.remove(&hash);
                return true;
            }
        }
        false
    }

    /// Evict least recently used prefix
    fn evict_lru(&mut self) {
        if let Some((&hash, _)) = self
            .cache
            .iter()
            .filter(|(_, v)| v.ref_count == 0)
            .min_by_key(|(_, v)| v.last_access)
        {
            self.cache.remove(&hash);
            self.stats.prefixes_evicted += 1;
        }
    }

    /// Get number of cached prefixes
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    /// Get cache statistics
    pub fn stats(&self) -> &PrefixCacheStats {
        &self.stats
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_counter = 0;
    }

    /// Get cache utilization (0.0 to 1.0)
    pub fn utilization(&self) -> f64 {
        if self.max_entries == 0 {
            0.0
        } else {
            self.cache.len() as f64 / self.max_entries as f64
        }
    }
}

impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(100)
    }
}

// ============================================================================
// KV CACHE QUANTIZATION (per llama.cpp Q8/Q4 KV)
// ============================================================================
//
// KV cache quantization reduces memory usage during inference:
// - Q8_0: 8-bit quantization, ~2x memory reduction, minimal quality loss
// - Q4_0: 4-bit quantization, ~4x memory reduction, some quality loss
//
// llama.cpp uses this for long-context inference where KV cache dominates memory.
// ============================================================================

/// KV cache quantization type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KvQuantType {
    /// Full precision (32-bit float)
    #[default]
    FP32,
    /// 8-bit quantization (Q8_0 format)
    Q8,
    /// 4-bit quantization (Q4_0 format)
    Q4,
}

impl KvQuantType {
    /// Bytes per value for this quantization type
    pub fn bytes_per_value(&self) -> f32 {
        match self {
            Self::FP32 => 4.0,
            Self::Q8 => 1.0, // 8 bits = 1 byte
            Self::Q4 => 0.5, // 4 bits = 0.5 bytes
        }
    }

    /// Memory reduction factor compared to FP32
    pub fn memory_reduction(&self) -> f32 {
        4.0 / self.bytes_per_value()
    }
}

/// Block size for KV quantization (matches GGML)
pub const KV_QUANT_BLOCK_SIZE: usize = 32;

/// Q8_0 quantized block for KV cache
#[derive(Debug, Clone)]
pub struct Q8KvBlock {
    /// Scale factor for the block
    pub scale: f32,
    /// Quantized values (int8, stored as i8)
    pub quants: [i8; KV_QUANT_BLOCK_SIZE],
}

impl Q8KvBlock {
    /// Create empty block
    pub fn new() -> Self {
        Self {
            scale: 0.0,
            quants: [0; KV_QUANT_BLOCK_SIZE],
        }
    }

    /// Quantize float values to Q8
    pub fn quantize(values: &[f32; KV_QUANT_BLOCK_SIZE]) -> Self {
        // Find max absolute value for scale
        let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        if amax < 1e-10 {
            return Self::new();
        }

        let scale = amax / 127.0;
        let inv_scale = 1.0 / scale;

        let mut quants = [0i8; KV_QUANT_BLOCK_SIZE];
        for (i, &v) in values.iter().enumerate() {
            let q = (v * inv_scale).round() as i32;
            quants[i] = q.clamp(-127, 127) as i8;
        }

        Self { scale, quants }
    }

    /// Dequantize to float values
    pub fn dequantize(&self) -> [f32; KV_QUANT_BLOCK_SIZE] {
        let mut result = [0.0f32; KV_QUANT_BLOCK_SIZE];
        for (i, &q) in self.quants.iter().enumerate() {
            result[i] = q as f32 * self.scale;
        }
        result
    }
}

impl Default for Q8KvBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Q4_0 quantized block for KV cache
#[derive(Debug, Clone)]
pub struct Q4KvBlock {
    /// Scale factor for the block
    pub scale: f32,
    /// Quantized values (4-bit, packed 2 per byte)
    pub quants: [u8; KV_QUANT_BLOCK_SIZE / 2],
}

impl Q4KvBlock {
    /// Create empty block
    pub fn new() -> Self {
        Self {
            scale: 0.0,
            quants: [0; KV_QUANT_BLOCK_SIZE / 2],
        }
    }

    /// Quantize float values to Q4
    pub fn quantize(values: &[f32; KV_QUANT_BLOCK_SIZE]) -> Self {
        // Find max absolute value for scale
        let amax = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        if amax < 1e-10 {
            return Self::new();
        }

        // Q4_0 uses signed 4-bit: -8 to 7
        let scale = amax / 7.0;
        let inv_scale = 1.0 / scale;

        let mut quants = [0u8; KV_QUANT_BLOCK_SIZE / 2];
        for i in 0..(KV_QUANT_BLOCK_SIZE / 2) {
            let v0 = values[i * 2];
            let v1 = values[i * 2 + 1];

            // Quantize to -8..7 range, then shift to 0..15 for unsigned storage
            let q0 = ((v0 * inv_scale).round() as i32).clamp(-8, 7) + 8;
            let q1 = ((v1 * inv_scale).round() as i32).clamp(-8, 7) + 8;

            // Pack two 4-bit values into one byte
            quants[i] = ((q1 as u8) << 4) | (q0 as u8);
        }

        Self { scale, quants }
    }

    /// Dequantize to float values
    pub fn dequantize(&self) -> [f32; KV_QUANT_BLOCK_SIZE] {
        let mut result = [0.0f32; KV_QUANT_BLOCK_SIZE];

        for (i, &packed) in self.quants.iter().enumerate() {
            // Unpack two 4-bit values
            let q0 = (packed & 0x0F) as i32 - 8;
            let q1 = ((packed >> 4) & 0x0F) as i32 - 8;

            result[i * 2] = q0 as f32 * self.scale;
            result[i * 2 + 1] = q1 as f32 * self.scale;
        }

        result
    }
}

impl Default for Q4KvBlock {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantized KV cache data for a single page
#[derive(Debug, Clone)]
pub enum QuantizedKvData {
    /// Full precision storage
    FP32 {
        /// Key cache: [block_size, num_heads, head_dim]
        keys: Vec<f32>,
        /// Value cache: [block_size, num_heads, head_dim]
        values: Vec<f32>,
    },
    /// Q8 quantized storage
    Q8 {
        /// Quantized key blocks
        key_blocks: Vec<Q8KvBlock>,
        /// Quantized value blocks
        value_blocks: Vec<Q8KvBlock>,
    },
    /// Q4 quantized storage
    Q4 {
        /// Quantized key blocks
        key_blocks: Vec<Q4KvBlock>,
        /// Quantized value blocks
        value_blocks: Vec<Q4KvBlock>,
    },
}
