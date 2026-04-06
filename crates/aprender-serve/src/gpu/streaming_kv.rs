//! Streaming KV Cache (PMAT-802)
//!
//! Memory-efficient key-value cache for transformer inference.

// ============================================================================
// M6: Memory Efficiency - StreamingKVCache
// ============================================================================

/// Streaming KV cache for memory-efficient inference
///
/// Implements a bounded circular buffer for key-value cache that allows
/// efficient inference on long sequences without unbounded memory growth.
///
/// ## Memory Bound
///
/// Total memory = num_layers * max_positions * num_heads * head_dim * 2 (K+V) * sizeof(f32)
///
/// For 7B model (32 layers, 2048 positions, 32 heads, 128 head_dim):
/// = 32 * 2048 * 32 * 128 * 2 * 4 = ~2GB
///
/// ## Usage
///
/// ```rust,ignore
/// let mut cache = StreamingKVCache::new(32, 2048, 32, 128);
/// cache.append(0, &key_vec, &value_vec);
/// let (keys, values) = cache.get_range(0, 0, 100);
/// ```
pub struct StreamingKVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum cached positions (context length)
    max_positions: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Key cache per layer [num_layers][max_positions * num_heads * head_dim]
    keys: Vec<Vec<f32>>,
    /// Value cache per layer
    values: Vec<Vec<f32>>,
    /// Current write position (circular)
    position: usize,
    /// Number of valid positions cached
    valid_positions: usize,
}

impl StreamingKVCache {
    /// Create a new streaming KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `max_positions` - Maximum context length to cache
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    #[must_use]
    pub fn new(num_layers: usize, max_positions: usize, num_heads: usize, head_dim: usize) -> Self {
        let kv_size = max_positions * num_heads * head_dim;
        Self {
            num_layers,
            max_positions,
            num_heads,
            head_dim,
            keys: vec![vec![0.0f32; kv_size]; num_layers],
            values: vec![vec![0.0f32; kv_size]; num_layers],
            position: 0,
            valid_positions: 0,
        }
    }

    /// Append key-value pair for a single position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `key` - Key vector [num_heads * head_dim]
    /// * `value` - Value vector [num_heads * head_dim]
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or key/value dimensions are wrong.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert_eq!(key.len(), kv_dim, "Key dimension mismatch");
        assert_eq!(value.len(), kv_dim, "Value dimension mismatch");

        let offset = self.position * kv_dim;
        self.keys[layer][offset..offset + kv_dim].copy_from_slice(key);
        self.values[layer][offset..offset + kv_dim].copy_from_slice(value);

        // Update position only after last layer
        if layer == self.num_layers - 1 {
            self.position = (self.position + 1) % self.max_positions;
            self.valid_positions = (self.valid_positions + 1).min(self.max_positions);
        }
    }

    /// Get keys and values for a range of positions
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `start` - Start position (inclusive)
    /// * `end` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) slices
    #[must_use]
    pub fn get_range(&self, layer: usize, start: usize, end: usize) -> (&[f32], &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        (
            &self.keys[layer][start_offset..end_offset],
            &self.values[layer][start_offset..end_offset],
        )
    }

    /// Get all valid cached keys and values for a layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) for all valid positions
    #[must_use]
    pub fn get_valid(&self, layer: usize) -> (&[f32], &[f32]) {
        self.get_range(layer, 0, self.valid_positions)
    }

    /// Get current number of valid cached positions
    #[must_use]
    pub fn len(&self) -> usize {
        self.valid_positions
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.valid_positions == 0
    }

    /// Get maximum positions (context length)
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Reset the cache
    pub fn clear(&mut self) {
        self.position = 0;
        self.valid_positions = 0;
        // Note: We don't zero the memory for performance
    }

    /// Calculate memory usage in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let kv_size = self.max_positions * self.num_heads * self.head_dim;
        // Keys + Values, f32 = 4 bytes
        self.num_layers * kv_size * 2 * 4
    }

    /// Calculate memory usage in megabytes
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

/// Streaming KV cache with FP16 storage for memory efficiency (M12)
///
/// Uses half-precision (FP16) storage to halve memory usage compared to FP32,
/// enabling support for ultra-long contexts (65536+) on consumer GPUs.
///
/// # Memory Efficiency
///
/// For 65536 context with 7B model config:
/// - FP32: 32 layers × 65536 pos × 32 heads × 128 dim × 2 × 4 bytes = 68.72 GB
/// - FP16: 32 layers × 65536 pos × 32 heads × 128 dim × 2 × 2 bytes = 34.36 GB
///
/// # Example
///
/// ```
/// use realizar::gpu::StreamingKVCacheFp16;
///
/// let mut cache = StreamingKVCacheFp16::new(32, 65536, 32, 128);
/// assert!(cache.memory_mb() < 36000.0); // < 36 GB
/// ```
pub struct StreamingKVCacheFp16 {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum cached positions (context length)
    max_positions: usize,
    /// Number of attention heads
    num_heads: usize,
    /// Dimension per head
    head_dim: usize,
    /// Key cache per layer [num_layers][max_positions * num_heads * head_dim] stored as FP16 bits
    keys: Vec<Vec<u16>>,
    /// Value cache per layer stored as FP16 bits
    values: Vec<Vec<u16>>,
    /// Current write position (circular)
    position: usize,
    /// Number of valid positions cached
    valid_positions: usize,
}

impl StreamingKVCacheFp16 {
    /// Create a new FP16 streaming KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `max_positions` - Maximum context length to cache
    /// * `num_heads` - Number of attention heads
    /// * `head_dim` - Dimension per attention head
    #[must_use]
    pub fn new(num_layers: usize, max_positions: usize, num_heads: usize, head_dim: usize) -> Self {
        let kv_size = max_positions * num_heads * head_dim;
        Self {
            num_layers,
            max_positions,
            num_heads,
            head_dim,
            keys: vec![vec![0u16; kv_size]; num_layers],
            values: vec![vec![0u16; kv_size]; num_layers],
            position: 0,
            valid_positions: 0,
        }
    }

    /// Convert f32 to FP16 bits
    #[inline]
    pub(crate) fn f32_to_f16(value: f32) -> u16 {
        half::f16::from_f32(value).to_bits()
    }

    /// Convert FP16 bits to f32
    #[inline]
    pub(crate) fn f16_to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }

    /// Append key-value pair for a single position (FP32 input, stored as FP16)
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0-indexed)
    /// * `key` - Key vector [num_heads * head_dim] as FP32
    /// * `value` - Value vector [num_heads * head_dim] as FP32
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or key/value dimensions are wrong.
    pub fn append(&mut self, layer: usize, key: &[f32], value: &[f32]) {
        let kv_dim = self.num_heads * self.head_dim;
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert_eq!(key.len(), kv_dim, "Key dimension mismatch");
        assert_eq!(value.len(), kv_dim, "Value dimension mismatch");

        let offset = self.position * kv_dim;

        // Convert FP32 to FP16 and store
        for (i, &k) in key.iter().enumerate() {
            self.keys[layer][offset + i] = Self::f32_to_f16(k);
        }
        for (i, &v) in value.iter().enumerate() {
            self.values[layer][offset + i] = Self::f32_to_f16(v);
        }

        // Update position only after last layer
        if layer == self.num_layers - 1 {
            self.position = (self.position + 1) % self.max_positions;
            self.valid_positions = (self.valid_positions + 1).min(self.max_positions);
        }
    }

    /// Get keys and values for a range of positions (converted back to FP32)
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `start` - Start position (inclusive)
    /// * `end` - End position (exclusive)
    ///
    /// # Returns
    ///
    /// Tuple of (keys, values) as Vec<f32>
    #[must_use]
    pub fn get_range_f32(&self, layer: usize, start: usize, end: usize) -> (Vec<f32>, Vec<f32>) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        let keys: Vec<f32> = self.keys[layer][start_offset..end_offset]
            .iter()
            .map(|&bits| Self::f16_to_f32(bits))
            .collect();

        let values: Vec<f32> = self.values[layer][start_offset..end_offset]
            .iter()
            .map(|&bits| Self::f16_to_f32(bits))
            .collect();

        (keys, values)
    }

    /// Get raw FP16 keys and values for a range of positions
    #[must_use]
    pub fn get_range_raw(&self, layer: usize, start: usize, end: usize) -> (&[u16], &[u16]) {
        let kv_dim = self.num_heads * self.head_dim;
        let start_offset = start * kv_dim;
        let end_offset = end * kv_dim;

        (
            &self.keys[layer][start_offset..end_offset],
            &self.values[layer][start_offset..end_offset],
        )
    }

    /// Get all valid cached keys and values for a layer (as FP32)
    #[must_use]
    pub fn get_valid_f32(&self, layer: usize) -> (Vec<f32>, Vec<f32>) {
        self.get_range_f32(layer, 0, self.valid_positions)
    }

    /// Get current number of valid cached positions
    #[must_use]
    pub fn len(&self) -> usize {
        self.valid_positions
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.valid_positions == 0
    }

    /// Get maximum positions (context length)
    #[must_use]
    pub fn max_positions(&self) -> usize {
        self.max_positions
    }

    /// Reset the cache
    pub fn clear(&mut self) {
        self.position = 0;
        self.valid_positions = 0;
    }

    /// Calculate memory usage in bytes (half of FP32 version)
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        let kv_size = self.max_positions * self.num_heads * self.head_dim;
        // Keys + Values, u16 (FP16) = 2 bytes
        self.num_layers * kv_size * 2 * 2
    }

    /// Calculate memory usage in megabytes
    #[must_use]
    pub fn memory_mb(&self) -> f64 {
        self.memory_bytes() as f64 / (1024.0 * 1024.0)
    }
}

include!("streaming_kv_streaming.rs");

// =========================================================================
// FALSIFY-KV: kv-cache-sizing-v1.yaml contract (realizar StreamingKVCache)
//
// Five-Whys (PMAT-354):
//   Why 1: realizar had KV cache tests but zero FALSIFY-KV-* tests
//   Why 2: tests verify append/get, not sizing formula invariants
//   Why 3: no mapping from kv-cache-sizing-v1.yaml to realizar test names
//   Why 4: realizar predates the provable-contracts YAML convention
//   Why 5: KV sizing was "obviously correct" (simple multiplication)
//
// References:
//   - provable-contracts/contracts/kv-cache-sizing-v1.yaml
// =========================================================================
#[cfg(test)]
mod kv_contract_tests {
    use super::*;

    /// FALSIFY-KV-001: Per-token KV bytes — 2 * n_kv * d_k * sizeof(f32)
    ///
    /// Formula: memory = num_layers * max_positions * num_heads * head_dim * 2 * 4
    #[test]
    fn falsify_kv_001_memory_formula() {
        let test_cases = vec![
            // (layers, max_pos, heads, head_dim)
            (1, 1, 1, 1),
            (32, 2048, 32, 128), // 7B model
            (40, 4096, 40, 128), // 13B model
            (1, 512, 8, 64),
        ];

        for (nl, mp, nh, hd) in test_cases {
            let cache = StreamingKVCache::new(nl, mp, nh, hd);
            let expected = nl * mp * nh * hd * 2 * 4;
            assert_eq!(
                cache.memory_bytes(),
                expected,
                "FALSIFIED KV-001: memory_bytes({nl}, {mp}, {nh}, {hd}) = {}, expected {expected}",
                cache.memory_bytes()
            );
        }
    }

    /// FALSIFY-KV-002: Monotonic — longer sequence => more memory
    #[test]
    fn falsify_kv_002_monotonic_sequence_length() {
        let seq_lengths = [128, 256, 512, 1024, 2048, 4096];
        let mut prev_bytes = 0;

        for &sl in &seq_lengths {
            let cache = StreamingKVCache::new(32, sl, 32, 128);
            let bytes = cache.memory_bytes();
            assert!(
                bytes > prev_bytes,
                "FALSIFIED KV-002: memory({sl}) = {bytes} not > memory(prev) = {prev_bytes}"
            );
            prev_bytes = bytes;
        }
    }

    /// FALSIFY-KV-002b: Monotonic — more layers => more memory
    #[test]
    fn falsify_kv_002b_monotonic_layers() {
        let layer_counts = [1, 8, 16, 32, 40, 64];
        let mut prev_bytes = 0;

        for &nl in &layer_counts {
            let cache = StreamingKVCache::new(nl, 2048, 32, 128);
            let bytes = cache.memory_bytes();
            assert!(
                bytes > prev_bytes,
                "FALSIFIED KV-002b: memory(layers={nl}) = {bytes} not > {prev_bytes}"
            );
            prev_bytes = bytes;
        }
    }

    /// FALSIFY-KV-001b: FP16 cache uses exactly half the memory of FP32
    #[test]
    fn falsify_kv_001b_fp16_half_memory() {
        let nl = 32;
        let mp = 2048;
        let nh = 32;
        let hd = 128;

        let f32_cache = StreamingKVCache::new(nl, mp, nh, hd);
        let f16_cache = StreamingKVCacheFp16::new(nl, mp, nh, hd);

        let f32_bytes = f32_cache.memory_bytes();
        let f16_bytes = f16_cache.memory_bytes();

        assert_eq!(
            f16_bytes * 2,
            f32_bytes,
            "FALSIFIED KV-001b: FP16 ({f16_bytes}) * 2 != FP32 ({f32_bytes})"
        );
    }
}
