//! Key-Value cache for efficient autoregressive generation
//!
//! Stores past key and value tensors to avoid recomputation during generation.
//! Enables O(1) per-token computation instead of O(n) for sequence of length n.
//!
//! ## Cache Types
//!
//! - [`KVCache`] - Basic KV cache with row-major storage
//! - [`OptimizedKVCache`] - Optimized cache with transposed V for better memory access
//!
//! ## Attention Functions
//!
//! - [`attention_with_cache`] - Scaled dot-product attention using cached K/V
//! - [`attention_with_transposed_v`] - Attention with transposed V storage

use super::{simd_dot, simd_softmax};

/// Key-Value cache for autoregressive generation
///
/// Stores past key and value tensors to avoid recomputation during generation.
/// Enables O(1) per-token computation instead of O(n) for sequence of length n.
///
/// # Example
///
/// ```
/// use realizar::inference::KVCache;
///
/// let mut cache = KVCache::new(12, 768, 2048);  // 12 layers, 768 hidden, 2048 max seq
/// assert!(cache.is_empty());
///
/// // Store KV for position 0
/// let k = vec![0.1; 768];
/// let v = vec![0.2; 768];
/// cache.store(0, &k, &v);
/// cache.advance();
///
/// assert_eq!(cache.len(), 1);
/// ```
#[derive(Clone)]
pub struct KVCache {
    /// Key cache: [num_layers, max_seq_len, kv_dim]
    k_cache: Vec<Vec<f32>>,
    /// Value cache: [num_layers, max_seq_len, kv_dim]
    v_cache: Vec<Vec<f32>>,
    /// Current sequence length
    seq_len: usize,
    /// Hidden dimension per head
    hidden_dim: usize,
}

impl KVCache {
    /// Create a new KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers
    /// * `hidden_dim` - Hidden dimension (total across all heads)
    /// * `max_seq_len` - Maximum sequence length to cache
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let k_cache = vec![vec![0.0; max_seq_len * hidden_dim]; num_layers];
        let v_cache = vec![vec![0.0; max_seq_len * hidden_dim]; num_layers];
        Self {
            k_cache,
            v_cache,
            seq_len: 0,
            hidden_dim,
        }
    }

    /// Store a new KV pair for a layer
    ///
    /// Stores the key and value vectors at the current sequence position.
    /// Does nothing if the cache is full.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index (0 to num_layers-1)
    /// * `k` - Key vector of length `hidden_dim`
    /// * `v` - Value vector of length `hidden_dim`
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        let start = self.seq_len * self.hidden_dim;
        let end = start + self.hidden_dim;

        if end <= self.k_cache[layer].len() {
            self.k_cache[layer][start..end].copy_from_slice(k);
            self.v_cache[layer][start..end].copy_from_slice(v);
        }
    }

    /// Advance the sequence position
    ///
    /// Call after storing all layers for the current position.
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Get cached keys for a layer
    ///
    /// Returns all keys from position 0 to current seq_len.
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        &self.k_cache[layer][..self.seq_len * self.hidden_dim]
    }

    /// Get cached values for a layer
    ///
    /// Returns all values from position 0 to current seq_len.
    #[must_use]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        &self.v_cache[layer][..self.seq_len * self.hidden_dim]
    }

    /// Get current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset the cache for a new sequence
    ///
    /// Clears the sequence position but keeps allocated memory.
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }
}

/// Compute attention with KV cache
///
/// Computes scaled dot-product attention: softmax(QK^T / sqrt(d)) V
/// Uses cached K and V from previous positions.
///
/// # Arguments
///
/// * `q` - Query vector [hidden_dim]
/// * `k_cache` - Cached keys [seq_len × hidden_dim]
/// * `v_cache` - Cached values [seq_len × hidden_dim]
/// * `current_k` - Key for current position [hidden_dim]
/// * `current_v` - Value for current position [hidden_dim]
/// * `num_heads` - Number of attention heads
///
/// # Returns
///
/// Attention output [hidden_dim]
///
/// # Example
///
/// ```
/// use realizar::inference::attention_with_cache;
///
/// let hidden_dim = 64;
/// let num_heads = 2;
///
/// let q = vec![0.1; hidden_dim];
/// let k_cache: Vec<f32> = vec![];  // No cached positions
/// let v_cache: Vec<f32> = vec![];
/// let current_k = vec![0.1; hidden_dim];
/// let current_v = vec![0.2; hidden_dim];
///
/// let output = attention_with_cache(&q, &k_cache, &v_cache, &current_k, &current_v, num_heads);
/// assert_eq!(output.len(), hidden_dim);
/// ```
#[must_use]
pub fn attention_with_cache(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    current_k: &[f32],
    current_v: &[f32],
    num_heads: usize,
) -> Vec<f32> {
    let hidden_dim = q.len();
    let head_dim = hidden_dim / num_heads;
    let cache_len = if hidden_dim > 0 {
        k_cache.len() / hidden_dim
    } else {
        0
    };
    let total_len = cache_len + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; hidden_dim];

    // Process each head independently
    for h in 0..num_heads {
        let head_offset = h * head_dim;
        let q_head = &q[head_offset..head_offset + head_dim];

        // Compute attention scores for all positions
        let mut scores = Vec::with_capacity(total_len);

        // Scores against cached K
        for pos in 0..cache_len {
            let k_start = pos * hidden_dim + head_offset;
            let k_head = &k_cache[k_start..k_start + head_dim];
            let score = simd_dot(q_head, k_head) * scale;
            scores.push(score);
        }

        // Score against current K
        let current_k_head = &current_k[head_offset..head_offset + head_dim];
        scores.push(simd_dot(q_head, current_k_head) * scale);

        // Softmax
        simd_softmax(&mut scores);

        // Weighted sum of V
        let out_head = &mut output[head_offset..head_offset + head_dim];

        for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
            let v_start = pos * hidden_dim + head_offset;
            let v_head = &v_cache[v_start..v_start + head_dim];
            for (i, &v) in v_head.iter().enumerate() {
                out_head[i] += weight * v;
            }
        }

        // Add contribution from current V
        let current_v_head = &current_v[head_offset..head_offset + head_dim];
        let current_weight = scores[cache_len];
        for (i, &v) in current_v_head.iter().enumerate() {
            out_head[i] += current_weight * v;
        }
    }

    output
}

/// Optimized KV cache with contiguous storage
///
/// Uses transposed V storage for better memory access during attention.
/// The value cache is stored as [hidden_dim × max_seq_len] instead of
/// [max_seq_len × hidden_dim], enabling better cache locality when
/// computing attention.
#[derive(Clone)]
pub struct OptimizedKVCache {
    /// Key cache: [num_layers][seq_len × hidden_dim]
    k_cache: Vec<Vec<f32>>,
    /// Value cache (transposed): [num_layers][hidden_dim × seq_len]
    v_cache: Vec<Vec<f32>>,
    /// Current sequence length
    seq_len: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
}

impl OptimizedKVCache {
    /// Create a new optimized KV cache
    #[must_use]
    pub fn new(num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let k_cache = vec![vec![0.0; max_seq_len * hidden_dim]; num_layers];
        let v_cache = vec![vec![0.0; hidden_dim * max_seq_len]; num_layers];
        Self {
            k_cache,
            v_cache,
            seq_len: 0,
            hidden_dim,
            max_seq_len,
        }
    }

    /// Store a new KV pair with transposed V storage
    ///
    /// Does nothing if the cache is at maximum capacity.
    pub fn store(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if self.seq_len >= self.max_seq_len {
            return;
        }

        // Store K in normal format
        let k_start = self.seq_len * self.hidden_dim;
        let k_end = k_start + self.hidden_dim;
        self.k_cache[layer][k_start..k_end].copy_from_slice(k);

        // Store V transposed: v[i] goes to v_cache[i * max_seq_len + seq_len]
        for (i, &val) in v.iter().enumerate() {
            self.v_cache[layer][i * self.max_seq_len + self.seq_len] = val;
        }
    }

    /// Advance the sequence position
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// Get cached keys for a layer
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        &self.k_cache[layer][..self.seq_len * self.hidden_dim]
    }

    /// Get cached values (transposed) for a layer
    #[must_use]
    pub fn get_v_transposed(&self, layer: usize) -> &[f32] {
        &self.v_cache[layer]
    }

    /// Get current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset the cache
    pub fn reset(&mut self) {
        self.seq_len = 0;
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Attention with transposed V cache for better memory access
///
/// Uses transposed V storage for improved cache locality during
/// the weighted sum computation.
#[must_use]
pub fn attention_with_transposed_v(
    q: &[f32],
    k_cache: &[f32],
    v_cache_transposed: &[f32],
    current_k: &[f32],
    current_v: &[f32],
    num_heads: usize,
    max_seq_len: usize,
) -> Vec<f32> {
    let hidden_dim = q.len();
    let head_dim = hidden_dim / num_heads;
    let cache_len = if hidden_dim > 0 {
        k_cache.len() / hidden_dim
    } else {
        0
    };
    let total_len = cache_len + 1;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut output = vec![0.0; hidden_dim];

    for h in 0..num_heads {
        let head_offset = h * head_dim;
        let q_head = &q[head_offset..head_offset + head_dim];

        // Compute attention scores
        let mut scores = Vec::with_capacity(total_len);

        for pos in 0..cache_len {
            let k_start = pos * hidden_dim + head_offset;
            let k_head = &k_cache[k_start..k_start + head_dim];
            scores.push(simd_dot(q_head, k_head) * scale);
        }

        let current_k_head = &current_k[head_offset..head_offset + head_dim];
        scores.push(simd_dot(q_head, current_k_head) * scale);

        simd_softmax(&mut scores);

        // Weighted sum with transposed V (better cache locality)
        let out_head = &mut output[head_offset..head_offset + head_dim];

        for i in 0..head_dim {
            let v_idx = (head_offset + i) * max_seq_len;
            let mut sum = 0.0;
            for (pos, &weight) in scores.iter().enumerate().take(cache_len) {
                sum += weight * v_cache_transposed[v_idx + pos];
            }
            sum += scores[cache_len] * current_v[head_offset + i];
            out_head[i] = sum;
        }
    }

    output
}

include!("kv_cache_cache.rs");
