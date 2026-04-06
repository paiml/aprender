//! Runtime types for GGUF model inference
//!
//! This module contains types used during inference runtime:
//!
//! - `QuantizedGenerateConfig`: Configuration for text generation
//! - `OwnedQuantizedKVCache`: KV cache for incremental decoding
//!
//! These are "leaf nodes" in the dependency graph - they don't depend
//! on other complex types, making them easy to extract.

use super::config::GGUFConfig;

// ============================================================================
// QuantizedGenerateConfig - Generation parameters
// ============================================================================

/// Configuration for quantized generation
///
/// Per benchmark-model-runners-spec.md "What's Remaining" item 1:
/// End-to-end Q4_K inference with generation config.
#[derive(Debug, Clone)]
pub struct QuantizedGenerateConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (1 = greedy)
    pub top_k: usize,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Random seed for sampling
    pub seed: u64,
    /// Repetition penalty (1.0 = no penalty)
    pub repeat_penalty: f32,
    /// Number of recent tokens to consider for repetition penalty
    pub repeat_last_n: usize,
    /// Stop token IDs
    pub stop_tokens: Vec<u32>,
    /// Enable inference tracing (PMAT-TRACE-GGUF-001)
    pub trace: bool,
    /// Return per-token log probabilities (realizr#191, F-QUALITY-01)
    pub logprobs: bool,
}

impl Default for QuantizedGenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            seed: 42,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            stop_tokens: Vec::new(),
            trace: false,
            logprobs: false,
        }
    }
}

impl QuantizedGenerateConfig {
    /// Create config for deterministic (greedy) generation
    #[must_use]
    pub fn deterministic(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
            trace: false,
            ..Default::default()
        }
    }

    /// Builder method to enable tracing (PMAT-TRACE-GGUF-001)
    #[must_use]
    pub fn with_trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        self
    }

    /// Builder method to set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Builder method to set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Builder method to set top_k
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Builder method to set stop tokens
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<u32>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }
}

// ============================================================================
// OwnedQuantizedKVCache - KV cache for incremental decoding
// ============================================================================

/// KV Cache for OwnedQuantizedModel incremental decoding (IMP-101c)
///
/// Stores Key and Value projections for all layers to enable O(n) per-token
/// decoding instead of O(n²). Reference: Spec Section 5.4 "Continuous Flow".
///
/// Memory layout: [num_layers, seq_len, hidden_dim]
#[derive(Debug, Clone)]
pub struct OwnedQuantizedKVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// KV dimension: num_kv_heads * head_dim (stored for future use)
    _hidden_dim: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Current sequence length (tokens processed)
    seq_len: usize,
    /// Key cache: [num_layers][seq_len][kv_dim]
    k_cache: Vec<Vec<f32>>,
    /// Value cache: [num_layers][seq_len][kv_dim]
    v_cache: Vec<Vec<f32>>,
}

/// PARITY-096: Default impl for std::mem::take optimization in batch_generate_gpu
impl Default for OwnedQuantizedKVCache {
    fn default() -> Self {
        Self {
            num_layers: 0,
            _hidden_dim: 0,
            max_seq_len: 0,
            seq_len: 0,
            k_cache: Vec::new(),
            v_cache: Vec::new(),
        }
    }
}

impl OwnedQuantizedKVCache {
    /// Create a new KV cache for the given model configuration
    ///
    /// # Arguments
    /// * `num_layers` - Number of transformer layers
    /// * `kv_dim` - KV dimension (`num_kv_heads * head_dim` for GQA, or `hidden_dim` for MHA)
    /// * `max_seq_len` - Maximum sequence length to cache
    #[must_use]
    pub fn new(num_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        Self {
            num_layers,
            _hidden_dim: kv_dim,
            max_seq_len,
            seq_len: 0,
            k_cache: vec![Vec::with_capacity(max_seq_len * kv_dim); num_layers],
            v_cache: vec![Vec::with_capacity(max_seq_len * kv_dim); num_layers],
        }
    }

    /// Create cache from model configuration
    ///
    /// ALB-102: Uses `num_kv_heads * head_dim` (not `hidden_dim`) so GQA models
    /// allocate only the KV cache they need instead of over-allocating by
    /// `num_heads / num_kv_heads`.
    #[must_use]
    pub fn from_config(config: &GGUFConfig, max_seq_len: usize) -> Self {
        let head_dim = config.head_dim();
        let kv_dim = config.num_kv_heads * head_dim;
        Self::new(config.num_layers, kv_dim, max_seq_len)
    }

    /// Append K and V vectors for a single position to a layer's cache
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `k` - Key vector [hidden_dim]
    /// * `v` - Value vector [hidden_dim]
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        if layer < self.num_layers && self.seq_len < self.max_seq_len {
            self.k_cache[layer].extend_from_slice(k);
            self.v_cache[layer].extend_from_slice(v);
        }
    }

    /// Advance the sequence position after processing a token
    pub fn advance(&mut self) {
        if self.seq_len < self.max_seq_len {
            self.seq_len += 1;
        }
    }

    /// PAR-097: Append multiple K/V entries to a layer's cache (for speculative decode)
    ///
    /// # Arguments
    /// * `layer` - Layer index
    /// * `k_all` - Key vectors for batch_size positions [batch_size × kv_dim]
    /// * `v_all` - Value vectors for batch_size positions [batch_size × kv_dim]
    pub fn append_kv(&mut self, layer: usize, k_all: &[f32], v_all: &[f32]) {
        if layer < self.num_layers {
            self.k_cache[layer].extend_from_slice(k_all);
            self.v_cache[layer].extend_from_slice(v_all);
        }
    }

    /// PAR-097: Advance sequence position by n tokens (for speculative decode)
    pub fn advance_by(&mut self, n: usize) {
        self.seq_len = (self.seq_len + n).min(self.max_seq_len);
    }

    /// PAR-098: Rollback cache to a previous position (for speculative decode rejection)
    ///
    /// When draft tokens are rejected, we need to remove their K/V entries.
    /// This truncates each layer's cache to keep only the first `new_len` positions.
    ///
    /// # Arguments
    /// * `new_len` - The new sequence length (must be <= current length)
    /// * `kv_dim` - The dimension of each K/V entry (num_kv_heads * head_dim)
    pub fn rollback_to(&mut self, new_len: usize, kv_dim: usize) {
        if new_len >= self.seq_len {
            return; // Nothing to rollback
        }
        let target_size = new_len * kv_dim;
        for layer_k in &mut self.k_cache {
            layer_k.truncate(target_size);
        }
        for layer_v in &mut self.v_cache {
            layer_v.truncate(target_size);
        }
        self.seq_len = new_len;
    }

    /// PAR-098: Get a snapshot of current cache lengths for rollback
    #[must_use]
    pub fn snapshot_len(&self) -> usize {
        self.seq_len
    }

    /// Get cached keys for a layer
    ///
    /// Returns slice of [seq_len, hidden_dim]
    #[must_use]
    pub fn get_k(&self, layer: usize) -> &[f32] {
        if layer < self.num_layers {
            &self.k_cache[layer]
        } else {
            &[]
        }
    }

    /// Get cached values for a layer
    ///
    /// Returns slice of [seq_len, hidden_dim]
    #[must_use]
    pub fn get_v(&self, layer: usize) -> &[f32] {
        if layer < self.num_layers {
            &self.v_cache[layer]
        } else {
            &[]
        }
    }

    /// Current sequence length
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Reset cache for new generation
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for layer_k in &mut self.k_cache {
            layer_k.clear();
        }
        for layer_v in &mut self.v_cache {
            layer_v.clear();
        }
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_len(&self) -> usize {
        self.max_seq_len
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_config_default() {
        let config = QuantizedGenerateConfig::default();
        assert_eq!(config.max_tokens, 64);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(config.stop_tokens.is_empty());
    }

    #[test]
    fn test_generate_config_deterministic() {
        let config = QuantizedGenerateConfig::deterministic(128);
        assert_eq!(config.max_tokens, 128);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
    }

    #[test]
    fn test_kv_cache_new() {
        let cache = OwnedQuantizedKVCache::new(4, 256, 512);
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.max_len(), 512);
    }

    #[test]
    fn test_kv_cache_append_advance() {
        let mut cache = OwnedQuantizedKVCache::new(2, 4, 10);

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
        assert_eq!(cache.get_k(0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(cache.get_v(0), &[5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = OwnedQuantizedKVCache::new(2, 4, 10);

        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];

        cache.append(0, &k, &v);
        cache.advance();

        cache.reset();

        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert!(cache.get_k(0).is_empty());
    }

    #[test]
    fn test_kv_cache_rollback() {
        let mut cache = OwnedQuantizedKVCache::new(1, 2, 10);

        // Append 3 tokens
        for i in 0..3 {
            let k = vec![i as f32, i as f32 + 0.5];
            let v = vec![i as f32 + 1.0, i as f32 + 1.5];
            cache.append(0, &k, &v);
            cache.advance();
        }

        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get_k(0).len(), 6); // 3 tokens * 2 dims

        // Rollback to position 1
        cache.rollback_to(1, 2);

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0).len(), 2); // 1 token * 2 dims
    }

    #[test]
    fn test_kv_cache_from_config() {
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 256,
            num_layers: 4,
            num_heads: 4,
            num_kv_heads: 4,
            vocab_size: 1000,
            intermediate_dim: 512,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let cache = OwnedQuantizedKVCache::from_config(&config, 512);
        assert_eq!(cache.max_len(), 512);
    }

    /// ALB-102: Verify GQA models allocate kv_dim (not hidden_dim) in from_config
    #[test]
    fn test_kv_cache_from_config_gqa() {
        // Albor 350M: hidden=1024, heads=16, kv_heads=4, head_dim=64
        // kv_dim = 4 * 64 = 256 (not 1024)
        let config = GGUFConfig {
            architecture: "llama".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("llama"),
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            vocab_size: 32000,
            intermediate_dim: 4096,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: None,
            bos_token_id: None,
            eos_token_id: None,
        };

        let kv_dim = config.num_kv_heads * config.head_dim(); // 4 * 64 = 256
        assert_eq!(kv_dim, 256);

        let mut cache = OwnedQuantizedKVCache::from_config(&config, 16);
        assert_eq!(cache.max_len(), 16);

        // Append a kv_dim-sized vector (256 floats, not 1024)
        let k = vec![1.0_f32; kv_dim];
        let v = vec![2.0_f32; kv_dim];
        cache.append(0, &k, &v);
        cache.advance();

        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get_k(0).len(), kv_dim); // 256, not 1024
        assert_eq!(cache.get_v(0).len(), kv_dim);
    }

    /// ALB-102: Verify explicit_head_dim is respected in from_config
    #[test]
    fn test_kv_cache_from_config_explicit_head_dim() {
        // Qwen3-0.6B style: hidden=1024, heads=16, kv_heads=4, explicit_head_dim=128
        // kv_dim = 4 * 128 = 512 (head_dim != hidden_dim / num_heads)
        let config = GGUFConfig {
            architecture: "qwen3".to_string(),
            constraints: crate::gguf::ArchConstraints::from_architecture("qwen3"),
            hidden_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            num_kv_heads: 4,
            vocab_size: 151936,
            intermediate_dim: 2048,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            rope_type: 0,
            explicit_head_dim: Some(128),
            bos_token_id: None,
            eos_token_id: None,
        };

        let kv_dim = config.num_kv_heads * config.head_dim(); // 4 * 128 = 512
        assert_eq!(kv_dim, 512);

        let cache = OwnedQuantizedKVCache::from_config(&config, 32);
        assert_eq!(cache.max_len(), 32);
    }

    #[test]
    fn test_kv_cache_default() {
        let cache = OwnedQuantizedKVCache::default();
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.max_len(), 0);
        assert!(cache.is_empty());
    }
}
