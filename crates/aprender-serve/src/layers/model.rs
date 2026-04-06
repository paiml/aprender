//! Model components for transformer inference
//!
//! Extracted from layers/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - KVCache: Key-Value cache for efficient autoregressive generation
//! - TransformerBlock: Single transformer layer (attention + FFN)
//! - Embedding: Token embedding layer
//! - Model: Full transformer model for inference
//! - ModelConfig: Configuration for transformer models

use crate::{
    error::{RealizarError, Result},
    generate::{sample_token, GenerationConfig},
    tensor::Tensor,
};

use super::{FeedForward, LayerNorm, Linear, MultiHeadAttention};

/// Key-Value Cache for efficient transformer inference
///
/// Stores key and value tensors from previous positions to avoid
/// recomputation during autoregressive generation. Each forward pass
/// only computes K/V for the new token and appends to the cache.
///
/// # Usage
///
/// 1. Create cache with `KVCache::new(num_layers, max_seq_len, head_dim)`
/// 2. At each generation step, call `update` with new K/V
/// 3. Use `get_key`/`get_value` to retrieve cached tensors
/// 4. Call `clear` to reset for new sequence
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Number of transformer layers
    num_layers: usize,
    /// Maximum sequence length
    max_seq_len: usize,
    /// Dimension per head
    head_dim: usize,
    /// Current sequence position
    current_pos: usize,
    /// Cached keys for each layer: `[num_layers][max_seq_len * head_dim]`
    keys: Vec<Vec<f32>>,
    /// Cached values for each layer: `[num_layers][max_seq_len * head_dim]`
    values: Vec<Vec<f32>>,
}

impl KVCache {
    /// Create a new KV cache
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of transformer layers to cache
    /// * `max_seq_len` - Maximum sequence length to cache
    /// * `head_dim` - Dimension per attention head
    ///
    /// # Errors
    ///
    /// Returns error if any dimension is zero
    pub fn new(num_layers: usize, max_seq_len: usize, head_dim: usize) -> Result<Self> {
        if num_layers == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_layers must be > 0".to_string(),
            });
        }
        if max_seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "max_seq_len must be > 0".to_string(),
            });
        }
        if head_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }

        let cache_size = max_seq_len * head_dim;
        let keys = vec![vec![0.0; cache_size]; num_layers];
        let values = vec![vec![0.0; cache_size]; num_layers];

        Ok(Self {
            num_layers,
            max_seq_len,
            head_dim,
            current_pos: 0,
            keys,
            values,
        })
    }

    /// Update cache with new key/value for a layer
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `key` - New key tensor `[head_dim]`
    /// * `value` - New value tensor `[head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds, cache is full, or tensor sizes don't match
    pub fn update(&mut self, layer: usize, key: &Tensor<f32>, value: &Tensor<f32>) -> Result<()> {
        contract_pre_update!();
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }
        if self.current_pos >= self.max_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Cache full at position {} (max {})",
                    self.current_pos, self.max_seq_len
                ),
            });
        }

        let k_data = key.data();
        let v_data = value.data();

        if k_data.len() != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Key size {} != head_dim {}", k_data.len(), self.head_dim),
            });
        }
        if v_data.len() != self.head_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!("Value size {} != head_dim {}", v_data.len(), self.head_dim),
            });
        }

        // Copy key and value into cache at current position
        let offset = self.current_pos * self.head_dim;
        self.keys[layer][offset..offset + self.head_dim].copy_from_slice(k_data);
        self.values[layer][offset..offset + self.head_dim].copy_from_slice(v_data);

        contract_post_update!(&());
        Ok(())
    }

    /// Advance to next position after updating all layers
    ///
    /// Call this after updating all layers for the current position
    pub fn advance(&mut self) {
        if self.current_pos < self.max_seq_len {
            self.current_pos += 1;
        }
    }

    /// Get cached keys for a layer up to current position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tensor with shape `[current_pos, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds
    pub fn get_key(&self, layer: usize) -> Result<Tensor<f32>> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }

        if self.current_pos == 0 {
            // Return empty tensor with shape [0, head_dim] is invalid
            // Return [1, head_dim] with zeros for consistency
            return Tensor::from_vec(vec![1, self.head_dim], vec![0.0; self.head_dim]);
        }

        let size = self.current_pos * self.head_dim;
        let data = self.keys[layer][..size].to_vec();
        Tensor::from_vec(vec![self.current_pos, self.head_dim], data)
    }

    /// Get cached values for a layer up to current position
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tensor with shape `[current_pos, head_dim]`
    ///
    /// # Errors
    ///
    /// Returns error if layer is out of bounds
    pub fn get_value(&self, layer: usize) -> Result<Tensor<f32>> {
        if layer >= self.num_layers {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Layer {} out of bounds (max {})",
                    layer,
                    self.num_layers - 1
                ),
            });
        }

        if self.current_pos == 0 {
            return Tensor::from_vec(vec![1, self.head_dim], vec![0.0; self.head_dim]);
        }

        let size = self.current_pos * self.head_dim;
        let data = self.values[layer][..size].to_vec();
        Tensor::from_vec(vec![self.current_pos, self.head_dim], data)
    }

    /// Clear cache and reset position to 0
    pub fn clear(&mut self) {
        self.current_pos = 0;
        // Optionally zero out the cache (not strictly necessary)
        for layer in 0..self.num_layers {
            self.keys[layer].fill(0.0);
            self.values[layer].fill(0.0);
        }
    }

    /// Get current sequence position
    #[must_use]
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    /// Get number of layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Get maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Check if cache is full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.current_pos >= self.max_seq_len
    }
}

/// Transformer Block (Pre-norm architecture)
///
/// A single transformer block combining self-attention and feed-forward layers
/// with residual connections and layer normalization.
///
/// # Architecture
///
/// ```text
/// Input
///   │
///   ├──────────────────┐
///   ▼                  │
/// LayerNorm            │
///   ▼                  │
/// Attention            │
///   ▼                  │
///   + <────────────────┘ (residual)
///   │
///   ├──────────────────┐
///   ▼                  │
/// LayerNorm            │
///   ▼                  │
/// FFN                  │
///   ▼                  │
///   + <────────────────┘ (residual)
///   │
/// Output
/// ```
///
/// This is the pre-norm architecture used in `LLaMA`, GPT-NeoX, and modern transformers.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Layer normalization before attention
    attn_norm: LayerNorm,
    /// Multi-head self-attention layer with Q/K/V/O projections
    attention: MultiHeadAttention,
    /// Layer normalization before FFN
    ffn_norm: LayerNorm,
    /// Feed-forward network
    ffn: FeedForward,
    /// Hidden dimension
    hidden_dim: usize,
    /// Number of attention heads
    num_heads: usize,
}

include!("model_transformer_block.rs");
include!("model_model.rs");
include!("model_cache.rs");
