//! Qwen2-0.5B-Instruct Model Implementation
//!
//! This module provides a complete Qwen2 model for inference, assembling
//! primitives from the `nn` module into a decoder-only transformer.
//!
//! # Architecture (Bai et al., 2023)
//!
//! ```text
//! Qwen2-0.5B-Instruct:
//! ├── hidden_size: 896
//! ├── num_attention_heads: 14 (query heads)
//! ├── num_kv_heads: 2 (grouped query attention)
//! ├── num_layers: 24
//! ├── intermediate_size: 4864 (FFN)
//! ├── vocab_size: 151936
//! ├── max_seq_len: 32768
//! └── rope_theta: 1,000,000
//! ```
//!
//! # Example
//!
//! ```ignore
//! use aprender::models::Qwen2Model;
//! use aprender::demo::Qwen2Config;
//!
//! let config = Qwen2Config::qwen2_0_5b_instruct();
//! let model = Qwen2Model::new(&config);
//! assert_eq!(model.num_layers(), 24);
//! ```
//!
//! **Note:** Inference (`forward`/`generate`) is handled exclusively by `realizar`.
//! This module provides model construction, weight loading, and introspection only.
//!
//! # References
//!
//! - Bai et al. (2023). "Qwen Technical Report"
//! - Ainslie et al. (2023). "GQA: Training Generalized Multi-Query Transformer Models"
//! - Su et al. (2021). "`RoFormer`: Enhanced Transformer with Rotary Position Embedding"
//! - Zhang & Sennrich (2019). "Root Mean Square Layer Normalization"

use crate::autograd::Tensor;
use crate::demo::Qwen2Config;
use crate::nn::{GroupedQueryAttention, Linear, Module, RMSNorm, RotaryPositionEmbedding};

// ============================================================================
// Embedding Layer
// ============================================================================

/// Token embedding lookup table.
///
/// Maps token IDs to dense vectors.
#[derive(Debug)]
pub struct Embedding {
    /// Weight matrix [`vocab_size`, `hidden_size`]
    weight: Tensor,
    vocab_size: usize,
    hidden_size: usize,
}

impl Embedding {
    /// Create a new embedding layer.
    #[must_use]
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        // Initialize with small random values
        let data: Vec<f32> = (0..vocab_size * hidden_size)
            .map(|i| {
                // Deterministic pseudo-random initialization
                (i as f32 * 0.0001).sin() * 0.02
            })
            .collect();

        Self {
            weight: Tensor::from_vec(data, &[vocab_size, hidden_size]),
            vocab_size,
            hidden_size,
        }
    }

    /// Create a placeholder embedding with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    /// Uses 1-element tensor instead of `vocab_size` * `hidden_size`.
    ///
    /// **IMPORTANT**: This layer will NOT work for inference until
    /// `set_weight()` is called with real weights.
    #[must_use]
    pub fn placeholder(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            weight: Tensor::new(&[0.0], &[1]),
            vocab_size,
            hidden_size,
        }
    }

    /// Look up embeddings for token IDs into a pre-allocated buffer.
    pub fn forward_into(&self, input_ids: &[u32], output: &mut [f32]) {
        for (s, &token_id) in input_ids.iter().enumerate() {
            let token_idx = token_id as usize;
            if token_idx >= self.vocab_size {
                // N-09: OOB token produces zeros (buffer already zero-initialized).
                // Contract: embedding-lookup-v1.yaml requires token_ids[i] < vocab_size.
                // We warn instead of panic because callers may be fuzzing or probing.
                eprintln!(
                    "Warning: token_id {token_id} >= vocab_size {} (N-09 OOB escape, zeros emitted)",
                    self.vocab_size
                );
                continue;
            }

            let src_offset = token_idx * self.hidden_size;
            let dst_offset = s * self.hidden_size;

            output[dst_offset..dst_offset + self.hidden_size]
                .copy_from_slice(&self.weight.data()[src_offset..src_offset + self.hidden_size]);
        }
    }

    /// Look up embeddings for token IDs.
    ///
    /// Contract: embedding-algebra-v1, equation "embedding_lookup"
    #[provable_contracts_macros::contract("embedding-algebra-v1", equation = "embedding_lookup")]
    #[must_use]
    #[allow(unused_variables)] // contract macro binds token_ids internally
    pub fn forward(&self, input_ids: &[u32]) -> Tensor {
        contract_pre_embedding_lookup!(input_ids);
        contract_pre_inference_determinism!();
        let batch_size = 1;
        let mut output = vec![0.0f32; batch_size * input_ids.len() * self.hidden_size];
        self.forward_into(input_ids, &mut output);
        let mut result = Tensor::new(&output, &[batch_size, input_ids.len(), self.hidden_size]);

        // PMAT-913 / OBLIG-EMBEDDING-BACKWARD-GRAD-FLOW: scatter-add the upstream
        // gradient back into the embedding TABLE rows by token index so the
        // token embeddings are trainable (previously severed via Tensor::new).
        self.record_embedding_backward(input_ids, &mut result);

        contract_post_embedding_lookup!(result.data());
        contract_post_inference_determinism!(result.data());
        result
    }

    /// Record the Embedding backward edge on the autograd tape (PMAT-913).
    ///
    /// Wires the embedding `weight` table as the sole graph input. The backward
    /// scatter-adds `grad_output[i]` into `dW[input_ids[i]]`; repeated token ids
    /// accumulate (ADD, not overwrite). Only runs when grad is enabled and the
    /// weight requires grad.
    fn record_embedding_backward(&self, input_ids: &[u32], result: &mut Tensor) {
        use crate::autograd::{is_grad_enabled, with_graph};
        use std::sync::Arc;

        if is_grad_enabled() && self.weight.requires_grad_enabled() {
            result.requires_grad_(true);
            let grad_fn = Arc::new(crate::autograd::grad_fn::EmbeddingBackward {
                indices: input_ids.to_vec(),
                vocab_size: self.vocab_size,
                hidden_size: self.hidden_size,
            });
            result.set_grad_fn(grad_fn.clone());
            with_graph(|graph| {
                graph.register_tensor(self.weight.clone());
                graph.record(result.id(), grad_fn, vec![self.weight.id()]);
            });
        }
    }

    /// Set weights from external tensor.
    pub fn set_weight(&mut self, weight: Tensor) {
        self.weight = weight;
    }

    /// Get weight tensor reference.
    #[must_use]
    pub fn weight(&self) -> &Tensor {
        contract_pre_q_projection!();
        contract_pre_kv_projection!();
        &self.weight
    }
}

// ============================================================================
// Qwen2 MLP (SwiGLU)
// ============================================================================

/// Qwen2 MLP with `SwiGLU` activation.
///
/// ```text
/// output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
/// ```
#[derive(Debug)]
#[allow(clippy::struct_field_names)] // Standard ML naming convention
pub struct Qwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2MLP {
    /// Create a new Qwen2 MLP layer.
    #[must_use]
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::new(hidden_size, intermediate_size),
            up_proj: Linear::new(hidden_size, intermediate_size),
            down_proj: Linear::new(intermediate_size, hidden_size),
        }
    }

    /// Create a placeholder MLP with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    #[must_use]
    pub fn placeholder(hidden_size: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::placeholder(hidden_size, intermediate_size),
            up_proj: Linear::placeholder(hidden_size, intermediate_size),
            down_proj: Linear::placeholder(intermediate_size, hidden_size),
        }
    }

    /// Forward pass with `SwiGLU` activation.
    #[must_use]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        contract_pre_swiglu!(x.data());
        let gate = self.gate_proj.forward(x);
        let gate_activated = silu(&gate);
        let up = self.up_proj.forward(x);
        let hidden = elementwise_mul(&gate_activated, &up);
        let result = self.down_proj.forward(&hidden);
        contract_post_swiglu!(result.data());
        result
    }

    /// Get mutable reference to gate projection layer.
    pub fn gate_proj_mut(&mut self) -> &mut Linear {
        &mut self.gate_proj
    }

    /// Get mutable reference to up projection layer.
    pub fn up_proj_mut(&mut self) -> &mut Linear {
        &mut self.up_proj
    }

    /// Get mutable reference to down projection layer.
    pub fn down_proj_mut(&mut self) -> &mut Linear {
        &mut self.down_proj
    }
}

// ============================================================================
// Qwen2 Decoder Layer
// ============================================================================

/// Single Qwen2 decoder layer.
///
/// ```text
/// residual = x
/// x = input_layernorm(x)
/// x = self_attn(x, x, x) + residual
///
/// residual = x
/// x = post_attention_layernorm(x)
/// x = mlp(x) + residual
/// ```
#[derive(Debug)]
pub struct Qwen2DecoderLayer {
    self_attn: GroupedQueryAttention,
    mlp: Qwen2MLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl Qwen2DecoderLayer {
    /// Create a new decoder layer.
    #[must_use]
    pub fn new(config: &Qwen2Config) -> Self {
        Self {
            self_attn: GroupedQueryAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads,
            ),
            mlp: Qwen2MLP::new(config.hidden_size, config.intermediate_size),
            input_layernorm: RMSNorm::new(&[config.hidden_size]),
            post_attention_layernorm: RMSNorm::new(&[config.hidden_size]),
        }
    }

    /// Create a placeholder decoder layer with minimal memory allocation.
    ///
    /// Used for lazy initialization when loading pre-trained weights.
    #[must_use]
    pub fn placeholder(config: &Qwen2Config) -> Self {
        Self {
            self_attn: GroupedQueryAttention::placeholder(
                config.hidden_size,
                config.num_attention_heads,
                config.num_kv_heads,
            ),
            mlp: Qwen2MLP::placeholder(config.hidden_size, config.intermediate_size),
            input_layernorm: RMSNorm::placeholder(&[config.hidden_size]),
            post_attention_layernorm: RMSNorm::placeholder(&[config.hidden_size]),
        }
    }

    /// Forward pass through the decoder layer.
    #[must_use]
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        _position_ids: &[usize],
        _rope: &RotaryPositionEmbedding,
        _attention_mask: Option<&Tensor>,
    ) -> Tensor {
        contract_pre_residual!(hidden_states.data());
        // Self-attention with pre-norm
        // Note: Attention mask handling is simplified - using None for now
        // Full implementation would reshape mask for multi-head attention
        let residual = hidden_states.clone();
        let hidden = self.input_layernorm.forward(hidden_states);
        let (attn_output, _attn_weights) = self.self_attn.forward_self(&hidden, None);
        let hidden = add_tensors(&residual, &attn_output);

        // MLP with pre-norm
        let residual = hidden.clone();
        let hidden = self.post_attention_layernorm.forward(&hidden);
        let mlp_output = self.mlp.forward(&hidden);
        add_tensors(&residual, &mlp_output)
    }

    /// Get mutable reference to self-attention layer.
    pub fn self_attn_mut(&mut self) -> &mut GroupedQueryAttention {
        &mut self.self_attn
    }

    /// Get mutable reference to MLP layer.
    pub fn mlp_mut(&mut self) -> &mut Qwen2MLP {
        &mut self.mlp
    }

    /// Get mutable reference to input layernorm.
    pub fn input_layernorm_mut(&mut self) -> &mut RMSNorm {
        &mut self.input_layernorm
    }

    /// Get mutable reference to post-attention layernorm.
    pub fn post_attention_layernorm_mut(&mut self) -> &mut RMSNorm {
        &mut self.post_attention_layernorm
    }
}

// ============================================================================
// Qwen2 Model
// ============================================================================
//
// NOTE: Autoregressive generation (and its `KVCache`) was removed when the
// non-realizar inference path was deleted (Refs #224, #1977). All inference —
// including KV caching — is owned exclusively by `realizar`
// (`StreamingKVCache`). This struct provides model construction, weight
// loading, and introspection only.

/// Complete Qwen2 model: construction, weight loading, and introspection.
///
/// Assembles embedding, decoder layers, and LM head into a complete model.
/// Inference (forward/generate) is owned exclusively by `realizar` — this
/// struct holds weights and metadata only.
#[derive(Debug)]
pub struct Qwen2Model {
    /// Token embeddings [`vocab_size`, `hidden_size`]
    embed_tokens: Embedding,
    /// Decoder layers
    layers: Vec<Qwen2DecoderLayer>,
    /// Final `RMSNorm`
    norm: RMSNorm,
    /// Language model head [`hidden_size`, `vocab_size`]
    lm_head: Linear,
    /// Rotary position embeddings (used by realizar for inference)
    #[allow(dead_code)]
    rope: RotaryPositionEmbedding,
    /// Model configuration
    config: Qwen2Config,
    /// Training mode flag
    training: bool,
}

#[cfg(test)]
#[path = "tests_embedding_contract.rs"]
mod tests_embedding_contract;

#[cfg(test)]
#[path = "tests_embedding_backward_gradflow.rs"]
mod tests_embedding_backward_gradflow;

include!("constructors.rs");
include!("element-wise.rs");
