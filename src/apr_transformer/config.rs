//! APR Transformer Configuration Types (PMAT-802)
//!
//! Configuration structs for APR transformer:
//! - AprKVCache: KV cache for efficient autoregressive generation
//! - GenerateConfig: Generation parameters
//! - AprTransformerConfig: Model architecture configuration
//! - AprTransformerLayer: Per-layer weights
//! - Q4KLayerWeights: Q4K quantized layer weights

use serde::{Deserialize, Serialize};

// ============================================================================

/// KV Cache for efficient autoregressive generation (Y4)
///
/// Pre-allocates storage for keys and values to avoid allocations during decode.
/// Each layer has separate K and V caches stored contiguously.
///
/// # Memory Layout
///
/// For each layer: `[K_pos0, K_pos1, ..., K_posN, V_pos0, V_pos1, ..., V_posN]`
/// where each K/V entry has shape `[num_kv_heads * head_dim]`.
#[derive(Debug, Clone)]
pub struct AprKVCache {
    /// Number of layers
    num_layers: usize,
    /// Number of KV heads
    num_kv_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Maximum context length (pre-allocated capacity)
    capacity: usize,
    /// Current sequence length (positions filled)
    len: usize,
    /// True if a position is currently being appended (layers 0..N-1 have written)
    in_progress: bool,
    /// K cache per layer: [num_layers][capacity * num_kv_heads * head_dim]
    k_cache: Vec<Vec<f32>>,
    /// V cache per layer: [num_layers][capacity * num_kv_heads * head_dim]
    v_cache: Vec<Vec<f32>>,
}

impl AprKVCache {
    /// Create a new KV cache with pre-allocated capacity
    ///
    /// # Arguments
    ///
    /// * `config` - Transformer configuration
    ///
    /// # Returns
    ///
    /// Empty KV cache with capacity for full context length
    #[must_use]
    pub fn new(config: &AprTransformerConfig) -> Self {
        let num_layers = config.num_layers;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = config.explicit_head_dim.unwrap_or_else(|| {
            if config.num_heads > 0 {
                config.hidden_dim / config.num_heads
            } else {
                0
            }
        });
        // N-03 (Meyer DbC): context_length may be 0 if metadata is missing.
        // Apply a safe minimum for KV cache allocation.
        let capacity = if config.context_length > 0 {
            config.context_length
        } else {
            2048
        };

        // Pre-allocate full capacity for each layer
        let kv_size = capacity * num_kv_heads * head_dim;
        let k_cache = (0..num_layers).map(|_| vec![0.0f32; kv_size]).collect();
        let v_cache = (0..num_layers).map(|_| vec![0.0f32; kv_size]).collect();

        Self {
            num_layers,
            num_kv_heads,
            head_dim,
            capacity,
            len: 0,
            in_progress: false,
            k_cache,
            v_cache,
        }
    }

    /// Get current sequence length (number of cached positions)
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if cache is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get pre-allocated capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get number of KV heads
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Get head dimension
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Append K and V for a single position
    ///
    /// When called with `layer == num_layers - 1` (last layer), this automatically
    /// increments `self.len` so that `get()` returns the newly appended data.
    /// Tests that only use layer 0 should call `advance()` after append.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    /// * `k` - Key tensor `[num_kv_heads * head_dim]`
    /// * `v` - Value tensor `[num_kv_heads * head_dim]`
    ///
    /// # Panics
    ///
    /// Panics if layer index is out of bounds or cache is full
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert!(layer < self.num_layers, "Layer index out of bounds");
        assert!(self.len < self.capacity, "KV cache is full");

        let kv_size = self.num_kv_heads * self.head_dim;
        let offset = self.len * kv_size;

        // Copy K and V into pre-allocated storage
        self.k_cache[layer][offset..offset + kv_size].copy_from_slice(k);
        self.v_cache[layer][offset..offset + kv_size].copy_from_slice(v);

        // Mark that we have in-progress data (so get() includes it)
        self.in_progress = true;

        // F-REGR-231 FIX: Increment len only on LAST layer to ensure:
        // 1. All layers write to the same offset (correct for single token)
        // 2. get() immediately sees new data after last layer appends
        // 3. No manual advance() calls needed in production code
        // Note: Tests using only layer 0 should call advance() manually.
        if layer == self.num_layers - 1 {
            self.len += 1;
            self.in_progress = false;
        }
    }

    /// Advance the cache position manually.
    ///
    /// Usually not needed - `append()` auto-advances after the last layer.
    /// Only use this if you need to advance without appending all layers (e.g., in tests).
    pub fn advance(&mut self) {
        self.len += 1;
        self.in_progress = false;
    }

    /// Get cached K and V for a layer
    ///
    /// If `in_progress` is true, returns data up to `len + 1` positions to include
    /// data appended by earlier layers in the current forward pass.
    ///
    /// # Arguments
    ///
    /// * `layer` - Layer index
    ///
    /// # Returns
    ///
    /// Tuple of (K cache slice, V cache slice) containing all cached positions
    #[must_use]
    pub fn get(&self, layer: usize) -> (&[f32], &[f32]) {
        let kv_size = self.num_kv_heads * self.head_dim;
        // Include in-progress position if any layer has appended
        let effective_len = self.len + (self.in_progress as usize);
        let used_size = effective_len * kv_size;

        (
            &self.k_cache[layer][..used_size],
            &self.v_cache[layer][..used_size],
        )
    }

    /// Clear the cache (reset to empty without deallocating)
    pub fn clear(&mut self) {
        self.len = 0;
        self.in_progress = false;
        // No need to zero memory - will be overwritten on next append
    }
}

/// Configuration for text generation
#[derive(Debug, Clone)]
pub struct GenerateConfig {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Top-p nucleus sampling threshold (optional)
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    /// Enable trace output (default: false)
    pub trace: bool,
    /// GH-330: EOS token IDs for stopping generation.
    ///
    /// **Design by Contract**: These come from the model config, not hardcoded.
    /// Empty means no EOS checking (generate until max_tokens).
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 32,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            trace: false,
            stop_tokens: Vec::new(),
        }
    }
}

/// Configuration for APR Transformer models
///
/// Mirrors `GGUFConfig` for compatibility but is serializable to APR format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AprTransformerConfig {
    /// Model architecture name (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Embedding/hidden dimension
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Maximum context length
    pub context_length: usize,
    /// RoPE theta for position encoding
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
    /// GH-330: EOS token ID (Design by Contract class invariant).
    ///
    /// After construction, the config carries the model's own EOS token.
    /// Callers must NOT use hardcoded fallbacks.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// GH-278: Explicit head dimension (Qwen3.5: 256)
    /// When present, overrides hidden_dim / num_heads calculation.
    #[serde(default)]
    pub explicit_head_dim: Option<usize>,
    /// GH-278: Per-layer attention type ("full_attention" or "linear_attention")
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
    /// GH-278: Linear attention key head dimension (Qwen3.5: 128)
    #[serde(default)]
    pub linear_key_head_dim: Option<usize>,
    /// GH-278: Linear attention value head dimension (Qwen3.5: 128)
    #[serde(default)]
    pub linear_value_head_dim: Option<usize>,
    /// GH-278: Number of key heads for linear attention (Qwen3.5: 16)
    #[serde(default)]
    pub linear_num_key_heads: Option<usize>,
    /// GH-278: Number of value heads for linear attention (Qwen3.5: 48)
    #[serde(default)]
    pub linear_num_value_heads: Option<usize>,
    /// GH-278: Conv1D kernel size for linear attention (Qwen3.5: 4)
    #[serde(default)]
    pub linear_conv_kernel_dim: Option<usize>,
    /// ALB-010: Number of MoE experts (Qwen3.5-35B-A3B: 256)
    #[serde(default)]
    pub num_experts: Option<usize>,
    /// ALB-010: Number of experts selected per token (Qwen3.5-35B-A3B: 8)
    #[serde(default)]
    pub num_experts_per_tok: Option<usize>,
    /// ALB-010: MoE expert intermediate/FFN dimension (Qwen3.5-35B-A3B: 512)
    #[serde(default)]
    pub expert_intermediate_size: Option<usize>,
}

impl Default for AprTransformerConfig {
    fn default() -> Self {
        Self {
            architecture: "unknown".to_string(),
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            num_kv_heads: 8,
            vocab_size: 32000,
            intermediate_dim: 2048,
            context_length: 2048,
            rope_theta: 10000.0,
            eps: 1e-5,
            eos_token_id: None,
            explicit_head_dim: None,
            layer_types: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_num_key_heads: None,
            linear_num_value_heads: None,
            linear_conv_kernel_dim: None,
            num_experts: None,
            num_experts_per_tok: None,
            expert_intermediate_size: None,
        }
    }
}

/// Weights for a single transformer layer (all F32)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AprTransformerLayer {
    /// Attention norm weight [hidden_dim]
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional) [hidden_dim]
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weight [hidden_dim, 3*hidden_dim]
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias (optional) [3*hidden_dim]
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection weight [hidden_dim, hidden_dim]
    pub attn_output_weight: Vec<f32>,
    /// Attention output projection bias (optional) [hidden_dim]
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN gate weight for SwiGLU (optional) [hidden_dim, intermediate_dim]
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// FFN gate bias (optional) [intermediate_dim]
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN up projection weight [hidden_dim, intermediate_dim]
    pub ffn_up_weight: Vec<f32>,
    /// FFN up projection bias (optional) [intermediate_dim]
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection weight [intermediate_dim, hidden_dim]
    pub ffn_down_weight: Vec<f32>,
    /// FFN down projection bias (optional) [hidden_dim]
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN norm weight (optional) [hidden_dim]
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional) [hidden_dim]
    pub ffn_norm_bias: Option<Vec<f32>>,
    /// GH-279: Per-head Q RMSNorm weight [head_dim] (Qwen3)
    pub attn_q_norm_weight: Option<Vec<f32>>,
    /// GH-279: Per-head K RMSNorm weight [head_dim] (Qwen3)
    pub attn_k_norm_weight: Option<Vec<f32>>,
    // =========================================================================
    // GH-278: Gated Delta Net weights (Qwen3.5 linear attention layers)
    // =========================================================================
    /// Gate projection weight (z): [value_dim, hidden_dim]
    /// Split from HF `in_proj_qkvz` combined projection.
    #[serde(default)]
    pub linear_attn_z_weight: Option<Vec<f32>>,
    /// Beta gate projection weight: [num_v_heads, hidden_dim]
    /// Split from HF `in_proj_ba` combined projection.
    #[serde(default)]
    pub linear_attn_b_weight: Option<Vec<f32>>,
    /// Decay projection weight (alpha): [num_v_heads, hidden_dim]
    /// Split from HF `in_proj_ba` combined projection.
    #[serde(default)]
    pub linear_attn_a_weight: Option<Vec<f32>>,
    /// Depthwise causal Conv1D weight: [conv_dim, kernel_size]
    /// HF stores as [conv_dim, 1, kernel_size]; middle dim squeezed at load.
    #[serde(default)]
    pub linear_attn_conv1d_weight: Option<Vec<f32>>,
    /// Logged decay base A_log: [num_v_heads]
    #[serde(default)]
    pub linear_attn_a_log: Option<Vec<f32>>,
    /// Time-step bias dt_bias: [num_v_heads]
    #[serde(default)]
    pub linear_attn_dt_bias: Option<Vec<f32>>,
    /// Gated RMSNorm weight: [value_dim]
    #[serde(default)]
    pub linear_attn_norm_weight: Option<Vec<f32>>,
    // =========================================================================
    // ALB-010: MoE expert weights (Qwen3.5-35B-A3B)
    // =========================================================================
    /// ALB-010: Router gate weight [num_experts, hidden_dim]
    #[serde(default)]
    pub moe_gate_weight: Option<Vec<f32>>,
    /// ALB-010: Packed expert gate+up projections [num_experts, 2*intermediate, hidden_dim]
    #[serde(default)]
    pub moe_expert_gate_up: Option<Vec<f32>>,
    /// ALB-010: Packed expert down projections [num_experts, hidden_dim, intermediate]
    #[serde(default)]
    pub moe_expert_down: Option<Vec<f32>>,
    /// ALB-010: Shared expert gate projection [intermediate, hidden_dim]
    #[serde(default)]
    pub moe_shared_gate: Option<Vec<f32>>,
    /// ALB-010: Shared expert up projection [intermediate, hidden_dim]
    #[serde(default)]
    pub moe_shared_up: Option<Vec<f32>>,
    /// ALB-010: Shared expert down projection [hidden_dim, intermediate]
    #[serde(default)]
    pub moe_shared_down: Option<Vec<f32>>,
    /// ALB-010: Shared expert gate weight [1, hidden_dim] for sigmoid scaling
    #[serde(default)]
    pub moe_shared_expert_gate_weight: Option<Vec<f32>>,
}

impl AprTransformerLayer {
    /// Create an empty layer with given dimensions (non-GQA: num_kv_heads == num_heads)
    pub fn empty(hidden_dim: usize, intermediate_dim: usize) -> Self {
        Self {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; hidden_dim * 3 * hidden_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }
    }

    /// Create an empty layer with GQA dimensions (num_kv_heads < num_heads)
    ///
    /// # Arguments
    /// * `hidden_dim` - Hidden dimension (num_heads * head_dim)
    /// * `num_heads` - Number of query heads
    /// * `num_kv_heads` - Number of key/value heads (< num_heads for GQA)
    /// * `intermediate_dim` - FFN intermediate dimension
    pub fn empty_gqa(
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        intermediate_dim: usize,
    ) -> Self {
        let head_dim = hidden_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;
        // QKV weight: [hidden_dim, Q_dim + K_dim + V_dim] = [hidden_dim, hidden_dim + 2*kv_dim]
        let qkv_out_dim = hidden_dim + 2 * kv_dim;

        Self {
            attn_norm_weight: vec![1.0; hidden_dim],
            attn_norm_bias: None,
            qkv_weight: vec![0.0; hidden_dim * qkv_out_dim],
            qkv_bias: None,
            attn_output_weight: vec![0.0; hidden_dim * hidden_dim],
            attn_output_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_up_weight: vec![0.0; hidden_dim * intermediate_dim],
            ffn_up_bias: None,
            ffn_down_weight: vec![0.0; intermediate_dim * hidden_dim],
            ffn_down_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            linear_attn_z_weight: None,
            linear_attn_b_weight: None,
            linear_attn_a_weight: None,
            linear_attn_conv1d_weight: None,
            linear_attn_a_log: None,
            linear_attn_dt_bias: None,
            linear_attn_norm_weight: None,
            moe_gate_weight: None,
            moe_expert_gate_up: None,
            moe_expert_down: None,
            moe_shared_gate: None,
            moe_shared_up: None,
            moe_shared_down: None,
            moe_shared_expert_gate_weight: None,
        }
    }

    /// Get total number of parameters in this layer
    #[must_use]
    pub fn num_parameters(&self) -> usize {
        let mut count = 0;
        count += self.attn_norm_weight.len();
        count += self.attn_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.qkv_weight.len();
        count += self.qkv_bias.as_ref().map_or(0, Vec::len);
        count += self.attn_output_weight.len();
        count += self.attn_output_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_gate_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_up_weight.len();
        count += self.ffn_up_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_down_weight.len();
        count += self.ffn_down_bias.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_weight.as_ref().map_or(0, Vec::len);
        count += self.ffn_norm_bias.as_ref().map_or(0, Vec::len);
        count += self.attn_q_norm_weight.as_ref().map_or(0, Vec::len);
        count += self.attn_k_norm_weight.as_ref().map_or(0, Vec::len);
        // GH-278: Linear attention weights
        count += self.linear_attn_z_weight.as_ref().map_or(0, Vec::len);
        count += self.linear_attn_b_weight.as_ref().map_or(0, Vec::len);
        count += self.linear_attn_a_weight.as_ref().map_or(0, Vec::len);
        count += self.linear_attn_conv1d_weight.as_ref().map_or(0, Vec::len);
        count += self.linear_attn_a_log.as_ref().map_or(0, Vec::len);
        count += self.linear_attn_dt_bias.as_ref().map_or(0, Vec::len);
        count += self.linear_attn_norm_weight.as_ref().map_or(0, Vec::len);
        count
    }
}

/// Q4K/Q6K raw weights for fused kernel inference (F-GPU-130)
///
/// When present, matmul operations use fused kernels (matmul_q4k_f32, matmul_q6k_f32)
/// instead of the F32 path, avoiding full dequantization overhead.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Q4KLayerWeights {
    /// QKV projection weight in Q4K format (combined, legacy)
    pub qkv_weight: Option<Vec<u8>>,
    /// Q projection weight in Q4K format (PMAT-103: separate for fused kernel)
    pub attn_q_weight: Option<Vec<u8>>,
    /// K projection weight in Q4K format (PMAT-103: separate for fused kernel)
    pub attn_k_weight: Option<Vec<u8>>,
    /// V projection weight in Q4K/Q6K format (PMAT-103: separate for fused kernel)
    pub attn_v_weight: Option<Vec<u8>>,
    /// V projection weight in Q6K format (when Q4K not available)
    pub attn_v_weight_q6k: Option<Vec<u8>>,
    /// Attention output projection in Q4K format
    pub attn_output_weight: Option<Vec<u8>>,
    /// FFN gate weight in Q4K format (for SwiGLU)
    pub ffn_gate_weight: Option<Vec<u8>>,
    /// FFN up projection in Q4K format
    pub ffn_up_weight: Option<Vec<u8>>,
    /// FFN down projection in Q4K format
    pub ffn_down_weight: Option<Vec<u8>>,
    /// FFN down projection in Q6K format (when Q4K not available)
    pub ffn_down_weight_q6k: Option<Vec<u8>>,
    /// FFN up projection in Q6K format (when Q4K not available)
    pub ffn_up_weight_q6k: Option<Vec<u8>>,
}

include!("config_apr.rs");
