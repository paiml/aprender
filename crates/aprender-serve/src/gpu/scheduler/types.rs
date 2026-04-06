//! GPU Model Types and Configuration (PMAT-COMPLY)
//!
//! Type definitions extracted from model.rs to improve file health.
//!
//! Contains:
//! - `BlockWeights` - Transformer block weight storage
//! - `WeightType` - Weight matrix type enum for split-borrow pattern
//! - `GpuModelConfig` - Model configuration
//! - `GpuGenerateConfig` - Generation parameters
//! - `AttentionBuffers` - Pre-allocated attention buffers

/// Weights for a single transformer block
///
/// Used by adapters (PMAT-106) to construct GpuModel from various formats.
pub struct BlockWeights {
    /// Attention layer norm weight
    pub attn_norm_weight: Vec<f32>,
    /// Attention layer norm bias
    pub attn_norm_bias: Vec<f32>,
    /// Combined QKV projection weights (hidden_dim x 3*hidden_dim)
    pub qkv_weight: Vec<f32>,
    /// QKV projection bias (reserved for future use)
    pub qkv_bias: Vec<f32>,
    /// Output projection weight (hidden_dim x hidden_dim)
    pub out_weight: Vec<f32>,
    /// Output projection bias
    pub out_bias: Vec<f32>,
    /// FFN layer norm weight
    pub ffn_norm_weight: Vec<f32>,
    /// FFN layer norm bias
    pub ffn_norm_bias: Vec<f32>,
    /// FFN first layer weight (up projection)
    pub ffn_fc1_weight: Vec<f32>,
    /// FFN first layer bias
    pub ffn_fc1_bias: Vec<f32>,
    /// FFN second layer weight (down projection)
    pub ffn_fc2_weight: Vec<f32>,
    /// FFN second layer bias
    pub ffn_fc2_bias: Vec<f32>,
    /// FFN gate projection weight for SwiGLU (optional)
    /// When present, FFN uses SwiGLU: down(SiLU(gate(x)) * up(x))
    /// When None, FFN uses GELU: down(GELU(up(x)))
    pub ffn_gate_weight: Option<Vec<f32>>,
    /// GH-278: Gated Delta Net weights for linear attention layers (Qwen3.5)
    ///
    /// When `Some`, this block uses the Gated Delta Net recurrent mechanism
    /// instead of standard softmax attention. The `qkv_weight` field stores
    /// `in_proj_qkv` and `out_weight` stores `out_proj`.
    ///
    /// Contract: `linear_attn.is_some()` ⟺ `config.is_linear_layer(block_idx)`
    pub linear_attn: Option<LinearAttnWeights>,
    /// ALB-010: MoE expert weights for Mixture-of-Experts layers (Qwen3.5-35B-A3B)
    ///
    /// When `Some`, this block uses MoE dispatch instead of dense FFN.
    /// The router selects top-k experts per token, runs SwiGLU on each,
    /// and combines with weighted sum + shared expert.
    ///
    /// Contract: `moe_experts.is_some()` ⟺ `config.num_experts.is_some()`
    pub moe_experts: Option<MoeExpertWeights>,
}

// =============================================================================
// GH-278: Gated Delta Net Types (Qwen3.5 Linear Attention)
// =============================================================================

/// Weights for a Gated Delta Net linear attention layer (GH-278)
///
/// Implements the recurrence from Qwen3.5 `Qwen3_5GatedDeltaNet`:
///
/// ```text
/// Equation (GDN-1): state_t = exp(g_t) · state_{t-1} + k_t ⊗ δ_t
/// Equation (GDN-2): δ_t = β_t · (v_t − state_{t-1}^T k_t)
/// Equation (GDN-3): output_t = state_t^T q_t
/// ```
///
/// where `g_t = −exp(A_log) · softplus(a_t + dt_bias)` is the decay factor,
/// `β_t = σ(b_t)` is the update gate, and Q/K are L2-normalized.
#[derive(Debug, Clone)]
pub struct LinearAttnWeights {
    /// Gate projection weight: [value_dim, hidden_dim]
    /// Projects input to gating signal z for output normalization
    pub z_weight: Vec<f32>,
    /// Beta gate projection weight: [num_v_heads, hidden_dim]
    /// Projects input to β = σ(b), controlling state update magnitude
    pub b_weight: Vec<f32>,
    /// Decay projection weight: [num_v_heads, hidden_dim]
    /// Projects input to a, used in decay g = −exp(A_log) · softplus(a + dt_bias)
    pub a_weight: Vec<f32>,
    /// Depthwise causal Conv1D weight: [conv_dim, kernel_size]
    /// Applied to concatenated QKV before SiLU activation.
    /// conv_dim = 2 * key_dim + value_dim. Depthwise: each channel independent.
    pub conv1d_weight: Vec<f32>,
    /// Logged decay base: [num_v_heads]
    /// A_log = log(A), where A controls exponential state decay rate
    pub a_log: Vec<f32>,
    /// Time-step bias: [num_v_heads]
    /// Added to decay projection before softplus: softplus(a + dt_bias)
    pub dt_bias: Vec<f32>,
    /// Gated RMSNorm weight: [head_v_dim]
    /// Applied as: RMSNorm(output) * SiLU(z)
    pub norm_weight: Vec<f32>,
}

// =============================================================================
// ALB-010: MoE Expert Types (Qwen3.5-35B-A3B)
// =============================================================================

/// Weights for Mixture-of-Experts dispatch (ALB-010)
///
/// Implements the MoE forward pass for Qwen3.5-35B-A3B:
/// 1. Router: `hidden_states @ gate_weight.T` → softmax → top-k → renormalize
/// 2. Per expert: `down(SiLU(gate(x)) * up(x))` (SwiGLU)
/// 3. Weighted sum of routed expert outputs + shared expert output
///
/// Contract: `moe-router-v1.yaml`, `moe-expert-dispatch-v1.yaml`
#[derive(Debug, Clone)]
pub struct MoeExpertWeights {
    /// Router gate weight: [num_experts, hidden_dim]
    /// Projects hidden states to expert scores for softmax routing.
    pub gate_weight: Vec<f32>,
    /// Fused gate+up projection for all routed experts: [num_experts, 2*intermediate, hidden_dim]
    /// Unfuse per expert: gate = [0:intermediate, :], up = [intermediate:2*intermediate, :]
    pub expert_gate_up: Vec<f32>,
    /// Down projection for all routed experts: [num_experts, hidden_dim, intermediate]
    pub expert_down: Vec<f32>,
    /// Shared expert gate projection: [intermediate, hidden_dim]
    /// Always-active expert (not gated by router).
    pub shared_gate: Vec<f32>,
    /// Shared expert up projection: [intermediate, hidden_dim]
    pub shared_up: Vec<f32>,
    /// Shared expert down projection: [hidden_dim, intermediate]
    pub shared_down: Vec<f32>,
    /// Shared expert gate: [1, hidden_dim] linear that scales shared expert output.
    /// Qwen3.5 applies sigmoid(gate(x)) * shared_expert(x). Empty = no gating.
    pub shared_expert_gate_weight: Vec<f32>,
    /// Number of experts (e.g., 256 for Qwen3.5-35B-A3B)
    pub num_experts: usize,
    /// Number of experts selected per token (e.g., 8)
    pub num_experts_per_tok: usize,
    /// Expert FFN intermediate dimension (e.g., 512)
    pub expert_intermediate: usize,
}

// =============================================================================
// Type-safe weight wrappers to prevent argument swaps
// =============================================================================
// Newtypes for lm_head_weight and lm_head_weight_t ensure that swapping them
// in from_apr_weights() is a compile-time error rather than a silent misuse.

/// LM head weight in original layout [vocab_size, hidden_dim]
/// Used for reference/debugging, NOT for GPU matmul
#[derive(Debug, Clone)]
pub struct LmHeadWeight(pub Vec<f32>);

/// LM head weight TRANSPOSED [hidden_dim, vocab_size]
/// This is what GPU matmul kernels expect
#[derive(Debug, Clone)]
pub struct LmHeadWeightTransposed(pub Vec<f32>);

impl LmHeadWeight {
    /// Get inner data
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    /// Borrow inner data
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

impl LmHeadWeightTransposed {
    /// Get inner data
    pub fn into_inner(self) -> Vec<f32> {
        self.0
    }

    /// Borrow inner data
    pub fn as_slice(&self) -> &[f32] {
        &self.0
    }
}

/// IMP-1007: Weight type for split-borrow matmul
///
/// This enum specifies which weight matrix to use in matmul_split,
/// enabling zero-clone matmul operations by using Rust's split borrow pattern.
#[derive(Debug, Clone, Copy)]
pub enum WeightType {
    /// QKV projection: [hidden_dim, qkv_dim]
    Qkv,
    /// Output projection: [hidden_dim, hidden_dim]
    Output,
    /// FFN FC1: [hidden_dim, intermediate_dim]
    FfnFc1,
    /// FFN FC2: [intermediate_dim, hidden_dim]
    FfnFc2,
    /// LM head: [hidden_dim, vocab_size]
    LmHead,
}

/// Configuration for GPU model
#[derive(Debug, Clone)]
pub struct GpuModelConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of attention heads (Q heads)
    pub num_heads: usize,
    /// Number of key-value heads for GQA (IMP-088)
    /// For standard MHA: num_kv_heads == num_heads
    /// For GQA (Qwen, Llama-3): num_kv_heads < num_heads
    pub num_kv_heads: usize,
    /// Number of transformer blocks
    pub num_layers: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Layer normalization epsilon
    pub eps: f32,
    /// RoPE theta for rotary position embeddings (Phase 21)
    /// Default: 10000.0 (standard LLaMA)
    pub rope_theta: f32,
    /// GH-278: Explicit head dimension override
    /// Qwen3.5 uses head_dim=256 (vs Qwen2's computed 128).
    /// When None, computed as hidden_dim / num_heads.
    pub explicit_head_dim: Option<usize>,
    /// GH-278: Per-layer attention type for hybrid models (Qwen3.5)
    /// None = all layers use standard attention
    /// Some(vec) = "attention" or "linear" per layer
    pub layer_types: Option<Vec<String>>,
    /// GH-278: Linear attention key head dimension (Qwen3.5: 128)
    pub linear_key_head_dim: Option<usize>,
    /// GH-278: Linear attention value head dimension (Qwen3.5: 128)
    pub linear_value_head_dim: Option<usize>,
    /// GH-278: Number of key heads in linear attention (Qwen3.5: 16)
    pub linear_num_key_heads: Option<usize>,
    /// GH-278: Number of value heads in linear attention (Qwen3.5: 32)
    pub linear_num_value_heads: Option<usize>,
    /// GH-278: Conv1D kernel size for linear attention (Qwen3.5: 4)
    pub linear_conv_kernel_dim: Option<usize>,
    /// GH-279: Architecture constraints for weight validation
    pub constraints: Option<crate::gguf::ArchConstraints>,
    /// ALB-010: Number of MoE experts per layer (None = dense FFN)
    /// Qwen3.5-35B-A3B: 256 experts
    pub num_experts: Option<usize>,
    /// ALB-010: Number of experts selected per token
    /// Qwen3.5-35B-A3B: 8 (top-8 routing)
    pub num_experts_per_tok: Option<usize>,
    /// ALB-010: Expert FFN intermediate dimension
    /// Qwen3.5-35B-A3B: 512 (vs 2048 for dense intermediate_dim)
    pub expert_intermediate_size: Option<usize>,
}

impl GpuModelConfig {
    /// Head dimension — uses explicit override if set, otherwise hidden_dim / num_heads
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.explicit_head_dim
            .unwrap_or_else(|| self.hidden_dim / self.num_heads)
    }

    /// K/V dimension for GQA (num_kv_heads * head_dim)
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// Total QKV projection output dimension
    /// Q = num_heads * head_dim, K = V = num_kv_heads * head_dim
    /// When explicit_head_dim is set (e.g., Qwen3-Coder: head_dim=128, hidden=2048),
    /// num_heads * head_dim != hidden_dim, so we must use the actual head_dim.
    #[inline]
    pub fn qkv_dim(&self) -> usize {
        let hd = self.head_dim();
        (self.num_heads + 2 * self.num_kv_heads) * hd
    }

    /// Q projection output dimension (num_heads * head_dim)
    /// May differ from hidden_dim when explicit_head_dim is set.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim()
    }

    /// Whether this is a GQA model (num_kv_heads < num_heads)
    #[inline]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads < self.num_heads
    }

    /// GH-278: Whether a specific layer uses linear attention
    ///
    /// Accepts both HF naming (`linear_attention`) and short naming (`linear`).
    #[inline]
    pub fn is_linear_layer(&self, block_idx: usize) -> bool {
        self.layer_types
            .as_ref()
            .and_then(|lt| lt.get(block_idx))
            .is_some_and(|t| t == "linear" || t == "linear_attention")
    }

    /// GH-278: Linear attention key dimension (num_key_heads * key_head_dim)
    #[inline]
    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads.unwrap_or(0) * self.linear_key_head_dim.unwrap_or(0)
    }

    /// GH-278: Linear attention value dimension (num_value_heads * value_head_dim)
    #[inline]
    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads.unwrap_or(0) * self.linear_value_head_dim.unwrap_or(0)
    }

    /// GH-278: Linear attention conv dimension (2 * key_dim + value_dim)
    #[inline]
    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }

    /// ALB-010: Whether this model uses Mixture-of-Experts
    #[inline]
    pub fn is_moe(&self) -> bool {
        self.num_experts.is_some_and(|n| n > 1)
    }
}

/// Configuration for GPU text generation (M14: E2E Inference)
#[derive(Debug, Clone)]
pub struct GpuGenerateConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Sampling temperature (0.0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (1 = greedy)
    pub top_k: usize,
    /// Stop token IDs
    pub stop_tokens: Vec<usize>,
    /// Enable debug tracing (F-COV-95: Added for cli/inference.rs compatibility)
    pub trace: bool,
}

impl Default for GpuGenerateConfig {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
            trace: false,
        }
    }
}

impl GpuGenerateConfig {
    /// Create config for deterministic (greedy) generation
    #[must_use]
    pub fn deterministic(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            temperature: 0.0,
            top_k: 1,
            stop_tokens: Vec::new(),
            trace: false,
        }
    }

    /// Create config with temperature and top-k sampling
    #[must_use]
    pub fn with_sampling(max_tokens: usize, temperature: f32, top_k: usize) -> Self {
        Self {
            max_tokens,
            temperature,
            top_k,
            stop_tokens: Vec::new(),
            trace: false,
        }
    }

    /// Add stop tokens to config
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<usize>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }
}

/// Pre-allocated attention buffers for optimized incremental decoding (M17)
///
/// Eliminates per-token memory allocation during incremental generation by
/// reusing pre-allocated buffers for Q, attention scores, and output.
#[derive(Debug)]
pub struct AttentionBuffers {
    /// Q buffer for single-token attention [hidden_dim]
    pub q_buffer: Vec<f32>,
    /// Attention scores buffer [num_heads * max_seq_len]
    pub scores_buffer: Vec<f32>,
    /// Attention output buffer [hidden_dim]
    pub output_buffer: Vec<f32>,
    /// K/V projection buffer [hidden_dim]
    pub kv_proj_buffer: Vec<f32>,
    /// Intermediate FFN buffer [intermediate_dim]
    pub ffn_buffer: Vec<f32>,
    /// Max sequence length these buffers support
    pub max_seq_len: usize,
}

impl AttentionBuffers {
    /// Create pre-allocated attention buffers from model config
    ///
    /// # Arguments
    ///
    /// * `config` - Model configuration
    /// * `max_seq_len` - Maximum sequence length to support
    #[must_use]
    pub fn new(config: &GpuModelConfig, max_seq_len: usize) -> Self {
        Self {
            q_buffer: vec![0.0; config.hidden_dim],
            scores_buffer: vec![0.0; config.num_heads * max_seq_len],
            output_buffer: vec![0.0; config.hidden_dim],
            kv_proj_buffer: vec![0.0; config.hidden_dim],
            ffn_buffer: vec![0.0; config.intermediate_dim],
            max_seq_len,
        }
    }

    /// Reset all buffers to zero (for reuse)
    pub fn reset(&mut self) {
        self.q_buffer.fill(0.0);
        self.scores_buffer.fill(0.0);
        self.output_buffer.fill(0.0);
        self.kv_proj_buffer.fill(0.0);
        self.ffn_buffer.fill(0.0);
    }
}

// =============================================================================
// PMAT-284: ValidatedGpuWeights — Poka-Yoke for GpuModel construction
// =============================================================================

/// PMAT-284: Validated weight bundle for GpuModel construction.
///
/// Groups all weight tensors with their config and validates dimensions
/// at construction. Makes it impossible to:
/// 1. Swap lm_head_weight and lm_head_weight_t (newtypes)
/// 2. Pass mismatched embedding/norm sizes (dimension checks)
/// 3. Pass wrong number of block weights (layer count check)
///
/// Private inner fields — can ONLY be constructed via `ValidatedGpuWeights::new()`.
pub struct ValidatedGpuWeights {
    pub(crate) config: GpuModelConfig,
    pub(crate) embedding_weights: Vec<f32>,
    pub(crate) block_weights: Vec<BlockWeights>,
    pub(crate) final_norm_weight: Vec<f32>,
    pub(crate) final_norm_bias: Vec<f32>,
    pub(crate) lm_head_weight: LmHeadWeight,
    pub(crate) lm_head_weight_t: LmHeadWeightTransposed,
    pub(crate) lm_head_bias: Vec<f32>,
}

/// Error from weight validation.
#[derive(Debug)]
pub struct GpuWeightError {
    /// Which field failed
    pub field: &'static str,
    /// Expected vs actual
    pub reason: String,
}

impl std::fmt::Display for GpuWeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PMAT-284 weight validation '{}': {}",
            self.field, self.reason
        )
    }
}

impl std::error::Error for GpuWeightError {}

impl ValidatedGpuWeights {
    /// Validate and construct a weight bundle.
    ///
    /// Checks:
    /// 1. embedding_weights.len() == vocab_size * hidden_dim
    /// 2. block_weights.len() == num_layers
    /// 3. final_norm_weight.len() == hidden_dim
    /// 4. lm_head_weight.len() == vocab_size * hidden_dim
    /// 5. lm_head_weight_t.len() == hidden_dim * vocab_size
    pub fn new(
        config: GpuModelConfig,
        embedding_weights: Vec<f32>,
        block_weights: Vec<BlockWeights>,
        final_norm_weight: Vec<f32>,
        final_norm_bias: Vec<f32>,
        lm_head_weight: LmHeadWeight,
        lm_head_weight_t: LmHeadWeightTransposed,
        lm_head_bias: Vec<f32>,
    ) -> std::result::Result<Self, GpuWeightError> {
        let expected_embed = config.vocab_size * config.hidden_dim;
        if embedding_weights.len() != expected_embed {
            return Err(GpuWeightError {
                field: "embedding_weights",
                reason: format!(
                    "expected {} (vocab_size={} * hidden_dim={}), got {}",
                    expected_embed,
                    config.vocab_size,
                    config.hidden_dim,
                    embedding_weights.len()
                ),
            });
        }

        if block_weights.len() != config.num_layers {
            return Err(GpuWeightError {
                field: "block_weights",
                reason: format!(
                    "expected {} layers, got {}",
                    config.num_layers,
                    block_weights.len()
                ),
            });
        }

        if final_norm_weight.len() != config.hidden_dim {
            return Err(GpuWeightError {
                field: "final_norm_weight",
                reason: format!(
                    "expected {} (hidden_dim), got {}",
                    config.hidden_dim,
                    final_norm_weight.len()
                ),
            });
        }

        let expected_lm = config.vocab_size * config.hidden_dim;
        if lm_head_weight.0.len() != expected_lm {
            return Err(GpuWeightError {
                field: "lm_head_weight",
                reason: format!(
                    "expected {} (vocab_size={} * hidden_dim={}), got {}",
                    expected_lm,
                    config.vocab_size,
                    config.hidden_dim,
                    lm_head_weight.0.len()
                ),
            });
        }

        if lm_head_weight_t.0.len() != expected_lm {
            return Err(GpuWeightError {
                field: "lm_head_weight_t",
                reason: format!(
                    "expected {} (hidden_dim={} * vocab_size={}), got {}",
                    expected_lm,
                    config.hidden_dim,
                    config.vocab_size,
                    lm_head_weight_t.0.len()
                ),
            });
        }

        Ok(Self {
            config,
            embedding_weights,
            block_weights,
            final_norm_weight,
            final_norm_bias,
            lm_head_weight,
            lm_head_weight_t,
            lm_head_bias,
        })
    }
}
