//! GPU Model (PMAT-802)
//!
//! GpuModel implementation. Types in types.rs for file health.

use super::super::{
    cpu_matmul, cpu_matmul_transposed_simd, exceeds_gpu_buffer_limit, HybridScheduler,
    StreamingKVCache,
};
#[cfg(feature = "cuda")]
use super::core::CudaScheduler;
use super::linear_attn::LinearAttnState;
use super::types::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModelConfig, ValidatedGpuWeights,
    WeightType,
};
use crate::apr_transformer::{ActivationStats, ForwardTrace, LayerActivation};
use crate::error::{RealizarError, Result};

/// GPU-accelerated model for M3 parity (128 tok/s target)
///
/// Wraps standard Model and uses HybridScheduler for GPU-accelerated
/// matrix multiplications in the forward pass.
///
/// # Phase 43: Test Executor Support
///
/// The model supports dependency injection via `test_executor` for testing
/// the forward pass without actual GPU hardware. Use `with_test_executor()`
/// to inject a mock executor.
pub struct GpuModel {
    /// Embedding weights (vocab_size x hidden_dim)
    pub(crate) embedding_weights: Vec<f32>,
    /// Linear layer weights for each block
    /// Each block has: attn_q, attn_k, attn_v, attn_out, ffn_fc1, ffn_fc2
    pub(crate) block_weights: Vec<BlockWeights>,
    /// Final layer norm weights
    pub(crate) final_norm_weight: Vec<f32>,
    pub(crate) final_norm_bias: Vec<f32>,
    /// LM head weights (hidden_dim x vocab_size)
    pub(crate) lm_head_weight: Vec<f32>,
    /// LM head weights transposed (vocab_size x hidden_dim) for fast CPU inference
    pub(crate) lm_head_weight_t: Vec<f32>,
    pub(crate) lm_head_bias: Vec<f32>,
    /// GPU scheduler (HybridScheduler - may force CPU for m=1)
    pub(crate) scheduler: HybridScheduler,
    /// IMP-1003: Optional CUDA-only scheduler that ALWAYS uses GPU
    /// When present, this scheduler is preferred over HybridScheduler for matmul
    #[cfg(feature = "cuda")]
    pub(crate) cuda_scheduler: Option<CudaScheduler>,
    /// Model configuration
    pub config: GpuModelConfig,
    /// Pre-allocated attention buffers for optimized incremental decoding (M17)
    pub(crate) attention_buffers: Option<AttentionBuffers>,
    /// Phase 43: Test executor for dependency injection
    ///
    /// When present, this executor is used instead of HybridScheduler or CudaScheduler.
    /// Enables testing forward pass logic without actual GPU hardware.
    ///
    /// Note: Explicit `+ Send + Sync` bounds required for axum Router compatibility.
    /// The trait already requires Send + Sync, but trait objects need explicit bounds.
    pub(crate) test_executor:
        Option<Box<dyn super::super::executor::GpuExecutorTrait + Send + Sync>>,
    /// GH-278: Persistent linear attention state for hybrid models (Qwen3.5)
    ///
    /// Holds recurrent state and conv buffers for Gated Delta Net layers.
    /// Initialized lazily on first forward pass when `config.layer_types` has
    /// "linear" entries. Reset when starting a new sequence.
    pub(crate) linear_attn_state: Option<LinearAttnState>,
}

include!("wrapper.rs");
include!("traced_forward.rs");
