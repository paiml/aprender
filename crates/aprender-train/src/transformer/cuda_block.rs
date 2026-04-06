//! CUDA-accelerated Transformer Block (ENT-147 through ENT-152)
//!
//! This module provides a fully GPU-accelerated transformer block using trueno-gpu kernels.
//! All operations run on CUDA to achieve >70% GPU utilization.
//!
//! # Phase 22 Implementation Status
//!
//! - ENT-147: CUDA RMSNorm integration ✅
//! - ENT-148: CUDA Softmax integration ✅
//! - ENT-149: CUDA SiLU activation ✅
//! - ENT-150: Fused SwiGLU kernel ✅
//! - ENT-151: CUDA backward pass ✅
//! - ENT-152: CudaTransformer wrapper ✅

#![allow(dead_code)]
// SAFETY: This module performs GPU memory transfers via CUDA driver FFI.
// The unsafe blocks are limited to copy_from_host_async / copy_to_host_async
// where we guarantee the host buffer outlives the async operation by syncing
// the stream before the buffer goes out of scope.
#![allow(unsafe_code)]

// PMAT-483/entrenar#328: Per-operation profiling indices for forward pass.
// Must match StepProfiler OP_* constants in step_profiler.rs.
#[cfg(feature = "cuda")]
const OP_RMSNORM_ATTN: usize = 0;
#[cfg(feature = "cuda")]
const OP_QKV_GEMM: usize = 1;
#[cfg(feature = "cuda")]
const OP_ATTENTION: usize = 2;
#[cfg(feature = "cuda")]
const OP_O_PROJ: usize = 3;
#[cfg(feature = "cuda")]
const OP_RMSNORM_FFN: usize = 4;
#[cfg(feature = "cuda")]
const OP_GATE_UP_GEMM: usize = 5;
#[cfg(feature = "cuda")]
const OP_SILU: usize = 6;
#[cfg(feature = "cuda")]
const OP_DOWN_GEMM: usize = 7;

// PMAT-483: Per-operation profiling indices for backward pass.
#[cfg(feature = "cuda")]
const OP_LORA_FWD: usize = 8;
#[cfg(feature = "cuda")]
const OP_DOWN_BWD: usize = 9;
#[cfg(feature = "cuda")]
const OP_SWIGLU_BWD: usize = 10;
#[cfg(feature = "cuda")]
const OP_GATE_UP_BWD: usize = 11;
#[cfg(feature = "cuda")]
const OP_ATTN_BWD: usize = 12;
#[cfg(feature = "cuda")]
const OP_QKV_BWD: usize = 13;
#[cfg(feature = "cuda")]
const OP_NORM_BWD: usize = 14;
#[cfg(feature = "cuda")]
const OP_LORA_BWD: usize = 15;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
#[inline]
fn saturating_u32(v: usize) -> u32 {
    v.min(u32::MAX as usize) as u32
}

/// Consume a value without running its destructor (prevents GPU double-free).
#[cfg(feature = "cuda")]
#[inline]
fn leak<T>(val: T) {
    let _ = std::mem::ManuallyDrop::new(val);
}

#[cfg(feature = "cuda")]
use trueno_gpu::driver::{CudaContext, CudaStream, GpuBuffer};

#[cfg(feature = "cuda")]
use crate::autograd::cuda_backward::{
    batched_softmax_backward, gemm_backward_a, gemm_backward_a_fp16_dispatch,
    gemm_backward_a_fp16_dispatch_accumulate, gemm_backward_b, rms_norm_backward, silu_backward,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_forward::{
    batched_4d_gemm_forward, batched_rope_neox_backward, batched_rope_neox_forward,
    batched_softmax_forward, batched_to_interleaved_forward, batched_transpose_forward,
    cast_f32_to_f16_gpu, elementwise_mul_forward, expand_kv_heads, fused_residual_rmsnorm_forward,
    fused_swiglu_forward, gemm_f16_to_f32_forward, gemm_forward, interleaved_to_batched_forward,
    per_head_rmsnorm_forward, residual_add_forward, rms_norm_forward, scale_forward, silu_forward,
};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_optim::{adamw_step_cuda, gradient_clip_cuda, squared_sum_cuda};
#[cfg(feature = "cuda")]
use crate::autograd::cuda_tensor::Result;

#[cfg(feature = "cuda")]
use super::config::TransformerConfig;

/// CUDA-accelerated transformer block
///
/// All operations run on GPU with minimal CPU<->GPU transfers.
#[cfg(feature = "cuda")]
pub struct CudaTransformerBlock {
    /// Configuration
    config: TransformerConfig,
    /// Layer index
    layer_idx: usize,
    /// Input RMSNorm weight (gamma)
    input_norm_weight: GpuBuffer<f32>,
    /// Post-attention RMSNorm weight (gamma)
    post_attn_norm_weight: GpuBuffer<f32>,
    /// Query projection weight (hidden_size x hidden_size)
    w_q: GpuBuffer<f32>,
    /// Key projection weight (hidden_size x kv_hidden_size)
    w_k: GpuBuffer<f32>,
    /// Value projection weight (hidden_size x kv_hidden_size)
    w_v: GpuBuffer<f32>,
    /// Output projection weight (hidden_size x hidden_size)
    w_o: GpuBuffer<f32>,
    /// FFN gate projection (hidden_size x intermediate_size)
    w_gate: GpuBuffer<f32>,
    /// FFN up projection (hidden_size x intermediate_size)
    w_up: GpuBuffer<f32>,
    /// FFN down projection (intermediate_size x hidden_size)
    w_down: GpuBuffer<f32>,
    /// CUDA context
    ctx: Arc<CudaContext>,
    /// Scratch buffers for intermediate results
    scratch: CudaBlockScratch,
    /// Pre-allocated host zero buffer for zeroing norm grad buffers [hidden_size]
    norm_zero_buf: Vec<f32>,
    /// ENT-270: QK-norm weights (per-head RMSNorm, shape=[head_dim])
    q_norm_weight: Option<GpuBuffer<f32>>,
    k_norm_weight: Option<GpuBuffer<f32>>,
}

/// Preallocated scratch buffers for transformer forward/backward pass.
///
/// For fp32 blocks: per-layer (backward reads forward activations).
/// For NF4 blocks: shared across all layers (forward-only, no backward).
///
/// # Contract (C-SCRATCH-001)
///
/// - **Precondition**: Allocated with matching `config` and `max_seq_len`
/// - **Postcondition**: All buffers sized for worst-case `max_seq_len`
/// - **Invariant**: NF4 layers run sequentially — one shared scratch is safe
#[cfg(feature = "cuda")]
pub(crate) struct CudaBlockScratch {
    /// After input RMSNorm (seq_len * hidden_size)
    norm1_out: GpuBuffer<f32>,
    /// Q projection output (seq_len * hidden_size)
    q: GpuBuffer<f32>,
    /// K projection output (seq_len * kv_hidden_size)
    k: GpuBuffer<f32>,
    /// V projection output (seq_len * kv_hidden_size)
    v: GpuBuffer<f32>,
    /// Attention scores (num_heads * seq_len * seq_len)
    attn_scores: GpuBuffer<f32>,
    /// Attention output (seq_len * q_dim)
    attn_out: GpuBuffer<f32>,
    /// Output projection result
    o_proj_out: GpuBuffer<f32>,
    /// Residual after attention
    residual1: GpuBuffer<f32>,
    /// After post-attention RMSNorm
    norm2_out: GpuBuffer<f32>,
    /// FFN gate output (seq_len * intermediate_size)
    gate_out: GpuBuffer<f32>,
    /// FFN up output (seq_len * intermediate_size)
    up_out: GpuBuffer<f32>,
    /// FFN fused SwiGLU output: SiLU(gate) * up (seq_len * intermediate_size)
    swiglu_out: GpuBuffer<f32>,
    /// FFN down projection output
    ffn_out: GpuBuffer<f32>,
    /// FP16 activation cast buffers for FP16 GEMM dispatch (PMAT-470)
    /// Allocated lazily on first use when FP16_GEMM=1.
    norm1_out_f16: Option<GpuBuffer<u16>>,
    attn_out_f16: Option<GpuBuffer<u16>>,
    norm2_out_f16: Option<GpuBuffer<u16>>,
    swiglu_out_f16: Option<GpuBuffer<u16>>,
    // === Seq-dependent backward scratch (per-layer for activation reuse) ===
    /// Gradient accumulator for hidden states
    grad_hidden: GpuBuffer<f32>,
    /// Gradient for SwiGLU intermediate
    grad_swiglu: GpuBuffer<f32>,
    // === Attention layout scratch buffers (GPU-only attention pipeline) ===
    /// Q in batched layout [num_heads, seq_len, head_dim]
    attn_q_batched: GpuBuffer<f32>,
    /// K/V layout temp buffer [num_heads, seq_len, head_dim]
    attn_kv_temp: GpuBuffer<f32>,
    /// K transposed / second temp [num_heads, head_dim, seq_len]
    attn_kv_temp2: GpuBuffer<f32>,
    // === Attention backward scratch (seq-dependent) ===
    /// Gradient for attention scores [num_heads * seq_len * seq_len]
    /// Kept separate from attn_scores because softmax backward reads y while writing grad_x
    grad_attn_scores: GpuBuffer<f32>,
    // === LoRA scratch buffers (ENT-153: QLoRA) ===
    /// LoRA intermediate: x @ A, sized [max_seq_len * max_lora_rank]
    lora_inter: GpuBuffer<f32>,
    /// LoRA temp for scaled addition, sized [max_seq_len * max_proj_dim]
    /// (reuses largest projection dimension for Q/V LoRA output)
    lora_temp: GpuBuffer<f32>,
    /// Sequential position indices [0, 1, ..., max_seq_len-1] for batched RoPE
    rope_positions: GpuBuffer<u32>,
    /// PMAT-420: Contiguous causal mask for current seq_len [seq_len * seq_len]
    causal_mask_contiguous: GpuBuffer<f32>,
    /// PMAT-420: seq_len that causal_mask_contiguous was generated for (cache key)
    pub(crate) causal_mask_cached_seq_len: usize,
    /// PMAT-483/entrenar#328: Per-operation timing accumulator (microseconds).
    /// Index matches StepProfiler OP_* constants. Accumulated across layers per step.
    /// Zero-overhead when profiling disabled (check op_profiling_enabled first).
    pub(crate) op_us: [u64; 16],
    /// Whether per-op profiling is active this step
    pub(crate) op_profiling_enabled: bool,
}

#[cfg(feature = "cuda")]
impl CudaBlockScratch {
    /// PMAT-483: Start timing an operation (no-op if profiling disabled).
    #[inline]
    pub(crate) fn op_begin(&self) -> Option<std::time::Instant> {
        if self.op_profiling_enabled {
            Some(std::time::Instant::now())
        } else {
            None
        }
    }

    /// PMAT-483: Record elapsed time for an operation.
    #[inline]
    pub(crate) fn op_end(&mut self, start: Option<std::time::Instant>, op: usize) {
        if let Some(t) = start {
            if op < 16 {
                self.op_us[op] += t.elapsed().as_micros() as u64;
            }
        }
    }

    /// Zero all forward scratch buffers to prevent backward gradient contamination.
    /// entrenar#318 Tier 1: GPU-side memset via cuMemsetD32Async (no PCIe transfer).
    /// Max sequence length this scratch was allocated for.
    pub(crate) fn max_seq_len(&self, hidden_size: usize) -> usize {
        self.norm1_out.len() / hidden_size.max(1)
    }

    #[rustfmt::skip]
    pub(crate) fn zero_forward_buffers(&mut self, stream: &CudaStream) {
        let z = |b: &mut GpuBuffer<f32>| { b.zero_async(stream).ok(); };
        z(&mut self.norm1_out); z(&mut self.q); z(&mut self.k); z(&mut self.v); z(&mut self.attn_scores); z(&mut self.attn_out);
        z(&mut self.o_proj_out); z(&mut self.residual1); z(&mut self.norm2_out); z(&mut self.gate_out); z(&mut self.up_out);
        z(&mut self.swiglu_out); z(&mut self.ffn_out); z(&mut self.attn_q_batched); z(&mut self.attn_kv_temp); z(&mut self.attn_kv_temp2);
        z(&mut self.grad_hidden); z(&mut self.grad_swiglu); z(&mut self.grad_attn_scores); z(&mut self.lora_inter); z(&mut self.lora_temp);
        self.causal_mask_cached_seq_len = 0;
    }

    /// Allocate scratch buffers for a given model config and max sequence length.
    ///
    /// # Contract (C-SCRATCH-001)
    ///
    /// All buffer sizes are deterministic from (config, max_seq_len).
    pub(crate) fn new(
        config: &TransformerConfig,
        max_seq_len: usize,
        ctx: &Arc<CudaContext>,
        lora_rank: usize,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim();
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_attention_heads;
        let head_dim = config.head_dim();

        // LoRA scratch: max(q_dim, kv_hidden) for the largest projection output
        let max_proj_dim = q_dim.max(kv_hidden_size);
        // Minimum 1 element to avoid zero-size GPU allocation
        let lora_inter_size = (max_seq_len * lora_rank).max(1);
        let lora_temp_size = (max_seq_len * max_proj_dim).max(1);

        // C-CAUSAL-001: Precompute causal mask [seq × seq] — shared across all heads
        // 4 MB for seq=1024. Applied per-head in compute_attention_cuda.
        let causal_mask_data: Vec<f32> = (0..max_seq_len * max_seq_len)
            .map(|idx| {
                let row = idx / max_seq_len;
                let col = idx % max_seq_len;
                if col <= row {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        Ok(Self {
            norm1_out: GpuBuffer::new(ctx, max_seq_len * hidden_size)?,
            q: GpuBuffer::new(ctx, max_seq_len * q_dim)?,
            k: GpuBuffer::new(ctx, max_seq_len * kv_hidden_size)?,
            v: GpuBuffer::new(ctx, max_seq_len * kv_hidden_size)?,
            attn_scores: GpuBuffer::new(ctx, num_heads * max_seq_len * max_seq_len)?,
            attn_out: GpuBuffer::new(ctx, max_seq_len * q_dim)?,
            o_proj_out: GpuBuffer::new(ctx, max_seq_len * hidden_size)?,
            residual1: GpuBuffer::new(ctx, max_seq_len * hidden_size)?,
            norm2_out: GpuBuffer::new(ctx, max_seq_len * hidden_size)?,
            gate_out: GpuBuffer::new(ctx, max_seq_len * intermediate_size)?,
            up_out: GpuBuffer::new(ctx, max_seq_len * intermediate_size)?,
            swiglu_out: GpuBuffer::new(ctx, max_seq_len * intermediate_size)?,
            ffn_out: GpuBuffer::new(ctx, max_seq_len * hidden_size)?,
            norm1_out_f16: None,
            attn_out_f16: None,
            norm2_out_f16: None,
            swiglu_out_f16: None,
            grad_hidden: GpuBuffer::new(ctx, max_seq_len * hidden_size)?,
            grad_swiglu: GpuBuffer::new(ctx, max_seq_len * intermediate_size)?,
            attn_q_batched: GpuBuffer::new(ctx, num_heads * max_seq_len * head_dim)?,
            attn_kv_temp: GpuBuffer::new(ctx, num_heads * max_seq_len * head_dim)?,
            attn_kv_temp2: GpuBuffer::new(ctx, num_heads * max_seq_len * head_dim)?,
            grad_attn_scores: GpuBuffer::new(
                ctx,
                num_heads * max_seq_len * max_seq_len.max(head_dim),
            )?,
            lora_inter: GpuBuffer::new(ctx, lora_inter_size)?,
            lora_temp: GpuBuffer::new(ctx, lora_temp_size)?,
            rope_positions: {
                let positions: Vec<u32> = (0..max_seq_len as u32).collect();
                let mut buf = GpuBuffer::new(ctx, max_seq_len)?;
                buf.copy_from_host(&positions)?;
                buf
            },
            causal_mask_contiguous: GpuBuffer::from_host(ctx, &causal_mask_data)?,
            causal_mask_cached_seq_len: max_seq_len,
            op_us: [0u64; 16],
            op_profiling_enabled: false,
        })
    }

    /// PMAT-420: Prepare a contiguous [seq_len * seq_len] causal mask. Cached: only regenerates
    /// when seq_len changes. Cost: O(seq_len^2) CPU + one H2D upload (~0.01ms for seq=256).
    pub(crate) fn prepare_causal_mask(
        &mut self,
        seq_len: usize,
        ctx: &Arc<CudaContext>,
    ) -> crate::autograd::cuda_tensor::Result<()> {
        if seq_len == self.causal_mask_cached_seq_len {
            return Ok(());
        }
        let mask_data: Vec<f32> = (0..seq_len * seq_len)
            .map(|idx| {
                let row = idx / seq_len;
                let col = idx % seq_len;
                if col <= row {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        self.causal_mask_contiguous = GpuBuffer::from_host(ctx, &mask_data)?;
        self.causal_mask_cached_seq_len = seq_len;
        Ok(())
    }
}

/// Shared gradient workspace for weight gradients (one per model, NOT per layer).
///
/// # Contract (C-GRADWS-001)
///
/// Backward processes layers sequentially — only one layer's weight gradients
/// are computed at a time. Sharing this workspace across layers saves
/// `(L-1) * per_layer_grad_weight_elements * 4` bytes of VRAM.
///
/// For Qwen3-4B: saves 35 * 372 MB = 13.0 GB.
///
/// - **Precondition**: Allocated once before training loop starts
/// - **Postcondition**: After backward() for layer i, contains layer i's weight gradients
/// - **Invariant**: Buffer sizes match model config; never reallocated during training
#[cfg(feature = "cuda")]
pub struct CudaGradWorkspace {
    /// Gradient for input norm weight [hidden_size]
    pub(crate) grad_input_norm: GpuBuffer<f32>,
    /// Gradient for post-attention norm weight [hidden_size]
    pub(crate) grad_post_attn_norm: GpuBuffer<f32>,
    /// Gradient for FFN gate projection [hidden_size * intermediate_size]
    pub(crate) grad_gate: GpuBuffer<f32>,
    /// Gradient for FFN up projection [hidden_size * intermediate_size]
    pub(crate) grad_up: GpuBuffer<f32>,
    /// Gradient for FFN down projection [intermediate_size * hidden_size]
    pub(crate) grad_down: GpuBuffer<f32>,
    /// Gradient for Q projection weight [q_dim * hidden_size]
    pub(crate) grad_w_q: GpuBuffer<f32>,
    /// Gradient for K projection weight [hidden_size * kv_hidden_size]
    pub(crate) grad_w_k: GpuBuffer<f32>,
    /// Gradient for V projection weight [hidden_size * kv_hidden_size]
    pub(crate) grad_w_v: GpuBuffer<f32>,
    /// Gradient for output projection weight [hidden_size * q_dim]
    pub(crate) grad_w_o: GpuBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl CudaGradWorkspace {
    /// Allocate shared gradient workspace for the given model config.
    ///
    /// Called once per training run. GEMM weight gradients are fully overwritten
    /// by each backward pass. Norm gradients use atomicAdd accumulation and MUST
    /// be zeroed before each rms_norm_backward call (see `zero_norm_grads`).
    pub fn new(ctx: &Arc<CudaContext>, config: &TransformerConfig) -> Result<Self> {
        let h = config.hidden_size;
        let q = config.q_dim();
        let kv = config.num_kv_heads * config.head_dim();
        let i = config.intermediate_size;

        Ok(Self {
            grad_input_norm: GpuBuffer::new(ctx, h)?,
            grad_post_attn_norm: GpuBuffer::new(ctx, h)?,
            grad_gate: GpuBuffer::new(ctx, h * i)?,
            grad_up: GpuBuffer::new(ctx, h * i)?,
            grad_down: GpuBuffer::new(ctx, i * h)?,
            grad_w_q: GpuBuffer::new(ctx, q * h)?,
            grad_w_k: GpuBuffer::new(ctx, h * kv)?,
            grad_w_v: GpuBuffer::new(ctx, h * kv)?,
            grad_w_o: GpuBuffer::new(ctx, h * q)?,
        })
    }

    /// Zero norm gradient buffers before rms_norm_backward calls.
    ///
    /// The BatchedRmsNormBackwardKernel accumulates grad_gamma via atomicAdd,
    /// so these buffers MUST be zeroed before each backward pass. Without this,
    /// grad_gamma accumulates across steps → exploding norm gradients.
    pub fn zero_norm_grads(&mut self, zero_buf: &[f32]) -> Result<()> {
        let n = self.grad_input_norm.len();
        self.grad_input_norm.copy_from_host(&zero_buf[..n]).map_err(|e| {
            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                "Failed to zero grad_input_norm: {e:?}"
            ))
        })?;
        self.grad_post_attn_norm.copy_from_host(&zero_buf[..n]).map_err(|e| {
            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                "Failed to zero grad_post_attn_norm: {e:?}"
            ))
        })?;
        Ok(())
    }
}

/// GPU-resident AdamW optimizer state for one transformer block.
///
/// Stores first (m) and second (v) moment estimates for all 9 weight tensors:
/// 7 matmul weights + 2 RMSNorm weights. All buffers live on GPU to avoid
/// CPU↔GPU transfers during training.
///
/// # Contract (C-OPTSTATE-001)
///
/// - **Precondition**: CUDA context valid, all buffers allocated to match weight dimensions
/// - **Postcondition**: m and v buffers initialized to zero (unbiased start)
/// - **Invariant**: Buffer sizes immutable after creation; m/v never reallocated
#[cfg(feature = "cuda")]
pub struct GpuBlockOptimizerState {
    // Attention projection optimizer states
    m_w_q: GpuBuffer<f32>,
    v_w_q: GpuBuffer<f32>,
    m_w_k: GpuBuffer<f32>,
    v_w_k: GpuBuffer<f32>,
    m_w_v: GpuBuffer<f32>,
    v_w_v: GpuBuffer<f32>,
    m_w_o: GpuBuffer<f32>,
    v_w_o: GpuBuffer<f32>,
    // FFN projection optimizer states
    m_w_gate: GpuBuffer<f32>,
    v_w_gate: GpuBuffer<f32>,
    m_w_up: GpuBuffer<f32>,
    v_w_up: GpuBuffer<f32>,
    m_w_down: GpuBuffer<f32>,
    v_w_down: GpuBuffer<f32>,
    // RMSNorm weight optimizer states
    m_input_norm: GpuBuffer<f32>,
    v_input_norm: GpuBuffer<f32>,
    m_post_attn_norm: GpuBuffer<f32>,
    v_post_attn_norm: GpuBuffer<f32>,
}

/// ALB-118: Download GPU optimizer state to host for checkpointing.
#[cfg(feature = "cuda")]
impl GpuBlockOptimizerState {
    /// Returns (suffix, data) pairs for all 18 m/v buffers.
    /// Suffix is e.g. "m.w_q", "v.w_gate" — caller prefixes with layer index.
    pub fn download_to_host(
        &self,
    ) -> crate::autograd::cuda_tensor::Result<Vec<(String, Vec<f32>)>> {
        let dl = |name: &str,
                  buf: &GpuBuffer<f32>|
         -> crate::autograd::cuda_tensor::Result<(String, Vec<f32>)> {
            let mut host = vec![0.0f32; buf.len()];
            buf.copy_to_host(&mut host).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "optimizer D2H {name}: {e}"
                ))
            })?;
            Ok((name.to_string(), host))
        };
        Ok(vec![
            dl("m.w_q", &self.m_w_q)?,
            dl("v.w_q", &self.v_w_q)?,
            dl("m.w_k", &self.m_w_k)?,
            dl("v.w_k", &self.v_w_k)?,
            dl("m.w_v", &self.m_w_v)?,
            dl("v.w_v", &self.v_w_v)?,
            dl("m.w_o", &self.m_w_o)?,
            dl("v.w_o", &self.v_w_o)?,
            dl("m.w_gate", &self.m_w_gate)?,
            dl("v.w_gate", &self.v_w_gate)?,
            dl("m.w_up", &self.m_w_up)?,
            dl("v.w_up", &self.v_w_up)?,
            dl("m.w_down", &self.m_w_down)?,
            dl("v.w_down", &self.v_w_down)?,
            dl("m.input_norm", &self.m_input_norm)?,
            dl("v.input_norm", &self.v_input_norm)?,
            dl("m.post_attn_norm", &self.m_post_attn_norm)?,
            dl("v.post_attn_norm", &self.v_post_attn_norm)?,
        ])
    }

    /// ALB-118: Upload host optimizer state to GPU (checkpoint resume).
    /// Missing keys are silently skipped (buffer stays zero-initialized).
    pub fn restore_from_host(
        &mut self,
        data: &std::collections::HashMap<String, Vec<f32>>,
    ) -> crate::autograd::cuda_tensor::Result<()> {
        let ul = |name: &str,
                  buf: &mut GpuBuffer<f32>,
                  data: &std::collections::HashMap<String, Vec<f32>>|
         -> crate::autograd::cuda_tensor::Result<()> {
            if let Some(host_data) = data.get(name) {
                if host_data.len() == buf.len() {
                    buf.copy_from_host(host_data).map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "optimizer H2D {name}: {e}"
                        ))
                    })?;
                }
            }
            Ok(())
        };
        ul("m.w_q", &mut self.m_w_q, data)?;
        ul("v.w_q", &mut self.v_w_q, data)?;
        ul("m.w_k", &mut self.m_w_k, data)?;
        ul("v.w_k", &mut self.v_w_k, data)?;
        ul("m.w_v", &mut self.m_w_v, data)?;
        ul("v.w_v", &mut self.v_w_v, data)?;
        ul("m.w_o", &mut self.m_w_o, data)?;
        ul("v.w_o", &mut self.v_w_o, data)?;
        ul("m.w_gate", &mut self.m_w_gate, data)?;
        ul("v.w_gate", &mut self.v_w_gate, data)?;
        ul("m.w_up", &mut self.m_w_up, data)?;
        ul("v.w_up", &mut self.v_w_up, data)?;
        ul("m.w_down", &mut self.m_w_down, data)?;
        ul("v.w_down", &mut self.v_w_down, data)?;
        ul("m.input_norm", &mut self.m_input_norm, data)?;
        ul("v.input_norm", &mut self.v_input_norm, data)?;
        ul("m.post_attn_norm", &mut self.m_post_attn_norm, data)?;
        ul("v.post_attn_norm", &mut self.v_post_attn_norm, data)?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl CudaTransformerBlock {
    /// Create a new CUDA transformer block from CPU tensors
    ///
    /// Uploads all weights to GPU memory.
    pub fn new(
        config: &TransformerConfig,
        layer_idx: usize,
        ctx: Arc<CudaContext>,
        input_norm_weight: &[f32],
        post_attn_norm_weight: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        max_seq_len: usize,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim(); // num_heads * head_dim (may differ from hidden_size)
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;
        let num_heads = config.num_attention_heads;

        // Upload weights to GPU
        let input_norm_weight = GpuBuffer::from_host(&ctx, input_norm_weight)?;
        let post_attn_norm_weight = GpuBuffer::from_host(&ctx, post_attn_norm_weight)?;
        let w_q = GpuBuffer::from_host(&ctx, w_q)?;
        let w_k = GpuBuffer::from_host(&ctx, w_k)?;
        let w_v = GpuBuffer::from_host(&ctx, w_v)?;
        let w_o = GpuBuffer::from_host(&ctx, w_o)?;
        let w_gate = GpuBuffer::from_host(&ctx, w_gate)?;
        let w_up = GpuBuffer::from_host(&ctx, w_up)?;
        let w_down = GpuBuffer::from_host(&ctx, w_down)?;

        // C-CAUSAL-001: Precompute causal mask for NF4 path
        let single_mask: Vec<f32> = (0..max_seq_len * max_seq_len)
            .map(|idx| {
                let row = idx / max_seq_len;
                let col = idx % max_seq_len;
                if col <= row {
                    0.0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect();
        // Allocate scratch buffers — Q and attn_out need q_dim, not hidden_size
        let scratch = CudaBlockScratch {
            norm1_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            q: GpuBuffer::new(&ctx, max_seq_len * q_dim)?,
            k: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            v: GpuBuffer::new(&ctx, max_seq_len * kv_hidden_size)?,
            attn_scores: GpuBuffer::new(&ctx, num_heads * max_seq_len * max_seq_len)?,
            attn_out: GpuBuffer::new(&ctx, max_seq_len * q_dim)?,
            o_proj_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            residual1: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            norm2_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            gate_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            up_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            swiglu_out: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            ffn_out: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            norm1_out_f16: None,
            attn_out_f16: None,
            norm2_out_f16: None,
            swiglu_out_f16: None,
            // Seq-dependent backward scratch
            grad_hidden: GpuBuffer::new(&ctx, max_seq_len * hidden_size)?,
            grad_swiglu: GpuBuffer::new(&ctx, max_seq_len * intermediate_size)?,
            // Attention layout scratch (all sized for num_heads, handles GQA expansion)
            attn_q_batched: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            attn_kv_temp2: GpuBuffer::new(&ctx, num_heads * max_seq_len * config.head_dim())?,
            // Attention backward gradient buffers (ENT-151b)
            // grad_attn_scores needs max(H*S*S, H*S*hd) for buffer reuse safety
            grad_attn_scores: GpuBuffer::new(
                &ctx,
                num_heads * max_seq_len * max_seq_len.max(config.head_dim()),
            )?,
            // LoRA scratch (unused for fp32 blocks, minimum allocation)
            lora_inter: GpuBuffer::new(&ctx, 1)?,
            lora_temp: GpuBuffer::new(&ctx, 1)?,
            rope_positions: {
                let positions: Vec<u32> = (0..max_seq_len as u32).collect();
                let mut buf = GpuBuffer::new(&ctx, max_seq_len)?;
                buf.copy_from_host(&positions)?;
                buf
            },
            causal_mask_contiguous: GpuBuffer::from_host(&ctx, &single_mask)?,
            causal_mask_cached_seq_len: max_seq_len,
            op_us: [0u64; 16],
            op_profiling_enabled: false,
        };

        Ok(Self {
            config: config.clone(),
            layer_idx,
            input_norm_weight,
            post_attn_norm_weight,
            w_q,
            w_k,
            w_v,
            w_o,
            w_gate,
            w_up,
            w_down,
            ctx,
            scratch,
            norm_zero_buf: vec![0.0f32; hidden_size],
            q_norm_weight: None, // ENT-270: set via set_qk_norm() after construction
            k_norm_weight: None,
        })
    }

    /// Set QK-norm weights (ENT-270). Called after construction when loading Qwen3 models.
    #[allow(dead_code)]
    pub fn set_qk_norm(&mut self, q_norm: &[f32], k_norm: &[f32]) -> Result<()> {
        self.q_norm_weight = Some(GpuBuffer::from_host(&self.ctx, q_norm)?);
        self.k_norm_weight = Some(GpuBuffer::from_host(&self.ctx, k_norm)?);
        Ok(())
    }

    /// Forward pass - all operations on GPU
    ///
    /// # Arguments
    /// * `input` - Input tensor on GPU (seq_len * hidden_size)
    /// * `output` - Output tensor on GPU (seq_len * hidden_size)
    /// * `seq_len` - Sequence length
    /// * `stream` - CUDA stream for async execution
    pub fn forward(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.q_dim();
        let kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let intermediate_size = self.config.intermediate_size;

        // === Pre-attention RMSNorm (ENT-147) ===
        rms_norm_forward(
            input,
            &self.input_norm_weight,
            &mut self.scratch.norm1_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Q, K, V Projections (CUDA GEMM) ===
        // C[seq,q_dim] = A[seq,hidden] @ B[hidden,q_dim]
        gemm_forward(
            &self.scratch.norm1_out,
            &self.w_q,
            &mut self.scratch.q,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(q_dim),
            stream,
        )?;

        gemm_forward(
            &self.scratch.norm1_out,
            &self.w_k,
            &mut self.scratch.k,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        gemm_forward(
            &self.scratch.norm1_out,
            &self.w_v,
            &mut self.scratch.v,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        // === Multi-Head Attention (GPU-only, zero CPU transfers) ===
        self.compute_attention_cuda(seq_len, stream)?;

        // === Output Projection ===
        // C[seq,hidden] = A[seq,q_dim] @ B[q_dim,hidden]
        gemm_forward(
            &self.scratch.attn_out,
            &self.w_o,
            &mut self.scratch.o_proj_out,
            saturating_u32(seq_len),
            saturating_u32(q_dim),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Residual Add (input + attention_output) ===
        cuda_add(
            input,
            &self.scratch.o_proj_out,
            &mut self.scratch.residual1,
            seq_len * hidden_size,
            stream,
        )?;

        // === Post-attention RMSNorm ===
        rms_norm_forward(
            &self.scratch.residual1,
            &self.post_attn_norm_weight,
            &mut self.scratch.norm2_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === FFN: Gate + Up Projections ===
        gemm_forward(
            &self.scratch.norm2_out,
            &self.w_gate,
            &mut self.scratch.gate_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        gemm_forward(
            &self.scratch.norm2_out,
            &self.w_up,
            &mut self.scratch.up_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // === FFN: Fused SwiGLU (ENT-150) - SiLU(gate) * up in single kernel ===
        fused_swiglu_forward(
            &self.scratch.gate_out,
            &self.scratch.up_out,
            &mut self.scratch.swiglu_out,
            saturating_u32(seq_len * intermediate_size),
            stream,
        )?;

        // === FFN: Down Projection ===
        gemm_forward(
            &self.scratch.swiglu_out,
            &self.w_down,
            &mut self.scratch.ffn_out,
            saturating_u32(seq_len),
            saturating_u32(intermediate_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Final Residual Add (residual1 + ffn_output) ===
        cuda_add(
            &self.scratch.residual1,
            &self.scratch.ffn_out,
            output,
            seq_len * hidden_size,
            stream,
        )?;

        Ok(())
    }

    /// Compute multi-head attention entirely on GPU (zero CPU transfers)
    ///
    /// # Contract (C-ATTN-001)
    ///
    /// - **Precondition**: Q [seq, hidden], K [seq, kv_hidden], V [seq, kv_hidden] on GPU
    /// - **Postcondition**: attn_out [seq, hidden] = concat(head_0..head_H) where
    ///   head_h = softmax(Q_h @ K_{kv(h)}^T / √d_k) @ V_{kv(h)}
    /// - **Invariant**: Zero gpu_to_vec / vec_to_gpu calls; numerically equivalent to CPU
    ///
    /// Uses existing trueno-gpu kernels:
    /// - `InterleavedToBatchedKernel` for Q/K/V layout conversion
    /// - `BatchedTransposeKernel` for K^T
    /// - `Batched4DGemmKernel` for Q@K^T and attn@V
    /// - `ScaleKernel` for 1/√d_k scaling
    /// - `BatchedSoftmaxKernel` for row-wise softmax
    /// - `BatchedToInterleavedKernel` for output layout conversion
    /// - D2D copies for GQA head expansion
    fn compute_attention_cuda(&mut self, seq_len: usize, stream: &CudaStream) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let seq = saturating_u32(seq_len);
        let nh = saturating_u32(num_heads);
        let nkv = saturating_u32(num_kv_heads);
        let hd = saturating_u32(head_dim);

        // PMAT-420: Ensure causal mask is contiguous for current seq_len.
        self.scratch.prepare_causal_mask(seq_len, &self.ctx)?;

        // ── ENT-270: Apply QK-norm (per-head RMSNorm) on Q and K ──────────
        // SAFETY: In-place GPU operations — CUDA kernels read all input before writing output.
        // Rust borrow checker cannot verify GPU kernel memory access patterns, so we use
        // raw pointer reborrow to allow the same buffer as both input and output.
        if let Some(ref q_norm) = self.q_norm_weight {
            for pos in 0..seq_len {
                let q_ref = unsafe { &*(std::ptr::addr_of!(self.scratch.q)) };
                per_head_rmsnorm_forward(q_ref, q_norm, &mut self.scratch.q, nh, hd, pos, stream)?;
            }
        }
        if let Some(ref k_norm) = self.k_norm_weight {
            for pos in 0..seq_len {
                let k_ref = unsafe { &*(std::ptr::addr_of!(self.scratch.k)) };
                per_head_rmsnorm_forward(k_ref, k_norm, &mut self.scratch.k, nkv, hd, pos, stream)?;
            }
        }

        // ── ENT-270: Apply RoPE (NeoX half-rotation) on Q and K ──────────
        // ALB-119: Batched launch (2 kernels) replaces per-position loop (2*seq_len kernels)
        let rope_theta = self.config.rope_theta;
        {
            let q_ref = unsafe { &*(std::ptr::addr_of!(self.scratch.q)) };
            batched_rope_neox_forward(
                q_ref,
                &mut self.scratch.q,
                &self.scratch.rope_positions,
                nh,
                hd,
                seq,
                rope_theta,
                stream,
            )?;
            let k_ref = unsafe { &*(std::ptr::addr_of!(self.scratch.k)) };
            batched_rope_neox_forward(
                k_ref,
                &mut self.scratch.k,
                &self.scratch.rope_positions,
                nkv,
                hd,
                seq,
                rope_theta,
                stream,
            )?;
        }

        // Step 1: Q interleaved [seq, num_heads * head_dim] → batched [num_heads, seq, head_dim]
        interleaved_to_batched_forward(
            &self.scratch.q,
            &mut self.scratch.attn_q_batched,
            seq,
            nh,
            hd,
            stream,
        )?;

        // Step 2: K interleaved [seq, num_kv_heads * head_dim] → batched [num_kv_heads, seq, head_dim]
        interleaved_to_batched_forward(
            &self.scratch.k,
            &mut self.scratch.attn_kv_temp,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // Step 3: GQA expansion + transpose for K
        if heads_per_kv == 1 {
            // MHA: transpose directly [num_heads, seq, head_dim] → [num_heads, head_dim, seq]
            batched_transpose_forward(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                nh,
                seq,
                hd,
                stream,
            )?;
        } else {
            // GQA: expand [num_kv_heads, seq, hd] → [num_heads, seq, hd] in attn_kv_temp2
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
            // Transpose expanded K: [num_heads, seq, hd] → [num_heads, hd, seq] in attn_kv_temp
            batched_transpose_forward(
                &self.scratch.attn_kv_temp2,
                &mut self.scratch.attn_kv_temp,
                nh,
                seq,
                hd,
                stream,
            )?;
            // Move K^T to attn_kv_temp2 for consistent naming below
            // (swap pointers via D2D copy — attn_kv_temp → attn_kv_temp2)
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            unsafe {
                self.scratch
                    .attn_kv_temp2
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "K^T buffer copy failed: {e}"
                        ))
                    })?;
            }
        }

        // Step 4: Q @ K^T → attn_scores [num_heads, seq, seq]
        // attn_q_batched: [1, num_heads, seq, head_dim]
        // attn_kv_temp2:  [1, num_heads, head_dim, seq] (K transposed)
        // attn_scores:    [1, num_heads, seq, seq]
        batched_4d_gemm_forward(
            &self.scratch.attn_q_batched,
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.attn_scores,
            1,
            nh,
            seq,
            seq,
            hd,
            stream,
        )?;

        // Step 5: Scale scores by 1/√d_k (in-place)
        let total_scores = nh * seq * seq;
        {
            // SAFETY: In-place aliasing is safe for element-wise operations where each
            // element is read before being written. ScaleKernel processes elements
            // independently. The view is forgotten to prevent double-free.
            let scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.attn_scores.as_ptr(),
                    self.scratch.attn_scores.len(),
                )
            };
            scale_forward(
                &scores_view,
                &mut self.scratch.attn_scores,
                scale,
                total_scores,
                stream,
            )?;
            leak(scores_view);
        }

        // Step 5.5 (C-CAUSAL-001): Apply causal mask — add -inf to future positions
        // Loop over heads, adding [seq, seq] mask to each head's scores slice.
        // PMAT-420: Use causal_mask_contiguous (correctly strided for seq_len)
        // instead of causal_mask (strided at max_seq_len, causes row misalignment
        // when seq_len < max_seq_len, leading to NaN after deep layers).
        {
            let seq_sq = (seq * seq) as usize;
            let mask_ptr = self.scratch.causal_mask_contiguous.as_ptr();
            let scores_base = self.scratch.attn_scores.as_ptr();
            for head in 0..nh as usize {
                let byte_offset = (head * seq_sq * 4) as u64; // f32 = 4 bytes
                let head_ptr = scores_base + byte_offset;
                // SAFETY: mask and scores_slice are non-overlapping GPU regions.
                // output aliases scores_slice — safe for element-wise add (read before write).
                // Views are leaked to prevent double-free of GPU memory.
                let mask_view = unsafe { GpuBuffer::<f32>::from_raw_parts(mask_ptr, seq_sq) };
                let scores_view = unsafe { GpuBuffer::<f32>::from_raw_parts(head_ptr, seq_sq) };
                let mut out_view = unsafe { GpuBuffer::<f32>::from_raw_parts(head_ptr, seq_sq) };
                residual_add_forward(&mask_view, &scores_view, &mut out_view, seq * seq, stream)?;
                leak(mask_view);
                leak(scores_view);
                leak(out_view);
            }
        }

        // Step 6: Row-wise softmax → attn_weights [num_heads * seq, seq] (in-place)
        let total_rows = nh * seq;
        {
            // SAFETY: In-place aliasing is safe for BatchedSoftmaxKernel which uses
            // shared memory for row-wise reduction. Each row is fully read into shared
            // memory before any output is written. The view is forgotten to prevent double-free.
            let scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.attn_scores.as_ptr(),
                    self.scratch.attn_scores.len(),
                )
            };
            batched_softmax_forward(
                &scores_view,
                &mut self.scratch.attn_scores,
                total_rows,
                seq,
                stream,
            )?;
            leak(scores_view);
        }

        // Step 7: V layout conversion + GQA expansion
        interleaved_to_batched_forward(
            &self.scratch.v,
            &mut self.scratch.attn_kv_temp,
            seq,
            nkv,
            hd,
            stream,
        )?;

        if heads_per_kv == 1 {
            // MHA: V already in [num_heads, seq, head_dim] in attn_kv_temp
        } else {
            // GQA: expand V [num_kv_heads, seq, hd] → [num_heads, seq, hd]
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
            // Copy expanded V back to attn_kv_temp for the GEMM
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            unsafe {
                self.scratch
                    .attn_kv_temp
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp2, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "V expanded buffer copy failed: {e}"
                        ))
                    })?;
            }
        }

        // Step 8: attn_weights @ V → attn_result [num_heads, seq, head_dim]
        // attn_scores:   [1, num_heads, seq, seq]
        // attn_kv_temp:  [1, num_heads, seq, head_dim]
        // → attn_q_batched: [1, num_heads, seq, head_dim] (reuse Q buffer)
        batched_4d_gemm_forward(
            &self.scratch.attn_scores,
            &self.scratch.attn_kv_temp,
            &mut self.scratch.attn_q_batched,
            1,
            nh,
            seq,
            hd,
            seq,
            stream,
        )?;

        // Step 9: Convert back to interleaved [seq, num_heads * head_dim] → attn_out
        batched_to_interleaved_forward(
            &self.scratch.attn_q_batched,
            &mut self.scratch.attn_out,
            seq,
            nh,
            hd,
            stream,
        )?;

        Ok(())
    }

    /// Get layer index
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig {
        &self.config
    }

    /// Backward pass - gradient computation on GPU (ENT-151)
    ///
    /// Computes gradients for all parameters given upstream gradient.
    ///
    /// # Arguments
    /// * `input` - Original input from forward pass (seq_len * hidden_size)
    /// * `grad_output` - Gradient from upstream layer (seq_len * hidden_size)
    /// * `grad_input` - Output: gradient w.r.t. input (seq_len * hidden_size)
    /// * `seq_len` - Sequence length
    /// * `stream` - CUDA stream for async execution
    ///
    /// # Returns
    /// Gradients are accumulated into the scratch buffers:
    /// - `scratch.grad_input_norm` - Gradient for input RMSNorm weight
    /// - `scratch.grad_post_attn_norm` - Gradient for post-attention RMSNorm weight
    /// - `scratch.grad_gate/up/down` - Gradients for FFN weights
    /// - `scratch.grad_w_q/w_k/w_v/w_o` - Gradients for attention projection weights
    #[provable_contracts_macros::contract("backward-pass-v1", equation = "backward")]
    pub fn backward(
        &mut self,
        input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let intermediate_size = self.config.intermediate_size;
        let eps = 1e-5_f32;

        // Zero norm gradient buffers before backward pass.
        // BatchedRmsNormBackwardKernel accumulates grad_gamma via atomicAdd,
        // so buffers must be zeroed before each call to prevent cross-step accumulation.
        grad_ws.zero_norm_grads(&self.norm_zero_buf)?;

        // Backward through final residual: output = residual1 + ffn_output
        // grad_output flows to BOTH residual1 (identity skip) and ffn_output path.
        self.backward_ffn(grad_output, seq_len, hidden_size, intermediate_size, stream, grad_ws)?;

        // Backward through post-attention RMSNorm (FFN path gradient only)
        self.backward_post_attn_norm(grad_input, seq_len, hidden_size, eps, stream, grad_ws)?;

        // C-RESIDUAL-001 / entrenar#313: Second residual skip gradient.
        // Forward: output = residual1 + ffn_output
        // The identity skip grad_residual1 = grad_output must bypass the
        // post-attention RMSNorm backward entirely — add AFTER norm backward.
        cuda_add_inplace(grad_input, grad_output, seq_len * hidden_size, stream)?;

        // Backward through attention: output projection, attention weights, Q/K/V projections
        // (ENT-151b: previously missing — attention params received no gradients)
        self.backward_attention(grad_input, seq_len, stream, grad_ws)?;

        // Backward through first residual connection and input RMSNorm
        self.backward_residual_and_input_norm(
            input,
            grad_output,
            grad_input,
            seq_len,
            hidden_size,
            eps,
            stream,
            grad_ws,
        )?;

        Ok(())
    }

    /// Backward through FFN: down projection, SwiGLU, gate+up projections.
    ///
    /// SwiGLU(gate, up) = silu(gate) * up
    /// ∂L/∂gate = ∂L/∂swiglu * up * silu'(gate)
    /// ∂L/∂up   = ∂L/∂swiglu * silu(gate)
    ///
    /// Buffer reuse plan (all [S,I] unless noted):
    ///   grad_swiglu  = ∂L/∂swiglu          (computed step 1, read steps 2/4/6)
    ///   swiglu_out  → temp1 (step 2)       → silu(gate) (step 4)
    ///   up_out      → grad_gate (step 3)
    ///   gate_out    → grad_up (step 6)
    ///   ffn_out     → grad_norm2_gate [S,H] (step 8)
    ///   grad_hidden → grad_norm2_up [S,H]   (step 9)
    ///   norm2_out   → accumulated grad [S,H] (step 10)
    fn backward_ffn(
        &mut self,
        grad_output: &GpuBuffer<f32>,
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        let n_inter = saturating_u32(seq_len * intermediate_size);
        let n_hidden = saturating_u32(seq_len * hidden_size);

        // Step 1: grad_swiglu = grad_ffn_out @ w_down^T  [S,I]
        gemm_backward_a(
            grad_output,
            &self.w_down,
            &mut self.scratch.grad_swiglu,
            saturating_u32(seq_len),
            saturating_u32(intermediate_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // Step 2: grad_w_down = swiglu_out^T @ grad_ffn_out  [I,H]
        // (swiglu_out free after this)
        gemm_backward_b(
            &self.scratch.swiglu_out,
            grad_output,
            &mut grad_ws.grad_down,
            saturating_u32(seq_len),
            saturating_u32(intermediate_size),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === SwiGLU backward: swiglu = silu(gate) * up ===

        // Step 3: temp1 = grad_swiglu * up_out → swiglu_out [S,I]
        elementwise_mul_forward(
            &self.scratch.grad_swiglu,
            &self.scratch.up_out,
            &mut self.scratch.swiglu_out,
            n_inter,
            stream,
        )?;

        // Step 4: grad_gate = silu_backward(gate_out, temp1) → up_out [S,I]
        // Computes: (grad_swiglu * up_out) * silu'(gate_out) = correct ∂L/∂gate
        silu_backward(
            &self.scratch.gate_out,
            &self.scratch.swiglu_out,
            &mut self.scratch.up_out,
            stream,
        )?;
        // up_out now holds grad_gate [S,I]

        // Step 5: silu_gate = silu(gate_out) → swiglu_out [S,I]
        silu_forward(&self.scratch.gate_out, &mut self.scratch.swiglu_out, n_inter, stream)?;

        // Step 6: grad_up = grad_swiglu * silu_gate → gate_out [S,I]
        elementwise_mul_forward(
            &self.scratch.grad_swiglu,
            &self.scratch.swiglu_out,
            &mut self.scratch.gate_out,
            n_inter,
            stream,
        )?;
        // gate_out now holds grad_up [S,I]

        // === Weight gradients ===

        // Step 7a: grad_w_gate = norm2_out^T @ grad_gate (in up_out)  [H,I]
        gemm_backward_b(
            &self.scratch.norm2_out,
            &self.scratch.up_out,
            &mut grad_ws.grad_gate,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // Step 7b: grad_w_up = norm2_out^T @ grad_up (in gate_out)  [H,I]
        gemm_backward_b(
            &self.scratch.norm2_out,
            &self.scratch.gate_out,
            &mut grad_ws.grad_up,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // === Input gradient (accumulate gate + up paths) ===

        // Step 8: grad_norm2_gate = grad_gate @ w_gate^T → ffn_out [S,H]
        gemm_backward_a(
            &self.scratch.up_out,
            &self.w_gate,
            &mut self.scratch.ffn_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // Step 9: grad_norm2_up = grad_up @ w_up^T → grad_hidden [S,H]
        gemm_backward_a(
            &self.scratch.gate_out,
            &self.w_up,
            &mut self.scratch.grad_hidden,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            saturating_u32(intermediate_size),
            stream,
        )?;

        // Step 10: norm2_out = grad_norm2_gate + grad_norm2_up  [S,H]
        residual_add_forward(
            &self.scratch.ffn_out,
            &self.scratch.grad_hidden,
            &mut self.scratch.norm2_out,
            n_hidden,
            stream,
        )?;

        Ok(())
    }

    /// Backward through post-attention RMSNorm.
    fn backward_post_attn_norm(
        &mut self,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        // D2D copy norm2_out → grad_hidden (avoids D2H + H2D round-trip)
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            self.scratch
                .grad_hidden
                .copy_from_buffer_async(&self.scratch.norm2_out, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "Backward norm D2D copy failed: {e}"
                    ))
                })?;
        }

        rms_norm_backward(
            &self.scratch.residual1,
            &self.post_attn_norm_weight,
            &self.scratch.grad_hidden,
            grad_input,
            &mut grad_ws.grad_post_attn_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )
    }

    /// Backward through multi-head attention (ENT-151b)
    ///
    /// Reverses the forward attention pipeline:
    /// output_proj → layout → attn_weights@V → softmax → scale → Q@K^T → layout → Q/K/V proj
    ///
    /// # Contract (C-ATTN-BACK-001)
    ///
    /// - **Precondition**: grad_input contains gradient from post-attention norm backward,
    ///   scratch.{q, k, v, attn_scores, attn_out, norm1_out} contain forward pass values
    /// - **Postcondition**: grad_hidden contains gradient w.r.t. norm1_out (input to Q/K/V proj),
    ///   grad_w_{q,k,v,o} contain weight gradients for attention projections
    /// - **Invariant**: Zero CPU-side data transfers; all operations on GPU
    fn backward_attention(
        &mut self,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.q_dim();
        let kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let heads_per_kv = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let seq = saturating_u32(seq_len);
        let nh = saturating_u32(num_heads);
        let nkv = saturating_u32(num_kv_heads);
        let hd = saturating_u32(head_dim);

        // === Step 4.1: Output projection backward ===
        // Forward: o_proj_out[seq,hidden] = attn_out[seq,q_dim] @ w_o[q_dim,hidden]
        //   m=seq, k=q_dim, n=hidden
        // grad_attn_out[seq,q_dim] = grad_o_proj[seq,hidden] @ w_o^T[hidden,q_dim]
        gemm_backward_a(
            grad_input,
            &self.w_o,
            &mut self.scratch.grad_hidden,
            seq,
            saturating_u32(q_dim),
            saturating_u32(hidden_size),
            stream,
        )?;

        // grad_w_o[q_dim,hidden] = attn_out^T[q_dim,seq] @ grad_o_proj[seq,hidden]
        gemm_backward_b(
            &self.scratch.attn_out,
            grad_input,
            &mut grad_ws.grad_w_o,
            seq,
            saturating_u32(q_dim),
            saturating_u32(hidden_size),
            stream,
        )?;

        // === Step 4.2: Layout conversion ===
        // grad_attn_out [seq, q_dim] → grad_attn_batched [num_heads, seq, head_dim]
        // Reuse attn_q_batched for grad_attn_batched
        interleaved_to_batched_forward(
            &self.scratch.grad_hidden,
            &mut self.scratch.attn_q_batched,
            seq,
            nh,
            hd,
            stream,
        )?;

        // === Step 4.3: Backward through attn_weights @ V ===
        // Forward was: attn_result = attn_weights @ V_batched
        // Reconstruct V_batched from preserved v
        interleaved_to_batched_forward(
            &self.scratch.v,
            &mut self.scratch.attn_kv_temp,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // GQA expand V if needed
        if heads_per_kv > 1 {
            expand_kv_heads(
                &self.scratch.attn_kv_temp,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
            // SAFETY: Both buffers are valid GPU allocations with matching sizes.
            unsafe {
                self.scratch
                    .attn_kv_temp
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp2, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "Attn backward V expand D2D copy failed: {e}"
                        ))
                    })?;
            }
        }
        // attn_kv_temp now has V_batched [num_heads, seq, head_dim]

        // Transpose V: [num_heads, seq, head_dim] → [num_heads, head_dim, seq]
        batched_transpose_forward(
            &self.scratch.attn_kv_temp,
            &mut self.scratch.attn_kv_temp2,
            nh,
            seq,
            hd,
            stream,
        )?;
        // attn_kv_temp2 = V^T [num_heads, head_dim, seq]

        // grad_attn_weights = grad_attn_batched @ V^T → grad_attn_scores [H, seq, seq]
        batched_4d_gemm_forward(
            &self.scratch.attn_q_batched,
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.grad_attn_scores,
            1,
            nh,
            seq,
            seq,
            hd,
            stream,
        )?;

        // grad_V = attn_weights^T @ grad_attn_batched → attn_kv_temp [H, seq, hd]
        //
        // BUG FIX: Cannot transpose attn_scores [H,S,S] into attn_kv_temp2 [H,S,hd]
        // because H*S*S >> H*S*hd when S > hd (e.g. 350M: 4.2M vs 524K = 8× overflow).
        //
        // Use identity: grad_V = (grad_attn_batched^T @ attn_scores)^T
        // All intermediates are [H, hd, S] = [H, S, hd] size — no H*S*S buffer needed.

        // Step A: transpose grad_attn_batched [H,S,hd] → [H,hd,S]
        batched_transpose_forward(
            &self.scratch.attn_q_batched,   // grad_attn_batched [H, S, hd]
            &mut self.scratch.attn_kv_temp, // temp: grad_attn_batched^T [H, hd, S]
            nh,
            seq,
            hd,
            stream,
        )?;

        // Step B: GEMM [H,hd,S] @ [H,S,S] → [H,hd,S] (= grad_V^T)
        batched_4d_gemm_forward(
            &self.scratch.attn_kv_temp,      // grad_attn_batched^T [H, hd, S]
            &self.scratch.attn_scores,       // attn_weights [H, S, S]
            &mut self.scratch.attn_kv_temp2, // grad_V^T [H, hd, S]
            1,
            nh,
            hd,  // m
            seq, // n
            seq, // k
            stream,
        )?;

        // Step C: transpose grad_V^T [H,hd,S] → grad_V [H,S,hd]
        batched_transpose_forward(
            &self.scratch.attn_kv_temp2,    // grad_V^T [H, hd, S]
            &mut self.scratch.attn_kv_temp, // grad_V [H, S, hd]
            nh,
            hd,
            seq,
            stream,
        )?;
        // attn_kv_temp = grad_V [num_heads, seq, head_dim]

        // === Step 4.4: Softmax backward ===
        // attn_scores contains softmax output from forward pass
        // In-place: grad_attn_scores is both input (grad_output) and output (grad_input)
        // This is safe because the kernel reads all elements in pass 1 before writing in pass 2.
        let total_rows = nh * seq;
        {
            // SAFETY: In-place aliasing is safe for BatchedSoftmaxBackwardKernel which uses
            // a two-pass approach: pass 1 reads all y[i]*gy[i] to compute dot product,
            // pass 2 writes grad_x[i]. The view is forgotten to prevent double-free.
            let grad_scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.grad_attn_scores.as_ptr(),
                    self.scratch.grad_attn_scores.len(),
                )
            };
            batched_softmax_backward(
                &self.scratch.attn_scores,
                &grad_scores_view,
                &mut self.scratch.grad_attn_scores,
                total_rows,
                seq,
                stream,
            )?;
            leak(grad_scores_view);
        }
        // grad_attn_scores now contains gradient through softmax

        // === Step 4.5: Scale backward ===
        // Forward scaled by 1/√d_k, backward is same scale (linear operation)
        let total_scores = nh * seq * seq;
        {
            // SAFETY: In-place aliasing safe for element-wise scale (independent elements).
            let scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    self.scratch.grad_attn_scores.as_ptr(),
                    self.scratch.grad_attn_scores.len(),
                )
            };
            scale_forward(
                &scores_view,
                &mut self.scratch.grad_attn_scores,
                scale,
                total_scores,
                stream,
            )?;
            leak(scores_view);
        }

        // === Step 4.6: Backward through Q @ K^T ===
        // Forward was: scores = Q_batched @ K^T
        // Reconstruct K_batched and expand for GQA
        interleaved_to_batched_forward(
            &self.scratch.k,
            &mut self.scratch.attn_kv_temp2,
            seq,
            nkv,
            hd,
            stream,
        )?;

        if heads_per_kv > 1 {
            // SAFETY: Both buffers are valid GPU allocations; attn_q_batched is about to be
            // overwritten anyway.
            unsafe {
                self.scratch
                    .attn_q_batched
                    .copy_from_buffer_async(&self.scratch.attn_kv_temp2, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "Attn backward K copy for GQA expand failed: {e}"
                        ))
                    })?;
            }
            expand_kv_heads(
                &self.scratch.attn_q_batched,
                &mut self.scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        }
        // attn_kv_temp2 = K_expanded [num_heads, seq, head_dim]

        // grad_Q = grad_scores @ K_expanded → attn_q_batched [H, seq, hd]
        batched_4d_gemm_forward(
            &self.scratch.grad_attn_scores,
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.attn_q_batched,
            1,
            nh,
            seq,
            hd,
            seq,
            stream,
        )?;

        // grad_K^T = Q^T @ grad_scores
        // First reconstruct Q_batched from preserved q
        // Reconstruct Q_batched into o_proj_out (attn_q_batched already overwritten by grad_Q).
        interleaved_to_batched_forward(
            &self.scratch.q,
            &mut self.scratch.o_proj_out, // temp buffer for Q_batched
            seq,
            nh,
            hd,
            stream,
        )?;

        // Transpose Q: [H, seq, hd] → [H, hd, seq]
        batched_transpose_forward(
            &self.scratch.o_proj_out,
            &mut self.scratch.attn_kv_temp2, // reuse for Q^T
            nh,
            seq,
            hd,
            stream,
        )?;

        // grad_K^T = Q^T @ grad_scores → ffn_out as temp [H, hd, seq]
        batched_4d_gemm_forward(
            &self.scratch.attn_kv_temp2,
            &self.scratch.grad_attn_scores,
            &mut self.scratch.ffn_out, // reuse as temp for grad_K^T [H, hd, seq]
            1,
            nh,
            hd,
            seq,
            seq,
            stream,
        )?;

        // Transpose grad_K^T → grad_K: [H, hd, seq] → [H, seq, hd]
        batched_transpose_forward(
            &self.scratch.ffn_out,
            &mut self.scratch.attn_kv_temp2, // grad_K [H, seq, hd]
            nh,
            hd,
            seq,
            stream,
        )?;

        // === Step 4.7: GQA gradient reduction ===
        // grad_K and grad_V are in [num_heads, seq, hd], need to reduce to [num_kv_heads, seq, hd]
        if heads_per_kv > 1 {
            self.reduce_gqa_gradients(num_kv_heads, heads_per_kv, seq_len, head_dim, stream)?;
        }

        // === Step 4.8: Convert gradients back to interleaved layout ===
        // grad_Q: attn_q_batched [H, seq, hd] → o_proj_out [seq, hidden] (interleaved)
        batched_to_interleaved_forward(
            &self.scratch.attn_q_batched,
            &mut self.scratch.o_proj_out,
            seq,
            nh,
            hd,
            stream,
        )?;

        // grad_K: attn_kv_temp2 [nkv, seq, hd] → norm2_out [seq, kv_hidden] (interleaved)
        batched_to_interleaved_forward(
            &self.scratch.attn_kv_temp2,
            &mut self.scratch.norm2_out,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // grad_V: attn_kv_temp [nkv, seq, hd] → ffn_out [seq, kv_hidden] (interleaved)
        batched_to_interleaved_forward(
            &self.scratch.attn_kv_temp,
            &mut self.scratch.ffn_out,
            seq,
            nkv,
            hd,
            stream,
        )?;

        // === Step 4.8b: RoPE backward (inverse rotation) ===
        // Forward applied RoPE to Q and K before attention. Backward must undo
        // the rotation so projection backward (step 4.9) gets unrotated gradients.
        // R^T(-θ): new_x0 = x0*cos + x1*sin, new_x1 = x1*cos - x0*sin
        let rope_theta = self.config.rope_theta;
        {
            // grad_Q in o_proj_out [seq, q_dim] — apply inverse rotation in-place
            let q_ref = unsafe { &*(std::ptr::addr_of!(self.scratch.o_proj_out)) };
            batched_rope_neox_backward(
                q_ref,
                &mut self.scratch.o_proj_out,
                &self.scratch.rope_positions,
                nh,
                hd,
                seq,
                rope_theta,
                stream,
            )?;
            // grad_K in norm2_out [seq, kv_hidden] — apply inverse rotation in-place
            let k_ref = unsafe { &*(std::ptr::addr_of!(self.scratch.norm2_out)) };
            batched_rope_neox_backward(
                k_ref,
                &mut self.scratch.norm2_out,
                &self.scratch.rope_positions,
                nkv,
                hd,
                seq,
                rope_theta,
                stream,
            )?;
        }

        // === Step 4.9: Q/K/V projection backward ===
        // Forward: q[seq,q_dim] = norm1[seq,hidden] @ w_q[hidden,q_dim]
        //   m=seq, k=hidden, n=q_dim
        // grad_norm1[seq,hidden] = grad_q[seq,q_dim] @ w_q^T[q_dim,hidden]
        gemm_backward_a(
            &self.scratch.o_proj_out, // grad_q interleaved [seq, q_dim]
            &self.w_q,
            &mut self.scratch.grad_hidden,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(q_dim),
            stream,
        )?;

        // grad_norm1 += grad_k @ w_k^T
        // Forward: k[seq,kv_hidden] = norm1[seq,hidden] @ w_k[hidden,kv_hidden]
        //   m=seq, k=hidden, n=kv_hidden
        // KAIZEN-057: cuda_add_inplace replaces residual_add_forward + D2D copy
        gemm_backward_a(
            &self.scratch.norm2_out, // grad_k interleaved
            &self.w_k,
            &mut self.scratch.grad_attn_scores, // temp for grad_k @ w_k^T
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;
        cuda_add_inplace(
            &mut self.scratch.grad_hidden,
            &self.scratch.grad_attn_scores,
            seq_len * hidden_size,
            stream,
        )?;

        // grad_norm1 += grad_v @ w_v^T
        // Forward: v[seq,kv_hidden] = norm1[seq,hidden] @ w_v[hidden,kv_hidden]
        //   m=seq, k=hidden, n=kv_hidden
        gemm_backward_a(
            &self.scratch.ffn_out, // grad_v interleaved
            &self.w_v,
            &mut self.scratch.grad_attn_scores, // temp for grad_v @ w_v^T
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;
        cuda_add_inplace(
            &mut self.scratch.grad_hidden,
            &self.scratch.grad_attn_scores,
            seq_len * hidden_size,
            stream,
        )?;

        // Weight gradients: grad_w_q[hidden,q_dim] = norm1_out^T[hidden,seq] @ grad_q[seq,q_dim]
        gemm_backward_b(
            &self.scratch.norm1_out,
            &self.scratch.o_proj_out, // grad_q [seq, q_dim]
            &mut grad_ws.grad_w_q,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(q_dim),
            stream,
        )?;

        // grad_w_k = norm1_out^T @ grad_k
        gemm_backward_b(
            &self.scratch.norm1_out,
            &self.scratch.norm2_out, // grad_k
            &mut grad_ws.grad_w_k,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        // grad_w_v = norm1_out^T @ grad_v
        gemm_backward_b(
            &self.scratch.norm1_out,
            &self.scratch.ffn_out, // grad_v
            &mut grad_ws.grad_w_v,
            seq,
            saturating_u32(hidden_size),
            saturating_u32(kv_hidden_size),
            stream,
        )?;

        // Copy grad_hidden → grad_input for downstream (residual backward)
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            grad_input.copy_from_buffer_async(&self.scratch.grad_hidden, stream).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "Attn backward grad_hidden → grad_input D2D copy failed: {e}"
                ))
            })?;
        }

        Ok(())
    }

    /// Reduce GQA head gradients from [num_heads] to [num_kv_heads] by summing groups.
    ///
    /// Reads grad_K from `attn_kv_temp2` and grad_V from `attn_kv_temp` (both [H, seq, hd]).
    /// Writes reduced grad_K to `attn_kv_temp2` and reduced grad_V to `attn_kv_temp`
    /// (both [nkv, seq, hd]).
    ///
    /// Uses `grad_attn_scores`, `ffn_out`, `o_proj_out`, `grad_hidden` as scratch.
    fn reduce_gqa_gradients(
        &mut self,
        num_kv_heads: usize,
        heads_per_kv: usize,
        seq_len: usize,
        head_dim: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let elems_per_head = seq_len * head_dim;

        // Reduce grad_K: attn_kv_temp2 [H] → grad_attn_scores [nkv]
        self.reduce_single_gqa_gradient(true, num_kv_heads, heads_per_kv, elems_per_head, stream)?;

        // Reduce grad_V: attn_kv_temp [H] → ffn_out [nkv]
        self.reduce_single_gqa_gradient(false, num_kv_heads, heads_per_kv, elems_per_head, stream)?;

        // Copy reduced results to known locations for step 4.8
        let kv_elems = num_kv_heads * elems_per_head;
        // SAFETY: Valid GPU allocations with sufficient size.
        unsafe {
            self.scratch
                .attn_kv_temp2
                .copy_from_buffer_at_async(&self.scratch.grad_attn_scores, 0, 0, kv_elems, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "GQA grad_K reduced final copy failed: {e}"
                    ))
                })?;
            self.scratch
                .attn_kv_temp
                .copy_from_buffer_at_async(&self.scratch.ffn_out, 0, 0, kv_elems, stream)
                .map_err(|e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "GQA grad_V reduced final copy failed: {e}"
                    ))
                })?;
        }
        Ok(())
    }

    /// Reduce one gradient tensor from [num_heads] to [num_kv_heads] by summing groups.
    ///
    /// When `is_k=true`: reads from `attn_kv_temp2`, writes to `grad_attn_scores`.
    /// When `is_k=false`: reads from `attn_kv_temp`, writes to `ffn_out`.
    /// Uses `o_proj_out` and `grad_hidden` as scratch.
    fn reduce_single_gqa_gradient(
        &mut self,
        is_k: bool,
        num_kv_heads: usize,
        heads_per_kv: usize,
        elems_per_head: usize,
        stream: &CudaStream,
    ) -> Result<()> {
        let label = if is_k { "K" } else { "V" };

        for kv_h in 0..num_kv_heads {
            let dst_offset = kv_h * elems_per_head;
            let first_h = kv_h * heads_per_kv;
            let src_offset = first_h * elems_per_head;

            // Copy first head of group as base
            // SAFETY: All offsets are within buffer bounds.
            unsafe {
                let (dst, src) = if is_k {
                    (&mut self.scratch.grad_attn_scores, &self.scratch.attn_kv_temp2)
                } else {
                    (&mut self.scratch.ffn_out, &self.scratch.attn_kv_temp)
                };
                dst.copy_from_buffer_at_async(src, dst_offset, src_offset, elems_per_head, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "GQA grad_{label} reduce base copy failed: {e}"
                        ))
                    })?;
            }

            // Add remaining heads in group
            for rep in 1..heads_per_kv {
                let h = kv_h * heads_per_kv + rep;
                let h_offset = h * elems_per_head;

                // Head extraction into o_proj_out buffer
                // SAFETY: Valid GPU allocations with sufficient size.
                unsafe {
                    let src =
                        if is_k { &self.scratch.attn_kv_temp2 } else { &self.scratch.attn_kv_temp };
                    self.scratch
                        .o_proj_out
                        .copy_from_buffer_at_async(src, 0, h_offset, elems_per_head, stream)
                        .map_err(|e| {
                            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                                "GQA grad_{label} reduce head copy failed: {e}"
                            ))
                        })?;
                }

                // Add: dst[dst_offset..] += o_proj_out[0..elems_per_head]
                // SAFETY: Creating non-owning views for arithmetic; forgotten to prevent double-free.
                unsafe {
                    let dst_buf =
                        if is_k { &self.scratch.grad_attn_scores } else { &self.scratch.ffn_out };
                    let dst_view = GpuBuffer::<f32>::from_raw_parts(
                        dst_buf.as_ptr() + (dst_offset as u64 * 4),
                        elems_per_head,
                    );
                    let src_view = GpuBuffer::<f32>::from_raw_parts(
                        self.scratch.o_proj_out.as_ptr(),
                        elems_per_head,
                    );
                    let mut sum_view = GpuBuffer::<f32>::from_raw_parts(
                        self.scratch.grad_hidden.as_ptr(),
                        elems_per_head,
                    );
                    residual_add_forward(
                        &dst_view,
                        &src_view,
                        &mut sum_view,
                        saturating_u32(elems_per_head),
                        stream,
                    )?;
                    // Copy sum back to dst at dst_offset
                    let dst_buf = if is_k {
                        &mut self.scratch.grad_attn_scores
                    } else {
                        &mut self.scratch.ffn_out
                    };
                    dst_buf
                        .copy_from_buffer_at_async(
                            &self.scratch.grad_hidden,
                            dst_offset,
                            0,
                            elems_per_head,
                            stream,
                        )
                        .map_err(|e| {
                            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                                "GQA grad_{label} reduce sum copy failed: {e}"
                            ))
                        })?;
                    leak(dst_view);
                    leak(src_view);
                    leak(sum_view);
                }
            }
        }
        Ok(())
    }

    /// Backward through first residual connection and input RMSNorm.
    fn backward_residual_and_input_norm(
        &mut self,
        input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        hidden_size: usize,
        eps: f32,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        // Forward was: norm_out = RMSNorm(input); attn_out = Attn(norm_out); residual1 = input + attn_out
        // Backward: grad_input = RMSNorm_backward(grad_through_attention) + grad_output
        //
        // C-RESIDUAL-001 / entrenar#313: The residual skip (grad_output) must be added
        // AFTER RMSNorm backward, not before. RMSNorm backward should only transform
        // the gradient that flows through the norm/attention path. The identity skip
        // bypasses the norm entirely.

        // D2D copy grad_input (attention path gradient) to grad_hidden
        // SAFETY: Both buffers are valid GPU allocations with matching sizes.
        unsafe {
            self.scratch.grad_hidden.copy_from_buffer_async(grad_input, stream).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "Backward residual grad_hidden D2D copy failed: {e}"
                ))
            })?;
        }

        // RMSNorm backward: only applied to the attention path gradient
        rms_norm_backward(
            input,
            &self.input_norm_weight,
            &self.scratch.grad_hidden,
            grad_input,
            &mut grad_ws.grad_input_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )?;

        // NOW add the residual skip: grad_input += grad_output (identity connection)
        cuda_add_inplace(grad_input, grad_output, seq_len * hidden_size, stream)
    }

    /// Initialize GPU-resident AdamW optimizer state for all block weights.
    ///
    /// Allocates zero-initialized first and second moment buffers for each of the
    /// 9 weight tensors (4 attention projections + 3 FFN projections + 2 RMSNorm).
    ///
    /// # Contract (C-OPTINIT-001)
    ///
    /// - **Precondition**: CUDA context is valid, sufficient GPU memory available
    /// - **Postcondition**: All m/v buffers are zero-initialized with dimensions
    ///   matching the corresponding weight tensors
    /// - **Invariant**: Total GPU memory for optimizer state = 2 × sum(weight_sizes) × 4 bytes
    pub fn init_optimizer_state(&self) -> Result<GpuBlockOptimizerState> {
        let hidden = self.config.hidden_size;
        let q_dim = self.config.q_dim();
        let kv_hidden = self.config.num_kv_heads * self.config.head_dim();
        let intermediate = self.config.intermediate_size;

        // CRITICAL: Must zero-initialize m/v buffers. GpuBuffer::new() does NOT
        // zero memory (cuMemAlloc returns uninitialized VRAM). Uninitialized m/v
        // causes v_new = beta2 * GARBAGE which can be negative → sqrt(neg) → NaN.
        let z = |n: usize| -> Result<GpuBuffer<f32>> {
            Ok(GpuBuffer::from_host(&self.ctx, &vec![0.0f32; n])?)
        };
        Ok(GpuBlockOptimizerState {
            m_w_q: z(q_dim * hidden)?,
            v_w_q: z(q_dim * hidden)?,
            m_w_k: z(hidden * kv_hidden)?,
            v_w_k: z(hidden * kv_hidden)?,
            m_w_v: z(hidden * kv_hidden)?,
            v_w_v: z(hidden * kv_hidden)?,
            m_w_o: z(hidden * q_dim)?,
            v_w_o: z(hidden * q_dim)?,
            m_w_gate: z(hidden * intermediate)?,
            v_w_gate: z(hidden * intermediate)?,
            m_w_up: z(hidden * intermediate)?,
            v_w_up: z(hidden * intermediate)?,
            m_w_down: z(intermediate * hidden)?,
            v_w_down: z(intermediate * hidden)?,
            m_input_norm: z(hidden)?,
            v_input_norm: z(hidden)?,
            m_post_attn_norm: z(hidden)?,
            v_post_attn_norm: z(hidden)?,
        })
    }

    /// Run GPU-resident AdamW optimizer step on all block weights.
    ///
    /// Updates weights in-place using gradients computed by `backward()`.
    /// All operations run on GPU — zero CPU↔GPU data transfers.
    ///
    /// # Contract (C-OPTSTEP-001)
    ///
    /// - **Precondition**: `backward()` completed for this block (scratch grad buffers valid),
    ///   `state` initialized via `init_optimizer_state()`, `step > 0`
    /// - **Postcondition**: All 9 weight tensors updated by AdamW rule,
    ///   m/v states updated with current gradient statistics
    /// - **Invariant**: Weight dimensions unchanged; no GPU memory allocated or freed
    pub fn optimizer_step(
        &mut self,
        state: &mut GpuBlockOptimizerState,
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: &CudaStream,
        grad_ws: &CudaGradWorkspace,
    ) -> Result<()> {
        debug_assert!(step > 0, "C-OPTSTEP-001: step must be > 0 for bias adjust");

        // Pre-capture lengths to avoid borrow conflicts (len is immutable borrow,
        // adamw_step_cuda takes mutable borrow on same buffer)
        let n_wq = self.w_q.len() as u32;
        let n_wk = self.w_k.len() as u32;
        let n_wv = self.w_v.len() as u32;
        let n_wo = self.w_o.len() as u32;
        let n_gate = self.w_gate.len() as u32;
        let n_up = self.w_up.len() as u32;
        let n_down = self.w_down.len() as u32;
        let n_inorm = self.input_norm_weight.len() as u32;
        let n_panorm = self.post_attn_norm_weight.len() as u32;

        // Attention projection weights
        adamw_step_cuda(
            &mut self.w_q,
            &grad_ws.grad_w_q,
            &mut state.m_w_q,
            &mut state.v_w_q,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_wq,
            stream,
        )?;
        adamw_step_cuda(
            &mut self.w_k,
            &grad_ws.grad_w_k,
            &mut state.m_w_k,
            &mut state.v_w_k,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_wk,
            stream,
        )?;
        adamw_step_cuda(
            &mut self.w_v,
            &grad_ws.grad_w_v,
            &mut state.m_w_v,
            &mut state.v_w_v,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_wv,
            stream,
        )?;
        adamw_step_cuda(
            &mut self.w_o,
            &grad_ws.grad_w_o,
            &mut state.m_w_o,
            &mut state.v_w_o,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_wo,
            stream,
        )?;

        // FFN projection weights
        adamw_step_cuda(
            &mut self.w_gate,
            &grad_ws.grad_gate,
            &mut state.m_w_gate,
            &mut state.v_w_gate,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_gate,
            stream,
        )?;
        adamw_step_cuda(
            &mut self.w_up,
            &grad_ws.grad_up,
            &mut state.m_w_up,
            &mut state.v_w_up,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_up,
            stream,
        )?;
        adamw_step_cuda(
            &mut self.w_down,
            &grad_ws.grad_down,
            &mut state.m_w_down,
            &mut state.v_w_down,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_down,
            stream,
        )?;

        // RMSNorm weights
        adamw_step_cuda(
            &mut self.input_norm_weight,
            &grad_ws.grad_input_norm,
            &mut state.m_input_norm,
            &mut state.v_input_norm,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_inorm,
            stream,
        )?;
        adamw_step_cuda(
            &mut self.post_attn_norm_weight,
            &grad_ws.grad_post_attn_norm,
            &mut state.m_post_attn_norm,
            &mut state.v_post_attn_norm,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            n_panorm,
            stream,
        )?;

        Ok(())
    }

    /// Download all weight data from GPU to host vectors.
    ///
    /// Used to synchronize GPU-updated weights back to CPU model for checkpointing.
    ///
    /// # Contract (C-DLWEIGHTS-001)
    ///
    /// - **Precondition**: Block weights are valid GPU allocations
    /// - **Postcondition**: Returned vectors have exact same length and content as GPU buffers
    /// - **Invariant**: GPU buffers are not modified
    pub fn download_weights(&self) -> Result<BlockWeights> {
        let download = |buf: &GpuBuffer<f32>| -> Result<Vec<f32>> {
            let mut host = vec![0.0f32; buf.len()];
            buf.copy_to_host(&mut host).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "Weight download failed: {e}"
                ))
            })?;
            Ok(host)
        };

        Ok(BlockWeights {
            w_q: download(&self.w_q)?,
            w_k: download(&self.w_k)?,
            w_v: download(&self.w_v)?,
            w_o: download(&self.w_o)?,
            w_gate: download(&self.w_gate)?,
            w_up: download(&self.w_up)?,
            w_down: download(&self.w_down)?,
            input_norm_weight: download(&self.input_norm_weight)?,
            post_attn_norm_weight: download(&self.post_attn_norm_weight)?,
        })
    }
}

/// Downloaded weight data from a CUDA transformer block.
///
/// # Contract (C-BLOCKWT-001)
///
/// - **Invariant**: Vector lengths match original weight dimensions
#[cfg(feature = "cuda")]
pub struct BlockWeights {
    pub w_q: Vec<f32>,
    pub w_k: Vec<f32>,
    pub w_v: Vec<f32>,
    pub w_o: Vec<f32>,
    pub w_gate: Vec<f32>,
    pub w_up: Vec<f32>,
    pub w_down: Vec<f32>,
    pub input_norm_weight: Vec<f32>,
    pub post_attn_norm_weight: Vec<f32>,
}

/// CUDA element-wise addition on GPU (zero CPU transfers)
///
/// Uses `ResidualAddKernel` — single kernel launch, no D2H/H2D transfers.
#[cfg(feature = "cuda")]
fn cuda_add(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: usize,
    stream: &CudaStream,
) -> Result<()> {
    residual_add_forward(a, b, output, saturating_u32(n), stream)
}

/// In-place add: `target += source` using residual add with aliased output.
///
/// # Safety
///
/// The ResidualAdd kernel reads `a[i]` and `b[i]` then writes `output[i] = a[i] + b[i]`.
/// When `a` and `output` alias the same GPU buffer, each element is read before written
/// (no inter-element dependency), so this is safe for elementwise operations.
#[cfg(feature = "cuda")]
pub(crate) fn cuda_add_inplace(
    target: &mut GpuBuffer<f32>,
    source: &GpuBuffer<f32>,
    n: usize,
    stream: &CudaStream,
) -> Result<()> {
    // SAFETY: ResidualAdd kernel is elementwise (output[i] = a[i] + b[i]).
    // Aliasing target as both input and output is safe because each element is
    // independent — the GPU reads a[i] before writing output[i] at the same address.
    let target_ref: &GpuBuffer<f32> = unsafe { &*std::ptr::from_ref::<GpuBuffer<f32>>(target) };
    residual_add_forward(target_ref, source, target, saturating_u32(n), stream)
}

/// CUDA element-wise multiplication on GPU (zero CPU transfers)
///
/// Uses `ElementwiseMulKernel` — single kernel launch, no D2H/H2D transfers.
#[cfg(feature = "cuda")]
fn cuda_mul(
    a: &GpuBuffer<f32>,
    b: &GpuBuffer<f32>,
    output: &mut GpuBuffer<f32>,
    n: usize,
    stream: &CudaStream,
) -> Result<()> {
    crate::autograd::cuda_forward::elementwise_mul_forward(a, b, output, saturating_u32(n), stream)
}

// CPU fallback stub
#[cfg(not(feature = "cuda"))]
pub struct CudaTransformerBlock;

#[cfg(not(feature = "cuda"))]
impl CudaTransformerBlock {
    pub fn layer_idx(&self) -> usize {
        0
    }
}

// =============================================================================
// CudaBlock — enum dispatching fp32 or NF4 transformer blocks
// =============================================================================

/// Unified enum for CUDA transformer blocks (fp32 or NF4-quantized).
///
/// The classify pipeline stores `Vec<CudaBlock>` and calls `forward()` without
/// caring which quantization format the frozen weights use.
#[cfg(feature = "cuda")]
pub enum CudaBlock {
    /// Standard fp32 weights (full precision, ~16 GB for Qwen3-4B)
    Fp32(CudaTransformerBlock),
    /// NF4 quantized weights (~2 GB for Qwen3-4B, ~8x compression)
    Nf4(CudaNf4TransformerBlock),
}

#[cfg(feature = "cuda")]
impl CudaBlock {
    /// Forward pass through the transformer block.
    ///
    /// For NF4 blocks, `shared_scratch` must be `Some` — shared across all layers (C-SCRATCH-001).
    /// For fp32 blocks, `shared_scratch` is ignored (each block owns its scratch for backward).
    pub(crate) fn forward(
        &mut self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        shared_scratch: Option<&mut CudaBlockScratch>,
    ) -> Result<()> {
        match self {
            CudaBlock::Fp32(b) => b.forward(input, output, seq_len, stream),
            CudaBlock::Nf4(b) => {
                let scratch =
                    shared_scratch.expect("C-SCRATCH-001: NF4 blocks require shared scratch");
                b.forward(input, output, seq_len, stream, scratch)
            }
        }
    }

    /// Get the layer index of this block.
    pub fn layer_idx(&self) -> usize {
        match self {
            CudaBlock::Fp32(b) => b.layer_idx(),
            CudaBlock::Nf4(b) => b.layer_idx,
        }
    }

    /// Backward pass (only supported for fp32 blocks).
    ///
    /// NF4 blocks are frozen — backward is never called when `quantize_nf4` is active
    /// because `gpu_training` is set to `None`.
    pub fn backward(
        &mut self,
        input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        grad_ws: &mut CudaGradWorkspace,
    ) -> Result<()> {
        match self {
            CudaBlock::Fp32(b) => {
                b.backward(input, grad_output, grad_input, seq_len, stream, grad_ws)
            }
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "backward not supported on NF4 blocks (frozen weights)".into(),
            )),
        }
    }

    /// Initialize optimizer state (only supported for fp32 blocks).
    pub fn init_optimizer_state(&self) -> Result<GpuBlockOptimizerState> {
        match self {
            CudaBlock::Fp32(b) => b.init_optimizer_state(),
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "init_optimizer_state not supported on NF4 blocks".into(),
            )),
        }
    }

    /// Download weights from GPU (only supported for fp32 blocks).
    pub fn download_weights(&self) -> Result<BlockWeights> {
        match self {
            CudaBlock::Fp32(b) => b.download_weights(),
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "download_weights not supported on NF4 blocks".into(),
            )),
        }
    }

    /// Optimizer step (only supported for fp32 blocks).
    pub fn optimizer_step(
        &mut self,
        state: &mut GpuBlockOptimizerState,
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: &CudaStream,
        grad_ws: &CudaGradWorkspace,
    ) -> Result<()> {
        match self {
            CudaBlock::Fp32(b) => {
                b.optimizer_step(state, step, lr, beta1, beta2, eps, weight_decay, stream, grad_ws)
            }
            CudaBlock::Nf4(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "optimizer_step not supported on NF4 blocks (frozen weights)".into(),
            )),
        }
    }

    /// NF4 backward pass with LoRA gradient computation (ENT-153).
    ///
    /// Only callable on NF4 blocks. Returns error for fp32 blocks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward_nf4(
        &self,
        layer_input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        output_scratch: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        shared_scratch: &mut CudaBlockScratch,
        grad_lora: &mut CudaLoraGradWorkspace,
    ) -> Result<()> {
        match self {
            CudaBlock::Nf4(b) => b.backward(
                layer_input,
                grad_output,
                grad_input,
                output_scratch,
                seq_len,
                stream,
                shared_scratch,
                grad_lora,
            ),
            CudaBlock::Fp32(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "backward_nf4 only supported on NF4 blocks".into(),
            )),
        }
    }

    /// Initialize LoRA optimizer state for NF4 blocks.
    pub(crate) fn init_lora_optimizer_state(&self) -> Result<GpuLoraOptimizerState> {
        match self {
            CudaBlock::Nf4(b) => b.init_lora_optimizer_state(),
            CudaBlock::Fp32(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "init_lora_optimizer_state only supported on NF4 blocks".into(),
            )),
        }
    }

    /// LoRA optimizer step for NF4 blocks.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lora_optimizer_step(
        &mut self,
        state: &mut GpuLoraOptimizerState,
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: &CudaStream,
        grad_lora: &CudaLoraGradWorkspace,
    ) -> Result<()> {
        match self {
            CudaBlock::Nf4(b) => b.lora_optimizer_step(
                state,
                step,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                stream,
                grad_lora,
            ),
            CudaBlock::Fp32(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "lora_optimizer_step only supported on NF4 blocks".into(),
            )),
        }
    }

    /// Download LoRA weights from NF4 blocks.
    pub fn download_lora_weights(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        match self {
            CudaBlock::Nf4(b) => b.download_lora_weights(),
            CudaBlock::Fp32(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "download_lora_weights only supported on NF4 blocks".into(),
            )),
        }
    }

    /// Upload LoRA weights to NF4 blocks for checkpoint resume (ENT-276).
    pub fn upload_lora_weights(
        &mut self,
        a_q: &[f32],
        b_q: &[f32],
        a_v: &[f32],
        b_v: &[f32],
    ) -> Result<()> {
        match self {
            CudaBlock::Nf4(b) => b.upload_lora_weights(a_q, b_q, a_v, b_v),
            CudaBlock::Fp32(_) => Err(crate::autograd::cuda_tensor::CudaTensorError::KernelError(
                "upload_lora_weights only supported on NF4 blocks".into(),
            )),
        }
    }
}

/// CPU fallback stub for CudaBlock.
#[cfg(not(feature = "cuda"))]
pub enum CudaBlock {
    Fp32(CudaTransformerBlock),
}

// =============================================================================
// NF4 Quantized Transformer Block (trueno#108: QLoRA support)
// =============================================================================

/// CUDA-accelerated transformer block with NF4-quantized frozen weights.
///
/// Stores the 7 projection weights as packed NF4 (4-bit) + per-block scales instead
/// of fp32, achieving ~8x compression. Norm weights remain fp32 (negligible size).
///
/// # VRAM Savings (Qwen3-4B example)
///
/// | Component | fp32 | NF4 |
/// |-----------|------|-----|
/// | Frozen weights (36L × 7 projections) | 16.0 GB | 2.1 GB |
///
/// # Forward Only
///
/// NF4 blocks are frozen — no backward pass needed. LoRA adapters (fp32) handle
/// the trainable parameters separately. The forward pass uses fused dequant+GEMM
/// kernels that read NF4 directly without materializing fp32 weights.
#[cfg(feature = "cuda")]
pub struct CudaNf4TransformerBlock {
    config: TransformerConfig,
    layer_idx: usize,
    // Norm weights stay fp32 (tiny: 2 × hidden_size floats)
    input_norm_weight: GpuBuffer<f32>,
    post_attn_norm_weight: GpuBuffer<f32>,
    // Projection weights: NF4 quantized (packed data + per-block scales)
    w_q_nf4: GpuBuffer<u8>,
    w_q_scales: GpuBuffer<f32>,
    w_k_nf4: GpuBuffer<u8>,
    w_k_scales: GpuBuffer<f32>,
    w_v_nf4: GpuBuffer<u8>,
    w_v_scales: GpuBuffer<f32>,
    w_o_nf4: GpuBuffer<u8>,
    w_o_scales: GpuBuffer<f32>,
    w_gate_nf4: GpuBuffer<u8>,
    w_gate_scales: GpuBuffer<f32>,
    w_up_nf4: GpuBuffer<u8>,
    w_up_scales: GpuBuffer<f32>,
    w_down_nf4: GpuBuffer<u8>,
    w_down_scales: GpuBuffer<f32>,
    // ENT-287: Pre-dequantized fp32 weights for cuBLAS GEMM (correct weight layout)
    w_q_fp32: GpuBuffer<f32>,
    w_k_fp32: GpuBuffer<f32>,
    w_v_fp32: GpuBuffer<f32>,
    w_o_fp32: GpuBuffer<f32>,
    w_gate_fp32: GpuBuffer<f32>,
    w_up_fp32: GpuBuffer<f32>,
    w_down_fp32: GpuBuffer<f32>,
    // LoRA adapters for Q and V projections (ENT-153: QLoRA backward)
    // None when LoRA is not active (inference-only or non-QLoRA training)
    lora_a_q: Option<GpuBuffer<f32>>, // [hidden_size, rank]
    lora_b_q: Option<GpuBuffer<f32>>, // [rank, q_dim]
    lora_a_v: Option<GpuBuffer<f32>>, // [hidden_size, rank]
    lora_b_v: Option<GpuBuffer<f32>>, // [rank, kv_hidden]
    lora_scale: f32,
    lora_rank: usize,
    // QK-norm weights (ENT-270: per-head RMSNorm on Q and K, shape=[head_dim])
    q_norm_weight: Option<GpuBuffer<f32>>,
    k_norm_weight: Option<GpuBuffer<f32>>,
    // FP16 weight buffers for Tier 2 parity (PMAT-470): halve memory BW
    // When set, forward uses gemm_f16_to_f32 (fp16 weights × fp16 activations → fp32 output)
    w_q_fp16: Option<GpuBuffer<u16>>,
    w_k_fp16: Option<GpuBuffer<u16>>,
    w_v_fp16: Option<GpuBuffer<u16>>,
    w_o_fp16: Option<GpuBuffer<u16>>,
    w_gate_fp16: Option<GpuBuffer<u16>>,
    w_up_fp16: Option<GpuBuffer<u16>>,
    w_down_fp16: Option<GpuBuffer<u16>>,
    ctx: Arc<CudaContext>,
    // NF4 blocks do NOT own scratch — shared across all layers (C-SCRATCH-001)
}

#[cfg(feature = "cuda")]
impl CudaNf4TransformerBlock {
    /// Create a new NF4 transformer block from fp32 CPU tensors.
    ///
    /// Quantizes all 7 projection weights to NF4 on CPU, then uploads the packed
    /// data and scales to GPU. Norm weights are uploaded as fp32.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: &TransformerConfig,
        layer_idx: usize,
        ctx: Arc<CudaContext>,
        input_norm_weight: &[f32],
        post_attn_norm_weight: &[f32],
        w_q: &[f32],
        w_k: &[f32],
        w_v: &[f32],
        w_o: &[f32],
        w_gate: &[f32],
        w_up: &[f32],
        w_down: &[f32],
        _max_seq_len: usize, // NF4 blocks use shared scratch (C-SCRATCH-001)
        // ENT-153: Optional LoRA adapters for Q and V projections
        q_lora: Option<(&[f32], &[f32])>,
        v_lora: Option<(&[f32], &[f32])>,
        lora_scale: f32,
        lora_rank: usize,
        // ENT-270: Optional QK-norm weights (per-head RMSNorm, shape=[head_dim])
        q_norm: Option<&[f32]>,
        k_norm: Option<&[f32]>,
    ) -> Result<Self> {
        use trueno_gpu::kernels::{quantize_nf4, NF4_BLOCK_SIZE};

        let hidden_size = config.hidden_size;
        let q_dim = config.q_dim(); // num_heads * head_dim (may differ from hidden_size)
        let kv_hidden_size = config.num_kv_heads * config.head_dim();
        let intermediate_size = config.intermediate_size;

        // ── C-NF4SHAPE-001: Weight shape contracts ──────────────────────
        // Ground truth: PMAT-331 validation in attention.rs from_pretrained()
        //   Q: [q_dim, hidden], K: [kv_hidden, hidden], V: [kv_hidden, hidden], O: [hidden, q_dim]
        //   gate: [intermediate, hidden], up: [intermediate, hidden], down: [hidden, intermediate]
        assert_eq!(
            w_q.len(),
            q_dim * hidden_size,
            "C-NF4SHAPE-001: w_q expected {}, got {} (q_dim={q_dim}, hidden={hidden_size})",
            q_dim * hidden_size,
            w_q.len()
        );
        assert_eq!(
            w_k.len(),
            kv_hidden_size * hidden_size,
            "C-NF4SHAPE-001: w_k expected {}, got {}",
            kv_hidden_size * hidden_size,
            w_k.len()
        );
        assert_eq!(
            w_v.len(),
            kv_hidden_size * hidden_size,
            "C-NF4SHAPE-001: w_v expected {}, got {}",
            kv_hidden_size * hidden_size,
            w_v.len()
        );
        assert_eq!(
            w_o.len(),
            hidden_size * q_dim,
            "C-NF4SHAPE-001: w_o expected {}, got {}",
            hidden_size * q_dim,
            w_o.len()
        );
        assert_eq!(
            w_gate.len(),
            intermediate_size * hidden_size,
            "C-NF4SHAPE-001: w_gate expected {}, got {}",
            intermediate_size * hidden_size,
            w_gate.len()
        );
        assert_eq!(
            w_up.len(),
            intermediate_size * hidden_size,
            "C-NF4SHAPE-001: w_up expected {}, got {}",
            intermediate_size * hidden_size,
            w_up.len()
        );
        assert_eq!(
            w_down.len(),
            hidden_size * intermediate_size,
            "C-NF4SHAPE-001: w_down expected {}, got {}",
            hidden_size * intermediate_size,
            w_down.len()
        );

        // Upload norm weights as fp32
        let input_norm_weight = GpuBuffer::from_host(&ctx, input_norm_weight)?;
        let post_attn_norm_weight = GpuBuffer::from_host(&ctx, post_attn_norm_weight)?;

        // Helper: quantize fp32 weight to NF4, upload packed data + scales to GPU
        // Returns (gpu_nf4, gpu_scales, cpu_quantized) — the CPU struct is retained
        // for dequantization into the cuBLAS fp32 buffer.
        let quantize_and_upload = |weights: &[f32],
                                   total: usize|
         -> Result<(
            GpuBuffer<u8>,
            GpuBuffer<f32>,
            trueno_gpu::kernels::Nf4Quantized,
        )> {
            assert_eq!(weights.len(), total, "weight length mismatch");
            assert!(
                total.is_multiple_of(NF4_BLOCK_SIZE),
                "weight count {total} not divisible by NF4 block size {NF4_BLOCK_SIZE}"
            );

            let q = quantize_nf4(weights, total / NF4_BLOCK_SIZE, NF4_BLOCK_SIZE);
            let nf4_buf = GpuBuffer::from_host(&ctx, &q.data)?;
            let scales_buf = GpuBuffer::from_host(&ctx, &q.scales)?;
            Ok((nf4_buf, scales_buf, q))
        };

        // Quantize all 7 projection weights (shape contracts already verified above)
        let (w_q_nf4, w_q_scales, w_q_nf4_q) = quantize_and_upload(w_q, q_dim * hidden_size)?;
        let (w_k_nf4, w_k_scales, w_k_nf4_q) =
            quantize_and_upload(w_k, kv_hidden_size * hidden_size)?;
        let (w_v_nf4, w_v_scales, w_v_nf4_q) =
            quantize_and_upload(w_v, kv_hidden_size * hidden_size)?;
        let (w_o_nf4, w_o_scales, w_o_nf4_q) = quantize_and_upload(w_o, hidden_size * q_dim)?;
        let (w_gate_nf4, w_gate_scales, w_gate_nf4_q) =
            quantize_and_upload(w_gate, intermediate_size * hidden_size)?;
        let (w_up_nf4, w_up_scales, w_up_nf4_q) =
            quantize_and_upload(w_up, intermediate_size * hidden_size)?;
        let (w_down_nf4, w_down_scales, w_down_nf4_q) =
            quantize_and_upload(w_down, hidden_size * intermediate_size)?;

        // ENT-287: Dequantize NF4 weights and TRANSPOSE to [K,N] for standard cuBLAS GEMM.
        //
        // bitsandbytes does: F.dequantize_4bit(B).to(dtype).t()
        // The .t() transposes [N,K] → [K,N] so torch.nn.functional.linear
        // can use standard C = A @ B^T internally.
        //
        // We replicate this exactly:
        // 1. dequantize_nf4(q) → flat [N*K] in [N,K] row-major order
        // 2. CPU transpose [N,K] → [K,N]
        // 3. Upload transposed buffer to GPU
        // 4. Use standard gemm_forward (NoTrans, NoTrans) — same as LoRA weights
        use trueno_gpu::kernels::dequantize_nf4;
        let dequant_transpose_upload = |q: &trueno_gpu::kernels::Nf4Quantized,
                                        n: usize,
                                        k: usize|
         -> std::result::Result<
            GpuBuffer<f32>,
            crate::autograd::cuda_tensor::CudaTensorError,
        > {
            let deq = dequantize_nf4(q); // [N*K] in [N,K] row-major
            let nonzero = deq.iter().filter(|&&x| x != 0.0).count();
            eprintln!(
                "[TRACE] dequant n={n} k={k} len={} nonzero={nonzero} first5={:?}",
                deq.len(),
                &deq[..5.min(deq.len())]
            );
            assert_eq!(deq.len(), n * k, "dequant size mismatch: {} vs {}x{}", deq.len(), n, k);
            // Transpose [N,K] → [K,N]
            let mut transposed = vec![0.0f32; n * k];
            for row in 0..n {
                for col in 0..k {
                    transposed[col * n + row] = deq[row * k + col];
                }
            }
            let buf = GpuBuffer::from_host(&ctx, &transposed).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "dequant transpose upload: {e:?}"
                ))
            })?;
            // Verify upload: must read FULL buffer then slice
            let mut verify_full = vec![0.0f32; buf.len()];
            let verify_ok = buf.copy_to_host(&mut verify_full).is_ok();
            let verify5: Vec<f32> = verify_full.iter().copied().take(5).collect();
            let nz = verify_full.iter().filter(|&&x| x != 0.0).count();
            eprintln!("[TRACE] uploaded ptr={:?} len={} copy_ok={verify_ok} nonzero={nz} verify[:5]={verify5:?}", buf.as_ptr(), buf.len());
            Ok(buf)
        };
        // Each weight is [out_features, in_features] = [N, K].
        // After transpose: [K, N] — standard cuBLAS B layout.
        let w_q_fp32 = dequant_transpose_upload(&w_q_nf4_q, q_dim, hidden_size)?;
        let w_k_fp32 = dequant_transpose_upload(&w_k_nf4_q, kv_hidden_size, hidden_size)?;
        let w_v_fp32 = dequant_transpose_upload(&w_v_nf4_q, kv_hidden_size, hidden_size)?;
        let w_o_fp32 = dequant_transpose_upload(&w_o_nf4_q, hidden_size, q_dim)?;
        let w_gate_fp32 = dequant_transpose_upload(&w_gate_nf4_q, intermediate_size, hidden_size)?;
        let w_up_fp32 = dequant_transpose_upload(&w_up_nf4_q, intermediate_size, hidden_size)?;
        let w_down_fp32 = dequant_transpose_upload(&w_down_nf4_q, hidden_size, intermediate_size)?;

        // NF4 blocks do NOT allocate scratch — shared across all layers (C-SCRATCH-001).
        // Pipeline allocates one CudaBlockScratch and passes &mut to each forward() call.
        // Saves (L-1) * 214 MB = 7.5 GB for Qwen3-4B (36 layers).

        // Upload LoRA adapters to GPU (ENT-153)
        // B matrices are pre-scaled by lora_scale to avoid a separate scale kernel in forward.
        let (lora_a_q, lora_b_q) = match q_lora {
            Some((a_data, b_data)) => {
                let a = GpuBuffer::from_host(&ctx, a_data)?;
                let scaled_b: Vec<f32> = b_data.iter().map(|&v| v * lora_scale).collect();
                let b = GpuBuffer::from_host(&ctx, &scaled_b)?;
                (Some(a), Some(b))
            }
            None => (None, None),
        };
        let (lora_a_v, lora_b_v) = match v_lora {
            Some((a_data, b_data)) => {
                let a = GpuBuffer::from_host(&ctx, a_data)?;
                let scaled_b: Vec<f32> = b_data.iter().map(|&v| v * lora_scale).collect();
                let b = GpuBuffer::from_host(&ctx, &scaled_b)?;
                (Some(a), Some(b))
            }
            None => (None, None),
        };

        // ENT-270: Upload QK-norm weights if present
        let q_norm_weight = match q_norm {
            Some(w) => {
                assert_eq!(
                    w.len(),
                    config.head_dim(),
                    "ENT-270: q_norm weight expected [head_dim={}], got [{}]",
                    config.head_dim(),
                    w.len()
                );
                Some(GpuBuffer::from_host(&ctx, w)?)
            }
            None => None,
        };
        let k_norm_weight = match k_norm {
            Some(w) => {
                assert_eq!(
                    w.len(),
                    config.head_dim(),
                    "ENT-270: k_norm weight expected [head_dim={}], got [{}]",
                    config.head_dim(),
                    w.len()
                );
                Some(GpuBuffer::from_host(&ctx, w)?)
            }
            None => None,
        };

        Ok(Self {
            config: config.clone(),
            layer_idx,
            input_norm_weight,
            post_attn_norm_weight,
            w_q_nf4,
            w_q_scales,
            w_k_nf4,
            w_k_scales,
            w_v_nf4,
            w_v_scales,
            w_o_nf4,
            w_o_scales,
            w_gate_nf4,
            w_gate_scales,
            w_up_nf4,
            w_up_scales,
            w_down_nf4,
            w_down_scales,
            w_q_fp32,
            w_k_fp32,
            w_v_fp32,
            w_o_fp32,
            w_gate_fp32,
            w_up_fp32,
            w_down_fp32,
            lora_a_q,
            lora_b_q,
            lora_a_v,
            lora_b_v,
            lora_scale,
            lora_rank,
            q_norm_weight,
            k_norm_weight,
            // FP16 weights: None by default, populated by set_fp16_weights() (PMAT-470)
            w_q_fp16: None,
            w_k_fp16: None,
            w_v_fp16: None,
            w_o_fp16: None,
            w_gate_fp16: None,
            w_up_fp16: None,
            w_down_fp16: None,
            ctx,
        })
    }

    /// Cast fp32→fp16 weights + drop fp32 (PMAT-470/472). Frees ~2.6 GB VRAM.
    pub fn set_fp16_weights(&mut self, stream: &CudaStream) -> Result<()> {
        let cast_weight = |w_fp32: &GpuBuffer<f32>, ctx: &CudaContext| -> Result<GpuBuffer<u16>> {
            let n = w_fp32.len();
            let mut w_fp16 = GpuBuffer::<u16>::new(ctx, n)?;
            cast_f32_to_f16_gpu(w_fp32, &mut w_fp16, n as u32, stream)?;
            Ok(w_fp16)
        };

        self.w_q_fp16 = Some(cast_weight(&self.w_q_fp32, &self.ctx)?);
        self.w_k_fp16 = Some(cast_weight(&self.w_k_fp32, &self.ctx)?);
        self.w_v_fp16 = Some(cast_weight(&self.w_v_fp32, &self.ctx)?);
        self.w_o_fp16 = Some(cast_weight(&self.w_o_fp32, &self.ctx)?);
        self.w_gate_fp16 = Some(cast_weight(&self.w_gate_fp32, &self.ctx)?);
        self.w_up_fp16 = Some(cast_weight(&self.w_up_fp32, &self.ctx)?);
        self.w_down_fp16 = Some(cast_weight(&self.w_down_fp32, &self.ctx)?);

        stream.synchronize().map_err(|e| {
            crate::autograd::cuda_tensor::CudaTensorError::KernelError(format!(
                "FP16 weight cast sync failed: {e:?}"
            ))
        })?;
        // PMAT-472: Drop fp32 weights — backward now uses fp16 via gemm_backward_a_fp16_dispatch.
        // Frees ~2.6 GB VRAM on yoga 8GB, allowing GPU embeddings to fit.
        let dummy = |ctx: &CudaContext| GpuBuffer::<f32>::new(ctx, 1).unwrap();
        self.w_q_fp32 = dummy(&self.ctx);
        self.w_k_fp32 = dummy(&self.ctx);
        self.w_v_fp32 = dummy(&self.ctx);
        self.w_o_fp32 = dummy(&self.ctx);
        self.w_gate_fp32 = dummy(&self.ctx);
        self.w_up_fp32 = dummy(&self.ctx);
        self.w_down_fp32 = dummy(&self.ctx);
        eprintln!("[FP16] Weights cast + fp32 dropped (~2.6 GB freed)");

        Ok(())
    }

    /// Forward pass: cuBLAS GEMM with pre-dequantized weights (ENT-287, C-SCRATCH-001).
    #[rustfmt::skip]
    pub(crate) fn forward(
        &self,
        input: &GpuBuffer<f32>,
        output: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        scratch: &mut CudaBlockScratch,
    ) -> Result<()> {
        use crate::autograd::cuda_forward::{gemm_forward, gemm_nf4_forward, gemm_nf4_tc_forward};

        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.q_dim();
        let kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let intermediate_size = self.config.intermediate_size;

        // entrenar#318: scratch zeroing moved to forward_cuda_training (once per step, not per layer)
        scratch.prepare_causal_mask(seq_len, &self.ctx)?;

        // === Pre-attention RMSNorm === (PMAT-483: per-op profiling)
        let _t = scratch.op_begin();
        rms_norm_forward(
            input,
            &self.input_norm_weight,
            &mut scratch.norm1_out,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            stream,
        )?;
        scratch.op_end(_t, OP_RMSNORM_ATTN);

        // === Q, K, V Projections ===
        // Backend selection:
        //   FP16_GEMM=1: fp16 tensor core GEMM (Tier 2 parity, 2x BW savings)
        //   NF4_FUSED_GEMM=1: fused dequant+GEMM (8x less BW, 100% GPU, but naive PTX)
        //   Default: cuBLAS fp32 (197 tok/s, 7% GPU, memory-BW bound)
        static USE_NF4_GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let nf4_gemm = *USE_NF4_GEMM.get_or_init(|| std::env::var("NF4_FUSED_GEMM").as_deref() == Ok("1"));
        static USE_NF4_TC_GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let nf4_tc_gemm = *USE_NF4_TC_GEMM.get_or_init(|| std::env::var("NF4_TC_GEMM").as_deref() == Ok("1"));
        static USE_FP16_GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fp16_gemm = *USE_FP16_GEMM.get_or_init(|| std::env::var("FP16_GEMM").as_deref() == Ok("1"));

        // FP16 path: cast activation once, reuse for all projections
        let act_n = (seq_len * hidden_size) as u32;
        if fp16_gemm && self.w_q_fp16.is_some() {
            // Lazy-allocate fp16 activation buffer
            if scratch.norm1_out_f16.is_none() {
                scratch.norm1_out_f16 = Some(GpuBuffer::new(&self.ctx, seq_len * hidden_size)?);
            }
            let f16_buf = scratch.norm1_out_f16.as_mut().unwrap();
            cast_f32_to_f16_gpu(&scratch.norm1_out, f16_buf, act_n, stream)?;
        }

        let _t = scratch.op_begin(); // QKV GEMM timing
        if fp16_gemm && self.w_q_fp16.is_some() {
            let f16_act = scratch.norm1_out_f16.as_ref().unwrap();
            gemm_f16_to_f32_forward(f16_act, self.w_q_fp16.as_ref().unwrap(), &mut scratch.q,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(q_dim), stream)?;
        } else if nf4_tc_gemm {
            gemm_nf4_tc_forward(&scratch.norm1_out, &self.w_q_nf4, &self.w_q_scales, &mut scratch.q,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(q_dim), stream)?;
        } else if nf4_gemm {
            gemm_nf4_forward(&scratch.norm1_out, &self.w_q_nf4, &self.w_q_scales, &mut scratch.q,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(q_dim), stream)?;
        } else {
            gemm_forward(&scratch.norm1_out, &self.w_q_fp32, &mut scratch.q,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(q_dim), stream)?;
        }

        // ENT-153: Q LoRA: q += (norm1_out @ A_q) @ B_q  (B_q pre-scaled by lora_scale)
        if let (Some(a_q), Some(b_q)) = (&self.lora_a_q, &self.lora_b_q) {
            let s = saturating_u32(seq_len);
            let h = saturating_u32(hidden_size);
            let r = saturating_u32(self.lora_rank);
            let qd = saturating_u32(q_dim);
            // lora_inter[seq, rank] = norm1_out[seq, hidden] @ A_q[hidden, rank]
            gemm_forward(&scratch.norm1_out, a_q, &mut scratch.lora_inter, s, h, r, stream)?;
            // lora_temp[seq, q_dim] = lora_inter[seq, rank] @ B_q[rank, q_dim]
            gemm_forward(&scratch.lora_inter, b_q, &mut scratch.lora_temp, s, r, qd, stream)?;
            // q += lora_temp (in-place add)
            cuda_add_inplace(&mut scratch.q, &scratch.lora_temp, seq_len * q_dim, stream)?;
        }

        if fp16_gemm && self.w_k_fp16.is_some() {
            let f16_act = scratch.norm1_out_f16.as_ref().unwrap();
            gemm_f16_to_f32_forward(f16_act, self.w_k_fp16.as_ref().unwrap(), &mut scratch.k,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(kv_hidden_size), stream)?;
            gemm_f16_to_f32_forward(f16_act, self.w_v_fp16.as_ref().unwrap(), &mut scratch.v,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(kv_hidden_size), stream)?;
        } else if nf4_tc_gemm {
            // PMAT-481: NF4 tensor core GEMM for K and V projections (separate)
            gemm_nf4_tc_forward(&scratch.norm1_out, &self.w_k_nf4, &self.w_k_scales, &mut scratch.k,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(kv_hidden_size), stream)?;
            gemm_nf4_tc_forward(&scratch.norm1_out, &self.w_v_nf4, &self.w_v_scales, &mut scratch.v,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(kv_hidden_size), stream)?;
        } else if nf4_gemm {
            // PMAT-478: Fused K+V — shared input load (same pattern as Gate+Up)
            crate::autograd::cuda_forward::gemm_nf4_gate_up_forward(
                &scratch.norm1_out,
                &self.w_k_nf4, &self.w_k_scales,
                &self.w_v_nf4, &self.w_v_scales,
                &mut scratch.k, &mut scratch.v,
                saturating_u32(seq_len), saturating_u32(hidden_size),
                saturating_u32(kv_hidden_size), stream,
            )?;
        } else {
            gemm_forward(&scratch.norm1_out, &self.w_k_fp32, &mut scratch.k,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(kv_hidden_size), stream)?;
            gemm_forward(&scratch.norm1_out, &self.w_v_fp32, &mut scratch.v,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(kv_hidden_size), stream)?;
        }

        scratch.op_end(_t, OP_QKV_GEMM); // End QKV timing (includes Q/K/V GEMMs + Q LoRA)

        // ENT-153: V LoRA: v += (norm1_out @ A_v) @ B_v  (B_v pre-scaled by lora_scale)
        if let (Some(a_v), Some(b_v)) = (&self.lora_a_v, &self.lora_b_v) {
            let s = saturating_u32(seq_len);
            let h = saturating_u32(hidden_size);
            let r = saturating_u32(self.lora_rank);
            let vd = saturating_u32(kv_hidden_size);
            // lora_inter[seq, rank] = norm1_out[seq, hidden] @ A_v[hidden, rank]
            gemm_forward(&scratch.norm1_out, a_v, &mut scratch.lora_inter, s, h, r, stream)?;
            // lora_temp[seq, kv_hidden] = lora_inter[seq, rank] @ B_v[rank, kv_hidden]
            gemm_forward(&scratch.lora_inter, b_v, &mut scratch.lora_temp, s, r, vd, stream)?;
            // v += lora_temp (in-place add)
            cuda_add_inplace(&mut scratch.v, &scratch.lora_temp, seq_len * kv_hidden_size, stream)?;
        }

        // === Multi-Head Attention (GPU-only, zero CPU transfers) ===
        let _t = scratch.op_begin();
        self.compute_attention_cuda(seq_len, stream, scratch)?;
        scratch.op_end(_t, OP_ATTENTION);

        // === Output Projection ===
        let _t = scratch.op_begin();
        if fp16_gemm && self.w_o_fp16.is_some() {
            if scratch.attn_out_f16.is_none() {
                scratch.attn_out_f16 = Some(GpuBuffer::new(&self.ctx, seq_len * q_dim)?);
            }
            let f16_buf = scratch.attn_out_f16.as_mut().unwrap();
            cast_f32_to_f16_gpu(&scratch.attn_out, f16_buf, (seq_len * q_dim) as u32, stream)?;
            gemm_f16_to_f32_forward(f16_buf, self.w_o_fp16.as_ref().unwrap(), &mut scratch.o_proj_out,
                saturating_u32(seq_len), saturating_u32(q_dim), saturating_u32(hidden_size), stream)?;
        } else if nf4_tc_gemm {
            gemm_nf4_tc_forward(&scratch.attn_out, &self.w_o_nf4, &self.w_o_scales, &mut scratch.o_proj_out,
                saturating_u32(seq_len), saturating_u32(q_dim), saturating_u32(hidden_size), stream)?;
        } else if nf4_gemm {
            gemm_nf4_forward(&scratch.attn_out, &self.w_o_nf4, &self.w_o_scales, &mut scratch.o_proj_out,
                saturating_u32(seq_len), saturating_u32(q_dim), saturating_u32(hidden_size), stream)?;
        } else {
            gemm_forward(&scratch.attn_out, &self.w_o_fp32, &mut scratch.o_proj_out,
                saturating_u32(seq_len), saturating_u32(q_dim), saturating_u32(hidden_size), stream)?;
        }

        scratch.op_end(_t, OP_O_PROJ);

        // === Fused Residual Add + RMSNorm (entrenar#321: eliminates NaN cascade) ===
        let _t = scratch.op_begin();
        // The separate cuda_add + rms_norm_forward allows activation explosion between
        // the two operations. Fusing them prevents NaN in layers 24-27 because RMSNorm
        // normalizes the residual sum immediately, before it can propagate.
        fused_residual_rmsnorm_forward(
            input,
            &scratch.o_proj_out,
            &mut scratch.residual1,
            &mut scratch.norm2_out,
            &self.post_attn_norm_weight,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            stream,
        )?;

        scratch.op_end(_t, OP_RMSNORM_FFN); // Fused residual + RMSNorm

        // === FFN: Gate + Up + SwiGLU + Down ===
        let _t = scratch.op_begin(); // Gate+Up GEMM timing
        if fp16_gemm && self.w_gate_fp16.is_some() {
            if scratch.norm2_out_f16.is_none() {
                scratch.norm2_out_f16 = Some(GpuBuffer::new(&self.ctx, seq_len * hidden_size)?);
            }
            let f16_buf = scratch.norm2_out_f16.as_mut().unwrap();
            cast_f32_to_f16_gpu(&scratch.norm2_out, f16_buf, (seq_len * hidden_size) as u32, stream)?;
            gemm_f16_to_f32_forward(f16_buf, self.w_gate_fp16.as_ref().unwrap(), &mut scratch.gate_out,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(intermediate_size), stream)?;
            gemm_f16_to_f32_forward(f16_buf, self.w_up_fp16.as_ref().unwrap(), &mut scratch.up_out,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(intermediate_size), stream)?;
        } else if nf4_tc_gemm {
            // PMAT-481: NF4 tensor core GEMM for Gate and Up projections (separate)
            gemm_nf4_tc_forward(&scratch.norm2_out, &self.w_gate_nf4, &self.w_gate_scales, &mut scratch.gate_out,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(intermediate_size), stream)?;
            gemm_nf4_tc_forward(&scratch.norm2_out, &self.w_up_nf4, &self.w_up_scales, &mut scratch.up_out,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(intermediate_size), stream)?;
        } else if nf4_gemm {
            // PMAT-475: Fused gate+up — shared input load, saves M×K×4 bytes DRAM
            crate::autograd::cuda_forward::gemm_nf4_gate_up_forward(
                &scratch.norm2_out,
                &self.w_gate_nf4, &self.w_gate_scales,
                &self.w_up_nf4, &self.w_up_scales,
                &mut scratch.gate_out, &mut scratch.up_out,
                saturating_u32(seq_len), saturating_u32(hidden_size),
                saturating_u32(intermediate_size), stream,
            )?;
        } else {
            gemm_forward(&scratch.norm2_out, &self.w_gate_fp32, &mut scratch.gate_out,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(intermediate_size), stream)?;
            gemm_forward(&scratch.norm2_out, &self.w_up_fp32, &mut scratch.up_out,
                saturating_u32(seq_len), saturating_u32(hidden_size), saturating_u32(intermediate_size), stream)?;
        }

        scratch.op_end(_t, OP_GATE_UP_GEMM);

        // === FFN: Fused SwiGLU ===
        let _t = scratch.op_begin();
        fused_swiglu_forward(&scratch.gate_out, &scratch.up_out, &mut scratch.swiglu_out,
            saturating_u32(seq_len * intermediate_size), stream)?;
        scratch.op_end(_t, OP_SILU);

        // === FFN: Down Projection ===
        let _t = scratch.op_begin();
        if fp16_gemm && self.w_down_fp16.is_some() {
            if scratch.swiglu_out_f16.is_none() {
                scratch.swiglu_out_f16 = Some(GpuBuffer::new(&self.ctx, seq_len * intermediate_size)?);
            }
            let f16_buf = scratch.swiglu_out_f16.as_mut().unwrap();
            cast_f32_to_f16_gpu(&scratch.swiglu_out, f16_buf, (seq_len * intermediate_size) as u32, stream)?;
            gemm_f16_to_f32_forward(f16_buf, self.w_down_fp16.as_ref().unwrap(), &mut scratch.ffn_out,
                saturating_u32(seq_len), saturating_u32(intermediate_size), saturating_u32(hidden_size), stream)?;
        } else if nf4_tc_gemm {
            gemm_nf4_tc_forward(&scratch.swiglu_out, &self.w_down_nf4, &self.w_down_scales, &mut scratch.ffn_out,
                saturating_u32(seq_len), saturating_u32(intermediate_size), saturating_u32(hidden_size), stream)?;
        } else if nf4_gemm {
            gemm_nf4_forward(&scratch.swiglu_out, &self.w_down_nf4, &self.w_down_scales, &mut scratch.ffn_out,
                saturating_u32(seq_len), saturating_u32(intermediate_size), saturating_u32(hidden_size), stream)?;
        } else {
            gemm_forward(&scratch.swiglu_out, &self.w_down_fp32, &mut scratch.ffn_out,
                saturating_u32(seq_len), saturating_u32(intermediate_size), saturating_u32(hidden_size), stream)?;
        }

        scratch.op_end(_t, OP_DOWN_GEMM);

        // === Final Residual Add ===
        cuda_add(&scratch.residual1, &scratch.ffn_out, output, seq_len * hidden_size, stream)?;

        Ok(())
    }

    /// Layer index accessor.
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }
}

/// Helper: delegate attention computation using shared scratch buffers.
///
/// `CudaNf4TransformerBlock` reuses the same attention pipeline as the fp32 block
/// since attention operates on fp32 activations (Q/K/V are already dequantized by GEMM).
#[cfg(feature = "cuda")]
impl CudaNf4TransformerBlock {
    fn compute_attention_cuda(
        &self,
        seq_len: usize,
        stream: &CudaStream,
        scratch: &mut CudaBlockScratch,
    ) -> Result<()> {
        let num_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let heads_per_kv = num_heads / num_kv_heads;

        let s = saturating_u32(seq_len);
        let nh = saturating_u32(num_heads);
        let nkv = saturating_u32(num_kv_heads);
        let hd = saturating_u32(head_dim);

        // ── ENT-270: Apply QK-norm (per-head RMSNorm) on Q and K ──────────
        // Must happen BEFORE RoPE, matching CPU path ordering:
        //   projection → QK-norm → RoPE → attention
        // SAFETY: In-place GPU operations — CUDA kernels read all input before writing output.
        if let Some(ref q_norm) = self.q_norm_weight {
            for pos in 0..seq_len {
                let q_ref = unsafe { &*(std::ptr::addr_of!(scratch.q)) };
                per_head_rmsnorm_forward(q_ref, q_norm, &mut scratch.q, nh, hd, pos, stream)?;
            }
        }
        if let Some(ref k_norm) = self.k_norm_weight {
            for pos in 0..seq_len {
                let k_ref = unsafe { &*(std::ptr::addr_of!(scratch.k)) };
                per_head_rmsnorm_forward(k_ref, k_norm, &mut scratch.k, nkv, hd, pos, stream)?;
            }
        }

        // ── ENT-270: Apply RoPE (NeoX half-rotation) on Q and K ──────────
        // ALB-119: Batched launch (2 kernels) replaces per-position loop (2*seq_len kernels)
        let rope_theta = self.config.rope_theta;
        {
            let q_ref = unsafe { &*(std::ptr::addr_of!(scratch.q)) };
            batched_rope_neox_forward(
                q_ref,
                &mut scratch.q,
                &scratch.rope_positions,
                nh,
                hd,
                s,
                rope_theta,
                stream,
            )?;
            let k_ref = unsafe { &*(std::ptr::addr_of!(scratch.k)) };
            batched_rope_neox_forward(
                k_ref,
                &mut scratch.k,
                &scratch.rope_positions,
                nkv,
                hd,
                s,
                rope_theta,
                stream,
            )?;
        }

        // Q: interleaved → batched layout
        interleaved_to_batched_forward(&scratch.q, &mut scratch.attn_q_batched, s, nh, hd, stream)?;

        // K: interleaved → batched, then GQA expand if needed
        interleaved_to_batched_forward(&scratch.k, &mut scratch.attn_kv_temp, s, nkv, hd, stream)?;

        if heads_per_kv > 1 {
            expand_kv_heads(
                &scratch.attn_kv_temp,
                &mut scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        } else {
            // SAFETY: D2D copy with matching buffer sizes
            unsafe {
                scratch
                    .attn_kv_temp2
                    .copy_from_buffer_async(&scratch.attn_kv_temp, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "K copy failed: {e:?}"
                        ))
                    })?;
            }
        }

        // K^T: transpose for attention scores
        batched_transpose_forward(
            &scratch.attn_kv_temp2,
            &mut scratch.attn_kv_temp,
            nh,
            s,
            hd,
            stream,
        )?;

        // Q @ K^T → attention scores
        batched_4d_gemm_forward(
            &scratch.attn_q_batched,
            &scratch.attn_kv_temp,
            &mut scratch.attn_scores,
            1,
            nh,
            s,
            s,
            hd,
            stream,
        )?;

        // Scale by 1/sqrt(head_dim)
        let scale_factor = 1.0 / (head_dim as f32).sqrt();
        let total_scores = num_heads * seq_len * seq_len;
        let scores_view = unsafe {
            GpuBuffer::<f32>::from_raw_parts(
                scratch.attn_scores.as_ptr(),
                scratch.attn_scores.len(),
            )
        };
        scale_forward(
            &scores_view,
            &mut scratch.attn_scores,
            scale_factor,
            saturating_u32(total_scores),
            stream,
        )?;
        leak(scores_view);

        // Softmax (in-place: input aliased with output via unsafe view)
        // C-CAUSAL-001: Apply causal mask before softmax (NF4 path)
        // PMAT-420: Use causal_mask_contiguous (correctly strided for seq_len)
        // instead of causal_mask (strided at max_seq_len, causes row misalignment
        // when seq_len < max_seq_len, leading to NaN after deep layers).
        {
            let seq_sq = seq_len * seq_len;
            let mask_ptr = scratch.causal_mask_contiguous.as_ptr();
            let scores_base = scratch.attn_scores.as_ptr();
            for head in 0..num_heads {
                let byte_offset = (head * seq_sq * 4) as u64;
                let head_ptr = scores_base + byte_offset;
                let mask_view = unsafe { GpuBuffer::<f32>::from_raw_parts(mask_ptr, seq_sq) };
                let scores_view = unsafe { GpuBuffer::<f32>::from_raw_parts(head_ptr, seq_sq) };
                let mut out_view = unsafe { GpuBuffer::<f32>::from_raw_parts(head_ptr, seq_sq) };
                residual_add_forward(
                    &mask_view,
                    &scores_view,
                    &mut out_view,
                    saturating_u32(seq_sq),
                    stream,
                )?;
                leak(mask_view);
                leak(scores_view);
                leak(out_view);
            }
        }

        // SAFETY: The softmax kernel reads each row completely into shared memory / registers
        // before writing output. The view is forgotten to prevent double-free.
        let scores_view = unsafe {
            GpuBuffer::<f32>::from_raw_parts(
                scratch.attn_scores.as_ptr(),
                scratch.attn_scores.len(),
            )
        };
        batched_softmax_forward(
            &scores_view,
            &mut scratch.attn_scores,
            saturating_u32(num_heads * seq_len),
            s,
            stream,
        )?;
        leak(scores_view);

        // V: interleaved → batched, then GQA expand
        interleaved_to_batched_forward(&scratch.v, &mut scratch.attn_kv_temp, s, nkv, hd, stream)?;

        if heads_per_kv > 1 {
            expand_kv_heads(
                &scratch.attn_kv_temp,
                &mut scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        } else {
            // SAFETY: async GPU buffer copy within same CUDA stream; both buffers are
            // pre-allocated scratch with matching sizes, and stream ordering guarantees
            // the source is fully written before this copy executes.
            unsafe {
                scratch
                    .attn_kv_temp2
                    .copy_from_buffer_async(&scratch.attn_kv_temp, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "V copy failed: {e:?}"
                        ))
                    })?;
            }
        }

        // attn_scores @ V → attention output
        batched_4d_gemm_forward(
            &scratch.attn_scores,
            &scratch.attn_kv_temp2,
            &mut scratch.attn_q_batched,
            1,
            nh,
            s,
            hd,
            s,
            stream,
        )?;

        // Batched → interleaved layout
        batched_to_interleaved_forward(
            &scratch.attn_q_batched,
            &mut scratch.attn_out,
            s,
            nh,
            hd,
            stream,
        )?;

        Ok(())
    }
}

// =============================================================================
// QLoRA Backward Pass Types (ENT-153)
// =============================================================================

/// Shared gradient workspace for LoRA weight gradients (one per model, NOT per layer).
///
/// Backward processes layers sequentially — only one layer's LoRA gradients
/// are computed at a time. Sharing this workspace saves
/// `(L-1) * per_layer_lora_grad_elements * 4` bytes of VRAM.
///
/// # Contract (C-LORAGRADWS-001)
///
/// - **Precondition**: Allocated once before training loop starts
/// - **Postcondition**: After backward() for layer i, contains layer i's LoRA gradients
/// - **Invariant**: Buffer sizes match model config; never reallocated during training
#[cfg(feature = "cuda")]
pub(crate) struct CudaLoraGradWorkspace {
    /// Gradient for LoRA A_q [hidden_size, rank]
    pub(crate) grad_lora_a_q: GpuBuffer<f32>,
    /// Gradient for LoRA B_q [rank, q_dim]
    pub(crate) grad_lora_b_q: GpuBuffer<f32>,
    /// Gradient for LoRA A_v [hidden_size, rank]
    pub(crate) grad_lora_a_v: GpuBuffer<f32>,
    /// Gradient for LoRA B_v [rank, kv_hidden]
    pub(crate) grad_lora_b_v: GpuBuffer<f32>,
    /// Gradient for input norm weight [hidden_size]
    pub(crate) grad_input_norm: GpuBuffer<f32>,
    /// Gradient for post-attention norm weight [hidden_size]
    pub(crate) grad_post_attn_norm: GpuBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl CudaLoraGradWorkspace {
    /// Allocate shared LoRA gradient workspace.
    pub(crate) fn new(
        ctx: &Arc<CudaContext>,
        config: &super::config::TransformerConfig,
        lora_rank: usize,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let q_dim = config.q_dim();
        let kv = config.num_kv_heads * config.head_dim();
        let r = lora_rank;

        Ok(Self {
            grad_lora_a_q: GpuBuffer::new(ctx, h * r)?,
            grad_lora_b_q: GpuBuffer::new(ctx, r * q_dim)?,
            grad_lora_a_v: GpuBuffer::new(ctx, h * r)?,
            grad_lora_b_v: GpuBuffer::new(ctx, r * kv)?,
            grad_input_norm: GpuBuffer::new(ctx, h)?,
            grad_post_attn_norm: GpuBuffer::new(ctx, h)?,
        })
    }

    /// ENT-265: Clip all 6 LoRA gradient buffers by global L2 norm.
    ///
    /// Computes the global L2 norm across A_q, B_q, A_v, B_v, input_norm,
    /// and post_attn_norm. If the norm exceeds `max_norm`, scales all buffers
    /// down by `max_norm / (total_norm + 1e-6)`.
    ///
    /// Two-phase design: phase 1 reads norms (immutable), phase 2 applies
    /// scale (mutable). This satisfies the borrow checker when the workspace
    /// is behind a mutable reference.
    pub(crate) fn clip_gradients(&mut self, max_norm: f32, stream: &CudaStream) {
        // Phase 1: compute global L2 norm
        let sq_a_q = squared_sum_cuda(&self.grad_lora_a_q, self.grad_lora_a_q.len() as u32, stream)
            .unwrap_or(0.0);
        let sq_b_q = squared_sum_cuda(&self.grad_lora_b_q, self.grad_lora_b_q.len() as u32, stream)
            .unwrap_or(0.0);
        let sq_a_v = squared_sum_cuda(&self.grad_lora_a_v, self.grad_lora_a_v.len() as u32, stream)
            .unwrap_or(0.0);
        let sq_b_v = squared_sum_cuda(&self.grad_lora_b_v, self.grad_lora_b_v.len() as u32, stream)
            .unwrap_or(0.0);
        let sq_in =
            squared_sum_cuda(&self.grad_input_norm, self.grad_input_norm.len() as u32, stream)
                .unwrap_or(0.0);
        let sq_pa = squared_sum_cuda(
            &self.grad_post_attn_norm,
            self.grad_post_attn_norm.len() as u32,
            stream,
        )
        .unwrap_or(0.0);
        let total_norm = (sq_a_q + sq_b_q + sq_a_v + sq_b_v + sq_in + sq_pa).sqrt();

        if total_norm <= max_norm {
            return;
        }

        // Phase 2: apply clip scale
        let clip_scale = max_norm / (total_norm + 1e-6);
        let n_aq = self.grad_lora_a_q.len() as u32;
        let n_bq = self.grad_lora_b_q.len() as u32;
        let n_av = self.grad_lora_a_v.len() as u32;
        let n_bv = self.grad_lora_b_v.len() as u32;
        let n_in = self.grad_input_norm.len() as u32;
        let n_pa = self.grad_post_attn_norm.len() as u32;
        let _ = gradient_clip_cuda(&mut self.grad_lora_a_q, clip_scale, n_aq, stream);
        let _ = gradient_clip_cuda(&mut self.grad_lora_b_q, clip_scale, n_bq, stream);
        let _ = gradient_clip_cuda(&mut self.grad_lora_a_v, clip_scale, n_av, stream);
        let _ = gradient_clip_cuda(&mut self.grad_lora_b_v, clip_scale, n_bv, stream);
        let _ = gradient_clip_cuda(&mut self.grad_input_norm, clip_scale, n_in, stream);
        let _ = gradient_clip_cuda(&mut self.grad_post_attn_norm, clip_scale, n_pa, stream);
    }
}

/// GPU-resident AdamW optimizer state for LoRA adapters in one NF4 block.
///
/// Stores first (m) and second (v) moment estimates for:
/// - 4 LoRA weight tensors (A_q, B_q, A_v, B_v)
/// - 2 RMSNorm weights (input_norm, post_attn_norm)
///
/// # Contract (C-LORAOPT-001)
///
/// - **Precondition**: CUDA context valid, buffers match weight dimensions
/// - **Postcondition**: m and v initialized to zero
/// - **Invariant**: Buffer sizes immutable after creation
#[cfg(feature = "cuda")]
pub(crate) struct GpuLoraOptimizerState {
    m_lora_a_q: GpuBuffer<f32>,
    v_lora_a_q: GpuBuffer<f32>,
    m_lora_b_q: GpuBuffer<f32>,
    v_lora_b_q: GpuBuffer<f32>,
    m_lora_a_v: GpuBuffer<f32>,
    v_lora_a_v: GpuBuffer<f32>,
    m_lora_b_v: GpuBuffer<f32>,
    v_lora_b_v: GpuBuffer<f32>,
    m_input_norm: GpuBuffer<f32>,
    v_input_norm: GpuBuffer<f32>,
    m_post_attn_norm: GpuBuffer<f32>,
    v_post_attn_norm: GpuBuffer<f32>,
}

#[cfg(feature = "cuda")]
impl GpuLoraOptimizerState {
    fn new(
        ctx: &Arc<CudaContext>,
        config: &super::config::TransformerConfig,
        lora_rank: usize,
    ) -> Result<Self> {
        let h = config.hidden_size;
        let q_dim = config.q_dim();
        let kv = config.num_kv_heads * config.head_dim();
        let r = lora_rank;

        // CRITICAL: Must zero-initialize m/v buffers. GpuBuffer::new() does NOT
        // zero memory (cuMemAlloc returns uninitialized VRAM).
        let z = |n: usize| -> Result<GpuBuffer<f32>> {
            Ok(GpuBuffer::from_host(ctx, &vec![0.0f32; n])?)
        };
        Ok(Self {
            m_lora_a_q: z(h * r)?,
            v_lora_a_q: z(h * r)?,
            m_lora_b_q: z(r * q_dim)?,
            v_lora_b_q: z(r * q_dim)?,
            m_lora_a_v: z(h * r)?,
            v_lora_a_v: z(h * r)?,
            m_lora_b_v: z(r * kv)?,
            v_lora_b_v: z(r * kv)?,
            m_input_norm: z(h)?,
            v_input_norm: z(h)?,
            m_post_attn_norm: z(h)?,
            v_post_attn_norm: z(h)?,
        })
    }
}

// =============================================================================
// NF4 Block Backward Pass (ENT-153)
// =============================================================================

#[cfg(feature = "cuda")]
impl CudaNf4TransformerBlock {
    /// Backward pass with activation checkpointing and LoRA gradient computation.
    ///
    /// # Activation Checkpointing
    ///
    /// Re-runs forward to regenerate intermediate activations. Only `layer_input`
    /// is saved per-layer (47 MB for 36 layers at seq_len=128). This is the standard
    /// NF4 QLoRA backward (C-QLORA-BWD-001): recompute activations, propagate gradients.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn backward(
        &self,
        layer_input: &GpuBuffer<f32>,
        grad_output: &GpuBuffer<f32>,
        grad_input: &mut GpuBuffer<f32>,
        output_scratch: &mut GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        scratch: &mut CudaBlockScratch,
        grad_lora: &mut CudaLoraGradWorkspace,
    ) -> Result<()> {
        let hidden_size = self.config.hidden_size;
        let _q_dim = self.config.q_dim();
        let _kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let intermediate_size = self.config.intermediate_size;
        let eps = 1e-5_f32;

        // === Step 0: Activation checkpointing — re-run forward ===
        // This repopulates scratch with all intermediates needed for backward.
        self.forward(layer_input, output_scratch, seq_len, stream, scratch).map_err(|e| {
            eprintln!(
                "[backward] Layer {} activation-checkpoint forward FAILED: {e:?}",
                self.layer_idx
            );
            e
        })?;

        // === Step 1: FFN backward (NF4 transpose, no weight grads for frozen projections) ===
        self.backward_nf4_ffn(
            grad_output,
            seq_len,
            hidden_size,
            intermediate_size,
            stream,
            scratch,
        )?;

        // === Step 2: Post-attn norm backward ===
        let _t = scratch.op_begin(); // OP_NORM_BWD timing (both norms)
        rms_norm_backward(
            &scratch.residual1,
            &self.post_attn_norm_weight,
            &scratch.grad_hidden, // grad_from_ffn is accumulated in grad_hidden by backward_nf4_ffn
            grad_input,           // temporarily store post-attn-norm grad here
            &mut grad_lora.grad_post_attn_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )?;

        // Add residual connection: grad flows through both ffn and skip path
        // grad_residual1 = grad_input (from norm backward) + grad_output (from residual skip)
        cuda_add_inplace(grad_input, grad_output, seq_len * hidden_size, stream)?;

        // === Step 3: Attention backward (NF4 + LoRA for Q/V) ===
        self.backward_nf4_attention(
            grad_input, // grad coming into attention (from residual1)
            seq_len, stream, scratch, grad_lora,
        )?;

        // === Step 4: Input norm backward + first residual ===
        // At this point, scratch.grad_hidden contains grad from attention block
        // (accumulated by backward_nf4_attention into norm1_out reusing grad_hidden)
        rms_norm_backward(
            layer_input,
            &self.input_norm_weight,
            &scratch.grad_hidden, // grad flowing into norm1
            grad_input,           // final grad_input for this layer
            &mut grad_lora.grad_input_norm,
            saturating_u32(seq_len),
            saturating_u32(hidden_size),
            eps,
            stream,
        )?;

        scratch.op_end(_t, OP_NORM_BWD);

        Ok(())
    }

    /// FFN backward for NF4 blocks (ENT-287: cuBLAS fp32 GEMM).
    ///
    /// Propagates gradient through: down_proj → SwiGLU → gate/up projections.
    /// Uses cuBLAS GEMM with pre-dequantized fp32 weights for correct layout.
    /// No weight gradients for frozen NF4 weights.
    fn backward_nf4_ffn(
        &self,
        grad_output: &GpuBuffer<f32>,
        seq_len: usize,
        hidden_size: usize,
        intermediate_size: usize,
        stream: &CudaStream,
        scratch: &mut CudaBlockScratch,
    ) -> Result<()> {
        let s = saturating_u32(seq_len);
        let h = saturating_u32(hidden_size);
        let i_size = saturating_u32(intermediate_size);
        let n_inter = saturating_u32(seq_len * intermediate_size);

        // PMAT-481: NF4 tensor core backward dispatch
        static USE_NF4_TC_BWD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let nf4_tc_bwd =
            *USE_NF4_TC_BWD.get_or_init(|| std::env::var("NF4_TC_BWD_GEMM").as_deref() == Ok("1"));

        // Step 1: grad_swiglu[S,I] = grad_output[S,H] @ W_down^T (PMAT-472: fp16 dispatch)
        let _t = scratch.op_begin(); // OP_DOWN_BWD timing
        if nf4_tc_bwd {
            // NF4 TC backward: fused dequant+WMMA, no separate dequant kernel
            crate::autograd::cuda_forward::gemm_nf4_tc_backward_a(
                grad_output,
                &self.w_down_nf4,
                &self.w_down_scales,
                &mut scratch.grad_swiglu,
                s,
                h,      // N = hidden_size (grad_output cols)
                i_size, // K = intermediate_size (output cols = W_down rows)
                stream,
            )?;
        } else {
            gemm_backward_a_fp16_dispatch(
                grad_output,
                self.w_down_fp16.as_ref(),
                &self.w_down_fp32,
                &mut scratch.grad_swiglu,
                s,
                i_size,
                h,
                stream,
                &self.ctx,
            )?;
        }

        scratch.op_end(_t, OP_DOWN_BWD);

        // Step 2: SwiGLU backward: swiglu = silu(gate) * up
        // d_gate = d_swiglu * up * silu'(gate)
        // d_up   = d_swiglu * silu(gate)
        let _t = scratch.op_begin(); // OP_SWIGLU_BWD timing

        // temp1 = d_swiglu * up_out → store in swiglu_out (reuse)
        elementwise_mul_forward(
            &scratch.grad_swiglu,
            &scratch.up_out,
            &mut scratch.swiglu_out,
            n_inter,
            stream,
        )?;

        // silu_backward: d_gate_raw = temp1 * silu'(gate_out)
        // silu'(x) = silu(x) * (1 + x*(1-silu(x)))
        // Reuse up_out as storage for d_gate
        silu_backward(
            &scratch.gate_out,
            &scratch.swiglu_out,
            &mut scratch.up_out, // d_gate stored here
            stream,
        )?;

        // d_up = d_swiglu * silu(gate) → store in gate_out (reuse)
        // Compute silu(gate) into swiglu_out (scratch) — NOT ffn_out which is [S,H] (too small)
        silu_forward(&scratch.gate_out, &mut scratch.swiglu_out, n_inter, stream)?;
        // d_up = d_swiglu * silu(gate)
        elementwise_mul_forward(
            &scratch.grad_swiglu,
            &scratch.swiglu_out,
            &mut scratch.gate_out, // d_up stored here
            n_inter,
            stream,
        )?;

        scratch.op_end(_t, OP_SWIGLU_BWD);

        // Step 3: gate/up backward (PMAT-472: fp16 dispatch, PMAT-481: TC dispatch)
        let _t = scratch.op_begin(); // OP_GATE_UP_BWD timing
                                     // PMAT-484: Fused backward — use cuBLAS beta=1.0 accumulate to eliminate
                                     // the separate cuda_add_inplace kernel launch (3 launches → 2).
        static USE_FUSED_BWD: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fused_bwd = *USE_FUSED_BWD
            .get_or_init(|| std::env::var("NF4_FUSED_BWD_GEMM").as_deref() == Ok("1"));

        if nf4_tc_bwd {
            // PMAT-481: NF4 tensor core backward — fused dequant+WMMA per projection
            // Up backward: grad_up[S,H] = d_up[S,I] @ W_up^T
            crate::autograd::cuda_forward::gemm_nf4_tc_backward_a(
                &scratch.gate_out, // d_up stored in gate_out buffer
                &self.w_up_nf4,
                &self.w_up_scales,
                &mut scratch.grad_hidden,
                s,
                i_size, // N = intermediate_size (d_up cols)
                h,      // K = hidden_size (output cols)
                stream,
            )?;
            // Gate backward: grad_gate[S,H] = d_gate[S,I] @ W_gate^T
            crate::autograd::cuda_forward::gemm_nf4_tc_backward_a(
                &scratch.up_out, // d_gate stored in up_out buffer
                &self.w_gate_nf4,
                &self.w_gate_scales,
                &mut scratch.ffn_out,
                s,
                i_size, // N = intermediate_size
                h,      // K = hidden_size
                stream,
            )?;
            // Accumulate: grad_hidden += grad_gate
            cuda_add_inplace(
                &mut scratch.grad_hidden,
                &scratch.ffn_out,
                seq_len * hidden_size,
                stream,
            )?;
        } else if fused_bwd {
            // Fused path: compute up backward into grad_hidden, then accumulate gate
            gemm_backward_a_fp16_dispatch(
                &scratch.gate_out,
                self.w_up_fp16.as_ref(),
                &self.w_up_fp32,
                &mut scratch.grad_hidden,
                s,
                h,
                i_size,
                stream,
                &self.ctx,
            )?;
            // Accumulate gate backward into grad_hidden (beta=1.0)
            gemm_backward_a_fp16_dispatch_accumulate(
                &scratch.up_out,
                self.w_gate_fp16.as_ref(),
                &self.w_gate_fp32,
                &mut scratch.grad_hidden,
                s,
                h,
                i_size,
                stream,
                &self.ctx,
            )?;
        } else {
            // Unfused path: separate GEMMs + explicit add
            gemm_backward_a_fp16_dispatch(
                &scratch.up_out,
                self.w_gate_fp16.as_ref(),
                &self.w_gate_fp32,
                &mut scratch.ffn_out,
                s,
                h,
                i_size,
                stream,
                &self.ctx,
            )?;
            gemm_backward_a_fp16_dispatch(
                &scratch.gate_out,
                self.w_up_fp16.as_ref(),
                &self.w_up_fp32,
                &mut scratch.grad_hidden,
                s,
                h,
                i_size,
                stream,
                &self.ctx,
            )?;

            // Accumulate: grad_hidden = grad_norm2_gate + grad_norm2_up
            cuda_add_inplace(
                &mut scratch.grad_hidden,
                &scratch.ffn_out,
                seq_len * hidden_size,
                stream,
            )?;
        }
        scratch.op_end(_t, OP_GATE_UP_BWD);

        Ok(())
    }

    /// Attention backward for NF4 blocks with LoRA gradient computation (ENT-287).
    ///
    /// Propagates gradient through O projection, attention mechanism, and Q/K/V projections.
    /// Computes LoRA weight gradients for Q and V projections.
    /// Uses cuBLAS GEMM with pre-dequantized fp32 weights.
    fn backward_nf4_attention(
        &self,
        grad_residual1: &GpuBuffer<f32>,
        seq_len: usize,
        stream: &CudaStream,
        scratch: &mut CudaBlockScratch,
        grad_lora: &mut CudaLoraGradWorkspace,
    ) -> Result<()> {
        use crate::autograd::cuda_forward::gemm_forward;

        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.q_dim();
        let kv_hidden_size = self.config.num_kv_heads * self.config.head_dim();
        let num_heads = self.config.num_attention_heads;
        let head_dim = self.config.head_dim();

        let s = saturating_u32(seq_len);
        let h = saturating_u32(hidden_size);
        let qd = saturating_u32(q_dim);
        let kvh = saturating_u32(kv_hidden_size);

        // Step 1: O projection backward (PMAT-481: TC dispatch)
        static USE_NF4_TC_BWD_O: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let nf4_tc_bwd_o = *USE_NF4_TC_BWD_O
            .get_or_init(|| std::env::var("NF4_TC_BWD_GEMM").as_deref() == Ok("1"));

        let _t = scratch.op_begin(); // OP_ATTN_BWD timing (O-proj + attention mechanism)
        if nf4_tc_bwd_o {
            crate::autograd::cuda_forward::gemm_nf4_tc_backward_a(
                grad_residual1,
                &self.w_o_nf4,
                &self.w_o_scales,
                &mut scratch.attn_out,
                s,
                h,  // N = hidden_size (grad_residual1 cols)
                qd, // K = q_dim (output cols = W_o rows)
                stream,
            )?;
        } else {
            gemm_backward_a_fp16_dispatch(
                grad_residual1,
                self.w_o_fp16.as_ref(),
                &self.w_o_fp32,
                &mut scratch.attn_out,
                s,
                qd,
                h,
                stream,
                &self.ctx,
            )?;
        }

        // Step 2: Attention mechanism backward
        // This is complex (softmax backward, batched GEMMs) — reuse the fp32 attention backward
        // infrastructure since attention operates on fp32 activations.
        self.backward_nf4_attention_mechanism(seq_len, num_heads, head_dim, stream, scratch)?;

        // After attention backward: scratch.norm1_out-related grads are accumulated.
        // grad_q is in scratch.q, grad_k in scratch.k, grad_v in scratch.v

        // Step 2b: RoPE backward (inverse rotation) on grad_q and grad_k
        // Forward applied RoPE to Q and K. Undo rotation so projection backward
        // gets gradients in the unrotated coordinate frame.
        let rope_theta = self.config.rope_theta;
        let num_kv_heads = self.config.num_kv_heads;
        let nkv = saturating_u32(num_kv_heads);
        let nh = saturating_u32(num_heads);
        let hd = saturating_u32(head_dim);
        {
            let q_ref = unsafe { &*(std::ptr::addr_of!(scratch.q)) };
            batched_rope_neox_backward(
                q_ref,
                &mut scratch.q,
                &scratch.rope_positions,
                nh,
                hd,
                s,
                rope_theta,
                stream,
            )?;
            let k_ref = unsafe { &*(std::ptr::addr_of!(scratch.k)) };
            batched_rope_neox_backward(
                k_ref,
                &mut scratch.k,
                &scratch.rope_positions,
                nkv,
                hd,
                s,
                rope_theta,
                stream,
            )?;
        }

        scratch.op_end(_t, OP_ATTN_BWD);

        // Q/K/V backward (PMAT-472: fp16 dispatch, PMAT-481: TC dispatch)
        static USE_NF4_TC_BWD_ATTN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let nf4_tc_bwd = *USE_NF4_TC_BWD_ATTN
            .get_or_init(|| std::env::var("NF4_TC_BWD_GEMM").as_deref() == Ok("1"));

        let _t = scratch.op_begin(); // OP_QKV_BWD timing
        if nf4_tc_bwd {
            // PMAT-481: NF4 tensor core backward for Q projection
            crate::autograd::cuda_forward::gemm_nf4_tc_backward_a(
                &scratch.q,
                &self.w_q_nf4,
                &self.w_q_scales,
                &mut scratch.o_proj_out,
                s,
                qd, // N = q_dim (grad_q cols)
                h,  // K = hidden_size (output cols)
                stream,
            )?;
        } else {
            gemm_backward_a_fp16_dispatch(
                &scratch.q,
                self.w_q_fp16.as_ref(),
                &self.w_q_fp32,
                &mut scratch.o_proj_out,
                s,
                h,
                qd,
                stream,
                &self.ctx,
            )?;
        }

        // LoRA Q backward: compute grad_A_q, grad_B_q, and add to grad_norm1
        if let (Some(a_q), Some(b_q)) = (&self.lora_a_q, &self.lora_b_q) {
            let r = saturating_u32(self.lora_rank);

            // Recompute: lora_inter_q = norm1_out @ A_q  [S, rank]
            gemm_forward(&scratch.norm1_out, a_q, &mut scratch.lora_inter, s, h, r, stream)?;

            // grad_B_q = lora_inter_q^T @ grad_q  [rank, q_dim]
            // (Note: B_q was pre-scaled, so grad_B_q includes the scale factor)
            gemm_backward_b(
                &scratch.lora_inter,
                &scratch.q,
                &mut grad_lora.grad_lora_b_q,
                s,
                r,
                qd,
                stream,
            )?;

            // grad_lora_inter = grad_q @ B_q^T  [S, rank]
            gemm_backward_a(
                &scratch.q,
                b_q,
                &mut scratch.lora_inter, // reuse for grad_lora_inter
                s,
                qd,
                r,
                stream,
            )?;

            // grad_A_q = norm1_out^T @ grad_lora_inter  [H, rank]
            gemm_backward_b(
                &scratch.norm1_out,
                &scratch.lora_inter,
                &mut grad_lora.grad_lora_a_q,
                s,
                h,
                r,
                stream,
            )?;

            // Add LoRA's contribution to grad_norm1: += grad_lora_inter @ A_q^T  [S, H]
            gemm_backward_a(
                &scratch.lora_inter,
                a_q,
                &mut scratch.lora_temp, // [S, H]
                s,
                r,
                h,
                stream,
            )?;
            cuda_add_inplace(
                &mut scratch.o_proj_out,
                &scratch.lora_temp,
                seq_len * hidden_size,
                stream,
            )?;
        }

        // K+V backward (PMAT-484: fused, PMAT-481: TC dispatch)
        static USE_FUSED_BWD_ATTN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let fused_bwd = *USE_FUSED_BWD_ATTN
            .get_or_init(|| std::env::var("NF4_FUSED_BWD_GEMM").as_deref() == Ok("1"));

        if nf4_tc_bwd {
            // PMAT-481: NF4 tensor core backward for K and V projections
            crate::autograd::cuda_forward::gemm_nf4_tc_backward_a(
                &scratch.k,
                &self.w_k_nf4,
                &self.w_k_scales,
                &mut scratch.ffn_out,
                s,
                kvh, // N = kv_hidden (grad_k cols)
                h,   // K = hidden_size (output cols)
                stream,
            )?;
            cuda_add_inplace(
                &mut scratch.o_proj_out,
                &scratch.ffn_out,
                seq_len * hidden_size,
                stream,
            )?;
            crate::autograd::cuda_forward::gemm_nf4_tc_backward_a(
                &scratch.v,
                &self.w_v_nf4,
                &self.w_v_scales,
                &mut scratch.ffn_out,
                s,
                kvh, // N = kv_hidden
                h,   // K = hidden_size
                stream,
            )?;
            cuda_add_inplace(
                &mut scratch.o_proj_out,
                &scratch.ffn_out,
                seq_len * hidden_size,
                stream,
            )?;
        } else if fused_bwd {
            // Fused: K and V backward accumulate directly into o_proj_out
            gemm_backward_a_fp16_dispatch_accumulate(
                &scratch.k,
                self.w_k_fp16.as_ref(),
                &self.w_k_fp32,
                &mut scratch.o_proj_out,
                s,
                h,
                kvh,
                stream,
                &self.ctx,
            )?;
            gemm_backward_a_fp16_dispatch_accumulate(
                &scratch.v,
                self.w_v_fp16.as_ref(),
                &self.w_v_fp32,
                &mut scratch.o_proj_out,
                s,
                h,
                kvh,
                stream,
                &self.ctx,
            )?;
        } else {
            // Unfused: K backward → temp, accumulate; V backward → temp, accumulate
            gemm_backward_a_fp16_dispatch(
                &scratch.k,
                self.w_k_fp16.as_ref(),
                &self.w_k_fp32,
                &mut scratch.ffn_out,
                s,
                h,
                kvh,
                stream,
                &self.ctx,
            )?;
            cuda_add_inplace(
                &mut scratch.o_proj_out,
                &scratch.ffn_out,
                seq_len * hidden_size,
                stream,
            )?;

            gemm_backward_a_fp16_dispatch(
                &scratch.v,
                self.w_v_fp16.as_ref(),
                &self.w_v_fp32,
                &mut scratch.ffn_out,
                s,
                h,
                kvh,
                stream,
                &self.ctx,
            )?;
            cuda_add_inplace(
                &mut scratch.o_proj_out,
                &scratch.ffn_out,
                seq_len * hidden_size,
                stream,
            )?;
        }

        // LoRA V backward
        if let (Some(a_v), Some(b_v)) = (&self.lora_a_v, &self.lora_b_v) {
            let r = saturating_u32(self.lora_rank);

            // Recompute: lora_inter_v = norm1_out @ A_v  [S, rank]
            gemm_forward(&scratch.norm1_out, a_v, &mut scratch.lora_inter, s, h, r, stream)?;

            // grad_B_v = lora_inter_v^T @ grad_v  [rank, kv_hidden]
            gemm_backward_b(
                &scratch.lora_inter,
                &scratch.v,
                &mut grad_lora.grad_lora_b_v,
                s,
                r,
                kvh,
                stream,
            )?;

            // grad_lora_inter = grad_v @ B_v^T  [S, rank]
            gemm_backward_a(&scratch.v, b_v, &mut scratch.lora_inter, s, kvh, r, stream)?;

            // grad_A_v = norm1_out^T @ grad_lora_inter  [H, rank]
            gemm_backward_b(
                &scratch.norm1_out,
                &scratch.lora_inter,
                &mut grad_lora.grad_lora_a_v,
                s,
                h,
                r,
                stream,
            )?;

            // Add LoRA V's contribution to grad_norm1
            gemm_backward_a(&scratch.lora_inter, a_v, &mut scratch.lora_temp, s, r, h, stream)?;
            cuda_add_inplace(
                &mut scratch.o_proj_out,
                &scratch.lora_temp,
                seq_len * hidden_size,
                stream,
            )?;
        }

        scratch.op_end(_t, OP_QKV_BWD);

        // Step 6: Accumulated grad_norm1 is in scratch.o_proj_out → move to scratch.grad_hidden
        unsafe {
            scratch.grad_hidden.copy_from_buffer_async(&scratch.o_proj_out, stream).map_err(
                |e| {
                    crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                        "grad_norm1 copy failed: {e}"
                    ))
                },
            )?;
        }

        Ok(())
    }

    /// Attention mechanism backward (softmax, Q@K^T backward) for NF4 blocks.
    ///
    /// After this call:
    /// - scratch.q contains grad_q [S, q_dim]
    /// - scratch.k contains grad_k [S, kv_hidden]
    /// - scratch.v contains grad_v [S, kv_hidden]
    ///
    /// PMAT-486: Full attention backward (replaces previous no-op).
    /// Mirrors the FP32 backward in `backward_attention` (lines 1470-1810).
    ///
    /// Forward: attn_out = softmax(Q @ K^T / √d) @ V
    /// Backward:
    ///   1. grad_scores = grad_attn_batched @ V^T
    ///   2. grad_V = (grad_attn_batched^T @ attn_weights)^T  (buffer-safe identity)
    ///   3. softmax_backward(grad_scores, attn_weights) → grad_raw
    ///   4. grad_raw *= 1/√d
    ///   5. grad_Q = grad_raw @ K_expanded
    ///   6. grad_K = (Q^T @ grad_raw)^T
    ///   7. GQA reduction + batched→interleaved conversion
    ///
    /// Contract: attention-backward-v1.yaml
    fn backward_nf4_attention_mechanism(
        &self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        stream: &CudaStream,
        scratch: &mut CudaBlockScratch,
    ) -> Result<()> {
        let num_kv_heads = self.config.num_kv_heads;
        let heads_per_kv = num_heads / num_kv_heads;
        let s = saturating_u32(seq_len);
        let nh = saturating_u32(num_heads);
        let nkv = saturating_u32(num_kv_heads);
        let hd = saturating_u32(head_dim);
        let scale = 1.0 / (head_dim as f32).sqrt();

        // grad_attn_out is in scratch.attn_out [S, q_dim]
        // Convert to batched layout [NH, S, HD]
        interleaved_to_batched_forward(
            &scratch.attn_out,
            &mut scratch.attn_q_batched, // grad_attn_batched [NH, S, HD]
            s,
            nh,
            hd,
            stream,
        )?;

        // === Step 1: Expand V for GQA and transpose ===
        // V is in scratch.v [S, kv_hidden] from activation checkpointing forward.
        // Convert to batched [NKV, S, HD], then GQA-expand to [NH, S, HD].
        interleaved_to_batched_forward(&scratch.v, &mut scratch.attn_kv_temp, s, nkv, hd, stream)?;

        if heads_per_kv > 1 {
            expand_kv_heads(
                &scratch.attn_kv_temp,
                &mut scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        } else {
            unsafe {
                scratch
                    .attn_kv_temp2
                    .copy_from_buffer_async(&scratch.attn_kv_temp, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "V copy for attn backward: {e:?}"
                        ))
                    })?;
            }
        }
        // attn_kv_temp2 = V_expanded [NH, S, HD]

        // Transpose V: [NH, S, HD] → [NH, HD, S]
        batched_transpose_forward(
            &scratch.attn_kv_temp2,
            &mut scratch.attn_kv_temp, // V^T [NH, HD, S]
            nh,
            s,
            hd,
            stream,
        )?;

        // === Step 2: grad_attn_scores = grad_attn_batched @ V^T ===
        // [NH, S, HD] @ [NH, HD, S] → [NH, S, S]
        batched_4d_gemm_forward(
            &scratch.attn_q_batched,
            &scratch.attn_kv_temp,
            &mut scratch.grad_attn_scores,
            1,
            nh,
            s,
            s,
            hd,
            stream,
        )?;

        // === Step 3: grad_V = (grad_attn_batched^T @ attn_weights)^T ===
        // Uses the identity to avoid needing an [NH, S, S] transpose buffer.
        // attn_weights in scratch.attn_scores [NH, S, S] from forward checkpoint.

        // Step 3a: transpose grad_attn_batched [NH, S, HD] → [NH, HD, S]
        batched_transpose_forward(
            &scratch.attn_q_batched,
            &mut scratch.attn_kv_temp, // reuse: grad_attn_batched^T [NH, HD, S]
            nh,
            s,
            hd,
            stream,
        )?;

        // Step 3b: [NH, HD, S] @ [NH, S, S] → [NH, HD, S] (= grad_V^T)
        batched_4d_gemm_forward(
            &scratch.attn_kv_temp,
            &scratch.attn_scores,       // attn_weights from forward
            &mut scratch.attn_kv_temp2, // grad_V^T [NH, HD, S]
            1,
            nh,
            hd,
            s,
            s,
            stream,
        )?;

        // Step 3c: transpose grad_V^T [NH, HD, S] → grad_V [NH, S, HD]
        batched_transpose_forward(
            &scratch.attn_kv_temp2,
            &mut scratch.attn_kv_temp, // grad_V [NH, S, HD]
            nh,
            hd,
            s,
            stream,
        )?;
        // attn_kv_temp = grad_V [NH, S, HD]

        // === Step 4: Softmax backward ===
        // In-place: grad_attn_scores is both input and output.
        let total_rows = nh * s;
        {
            let grad_scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    scratch.grad_attn_scores.as_ptr(),
                    scratch.grad_attn_scores.len(),
                )
            };
            batched_softmax_backward(
                &scratch.attn_scores,
                &grad_scores_view,
                &mut scratch.grad_attn_scores,
                total_rows,
                s,
                stream,
            )?;
            leak(grad_scores_view);
        }

        // === Step 5: Scale backward (1/√d) ===
        let total_scores = saturating_u32(num_heads * seq_len * seq_len);
        {
            let scores_view = unsafe {
                GpuBuffer::<f32>::from_raw_parts(
                    scratch.grad_attn_scores.as_ptr(),
                    scratch.grad_attn_scores.len(),
                )
            };
            scale_forward(
                &scores_view,
                &mut scratch.grad_attn_scores,
                scale,
                total_scores,
                stream,
            )?;
            leak(scores_view);
        }

        // === Step 6: Reconstruct K, GQA expand, compute grad_Q ===
        interleaved_to_batched_forward(&scratch.k, &mut scratch.attn_kv_temp2, s, nkv, hd, stream)?;

        if heads_per_kv > 1 {
            unsafe {
                scratch
                    .attn_q_batched
                    .copy_from_buffer_async(&scratch.attn_kv_temp2, stream)
                    .map_err(|e| {
                        crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                            "K copy for GQA expand: {e}"
                        ))
                    })?;
            }
            expand_kv_heads(
                &scratch.attn_q_batched,
                &mut scratch.attn_kv_temp2,
                num_kv_heads,
                heads_per_kv,
                seq_len * head_dim,
                stream,
            )?;
        }
        // attn_kv_temp2 = K_expanded [NH, S, HD]

        // grad_Q = grad_raw_scores @ K_expanded → attn_q_batched [NH, S, HD]
        batched_4d_gemm_forward(
            &scratch.grad_attn_scores,
            &scratch.attn_kv_temp2,
            &mut scratch.attn_q_batched,
            1,
            nh,
            s,
            hd,
            s,
            stream,
        )?;

        // === Step 7: Compute grad_K ===
        // grad_K^T = Q^T @ grad_raw_scores
        // Reconstruct Q_batched into o_proj_out (attn_q_batched has grad_Q now)
        interleaved_to_batched_forward(
            &scratch.q,
            &mut scratch.o_proj_out, // temp for Q_batched
            s,
            nh,
            hd,
            stream,
        )?;

        // Transpose Q: [NH, S, HD] → [NH, HD, S]
        batched_transpose_forward(
            &scratch.o_proj_out,
            &mut scratch.attn_kv_temp2, // Q^T [NH, HD, S]
            nh,
            s,
            hd,
            stream,
        )?;

        // grad_K^T = Q^T @ grad_raw_scores → ffn_out as temp [NH, HD, S]
        batched_4d_gemm_forward(
            &scratch.attn_kv_temp2,
            &scratch.grad_attn_scores,
            &mut scratch.ffn_out, // grad_K^T [NH, HD, S]
            1,
            nh,
            hd,
            s,
            s,
            stream,
        )?;

        // Transpose grad_K^T → grad_K: [NH, HD, S] → [NH, S, HD]
        batched_transpose_forward(
            &scratch.ffn_out,
            &mut scratch.attn_kv_temp2, // grad_K [NH, S, HD]
            nh,
            hd,
            s,
            stream,
        )?;

        // === Step 8: GQA gradient reduction ===
        if heads_per_kv > 1 {
            self.reduce_gqa_gradients_nf4(
                num_kv_heads,
                heads_per_kv,
                seq_len,
                head_dim,
                stream,
                scratch,
            )?;
        }

        // === Step 9: Convert batched gradients → interleaved, store in scratch.q/k/v ===
        // grad_Q: attn_q_batched [NH, S, HD] → scratch.q [S, q_dim]
        batched_to_interleaved_forward(&scratch.attn_q_batched, &mut scratch.q, s, nh, hd, stream)?;

        // grad_K: attn_kv_temp2 [NKV, S, HD] → scratch.k [S, kv_hidden]
        batched_to_interleaved_forward(&scratch.attn_kv_temp2, &mut scratch.k, s, nkv, hd, stream)?;

        // grad_V: attn_kv_temp [NKV, S, HD] → scratch.v [S, kv_hidden]
        // (After GQA reduction, grad_V was reduced from [NH] to [NKV] heads)
        batched_to_interleaved_forward(&scratch.attn_kv_temp, &mut scratch.v, s, nkv, hd, stream)?;

        Ok(())
    }

    /// GQA gradient reduction for NF4 blocks.
    /// Reduces grad_K and grad_V from [num_heads, S, HD] to [num_kv_heads, S, HD]
    /// by summing across Q heads sharing each KV head.
    fn reduce_gqa_gradients_nf4(
        &self,
        num_kv_heads: usize,
        heads_per_kv: usize,
        seq_len: usize,
        head_dim: usize,
        stream: &CudaStream,
        scratch: &mut CudaBlockScratch,
    ) -> Result<()> {
        let chunk = seq_len * head_dim;
        for g in 0..num_kv_heads {
            let dst_off = g * chunk;
            // First Q head in group → initialize destination
            let src_off = g * heads_per_kv * chunk;
            // Copy first head
            {
                let src = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp2.as_ptr() + (src_off * 4) as u64,
                        chunk,
                    )
                };
                let mut dst = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp2.as_ptr() + (dst_off * 4) as u64,
                        chunk,
                    )
                };
                if src_off != dst_off {
                    unsafe {
                        dst.copy_from_buffer_async(&src, stream).map_err(|e| {
                            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                                "GQA K reduce copy: {e}"
                            ))
                        })?;
                    }
                }
                leak(src);
                leak(dst);
            }
            // Accumulate remaining heads
            for h in 1..heads_per_kv {
                let add_off = (g * heads_per_kv + h) * chunk;
                let src = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp2.as_ptr() + (add_off * 4) as u64,
                        chunk,
                    )
                };
                let mut dst = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp2.as_ptr() + (dst_off * 4) as u64,
                        chunk,
                    )
                };
                cuda_add_inplace(&mut dst, &src, chunk, stream)?;
                leak(src);
                leak(dst);
            }
            // Same for grad_V (in attn_kv_temp)
            {
                let src = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp.as_ptr() + (src_off * 4) as u64,
                        chunk,
                    )
                };
                let mut dst = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp.as_ptr() + (dst_off * 4) as u64,
                        chunk,
                    )
                };
                if src_off != dst_off {
                    unsafe {
                        dst.copy_from_buffer_async(&src, stream).map_err(|e| {
                            crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                                "GQA V reduce copy: {e}"
                            ))
                        })?;
                    }
                }
                leak(src);
                leak(dst);
            }
            for h in 1..heads_per_kv {
                let add_off = (g * heads_per_kv + h) * chunk;
                let src = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp.as_ptr() + (add_off * 4) as u64,
                        chunk,
                    )
                };
                let mut dst = unsafe {
                    GpuBuffer::<f32>::from_raw_parts(
                        scratch.attn_kv_temp.as_ptr() + (dst_off * 4) as u64,
                        chunk,
                    )
                };
                cuda_add_inplace(&mut dst, &src, chunk, stream)?;
                leak(src);
                leak(dst);
            }
        }
        Ok(())
    }

    /// Initialize LoRA optimizer state for this block.
    pub(crate) fn init_lora_optimizer_state(&self) -> Result<GpuLoraOptimizerState> {
        GpuLoraOptimizerState::new(&self.ctx, &self.config, self.lora_rank)
    }

    /// LoRA optimizer step: update A_q, B_q, A_v, B_v and norm weights using AdamW.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn lora_optimizer_step(
        &mut self,
        state: &mut GpuLoraOptimizerState,
        step: u32,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        stream: &CudaStream,
        grad_lora: &CudaLoraGradWorkspace,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let q_dim = self.config.q_dim();
        let kv = self.config.num_kv_heads * self.config.head_dim();
        let r = self.lora_rank;

        // AdamW step for each LoRA weight
        if let Some(ref mut a_q) = self.lora_a_q {
            adamw_step_cuda(
                a_q,
                &grad_lora.grad_lora_a_q,
                &mut state.m_lora_a_q,
                &mut state.v_lora_a_q,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                saturating_u32(h * r),
                stream,
            )?;
        }
        if let Some(ref mut b_q) = self.lora_b_q {
            adamw_step_cuda(
                b_q,
                &grad_lora.grad_lora_b_q,
                &mut state.m_lora_b_q,
                &mut state.v_lora_b_q,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                saturating_u32(r * q_dim),
                stream,
            )?;
        }
        if let Some(ref mut a_v) = self.lora_a_v {
            adamw_step_cuda(
                a_v,
                &grad_lora.grad_lora_a_v,
                &mut state.m_lora_a_v,
                &mut state.v_lora_a_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                saturating_u32(h * r),
                stream,
            )?;
        }
        if let Some(ref mut b_v) = self.lora_b_v {
            adamw_step_cuda(
                b_v,
                &grad_lora.grad_lora_b_v,
                &mut state.m_lora_b_v,
                &mut state.v_lora_b_v,
                lr,
                beta1,
                beta2,
                eps,
                weight_decay,
                step,
                saturating_u32(r * kv),
                stream,
            )?;
        }

        // AdamW step for norm weights
        adamw_step_cuda(
            &mut self.input_norm_weight,
            &grad_lora.grad_input_norm,
            &mut state.m_input_norm,
            &mut state.v_input_norm,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            saturating_u32(h),
            stream,
        )?;
        adamw_step_cuda(
            &mut self.post_attn_norm_weight,
            &grad_lora.grad_post_attn_norm,
            &mut state.m_post_attn_norm,
            &mut state.v_post_attn_norm,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
            saturating_u32(h),
            stream,
        )?;

        Ok(())
    }

    /// Download LoRA weights from GPU to CPU for checkpoint saving.
    ///
    /// Returns (A_q, B_q, A_v, B_v) as flat f32 vectors.
    /// B matrices are returned WITH the baked-in scale (caller can divide by lora_scale
    /// if they need the unscaled version).
    pub fn download_lora_weights(&self) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
        let download = |buf: &GpuBuffer<f32>| -> Result<Vec<f32>> {
            let mut host = vec![0.0f32; buf.len()];
            buf.copy_to_host(&mut host).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "LoRA weight download failed: {e}"
                ))
            })?;
            Ok(host)
        };
        let a_q = self.lora_a_q.as_ref().map(&download).transpose()?.unwrap_or_default();
        let b_q = self.lora_b_q.as_ref().map(&download).transpose()?.unwrap_or_default();
        let a_v = self.lora_a_v.as_ref().map(&download).transpose()?.unwrap_or_default();
        let b_v = self.lora_b_v.as_ref().map(&download).transpose()?.unwrap_or_default();
        Ok((a_q, b_q, a_v, b_v))
    }

    /// Upload LoRA weights from CPU to GPU for checkpoint resume (ENT-276).
    ///
    /// Overwrites the current LoRA adapter buffers with trained weights
    /// restored from a checkpoint. Call after `new()` to replace the fresh
    /// random init with previously trained adapters.
    pub fn upload_lora_weights(
        &mut self,
        a_q: &[f32],
        b_q: &[f32],
        a_v: &[f32],
        b_v: &[f32],
    ) -> Result<()> {
        let upload = |buf: &mut GpuBuffer<f32>, data: &[f32], name: &str| -> Result<()> {
            if data.len() != buf.len() {
                return Err(crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(
                    format!(
                        "LoRA {name} size mismatch: checkpoint has {} but GPU buffer expects {}",
                        data.len(),
                        buf.len()
                    ),
                ));
            }
            buf.copy_from_host(data).map_err(|e| {
                crate::autograd::cuda_tensor::CudaTensorError::TransferFailed(format!(
                    "LoRA {name} upload failed: {e}"
                ))
            })
        };
        if let Some(ref mut buf) = self.lora_a_q {
            upload(buf, a_q, "a_q")?;
        }
        if let Some(ref mut buf) = self.lora_b_q {
            upload(buf, b_q, "b_q")?;
        }
        if let Some(ref mut buf) = self.lora_a_v {
            upload(buf, a_v, "a_v")?;
        }
        if let Some(ref mut buf) = self.lora_b_v {
            upload(buf, b_v, "b_v")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_cuda_block_compiles() {
        // Basic compilation test
        #[cfg(feature = "cuda")]
        {
            use super::*;
            let _ = std::mem::size_of::<CudaTransformerBlock>();
            let _ = std::mem::size_of::<CudaNf4TransformerBlock>();
        }
    }
}
