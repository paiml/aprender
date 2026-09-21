//! PMAT-3477 / aprender#3090: the Qwen3.5 Gated `DeltaNet` layer on the GPU.
//!
//! [`Qwen35CudaModel`] is built from the CPU [`Qwen35Model`] plus a
//! [`CudaExecutor`]: every `DeltaNet` layer's quantized projections go into the
//! executor's cached-weight store and its f32 vectors into device buffers, and
//! [`Qwen35CudaModel::forward_deltanet_layer`] reproduces
//! `Qwen35Model::forward_deltanet` operation for operation on a device-resident
//! hidden state.
//!
//! The CPU function is the specification. Op order, verbatim:
//!
//! ```text
//! rms_norm -> attn_qkv GEMV -> causal_conv1d(+SiLU) -> split q|k|v
//!   -> per-head L2 on q,k -> ssm_alpha/ssm_beta GEMV -> dt/beta gates
//!   -> attn_gate GEMV -> delta rule -> gated rmsnorm -> ssm_out GEMV
//!   -> residual -> post_attention_norm -> SwiGLU FFN -> residual
//! ```
//!
//! The interleaved full-attention layers run here too (PMAT-3477 phase 2), from
//! the same specification — `Qwen35Model::forward_attention`:
//!
//! ```text
//! rms_norm -> attn_q|attn_k|attn_v GEMV -> split q|gate -> per-head RMSNorm on
//!   q and k -> partial NEOX rope on q and k -> KV append at `position`
//!   -> GQA decode attention -> sigmoid gate -> attn_output GEMV -> residual
//!   -> post_attention_norm -> SwiGLU FFN -> residual
//! ```
//!
//! and [`Qwen35CudaModel::forward_single`] runs a whole token through both layer
//! kinds, the output norm and the `lm_head`.
//!
//! Every state buffer is sized from the config
//! (`num_v_heads * head_v_dim * head_k_dim`, `conv_dim * (conv_kernel - 1)`,
//! `max_seq_len * num_kv_heads * head_dim`) — never from a constant.

use super::{OwnedQuantizedTensor, RealizarError, Result};
use crate::cuda::types::WeightQuantType;
use crate::cuda::CudaExecutor;
use crate::gguf::forward_qwen35::{Qwen35Model, Qwen35OwnedLayer};
use trueno_gpu::driver::GpuBuffer;

/// Positions the device KV cache holds unless a caller asks for more.
///
/// The CPU model takes `max_seq_len` at `new_state`; this model takes it at
/// build time because the caches are allocated with the weights.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// A quantized projection already resident in the executor's weight cache.
struct CudaQuantWeight {
    /// Device pointer into the executor's quantized weight cache.
    ptr: u64,
    /// The GEMV kernel family this tensor's GGML type binds to.
    qtype: WeightQuantType,
    /// Output width (`out_dim`).
    n: u32,
    /// Input width (`in_dim`).
    k: u32,
}

/// One Gated `DeltaNet` layer's device-resident weights.
struct CudaDeltaNetLayer {
    attn_norm: GpuBuffer<f32>,
    conv1d_weight: GpuBuffer<f32>,
    ssm_a: GpuBuffer<f32>,
    ssm_dt_bias: GpuBuffer<f32>,
    ssm_norm_weight: GpuBuffer<f32>,
    post_attention_norm: GpuBuffer<f32>,
    attn_qkv: CudaQuantWeight,
    ssm_alpha: CudaQuantWeight,
    ssm_beta: CudaQuantWeight,
    attn_gate: CudaQuantWeight,
    ssm_out: CudaQuantWeight,
    ffn_gate: CudaQuantWeight,
    ffn_up: CudaQuantWeight,
    ffn_down: CudaQuantWeight,
}

/// One full-attention layer's device-resident weights.
struct CudaAttentionLayer {
    attn_norm: GpuBuffer<f32>,
    attn_q_norm: GpuBuffer<f32>,
    attn_k_norm: GpuBuffer<f32>,
    post_attention_norm: GpuBuffer<f32>,
    attn_q: CudaQuantWeight,
    attn_k: CudaQuantWeight,
    attn_v: CudaQuantWeight,
    attn_output: CudaQuantWeight,
    ffn_gate: CudaQuantWeight,
    ffn_up: CudaQuantWeight,
    ffn_down: CudaQuantWeight,
}

/// One resident layer, of either kind — the GPU mirror of `Qwen35OwnedLayer`.
enum CudaLayer {
    DeltaNet(Box<CudaDeltaNetLayer>),
    Attention(Box<CudaAttentionLayer>),
}

/// Per-sequence Qwen3.5 decode state on the device: one causal-conv window and
/// one recurrent state per Gated `DeltaNet` layer, one K/V cache per
/// full-attention layer, all sized from the config.
pub struct Qwen35CudaState {
    conv: Vec<GpuBuffer<f32>>,
    ssm: Vec<GpuBuffer<f32>>,
    /// `Some((k, v))` for a full-attention layer, `None` for a `DeltaNet` one.
    /// Each is `[max_seq_len][num_kv_heads * head_dim]`, the CPU KV cache's
    /// layout exactly.
    kv: Vec<Option<(GpuBuffer<f32>, GpuBuffer<f32>)>>,
    /// `conv_dim * (conv_kernel - 1)`.
    conv_len: usize,
    /// `num_v_heads * head_v_dim * head_k_dim`.
    ssm_len: usize,
    /// `num_kv_heads * head_dim` — one KV cache row.
    kv_row: usize,
    /// Positions the KV cache can hold.
    max_seq_len: usize,
    /// KV rows written so far (the CPU cache's `seq_len` after `advance`).
    kv_len: usize,
}

impl Qwen35CudaState {
    /// Elements in one layer's causal-conv window.
    #[must_use]
    pub const fn conv_len(&self) -> usize {
        self.conv_len
    }

    /// Elements in one layer's recurrent state.
    #[must_use]
    pub const fn ssm_len(&self) -> usize {
        self.ssm_len
    }

    /// Elements in one KV cache row (`num_kv_heads * head_dim`).
    #[must_use]
    pub const fn kv_row(&self) -> usize {
        self.kv_row
    }

    /// Positions the KV cache can hold.
    #[must_use]
    pub const fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// KV rows written so far.
    #[must_use]
    pub const fn kv_len(&self) -> usize {
        self.kv_len
    }

    /// A placeholder holding no device memory.
    ///
    /// Only ever alive inside [`Qwen35CudaModel::with_own_state`], which swaps
    /// the real state back before returning — the layer bodies take the state as
    /// an argument, so the model cannot lend it to itself.
    const fn detached() -> Self {
        Self {
            conv: Vec::new(),
            ssm: Vec::new(),
            kv: Vec::new(),
            conv_len: 0,
            ssm_len: 0,
            kv_row: 0,
            max_seq_len: 0,
            kv_len: 0,
        }
    }
}

/// Scratch device buffers for one `DeltaNet` layer step, allocated once.
struct Qwen35CudaScratch {
    normed: GpuBuffer<f32>,
    conv_in: GpuBuffer<f32>,
    conv_out: GpuBuffer<f32>,
    alpha_raw: GpuBuffer<f32>,
    beta_raw: GpuBuffer<f32>,
    dt: GpuBuffer<f32>,
    beta: GpuBuffer<f32>,
    gate: GpuBuffer<f32>,
    out_h: GpuBuffer<f32>,
    ssm_out_in: GpuBuffer<f32>,
    ssm_out: GpuBuffer<f32>,
    post_normed: GpuBuffer<f32>,
    ffn_gate: GpuBuffer<f32>,
    ffn_up: GpuBuffer<f32>,
    ffn_act: GpuBuffer<f32>,
    ffn_down: GpuBuffer<f32>,
}

/// Scratch device buffers for one full-attention layer step, allocated once.
///
/// `post_normed` and the FFN buffers are shared with [`Qwen35CudaScratch`] —
/// the two layer kinds have the same FFN.
struct Qwen35AttnScratch {
    /// The joint `[q | gate]` projection, `[num_heads * 2 * head_dim]`.
    q_full: GpuBuffer<f32>,
    /// `q` out of the split, before the per-head norm.
    q: GpuBuffer<f32>,
    /// `q` after the per-head norm (and rotated in place).
    q_normed: GpuBuffer<f32>,
    /// `gate` out of the split.
    gate: GpuBuffer<f32>,
    /// `k` out of the GEMV, before the per-head norm (which writes the cache).
    k_raw: GpuBuffer<f32>,
    /// The attention output before the output projection.
    attn_out_in: GpuBuffer<f32>,
    /// The output projection's result.
    attn_out: GpuBuffer<f32>,
}

/// The shapes the layer forward reads, all derived from the model config.
#[derive(Debug, Clone, Copy)]
struct Qwen35CudaDims {
    hidden_dim: u32,
    intermediate_dim: u32,
    conv_dim: u32,
    k_dim: u32,
    v_dim: u32,
    head_k_dim: u32,
    num_k_heads: u32,
    head_v_dim: u32,
    num_v_heads: u32,
    conv_kernel: u32,
    eps: f32,
    /// `config.num_heads` — the full-attention query heads.
    num_heads: u32,
    /// `config.num_kv_heads`.
    num_kv_heads: u32,
    /// `attn_q_norm.len()` (256 for the 0.8B file), NOT `hidden_dim / num_heads`.
    attn_head_dim: u32,
    /// `2 * sum(rope.dimension_sections)`.
    n_rot: u32,
    /// `freq_base.powf(-2.0 / n_rot)`, computed on the HOST exactly as the CPU
    /// reference does — see `gdn_partial_neox_rope_into`.
    theta_scale: f32,
    vocab_size: u32,
}

/// Qwen3.5's Gated `DeltaNet` block, resident on one CUDA device (#3090).
///
/// Built from the CPU [`Qwen35Model`], which stays the specification and the
/// parity reference.
pub struct Qwen35CudaModel<'a> {
    model: &'a Qwen35Model<'a>,
    executor: CudaExecutor,
    layers: Vec<CudaLayer>,
    state: Qwen35CudaState,
    scratch: Qwen35CudaScratch,
    attn_scratch: Qwen35AttnScratch,
    /// The final `output_norm` gamma, device-resident — `base.output_norm_weight()`.
    output_norm: GpuBuffer<f32>,
    /// The `lm_head` (`output.weight`) in the executor's quantized weight cache.
    lm_head: CudaQuantWeight,
    /// `[hidden_dim]`, the output norm's result. Feeds the `lm_head` GEMV and is
    /// never read by the host.
    out_normed: GpuBuffer<f32>,
    /// `[vocab_size]`, the logits — the ONE buffer a token's forward downloads.
    logits_buf: GpuBuffer<f32>,
    dims: Qwen35CudaDims,
    /// Positions the device KV caches hold.
    max_seq_len: usize,
}

/// Map a GPU error into the crate error type with the operation that raised it.
fn gpu_err(operation: &str, e: &trueno_gpu::GpuError) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: operation.to_string(),
        reason: format!("{e}"),
    }
}

impl<'a> Qwen35CudaModel<'a> {
    /// Upload one quantized projection into the executor's cache and bind its
    /// GEMV kernel family.
    fn upload_quant(
        executor: &mut CudaExecutor,
        name: &str,
        tensor: &OwnedQuantizedTensor,
    ) -> Result<CudaQuantWeight> {
        let qtype = WeightQuantType::from_ggml_type(tensor.qtype).ok_or_else(|| {
            RealizarError::UnsupportedOperation {
                operation: "qwen35_cuda_upload".to_string(),
                reason: format!(
                    "'{name}': GGML type {} has no verified GPU GEMV kernel",
                    tensor.qtype
                ),
            }
        })?;
        if tensor.data.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "qwen35_cuda_upload".to_string(),
                reason: format!("'{name}': tensor data is empty (it lay outside the file)"),
            });
        }
        executor
            .load_quantized_weights_with_type(name, &tensor.data, tensor.qtype)
            .map_err(|e| gpu_err("qwen35_cuda_upload", &e))?;
        let ptr = executor
            .get_quantized_weight_ptr(name)
            .map_err(|e| gpu_err("qwen35_cuda_upload", &e))?;
        Ok(CudaQuantWeight {
            ptr,
            qtype,
            n: u32::try_from(tensor.out_dim).unwrap_or(0),
            k: u32::try_from(tensor.in_dim).unwrap_or(0),
        })
    }

    /// Upload an f32 vector as a device buffer.
    fn upload_f32(executor: &CudaExecutor, name: &str, v: &[f32]) -> Result<GpuBuffer<f32>> {
        if v.is_empty() {
            return Err(RealizarError::UnsupportedOperation {
                operation: "qwen35_cuda_upload".to_string(),
                reason: format!("'{name}': f32 vector is empty"),
            });
        }
        GpuBuffer::from_host(executor.context(), v).map_err(|e| gpu_err("qwen35_cuda_upload", &e))
    }

    /// Zero-filled device buffer of `len` f32.
    fn zeros(executor: &CudaExecutor, len: usize) -> Result<GpuBuffer<f32>> {
        GpuBuffer::from_host(executor.context(), &vec![0.0f32; len])
            .map_err(|e| gpu_err("qwen35_cuda_alloc", &e))
    }

    /// The shapes, read from the model — never hard-coded.
    fn dims_of(model: &Qwen35Model<'_>) -> Qwen35CudaDims {
        let k_dim = model.head_k_dim * model.num_k_heads;
        let v_dim = model.head_v_dim * model.num_v_heads;
        let n_rot = 2 * model.rope_sections.iter().sum::<usize>();
        // The CPU computes theta_scale once per call with f32::powf; the kernel
        // must be given THAT value, not one derived on the device.
        let theta_scale = trueno_gpu::kernels::gdn::PartialNeoxRopeKernel::new(
            model.base.config.num_heads as u32,
            model.head_dim as u32,
            n_rot as u32,
        )
        .theta_scale(model.base.config.rope_theta);
        Qwen35CudaDims {
            num_heads: model.base.config.num_heads as u32,
            num_kv_heads: model.base.config.num_kv_heads as u32,
            attn_head_dim: model.head_dim as u32,
            n_rot: n_rot as u32,
            theta_scale,
            vocab_size: model.base.config.vocab_size as u32,
            hidden_dim: model.base.config.hidden_dim as u32,
            intermediate_dim: model.base.config.intermediate_dim as u32,
            conv_dim: (k_dim * 2 + v_dim) as u32,
            k_dim: k_dim as u32,
            v_dim: v_dim as u32,
            head_k_dim: model.head_k_dim as u32,
            num_k_heads: model.num_k_heads as u32,
            head_v_dim: model.head_v_dim as u32,
            num_v_heads: model.num_v_heads as u32,
            conv_kernel: model.conv_kernel as u32,
            eps: model.base.config.eps,
        }
    }

    /// The delta rule maps value head `h` onto key head `h % num_k_heads`, so the
    /// ONLY shape it cannot serve is one whose value heads are not a whole number
    /// of key-head groups (PMAT-3477, #3346/#3510).
    ///
    /// This used to refuse every `num_k_heads != num_v_heads` file outright, which
    /// is what kept Qwen3.5-4B/9B (32 value heads against 16 key heads) and -27B
    /// (48 against 16) off the GPU. `head_k_dim != head_v_dim` is likewise no
    /// longer a refusal: the kernel sizes the state row from the key dim and the
    /// row count from the value dim.
    /// Renamed from `check_head_symmetry`: it no longer asks for symmetry, and a
    /// predicate whose name outlives what it tests is the next reader's wrong
    /// diagnosis.
    fn check_head_grouping(d: Qwen35CudaDims) -> Result<()> {
        if d.num_k_heads > 0 && d.num_v_heads % d.num_k_heads == 0 {
            return Ok(());
        }
        Err(RealizarError::UnsupportedOperation {
            operation: "qwen35_cuda_deltanet".to_string(),
            reason: format!(
                "the delta rule maps value head h onto key head h % num_k_heads, so \
                 num_v_heads must be a positive multiple of num_k_heads: num_k_heads {} \
                 does not divide num_v_heads {}",
                d.num_k_heads, d.num_v_heads
            ),
        })
    }

    /// Upload one `DeltaNet` layer's tensors.
    fn build_layer(
        executor: &mut CudaExecutor,
        il: usize,
        d: &crate::gguf::forward_qwen35::Qwen35OwnedDeltaNetLayer,
    ) -> Result<CudaDeltaNetLayer> {
        let q = |e: &mut CudaExecutor, suffix: &str, t: &OwnedQuantizedTensor| {
            Self::upload_quant(e, &format!("qwen35.blk.{il}.{suffix}"), t)
        };
        Ok(CudaDeltaNetLayer {
            attn_norm: Self::upload_f32(executor, "attn_norm", &d.attn_norm)?,
            conv1d_weight: Self::upload_f32(executor, "ssm_conv1d", &d.ssm_conv1d_weight)?,
            ssm_a: Self::upload_f32(executor, "ssm_a", &d.ssm_a)?,
            ssm_dt_bias: Self::upload_f32(executor, "ssm_dt_bias", &d.ssm_dt_bias)?,
            ssm_norm_weight: Self::upload_f32(executor, "ssm_norm", &d.ssm_norm_weight)?,
            post_attention_norm: Self::upload_f32(
                executor,
                "post_attention_norm",
                &d.post_attention_norm,
            )?,
            attn_qkv: q(executor, "attn_qkv.weight", &d.attn_qkv)?,
            ssm_alpha: q(executor, "ssm_alpha.weight", &d.ssm_alpha)?,
            ssm_beta: q(executor, "ssm_beta.weight", &d.ssm_beta)?,
            attn_gate: q(executor, "attn_gate.weight", &d.attn_gate)?,
            ssm_out: q(executor, "ssm_out.weight", &d.ssm_out)?,
            ffn_gate: q(executor, "ffn_gate.weight", &d.ffn_gate)?,
            ffn_up: q(executor, "ffn_up.weight", &d.ffn_up)?,
            ffn_down: q(executor, "ffn_down.weight", &d.ffn_down)?,
        })
    }

    /// Upload one full-attention layer's tensors.
    fn build_attention_layer(
        executor: &mut CudaExecutor,
        il: usize,
        a: &crate::gguf::forward_qwen35::Qwen35OwnedAttentionLayer,
    ) -> Result<CudaAttentionLayer> {
        let q = |e: &mut CudaExecutor, suffix: &str, t: &OwnedQuantizedTensor| {
            Self::upload_quant(e, &format!("qwen35.blk.{il}.{suffix}"), t)
        };
        Ok(CudaAttentionLayer {
            attn_norm: Self::upload_f32(executor, "attn_norm", &a.attn_norm)?,
            attn_q_norm: Self::upload_f32(executor, "attn_q_norm", &a.attn_q_norm)?,
            attn_k_norm: Self::upload_f32(executor, "attn_k_norm", &a.attn_k_norm)?,
            post_attention_norm: Self::upload_f32(
                executor,
                "post_attention_norm",
                &a.post_attention_norm,
            )?,
            attn_q: q(executor, "attn_q.weight", &a.attn_q)?,
            attn_k: q(executor, "attn_k.weight", &a.attn_k)?,
            attn_v: q(executor, "attn_v.weight", &a.attn_v)?,
            attn_output: q(executor, "attn_output.weight", &a.attn_output)?,
            ffn_gate: q(executor, "ffn_gate.weight", &a.ffn_gate)?,
            ffn_up: q(executor, "ffn_up.weight", &a.ffn_up)?,
            ffn_down: q(executor, "ffn_down.weight", &a.ffn_down)?,
        })
    }

    /// Allocate the scratch buffers one full-attention layer step needs.
    fn build_attn_scratch(executor: &CudaExecutor, d: Qwen35CudaDims) -> Result<Qwen35AttnScratch> {
        let q_dim = (d.num_heads * d.attn_head_dim) as usize;
        let kv_dim = (d.num_kv_heads * d.attn_head_dim) as usize;
        Ok(Qwen35AttnScratch {
            q_full: Self::zeros(executor, q_dim * 2)?,
            q: Self::zeros(executor, q_dim)?,
            q_normed: Self::zeros(executor, q_dim)?,
            gate: Self::zeros(executor, q_dim)?,
            k_raw: Self::zeros(executor, kv_dim)?,
            attn_out_in: Self::zeros(executor, q_dim)?,
            attn_out: Self::zeros(executor, d.hidden_dim as usize)?,
        })
    }

    /// Allocate the scratch buffers one layer step needs.
    fn build_scratch(executor: &CudaExecutor, d: Qwen35CudaDims) -> Result<Qwen35CudaScratch> {
        let hidden = d.hidden_dim as usize;
        let inter = d.intermediate_dim as usize;
        let conv = d.conv_dim as usize;
        let v = d.v_dim as usize;
        let nv = d.num_v_heads as usize;
        Ok(Qwen35CudaScratch {
            normed: Self::zeros(executor, hidden)?,
            conv_in: Self::zeros(executor, conv)?,
            conv_out: Self::zeros(executor, conv)?,
            alpha_raw: Self::zeros(executor, nv)?,
            beta_raw: Self::zeros(executor, nv)?,
            dt: Self::zeros(executor, nv)?,
            beta: Self::zeros(executor, nv)?,
            gate: Self::zeros(executor, v)?,
            out_h: Self::zeros(executor, v)?,
            ssm_out_in: Self::zeros(executor, v)?,
            ssm_out: Self::zeros(executor, hidden)?,
            post_normed: Self::zeros(executor, hidden)?,
            ffn_gate: Self::zeros(executor, inter)?,
            ffn_up: Self::zeros(executor, inter)?,
            ffn_act: Self::zeros(executor, inter)?,
            ffn_down: Self::zeros(executor, hidden)?,
        })
    }

    /// Build the GPU model from the CPU model and an executor.
    ///
    /// Uploads every `DeltaNet` layer's quantized projections into the
    /// executor's cache and its f32 vectors into device buffers, and allocates
    /// the per-layer state from the config.
    ///
    /// # Errors
    /// A tensor whose GGML type has no GPU GEMV kernel, an empty tensor, a
    /// key/value head mismatch the delta-rule kernel cannot index, or any CUDA
    /// allocation failure.
    pub fn new(model: &'a Qwen35Model<'a>, executor: CudaExecutor) -> Result<Self> {
        Self::with_max_seq_len(model, executor, DEFAULT_MAX_SEQ_LEN)
    }

    /// Build the GPU model with room for `max_seq_len` KV cache positions.
    ///
    /// # Errors
    /// As [`Self::new`], plus a `max_seq_len` of zero.
    pub fn with_max_seq_len(
        model: &'a Qwen35Model<'a>,
        mut executor: CudaExecutor,
        max_seq_len: usize,
    ) -> Result<Self> {
        let dims = Self::dims_of(model);
        Self::check_head_grouping(dims)?;
        if max_seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "qwen35_cuda: max_seq_len must be at least 1".to_string(),
            });
        }

        // PRODUCTION DEFAULT, not a test affordance (PMAT-3477 / #3090): this
        // architecture runs the FLOAT Q4_K/Q6_K GEMV kernels, never the DP4A
        // ones `GpuProfile::detect` picks for a dense decode. The DP4A kernels
        // quantize the ACTIVATION to int8, and Qwen3.5 feeds its projections
        // straight into a recurrence, which compounds that error instead of
        // absorbing it. Measured on the real 0.8B file: with the float variants
        // pinned, a whole DeltaNet layer's output is 0.000 relative from a
        // second float run and inside the layer budget against the CPU; with
        // `HwDp4a` the DeltaNet-only path lands **1.656 relative** away, and the
        // end-to-end argmax is garbage — a wrong token at position 0, not a
        // rounding difference. The falsifier lives in the tests file
        // (`qwen35_cuda_dp4a_gemv_is_catastrophic_through_the_recurrence`).
        //
        // Recovering the DP4A throughput for this architecture (a higher-
        // precision activation quantization, or DP4A only on the layers that do
        // not feed the recurrence) is the DP4A-through-recurrence ticket,
        // 0.69.0. Until it lands, correctness is not optional here.
        Self::pin_float_gemv(&mut executor.gpu_profile);

        let mut layers = Vec::with_capacity(model.layers.len());
        for (il, layer) in model.layers.iter().enumerate() {
            layers.push(match layer {
                Qwen35OwnedLayer::DeltaNet(d) => {
                    CudaLayer::DeltaNet(Box::new(Self::build_layer(&mut executor, il, d)?))
                },
                Qwen35OwnedLayer::Attention(a) => CudaLayer::Attention(Box::new(
                    Self::build_attention_layer(&mut executor, il, a)?,
                )),
            });
        }

        // The tail (output norm + lm_head) runs HERE, on the device buffer the
        // layers left the hidden state in — not through `hidden_to_logits`,
        // which would take the hidden state down to the host and put it back.
        // The gamma is ours; the lm_head goes in the executor's weight cache
        // under the name its own path also expects.
        let output_norm =
            Self::upload_f32(&executor, "output_norm", model.base.output_norm_weight())?;
        executor
            .preload_output_norm(model.base.output_norm_weight())
            .map_err(|e| gpu_err("qwen35_cuda_upload", &e))?;
        let lm_head =
            Self::upload_quant(&mut executor, "output.weight", model.base.lm_head_weight())?;
        if lm_head.n != dims.vocab_size || lm_head.k != dims.hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda: the lm_head is {}x{}, the config says vocab {} x hidden {}",
                    lm_head.n, lm_head.k, dims.vocab_size, dims.hidden_dim
                ),
            });
        }

        // The DP4A GEMV kernels quantize the activation into
        // `workspace.q8_activation_buf`, which `init_workspace` sizes from
        // `max(hidden_dim, intermediate_dim, q_dim)`. Our widest GEMV input is
        // `v_dim` (the ssm_out projection) or `num_heads * attn_head_dim` (the
        // attn_output projection), neither of which is one of those — pass the
        // widest as the "hidden" width so the buffer covers every input this
        // model feeds a kernel. Without this the Q6_K path panics on an
        // uninitialized buffer.
        let widest_gemv_input = dims
            .hidden_dim
            .max(dims.v_dim)
            .max(dims.num_heads * dims.attn_head_dim) as usize;
        executor
            .init_workspace(widest_gemv_input, dims.intermediate_dim as usize)
            .map_err(|e| gpu_err("qwen35_cuda_workspace", &e))?;

        let scratch = Self::build_scratch(&executor, dims)?;
        let attn_scratch = Self::build_attn_scratch(&executor, dims)?;
        let out_normed = Self::zeros(&executor, dims.hidden_dim as usize)?;
        let logits_buf = Self::zeros(&executor, dims.vocab_size as usize)?;
        // #3596: the model's OWN state serves only the single-layer handles
        // (`forward_attention_layer`, `upload_attention_kv`, …), never a generation —
        // `qwen35_gpu_decode` and the F2 probe each allocate theirs. Sizing it to the
        // request's `max_seq_len` put a second full-length KV on the device next to the
        // decode state: 2 × 16 GiB on the 9B at 262,144 positions.
        let state = Self::build_state(
            &executor,
            &layers,
            dims,
            max_seq_len.min(DEFAULT_MAX_SEQ_LEN),
        )?;
        Ok(Self {
            model,
            executor,
            layers,
            state,
            scratch,
            attn_scratch,
            output_norm,
            lm_head,
            out_normed,
            logits_buf,
            dims,
            max_seq_len,
        })
    }

    /// Allocate a fresh decode state: every layer's conv window and recurrent
    /// state, and a K/V cache for each full-attention layer.
    fn build_state(
        executor: &CudaExecutor,
        layers: &[CudaLayer],
        dims: Qwen35CudaDims,
        max_seq_len: usize,
    ) -> Result<Qwen35CudaState> {
        let conv_len = (dims.conv_dim * (dims.conv_kernel - 1)) as usize;
        // The recurrent state of one value head is [head_v_dim rows x head_k_dim],
        // laid out `s[j * head_k_dim + i] == S[i][j]` — the CPU reference's own
        // layout. Sizing it from head_v_dim twice was only right because every
        // file so far ships head_k_dim == head_v_dim (PMAT-3477).
        let ssm_len = (dims.num_v_heads * dims.head_v_dim * dims.head_k_dim) as usize;
        let kv_row = (dims.num_kv_heads * dims.attn_head_dim) as usize;
        let mut conv = Vec::with_capacity(layers.len());
        let mut ssm = Vec::with_capacity(layers.len());
        let mut kv = Vec::with_capacity(layers.len());
        for layer in layers {
            conv.push(Self::zeros(executor, conv_len)?);
            ssm.push(Self::zeros(executor, ssm_len)?);
            kv.push(match layer {
                CudaLayer::DeltaNet(_) => None,
                CudaLayer::Attention(_) => Some((
                    Self::zeros(executor, kv_row * max_seq_len)?,
                    Self::zeros(executor, kv_row * max_seq_len)?,
                )),
            });
        }
        Ok(Qwen35CudaState {
            conv,
            ssm,
            kv,
            conv_len,
            ssm_len,
            kv_row,
            max_seq_len,
            kv_len: 0,
        })
    }

    /// A fresh decode state, independent of the model's own.
    ///
    /// The GPU analogue of `Qwen35Model::new_state`; `max_seq_len` is the one
    /// this model was built with.
    ///
    /// # Errors
    /// Any CUDA allocation failure.
    pub fn new_state(&self) -> Result<Qwen35CudaState> {
        Self::build_state(&self.executor, &self.layers, self.dims, self.max_seq_len)
    }

    /// A fresh decode state holding `max_seq_len` positions — for a caller that
    /// needs fewer than the model was built for (#3596: the F2 probe needs its
    /// probe plus one decode step, not the whole request's KV).
    ///
    /// # Errors
    /// A `max_seq_len` of zero, or any CUDA allocation failure.
    pub fn new_state_with_len(&self, max_seq_len: usize) -> Result<Qwen35CudaState> {
        if max_seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "qwen35_cuda: a state must hold at least one position".to_string(),
            });
        }
        Self::build_state(&self.executor, &self.layers, self.dims, max_seq_len)
    }

    /// Run `f` with the model's own state detached.
    ///
    /// The layer bodies take `&mut Qwen35CudaState` so a caller can drive an
    /// external state ([`Self::forward_single`]); a method that uses the
    /// model's own state cannot pass `&mut self.state` and `&mut self` at once,
    /// so it swaps the state out for the call and back after.
    fn with_own_state<R>(&mut self, f: impl FnOnce(&mut Self, &mut Qwen35CudaState) -> R) -> R {
        let mut state = std::mem::replace(&mut self.state, Qwen35CudaState::detached());
        let out = f(self, &mut state);
        self.state = state;
        out
    }

    /// The device state buffers (their configured lengths).
    #[must_use]
    pub const fn state(&self) -> &Qwen35CudaState {
        &self.state
    }

    /// The executor, for a caller that needs to synchronise or inspect it.
    pub fn executor_mut(&mut self) -> &mut CudaExecutor {
        &mut self.executor
    }

    /// Pin the FLOAT (non-DP4A) Q4_K / Q6_K GEMV kernels for this model.
    ///
    /// The CPU reference dequantizes to f32 and accumulates in f32; the DP4A
    /// kernels quantize the activation to int8 first, which is an intentional
    /// and already-measured approximation (the load-time parity gate budgets
    /// cosine 0.9887 for it versus 0.9999 for the float path). Measured here on
    /// the real 0.8B file, the DP4A path puts the `DeltaNet` layer output
    /// 1.1e-2 relative from the CPU while the float path is inside 1e-3 — so a
    /// test that means to measure the Gated `DeltaNet` kernels, and not the GEMV
    /// quantization choice, pins this first.
    ///
    /// This is ALSO what [`Self::with_max_seq_len`] does at build time for
    /// every model of this architecture (see the comment there and
    /// [`Self::gemv_variants`]); the method stays because a test that wants to
    /// say "the float GEMV, explicitly" should be able to, and because a caller
    /// that has re-armed DP4A on the executor can get back to the pinned state.
    pub fn pin_reference_gemv(&mut self) {
        Self::pin_float_gemv(&mut self.executor.gpu_profile);
    }

    /// Set the float (non-DP4A) Q4_K / Q6_K variants on a profile.
    fn pin_float_gemv(profile: &mut crate::cuda::gpu_profile::GpuProfile) {
        profile.q4k = crate::cuda::gpu_profile::Q4kVariant::Mwv;
        profile.q6k = crate::cuda::gpu_profile::Q6kVariant::Mwv;
    }

    /// The Q4_K and Q6_K GEMV variants this model will actually dispatch.
    ///
    /// A freshly built model reports the float pair — see the pinning comment
    /// in [`Self::with_max_seq_len`]. A caller that overrides the executor's
    /// profile afterwards sees its own choice here, which is what the DP4A
    /// falsifier test reads.
    #[must_use]
    pub const fn gemv_variants(
        &self,
    ) -> (
        crate::cuda::gpu_profile::Q4kVariant,
        crate::cuda::gpu_profile::Q6kVariant,
    ) {
        (self.executor.gpu_profile.q4k, self.executor.gpu_profile.q6k)
    }

    /// Read layer `il`'s state back to the host — the parity tests' observation
    /// point. What comes back depends on the layer kind:
    ///
    /// | layer | pair |
    /// |-------|------|
    /// | Gated `DeltaNet` | `(causal-conv window, recurrent state)` |
    /// | full attention | `(K rows written so far, V rows written so far)`, each `kv_len * kv_row` long |
    ///
    /// An attention layer at position 0 of a fresh state therefore returns two
    /// EMPTY vectors, not two zero-filled ones — nothing has been written.
    ///
    /// # Errors
    /// A device-to-host copy failure, or a layer index out of range.
    pub fn download_layer(&mut self, il: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        match self.layers.get(il) {
            Some(CudaLayer::DeltaNet(_)) => {},
            Some(CudaLayer::Attention(_)) => return self.download_attention_layer(il),
            None => {
                return Err(RealizarError::InvalidShape {
                    reason: format!("qwen35_cuda: layer {il} is out of range"),
                })
            },
        }
        self.executor
            .sync_stream()
            .map_err(|e| gpu_err("qwen35_cuda_download", &e))?;
        let mut conv = vec![0.0f32; self.state.conv_len];
        let mut ssm = vec![0.0f32; self.state.ssm_len];
        self.state.conv[il]
            .copy_to_host(&mut conv)
            .map_err(|e| gpu_err("qwen35_cuda_download", &e))?;
        self.state.ssm[il]
            .copy_to_host(&mut ssm)
            .map_err(|e| gpu_err("qwen35_cuda_download", &e))?;
        Ok((conv, ssm))
    }

    /// The K and V rows written so far for full-attention layer `il`.
    fn download_attention_layer(&mut self, il: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.executor
            .sync_stream()
            .map_err(|e| gpu_err("qwen35_cuda_download", &e))?;
        let n = self.state.kv_len * self.state.kv_row;
        let (kc, vc) = self.state.kv[il]
            .as_ref()
            .expect("an attention layer owns a KV cache");
        let mut k = vec![0.0f32; n];
        let mut v = vec![0.0f32; n];
        if n > 0 {
            kc.copy_to_host_at(&mut k, 0)
                .map_err(|e| gpu_err("qwen35_cuda_download", &e))?;
            vc.copy_to_host_at(&mut v, 0)
                .map_err(|e| gpu_err("qwen35_cuda_download", &e))?;
        }
        Ok((k, v))
    }

    /// Write the K and V rows of full-attention layer `il` — teacher forcing
    /// for the parity tests. `k` and `v` are whole rows (`kv_row` each) and set
    /// the state's `kv_len`.
    ///
    /// # Errors
    /// A length that is not a whole number of rows or exceeds `max_seq_len`, a
    /// host-to-device copy failure, or a layer that is not a full-attention one.
    pub fn upload_attention_kv(&mut self, il: usize, k: &[f32], v: &[f32]) -> Result<()> {
        self.require_attention(il)?;
        let row = self.state.kv_row;
        if k.len() != v.len() || k.len() % row != 0 || k.len() > row * self.state.max_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda upload_attention_kv({il}): expected equal K/V lengths that are a \
                     multiple of {row} and at most {}, got {} / {}",
                    row * self.state.max_seq_len,
                    k.len(),
                    v.len()
                ),
            });
        }
        let rows = k.len() / row;
        {
            let (kc, vc) = self.state.kv[il]
                .as_mut()
                .expect("require_attention checked this layer");
            if !k.is_empty() {
                kc.copy_from_host_at(k, 0)
                    .map_err(|e| gpu_err("qwen35_cuda_upload_state", &e))?;
                vc.copy_from_host_at(v, 0)
                    .map_err(|e| gpu_err("qwen35_cuda_upload_state", &e))?;
            }
        }
        self.state.kv_len = rows;
        Ok(())
    }

    /// Write layer `il`'s causal-conv window and recurrent state — teacher
    /// forcing for the parity tests.
    ///
    /// # Errors
    /// A length that does not match the configured state size, a host-to-device
    /// copy failure, or a layer index that is not a `DeltaNet` layer.
    pub fn upload_layer(&mut self, il: usize, conv: &[f32], ssm: &[f32]) -> Result<()> {
        self.require_deltanet(il)?;
        if conv.len() != self.state.conv_len || ssm.len() != self.state.ssm_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda upload_layer({il}): expected conv {} / ssm {}, got {} / {}",
                    self.state.conv_len,
                    self.state.ssm_len,
                    conv.len(),
                    ssm.len()
                ),
            });
        }
        self.state.conv[il]
            .copy_from_host(conv)
            .map_err(|e| gpu_err("qwen35_cuda_upload_state", &e))?;
        self.state.ssm[il]
            .copy_from_host(ssm)
            .map_err(|e| gpu_err("qwen35_cuda_upload_state", &e))?;
        Ok(())
    }

    /// Refuse a layer index that is not a resident `DeltaNet` layer.
    fn require_deltanet(&self, il: usize) -> Result<()> {
        match self.layers.get(il) {
            Some(CudaLayer::DeltaNet(_)) => Ok(()),
            Some(CudaLayer::Attention(_)) => Err(RealizarError::UnsupportedOperation {
                operation: "qwen35_cuda_attention".to_string(),
                reason: format!("layer {il} is a full-attention layer, not a Gated DeltaNet one"),
            }),
            None => Err(RealizarError::InvalidShape {
                reason: format!("qwen35_cuda: layer {il} is out of range"),
            }),
        }
    }

    /// Refuse a layer index that is not a resident full-attention layer.
    fn require_attention(&self, il: usize) -> Result<()> {
        match self.layers.get(il) {
            Some(CudaLayer::Attention(_)) => Ok(()),
            Some(CudaLayer::DeltaNet(_)) => Err(RealizarError::UnsupportedOperation {
                operation: "qwen35_cuda_deltanet".to_string(),
                reason: format!("layer {il} is a Gated DeltaNet layer, not a full-attention one"),
            }),
            None => Err(RealizarError::InvalidShape {
                reason: format!("qwen35_cuda: layer {il} is out of range"),
            }),
        }
    }

    /// Run one full-attention layer in place on a device-resident hidden state,
    /// appending this token's K and V at `position`.
    ///
    /// Mirrors `Qwen35Model::forward_attention` operation for operation, on the
    /// model's own state.
    ///
    /// # Errors
    /// A kernel launch or GEMV dispatch failure, a layer index that is not a
    /// full-attention layer, or a position past the state's `max_seq_len`.
    pub fn forward_attention_layer(
        &mut self,
        il: usize,
        hidden: &GpuBuffer<f32>,
        position: usize,
    ) -> Result<()> {
        self.require_attention(il)?;
        self.with_own_state(|m, s| m.attention_layer(s, il, hidden, position))
    }

    /// The attention body against an explicit state.
    fn attention_layer(
        &mut self,
        state: &mut Qwen35CudaState,
        il: usize,
        hidden: &GpuBuffer<f32>,
        position: usize,
    ) -> Result<()> {
        self.require_attention(il)?;
        if position >= state.max_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda: position {position} is past the KV cache ({} rows)",
                    state.max_seq_len
                ),
            });
        }
        self.attention_layer_inner(state, il, hidden, position)
            .map_err(|e| gpu_err("qwen35_cuda_attention", &e))
    }

    /// The op-for-op attention body; every error here is a GPU error.
    #[allow(clippy::too_many_lines)]
    fn attention_layer_inner(
        &mut self,
        state: &mut Qwen35CudaState,
        il: usize,
        hidden: &GpuBuffer<f32>,
        position: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let CudaLayer::Attention(w) = &self.layers[il] else {
            unreachable!("require_attention checked this layer")
        };
        let s = &self.scratch;
        let a = &self.attn_scratch;
        let ex = &mut self.executor;
        let q_dim = d.num_heads * d.attn_head_dim;
        let kv_dim = d.num_kv_heads * d.attn_head_dim;
        let (k_cache, v_cache) = state.kv[il]
            .as_ref()
            .expect("an attention layer owns a KV cache");

        // This token's row in the KV cache: the GEMVs and the norm write
        // straight into it, which IS the CPU's `kv_cache.append`.
        let k_row = Self::view(
            k_cache,
            u32::try_from(position).unwrap_or(0) * kv_dim,
            kv_dim,
        );
        let v_row = Self::view(
            v_cache,
            u32::try_from(position).unwrap_or(0) * kv_dim,
            kv_dim,
        );

        // rms_norm(hidden, attn_norm)
        ex.rmsnorm_into(hidden, &w.attn_norm, &s.normed, d.hidden_dim, d.eps)?;

        // attn_q . normed -> [q | gate] per head ; attn_k, attn_v -> the cache row
        ex.gemv_dispatch(
            w.attn_q.qtype,
            w.attn_q.ptr,
            &s.normed,
            &a.q_full,
            w.attn_q.n,
            w.attn_q.k,
        )?;
        ex.gemv_dispatch(
            w.attn_k.qtype,
            w.attn_k.ptr,
            &s.normed,
            &a.k_raw,
            w.attn_k.n,
            w.attn_k.k,
        )?;
        ex.gemv_dispatch(
            w.attn_v.qtype,
            w.attn_v.ptr,
            &s.normed,
            &v_row,
            w.attn_v.n,
            w.attn_v.k,
        )?;

        // split [q | gate] per head
        ex.gdn_split_interleaved_into(&a.q_full, &a.q, &a.gate, d.num_heads, d.attn_head_dim)?;

        // per-head RMSNorm on q (num_heads) and k (num_kv_heads); k's result is
        // written into its cache row.
        ex.per_head_rmsnorm_into(
            &a.q,
            &w.attn_q_norm,
            &a.q_normed,
            d.attn_head_dim,
            d.num_heads,
            d.eps,
        )?;
        ex.per_head_rmsnorm_into(
            &a.k_raw,
            &w.attn_k_norm,
            &k_row,
            d.attn_head_dim,
            d.num_kv_heads,
            d.eps,
        )?;

        // partial NEOX rope on q and k, in place, at this position
        let pos32 = u32::try_from(position).unwrap_or(u32::MAX);
        ex.gdn_partial_neox_rope_into(
            &a.q_normed,
            d.num_heads,
            d.attn_head_dim,
            d.n_rot,
            pos32,
            d.theta_scale,
        )?;
        ex.gdn_partial_neox_rope_into(
            &k_row,
            d.num_kv_heads,
            d.attn_head_dim,
            d.n_rot,
            pos32,
            d.theta_scale,
        )?;

        // GQA decode attention over positions 0..=position
        ex.gdn_decode_attention_into(
            &a.q_normed,
            k_cache,
            v_cache,
            &a.attn_out_in,
            d.num_heads,
            d.num_kv_heads,
            d.attn_head_dim,
            pos32 + 1,
        )?;
        std::mem::forget(k_row);
        std::mem::forget(v_row);

        // the output gate, then the output projection and the first residual
        ex.gdn_sigmoid_gate_into(&a.attn_out_in, &a.gate, q_dim)?;
        ex.gemv_dispatch(
            w.attn_output.qtype,
            w.attn_output.ptr,
            &a.attn_out_in,
            &a.attn_out,
            w.attn_output.n,
            w.attn_output.k,
        )?;
        ex.residual_add_into(hidden, &a.attn_out, hidden, d.hidden_dim)?;

        // post_attention_norm -> SwiGLU FFN -> the second residual
        ex.rmsnorm_into(
            hidden,
            &w.post_attention_norm,
            &s.post_normed,
            d.hidden_dim,
            d.eps,
        )?;
        ex.gemv_dispatch(
            w.ffn_gate.qtype,
            w.ffn_gate.ptr,
            &s.post_normed,
            &s.ffn_gate,
            w.ffn_gate.n,
            w.ffn_gate.k,
        )?;
        ex.gemv_dispatch(
            w.ffn_up.qtype,
            w.ffn_up.ptr,
            &s.post_normed,
            &s.ffn_up,
            w.ffn_up.n,
            w.ffn_up.k,
        )?;
        ex.fused_swiglu_into(&s.ffn_gate, &s.ffn_up, &s.ffn_act, d.intermediate_dim)?;
        ex.gemv_dispatch(
            w.ffn_down.qtype,
            w.ffn_down.ptr,
            &s.ffn_act,
            &s.ffn_down,
            w.ffn_down.n,
            w.ffn_down.k,
        )?;
        ex.residual_add_into(hidden, &s.ffn_down, hidden, d.hidden_dim)?;

        // NO sync here (#3090 review). Every op above is enqueued on the one
        // stream this model uses, so the next layer's first kernel is already
        // ordered after this layer's last one; a sync inside the per-token loop
        // only stalls the host. The bookkeeping below is a host-side counter of
        // rows the stream has been ASKED to write, which no device read
        // observes — the syncs that matter are the ones in front of a host read
        // (`download_layer`, `dump_stage`, the logits download).
        //
        // The row is written: the cache now holds `position + 1` rows.
        state.kv_len = state.kv_len.max(position + 1);
        Ok(())
    }

    /// Read one named scratch buffer back to the host.
    ///
    /// The op-by-op observation point: a per-layer L∞ that fails says only THAT
    /// the layer diverged, never WHERE, and the twelve intermediates between
    /// `attn_qkv` and the second residual are what tell the two apart.
    ///
    /// # Panics
    /// On an unknown stage name, or a failed device-to-host copy.
    #[cfg(test)]
    pub(crate) fn dump_stage(&mut self, which: &str) -> Vec<f32> {
        self.executor.sync_stream().expect("sync");
        let s = &self.scratch;
        let a = &self.attn_scratch;
        let buf = match which {
            "q_full" => &a.q_full,
            "q" => &a.q,
            "q_normed" => &a.q_normed,
            "attn_gate" => &a.gate,
            "k_raw" => &a.k_raw,
            "attn_out_in" => &a.attn_out_in,
            "attn_out" => &a.attn_out,
            "normed" => &s.normed,
            "conv_in" => &s.conv_in,
            "conv_out" => &s.conv_out,
            "alpha_raw" => &s.alpha_raw,
            "beta_raw" => &s.beta_raw,
            "dt" => &s.dt,
            "beta" => &s.beta,
            "gate" => &s.gate,
            "out_h" => &s.out_h,
            "ssm_out_in" => &s.ssm_out_in,
            "ssm_out" => &s.ssm_out,
            "post_normed" => &s.post_normed,
            "ffn_gate" => &s.ffn_gate,
            "ffn_up" => &s.ffn_up,
            "ffn_act" => &s.ffn_act,
            "ffn_down" => &s.ffn_down,
            other => panic!("qwen35_cuda: no scratch stage named '{other}'"),
        };
        let mut host = vec![0.0f32; buf.len()];
        buf.copy_to_host(&mut host).expect("download stage");
        host
    }

    /// Run attention layer `il`'s `attn_q` projection on a host input vector and
    /// read the result back — the observation point for the parity-floor test.
    ///
    /// The whole-layer and end-to-end comparisons are bounded from below by the
    /// projection GEMVs, and "is the GPU wrong or is the reference?" is not
    /// answerable from a GPU-vs-CPU number alone. This exposes one GEMV so a
    /// test can put BOTH sides against an exact dequantized f64 reference.
    ///
    /// # Errors
    /// Any allocation, dispatch or transfer failure.
    #[cfg(test)]
    pub(crate) fn attn_q_gemv_of_host_input(&mut self, il: usize, x: &[f32]) -> Result<Vec<f32>> {
        let CudaLayer::Attention(w) = &self.layers[il] else {
            unreachable!("an attention layer")
        };
        let (qtype, ptr, n, k) = (w.attn_q.qtype, w.attn_q.ptr, w.attn_q.n, w.attn_q.k);
        let dev =
            GpuBuffer::from_host(self.executor.context(), x).map_err(|e| gpu_err("diag", &e))?;
        let out = Self::zeros(&self.executor, n as usize)?;
        self.executor
            .gemv_dispatch(qtype, ptr, &dev, &out, n, k)
            .map_err(|e| gpu_err("diag", &e))?;
        self.executor
            .sync_stream()
            .map_err(|e| gpu_err("diag", &e))?;
        let mut host = vec![0.0f32; n as usize];
        out.copy_to_host(&mut host)
            .map_err(|e| gpu_err("diag", &e))?;
        Ok(host)
    }

    /// A non-owning view of `elems` f32 starting `offset` elements into `buf`.
    ///
    /// The delta rule wants q, k and v as three separate pointers; the conv
    /// output holds them contiguously, exactly as the CPU reference slices it.
    /// The returned buffer must be `std::mem::forget`-ed — it does not own the
    /// allocation.
    fn view(buf: &GpuBuffer<f32>, offset: u32, elems: u32) -> GpuBuffer<f32> {
        let ptr = buf.as_ptr() + u64::from(offset) * 4;
        // SAFETY: `offset + elems <= buf.len()` at every call site below
        // (0/k_dim/2*k_dim into a conv_dim-long buffer), and the view is
        // forgotten before it can free memory it does not own.
        unsafe { GpuBuffer::<f32>::from_raw_parts(ptr, elems as usize) }
    }

    /// Run one Gated `DeltaNet` layer in place on a device-resident hidden state.
    ///
    /// Mirrors `Qwen35Model::forward_deltanet` operation for operation. The
    /// layer's conv window and recurrent state are read and written in place, so
    /// a caller that wants a specific starting state sets it with
    /// [`Self::upload_layer`] first.
    ///
    /// # Errors
    /// A kernel launch or GEMV dispatch failure, or a layer index that is not a
    /// `DeltaNet` layer.
    pub fn forward_deltanet_layer(&mut self, il: usize, hidden: &GpuBuffer<f32>) -> Result<()> {
        self.require_deltanet(il)?;
        self.with_own_state(|m, s| m.deltanet_layer(s, il, hidden))
    }

    /// The `DeltaNet` body against an explicit state.
    fn deltanet_layer(
        &mut self,
        state: &mut Qwen35CudaState,
        il: usize,
        hidden: &GpuBuffer<f32>,
    ) -> Result<()> {
        self.require_deltanet(il)?;
        self.deltanet_layer_inner(state, il, hidden)
            .map_err(|e| gpu_err("qwen35_cuda_deltanet", &e))
    }

    /// The op-for-op body; every error here is a GPU error.
    #[allow(clippy::too_many_lines)]
    fn deltanet_layer_inner(
        &mut self,
        state: &Qwen35CudaState,
        il: usize,
        hidden: &GpuBuffer<f32>,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let CudaLayer::DeltaNet(w) = &self.layers[il] else {
            unreachable!("require_deltanet checked this layer")
        };
        let s = &self.scratch;
        let ex = &mut self.executor;

        // rms_norm(hidden, attn_norm)
        ex.rmsnorm_into(hidden, &w.attn_norm, &s.normed, d.hidden_dim, d.eps)?;

        // attn_qkv . normed -> conv_in
        ex.gemv_dispatch(
            w.attn_qkv.qtype,
            w.attn_qkv.ptr,
            &s.normed,
            &s.conv_in,
            w.attn_qkv.n,
            w.attn_qkv.k,
        )?;

        // causal_conv1d + SiLU (the window is updated in place)
        ex.gdn_causal_conv1d_silu_into(
            &s.conv_in,
            &state.conv[il],
            &w.conv1d_weight,
            &s.conv_out,
            d.conv_dim,
            d.conv_kernel,
        )?;

        // split q | k | v out of conv_out, then per-head L2 on q and k only
        let q_view = Self::view(&s.conv_out, 0, d.k_dim);
        let k_view = Self::view(&s.conv_out, d.k_dim, d.k_dim);
        let v_view = Self::view(&s.conv_out, d.k_dim * 2, d.v_dim);
        ex.gdn_per_head_l2_norm_into(&q_view, d.head_k_dim, d.num_k_heads, d.eps)?;
        ex.gdn_per_head_l2_norm_into(&k_view, d.head_k_dim, d.num_k_heads, d.eps)?;

        // Gated DeltaNet applies NO RoPE: position comes from the causal conv.

        // dt = softplus(ssm_alpha . x + dt_bias) * a ; beta = sigmoid(ssm_beta . x)
        ex.gemv_dispatch(
            w.ssm_alpha.qtype,
            w.ssm_alpha.ptr,
            &s.normed,
            &s.alpha_raw,
            w.ssm_alpha.n,
            w.ssm_alpha.k,
        )?;
        ex.gemv_dispatch(
            w.ssm_beta.qtype,
            w.ssm_beta.ptr,
            &s.normed,
            &s.beta_raw,
            w.ssm_beta.n,
            w.ssm_beta.k,
        )?;
        ex.gdn_gates_into(
            &s.alpha_raw,
            &w.ssm_dt_bias,
            &w.ssm_a,
            &s.beta_raw,
            &s.dt,
            &s.beta,
            d.num_v_heads,
        )?;

        // attn_gate . x
        ex.gemv_dispatch(
            w.attn_gate.qtype,
            w.attn_gate.ptr,
            &s.normed,
            &s.gate,
            w.attn_gate.n,
            w.attn_gate.k,
        )?;

        // the delta rule (the recurrent state is updated in place)
        ex.gdn_delta_rule_into(
            &q_view,
            &k_view,
            &v_view,
            &s.beta,
            &s.dt,
            &state.ssm[il],
            &s.out_h,
            d.num_k_heads,
            d.head_k_dim,
            d.num_v_heads,
            d.head_v_dim,
        )?;
        std::mem::forget(q_view);
        std::mem::forget(k_view);
        std::mem::forget(v_view);

        // gated rmsnorm, then ssm_out, then the first residual
        ex.gdn_gated_rmsnorm_into(
            &s.out_h,
            &s.gate,
            &w.ssm_norm_weight,
            &s.ssm_out_in,
            d.head_v_dim,
            d.num_v_heads,
            d.eps,
        )?;
        ex.gemv_dispatch(
            w.ssm_out.qtype,
            w.ssm_out.ptr,
            &s.ssm_out_in,
            &s.ssm_out,
            w.ssm_out.n,
            w.ssm_out.k,
        )?;
        ex.residual_add_into(hidden, &s.ssm_out, hidden, d.hidden_dim)?;

        // post_attention_norm -> SwiGLU FFN -> the second residual
        ex.rmsnorm_into(
            hidden,
            &w.post_attention_norm,
            &s.post_normed,
            d.hidden_dim,
            d.eps,
        )?;
        ex.gemv_dispatch(
            w.ffn_gate.qtype,
            w.ffn_gate.ptr,
            &s.post_normed,
            &s.ffn_gate,
            w.ffn_gate.n,
            w.ffn_gate.k,
        )?;
        ex.gemv_dispatch(
            w.ffn_up.qtype,
            w.ffn_up.ptr,
            &s.post_normed,
            &s.ffn_up,
            w.ffn_up.n,
            w.ffn_up.k,
        )?;
        ex.fused_swiglu_into(&s.ffn_gate, &s.ffn_up, &s.ffn_act, d.intermediate_dim)?;
        ex.gemv_dispatch(
            w.ffn_down.qtype,
            w.ffn_down.ptr,
            &s.ffn_act,
            &s.ffn_down,
            w.ffn_down.n,
            w.ffn_down.k,
        )?;
        ex.residual_add_into(hidden, &s.ffn_down, hidden, d.hidden_dim)?;

        // NO sync here — see `attention_layer_inner`. Stream order IS the
        // dependency; the host only has to wait where it reads.
        Ok(())
    }

    /// Run one token at `position` through every layer of both kinds, the
    /// output norm and the `lm_head`, and return the logits — the GPU twin of
    /// `Qwen35Model::forward_single_qwen35`.
    ///
    /// `state` is updated in place (conv windows, recurrent states and the K/V
    /// caches) and its KV length advanced to `position + 1`, exactly as the CPU
    /// advances `cache.kv_cache` at the end of the token.
    ///
    /// # Errors
    /// A token id outside the vocabulary, a position past the state's
    /// `max_seq_len`, or any kernel / GEMV / transfer failure.
    pub fn forward_single(
        &mut self,
        token: u32,
        state: &mut Qwen35CudaState,
        position: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.dims.hidden_dim as usize;
        let embedding = self.model.base.token_embedding();
        let start = (token as usize) * hidden_dim;
        if start + hidden_dim > embedding.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda: token {token} is outside the {}-row embedding table",
                    embedding.len() / hidden_dim
                ),
            });
        }
        if position >= state.max_seq_len {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda: position {position} is past the KV cache ({} rows)",
                    state.max_seq_len
                ),
            });
        }

        let dev = GpuBuffer::from_host(
            self.executor.context(),
            &embedding[start..start + hidden_dim],
        )
        .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;
        for il in 0..self.layers.len() {
            match self.layers[il] {
                CudaLayer::DeltaNet(_) => self.deltanet_layer(state, il, &dev)?,
                CudaLayer::Attention(_) => self.attention_layer(state, il, &dev, position)?,
            }
        }
        // The tail stays on the device (#3090 review). `hidden_to_logits` would
        // sync, copy `dev` to the host and upload it again; the hidden state is
        // already where the output norm wants it, so run the norm into
        // `out_normed` and the lm_head GEMV into `logits_buf` on the same
        // stream. The CPU reference applies no lm_head bias (`lm_head_bias` is
        // None for this architecture — `forward_single_qwen35` goes straight
        // from `rms_norm_into` to `fused_matmul_into`), so neither does this.
        let d = self.dims;
        self.executor
            .rmsnorm_into(
                &dev,
                &self.output_norm,
                &self.out_normed,
                d.hidden_dim,
                d.eps,
            )
            .map_err(|e| gpu_err("qwen35_cuda_lm_head", &e))?;
        self.executor
            .gemv_dispatch(
                self.lm_head.qtype,
                self.lm_head.ptr,
                &self.out_normed,
                &self.logits_buf,
                self.lm_head.n,
                self.lm_head.k,
            )
            .map_err(|e| gpu_err("qwen35_cuda_lm_head", &e))?;

        // The ONE sync of the whole token, in front of the ONE download.
        self.executor
            .sync_stream()
            .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;
        let mut logits = vec![0.0f32; d.vocab_size as usize];
        self.logits_buf
            .copy_to_host(&mut logits)
            .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;

        state.kv_len = state.kv_len.max(position + 1);
        Ok(logits)
    }

    /// Run every Gated `DeltaNet` layer over a host hidden state and read the
    /// result back — the tests' end-to-end handle on the GPU block.
    ///
    /// Full-attention layers are skipped, so this is NOT a decoder forward
    /// pass; it is exactly "the `DeltaNet` layers, in order".
    ///
    /// # Errors
    /// A hidden state of the wrong width, or any per-layer failure.
    pub fn forward_hidden_deltanet_only(&mut self, hidden_host: &[f32]) -> Result<Vec<f32>> {
        let hidden_dim = self.dims.hidden_dim as usize;
        if hidden_host.len() != hidden_dim {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen35_cuda: hidden state is {} wide, the model's is {hidden_dim}",
                    hidden_host.len()
                ),
            });
        }
        let dev = GpuBuffer::from_host(self.executor.context(), hidden_host)
            .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;
        for il in 0..self.model.layers.len() {
            if matches!(self.model.layers[il], Qwen35OwnedLayer::DeltaNet(_)) {
                self.forward_deltanet_layer(il, &dev)?;
            }
        }
        self.executor
            .sync_stream()
            .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;
        let mut out = vec![0.0f32; hidden_dim];
        dev.copy_to_host(&mut out)
            .map_err(|e| gpu_err("qwen35_cuda_forward", &e))?;
        Ok(out)
    }
}

/// PMAT-3596 (#3596): the batched (chunked) prefill — [`Qwen35CudaModel::prefill`].
#[path = "forward_qwen35_cuda_prefill.rs"]
mod prefill;
pub use prefill::{PREFILL_MAX_CHUNK_ROWS, PREFILL_SCORES_BUDGET_BYTES};

/// Per-layer CPU parity on the real Qwen3.5-0.8B file.
#[cfg(test)]
#[path = "forward_qwen35_cuda_tests.rs"]
mod qwen35_cuda_tests;
