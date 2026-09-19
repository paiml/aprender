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
//! Full-attention layers are NOT in this phase: the head-256 / partial-RoPE /
//! gated-split kernels are still being written, so
//! [`Qwen35CudaModel::forward_attention_layer`] is a named seam that refuses.
//!
//! Every state buffer is sized from the config
//! (`num_v_heads * head_v_dim * head_v_dim`, `conv_dim * (conv_kernel - 1)`) —
//! never from a constant.

use super::{OwnedQuantizedTensor, RealizarError, Result};
use crate::cuda::types::WeightQuantType;
use crate::cuda::CudaExecutor;
use crate::gguf::forward_qwen35::{Qwen35Model, Qwen35OwnedLayer};
use trueno_gpu::driver::GpuBuffer;

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

/// Per-sequence Gated `DeltaNet` decode state on the device: one causal-conv
/// window and one recurrent state per layer, both sized from the config.
///
/// Attention layers own no entry here (their KV cache is the CPU model's).
pub struct Qwen35CudaState {
    conv: Vec<GpuBuffer<f32>>,
    ssm: Vec<GpuBuffer<f32>>,
    /// `conv_dim * (conv_kernel - 1)`.
    conv_len: usize,
    /// `num_v_heads * head_v_dim * head_v_dim`.
    ssm_len: usize,
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
}

/// Qwen3.5's Gated `DeltaNet` block, resident on one CUDA device (#3090).
///
/// Built from the CPU [`Qwen35Model`], which stays the specification and the
/// parity reference.
pub struct Qwen35CudaModel<'a> {
    model: &'a Qwen35Model<'a>,
    executor: CudaExecutor,
    /// `None` for a full-attention layer (phase 2), `Some` for a `DeltaNet` one.
    layers: Vec<Option<CudaDeltaNetLayer>>,
    state: Qwen35CudaState,
    scratch: Qwen35CudaScratch,
    dims: Qwen35CudaDims,
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
        Qwen35CudaDims {
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

    /// The delta rule reads q, k and v with one head stride, so a file whose key
    /// heads differ from its value heads cannot run this kernel — refuse rather
    /// than index past a head.
    fn check_head_symmetry(d: Qwen35CudaDims) -> Result<()> {
        if d.head_k_dim == d.head_v_dim && d.num_k_heads == d.num_v_heads {
            return Ok(());
        }
        Err(RealizarError::UnsupportedOperation {
            operation: "qwen35_cuda_deltanet".to_string(),
            reason: format!(
                "the delta-rule kernel indexes q/k/v with one head stride: \
                 head_k_dim {} != head_v_dim {} or num_k_heads {} != num_v_heads {}",
                d.head_k_dim, d.head_v_dim, d.num_k_heads, d.num_v_heads
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
    pub fn new(model: &'a Qwen35Model<'a>, mut executor: CudaExecutor) -> Result<Self> {
        let dims = Self::dims_of(model);
        Self::check_head_symmetry(dims)?;

        let mut layers = Vec::with_capacity(model.layers.len());
        for (il, layer) in model.layers.iter().enumerate() {
            match layer {
                Qwen35OwnedLayer::DeltaNet(d) => {
                    layers.push(Some(Self::build_layer(&mut executor, il, d)?));
                },
                Qwen35OwnedLayer::Attention(_) => layers.push(None),
            }
        }

        let conv_len = (dims.conv_dim * (dims.conv_kernel - 1)) as usize;
        let ssm_len = (dims.num_v_heads * dims.head_v_dim * dims.head_v_dim) as usize;
        let mut conv = Vec::with_capacity(model.layers.len());
        let mut ssm = Vec::with_capacity(model.layers.len());
        for _ in 0..model.layers.len() {
            conv.push(Self::zeros(&executor, conv_len)?);
            ssm.push(Self::zeros(&executor, ssm_len)?);
        }

        // The DP4A GEMV kernels quantize the activation into
        // `workspace.q8_activation_buf`, which `init_workspace` sizes from
        // `max(hidden_dim, intermediate_dim, q_dim)`. Our widest GEMV input is
        // `v_dim` (the ssm_out projection), which is none of those — pass it as
        // the "hidden" width so the buffer covers every input this block feeds a
        // kernel. Without this the Q6_K path panics on an uninitialized buffer.
        let widest_gemv_input = dims.hidden_dim.max(dims.v_dim) as usize;
        executor
            .init_workspace(widest_gemv_input, dims.intermediate_dim as usize)
            .map_err(|e| gpu_err("qwen35_cuda_workspace", &e))?;

        let scratch = Self::build_scratch(&executor, dims)?;
        Ok(Self {
            model,
            executor,
            layers,
            state: Qwen35CudaState {
                conv,
                ssm,
                conv_len,
                ssm_len,
            },
            scratch,
            dims,
        })
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
    /// Not the default: the default is whatever `GpuProfile::detect` chose for
    /// the device, which is the production decode path.
    pub fn pin_reference_gemv(&mut self) {
        self.executor.gpu_profile.q4k = crate::cuda::gpu_profile::Q4kVariant::Mwv;
        self.executor.gpu_profile.q6k = crate::cuda::gpu_profile::Q6kVariant::Mwv;
    }

    /// Read layer `il`'s causal-conv window and recurrent state back to the
    /// host — the parity tests' observation point.
    ///
    /// # Errors
    /// A device-to-host copy failure, or a layer index that is not a `DeltaNet`
    /// layer.
    pub fn download_layer(&mut self, il: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        self.require_deltanet(il)?;
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
            Some(Some(_)) => Ok(()),
            Some(None) => Err(RealizarError::UnsupportedOperation {
                operation: "qwen35_cuda_attention".to_string(),
                reason: "#3090 phase 2".to_string(),
            }),
            None => Err(RealizarError::InvalidShape {
                reason: format!("qwen35_cuda: layer {il} is out of range"),
            }),
        }
    }

    /// The full-attention layer forward — NOT in this phase.
    ///
    /// The head-256 / partial-RoPE / gated-split kernels are being written in
    /// `aprender-gpu`; this is the named seam they land behind.
    ///
    /// # Errors
    /// Always: `UnsupportedOperation { operation: "qwen35_cuda_attention" }`.
    #[allow(clippy::unused_self)]
    pub fn forward_attention_layer(&mut self, _il: usize, _hidden: &GpuBuffer<f32>) -> Result<()> {
        Err(RealizarError::UnsupportedOperation {
            operation: "qwen35_cuda_attention".to_string(),
            reason: "#3090 phase 2".to_string(),
        })
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
        let buf = match which {
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
        self.deltanet_layer_inner(il, hidden)
            .map_err(|e| gpu_err("qwen35_cuda_deltanet", &e))
    }

    /// The op-for-op body; every error here is a GPU error.
    #[allow(clippy::too_many_lines)]
    fn deltanet_layer_inner(
        &mut self,
        il: usize,
        hidden: &GpuBuffer<f32>,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let w = self.layers[il]
            .as_ref()
            .expect("require_deltanet checked this layer");
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
            &self.state.conv[il],
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
            &self.state.ssm[il],
            &s.out_h,
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

        ex.sync_stream()
    }

    /// Run every Gated `DeltaNet` layer over a host hidden state and read the
    /// result back — the tests' end-to-end handle on the GPU block.
    ///
    /// Full-attention layers are skipped (phase 2), so this is NOT a decoder
    /// forward pass; it is exactly "the `DeltaNet` layers, in order".
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

/// Per-layer CPU parity on the real Qwen3.5-0.8B file.
#[cfg(test)]
#[path = "forward_qwen35_cuda_tests.rs"]
mod qwen35_cuda_tests;
