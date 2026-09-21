//! PMAT-3714 / aprender#3714: the Qwen3-MoE decoder, resident on one CUDA device.
//!
//! [`Qwen3MoeCudaModel`] is built from the CPU [`OwnedQuantizedModel`] (which
//! the MoE loader fills with attention, norms and `lm_head`) plus the per-layer
//! [`Qwen3MoeQuantizedLayer`] descriptors that locate the router and the three
//! stacked expert tensors in the mapped file. Every weight goes to the device
//! once; the hidden state never leaves it inside a layer.
//!
//! The CPU function is the specification —
//! `OwnedQuantizedModel::forward_single_qwen3_moe_with_cache`. Op order, verbatim:
//!
//! ```text
//! rms_norm -> attn_q | attn_k | attn_v GEMV -> per-head RMSNorm on q and k
//!   -> NEOX rope on q and k -> KV append at `position` -> GQA decode attention
//!   -> attn_output GEMV -> residual -> ffn_norm
//!   -> router GEMV (F32) -> softmax, top-k, renormalize (`route_top_k`)
//!   -> for each routed expert: gate GEMV, up GEMV, SwiGLU, down GEMV,
//!      weighted into the residual
//! ```
//!
//! then the output norm and the `lm_head`.
//!
//! ## Kernels
//!
//! Every kernel here already existed on the executor (#3714 gap list, G3): the
//! expert GEMVs are the ordinary Q4_K/Q6_K GEMVs pointed at `base + e * stride`
//! inside one stacked-expert allocation — each expert is contiguous in the GGUF,
//! so no repacking. The router weight `w_e` is folded into the SwiGLU activation
//! (`act ⊙ w_e`) before the down GEMV, which is linear, so the down GEMV emits
//! the already-weighted expert output and it adds straight into the residual.
//!
//! ## Routing stays on the host (v1)
//!
//! The router logits (one F32 GEMV, `num_experts` floats) come back to the host
//! once per layer and go through [`route_top_k`] — the SAME function the CPU
//! layers call — so a GPU token can differ from a CPU token only in the logits
//! the router computed, never in how an expert was picked from them. That is
//! one sync per layer. Moving the top-k onto the device (so the token can be
//! graph-captured) is #3714 R3, and is where new kernels start.

use super::{OwnedQKVWeights, OwnedQuantizedModel, OwnedQuantizedTensor, RealizarError, Result};
use crate::cuda::types::WeightQuantType;
use crate::cuda::CudaExecutor;
use crate::gguf::qwen3_moe_load::{route_top_k, Qwen3MoeQuantizedLayer};
use trueno_gpu::driver::GpuBuffer;

/// Positions the device KV cache holds unless a caller asks for more.
pub const DEFAULT_MAX_SEQ_LEN: usize = 512;

/// The MoE shape the GGUF metadata declares (`{arch}.expert_count`,
/// `{arch}.expert_used_count`, `{arch}.expert_feed_forward_length`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Qwen3MoeShape {
    /// Experts per layer (128 for Qwen3-30B-A3B).
    pub num_experts: usize,
    /// Experts routed per token (8).
    pub num_experts_per_tok: usize,
    /// Each expert's FFN width (768).
    pub expert_dim: usize,
}

/// A 2-D quantized projection resident in the executor's weight cache.
#[derive(Debug, Clone, Copy)]
struct QuantWeight {
    ptr: u64,
    qtype: WeightQuantType,
    /// Output width.
    n: u32,
    /// Input width.
    k: u32,
}

/// One stacked expert tensor `[num_experts][n][k]`, resident as ONE allocation.
#[derive(Debug, Clone, Copy)]
struct ExpertStack {
    base: u64,
    /// Bytes per expert: `byte_size / num_experts`.
    stride: u64,
    qtype: WeightQuantType,
    n: u32,
    k: u32,
}

impl ExpertStack {
    /// Device pointer of expert `e`'s `[n][k]` matrix.
    fn expert(&self, e: usize) -> u64 {
        self.base + self.stride * e as u64
    }
}

/// One decoder layer's device-resident weights.
struct MoeLayer {
    attn_norm: GpuBuffer<f32>,
    attn_q_norm: GpuBuffer<f32>,
    attn_k_norm: GpuBuffer<f32>,
    ffn_norm: GpuBuffer<f32>,
    attn_q: QuantWeight,
    attn_k: QuantWeight,
    attn_v: QuantWeight,
    attn_output: QuantWeight,
    router: QuantWeight,
    gate: ExpertStack,
    up: ExpertStack,
    down: ExpertStack,
}

/// Per-sequence decode state on the device: one K and one V cache per layer,
/// each `[max_seq_len][num_kv_heads * head_dim]` — the CPU cache's layout.
pub struct Qwen3MoeCudaState {
    kv: Vec<(GpuBuffer<f32>, GpuBuffer<f32>)>,
    /// `num_kv_heads * head_dim`.
    kv_row: usize,
    /// Positions the KV cache can hold.
    max_seq_len: usize,
    /// KV rows written so far.
    kv_len: usize,
}

impl Qwen3MoeCudaState {
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
}

/// Scratch device buffers for one layer step, allocated once.
struct Scratch {
    normed: GpuBuffer<f32>,
    q: GpuBuffer<f32>,
    q_normed: GpuBuffer<f32>,
    k_raw: GpuBuffer<f32>,
    attn_out_in: GpuBuffer<f32>,
    attn_out: GpuBuffer<f32>,
    ffn_normed: GpuBuffer<f32>,
    router_logits: GpuBuffer<f32>,
    e_gate: GpuBuffer<f32>,
    e_up: GpuBuffer<f32>,
    e_act: GpuBuffer<f32>,
    e_act_w: GpuBuffer<f32>,
    e_down: GpuBuffer<f32>,
    /// `[num_experts_per_tok][expert_dim]`: routed slot `s`'s weight, repeated
    /// across its row — the second operand of the `act ⊙ w` multiply.
    route_w: GpuBuffer<f32>,
}

/// The shapes the forward reads, all derived from the model config.
#[derive(Debug, Clone, Copy)]
struct Dims {
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    /// `rope_theta.powf(-2.0 / head_dim)` — see `PartialNeoxRopeKernel`.
    theta_scale: f32,
    eps: f32,
    vocab_size: u32,
    num_experts: u32,
    num_experts_per_tok: u32,
    expert_dim: u32,
}

/// Qwen3-MoE, resident on one CUDA device (#3714).
///
/// Built from the CPU [`OwnedQuantizedModel`], which stays the specification
/// and the parity reference.
pub struct Qwen3MoeCudaModel<'a> {
    model: &'a OwnedQuantizedModel,
    executor: CudaExecutor,
    layers: Vec<MoeLayer>,
    scratch: Scratch,
    output_norm: GpuBuffer<f32>,
    lm_head: QuantWeight,
    out_normed: GpuBuffer<f32>,
    logits_buf: GpuBuffer<f32>,
    dims: Dims,
    max_seq_len: usize,
    /// Host staging for the router logits of one layer.
    router_host: Vec<f32>,
    /// Host staging for `Scratch::route_w`.
    route_w_host: Vec<f32>,
    /// The routing each layer chose for the LAST token forwarded — the parity
    /// tests' observation point for a routing divergence.
    last_routes: Vec<Vec<(usize, f32)>>,
    /// Test builds only: a routing fault to inject, so the parity tests can
    /// prove they reject a wrong forward (`tests::RoutingFault`).
    #[cfg(test)]
    fault: Option<tests::RoutingFault>,
}

/// Map a GPU error into the crate error type with the operation that raised it.
fn gpu_err(operation: &str, e: &trueno_gpu::GpuError) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: operation.to_string(),
        reason: format!("{e}"),
    }
}

/// A refusal to build: this model cannot serve the file, and says why.
fn refuse(reason: String) -> RealizarError {
    RealizarError::UnsupportedOperation {
        operation: "qwen3moe_cuda_build".to_string(),
        reason,
    }
}

const MIB: usize = 1024 * 1024;

/// The capacity plan's inputs for this model (#3596's one fit computation,
/// `crate::capacity`): quantized weights as uploaded, an f32 K and V row per
/// layer per position, and the per-layer scratch as workspace. Pure, so the
/// numbers are tested without a device.
fn capacity_inputs(
    weights_bytes: usize,
    layers: usize,
    kv_row: usize,
    max_seq_len: usize,
    scratch_bytes: usize,
    memory: crate::capacity::DeviceMemory,
) -> crate::capacity::CapacityInputs {
    let (free, total) = memory.plan_free_total();
    crate::capacity::CapacityInputs {
        weights_bytes: weights_bytes as u64,
        kv_bytes_per_token_f32: (2 * layers * kv_row * 4) as u64,
        seq_len: max_seq_len as u64,
        workspace_bytes: scratch_bytes as u64,
        overhead_bytes: crate::capacity::OVERHEAD_BYTES,
        gpu_free_bytes: free,
        gpu_total_bytes: total,
        // This model's KV cache is f32 only: a plan that fits only at f16 is a
        // refusal here, never a silent downgrade.
        f16_kv_decode_available: false,
        memory: Some(memory),
    }
}

impl<'a> Qwen3MoeCudaModel<'a> {
    /// Build with room for [`DEFAULT_MAX_SEQ_LEN`] positions.
    ///
    /// # Errors
    /// As [`Self::with_max_seq_len`].
    pub fn new(
        model: &'a OwnedQuantizedModel,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        shape: Qwen3MoeShape,
        data: &[u8],
        executor: CudaExecutor,
    ) -> Result<Self> {
        Self::with_max_seq_len(
            model,
            moe_layers,
            shape,
            data,
            executor,
            DEFAULT_MAX_SEQ_LEN,
        )
    }

    /// Build the GPU model with room for `max_seq_len` KV cache positions.
    ///
    /// `data` is the mapped GGUF the `moe_layers` descriptors index into.
    ///
    /// # Errors
    /// A config this forward does not implement (named, never guessed), a
    /// device without room for the weights plus the KV cache (the refusal
    /// carries the arithmetic), or any CUDA upload failure.
    pub fn with_max_seq_len(
        model: &'a OwnedQuantizedModel,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        shape: Qwen3MoeShape,
        data: &[u8],
        mut executor: CudaExecutor,
        max_seq_len: usize,
    ) -> Result<Self> {
        let dims = Self::dims_of(model, shape)?;
        Self::check_supported(model, moe_layers, shape)?;
        if max_seq_len == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "qwen3moe_cuda: max_seq_len must be at least 1".to_string(),
            });
        }
        Self::check_fits(&executor, model, moe_layers, dims, max_seq_len)?;

        // Correctness first (#3714 R1): the float Q4_K/Q6_K GEMV variants, as
        // the #3090 hybrid pins them. The DP4A variants quantize every GEMV's
        // activation to int8; the router then picks 8 of 128 experts from
        // logits computed off those activations, and whether that keeps the
        // routing inside the parity floors is a measurement, not an
        // assumption — it belongs to the throughput row (R3), which must show
        // it on both hosts before this pin moves.
        executor.gpu_profile.q4k = crate::cuda::gpu_profile::Q4kVariant::Mwv;
        executor.gpu_profile.q6k = crate::cuda::gpu_profile::Q6kVariant::Mwv;

        let mut layers = Vec::with_capacity(model.layers.len());
        for (il, (layer, moe)) in model.layers.iter().zip(moe_layers).enumerate() {
            layers.push(Self::build_layer(
                &mut executor,
                il,
                layer,
                moe,
                shape,
                data,
            )?);
        }

        let output_norm = Self::upload_f32(&executor, "output_norm", &model.output_norm_weight)?;
        let lm_head = Self::upload_quant(&mut executor, "output.weight", &model.lm_head_weight)?;
        if lm_head.n != dims.vocab_size || lm_head.k != dims.hidden_dim {
            return Err(refuse(format!(
                "the lm_head is {}x{}, the config says vocab {} x hidden {}",
                lm_head.n, lm_head.k, dims.vocab_size, dims.hidden_dim
            )));
        }

        // The quantized GEMVs read their activation through the executor
        // workspace, sized from the widest input this model feeds one.
        let q_dim = (dims.num_heads * dims.head_dim) as usize;
        let widest_gemv_input = (dims.hidden_dim as usize).max(q_dim);
        executor
            .init_workspace(widest_gemv_input, dims.expert_dim as usize)
            .map_err(|e| gpu_err("qwen3moe_cuda_workspace", &e))?;

        let scratch = Self::build_scratch(&executor, dims)?;
        let out_normed = Self::zeros(&executor, dims.hidden_dim as usize)?;
        let logits_buf = Self::zeros(&executor, dims.vocab_size as usize)?;
        let num_layers = layers.len();
        Ok(Self {
            model,
            executor,
            layers,
            scratch,
            output_norm,
            lm_head,
            out_normed,
            logits_buf,
            dims,
            max_seq_len,
            router_host: vec![0.0; shape.num_experts],
            route_w_host: vec![0.0; shape.num_experts_per_tok * shape.expert_dim],
            last_routes: vec![Vec::new(); num_layers],
            #[cfg(test)]
            fault: None,
        })
    }

    /// The shapes, read from the model — never hard-coded.
    fn dims_of(model: &OwnedQuantizedModel, shape: Qwen3MoeShape) -> Result<Dims> {
        let c = &model.config;
        let head_dim = c.head_dim();
        let theta_scale = trueno_gpu::kernels::gdn::PartialNeoxRopeKernel::new(
            c.num_heads as u32,
            head_dim as u32,
            head_dim as u32,
        )
        .theta_scale(c.rope_theta);
        let u = |v: usize, what: &str| {
            u32::try_from(v)
                .map_err(|_| refuse(format!("{what} = {v} does not fit a kernel's u32")))
        };
        Ok(Dims {
            hidden_dim: u(c.hidden_dim, "hidden_dim")?,
            num_heads: u(c.num_heads, "num_heads")?,
            num_kv_heads: u(c.num_kv_heads, "num_kv_heads")?,
            head_dim: u(head_dim, "head_dim")?,
            theta_scale,
            eps: c.eps,
            vocab_size: u(c.vocab_size, "vocab_size")?,
            num_experts: u(shape.num_experts, "num_experts")?,
            num_experts_per_tok: u(shape.num_experts_per_tok, "num_experts_per_tok")?,
            expert_dim: u(shape.expert_dim, "expert_dim")?,
        })
    }

    /// Everything the CPU specification does that this forward does NOT
    /// implement is refused here, by name — a silent mismatch would only show
    /// up as a parity failure far from its cause.
    fn check_supported(
        model: &OwnedQuantizedModel,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        shape: Qwen3MoeShape,
    ) -> Result<()> {
        let c = &model.config;
        if moe_layers.len() != model.layers.len() {
            return Err(refuse(format!(
                "{} MoE layer descriptors for {} decoder layers",
                moe_layers.len(),
                model.layers.len()
            )));
        }
        if shape.num_experts == 0
            || shape.num_experts_per_tok == 0
            || shape.expert_dim == 0
            || shape.num_experts_per_tok > shape.num_experts
        {
            return Err(refuse(format!("incomplete MoE shape {shape:?}")));
        }
        if !c.constraints.uses_rmsnorm() {
            return Err(refuse("only RMSNorm is implemented".to_string()));
        }
        if !c.constraints.uses_rope() || c.rope_type != 2 {
            return Err(refuse(format!(
                "only NEOX rope (rope_type 2) is implemented; the config has uses_rope={} \
                 rope_type={}",
                c.constraints.uses_rope(),
                c.rope_type
            )));
        }
        if c.constraints.uses_absolute_positions() {
            return Err(refuse(
                "absolute position embeddings are not implemented".to_string(),
            ));
        }
        if c.num_kv_heads == 0 || c.num_heads % c.num_kv_heads != 0 {
            return Err(refuse(format!(
                "num_heads {} is not a multiple of num_kv_heads {}",
                c.num_heads, c.num_kv_heads
            )));
        }
        if model.lm_head_bias.is_some() {
            return Err(refuse("an lm_head bias is not implemented".to_string()));
        }
        for (il, layer) in model.layers.iter().enumerate() {
            let missing = if layer.qkv_bias.is_some() || layer.attn_output_bias.is_some() {
                Some("attention biases")
            } else if layer.attn_norm_bias.is_some() {
                Some("an attention-norm bias")
            } else if layer.attn_q_norm_weight.is_none() || layer.attn_k_norm_weight.is_none() {
                Some("the per-head Q/K RMSNorm Qwen3 requires (absent here)")
            } else if layer.ffn_norm_weight.is_none() {
                Some("the ffn_norm Qwen3-MoE requires (absent here)")
            } else if !matches!(layer.qkv_weight, OwnedQKVWeights::Separate { .. }) {
                Some("a fused QKV tensor")
            } else {
                None
            };
            if let Some(what) = missing {
                return Err(refuse(format!("layer {il}: {what} is not implemented")));
            }
        }
        Ok(())
    }

    /// Refuse, with the arithmetic, a device that cannot hold the model.
    ///
    /// #3714 done_when 1: the 30B-A3B at Q4_K_M must fit the 4090 "or the
    /// refusal names the memory arithmetic". A 17.7 GiB upload that dies half
    /// way with an allocator error names nothing.
    fn check_fits(
        executor: &CudaExecutor,
        model: &OwnedQuantizedModel,
        moe_layers: &[Qwen3MoeQuantizedLayer],
        d: Dims,
        max_seq_len: usize,
    ) -> Result<()> {
        let attn: usize = model
            .layers
            .iter()
            .map(|l| {
                let qkv = match &l.qkv_weight {
                    OwnedQKVWeights::Separate { q, k, v } => {
                        q.data.len() + k.data.len() + v.data.len()
                    },
                    OwnedQKVWeights::Fused(t) => t.data.len(),
                };
                qkv + l.attn_output_weight.data.len()
            })
            .sum();
        let experts: usize = moe_layers
            .iter()
            .map(|m| {
                m.router.byte_size
                    + m.gate_exps.byte_size
                    + m.up_exps.byte_size
                    + m.down_exps.byte_size
            })
            .sum();
        let kv_row = (d.num_kv_heads * d.head_dim) as usize;
        let q_dim = (d.num_heads * d.head_dim) as usize;
        let e = d.expert_dim as usize;
        let scratch_bytes = 4
            * (6 * d.hidden_dim as usize
                + 4 * q_dim
                + kv_row
                + d.num_experts as usize
                + 4 * e
                + d.num_experts_per_tok as usize * e);
        let memory = crate::capacity::measure_device_memory(executor)
            .map_err(|e| refuse(format!("the device memory could not be measured: {e}")))?;
        let inputs = capacity_inputs(
            attn + experts + model.lm_head_weight.data.len(),
            model.layers.len(),
            kv_row,
            max_seq_len,
            scratch_bytes,
            memory,
        );
        match crate::capacity::plan(&inputs) {
            crate::capacity::CapacityVerdict::Fits(_) => Ok(()),
            crate::capacity::CapacityVerdict::Refused(r) => {
                Err(RealizarError::CapacityRefused(Box::new(r)))
            },
        }
    }

    /// Upload one quantized projection into the executor's cache.
    fn upload_quant(
        executor: &mut CudaExecutor,
        name: &str,
        tensor: &OwnedQuantizedTensor,
    ) -> Result<QuantWeight> {
        let qtype = WeightQuantType::from_ggml_type(tensor.qtype).ok_or_else(|| {
            refuse(format!(
                "'{name}': GGML type {} has no GPU GEMV kernel",
                tensor.qtype
            ))
        })?;
        if tensor.data.is_empty() {
            return Err(refuse(format!("'{name}': tensor data is empty")));
        }
        executor
            .load_quantized_weights_with_type(name, &tensor.data, tensor.qtype)
            .map_err(|e| gpu_err("qwen3moe_cuda_upload", &e))?;
        let ptr = executor
            .get_quantized_weight_ptr(name)
            .map_err(|e| gpu_err("qwen3moe_cuda_upload", &e))?;
        Ok(QuantWeight {
            ptr,
            qtype,
            n: u32::try_from(tensor.out_dim).unwrap_or(0),
            k: u32::try_from(tensor.in_dim).unwrap_or(0),
        })
    }

    /// Upload a tensor the MoE loader left in the mapped file (router, experts)
    /// straight from its bytes.
    fn upload_mapped(
        executor: &mut CudaExecutor,
        name: &str,
        tensor: &crate::gguf::quantized::QuantizedTensorRef,
        data: &[u8],
    ) -> Result<(u64, WeightQuantType)> {
        let qtype = WeightQuantType::from_ggml_type(tensor.qtype).ok_or_else(|| {
            refuse(format!(
                "'{name}': GGML type {} has no GPU GEMV kernel",
                tensor.qtype
            ))
        })?;
        let end = tensor
            .offset
            .checked_add(tensor.byte_size)
            .filter(|&e| e <= data.len());
        let Some(end) = end else {
            return Err(refuse(format!(
                "'{name}': bytes [{}, +{}) lie outside the {}-byte file",
                tensor.offset,
                tensor.byte_size,
                data.len()
            )));
        };
        executor
            .load_quantized_weights_with_type(name, &data[tensor.offset..end], tensor.qtype)
            .map_err(|e| gpu_err("qwen3moe_cuda_upload", &e))?;
        let ptr = executor
            .get_quantized_weight_ptr(name)
            .map_err(|e| gpu_err("qwen3moe_cuda_upload", &e))?;
        Ok((ptr, qtype))
    }

    /// Upload one stacked expert tensor and check that `num_experts` equal
    /// `[n][k]` matrices of its type tile it exactly.
    fn upload_experts(
        executor: &mut CudaExecutor,
        name: &str,
        tensor: &crate::gguf::quantized::QuantizedTensorRef,
        data: &[u8],
        num_experts: usize,
        n: usize,
        k: usize,
    ) -> Result<ExpertStack> {
        let (base, qtype) = Self::upload_mapped(executor, name, tensor, data)?;
        let stride = tensor.byte_size / num_experts;
        if stride * num_experts != tensor.byte_size || !qtype.matches_size(stride, n, k) {
            return Err(refuse(format!(
                "'{name}': {} bytes do not tile into {num_experts} experts of [{n}][{k}] {:?}",
                tensor.byte_size, qtype
            )));
        }
        Ok(ExpertStack {
            base,
            stride: stride as u64,
            qtype,
            n: n as u32,
            k: k as u32,
        })
    }

    /// Upload an f32 vector as a device buffer.
    fn upload_f32(executor: &CudaExecutor, name: &str, v: &[f32]) -> Result<GpuBuffer<f32>> {
        if v.is_empty() {
            return Err(refuse(format!("'{name}': f32 vector is empty")));
        }
        GpuBuffer::from_host(executor.context(), v).map_err(|e| gpu_err("qwen3moe_cuda_upload", &e))
    }

    /// Zero-filled device buffer of `len` f32.
    fn zeros(executor: &CudaExecutor, len: usize) -> Result<GpuBuffer<f32>> {
        GpuBuffer::from_host(executor.context(), &vec![0.0f32; len])
            .map_err(|e| gpu_err("qwen3moe_cuda_alloc", &e))
    }

    /// Upload one decoder layer.
    fn build_layer(
        executor: &mut CudaExecutor,
        il: usize,
        layer: &super::super::quantized::OwnedQuantizedLayer,
        moe: &Qwen3MoeQuantizedLayer,
        shape: Qwen3MoeShape,
        data: &[u8],
    ) -> Result<MoeLayer> {
        let name = |suffix: &str| format!("qwen3moe.blk.{il}.{suffix}");
        let OwnedQKVWeights::Separate { q, k, v } = &layer.qkv_weight else {
            return Err(refuse(format!(
                "layer {il}: a fused QKV tensor is not implemented"
            )));
        };
        fn required<'v>(il: usize, what: &str, v: &'v Option<Vec<f32>>) -> Result<&'v [f32]> {
            v.as_deref()
                .ok_or_else(|| refuse(format!("layer {il}: {what} is absent")))
        }
        let norm = |what: &str, v| required(il, what, v);
        let hidden = layer.attn_norm_weight.len();
        let (router_ptr, router_qtype) =
            Self::upload_mapped(executor, &name("ffn_gate_inp.weight"), &moe.router, data)?;
        if router_qtype != WeightQuantType::F32
            || moe.router.byte_size != shape.num_experts * hidden * 4
        {
            return Err(refuse(format!(
                "layer {il}: the router is {:?} with {} bytes; only an F32 [{}][{hidden}] router \
                 is implemented",
                router_qtype, moe.router.byte_size, shape.num_experts
            )));
        }
        Ok(MoeLayer {
            attn_norm: Self::upload_f32(executor, "attn_norm", &layer.attn_norm_weight)?,
            attn_q_norm: Self::upload_f32(
                executor,
                "attn_q_norm",
                norm("attn_q_norm", &layer.attn_q_norm_weight)?,
            )?,
            attn_k_norm: Self::upload_f32(
                executor,
                "attn_k_norm",
                norm("attn_k_norm", &layer.attn_k_norm_weight)?,
            )?,
            ffn_norm: Self::upload_f32(
                executor,
                "ffn_norm",
                norm("ffn_norm", &layer.ffn_norm_weight)?,
            )?,
            attn_q: Self::upload_quant(executor, &name("attn_q.weight"), q)?,
            attn_k: Self::upload_quant(executor, &name("attn_k.weight"), k)?,
            attn_v: Self::upload_quant(executor, &name("attn_v.weight"), v)?,
            attn_output: Self::upload_quant(
                executor,
                &name("attn_output.weight"),
                &layer.attn_output_weight,
            )?,
            router: QuantWeight {
                ptr: router_ptr,
                qtype: WeightQuantType::F32,
                n: shape.num_experts as u32,
                k: hidden as u32,
            },
            gate: Self::upload_experts(
                executor,
                &name("ffn_gate_exps.weight"),
                &moe.gate_exps,
                data,
                shape.num_experts,
                shape.expert_dim,
                hidden,
            )?,
            up: Self::upload_experts(
                executor,
                &name("ffn_up_exps.weight"),
                &moe.up_exps,
                data,
                shape.num_experts,
                shape.expert_dim,
                hidden,
            )?,
            down: Self::upload_experts(
                executor,
                &name("ffn_down_exps.weight"),
                &moe.down_exps,
                data,
                shape.num_experts,
                hidden,
                shape.expert_dim,
            )?,
        })
    }

    /// Allocate the scratch buffers one layer step needs.
    fn build_scratch(executor: &CudaExecutor, d: Dims) -> Result<Scratch> {
        let hidden = d.hidden_dim as usize;
        let q_dim = (d.num_heads * d.head_dim) as usize;
        let kv_dim = (d.num_kv_heads * d.head_dim) as usize;
        let e = d.expert_dim as usize;
        Ok(Scratch {
            normed: Self::zeros(executor, hidden)?,
            q: Self::zeros(executor, q_dim)?,
            q_normed: Self::zeros(executor, q_dim)?,
            k_raw: Self::zeros(executor, kv_dim)?,
            attn_out_in: Self::zeros(executor, q_dim)?,
            attn_out: Self::zeros(executor, hidden)?,
            ffn_normed: Self::zeros(executor, hidden)?,
            router_logits: Self::zeros(executor, d.num_experts as usize)?,
            e_gate: Self::zeros(executor, e)?,
            e_up: Self::zeros(executor, e)?,
            e_act: Self::zeros(executor, e)?,
            e_act_w: Self::zeros(executor, e)?,
            e_down: Self::zeros(executor, hidden)?,
            route_w: Self::zeros(executor, d.num_experts_per_tok as usize * e)?,
        })
    }

    /// A fresh decode state: an empty K/V cache per layer.
    ///
    /// # Errors
    /// A CUDA allocation failure.
    pub fn new_state(&self) -> Result<Qwen3MoeCudaState> {
        let kv_row = (self.dims.num_kv_heads * self.dims.head_dim) as usize;
        let mut kv = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            kv.push((
                Self::zeros(&self.executor, self.max_seq_len * kv_row)?,
                Self::zeros(&self.executor, self.max_seq_len * kv_row)?,
            ));
        }
        Ok(Qwen3MoeCudaState {
            kv,
            kv_row,
            max_seq_len: self.max_seq_len,
            kv_len: 0,
        })
    }

    /// The routing each layer chose for the last token forwarded:
    /// `(expert, renormalized weight)` pairs in rank order.
    #[must_use]
    pub fn last_routes(&self) -> &[Vec<(usize, f32)>] {
        &self.last_routes
    }

    /// The device this model runs on, and its total memory in MiB.
    #[must_use]
    pub fn device_summary(&self) -> (String, usize) {
        let name = self
            .executor
            .device_name()
            .unwrap_or_else(|_| "Unknown GPU".to_string());
        let total = self.executor.memory_info().map_or(0, |(_, t)| t / MIB);
        (name, total)
    }

    /// A non-owning view of `elems` f32 starting `offset` elements into `buf`.
    /// The returned buffer must be `std::mem::forget`-ed.
    fn view(buf: &GpuBuffer<f32>, offset: usize, elems: usize) -> GpuBuffer<f32> {
        let ptr = buf.as_ptr() + (offset as u64) * 4;
        // SAFETY: every call site passes `offset + elems <= buf.len()` (a KV
        // row below `max_seq_len`, or a routed slot below `num_experts_per_tok`),
        // and the view is forgotten before it can free memory it does not own.
        unsafe { GpuBuffer::<f32>::from_raw_parts(ptr, elems) }
    }

    /// Run one token at `position` through the whole decoder and return its
    /// logits — the GPU twin of `forward_single_qwen3_moe_with_cache`.
    ///
    /// # Errors
    /// A token outside the vocabulary, a position past the state's KV cache,
    /// or any kernel / GEMV / transfer failure.
    pub fn forward_single(
        &mut self,
        token: u32,
        state: &mut Qwen3MoeCudaState,
        position: usize,
    ) -> Result<Vec<f32>> {
        if token >= self.dims.vocab_size {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen3moe_cuda: token {token} is outside the {}-token vocabulary",
                    self.dims.vocab_size
                ),
            });
        }
        if position >= state.max_seq_len || state.kv.len() != self.layers.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "qwen3moe_cuda: position {position} is past the KV cache ({} rows), or the \
                     state was built for another model",
                    state.max_seq_len
                ),
            });
        }

        let hidden = GpuBuffer::from_host(self.executor.context(), &self.model.embed(&[token]))
            .map_err(|e| gpu_err("qwen3moe_cuda_forward", &e))?;
        for il in 0..self.layers.len() {
            self.attention_layer(state, il, &hidden, position)
                .map_err(|e| gpu_err("qwen3moe_cuda_attention", &e))?;
            let routes = self
                .moe_layer(il, &hidden)
                .map_err(|e| gpu_err("qwen3moe_cuda_moe", &e))?;
            self.last_routes[il] = routes;
        }

        let d = self.dims;
        let ex = &mut self.executor;
        ex.rmsnorm_into(
            &hidden,
            &self.output_norm,
            &self.out_normed,
            d.hidden_dim,
            d.eps,
        )
        .map_err(|e| gpu_err("qwen3moe_cuda_lm_head", &e))?;
        ex.gemv_dispatch(
            self.lm_head.qtype,
            self.lm_head.ptr,
            &self.out_normed,
            &self.logits_buf,
            self.lm_head.n,
            self.lm_head.k,
        )
        .map_err(|e| gpu_err("qwen3moe_cuda_lm_head", &e))?;
        ex.sync_stream()
            .map_err(|e| gpu_err("qwen3moe_cuda_forward", &e))?;
        let mut logits = vec![0.0f32; d.vocab_size as usize];
        self.logits_buf
            .copy_to_host(&mut logits)
            .map_err(|e| gpu_err("qwen3moe_cuda_forward", &e))?;

        state.kv_len = state.kv_len.max(position + 1);
        Ok(logits)
    }

    /// Attention, in place on the device hidden state: the first half of the
    /// CPU layer, up to and including the first residual.
    fn attention_layer(
        &mut self,
        state: &Qwen3MoeCudaState,
        il: usize,
        hidden: &GpuBuffer<f32>,
        position: usize,
    ) -> std::result::Result<(), trueno_gpu::GpuError> {
        let d = self.dims;
        let w = &self.layers[il];
        let s = &self.scratch;
        let ex = &mut self.executor;
        let kv_dim = state.kv_row;
        let (k_cache, v_cache) = &state.kv[il];

        // This token's row in the KV cache: the k norm and the v GEMV write
        // straight into it, which IS the CPU's `cache.append`.
        let k_row = Self::view(k_cache, position * kv_dim, kv_dim);
        let v_row = Self::view(v_cache, position * kv_dim, kv_dim);

        ex.rmsnorm_into(hidden, &w.attn_norm, &s.normed, d.hidden_dim, d.eps)?;
        ex.gemv_dispatch(
            w.attn_q.qtype,
            w.attn_q.ptr,
            &s.normed,
            &s.q,
            w.attn_q.n,
            w.attn_q.k,
        )?;
        ex.gemv_dispatch(
            w.attn_k.qtype,
            w.attn_k.ptr,
            &s.normed,
            &s.k_raw,
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

        ex.per_head_rmsnorm_into(
            &s.q,
            &w.attn_q_norm,
            &s.q_normed,
            d.head_dim,
            d.num_heads,
            d.eps,
        )?;
        ex.per_head_rmsnorm_into(
            &s.k_raw,
            &w.attn_k_norm,
            &k_row,
            d.head_dim,
            d.num_kv_heads,
            d.eps,
        )?;

        let pos = u32::try_from(position).unwrap_or(u32::MAX);
        ex.gdn_partial_neox_rope_into(
            &s.q_normed,
            d.num_heads,
            d.head_dim,
            d.head_dim,
            pos,
            d.theta_scale,
        )?;
        ex.gdn_partial_neox_rope_into(
            &k_row,
            d.num_kv_heads,
            d.head_dim,
            d.head_dim,
            pos,
            d.theta_scale,
        )?;

        ex.gdn_decode_attention_into(
            &s.q_normed,
            k_cache,
            v_cache,
            &s.attn_out_in,
            d.num_heads,
            d.num_kv_heads,
            d.head_dim,
            pos + 1,
        )?;
        std::mem::forget(k_row);
        std::mem::forget(v_row);

        ex.gemv_dispatch(
            w.attn_output.qtype,
            w.attn_output.ptr,
            &s.attn_out_in,
            &s.attn_out,
            w.attn_output.n,
            w.attn_output.k,
        )?;
        ex.residual_add_into(hidden, &s.attn_out, hidden, d.hidden_dim)
    }

    /// The routed-expert FFN, in place on the device hidden state: the second
    /// half of the CPU layer. Returns the routing it applied.
    fn moe_layer(
        &mut self,
        il: usize,
        hidden: &GpuBuffer<f32>,
    ) -> std::result::Result<Vec<(usize, f32)>, trueno_gpu::GpuError> {
        let d = self.dims;
        let w = &self.layers[il];
        let s = &mut self.scratch;
        let ex = &mut self.executor;
        let e_dim = d.expert_dim as usize;

        ex.rmsnorm_into(hidden, &w.ffn_norm, &s.ffn_normed, d.hidden_dim, d.eps)?;
        ex.gemv_dispatch(
            w.router.qtype,
            w.router.ptr,
            &s.ffn_normed,
            &s.router_logits,
            w.router.n,
            w.router.k,
        )?;

        // The one host round trip of the layer: the router's logits come back,
        // the shared routing rule picks the experts, their weights go down.
        // The sync also retires the previous layer's reads of `route_w`, so
        // overwriting it below cannot race them.
        ex.sync_stream()?;
        s.router_logits.copy_to_host(&mut self.router_host)?;
        let routes = route_top_k(&self.router_host, d.num_experts_per_tok as usize);
        #[cfg(test)]
        let routes = tests::inject(self.fault, routes, d.num_experts as usize);
        for (slot, &(_, weight)) in routes.iter().enumerate() {
            self.route_w_host[slot * e_dim..(slot + 1) * e_dim].fill(weight);
        }
        s.route_w.copy_from_host(&self.route_w_host)?;

        for (slot, &(expert, _)) in routes.iter().enumerate() {
            ex.gemv_dispatch(
                w.gate.qtype,
                w.gate.expert(expert),
                &s.ffn_normed,
                &s.e_gate,
                w.gate.n,
                w.gate.k,
            )?;
            ex.gemv_dispatch(
                w.up.qtype,
                w.up.expert(expert),
                &s.ffn_normed,
                &s.e_up,
                w.up.n,
                w.up.k,
            )?;
            ex.fused_swiglu_into(&s.e_gate, &s.e_up, &s.e_act, d.expert_dim)?;
            let weight_row = Self::view(&s.route_w, slot * e_dim, e_dim);
            ex.elementwise_mul_into(&s.e_act, &weight_row, &s.e_act_w, d.expert_dim)?;
            std::mem::forget(weight_row);
            ex.gemv_dispatch(
                w.down.qtype,
                w.down.expert(expert),
                &s.e_act_w,
                &s.e_down,
                w.down.n,
                w.down.k,
            )?;
            ex.residual_add_into(hidden, &s.e_down, hidden, d.hidden_dim)?;
        }
        Ok(routes)
    }
}

/// Unit tests that need no device, and the device parity test (which skips
/// itself without a GPU and without the model file).
#[cfg(test)]
#[path = "forward_qwen3_moe_resident_tests.rs"]
mod tests;
