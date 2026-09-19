//! Quantized GGUF transformer types
//!
//! This module contains the quantized transformer layer and model structures
//! that enable fused dequantization operations for memory-efficient inference.

use crate::error::{RealizarError, Result};

use super::config::{GGUFConfig, ValidatedModelConfig};
use super::quantized::{QKVWeights, QuantizedTensorRef};
// #3432: the block sizes themselves now live in `ggml_type_table`; only the ids
// this module names in code (the F32 placeholder, the PAR-058 resolver, the
// legacy flat-sized set) are imported here.
use super::types::{
    GGUFModel, GGUF_TYPE_F32, GGUF_TYPE_Q2_K, GGUF_TYPE_Q4_0, GGUF_TYPE_Q4_1, GGUF_TYPE_Q5_0,
    GGUF_TYPE_Q8_0,
};
// The remaining ids appear only in this module's tests (including the
// `include!`d `transformer_quantized_layer_field.rs`), which reach them through
// `use super::*`.
#[cfg(test)]
use super::types::{GGUF_TYPE_BF16, GGUF_TYPE_F16, GGUF_TYPE_Q4_K, GGUF_TYPE_Q5_K, GGUF_TYPE_Q6_K};

/// Quantized transformer layer weights (stored as byte references)
///
/// Unlike `GGUFTransformerLayer` which stores dequantized Vec<f32>,
/// this stores references to quantized data for fused operations.
pub struct QuantizedGGUFTransformerLayer {
    /// Attention norm weight (kept as f32 - small, read once per token)
    pub attn_norm_weight: Vec<f32>,
    /// Attention norm bias (optional)
    pub attn_norm_bias: Option<Vec<f32>>,
    /// QKV projection weights (quantized) - supports fused or separate
    pub qkv_weight: QKVWeights,
    /// QKV bias (optional, f32)
    pub qkv_bias: Option<Vec<f32>>,
    /// Attention output projection (quantized)
    pub attn_output_weight: QuantizedTensorRef,
    /// Attention output bias (optional, f32)
    pub attn_output_bias: Option<Vec<f32>>,
    /// FFN up projection (quantized)
    pub ffn_up_weight: QuantizedTensorRef,
    /// FFN up bias (optional, f32)
    pub ffn_up_bias: Option<Vec<f32>>,
    /// FFN down projection (quantized)
    pub ffn_down_weight: QuantizedTensorRef,
    /// FFN down bias (optional, f32)
    pub ffn_down_bias: Option<Vec<f32>>,
    /// FFN gate projection (quantized, SwiGLU models like LLaMA)
    pub ffn_gate_weight: Option<QuantizedTensorRef>,
    /// FFN gate bias (optional, f32)
    pub ffn_gate_bias: Option<Vec<f32>>,
    /// FFN norm weight (pre-FFN layer norm, LLaMA-style)
    pub ffn_norm_weight: Option<Vec<f32>>,
    /// FFN norm bias (optional, f32)
    pub ffn_norm_bias: Option<Vec<f32>>,
    /// GH-279: Per-head Q RMSNorm weight [head_dim] (Qwen3)
    pub attn_q_norm_weight: Option<Vec<f32>>,
    /// GH-279: Per-head K RMSNorm weight [head_dim] (Qwen3)
    pub attn_k_norm_weight: Option<Vec<f32>>,
    /// PMAT-810: Gemma2 POST-attention RMSNorm weight (`blk.N.post_attention_norm.weight`).
    /// Gemma2 sandwiches the attention block: `x + post_attn_norm(attn(input_norm(x)))`.
    /// `None` for every other architecture (LLaMA/Qwen/Gemma1 have no post-norm).
    pub post_attn_norm_weight: Option<Vec<f32>>,
    /// PMAT-810: Gemma2 POST-feedforward RMSNorm weight (`blk.N.post_ffw_norm.weight`).
    /// Gemma2 sandwiches the FFN block: `h + post_ffw_norm(ffn(pre_ffn_norm(h)))`.
    /// `None` for every other architecture.
    pub post_ffw_norm_weight: Option<Vec<f32>>,
}

/// Reason this GGUF cannot be run by the quantized transformer path, or `None`.
///
/// GH-704 put the SSM (Gated Delta Net) refusal inline in
/// [`QuantizedGGUFTransformer::from_gguf`], so it only fired for callers that
/// actually built a transformer. Every other tool reading the same file was free
/// to invent a story about it — `apr ptx-map` printed a full dense-transformer
/// kernel sequence, exit 0, for a Qwen3.5 GGUF that `apr check` refuses to load
/// (dogfood-0.63.0, issue #2399 finding 2). Both surfaces now ask this one
/// function, so they cannot drift.
///
/// Takes tensor names rather than a `GGUFModel` so the predicate is directly
/// testable and callable from any crate that has already parsed the header.
///
/// #3477 (operator ruling 2026-09-19, GPU support for Qwen3.5 is required):
/// the hybrid forward now exists on BOTH backends for the architecture
/// [`hybrid_forward_handles`] names — `forward_qwen35` on the CPU (#3091) and
/// `Qwen35CudaModel` on the GPU (#3090) — so this predicate no longer refuses
/// it. Every other SSM architecture keeps its refusal, reworded: the old
/// sentence claimed the CPU cannot run Gated DeltaNet, which has not been true
/// since #3091.
pub fn unsupported_architecture_reason<'n>(
    architecture: &str,
    tensor_names: impl IntoIterator<Item = &'n str>,
) -> Option<String> {
    let ssm_tensor = tensor_names
        .into_iter()
        .find(|name| name.contains("ssm_") || name.contains("ssm."));
    let tensor_name = ssm_tensor?;
    if hybrid_forward_handles(architecture) {
        return None;
    }
    Some(format!(
        "Architecture '{architecture}' uses SSM/Gated Delta Net layers (detected tensor '{tensor_name}'). \
         There is no CPU or GPU forward for architecture '{architecture}' with SSM tensors. \
         Tracking issues: #3090 (GPU) and #3091 (CPU) — both implement the Qwen3.5 hybrid ('qwen35'), not this architecture. \
         Use a standard transformer model (e.g., Qwen2.5, LLaMA, Mistral) or wait for SSM support in a future release."
    ))
}

/// Is this the architecture string the hybrid (Gated `DeltaNet` + gated
/// attention) forward handles?
///
/// This is the ONE spelling set the runtime dispatches on: `inference_result.rs`
/// compares the GGUF architecture to the literal `"qwen35"` and, on a match,
/// builds `forward_qwen35::Qwen35Model` (CPU, #3091), which
/// `cuda::forward_qwen35_cuda::Qwen35CudaModel` wraps on the GPU (#3090).
/// `apr qa`'s capability gate, `apr parity`'s refusal and `apr ptx-map` all ask
/// THIS function rather than keeping private copies, so a tool can no longer
/// admit a spelling the runtime refuses (or refuse one it serves).
///
/// Deliberately exact, not case-insensitive and not a spelling family:
/// `qwen3_5` / `qwen3.5` reach no forward, so calling them handled would have
/// the tooling promise a load `apr run` cannot perform.
#[must_use]
pub fn hybrid_forward_handles(architecture: &str) -> bool {
    architecture == "qwen35"
}

/// Why the DENSE loader ([`QuantizedGGUFTransformer::from_gguf`]) cannot load
/// this file, even though the hybrid forward can run it.
///
/// #3477: lifting the SSM refusal from [`unsupported_architecture_reason`] would
/// otherwise let `apr check`-style callers walk into the dense tensor lookups
/// with a Qwen3.5 GGUF and fail on a missing `blk.0.attn_q.weight` — the cryptic
/// surface the GH-704 refusal was introduced to replace. The dense loader is not
/// the hybrid path, so it refuses with a message that names the path that IS.
pub(crate) fn dense_loader_refusal<'n>(
    architecture: &str,
    tensor_names: impl IntoIterator<Item = &'n str>,
) -> Option<String> {
    if !hybrid_forward_handles(architecture) {
        return None;
    }
    let tensor_name = tensor_names
        .into_iter()
        .find(|name| name.contains("ssm_") || name.contains("ssm."))?;
    Some(format!(
        "Architecture '{architecture}' is the Qwen3.5 hybrid (Gated Delta Net, detected tensor '{tensor_name}'): \
         it runs through `Qwen35Model` (realizar::gguf::forward_qwen35, CPU #3091) or `Qwen35CudaModel` (GPU #3090), \
         NOT through the dense QuantizedGGUFTransformer loader, whose per-layer tensor names this file does not carry. \
         Load it with `apr run` / `apr chat`, which dispatch this architecture to the hybrid forward."
    ))
}

/// The GPU-quant refusal for a hybrid (Gated `DeltaNet`) GGUF, or `None`.
///
/// The tensor set and the whitelist both live in `dtype.rs` beside
/// `gpu_unsupported_quant_qtype`, the single source of truth this delegates to
/// (PMAT-785); this is only the public, header-level entry point the tools call.
///
/// Answers from the header alone — `(tensor name, GGML type)` pairs — so a
/// caller refuses BEFORE uploading a single weight. `None` for every
/// architecture the hybrid forward does not handle: the dense quant gate
/// already speaks for those.
pub fn hybrid_gpu_quant_refusal<'n>(
    architecture: &str,
    tensors: impl IntoIterator<Item = (&'n str, u32)>,
) -> Option<String> {
    if !hybrid_forward_handles(architecture) {
        return None;
    }
    let (name, qtype) = super::loader::hybrid_gpu_unsupported_quant_tensor(tensors)?;
    Some(format!(
        "Architecture '{architecture}': tensor '{name}' has GGML type {qtype}, which has no verified GPU GEMV kernel. \
         The GPU weight upload would decode it as Q4_K and produce garbage logits (PMAT-781/783/785). \
         Run this model on the CPU (`apr run --no-gpu`), or requantize to Q4_0/Q4_1/Q5_0/Q8_0/Q4_K/Q5_K/Q6_K."
    ))
}

/// Quantized GGUF Transformer for fused inference
///
/// Per Williams et al. (2009) roofline model, LLM inference is memory-bound.
/// This transformer stores weights in quantized form and uses fused
/// dequant+dot operations to minimize memory bandwidth.
///
/// # Performance Benefits
///
/// - **8x bandwidth reduction** for Q4_K vs f32 (144 bytes vs 1024 bytes per 256 values)
/// - **Zero intermediate buffers** - dequantization happens inline with dot product
/// - **SIMD acceleration** - AVX2/FMA fused operations when available
/// - **Zero-copy loading** - weights stay in memory-mapped file
///
/// # Architecture
///
/// ```text
/// [Memory-mapped Q4_K bytes] → [fused_q4k_dot_simd] → [f32 result]
///                               ↑
///                         No intermediate Vec<f32>!
/// ```
pub struct QuantizedGGUFTransformer<'a> {
    /// Model configuration
    pub config: GGUFConfig,
    /// Reference to memory-mapped file data
    pub data: &'a [u8],
    /// Token embedding (kept as f32 for lookup)
    pub token_embedding: Vec<f32>,
    /// GH-278: Position embedding [context_length, hidden_dim] (GPT-2 only)
    pub position_embedding: Option<Vec<f32>>,
    /// Quantized layer weights
    pub layers: Vec<QuantizedGGUFTransformerLayer>,
    /// M32c.2: Per-layer MoE expert tensor descriptors when loaded
    /// via `from_gguf_for_moe`. Empty `Vec` for dense models loaded
    /// via the standard `from_gguf` constructor. When populated,
    /// `moe_layers.len() == layers.len()` and each entry holds the
    /// 4 quantized tensor refs for `qwen3_moe`'s router + per-expert
    /// gate/up/down. The `layers[i].ffn_up_weight` etc. fields are
    /// stubbed with empty `QuantizedTensorRef` placeholders for MoE
    /// layers; consumers MUST check `moe_layers[i].is_some()` before
    /// dispatching the FFN.
    pub moe_layers: Vec<Option<crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer>>,
    /// Output norm weight (f32)
    pub output_norm_weight: Vec<f32>,
    /// Output norm bias (optional)
    pub output_norm_bias: Option<Vec<f32>>,
    /// LM head weight (quantized for large vocab)
    pub lm_head_weight: QuantizedTensorRef,
    /// LM head bias (optional, f32)
    pub lm_head_bias: Option<Vec<f32>>,
}

impl<'a> QuantizedGGUFTransformer<'a> {
    /// Load quantized transformer from memory-mapped GGUF model
    ///
    /// # Arguments
    ///
    /// * `model` - Parsed GGUF model metadata
    /// * `data` - Memory-mapped file data (zero-copy)
    ///
    /// # Errors
    ///
    /// Returns error if required tensors are missing or have unsupported format
    pub fn from_gguf(model: &GGUFModel, data: &'a [u8]) -> Result<Self> {
        // Phase 2: Validate config at construction boundary.
        let config = ValidatedModelConfig::from_gguf(model)?.into_inner();

        // GH-704: Detect hybrid SSM architectures (Qwen3.5 Gated Delta Net) early.
        // These require a dedicated SSM inference path not yet implemented.
        // The predicate lives in `unsupported_architecture_reason` so read-only
        // tools (apr ptx-map) refuse the same files with the same words (#2399).
        if let Some(reason) = unsupported_architecture_reason(
            &config.architecture,
            model.tensors.iter().map(|t| t.name.as_str()),
        ) {
            return Err(crate::RealizarError::FormatError { reason });
        }

        // #3477: the hybrid architectures are no longer refused by the shared
        // predicate — `apr run` serves them through `Qwen35Model` /
        // `Qwen35CudaModel`. THIS loader is still the dense one, so it refuses
        // them here with a message that names the path that does load them,
        // rather than falling through to a missing `blk.0.attn_q.weight`.
        if let Some(reason) = dense_loader_refusal(
            &config.architecture,
            model.tensors.iter().map(|t| t.name.as_str()),
        ) {
            return Err(crate::RealizarError::FormatError { reason });
        }

        // M32b: refuse Mixture-of-Experts architectures with a structured,
        // contract-named error before reaching the dense-FFN tensor lookup.
        // Replaces the pre-M32 cryptic "Tensor 'blk.0.ffn_up.weight' not
        // found" surface captured by FALSIFY-QW3-MOE-FORWARD-001 in
        // contracts/qwen3-moe-forward-v1.yaml.
        // M32c.2.1: dispatch qwen3_moe arch to the MoE-aware constructor
        // (M32c.2's `from_gguf_for_moe`). Loading now succeeds end-to-end;
        // the forward path emits the contract-named UnsupportedOperation
        // when it encounters the placeholder dense FFN — see M32c.2.2 for
        // the actual MoE forward wiring. Replaces M32b's load-time refusal.
        // See contracts/qwen3-moe-forward-v1.yaml.
        let canonical_arch = crate::tensor_names::normalize_architecture(&config.architecture);
        if canonical_arch == "qwen3_moe" {
            return Self::from_gguf_for_moe(model, data);
        }

        // Token embedding - keep as f32 for efficient lookup
        let token_embedding = model.get_tensor_f32("token_embd.weight", data)?;
        // GH-278: Position embedding — standard GGUF + legacy + aprender export fallback
        let position_embedding = model
            .get_tensor_f32("position_embd.weight", data)
            .or_else(|_| model.get_tensor_f32("token_pos_embd.weight", data))
            .or_else(|_| model.get_tensor_f32("model.position_embedding.weight", data))
            .ok();

        // Load layers with quantized weight references
        let mut layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = Self::load_quantized_layer(model, data, layer_idx)?;
            layers.push(layer);
        }

        // Output norm - small, keep as f32
        let output_norm_weight = model.get_tensor_f32("output_norm.weight", data)?;
        // GH-278: Output norm bias — standard + aprender fallback
        let output_norm_bias = model
            .get_tensor_f32("output_norm.bias", data)
            .or_else(|_| model.get_tensor_f32("model.norm.bias", data))
            .ok();

        // LM head - large, keep quantized
        // Fall back to token_embd.weight for tied embeddings (Qwen2, some LLaMA variants)
        let lm_head_weight = Self::get_tensor_ref(model, data, "output.weight")
            .or_else(|_| Self::get_tensor_ref(model, data, "token_embd.weight"))?;
        let lm_head_bias = model.get_tensor_f32("output.bias", data).ok();

        Ok(Self {
            config,
            data,
            token_embedding,
            position_embedding,
            layers,
            moe_layers: Vec::new(),
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }

    /// M32c.2: Load a `qwen3_moe`-arch GGUF, populating both the
    /// non-FFN dense fields and the per-layer MoE expert tensor
    /// descriptors. This is the qwen3_moe-aware sibling of
    /// `from_gguf` — call it instead when the architecture has been
    /// canonicalized to `qwen3_moe`.
    ///
    /// Forward dispatch is NOT yet wired (M32c.2.1). This
    /// constructor exists so M32c.2 can prove that the
    /// load infrastructure (M32c.1's `load_qwen3_moe_layer` +
    /// shared dense-FFN-skip path) works end-to-end against the
    /// real 17.3 GB Qwen3-Coder GGUF without going through the
    /// M32b load-time refusal.
    ///
    /// # Arguments
    /// * `model` - Parsed GGUF model. The caller MUST have verified
    ///   that `tensor_names::normalize_architecture(&config.architecture) == "qwen3_moe"`.
    /// * `data` - Memory-mapped file data (zero-copy).
    ///
    /// # Errors
    /// Returns an error if any of:
    /// - SSM tensor names appear (mutually exclusive with MoE)
    /// - Required non-FFN tensors are missing (token_embd, attn_*,
    ///   output_norm, output)
    /// - Any MoE tensor declared by `tensor-names-v1` v1.1.0 is
    ///   missing for any layer
    ///
    /// On success, every `layers[i]` has placeholder dense FFN
    /// `QuantizedTensorRef`s (offset=0, byte_size=0, num_elements=0,
    /// qtype=GGUF_TYPE_F32) — consumers MUST check
    /// `moe_layers[i].is_some()` before attempting any dense FFN
    /// dequantization.
    pub fn from_gguf_for_moe(model: &GGUFModel, data: &'a [u8]) -> Result<Self> {
        let config = ValidatedModelConfig::from_gguf(model)?.into_inner();

        let canonical_arch = crate::tensor_names::normalize_architecture(&config.architecture);
        if canonical_arch != "qwen3_moe" {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!(
                    "from_gguf_for_moe: architecture '{}' (canonical '{}') is not qwen3_moe — \
                     caller should dispatch to from_gguf instead",
                    config.architecture, canonical_arch
                ),
            });
        }

        let has_ssm = model
            .tensors
            .iter()
            .any(|t| t.name.contains("ssm_") || t.name.contains("ssm."));
        if has_ssm {
            return Err(crate::RealizarError::FormatError {
                reason: format!(
                    "Architecture '{}' has both qwen3_moe arch tag AND SSM tensors — \
                     unsupported hybrid configuration",
                    config.architecture
                ),
            });
        }

        let token_embedding = model.get_tensor_f32("token_embd.weight", data)?;
        let position_embedding = model
            .get_tensor_f32("position_embd.weight", data)
            .or_else(|_| model.get_tensor_f32("token_pos_embd.weight", data))
            .or_else(|_| model.get_tensor_f32("model.position_embedding.weight", data))
            .ok();

        let mut layers = Vec::with_capacity(config.num_layers);
        let mut moe_layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            layers.push(Self::load_quantized_layer_moe_skeleton(
                model, data, layer_idx,
            )?);
            moe_layers.push(Some(crate::gguf::qwen3_moe_load::load_qwen3_moe_layer(
                model, data, layer_idx,
            )?));
        }

        let output_norm_weight = model.get_tensor_f32("output_norm.weight", data)?;
        let output_norm_bias = model
            .get_tensor_f32("output_norm.bias", data)
            .or_else(|_| model.get_tensor_f32("model.norm.bias", data))
            .ok();

        let lm_head_weight = Self::get_tensor_ref(model, data, "output.weight")
            .or_else(|_| Self::get_tensor_ref(model, data, "token_embd.weight"))?;
        let lm_head_bias = model.get_tensor_f32("output.bias", data).ok();

        Ok(Self {
            config,
            data,
            token_embedding,
            position_embedding,
            layers,
            moe_layers,
            output_norm_weight,
            output_norm_bias,
            lm_head_weight,
            lm_head_bias,
        })
    }

    /// M32c.2 helper: load the non-FFN portion of a transformer layer.
    /// Dense FFN fields are stubbed with empty `QuantizedTensorRef`
    /// placeholders — the caller MUST populate `moe_layers[i]` for
    /// these layers via `load_qwen3_moe_layer`.
    fn load_quantized_layer_moe_skeleton(
        model: &GGUFModel,
        data: &[u8],
        layer_idx: usize,
    ) -> Result<QuantizedGGUFTransformerLayer> {
        let prefix = format!("blk.{layer_idx}");

        let attn_norm_weight = model.get_tensor_f32(&format!("{prefix}.attn_norm.weight"), data)?;
        let attn_norm_bias = model
            .get_tensor_f32(&format!("{prefix}.attn_norm.bias"), data)
            .or_else(|_| model.get_tensor_f32(&format!("{prefix}.input_layernorm.bias"), data))
            .ok();

        // qwen3_moe uses separate Q/K/V (llama-style); fused QKV is unused for this arch.
        let q = Self::get_tensor_ref(model, data, &format!("{prefix}.attn_q.weight"))?;
        let k = Self::get_tensor_ref(model, data, &format!("{prefix}.attn_k.weight"))?;
        let v = Self::get_tensor_ref(model, data, &format!("{prefix}.attn_v.weight"))?;
        let q_bias = model
            .get_tensor_f32(&format!("{prefix}.attn_q.bias"), data)
            .ok();
        let k_bias = model
            .get_tensor_f32(&format!("{prefix}.attn_k.bias"), data)
            .ok();
        let v_bias = model
            .get_tensor_f32(&format!("{prefix}.attn_v.bias"), data)
            .ok();
        let qkv_bias = match (q_bias, k_bias, v_bias) {
            (Some(qb), Some(kb), Some(vb)) => {
                let mut combined = Vec::with_capacity(qb.len() + kb.len() + vb.len());
                combined.extend_from_slice(&qb);
                combined.extend_from_slice(&kb);
                combined.extend_from_slice(&vb);
                Some(combined)
            },
            _ => None,
        };
        let qkv_weight = QKVWeights::Separate { q, k, v };

        let attn_output_weight =
            Self::get_tensor_ref(model, data, &format!("{prefix}.attn_output.weight"))?;
        let attn_output_bias = model
            .get_tensor_f32(&format!("{prefix}.attn_output.bias"), data)
            .ok();

        // FFN fields stubbed — see moe_layers field for the real expert tensors.
        let dense_ffn_placeholder = QuantizedTensorRef {
            offset: 0,
            byte_size: 0,
            num_elements: 0,
            qtype: GGUF_TYPE_F32,
        };

        let ffn_norm_weight = model
            .get_tensor_f32(&format!("{prefix}.ffn_norm.weight"), data)
            .ok();
        let ffn_norm_bias = model
            .get_tensor_f32(&format!("{prefix}.ffn_norm.bias"), data)
            .or_else(|_| {
                model.get_tensor_f32(&format!("{prefix}.post_attention_layernorm.bias"), data)
            })
            .ok();

        let attn_q_norm_weight = model
            .get_tensor_f32(&format!("{prefix}.attn_q_norm.weight"), data)
            .ok();
        let attn_k_norm_weight = model
            .get_tensor_f32(&format!("{prefix}.attn_k_norm.weight"), data)
            .ok();

        // PMAT-810: Gemma2 post-attention / post-FFN RMSNorm (absent elsewhere).
        let post_attn_norm_weight = model
            .get_tensor_f32(&format!("{prefix}.post_attention_norm.weight"), data)
            .ok();
        let post_ffw_norm_weight = model
            .get_tensor_f32(&format!("{prefix}.post_ffw_norm.weight"), data)
            .ok();

        Ok(QuantizedGGUFTransformerLayer {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_up_weight: dense_ffn_placeholder.clone(),
            ffn_up_bias: None,
            ffn_down_weight: dense_ffn_placeholder,
            ffn_down_bias: None,
            ffn_gate_weight: None,
            ffn_gate_bias: None,
            ffn_norm_weight,
            ffn_norm_bias,
            attn_q_norm_weight,
            attn_k_norm_weight,
            post_attn_norm_weight,
            post_ffw_norm_weight,
        })
    }

    /// Calculate byte size for a quantized tensor based on its type and dimensions.
    ///
    /// #3432: the sizes come from the ONE ggml type table
    /// ([`crate::gguf::ggml_type_table`]) rather than an 11-arm match that listed
    /// only the types its authors had met. The match refused IQ2_XXS (16) and
    /// IQ4_XS (23) — the types real unsloth Qwen3.5 GGUFs use — at *size* time,
    /// so `apr run` could not load the file at all. Sizing a type is not a claim
    /// that a kernel exists for it; `fused_matmul_into` still fails loud for any
    /// qtype it cannot dequantize.
    ///
    /// # Errors
    ///
    /// `UnsupportedOperation` when the ggml table has no layout for `qtype`, or
    /// when the size overflows `usize`.
    fn tensor_byte_size(qtype: u32, num_elements: usize, dims: &[u64]) -> Result<usize> {
        // Types the pre-#3432 match sized from the FLAT element count. A 2-D
        // tensor whose rows are not whole blocks gets a different (smaller)
        // number this way than row-padding gives, so these ids keep their
        // historical sizing rather than silently re-sizing existing models.
        // Every other block type is row-padded, as `k_quant_bytes` always did.
        const LEGACY_FLAT_SIZED: &[u32] = &[
            GGUF_TYPE_Q4_0,
            GGUF_TYPE_Q4_1,
            GGUF_TYPE_Q5_0,
            GGUF_TYPE_Q8_0,
            GGUF_TYPE_Q2_K,
        ];

        let scalar = super::ggml_type_table::traits(qtype).is_some_and(|t| t.blck_size == 1);
        let sized = if scalar || LEGACY_FLAT_SIZED.contains(&qtype) {
            super::ggml_type_table::flat_byte_size(qtype, num_elements)
        } else {
            super::ggml_type_table::byte_size(qtype, dims)
        };
        sized.map_err(|reason| RealizarError::UnsupportedOperation {
            operation: "tensor_byte_size".to_string(),
            reason,
        })
    }

    /// PAR-058: Auto-correct qtype when header claims wrong type.
    fn resolve_qtype(
        name: &str,
        claimed_qtype: u32,
        byte_size: usize,
        num_elements: usize,
        offset: usize,
        data_len: usize,
    ) -> (usize, u32) {
        if offset + byte_size <= data_len {
            return (byte_size, claimed_qtype);
        }
        let avail = data_len.saturating_sub(offset);
        let q4_0_size = num_elements.div_ceil(32) * 18;
        if q4_0_size <= avail && q4_0_size > 0 {
            eprintln!(
                "[PAR-058-RESOLVED] Tensor '{name}' qtype mismatch: header says {claimed_qtype} but byte size suggests Q4_0. Using Q4_0."
            );
            return (q4_0_size, GGUF_TYPE_Q4_0);
        }
        let q8_0_size = num_elements.div_ceil(32) * 34;
        if q8_0_size <= avail && q8_0_size > 0 {
            eprintln!(
                "[PAR-058-RESOLVED] Tensor '{name}' qtype mismatch: header says {claimed_qtype} but byte size suggests Q8_0. Using Q8_0."
            );
            return (q8_0_size, GGUF_TYPE_Q8_0);
        }
        (byte_size, claimed_qtype)
    }

    /// Get tensor reference (offset + size + qtype) without dequantization
    pub(crate) fn get_tensor_ref(
        model: &GGUFModel,
        data: &[u8],
        name: &str,
    ) -> Result<QuantizedTensorRef> {
        let tensor = model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: format!("Tensor '{}' not found", name),
            })?;

        let num_elements: usize = tensor.dims.iter().map(|&d| d as usize).product();
        let offset = model.tensor_data_start + tensor.offset as usize;
        let byte_size = Self::tensor_byte_size(tensor.qtype, num_elements, &tensor.dims)?;
        let (byte_size, actual_qtype) = Self::resolve_qtype(
            name,
            tensor.qtype,
            byte_size,
            num_elements,
            offset,
            data.len(),
        );

        if offset + byte_size > data.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "Tensor '{}' data range [{}, {}) exceeds file size {}",
                    name,
                    offset,
                    offset + byte_size,
                    data.len()
                ),
            });
        }

        Ok(QuantizedTensorRef {
            offset,
            byte_size,
            num_elements,
            qtype: actual_qtype,
        })
    }

    /// Load a single quantized transformer layer
    fn load_quantized_layer(
        model: &GGUFModel,
        data: &[u8],
        layer_idx: usize,
    ) -> Result<QuantizedGGUFTransformerLayer> {
        let prefix = format!("blk.{}", layer_idx);

        // Attention norm - small, keep as f32
        let attn_norm_weight =
            model.get_tensor_f32(&format!("{}.attn_norm.weight", prefix), data)?;
        // GH-278: Attention norm bias — standard GGUF + aprender fallback
        let attn_norm_bias = model
            .get_tensor_f32(&format!("{}.attn_norm.bias", prefix), data)
            .or_else(|_| model.get_tensor_f32(&format!("{}.input_layernorm.bias", prefix), data))
            .ok();

        // QKV - large, keep quantized
        // Try fused first (phi-2 style), fall back to separate (llama style)
        let (qkv_weight, qkv_bias) = if let Ok(fused) =
            Self::get_tensor_ref(model, data, &format!("{}.attn_qkv.weight", prefix))
        {
            // phi-2 style: fused QKV tensor
            let bias = model
                .get_tensor_f32(&format!("{}.attn_qkv.bias", prefix), data)
                .ok();
            (QKVWeights::Fused(fused), bias)
        } else {
            // llama style: separate Q, K, V tensors
            let q = Self::get_tensor_ref(model, data, &format!("{}.attn_q.weight", prefix))?;
            let k = Self::get_tensor_ref(model, data, &format!("{}.attn_k.weight", prefix))?;
            let v = Self::get_tensor_ref(model, data, &format!("{}.attn_v.weight", prefix))?;

            // Try to get biases (llama usually doesn't have them)
            let q_bias = model
                .get_tensor_f32(&format!("{}.attn_q.bias", prefix), data)
                .ok();
            let k_bias = model
                .get_tensor_f32(&format!("{}.attn_k.bias", prefix), data)
                .ok();
            let v_bias = model
                .get_tensor_f32(&format!("{}.attn_v.bias", prefix), data)
                .ok();

            let bias = match (q_bias, k_bias, v_bias) {
                (Some(qb), Some(kb), Some(vb)) => {
                    let mut combined = Vec::with_capacity(qb.len() + kb.len() + vb.len());
                    combined.extend_from_slice(&qb);
                    combined.extend_from_slice(&kb);
                    combined.extend_from_slice(&vb);
                    Some(combined)
                },
                _ => None,
            };

            (QKVWeights::Separate { q, k, v }, bias)
        };

        // Attention output - large, keep quantized
        let attn_output_weight =
            Self::get_tensor_ref(model, data, &format!("{}.attn_output.weight", prefix))?;
        let attn_output_bias = model
            .get_tensor_f32(&format!("{}.attn_output.bias", prefix), data)
            .ok();

        // FFN - large, keep quantized
        let ffn_up_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_up.weight", prefix))?;
        // GH-278: FFN biases — standard GGUF + aprender fallback
        let ffn_up_bias = model
            .get_tensor_f32(&format!("{}.ffn_up.bias", prefix), data)
            .or_else(|_| model.get_tensor_f32(&format!("{}.mlp.up_proj.bias", prefix), data))
            .ok();
        let ffn_down_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_down.weight", prefix))?;
        let ffn_down_bias = model
            .get_tensor_f32(&format!("{}.ffn_down.bias", prefix), data)
            .or_else(|_| model.get_tensor_f32(&format!("{}.mlp.down_proj.bias", prefix), data))
            .ok();

        // FFN gate - SwiGLU models like LLaMA have this
        let ffn_gate_weight =
            Self::get_tensor_ref(model, data, &format!("{}.ffn_gate.weight", prefix)).ok();
        let ffn_gate_bias = model
            .get_tensor_f32(&format!("{}.ffn_gate.bias", prefix), data)
            .ok();

        // FFN norm - LLaMA-style pre-FFN layer norm
        let ffn_norm_weight = model
            .get_tensor_f32(&format!("{}.ffn_norm.weight", prefix), data)
            .ok();
        // GH-278: FFN norm bias — standard GGUF + aprender fallback
        let ffn_norm_bias = model
            .get_tensor_f32(&format!("{}.ffn_norm.bias", prefix), data)
            .or_else(|_| {
                model.get_tensor_f32(&format!("{}.post_attention_layernorm.bias", prefix), data)
            })
            .ok();

        // GH-279: QK norm weights (Qwen3 per-head RMSNorm on Q and K)
        let attn_q_norm_weight = model
            .get_tensor_f32(&format!("{}.attn_q_norm.weight", prefix), data)
            .ok();
        let attn_k_norm_weight = model
            .get_tensor_f32(&format!("{}.attn_k_norm.weight", prefix), data)
            .ok();

        // PMAT-810: Gemma2 post-attention / post-FFN RMSNorm (absent for LLaMA/
        // Qwen/Gemma1). Gemma2 sandwiches each block:
        //   x = x + post_attn_norm(attn(attn_norm(x)))
        //   h = h + post_ffw_norm(ffn(ffn_norm(h)))
        let post_attn_norm_weight = model
            .get_tensor_f32(&format!("{}.post_attention_norm.weight", prefix), data)
            .ok();
        let post_ffw_norm_weight = model
            .get_tensor_f32(&format!("{}.post_ffw_norm.weight", prefix), data)
            .ok();

        Ok(QuantizedGGUFTransformerLayer {
            attn_norm_weight,
            attn_norm_bias,
            qkv_weight,
            qkv_bias,
            attn_output_weight,
            attn_output_bias,
            ffn_up_weight,
            ffn_up_bias,
            ffn_down_weight,
            ffn_down_bias,
            ffn_gate_weight,
            ffn_gate_bias,
            ffn_norm_weight,
            ffn_norm_bias,
            attn_q_norm_weight,
            attn_k_norm_weight,
            post_attn_norm_weight,
            post_ffw_norm_weight,
        })
    }
}

#[cfg(test)]
mod tensor_byte_size_tests {
    use super::*;
    use crate::gguf::types::GGUF_TYPE_Q3_K;

    // PMAT-788: F16 (ggml type 1) is the most common GGUF weight format. Its
    // byte-size arm was missing, so `from_gguf` crashed on every F16 GGUF on
    // both CPU and GPU paths before the fail-closed quant gate could route it.
    #[test]
    fn f16_byte_size_is_two_bytes_per_element() {
        let n = 1024;
        let got = QuantizedGGUFTransformer::tensor_byte_size(GGUF_TYPE_F16, n, &[1024])
            .expect("F16 must have a known byte size (PMAT-788)");
        assert_eq!(got, n * 2, "F16 is 2 bytes/element");
    }

    #[test]
    fn bf16_byte_size_is_two_bytes_per_element() {
        let n = 768;
        let got = QuantizedGGUFTransformer::tensor_byte_size(GGUF_TYPE_BF16, n, &[768])
            .expect("BF16 must have a known byte size");
        assert_eq!(got, n * 2);
    }

    // #3432: Q3_K (type 11) HAS a defined size in ggml's type_traits table
    // (256-element super-block, 110 bytes), so sizing it is not a claim that the
    // forward can run it. The kernel refusal is downstream and unchanged:
    // `fused_matmul_into.rs` fails loud with "Unsupported quantization type" for
    // any qtype it has no dequantizer for. Before #3432 this arm was missing and
    // the loader refused at *size* time — which also refused IQ2_XXS/IQ4_XS, the
    // types real unsloth Qwen3.5 files use.
    #[test]
    fn q3_k_byte_size_is_the_ggml_super_block_size() {
        let got = QuantizedGGUFTransformer::tensor_byte_size(GGUF_TYPE_Q3_K, 256, &[256])
            .expect("Q3_K has a defined ggml block size (#3432)");
        assert_eq!(got, 110, "Q3_K is 110 bytes per 256-element super-block");
    }

    /// #3432 (RED first): real unsloth Qwen3.5 GGUFs are IQ2_XXS (type 16) and
    /// IQ4_XS (type 23). The 11-arm match refused both with "Unsupported
    /// quantization type", so `apr run` could not even load the file.
    #[test]
    fn iq_quants_used_by_real_unsloth_files_are_sized() {
        // [rows, cols] with cols a whole number of 256-element super-blocks.
        let dims = [4u64, 512];
        let n = 4 * 512;
        let iq2_xxs = QuantizedGGUFTransformer::tensor_byte_size(16, n, &dims)
            .expect("IQ2_XXS (type 16) must be sized — Qwen3.5-0.8B-UD-IQ2_XXS.gguf (#3432)");
        assert_eq!(iq2_xxs, 4 * 2 * 66, "IQ2_XXS: 66 bytes per super-block");

        let iq4_xs = QuantizedGGUFTransformer::tensor_byte_size(23, n, &dims)
            .expect("IQ4_XS (type 23) must be sized — Qwen3.5-0.8B-IQ4_XS.gguf (#3432)");
        assert_eq!(iq4_xs, 4 * 2 * 136, "IQ4_XS: 136 bytes per super-block");
    }

    /// The whole point of a table is that an id it does not know still fails
    /// loud, rather than being sized as something else.
    #[test]
    fn an_unknown_type_id_is_still_refused() {
        let err = QuantizedGGUFTransformer::tensor_byte_size(9_999, 256, &[256])
            .expect_err("an id outside the ggml table must not be sized");
        let msg = err.to_string();
        assert!(
            msg.contains("9999"),
            "the refusal must name the id it was given, got: {msg}"
        );
    }

    /// Regression snapshot: every id the 11-arm match supported keeps EXACTLY
    /// the byte count it returned before the table (#3432). The dims are
    /// deliberately NOT block-aligned on the last axis, which is where the
    /// legacy flat-element-count arms (Q4_0/Q4_1/Q5_0/Q8_0/Q2_K) and the
    /// row-padded K-quant arms disagree — so this pins both semantics.
    #[test]
    fn previously_supported_ids_keep_their_byte_sizes() {
        let dims = [4u64, 1000];
        let n = 4000usize;
        let cases: [(u32, usize); 11] = [
            (GGUF_TYPE_F32, 16_000),
            (GGUF_TYPE_F16, 8_000),
            (GGUF_TYPE_BF16, 8_000),
            (GGUF_TYPE_Q4_0, 2_250),
            (GGUF_TYPE_Q4_1, 2_500),
            (GGUF_TYPE_Q5_0, 2_750),
            (GGUF_TYPE_Q8_0, 4_250),
            (GGUF_TYPE_Q2_K, 1_344),
            (GGUF_TYPE_Q4_K, 2_304),
            (GGUF_TYPE_Q5_K, 2_816),
            (GGUF_TYPE_Q6_K, 3_360),
        ];
        for (qtype, expected) in cases {
            let got = QuantizedGGUFTransformer::tensor_byte_size(qtype, n, &dims)
                .expect("previously supported qtype must stay supported");
            assert_eq!(got, expected, "qtype {qtype} byte size changed");
        }
    }

    /// The 1-D snapshot (vectors, norms) for the same ids.
    #[test]
    fn previously_supported_ids_keep_their_1d_byte_sizes() {
        let dims = [1000u64];
        let n = 1000usize;
        let cases: [(u32, usize); 4] = [
            (GGUF_TYPE_F32, 4_000),
            (GGUF_TYPE_Q4_0, 32 * 18),
            (GGUF_TYPE_Q2_K, 4 * 84),
            (GGUF_TYPE_Q4_K, 4 * 144),
        ];
        for (qtype, expected) in cases {
            let got = QuantizedGGUFTransformer::tensor_byte_size(qtype, n, &dims)
                .expect("previously supported qtype must stay supported");
            assert_eq!(got, expected, "qtype {qtype} 1-D byte size changed");
        }
    }
}

#[cfg(test)]
mod unsupported_architecture_tests {
    use super::{
        dense_loader_refusal, hybrid_forward_handles, hybrid_gpu_quant_refusal,
        unsupported_architecture_reason,
    };

    // #3477 (operator ruling 2026-09-19): the hybrid forward now exists on both
    // backends for `qwen35` — `forward_qwen35` (CPU, #3091) and
    // `Qwen35CudaModel` (GPU, #3090). The shared predicate must stop refusing
    // it, or every tool that asks it (apr qa, apr parity, apr ptx-map) keeps
    // contradicting the runtime.
    #[test]
    fn qwen35_gated_delta_net_tensors_are_admitted() {
        assert_eq!(
            unsupported_architecture_reason(
                "qwen35",
                [
                    "token_embd.weight",
                    "blk.0.ssm_conv1d.weight",
                    "blk.0.attn_q.weight",
                ],
            ),
            None,
            "the hybrid forward runs qwen35 on both backends (#3090/#3091)"
        );
        assert_eq!(
            unsupported_architecture_reason("qwen35", ["blk.0.ssm.a"]),
            None,
            "the dotted `ssm.` spelling is the same architecture"
        );
    }

    // dogfood-0.63.0 #2399 finding 2: the SSM refusal must still be answerable
    // from tensor names alone for the architectures NO forward handles, so
    // read-only tools get the same verdict `from_gguf` gives.
    #[test]
    fn an_ssm_architecture_with_no_forward_is_still_refused() {
        let reason = unsupported_architecture_reason(
            "mamba",
            [
                "token_embd.weight",
                "blk.0.ssm_conv1d.weight",
                "blk.0.attn_q.weight",
            ],
        )
        .expect("an SSM architecture with no forward must be refused");
        assert!(
            reason.contains("mamba") && reason.contains("SSM/Gated Delta Net"),
            "refusal must name the architecture and the reason, got: {reason}"
        );
        assert!(reason.contains("#3090"), "must mention GPU #3090");
        assert!(reason.contains("#3091"), "must mention CPU #3091");
        assert!(
            reason.contains("blk.0.ssm_conv1d.weight"),
            "must mention tensor name"
        );
        // #3091 gave the CPU a Gated DeltaNet forward and #3090 the GPU one;
        // the old sentence claimed neither backend implements it at all.
        assert!(
            !reason.contains("NEITHER the CPU nor the GPU"),
            "the 'neither backend implements Gated DeltaNet' claim is false since #3091: {reason}"
        );
        // The dotted spelling (`blk.0.ssm.a`) is the other form seen in the wild.
        assert!(
            unsupported_architecture_reason("mamba", ["blk.0.ssm.a"]).is_some(),
            "`ssm.` spelling must be refused too"
        );
    }

    // The admission mirrors the runtime dispatch, which compares the GGUF
    // architecture to the literal "qwen35" (infer/inference_result.rs). A
    // spelling the runtime does not dispatch must keep the refusal, or the
    // tooling would promise a load `apr run` cannot perform.
    #[test]
    fn unsupported_architecture_handled_set_is_the_runtime_dispatch_literal() {
        assert!(hybrid_forward_handles("qwen35"));
        for other in ["qwen3_5", "qwen3.5", "QWEN35", "mamba", "qwen2", "llama"] {
            assert!(
                !hybrid_forward_handles(other),
                "'{other}' reaches no hybrid forward"
            );
            assert!(
                unsupported_architecture_reason(other, ["blk.0.ssm_a"]).is_some(),
                "'{other}' carries SSM tensors and has no forward — it must be refused"
            );
        }
    }

    // The DENSE loader is not the hybrid path: it must refuse qwen35 with a
    // message naming the path that does load it, instead of falling through to
    // a missing `blk.0.attn_q.weight`.
    #[test]
    fn unsupported_architecture_dense_loader_still_refuses_the_hybrid() {
        let reason = dense_loader_refusal("qwen35", ["token_embd.weight", "blk.0.ssm_a"])
            .expect("the dense loader cannot load a hybrid GGUF");
        assert!(
            reason.contains("Qwen35Model") && reason.contains("forward_qwen35"),
            "the refusal must name the loader that DOES work: {reason}"
        );
        assert!(
            !reason.contains("NEITHER the CPU nor the GPU"),
            "the dense loader's limits are not the runtime's: {reason}"
        );
        assert_eq!(
            dense_loader_refusal("qwen2", ["blk.0.attn_q.weight"]),
            None,
            "a dense model must still load through the dense loader"
        );
        assert_eq!(
            dense_loader_refusal("mamba", ["blk.0.ssm_a"]),
            None,
            "an architecture no forward handles is refused by the shared predicate, not here"
        );
    }

    // PMAT-781/783/785: the hybrid GPU upload decodes an unrecognized GGML type
    // as Q4_K. The dense quant gate cannot see the Gated DeltaNet tensors (the
    // qwen35 base model carries an empty layer list), so the header-level gate
    // has to judge them by name.
    #[test]
    fn unsupported_architecture_hybrid_quant_gate_judges_the_deltanet_tensors() {
        let reason = hybrid_gpu_quant_refusal(
            "qwen35",
            [
                ("token_embd.weight", 12u32),
                ("blk.0.ssm_alpha.weight", 11), // Q3_K: no GPU GEMV kernel
                ("blk.0.ffn_down.weight", 12),
            ],
        )
        .expect("a Q3_K DeltaNet projection has no GPU kernel — it must be refused");
        assert!(
            reason.contains("blk.0.ssm_alpha.weight") && reason.contains("11"),
            "the refusal must name the tensor and its GGML type: {reason}"
        );
        assert_eq!(
            hybrid_gpu_quant_refusal(
                "qwen35",
                [
                    ("blk.0.attn_qkv.weight", 8u32),
                    ("blk.0.ssm_alpha.weight", 12),
                    ("blk.0.ssm_out.weight", 14),
                ],
            ),
            None,
            "Q8_0/Q4_K/Q6_K all have verified GPU GEMV kernels"
        );
        assert_eq!(
            hybrid_gpu_quant_refusal("qwen2", [("blk.0.attn_q.weight", 11u32)]),
            None,
            "a dense model is judged by the dense quant gate, not this one"
        );
    }

    // A plain transformer must NOT be refused — a predicate that says "no" to
    // everything would pass the two tests above while breaking every model.
    #[test]
    fn standard_transformer_tensors_are_accepted() {
        assert_eq!(
            unsupported_architecture_reason(
                "qwen2",
                [
                    "token_embd.weight",
                    "blk.0.attn_q.weight",
                    "blk.0.ffn_down.weight",
                    "output.weight",
                ],
            ),
            None,
            "a dense transformer must load"
        );
    }
}

include!("transformer_quantized_layer_field.rs");
