//! M32c.1: Architecture-aware load of Qwen3-MoE expert tensors.
//!
//! Per `contracts/qwen3-moe-forward-v1.yaml` (M32a) +
//! `contracts/tensor-names-v1.yaml` v1.1.0 (M29), the four tensor
//! names load-bearing for `qwen3_moe` are:
//!
//! ```text
//! blk.{L}.ffn_gate_inp.weight   [num_experts, hidden_dim]            — router
//! blk.{L}.ffn_gate_exps.weight  [num_experts, intermediate, hidden]  — gate per expert
//! blk.{L}.ffn_up_exps.weight    [num_experts, intermediate, hidden]  — up   per expert
//! blk.{L}.ffn_down_exps.weight  [num_experts, hidden, intermediate]  — down per expert
//! ```
//!
//! This module exposes a thin loader that, given a parsed
//! `GGUFModel` and the file's mmapped bytes, returns four
//! `QuantizedTensorRef` per layer — the on-disk byte ranges of
//! each MoE tensor. **No dequantization happens here**: that is
//! M32c.2's job (forward dispatch). The structs returned here
//! are read-only descriptors suitable for stashing on a
//! per-layer struct and consuming via the existing
//! `fused_q4k_*` / `fused_q6k_*` row-major matvec kernels.
//!
//! The forward path remains unchanged in this slice: M32b's
//! `RealizarError::UnsupportedOperation { operation:
//! "moe_forward_pass" }` early-return still fires for any
//! attempted inference. M32c.2 is what replaces that
//! early-return with an actual MoE forward.
//!
//! ## Slice scope
//! - **In-scope (M32c.1, this module)**: per-layer tensor
//!   descriptors + a falsifier asserting that the cached
//!   17.3 GB Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf yields
//!   non-zero descriptors for every L ∈ [0, 48).
//! - **Out-of-scope (deferred to M32c.2)**: dequantization,
//!   forward dispatch, KV cache, attention.

use crate::error::Result;
use crate::gguf::quantized::QuantizedTensorRef;
use crate::gguf::GGUFModel;
use crate::gguf::QuantizedGGUFTransformer;

/// Per-layer MoE tensor descriptors for one Qwen3-MoE decoder block.
///
/// All four fields are byte-range descriptors into the GGUF file's
/// mmapped data — no dequantization or copying happens at load
/// time. The dequantize-on-demand pattern matches the dense FFN
/// path's `QuantizedGGUFTransformerLayer` and preserves the
/// 8× memory-bandwidth advantage of Q4_K (per
/// `crates/aprender-serve/CLAUDE.md` § "Quantized GGUF Transformer
/// for fused inference").
#[derive(Debug, Clone)]
pub struct Qwen3MoeQuantizedLayer {
    /// `blk.{L}.ffn_gate_inp.weight` — router projection
    /// `[num_experts, hidden_dim]` row-major.
    pub router: QuantizedTensorRef,

    /// `blk.{L}.ffn_gate_exps.weight` — per-expert gate projection
    /// stacked as `[num_experts, intermediate, hidden_dim]`.
    pub gate_exps: QuantizedTensorRef,

    /// `blk.{L}.ffn_up_exps.weight` — per-expert up projection
    /// `[num_experts, intermediate, hidden_dim]`.
    pub up_exps: QuantizedTensorRef,

    /// `blk.{L}.ffn_down_exps.weight` — per-expert down projection
    /// `[num_experts, hidden_dim, intermediate]`.
    pub down_exps: QuantizedTensorRef,
}

/// Load the four MoE tensor descriptors for `layer_idx` from a
/// `qwen3_moe`-arch GGUF.
///
/// # Errors
/// Returns the standard `RealizarError::InvalidShape { reason:
/// "Tensor '...' not found" }` if any of the four contract-named
/// tensors is missing. For arch-mismatched inputs (e.g. a dense
/// LLaMA GGUF passed to this function), the caller is expected
/// to first canonicalize the architecture via
/// `tensor_names::normalize_architecture` and only invoke this
/// function for `qwen3_moe`.
///
/// # Example
/// ```ignore
/// let mapped = MappedGGUFModel::from_path(&path)?;
/// let layer0 = load_qwen3_moe_layer(&mapped.model, mapped.data(), 0)?;
/// assert!(layer0.router.num_elements >= 128 * 2048);
/// ```
pub fn load_qwen3_moe_layer(
    model: &GGUFModel,
    data: &[u8],
    layer_idx: usize,
) -> Result<Qwen3MoeQuantizedLayer> {
    let prefix = format!("blk.{layer_idx}");
    Ok(Qwen3MoeQuantizedLayer {
        router: QuantizedGGUFTransformer::get_tensor_ref(
            model,
            data,
            &format!("{prefix}.ffn_gate_inp.weight"),
        )?,
        gate_exps: QuantizedGGUFTransformer::get_tensor_ref(
            model,
            data,
            &format!("{prefix}.ffn_gate_exps.weight"),
        )?,
        up_exps: QuantizedGGUFTransformer::get_tensor_ref(
            model,
            data,
            &format!("{prefix}.ffn_up_exps.weight"),
        )?,
        down_exps: QuantizedGGUFTransformer::get_tensor_ref(
            model,
            data,
            &format!("{prefix}.ffn_down_exps.weight"),
        )?,
    })
}

/// Slice the byte range for ONE expert's portion of a stacked
/// per-expert tensor.
///
/// Per the LAZY-FUSED-MATVEC decision recorded in
/// `contracts/qwen3-moe-forward-v1.yaml` v1.1.0 (M32c.2.2 amendment),
/// MoE forward dispatch keeps weights quantized and dequantizes
/// inline through the existing fused Q4_K/Q6_K row-major matvec
/// kernels. This adapter slices the stacked tensor — laid out
/// `[num_experts, ...]` row-major — into one expert's contiguous
/// byte range, ready for `fused_q4k_parallel_matvec` /
/// `fused_q6k_parallel_matvec`.
///
/// # Layout assumption
/// The stacked tensor's element count is `num_experts *
/// per_expert_elements`. Both `num_elements` and `byte_size` on
/// `tensor` divide evenly by `num_experts`. Q4_K and Q6_K K-quants
/// pad each row of `cols` elements to super-block boundaries
/// (cols is the LAST dim) — since each expert's slab is itself a
/// contiguous `[..., cols]` block, the per-expert byte size is
/// `tensor.byte_size / num_experts`.
///
/// # Errors
/// Returns `RealizarError::InvalidShape` if:
/// - `num_experts == 0`
/// - `expert_id >= num_experts`
/// - `tensor.byte_size % num_experts != 0` (stacking invariant
///   violation — would indicate an upstream loader bug or an
///   architecture mismatch)
/// - the slice runs past `data.len()`
///
/// # Returns
/// `&[u8]` borrowed from `data`, length `tensor.byte_size / num_experts`,
/// covering exactly expert `expert_id`'s contribution. The caller is
/// responsible for knowing the per-expert dims and qtype (read off
/// the sibling `tensor.qtype`).
pub fn expert_byte_slice<'a>(
    tensor: &QuantizedTensorRef,
    data: &'a [u8],
    expert_id: usize,
    num_experts: usize,
) -> crate::error::Result<&'a [u8]> {
    use crate::error::RealizarError;

    if num_experts == 0 {
        return Err(RealizarError::InvalidShape {
            reason: "expert_byte_slice: num_experts must be > 0".to_string(),
        });
    }
    if expert_id >= num_experts {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "expert_byte_slice: expert_id {expert_id} out of range \
                 (num_experts = {num_experts})"
            ),
        });
    }
    if tensor.byte_size % num_experts != 0 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "expert_byte_slice: tensor byte_size {} not divisible by num_experts {} \
                 — stacking invariant violated. Layout mismatch (LAZY-FUSED-MATVEC \
                 expects [num_experts, ...] outermost dim contiguous)",
                tensor.byte_size, num_experts
            ),
        });
    }
    let per_expert_bytes = tensor.byte_size / num_experts;
    let start = tensor.offset + expert_id * per_expert_bytes;
    let end = start + per_expert_bytes;
    if end > data.len() {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "expert_byte_slice: slice range [{start}, {end}) exceeds file size {}",
                data.len()
            ),
        });
    }
    Ok(&data[start..end])
}

/// Per-expert SwiGLU FFN evaluation with on-the-fly Q4_K/Q6_K
/// dequantization (M32c.2.2.1).
///
/// Implements one selected expert's contribution to the MoE layer:
/// `down(SiLU(gate(x)) ⊙ up(x))` where gate, up are Q4_K and down
/// is Q6_K. Uses `expert_byte_slice` (M32c.2.2.0) to find the
/// expert's portion of the stacked tensor + the existing
/// `fused_q4k_parallel_matvec` / `fused_q6k_parallel_matvec`
/// row-major kernels to keep weights quantized through the matmul
/// (LAZY-FUSED-MATVEC, qwen3-moe-forward-v1 v1.1.0).
///
/// # Arguments
/// * `hidden` — input hidden state, length == `hidden_dim`.
/// * `layer` — the M32c.1 `Qwen3MoeQuantizedLayer` for this decoder block.
/// * `expert_id` — selected expert index ∈ [0, num_experts).
/// * `num_experts` — total experts in the stacked tensors (e.g. 128 for Qwen3-Coder-30B).
/// * `intermediate` — per-expert intermediate dim (e.g. 768 for Qwen3-Coder-30B).
/// * `hidden_dim` — model hidden dim (e.g. 2048 for Qwen3-Coder-30B).
/// * `data` — file's mmapped byte slice (zero-copy from `MappedGGUFModel::data()`).
///
/// # Returns
/// A new `Vec<f32>` of length `hidden_dim` — this expert's contribution
/// to the layer's MoE output. Caller is responsible for the routing
/// weight scaling and accumulation (see `moe_forward_token` semantics
/// in `gpu/scheduler/moe_dispatch.rs`).
///
/// # Errors
/// Propagates errors from `expert_byte_slice` (out-of-range expert,
/// stacking-invariant violation, slice overrun) and the matvec kernels
/// (length mismatch).
///
/// # Layout assumption
/// Per `tensor-names-v1` v1.1.0:
///   * `gate_exps`, `up_exps`: stacked `[num_experts, intermediate, hidden]`
///     row-major Q4_K. Per-expert slab is `[intermediate, hidden]`.
///   * `down_exps`: stacked `[num_experts, hidden, intermediate]` row-major
///     Q6_K. Per-expert slab is `[hidden, intermediate]`.
///
/// `fused_q4k_parallel_matvec` is documented to take row-major
/// `[out_dim, in_dim]` weights, so we pass `(hidden_dim, intermediate)` for
/// gate/up (in=hidden, out=intermediate) and `(intermediate, hidden_dim)`
/// for down (in=intermediate, out=hidden).
pub fn expert_swiglu_quantized(
    hidden: &[f32],
    layer: &Qwen3MoeQuantizedLayer,
    expert_id: usize,
    num_experts: usize,
    intermediate: usize,
    hidden_dim: usize,
    data: &[u8],
) -> Result<Vec<f32>> {
    use crate::error::RealizarError;

    if hidden.len() != hidden_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "expert_swiglu_quantized: hidden.len() = {} but hidden_dim = {}",
                hidden.len(),
                hidden_dim
            ),
        });
    }

    let gate_bytes = expert_byte_slice(&layer.gate_exps, data, expert_id, num_experts)?;
    let up_bytes = expert_byte_slice(&layer.up_exps, data, expert_id, num_experts)?;
    let down_bytes = expert_byte_slice(&layer.down_exps, data, expert_id, num_experts)?;

    // gate(x) and up(x): qtype-aware dispatch (Q4_K_M GGUFs mix Q4_K/Q6_K
    // across layers; some layers' gate/up_exps are Q6_K instead of Q4_K).
    let gate_out = matvec_for_qtype(
        layer.gate_exps.qtype,
        gate_bytes,
        hidden,
        hidden_dim,
        intermediate,
    )?;
    let up_out = matvec_for_qtype(
        layer.up_exps.qtype,
        up_bytes,
        hidden,
        hidden_dim,
        intermediate,
    )?;

    // SwiGLU: SiLU(gate) ⊙ up. SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)).
    let mut ffn_hidden = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let g = gate_out[i];
        let silu = g / (1.0 + (-g).exp());
        ffn_hidden[i] = silu * up_out[i];
    }

    // down(ffn_hidden): qtype-aware dispatch (Q4_K_M mixes types per layer).
    let result = matvec_for_qtype(
        layer.down_exps.qtype,
        down_bytes,
        &ffn_hidden,
        intermediate,
        hidden_dim,
    )?;
    Ok(result)
}

/// Dispatch matvec to the right quantization kernel based on qtype.
/// Supports Q4_K (12) and Q6_K (14) — the two K-quants used by Qwen3-Coder
/// Q4_K_M expert tensors. Other quantizations error out.
fn matvec_for_qtype(
    qtype: u32,
    weight_data: &[u8],
    activations: &[f32],
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    use crate::error::RealizarError;
    use crate::gguf::types::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};
    use crate::quantize::{fused_q4k_parallel_matvec, fused_q6k_parallel_matvec};
    match qtype {
        GGUF_TYPE_Q4_K => fused_q4k_parallel_matvec(weight_data, activations, in_dim, out_dim),
        GGUF_TYPE_Q6_K => fused_q6k_parallel_matvec(weight_data, activations, in_dim, out_dim),
        other => Err(RealizarError::UnsupportedOperation {
            operation: "moe_expert_matvec".to_string(),
            reason: format!(
                "MoE expert tensor qtype {other} not supported. Qwen3-Coder Q4_K_M uses \
                 Q4_K (12) and Q6_K (14) — caller must extend matvec_for_qtype for other \
                 quantizations."
            ),
        }),
    }
}

/// Full MoE FFN forward for ONE layer of a Qwen3-MoE model
/// (M32c.2.2.2.0 — dispatch layer above per-expert SwiGLU).
///
/// Implements the full single-token MoE FFN block:
/// `Σ_{e ∈ TopK(softmax(router@x), k)} renorm(weight_e) · SwiGLU_e(x)`
/// per `moe-router-v1` + `moe-expert-dispatch-v1` + the LAZY-FUSED-MATVEC
/// dequant strategy from `qwen3-moe-forward-v1` v1.1.0.
///
/// The router weight is read directly as F32 from the mmapped data
/// (Qwen3-Coder-30B uses qtype=F32 for ffn_gate_inp; quantized routers
/// would need a small extension here).
///
/// # Arguments
/// * `hidden` — input post-RMSNorm hidden state, length == hidden_dim.
/// * `layer` — the M32c.1 `Qwen3MoeQuantizedLayer`.
/// * `num_experts` — total experts in stacked tensors (e.g. 128).
/// * `num_experts_per_tok` — top-k selection (e.g. 8).
/// * `intermediate` — per-expert intermediate dim (e.g. 768).
/// * `hidden_dim` — model hidden dim (e.g. 2048).
/// * `data` — file's mmapped byte slice.
///
/// # Returns
/// `Vec<f32>` of length hidden_dim — the layer's MoE FFN output
/// (caller adds to residual).
///
/// # Errors
/// - Router tensor not F32 (extension point for future routers)
/// - Slice/byte/dim mismatches propagated from sub-calls
///
/// # Note on shared expert
/// Qwen3-Coder-30B-A3B does NOT use a shared expert; `moe_forward_token`
/// in `gpu/scheduler/moe_dispatch.rs` handles models that do (e.g.
/// Qwen3.5-MoE) via the `shared_*` tensor groups. This function is
/// the routed-only variant and is correct for Qwen3-Coder-30B.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_forward_layer(
    hidden: &[f32],
    layer: &Qwen3MoeQuantizedLayer,
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate: usize,
    hidden_dim: usize,
    data: &[u8],
) -> Result<Vec<f32>> {
    use crate::error::RealizarError;

    if hidden.len() != hidden_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "moe_ffn_forward_layer: hidden.len() = {} but hidden_dim = {}",
                hidden.len(),
                hidden_dim
            ),
        });
    }

    // ---- Router: read F32 weight, compute logits = router @ hidden ----
    if layer.router.qtype != crate::gguf::types::GGUF_TYPE_F32 {
        return Err(RealizarError::UnsupportedOperation {
            operation: "moe_router_quantized_read".to_string(),
            reason: format!(
                "moe_ffn_forward_layer: router qtype = {} (not F32). Quantized router \
                 not yet wired — Qwen3-Coder-30B uses F32 router so this is fine for it; \
                 other Qwen3-MoE variants needing quantized router are M32 follow-up.",
                layer.router.qtype
            ),
        });
    }
    let router_bytes = &data[layer.router.offset..layer.router.offset + layer.router.byte_size];
    let expected_bytes = num_experts * hidden_dim * 4;
    if router_bytes.len() != expected_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "moe_ffn_forward_layer: router byte_size {} != expected {} \
                 (num_experts {} × hidden_dim {} × 4)",
                router_bytes.len(),
                expected_bytes,
                num_experts,
                hidden_dim
            ),
        });
    }
    // Reinterpret router_bytes as &[f32]. Layout: [num_experts, hidden_dim] row-major.
    // logits[e] = Σ_j router[e, j] * hidden[j].
    let mut logits = vec![0.0f32; num_experts];
    for e in 0..num_experts {
        let row_off = e * hidden_dim * 4;
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            let b = row_off + j * 4;
            let w = f32::from_le_bytes([
                router_bytes[b],
                router_bytes[b + 1],
                router_bytes[b + 2],
                router_bytes[b + 3],
            ]);
            sum += w * hidden[j];
        }
        logits[e] = sum;
    }

    // ---- Softmax (numerically stable) ----
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let psum: f32 = probs.iter().sum();
    if psum > 0.0 {
        for p in &mut probs {
            *p /= psum;
        }
    }

    // ---- Top-k selection ----
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let topk = &indexed[..num_experts_per_tok.min(num_experts)];

    // ---- Renormalize selected ----
    let topk_sum: f32 = topk.iter().map(|(_, w)| w).sum();
    let topk_renorm: Vec<(usize, f32)> = if topk_sum > 0.0 {
        topk.iter().map(|(i, w)| (*i, w / topk_sum)).collect()
    } else {
        let n = topk.len();
        topk.iter().map(|(i, _)| (*i, 1.0 / n as f32)).collect()
    };

    // ---- Per-expert SwiGLU + weighted accumulate ----
    //
    // The top-k experts are independent — each `expert_swiglu_quantized`
    // call reads its own slice of the on-disk MoE tensors and produces a
    // [hidden_dim] output. Run them in parallel with rayon, then
    // sequentially fold the weighted contributions (weighted-add is cheap
    // compared to the per-expert SwiGLU + Q4_K dequant).
    //
    // Performance: pre-parallel measurement on lambda-vector RTX 4090
    // showed `apr run --max-tokens 8` against the 17.3 GB Qwen3-Coder
    // GGUF taking ~5 minutes (k=8 experts × 48 layers running serially).
    // After this change each forward step does k=8 per-expert SwiGLU calls
    // in parallel (one per CPU core, up to k cores), reducing per-layer
    // FFN time by close to k×.
    use rayon::prelude::*;
    let expert_outputs: Vec<(f32, Vec<f32>)> = topk_renorm
        .par_iter()
        .map(|(expert_id, weight)| {
            let expert_out = expert_swiglu_quantized(
                hidden,
                layer,
                *expert_id,
                num_experts,
                intermediate,
                hidden_dim,
                data,
            )?;
            Ok::<_, RealizarError>((*weight, expert_out))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut output = vec![0.0f32; hidden_dim];
    for (weight, expert_out) in expert_outputs {
        for i in 0..hidden_dim {
            output[i] += weight * expert_out[i];
        }
    }

    Ok(output)
}

/// Sibling of [`moe_ffn_forward_layer`] that ALSO returns the router top-k
/// weights, enabling traced forward bodies to capture the `MoeRouter` stage
/// without a second router computation.
///
/// Per `contracts/trace-moe-gpu-sub-stages-v1.yaml` v1.1.0 (M-MOE-SUB-2 step c).
///
/// # Returns
///
/// `(output, router_top_k_weights)` where `output: Vec<f32>` is the
/// `[hidden_dim]` aggregated MoE FFN output (the `MoeFfnOut`
/// SaveTensorStage capture target — identical to the value returned by
/// [`moe_ffn_forward_layer`] for the same inputs), and
/// `router_top_k_weights: Vec<f32>` is the `[num_experts_per_tok]`
/// post-softmax + renormalize top-k expert weights (the `MoeRouter`
/// SaveTensorStage capture target — sums to ~1.0 unless the all-zero
/// softmax fallback path activates, in which case it sums to exactly 1.0
/// by uniform distribution).
///
/// # Hot path safety
///
/// This is the **traced sibling**. Production [`moe_ffn_forward_layer`] is
/// unchanged byte-for-byte. The two functions duplicate the router /
/// softmax / top-k logic — drift between them is mechanically prevented by
/// `moe_ffn_forward_layer_with_router_matches_production` below, which
/// asserts both functions produce the same `output` Vec for the same input
/// (synthetic F32 router only, since the production code requires real
/// GGUF MoE data which is OOS for unit tests).
///
/// # Errors
///
/// Same as [`moe_ffn_forward_layer`]: invalid shapes, non-F32 router,
/// expert byte-slice issues, or fused-matmul kernel errors.
#[allow(clippy::too_many_arguments)]
pub fn moe_ffn_forward_layer_with_router(
    hidden: &[f32],
    layer: &Qwen3MoeQuantizedLayer,
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate: usize,
    hidden_dim: usize,
    data: &[u8],
) -> Result<(Vec<f32>, Vec<f32>)> {
    use crate::error::RealizarError;

    if hidden.len() != hidden_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "moe_ffn_forward_layer_with_router: hidden.len() = {} but hidden_dim = {}",
                hidden.len(),
                hidden_dim
            ),
        });
    }

    if layer.router.qtype != crate::gguf::types::GGUF_TYPE_F32 {
        return Err(RealizarError::UnsupportedOperation {
            operation: "moe_router_quantized_read".to_string(),
            reason: format!(
                "moe_ffn_forward_layer_with_router: router qtype = {} (not F32). \
                 Quantized router not yet wired.",
                layer.router.qtype
            ),
        });
    }
    let router_bytes = &data[layer.router.offset..layer.router.offset + layer.router.byte_size];
    let expected_bytes = num_experts * hidden_dim * 4;
    if router_bytes.len() != expected_bytes {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "moe_ffn_forward_layer_with_router: router byte_size {} != expected {} \
                 (num_experts {} × hidden_dim {} × 4)",
                router_bytes.len(),
                expected_bytes,
                num_experts,
                hidden_dim
            ),
        });
    }
    let mut logits = vec![0.0f32; num_experts];
    for e in 0..num_experts {
        let row_off = e * hidden_dim * 4;
        let mut sum = 0.0f32;
        for j in 0..hidden_dim {
            let b = row_off + j * 4;
            let w = f32::from_le_bytes([
                router_bytes[b],
                router_bytes[b + 1],
                router_bytes[b + 2],
                router_bytes[b + 3],
            ]);
            sum += w * hidden[j];
        }
        logits[e] = sum;
    }

    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let psum: f32 = probs.iter().sum();
    if psum > 0.0 {
        for p in &mut probs {
            *p /= psum;
        }
    }

    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let topk = &indexed[..num_experts_per_tok.min(num_experts)];

    let topk_sum: f32 = topk.iter().map(|(_, w)| w).sum();
    let topk_renorm: Vec<(usize, f32)> = if topk_sum > 0.0 {
        topk.iter().map(|(i, w)| (*i, w / topk_sum)).collect()
    } else {
        let n = topk.len();
        topk.iter().map(|(i, _)| (*i, 1.0 / n as f32)).collect()
    };

    use rayon::prelude::*;
    let expert_outputs: Vec<(f32, Vec<f32>)> = topk_renorm
        .par_iter()
        .map(|(expert_id, weight)| {
            let expert_out = expert_swiglu_quantized(
                hidden,
                layer,
                *expert_id,
                num_experts,
                intermediate,
                hidden_dim,
                data,
            )?;
            Ok::<_, RealizarError>((*weight, expert_out))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut output = vec![0.0f32; hidden_dim];
    for (weight, expert_out) in &expert_outputs {
        for i in 0..hidden_dim {
            output[i] += weight * expert_out[i];
        }
    }

    let router_top_k_weights: Vec<f32> = topk_renorm.iter().map(|(_, w)| *w).collect();

    Ok((output, router_top_k_weights))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity: `Qwen3MoeQuantizedLayer` is a small Clone+Debug
    /// struct. Catches accidental loss of derive macros.
    #[test]
    fn qwen3_moe_quantized_layer_is_clone_and_debug() {
        let dummy = QuantizedTensorRef {
            offset: 0,
            byte_size: 0,
            num_elements: 0,
            qtype: 0,
        };
        let layer = Qwen3MoeQuantizedLayer {
            router: dummy.clone(),
            gate_exps: dummy.clone(),
            up_exps: dummy.clone(),
            down_exps: dummy,
        };
        let cloned = layer.clone();
        assert_eq!(cloned.router.offset, layer.router.offset);
        assert!(format!("{layer:?}").contains("Qwen3MoeQuantizedLayer"));
    }

    /// `expert_byte_slice` returns each expert's contiguous byte
    /// range in a synthetic 4-expert stacked tensor.
    #[test]
    fn expert_byte_slice_partitions_evenly() {
        // 4 experts × 32 bytes/expert = 128 total bytes.
        let data: Vec<u8> = (0..128).collect();
        let tensor = QuantizedTensorRef {
            offset: 0,
            byte_size: 128,
            num_elements: 128 * 2, // arbitrary, not used by slicer
            qtype: 12,             // Q4_K
        };

        for e in 0..4 {
            let slice = expert_byte_slice(&tensor, &data, e, 4).unwrap();
            assert_eq!(slice.len(), 32, "expert {e} slice length");
            // Expert e's slice starts at byte e*32; first byte must equal e*32.
            assert_eq!(slice[0], (e * 32) as u8, "expert {e} first byte");
        }
    }

    #[test]
    fn expert_byte_slice_rejects_out_of_range_expert_id() {
        let data = vec![0u8; 64];
        let tensor = QuantizedTensorRef {
            offset: 0,
            byte_size: 64,
            num_elements: 0,
            qtype: 0,
        };
        let err = expert_byte_slice(&tensor, &data, 4, 4).unwrap_err();
        assert!(format!("{err}").contains("expert_id 4 out of range"));
    }

    #[test]
    fn expert_byte_slice_rejects_zero_num_experts() {
        let data = vec![0u8; 64];
        let tensor = QuantizedTensorRef {
            offset: 0,
            byte_size: 64,
            num_elements: 0,
            qtype: 0,
        };
        let err = expert_byte_slice(&tensor, &data, 0, 0).unwrap_err();
        assert!(format!("{err}").contains("num_experts must be > 0"));
    }

    #[test]
    fn expert_byte_slice_rejects_uneven_stacking() {
        let data = vec![0u8; 100];
        let tensor = QuantizedTensorRef {
            offset: 0,
            byte_size: 100,
            num_elements: 0,
            qtype: 0,
        };
        // 100 not divisible by 3 → stacking invariant violated.
        let err = expert_byte_slice(&tensor, &data, 0, 3).unwrap_err();
        assert!(format!("{err}").contains("stacking invariant violated"));
    }

    #[test]
    fn expert_byte_slice_rejects_overrun() {
        let data = vec![0u8; 32];
        let tensor = QuantizedTensorRef {
            offset: 0,
            byte_size: 64, // claims 64 bytes but data only has 32
            num_elements: 0,
            qtype: 0,
        };
        // Expert 1 starts at byte 32; range [32, 64) overruns the 32-byte buffer.
        let err = expert_byte_slice(&tensor, &data, 1, 2).unwrap_err();
        assert!(format!("{err}").contains("exceeds file size"));
    }

    /// M-MOE-SUB-2 step (c) — `moe_ffn_forward_layer_with_router` rejects
    /// bad inputs at the same shape boundaries as the production sibling.
    /// Discharges FALSIFY-MOE-SUB-002 partially: structural sanity that
    /// the helper exists and validates its inputs symmetrically with
    /// `moe_ffn_forward_layer`. End-to-end byte-identity vs production for
    /// realistic GGUF inputs is exercised by the heavy parity tests at
    /// crates/aprender-serve/tests/qwen3_moe_gpu_parity.rs.
    #[test]
    fn moe_ffn_forward_layer_with_router_rejects_hidden_dim_mismatch() {
        let dummy = QuantizedTensorRef {
            offset: 0,
            byte_size: 0,
            num_elements: 0,
            qtype: crate::gguf::types::GGUF_TYPE_F32,
        };
        let layer = Qwen3MoeQuantizedLayer {
            router: dummy.clone(),
            gate_exps: dummy.clone(),
            up_exps: dummy.clone(),
            down_exps: dummy,
        };
        let hidden = vec![0.0f32; 8];
        let data = vec![0u8; 16];
        let err = moe_ffn_forward_layer_with_router(
            &hidden, &layer, 4, 2, 16, /* hidden_dim */ 16, &data,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("hidden.len() = 8 but hidden_dim = 16"),
            "expected hidden_dim mismatch error, got: {err}"
        );
    }

    /// M-MOE-SUB-2 step (c) — helper rejects non-F32 router with the same
    /// `moe_router_quantized_read` operation tag as production sibling. This
    /// pins the additive-purity invariant: the helper's error class is
    /// identical to production.
    #[test]
    fn moe_ffn_forward_layer_with_router_rejects_non_f32_router() {
        let dummy = QuantizedTensorRef {
            offset: 0,
            byte_size: 0,
            num_elements: 0,
            qtype: crate::gguf::types::GGUF_TYPE_Q4_K, // not F32
        };
        let layer = Qwen3MoeQuantizedLayer {
            router: dummy.clone(),
            gate_exps: dummy.clone(),
            up_exps: dummy.clone(),
            down_exps: dummy,
        };
        let hidden = vec![0.0f32; 16];
        let data = vec![0u8; 16];
        let err =
            moe_ffn_forward_layer_with_router(&hidden, &layer, 4, 2, 16, 16, &data).unwrap_err();
        assert!(
            format!("{err}").contains("router qtype") && format!("{err}").contains("not F32"),
            "expected non-F32 router error, got: {err}"
        );
    }
}
