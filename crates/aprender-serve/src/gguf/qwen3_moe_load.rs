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
}
