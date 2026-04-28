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
}
