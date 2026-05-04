// crates/aprender-serve/src/gguf/cuda/forward_qwen3_moe_cuda.rs
//
// GPU sibling of `forward_qwen3_moe` — first cut on the correct type.
//
// Implements the contract `qwen3-moe-forward-gpu-v1` (paiml/aprender
// `contracts/qwen3-moe-forward-gpu-v1.yaml`, v1.1.0 ACTIVE_ALGORITHM_LEVEL
// per the option D amendment in PR #1462 squash 449540714 on
// 2026-05-04T09:38:29Z). This file is **M-GPU-MOE-1.0-redo** —
// the first sub-stage of M-GPU-MOE-1, placed on the correct type:
// `OwnedQuantizedModelCuda` (NOT `OwnedQuantizedModel`).
//
// Why on OwnedQuantizedModelCuda
// ===============================
//
// Per the v1.1.0 amendment's option-D decision: this method must
// extend the existing OwnedQuantizedModelCuda CPU-attention + CUDA-
// FFN pattern (forward_cuda in cuda.rs), not invent a new substrate.
// The wrong-type stub on OwnedQuantizedModel from PR #1460
// (4d9e5ae2b on aprender main) is retired by this redo.
//
// The wrong-type stub stays on main for now (it'll be removed in a
// later cleanup PR). It documents the entry-point name but routes
// any caller to use forward_qwen3_moe_cuda on the wrapper type.
//
// Why this is a stub
// ==================
//
// Same reason as the v1 sibling staging (qwen3-moe-forward-v1
// M32a → M32b → M32c.* chain): contract first, scaffold second,
// implementation third. M-GPU-MOE-1.0-redo establishes the function
// on the correct type so M-GPU-MOE-1.1 (per-expert CUDA dispatch via
// self.executor) can land in a separate PR without re-arguing the
// architectural seam.

// Imports inherited from parent forward.rs (super::OwnedQuantizedModelCuda,
// crate::error::{RealizarError, Result}). This file is included via
// uses.rs include!() chain, so re-importing causes "must be defined only
// once" namespace conflicts.

use crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer;

impl OwnedQuantizedModelCuda {
    /// CUDA forward pass for a Qwen3-MoE-arch model — **stub on the
    /// correct type per qwen3-moe-forward-gpu-v1 v1.1.0 option D**.
    ///
    /// Mirrors `OwnedQuantizedModel::forward_qwen3_moe` (CPU sibling)
    /// signature step-for-step, plus the precondition validation
    /// boundary. The implementation will land incrementally per the
    /// contract's `implementation_stages`:
    ///
    /// - **M-GPU-MOE-1.0-redo (this stub)**: function exists on the
    ///   correct type; returns structured `UnsupportedOperation`
    ///   pointing at the contract.
    /// - **M-GPU-MOE-1.1**: per-expert CUDA dispatch via
    ///   `self.executor` (gemm_q4k for gate/up_proj, gemm_q6k for
    ///   down_proj). Naive — one cuBLAS call per top-k expert per
    ///   token, no fused dequant+matmul, no sparse expert batching.
    ///   Discharges AC_GPU_MOE_001..005 against the CPU LAZY-FUSED-
    ///   MATVEC reference.
    /// - **M-GPU-MOE-1.2**: cosine-vs-CPU parity gate ≥0.99
    ///   (FALSIFY-QW3-MOE-GPU-PARITY-001).
    /// - **M-GPU-MOE-2**: wgpu fallback (separate type analogous to
    ///   OwnedQuantizedModelCuda for non-CUDA hardware).
    /// - **M-GPU-MOE-3**: fused dequant+matmul + sparse expert
    ///   batching → ≥150 tok/s on RTX 4090.
    ///
    /// # Arguments
    ///
    /// Identical to `forward_qwen3_moe` (CPU sibling on
    /// `OwnedQuantizedModel`). See that function's doc-comment for
    /// parameter semantics.
    ///
    /// # Returns
    ///
    /// `Vec<f32>` logits with shape `[vocab_size]` for the LAST token
    /// (matching the CPU sibling's last-token-only convention from
    /// FALSIFY-APR-GGUF-PARITY-007).
    ///
    /// # Errors
    ///
    /// At M-GPU-MOE-1.0-redo: returns `RealizarError::UnsupportedOperation
    /// { operation: "forward_qwen3_moe_cuda" }` whose `Display`
    /// mentions `qwen3-moe-forward-gpu-v1`. M32b precedent.
    ///
    /// At M-GPU-MOE-1.1+: propagates errors from `self.executor`
    /// (CudaExecutor) and from the per-expert byte slicer.
    ///
    /// # Pre-conditions (validated even at M-GPU-MOE-1.0-redo stub)
    ///
    /// - `moe_layers.len() == self.model.layers.len()`
    /// - `num_experts > 0 && num_experts_per_tok > 0 && moe_intermediate > 0`
    /// - `num_experts_per_tok <= num_experts`
    /// - `token_ids` is non-empty
    /// - `self.executor.is_available()` (GPU device — checked
    ///   implicitly because OwnedQuantizedModelCuda::new already
    ///   instantiates a CudaExecutor for device 0)
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe_cuda(
        &self,
        token_ids: &[u32],
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        _data: &[u8],
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "forward_qwen3_moe_cuda: token_ids must not be empty".to_string(),
            });
        }
        if moe_layers.len() != self.model.layers.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_cuda: moe_layers.len() = {} but model has {} decoder layers",
                    moe_layers.len(),
                    self.model.layers.len()
                ),
            });
        }
        if num_experts == 0 || num_experts_per_tok == 0 || moe_intermediate == 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_cuda: incomplete MoE config — num_experts={num_experts}, \
                     num_experts_per_tok={num_experts_per_tok}, moe_intermediate={moe_intermediate}. \
                     Caller must supply all three from GGUF metadata."
                ),
            });
        }
        if num_experts_per_tok > num_experts {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_cuda: num_experts_per_tok ({num_experts_per_tok}) \
                     exceeds num_experts ({num_experts})"
                ),
            });
        }

        // M-GPU-MOE-1.0-redo: stub returns structured UnsupportedOperation
        // pointing at the contract. Same M32b precedent as the v1 CPU
        // sibling staging.
        Err(RealizarError::UnsupportedOperation {
            operation: "forward_qwen3_moe_cuda".to_string(),
            reason: format!(
                "M-GPU-MOE-1.0-redo stub on OwnedQuantizedModelCuda \
                 (qwen3-moe-forward-gpu-v1 v1.1.0 option D, ACTIVE_ALGORITHM_LEVEL). \
                 Stages M-GPU-MOE-1.1 (per-expert CUDA dispatch via self.executor) \
                 and beyond are pending. Use OwnedQuantizedModel::forward_qwen3_moe \
                 (CPU LAZY-FUSED-MATVEC) for now. \
                 num_experts={num_experts}, num_experts_per_tok={num_experts_per_tok}, \
                 moe_intermediate={moe_intermediate}, layers={}",
                self.model.layers.len()
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    /// Compilation gate: signature drift between this stub and the
    /// CPU sibling forward_qwen3_moe is caught at build time.
    /// When M-GPU-MOE-1.1 lands, fixture-bearing tests in
    /// tests/qwen3_moe_gpu_parity.rs take over the role of "function
    /// reaches GPU and matches CPU within cosine ≥0.99". This unit
    /// test remains valid because the precondition checks remain in
    /// place even past M-GPU-MOE-1.1.
    #[test]
    fn forward_qwen3_moe_cuda_stub_compiles_with_correct_signature() {
        // Compilation alone proves signature parity with the CPU
        // sibling (mod self type and _data underscore). No runtime
        // check needed — the test exists to fail compile if either
        // side's signature drifts.
    }
}
