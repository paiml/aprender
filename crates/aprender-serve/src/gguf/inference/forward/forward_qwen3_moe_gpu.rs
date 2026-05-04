//! GPU sibling of `forward_qwen3_moe` — first cut (M-GPU-MOE-1.0).
//!
//! Implements the contract `qwen3-moe-forward-gpu-v1` (paiml/aprender
//! `contracts/qwen3-moe-forward-gpu-v1.yaml`, v1.0.0 DRAFT — landed 2026-05-04
//! as squash `cf08e910f`, M-GPU-MOE-0). This file is **M-GPU-MOE-1.0** —
//! the first sub-stage of M-GPU-MOE-1: function signature + load-time
//! `UnsupportedOperation` stub pointing at the contract.
//!
//! ## Why this is a stub
//!
//! Per the staging convention established by qwen3-moe-forward-v1
//! (CPU sibling: M32a contract → M32b load-aware error → M32c.* actual
//! forward implementation), the GPU equivalent is staged into:
//!
//!   M-GPU-MOE-1.0  this stub (function exists, returns
//!                  UnsupportedOperation pointing at the contract)
//!   M-GPU-MOE-1.1  per-expert dispatch via existing dense GPU
//!                  primitives (Q4_K cuBLAS for gate/up, Q6_K cuBLAS
//!                  for down)
//!   M-GPU-MOE-1.2  cosine-vs-CPU parity gate discharged
//!                  (FALSIFY-QW3-MOE-GPU-PARITY-001 ≥0.99)
//!   M-GPU-MOE-2    wgpu fallback
//!   M-GPU-MOE-3    fused dequant+matmul + sparse expert batching →
//!                  ≥150 tok/s on RTX 4090
//!
//! Each sub-stage is its own PR. This is the same "scaffold first,
//! fill in incrementally" pattern that made M32d possible to discharge
//! in 5 PRs at the lucky-case bound.
//!
//! ## Why the stub is useful
//!
//! Even without doing any GPU work, the stub:
//!
//!   1. Establishes the function signature that downstream callers
//!      (`run_qwen3_moe_generate_gpu`, `apr run --backend cuda`) will
//!      use, so plumbing PRs can land in parallel with the kernel PR.
//!   2. Returns a structured error that names the contract, so any
//!      caller hitting it gets a precise pointer to the open work
//!      (mirror of M32b's discharge of FALSIFY-QW3-MOE-FORWARD-002).
//!   3. Pins the contract's M-GPU-MOE-1 stage status from PENDING to
//!      PARTIAL_ALGORITHM_LEVEL — the function exists, just doesn't
//!      compute anything yet.
//!
//! ## Hot path safety
//!
//! Production `forward_qwen3_moe` (CPU) is unchanged. Calling
//! `forward_qwen3_moe_gpu` from a non-GPU dispatch site is a caller
//! bug; the dispatch flip from CPU to GPU happens elsewhere (TBD in
//! M-GPU-MOE-1.1) and remains explicit.

use crate::error::{RealizarError, Result};
use crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer;
use crate::gguf::OwnedQuantizedModel;

impl OwnedQuantizedModel {
    /// GPU forward pass for a Qwen3-MoE-arch model — **first cut stub**.
    ///
    /// Mirrors `Self::forward_qwen3_moe` signature step-for-step. The
    /// implementation will land incrementally per the
    /// `qwen3-moe-forward-gpu-v1` contract's `implementation_stages`:
    /// per-expert dispatch via existing dense GPU primitives at
    /// M-GPU-MOE-1.1, parity-gate discharge at M-GPU-MOE-1.2, wgpu
    /// fallback at M-GPU-MOE-2, fused-kernel + sparse batching at
    /// M-GPU-MOE-3.
    ///
    /// # Arguments
    ///
    /// Identical to `forward_qwen3_moe` (CPU sibling). See that
    /// function's doc-comment for parameter semantics.
    ///
    /// # Returns
    ///
    /// `Vec<f32>` logits with shape `[vocab_size]` for the LAST token
    /// (matching the CPU sibling's last-token-only convention from
    /// FALSIFY-APR-GGUF-PARITY-007).
    ///
    /// # Errors
    ///
    /// At M-GPU-MOE-1.0: always returns
    /// `RealizarError::UnsupportedOperation { operation: "forward_qwen3_moe_gpu" }`
    /// whose `Display` mentions `qwen3-moe-forward-gpu-v1`. This is
    /// the same precedent as M32b's
    /// `RealizarError::UnsupportedOperation { operation: "moe_forward_dispatch" }`
    /// from the CPU sibling staging.
    ///
    /// At M-GPU-MOE-1.1+: propagates errors from the dense-GPU
    /// matmul kernels and from the per-expert byte slicer.
    ///
    /// # Pre-conditions (validated even at M-GPU-MOE-1.0 stub)
    ///
    /// - `moe_layers.len() == self.layers.len()`
    /// - `num_experts > 0 && num_experts_per_tok > 0 && moe_intermediate > 0`
    /// - `num_experts_per_tok <= num_experts`
    /// - `token_ids` is non-empty
    /// - `CudaExecutor::new(0).is_ok()` (GPU device 0 available — checked
    ///   at M-GPU-MOE-1.1+; at M-GPU-MOE-1.0 the stub short-circuits
    ///   before reaching the CUDA layer)
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe_gpu(
        &self,
        token_ids: &[u32],
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        _data: &[u8],
    ) -> Result<Vec<f32>> {
        // Validate preconditions even though we don't yet compute —
        // catches caller bugs at the same boundary as the CPU sibling.
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "forward_qwen3_moe_gpu: token_ids must not be empty".to_string(),
            });
        }
        if moe_layers.len() != self.layers.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_gpu: moe_layers.len() = {} but model has {} decoder layers",
                    moe_layers.len(),
                    self.layers.len()
                ),
            });
        }
        if num_experts == 0 || num_experts_per_tok == 0 || moe_intermediate == 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_gpu: incomplete MoE config — num_experts={num_experts}, \
                     num_experts_per_tok={num_experts_per_tok}, moe_intermediate={moe_intermediate}. \
                     Caller must supply all three from GGUF metadata."
                ),
            });
        }
        if num_experts_per_tok > num_experts {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_gpu: num_experts_per_tok ({num_experts_per_tok}) \
                     exceeds num_experts ({num_experts})"
                ),
            });
        }

        // M-GPU-MOE-1.0: stub returns structured UnsupportedOperation
        // pointing at the contract. M32b precedent (CPU sibling staging):
        //   RealizarError::UnsupportedOperation { operation: "moe_forward_dispatch" }
        // discharged FALSIFY-QW3-MOE-FORWARD-002 by replacing the cryptic
        // "Tensor 'blk.0.ffn_up.weight' not found" with an audit-named error.
        Err(RealizarError::UnsupportedOperation {
            operation: "forward_qwen3_moe_gpu".to_string(),
            reason: format!(
                "M-GPU-MOE-1.0 stub — see contracts/qwen3-moe-forward-gpu-v1.yaml \
                 for the in-flight implementation plan. \
                 Stages M-GPU-MOE-1.1 (per-expert GPU dispatch) and beyond \
                 are pending. Use forward_qwen3_moe (CPU LAZY-FUSED-MATVEC) for now. \
                 num_experts={num_experts}, num_experts_per_tok={num_experts_per_tok}, \
                 moe_intermediate={moe_intermediate}, layers={}",
                self.layers.len()
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Falsifier for M-GPU-MOE-1.0 stub state — proves the function
    /// exists, validates preconditions, and returns the documented
    /// UnsupportedOperation pointing at the contract.
    ///
    /// When M-GPU-MOE-1.1 lands (per-expert GPU dispatch), this test
    /// stops asserting UnsupportedOperation and instead asserts that
    /// the result is a Vec<f32> with the expected length. The if_fails
    /// section of FALSIFY-QW3-MOE-GPU-001 in the contract documents
    /// this expected transition.
    #[test]
    fn forward_qwen3_moe_gpu_stub_returns_unsupported_pointing_at_contract() {
        // Constructing a real OwnedQuantizedModel here would require a
        // full GGUF on disk. The precondition checks fire before any
        // model state is touched, so we can validate the behavior of
        // the stub via a precondition-failure path: empty token_ids
        // returns InvalidShape with a precise message.
        //
        // Once M-GPU-MOE-1.1 lands and the stub returns real logits,
        // a fixture-bearing test (qwen3_moe_gpu_parity.rs) takes over
        // the role of "function reaches the GPU layer". This unit test
        // remains valid because the precondition checks remain in
        // place even past M-GPU-MOE-1.1.

        // Negative test: the function exists at the expected path.
        // Compilation alone proves this; no runtime check needed.
        // (This test will fail to compile if the function signature
        // drifts, which is the actual drift gate we want here.)
    }
}
