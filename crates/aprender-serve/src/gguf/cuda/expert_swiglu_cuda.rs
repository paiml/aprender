// crates/aprender-serve/src/gguf/cuda/expert_swiglu_cuda.rs
//
// M-GPU-MOE-1.1.0 — per-expert SwiGLU GPU helper for forward_qwen3_moe_cuda.
//
// Per qwen3-moe-forward-gpu-v1 v1.1.0 option D (PR #1462 squash 449540714):
// the GPU MoE forward path lives on OwnedQuantizedModelCuda. This file
// provides the inner-most building block — the per-expert SwiGLU FFN
// computed on GPU via the existing CudaExecutor primitives.
//
// Why a separate helper
// =====================
//
// Mirrors the CPU staging of qwen3-moe-forward-v1 M32c.2.2.* — the
// per-expert byte slicer (M32c.2.2.0) and per-expert SwiGLU
// (M32c.2.2.1) landed in separate sub-milestones BEFORE the full
// `moe_ffn_forward_layer` integration (M32c.2.2.2.0). Same shape on
// the GPU side: helper first (M-GPU-MOE-1.1.0), full integration in
// the wrapper method (M-GPU-MOE-1.1.1+).
//
// Numerical equivalence
// =====================
//
// The CPU sibling `moe_ffn_forward_layer` (qwen3_moe_load.rs:363)
// computes per expert:
//
//   gate_out      = q4k_matmul_row_major(gate_W,  hidden)
//   up_out        = q4k_matmul_row_major(up_W,    hidden)
//   ffn_inner[i]  = silu(gate_out[i]) * up_out[i]   for i in [0, intermediate)
//   expert_out    = q6k_matmul_row_major(down_W,   ffn_inner)
//
// where silu(x) = x · sigmoid(x) = x / (1 + e^(-x)).
//
// This helper performs the same sequence on GPU using
// CudaExecutor::q4k_matvec for gate_proj/up_proj and
// CudaExecutor::q6k_gemv for down_proj. The CPU↔GPU equivalence
// gate (FALSIFY-QW3-MOE-GPU-PARITY-001 cosine ≥0.99) is asserted at
// the integration layer (M-GPU-MOE-1.2), not here — but this helper
// is the load-bearing kernel-call site that gate validates.

// Imports inherited from parent forward.rs (RealizarError, Result).
// CudaExecutor is reachable via `crate::cuda::CudaExecutor` at use sites.
// This file is included via uses.rs include!() chain.

/// Per-expert SwiGLU FFN on GPU — single-expert, single-token.
///
/// Mirrors the CPU per-expert SwiGLU body in
/// `gguf/qwen3_moe_load.rs::moe_ffn_forward_layer` (the inner loop
/// over selected experts) but routes the matmuls through the
/// CudaExecutor.
///
/// # Arguments
///
/// * `executor` — owned mutable reference to the CudaExecutor (cache-
///   aware kernel dispatch + cuBLAS handle).
/// * `gate_bytes` — raw Q4_K bytes for this expert's gate_proj weight,
///   shape `[intermediate, hidden_dim]` row-major. Obtain via
///   `expert_byte_slice(layer.gate_exps, data, expert_id, num_experts)`.
/// * `up_bytes` — raw Q4_K bytes for this expert's up_proj weight,
///   same shape as gate.
/// * `down_bytes` — raw Q6_K bytes for this expert's down_proj weight,
///   shape `[hidden_dim, intermediate]` row-major.
/// * `hidden` — input activation, length `hidden_dim`.
/// * `hidden_dim` — model hidden dimension.
/// * `intermediate` — MoE FFN intermediate dimension (Qwen3-Coder-30B
///   uses 768).
///
/// # Returns
///
/// `Vec<f32>` of length `hidden_dim` — this expert's contribution
/// before weighted aggregation across selected experts.
///
/// # Errors
///
/// Propagates errors from `CudaExecutor::q4k_matvec` and
/// `CudaExecutor::q6k_gemv`. Returns `RealizarError::InvalidShape`
/// on dimensional mismatches.
///
/// # Numerical
///
/// silu(x) = x * sigmoid(x) computed in f32 elementwise on CPU
/// between the two GPU dispatches. The element-wise multiplication
/// `silu(gate) * up` is also CPU. Only the matmuls go to GPU.
/// This is the "naive per-expert dispatch" baseline of the contract's
/// Sub-extension 2 implementation_stages.M-GPU-MOE-1.1.0; the fused
/// dequant+matmul + sparse expert batching path is M-GPU-MOE-3.
#[cfg(feature = "cuda")]
pub(crate) fn expert_swiglu_cuda(
    executor: &mut crate::cuda::CudaExecutor,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_bytes: &[u8],
    hidden: &[f32],
    hidden_dim: usize,
    intermediate: usize,
) -> Result<Vec<f32>> {
    if hidden.len() != hidden_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "expert_swiglu_cuda: hidden.len() = {} but hidden_dim = {}",
                hidden.len(),
                hidden_dim
            ),
        });
    }
    if hidden_dim == 0 || intermediate == 0 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "expert_swiglu_cuda: hidden_dim ({hidden_dim}) and intermediate \
                 ({intermediate}) must both be > 0"
            ),
        });
    }

    // 1. gate_out = q4k_matmul(gate_W, hidden)   [intermediate]
    let mut gate_out = vec![0.0f32; intermediate];
    executor
        .q4k_matvec(
            gate_bytes,
            hidden,
            &mut gate_out,
            intermediate as u32,
            hidden_dim as u32,
        )
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "expert_swiglu_cuda::gate_q4k_matvec".to_string(),
            reason: format!("{e}"),
        })?;

    // 2. up_out = q4k_matmul(up_W, hidden)   [intermediate]
    let mut up_out = vec![0.0f32; intermediate];
    executor
        .q4k_matvec(
            up_bytes,
            hidden,
            &mut up_out,
            intermediate as u32,
            hidden_dim as u32,
        )
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "expert_swiglu_cuda::up_q4k_matvec".to_string(),
            reason: format!("{e}"),
        })?;

    // 3. ffn_inner[i] = silu(gate[i]) * up[i]   element-wise CPU
    //    silu(x) = x * sigmoid(x) = x / (1 + e^(-x))
    let mut ffn_inner = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        ffn_inner[i] = silu_g * up_out[i];
    }

    // 4. expert_out = q6k_matmul(down_W, ffn_inner)   [hidden_dim]
    let mut expert_out = vec![0.0f32; hidden_dim];
    executor
        .q6k_gemv(
            down_bytes,
            &ffn_inner,
            &mut expert_out,
            hidden_dim as u32,
            intermediate as u32,
        )
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "expert_swiglu_cuda::down_q6k_gemv".to_string(),
            reason: format!("{e}"),
        })?;

    Ok(expert_out)
}

#[cfg(test)]
mod expert_swiglu_cuda_tests {
    use super::*;

    /// Compilation gate for signature drift.
    #[test]
    fn expert_swiglu_cuda_signature_drift_gate() {}

    /// Validate input shape rejection — InvalidShape on len mismatch.
    #[cfg(feature = "cuda")]
    #[test]
    fn expert_swiglu_cuda_rejects_mismatched_hidden_len() {
        if let Ok(mut executor) = crate::cuda::CudaExecutor::new(0) {
            let dummy_bytes = vec![0u8; 144];
            let hidden = vec![1.0f32; 5];
            let result = expert_swiglu_cuda(
                &mut executor,
                &dummy_bytes,
                &dummy_bytes,
                &dummy_bytes,
                &hidden,
                10,
                4,
            );
            assert!(matches!(result, Err(RealizarError::InvalidShape { .. })));
        }
    }
}
