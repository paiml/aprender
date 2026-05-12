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
/// `gguf/qwen3_moe_load.rs::expert_swiglu_quantized` (which dispatches
/// via `matvec_for_qtype`) and routes the matmuls through the
/// CudaExecutor with **qtype-aware dispatch** (Q4_K vs Q6_K).
///
/// # Why qtype-aware dispatch
///
/// Qwen3-Coder-30B-A3B-Instruct Q4_K_M is a MIXED quantization — its
/// expert tensors per layer can be either Q4_K (12) or Q6_K (14). The
/// CPU sibling `expert_swiglu_quantized` already does this; the GPU
/// path now mirrors it (M-GPU-MOE-1.4 step (c) fix per
/// `qwen3-moe-forward-gpu-v1` v1.6.0).
///
/// # Arguments
///
/// * `executor` — owned mutable reference to the CudaExecutor.
/// * `gate_bytes` / `gate_qtype` — raw bytes + GGUF qtype for this
///   expert's gate_proj weight, shape `[intermediate, hidden_dim]`
///   row-major.
/// * `up_bytes` / `up_qtype` — raw bytes + qtype for this expert's
///   up_proj weight, same shape as gate.
/// * `down_bytes` / `down_qtype` — raw bytes + qtype for this expert's
///   down_proj weight, shape `[hidden_dim, intermediate]` row-major.
/// * `hidden` — input activation, length `hidden_dim`.
/// * `hidden_dim` — model hidden dimension.
/// * `intermediate` — MoE FFN intermediate dimension.
///
/// # Returns
///
/// `Vec<f32>` of length `hidden_dim` — this expert's contribution
/// before weighted aggregation across selected experts.
///
/// # Errors
///
/// - `InvalidShape` on dimensional mismatches.
/// - `UnsupportedOperation` on unknown qtype (anything other than
///   Q4_K (12) or Q6_K (14) — same set as CPU `matvec_for_qtype`).
/// - Propagates errors from `CudaExecutor::q4k_matvec` and
///   `CudaExecutor::q6k_gemv`.
///
/// # Numerical
///
/// silu(x) = x * sigmoid(x) computed in f32 elementwise on CPU
/// between the two GPU dispatches. The element-wise multiplication
/// `silu(gate) * up` is also CPU. Only the matmuls go to GPU.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_swiglu_cuda(
    executor: &mut crate::cuda::CudaExecutor,
    gate_bytes: &[u8],
    gate_qtype: u32,
    up_bytes: &[u8],
    up_qtype: u32,
    down_bytes: &[u8],
    down_qtype: u32,
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

    // 1. gate_out = matvec_for_qtype(gate_W, hidden)   [intermediate]
    let mut gate_out = vec![0.0f32; intermediate];
    matvec_qtype_cuda(
        executor,
        gate_qtype,
        gate_bytes,
        hidden,
        &mut gate_out,
        intermediate,
        hidden_dim,
        "gate",
    )?;

    // 2. up_out = matvec_for_qtype(up_W, hidden)   [intermediate]
    let mut up_out = vec![0.0f32; intermediate];
    matvec_qtype_cuda(
        executor,
        up_qtype,
        up_bytes,
        hidden,
        &mut up_out,
        intermediate,
        hidden_dim,
        "up",
    )?;

    // 3. ffn_inner[i] = silu(gate[i]) * up[i]   element-wise CPU
    //    silu(x) = x * sigmoid(x) = x / (1 + e^(-x))
    let mut ffn_inner = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let g = gate_out[i];
        let silu_g = g / (1.0 + (-g).exp());
        ffn_inner[i] = silu_g * up_out[i];
    }

    // 4. expert_out = matvec_for_qtype(down_W, ffn_inner)   [hidden_dim]
    let mut expert_out = vec![0.0f32; hidden_dim];
    matvec_qtype_cuda(
        executor,
        down_qtype,
        down_bytes,
        &ffn_inner,
        &mut expert_out,
        hidden_dim,
        intermediate,
        "down",
    )?;

    Ok(expert_out)
}

/// Dispatch a single matvec to either q4k_matvec or q6k_gemv based on qtype.
/// Mirrors the CPU `matvec_for_qtype` (in `qwen3_moe_load.rs`) shape so the
/// GPU and CPU paths stay numerically equivalent for both Q4_K and Q6_K
/// expert tensors. M-GPU-MOE-1.4 step (c) per qwen3-moe-forward-gpu-v1 v1.6.0.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn matvec_qtype_cuda(
    executor: &mut crate::cuda::CudaExecutor,
    qtype: u32,
    bytes: &[u8],
    activations: &[f32],
    out: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    role: &'static str,
) -> Result<()> {
    use crate::gguf::types::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};
    match qtype {
        GGUF_TYPE_Q4_K => executor
            .q4k_matvec(bytes, activations, out, out_dim as u32, in_dim as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: format!("expert_swiglu_cuda::{role}_q4k_matvec"),
                reason: format!("{e}"),
            }),
        GGUF_TYPE_Q6_K => executor
            .q6k_gemv(bytes, activations, out, out_dim as u32, in_dim as u32)
            .map_err(|e| RealizarError::UnsupportedOperation {
                operation: format!("expert_swiglu_cuda::{role}_q6k_gemv"),
                reason: format!("{e}"),
            }),
        other => Err(RealizarError::UnsupportedOperation {
            operation: format!("expert_swiglu_cuda::{role}_matvec"),
            reason: format!(
                "MoE expert tensor qtype {other} not supported. Qwen3-Coder Q4_K_M uses \
                 Q4_K (12) and Q6_K (14) — caller must extend matvec_qtype_cuda for other \
                 quantizations (mirror of CPU matvec_for_qtype in qwen3_moe_load.rs)."
            ),
        }),
    }
}

#[cfg(test)]
mod expert_swiglu_cuda_tests {
    use super::*;
    use crate::gguf::types::{GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K};

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
                GGUF_TYPE_Q4_K,
                &dummy_bytes,
                GGUF_TYPE_Q4_K,
                &dummy_bytes,
                GGUF_TYPE_Q6_K,
                &hidden,
                10,
                4,
            );
            assert!(matches!(result, Err(RealizarError::InvalidShape { .. })));
        }
    }

    /// FALSIFY-MOE-SUB-004 / M-GPU-MOE-1.4 step (c) drift gate:
    /// `expert_swiglu_cuda` must reject any qtype other than Q4_K (12)
    /// or Q6_K (14) with `UnsupportedOperation`. Mirrors the CPU
    /// `matvec_for_qtype` rejection set. Lib-only test — no CUDA
    /// device required because the qtype dispatch happens before any
    /// GPU call.
    #[cfg(feature = "cuda")]
    #[test]
    fn falsify_qw3_moe_gpu_qtype_aware_dispatch_rejects_unknown() {
        if let Ok(mut executor) = crate::cuda::CudaExecutor::new(0) {
            // Use sufficient bytes so the shape check passes (4 hidden,
            // 10 intermediate); we need to reach the qtype-dispatch step.
            let dummy_bytes = vec![0u8; 1024];
            let hidden = vec![1.0f32; 4];
            // Q8_0 = 8 — unknown to expert_swiglu_cuda (CPU also rejects).
            const GGUF_TYPE_Q8_0: u32 = 8;
            let result = expert_swiglu_cuda(
                &mut executor,
                &dummy_bytes,
                GGUF_TYPE_Q8_0, // unknown qtype on gate
                &dummy_bytes,
                GGUF_TYPE_Q4_K,
                &dummy_bytes,
                GGUF_TYPE_Q6_K,
                &hidden,
                4,
                10,
            );
            assert!(
                matches!(result, Err(RealizarError::UnsupportedOperation { .. })),
                "expected UnsupportedOperation for unknown gate qtype, got {result:?}"
            );
        }
    }

    /// FALSIFY-MOE-SUB-004 / M-GPU-MOE-1.4 step (c) drift gate:
    /// `expert_swiglu_cuda` accepts the {Q4_K, Q6_K} qtype combinatorial
    /// surface mirroring CPU `matvec_for_qtype`. Compilation gate —
    /// asserts the signature has 3 separate qtype params (one per
    /// matvec target: gate, up, down). If a future refactor collapses
    /// them, this test will fail to compile.
    #[test]
    fn expert_swiglu_cuda_signature_has_three_qtype_params() {
        // Reference the function path with all three qtype params spelled
        // explicitly; compilation alone proves they exist.
        #[cfg(feature = "cuda")]
        let _f: fn(
            &mut crate::cuda::CudaExecutor,
            &[u8],
            u32, // gate_qtype
            &[u8],
            u32, // up_qtype
            &[u8],
            u32, // down_qtype
            &[f32],
            usize,
            usize,
        ) -> Result<Vec<f32>> = expert_swiglu_cuda;
    }
}
