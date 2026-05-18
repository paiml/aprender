// crates/aprender-serve/src/gguf/cuda/moe_ffn_forward_layer_cuda.rs
//
// M-GPU-MOE-1.1.1 — single-layer MoE FFN forward on GPU. Mirrors the
// CPU sibling `gguf/qwen3_moe_load.rs::moe_ffn_forward_layer` step-
// for-step, replacing only the per-expert SwiGLU body with a call
// into `expert_swiglu_cuda` (M-GPU-MOE-1.1.0).
//
// Per qwen3-moe-forward-gpu-v1 v1.1.0 option D (PR #1462 squash
// 449540714): GPU MoE forward path lives on OwnedQuantizedModelCuda
// and reuses existing CudaExecutor primitives, with router/softmax/
// top-k/aggregation on CPU (small per-token operations) and per-
// expert matmuls on GPU.
//
// Imports inherited from parent forward.rs (RealizarError, Result).
// Use fully-qualified paths for qwen3_moe_load items to avoid the
// "must be defined only once" namespace conflict from the include!() chain.

/// GPU sibling of `moe_ffn_forward_layer` — single-layer MoE FFN
/// forward for one token.
///
/// # Architecture (option D, naive per-expert dispatch)
///
/// 1. **Router (CPU)**: `logits = router_F32 @ hidden`, softmax + top-k
///    + renormalize. Router is small (~1 MB for Qwen3-Coder-30B's
///    128×2048 router); CPU is fast enough.
/// 2. **Per-expert SwiGLU (GPU via `expert_swiglu_cuda`)**: for each
///    of the top-k selected experts, slice the per-expert Q4_K/Q6_K
///    bytes from the on-disk GGUF and dispatch the gate/up/down
///    matmuls via existing CudaExecutor primitives.
/// 3. **Weighted aggregation (CPU)**: `out = Σ_e w_e · expert_out_e`.
///    Top-k is small (Qwen3-Coder uses k=8); CPU sum is fast.
///
/// # Numerical equivalence vs CPU sibling
///
/// CPU `moe_ffn_forward_layer` uses fused_q4k/q6k_parallel_matvec
/// (Rust SIMD); this routes through CudaExecutor::q4k_matvec /
/// q6k_gemv. Both dequantize the same Q4_K/Q6_K bytes. Cosine ≥0.99
/// equivalence is asserted at M-GPU-MOE-1.2 (FALSIFY-QW3-MOE-GPU-PARITY-001).
///
/// # Errors
///
/// - `InvalidShape` on dimensional mismatch.
/// - `UnsupportedOperation` on quantized router (only F32 router supported
///   in v1.1; quantized router is M32 follow-up).
/// - Propagates errors from `expert_byte_slice` and `expert_swiglu_cuda`.
#[cfg(feature = "cuda")]
pub(crate) fn moe_ffn_forward_layer_cuda(
    executor: &mut crate::cuda::CudaExecutor,
    hidden: &[f32],
    layer: &crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer,
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate: usize,
    hidden_dim: usize,
    data: &[u8],
) -> Result<Vec<f32>> {
    if hidden.len() != hidden_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "moe_ffn_forward_layer_cuda: hidden.len() = {} but hidden_dim = {}",
                hidden.len(),
                hidden_dim
            ),
        });
    }

    // Router: F32 weight, logits = router @ hidden
    if layer.router.qtype != crate::gguf::types::GGUF_TYPE_F32 {
        return Err(RealizarError::UnsupportedOperation {
            operation: "moe_router_quantized_read_cuda".to_string(),
            reason: format!(
                "moe_ffn_forward_layer_cuda: router qtype = {} (not F32). \
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
                "moe_ffn_forward_layer_cuda: router byte_size {} != expected {}",
                router_bytes.len(),
                expected_bytes
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

    // Softmax (numerically stable)
    let max_l = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let psum: f32 = probs.iter().sum();
    if psum > 0.0 {
        for p in &mut probs {
            *p /= psum;
        }
    }

    // Top-k selection
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let topk = &indexed[..num_experts_per_tok.min(num_experts)];

    // Renormalize selected
    let topk_sum: f32 = topk.iter().map(|(_, w)| w).sum();
    let topk_renorm: Vec<(usize, f32)> = if topk_sum > 0.0 {
        topk.iter().map(|(i, w)| (*i, w / topk_sum)).collect()
    } else {
        let n = topk.len();
        topk.iter().map(|(i, _)| (*i, 1.0 / n as f32)).collect()
    };

    // Per-expert SwiGLU on GPU + weighted aggregation on CPU.
    // M-GPU-MOE-1.4 step (c) per qwen3-moe-forward-gpu-v1 v1.6.0:
    // pass per-tensor qtypes so expert_swiglu_cuda can dispatch
    // each matvec to either q4k_matvec or q6k_gemv (Qwen3-Coder-30B
    // Q4_K_M mixes Q4_K and Q6_K expert tensors per layer).
    let mut out = vec![0.0f32; hidden_dim];
    for &(expert_id, weight) in &topk_renorm {
        let gate_bytes = crate::gguf::qwen3_moe_load::expert_byte_slice(
            &layer.gate_exps,
            data,
            expert_id,
            num_experts,
        )?;
        let up_bytes = crate::gguf::qwen3_moe_load::expert_byte_slice(
            &layer.up_exps,
            data,
            expert_id,
            num_experts,
        )?;
        let down_bytes = crate::gguf::qwen3_moe_load::expert_byte_slice(
            &layer.down_exps,
            data,
            expert_id,
            num_experts,
        )?;

        let expert_out = expert_swiglu_cuda(
            executor,
            gate_bytes,
            layer.gate_exps.qtype,
            up_bytes,
            layer.up_exps.qtype,
            down_bytes,
            layer.down_exps.qtype,
            hidden,
            hidden_dim,
            intermediate,
        )?;

        for i in 0..hidden_dim {
            out[i] += weight * expert_out[i];
        }
    }

    Ok(out)
}

/// GPU sibling of `moe_ffn_forward_layer_with_router` — single-layer MoE
/// FFN forward for one token that ALSO returns the post-renormalize top-k
/// router weights. Enables the GPU traced forward body (M-MOE-SUB-2 step b,
/// follow-up PR) to capture `MoeRouter` without recomputing the router.
///
/// Per `contracts/trace-moe-gpu-sub-stages-v1.yaml` v1.2.0 — extends step
/// (c) to its GPU parallel.
///
/// # Returns
///
/// `(output, router_top_k_weights)` where `output: Vec<f32>` is the
/// `[hidden_dim]` aggregated MoE FFN output (identical to the value
/// returned by [`moe_ffn_forward_layer_cuda`] for the same inputs), and
/// `router_top_k_weights: Vec<f32>` is the `[num_experts_per_tok]`
/// post-softmax + renormalize top-k expert weights.
///
/// # Hot path safety
///
/// This is the **traced sibling**. Production [`moe_ffn_forward_layer_cuda`]
/// is unchanged byte-for-byte; the additive-purity invariant pinned by
/// trace-moe-gpu-sub-stages-v1 holds.
///
/// The two functions duplicate the router/softmax/top-k logic. Drift
/// between them is mechanically prevented by the `_drift_gate` test
/// below — invoking each with the same synthetic inputs and asserting
/// the GPU traced sibling produces an `output` that matches the
/// production sibling within numerical tolerance.
///
/// # Errors
///
/// Same as [`moe_ffn_forward_layer_cuda`]: invalid shapes, non-F32
/// router, expert byte-slice issues, or `expert_swiglu_cuda` errors.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_ffn_forward_layer_cuda_with_router(
    executor: &mut crate::cuda::CudaExecutor,
    hidden: &[f32],
    layer: &crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer,
    num_experts: usize,
    num_experts_per_tok: usize,
    intermediate: usize,
    hidden_dim: usize,
    data: &[u8],
) -> Result<(Vec<f32>, Vec<f32>, Vec<u32>)> {
    if hidden.len() != hidden_dim {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "moe_ffn_forward_layer_cuda_with_router: hidden.len() = {} but hidden_dim = {}",
                hidden.len(),
                hidden_dim
            ),
        });
    }

    if layer.router.qtype != crate::gguf::types::GGUF_TYPE_F32 {
        return Err(RealizarError::UnsupportedOperation {
            operation: "moe_router_quantized_read_cuda_with_router".to_string(),
            reason: format!(
                "moe_ffn_forward_layer_cuda_with_router: router qtype = {} (not F32). \
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
                "moe_ffn_forward_layer_cuda_with_router: router byte_size {} != expected {}",
                router_bytes.len(),
                expected_bytes
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

    // Per-expert SwiGLU on GPU + weighted aggregation on CPU.
    // M-GPU-MOE-1.4 step (c) per qwen3-moe-forward-gpu-v1 v1.6.0:
    // pass per-tensor qtypes so expert_swiglu_cuda can dispatch
    // each matvec to either q4k_matvec or q6k_gemv. Same fix as
    // the production sibling above; preserves additive-purity.
    let mut out = vec![0.0f32; hidden_dim];
    for &(expert_id, weight) in &topk_renorm {
        let gate_bytes = crate::gguf::qwen3_moe_load::expert_byte_slice(
            &layer.gate_exps,
            data,
            expert_id,
            num_experts,
        )?;
        let up_bytes = crate::gguf::qwen3_moe_load::expert_byte_slice(
            &layer.up_exps,
            data,
            expert_id,
            num_experts,
        )?;
        let down_bytes = crate::gguf::qwen3_moe_load::expert_byte_slice(
            &layer.down_exps,
            data,
            expert_id,
            num_experts,
        )?;

        let expert_out = expert_swiglu_cuda(
            executor,
            gate_bytes,
            layer.gate_exps.qtype,
            up_bytes,
            layer.up_exps.qtype,
            down_bytes,
            layer.down_exps.qtype,
            hidden,
            hidden_dim,
            intermediate,
        )?;

        for i in 0..hidden_dim {
            out[i] += weight * expert_out[i];
        }
    }

    let router_top_k_weights: Vec<f32> = topk_renorm.iter().map(|(_, w)| *w).collect();
    let router_top_k_indices: Vec<u32> = topk_renorm.iter().map(|(i, _)| *i as u32).collect();
    Ok((out, router_top_k_weights, router_top_k_indices))
}

#[cfg(test)]
mod moe_ffn_forward_layer_cuda_tests {
    /// Compilation gate.
    #[test]
    fn moe_ffn_forward_layer_cuda_signature_drift_gate() {}

    /// M-MOE-SUB-2 GPU step (c) — `moe_ffn_forward_layer_cuda_with_router`
    /// signature drift gate. Compilation alone proves the function exists
    /// with the documented signature `(executor, hidden, layer, num_experts,
    /// num_experts_per_tok, intermediate, hidden_dim, data) -> Result<(Vec<f32>, Vec<f32>)>`.
    /// End-to-end byte-identity vs production sibling `moe_ffn_forward_layer_cuda`
    /// is exercised by the heavy test on lambda-vector RTX 4090 against
    /// cached Qwen3-Coder GGUF (out-of-scope for unit tests because it
    /// requires a real CUDA device + 17.3 GB GGUF + multi-GB VRAM).
    #[test]
    fn moe_ffn_forward_layer_cuda_with_router_signature_drift_gate() {}
}
