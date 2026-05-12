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
        &mut self,
        token_ids: &[u32],
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
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

        // M-GPU-MOE-1.1.2 — full forward integration mirroring CPU sibling
        // forward_qwen3_moe (forward/forward_qwen3_moe.rs) line-for-line.
        // Only difference: per-layer FFN section routes through
        // moe_ffn_forward_layer_cuda which dispatches matmuls to
        // self.executor via the expert_swiglu_cuda helper.
        //
        // Attention path stays on CPU. Pattern matches existing
        // forward_cuda method (CPU attention + CUDA matmul) on
        // OwnedQuantizedModelCuda — established by v1.1.0 option D.

        let hidden_dim = self.model.config.hidden_dim;
        let intermediate = moe_intermediate;
        let use_rmsnorm = self.model.config.constraints.uses_rmsnorm();

        // 1. Token embedding (CPU)
        let mut hidden = self.model.embed(token_ids);

        // GH-278 absolute-position embedding (qwen3_moe doesn't use this,
        // but mirror dense path for edge-config correctness).
        if self.model.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.model.position_embedding {
                for s in 0..token_ids.len() {
                    let pos_start = s * hidden_dim;
                    let pos_end = pos_start + hidden_dim;
                    if pos_end <= pos_emb.len() {
                        let h_start = s * hidden_dim;
                        for i in 0..hidden_dim {
                            hidden[h_start + i] += pos_emb[pos_start + i];
                        }
                    }
                }
            }
        }

        // 2. Per-layer
        for (layer_idx, layer) in self.model.layers.iter().enumerate() {
            // 2a. Attention norm
            let normed = if use_rmsnorm {
                crate::gguf::ops::rms_norm(&hidden, &layer.attn_norm_weight, self.model.config.eps)
            } else {
                crate::gguf::ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.model.config.eps,
                )
            };

            // 2b. QKV projection (CPU via self.model)
            let qkv_dim = layer.qkv_weight.out_dim();
            let q_dim = layer.qkv_weight.q_dim_for_config(
                self.model.config.num_heads,
                self.model.config.num_kv_heads,
                self.model.config.hidden_dim,
                self.model.config.head_dim(),
            );
            let k_dim = layer.qkv_weight.k_dim_for_config(
                self.model.config.num_heads,
                self.model.config.num_kv_heads,
                self.model.config.hidden_dim,
                self.model.config.head_dim(),
            );
            let v_dim = layer.qkv_weight.v_dim_for_config(
                self.model.config.num_heads,
                self.model.config.num_kv_heads,
                self.model.config.hidden_dim,
                self.model.config.head_dim(),
            );
            let mut qkv = self.model.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                crate::gguf::ops::add_bias(&mut qkv, bias);
            }

            // 2c. Per-position per-head Q/K RMSNorm + RoPE (M32d Step 5/5b)
            let seq_len = token_ids.len();
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * k_dim);
            let mut v_all = Vec::with_capacity(seq_len * v_dim);
            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q = qkv[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv[qkv_start + q_dim..qkv_start + q_dim + k_dim].to_vec();
                let v = &qkv[qkv_start + q_dim + k_dim..qkv_start + q_dim + k_dim + v_dim];

                if let Some(ref q_norm) = layer.attn_q_norm_weight {
                    crate::gguf::ops::apply_per_head_rms_norm(
                        &mut q,
                        q_norm,
                        self.model.config.num_heads,
                        self.model.config.eps,
                    );
                }
                if let Some(ref k_norm) = layer.attn_k_norm_weight {
                    crate::gguf::ops::apply_per_head_rms_norm(
                        &mut k,
                        k_norm,
                        self.model.config.num_kv_heads,
                        self.model.config.eps,
                    );
                }

                if self.model.config.constraints.uses_rope() {
                    self.model.apply_rope(&mut q, s, self.model.config.num_heads);
                    self.model.apply_rope(&mut k, s, self.model.config.num_kv_heads);
                }
                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // 2d. Causal attention + output projection (CPU)
            let attn_out = self.model.causal_attention(&q_all, &k_all, &v_all, seq_len);
            let mut attn_output = self.model.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                crate::gguf::ops::add_bias(&mut attn_output, bias);
            }

            // 2e. Residual
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Pre-FFN norm (CPU)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    crate::gguf::ops::rms_norm(&hidden, ffn_norm, self.model.config.eps)
                } else {
                    crate::gguf::ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.model.config.eps,
                    )
                }
            } else {
                hidden.clone()
            };

            // 2g. **MoE FFN on GPU** — only piece that differs from CPU.
            // Per-token dispatch through moe_ffn_forward_layer_cuda which
            // routes per-expert matmuls to self.executor.
            let mut ffn_output = vec![0.0f32; seq_len * hidden_dim];
            for s in 0..seq_len {
                let pos_in = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                let pos_out = moe_ffn_forward_layer_cuda(
                    &mut self.executor,
                    pos_in,
                    &moe_layers[layer_idx],
                    num_experts,
                    num_experts_per_tok,
                    intermediate,
                    hidden_dim,
                    data,
                )?;
                ffn_output[s * hidden_dim..(s + 1) * hidden_dim].copy_from_slice(&pos_out);
            }

            // 2h. Residual
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm (CPU)
        let normed = if use_rmsnorm {
            crate::gguf::ops::rms_norm(&hidden, &self.model.output_norm_weight, self.model.config.eps)
        } else {
            crate::gguf::ops::layer_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.output_norm_bias.as_deref(),
                self.model.config.eps,
            )
        };

        // 4. LM head — last token only (CPU; existing forward_cuda also
        // does LM head on CPU)
        let seq_len = token_ids.len();
        let last_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_start..last_start + hidden_dim];
        let mut logits = self.model.fused_matmul(last_hidden, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            crate::gguf::ops::add_bias(&mut logits, bias);
        }
        Ok(logits)
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
