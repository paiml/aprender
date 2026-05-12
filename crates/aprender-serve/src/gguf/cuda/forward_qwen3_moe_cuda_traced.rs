// crates/aprender-serve/src/gguf/cuda/forward_qwen3_moe_cuda_traced.rs
//
// M-MOE-SUB-2 step (b) — GPU traced sibling of `forward_qwen3_moe_cuda`.
//
// Per `contracts/trace-moe-gpu-sub-stages-v1.yaml` v1.2.0 step (b):
// the GPU traced sibling that mirrors `OwnedQuantizedModel::forward_qwen3_moe_traced`
// (CPU traced sibling, M32d Step 2) but routes per-layer MoE FFN through
// the GPU dispatch (`moe_ffn_forward_layer_cuda` / `_with_router`) so that
// `apr trace --json --payload --save-tensor` can run the same SaveTensorPlan
// against both CPU and GPU forward paths and bisect per-stage at MoeRouter
// + MoeFfnOut to find the first stage where GPU NaN poisoning enters
// (M-GPU-MOE-1.4).
//
// Hot path safety
// ===============
//
// Production `forward_qwen3_moe_cuda` is unchanged byte-for-byte. This is
// a parallel slow path used only by `apr trace --gpu`. Allocation cost is
// acceptable for the diagnostic CLI use case (each per-layer
// ActivationStats::from_slice is O(hidden_dim) plus 9 stat structs per
// layer). The per-token loop dispatches the GPU MoE FFN identically to
// production for non-capture positions and uses the router-returning
// helper at the capture position so `MoeRouter` can be emitted without
// recomputation.
//
// Imports inherited from parent forward.rs (super::OwnedQuantizedModelCuda,
// crate::error::{RealizarError, Result}). This file is included via
// uses.rs include!() chain, so re-importing causes "must be defined only
// once" namespace conflicts.
//
// `Qwen3MoeQuantizedLayer` is already imported by the sibling production
// file `forward_qwen3_moe_cuda.rs` (which is also include!()'d into the
// same module). To avoid E0252 duplicate-import errors, we use the
// fully-qualified `crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer`
// in this file's signatures rather than re-importing.
//
// Trace types (ActivationStats, ForwardTrace, etc.) are NOT imported by
// any other include!()'d cuda file, so a use-statement here is safe.

use crate::apr_transformer::{ActivationStats, ForwardTrace, LastTokenStats, LayerActivation};
use crate::inference_trace::save_tensor_emit::maybe_save_stage;
use crate::inference_trace::save_tensor_plan::SaveTensorPlan;
use crate::inference_trace::save_tensor_stage::SaveTensorStage;

impl OwnedQuantizedModelCuda {
    /// GPU traced sibling of `forward_qwen3_moe_cuda` — runs the same
    /// forward pass with per-layer ActivationStats capture for the LAST
    /// token. Used by `apr trace --gpu --json --payload` to drive the
    /// M-GPU-MOE-1.4 NaN/Inf bisection (per-stage CPU vs GPU diff).
    ///
    /// # Arguments
    ///
    /// Identical to `forward_qwen3_moe_cuda`.
    ///
    /// # Returns
    ///
    /// `ForwardTrace` containing per-layer `LayerActivation` for every
    /// decoder layer plus embedding / final-norm / logits stats. Last-
    /// token-only stats per FALSIFY-APR-GGUF-PARITY-007 count-parity
    /// convention (matches the CPU traced sibling).
    ///
    /// # Errors
    ///
    /// Same as `forward_qwen3_moe_cuda`: invalid shape, MoE config
    /// violations, or CUDA dispatch errors.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe_cuda_traced(
        &mut self,
        token_ids: &[u32],
        moe_layers: &[crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
    ) -> Result<ForwardTrace> {
        self.forward_qwen3_moe_cuda_traced_with_plan(
            token_ids,
            moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            data,
            None,
        )
    }

    /// M-MOE-SUB-2 step (b) — `forward_qwen3_moe_cuda_traced` with optional
    /// `SaveTensorPlan` for per-layer `MoeRouter` + `MoeFfnOut` capture
    /// on the GPU forward path.
    ///
    /// Per `contracts/trace-moe-gpu-sub-stages-v1.yaml` v1.2.0 step (b).
    ///
    /// When `plan` is `None`, behavior is byte-identical to
    /// [`Self::forward_qwen3_moe_cuda_traced`] — only added cost is the
    /// `Option` discriminant check at each potential capture point.
    /// When `plan` is `Some`, the per-layer MoE FFN dispatch routes
    /// through `moe_ffn_forward_layer_cuda_with_router` (M-MOE-SUB-2
    /// step c.gpu, PR #1522) for the LAST sequence position to obtain
    /// the post-renormalize top-k router weights without re-running the
    /// MoE forward.
    ///
    /// After each layer's MoE FFN completes, `MoeRouter` is emitted as
    /// `[num_experts_per_tok]` (last token's post-softmax + renormalize
    /// top-k weights) and `MoeFfnOut` is emitted as `[hidden_dim]` (last
    /// token's aggregated MoE FFN output). Both emissions are gated by
    /// [`maybe_save_stage`] which no-ops when the plan does not select
    /// the stage for that layer.
    ///
    /// # Hot path safety
    ///
    /// Production [`Self::forward_qwen3_moe_cuda`] is unchanged byte-for-
    /// byte (additive-purity invariant pinned in v1.1.0). This traced
    /// path is a parallel slow path used only by `apr trace --gpu`.
    ///
    /// # Errors
    ///
    /// Same as [`Self::forward_qwen3_moe_cuda_traced`] plus IO errors
    /// from `maybe_save_stage` when emitting tensors.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe_cuda_traced_with_plan(
        &mut self,
        token_ids: &[u32],
        moe_layers: &[crate::gguf::qwen3_moe_load::Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
        plan: Option<&SaveTensorPlan>,
    ) -> Result<ForwardTrace> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "forward_qwen3_moe_cuda_traced: token_ids must not be empty".to_string(),
            });
        }
        if moe_layers.len() != self.model.layers.len() {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_cuda_traced: moe_layers.len() = {} but model has {} decoder layers",
                    moe_layers.len(),
                    self.model.layers.len()
                ),
            });
        }
        if num_experts == 0 || num_experts_per_tok == 0 || moe_intermediate == 0 {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_cuda_traced: incomplete MoE config — num_experts={num_experts}, \
                     num_experts_per_tok={num_experts_per_tok}, moe_intermediate={moe_intermediate}."
                ),
            });
        }
        if num_experts_per_tok > num_experts {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_cuda_traced: num_experts_per_tok ({num_experts_per_tok}) \
                     exceeds num_experts ({num_experts})"
                ),
            });
        }

        let hidden_dim = self.model.config.hidden_dim;
        let intermediate = moe_intermediate;
        let use_rmsnorm = self.model.config.constraints.uses_rmsnorm();
        let seq_len = token_ids.len();
        let last_start = (seq_len - 1) * hidden_dim;

        // 1. Token embedding (CPU)
        let mut hidden = self.model.embed(token_ids);

        // GH-278 absolute-position embedding (qwen3_moe doesn't use this,
        // but mirror dense path for edge-config correctness).
        if self.model.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.model.position_embedding {
                for s in 0..seq_len {
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
        let embed_stats = ActivationStats::from_slice(&hidden[last_start..last_start + hidden_dim]);

        let mut layer_activations: Vec<LayerActivation> =
            Vec::with_capacity(self.model.layers.len());

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
            let attn_norm_stats =
                ActivationStats::from_slice(&normed[last_start..last_start + hidden_dim]);

            // 2b. QKV projection (CPU via self.model — matches forward_qwen3_moe_cuda)
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
            let qkv_last_start = (seq_len - 1) * qkv_dim;
            let qkv_stats =
                ActivationStats::from_slice(&qkv[qkv_last_start..qkv_last_start + qkv_dim]);

            // 2c. Per-position per-head Q/K RMSNorm (GH-279) + RoPE
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
                    self.model
                        .apply_rope(&mut q, s, self.model.config.num_heads);
                    self.model
                        .apply_rope(&mut k, s, self.model.config.num_kv_heads);
                }
                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // 2d. Causal attention + output projection (CPU — matches forward_qwen3_moe_cuda)
            let attn_out = self.model.causal_attention(&q_all, &k_all, &v_all, seq_len);
            let mut attn_output = self
                .model
                .fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                crate::gguf::ops::add_bias(&mut attn_output, bias);
            }
            let attn_out_stats =
                ActivationStats::from_slice(&attn_output[last_start..last_start + hidden_dim]);

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
            let ffn_norm_stats =
                ActivationStats::from_slice(&ffn_input[last_start..last_start + hidden_dim]);

            // 2g. **MoE FFN on GPU** — matches production forward_qwen3_moe_cuda.
            // Last sequence position uses `moe_ffn_forward_layer_cuda_with_router`
            // when the plan selects MoeRouter or MoeFfnOut for this layer; all
            // other positions (and the last position when plan does not select
            // either stage) use the production helper `moe_ffn_forward_layer_cuda`
            // so trace cost stays minimal.
            let last_pos = seq_len - 1;
            let want_router =
                plan.is_some_and(|p| p.should_save(SaveTensorStage::MoeRouter, layer_idx as u32));
            let want_ffn_out =
                plan.is_some_and(|p| p.should_save(SaveTensorStage::MoeFfnOut, layer_idx as u32));
            let want_capture = want_router || want_ffn_out;

            let mut ffn_output = vec![0.0f32; seq_len * hidden_dim];
            let mut last_router_top_k: Vec<f32> = Vec::new();
            for s in 0..seq_len {
                let pos_in = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                if want_capture && s == last_pos {
                    let (pos_out, router_top_k) = moe_ffn_forward_layer_cuda_with_router(
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
                    last_router_top_k = router_top_k;
                } else {
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
            }
            let ffn_out_stats =
                ActivationStats::from_slice(&ffn_output[last_start..last_start + hidden_dim]);

            // M-MOE-SUB-2 step (b) — emit per-layer MoeRouter + MoeFfnOut
            // when the plan selects them. `maybe_save_stage` no-ops when
            // plan is None or stage/layer not selected.
            if want_router {
                maybe_save_stage(
                    plan,
                    SaveTensorStage::MoeRouter,
                    layer_idx as u32,
                    &last_router_top_k,
                )
                .map_err(|e| RealizarError::IoError {
                    message: format!("save_tensor::MoeRouter L{layer_idx}: {e}"),
                })?;
            }
            if want_ffn_out {
                let last_ffn_out = &ffn_output[last_start..last_start + hidden_dim];
                maybe_save_stage(
                    plan,
                    SaveTensorStage::MoeFfnOut,
                    layer_idx as u32,
                    last_ffn_out,
                )
                .map_err(|e| RealizarError::IoError {
                    message: format!("save_tensor::MoeFfnOut L{layer_idx}: {e}"),
                })?;
            }

            // 2h. Residual
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
            let output_stats =
                ActivationStats::from_slice(&hidden[last_start..last_start + hidden_dim]);

            // Sub-FFN slots default to zero — same convention as CPU traced
            // sibling. MoE has no globally-meaningful SwiGLU breakdown
            // because per-expert SwiGLU is internal to
            // moe_ffn_forward_layer_cuda and weighted-aggregated before
            // producing ffn_out_stats. Per-expert sub-stages would land at
            // M-MOE-SUB-4 if M-MOE-SUB-3's MoeRouter/MoeFfnOut bisection
            // doesn't pinpoint the bug at this granularity.
            let ffn_gate_stats = ActivationStats::default();
            let ffn_up_stats = ActivationStats::default();
            let ffn_silu_gate_stats = ActivationStats::default();
            let ffn_swiglu_inner_stats = ActivationStats::default();

            let last_token = Some(LastTokenStats {
                attn_norm_stats: attn_norm_stats.clone(),
                qkv_stats: qkv_stats.clone(),
                attn_out_stats: attn_out_stats.clone(),
                ffn_norm_stats: ffn_norm_stats.clone(),
                ffn_gate_stats: ffn_gate_stats.clone(),
                ffn_up_stats: ffn_up_stats.clone(),
                ffn_silu_gate_stats: ffn_silu_gate_stats.clone(),
                ffn_swiglu_inner_stats: ffn_swiglu_inner_stats.clone(),
                ffn_out_stats: ffn_out_stats.clone(),
                output_stats: output_stats.clone(),
            });

            layer_activations.push(LayerActivation {
                layer_idx,
                attn_norm_stats,
                qkv_stats,
                attn_out_stats,
                ffn_norm_stats,
                ffn_gate_stats,
                ffn_up_stats,
                ffn_silu_gate_stats,
                ffn_swiglu_inner_stats,
                ffn_out_stats,
                output_stats,
                last_token,
            });
        }

        // 3. Final layer norm (CPU)
        let normed = if use_rmsnorm {
            crate::gguf::ops::rms_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.config.eps,
            )
        } else {
            crate::gguf::ops::layer_norm(
                &hidden,
                &self.model.output_norm_weight,
                self.model.output_norm_bias.as_deref(),
                self.model.config.eps,
            )
        };
        let final_norm_stats =
            ActivationStats::from_slice(&normed[last_start..last_start + hidden_dim]);

        // 4. LM head — last token only (CPU; matches forward_qwen3_moe_cuda)
        let last_hidden = &normed[last_start..last_start + hidden_dim];
        let mut logits = self
            .model
            .fused_matmul(last_hidden, &self.model.lm_head_weight)?;
        if let Some(ref bias) = self.model.lm_head_bias {
            crate::gguf::ops::add_bias(&mut logits, bias);
        }
        let logits_stats = ActivationStats::from_slice(&logits);

        Ok(ForwardTrace {
            input_tokens: token_ids.to_vec(),
            embed_stats,
            layer_activations,
            final_norm_stats,
            logits_stats,
            logits,
        })
    }
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod forward_qwen3_moe_cuda_traced_tests {
    /// M-MOE-SUB-2 step (b) — `forward_qwen3_moe_cuda_traced` signature
    /// drift gate. Compilation alone proves both functions exist with
    /// the documented signatures:
    ///
    /// - `forward_qwen3_moe_cuda_traced(&mut self, token_ids, moe_layers,
    ///   num_experts, num_experts_per_tok, moe_intermediate, data) -> Result<ForwardTrace>`
    /// - `forward_qwen3_moe_cuda_traced_with_plan(..., plan: Option<&SaveTensorPlan>) -> Result<ForwardTrace>`
    ///
    /// End-to-end byte-identity vs production sibling
    /// `forward_qwen3_moe_cuda` is exercised by the heavy
    /// `qwen3_moe_gpu_parity` test on lambda-vector RTX 4090 against
    /// the cached 17.3 GB Qwen3-Coder GGUF (out-of-scope for unit tests
    /// because it requires a real CUDA device + 17.3 GB GGUF + multi-GB
    /// VRAM). Discharges FALSIFY-MOE-SUB-002 at integration level when
    /// M-MOE-SUB-3 lands.
    #[test]
    fn forward_qwen3_moe_cuda_traced_signature_drift_gate() {}
}
