//! M32d Step 2 (companion `claude-code-parity-apr-poc.md` § "M32d FAST PATH") —
//! `forward_qwen3_moe_traced` per-layer ActivationStats variant of
//! `forward_qwen3_moe`.
//!
//! ## Purpose
//!
//! Step 2 of the M34 five-whys FAST PATH plan converts `apr trace --json
//! --payload` from "returns null per-layer stats for qwen3_moe" to "returns
//! finite per-layer L2 + dim-mean/std for every transformer block". Without
//! this, Step 3 (per-layer cosine bisection vs HF FP16 reference) has no
//! input.
//!
//! ## Method
//!
//! Mirrors `OwnedQuantizedModel::forward_qwen3_moe` step-for-step. After each
//! stat boundary in the layer loop, grab the LAST token's hidden-state slice
//! and compute `ActivationStats::from_slice`. Sub-FFN slots
//! (`ffn_gate_stats`, `ffn_up_stats`, `ffn_silu_gate_stats`,
//! `ffn_swiglu_inner_stats`) default to zero — MoE has no globally meaningful
//! SwiGLU breakdown because the per-expert SwiGLU is internal to
//! `moe_ffn_forward_layer` and is weighted+aggregated across top-k experts
//! before producing `ffn_out`. If component-level breakdown becomes needed
//! at Step 4, the MoE-specific breakdown is a separate add (router output,
//! per-expert contribution, etc.).
//!
//! ## Hot path safety
//!
//! Production `forward_qwen3_moe` is unchanged. This is a parallel slow path
//! used only by `apr trace`. Allocation cost is acceptable for the diagnostic
//! CLI use case.

use crate::apr_transformer::{ActivationStats, ForwardTrace, LastTokenStats, LayerActivation};
use crate::error::Result;
use crate::gguf::ops;
use crate::gguf::qwen3_moe_load::{
    moe_ffn_forward_layer, moe_ffn_forward_layer_with_router, Qwen3MoeQuantizedLayer,
};
use crate::gguf::OwnedQuantizedModel;
use crate::inference_trace::save_tensor_emit::maybe_save_stage;
use crate::inference_trace::save_tensor_plan::SaveTensorPlan;
use crate::inference_trace::save_tensor_stage::SaveTensorStage;

impl OwnedQuantizedModel {
    /// Run a single forward pass for a Qwen3-MoE-arch model and capture
    /// per-layer activation statistics for the LAST token.
    ///
    /// Mirrors `Self::forward_qwen3_moe` numerically; differs only in stat
    /// capture. Used by `apr trace --json --payload` to drive M34 FAST PATH
    /// Step 3 (per-layer cosine bisection vs HF FP16 reference).
    ///
    /// # Arguments
    ///
    /// Identical to `forward_qwen3_moe`.
    ///
    /// # Returns
    ///
    /// `ForwardTrace` containing per-layer `LayerActivation` for every
    /// decoder layer plus embedding/final-norm/logit stats. Last-token-only
    /// stats per FALSIFY-APR-GGUF-PARITY-007 count-parity convention.
    ///
    /// # Errors
    ///
    /// Same as `forward_qwen3_moe`: invalid shape, MoE config violations,
    /// or fused-matmul kernel errors.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe_traced(
        &self,
        token_ids: &[u32],
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
    ) -> Result<ForwardTrace> {
        self.forward_qwen3_moe_traced_with_plan(
            token_ids,
            moe_layers,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            data,
            None,
        )
    }

    /// M-MOE-SUB-2 step (a) — `forward_qwen3_moe_traced` with optional
    /// `SaveTensorPlan` for per-layer `MoeRouter` + `MoeFfnOut` capture.
    ///
    /// Per `contracts/trace-moe-gpu-sub-stages-v1.yaml` v1.1.0 step (a).
    ///
    /// When `plan` is `None`, behavior is byte-identical to
    /// [`Self::forward_qwen3_moe_traced`] — the only added cost is the
    /// `Option` discriminant check at each potential capture point. The
    /// plan-aware code path uses [`moe_ffn_forward_layer_with_router`]
    /// (M-MOE-SUB-2 step c, M68) at the last sequence position to obtain
    /// the top-k router weights without re-running the MoE forward.
    ///
    /// When `plan` is `Some`, after each layer's MoE FFN completes,
    /// `MoeRouter` is emitted as `[num_experts_per_tok]` post-softmax
    /// renormalize top-k weights for the last token's MoE router output
    /// (per the contract's tensor shape spec), and `MoeFfnOut` is
    /// emitted as `[hidden_dim]` aggregated MoE FFN output for the last
    /// token.
    ///
    /// Both emissions are gated by [`maybe_save_stage`] which no-ops
    /// when the plan does not select the stage for that layer.
    ///
    /// # Hot path safety
    ///
    /// Production [`OwnedQuantizedModel::forward_qwen3_moe`] is unchanged
    /// (additive-purity invariant pinned in v1.1.0). The traced path
    /// `forward_qwen3_moe_traced` was already a slow path used only by
    /// `apr trace`; this adds an `Option` parameter to it without
    /// changing its `plan == None` behavior. The default-arg-style
    /// delegate above preserves the public API.
    ///
    /// # Errors
    ///
    /// Same as [`Self::forward_qwen3_moe_traced`] plus IO errors from
    /// `maybe_save_stage` when emitting tensors.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe_traced_with_plan(
        &self,
        token_ids: &[u32],
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
        plan: Option<&SaveTensorPlan>,
    ) -> Result<ForwardTrace> {
        let hidden_dim = self.config.hidden_dim;

        if token_ids.is_empty() {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: "forward_qwen3_moe_traced: token_ids must not be empty".to_string(),
            });
        }
        if moe_layers.len() != self.layers.len() {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_traced: moe_layers.len() = {} but model has {} decoder layers",
                    moe_layers.len(),
                    self.layers.len()
                ),
            });
        }
        if num_experts == 0 || num_experts_per_tok == 0 || moe_intermediate == 0 {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe_traced: incomplete MoE config — num_experts={num_experts}, \
                     num_experts_per_tok={num_experts_per_tok}, moe_intermediate={moe_intermediate}."
                ),
            });
        }

        let seq_len = token_ids.len();
        let last_start = (seq_len - 1) * hidden_dim;

        // 1. Token embedding
        let mut hidden = self.embed(token_ids);
        if self.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.position_embedding {
                for (s, _) in token_ids.iter().enumerate() {
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

        let use_rmsnorm = self.config.constraints.uses_rmsnorm();
        let intermediate = moe_intermediate;

        let mut layer_activations: Vec<LayerActivation> = Vec::with_capacity(self.layers.len());

        // 2. Per-layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention norm
            let normed = if use_rmsnorm {
                ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(
                    &hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                )
            };
            let attn_norm_stats =
                ActivationStats::from_slice(&normed[last_start..last_start + hidden_dim]);

            // 2b. QKV projection
            let qkv_dim = layer.qkv_weight.out_dim();
            let q_dim = layer.qkv_weight.q_dim_for_config(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.hidden_dim,
                self.config.head_dim(),
            );
            let k_dim = layer.qkv_weight.k_dim_for_config(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.hidden_dim,
                self.config.head_dim(),
            );
            let v_dim = layer.qkv_weight.v_dim_for_config(
                self.config.num_heads,
                self.config.num_kv_heads,
                self.config.hidden_dim,
                self.config.head_dim(),
            );
            let mut qkv = self.qkv_matmul(&normed, &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias {
                ops::add_bias(&mut qkv, bias);
            }
            let qkv_last_start = (seq_len - 1) * qkv_dim;
            let qkv_stats =
                ActivationStats::from_slice(&qkv[qkv_last_start..qkv_last_start + qkv_dim]);

            // 2c. Per-position per-head Q/K RMSNorm (GH-279, Qwen3) + RoPE +
            // extract Q/K/V. Mirrors forward_qwen3_moe::forward_qwen3_moe
            // post-Step-5 fix (M32d FAST PATH) so the diagnostic trace shows
            // the same numerics as the production path.
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * k_dim);
            let mut v_all = Vec::with_capacity(seq_len * v_dim);
            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q = qkv[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv[qkv_start + q_dim..qkv_start + q_dim + k_dim].to_vec();
                let v = &qkv[qkv_start + q_dim + k_dim..qkv_start + q_dim + k_dim + v_dim];

                // GH-279: per-head Q/K RMSNorm AFTER bias, BEFORE RoPE.
                if let Some(ref q_norm) = layer.attn_q_norm_weight {
                    ops::apply_per_head_rms_norm(
                        &mut q,
                        q_norm,
                        self.config.num_heads,
                        self.config.eps,
                    );
                }
                if let Some(ref k_norm) = layer.attn_k_norm_weight {
                    ops::apply_per_head_rms_norm(
                        &mut k,
                        k_norm,
                        self.config.num_kv_heads,
                        self.config.eps,
                    );
                }

                if self.config.constraints.uses_rope() {
                    self.apply_rope(&mut q, s, self.config.num_heads);
                    self.apply_rope(&mut k, s, self.config.num_kv_heads);
                }
                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // 2d. Causal attention + output projection
            let attn_out = self.causal_attention(&q_all, &k_all, &v_all, seq_len);
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }
            let attn_out_stats =
                ActivationStats::from_slice(&attn_output[last_start..last_start + hidden_dim]);

            // 2e. Residual
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Pre-FFN norm
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm(&hidden, ffn_norm, self.config.eps)
                } else {
                    ops::layer_norm(
                        &hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                    )
                }
            } else {
                hidden.clone()
            };
            let ffn_norm_stats =
                ActivationStats::from_slice(&ffn_input[last_start..last_start + hidden_dim]);

            // 2g. MoE FFN
            //
            // M-MOE-SUB-2 step (a): when `plan` selects MoeRouter or
            // MoeFfnOut for this layer, capture the LAST sequence
            // position's router weights via `moe_ffn_forward_layer_with_router`.
            // For all other positions (and for the last position when
            // plan does not select either stage), use the production
            // helper `moe_ffn_forward_layer` so trace cost stays minimal.
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
                    let (pos_out, router_top_k) = moe_ffn_forward_layer_with_router(
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
                    let pos_out = moe_ffn_forward_layer(
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

            // M-MOE-SUB-2 step (a) — emit per-layer MoeRouter + MoeFfnOut
            // when the plan selects them. `maybe_save_stage` no-ops when
            // plan is None or stage/layer not selected.
            if want_router {
                maybe_save_stage(
                    plan,
                    SaveTensorStage::MoeRouter,
                    layer_idx as u32,
                    &last_router_top_k,
                )
                .map_err(|e| crate::error::RealizarError::IoError {
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
                .map_err(|e| crate::error::RealizarError::IoError {
                    message: format!("save_tensor::MoeFfnOut L{layer_idx}: {e}"),
                })?;
            }

            // 2h. Residual into hidden
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
            let output_stats =
                ActivationStats::from_slice(&hidden[last_start..last_start + hidden_dim]);

            // Sub-FFN slots default to zero — MoE has no globally-meaningful
            // SwiGLU breakdown (per-expert SwiGLU is internal to
            // moe_ffn_forward_layer and weighted-aggregated before producing
            // ffn_out_stats). Step 4 of M34 FAST PATH may add MoE-specific
            // sub-component breakdown (router output, per-expert
            // contribution).
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

        // 3. Final layer norm
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &self.output_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(
                &hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
            )
        };
        let final_norm_stats =
            ActivationStats::from_slice(&normed[last_start..last_start + hidden_dim]);

        // 4. LM head — last token only
        let last_hidden = &normed[last_start..last_start + hidden_dim];
        let mut logits = self.fused_matmul(last_hidden, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
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
