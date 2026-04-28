// SHIP-007 PR A: GGUF forward_traced scaffold
//
// Implements `OwnedQuantizedModel::forward_traced` mirroring the APR-side
// `AprTransformer::forward_traced`. PR A populates the 6 non-FFN
// `LayerActivation` fields per layer (attn_norm, qkv, attn_out, ffn_norm,
// ffn_out, output) and default-zeros the 4 sub-FFN fields (ffn_gate,
// ffn_up, ffn_silu_gate, ffn_swiglu_inner) — those are filled in PR B.
//
// Methodology: per `project_ship_007_gguf_forward_traced_plan.md` Option A
// (full clone of the orchestrator). Hot-path safety preserved — production
// `forward_single_with_scratch` is unchanged.
//
// Spec reference: SHIP-TWO-001 §17 + §23 (layer-3 ffn_swigl narrowed to
// the SHIP-007 surface) + §26.4 (P3 binding criterion: APR vs GGUF
// layer-3 ffn_swigl ratio).

use crate::apr_transformer::{ActivationStats, ForwardTrace, LayerActivation, TracedForward};
use crate::error::Result;
use crate::gguf::inference_types::InferenceScratchBuffer;
use crate::gguf::model::OwnedQuantizedModel;
use crate::gguf::ops;
use crate::gguf::runtime::OwnedQuantizedKVCache;

const GGUF_TYPE_Q4_K: u32 = 12; // mirror of constant in results.rs

impl OwnedQuantizedModel {
    /// Run forward pass and capture per-layer activation statistics.
    ///
    /// Mirrors `AprTransformer::forward_traced` so the GGUF inference path
    /// emits comparable telemetry. Used by `apr trace --payload` to
    /// produce the per-layer ffn_swigl std needed for the SHIP-007
    /// APR-vs-GGUF bisection (spec §26.4).
    ///
    /// PR A populates 6 non-FFN fields per layer; the 4 sub-FFN fields
    /// (`ffn_gate_stats`, `ffn_up_stats`, `ffn_silu_gate_stats`,
    /// `ffn_swiglu_inner_stats`) default to zero. PR B clones
    /// `scratch_swiglu_ffn` into `_traced` and populates them.
    ///
    /// # Errors
    ///
    /// Returns error if inference fails or if `token_ids` is empty.
    pub fn forward_traced(&self, token_ids: &[u32]) -> Result<ForwardTrace> {
        if token_ids.is_empty() {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        // Encoder-decoder paths (T5/Whisper) are not supported in PR A —
        // the orchestrator structure differs and would need its own clone.
        // This is acknowledged in the plan memory as out of scope for PR A.
        if !self.encoder_layers.is_empty() {
            return Err(crate::error::RealizarError::UnsupportedOperation {
                operation: "forward_traced".to_string(),
                reason: "encoder-decoder models not supported in PR A (decoder-only scaffold)"
                    .to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let kv_dim = self.config.kv_dim();
        let num_layers = self.layers().len();
        let max_seq = token_ids.len();

        // Allocate scratch + cache. PR A is a slow path; allocation cost is
        // acceptable for the diagnostic CLI use case.
        let mut scratch = InferenceScratchBuffer::from_config(&self.config);
        let mut cache = OwnedQuantizedKVCache::new(num_layers, kv_dim, max_seq);

        // Phase 1: prefill all tokens except the last via the existing
        // production path. This fills the KV cache and matches production
        // semantics exactly. We do NOT capture stats from these tokens —
        // per the plan memory, the LAST token's layer states are the
        // capture target (matches APR semantics: one LayerActivation per
        // layer).
        for (pos, &tok) in token_ids[..token_ids.len() - 1].iter().enumerate() {
            self.forward_single_with_scratch(tok, &mut cache, pos, &mut scratch)?;
        }

        // Phase 2: process the LAST token through an INLINED orchestrator
        // that captures stats at each layer boundary. The structure mirrors
        // `forward_single_with_scratch` (results.rs:447-587) exactly so the
        // numerical path is identical — only stat capture is added.
        let last_token = *token_ids.last().expect("non-empty checked above");
        let last_position = token_ids.len() - 1;
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();
        let use_q8k_path = hidden_dim.is_multiple_of(256);

        // 1. Token embedding + position (matches results.rs:463-477)
        self.embed_into(last_token, &mut scratch.hidden);
        if self.config.constraints.uses_absolute_positions() {
            if let Some(pos_emb) = self.position_embedding() {
                let start = last_position * hidden_dim;
                let end = start + hidden_dim;
                if end <= pos_emb.len() {
                    for i in 0..hidden_dim {
                        scratch.hidden[i] += pos_emb[start + i];
                    }
                }
            }
        }
        let embed_stats = ActivationStats::from_slice(&scratch.hidden[..hidden_dim]);

        let mut layer_activations: Vec<LayerActivation> = Vec::with_capacity(num_layers);

        // 2. Layer loop with inline stat capture
        for (layer_idx, layer) in self.layers().iter().enumerate() {
            // 2a. Attn norm → scratch.normed
            if use_rmsnorm {
                ops::rms_norm_into(
                    &scratch.hidden,
                    &layer.attn_norm_weight,
                    self.config.eps,
                    &mut scratch.normed,
                );
            } else {
                ops::layer_norm_into(
                    &scratch.hidden,
                    &layer.attn_norm_weight,
                    layer.attn_norm_bias.as_deref(),
                    self.config.eps,
                    &mut scratch.normed,
                );
            }
            let attn_norm_stats = ActivationStats::from_slice(&scratch.normed[..hidden_dim]);

            // 2b-2e. Attention block (QKV proj, attention, output proj, residual)
            // Encapsulated helper writes scratch.qkv (post-projection) and
            // scratch.attn_proj (post-output-projection); residual lands in
            // scratch.hidden.
            self.scratch_attention_block(
                layer_idx,
                layer,
                &mut scratch,
                &mut cache,
                last_position,
                use_q8k_path,
                hidden_dim,
            )?;
            // QKV stats: read combined QKV projection (q_dim + 2 * kv_dim)
            let q_dim = self.config.q_dim();
            let qkv_dim = q_dim + 2 * kv_dim;
            let qkv_stats = ActivationStats::from_slice(&scratch.qkv[..qkv_dim]);
            // attn_out stats: read attention output projection (hidden_dim)
            let attn_out_stats = ActivationStats::from_slice(&scratch.attn_proj[..hidden_dim]);

            // 2f. Pre-FFN norm → scratch.normed
            if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm {
                    ops::rms_norm_into(
                        &scratch.hidden,
                        ffn_norm,
                        self.config.eps,
                        &mut scratch.normed,
                    );
                } else {
                    ops::layer_norm_into(
                        &scratch.hidden,
                        ffn_norm,
                        layer.ffn_norm_bias.as_deref(),
                        self.config.eps,
                        &mut scratch.normed,
                    );
                }
            } else {
                scratch.normed[..hidden_dim].copy_from_slice(&scratch.hidden[..hidden_dim]);
            }
            let ffn_norm_stats = ActivationStats::from_slice(&scratch.normed[..hidden_dim]);

            // 2g. FFN. PR B populates the 4 sub-FFN stat slots via the
            // `_traced` helper for the SwiGLU path; the GELU path keeps
            // sub-FFN slots at default-zero (no SwiGLU components).
            let mut ffn_gate_stats = ActivationStats::default();
            let mut ffn_up_stats = ActivationStats::default();
            let mut ffn_silu_gate_stats = ActivationStats::default();
            let mut ffn_swiglu_inner_stats = ActivationStats::default();
            if self.config.constraints.has_gate_ffn() {
                self.scratch_swiglu_ffn_traced(
                    layer_idx,
                    &mut scratch,
                    use_q8k_path,
                    hidden_dim,
                    intermediate_dim,
                    &mut ffn_gate_stats,
                    &mut ffn_up_stats,
                    &mut ffn_silu_gate_stats,
                    &mut ffn_swiglu_inner_stats,
                )?;
            } else {
                // GELU path: no SwiGLU sub-FFN components — leave the 4
                // sub-FFN slots at default-zero. This matches APR's
                // forward_traced semantics for non-SwiGLU models.
                self.scratch_gelu_ffn(
                    layer_idx,
                    &mut scratch,
                    use_q8k_path,
                    hidden_dim,
                    intermediate_dim,
                )?;
            }
            // ffn_out stats: read FFN down-projection output (hidden_dim)
            let ffn_out_stats = ActivationStats::from_slice(&scratch.ffn_down[..hidden_dim]);

            // 2h. FFN residual into scratch.hidden
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_down[i];
            }
            // output stats: post-residual hidden (the layer's output)
            let output_stats = ActivationStats::from_slice(&scratch.hidden[..hidden_dim]);

            // Construct LayerActivation. Order matches APR's
            // inference.rs:231-242. PR B: all 10 fields populated for
            // SwiGLU; GELU path leaves 4 sub-FFN at default-zero.
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
            });
        }

        // 3. Final layer norm → scratch.normed
        if use_rmsnorm {
            ops::rms_norm_into(
                &scratch.hidden,
                self.output_norm_weight(),
                self.config.eps,
                &mut scratch.normed,
            );
        } else {
            ops::layer_norm_into(
                &scratch.hidden,
                self.output_norm_weight(),
                self.output_norm_bias(),
                self.config.eps,
                &mut scratch.normed,
            );
        }
        let final_norm_stats = ActivationStats::from_slice(&scratch.normed[..hidden_dim]);

        // 4. LM head → scratch.logits (mirror results.rs:557-584)
        let use_q8k_lm =
            hidden_dim.is_multiple_of(256) && self.lm_head_weight().qtype == GGUF_TYPE_Q4_K;
        if use_q8k_lm {
            use crate::quantize::{
                fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
            };
            let hidden_sb = hidden_dim / 256;
            quantize_activations_q8k_into(
                &scratch.normed[..hidden_dim],
                &mut scratch.q8k_hidden_scales[..hidden_sb],
                &mut scratch.q8k_hidden_quants[..hidden_dim],
            )?;
            fused_q4k_q8k_parallel_matvec_into(
                &self.lm_head_weight().data,
                &scratch.q8k_hidden_scales[..hidden_sb],
                &scratch.q8k_hidden_quants[..hidden_dim],
                self.lm_head_weight().in_dim,
                self.lm_head_weight().out_dim,
                &mut scratch.logits,
            )?;
        } else {
            self.fused_matmul_into(
                &scratch.normed[..hidden_dim],
                self.lm_head_weight(),
                &mut scratch.logits,
            )?;
        }
        let logits_stats = ActivationStats::from_slice(&scratch.logits);

        Ok(ForwardTrace {
            input_tokens: token_ids.to_vec(),
            embed_stats,
            layer_activations,
            final_norm_stats,
            logits_stats,
            logits: scratch.logits.clone(),
        })
    }
}

/// PMAT-216: Implement TracedForward trait for the GGUF backend.
///
/// Mirrors the CPU impl pattern from `apr_transformer::traced_forward.rs`:
/// the trait method delegates to the immutable inherent method.
impl TracedForward for OwnedQuantizedModel {
    fn forward_traced(&mut self, tokens: &[u32]) -> Result<ForwardTrace> {
        OwnedQuantizedModel::forward_traced(self, tokens)
    }
}
