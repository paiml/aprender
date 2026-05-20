//! M32c.2.2.2.1.1 — `forward_qwen3_moe` method on `OwnedQuantizedModel`.
//!
//! Per the integration strategy in `contracts/qwen3-moe-forward-v1.yaml` v1.2.0
//! (PR #1123), this is the per-token forward pass for Qwen3-MoE-arch GGUF
//! models. It mirrors `OwnedQuantizedModel::forward` (the dense path in
//! `forward_fused_q4k.rs`) step-for-step EXCEPT at the FFN site, where it
//! calls `moe_ffn_forward_layer` (M32c.2.2.2.0) instead of the dense
//! gate/up/down dispatch.
//!
//! ## Reuse of existing primitives
//! All non-FFN steps (embedding, attention norm, QKV projection, RoPE,
//! causal attention, output projection, LM head) call the EXISTING
//! `&self` methods on `OwnedQuantizedModel`. No code is duplicated.
//!
//! ## What's NEW vs `forward`
//! - Two new parameters: `moe_layers: &[Qwen3MoeQuantizedLayer]` (M32c.1)
//!   + `data: &[u8]` (the file's mmapped bytes — caller holds the
//!     `MappedGGUFModel` for the lifetime of this call).
//! - At the FFN dispatch site, calls `moe_ffn_forward_layer` instead
//!   of the SwiGLU/GELU branch.
//!
//! ## What's UNCHANGED
//! - `OwnedQuantizedModel` struct fields. No new fields, no 99-site
//!   blast radius.
//! - All forward path components except FFN: bit-identical to the
//!   existing dense path.
//!
//! ## Stage in M32c.2.2.2.1
//! This is sub-slice .1.1. M32c.2.2.2.1.0 (helper extraction) was
//! found unnecessary — the existing `&self` methods on
//! `OwnedQuantizedModel` already serve as helpers for this method.
//! Sub-slices .1.2 (`run_qwen3_moe_generate`), .1.3 (dispatch flip),
//! .1.4 (live falsifier) follow.

use crate::error::Result;
use crate::gguf::ops;
use crate::gguf::qwen3_moe_load::{moe_ffn_forward_layer, Qwen3MoeQuantizedLayer};
use crate::gguf::OwnedQuantizedModel;

impl OwnedQuantizedModel {
    /// Run a single forward pass for a Qwen3-MoE-arch model.
    ///
    /// Mirrors `Self::forward` step-for-step except the FFN section,
    /// which calls `moe_ffn_forward_layer` per layer instead of the
    /// dense SwiGLU dispatch.
    ///
    /// # Arguments
    /// * `token_ids` — input token IDs.
    /// * `moe_layers` — per-layer Qwen3MoE expert tensor descriptors;
    ///   length must equal `self.layers.len()`. Built once via
    ///   `load_qwen3_moe_layer` per layer.
    /// * `data` — the file's mmapped byte slice (zero-copy from
    ///   `MappedGGUFModel::data()`). Borrowed by `moe_ffn_forward_layer`
    ///   for in-place fused dequant+matvec on each selected expert.
    ///
    /// # Returns
    /// Logits for the next-token prediction, length == `vocab_size`.
    ///
    /// # Errors
    /// Propagates errors from `moe_ffn_forward_layer` (mismatched
    /// dims, out-of-range expert, etc.) and from
    /// `OwnedQuantizedModel`'s existing fused-matmul kernels.
    ///
    /// # Pre-conditions
    /// - `moe_layers.len() == self.layers.len()`
    /// - `self.config.architecture` should canonicalize to
    ///   `"qwen3_moe"` (caller's responsibility).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_qwen3_moe(
        &self,
        token_ids: &[u32],
        moe_layers: &[Qwen3MoeQuantizedLayer],
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;

        if moe_layers.len() != self.layers.len() {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe: moe_layers.len() = {} but model has {} decoder layers",
                    moe_layers.len(),
                    self.layers.len()
                ),
            });
        }
        if num_experts == 0 || num_experts_per_tok == 0 || moe_intermediate == 0 {
            return Err(crate::error::RealizarError::InvalidShape {
                reason: format!(
                    "forward_qwen3_moe: incomplete MoE config — num_experts={num_experts}, \
                     num_experts_per_tok={num_experts_per_tok}, moe_intermediate={moe_intermediate}. \
                     Caller must supply all three from GGUF metadata."
                ),
            });
        }

        // 1. Token embedding
        let mut hidden = self.embed(token_ids);

        // GH-278: absolute-position embedding (qwen3_moe doesn't use this, but
        // mirror the dense path for correctness on edge configurations).
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

        let use_rmsnorm = self.config.constraints.uses_rmsnorm();
        let intermediate = moe_intermediate;

        // 2. Per-layer: attention (existing primitives) + MoE FFN (new)
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

            // 2c. Per-position per-head Q/K RMSNorm (GH-279, Qwen3) + RoPE +
            // extract Q/K/V.
            //
            // M32d FAST PATH Step 5 fix
            // (companion claude-code-parity-apr docs/specifications/
            //  claude-code-parity-apr-poc.md § "M32d FAST PATH"):
            // Qwen3 applies per-head RMSNorm to Q and K BETWEEN bias and
            // RoPE — see adaptive_ffn.rs:174-179 (GH-279) for the dense
            // path's reference implementation. This was missing from
            // forward_qwen3_moe and was the rank-3 prior (15%) in the
            // FAST PATH component-prior table. Surfaced by `apr trace
            // --payload`: layer std-dev grew 40× over 48 layers
            // (layer[0]=0.07 → layer[47]=2.82) — exact signature of
            // missing Q/K norm letting attention scores compound.
            let seq_len = token_ids.len();
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

            // 2e. Residual
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f+2g. Per-position MoE FFN block (norm + dispatch + residual).
            // Lifted into `moe_ffn_layer` helper (M32d Day 1 prep refactor)
            // so the upcoming `forward_single_qwen3_moe_with_cache` (M32d
            // step 3) can call the SAME per-token MoE FFN implementation.
            //
            // Math is byte-identical to the previous batch-norm + per-position
            // dispatch + outer residual version because rms_norm/layer_norm
            // are per-vector ops (no cross-token interaction).
            let mut tok_hidden = vec![0.0f32; hidden_dim];
            for s in 0..seq_len {
                tok_hidden.copy_from_slice(&hidden[s * hidden_dim..(s + 1) * hidden_dim]);
                self.moe_ffn_layer(
                    &mut tok_hidden,
                    layer,
                    &moe_layers[layer_idx],
                    num_experts,
                    num_experts_per_tok,
                    intermediate,
                    data,
                    use_rmsnorm,
                )?;
                hidden[s * hidden_dim..(s + 1) * hidden_dim].copy_from_slice(&tok_hidden);
            }
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

        // 4. LM head — last token only
        let seq_len = token_ids.len();
        let last_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_start..last_start + hidden_dim];
        let mut logits = self.fused_matmul(last_hidden, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }
        Ok(logits)
    }

    /// Per-token MoE FFN block — M32d Day 1 prep refactor.
    ///
    /// Mirrors the dense path's per-layer FFN block but routes the
    /// computation through `moe_ffn_forward_layer` (per-expert dispatch)
    /// instead of the dense gate × up × SwiGLU × down sequence.
    ///
    /// The body is lifted verbatim from `forward_qwen3_moe`'s per-position
    /// FFN block (formerly at the bottom of the per-layer loop). The
    /// outer `forward_qwen3_moe` calls this once per token in its
    /// `for s in 0..seq_len` loop. The upcoming
    /// `forward_single_qwen3_moe_with_cache` (M32d step 3) calls this
    /// exactly once per layer per generated token.
    ///
    /// Mutates `hidden_token` via the post-MoE residual add.
    ///
    /// # Arguments
    /// * `hidden_token` - Single-token hidden state, `[hidden_dim]`; mutated
    /// * `layer` - Dense layer struct (read for `ffn_norm_weight` + bias)
    /// * `moe_layer` - Per-layer MoE expert tensor descriptors
    /// * `num_experts`, `num_experts_per_tok`, `moe_intermediate` - MoE config
    /// * `data` - Mmapped GGUF byte slice (borrowed by `moe_ffn_forward_layer`
    ///   for in-place fused dequant+matvec on each selected expert)
    /// * `use_rmsnorm` - True for RMSNorm models; false for LayerNorm
    ///
    /// # Errors
    /// Propagates errors from `moe_ffn_forward_layer` (mismatched dims,
    /// out-of-range expert, etc.).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn moe_ffn_layer(
        &self,
        hidden_token: &mut [f32],
        layer: &crate::gguf::OwnedQuantizedLayer,
        moe_layer: &Qwen3MoeQuantizedLayer,
        num_experts: usize,
        num_experts_per_tok: usize,
        moe_intermediate: usize,
        data: &[u8],
        use_rmsnorm: bool,
    ) -> Result<()> {
        let hidden_dim = self.config.hidden_dim;
        debug_assert_eq!(
            hidden_token.len(),
            hidden_dim,
            "moe_ffn_layer expects per-token hidden of length hidden_dim"
        );

        // 2f. Pre-FFN norm — per-vector, math is identical whether applied
        // batched or per-token (rms_norm/layer_norm only mix elements
        // within a single hidden_dim vector, never across tokens).
        let ffn_input: Vec<f32> = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
            if use_rmsnorm {
                ops::rms_norm(hidden_token, ffn_norm, self.config.eps)
            } else {
                ops::layer_norm(
                    hidden_token,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            }
        } else {
            hidden_token.to_vec()
        };

        // 2g. MoE FFN — router + top-K experts + weighted sum.
        let pos_out = moe_ffn_forward_layer(
            &ffn_input,
            moe_layer,
            num_experts,
            num_experts_per_tok,
            moe_intermediate,
            hidden_dim,
            data,
        )?;

        // Residual add into the caller's hidden_token slice.
        for i in 0..hidden_dim {
            hidden_token[i] += pos_out[i];
        }
        Ok(())
    }
}
