
impl OwnedQuantizedModel {

    /// Q8K-accelerated up+gate computation for SwiGLU FFN (PMAT-260)
    ///
    /// Handles the 3 paths: fused Q4K up+gate, separate Q8K rayon matmuls, F32 fallback.
    /// Results are written into scratch.ffn_up and scratch.ffn_gate.
    fn scratch_q8k_up_gate(
        &self,
        layer_idx: usize,
        scratch: &mut InferenceScratchBuffer,
        use_q8k_path: bool,
        hidden_dim: usize,
    ) -> Result<()> {
        let layer = &self.layers[layer_idx];
        let gate_weight = layer.ffn_gate_weight.as_ref().expect("SwiGLU requires gate weight");

        if use_q8k_path {
            use crate::quantize::quantize_activations_q8k_into;
            let hidden_sb = hidden_dim / 256;
            quantize_activations_q8k_into(
                &scratch.normed[..hidden_dim],
                &mut scratch.q8k_hidden_scales[..hidden_sb],
                &mut scratch.q8k_hidden_quants[..hidden_dim],
            )?;

            let up_weight = &layer.ffn_up_weight;
            let q8k_scales = &scratch.q8k_hidden_scales[..hidden_sb];
            let q8k_quants = &scratch.q8k_hidden_quants[..hidden_dim];

            if up_weight.qtype == GGUF_TYPE_Q4_K && gate_weight.qtype == GGUF_TYPE_Q4_K {
                use crate::quantize::fused_q4k_q8k_ffn_up_gate_into;
                fused_q4k_q8k_ffn_up_gate_into(
                    &up_weight.data,
                    &gate_weight.data,
                    q8k_scales,
                    q8k_quants,
                    up_weight.in_dim,
                    up_weight.out_dim,
                    &mut scratch.ffn_up,
                    &mut scratch.ffn_gate,
                )?;
            } else if up_weight.qtype == gate_weight.qtype
                && up_weight.in_dim == gate_weight.in_dim
                && up_weight.out_dim == gate_weight.out_dim
                && (up_weight.qtype == GGUF_TYPE_Q4_K
                    || up_weight.qtype == GGUF_TYPE_Q5_K
                    || up_weight.qtype == GGUF_TYPE_Q6_K)
            {
                // Fused gate+up for same-type K-quant weights (PMAT-FFN-FUSION)
                self.fused_gate_up_matmul_into(
                    &scratch.normed[..hidden_dim],
                    gate_weight,
                    up_weight,
                    &mut scratch.ffn_gate,
                    &mut scratch.ffn_up,
                )?;
            } else {
                use crate::quantize::fused_q4k_q8k_parallel_matvec_into;
                let (up_result, gate_result) = rayon::join(
                    || {
                        if up_weight.qtype == GGUF_TYPE_Q4_K {
                            fused_q4k_q8k_parallel_matvec_into(
                                &up_weight.data,
                                q8k_scales,
                                q8k_quants,
                                up_weight.in_dim,
                                up_weight.out_dim,
                                &mut scratch.ffn_up,
                            )
                        } else {
                            self.fused_matmul_into(
                                &scratch.normed[..hidden_dim],
                                up_weight,
                                &mut scratch.ffn_up,
                            )
                        }
                    },
                    || {
                        if gate_weight.qtype == GGUF_TYPE_Q4_K {
                            fused_q4k_q8k_parallel_matvec_into(
                                &gate_weight.data,
                                q8k_scales,
                                q8k_quants,
                                gate_weight.in_dim,
                                gate_weight.out_dim,
                                &mut scratch.ffn_gate,
                            )
                        } else {
                            self.fused_matmul_into(
                                &scratch.normed[..hidden_dim],
                                gate_weight,
                                &mut scratch.ffn_gate,
                            )
                        }
                    },
                );
                up_result?;
                gate_result?;
            }
        } else {
            let up_weight = &layer.ffn_up_weight;
            self.fused_gate_up_matmul_into(
                &scratch.normed[..hidden_dim],
                gate_weight,
                up_weight,
                &mut scratch.ffn_gate,
                &mut scratch.ffn_up,
            )?;
        }
        Ok(())
    }

    /// Q8K-accelerated down projection for SwiGLU FFN (PMAT-260)
    fn scratch_q8k_down_projection(
        &self,
        layer_idx: usize,
        scratch: &mut InferenceScratchBuffer,
        intermediate_dim: usize,
        hidden_dim: usize,
    ) -> Result<()> {
        let layer = &self.layers[layer_idx];
        let use_q8k_down = intermediate_dim.is_multiple_of(256)
            && layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K;

        if use_q8k_down {
            use crate::quantize::{
                fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
            };
            let inter_sb = intermediate_dim / 256;
            quantize_activations_q8k_into(
                &scratch.ffn_gate[..intermediate_dim],
                &mut scratch.q8k_inter_scales[..inter_sb],
                &mut scratch.q8k_inter_quants[..intermediate_dim],
            )?;
            fused_q4k_q8k_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &scratch.q8k_inter_scales[..inter_sb],
                &scratch.q8k_inter_quants[..intermediate_dim],
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
                &mut scratch.ffn_down,
            )?;
        } else {
            self.fused_matmul_into(
                &scratch.ffn_gate[..intermediate_dim],
                &layer.ffn_down_weight,
                &mut scratch.ffn_down,
            )?;
        }
        if let Some(ref bias) = layer.ffn_down_bias {
            for i in 0..hidden_dim {
                scratch.ffn_down[i] += bias[i];
            }
        }
        Ok(())
    }

    /// Scratch-buffer attention block: QKV projection, attention, output projection (PMAT-260)
    ///
    /// Handles the full attention computation for a single layer:
    /// 1. QKV projection (Q8K-accelerated or F32 fallback)
    /// 2. QKV split, bias, RoPE
    /// 3. Attention with KV cache (GQA-aware)
    /// 4. Output projection (Q8K-accelerated or F32 fallback)
    /// 5. Residual connection into scratch.hidden
    fn scratch_attention_block(
        &self,
        layer_idx: usize,
        layer: &OwnedQuantizedLayer,
        scratch: &mut InferenceScratchBuffer,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        use_q8k_path: bool,
        hidden_dim: usize,
    ) -> Result<()> {
        // GH-305: Use config.head_dim() (from GGUF metadata) — NOT hidden_dim / num_heads.
        // Qwen3-0.6B: hidden_dim=1024, num_heads=16, head_dim=128 → q_dim=2048 ≠ hidden_dim.
        let head_dim = self.config.head_dim();
        let num_kv_heads = self.config.num_kv_heads;
        let kv_dim = num_kv_heads * head_dim;
        let q_dim = self.config.q_dim();
        let k_dim = kv_dim;
        let v_dim = kv_dim;
        let qkv_dim = q_dim + k_dim + v_dim;

        // QKV projection (Q8K or F32)
        if use_q8k_path {
            use crate::quantize::quantize_activations_q8k_into;
            let hidden_sb = hidden_dim / 256;
            quantize_activations_q8k_into(
                &scratch.normed[..hidden_dim],
                &mut scratch.q8k_hidden_scales[..hidden_sb],
                &mut scratch.q8k_hidden_quants[..hidden_dim],
            )?;
            self.qkv_matmul_q8k_into(
                &scratch.normed,
                &layer.qkv_weight,
                &mut scratch.qkv[..qkv_dim],
                &scratch.q8k_hidden_scales[..hidden_sb],
                &scratch.q8k_hidden_quants[..hidden_dim],
            )?;
        } else {
            self.qkv_matmul_into(
                &scratch.normed,
                &layer.qkv_weight,
                &mut scratch.qkv[..qkv_dim],
            )?;
        }

        // Split QKV and apply bias + RoPE
        scratch.q[..q_dim].copy_from_slice(&scratch.qkv[..q_dim]);
        scratch.k[..k_dim].copy_from_slice(&scratch.qkv[q_dim..q_dim + k_dim]);
        scratch.v[..v_dim].copy_from_slice(&scratch.qkv[q_dim + k_dim..qkv_dim]);

        if let Some(ref bias) = layer.qkv_bias {
            for i in 0..q_dim {
                scratch.q[i] += bias[i];
            }
            for i in 0..k_dim {
                scratch.k[i] += bias[q_dim + i];
            }
            for i in 0..v_dim {
                scratch.v[i] += bias[q_dim + k_dim + i];
            }
        }

        // GH-279: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
        if let Some(ref q_norm) = layer.attn_q_norm_weight {
            ops::apply_per_head_rms_norm(
                &mut scratch.q[..q_dim],
                q_norm,
                self.config.num_heads,
                self.config.eps,
            );
        }
        if let Some(ref k_norm) = layer.attn_k_norm_weight {
            ops::apply_per_head_rms_norm(
                &mut scratch.k[..k_dim],
                k_norm,
                self.config.num_kv_heads,
                self.config.eps,
            );
        }

        // GH-278: Skip RoPE for models with learned position embeddings (GPT-2)
        if self.config.constraints.uses_rope() {
            self.apply_rope(&mut scratch.q[..q_dim], position, self.config.num_heads);
            self.apply_rope(&mut scratch.k[..k_dim], position, self.config.num_kv_heads);
        }

        // Attention computation
        let k_cache = cache.get_k(layer_idx);
        let v_cache = cache.get_v(layer_idx);

        if k_cache.is_empty() {
            if self.config.num_kv_heads < self.config.num_heads {
                let group_size = self.config.num_heads / self.config.num_kv_heads;
                for h in 0..self.config.num_heads {
                    let kv_head = h / group_size;
                    let src_start = kv_head * head_dim;
                    let dst_start = h * head_dim;
                    scratch.attn_out[dst_start..dst_start + head_dim]
                        .copy_from_slice(&scratch.v[src_start..src_start + head_dim]);
                }
            } else {
                scratch.attn_out[..q_dim].copy_from_slice(&scratch.v[..q_dim]);
            }
        } else {
            self.attention_with_cache_gqa_into(
                &scratch.q[..q_dim],
                k_cache,
                v_cache,
                &scratch.k[..k_dim],
                &scratch.v[..v_dim],
                &mut scratch.attn_out,
            );
        }

        cache.append(layer_idx, &scratch.k[..k_dim], &scratch.v[..v_dim]);

        // Attention output projection (Q8K or F32)
        // GH-305: attn_out has q_dim elements (may differ from hidden_dim), output is hidden_dim.
        let use_q8k_attn_out = use_q8k_path
            && q_dim.is_multiple_of(256)
            && layer.attn_output_weight.qtype == GGUF_TYPE_Q4_K;
        if use_q8k_attn_out {
            use crate::quantize::{
                fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
            };
            let q_sb = q_dim / 256;
            quantize_activations_q8k_into(
                &scratch.attn_out[..q_dim],
                &mut scratch.q8k_hidden_scales[..q_sb],
                &mut scratch.q8k_hidden_quants[..q_dim],
            )?;
            fused_q4k_q8k_parallel_matvec_into(
                &layer.attn_output_weight.data,
                &scratch.q8k_hidden_scales[..q_sb],
                &scratch.q8k_hidden_quants[..q_dim],
                layer.attn_output_weight.in_dim,
                layer.attn_output_weight.out_dim,
                &mut scratch.attn_proj,
            )?;
        } else {
            self.fused_matmul_into(
                &scratch.attn_out[..q_dim],
                &layer.attn_output_weight,
                &mut scratch.attn_proj,
            )?;
        }
        if let Some(ref bias) = layer.attn_output_bias {
            for i in 0..hidden_dim {
                scratch.attn_proj[i] += bias[i];
            }
        }

        // Residual
        for i in 0..hidden_dim {
            scratch.hidden[i] += scratch.attn_proj[i];
        }
        Ok(())
    }

    /// SwiGLU FFN path with Q8K acceleration for scratch-buffer forward pass
    ///
    /// Computes FFN up + gate projections with optional Q8K VNNI acceleration,
    /// applies SwiGLU activation, then projects down. Results written to scratch buffers.
    fn scratch_swiglu_ffn(
        &self,
        layer_idx: usize,
        scratch: &mut InferenceScratchBuffer,
        use_q8k_path: bool,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<()> {
        let layer = &self.layers[layer_idx];

        // Up + gate projections (Q8K or F32)
        self.scratch_q8k_up_gate(layer_idx, scratch, use_q8k_path, hidden_dim)?;

        // Apply biases
        if let Some(ref bias) = layer.ffn_up_bias {
            for i in 0..intermediate_dim {
                scratch.ffn_up[i] += bias[i];
            }
        }
        if let Some(ref bias) = layer.ffn_gate_bias {
            for i in 0..intermediate_dim {
                scratch.ffn_gate[i] += bias[i];
            }
        }

        // SiLU on gate, multiply with up
        ops::silu(&mut scratch.ffn_gate[..intermediate_dim]);
        for i in 0..intermediate_dim {
            scratch.ffn_gate[i] *= scratch.ffn_up[i];
        }

        // Down projection (Q8K or F32)
        self.scratch_q8k_down_projection(layer_idx, scratch, intermediate_dim, hidden_dim)
    }

    /// GELU FFN path with Q8K acceleration for scratch-buffer forward pass
    ///
    /// Computes FFN up projection with optional Q8K VNNI acceleration,
    /// applies GELU activation, then projects down. Results written to scratch buffers.
    fn scratch_gelu_ffn(
        &self,
        layer_idx: usize,
        scratch: &mut InferenceScratchBuffer,
        use_q8k_path: bool,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<()> {
        let layer = &self.layers[layer_idx];

        // PAR-129: Use Q8K-accelerated FFN for GELU models (Q4K only)
        let use_q8k_gelu_up = use_q8k_path && layer.ffn_up_weight.qtype == GGUF_TYPE_Q4_K;
        let use_q8k_gelu_down = intermediate_dim.is_multiple_of(256)
            && layer.ffn_down_weight.qtype == GGUF_TYPE_Q4_K;

        if use_q8k_gelu_up {
            // Reuse already-quantized hidden from QKV (scratch.q8k_hidden_*)
            use crate::quantize::fused_q4k_q8k_parallel_matvec_into;
            let hidden_sb = hidden_dim / 256;
            fused_q4k_q8k_parallel_matvec_into(
                &layer.ffn_up_weight.data,
                &scratch.q8k_hidden_scales[..hidden_sb],
                &scratch.q8k_hidden_quants[..hidden_dim],
                layer.ffn_up_weight.in_dim,
                layer.ffn_up_weight.out_dim,
                &mut scratch.ffn_up,
            )?;
        } else {
            self.fused_matmul_into(
                &scratch.normed[..hidden_dim],
                &layer.ffn_up_weight,
                &mut scratch.ffn_up,
            )?;
        }
        if let Some(ref bias) = layer.ffn_up_bias {
            for i in 0..intermediate_dim {
                scratch.ffn_up[i] += bias[i];
            }
        }
        ops::gelu(&mut scratch.ffn_up[..intermediate_dim]);

        if use_q8k_gelu_down {
            use crate::quantize::{
                fused_q4k_q8k_parallel_matvec_into, quantize_activations_q8k_into,
            };
            let inter_sb = intermediate_dim / 256;
            quantize_activations_q8k_into(
                &scratch.ffn_up[..intermediate_dim],
                &mut scratch.q8k_inter_scales[..inter_sb],
                &mut scratch.q8k_inter_quants[..intermediate_dim],
            )?;
            fused_q4k_q8k_parallel_matvec_into(
                &layer.ffn_down_weight.data,
                &scratch.q8k_inter_scales[..inter_sb],
                &scratch.q8k_inter_quants[..intermediate_dim],
                layer.ffn_down_weight.in_dim,
                layer.ffn_down_weight.out_dim,
                &mut scratch.ffn_down,
            )?;
        } else {
            self.fused_matmul_into(
                &scratch.ffn_up[..intermediate_dim],
                &layer.ffn_down_weight,
                &mut scratch.ffn_down,
            )?;
        }
        if let Some(ref bias) = layer.ffn_down_bias {
            for i in 0..hidden_dim {
                scratch.ffn_down[i] += bias[i];
            }
        }

        Ok(())
    }

    /// Zero-allocation forward pass using scratch buffers (IMP-131)
    ///
    /// All intermediate results are written to pre-allocated scratch buffers.
    /// Output logits are stored in `scratch.logits`.
    pub(crate) fn forward_single_with_scratch(
        &self,
        token_id: u32,
        cache: &mut OwnedQuantizedKVCache,
        position: usize,
        scratch: &mut InferenceScratchBuffer,
    ) -> Result<()> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // GH-278: Use contract-derived norm type.
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

        // Q8K requires hidden_dim to be multiple of 256; smaller models fall back to f32
        let use_q8k_path = hidden_dim.is_multiple_of(256);

        // 1. Token embedding lookup -> scratch.hidden
        self.embed_into(token_id, &mut scratch.hidden);

        // GH-278: Add learned position embedding for absolute encoding (GPT-2, BERT, whisper)
        if self.config.constraints.uses_absolute_positions() {
            if let Some(ref pos_emb) = self.position_embedding {
                let start = position * hidden_dim;
                let end = start + hidden_dim;
                if end <= pos_emb.len() {
                    for i in 0..hidden_dim {
                        scratch.hidden[i] += pos_emb[start + i];
                    }
                }
            }
        }

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // 2a. Attention layer norm -> scratch.normed
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

            // 2b-2e. Attention block (QKV, attention, output projection, residual)
            self.scratch_attention_block(
                layer_idx, layer, scratch, cache, position, use_q8k_path, hidden_dim,
            )?;

            // 2f. Pre-FFN layer norm -> scratch.normed
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

            // 2g. FFN (contract-driven activation selection, GH-278)
            if self.config.constraints.has_gate_ffn() {
                self.scratch_swiglu_ffn(layer_idx, scratch, use_q8k_path, hidden_dim, intermediate_dim)?;
            } else {
                self.scratch_gelu_ffn(layer_idx, scratch, use_q8k_path, hidden_dim, intermediate_dim)?;
            }

            // 2h. FFN residual
            for i in 0..hidden_dim {
                scratch.hidden[i] += scratch.ffn_down[i];
            }
        }

        // 3. Final layer norm -> scratch.normed
        if use_rmsnorm {
            ops::rms_norm_into(
                &scratch.hidden,
                &self.output_norm_weight,
                self.config.eps,
                &mut scratch.normed,
            );
        } else {
            ops::layer_norm_into(
                &scratch.hidden,
                &self.output_norm_weight,
                self.output_norm_bias.as_deref(),
                self.config.eps,
                &mut scratch.normed,
            );
        }

        // 4. LM head -> scratch.logits (Q8K integer path for Q4K weights)
        let use_q8k_lm = hidden_dim.is_multiple_of(256)
            && self.lm_head_weight.qtype == GGUF_TYPE_Q4_K;
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
                &self.lm_head_weight.data,
                &scratch.q8k_hidden_scales[..hidden_sb],
                &scratch.q8k_hidden_quants[..hidden_dim],
                self.lm_head_weight.in_dim,
                self.lm_head_weight.out_dim,
                &mut scratch.logits,
            )?;
        } else {
            self.fused_matmul_into(
                &scratch.normed[..hidden_dim],
                &self.lm_head_weight,
                &mut scratch.logits,
            )?;
        }

        Ok(())
    }
}

include!("ffn_block.rs");
include!("adaptive_ffn.rs");
