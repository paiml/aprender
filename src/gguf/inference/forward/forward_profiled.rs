impl OwnedQuantizedModel {

    /// Forward pass with per-operation BrickProfiler instrumentation
    ///
    /// Identical to `forward()` but wraps each operation with
    /// `profiler.start()`/`profiler.stop()` using BrickId-aligned names.
    /// Zero overhead on production path (use `forward()` instead).
    ///
    /// Returns (logits, ProfileReport) with real per-operation timing.
    pub fn forward_profiled(
        &self,
        token_ids: &[u32],
        profiler: &mut BrickProfiler,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let _ = self.config.intermediate_dim;

        // 1. Token embedding
        profiler.start("Embedding");
        let mut hidden = self.embed(token_ids);
        // GH-278: Add learned position embeddings for absolute encoding (GPT-2, BERT, whisper)
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
        profiler.stop("Embedding");

        let use_rmsnorm = self.config.constraints.uses_rmsnorm();

        // 2. Transformer layers with per-operation instrumentation
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            profiler.set_current_layer(layer_idx);

            // 2a. Attention norm
            profiler.start("RmsNorm");
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
            profiler.stop("RmsNorm");

            // 2b. QKV projection
            profiler.start("QkvProjection");
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
            profiler.stop("QkvProjection");

            // 2c. RoPE
            profiler.start("RopeEmbedding");
            let seq_len = token_ids.len();
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * k_dim);
            let mut v_all = Vec::with_capacity(seq_len * v_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q = qkv[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv[qkv_start + q_dim..qkv_start + q_dim + k_dim].to_vec();
                let v = &qkv[qkv_start + q_dim + k_dim..qkv_start + q_dim + k_dim + v_dim];

                // GH-278: Skip RoPE for models with learned position embeddings (GPT-2)
                if self.config.constraints.uses_rope() {
                    self.apply_rope(&mut q, s, self.config.num_heads);
                    self.apply_rope(&mut k, s, self.config.num_kv_heads);
                }

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }
            profiler.stop("RopeEmbedding");

            // 2d. Attention score + softmax + output
            profiler.start("AttentionScore");
            let attn_out = self.causal_attention(&q_all, &k_all, &v_all, seq_len);
            profiler.stop("AttentionScore");

            // 2e. Output projection
            profiler.start("OutputProjection");
            let mut attn_output = self.fused_matmul(&attn_out, &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias {
                ops::add_bias(&mut attn_output, bias);
            }
            profiler.stop("OutputProjection");

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN norm
            profiler.start("RmsNorm");
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
            profiler.stop("RmsNorm");

            // 2g. FFN (contract-driven activation selection, GH-278)
            let ffn_activated = if self.config.constraints.has_gate_ffn() {
                // SwiGLU path
                if let Some(ref gate_weight) = layer.ffn_gate_weight {
                    // Separate gate/up weights (LLaMA, Qwen2, Mistral)
                    profiler.start("UpProjection");
                    let mut ffn_up = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    if let Some(ref bias) = layer.ffn_up_bias {
                        ops::add_bias(&mut ffn_up, bias);
                    }
                    profiler.stop("UpProjection");

                    profiler.start("GateProjection");
                    let mut ffn_gate = self.fused_matmul(&ffn_input, gate_weight)?;
                    if let Some(ref bias) = layer.ffn_gate_bias {
                        ops::add_bias(&mut ffn_gate, bias);
                    }
                    profiler.stop("GateProjection");

                    profiler.start("Activation");
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    profiler.stop("Activation");

                    ffn_gate
                } else {
                    // GH-306: Fused gate_up weight (Phi-3.5) — single matmul, split in half
                    profiler.start("FusedGateUpProjection");
                    let fused = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                    let half = fused.len() / 2;
                    let mut ffn_gate = fused[..half].to_vec();
                    let mut ffn_up = fused[half..].to_vec();
                    if let Some(ref bias) = layer.ffn_up_bias {
                        let bias_half = bias.len() / 2;
                        ops::add_bias(&mut ffn_gate, &bias[..bias_half]);
                        ops::add_bias(&mut ffn_up, &bias[bias_half..]);
                    }
                    profiler.stop("FusedGateUpProjection");

                    profiler.start("Activation");
                    ops::silu(&mut ffn_gate);
                    for i in 0..ffn_gate.len() {
                        ffn_gate[i] *= ffn_up[i];
                    }
                    profiler.stop("Activation");

                    ffn_gate
                }
            } else {
                // GELU path
                profiler.start("UpProjection");
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                profiler.stop("UpProjection");

                profiler.start("Activation");
                ops::gelu(&mut ffn_hidden);
                profiler.stop("Activation");

                ffn_hidden
            };

            // 2h. Down projection
            profiler.start("DownProjection");
            let mut ffn_output = self.fused_matmul(&ffn_activated, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias {
                ops::add_bias(&mut ffn_output, bias);
            }
            profiler.stop("DownProjection");

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        profiler.start("RmsNorm");
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
        profiler.stop("RmsNorm");

        // 4. LM head projection (only last token)
        profiler.start("LmHead");
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];
        let mut logits = self.fused_matmul(last_hidden, &self.lm_head_weight)?;
        if let Some(ref bias) = self.lm_head_bias {
            ops::add_bias(&mut logits, bias);
        }
        profiler.stop("LmHead");

        Ok(logits)
    }
}
