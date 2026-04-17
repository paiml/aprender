impl AprTransformer {

    /// Sequential Q4K matmul across sequence positions (PMAT-260)
    fn seq_matmul_q4k(
        q4k_bytes: &[u8],
        input: &[f32],
        seq_len: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(seq_len * out_dim);
        for s in 0..seq_len {
            let input_slice = &input[s * in_dim..(s + 1) * in_dim];
            let pos_out = matmul_q4k_rowmajor(q4k_bytes, input_slice, out_dim, in_dim)?;
            output.extend(pos_out);
        }
        Ok(output)
    }

    /// Sequential Q6K matmul across sequence positions (PMAT-260)
    fn seq_matmul_q6k(
        q6k_bytes: &[u8],
        input: &[f32],
        seq_len: usize,
        out_dim: usize,
        in_dim: usize,
    ) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(seq_len * out_dim);
        for s in 0..seq_len {
            let input_slice = &input[s * in_dim..(s + 1) * in_dim];
            let pos_out = matmul_q6k_rowmajor(q6k_bytes, input_slice, out_dim, in_dim)?;
            output.extend(pos_out);
        }
        Ok(output)
    }

    /// Compute causal GQA scaled dot-product attention (PMAT-260)
    ///
    /// Implements the 4-level nested loop: heads x positions x past x head_dim
    /// with causal masking and softmax normalization.
    fn compute_causal_gqa_attention(
        q_all: &[f32],
        k_all: &[f32],
        v_all: &[f32],
        seq_len: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        hidden_dim: usize,
        scale: f32,
    ) -> Vec<f32> {
        let group_size = num_heads / num_kv_heads;
        let kv_dim = num_kv_heads * head_dim;
        let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
        for head in 0..num_heads {
            let kv_head = head / group_size;
            let q_head_offset = head * head_dim;
            let kv_head_offset = kv_head * head_dim;

            for i in 0..seq_len {
                let mut scores = Vec::with_capacity(i + 1);
                let q_start = i * hidden_dim + q_head_offset;

                for j in 0..=i {
                    let k_start = j * kv_dim + kv_head_offset;
                    let mut score = 0.0f32;
                    for d in 0..head_dim {
                        score += q_all[q_start + d] * k_all[k_start + d];
                    }
                    scores.push(score * scale);
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }
                if exp_sum > 0.0 {
                    for s in &mut scores {
                        *s /= exp_sum;
                    }
                }

                // Weighted sum of V
                let out_start = i * hidden_dim + q_head_offset;
                for (j, &weight) in scores.iter().enumerate() {
                    let v_start = j * kv_dim + kv_head_offset;
                    for d in 0..head_dim {
                        attn_out[out_start + d] += weight * v_all[v_start + d];
                    }
                }
            }
        }
        attn_out
    }

    /// Q4K or F32 attention output projection (PMAT-260)
    fn apr_attn_output_projection(
        &self,
        attn_out: &[f32],
        q4k_layer: Option<&Q4KLayerWeights>,
        layer: &AprTransformerLayer,
        seq_len: usize,
        hidden_dim: usize,
        layer_idx: usize,
        trace_enabled: bool,
    ) -> Result<Vec<f32>> {
        if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.attn_output_weight.as_ref()) {
            if trace_enabled && layer_idx == 0 {
                eprintln!("[TRACE] Layer {layer_idx}: attn_output using Q4K fused kernel");
            }
            Self::seq_matmul_q4k(q4k_bytes, attn_out, seq_len, hidden_dim, hidden_dim)
        } else {
            if trace_enabled && layer_idx == 0 {
                eprintln!("[TRACE] Layer {layer_idx}: attn_output using F32 fallback (slow!)");
            }
            Ok(self.matmul(attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim))
        }
    }

    /// SwiGLU FFN computation for APR transformer layers (PMAT-260)
    ///
    /// Computes: down(SiLU(gate(x)) * up(x)) with Q4K/Q6K fused kernel support.
    /// Gate and up projections are computed in parallel via rayon::join.
    ///
    /// # Arguments
    ///
    /// * `ffn_input` - Normalized hidden state input to FFN
    /// * `layer_idx` - Layer index for accessing Q4K weights and trace logging
    /// * `q4k_layer` - Optional Q4K weights for fused kernel path
    /// * `seq_len` - Sequence length (number of tokens)
    /// * `hidden_dim` - Hidden dimension size
    /// * `intermediate_dim` - Intermediate FFN dimension size
    /// * `trace_enabled` - Whether REALIZE_TRACE is set
    #[allow(clippy::too_many_arguments)]
    fn apr_swiglu_ffn(
        &self,
        ffn_input: &[f32],
        layer_idx: usize,
        q4k_layer: Option<&Q4KLayerWeights>,
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
        trace_enabled: bool,
    ) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_idx];
        let gate_weight = layer
            .ffn_gate_weight
            .as_ref()
            .expect("apr_swiglu_ffn called without gate weight");

        // GH-192/199: Compute gate and up in parallel (like GGUF path)
        let q4k_gate = q4k_layer.and_then(|q| q.ffn_gate_weight.as_ref());
        let q4k_up = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref());
        let (gate_result, up_result) = rayon::join(
            || -> Result<Vec<f32>> {
                if let Some(q4k_bytes) = q4k_gate {
                    Self::seq_matmul_q4k(q4k_bytes, ffn_input, seq_len, intermediate_dim, hidden_dim)
                } else {
                    Ok(self.matmul(ffn_input, gate_weight, hidden_dim, intermediate_dim))
                }
            },
            || -> Result<Vec<f32>> {
                if let Some(q4k_bytes) = q4k_up {
                    Self::seq_matmul_q4k(q4k_bytes, ffn_input, seq_len, intermediate_dim, hidden_dim)
                } else {
                    Ok(self.matmul(ffn_input, &layer.ffn_up_weight, hidden_dim, intermediate_dim))
                }
            },
        );
        let gate = gate_result?;
        let up = up_result?;

        if trace_enabled && layer_idx == 0 {
            let kernel = if q4k_gate.is_some() { "Q4K" } else { "F32" };
            eprintln!("[TRACE] Layer 0: ffn_gate/up using {kernel} kernel");
        }

        // SiLU(gate) * up, then down projection
        let mut ffn_hidden = Vec::with_capacity(gate.len());
        for (g, u) in gate.iter().zip(up.iter()) {
            let silu_g = g / (1.0 + (-g).exp());
            ffn_hidden.push(silu_g * u);
        }

        // Down projection with Q4K/Q6K/F32 dispatch
        let mut out = if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight.as_ref()) {
            Self::seq_matmul_q4k(q4k_bytes, &ffn_hidden, seq_len, hidden_dim, intermediate_dim)?
        } else if let Some(q6k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight_q6k.as_ref()) {
            Self::seq_matmul_q6k(q6k_bytes, &ffn_hidden, seq_len, hidden_dim, intermediate_dim)?
        } else {
            self.matmul(&ffn_hidden, &layer.ffn_down_weight, intermediate_dim, hidden_dim)
        };
        if let Some(ref bias) = layer.ffn_down_bias {
            self.add_bias(&mut out, bias);
        }
        Ok(out)
    }

    /// Standard GELU MLP FFN computation for APR transformer layers (PMAT-260)
    ///
    /// Computes: down(GELU(up(x))) with Q4K fused kernel support.
    ///
    /// # Arguments
    ///
    /// * `ffn_input` - Normalized hidden state input to FFN
    /// * `layer_idx` - Layer index for accessing Q4K weights
    /// * `q4k_layer` - Optional Q4K weights for fused kernel path
    /// * `seq_len` - Sequence length (number of tokens)
    /// * `hidden_dim` - Hidden dimension size
    /// * `intermediate_dim` - Intermediate FFN dimension size
    fn apr_gelu_ffn(
        &self,
        ffn_input: &[f32],
        layer_idx: usize,
        q4k_layer: Option<&Q4KLayerWeights>,
        seq_len: usize,
        hidden_dim: usize,
        intermediate_dim: usize,
    ) -> Result<Vec<f32>> {
        let layer = &self.layers[layer_idx];

        // Standard MLP: down(GELU(up(x)))
        // PMAT-103: Check for Q4K up weight
        let mut ffn_hidden =
            if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_up_weight.as_ref()) {
                let mut output = Vec::with_capacity(seq_len * intermediate_dim);
                for s in 0..seq_len {
                    let input_slice = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                    // PMAT-103 FIX: Q4K kernel expects (ne0=output_dim, ne1=input_dim)
                    // ffn_up: [intermediate_dim, hidden_dim] maps hidden[1536] -> intermediate[8960]
                    let pos_out = matmul_q4k_rowmajor(
                        q4k_bytes,
                        input_slice,
                        intermediate_dim,
                        hidden_dim,
                    )?;
                    output.extend(pos_out);
                }
                output
            } else {
                self.matmul(
                    ffn_input,
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                )
            };
        if let Some(ref bias) = layer.ffn_up_bias {
            self.add_bias(&mut ffn_hidden, bias);
        }
        self.gelu(&mut ffn_hidden);

        // PMAT-103: Check for Q4K down weight
        let mut out =
            if let Some(q4k_bytes) = q4k_layer.and_then(|q| q.ffn_down_weight.as_ref()) {
                let mut output = Vec::with_capacity(seq_len * hidden_dim);
                for s in 0..seq_len {
                    let input_slice =
                        &ffn_hidden[s * intermediate_dim..(s + 1) * intermediate_dim];
                    let pos_out = matmul_q4k_rowmajor(
                        q4k_bytes,
                        input_slice,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    output.extend(pos_out);
                }
                output
            } else {
                self.matmul(
                    &ffn_hidden,
                    &layer.ffn_down_weight,
                    intermediate_dim,
                    hidden_dim,
                )
            };
        if let Some(ref bias) = layer.ffn_down_bias {
            self.add_bias(&mut out, bias);
        }
        Ok(out)
    }

    /// Forward pass through the transformer
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary for next token prediction
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        // NOISY-GUARD: Only print trace messages when REALIZE_TRACE is set
        let trace_enabled = std::env::var("REALIZE_TRACE").is_ok();

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);

        // 2. Process through transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // PMAT-103: Get Q4K weights for this layer (if available)
            let q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );

            // 2b. QKV projection
            // Calculate qkv_dim from actual weight size (handles GQA models)
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }

            // 2c. Proper attention with GQA support and RoPE
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let kv_dim = num_kv_heads * head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // Split QKV and apply RoPE
            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V (layout: [Q..., K..., V...])
                let mut q_pos = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k_pos =
                    qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v_pos =
                    &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                // GH-279: Per-head QK RMSNorm (Qwen3) — after bias, before RoPE
                if let Some(ref q_norm) = layer.attn_q_norm_weight {
                    crate::gguf::ops::apply_per_head_rms_norm(
                        &mut q_pos,
                        q_norm,
                        self.config.num_heads,
                        self.config.eps,
                    );
                }
                if let Some(ref k_norm) = layer.attn_k_norm_weight {
                    crate::gguf::ops::apply_per_head_rms_norm(
                        &mut k_pos,
                        k_norm,
                        num_kv_heads,
                        self.config.eps,
                    );
                }

                // Apply RoPE to Q and K
                self.apply_rope_f32(&mut q_pos, s, self.config.num_heads, head_dim);
                self.apply_rope_f32(&mut k_pos, s, num_kv_heads, head_dim);

                q_all.extend_from_slice(&q_pos);
                k_all.extend_from_slice(&k_pos);
                v_all.extend_from_slice(v_pos);
            }

            // Compute scaled dot-product attention with causal mask
            let attn_out = Self::compute_causal_gqa_attention(
                &q_all, &k_all, &v_all, seq_len, self.config.num_heads, num_kv_heads, head_dim, hidden_dim, scale,
            );

            // 2d. Attention output projection
            let mut attn_output = self.apr_attn_output_projection(
                &attn_out, q4k_layer, layer, seq_len, hidden_dim, layer_idx, trace_enabled,
            )?;
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }

            // 2e. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. Apply FFN norm if present (post_attention_layernorm)
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                self.layer_norm(
                    &hidden,
                    ffn_norm,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                )
            } else {
                hidden.clone()
            };

            // 2g. FFN projection (SwiGLU or standard GELU)
            // PMAT-103: Use Q4K fused kernel when available for FFN
            let seq_len = token_ids.len();
            let ffn_output = if layer.ffn_gate_weight.is_some() {
                self.apr_swiglu_ffn(
                    &ffn_input,
                    layer_idx,
                    q4k_layer,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                    trace_enabled,
                )?
            } else {
                self.apr_gelu_ffn(
                    &ffn_input,
                    layer_idx,
                    q4k_layer,
                    seq_len,
                    hidden_dim,
                    intermediate_dim,
                )?
            };

            // 2h. Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );

        // 4. LM head projection (only last token)
        let seq_len = token_ids.len();
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &normed[last_hidden_start..last_hidden_start + hidden_dim];

        let mut logits = self.matmul(
            last_hidden,
            &self.lm_head_weight,
            hidden_dim,
            self.config.vocab_size,
        );
        if let Some(ref bias) = self.lm_head_bias {
            self.add_bias(&mut logits, bias);
        }

        Ok(logits)
    }
}
