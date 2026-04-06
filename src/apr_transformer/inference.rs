impl AprTransformer {

    /// Forward pass with layer-by-layer activation tracing.
    ///
    /// This is identical to `forward()` but collects statistics at each layer
    /// for debugging inference divergence issues.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// `ForwardTrace` containing logits and per-layer activation statistics
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn forward_traced(&self, token_ids: &[u32]) -> Result<ForwardTrace> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;

        // 1. Token embedding lookup
        let mut hidden = self.embed(token_ids);
        let embed_stats = ActivationStats::from_slice(&hidden);

        let mut layer_activations = Vec::with_capacity(self.layers.len());

        // 2. Process through transformer layers with tracing
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Note: Q4K layers not used in traced forward (uses F32 for accuracy)
            let _q4k_layer = self.q4k_layers.as_ref().and_then(|l| l.get(layer_idx));

            // 2a. Attention layer norm
            let normed = self.layer_norm(
                &hidden,
                &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(),
                self.config.eps,
            );
            let attn_norm_stats = ActivationStats::from_slice(&normed);

            // 2b. QKV projection
            let qkv_dim = layer.qkv_weight.len() / hidden_dim;
            let mut qkv = self.matmul(&normed, &layer.qkv_weight, hidden_dim, qkv_dim);
            if let Some(ref bias) = layer.qkv_bias {
                self.add_bias(&mut qkv, bias);
            }
            let qkv_stats = ActivationStats::from_slice(&qkv);

            // 2c. Attention computation (simplified for trace - same logic as forward)
            let seq_len = token_ids.len();
            let head_dim = hidden_dim / self.config.num_heads;
            let num_kv_heads = self.config.num_kv_heads;
            let kv_dim = num_kv_heads * head_dim;
            let group_size = self.config.num_heads / num_kv_heads;
            let scale = 1.0 / (head_dim as f32).sqrt();

            let mut q_all = Vec::with_capacity(seq_len * hidden_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;
                let mut q_pos = qkv[qkv_start..qkv_start + hidden_dim].to_vec();
                let mut k_pos =
                    qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim].to_vec();
                let v_pos =
                    &qkv[qkv_start + hidden_dim + kv_dim..qkv_start + hidden_dim + 2 * kv_dim];

                self.apply_rope_f32(&mut q_pos, s, self.config.num_heads, head_dim);
                self.apply_rope_f32(&mut k_pos, s, num_kv_heads, head_dim);

                q_all.extend_from_slice(&q_pos);
                k_all.extend_from_slice(&k_pos);
                v_all.extend_from_slice(v_pos);
            }

            // Attention output
            let mut attn_out = vec![0.0f32; seq_len * hidden_dim];
            for head in 0..self.config.num_heads {
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
                    let exp_scores: Vec<f32> =
                        scores.iter().map(|s| (s - max_score).exp()).collect();
                    let sum_exp: f32 = exp_scores.iter().sum();
                    let probs: Vec<f32> = exp_scores.iter().map(|e| e / sum_exp).collect();

                    // Weighted sum of values
                    let out_start = i * hidden_dim + q_head_offset;
                    for (j, &p) in probs.iter().enumerate() {
                        let v_start = j * kv_dim + kv_head_offset;
                        for d in 0..head_dim {
                            attn_out[out_start + d] += p * v_all[v_start + d];
                        }
                    }
                }
            }

            // Output projection
            let mut attn_output =
                self.matmul(&attn_out, &layer.attn_output_weight, hidden_dim, hidden_dim);
            if let Some(ref bias) = layer.attn_output_bias {
                self.add_bias(&mut attn_output, bias);
            }
            let attn_out_stats = ActivationStats::from_slice(&attn_output);

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += attn_output[i];
            }

            // 2f. FFN layer norm (if present)
            let ffn_input = if let Some(ref norm_weight) = layer.ffn_norm_weight {
                let normed = self.layer_norm(
                    &hidden,
                    norm_weight,
                    layer.ffn_norm_bias.as_deref(),
                    self.config.eps,
                );
                normed
            } else {
                hidden.clone()
            };
            let ffn_norm_stats = ActivationStats::from_slice(&ffn_input);

            // 2g. FFN - check if gated MLP (SwiGLU) by presence of gate weight
            let ffn_output = if let Some(ref gate_weight) = layer.ffn_gate_weight {
                let gate = self.matmul(&ffn_input, gate_weight, hidden_dim, intermediate_dim);
                let up = self.matmul(
                    &ffn_input,
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );

                let mut ffn_hidden = Vec::with_capacity(gate.len());
                for (g, u) in gate.iter().zip(up.iter()) {
                    let silu_g = g / (1.0 + (-g).exp());
                    ffn_hidden.push(silu_g * u);
                }

                let mut out = self.matmul(
                    &ffn_hidden,
                    &layer.ffn_down_weight,
                    intermediate_dim,
                    hidden_dim,
                );
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            } else {
                // Standard MLP without gating
                let mut ffn_hidden = self.matmul(
                    &ffn_input,
                    &layer.ffn_up_weight,
                    hidden_dim,
                    intermediate_dim,
                );
                if let Some(ref bias) = layer.ffn_up_bias {
                    self.add_bias(&mut ffn_hidden, bias);
                }
                for h in &mut ffn_hidden {
                    let gelu_approx =
                        0.5 * *h * (1.0 + (0.797_884_6 * (*h + 0.044_715 * *h * *h * *h)).tanh());
                    *h = gelu_approx;
                }
                let mut out = self.matmul(
                    &ffn_hidden,
                    &layer.ffn_down_weight,
                    intermediate_dim,
                    hidden_dim,
                );
                if let Some(ref bias) = layer.ffn_down_bias {
                    self.add_bias(&mut out, bias);
                }
                out
            };
            let ffn_out_stats = ActivationStats::from_slice(&ffn_output);

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_output[i];
            }
            let output_stats = ActivationStats::from_slice(&hidden);

            layer_activations.push(LayerActivation {
                layer_idx,
                attn_norm_stats,
                qkv_stats,
                attn_out_stats,
                ffn_norm_stats,
                ffn_out_stats,
                output_stats,
            });
        }

        // 3. Final layer norm
        let normed = self.layer_norm(
            &hidden,
            &self.output_norm_weight,
            self.output_norm_bias.as_deref(),
            self.config.eps,
        );
        let final_norm_stats = ActivationStats::from_slice(&normed);

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

    /// Predict next token (greedy decoding)
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Token ID with highest probability
    ///
    /// # Errors
    ///
    /// Returns error if inference fails
    pub fn predict_next(&self, token_ids: &[u32]) -> Result<u32> {
        let logits = self.forward(token_ids)?;

        // Argmax
        let (max_idx, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Empty logits".to_string(),
            })?;

        Ok(max_idx as u32)
    }
}
