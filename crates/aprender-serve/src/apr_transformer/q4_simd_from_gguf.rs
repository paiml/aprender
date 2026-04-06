impl QuantizedAprTransformerQ4 {
    /// Create from GGUF OwnedQuantizedModel (extracts Q4_0 bytes)
    ///
    /// # Arguments
    ///
    /// * `gguf` - Source GGUF model with Q4_0 weights
    ///
    /// # Returns
    ///
    /// Quantized APR transformer with same weights
    pub fn from_gguf(gguf: &crate::gguf::OwnedQuantizedModel) -> Self {
        use crate::gguf::OwnedQKVWeights;

        let config = AprTransformerConfig {
            architecture: gguf.config.architecture.clone(),
            hidden_dim: gguf.config.hidden_dim,
            num_layers: gguf.config.num_layers,
            num_heads: gguf.config.num_heads,
            num_kv_heads: gguf.config.num_kv_heads,
            vocab_size: gguf.config.vocab_size,
            intermediate_dim: gguf.config.intermediate_dim,
            context_length: gguf.config.context_length,
            rope_theta: gguf.config.rope_theta,
            eps: gguf.config.eps,
            eos_token_id: gguf.config.eos_token_id,
        ..Default::default()
        };

        let layers =
            gguf.layers
                .iter()
                .map(|l| {
                    // Extract QKV weight data
                    let qkv_weight = match &l.qkv_weight {
                        OwnedQKVWeights::Fused(t) => {
                            QuantizedAprTensorQ4::new(t.data.clone(), t.in_dim, t.out_dim)
                        },
                        OwnedQKVWeights::Separate { q, k, v } => {
                            // Concatenate Q, K, V for fused format
                            let mut data =
                                Vec::with_capacity(q.data.len() + k.data.len() + v.data.len());
                            data.extend_from_slice(&q.data);
                            data.extend_from_slice(&k.data);
                            data.extend_from_slice(&v.data);
                            QuantizedAprTensorQ4::new(
                                data,
                                q.in_dim,                          // hidden_dim
                                q.out_dim + k.out_dim + v.out_dim, // qkv_dim
                            )
                        },
                    };

                    QuantizedAprLayerQ4 {
                        attn_norm_weight: l.attn_norm_weight.clone(),
                        qkv_weight,
                        attn_output_weight: QuantizedAprTensorQ4::new(
                            l.attn_output_weight.data.clone(),
                            l.attn_output_weight.in_dim,
                            l.attn_output_weight.out_dim,
                        ),
                        ffn_up_weight: QuantizedAprTensorQ4::new(
                            l.ffn_up_weight.data.clone(),
                            l.ffn_up_weight.in_dim,
                            l.ffn_up_weight.out_dim,
                        ),
                        ffn_down_weight: QuantizedAprTensorQ4::new(
                            l.ffn_down_weight.data.clone(),
                            l.ffn_down_weight.in_dim,
                            l.ffn_down_weight.out_dim,
                        ),
                        ffn_gate_weight: l.ffn_gate_weight.as_ref().map(|g| {
                            QuantizedAprTensorQ4::new(g.data.clone(), g.in_dim, g.out_dim)
                        }),
                        ffn_norm_weight: l.ffn_norm_weight.clone(),
                    }
                })
                .collect();

        let lm_head_weight = QuantizedAprTensorQ4::new(
            gguf.lm_head_weight.data.clone(),
            gguf.lm_head_weight.in_dim,
            gguf.lm_head_weight.out_dim,
        );

        Self {
            config,
            token_embedding: gguf.token_embedding.clone(),
            layers,
            output_norm_weight: gguf.output_norm_weight.clone(),
            lm_head_weight,
        }
    }

    /// Get model configuration
    #[must_use]
    pub fn config(&self) -> &AprTransformerConfig {
        &self.config
    }

    /// Create a scratch buffer for zero-allocation inference
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let model = QuantizedAprTransformerQ4::from_gguf(&gguf);
    /// let mut scratch = model.create_scratch();
    ///
    /// // Reuse scratch across multiple forward passes
    /// for token_id in token_ids {
    ///     let logits = model.forward_single_with_scratch(token_id, &mut scratch)?;
    /// }
    /// ```
    #[must_use]
    pub fn create_scratch(&self) -> AprInferenceScratch {
        AprInferenceScratch::from_config(&self.config)
    }

    /// Forward pass using SIMD-accelerated Q4_0×Q8_0 matmul
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits over vocabulary
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;

        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // 1. Token embedding lookup (F32)
        let seq_len = token_ids.len();
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            let offset = (token_id as usize) * hidden_dim;
            if offset + hidden_dim <= self.token_embedding.len() {
                hidden.extend_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
            } else {
                hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
            }
        }

        // 2. Process through transformer layers
        for layer in &self.layers {
            // Pre-attention RMS norm
            let mut normed = Vec::with_capacity(hidden.len());
            for s in 0..seq_len {
                let start = s * hidden_dim;
                let slice = &hidden[start..start + hidden_dim];
                let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                for (i, &x) in slice.iter().enumerate() {
                    normed.push(x / rms * layer.attn_norm_weight[i]);
                }
            }

            // QKV projection using SIMD matmul
            let qkv_dim = layer.qkv_weight.out_dim;
            let mut qkv_out = Vec::with_capacity(seq_len * qkv_dim);
            for s in 0..seq_len {
                let input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
                let qkv = fused_q4_0_q8_0_parallel_matvec(
                    &layer.qkv_weight.data,
                    input,
                    hidden_dim,
                    qkv_dim,
                )?;
                qkv_out.extend(qkv);
            }

            // Proper attention with RoPE and causal mask
            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;

            // Extract Q, K, V and apply RoPE to Q and K
            let mut q_all = Vec::with_capacity(seq_len * q_dim);
            let mut k_all = Vec::with_capacity(seq_len * kv_dim);
            let mut v_all = Vec::with_capacity(seq_len * kv_dim);

            for s in 0..seq_len {
                let qkv_start = s * qkv_dim;

                // Extract Q, K, V for this position (QKV layout: [Q..., K..., V...])
                let mut q = qkv_out[qkv_start..qkv_start + q_dim].to_vec();
                let mut k = qkv_out[qkv_start + q_dim..qkv_start + q_dim + kv_dim].to_vec();
                let v = &qkv_out[qkv_start + q_dim + kv_dim..qkv_start + q_dim + 2 * kv_dim];

                // Apply RoPE to Q and K (position-dependent rotation)
                self.apply_rope(&mut q, s, num_heads);
                self.apply_rope(&mut k, s, num_kv_heads);

                q_all.extend_from_slice(&q);
                k_all.extend_from_slice(&k);
                v_all.extend_from_slice(v);
            }

            // Compute scaled dot-product attention with causal mask
            let attn_output = self.causal_attention(&q_all, &k_all, &v_all, seq_len);

            // Output projection using SIMD matmul
            let mut proj_out = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let input = &attn_output[s * hidden_dim..(s + 1) * hidden_dim];
                let proj = fused_q4_0_q8_0_parallel_matvec(
                    &layer.attn_output_weight.data,
                    input,
                    layer.attn_output_weight.in_dim,
                    layer.attn_output_weight.out_dim,
                )?;
                proj_out.extend(proj);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += proj_out[i];
            }

            // Pre-FFN norm (if present)
            let ffn_input = if let Some(ffn_norm) = &layer.ffn_norm_weight {
                let mut normed_ffn = Vec::with_capacity(hidden.len());
                for s in 0..seq_len {
                    let start = s * hidden_dim;
                    let slice = &hidden[start..start + hidden_dim];
                    let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
                    let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
                    for (i, &x) in slice.iter().enumerate() {
                        normed_ffn.push(x / rms * ffn_norm[i]);
                    }
                }
                normed_ffn
            } else {
                normed.clone()
            };

            // FFN with SwiGLU (sequential to avoid nested parallelism overhead)
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            let ffn_up = if let Some(gate) = &layer.ffn_gate_weight {
                // SwiGLU: Sequential up + gate (both matmuls use internal parallelism)
                let mut ffn_up_out = Vec::with_capacity(seq_len * intermediate_dim);
                let mut ffn_gate_out = Vec::with_capacity(seq_len * intermediate_dim);

                for s in 0..seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];

                    // Up projection
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    ffn_up_out.extend(u);

                    // Gate projection
                    let g = fused_q4_0_q8_0_parallel_matvec(
                        &gate.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    ffn_gate_out.extend(g);
                }

                // Apply SiLU to gate and multiply with up
                for i in 0..ffn_up_out.len() {
                    let silu = ffn_gate_out[i] / (1.0 + (-ffn_gate_out[i]).exp());
                    ffn_up_out[i] *= silu;
                }
                ffn_up_out
            } else {
                // Non-SwiGLU: Sequential up projection + GELU
                let mut up = Vec::with_capacity(seq_len * intermediate_dim);
                for s in 0..seq_len {
                    let input = &ffn_input[s * hidden_dim..(s + 1) * hidden_dim];
                    let u = fused_q4_0_q8_0_parallel_matvec(
                        &layer.ffn_up_weight.data,
                        input,
                        hidden_dim,
                        intermediate_dim,
                    )?;
                    up.extend(u);
                }
                // GELU activation (tanh approximation)
                const SQRT_2_OVER_PI: f32 = 0.797_884_6;
                const GELU_COEFF: f32 = 0.044_715;
                for x in &mut up {
                    let t = (SQRT_2_OVER_PI * (*x + GELU_COEFF * *x * *x * *x)).tanh();
                    *x = 0.5 * *x * (1.0 + t);
                }
                up
            };

            // FFN: down projection
            let mut ffn_down = Vec::with_capacity(seq_len * hidden_dim);
            for s in 0..seq_len {
                let input = &ffn_up[s * intermediate_dim..(s + 1) * intermediate_dim];
                let down = fused_q4_0_q8_0_parallel_matvec(
                    &layer.ffn_down_weight.data,
                    input,
                    intermediate_dim,
                    hidden_dim,
                )?;
                ffn_down.extend(down);
            }

            // Residual connection
            for i in 0..hidden.len() {
                hidden[i] += ffn_down[i];
            }
        }

        // 3. Final RMS norm
        let last_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_start..last_start + hidden_dim];
        let sq_sum: f32 = last_hidden.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        let normed_final: Vec<f32> = last_hidden
            .iter()
            .enumerate()
            .map(|(i, &x)| x / rms * self.output_norm_weight[i])
            .collect();

        // 4. LM head projection using SIMD matmul
        let vocab_size = self.config.vocab_size;
        let logits = fused_q4_0_q8_0_parallel_matvec(
            &self.lm_head_weight.data,
            &normed_final,
            hidden_dim,
            vocab_size,
        )?;

        Ok(logits)
    }

    /// Create a KV cache for this model
    #[must_use]
    pub fn create_kv_cache(&self) -> AprKVCache {
        AprKVCache::new(&self.config)
    }
}
