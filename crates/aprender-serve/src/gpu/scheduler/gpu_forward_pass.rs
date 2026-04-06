impl GpuModel {

    /// GPU-accelerated forward pass
    ///
    /// Uses HybridScheduler for matrix multiplications.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token IDs
    ///
    /// # Returns
    ///
    /// Logits tensor with shape `[seq_len, vocab_size]`
    ///
    /// # Errors
    ///
    /// Returns error if forward pass fails
    pub fn forward_gpu(&mut self, token_ids: &[usize]) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token IDs cannot be empty".to_string(),
            });
        }

        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_dim;

        // Step 1: Embed tokens
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            if token_id >= self.config.vocab_size {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Token ID {} out of bounds (vocab_size={})",
                        token_id, self.config.vocab_size
                    ),
                });
            }
            let offset = token_id * hidden_dim;
            hidden.extend_from_slice(&self.embedding_weights[offset..offset + hidden_dim]);
        }

        // Step 2: Pass through transformer blocks
        for block_idx in 0..self.block_weights.len() {
            hidden = self.forward_block_idx(&hidden, seq_len, block_idx)?;
        }

        // Step 3: Final layer norm
        hidden = self.layer_norm(&hidden, &self.final_norm_weight, &self.final_norm_bias);

        // Step 4: LM head projection
        // [seq_len, hidden_dim] @ [hidden_dim, vocab_size] -> [seq_len, vocab_size]
        // Phase 22 FIX: Use lm_head_weight_t (transposed) which is [hidden_dim, vocab_size]
        // The original lm_head_weight is [vocab_size, hidden_dim] (APR convention)
        // IMP-090: Use CPU fallback for large vocab to avoid GPU buffer overflow
        let lm_head_elements = hidden_dim * self.config.vocab_size;
        let logits = if exceeds_gpu_buffer_limit(lm_head_elements) {
            // CPU fallback for large vocab (>256MB weight matrix)
            cpu_matmul(
                &hidden,
                &self.lm_head_weight_t,
                seq_len,
                hidden_dim,
                self.config.vocab_size,
            )
        } else {
            // GPU path for smaller vocab (IMP-1005: use do_matmul for CUDA)
            // Clone weights to avoid borrow conflict with &mut self in do_matmul
            let lm_weight = self.lm_head_weight_t.clone();
            self.do_matmul(
                &hidden,
                &lm_weight,
                seq_len,
                hidden_dim,
                self.config.vocab_size,
            )?
        };

        // Add bias
        let mut output = logits;
        for i in 0..seq_len {
            for j in 0..self.config.vocab_size {
                output[i * self.config.vocab_size + j] += self.lm_head_bias[j];
            }
        }

        Ok(output)
    }

    /// GPU-accelerated forward pass with layer-by-layer tracing (PMAT-216)
    ///
    /// This method enables diagnostics by capturing activation statistics
    /// at each layer, matching the CPU `AprTransformer::forward_traced()` API.
    ///
    /// # Five Whys Root Cause (PMAT-216)
    ///
    /// The GPU path previously lacked tracing, allowing bugs to ship undetected.
    /// This method ensures GPU/CPU parity can be verified at each layer.
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
    /// Returns error if forward pass fails
    pub fn forward_traced_gpu(&mut self, token_ids: &[usize]) -> Result<ForwardTrace> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token IDs cannot be empty".to_string(),
            });
        }

        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_dim;

        // Step 1: Embed tokens
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in token_ids {
            if token_id >= self.config.vocab_size {
                return Err(RealizarError::InvalidShape {
                    reason: format!(
                        "Token ID {} out of bounds (vocab_size={})",
                        token_id, self.config.vocab_size
                    ),
                });
            }
            let offset = token_id * hidden_dim;
            hidden.extend_from_slice(&self.embedding_weights[offset..offset + hidden_dim]);
        }
        let embed_stats = ActivationStats::from_slice(&hidden);

        // Step 2: Pass through transformer blocks with tracing
        let mut layer_activations = Vec::with_capacity(self.block_weights.len());

        for block_idx in 0..self.block_weights.len() {
            let layer_trace = self.forward_block_traced(&hidden, seq_len, block_idx)?;
            hidden = layer_trace.0;
            layer_activations.push(layer_trace.1);
        }

        // Step 3: Final layer norm
        hidden = self.layer_norm(&hidden, &self.final_norm_weight, &self.final_norm_bias);
        let final_norm_stats = ActivationStats::from_slice(&hidden);

        // Step 4: LM head projection (only last token for logits)
        let last_hidden_start = (seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_hidden_start..last_hidden_start + hidden_dim];

        // Use lm_head_weight_t (transposed) for matmul
        let logits = cpu_matmul(
            last_hidden,
            &self.lm_head_weight_t,
            1,
            hidden_dim,
            self.config.vocab_size,
        );

        // Add bias
        let mut logits_with_bias = logits;
        for j in 0..self.config.vocab_size {
            logits_with_bias[j] += self.lm_head_bias[j];
        }

        let logits_stats = ActivationStats::from_slice(&logits_with_bias);

        Ok(ForwardTrace {
            input_tokens: token_ids.iter().map(|&x| x as u32).collect(),
            embed_stats,
            layer_activations,
            final_norm_stats,
            logits_stats,
            logits: logits_with_bias,
        })
    }

    /// Forward pass through a single block with tracing (PMAT-216)
    ///
    /// Returns (output_hidden, LayerActivation) for diagnostics.
    fn forward_block_traced(
        &mut self,
        input: &[f32],
        seq_len: usize,
        block_idx: usize,
    ) -> Result<(Vec<f32>, LayerActivation)> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = self.config.intermediate_dim;
        let qkv_dim = self.config.qkv_dim();

        let block = &self.block_weights[block_idx];

        // Pre-norm
        let normed = Self::layer_norm_static(
            input,
            &block.attn_norm_weight,
            &block.attn_norm_bias,
            hidden_dim,
            self.config.eps,
        );
        let attn_norm_stats = ActivationStats::from_slice(&normed);

        // QKV projection
        let qkv_weight = self.block_weights[block_idx].qkv_weight.clone();
        let mut qkv = self.do_matmul(&normed, &qkv_weight, seq_len, hidden_dim, qkv_dim)?;

        // Apply RoPE
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = self.config.head_dim();
        let kv_dim = self.config.kv_dim();
        let rope_theta = self.config.rope_theta;

        for pos in 0..seq_len {
            let qkv_start = pos * qkv_dim;
            Self::apply_rope_inline(
                &mut qkv[qkv_start..qkv_start + hidden_dim],
                num_heads,
                head_dim,
                rope_theta,
                pos,
            );
            Self::apply_rope_inline(
                &mut qkv[qkv_start + hidden_dim..qkv_start + hidden_dim + kv_dim],
                num_kv_heads,
                head_dim,
                rope_theta,
                pos,
            );
        }
        let qkv_stats = ActivationStats::from_slice(&qkv);

        // Attention
        let attn_out = self.optimized_gqa_attention(&qkv, seq_len)?;

        // Clone all weights/biases upfront to avoid borrow conflicts with do_matmul
        let out_weight = self.block_weights[block_idx].out_weight.clone();
        let out_bias = self.block_weights[block_idx].out_bias.clone();
        let ffn_norm_weight = self.block_weights[block_idx].ffn_norm_weight.clone();
        let ffn_norm_bias = self.block_weights[block_idx].ffn_norm_bias.clone();
        let ffn_fc1_weight = self.block_weights[block_idx].ffn_fc1_weight.clone();
        let ffn_fc1_bias = self.block_weights[block_idx].ffn_fc1_bias.clone();
        let ffn_gate_weight = self.block_weights[block_idx].ffn_gate_weight.clone();
        let ffn_fc2_weight = self.block_weights[block_idx].ffn_fc2_weight.clone();
        let ffn_fc2_bias = self.block_weights[block_idx].ffn_fc2_bias.clone();

        // Output projection
        let projected = self.do_matmul(&attn_out, &out_weight, seq_len, hidden_dim, hidden_dim)?;

        // Residual 1
        let mut residual1: Vec<f32> = input
            .iter()
            .zip(projected.iter())
            .enumerate()
            .map(|(i, (&inp, &proj))| inp + proj + out_bias[i % hidden_dim])
            .collect();
        let attn_out_stats = ActivationStats::from_slice(&residual1);

        // FFN pre-norm
        let ffn_normed = Self::layer_norm_static(
            &residual1,
            &ffn_norm_weight,
            &ffn_norm_bias,
            hidden_dim,
            self.config.eps,
        );
        let ffn_norm_stats = ActivationStats::from_slice(&ffn_normed);

        // FFN
        let ffn_output: Vec<f32> = if let Some(gate_weight) = ffn_gate_weight {
            // SwiGLU
            let up_out = self.do_matmul(
                &ffn_normed,
                &ffn_fc1_weight,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;
            let gate_out = self.do_matmul(
                &ffn_normed,
                &gate_weight,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;

            let activated: Vec<f32> = gate_out
                .iter()
                .zip(up_out.iter())
                .map(|(&g, &u)| {
                    let silu = g / (1.0 + (-g).exp());
                    silu * u
                })
                .collect();

            let down = self.do_matmul(
                &activated,
                &ffn_fc2_weight,
                seq_len,
                intermediate_dim,
                hidden_dim,
            )?;
            down.iter()
                .enumerate()
                .map(|(i, &d)| d + ffn_fc2_bias[i % hidden_dim])
                .collect()
        } else {
            // Standard GELU MLP
            let up_out = self.do_matmul(
                &ffn_normed,
                &ffn_fc1_weight,
                seq_len,
                hidden_dim,
                intermediate_dim,
            )?;
            let activated: Vec<f32> = up_out
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let biased = x + ffn_fc1_bias[i % intermediate_dim];
                    0.5 * biased
                        * (1.0 + (0.797_884_6 * (biased + 0.044_715 * biased.powi(3))).tanh())
                })
                .collect();

            let down = self.do_matmul(
                &activated,
                &ffn_fc2_weight,
                seq_len,
                intermediate_dim,
                hidden_dim,
            )?;
            down.iter()
                .enumerate()
                .map(|(i, &d)| d + ffn_fc2_bias[i % hidden_dim])
                .collect()
        };
        let ffn_out_stats = ActivationStats::from_slice(&ffn_output);

        // Residual 2
        for i in 0..residual1.len() {
            residual1[i] += ffn_output[i];
        }
        let output_stats = ActivationStats::from_slice(&residual1);

        Ok((
            residual1,
            LayerActivation {
                layer_idx: block_idx,
                attn_norm_stats,
                qkv_stats,
                attn_out_stats,
                ffn_norm_stats,
                ffn_out_stats,
                output_stats,
            },
        ))
    }
}
