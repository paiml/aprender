/// Apply RMS norm in-place from `src` into `dst`, weighted by `gamma`.
fn rms_norm_weighted(src: &[f32], gamma: &[f32], dst: &mut [f32], eps: f32) {
    let n = src.len();
    let sq_sum: f32 = src.iter().map(|x| x * x).sum();
    let rms = (sq_sum / n as f32 + eps).sqrt();
    for i in 0..n {
        dst[i] = src[i] / rms * gamma[i];
    }
}

/// Apply RMS norm to a multi-token hidden state, returning normalized output.
fn rms_norm_batched(hidden: &[f32], gamma: &[f32], hidden_dim: usize, eps: f32) -> Vec<f32> {
    let seq_len = hidden.len() / hidden_dim;
    let mut out = Vec::with_capacity(hidden.len());
    for s in 0..seq_len {
        let slice = &hidden[s * hidden_dim..(s + 1) * hidden_dim];
        let sq_sum: f32 = slice.iter().map(|x| x * x).sum();
        let rms = (sq_sum / hidden_dim as f32 + eps).sqrt();
        for (i, &x) in slice.iter().enumerate() {
            out.push(x / rms * gamma[i]);
        }
    }
    out
}

/// Apply SwiGLU activation: silu(gate) * up, in-place on `up`.
fn apply_swiglu(up: &mut [f32], gate: &[f32]) {
    for i in 0..up.len() {
        let silu = gate[i] / (1.0 + (-gate[i]).exp());
        up[i] *= silu;
    }
}

/// Apply GELU activation in-place.
fn apply_gelu(data: &mut [f32]) {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    const GELU_COEFF: f32 = 0.044_715;
    for x in data.iter_mut() {
        let t = (SQRT_2_OVER_PI * (*x + GELU_COEFF * *x * *x * *x)).tanh();
        *x = 0.5 * *x * (1.0 + t);
    }
}

/// Embed token IDs into hidden states.
fn embed_tokens(token_ids: &[u32], embedding: &[f32], hidden_dim: usize) -> Vec<f32> {
    let mut hidden = Vec::with_capacity(token_ids.len() * hidden_dim);
    for &token_id in token_ids {
        let offset = (token_id as usize) * hidden_dim;
        if offset + hidden_dim <= embedding.len() {
            hidden.extend_from_slice(&embedding[offset..offset + hidden_dim]);
        } else {
            hidden.extend(std::iter::repeat_n(0.0, hidden_dim));
        }
    }
    hidden
}

/// Add `src` elementwise into `dst`.
fn residual_add(dst: &mut [f32], src: &[f32]) {
    for i in 0..dst.len() {
        dst[i] += src[i];
    }
}

impl QuantizedAprTransformerQ4 {
    /// Forward pass for a single token using scratch buffer (zero allocation).
    pub fn forward_single_with_scratch(
        &self,
        token_id: u32,
        scratch: &mut AprInferenceScratch,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec_into;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let eps = self.config.eps;

        // 1. Token embedding lookup
        let offset = (token_id as usize) * hidden_dim;
        if offset + hidden_dim <= self.token_embedding.len() {
            scratch.hidden[..hidden_dim]
                .copy_from_slice(&self.token_embedding[offset..offset + hidden_dim]);
        } else {
            scratch.hidden[..hidden_dim].fill(0.0);
        }

        // 2. Process through transformer layers
        for layer in &self.layers {
            rms_norm_weighted(&scratch.hidden, &layer.attn_norm_weight, &mut scratch.normed, eps);

            let qkv_dim = layer.qkv_weight.out_dim;
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.qkv_weight.data, &scratch.normed[..hidden_dim],
                hidden_dim, &mut scratch.qkv_out[..qkv_dim],
            )?;

            let q_dim = num_heads * head_dim;
            let kv_dim = num_kv_heads * head_dim;
            scratch.q[..q_dim].copy_from_slice(&scratch.qkv_out[..q_dim]);
            scratch.k[..kv_dim].copy_from_slice(&scratch.qkv_out[q_dim..q_dim + kv_dim]);
            scratch.v[..kv_dim]
                .copy_from_slice(&scratch.qkv_out[q_dim + kv_dim..q_dim + 2 * kv_dim]);

            self.apply_rope(&mut scratch.q[..q_dim], 0, num_heads);
            self.apply_rope(&mut scratch.k[..kv_dim], 0, num_kv_heads);

            // Single token: attention output = V (softmax of 1 element = 1.0)
            let group_size = num_heads / num_kv_heads;
            for head in 0..num_heads {
                let kv_head = head / group_size;
                scratch.attn_out[head * head_dim..(head + 1) * head_dim]
                    .copy_from_slice(&scratch.v[kv_head * head_dim..(kv_head + 1) * head_dim]);
            }

            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.attn_output_weight.data, &scratch.attn_out[..hidden_dim],
                layer.attn_output_weight.in_dim,
                &mut scratch.ffn_out[..layer.attn_output_weight.out_dim],
            )?;
            residual_add(&mut scratch.hidden[..hidden_dim], &scratch.ffn_out[..hidden_dim]);

            // Pre-FFN norm
            if let Some(ffn_norm) = &layer.ffn_norm_weight {
                rms_norm_weighted(&scratch.hidden, ffn_norm, &mut scratch.ffn_input, eps);
            } else {
                scratch.ffn_input[..hidden_dim].copy_from_slice(&scratch.normed[..hidden_dim]);
            }

            // FFN
            let intermediate_dim = layer.ffn_up_weight.out_dim;
            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.ffn_up_weight.data, &scratch.ffn_input[..hidden_dim],
                hidden_dim, &mut scratch.ffn_up[..intermediate_dim],
            )?;
            if let Some(gate) = &layer.ffn_gate_weight {
                fused_q4_0_q8_0_parallel_matvec_into(
                    &gate.data, &scratch.ffn_input[..hidden_dim],
                    hidden_dim, &mut scratch.ffn_gate[..intermediate_dim],
                )?;
                apply_swiglu(&mut scratch.ffn_up[..intermediate_dim], &scratch.ffn_gate[..intermediate_dim]);
            } else {
                apply_gelu(&mut scratch.ffn_up[..intermediate_dim]);
            }

            fused_q4_0_q8_0_parallel_matvec_into(
                &layer.ffn_down_weight.data, &scratch.ffn_up[..intermediate_dim],
                intermediate_dim, &mut scratch.ffn_out[..hidden_dim],
            )?;
            residual_add(&mut scratch.hidden[..hidden_dim], &scratch.ffn_out[..hidden_dim]);
        }

        // 3. Final RMS norm
        rms_norm_weighted(&scratch.hidden, &self.output_norm_weight, &mut scratch.normed, eps);

        // 4. LM head projection
        let vocab_size = self.config.vocab_size;
        let mut logits = vec![0.0f32; vocab_size];
        fused_q4_0_q8_0_parallel_matvec_into(
            &self.lm_head_weight.data, &scratch.normed[..hidden_dim],
            hidden_dim, &mut logits,
        )?;

        Ok(logits)
    }

    /// Forward pass with KV cache for efficient autoregressive generation.
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        cache: &mut AprKVCache,
    ) -> Result<Vec<f32>> {
        if token_ids.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Token sequence cannot be empty".to_string(),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let eps = self.config.eps;
        let cache_len = cache.len();
        let new_seq_len = token_ids.len();

        let mut hidden = embed_tokens(token_ids, &self.token_embedding, hidden_dim);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let normed = rms_norm_batched(&hidden, &layer.attn_norm_weight, hidden_dim, eps);

            let attn_out = self.cached_layer_attention(
                &normed, layer, layer_idx, cache, cache_len, new_seq_len,
            )?;

            let proj_out = self.batched_matvec(&layer.attn_output_weight, &attn_out, new_seq_len)?;
            residual_add(&mut hidden, &proj_out);

            let ffn_input = match &layer.ffn_norm_weight {
                Some(ffn_norm) => rms_norm_batched(&hidden, ffn_norm, hidden_dim, eps),
                None => normed.clone(),
            };

            let ffn_out = self.cached_layer_ffn(layer, &ffn_input, new_seq_len)?;
            residual_add(&mut hidden, &ffn_out);
        }

        // Final norm + LM head (last token only)
        let last_start = (new_seq_len - 1) * hidden_dim;
        let last_hidden = &hidden[last_start..last_start + hidden_dim];
        let mut normed_final = vec![0.0f32; hidden_dim];
        rms_norm_weighted(last_hidden, &self.output_norm_weight, &mut normed_final, eps);

        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;
        fused_q4_0_q8_0_parallel_matvec(
            &self.lm_head_weight.data, &normed_final, hidden_dim, self.config.vocab_size,
        )
    }

    /// Attention sublayer for cached forward pass.
    fn cached_layer_attention(
        &self,
        normed: &[f32],
        layer: &QuantizedAprLayerQ4,
        layer_idx: usize,
        cache: &mut AprKVCache,
        cache_len: usize,
        new_seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;

        let hidden_dim = self.config.hidden_dim;
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let head_dim = hidden_dim / num_heads;
        let qkv_dim = layer.qkv_weight.out_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let mut qkv_out = Vec::with_capacity(new_seq_len * qkv_dim);
        for s in 0..new_seq_len {
            let input = &normed[s * hidden_dim..(s + 1) * hidden_dim];
            let qkv = fused_q4_0_q8_0_parallel_matvec(
                &layer.qkv_weight.data, input, hidden_dim, qkv_dim,
            )?;
            qkv_out.extend(qkv);
        }

        let mut new_q = Vec::with_capacity(new_seq_len * q_dim);
        for s in 0..new_seq_len {
            let base = s * qkv_dim;
            let position = cache_len + s;
            let mut q = qkv_out[base..base + q_dim].to_vec();
            let mut k = qkv_out[base + q_dim..base + q_dim + kv_dim].to_vec();
            let v = qkv_out[base + q_dim + kv_dim..base + q_dim + 2 * kv_dim].to_vec();
            self.apply_rope(&mut q, position, num_heads);
            self.apply_rope(&mut k, position, num_kv_heads);
            new_q.extend_from_slice(&q);
            cache.append(layer_idx, &k, &v);
        }

        let (full_k, full_v) = cache.get(layer_idx);
        let total_seq_len = cache.len();
        Ok(self.causal_attention_cached(&new_q, full_k, full_v, new_seq_len, total_seq_len, cache_len))
    }

    /// FFN sublayer for cached forward pass (SwiGLU or GELU).
    fn cached_layer_ffn(
        &self,
        layer: &QuantizedAprLayerQ4,
        ffn_input: &[f32],
        new_seq_len: usize,
    ) -> Result<Vec<f32>> {
        let hidden_dim = self.config.hidden_dim;
        let intermediate_dim = layer.ffn_up_weight.out_dim;

        let mut up = self.batched_matvec(&layer.ffn_up_weight, ffn_input, new_seq_len)?;

        if let Some(gate) = &layer.ffn_gate_weight {
            let gate_out = self.batched_matvec(gate, ffn_input, new_seq_len)?;
            apply_swiglu(&mut up, &gate_out);
        } else {
            apply_gelu(&mut up);
        }

        self.batched_matvec_custom(&layer.ffn_down_weight, &up, new_seq_len, intermediate_dim, hidden_dim)
    }

    /// Batched matvec: multiply each token's hidden state by a quantized weight matrix.
    fn batched_matvec(
        &self,
        weight: &QuantizedAprTensorQ4,
        input: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;
        let in_dim = weight.in_dim;
        let out_dim = weight.out_dim;
        let mut out = Vec::with_capacity(seq_len * out_dim);
        for s in 0..seq_len {
            let slice = &input[s * in_dim..(s + 1) * in_dim];
            let row = fused_q4_0_q8_0_parallel_matvec(&weight.data, slice, in_dim, out_dim)?;
            out.extend(row);
        }
        Ok(out)
    }

    /// Batched matvec with explicit dimensions (for down projection where dims differ).
    fn batched_matvec_custom(
        &self,
        weight: &QuantizedAprTensorQ4,
        input: &[f32],
        seq_len: usize,
        in_dim: usize,
        out_dim: usize,
    ) -> Result<Vec<f32>> {
        use crate::quantize::fused_q4_0_q8_0_parallel_matvec;
        let mut out = Vec::with_capacity(seq_len * out_dim);
        for s in 0..seq_len {
            let slice = &input[s * in_dim..(s + 1) * in_dim];
            let row = fused_q4_0_q8_0_parallel_matvec(&weight.data, slice, in_dim, out_dim)?;
            out.extend(row);
        }
        Ok(out)
    }
}
