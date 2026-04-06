impl OwnedQuantizedModel {
    /// GH-177: Run the decoder forward pass with cross-attention.
///
/// Processes decoder tokens through decoder layers using:
/// 1. Causal self-attention (each position only sees previous)
/// 2. Cross-attention (Q from decoder, K/V from encoder output)
/// 3. FFN (GELU)
///
/// Returns logits over the vocabulary for the last decoder position.
pub fn decode(
    &self,
    decoder_tokens: &[u32],
    encoder_output: &EncoderOutput,
) -> Result<Vec<f32>> {
    if !self.config.is_encoder_decoder() {
        return Err(crate::error::RealizarError::UnsupportedOperation {
            operation: "decode".to_string(),
            reason: format!(
                "decode() requires encoder-decoder architecture, got '{}'",
                self.config.architecture
            ),
        });
    }

    let hidden_dim = self.config.hidden_dim;
    let decoder_len = decoder_tokens.len();
    let head_dim = self.config.head_dim();
    let num_heads = self.config.num_heads;
    let num_kv_heads = self.config.num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let tile_size = 64;
    let use_rmsnorm = self.config.constraints.uses_rmsnorm();

    // Token embedding lookup for decoder
    let mut hidden = Vec::with_capacity(decoder_len * hidden_dim);
    for &token_id in decoder_tokens {
        let start = token_id as usize * hidden_dim;
        let end = start + hidden_dim;
        if end > self.token_embedding.len() {
            return Err(crate::error::RealizarError::InferenceError(format!(
                "Decoder token ID {} out of range", token_id
            )));
        }
        hidden.extend_from_slice(&self.token_embedding[start..end]);
    }

    let q_dim = num_heads * head_dim;
    let kv_dim_per = num_kv_heads * head_dim;
    let group_size = if num_kv_heads > 0 { num_heads / num_kv_heads } else { 1 };

    for layer in self.layers.iter() {
        // === Causal self-attention ===
        let normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(&hidden, &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(), self.config.eps)
        };

        let (mut full_q, mut full_k, mut full_v) = (
            Vec::with_capacity(decoder_len * q_dim),
            Vec::with_capacity(decoder_len * kv_dim_per),
            Vec::with_capacity(decoder_len * kv_dim_per),
        );
        for pos in 0..decoder_len {
            let mut qkv = self.qkv_matmul(&normed[pos * hidden_dim..(pos + 1) * hidden_dim], &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias { ops::add_bias(&mut qkv, bias); }
            full_q.extend_from_slice(&qkv[0..q_dim]);
            full_k.extend_from_slice(&qkv[q_dim..q_dim + kv_dim_per]);
            full_v.extend_from_slice(&qkv[q_dim + kv_dim_per..q_dim + 2 * kv_dim_per]);
        }

        let mut self_attn_out = vec![0.0f32; decoder_len * hidden_dim];
        for h in 0..num_heads {
            let kv_h = h / group_size;
            let (mut h_q, mut h_k, mut h_v) = (
                Vec::with_capacity(decoder_len * head_dim),
                Vec::with_capacity(decoder_len * head_dim),
                Vec::with_capacity(decoder_len * head_dim),
            );
            for pos in 0..decoder_len {
                h_q.extend_from_slice(&full_q[pos * q_dim + h * head_dim..pos * q_dim + h * head_dim + head_dim]);
                h_k.extend_from_slice(&full_k[pos * kv_dim_per + kv_h * head_dim..pos * kv_dim_per + kv_h * head_dim + head_dim]);
                h_v.extend_from_slice(&full_v[pos * kv_dim_per + kv_h * head_dim..pos * kv_dim_per + kv_h * head_dim + head_dim]);
            }
            let head_out = self.tiled_causal_attention(&h_q, &h_k, &h_v, decoder_len, head_dim, scale, tile_size)?;
            for pos in 0..decoder_len {
                let dst = pos * hidden_dim + h * head_dim;
                self_attn_out[dst..dst + head_dim].copy_from_slice(&head_out[pos * head_dim..(pos + 1) * head_dim]);
            }
        }

        // Self-attention output projection + residual
        for pos in 0..decoder_len {
            let mut proj = self.fused_matmul(&self_attn_out[pos * hidden_dim..(pos + 1) * hidden_dim], &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias { ops::add_bias(&mut proj, bias); }
            for i in 0..hidden_dim { hidden[pos * hidden_dim + i] += proj[i]; }
        }

        // === Cross-attention (Q from decoder, K/V from encoder) ===
        let cross_normed = if use_rmsnorm {
            ops::rms_norm(&hidden, &layer.attn_norm_weight, self.config.eps)
        } else {
            ops::layer_norm(&hidden, &layer.attn_norm_weight,
                layer.attn_norm_bias.as_deref(), self.config.eps)
        };

        let enc_len = encoder_output.seq_len;
        let mut dec_q_all = Vec::with_capacity(decoder_len * q_dim);
        for pos in 0..decoder_len {
            let mut qkv = self.qkv_matmul(&cross_normed[pos * hidden_dim..(pos + 1) * hidden_dim], &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias { ops::add_bias(&mut qkv, bias); }
            dec_q_all.extend_from_slice(&qkv[0..q_dim]);
        }
        let (mut enc_k_all, mut enc_v_all) = (
            Vec::with_capacity(enc_len * kv_dim_per),
            Vec::with_capacity(enc_len * kv_dim_per),
        );
        for pos in 0..enc_len {
            let mut qkv = self.qkv_matmul(&encoder_output.hidden_states[pos * hidden_dim..(pos + 1) * hidden_dim], &layer.qkv_weight)?;
            if let Some(ref bias) = layer.qkv_bias { ops::add_bias(&mut qkv, bias); }
            enc_k_all.extend_from_slice(&qkv[q_dim..q_dim + kv_dim_per]);
            enc_v_all.extend_from_slice(&qkv[q_dim + kv_dim_per..q_dim + 2 * kv_dim_per]);
        }

        let mut cross_attn_out = vec![0.0f32; decoder_len * hidden_dim];
        for h in 0..num_heads {
            let kv_h = h / group_size;
            let (mut h_q, mut h_k, mut h_v) = (
                Vec::with_capacity(decoder_len * head_dim),
                Vec::with_capacity(enc_len * head_dim),
                Vec::with_capacity(enc_len * head_dim),
            );
            for pos in 0..decoder_len {
                h_q.extend_from_slice(&dec_q_all[pos * q_dim + h * head_dim..pos * q_dim + h * head_dim + head_dim]);
            }
            for pos in 0..enc_len {
                h_k.extend_from_slice(&enc_k_all[pos * kv_dim_per + kv_h * head_dim..pos * kv_dim_per + kv_h * head_dim + head_dim]);
                h_v.extend_from_slice(&enc_v_all[pos * kv_dim_per + kv_h * head_dim..pos * kv_dim_per + kv_h * head_dim + head_dim]);
            }
            let head_out = self.tiled_cross_attention(&h_q, &h_k, &h_v, decoder_len, enc_len, head_dim, scale, tile_size)?;
            for pos in 0..decoder_len {
                let dst = pos * hidden_dim + h * head_dim;
                cross_attn_out[dst..dst + head_dim].copy_from_slice(&head_out[pos * head_dim..(pos + 1) * head_dim]);
            }
        }

        // Cross-attention output projection + residual
        for pos in 0..decoder_len {
            let mut proj = self.fused_matmul(&cross_attn_out[pos * hidden_dim..(pos + 1) * hidden_dim], &layer.attn_output_weight)?;
            if let Some(ref bias) = layer.attn_output_bias { ops::add_bias(&mut proj, bias); }
            for i in 0..hidden_dim { hidden[pos * hidden_dim + i] += proj[i]; }
        }

        // === FFN block ===
        for pos in 0..decoder_len {
            let pos_h = &hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                if use_rmsnorm { ops::rms_norm(pos_h, ffn_norm, self.config.eps) }
                else { ops::layer_norm(pos_h, ffn_norm, layer.ffn_norm_bias.as_deref(), self.config.eps) }
            } else { pos_h.to_vec() };

            let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
            if let Some(ref bias) = layer.ffn_up_bias { ops::add_bias(&mut ffn_hidden, bias); }
            ops::gelu(&mut ffn_hidden);
            let mut ffn_out = self.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
            if let Some(ref bias) = layer.ffn_down_bias { ops::add_bias(&mut ffn_out, bias); }
            for i in 0..hidden_dim { hidden[pos * hidden_dim + i] += ffn_out[i]; }
        }
    }

    // GH-177 Item 4: Final norm + LM head projection
    let last_pos = decoder_len - 1;
    let last_hidden = &hidden[last_pos * hidden_dim..(last_pos + 1) * hidden_dim];
    let normed = if use_rmsnorm {
        ops::rms_norm(last_hidden, &self.output_norm_weight, self.config.eps)
    } else {
        ops::layer_norm(last_hidden, &self.output_norm_weight, self.output_norm_bias.as_deref(), self.config.eps)
    };
    let mut logits = self.fused_matmul(&normed, &self.lm_head_weight)?;
    if let Some(ref bias) = self.lm_head_bias { ops::add_bias(&mut logits, bias); }
    Ok(logits)
}
}
