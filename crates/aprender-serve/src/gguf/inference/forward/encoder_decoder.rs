//! GH-177: Encoder-decoder forward pass for T5/Whisper
//!
//! Provides `encode()` and `decode()` methods on OwnedQuantizedModel
//! for encoder-decoder architectures (T5, Whisper).
//!
//! The encoder runs bidirectional attention (no causal mask).
//! The decoder runs causal self-attention + cross-attention
//! (Q from decoder, K/V from encoder output).

use crate::error::Result;
use crate::gguf::ops;
use crate::gguf::OwnedQuantizedModel;

/// Encoder output: hidden states from the encoder.
///
/// Stored and passed to the decoder for cross-attention.
/// For T5-small (6 layers, hidden=512), this is ~2KB per token.
#[derive(Debug, Clone)]
pub struct EncoderOutput {
    /// Hidden states: [seq_len, hidden_dim]
    pub hidden_states: Vec<f32>,
    /// Sequence length of the encoder input
    pub seq_len: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
}

impl OwnedQuantizedModel {
    /// GH-177: Run the encoder forward pass.
    ///
    /// Processes input tokens through the encoder layers using
    /// bidirectional attention (each position attends to all others).
    ///
    /// Each encoder layer: LayerNorm → self-attn (bidirectional) →
    /// residual → LayerNorm → FFN (GELU) → residual.
    pub fn encode(&self, input_tokens: &[u32]) -> Result<EncoderOutput> {
        if !self.config.is_encoder_decoder() {
            return Err(crate::error::RealizarError::UnsupportedOperation {
                operation: "encode".to_string(),
                reason: format!(
                    "encode() requires encoder-decoder architecture, \
                     got '{}'",
                    self.config.architecture
                ),
            });
        }

        let hidden_dim = self.config.hidden_dim;
        let seq_len = input_tokens.len();

        // Token embedding lookup
        let mut hidden = Vec::with_capacity(seq_len * hidden_dim);
        for &token_id in input_tokens {
            let start = token_id as usize * hidden_dim;
            let end = start + hidden_dim;
            if end > self.token_embedding.len() {
                return Err(crate::error::RealizarError::InferenceError(format!(
                    "Token ID {} out of range (embedding size: {})",
                    token_id,
                    self.token_embedding.len() / hidden_dim
                )));
            }
            hidden.extend_from_slice(&self.token_embedding[start..end]);
        }

        // GH-177 Item 2: Run through encoder layers with bidirectional attention.
        // Uses encoder_layers if populated, falls back to self.layers for
        // models where encoder/decoder share the same layer format.
        let encoder_layers = if self.encoder_layers.is_empty() {
            &self.layers
        } else {
            &self.encoder_layers
        };

        let head_dim = self.config.head_dim();
        let num_heads = self.config.num_heads;
        let num_kv_heads = self.config.num_kv_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let tile_size = 64;

        for layer in encoder_layers.iter() {
            // Pre-attention LayerNorm (T5 uses LayerNorm, not RMSNorm)
            let use_rmsnorm = self.config.constraints.uses_rmsnorm();
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

            // QKV projection for all positions
            let mut all_attn_out = vec![0.0f32; seq_len * hidden_dim];

            // Project Q, K, V for entire sequence
            let q_dim = num_heads * head_dim;
            let kv_dim_per = num_kv_heads * head_dim;

            let mut full_q = Vec::with_capacity(seq_len * q_dim);
            let mut full_k = Vec::with_capacity(seq_len * kv_dim_per);
            let mut full_v = Vec::with_capacity(seq_len * kv_dim_per);

            for pos in 0..seq_len {
                let pos_hidden = &normed[pos * hidden_dim..(pos + 1) * hidden_dim];
                let mut qkv = self.qkv_matmul(pos_hidden, &layer.qkv_weight)?;
                if let Some(ref bias) = layer.qkv_bias {
                    ops::add_bias(&mut qkv, bias);
                }

                let q = &qkv[0..q_dim];
                let k = &qkv[q_dim..q_dim + kv_dim_per];
                let v = &qkv[q_dim + kv_dim_per..q_dim + 2 * kv_dim_per];

                full_q.extend_from_slice(q);
                full_k.extend_from_slice(k);
                full_v.extend_from_slice(v);
            }

            // Note: T5 uses relative position bias, not RoPE.
            // For now, skip positional encoding (relative bias not yet impl).
            // This is correct for models that don't use any positional encoding
            // in the attention computation itself (e.g., absolute positions
            // added at embedding time).

            // Multi-head bidirectional attention
            let group_size = if num_kv_heads > 0 {
                num_heads / num_kv_heads
            } else {
                1
            };

            for h in 0..num_heads {
                let kv_h = h / group_size;

                // Extract per-head Q, K, V across all positions
                let mut h_q = Vec::with_capacity(seq_len * head_dim);
                let mut h_k = Vec::with_capacity(seq_len * head_dim);
                let mut h_v = Vec::with_capacity(seq_len * head_dim);

                for pos in 0..seq_len {
                    let q_start = pos * q_dim + h * head_dim;
                    h_q.extend_from_slice(&full_q[q_start..q_start + head_dim]);

                    let k_start = pos * kv_dim_per + kv_h * head_dim;
                    h_k.extend_from_slice(&full_k[k_start..k_start + head_dim]);

                    let v_start = pos * kv_dim_per + kv_h * head_dim;
                    h_v.extend_from_slice(&full_v[v_start..v_start + head_dim]);
                }

                // Bidirectional attention (all positions attend to all)
                let head_out = self.tiled_bidirectional_attention(
                    &h_q, &h_k, &h_v, seq_len, head_dim, scale, tile_size,
                )?;

                // Scatter back to multi-head output
                for pos in 0..seq_len {
                    let src = &head_out[pos * head_dim..(pos + 1) * head_dim];
                    let dst_start = pos * hidden_dim + h * head_dim;
                    all_attn_out[dst_start..dst_start + head_dim].copy_from_slice(src);
                }
            }

            // Output projection + residual
            for pos in 0..seq_len {
                let attn_pos = &all_attn_out[pos * hidden_dim..(pos + 1) * hidden_dim];
                let mut proj = self.fused_matmul(attn_pos, &layer.attn_output_weight)?;
                if let Some(ref bias) = layer.attn_output_bias {
                    ops::add_bias(&mut proj, bias);
                }
                for i in 0..hidden_dim {
                    hidden[pos * hidden_dim + i] += proj[i];
                }
            }

            // FFN block: LayerNorm → up → GELU → down → residual
            for pos in 0..seq_len {
                let pos_hidden = &hidden[pos * hidden_dim..(pos + 1) * hidden_dim];

                let ffn_input = if let Some(ref ffn_norm) = layer.ffn_norm_weight {
                    if use_rmsnorm {
                        ops::rms_norm(pos_hidden, ffn_norm, self.config.eps)
                    } else {
                        ops::layer_norm(
                            pos_hidden,
                            ffn_norm,
                            layer.ffn_norm_bias.as_deref(),
                            self.config.eps,
                        )
                    }
                } else {
                    pos_hidden.to_vec()
                };

                // T5 uses GELU FFN (no gate weight)
                let mut ffn_hidden = self.fused_matmul(&ffn_input, &layer.ffn_up_weight)?;
                if let Some(ref bias) = layer.ffn_up_bias {
                    ops::add_bias(&mut ffn_hidden, bias);
                }
                ops::gelu(&mut ffn_hidden);

                let mut ffn_out = self.fused_matmul(&ffn_hidden, &layer.ffn_down_weight)?;
                if let Some(ref bias) = layer.ffn_down_bias {
                    ops::add_bias(&mut ffn_out, bias);
                }

                // Residual
                for i in 0..hidden_dim {
                    hidden[pos * hidden_dim + i] += ffn_out[i];
                }
            }
        }

        // Encoder final norm
        let use_rmsnorm = self.config.constraints.uses_rmsnorm();
        let enc_norm_weight = self
            .encoder_output_norm_weight
            .as_deref()
            .unwrap_or(&self.output_norm_weight);
        let enc_norm_bias = self
            .encoder_output_norm_bias
            .as_deref()
            .or(self.output_norm_bias.as_deref());

        let mut normed_hidden = Vec::with_capacity(seq_len * hidden_dim);
        for pos in 0..seq_len {
            let pos_h = &hidden[pos * hidden_dim..(pos + 1) * hidden_dim];
            let normed = if use_rmsnorm {
                ops::rms_norm(pos_h, enc_norm_weight, self.config.eps)
            } else {
                ops::layer_norm(pos_h, enc_norm_weight, enc_norm_bias, self.config.eps)
            };
            normed_hidden.extend_from_slice(&normed);
        }

        Ok(EncoderOutput {
            hidden_states: normed_hidden,
            seq_len,
            hidden_dim,
        })
    }
}

include!("encoder_decoder_decode.rs");
