//! `BertLayer` — single transformer encoder block (post-norm, BERT-style).
//!
//! Differs from `nn::transformer::TransformerEncoderLayer` (which is pre-norm):
//! BERT applies LayerNorm AFTER the residual add, matching the original
//! "Attention Is All You Need" paper and the HuggingFace reference.

use crate::autograd::Tensor;
use crate::models::bert::config::BertConfig;
use crate::nn::functional::gelu;
use crate::nn::transformer::MultiHeadAttention;
use crate::nn::{LayerNorm, Linear, Module};

/// Single BERT encoder layer.
///
/// Post-norm sublayer pattern:
/// ```text
///   hidden = LayerNorm(hidden + MultiHeadAttention(hidden))
///   hidden = LayerNorm(hidden + Linear(GELU(Linear(hidden))))
/// ```
pub struct BertLayer {
    /// Self-attention (Q/K/V/O projections with bias).
    attention: MultiHeadAttention,
    /// LayerNorm applied after attention residual.
    attention_norm: LayerNorm,
    /// FFN expand projection (`hidden_dim` → `intermediate_dim`).
    intermediate: Linear,
    /// FFN contract projection (`intermediate_dim` → `hidden_dim`).
    output_dense: Linear,
    /// LayerNorm applied after FFN residual.
    output_norm: LayerNorm,
}

impl BertLayer {
    /// Construct a layer with zero/identity-initialized weights of the right shape.
    #[must_use]
    pub fn new(config: &BertConfig) -> Self {
        let h = config.hidden_dim;
        let intermediate = config.intermediate_dim;
        Self {
            attention: MultiHeadAttention::new(h, config.num_heads),
            attention_norm: LayerNorm::with_eps(&[h], config.layer_norm_eps),
            intermediate: Linear::new(h, intermediate),
            output_dense: Linear::new(intermediate, h),
            output_norm: LayerNorm::with_eps(&[h], config.layer_norm_eps),
        }
    }

    /// Forward pass on `[seq_len, hidden_dim]`.
    ///
    /// `attn_mask` is the optional additive mask broadcast to attention scores
    /// (use large-negative values to mask out positions, e.g. for padding).
    #[must_use]
    pub fn forward(&self, hidden: &Tensor, attn_mask: Option<&Tensor>) -> Tensor {
        // Self-attention + residual + LayerNorm (post-norm).
        let (attn_out, _) = self.attention.forward_self(hidden, attn_mask);
        let attn_residual = hidden.add(&attn_out);
        let attn_normalized = self.attention_norm.forward(&attn_residual);

        // FFN: Linear → GELU → Linear; then residual + LayerNorm.
        let intermediate = self.intermediate.forward(&attn_normalized);
        let intermediate_act = gelu(&intermediate);
        let ffn_out = self.output_dense.forward(&intermediate_act);
        let ffn_residual = attn_normalized.add(&ffn_out);
        self.output_norm.forward(&ffn_residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bert_layer_preserves_shape() {
        let config = BertConfig::minilm_l6();
        let layer = BertLayer::new(&config);
        let seq_len = 5;
        let h = config.hidden_dim;
        let input = Tensor::from_vec(vec![0.0; seq_len * h], &[1, seq_len, h]);
        let out = layer.forward(&input, None);
        assert_eq!(out.shape(), &[1, seq_len, h]);
    }

    #[test]
    fn bert_layer_handles_long_seq() {
        let config = BertConfig::minilm_l6();
        let layer = BertLayer::new(&config);
        let seq_len = 128;
        let h = config.hidden_dim;
        let input = Tensor::from_vec(vec![0.0; seq_len * h], &[1, seq_len, h]);
        let out = layer.forward(&input, None);
        assert_eq!(out.shape(), &[1, seq_len, h]);
    }
}
