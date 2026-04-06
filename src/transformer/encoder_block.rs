//! Encoder transformer block (BERT/RoBERTa architecture)
//!
//! Unlike decoder blocks (pre-norm RMSNorm + SwiGLU FFN), encoder blocks use:
//! - Post-norm architecture: residual + LayerNorm (not pre-norm)
//! - Bidirectional attention (no causal mask — same as existing `MultiHeadAttention`)
//! - GELU FFN with 2 projections (not SwiGLU with 3)
//! - LayerNorm with bias (not RMSNorm)
//!
//! # Contract (CLF-001)
//! - Output shape == input shape (seq_len × hidden_size)
//! - No NaN or Inf in output for finite input

use crate::autograd::add;
use crate::Tensor;
use std::collections::HashMap;

use super::attention::MultiHeadAttention;
use super::config::TransformerConfig;
use super::feedforward::EncoderFeedForward;
use super::norm::LayerNorm;

/// Encoder transformer block (BERT/RoBERTa).
///
/// Architecture: x → Attn(x) + x → LayerNorm → FFN + residual → LayerNorm
/// (post-norm, matching HuggingFace BERT implementation)
pub struct EncoderBlock {
    /// Layer index
    layer_idx: usize,
    /// Self-attention (bidirectional — no causal mask)
    pub self_attn: MultiHeadAttention,
    /// Post-attention LayerNorm
    pub attn_layernorm: LayerNorm,
    /// Feed-forward network (GELU, 2 projections)
    pub ffn: EncoderFeedForward,
    /// Post-FFN LayerNorm
    pub ffn_layernorm: LayerNorm,
    /// Hidden size for batched operations
    hidden_size: usize,
}

impl EncoderBlock {
    /// Create new encoder block with default initialization
    pub fn new(config: &TransformerConfig, layer_idx: usize) -> Self {
        let eps = config.rms_norm_eps; // reuse epsilon field
        Self {
            layer_idx,
            self_attn: MultiHeadAttention::new(config),
            attn_layernorm: LayerNorm::new(config.hidden_size, eps),
            ffn: EncoderFeedForward::new(config),
            ffn_layernorm: LayerNorm::new(config.hidden_size, eps),
            hidden_size: config.hidden_size,
        }
    }

    /// Create encoder block from pre-trained parameters.
    ///
    /// Expected weight names (after RoBERTa mapping):
    /// - `encoder.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `encoder.layers.{i}.input_layernorm.{weight,bias}`
    /// - `encoder.layers.{i}.mlp.intermediate.dense.{weight,bias}`
    /// - `encoder.layers.{i}.mlp.output.dense.{weight,bias}`
    /// - `encoder.layers.{i}.post_attention_layernorm.{weight,bias}`
    pub fn from_params(
        config: &TransformerConfig,
        params: &HashMap<String, Tensor>,
        layer_idx: usize,
    ) -> Option<Self> {
        let prefix = format!("encoder.layers.{layer_idx}");
        let eps = config.rms_norm_eps;

        let self_attn =
            MultiHeadAttention::from_params(config, params, &format!("{prefix}.self_attn"))?;

        let attn_layernorm = LayerNorm::from_params(
            params,
            &format!("{prefix}.input_layernorm"),
            eps,
            config.hidden_size,
        )?;

        let ffn = EncoderFeedForward::from_params(config, params, &format!("{prefix}.mlp"))?;

        let ffn_layernorm = LayerNorm::from_params(
            params,
            &format!("{prefix}.post_attention_layernorm"),
            eps,
            config.hidden_size,
        )?;

        Some(Self {
            layer_idx,
            self_attn,
            attn_layernorm,
            ffn,
            ffn_layernorm,
            hidden_size: config.hidden_size,
        })
    }

    /// Forward pass (post-norm encoder architecture).
    ///
    /// ```text
    /// h = LayerNorm(x + Attention(x))
    /// out = LayerNorm(h + FFN(h))
    /// ```
    pub fn forward(&self, x: &Tensor, seq_len: usize) -> Tensor {
        let h = self.hidden_size;

        // Self-attention + residual + LayerNorm
        let attn_out = self.self_attn.forward(x, seq_len);
        let residual1 = add(x, &attn_out);
        let norm1 = self.attn_layernorm.forward_batched(&residual1, seq_len, h);

        // FFN + residual + LayerNorm
        let ffn_out = self.ffn.forward(&norm1, seq_len);
        let residual2 = add(&norm1, &ffn_out);
        self.ffn_layernorm.forward_batched(&residual2, seq_len, h)
    }

    /// Get layer index
    pub fn layer_idx(&self) -> usize {
        self.layer_idx
    }

    /// Get all parameters (immutable)
    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.push(&self.attn_layernorm.weight);
        params.push(&self.attn_layernorm.bias);
        params.extend(self.ffn.parameters());
        params.push(&self.ffn_layernorm.weight);
        params.push(&self.ffn_layernorm.bias);
        params
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn clf_001_encoder_block_output_shape() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 0);
        let seq_len = 4;
        let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
        let output = block.forward(&x, seq_len);
        assert_eq!(output.len(), seq_len * config.hidden_size);
    }

    #[test]
    fn clf_001_encoder_block_output_finite() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 0);
        let seq_len = 3;
        let x = Tensor::from_vec(vec![0.5; seq_len * config.hidden_size], true);
        let output = block.forward(&x, seq_len);
        let data = output.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()), "encoder block output must be finite");
    }

    #[test]
    fn clf_001_encoder_block_layer_idx() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 7);
        assert_eq!(block.layer_idx(), 7);
    }

    #[test]
    fn clf_001_encoder_block_parameters_count() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 0);
        let params = block.parameters();
        // self_attn: 4 (Q,K,V,O) + attn_layernorm: 2 (w,b)
        // ffn: 4 (w_up,b_up,w_down,b_down) + ffn_layernorm: 2 (w,b)
        assert_eq!(params.len(), 12);
    }

    #[test]
    fn test_encoder_block_different_layer_indices() {
        let config = TransformerConfig::codebert();
        for idx in [0, 1, 5, 11] {
            let block = EncoderBlock::new(&config, idx);
            assert_eq!(block.layer_idx(), idx);
        }
    }

    #[test]
    fn test_encoder_block_forward_preserves_shape() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 0);
        for seq_len in [1, 2, 4, 8] {
            let x = Tensor::from_vec(vec![0.1; seq_len * config.hidden_size], true);
            let output = block.forward(&x, seq_len);
            assert_eq!(
                output.len(),
                seq_len * config.hidden_size,
                "Shape mismatch for seq_len={seq_len}"
            );
        }
    }

    #[test]
    fn test_encoder_block_deterministic() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 0);
        let seq_len = 3;
        let x = Tensor::from_vec(vec![0.3; seq_len * config.hidden_size], true);

        let out1 = block.forward(&x, seq_len);
        let out2 = block.forward(&x, seq_len);

        let d1 = out1.data();
        let d2 = out2.data();
        let s1 = d1.as_slice().unwrap();
        let s2 = d2.as_slice().unwrap();
        assert_eq!(s1, s2, "Encoder block should be deterministic");
    }

    #[test]
    fn test_encoder_block_from_params_missing() {
        let config = TransformerConfig::codebert();
        let empty_params: HashMap<String, Tensor> = HashMap::new();
        let result = EncoderBlock::from_params(&config, &empty_params, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_encoder_block_hidden_size() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 0);
        assert_eq!(block.hidden_size, config.hidden_size);
    }

    #[test]
    fn test_encoder_block_parameters_nonzero_length() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 0);
        let params = block.parameters();
        for (i, p) in params.iter().enumerate() {
            assert!(!p.is_empty(), "Parameter {i} should have non-zero length");
        }
    }

    #[test]
    fn test_encoder_block_single_token() {
        let config = TransformerConfig::codebert();
        let block = EncoderBlock::new(&config, 3);
        let x = Tensor::from_vec(vec![0.2; config.hidden_size], true);
        let output = block.forward(&x, 1);
        assert_eq!(output.len(), config.hidden_size);
        let data = output.data();
        let slice = data.as_slice().unwrap();
        assert!(slice.iter().all(|v| v.is_finite()));
    }
}
