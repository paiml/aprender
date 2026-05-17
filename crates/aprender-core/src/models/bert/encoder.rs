//! `BertEncoder` — stack of N `BertLayer` blocks.
//!
//! Maps to HuggingFace `bert.encoder` — the model body between embeddings
//! and the task head (pooler/classifier).

use crate::autograd::Tensor;
use crate::models::bert::config::BertConfig;
use crate::models::bert::layer::BertLayer;

/// Stack of BERT encoder layers.
pub struct BertEncoder {
    layers: Vec<BertLayer>,
}

impl BertEncoder {
    /// Construct an encoder with `num_layers` zero-initialized `BertLayer`s.
    #[must_use]
    pub fn new(config: &BertConfig) -> Self {
        let layers = (0..config.num_layers)
            .map(|_| BertLayer::new(config))
            .collect();
        Self { layers }
    }

    /// Forward pass over the entire stack.
    ///
    /// `hidden`: `[seq_len, hidden_dim]` — typically from `BertEmbeddings::forward`.
    /// `attn_mask`: optional additive mask broadcast to attention scores.
    ///
    /// Returns the final hidden states `[seq_len, hidden_dim]`.
    #[must_use]
    pub fn forward(&self, hidden: &Tensor, attn_mask: Option<&Tensor>) -> Tensor {
        let mut h = hidden.clone();
        for layer in &self.layers {
            h = layer.forward(&h, attn_mask);
        }
        h
    }

    /// Number of layers in the encoder stack.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_preserves_shape() {
        let config = BertConfig::minilm_l6();
        let encoder = BertEncoder::new(&config);
        assert_eq!(encoder.num_layers(), 6);

        let seq_len = 5;
        let h = config.hidden_dim;
        let input = Tensor::from_vec(vec![0.0; seq_len * h], &[1, seq_len, h]);
        let out = encoder.forward(&input, None);
        assert_eq!(out.shape(), &[1, seq_len, h]);
    }

    #[test]
    fn encoder_handles_bert_base_dims() {
        let config = BertConfig::default();
        let encoder = BertEncoder::new(&config);
        assert_eq!(encoder.num_layers(), 12);

        let seq_len = 4;
        let h = config.hidden_dim;
        let input = Tensor::from_vec(vec![0.0; seq_len * h], &[1, seq_len, h]);
        let out = encoder.forward(&input, None);
        assert_eq!(out.shape(), &[1, seq_len, h]);
    }
}
