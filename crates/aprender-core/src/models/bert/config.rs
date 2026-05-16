//! `BertConfig` — hyperparameters for BERT encoder + cross-encoder.
//!
//! Defaults match `bert-base-uncased` (110M params). Override for distilled
//! variants like MiniLM-L-6 (22M, hidden_dim=384) or bge-reranker-base.

/// BERT model configuration.
///
/// Maps to the relevant fields of HuggingFace `BertConfig`.
#[derive(Debug, Clone, PartialEq)]
pub struct BertConfig {
    /// Hidden dimension of each token (e.g. 768 for base, 384 for MiniLM-L-6).
    pub hidden_dim: usize,
    /// Number of encoder layers (e.g. 12 for base, 6 for MiniLM-L-6).
    pub num_layers: usize,
    /// Number of attention heads. `hidden_dim` must divide evenly by this.
    pub num_heads: usize,
    /// FFN intermediate dimension (typically 4 × `hidden_dim`).
    pub intermediate_dim: usize,
    /// Token vocabulary size (e.g. 30522 for `bert-base-uncased`).
    pub vocab_size: usize,
    /// Maximum sequence length supported by position embeddings.
    pub max_position_embeddings: usize,
    /// Token-type vocabulary size (typically 2 for `[A]` / `[B]` segments).
    pub type_vocab_size: usize,
    /// LayerNorm epsilon (HuggingFace default 1e-12).
    pub layer_norm_eps: f32,
    /// Pad token id (typically 0). Used for attention masking.
    pub pad_token_id: u32,
}

impl Default for BertConfig {
    /// `bert-base-uncased` defaults.
    fn default() -> Self {
        Self {
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            intermediate_dim: 3072,
            vocab_size: 30522,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }
}

impl BertConfig {
    /// Compute the per-head dimension. `hidden_dim` must be divisible by `num_heads`.
    #[must_use]
    pub const fn head_dim(&self) -> usize {
        self.hidden_dim / self.num_heads
    }

    /// MiniLM-L-6 preset (22M params, 384 hidden, 12 heads, 6 layers).
    #[must_use]
    pub fn minilm_l6() -> Self {
        Self {
            hidden_dim: 384,
            num_layers: 6,
            num_heads: 12,
            intermediate_dim: 1536,
            vocab_size: 30522,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bert_base_default_hyperparams() {
        let c = BertConfig::default();
        assert_eq!(c.hidden_dim, 768);
        assert_eq!(c.num_layers, 12);
        assert_eq!(c.num_heads, 12);
        assert_eq!(c.intermediate_dim, 3072);
        assert_eq!(c.head_dim(), 64);
    }

    #[test]
    fn minilm_l6_preset() {
        let c = BertConfig::minilm_l6();
        assert_eq!(c.hidden_dim, 384);
        assert_eq!(c.num_layers, 6);
        assert_eq!(c.head_dim(), 32);
    }

    #[test]
    fn head_dim_divides_evenly() {
        let c = BertConfig::default();
        assert_eq!(c.head_dim() * c.num_heads, c.hidden_dim);
    }
}
