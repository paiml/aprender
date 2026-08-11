//! `BertEmbeddings` — word + position + token_type embeddings → LayerNorm.
//!
//! HuggingFace reference:
//! `bert.embeddings.{word,position,token_type}_embeddings.weight`
//! `bert.embeddings.LayerNorm.{weight,bias}`

use crate::autograd::Tensor;
use crate::models::bert::config::BertConfig;
use crate::nn::{LayerNorm, Module};

/// BERT input embedding layer.
///
/// Takes (input_ids, token_type_ids) and returns per-token hidden states:
/// `hidden = LayerNorm(word_embed[ids] + position_embed[0..seq_len] + token_type_embed[type_ids])`
pub struct BertEmbeddings {
    /// Word embedding table: `[vocab_size, hidden_dim]`.
    pub(crate) word_embeddings: Tensor,
    /// Position embedding table: `[max_position_embeddings, hidden_dim]`.
    pub(crate) position_embeddings: Tensor,
    /// Token-type embedding table: `[type_vocab_size, hidden_dim]`.
    pub(crate) token_type_embeddings: Tensor,
    /// Final LayerNorm applied to summed embeddings.
    pub(crate) layer_norm: LayerNorm,
    /// Cached hidden_dim for slicing.
    hidden_dim: usize,
    /// Cached max_position_embeddings for bound check.
    max_position_embeddings: usize,
}

impl BertEmbeddings {
    /// Construct with zero-initialized weights of the right shape.
    /// Real inference replaces these via direct field assignment after load.
    #[must_use]
    pub fn new(config: &BertConfig) -> Self {
        let h = config.hidden_dim;
        let we = vec![0.0; config.vocab_size * h];
        let pe = vec![0.0; config.max_position_embeddings * h];
        let te = vec![0.0; config.type_vocab_size * h];
        Self {
            word_embeddings: Tensor::from_vec(we, &[config.vocab_size, h]),
            position_embeddings: Tensor::from_vec(pe, &[config.max_position_embeddings, h]),
            token_type_embeddings: Tensor::from_vec(te, &[config.type_vocab_size, h]),
            layer_norm: LayerNorm::with_eps(&[h], config.layer_norm_eps),
            hidden_dim: h,
            max_position_embeddings: config.max_position_embeddings,
        }
    }

    /// Load embeddings from an APR v2 reader (GH-326 Phase 6 — embed-only).
    ///
    /// # Errors
    ///
    /// Returns [`BertLoadError`](crate::models::bert::load::BertLoadError)
    /// on the first missing tensor or shape mismatch.
    pub fn load_from_reader(
        &mut self,
        reader: &crate::format::v2::AprV2Reader,
        config: &crate::models::bert::BertConfig,
    ) -> Result<(), crate::models::bert::load::BertLoadError> {
        crate::models::bert::load::load_embeddings_from_reader(self, reader, config)
    }

    /// Forward pass: produces `[1, seq_len, hidden_dim]` of post-LN embeddings
    /// (batch dim is implicit batch=1 for single-pair scoring).
    ///
    /// # Panics
    ///
    /// Panics if `input_ids.len() != token_type_ids.len()`, if
    /// `input_ids.len() > max_position_embeddings`, or if any id is at or
    /// above its table size (the embedding rows are sliced unchecked).
    /// Callers handling untrusted ids must pre-check with
    /// [`BertConfig::validate_ids`](crate::models::bert::BertConfig::validate_ids)
    /// and surface the error rather than aborting the process.
    #[must_use]
    pub fn forward(&self, input_ids: &[u32], token_type_ids: &[u32]) -> Tensor {
        assert_eq!(
            input_ids.len(),
            token_type_ids.len(),
            "input_ids and token_type_ids must have the same length"
        );
        assert!(
            input_ids.len() <= self.max_position_embeddings,
            "sequence length {} exceeds max_position_embeddings {}",
            input_ids.len(),
            self.max_position_embeddings
        );

        let seq_len = input_ids.len();
        let h = self.hidden_dim;
        let mut summed = vec![0.0f32; seq_len * h];

        let we_data = self.word_embeddings.data();
        let pe_data = self.position_embeddings.data();
        let te_data = self.token_type_embeddings.data();

        for (i, (&wid, &tid)) in input_ids.iter().zip(token_type_ids).enumerate() {
            let dst = &mut summed[i * h..(i + 1) * h];
            let w_row = &we_data[wid as usize * h..(wid as usize + 1) * h];
            let p_row = &pe_data[i * h..(i + 1) * h];
            let t_row = &te_data[tid as usize * h..(tid as usize + 1) * h];
            for j in 0..h {
                dst[j] = w_row[j] + p_row[j] + t_row[j];
            }
        }

        // Reshape to [1, seq_len, hidden_dim] so the encoder/MHA stack
        // (which expects 3D batched input) can consume it directly.
        let summed_tensor = Tensor::from_vec(summed, &[1, seq_len, h]);
        self.layer_norm.forward(&summed_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeddings_shape_correct() {
        let config = BertConfig::minilm_l6();
        let emb = BertEmbeddings::new(&config);
        let input_ids = vec![101u32, 2024, 102];
        let token_type_ids = vec![0u32, 0, 0];
        let out = emb.forward(&input_ids, &token_type_ids);
        assert_eq!(out.shape(), &[1, 3, 384]);
    }

    #[test]
    #[should_panic(expected = "must have the same length")]
    fn embeddings_mismatched_ids_panics() {
        let config = BertConfig::minilm_l6();
        let emb = BertEmbeddings::new(&config);
        emb.forward(&[101u32, 2024], &[0u32]);
    }

    #[test]
    fn embeddings_handles_paired_input() {
        // [CLS] q [SEP] p [SEP] — cross-encoder layout
        let config = BertConfig::minilm_l6();
        let emb = BertEmbeddings::new(&config);
        let input_ids = vec![101u32, 2024, 102, 3456, 102];
        // token_type 0 for query, 1 for passage
        let token_type_ids = vec![0u32, 0, 0, 1, 1];
        let out = emb.forward(&input_ids, &token_type_ids);
        assert_eq!(out.shape(), &[1, 5, 384]);
    }

    /// The hazard `BertConfig::validate_ids` exists to guard: the embedding
    /// rows are sliced with the raw id, so an id past the end of the table
    /// aborts the process. Pinning it here keeps the guard load-bearing — if
    /// this ever stops panicking, the pre-check is no longer the only thing
    /// standing between user input and a slice-range abort.
    #[test]
    #[should_panic(expected = "out of range for slice")]
    fn embeddings_out_of_range_id_panics_without_a_pre_check() {
        let config = BertConfig::minilm_l6();
        let emb = BertEmbeddings::new(&config);
        let _ = emb.forward(&[101u32, 999_999, 102], &[0u32, 0, 0]);
    }

    /// The same ids, run through the pre-check first, are rejected with a
    /// recoverable error instead of aborting.
    #[test]
    fn validate_ids_rejects_what_forward_would_abort_on() {
        let config = BertConfig::minilm_l6();
        let err = config
            .validate_ids(&[101, 999_999, 102], &[0, 0, 0])
            .expect_err("id past the word-embedding table must be rejected");
        assert_eq!(err.field, "input_ids[1]");
    }
}
