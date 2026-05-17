//! `CrossEncoder` — BERT-based relevance scorer for query/passage pairs.
//!
//! Wraps `BertEmbeddings` + `BertEncoder` with a classifier head:
//! ```text
//!   score = sigmoid(classifier(encoder(embed(tokens))[CLS]))
//! ```
//!
//! Trained models (e.g. `BAAI/bge-reranker-base`, `cross-encoder/ms-marco-MiniLM-L-6-v2`)
//! output a scalar relevance score in `[0, 1]` per (query, passage) pair.

use crate::autograd::Tensor;
use crate::models::bert::config::BertConfig;
use crate::models::bert::embeddings::BertEmbeddings;
use crate::models::bert::encoder::BertEncoder;
use crate::nn::{Linear, Module};

/// BERT cross-encoder scorer.
pub struct CrossEncoder {
    embeddings: BertEmbeddings,
    encoder: BertEncoder,
    /// Optional pooler dense layer (HuggingFace BERT classifier path goes
    /// through `bert.pooler.dense` → tanh before classification).
    pooler: Option<Linear>,
    /// Classifier head: `hidden_dim` → `num_labels` (typically 1 for
    /// regression-style relevance scoring).
    classifier: Linear,
    hidden_dim: usize,
    /// Cached `num_labels` so the loader can validate the classifier-head
    /// shape without coupling to `Linear::out_features` (GH-326).
    num_labels: usize,
}

impl CrossEncoder {
    /// Construct a cross-encoder with zero-initialized weights.
    ///
    /// `num_labels` is typically 1 (regression head) for reranking;
    /// pass 2 for binary classification heads.
    #[must_use]
    pub fn new(config: &BertConfig, num_labels: usize, with_pooler: bool) -> Self {
        let h = config.hidden_dim;
        Self {
            embeddings: BertEmbeddings::new(config),
            encoder: BertEncoder::new(config),
            pooler: if with_pooler {
                Some(Linear::new(h, h))
            } else {
                None
            },
            classifier: Linear::new(h, num_labels),
            hidden_dim: h,
            num_labels,
        }
    }

    /// Number of output labels (1 for relevance regression, 2+ for classification).
    #[must_use]
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Mutable access to the embeddings table (GH-326 weight loading).
    pub fn embeddings_mut(&mut self) -> &mut BertEmbeddings {
        &mut self.embeddings
    }

    /// Mutable access to the encoder stack (GH-326 weight loading).
    pub fn encoder_mut(&mut self) -> &mut BertEncoder {
        &mut self.encoder
    }

    /// Mutable access to the pooler if present (GH-326 weight loading).
    pub fn pooler_mut(&mut self) -> Option<&mut Linear> {
        self.pooler.as_mut()
    }

    /// Mutable access to the classifier head (GH-326 weight loading).
    pub fn classifier_mut(&mut self) -> &mut Linear {
        &mut self.classifier
    }

    /// Load all weights from a pre-trained `.apr` reader (GH-326 Phase 1).
    ///
    /// Reads embeddings + encoder layers + optional pooler + classifier
    /// head. The tensor names follow the HuggingFace BERT convention
    /// preserved through `Architecture::Bert.bert_map_name` (identity).
    ///
    /// # Errors
    ///
    /// Returns [`BertLoadError`] on the first missing tensor or shape
    /// mismatch. Future PRs may add a strict/non-strict mode to allow
    /// missing pooler weights for encoder-only checkpoints.
    pub fn load_from_reader(
        &mut self,
        reader: &crate::format::v2::AprV2Reader,
        config: &BertConfig,
    ) -> Result<(), crate::models::bert::load::BertLoadError> {
        crate::models::bert::load::load_cross_encoder_from_reader(self, reader, config)
    }

    /// Score a single (input_ids, token_type_ids) pair.
    ///
    /// Returns the raw logit `[num_labels]` (apply sigmoid for relevance probability,
    /// or softmax for multi-class probabilities — left to the caller per
    /// the trained model's convention).
    ///
    /// # Panics
    ///
    /// Same panic conditions as [`BertEmbeddings::forward`].
    #[must_use]
    pub fn forward(&self, input_ids: &[u32], token_type_ids: &[u32]) -> Tensor {
        let embeddings = self.embeddings.forward(input_ids, token_type_ids);
        let hidden = self.encoder.forward(&embeddings, None);

        // CLS pooling — take the first token's hidden state.
        let hidden_data = hidden.data();
        let cls = hidden_data[..self.hidden_dim].to_vec();
        let cls_tensor = Tensor::from_vec(cls, &[1, self.hidden_dim]);

        // Optional pooler (tanh squashed dense layer, BERT classification head pattern).
        let pooled = if let Some(pooler) = &self.pooler {
            let dense = pooler.forward(&cls_tensor);
            tanh(&dense)
        } else {
            cls_tensor
        };

        self.classifier.forward(&pooled)
    }

    /// Score a single query/passage pair after tokenization.
    ///
    /// The caller is responsible for tokenizing
    /// `[CLS] query [SEP] passage [SEP]` and providing the matching
    /// `token_type_ids` (0 for query, 1 for passage).
    ///
    /// Applies sigmoid to the raw logit for relevance interpretation.
    #[must_use]
    pub fn score(&self, input_ids: &[u32], token_type_ids: &[u32]) -> f32 {
        let logit_tensor = self.forward(input_ids, token_type_ids);
        let logit = logit_tensor.data()[0];
        sigmoid(logit)
    }
}

/// Sigmoid activation for relevance probability conversion.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Per-element tanh applied to a tensor (used by the BERT pooler).
fn tanh(x: &Tensor) -> Tensor {
    let data: Vec<f32> = x.data().iter().map(|v| v.tanh()).collect();
    Tensor::from_vec(data, x.shape())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cross_encoder_returns_scalar_logit() {
        let config = BertConfig::minilm_l6();
        let model = CrossEncoder::new(&config, 1, true);
        let input_ids = vec![101u32, 2024, 102, 3456, 102];
        let token_type_ids = vec![0u32, 0, 0, 1, 1];
        let out = model.forward(&input_ids, &token_type_ids);
        assert_eq!(out.shape(), &[1, 1]);
    }

    #[test]
    fn cross_encoder_score_returns_finite() {
        let config = BertConfig::minilm_l6();
        let model = CrossEncoder::new(&config, 1, true);
        let input_ids = vec![101u32, 2024, 102, 3456, 102];
        let token_type_ids = vec![0u32, 0, 0, 1, 1];
        let score = model.score(&input_ids, &token_type_ids);
        // With zero weights the logit is 0 → sigmoid(0) = 0.5
        assert!(score.is_finite());
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn cross_encoder_without_pooler() {
        let config = BertConfig::minilm_l6();
        let model = CrossEncoder::new(&config, 1, false);
        let input_ids = vec![101u32, 2024, 102];
        let token_type_ids = vec![0u32, 0, 0];
        let out = model.forward(&input_ids, &token_type_ids);
        assert_eq!(out.shape(), &[1, 1]);
    }

    #[test]
    fn cross_encoder_num_labels_dimension() {
        // Some cross-encoders use 2-label classification heads.
        let config = BertConfig::minilm_l6();
        let model = CrossEncoder::new(&config, 2, false);
        let input_ids = vec![101u32, 2024, 102];
        let token_type_ids = vec![0u32, 0, 0];
        let out = model.forward(&input_ids, &token_type_ids);
        assert_eq!(out.shape(), &[1, 2]);
    }
}
