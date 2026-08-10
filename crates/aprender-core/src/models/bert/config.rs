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

    /// Check the structural invariants that model construction **asserts**.
    ///
    /// `MultiHeadAttention::new` (reached via `BertEncoder::new` /
    /// `CrossEncoder::new`) asserts that `hidden_dim` is a multiple of
    /// `num_heads`, and `head_dim()` divides by `num_heads`. Callers that
    /// build a config from untrusted input (CLI overrides, config files) must
    /// call this first and surface the error, instead of letting the assert
    /// abort the process.
    ///
    /// # Errors
    ///
    /// Returns `BertConfigError` naming the offending field when
    /// `hidden_dim` or `num_heads` is zero, or when `hidden_dim` is not
    /// divisible by `num_heads`.
    pub fn validate(&self) -> Result<(), BertConfigError> {
        if self.hidden_dim == 0 {
            return Err(BertConfigError {
                field: "hidden_dim".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if self.num_heads == 0 {
            return Err(BertConfigError {
                field: "num_heads".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        if !self.hidden_dim.is_multiple_of(self.num_heads) {
            return Err(BertConfigError {
                field: "num_heads".to_string(),
                reason: format!(
                    "hidden_dim ({}) is not divisible by num_heads ({})",
                    self.hidden_dim, self.num_heads
                ),
            });
        }
        Ok(())
    }

    /// Check a token-id batch against this config before a forward pass.
    ///
    /// `BertEmbeddings::forward` indexes the word/position/token-type tables
    /// with the raw ids and slices them unchecked, so an id at or above the
    /// corresponding table size aborts the process with a slice-range panic.
    /// This is the fallible pre-check for any caller feeding user-supplied
    /// ids.
    ///
    /// # Errors
    ///
    /// Returns `BertConfigError` for a length mismatch, a sequence longer
    /// than `max_position_embeddings`, an input id `>= vocab_size`, or a
    /// token-type id `>= type_vocab_size`.
    pub fn validate_ids(
        &self,
        input_ids: &[u32],
        token_type_ids: &[u32],
    ) -> Result<(), BertConfigError> {
        if input_ids.len() != token_type_ids.len() {
            return Err(BertConfigError {
                field: "token_type_ids".to_string(),
                reason: format!(
                    "length {} does not match input_ids length {}",
                    token_type_ids.len(),
                    input_ids.len()
                ),
            });
        }
        if input_ids.len() > self.max_position_embeddings {
            return Err(BertConfigError {
                field: "input_ids".to_string(),
                reason: format!(
                    "sequence length {} exceeds max_position_embeddings {}",
                    input_ids.len(),
                    self.max_position_embeddings
                ),
            });
        }
        if let Some((i, &id)) = input_ids
            .iter()
            .enumerate()
            .find(|(_, &id)| id as usize >= self.vocab_size)
        {
            return Err(BertConfigError {
                field: format!("input_ids[{i}]"),
                reason: format!(
                    "token id {id} is out of range for vocab_size {}",
                    self.vocab_size
                ),
            });
        }
        if let Some((i, &id)) = token_type_ids
            .iter()
            .enumerate()
            .find(|(_, &id)| id as usize >= self.type_vocab_size)
        {
            return Err(BertConfigError {
                field: format!("token_type_ids[{i}]"),
                reason: format!(
                    "token-type id {id} is out of range for type_vocab_size {}",
                    self.type_vocab_size
                ),
            });
        }
        Ok(())
    }
}

/// Error returned when a `BertConfig` — or a token-id batch measured against
/// one — violates an invariant that the forward path would otherwise hit as a
/// panic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BertConfigError {
    /// Config field or input position at fault (e.g. `num_heads`, `input_ids[3]`).
    pub field: String,
    /// One-line description of what went wrong.
    pub reason: String,
}

impl std::fmt::Display for BertConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.field, self.reason)
    }
}

impl std::error::Error for BertConfigError {}

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

    /// `validate` accepts the shipped presets — the guard must not reject
    /// working configs.
    #[test]
    fn validate_accepts_presets() {
        BertConfig::default()
            .validate()
            .expect("bert-base is valid");
        BertConfig::minilm_l6()
            .validate()
            .expect("MiniLM-L-6 is valid");
    }

    /// Regression: `apr rerank --hidden-dim 999` aborted with
    /// `embed_dim (999) must be divisible by num_heads (12)` raised by
    /// `MultiHeadAttention::new`. `validate` must reject the config first
    /// and name the offending flag.
    #[test]
    fn validate_rejects_indivisible_hidden_dim() {
        let c = BertConfig {
            hidden_dim: 999,
            ..BertConfig::minilm_l6()
        };
        let err = c.validate().expect_err("999 % 12 != 0 must be rejected");
        assert_eq!(err.field, "num_heads");
        assert!(
            err.reason.contains("999") && err.reason.contains("12"),
            "{err}"
        );
    }

    /// Same defect through the other flag: `--num-heads 7` against the
    /// model's own hidden_dim 384.
    #[test]
    fn validate_rejects_indivisible_num_heads() {
        let c = BertConfig {
            num_heads: 7,
            ..BertConfig::minilm_l6()
        };
        let err = c.validate().expect_err("384 % 7 != 0 must be rejected");
        assert_eq!(err.field, "num_heads");
    }

    #[test]
    fn validate_rejects_zero_dims() {
        let zero_heads = BertConfig {
            num_heads: 0,
            ..BertConfig::minilm_l6()
        };
        assert_eq!(
            zero_heads
                .validate()
                .expect_err("num_heads 0 must be rejected")
                .field,
            "num_heads"
        );
        let zero_hidden = BertConfig {
            hidden_dim: 0,
            ..BertConfig::minilm_l6()
        };
        assert_eq!(
            zero_hidden
                .validate()
                .expect_err("hidden_dim 0 must be rejected")
                .field,
            "hidden_dim"
        );
    }

    /// `validate_ids` accepts a well-formed `[CLS] q [SEP] p [SEP]` pair.
    #[test]
    fn validate_ids_accepts_well_formed_pair() {
        let c = BertConfig::minilm_l6();
        c.validate_ids(&[101, 2024, 102, 3456, 102], &[0, 0, 0, 1, 1])
            .expect("in-range ids are valid");
    }

    /// Regression: `apr rerank --input-ids 101,999999,102` aborted with
    /// `range start index 383999616 out of range for slice of length …`
    /// from the unchecked slice in `BertEmbeddings::forward`.
    #[test]
    fn validate_ids_rejects_token_id_at_or_above_vocab_size() {
        let c = BertConfig::minilm_l6();
        let err = c
            .validate_ids(&[101, 999_999, 102], &[0, 0, 0])
            .expect_err("999999 >= 30522 must be rejected");
        assert_eq!(err.field, "input_ids[1]");
        assert!(
            err.reason.contains("999999") && err.reason.contains("30522"),
            "{err}"
        );
        // Boundary: exactly vocab_size is out of range, vocab_size-1 is in.
        assert!(c.validate_ids(&[30522], &[0]).is_err());
        c.validate_ids(&[30521], &[0]).expect("last id is in range");
    }

    /// Same panic class through `token_type_ids` (table has only
    /// `type_vocab_size` rows).
    #[test]
    fn validate_ids_rejects_token_type_id_out_of_range() {
        let c = BertConfig::minilm_l6();
        let err = c
            .validate_ids(&[101, 2024, 102], &[0, 7, 0])
            .expect_err("token-type 7 >= 2 must be rejected");
        assert_eq!(err.field, "token_type_ids[1]");
        c.validate_ids(&[101], &[1]).expect("type id 1 is in range");
    }

    /// Regression: a 600-token pair aborted with `sequence length 600
    /// exceeds max_position_embeddings 512`.
    #[test]
    fn validate_ids_rejects_sequence_longer_than_position_table() {
        let c = BertConfig::minilm_l6();
        let ids = vec![101u32; 600];
        let tt = vec![0u32; 600];
        let err = c
            .validate_ids(&ids, &tt)
            .expect_err("600 > 512 must be rejected");
        assert_eq!(err.field, "input_ids");
        assert!(
            err.reason.contains("600") && err.reason.contains("512"),
            "{err}"
        );
        // Boundary: exactly max_position_embeddings is accepted.
        c.validate_ids(&vec![101u32; 512], &vec![0u32; 512])
            .expect("512 fits the position table");
    }

    #[test]
    fn validate_ids_rejects_length_mismatch() {
        let c = BertConfig::minilm_l6();
        let err = c
            .validate_ids(&[101, 2024, 102], &[0, 0])
            .expect_err("length mismatch must be rejected");
        assert_eq!(err.field, "token_type_ids");
    }

    /// The whole point of the guard: everything `validate` + `validate_ids`
    /// accept must survive a real `CrossEncoder` construction and forward
    /// pass without panicking. Proves the guard is not merely a string check
    /// that happens to be stricter/looser than the assert it replaces.
    #[test]
    fn accepted_config_and_ids_survive_a_real_forward_pass() {
        use crate::models::bert::CrossEncoder;
        let c = BertConfig {
            hidden_dim: 32,
            num_layers: 1,
            num_heads: 4,
            intermediate_dim: 64,
            vocab_size: 64,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            ..BertConfig::minilm_l6()
        };
        c.validate().expect("config is valid");
        let input_ids = [1u32, 63, 2];
        let token_type_ids = [0u32, 1, 1];
        c.validate_ids(&input_ids, &token_type_ids)
            .expect("ids are valid");
        let ce = CrossEncoder::new(&c, 1, true);
        let out = ce.forward(&input_ids, &token_type_ids);
        assert_eq!(out.data().len(), 1);
    }
}
