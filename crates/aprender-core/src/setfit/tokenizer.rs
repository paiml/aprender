//! The MiniLM tokenizer boundary (ENC-02, D-07).
//!
//! Wraps the pinned HuggingFace `tokenizers` WordPiece tokenizer and returns a
//! [`SentenceBatch`]: ordered **canonical** vocabulary ids, token type ids, an
//! attention mask, per-input truncation facts, per-input provenance, and the
//! sha256 of the tokenizer that produced them.
//!
//! # Canonical ids stay canonical
//!
//! `input_ids` are always the tokenizer's real vocabulary ids. The slice
//! fixtures' `orig_to_slice` remap is carried by the *import* and applied inside
//! the encoder at gather time — a `SentenceBatch` is never rewritten to fit a
//! slice. Rewriting it would make tokenizer identity a function of which model
//! happened to consume the batch.
//!
//! # Truncation-fact strategy
//!
//! STRATEGY IN FORCE: **one user-facing call; the batch is tokenized once, and
//! rows that were actually truncated are re-tokenized once more without
//! truncation to recover their true length.** The extra pass is restricted to
//! truncated rows and is therefore empty for every non-truncating input.
//!
//! This is recorded rather than glossed because the plan's preferred
//! single-pass derivation — `original_len = ids.len() + sum(overflowing.len())`
//! — does **not** hold for the pinned `tokenizers` 0.23.1 with padding enabled.
//! Measured on the frozen `truncation_long` fixture (482 real tokens):
//! post-processing adds `[CLS]`/`[SEP]` to *each* overflow chunk and padding
//! then pads each chunk to the batch-longest width, so the naive sum reports
//! **512**, not 482. Deriving the count from a formula that is off by the
//! specials-and-padding of every overflow chunk would make ENC-02's reported
//! `original_len` quietly wrong exactly on the inputs it exists to describe.
//! ENC-02's guarantee is therefore stated as "one user-facing call", not
//! "one tokenizer pass".
//!
//! # Sealing (D-08 + W1)
//!
//! [`MiniLmTokenizer::from_bytes`] is `pub(crate)`: out-of-crate callers obtain a
//! tokenizer only through `SetFitMiniLm`, which builds the tokenizer and the
//! encoder together from one source, so a mismatched pair is not constructible.
//! [`SentenceBatch`]'s fields are `pub(crate)` with read-only accessors, so
//! out-of-crate code can neither forge a batch stamped with a borrowed
//! `tokenizer_sha256` nor mutate the ids of a batch it legitimately received.

use sha2::{Digest, Sha256};

use super::error::SetFitError;

/// Maximum sequence length of the pinned sentence-transformers configuration.
///
/// `sentence_bert_config.json` for all-MiniLM-L6-v2 sets `max_seq_length: 256`;
/// the frozen fixtures were generated at that bound.
pub const MAX_SEQUENCE_LENGTH: usize = 256;

/// What truncation did to one input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TruncationFact {
    /// Whether the input was longer than [`MAX_SEQUENCE_LENGTH`].
    pub truncated: bool,
    /// Token count of the input **before** truncation, special tokens included.
    pub original_len: usize,
}

/// Which input produced a row, and what that input was.
///
/// Carried so a downstream embedding can be traced back to the exact bytes that
/// produced it without retaining the text itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputProvenance {
    /// Position of this input in the `texts` slice passed to `encode_batch`.
    pub index: usize,
    /// Lowercase-hex sha256 of the input text's UTF-8 bytes.
    pub text_sha256: String,
}

/// A tokenized batch, ready for the encoder.
///
/// # Read-only outside the crate (W1)
///
/// Every field is `pub(crate)`. In-crate code (the encoder, 01-06) reads the
/// fields directly; out-of-crate code reads through the accessors below and can
/// neither build a `SentenceBatch` literal nor mutate one it received.
///
/// This is what makes the encoder's `tokenizer_sha256` equality check
/// meaningful. With `pub` fields the check would compare a value the caller
/// controls: out-of-crate code could hand-build a batch stamped with a
/// legitimate hash, or take a batch from `SetFitMiniLm::tokenize()` and mutate
/// `input_ids` while leaving the hash intact. Both defeat the check while the
/// D-08 constructor seal remains formally intact, so the seal is enforced at the
/// data layer too.
///
/// `#[non_exhaustive]` is deliberately NOT used: it is redundant once the fields
/// are `pub(crate)`, which already blocks out-of-crate struct-literal
/// construction and exhaustive destructuring, and it is a no-op within the
/// defining crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SentenceBatch {
    /// Row-major `[batch * seq]` CANONICAL vocabulary ids.
    pub(crate) input_ids: Vec<u32>,
    /// Row-major `[batch * seq]` token type ids.
    pub(crate) token_type_ids: Vec<u32>,
    /// Row-major `[batch * seq]` mask: `1` keep, `0` padding.
    pub(crate) attention_mask: Vec<u8>,
    /// Number of inputs in the batch.
    pub(crate) batch: usize,
    /// Padded sequence length (longest row in the batch, capped at the max).
    pub(crate) seq: usize,
    /// Per-input truncation facts, in input order.
    pub(crate) truncation: Vec<TruncationFact>,
    /// Per-input provenance, in input order.
    pub(crate) provenance: Vec<InputProvenance>,
    /// Sha256 of the tokenizer that produced this batch (D-08 defense in depth).
    pub(crate) tokenizer_sha256: String,
}

impl SentenceBatch {
    /// Row-major `[batch * seq]` canonical vocabulary ids.
    #[must_use]
    pub fn input_ids(&self) -> &[u32] {
        &self.input_ids
    }

    /// Row-major `[batch * seq]` token type ids.
    #[must_use]
    pub fn token_type_ids(&self) -> &[u32] {
        &self.token_type_ids
    }

    /// Row-major `[batch * seq]` attention mask (`1` keep, `0` padding).
    #[must_use]
    pub fn attention_mask(&self) -> &[u8] {
        &self.attention_mask
    }

    /// Number of inputs in the batch.
    #[must_use]
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Padded sequence length.
    #[must_use]
    pub fn seq(&self) -> usize {
        self.seq
    }

    /// Per-input truncation facts, in input order.
    #[must_use]
    pub fn truncation(&self) -> &[TruncationFact] {
        &self.truncation
    }

    /// Per-input provenance, in input order.
    #[must_use]
    pub fn provenance(&self) -> &[InputProvenance] {
        &self.provenance
    }

    /// Sha256 of the tokenizer that produced this batch.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }
}

/// The padding mode the tokenizer is configured with.
///
/// A NAMED constant rather than a string a downstream artifact writer invents,
/// for the same reason [`MAX_SEQUENCE_LENGTH`] is one: a persistence layer that
/// records "batch_longest" from its own literal is recording what it believes,
/// and the belief and the configuration can drift apart without either side
/// becoming red. This is the single definition, and `with_padding` below is the
/// single place it is applied.
pub const PADDING_MODE: &str = "batch_longest";

/// The pinned MiniLM WordPiece tokenizer.
pub struct MiniLmTokenizer {
    /// Configured with truncation at [`MAX_SEQUENCE_LENGTH`] and
    /// batch-longest padding.
    inner: tokenizers::Tokenizer,
    /// Same vocabulary, no truncation and no padding. Used only to recover the
    /// true length of rows that the truncating pass actually cut.
    untruncated: tokenizers::Tokenizer,
    /// The exact `tokenizer.json` bytes this tokenizer was built from.
    ///
    /// RETAINED, not merely hashed (plan 03-08). A persistence artifact that
    /// carries only [`Self::tokenizer_sha256`] can *detect* a substituted
    /// tokenizer and cannot *rebuild* the right one, so a "reload" from such an
    /// artifact could never re-encode a single string. The hash stays as the
    /// identity check; these bytes are what makes the reload real.
    source_bytes: Vec<u8>,
    /// Lowercase-hex sha256 of the bytes this tokenizer was built from.
    tokenizer_sha256: String,
}

impl std::fmt::Debug for MiniLmTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniLmTokenizer")
            .field("tokenizer_sha256", &self.tokenizer_sha256)
            .field("source_bytes_len", &self.source_bytes.len())
            .field("max_sequence_length", &MAX_SEQUENCE_LENGTH)
            .finish()
    }
}

/// Lowercase-hex sha256 of a byte slice.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    // `GenericArray`'s LowerHex renders the digest directly — no per-byte loop and
    // no infallible-Result discard to justify.
    format!("{:x}", Sha256::digest(bytes))
}

impl MiniLmTokenizer {
    /// Build a tokenizer from `tokenizer.json` bytes.
    ///
    /// SEALED (D-08): `pub(crate)`. Out-of-crate callers reach a tokenizer only
    /// via `SetFitMiniLm`, which constructs the tokenizer and the encoder
    /// together — so a mismatched pair is not constructible.
    ///
    /// # Errors
    ///
    /// [`SetFitError::TokenizerLoad`] if the bytes are not a parseable
    /// `tokenizers` serialization, or if truncation/padding cannot be
    /// configured on it.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, SetFitError> {
        let mut inner =
            tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| SetFitError::TokenizerLoad {
                reason: e.to_string(),
            })?;

        // Truncation and padding come from the tokenizers API, never from
        // hand-rolled slicing: the library reserves room for the post-processor's
        // special tokens, so the padded row is exactly MAX_SEQUENCE_LENGTH with
        // [CLS]/[SEP] intact. Hand-rolling would silently drop [SEP].
        inner
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQUENCE_LENGTH,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                stride: 0,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| SetFitError::TokenizerLoad {
                reason: format!("cannot configure truncation: {e}"),
            })?;
        // `PADDING_MODE` names exactly this strategy; the two are adjacent so a
        // change to one is a change a reader of the other cannot miss.
        debug_assert_eq!(PADDING_MODE, "batch_longest");
        inner.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: 0,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        // Second view over the SAME bytes with neither truncation nor padding.
        // Used only to recover the true length of rows the first pass cut; see
        // the truncation-fact strategy in the module docs.
        let mut untruncated =
            tokenizers::Tokenizer::from_bytes(bytes).map_err(|e| SetFitError::TokenizerLoad {
                reason: e.to_string(),
            })?;
        untruncated
            .with_truncation(None)
            .map_err(|e| SetFitError::TokenizerLoad {
                reason: format!("cannot clear truncation: {e}"),
            })?;
        untruncated.with_padding(None);

        Ok(Self {
            inner,
            untruncated,
            source_bytes: bytes.to_vec(),
            tokenizer_sha256: sha256_hex(bytes),
        })
    }

    /// Sha256 of the bytes this tokenizer was built from.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// The exact `tokenizer.json` bytes this tokenizer was built from.
    ///
    /// The hash and these bytes are set from the same argument in the same
    /// expression, so they cannot describe different tokenizers; the test
    /// `tokenizer_bytes_hash_agrees_with_the_recorded_sha256` re-derives the
    /// digest from what this returns so the two can never drift apart silently.
    #[must_use]
    pub fn source_bytes(&self) -> &[u8] {
        &self.source_bytes
    }

    /// Tokenize a batch of texts.
    ///
    /// Truncates at [`MAX_SEQUENCE_LENGTH`] and pads to the longest row in the
    /// batch. Ids are canonical.
    ///
    /// # Errors
    ///
    /// [`SetFitError::BatchInvalid`] if `texts` is empty or the tokenizer
    /// returns a malformed encoding; [`SetFitError::TokenizerLoad`] if the
    /// underlying tokenizer fails on an input.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<SentenceBatch, SetFitError> {
        if texts.is_empty() {
            return Err(SetFitError::BatchInvalid {
                reason: "empty text list: a batch needs at least one input".to_string(),
            });
        }

        let encodings = self.inner.encode_batch(texts.to_vec(), true).map_err(|e| {
            SetFitError::TokenizerLoad {
                reason: format!("encode_batch failed: {e}"),
            }
        })?;

        if encodings.len() != texts.len() {
            return Err(SetFitError::BatchInvalid {
                reason: format!(
                    "tokenizer returned {} encodings for {} inputs",
                    encodings.len(),
                    texts.len()
                ),
            });
        }

        let batch = texts.len();
        let seq = encodings[0].get_ids().len();
        if seq == 0 {
            return Err(SetFitError::BatchInvalid {
                reason: "tokenizer produced a zero-length row".to_string(),
            });
        }

        // BatchLongest padding must make every row the same width. Checked
        // rather than assumed: an unequal row would silently corrupt the
        // row-major flattening below.
        for (i, e) in encodings.iter().enumerate() {
            if e.get_ids().len() != seq {
                return Err(SetFitError::BatchInvalid {
                    reason: format!(
                        "row {i} has length {} but row 0 has {seq}; padding did not apply",
                        e.get_ids().len()
                    ),
                });
            }
        }

        // Which rows were actually cut. `get_overflowing()` is non-empty exactly
        // when truncation removed content.
        let cut: Vec<usize> = encodings
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.get_overflowing().is_empty())
            .map(|(i, _)| i)
            .collect();

        // Recover true lengths for cut rows only. The naive single-pass formula
        // `ids.len() + sum(overflowing.len())` is WRONG here: measured against
        // the frozen `truncation_long` fixture it reports 512 for a 482-token
        // input, because post-processing adds [CLS]/[SEP] to each overflow chunk
        // and padding then widens each chunk to the batch-longest width.
        //
        // For a row that was NOT cut, the true length is the number of kept
        // positions, i.e. the attention-mask population count. It is emphatically
        // NOT `get_ids().len()`, which is the batch-longest PADDED width — that
        // reads 20 for a 5-token input in the frozen `mixed_length_pair` case.
        let mut original_lens: Vec<usize> = encodings
            .iter()
            .map(|e| e.get_attention_mask().iter().filter(|m| **m == 1).count())
            .collect();
        if !cut.is_empty() {
            let cut_texts: Vec<&str> = cut.iter().map(|i| texts[*i]).collect();
            let full = self
                .untruncated
                .encode_batch(cut_texts, true)
                .map_err(|e| SetFitError::TokenizerLoad {
                    reason: format!("untruncated pass failed: {e}"),
                })?;
            if full.len() != cut.len() {
                return Err(SetFitError::BatchInvalid {
                    reason: "untruncated pass returned a different row count".to_string(),
                });
            }
            for (slot, e) in cut.iter().zip(full.iter()) {
                original_lens[*slot] = e.get_ids().len();
            }
        }

        let n = batch
            .checked_mul(seq)
            .ok_or_else(|| SetFitError::BatchInvalid {
                reason: format!("batch {batch} x seq {seq} overflows usize"),
            })?;
        let mut input_ids = Vec::with_capacity(n);
        let mut token_type_ids = Vec::with_capacity(n);
        let mut attention_mask = Vec::with_capacity(n);
        let mut truncation = Vec::with_capacity(batch);
        let mut provenance = Vec::with_capacity(batch);

        for (i, e) in encodings.iter().enumerate() {
            input_ids.extend_from_slice(e.get_ids());
            token_type_ids.extend_from_slice(e.get_type_ids());
            // tokenizers reports the mask as u32; narrow it explicitly and reject
            // anything that is not 0/1 rather than truncating a stray value into
            // a plausible-looking keep.
            for (pos, m) in e.get_attention_mask().iter().enumerate() {
                let bit = u8::try_from(*m).map_err(|_| SetFitError::BatchInvalid {
                    reason: format!("attention mask value {m} at row {i} position {pos}"),
                })?;
                if bit > 1 {
                    return Err(SetFitError::BatchInvalid {
                        reason: format!(
                            "non-binary attention mask value {bit} at row {i} position {pos}"
                        ),
                    });
                }
                attention_mask.push(bit);
            }
            truncation.push(TruncationFact {
                truncated: !e.get_overflowing().is_empty(),
                original_len: original_lens[i],
            });
            provenance.push(InputProvenance {
                index: i,
                text_sha256: sha256_hex(texts[i].as_bytes()),
            });
        }

        Ok(SentenceBatch {
            input_ids,
            token_type_ids,
            attention_mask,
            batch,
            seq,
            truncation,
            provenance,
            tokenizer_sha256: self.tokenizer_sha256.clone(),
        })
    }
}

#[cfg(all(test, feature = "setfit"))]
#[path = "tokenizer_tests.rs"]
mod tokenizer_tests;
