//! The pinned-revision import contract (ENC-01).
//!
//! Two constructors, deliberately separate (RESEARCH Pitfall 3):
//!
//! * [`MiniLmImport::open`] — the **full pin**. Every behavior-affecting field
//!   of `config.json` must equal the pinned all-MiniLM-L6-v2 architecture, the
//!   sentence-transformers module stack must be mean-pool + L2-normalize, and
//!   the tokenizer bytes must hash to the pinned digest.
//! * [`MiniLmImport::open_slice_fixture`] — **fixtures only**, behind
//!   `conformance-fixtures`. Bypasses ONLY the equality-with-the-pin checks.
//!   Every structural, shape-consistency and finiteness check still runs.
//!
//! Keeping them separate is what stops the fixture gates and the pin gates from
//! contradicting each other. The tempting "fix" — parameterising the pin path by
//! caller-supplied dimensions — is PF-011's exact failure mode: it turns the pin
//! into a formality that agrees with whatever it is handed.
//!
//! # Unknown metadata is tolerated, on purpose
//!
//! `serde(deny_unknown_fields)` is **not** used. The real pinned `config.json`
//! carries `_name_or_path`, `gradient_checkpointing`, `initializer_range`,
//! `model_type`, `transformers_version` and `use_cache`, none of which this
//! struct models. Denying unknown fields would reject the pinned model itself
//! and turn ENC-01 into a gate that fails on the correct artifact. The security
//! property comes from validating every field that can CHANGE BEHAVIOR, not from
//! refusing to parse metadata.
//!
//! # Sealing (D-08)
//!
//! Both constructors are `pub(crate)`. Nothing outside the crate builds a
//! `MiniLmImport`; `SetFitMiniLm::from_pretrained_dir` / `from_slice_fixture`
//! (01-07) are the public entry points and they load the tokenizer from the same
//! source, so a mismatched tokenizer/encoder pair cannot be assembled.

// D32 CLOSED (01-07): the module-wide `#![allow(dead_code)]` is GONE.
//
// The history is worth keeping because it is the removal condition being met,
// not a guess. 01-05 added the allow when every constructor here became
// `pub(crate)` under the D-08 seal with no in-crate caller: ~15 findings, none
// of them a defect. 01-06 wired the encoder and MEASURED the surface down to
// exactly three — `VocabRemap::from_json_bytes`, `SliceConfig::from_json_bytes`
// and `validate_pooling` — and left the allow in place because all three were
// still reachable only from tests. `SetFitMiniLm::from_pretrained_dir` calls
// `open` (which calls `validate_pooling`) and `from_slice_fixture` calls both
// `from_json_bytes` constructors, so all three now have a library caller. Zero
// dead-code findings remain in this file, measured with
// `cargo check -p aprender-core --features setfit`.

use std::collections::HashMap;
use std::path::Path;

use crate::autograd::Tensor;
use crate::format::v2::AprV2Reader;
use crate::models::bert::config::BertConfig;
use crate::models::bert::load::{detect_bert_prefix, read_tensor};

use super::error::SetFitError;
use super::tokenizer::sha256_hex;

/// The pinned upstream revision of `sentence-transformers/all-MiniLM-L6-v2`.
///
/// Recorded as **data**, never fetched by branch name (T-1-10): a mutable ref
/// can be repointed, an immutable commit sha cannot. This value is asserted
/// against `tests/fixtures/setfit/upstream_manifest.json` by
/// `import_pin_revision_agrees_with_the_frozen_upstream_manifest`, so it cannot
/// drift from the artifact set 01-04 froze.
pub const PINNED_REVISION: &str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41";

/// Sha256 of `tokenizer.json` at [`PINNED_REVISION`].
///
/// Also asserted against `upstream_manifest.json` rather than transcribed on
/// trust.
pub const PINNED_TOKENIZER_SHA256: &str =
    "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037";

/// The pinned sentence-transformers maximum sequence length.
pub const PINNED_MAX_SEQ_LENGTH: usize = 256;

/// The only activation this crate implements for the pinned model.
///
/// `"gelu"` is the exact **erf** form (`Tensor::gelu_exact`, 01-09).
/// `"gelu_new"` / `"gelu_pytorch_tanh"` select the tanh approximation, which
/// differs from it by a measured 4.734993e-04 — two orders above the frozen
/// activation tolerance of 4.47e-06 — so they are rejected, never coerced.
pub const PINNED_ACTIVATION: &str = "gelu";

// ---------------------------------------------------------------------------
// Vocabulary remap (slice fixtures only)
// ---------------------------------------------------------------------------

/// Canonical vocabulary id <-> slice embedding row.
///
/// Deserialized from `vocab_remap.json` (01-04). The encoder applies it to
/// canonical ids at gather time; a [`super::SentenceBatch`] is never rewritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VocabRemap {
    orig_to_slice: HashMap<u32, u32>,
    slice_to_orig: Vec<u32>,
}

impl VocabRemap {
    /// Parse and fully validate a remap against the slice vocabulary size.
    ///
    /// # Errors
    ///
    /// [`SetFitError::RemapInvalid`] if the two directions disagree, if any
    /// slice row is out of range, or if the arity does not match `slice_vocab`.
    #[cfg(feature = "conformance-fixtures")]
    pub(crate) fn from_json_bytes(bytes: &[u8], slice_vocab: usize) -> Result<Self, SetFitError> {
        let wire: VocabRemapWire =
            serde_json::from_slice(bytes).map_err(|e| SetFitError::RemapInvalid {
                reason: format!("vocab_remap.json is not parseable: {e}"),
            })?;

        if wire.slice_to_orig.len() != slice_vocab {
            return Err(SetFitError::RemapInvalid {
                reason: format!(
                    "slice_to_orig has {} entries but the slice vocabulary is {slice_vocab}",
                    wire.slice_to_orig.len()
                ),
            });
        }
        if wire.orig_to_slice.len() != slice_vocab {
            return Err(SetFitError::RemapInvalid {
                reason: format!(
                    "orig_to_slice has {} entries but the slice vocabulary is {slice_vocab}",
                    wire.orig_to_slice.len()
                ),
            });
        }

        let bound = u32::try_from(slice_vocab).map_err(|_| SetFitError::RemapInvalid {
            reason: format!("slice vocabulary {slice_vocab} does not fit in u32"),
        })?;

        // Range: no slice row may address a table row that does not exist. This
        // is checked BEFORE any gather, because an out-of-range row would
        // otherwise become an out-of-bounds read or a silent zero row.
        for (canonical, slice_row) in &wire.orig_to_slice {
            if *slice_row >= bound {
                return Err(SetFitError::RemapInvalid {
                    reason: format!(
                        "canonical id {canonical} maps to slice row {slice_row}, \
                         which is outside a {slice_vocab}-row table"
                    ),
                });
            }
        }

        // Mutual inverse, both directions. Checking only one direction admits a
        // remap where two canonical ids collide onto one slice row — which would
        // silently merge two distinct tokens' embeddings.
        for (canonical, slice_row) in &wire.orig_to_slice {
            let back = wire.slice_to_orig[*slice_row as usize];
            if back != *canonical {
                return Err(SetFitError::RemapInvalid {
                    reason: format!(
                        "orig_to_slice[{canonical}] = {slice_row} but slice_to_orig[{slice_row}] \
                         = {back}; the two directions disagree"
                    ),
                });
            }
        }
        for (row, canonical) in wire.slice_to_orig.iter().enumerate() {
            match wire.orig_to_slice.get(canonical) {
                Some(back) if usize::try_from(*back).ok() == Some(row) => {}
                Some(back) => {
                    return Err(SetFitError::RemapInvalid {
                        reason: format!(
                            "slice_to_orig[{row}] = {canonical} but orig_to_slice[{canonical}] \
                             = {back}; the two directions disagree"
                        ),
                    })
                }
                None => {
                    return Err(SetFitError::RemapInvalid {
                        reason: format!(
                            "slice_to_orig[{row}] = {canonical} has no orig_to_slice entry; \
                             the two directions disagree"
                        ),
                    })
                }
            }
        }

        Ok(Self {
            orig_to_slice: wire.orig_to_slice,
            slice_to_orig: wire.slice_to_orig,
        })
    }

    /// Map a canonical vocabulary id to its slice embedding row.
    ///
    /// # Errors
    ///
    /// [`SetFitError::VocabOutOfSlice`] when the id is outside the slice
    /// closure. Returned rather than zero-filled: a zero row is
    /// indistinguishable from a legitimately zero embedding downstream.
    pub fn to_slice_row(&self, canonical: u32) -> Result<u32, SetFitError> {
        self.orig_to_slice
            .get(&canonical)
            .copied()
            .ok_or(SetFitError::VocabOutOfSlice {
                canonical_id: canonical,
            })
    }

    /// Rebuild a remap from its `slice_to_orig` table alone (plan 03-08).
    ///
    /// The reverse direction is DERIVED here rather than transported, which is
    /// what makes the two directions mutually inverse by construction instead of
    /// by a check on two independently supplied tables. The injectivity check
    /// stays, because a `slice_to_orig` carrying the same canonical id twice
    /// would silently collapse two rows into one and there is nothing about a
    /// derived inverse that prevents that.
    ///
    /// `pub(crate)`: the public door is `SetFitMiniLm::from_bundle_parts`.
    ///
    /// # Errors
    ///
    /// [`SetFitError::RemapInvalid`] if the table is empty, does not fit `u32`,
    /// or maps two rows to one canonical id.
    pub(crate) fn from_slice_to_orig(slice_to_orig: Vec<u32>) -> Result<Self, SetFitError> {
        if slice_to_orig.is_empty() {
            return Err(SetFitError::RemapInvalid {
                reason: "slice_to_orig is empty; a slice with no vocabulary cannot gather"
                    .to_string(),
            });
        }
        u32::try_from(slice_to_orig.len()).map_err(|_| SetFitError::RemapInvalid {
            reason: format!(
                "slice vocabulary {} does not fit in u32",
                slice_to_orig.len()
            ),
        })?;

        let mut orig_to_slice: HashMap<u32, u32> = HashMap::with_capacity(slice_to_orig.len());
        for (row, canonical) in slice_to_orig.iter().enumerate() {
            // `row` is bounded by the length check above, so the cast is exact.
            let row_u32 = u32::try_from(row).map_err(|_| SetFitError::RemapInvalid {
                reason: format!("slice row {row} does not fit in u32"),
            })?;
            if let Some(first) = orig_to_slice.insert(*canonical, row_u32) {
                return Err(SetFitError::RemapInvalid {
                    reason: format!(
                        "canonical id {canonical} appears at slice rows {first} and {row}; \
                         the map is not injective and two tokens would share one embedding row"
                    ),
                });
            }
        }
        Ok(Self {
            orig_to_slice,
            slice_to_orig,
        })
    }

    /// The `slice_to_orig` table, in row order.
    ///
    /// The whole remap as one serializable value: the reverse direction is
    /// derivable from it (see [`Self::from_slice_to_orig`]), so transporting
    /// both would be transporting one fact twice.
    #[must_use]
    pub fn slice_to_orig(&self) -> &[u32] {
        &self.slice_to_orig
    }

    /// Number of rows in the slice embedding table.
    #[must_use]
    pub fn slice_vocab(&self) -> usize {
        self.slice_to_orig.len()
    }

    /// Canonical id for a slice row, if the row exists.
    #[must_use]
    pub fn to_canonical(&self, slice_row: u32) -> Option<u32> {
        self.slice_to_orig.get(slice_row as usize).copied()
    }
}

/// Wire form of `vocab_remap.json`.
///
/// Gated exactly like its sole constructor, [`VocabRemap::from_json_bytes`]:
/// `vocab_remap.json` belongs to the slice-fixture path, and a `--features
/// setfit` build has no way to reach it. Keeping the gate on one half only is
/// what left a genuine dead-code finding behind after 01-07 removed this
/// module's `#![allow(dead_code)]` — the targeted `cfg` is the fix, not an
/// allow.
#[cfg(feature = "conformance-fixtures")]
#[derive(serde::Deserialize)]
struct VocabRemapWire {
    orig_to_slice: HashMap<u32, u32>,
    slice_to_orig: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Model dimensions and slice configuration
// ---------------------------------------------------------------------------

/// The dimensions an import actually loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelDims {
    /// Hidden width.
    pub hidden: usize,
    /// Encoder layer count.
    pub layers: usize,
    /// Attention head count.
    pub heads: usize,
    /// FFN intermediate width.
    pub intermediate: usize,
    /// Embedding table rows.
    pub vocab: usize,
    /// Position table rows.
    pub max_positions: usize,
    /// Token-type table rows.
    pub type_vocab: usize,
    /// Padding token id.
    pub pad_token_id: u32,
}

/// Wire form of `slice_config.json` (01-04).
///
/// The `source_*` fields the generator also writes are intentionally unmodelled
/// — same reason `deny_unknown_fields` is absent from the pin path.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SliceConfig {
    /// Hidden width of the slice.
    pub hidden: usize,
    /// Attention heads retained.
    pub heads: usize,
    /// Per-head dimension (unchanged from the source model).
    pub head_dim: usize,
    /// Encoder layers retained.
    pub num_layers: usize,
    /// FFN intermediate width.
    pub intermediate: usize,
    /// Rows of the remapped embedding table.
    pub vocab: usize,
    /// Position table rows.
    pub positions: usize,
    /// Token-type table rows.
    pub type_vocab_size: usize,
    /// LayerNorm epsilon.
    pub layer_norm_eps: f64,
    /// Padding token id.
    pub pad_token_id: u32,
    /// Activation; must still be the pinned exact-erf `"gelu"`.
    pub hidden_act: String,
    /// Upstream revision the slice was cut from.
    pub source_revision: String,
    /// Sha256 of the tokenizer the slice was cut against.
    pub tokenizer_sha256: String,
}

impl SliceConfig {
    /// Parse `slice_config.json`.
    ///
    /// # Errors
    ///
    /// [`SetFitError::ImportIo`] if the bytes are not parseable.
    #[cfg(feature = "conformance-fixtures")]
    pub(crate) fn from_json_bytes(bytes: &[u8]) -> Result<Self, SetFitError> {
        serde_json::from_slice(bytes).map_err(|e| SetFitError::ImportIo {
            path: "slice_config.json".to_string(),
            reason: e.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// HuggingFace config wire forms
// ---------------------------------------------------------------------------

/// The BEHAVIOR-AFFECTING subset of `config.json`.
///
/// No `deny_unknown_fields` — see the module docs.
#[derive(Debug, Clone, serde::Deserialize)]
struct HfBertConfig {
    architectures: Vec<String>,
    attention_probs_dropout_prob: f64,
    hidden_act: String,
    hidden_dropout_prob: f64,
    hidden_size: usize,
    intermediate_size: usize,
    layer_norm_eps: f64,
    max_position_embeddings: usize,
    model_type: String,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    pad_token_id: u32,
    position_embedding_type: String,
    type_vocab_size: usize,
    vocab_size: usize,
}

/// One entry of `modules.json`.
#[derive(Debug, Clone, serde::Deserialize)]
struct SentenceTransformerModule {
    #[allow(dead_code)]
    idx: usize,
    #[allow(dead_code)]
    path: String,
    #[serde(rename = "type")]
    kind: String,
}

/// `1_Pooling/config.json`.
#[derive(Debug, Clone, serde::Deserialize)]
struct PoolingConfig {
    word_embedding_dimension: usize,
    pooling_mode_cls_token: bool,
    pooling_mode_mean_tokens: bool,
    pooling_mode_max_tokens: bool,
    pooling_mode_mean_sqrt_len_tokens: bool,
}

/// `sentence_bert_config.json`, when the checkout carries one.
#[derive(Debug, Clone, serde::Deserialize)]
struct SentenceBertConfig {
    max_seq_length: usize,
}

// ---------------------------------------------------------------------------
// The import
// ---------------------------------------------------------------------------

/// A validated MiniLM checkpoint: dimensions, weights reader, and provenance.
pub struct MiniLmImport {
    dims: ModelDims,
    layer_norm_eps: f32,
    reader: AprV2Reader,
    tensor_prefix: &'static str,
    revision: String,
    tokenizer_sha256: String,
    vocab_remap: Option<VocabRemap>,
}

impl std::fmt::Debug for MiniLmImport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MiniLmImport")
            .field("dims", &self.dims)
            .field("revision", &self.revision)
            .field("tokenizer_sha256", &self.tokenizer_sha256)
            .field("is_slice", &self.vocab_remap.is_some())
            .finish()
    }
}

impl MiniLmImport {
    /// Open a pinned all-MiniLM-L6-v2 checkout.
    ///
    /// SEALED (D-08): `pub(crate)`. The public full-pin entry point is
    /// `SetFitMiniLm::from_pretrained_dir` (01-07).
    ///
    /// # Errors
    ///
    /// A typed [`SetFitError`] naming the field, module, or tensor that failed.
    pub(crate) fn open(dir: &Path) -> Result<Self, SetFitError> {
        // Order matters. Configuration is validated BEFORE a single weight byte
        // is read, so a mutated checkout is rejected on the field that was
        // mutated rather than on whatever the corrupted weights happen to do.
        let cfg = parse_hf_config(&read_required(dir, "config.json")?)?;
        let (dims, layer_norm_eps) = validate_against_pin(&cfg)?;
        validate_module_stack(&read_required(dir, "modules.json")?)?;
        validate_pooling(&read_required(dir, "1_Pooling/config.json")?, dims.hidden)?;

        // sentence_bert_config.json is validated WHEN PRESENT. The pinned
        // upstream file set recorded in 01-04's upstream_manifest.json does not
        // include it, so requiring it would reject the very checkout the D-10
        // gated suite materialises. Its absence cannot change behaviour: the
        // tokenizer bound is MAX_SEQUENCE_LENGTH in this crate's own code, not
        // read from the checkout. A PRESENT-but-different value is a real
        // disagreement about the model's contract and is rejected.
        if let Some(bytes) = read_optional(dir, "sentence_bert_config.json")? {
            validate_sentence_bert(&bytes)?;
        }

        let tokenizer_bytes = read_required(dir, "tokenizer.json")?;
        let got = sha256_hex(&tokenizer_bytes);
        if got != PINNED_TOKENIZER_SHA256 {
            return Err(SetFitError::TokenizerHashMismatch {
                expected: PINNED_TOKENIZER_SHA256.to_string(),
                got,
            });
        }

        let (weights_name, weights) = read_weights(dir)?;
        let reader = parse_apr(&weights, &weights_name)?;
        let tensor_prefix = detect_bert_prefix(&reader);
        load_and_check_tensors(&reader, tensor_prefix, &dims)?;

        Ok(Self {
            dims,
            layer_norm_eps,
            reader,
            tensor_prefix,
            // Recorded as immutable data. Nothing in this module ever resolves a
            // branch name (T-1-10).
            revision: PINNED_REVISION.to_string(),
            tokenizer_sha256: got,
            vocab_remap: None,
        })
    }

    /// Open a slice fixture APR.
    ///
    /// Bypasses ONLY the equality-with-the-pin checks. Structural consistency,
    /// tensor presence/shape, remap validity and finiteness all still apply.
    ///
    /// SEALED (D-08): `pub(crate)`. The public fixture entry point is
    /// `SetFitMiniLm::from_slice_fixture` (01-07).
    ///
    /// # Errors
    ///
    /// A typed [`SetFitError`] naming what failed.
    #[cfg(feature = "conformance-fixtures")]
    pub(crate) fn open_slice_fixture(
        apr: &Path,
        config: &SliceConfig,
        remap: &VocabRemap,
    ) -> Result<Self, SetFitError> {
        // The bypass covers EXACTLY ONE thing: equality with the pinned
        // architecture's dimensions. Everything else below still runs.
        if config.hidden_act != PINNED_ACTIVATION {
            return Err(SetFitError::UnsupportedActivation {
                got: config.hidden_act.clone(),
            });
        }
        if config.source_revision != PINNED_REVISION {
            return Err(SetFitError::ImportConfigMismatch {
                field: "source_revision".to_string(),
                expected: PINNED_REVISION.to_string(),
                got: config.source_revision.clone(),
            });
        }
        if config.tokenizer_sha256 != PINNED_TOKENIZER_SHA256 {
            return Err(SetFitError::TokenizerHashMismatch {
                expected: PINNED_TOKENIZER_SHA256.to_string(),
                got: config.tokenizer_sha256.clone(),
            });
        }

        // Structural self-consistency. A slice whose heads do not tile its
        // hidden width is not a slice of anything.
        if config.heads == 0 || config.head_dim == 0 || config.hidden == 0 {
            return Err(SetFitError::ImportConfigMismatch {
                field: "heads".to_string(),
                expected: "non-zero heads, head_dim and hidden".to_string(),
                got: format!(
                    "heads {} head_dim {} hidden {}",
                    config.heads, config.head_dim, config.hidden
                ),
            });
        }
        if config.heads * config.head_dim != config.hidden {
            return Err(SetFitError::ImportConfigMismatch {
                field: "heads".to_string(),
                expected: format!("hidden {} / head_dim {}", config.hidden, config.head_dim),
                got: format!(
                    "heads {} (heads * head_dim = {})",
                    config.heads,
                    config.heads * config.head_dim
                ),
            });
        }
        for (field, value) in [
            ("num_layers", config.num_layers),
            ("intermediate", config.intermediate),
            ("vocab", config.vocab),
            ("positions", config.positions),
            ("type_vocab_size", config.type_vocab_size),
        ] {
            if value == 0 {
                return Err(SetFitError::ImportConfigMismatch {
                    field: field.to_string(),
                    expected: "non-zero".to_string(),
                    got: "0".to_string(),
                });
            }
        }
        let layer_norm_eps = narrow_eps(config.layer_norm_eps)?;

        // The remap must describe THIS slice, not some other one.
        if remap.slice_vocab() != config.vocab {
            return Err(SetFitError::RemapInvalid {
                reason: format!(
                    "remap covers {} rows but the slice vocabulary is {}",
                    remap.slice_vocab(),
                    config.vocab
                ),
            });
        }

        let dims = ModelDims {
            hidden: config.hidden,
            layers: config.num_layers,
            heads: config.heads,
            intermediate: config.intermediate,
            vocab: config.vocab,
            max_positions: config.positions,
            type_vocab: config.type_vocab_size,
            pad_token_id: config.pad_token_id,
        };

        let bytes = std::fs::read(apr).map_err(|e| SetFitError::ImportIo {
            path: apr.display().to_string(),
            reason: e.to_string(),
        })?;
        let reader = parse_apr(&bytes, &apr.display().to_string())?;
        let tensor_prefix = detect_bert_prefix(&reader);
        load_and_check_tensors(&reader, tensor_prefix, &dims)?;

        Ok(Self {
            dims,
            layer_norm_eps,
            reader,
            tensor_prefix,
            revision: config.source_revision.clone(),
            tokenizer_sha256: config.tokenizer_sha256.clone(),
            vocab_remap: Some(remap.clone()),
        })
    }

    /// Dimensions this import loaded.
    #[must_use]
    pub fn dims(&self) -> &ModelDims {
        &self.dims
    }

    /// LayerNorm epsilon.
    #[must_use]
    pub fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }

    /// Upstream revision this import was validated against.
    #[must_use]
    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Sha256 of the tokenizer this import is paired with.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        &self.tokenizer_sha256
    }

    /// `None` after a full-pin open; `Some` after a slice open.
    #[must_use]
    pub fn vocab_remap(&self) -> Option<&VocabRemap> {
        self.vocab_remap.as_ref()
    }

    /// The weights reader, for the encoder (01-06).
    pub(crate) fn reader(&self) -> &AprV2Reader {
        &self.reader
    }

    /// The `bert.` / `` prefix the checkpoint uses.
    pub(crate) fn tensor_prefix(&self) -> &'static str {
        self.tensor_prefix
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// The pinned dropout probabilities.
///
/// These are not in [`BertConfig`], which models no dropout, so they are read
/// off the pinned `config.json`. They are not taken on trust: the pinned config
/// is embedded byte-identically in the tests (digest-checked against
/// `upstream_manifest.json`) and asserted to be ACCEPTED, so a wrong constant
/// here fails `import_pin_accepts_the_real_config_with_its_unmodelled_extra_fields`.
/// `pub(crate)` (plan 01-06): the encoder places dropout at the four
/// HF-verified sites and must use the SAME probability the pin enforces. Two
/// unlinked copies of a pinned number is precisely how one of them drifts, so
/// `encoder_dropout_probability_agrees_with_the_enc01_pin` asserts equality at
/// `f32` rather than trusting two literals to stay in step.
pub(crate) const PINNED_HIDDEN_DROPOUT_PROB: f64 = 0.1;
/// See [`PINNED_HIDDEN_DROPOUT_PROB`].
pub(crate) const PINNED_ATTENTION_DROPOUT_PROB: f64 = 0.1;
/// The pinned position-embedding scheme. `relative_key` and friends change the
/// attention computation itself, so they are rejected rather than ignored.
const PINNED_POSITION_EMBEDDING_TYPE: &str = "absolute";
/// The pinned `architectures` entry.
const PINNED_ARCHITECTURE: &str = "BertModel";
/// The pinned `model_type`.
const PINNED_MODEL_TYPE: &str = "bert";
/// Weight-file names an open() checkout may carry, most specific first.
const WEIGHT_FILE_CANDIDATES: [&str; 2] = ["full_model.apr", "model.apr"];

/// Read one file from a model directory, naming it in any failure.
///
/// Shared with `setfit::from_pretrained_dir` / `from_slice_fixture`, which read the
/// same directory — a change to how a missing file is reported belongs in one place.
pub(super) fn read_required(dir: &Path, name: &str) -> Result<Vec<u8>, SetFitError> {
    std::fs::read(dir.join(name)).map_err(|e| SetFitError::ImportIo {
        path: name.to_string(),
        reason: e.to_string(),
    })
}

fn read_optional(dir: &Path, name: &str) -> Result<Option<Vec<u8>>, SetFitError> {
    let path = dir.join(name);
    if !path.exists() {
        return Ok(None);
    }
    read_required(dir, name).map(Some)
}

fn read_weights(dir: &Path) -> Result<(String, Vec<u8>), SetFitError> {
    for name in WEIGHT_FILE_CANDIDATES {
        if let Some(bytes) = read_optional(dir, name)? {
            return Ok((name.to_string(), bytes));
        }
    }
    Err(SetFitError::ImportIo {
        path: WEIGHT_FILE_CANDIDATES[0].to_string(),
        reason: format!(
            "no APR weights present (looked for {})",
            WEIGHT_FILE_CANDIDATES.join(", ")
        ),
    })
}

fn parse_apr(bytes: &[u8], name: &str) -> Result<AprV2Reader, SetFitError> {
    AprV2Reader::from_bytes(bytes).map_err(|e| SetFitError::ImportIo {
        path: name.to_string(),
        reason: format!("not a readable APR v2 container: {e}"),
    })
}

fn parse_hf_config(bytes: &[u8]) -> Result<HfBertConfig, SetFitError> {
    serde_json::from_slice(bytes).map_err(|e| SetFitError::ImportIo {
        path: "config.json".to_string(),
        reason: e.to_string(),
    })
}

fn mismatch(
    field: &str,
    expected: impl std::fmt::Display,
    got: impl std::fmt::Display,
) -> SetFitError {
    SetFitError::ImportConfigMismatch {
        field: field.to_string(),
        expected: expected.to_string(),
        got: got.to_string(),
    }
}

fn check_usize(field: &str, got: usize, expected: usize) -> Result<(), SetFitError> {
    if got == expected {
        Ok(())
    } else {
        Err(mismatch(field, expected, got))
    }
}

/// Narrow a JSON `f64` epsilon to the `f32` the model actually computes in.
///
/// Comparison happens at `f32` deliberately: `1e-12f32` and `1e-12f64` are
/// different numbers, so comparing the parsed `f64` against the `BertConfig`
/// constant in `f64` would reject the pinned config's own value.
fn narrow_eps(value: f64) -> Result<f32, SetFitError> {
    #[allow(clippy::cast_possible_truncation)]
    let narrowed = value as f32;
    if !narrowed.is_finite() || narrowed <= 0.0 {
        return Err(mismatch("layer_norm_eps", "a finite positive value", value));
    }
    Ok(narrowed)
}

/// Validate every BEHAVIOR-AFFECTING field against the pin.
fn validate_against_pin(cfg: &HfBertConfig) -> Result<(ModelDims, f32), SetFitError> {
    let pin = BertConfig::minilm_l6();

    if cfg.architectures.len() != 1 || cfg.architectures[0] != PINNED_ARCHITECTURE {
        return Err(SetFitError::UnsupportedArchitecture {
            got: format!("{:?}", cfg.architectures),
        });
    }
    if cfg.model_type != PINNED_MODEL_TYPE {
        return Err(mismatch("model_type", PINNED_MODEL_TYPE, &cfg.model_type));
    }
    if cfg.hidden_act != PINNED_ACTIVATION {
        return Err(SetFitError::UnsupportedActivation {
            got: cfg.hidden_act.clone(),
        });
    }
    if cfg.position_embedding_type != PINNED_POSITION_EMBEDDING_TYPE {
        return Err(mismatch(
            "position_embedding_type",
            PINNED_POSITION_EMBEDDING_TYPE,
            &cfg.position_embedding_type,
        ));
    }

    check_usize("hidden_size", cfg.hidden_size, pin.hidden_dim)?;
    check_usize("num_hidden_layers", cfg.num_hidden_layers, pin.num_layers)?;
    check_usize(
        "num_attention_heads",
        cfg.num_attention_heads,
        pin.num_heads,
    )?;
    check_usize(
        "intermediate_size",
        cfg.intermediate_size,
        pin.intermediate_dim,
    )?;
    check_usize("vocab_size", cfg.vocab_size, pin.vocab_size)?;
    check_usize(
        "max_position_embeddings",
        cfg.max_position_embeddings,
        pin.max_position_embeddings,
    )?;
    check_usize("type_vocab_size", cfg.type_vocab_size, pin.type_vocab_size)?;
    if cfg.pad_token_id != pin.pad_token_id {
        return Err(mismatch("pad_token_id", pin.pad_token_id, cfg.pad_token_id));
    }

    let eps = narrow_eps(cfg.layer_norm_eps)?;
    if eps != pin.layer_norm_eps {
        return Err(mismatch(
            "layer_norm_eps",
            pin.layer_norm_eps,
            cfg.layer_norm_eps,
        ));
    }

    // Dropout is inference-inert but training-critical, and a checkpoint that
    // declares a different rate is not the model the fixtures were generated
    // from — so it is a pin field, not a comment.
    if cfg.hidden_dropout_prob != PINNED_HIDDEN_DROPOUT_PROB {
        return Err(mismatch(
            "hidden_dropout_prob",
            PINNED_HIDDEN_DROPOUT_PROB,
            cfg.hidden_dropout_prob,
        ));
    }
    if cfg.attention_probs_dropout_prob != PINNED_ATTENTION_DROPOUT_PROB {
        return Err(mismatch(
            "attention_probs_dropout_prob",
            PINNED_ATTENTION_DROPOUT_PROB,
            cfg.attention_probs_dropout_prob,
        ));
    }

    Ok((
        ModelDims {
            hidden: cfg.hidden_size,
            layers: cfg.num_hidden_layers,
            heads: cfg.num_attention_heads,
            intermediate: cfg.intermediate_size,
            vocab: cfg.vocab_size,
            max_positions: cfg.max_position_embeddings,
            type_vocab: cfg.type_vocab_size,
            pad_token_id: cfg.pad_token_id,
        },
        eps,
    ))
}

/// The sentence-transformers module graph must be Transformer -> Pooling ->
/// Normalize. The trailing `Normalize` IS the normalize flag: dropping it
/// changes the embedding the model emits.
fn validate_module_stack(bytes: &[u8]) -> Result<(), SetFitError> {
    let modules: Vec<SentenceTransformerModule> =
        serde_json::from_slice(bytes).map_err(|e| SetFitError::ImportIo {
            path: "modules.json".to_string(),
            reason: e.to_string(),
        })?;
    let kinds: Vec<&str> = modules.iter().map(|m| m.kind.as_str()).collect();
    let expected = [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
    ];
    if kinds != expected {
        return Err(SetFitError::UnsupportedPooling {
            got: format!(
                "modules.json declares {kinds:?}; the pin requires \
                 Transformer -> Pooling -> Normalize (the trailing Normalize is the \
                 normalize flag)"
            ),
        });
    }
    Ok(())
}

/// Mean pooling, and only mean pooling.
fn validate_pooling(bytes: &[u8], hidden: usize) -> Result<(), SetFitError> {
    let pooling: PoolingConfig =
        serde_json::from_slice(bytes).map_err(|e| SetFitError::ImportIo {
            path: "1_Pooling/config.json".to_string(),
            reason: e.to_string(),
        })?;
    if !pooling.pooling_mode_mean_tokens {
        return Err(SetFitError::UnsupportedPooling {
            got: "pooling_mode_mean_tokens = false; the pin is mean pooling".to_string(),
        });
    }
    for (field, enabled) in [
        ("pooling_mode_cls_token", pooling.pooling_mode_cls_token),
        ("pooling_mode_max_tokens", pooling.pooling_mode_max_tokens),
        (
            "pooling_mode_mean_sqrt_len_tokens",
            pooling.pooling_mode_mean_sqrt_len_tokens,
        ),
    ] {
        if enabled {
            return Err(SetFitError::UnsupportedPooling {
                got: format!("{field} = true; the pin is mean pooling ONLY"),
            });
        }
    }
    if pooling.word_embedding_dimension != hidden {
        return Err(mismatch(
            "word_embedding_dimension",
            hidden,
            pooling.word_embedding_dimension,
        ));
    }
    Ok(())
}

fn validate_sentence_bert(bytes: &[u8]) -> Result<(), SetFitError> {
    let sbert: SentenceBertConfig =
        serde_json::from_slice(bytes).map_err(|e| SetFitError::ImportIo {
            path: "sentence_bert_config.json".to_string(),
            reason: e.to_string(),
        })?;
    check_usize(
        "max_seq_length",
        sbert.max_seq_length,
        PINNED_MAX_SEQ_LENGTH,
    )
}

/// Every tensor the encoder will read, with the shape it must have.
fn expected_tensor_specs(prefix: &str, dims: &ModelDims) -> Vec<(String, Vec<usize>)> {
    let h = dims.hidden;
    let im = dims.intermediate;
    let mut specs = vec![
        (
            format!("{prefix}embeddings.word_embeddings.weight"),
            vec![dims.vocab, h],
        ),
        (
            format!("{prefix}embeddings.position_embeddings.weight"),
            vec![dims.max_positions, h],
        ),
        (
            format!("{prefix}embeddings.token_type_embeddings.weight"),
            vec![dims.type_vocab, h],
        ),
        (format!("{prefix}embeddings.LayerNorm.weight"), vec![h]),
        (format!("{prefix}embeddings.LayerNorm.bias"), vec![h]),
    ];
    for idx in 0..dims.layers {
        let p = format!("{prefix}encoder.layer.{idx}");
        for proj in ["query", "key", "value"] {
            specs.push((format!("{p}.attention.self.{proj}.weight"), vec![h, h]));
            specs.push((format!("{p}.attention.self.{proj}.bias"), vec![h]));
        }
        specs.push((format!("{p}.attention.output.dense.weight"), vec![h, h]));
        specs.push((format!("{p}.attention.output.dense.bias"), vec![h]));
        specs.push((format!("{p}.attention.output.LayerNorm.weight"), vec![h]));
        specs.push((format!("{p}.attention.output.LayerNorm.bias"), vec![h]));
        specs.push((format!("{p}.intermediate.dense.weight"), vec![im, h]));
        specs.push((format!("{p}.intermediate.dense.bias"), vec![im]));
        specs.push((format!("{p}.output.dense.weight"), vec![h, im]));
        specs.push((format!("{p}.output.dense.bias"), vec![h]));
        specs.push((format!("{p}.output.LayerNorm.weight"), vec![h]));
        specs.push((format!("{p}.output.LayerNorm.bias"), vec![h]));
    }
    specs
}

/// Read every expected tensor through the checked reader and scan it for
/// non-finite values BEFORE anything downstream can consume it (PF-011).
fn load_and_check_tensors(
    reader: &AprV2Reader,
    prefix: &str,
    dims: &ModelDims,
) -> Result<(), SetFitError> {
    for (name, shape) in expected_tensor_specs(prefix, dims) {
        // The A-01 amendment exists for this call: the checked-read semantics
        // (presence, dtype path, element count) are reused, not reimplemented.
        let tensor: Tensor = read_tensor(reader, &name, &shape)?;
        if let Some(position) = tensor.data().iter().position(|v| !v.is_finite()) {
            return Err(SetFitError::NonFiniteTensor {
                tensor: name,
                position,
            });
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "setfit"))]
#[path = "import_tests.rs"]
mod import_tests;
