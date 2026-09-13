//! SetFit / all-MiniLM-L6-v2 conformance boundary (feature `setfit`).
//!
//! Contract: `setfit-encoder-conformance-v1`. Requirements ENC-01 (pinned
//! import with typed rejection of every unsupported variant) and ENC-02 (exact
//! tokenizer parity against the frozen fixtures in
//! `crates/aprender-core/tests/fixtures/setfit/`).
//!
//! # Feature isolation (D-05 / D-06)
//!
//! The whole module is behind `setfit`, which is the only feature that enables
//! the `tokenizers` dependency. A build without `setfit` must not contain a
//! `tokenizers` node at all — minimal consumers of this crate pay nothing for a
//! conformance path they do not use. `setfit` also enables `sha2`, which the
//! module uses in production code to hash tokenizer bytes and input text, so the
//! feature is dependency-closed and `--features setfit` builds on its own.
//!
//! `conformance-fixtures` implies `setfit` and additionally unlocks the
//! **test-only** slice constructor. That separation is deliberate (RESEARCH
//! Pitfall 3): the fixture gates run against a 2-layer/64-hidden index slice,
//! while the pin gates run against the full pinned architecture. Without two
//! distinct constructors the two families of gate contradict each other, and
//! the usual "fix" — parameterising the pin path by caller-supplied dims — is
//! precisely the failure mode PF-011 records.
//!
//! # Visibility rule for this module (D-08, user decision 2026-08-08)
//!
//! Every constructor that produces a tokenizer or an import is `pub(crate)`, and
//! [`SentenceBatch`]'s fields are `pub(crate)` with read-only public accessors.
//! The bound type (`SetFitMiniLm`, 01-07) is the sole public entry point, and it
//! builds the tokenizer and the encoder together from one source — so a
//! mismatched tokenizer/encoder pair is **not constructible** from outside the
//! crate, rather than merely being detected at runtime. The `tokenizer_sha256`
//! equality check the encoder performs at every forward call is retained as
//! defense in depth for in-crate misuse; it is meaningful precisely because the
//! value it compares is not out-of-crate-writable.
//!
//! Re-exports below are **types only**. Never re-export a sealed constructor as
//! a free function — that would reopen the seal through a path the source
//! assertions do not scan.

pub mod artifact;
pub mod classify;
pub mod dropout_rng;
pub mod encoder;
pub mod error;
pub mod import;
pub mod loss;
pub mod tokenizer;

pub use artifact::{
    artifact_sha256_hex, load_setfit_apr, read_setfit_apr_bytes_bounded, read_setfit_apr_parts,
    write_setfit_apr, ProbeReplayDivergence, SetFitAprParts, SetFitArtifactDoc,
    SetFitArtifactError, SetFitArtifactView, SetFitHeadDoc, SetFitPreprocessingDoc,
    SetFitProbeRecord, VerifiedSetFitModel, MAX_ARTIFACT_BYTES, MAX_ENCODER_LAYERS,
    NULLABLE_PATH_ALLOWLIST, PROBE_EMBEDDING_ABS_TOLERANCE, PROBE_LOGITS_ABS_TOLERANCE,
    PROBE_PROBABILITIES_ABS_TOLERANCE, WALKED_SUBDOCUMENTS,
};
pub use classify::{
    ClassifyError, ClassifyRequestDocument, ClassifyResponse, ClassifyResult,
    CLASSIFY_SCHEMA_VERSION, MAX_BATCH_TEXTS, MAX_REQUEST_BODY_BYTES,
    PROBABILITY_MASS_ABS_TOLERANCE,
};
pub use dropout_rng::{DropoutRngError, SiteDropout};
pub use encoder::{
    BertSentenceEncoder, ExecutionBackend, L2_EPS, NORMALIZATION_POLICY, POOLING_POLICY,
};
pub use error::SetFitError;
pub use import::{
    MiniLmImport, ModelDims, SliceConfig, VocabRemap, PINNED_ACTIVATION, PINNED_MAX_SEQ_LENGTH,
    PINNED_REVISION, PINNED_TOKENIZER_SHA256,
};
pub use loss::pair_cosine_mse;
pub use tokenizer::{
    InputProvenance, MiniLmTokenizer, SentenceBatch, TruncationFact, MAX_SEQUENCE_LENGTH,
    PADDING_MODE,
};

use std::collections::BTreeMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// The reload surface's architecture record (plan 03-08, D-07)
// ---------------------------------------------------------------------------

/// Everything a rebuild needs that is NOT a tensor and NOT the tokenizer bytes.
///
/// The field set is [`SliceConfig`]'s, plus the vocabulary remap. The remap is the
/// addition that makes the record COMPLETE rather than merely plausible: a slice
/// encoder gathers through `slice_to_orig`, so a record without it rebuilds an
/// encoder that reads the wrong embedding row for every token and still looks
/// structurally valid.
///
/// `SliceConfig` is deliberately NOT reused as the transport type, and there is no
/// `From<&SliceConfig>` for this: that file is parsed only on the
/// `conformance-fixtures` path, it does not exist on the full-pin path, and it
/// carries no remap — so such a conversion would be a door that mints incomplete
/// records for exactly the models that need the missing field.
///
/// # It lives HERE and not in `import.rs`
///
/// `import.rs` is the pinned-config path, and `import_pin_constructors_are_sealed`
/// asserts that NO wire form in that file denies unknown fields — because the real
/// `config.json` carries metadata this crate does not model, so denying unknown
/// fields there would reject the pinned model itself. This record has the opposite
/// obligation: it is a closed artifact schema and an unknown field in it is a
/// rejection. Putting it in `import.rs` would have forced that guard to be
/// weakened to accommodate a type it was never about.
///
/// Produced by exactly one function, [`SetFitMiniLm::architecture`], which reads
/// every field off the encoder that is going to be rebuilt.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderArchitecture {
    /// Hidden width.
    pub hidden: usize,
    /// Attention heads.
    pub heads: usize,
    /// Per-head dimension (`hidden / heads`).
    pub head_dim: usize,
    /// Encoder layers.
    pub num_layers: usize,
    /// FFN intermediate width.
    pub intermediate: usize,
    /// Embedding table rows.
    pub vocab: usize,
    /// Position table rows.
    pub positions: usize,
    /// Token-type table rows.
    pub type_vocab_size: usize,
    /// LayerNorm epsilon, widened from the `f32` the encoder holds.
    ///
    /// `f32 -> f64 -> f32` is lossless, so the rebuild recovers the exact constant
    /// the original normalized with.
    pub layer_norm_eps: f64,
    /// Padding token id.
    pub pad_token_id: u32,
    /// Activation; must still be the pinned exact-erf [`PINNED_ACTIVATION`].
    pub hidden_act: String,
    /// Upstream revision this encoder was loaded from.
    pub source_revision: String,
    /// Sha256 of the tokenizer this encoder is paired with.
    pub tokenizer_sha256: String,
    /// `slice_to_orig` for a slice encoder; `None` for the full pin.
    pub vocab_remap: Option<Vec<u32>>,
}

use crate::autograd::Tensor;
use crate::nn::Module;

// ---------------------------------------------------------------------------
// Freeze groups (D-20 / D-21 / D-22)
// ---------------------------------------------------------------------------

/// A named, validated slice of the encoder's parameters (D-22).
///
/// A structured enum, deliberately **not** a glob/string DSL. A string pattern
/// API has two failure modes this cannot have: a typo silently addresses
/// nothing, and a pattern that is correct today silently re-addresses a
/// different set the moment a name changes upstream. Every variant here maps to
/// an exact prefix set over the HF dotted names 01-06 pins against
/// `gradients.json`'s `parameter_order`, and a group that matches zero names is
/// a typed error rather than a no-op.
///
/// The four variants are exactly ENC-04's components: the embeddings block, and
/// per layer the attention, feed-forward and normalization sub-blocks. Coarser
/// policies ("freeze the bottom two layers") are expressible as a list of
/// groups, so no additional API is needed for them.
///
/// `Ord` is derived because [`SetFitMiniLm::apply_freeze`] normalizes its
/// argument by sorting and deduplicating — that single step is what makes the
/// policy order-insensitive and duplicate-tolerant without three special cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FreezeGroup {
    /// `embeddings.*` — the word/position/token-type tables and the embeddings
    /// LayerNorm.
    Embeddings,
    /// `encoder.layer.N.attention.self.*` plus
    /// `encoder.layer.N.attention.output.dense.*`.
    ///
    /// NOT `attention.output.LayerNorm`, which belongs to [`Self::LayerNorm`].
    /// The boundary matters: freezing a block's normalization along with its
    /// projections is a different experiment from freezing the projections
    /// alone, and a mapping that quietly included it would make the two
    /// indistinguishable.
    LayerAttention(usize),
    /// `encoder.layer.N.intermediate.*` plus `encoder.layer.N.output.dense.*`.
    LayerFfn(usize),
    /// Both LayerNorms of layer N: `attention.output.LayerNorm.*` and
    /// `output.LayerNorm.*`.
    LayerNorm(usize),
}

impl FreezeGroup {
    /// The layer index this group addresses, or `None` for [`Self::Embeddings`].
    #[must_use]
    pub fn layer(self) -> Option<usize> {
        match self {
            Self::Embeddings => None,
            Self::LayerAttention(n) | Self::LayerFfn(n) | Self::LayerNorm(n) => Some(n),
        }
    }

    /// The exact HF dotted-name prefixes this group addresses.
    ///
    /// THE single definition of the group -> parameter mapping; nothing else in
    /// this module hardcodes a prefix. Every prefix ends with `.` on purpose, so
    /// `encoder.layer.1.` cannot match `encoder.layer.10.…` when a future model
    /// has ten or more layers.
    #[must_use]
    pub fn name_prefixes(self) -> Vec<String> {
        match self {
            // The word/position/token-type tables AND the embeddings LayerNorm:
            // ENC-04 treats them as one component, and 01-06's gradient gate
            // aggregates them the same way.
            Self::Embeddings => vec!["embeddings.".to_string()],
            // `attention.self.*` is Q/K/V; `attention.output.dense.*` is the
            // out-projection `MultiHeadAttention` applies inside forward_self.
            // NOT `attention.output.LayerNorm.*` — that is LayerNorm(n).
            Self::LayerAttention(n) => vec![
                format!("encoder.layer.{n}.attention.self."),
                format!("encoder.layer.{n}.attention.output.dense."),
            ],
            Self::LayerFfn(n) => vec![
                format!("encoder.layer.{n}.intermediate."),
                format!("encoder.layer.{n}.output.dense."),
            ],
            Self::LayerNorm(n) => vec![
                format!("encoder.layer.{n}.attention.output.LayerNorm."),
                format!("encoder.layer.{n}.output.LayerNorm."),
            ],
        }
    }

    /// True when `name` falls inside this group.
    #[must_use]
    pub fn matches(self, name: &str) -> bool {
        self.name_prefixes().iter().any(|p| name.starts_with(p))
    }
}

// ---------------------------------------------------------------------------
// The bound model (D-08)
// ---------------------------------------------------------------------------

/// A MiniLM tokenizer bound to the encoder built from the SAME source.
///
/// # The seal, in one sentence
///
/// [`MiniLmTokenizer::from_bytes`], [`MiniLmImport::open`],
/// [`MiniLmImport::open_slice_fixture`] and [`BertSentenceEncoder::from_import`]
/// are all `pub(crate)`, so the only way an out-of-crate caller obtains a
/// tokenizer paired with an encoder is through this type — which loads both
/// halves from one directory. A mismatched pair is therefore **not
/// constructible** rather than merely detected (D-08 as written, user decision
/// 2026-08-08). The `tokenizer_sha256` equality the encoder enforces at every
/// forward call is kept as defense in depth against IN-crate misuse.
///
/// # Freeze policy
///
/// Default is all-trainable (D-20). [`Self::apply_freeze`] has REPLACEMENT
/// semantics: the argument fully defines the policy, so any group not listed
/// becomes trainable again. It is idempotent, order-insensitive and
/// duplicate-tolerant, and `apply_freeze(&[])` is exactly [`Self::clear_freeze`].
///
/// Freezing is BY EXCLUSION: a frozen parameter is dropped from
/// [`Self::trainable_parameters_mut`] — the set an optimizer is built from — and
/// additionally has `requires_grad` cleared. The exclusion is the load-bearing
/// half; the flag is the belt to its braces.
pub struct SetFitMiniLm {
    tokenizer: MiniLmTokenizer,
    encoder: BertSentenceEncoder,
    /// Normalized (sorted, deduplicated) freeze policy. Empty means D-20's
    /// all-trainable default.
    freeze: Vec<FreezeGroup>,
}

impl std::fmt::Debug for SetFitMiniLm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetFitMiniLm")
            .field("encoder", &self.encoder)
            .field("freeze", &self.freeze)
            .finish_non_exhaustive()
    }
}

impl SetFitMiniLm {
    /// Load a pinned all-MiniLM-L6-v2 checkout: tokenizer and encoder together.
    ///
    /// Returns a model in **eval** mode, matching HuggingFace
    /// `from_pretrained`, which is also the mode the frozen fixtures were
    /// generated in (D-16). Training callers flip it with
    /// [`Self::set_training`].
    ///
    /// # Errors
    ///
    /// Any typed [`SetFitError`] the ENC-01 pin, the tokenizer load or the
    /// tensor read produces, naming the field/file/tensor that failed.
    pub fn from_pretrained_dir(dir: &Path, root_seed: u64) -> Result<Self, SetFitError> {
        // ONE source for both halves. `MiniLmImport::open` additionally requires
        // these very bytes to hash to `PINNED_TOKENIZER_SHA256`, so the pairing
        // is correct by construction and not by a check that could be skipped.
        let tokenizer =
            MiniLmTokenizer::from_bytes(&import::read_required(dir, "tokenizer.json")?)?;
        let import = MiniLmImport::open(dir)?;
        let encoder = BertSentenceEncoder::from_import(&import, root_seed)?;
        Ok(Self {
            tokenizer,
            encoder,
            freeze: Vec::new(),
        })
    }

    /// Load the frozen conformance slice: tokenizer and encoder together.
    ///
    /// Reads `tokenizer.json`, `slice_config.json`, `vocab_remap.json` and
    /// `slice_model.apr` from `fixture_dir`. Returns a model in eval mode.
    ///
    /// # Errors
    ///
    /// As [`Self::from_pretrained_dir`], plus [`SetFitError::RemapInvalid`] if
    /// the remap does not describe this slice.
    #[cfg(feature = "conformance-fixtures")]
    pub fn from_slice_fixture(fixture_dir: &Path, root_seed: u64) -> Result<Self, SetFitError> {
        let tokenizer =
            MiniLmTokenizer::from_bytes(&import::read_required(fixture_dir, "tokenizer.json")?)?;
        let config = SliceConfig::from_json_bytes(&import::read_required(
            fixture_dir,
            "slice_config.json",
        )?)?;
        let remap = VocabRemap::from_json_bytes(
            &import::read_required(fixture_dir, "vocab_remap.json")?,
            config.vocab,
        )?;
        // `open_slice_fixture` requires `config.tokenizer_sha256` to equal the
        // pin, and the tokenizer above hashes the bytes from the same directory,
        // so the two halves cannot disagree.
        let import = MiniLmImport::open_slice_fixture(
            &fixture_dir.join("slice_model.apr"),
            &config,
            &remap,
        )?;
        let encoder = BertSentenceEncoder::from_import(&import, root_seed)?;
        Ok(Self {
            tokenizer,
            encoder,
            freeze: Vec::new(),
        })
    }

    /// Rebuild a model from an artifact's bytes alone (plan 03-08, D-07).
    ///
    /// THE reload surface. It takes the exact tokenizer bytes, the architecture
    /// record and every named encoder tensor, and it takes nothing else — in
    /// particular it takes no resolved runtime configuration, because a resolved
    /// device is a fact about the host that is running now, not about the host
    /// that wrote the file.
    ///
    /// The tokenizer hash is checked FIRST, before a tensor is touched: the
    /// architecture record's `tokenizer_sha256` is what pairs the two halves, and
    /// a rebuild that installed the tensors before noticing the tokenizer was a
    /// different one would have done all its work on a mismatched pair.
    ///
    /// The returned model is in eval mode, like every other constructor here.
    ///
    /// # Errors
    ///
    /// [`SetFitError::TokenizerHashMismatch`] if the bytes do not hash to the
    /// record's digest, [`SetFitError::TokenizerLoad`] if they do not parse, and
    /// anything [`BertSentenceEncoder::from_named_tensors`] rejects — each naming
    /// the tensor, field or remap row that failed.
    pub fn from_bundle_parts(
        tokenizer_bytes: &[u8],
        arch: &EncoderArchitecture,
        tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
        root_seed: u64,
    ) -> Result<Self, SetFitError> {
        let observed = tokenizer::sha256_hex(tokenizer_bytes);
        if observed != arch.tokenizer_sha256 {
            return Err(SetFitError::TokenizerHashMismatch {
                expected: arch.tokenizer_sha256.clone(),
                got: observed,
            });
        }
        let tokenizer = MiniLmTokenizer::from_bytes(tokenizer_bytes)?;
        let encoder = BertSentenceEncoder::from_named_tensors(arch, tensors, root_seed)?;
        Ok(Self {
            tokenizer,
            encoder,
            freeze: Vec::new(),
        })
    }

    /// Sha256 of the tokenizer half of this pair.
    #[must_use]
    pub fn tokenizer_sha256(&self) -> &str {
        self.tokenizer.tokenizer_sha256()
    }

    /// The exact `tokenizer.json` bytes this model's tokenizer was built from.
    ///
    /// The other half of [`Self::tokenizer_sha256`], and the reason a persistence
    /// artifact can rebuild a working tokenizer rather than only detect a wrong
    /// one. `sha256(tokenizer_bytes()) == tokenizer_sha256()` is asserted by a
    /// test so the two cannot drift.
    #[must_use]
    pub fn tokenizer_bytes(&self) -> &[u8] {
        self.tokenizer.source_bytes()
    }

    /// This model's architecture record — the non-tensor half of a reload.
    ///
    /// Every field is READ OFF the encoder that would be rebuilt, never restated
    /// from a configuration file: a record assembled from the config a caller
    /// intended describes the model that was meant to be built, which is exactly
    /// the claim a reload is supposed to check.
    #[must_use]
    pub fn architecture(&self) -> EncoderArchitecture {
        let dims = self.encoder.dims();
        EncoderArchitecture {
            hidden: dims.hidden,
            heads: dims.heads,
            // Derived, not stored: `ModelDims` has no head_dim, and the encoder's
            // attention is built as `hidden / heads` in exactly one place.
            head_dim: dims.hidden / dims.heads.max(1),
            num_layers: dims.layers,
            intermediate: dims.intermediate,
            vocab: dims.vocab,
            positions: dims.max_positions,
            type_vocab_size: dims.type_vocab,
            layer_norm_eps: f64::from(self.encoder.layer_norm_eps()),
            pad_token_id: dims.pad_token_id,
            // The import rejects every other activation, so this is the only
            // value an existing encoder can have been built with.
            hidden_act: PINNED_ACTIVATION.to_string(),
            source_revision: self.encoder.source_revision().to_string(),
            tokenizer_sha256: self.tokenizer.tokenizer_sha256().to_string(),
            vocab_remap: self
                .encoder
                .vocab_remap()
                .map(|remap| remap.slice_to_orig().to_vec()),
        }
    }

    /// Every named encoder parameter, in `Module::named_parameters` order.
    ///
    /// A READ accessor: shared borrows out, no construction and no mutation. It
    /// exists because a persistence artifact must write EVERY tensor, and the two
    /// existing enumerations ([`Self::trainable_parameters_mut`] and
    /// [`Self::frozen_parameters`]) partition that set by freeze policy — an
    /// artifact assembled from their union would silently depend on the policy in
    /// force when it was written.
    #[must_use]
    pub fn named_parameters(&self) -> Vec<(String, &Tensor)> {
        self.encoder.named_parameters()
    }

    /// This model's architecture fingerprint — see
    /// [`BertSentenceEncoder::architecture_fingerprint`].
    ///
    /// Forwarded rather than reached through `encoder()`, which is gated on
    /// `conformance-fixtures` and therefore absent from production builds.
    #[must_use]
    pub fn architecture_fingerprint(&self) -> String {
        self.encoder.architecture_fingerprint()
    }

    /// Encoder layers, for freeze-group validation and introspection.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.encoder.num_layers()
    }

    /// The root seed every dropout site's stream is derived from.
    ///
    /// Read off the encoder, not off a configuration: a persistence artifact must
    /// record the seed the model was BUILT with, and a run whose configuration
    /// seed differed from its encoder's would otherwise write a seed that rebuilds
    /// a different dropout schedule.
    #[must_use]
    pub fn root_seed(&self) -> u64 {
        self.encoder.root_seed()
    }

    /// Whether the encoder is in training mode.
    #[must_use]
    pub fn training(&self) -> bool {
        self.encoder.training()
    }

    /// Tokenize once, then encode: `[B, H]` unit-norm sentence embeddings.
    ///
    /// # Errors
    ///
    /// [`SetFitError::BatchInvalid`] for an empty input list, plus anything the
    /// encoder's boundary validation or the pooling/normalize primitives reject.
    pub fn encode_texts(&self, texts: &[&str]) -> Result<Tensor, SetFitError> {
        let batch = self.tokenizer.encode_batch(texts)?;
        self.encoder.encode(&batch)
    }

    /// [`Self::encode_texts`], plus the identity of the path that ran it (D-12).
    ///
    /// [`Self::encode_texts`] is left EXACTLY as it was: the training path and
    /// every Phase 1 conformance fixture call it, and their bytes must not move
    /// because a reporting channel was added beside them.
    ///
    /// # Errors
    ///
    /// Exactly [`Self::encode_texts`]'s.
    pub fn encode_texts_traced(
        &self,
        texts: &[&str],
    ) -> Result<(Tensor, ExecutionBackend), SetFitError> {
        let batch = self.tokenize_batch(texts)?;
        self.encode_batch_traced(&batch)
    }

    /// Tokenize with THIS model's tokenizer — the in-crate half of
    /// [`Self::encode_texts_traced`].
    ///
    /// `pub(crate)`: [`Self::tokenize`] is gated on `conformance-fixtures` and
    /// therefore absent from production builds, but
    /// [`crate::setfit::artifact::VerifiedSetFitModel::classify`] needs the
    /// batch's ordered truncation and attention-mask facts in every build. It
    /// stays crate-private so the D-08 seal is untouched.
    ///
    /// # Errors
    ///
    /// As [`Self::tokenize`].
    pub(crate) fn tokenize_batch(&self, texts: &[&str]) -> Result<SentenceBatch, SetFitError> {
        self.tokenizer.encode_batch(texts)
    }

    /// Encode an already-tokenized batch, returning the executing backend.
    ///
    /// Split from [`Self::encode_texts_traced`] so `classify` tokenizes ONCE and
    /// reads its token facts off the very batch that was encoded — rather than
    /// tokenizing a second time and reporting facts about a different call.
    ///
    /// # Errors
    ///
    /// As [`BertSentenceEncoder::encode`].
    pub(crate) fn encode_batch_traced(
        &self,
        batch: &SentenceBatch,
    ) -> Result<(Tensor, ExecutionBackend), SetFitError> {
        self.encoder.encode_with_backend(batch)
    }

    /// ENC-05 mode propagation, forwarded to the encoder.
    pub fn set_training(&mut self, training: bool) {
        self.encoder.set_training(training);
    }

    /// Point every dropout site at forward-call ordinal `forward_ordinal` (D-15).
    ///
    /// The ordinal is `2 * training_step + branch`, with `branch` 0 for the pair's
    /// A sentence and 1 for its B sentence — see
    /// [`dropout_rng::forward_ordinal`]. Two coordinates, not one, because
    /// [`pair_cosine_mse`] takes TWO `[B,H]` matrices: a training step runs two
    /// separate encoder forwards, and keying on the step alone would hand both
    /// siamese branches the identical mask at every element.
    ///
    /// # Errors
    ///
    /// [`SetFitError::DropoutRng`] if the ordinal does not fit the `u32` counter
    /// lane; every site is left at its previous ordinal in that case.
    pub fn set_forward_ordinal(&mut self, forward_ordinal: u64) -> Result<(), SetFitError> {
        self.encoder.set_forward_ordinal(forward_ordinal)
    }

    /// The forward-call ordinal every dropout site currently draws at.
    #[must_use]
    pub fn forward_ordinal(&self) -> u64 {
        self.encoder.forward_ordinal()
    }

    // -----------------------------------------------------------------------
    // Conformance-only READ accessors
    //
    // Borrows out, never construction. They exist because the lower-level
    // constructors are sealed and 01-08's out-of-crate harness still needs
    // per-layer introspection and the tokenized batch. Gated on
    // `conformance-fixtures`, not on `setfit`, so an ordinary consumer never
    // sees them.
    // -----------------------------------------------------------------------

    /// Borrow the encoder — 01-08 reaches `forward_tokens_per_layer` through it.
    #[cfg(feature = "conformance-fixtures")]
    #[must_use]
    pub fn encoder(&self) -> &BertSentenceEncoder {
        &self.encoder
    }

    /// Tokenize with THIS model's tokenizer.
    ///
    /// The returned batch is already stamped with the matching
    /// `tokenizer_sha256` and its fields are `pub(crate)` (01-05 W1), so an
    /// out-of-crate caller can read it but cannot re-point the stamp. A
    /// mismatched pair is therefore not assemblable from this accessor either.
    ///
    /// # Errors
    ///
    /// [`SetFitError::BatchInvalid`] for an empty list;
    /// [`SetFitError::TokenizerLoad`] if the tokenizer fails on an input.
    #[cfg(feature = "conformance-fixtures")]
    pub fn tokenize(&self, texts: &[&str]) -> Result<SentenceBatch, SetFitError> {
        self.tokenizer.encode_batch(texts)
    }

    // -----------------------------------------------------------------------
    // Freeze policy
    // -----------------------------------------------------------------------

    /// Replace the freeze policy.
    ///
    /// See the type docs for the semantics. Validation of EVERY group completes
    /// before any flag is touched, so a rejected call leaves the previous policy
    /// and every `requires_grad` exactly as it found them.
    ///
    /// # Errors
    ///
    /// [`SetFitError::FreezeGroupInvalid`] for a layer index outside
    /// `0..num_layers()`, or for a structurally valid group whose prefix set
    /// addresses zero named parameters (the naming-drift guard).
    pub fn apply_freeze(&mut self, groups: &[FreezeGroup]) -> Result<(), SetFitError> {
        let layers = self.encoder.num_layers();
        let all: Vec<String> = self
            .encoder
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();

        // (1) Validate EVERY group before touching anything. This ordering is
        //     what makes "no partial application" true: a validate-as-you-go
        //     loop would already have frozen the valid prefix of the list by the
        //     time it rejected a later group, leaving the model in a state no
        //     caller asked for and no return value describes.
        for g in groups {
            if let Some(n) = g.layer() {
                if n >= layers {
                    return Err(SetFitError::FreezeGroupInvalid {
                        reason: format!(
                            "{g:?} names layer {n}, but this encoder has {layers} layers \
                             (valid indices 0..{})",
                            layers.saturating_sub(1)
                        ),
                    });
                }
            }
            // Naming-drift guard. A structurally valid group that addresses
            // nothing means 01-06's dotted names moved; a policy that silently
            // freezes nothing is worse than one that fails, because the run
            // still looks like a successful partial freeze.
            if !all.iter().any(|name| g.matches(name)) {
                return Err(SetFitError::FreezeGroupInvalid {
                    reason: format!(
                        "{g:?} addresses ZERO named parameters (prefixes {:?}); the encoder's \
                         parameter naming has drifted from the freeze mapping",
                        g.name_prefixes()
                    ),
                });
            }
        }

        // (2) Normalize once. Sorting and deduplicating here is what delivers
        //     order-insensitivity and duplicate-tolerance from ONE code path
        //     instead of three special cases.
        let mut normalized = groups.to_vec();
        normalized.sort_unstable();
        normalized.dedup();

        // (3) Reset, then (4) apply. Resetting first is what delivers
        //     REPLACEMENT semantics: a group absent from the new policy becomes
        //     trainable again without anyone having to remember to un-freeze it.
        self.set_all_requires_grad(true);
        self.freeze = normalized;
        let frozen = self.frozen_names();
        for (name, t) in self.encoder.named_parameters_mut() {
            if frozen.contains(&name) {
                t.requires_grad_(false);
            }
        }
        Ok(())
    }

    /// Restore the D-20 default: every parameter trainable.
    pub fn clear_freeze(&mut self) {
        self.freeze.clear();
        self.set_all_requires_grad(true);
    }

    /// Names the current policy freezes, in `named_parameters()` order.
    fn frozen_names(&self) -> Vec<String> {
        self.encoder
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| self.freeze.iter().any(|g| g.matches(n)))
            .collect()
    }

    fn set_all_requires_grad(&mut self, requires: bool) {
        for (_, t) in self.encoder.named_parameters_mut() {
            t.requires_grad_(requires);
        }
    }

    /// The applied policy, normalized (deduplicated and sorted).
    #[must_use]
    pub fn freeze_policy(&self) -> Vec<FreezeGroup> {
        self.freeze.clone()
    }

    /// Named parameters an optimizer should update, honoring the freeze policy.
    ///
    /// This is the set 01-08's AdamW is built from, which is why freezing works
    /// by EXCLUSION here rather than only by clearing a flag.
    #[must_use]
    pub fn trainable_parameters_mut(&mut self) -> Vec<(String, &mut Tensor)> {
        let frozen = self.frozen_names();
        self.encoder
            .named_parameters_mut()
            .into_iter()
            .filter(|(n, _)| !frozen.contains(n))
            .collect()
    }

    /// Named parameters the freeze policy excludes from optimization.
    #[must_use]
    pub fn frozen_parameters(&self) -> Vec<(String, &Tensor)> {
        self.encoder
            .named_parameters()
            .into_iter()
            .filter(|(n, _)| self.freeze.iter().any(|g| g.matches(n)))
            .collect()
    }
}

#[cfg(all(test, feature = "setfit"))]
#[path = "model_tests.rs"]
mod model_tests;
