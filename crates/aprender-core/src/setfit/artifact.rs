//! The `setfit-apr-v1` WRITER: one pure deterministic function from a complete
//! artifact view to APR v2 bytes.
//!
//! Contract: `contracts/setfit-apr-v1.yaml` (APR-01). Everything this module
//! encodes — the storage map, the normative `SetFitArtifactDoc` field list, the
//! canonical tensor-name table, the four-path nullable allowlist over five
//! walked sub-documents, the six synthetic probe inputs — is READ FROM that
//! contract rather than restated here in a second, subtly different form. A
//! schema restated in three places is a schema that disagrees with itself.
//!
//! # What the writer guarantees
//!
//! `write_setfit_apr(view)` is a function of `view` and NOTHING ELSE: no clock,
//! no environment, no host name, no filesystem lookup, no random state. Two
//! calls on the same view — in the same process or in two different ones —
//! produce byte-identical output with an identical SHA-256. That is not a
//! nicety: the artifact hash is the identity every Phase 4 response carries, and
//! the codec's byte-canonical closure obligation (`serialize(deserialize(bytes))
//! == bytes`) fails outright the moment any writer input is not derivable from
//! the view.
//!
//! # Why the head is TENSORS and not metadata (review B1)
//!
//! `setfit.head.weight` and `setfit.head.bias` are first-class named F32 tensors.
//! A `K*d` float array inside the JSON document would be lossy at the text
//! boundary, unaligned, invisible to `apr tensors` / `apr diff` / `apr qa`, and
//! would push the metadata section toward the container's 16 MiB
//! `MAX_METADATA_SIZE`. Because they are tensors, the architecture-derived
//! tensor-set rule can REFUSE an artifact that is missing either one.
//!
//! # Why exactly ONE custom metadata key (Pitfall 1)
//!
//! `AprV2Metadata.custom` is a `HashMap<String, Value>` with `#[serde(flatten)]`
//! (header_impl.rs:297-299) and `HashMap` iteration order is unspecified. With N
//! top-level custom keys the serialized JSON key order varies per process, so the
//! container checksum over it is not reproducible and the closure check fails
//! intermittently — the worst kind of red. ONE key holding one
//! `serde_json::Map` is reproducible, because `serde_json::Map` is BTreeMap-backed
//! and no workspace crate enables `preserve_order`.
//!
//! # Why `created_at` stays `None` (Pitfall 2)
//!
//! A timestamp would make two artifacts of the same run differ, breaking both the
//! closure equation and the artifact-hash identity. The container's own
//! `license` / `data_source` / `data_license` fields serialize as explicit
//! `null` BY DESIGN (no `skip_serializing_if`, required by FALSIFY-SHIP-022);
//! that is deterministic, expected, and deliberately OUTSIDE the null walk's
//! scope — see [`WALKED_SUBDOCUMENTS`].

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::Deserialize as _;
use serde_json::{Map as JsonMap, Value};

use super::tokenizer::sha256_hex;
use super::{EncoderArchitecture, SetFitMiniLm};
use crate::classification::MultinomialLogisticRegression;
use crate::format::v2::{
    AprV2Metadata, AprV2Reader, AprV2Writer, TensorDType, V2FormatError, VERSION_V2,
};

// ===========================================================================
// Contract-resident constants
// ===========================================================================

/// The schema identifier written into the doc's first-level `schema` field.
pub const ARTIFACT_SCHEMA: &str = "setfit-apr-v1";

/// The schema version written into the doc's first-level `schema_version` field.
pub const ARTIFACT_SCHEMA_VERSION: u32 = 1;

/// The ONLY typed container metadata key this writer sets (D-02(a) amendment,
/// D-04 detection). A SetFit-shaped tensor set without this tag is a plain APR.
pub const MODEL_TYPE_TAG: &str = "setfit";

/// The ONE custom metadata key — see the module docs for why N keys are unsafe.
pub const CUSTOM_METADATA_KEY: &str = "setfit";

/// Canonical name of the classifier weight tensor, `[num_labels, n_features]`.
pub const HEAD_WEIGHT_TENSOR: &str = "setfit.head.weight";

/// Canonical name of the classifier intercept tensor, `[num_labels]`.
pub const HEAD_BIAS_TENSOR: &str = "setfit.head.bias";

/// Canonical name of the raw `tokenizer.json` payload, dtype U8.
pub const TOKENIZER_BLOB_TENSOR: &str = "tokenizer.blob";

/// The normative `SetFitArtifactDoc` field list, in declaration order.
///
/// Read from `contracts/setfit-apr-v1.yaml` equation `artifact_doc_schema`
/// (review B2). A field missing from this list is a finding, not a
/// simplification; the doc-key-set test compares against this literal so an
/// added or renamed field is a LOUD failure rather than a silent widening.
pub const SETFIT_ARTIFACT_DOC_FIELDS: [&str; 16] = [
    "schema",
    "schema_version",
    "bundle_schema_version",
    "format_id",
    "architecture",
    "tokenizer_sha256",
    "preprocessing",
    "root_seed",
    "head",
    "ordered_labels",
    "requested_config",
    "resolved_config",
    "evidence",
    "provenance",
    "hf_name_map",
    "probes",
];

/// The FIVE embedded sub-documents the null walk visits.
///
/// # The walk is FIVE and the allowlist is FOUR, deliberately
///
/// `resolved_config` (`ResolvedConfigRecord`, one `String`) and `provenance`
/// (`ProvenanceRecord`, four `String` + one `u64` + one `u32`) contribute ZERO
/// allowlisted paths today, and they are WALKED ANYWAY. That is not redundancy;
/// it is the whole point. An UNWALKED subtree cannot reject anything, so a
/// future `Option` field added to either type would fail SILENTLY on the first
/// production artifact instead of loudly at this guard — the exact inversion of
/// the fail-loud asymmetry the allowlist design rests on (checker warning W-A).
///
/// The second half of the same guarantee lives in `aprender-train` (plan 04-13),
/// the only crate that can NAME all five types: its completeness gate asserts
/// that serializing an all-`None` instance of each produces exactly
/// [`NULLABLE_PATH_ALLOWLIST`], with `resolved_config` and `provenance` each
/// asserted BY NAME to contribute an empty set.
pub const WALKED_SUBDOCUMENTS: [&str; 5] = [
    "architecture",
    "requested_config",
    "resolved_config",
    "evidence",
    "provenance",
];

/// The four paths at which a `null` is LEGITIMATE, read from the contract.
///
/// # The derivation rule
///
/// This is exactly the set of paths at which an `Option`-typed field of the FIVE
/// [`WALKED_SUBDOCUMENTS`] can serialize. None of those five types carries
/// `skip_serializing_if`, so every `None` emits an EXPLICIT `null` the walk can
/// see. The list is NOT hand-picked, and no path may join it without the same
/// written analysis the contract requires: owning type, `Option<...>` field
/// type, source location, whether `None` is the production shape, and whether
/// the `Option` wraps a float.
///
/// | path | owning type | `Option` field type |
/// |------|-------------|---------------------|
/// | `architecture.vocab_remap` | `EncoderArchitecture` | `Option<Vec<u32>>` |
/// | `requested_config.pair_config.budget` | `PairConfigWire` | `Option<u64>` |
/// | `requested_config.pair_config.hard_cap` | `PairConfigWire` | `Option<u64>` |
/// | `evidence.epsilon_used` | `EvidenceSummary` | `Option<f64>` |
///
/// # Why this is an allowlisted NULL SCAN and not an enumerated float-path check
///
/// `serde_json` maps EVERY non-finite `f64` to `null` SILENTLY — `+inf`, `-inf`
/// and every `NaN` payload render identically — so a stray `null` IS the CR-03
/// signature. `UpdateEvidence::to_canonical_bytes` (evidence.rs:456-483) already
/// records, in its own words, why a scan beats a field list: "An enumerated field
/// list is the guard that silently stops covering the newest field, which is how
/// these checks rot." That guard's blanket form is EXACT only because
/// `UpdateEvidence` has no `Option` field. THREE of the five sub-documents here
/// DO have `Option` fields, so a blanket scan would refuse honest artifacts. This
/// allowlist is the minimum widening that keeps the guard armed.
///
/// # The residual, recorded rather than claimed away
///
/// At the three `Option<non-float>` paths a `null` is UNAMBIGUOUS: no `f64` can
/// produce a `null` at a `Vec<u32>` or `u64` field. At `evidence.epsilon_used`,
/// which is `Option<f64>`, `None` and a non-finite epsilon render IDENTICALLY and
/// this writer cannot separate them. Today that residual is EMPTY —
/// `epsilon_used` has exactly one assignment in the tree and it is `None`
/// (evidence.rs:655, asserted at evidence.rs:1144).
///
/// # `skip_serializing_if` on any of the five types is FORBIDDEN
///
/// It would change `SetFitBundle::to_canonical_bytes` output and break Phase 3's
/// committed closure tests, AND it would silently empty this allowlist, turning a
/// guarded schema into an unguarded one with no test turning red. The fix for a
/// new nullable field is a new allowlist entry plus 04-13's completeness gate.
pub const NULLABLE_PATH_ALLOWLIST: [&str; 4] = [
    // `EncoderArchitecture::vocab_remap: Option<Vec<u32>>`
    // (aprender-core/src/setfit/mod.rs:126-127). `None` is the FULL PIN — the
    // PRODUCTION shape (import.rs:501); `Some` occurs only on the slice-fixture
    // path (import.rs:620). An allowlist omitting this path would refuse EVERY
    // pinned MiniLM artifact while every slice-fixture test stayed green.
    "architecture.vocab_remap",
    // `PairConfigWire::budget: Option<u64>`
    // (aprender-train/src/train/setfit/config.rs:716).
    "requested_config.pair_config.budget",
    // `PairConfigWire::hard_cap: Option<u64>`
    // (aprender-train/src/train/setfit/config.rs:717).
    "requested_config.pair_config.hard_cap",
    // `EvidenceSummary::epsilon_used: Option<f64>`
    // (aprender-train/src/train/setfit/evidence.rs:599). The ONE allowlisted
    // path whose `Option` wraps a float — see the residual note above.
    "evidence.epsilon_used",
];

/// Number of contract-resident synthetic probes.
pub const PROBE_COUNT: usize = 6;

/// The unit `probe_truncation_boundary` repeats (note the trailing space).
pub const PROBE_TRUNCATION_REPEAT_UNIT: &str = "few shot classification with contrastive pairs ";

/// How many times [`PROBE_TRUNCATION_REPEAT_UNIT`] repeats, with no separator.
pub const PROBE_TRUNCATION_REPEAT_COUNT: usize = 64;

/// The contract's probe identifiers, in probe order.
pub const PROBE_IDS: [&str; PROBE_COUNT] = [
    "probe_ascii_pangram",
    "probe_unicode",
    "probe_truncation_boundary",
    "probe_minimal",
    "probe_social",
    "probe_whitespace",
];

/// The six FIXED, SYNTHETIC, contract-resident probe inputs, in probe order.
///
/// They are NEVER sampled from the dataset or the selection (Pitfall 8): the
/// train-time `VerifyProbe` rows come from the selection, and reusing that source
/// would embed TweetEval text in every shipped artifact — colliding with DATA-01's
/// no-vendored-corpus posture AND making probes dataset-dependent, hence useless
/// for Phase 5 cross-cell comparison.
///
/// The six cover six DIFFERENT failure shapes, not six samples of one: ASCII
/// baseline, multi-byte UTF-8, the truncation boundary, a minimal input,
/// social-media punctuation, and embedded control whitespace.
#[must_use]
pub fn probe_inputs() -> Vec<String> {
    vec![
        "the quick brown fox jumps over the lazy dog".to_string(),
        "El rapido zorro marron salta sobre el perro perezoso — naive cafe, pi = 3.14159"
            .to_string(),
        PROBE_TRUNCATION_REPEAT_UNIT.repeat(PROBE_TRUNCATION_REPEAT_COUNT),
        "ok".to_string(),
        "Stance detection: I firmly support this position!!! #debate @user123 https://example.com"
            .to_string(),
        "line one\nline two\ttabbed   spaced".to_string(),
    ]
}

// ---------------------------------------------------------------------------
// The canonical tensor-name table (contract equation `canonical_tensor_names`)
// ---------------------------------------------------------------------------

/// The five global `(hf_name, canonical_name)` pairs.
///
/// D-01 AMENDMENT (recorded in-contract, not silent): four of these five have NO
/// canonical form in `tensor-names-v1` — `token_type_embeddings` and both
/// `embeddings.LayerNorm` leaves have no role at all, and `position_embedding`
/// has a `bert:` alias whose `_fallback` list is EMPTY. `setfit-apr-v1` RESERVES
/// their names, following the same global convention so generic tooling sees one
/// coherent scheme, and the non-collision against `tensor-names-v1`'s complete
/// `_fallback` set was enumerated and checked rather than asserted.
const GLOBAL_NAME_TABLE: [(&str, &str); 5] = [
    // tensor-names-v1 global_roles.embedding _fallback
    ("embeddings.word_embeddings.weight", "token_embd.weight"),
    // SETFIT-SCHEMA-OWNED: position_embedding's _fallback is empty
    (
        "embeddings.position_embeddings.weight",
        "position_embd.weight",
    ),
    // SETFIT-SCHEMA-OWNED: no tensor-names-v1 role
    (
        "embeddings.token_type_embeddings.weight",
        "token_types.weight",
    ),
    // SETFIT-SCHEMA-OWNED: no tensor-names-v1 role
    ("embeddings.LayerNorm.weight", "token_embd_norm.weight"),
    // SETFIT-SCHEMA-OWNED: no tensor-names-v1 role
    ("embeddings.LayerNorm.bias", "token_embd_norm.bias"),
];

/// The sixteen per-layer `(hf_template, canonical_template)` pairs.
///
/// `{n}` is expanded over `architecture.num_layers` — the expected set is a
/// FUNCTION of the architecture, never a hardcoded six-layer list. That is what
/// makes the identical rule apply to the pinned model and to a reduced fixture.
///
/// Three of the sixteen are SETFIT-SCHEMA-OWNED under the same D-01 amendment:
/// `tensor-names-v1` has no `o_proj_bias`, `ffn_up_bias` or `ffn_down_bias` role.
const LAYER_NAME_TABLE: [(&str, &str); 16] = [
    (
        "encoder.layer.{n}.attention.self.query.weight",
        "blk.{n}.attn_q.weight",
    ),
    (
        "encoder.layer.{n}.attention.self.query.bias",
        "blk.{n}.attn_q.bias",
    ),
    (
        "encoder.layer.{n}.attention.self.key.weight",
        "blk.{n}.attn_k.weight",
    ),
    (
        "encoder.layer.{n}.attention.self.key.bias",
        "blk.{n}.attn_k.bias",
    ),
    (
        "encoder.layer.{n}.attention.self.value.weight",
        "blk.{n}.attn_v.weight",
    ),
    (
        "encoder.layer.{n}.attention.self.value.bias",
        "blk.{n}.attn_v.bias",
    ),
    (
        "encoder.layer.{n}.attention.output.dense.weight",
        "blk.{n}.attn_output.weight",
    ),
    // SETFIT-SCHEMA-OWNED: there is no o_proj_bias role
    (
        "encoder.layer.{n}.attention.output.dense.bias",
        "blk.{n}.attn_output.bias",
    ),
    (
        "encoder.layer.{n}.attention.output.LayerNorm.weight",
        "blk.{n}.attn_norm.weight",
    ),
    (
        "encoder.layer.{n}.attention.output.LayerNorm.bias",
        "blk.{n}.attn_norm.bias",
    ),
    (
        "encoder.layer.{n}.intermediate.dense.weight",
        "blk.{n}.ffn_up.weight",
    ),
    // SETFIT-SCHEMA-OWNED: there is no ffn_up_bias role
    (
        "encoder.layer.{n}.intermediate.dense.bias",
        "blk.{n}.ffn_up.bias",
    ),
    (
        "encoder.layer.{n}.output.dense.weight",
        "blk.{n}.ffn_down.weight",
    ),
    // SETFIT-SCHEMA-OWNED: there is no ffn_down_bias role
    (
        "encoder.layer.{n}.output.dense.bias",
        "blk.{n}.ffn_down.bias",
    ),
    (
        "encoder.layer.{n}.output.LayerNorm.weight",
        "blk.{n}.ffn_norm.weight",
    ),
    (
        "encoder.layer.{n}.output.LayerNorm.bias",
        "blk.{n}.ffn_norm.bias",
    ),
];

// ===========================================================================
// The view: the writer's complete input
// ===========================================================================

/// Everything `write_setfit_apr` is allowed to know.
///
/// The field list mirrors `SetFitBundle`'s TWENTY fields 1:1 so plan 04-05's
/// codec can populate it without inventing anything (review B3). Everything in
/// the artifact MUST be derivable from this value — no clock, no environment, no
/// host name, no lookup — because anything else breaks the closure equation.
///
/// # `architecture` is the TYPED record, and the sub-document is derived from it
///
/// The contract's forward bijection row is `architecture = to_value(bundle.architecture)`
/// — a FUNCTION of the typed value. Carrying both the typed record and a
/// separately-supplied `serde_json::Value` copy would be transporting one fact
/// twice, and two copies of one fact are two values that can disagree: the codec
/// would have to keep them in sync, and a divergence would break the bijection
/// with nothing turning red. So the view carries the typed record and the writer
/// computes `serde_json::to_value` itself. A pleasant consequence: the ONLY
/// `null` the `architecture` subtree can emit is the allowlisted
/// `vocab_remap`, because every other field is non-`Option`. The subtree is
/// walked anyway, so a future `Option` there is caught — see
/// [`WALKED_SUBDOCUMENTS`].
#[derive(Debug, Clone)]
pub struct SetFitArtifactView {
    /// The `SetFitBundle` wire version this artifact is written from.
    pub bundle_schema_version: u32,
    /// The codec identifier that wrote the payload.
    pub format_id: String,
    /// The encoder architecture record — typed, so the rebuild can use it.
    pub architecture: EncoderArchitecture,
    /// The exact `tokenizer.json` bytes, byte-identical to upstream.
    pub tokenizer_bytes: Vec<u8>,
    /// The pooling policy identifier the encoder applied.
    pub pooling: String,
    /// The normalization policy identifier the encoder applied.
    pub normalization: String,
    /// The epsilon the L2 normalization clamped with.
    pub l2_epsilon: f32,
    /// The tokenizer's truncation bound.
    pub truncation_max_sequence_length: u32,
    /// The tokenizer's padding mode.
    pub padding_mode: String,
    /// The run's requested max sequence length.
    pub max_length: u32,
    /// The root seed every dropout stream derives from.
    pub root_seed: u64,
    /// Every named encoder tensor, keyed by HF dotted name, `(shape, data)`.
    pub tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    /// The head's `K * d` weights, row-major; row `i` belongs to `ordered_labels[i]`.
    pub head_weights: Vec<f32>,
    /// The head's `K` intercepts.
    pub head_intercepts: Vec<f32>,
    /// The head's fitted feature dimension.
    pub head_n_features: usize,
    /// The ordered labels the head's weight rows are indexed by.
    pub ordered_labels: Vec<String>,
    /// `to_value(SetFitTrainConfig)` — opaque here; `aprender-core` cannot name it.
    pub requested_config: Value,
    /// `to_value(ResolvedConfigRecord)` — opaque here.
    pub resolved_config: Value,
    /// `to_value(EvidenceSummary)` — opaque here.
    pub evidence: Value,
    /// `to_value(ProvenanceRecord)` — opaque here (bundle field 20, plan 04-13).
    pub provenance: Value,
}

// ===========================================================================
// The typed failure
// ===========================================================================

/// A `setfit-apr-v1` write (and, from 04-03, load) failure.
///
/// `Debug + Clone + PartialEq` is REQUIRED, not incidental: plan 04-05 embeds
/// this type inside `CodecError`, which derives exactly those three. A variant
/// carrying a non-`Clone` payload would make the typed-source mapping impossible
/// and force the stringification review B3 flagged.
///
/// `#[non_exhaustive]`: 04-03 adds the loader's rungs to this same enum, so a
/// downstream `match` must already be written to tolerate new variants.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum SetFitArtifactError {
    /// A supplied HF tensor name has no entry in the architecture-derived map.
    UnmappedTensorName {
        /// The HF dotted name with no canonical form.
        hf_name: String,
    },

    /// The view is missing tensors the architecture-derived set requires.
    ///
    /// Names the MISSING tensors, not merely the two counts: "104 != 103" is not
    /// a diagnosis.
    IncompleteTensorSet {
        /// The expected HF dotted names that the view does not carry, sorted.
        missing: Vec<String>,
    },

    /// The view's parts contradict one another (shape vs element count, head
    /// arity vs label count, head feature dimension vs encoder width, a
    /// canonical-name collision).
    InconsistentTensorSet {
        /// What disagreed with what.
        reason: String,
    },

    /// A `NaN`/`±Inf` float, or a `null` at a path outside
    /// [`NULLABLE_PATH_ALLOWLIST`], was found BEFORE serialization.
    ///
    /// `path` names the exact dotted path of the FIRST offender in document
    /// order, so the diagnosis does not require reading the contract.
    NonFiniteValue {
        /// Dotted path of the offending value.
        path: String,
    },

    /// The tokenizer bytes do not hash to the digest the architecture records.
    ///
    /// An artifact whose tokenizer does not match its encoder produces
    /// confidently wrong embeddings for every input while looking structurally
    /// valid the whole time.
    TokenizerHashMismatch {
        /// The digest the architecture record claims.
        expected: String,
        /// The digest of the bytes actually supplied.
        got: String,
    },

    /// The APR v2 container refused the write.
    ContainerWrite {
        /// The container's own diagnostic.
        reason: String,
    },

    /// A probe could not be computed: the rebuild, the head or the encode failed.
    ProbeComputation {
        /// Which probe (or `<rebuild>` / `<head>` for the shared setup).
        probe: String,
        /// The underlying diagnostic, forwarded rather than flattened.
        reason: String,
    },

    // -----------------------------------------------------------------------
    // The loader's rungs (plan 04-03). One variant per corruption class, so a
    // refusal names the RUNG and what it observed — "invalid artifact" is not a
    // diagnosis (contract `load_validation_ladder`).
    // -----------------------------------------------------------------------
    /// Rungs 1-2, the two length bounds: an over-cap length.
    ///
    /// `what` distinguishes the three checks the contract requires — a
    /// `declared_length` refusal happened BEFORE a byte was read, a `stream`
    /// refusal means the declared length lied, and an `input_bytes` refusal is
    /// the in-memory door's own defense-in-depth check.
    ArtifactTooLarge {
        /// `declared_length`, `stream` or `input_bytes`.
        what: &'static str,
        /// The length observed at that check.
        observed: u64,
        /// The cap it was judged against.
        cap: u64,
    },

    /// The bounded read's underlying source failed.
    ArtifactRead {
        /// The I/O diagnostic, forwarded rather than flattened.
        reason: String,
    },

    /// Rung 3: magic, version, header CRC, row-major flag or footer CRC.
    ///
    /// CRC32 is not cryptographic, so this rung cannot see semantic corruption —
    /// that is what rung 8's probe replay is for.
    ContainerIntegrity {
        /// `magic`, `container_version`, `header_checksum`, `row_major_flag`,
        /// `footer_length`, `footer_checksum` or `container_parse`.
        what: &'static str,
        /// What was observed, in full.
        reason: String,
    },

    /// Rung 4: the container is not tagged as a SetFit artifact (D-04).
    ///
    /// Detection is EXPLICIT-TAG-ONLY: a SetFit-shaped tensor set without the
    /// typed `model_type` key is a plain APR and is refused here.
    NotASetFitArtifact {
        /// The `model_type` the container declares.
        model_type: String,
    },

    /// Rung 4: the ONE custom metadata document is absent or unusable.
    ArtifactDocumentMissing {
        /// What was expected and what was found.
        reason: String,
    },

    /// Rung 4: the document declares a schema identifier this build does not own.
    UnsupportedSchema {
        /// The identifier found.
        got: String,
        /// The identifier this build reads and writes.
        supported: &'static str,
    },

    /// Rung 4: the document declares a schema version this build does not implement.
    ///
    /// Checked BEFORE any other field is read, so a future schema is refused
    /// rather than partially interpreted by this build.
    UnsupportedSchemaVersion {
        /// The version found.
        got: u64,
        /// The version this build implements.
        supported: u32,
    },

    /// Rung 4: the document did not parse under `deny_unknown_fields`.
    ArtifactDocumentParse {
        /// serde's diagnostic, including the position when it has one.
        detail: String,
    },

    /// Rung 5: ONE index entry contradicts its own declared shape, dtype or size.
    ///
    /// Distinct from [`Self::InconsistentTensorSet`], which is a SET-level
    /// disagreement: "this tensor lies about itself" and "the collection is the
    /// wrong collection" are two different operator errors.
    InconsistentTensor {
        /// The tensor that contradicts itself.
        tensor: String,
        /// What disagreed with what.
        reason: String,
    },

    /// Rung 5: the document's carried `hf_name_map` is not usable as an inversion.
    InconsistentNameMap {
        /// What was wrong: not injective, not total, or naming an absent tensor.
        reason: String,
    },

    /// Rung 7: the encoder, tokenizer or head could not be rebuilt.
    ArtifactRebuildFailed {
        /// `encoder` or `head`.
        what: &'static str,
        /// The underlying diagnostic, forwarded rather than flattened.
        reason: String,
    },

    /// Rung 8: a probe did not replay within tolerance.
    ///
    /// This is the rung a checksum cannot reach: corrupted-but-checksummed
    /// states, wrong-loader states and platform math divergence all arrive here.
    ///
    /// The payload is BOXED. Inlined, its seven fields (four of them `String`)
    /// would make `SetFitArtifactError` 136 bytes, and every
    /// `Result<_, SetFitArtifactError>` in this module — including
    /// `write_setfit_apr`'s, which returns a `Vec<u8>` on success — would pay
    /// that width on its SUCCESS path (`clippy::result_large_err`, 32 sites). One
    /// heap allocation on the failure path is the right trade.
    ProbeReplayFailed(Box<ProbeReplayDivergence>),

    /// [`VerifiedSetFitModel::embed`] was handed an empty batch.
    ///
    /// A typed refusal rather than an empty result: "embed nothing" is a caller
    /// mistake, and returning `Ok(vec![])` would let it travel silently.
    EmptyEmbedBatch,

    /// [`VerifiedSetFitModel::embed`]'s tokenize-or-encode step failed.
    EncodeFailed {
        /// The underlying diagnostic, forwarded rather than flattened.
        reason: String,
    },
}

/// Everything a rung-8 probe divergence reports.
///
/// A named struct behind [`SetFitArtifactError::ProbeReplayFailed`]'s `Box`, so
/// the diagnosis stays STRUCTURED — an operator gets the probe, the component and
/// both values rather than one flattened sentence — without widening every
/// `Result` in the module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeReplayDivergence {
    /// The probe's index in the artifact's probe array.
    pub probe: usize,
    /// The contract's identifier for that probe.
    pub probe_id: String,
    /// `probe_count`, `input`, `embedding_width`, `embedding`, `logit_count`,
    /// `logit`, `probability_count`, `probability` or `label`.
    pub component: &'static str,
    /// The component's index inside the probe, or 0 where there is none.
    pub index: usize,
    /// The artifact's recorded expectation.
    pub expected: String,
    /// What this process actually produced.
    pub observed: String,
    /// The bound the comparison used, formatted; `exact` where none exists.
    pub tolerance: String,
}

impl std::fmt::Display for SetFitArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnmappedTensorName { hf_name } => write!(
                f,
                "SetFitArtifactError::UnmappedTensorName({hf_name} has no canonical form in the \
                 architecture-derived name map)"
            ),
            Self::IncompleteTensorSet { missing } => write!(
                f,
                "SetFitArtifactError::IncompleteTensorSet(missing {} tensor(s): {})",
                missing.len(),
                missing.join(", ")
            ),
            Self::InconsistentTensorSet { reason } => {
                write!(f, "SetFitArtifactError::InconsistentTensorSet({reason})")
            }
            Self::NonFiniteValue { path } => write!(
                f,
                "SetFitArtifactError::NonFiniteValue(at {path}; a null outside the four-path \
                 allowlist is a silently destroyed non-finite value)"
            ),
            Self::TokenizerHashMismatch { expected, got } => write!(
                f,
                "SetFitArtifactError::TokenizerHashMismatch(expected {expected}, got {got})"
            ),
            Self::ContainerWrite { reason } => {
                write!(f, "SetFitArtifactError::ContainerWrite({reason})")
            }
            Self::ProbeComputation { probe, reason } => {
                write!(
                    f,
                    "SetFitArtifactError::ProbeComputation({probe}: {reason})"
                )
            }
            Self::ArtifactTooLarge {
                what,
                observed,
                cap,
            } => write!(
                f,
                "SetFitArtifactError::ArtifactTooLarge(check {what} observed {observed} bytes \
                 against the {cap}-byte cap; the payload is refused before the allocation it \
                 would have requested)"
            ),
            Self::ArtifactRead { reason } => {
                write!(f, "SetFitArtifactError::ArtifactRead({reason})")
            }
            Self::ContainerIntegrity { what, reason } => write!(
                f,
                "SetFitArtifactError::ContainerIntegrity(rung 3, {what}: {reason})"
            ),
            Self::NotASetFitArtifact { model_type } => write!(
                f,
                "SetFitArtifactError::NotASetFitArtifact(rung 4: model_type is {model_type:?}, \
                 not \"setfit\"; detection is explicit-tag-only, so a SetFit-shaped tensor set \
                 without the tag is a plain APR)"
            ),
            Self::ArtifactDocumentMissing { reason } => write!(
                f,
                "SetFitArtifactError::ArtifactDocumentMissing(rung 4: {reason})"
            ),
            Self::UnsupportedSchema { got, supported } => write!(
                f,
                "SetFitArtifactError::UnsupportedSchema(rung 4: document declares schema {got:?}, \
                 this build owns {supported:?})"
            ),
            Self::UnsupportedSchemaVersion { got, supported } => write!(
                f,
                "SetFitArtifactError::UnsupportedSchemaVersion(rung 4: document declares version \
                 {got}, this build implements {supported}; a document from a different schema is \
                 refused rather than partially interpreted)"
            ),
            Self::ArtifactDocumentParse { detail } => write!(
                f,
                "SetFitArtifactError::ArtifactDocumentParse(rung 4, deny_unknown_fields: {detail})"
            ),
            Self::InconsistentTensor { tensor, reason } => write!(
                f,
                "SetFitArtifactError::InconsistentTensor(rung 5, {tensor}: {reason})"
            ),
            Self::InconsistentNameMap { reason } => write!(
                f,
                "SetFitArtifactError::InconsistentNameMap(rung 5: {reason})"
            ),
            Self::ArtifactRebuildFailed { what, reason } => write!(
                f,
                "SetFitArtifactError::ArtifactRebuildFailed(rung 7, {what}: {reason})"
            ),
            Self::ProbeReplayFailed(divergence) => write!(
                f,
                "SetFitArtifactError::ProbeReplayFailed(rung 8, probe {} ({}), {}[{}]: expected \
                 {}, observed {}, tolerance {})",
                divergence.probe,
                divergence.probe_id,
                divergence.component,
                divergence.index,
                divergence.expected,
                divergence.observed,
                divergence.tolerance
            ),
            Self::EmptyEmbedBatch => write!(
                f,
                "SetFitArtifactError::EmptyEmbedBatch(embed was handed no texts)"
            ),
            Self::EncodeFailed { reason } => {
                write!(f, "SetFitArtifactError::EncodeFailed({reason})")
            }
        }
    }
}

impl std::error::Error for SetFitArtifactError {}

// ===========================================================================
// Public API
// ===========================================================================

/// Lowercase-hex SHA-256 of artifact bytes.
///
/// A TRUSTED free function in core: the codec must never hash its own output
/// (verify.rs:198-205 discipline), so the hash a run records and the hash a
/// consumer recomputes come from ONE implementation. It delegates to the same
/// `sha256_hex` the tokenizer digest uses, so there is exactly one hashing path
/// in this crate.
#[must_use]
pub fn artifact_sha256_hex(bytes: &[u8]) -> String {
    sha256_hex(bytes)
}

/// Write a complete view as `setfit-apr-v1` bytes.
///
/// # Order is load-bearing
///
/// 1. derive the architecture-expected canonical set and validate the view's
///    tensor names map onto it 1:1;
/// 2. scan every `f32` the view carries (tensors, head weights, head intercepts);
/// 3. check the tokenizer bytes hash to the digest the architecture records —
///    BEFORE the rebuild, so a probe mismatch can never be a mis-paired
///    tokenizer wearing a math-divergence diagnosis;
/// 4. compute the six probes from the view's OWN parts;
/// 5. build the one-key document;
/// 6. walk all FIVE embedded sub-documents for `null`s outside the allowlist;
/// 7. hand the tensors and the document to the container.
///
/// These are STEPS, deliberately not "rungs". The rung vocabulary is
/// contract-normative for the LOADER ladder, where rung 5 is structure and rung 8
/// is probe replay. Reusing those numbers here for different checks made "rung 5"
/// mean two things in one file.
///
/// Nothing is written until every rung has passed: there is no partial artifact.
///
/// # Errors
///
/// [`SetFitArtifactError`], each variant naming the specific tensor, path, probe
/// or digest that failed.
pub fn write_setfit_apr(view: &SetFitArtifactView) -> Result<Vec<u8>, SetFitArtifactError> {
    // (1) NAMES. The expected set is a FUNCTION of the view's own declared
    //     architecture, so the identical rule judges the pinned model and a
    //     reduced fixture.
    let hf_name_map = build_hf_name_map(view.architecture.num_layers);
    validate_view_structure(view, &hf_name_map)?;

    // (2) EVERY f32 THE VIEW CARRIES, before any of it can reach a payload.
    scan_view_floats(view)?;

    // (3) TOKENIZER IDENTITY, BEFORE THE REBUILD. An encoder rebuilt with a
    //     substituted tokenizer produces confidently wrong embeddings for every
    //     input and looks structurally valid the whole time — so a probe
    //     mismatch must never be able to arrive wearing a math-divergence
    //     diagnosis when it is really a mis-paired tokenizer.
    let observed = sha256_hex(&view.tokenizer_bytes);
    if observed != view.architecture.tokenizer_sha256 {
        return Err(SetFitArtifactError::TokenizerHashMismatch {
            expected: view.architecture.tokenizer_sha256.clone(),
            got: observed,
        });
    }

    // (4) PROBES, computed from the view's OWN parts — never carried, which is
    //     what makes them closure-safe (a carried probe set would be a 21st
    //     bundle field with no source).
    let probes = compute_probes(view)?;

    // (5) THE ONE DOCUMENT.
    let doc = build_artifact_doc(view, &hf_name_map, &probes)?;

    // (6) THE FIVE-SUBDOCUMENT NULL WALK. Runs at WRITE time and not only as a
    //     byte comparison: an allowlisted `null` round-trips to `None` and back
    //     to `null`, so closure HOLDS while the value is GONE.
    guard_subdocument_nulls(&doc)?;

    // (7) THE CONTAINER. Nothing has been written until here, so a refusal at
    //     any rung above leaves no partial artifact.
    write_container(view, &hf_name_map, doc)
}

/// Step 1 of [`write_setfit_apr`]: the view's tensor names map 1:1 onto the
/// architecture-derived set,
/// and its parts do not contradict one another.
fn validate_view_structure(
    view: &SetFitArtifactView,
    hf_name_map: &BTreeMap<String, String>,
) -> Result<(), SetFitArtifactError> {
    // An HF name with no canonical entry is a typed error, NEVER a silently
    // dropped tensor.
    for hf in view.tensors.keys() {
        if !hf_name_map.contains_key(hf) {
            return Err(SetFitArtifactError::UnmappedTensorName {
                hf_name: hf.clone(),
            });
        }
    }
    let missing: Vec<String> = hf_name_map
        .keys()
        .filter(|expected| !view.tensors.contains_key(*expected))
        .cloned()
        .collect();
    if !missing.is_empty() {
        return Err(SetFitArtifactError::IncompleteTensorSet { missing });
    }
    // Injectivity: two HF names resolving to one canonical name would OVERWRITE
    // a tensor in the container's index rather than fail.
    let canonical: BTreeSet<&String> = hf_name_map.values().collect();
    if canonical.len() != hf_name_map.len() {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "the canonical name map is not injective: {} HF names resolve to {} canonical names",
                hf_name_map.len(),
                canonical.len()
            ),
        });
    }
    // The structural per-entry rule, applied to the DECLARED shape before any
    // payload is written.
    for (hf, (shape, data)) in &view.tensors {
        if shape.is_empty() {
            return Err(SetFitArtifactError::InconsistentTensorSet {
                reason: format!("{hf}: a tensor with no declared shape is not writable"),
            });
        }
        let elements: usize = shape.iter().product();
        if elements != data.len() {
            return Err(SetFitArtifactError::InconsistentTensorSet {
                reason: format!(
                    "{hf}: shape {shape:?} implies {elements} elements but {} were supplied",
                    data.len()
                ),
            });
        }
    }

    let num_labels = view.ordered_labels.len();
    if num_labels < 2 {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!("a classifier head needs at least two labels, got {num_labels}"),
        });
    }
    // The head must be able to CONSUME this encoder's embedding. A head fitted
    // at a different width would refuse every probe at replay time, in a fresh
    // process, long after the artifact shipped.
    if view.head_n_features != view.architecture.hidden {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "head_n_features {} does not match the encoder's hidden width {}",
                view.head_n_features, view.architecture.hidden
            ),
        });
    }
    let expected_weights = num_labels.saturating_mul(view.head_n_features);
    if view.head_weights.len() != expected_weights {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "setfit.head.weight declares [{num_labels}, {}] = {expected_weights} values but {} were supplied",
                view.head_n_features,
                view.head_weights.len()
            ),
        });
    }
    if view.head_intercepts.len() != num_labels {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "setfit.head.bias declares [{num_labels}] values but {} were supplied",
                view.head_intercepts.len()
            ),
        });
    }
    Ok(())
}

/// Step 2 of [`write_setfit_apr`]: every `f32` the view carries is finite, named
/// by its exact path.
fn scan_view_floats(view: &SetFitArtifactView) -> Result<(), SetFitArtifactError> {
    for (hf, (_, data)) in &view.tensors {
        for (index, value) in data.iter().enumerate() {
            if !value.is_finite() {
                return Err(SetFitArtifactError::NonFiniteValue {
                    path: format!("tensors.{hf}[{index}]"),
                });
            }
        }
    }
    for (array, values) in [
        ("head_weights", &view.head_weights),
        ("head_intercepts", &view.head_intercepts),
    ] {
        for (index, value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(SetFitArtifactError::NonFiniteValue {
                    path: format!("{array}[{index}]"),
                });
            }
        }
    }
    if !view.l2_epsilon.is_finite() {
        return Err(SetFitArtifactError::NonFiniteValue {
            path: "preprocessing.l2_epsilon".to_string(),
        });
    }
    Ok(())
}

/// Step 4 of [`write_setfit_apr`]: the six contract-resident probes, replayed
/// through a model rebuilt
/// from the view's own parts.
///
/// The tensor map is CLONED because `from_bundle_parts` takes ownership and
/// drains it (encoder.rs:408-419, where taking by value is what avoids a
/// per-tensor copy on the reload path), while this writer only borrows the view.
/// That is a real cost on a full pin and it is the right trade: computing the
/// probes from anything other than the view's OWN tensors would record
/// expectations for a model the artifact does not contain.
fn compute_probes(view: &SetFitArtifactView) -> Result<Vec<Value>, SetFitArtifactError> {
    let model = SetFitMiniLm::from_bundle_parts(
        &view.tokenizer_bytes,
        &view.architecture,
        view.tensors.clone(),
        view.root_seed,
    )
    .map_err(|e| SetFitArtifactError::ProbeComputation {
        probe: "<rebuild>".to_string(),
        reason: e.to_string(),
    })?;
    let head = MultinomialLogisticRegression::from_stored_coefficients(
        view.ordered_labels.clone(),
        view.head_n_features,
        view.head_weights.clone(),
        view.head_intercepts.clone(),
    )
    .map_err(|e| SetFitArtifactError::ProbeComputation {
        probe: "<head>".to_string(),
        reason: e.to_string(),
    })?;

    let d = view.head_n_features;
    let mut records = Vec::with_capacity(PROBE_COUNT);

    for (index, input) in probe_inputs().into_iter().enumerate() {
        let id = PROBE_IDS[index];
        let fail = |reason: String| SetFitArtifactError::ProbeComputation {
            probe: id.to_string(),
            reason,
        };

        let pooled = model
            .encode_texts(&[input.as_str()])
            .map_err(|e| fail(e.to_string()))?;
        if pooled.shape().to_vec() != vec![1, d] {
            return Err(fail(format!(
                "the encode produced shape {:?}, expected [1, {d}]",
                pooled.shape()
            )));
        }
        let embedding: Vec<f32> = pooled.data().to_vec();
        for (position, value) in embedding.iter().enumerate() {
            if !value.is_finite() {
                return Err(SetFitArtifactError::NonFiniteValue {
                    path: format!("probes.{index}.embedding_hex.{position}"),
                });
            }
        }

        // The recorded logits come from `predict_logits` — THE single logit
        // implementation — rather than a second accumulation written here, so the
        // recorded logits and the recorded probabilities below cannot describe two
        // different computations. `predict_proba` is literally `predict_logits` +
        // softmax, so they now share the accumulation rather than agreeing by hand.
        // Only the narrowing to `f32` for storage happens here; it is deterministic,
        // and the contract's replay tolerance (1.0e-5 absolute) is orders above
        // `f32` epsilon at these magnitudes.
        let logits: Vec<f32> = head
            .predict_logits(&[embedding.clone()])
            .map_err(|e| fail(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| fail("predict_logits returned no rows".to_string()))?
            .into_iter()
            .map(|z| z as f32)
            .collect();

        let probabilities: Vec<f32> = head
            .predict_proba(&[embedding.clone()])
            .map_err(|e| fail(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| fail("predict_proba returned no rows".to_string()))?
            .into_iter()
            .map(|p| p as f32)
            .collect();

        for (field, values) in [
            ("logits_hex", &logits),
            ("probabilities_hex", &probabilities),
        ] {
            for (position, value) in values.iter().enumerate() {
                if !value.is_finite() {
                    return Err(SetFitArtifactError::NonFiniteValue {
                        path: format!("probes.{index}.{field}.{position}"),
                    });
                }
            }
        }

        // The label goes through the head's own `predict`, so the tie-break
        // (lowest index on an exact tie) is the house rule and not a second one
        // written here.
        let label = head
            .predict(&[embedding.clone()])
            .map_err(|e| fail(e.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| fail("predict returned no rows".to_string()))?;

        let mut record = JsonMap::new();
        record.insert("input".to_string(), Value::String(input));
        record.insert(
            "embedding_hex".to_string(),
            Value::Array(f32_slice_hex(&embedding)),
        );
        record.insert(
            "logits_hex".to_string(),
            Value::Array(f32_slice_hex(&logits)),
        );
        record.insert(
            "probabilities_hex".to_string(),
            Value::Array(f32_slice_hex(&probabilities)),
        );
        record.insert("label".to_string(), Value::String(label));
        records.push(Value::Object(record));
    }
    Ok(records)
}

/// Step 7 of [`write_setfit_apr`]: hand the tensors and the ONE document to the
/// APR v2 container.
fn write_container(
    view: &SetFitArtifactView,
    hf_name_map: &BTreeMap<String, String>,
    doc: JsonMap<String, Value>,
) -> Result<Vec<u8>, SetFitArtifactError> {
    let mut custom: HashMap<String, Value> = HashMap::with_capacity(1);
    custom.insert(CUSTOM_METADATA_KEY.to_string(), Value::Object(doc));

    let metadata = AprV2Metadata {
        model_type: MODEL_TYPE_TAG.to_string(),
        // NO TIMESTAMP IS WRITTEN — see the module docs. Written explicitly
        // rather than left to `Default` so the choice is visible at the site
        // that makes it.
        created_at: None,
        custom,
        ..Default::default()
    };

    // `AprV2Writer::new` already sets LAYOUT_ROW_MAJOR (writer.rs:30-39); there
    // is no GGUF import path into this writer, so there is no transpose at this
    // boundary and no column-major kernel may ever be pointed at these tensors.
    let mut writer = AprV2Writer::new(metadata);
    for (hf, (shape, data)) in &view.tensors {
        let canonical =
            hf_name_map
                .get(hf)
                .ok_or_else(|| SetFitArtifactError::UnmappedTensorName {
                    hf_name: hf.clone(),
                })?;
        writer.add_f32_tensor(canonical.clone(), shape.clone(), data);
    }
    let num_labels = view.ordered_labels.len();
    writer.add_f32_tensor(
        HEAD_WEIGHT_TENSOR,
        vec![num_labels, view.head_n_features],
        &view.head_weights,
    );
    writer.add_f32_tensor(HEAD_BIAS_TENSOR, vec![num_labels], &view.head_intercepts);
    // dtype U8 makes the tokenizer's byte-exactness STRUCTURAL: there is no
    // float path it could be rounded through.
    writer.add_tensor(
        TOKENIZER_BLOB_TENSOR,
        TensorDType::U8,
        vec![view.tokenizer_bytes.len()],
        view.tokenizer_bytes.clone(),
    );

    writer
        .write()
        .map_err(|e| SetFitArtifactError::ContainerWrite {
            reason: e.to_string(),
        })
}

// ===========================================================================
// Name derivation
// ===========================================================================

/// The architecture-derived HF -> canonical tensor-name map.
///
/// A FUNCTION of `num_layers`: the sixteen per-layer templates are expanded over
/// `0..num_layers`, so the same rule judges the pinned six-layer model and a
/// reduced fixture. There is no test-only schema exception.
#[must_use]
pub fn build_hf_name_map(num_layers: usize) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for (hf, canonical) in GLOBAL_NAME_TABLE {
        map.insert(hf.to_string(), canonical.to_string());
    }
    for n in 0..num_layers {
        let index = n.to_string();
        for (hf, canonical) in LAYER_NAME_TABLE {
            map.insert(hf.replace("{n}", &index), canonical.replace("{n}", &index));
        }
    }
    map
}

/// Every tensor name the container is expected to carry, including the three
/// schema-owned entries. `|expected| = 5 + 16 * num_layers + 3`.
#[must_use]
pub fn expected_tensor_names(num_layers: usize) -> BTreeSet<String> {
    expected_tensor_names_from(&build_hf_name_map(num_layers))
}

/// [`expected_tensor_names`] over a map the caller has ALREADY expanded.
///
/// The composition rule — encoder names plus the three schema-owned ones — lives
/// here once; the public door above is that rule applied to a fresh expansion,
/// and rung 5 is that same rule applied to the one expansion it already made.
fn expected_tensor_names_from(derived: &BTreeMap<String, String>) -> BTreeSet<String> {
    let mut names: BTreeSet<String> = derived.values().cloned().collect();
    names.insert(HEAD_WEIGHT_TENSOR.to_string());
    names.insert(HEAD_BIAS_TENSOR.to_string());
    names.insert(TOKENIZER_BLOB_TENSOR.to_string());
    names
}

/// The canonical form of ONE HF dotted name under an architecture of
/// `num_layers` layers, or `None` if the name has no canonical entry.
///
/// A convenience over [`build_hf_name_map`], deliberately implemented BY calling
/// it rather than by re-parsing `encoder.layer.{n}.` prefixes: a second
/// implementation of the mapping would be a second table to keep in step, and
/// the whole point of `canonical_tensor_names` is that there is one.
#[must_use]
pub fn canonical_name_for_hf(hf_name: &str, num_layers: usize) -> Option<String> {
    build_hf_name_map(num_layers).remove(hf_name)
}

// ===========================================================================
// Bit-pattern hex (never decimal text)
// ===========================================================================

/// One `f32` as the lowercase hex of its LITTLE-ENDIAN bit pattern.
///
/// Byte order matches `bundle.rs`'s `f32_to_hex` EXACTLY (bundle.rs:699-715,
/// `hex::encode(value.to_bits().to_le_bytes())`), because the contract's forward
/// bijection row is `preprocessing.l2_epsilon_hex = f32_to_hex(bundle.l2_epsilon)`
/// — a different byte order here would make the doc and the bundle disagree about
/// the same number while both looked like "the bit pattern in hex".
///
/// Decimal text is not the identity on `f32`, and a `null` from a non-finite
/// value would be indistinguishable from a missing one; hex keeps every
/// doc-OWNED float exact and off the null path entirely.
fn f32_bits_hex(value: f32) -> String {
    let mut out = String::with_capacity(8);
    for byte in value.to_bits().to_le_bytes() {
        out.push(nibble_hex(byte >> 4));
        out.push(nibble_hex(byte & 0x0F));
    }
    out
}

fn nibble_hex(n: u8) -> char {
    match n {
        0..=9 => char::from(b'0' + n),
        // The caller masks to 4 bits, so 10..=15 is the only other reachable
        // range; the arm is written as a catch-all rather than 10..=15 plus an
        // unreachable arm, because an unreachable arm here would be a panic
        // path in a codec whose contract is that it does not panic.
        _ => char::from(b'a' + (n & 0x0F) - 10),
    }
}

/// A slice of `f32` as an array of bit-pattern hex strings.
fn f32_slice_hex(values: &[f32]) -> Vec<Value> {
    values
        .iter()
        .map(|v| Value::String(f32_bits_hex(*v)))
        .collect()
}

// ===========================================================================
// The null walk (contract equation `nullable_path_allowlist`)
// ===========================================================================

/// Join one path segment onto a prefix.
///
/// The `first_null_path` / `join_path` shape of evidence.rs:199-224, inverted to
/// build TOP-DOWN: that precedent returns only the FIRST offender and assembles
/// the path on the way back up, while this walk must collect EVERY offender so a
/// rejection can be reasoned about and the accept tests can assert the exact
/// observed set.
fn join_path(prefix: &str, segment: &str) -> String {
    if prefix.is_empty() {
        segment.to_string()
    } else {
        format!("{prefix}.{segment}")
    }
}

/// Collect the dotted path of EVERY `Value::Null` under `value`, in document order.
fn collect_null_paths(prefix: &str, value: &Value, out: &mut Vec<String>) {
    match value {
        Value::Null => out.push(prefix.to_string()),
        Value::Object(map) => {
            for (key, child) in map {
                collect_null_paths(&join_path(prefix, key), child, out);
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                collect_null_paths(&join_path(prefix, &index.to_string()), child, out);
            }
        }
        _ => {}
    }
}

/// Every `null` path observed across the FIVE walked sub-documents, allowlisted
/// or not, in document order.
///
/// A missing sub-document contributes nothing rather than panicking: the doc
/// builder is the only producer and it always inserts all five, and the
/// `SETFIT_ARTIFACT_DOC_FIELDS` test is what proves that.
fn observed_null_paths(doc: &JsonMap<String, Value>) -> Vec<String> {
    let mut out = Vec::new();
    for name in WALKED_SUBDOCUMENTS {
        if let Some(sub) = doc.get(name) {
            collect_null_paths(name, sub, &mut out);
        }
    }
    out
}

/// The observed `null` paths that are NOT in [`NULLABLE_PATH_ALLOWLIST`].
fn disallowed_null_paths(doc: &JsonMap<String, Value>) -> Vec<String> {
    observed_null_paths(doc)
        .into_iter()
        .filter(|path| !NULLABLE_PATH_ALLOWLIST.contains(&path.as_str()))
        .collect()
}

/// The FIRST `null` path outside [`NULLABLE_PATH_ALLOWLIST`], in document order.
///
/// The contract's `nullable_path_allowlist` equation names this concept and the
/// binding registry names this function; it is the whole decision the guard
/// makes. The walk collects EVERY offender first — so the full set is available
/// to a caller and the accept tests can assert the exact observed set — and only
/// the error's single `path` field is narrowed to the first, which is what the
/// contract's postcondition requires ("names the exact dotted path of the first
/// offending null").
fn first_unallowed_null_path(doc: &JsonMap<String, Value>) -> Option<String> {
    disallowed_null_paths(doc).into_iter().next()
}

/// Refuse a document carrying a `null` outside the allowlist.
fn guard_subdocument_nulls(doc: &JsonMap<String, Value>) -> Result<(), SetFitArtifactError> {
    if let Some(path) = first_unallowed_null_path(doc) {
        return Err(SetFitArtifactError::NonFiniteValue { path });
    }
    Ok(())
}

// ===========================================================================
// Document construction
// ===========================================================================

/// Build the normative `SetFitArtifactDoc` as ONE `serde_json::Map`.
///
/// The insertion order below is the contract's declaration order, for a reader's
/// benefit only: `serde_json::Map` is BTreeMap-backed (no workspace crate enables
/// `preserve_order`), so the SERIALIZED key order is sorted and does not depend
/// on the order of these calls. That is precisely what makes the bytes
/// reproducible.
fn build_artifact_doc(
    view: &SetFitArtifactView,
    hf_name_map: &BTreeMap<String, String>,
    probes: &[Value],
) -> Result<JsonMap<String, Value>, SetFitArtifactError> {
    // The architecture sub-document is DERIVED from the typed record — see
    // `SetFitArtifactView`. One fact, one copy.
    let architecture = serde_json::to_value(&view.architecture).map_err(|e| {
        SetFitArtifactError::InconsistentTensorSet {
            reason: format!("the architecture record does not serialize: {e}"),
        }
    })?;

    let mut preprocessing = JsonMap::new();
    preprocessing.insert("pooling".to_string(), Value::String(view.pooling.clone()));
    preprocessing.insert(
        "normalization".to_string(),
        Value::String(view.normalization.clone()),
    );
    // A bit-pattern hex string, never a JSON number: decimal text is not the
    // identity on f32, and a non-finite value would render as an indistinguishable
    // `null`. Hex keeps every doc-OWNED float exact and off the null path.
    preprocessing.insert(
        "l2_epsilon_hex".to_string(),
        Value::String(f32_bits_hex(view.l2_epsilon)),
    );
    preprocessing.insert(
        "truncation_max_sequence_length".to_string(),
        Value::from(view.truncation_max_sequence_length),
    );
    preprocessing.insert(
        "padding_mode".to_string(),
        Value::String(view.padding_mode.clone()),
    );
    preprocessing.insert("max_length".to_string(), Value::from(view.max_length));

    let mut head = JsonMap::new();
    head.insert("n_features".to_string(), Value::from(view.head_n_features));
    head.insert(
        "num_labels".to_string(),
        Value::from(view.ordered_labels.len()),
    );

    // The map is WRITTEN INTO the artifact so `deserialize` inverts a map the
    // artifact carries rather than re-deriving names from a table that may have
    // moved between the write and the read.
    let mut names = JsonMap::new();
    for (hf, canonical) in hf_name_map {
        names.insert(hf.clone(), Value::String(canonical.clone()));
    }

    let mut doc = JsonMap::new();
    doc.insert(
        "schema".to_string(),
        Value::String(ARTIFACT_SCHEMA.to_string()),
    );
    doc.insert(
        "schema_version".to_string(),
        Value::from(ARTIFACT_SCHEMA_VERSION),
    );
    doc.insert(
        "bundle_schema_version".to_string(),
        Value::from(view.bundle_schema_version),
    );
    doc.insert(
        "format_id".to_string(),
        Value::String(view.format_id.clone()),
    );
    doc.insert("architecture".to_string(), architecture);
    doc.insert(
        "tokenizer_sha256".to_string(),
        Value::String(view.architecture.tokenizer_sha256.clone()),
    );
    doc.insert("preprocessing".to_string(), Value::Object(preprocessing));
    doc.insert("root_seed".to_string(), Value::from(view.root_seed));
    doc.insert("head".to_string(), Value::Object(head));
    doc.insert(
        "ordered_labels".to_string(),
        Value::Array(
            view.ordered_labels
                .iter()
                .map(|label| Value::String(label.clone()))
                .collect(),
        ),
    );
    doc.insert(
        "requested_config".to_string(),
        view.requested_config.clone(),
    );
    doc.insert("resolved_config".to_string(), view.resolved_config.clone());
    doc.insert("evidence".to_string(), view.evidence.clone());
    doc.insert("provenance".to_string(), view.provenance.clone());
    doc.insert("hf_name_map".to_string(), Value::Object(names));
    doc.insert("probes".to_string(), Value::Array(probes.to_vec()));

    debug_assert_eq!(
        doc.len(),
        SETFIT_ARTIFACT_DOC_FIELDS.len(),
        "the doc must carry exactly the contract's field list"
    );
    Ok(doc)
}

// ===========================================================================
// The loader (plan 04-03): the bounded read, the fail-closed ladder, and the
// `VerifiedSetFitModel` typestate
//
// Contract: `contracts/setfit-apr-v1.yaml`, equations `artifact_size_bounds`,
// `bounded_read`, `artifact_doc_schema`, `architecture_derived_tensor_set`,
// `probe_policy`, `probe_and_parity_tolerances` and `load_validation_ladder`.
//
// # The rungs, and why the ORDER is part of the contract
//
// THE NUMBERS ARE THE CONTRACT'S, NOT THIS FILE'S. `load_validation_ladder`
// (contracts/setfit-apr-v1.yaml, `rungs:`) enumerates EIGHT, and a refusal here
// names the rung an operator will look up there. An earlier draft numbered its
// own seven functions 1-7 by folding the bounded read out of the count, so every
// `rung N` this module printed pointed at a different rung of the normative
// ladder — `rung 4` on a corrupt tensor index resolved to "typed tag / document
// parse" in the contract, the wrong subsystem entirely. One numbering, and it is
// the published one.
//
// | rung | what it bounds | where it lives | why it cannot move |
// |------|----------------|----------------|--------------------|
// | 1 | declared length, before the reader is touched | [`read_setfit_apr_bytes_bounded`] | the allocation must be refused before it is requested |
// | 2 | raw length vs the cap | [`rung2_raw_length`] | a parse of an unbounded buffer is the attack |
// | 3 | magic, version, header CRC, row-major flag, footer CRC | [`rung3_container`] | nothing may be believed before integrity |
// | 4 | typed tag, ONE custom key, `schema`/`schema_version`, `deny_unknown_fields` | [`rung4_document`] | a future schema must not be partially interpreted |
// | 5 | architecture-derived tensor set, per-entry size, head shapes, tokenizer digest | [`rung5_structure`] | the tokenizer must be paired BEFORE a tensor is installed |
// | 6 | non-finite scan over every decoded `f32` | [`rung6_finite_payloads`] | a NaN weight must not reach a rebuild |
// | 7 | rebuild encoder + tokenizer + head | [`rung7_rebuild`] | only from bytes that passed 1-6 |
// | 8 | replay all six probes within tolerance | [`rung8_replay_probes`] | the last word, before a classify-capable value exists |
//
// RUNG 1 IS A DIFFERENT KIND OF RUNG, and saying so is the honest form. It bounds
// a SOURCE, so it belongs at the boundary where bytes are acquired and cannot be
// applied to a `&[u8]` a caller already holds. Rung 2 is what the in-memory doors
// run for that caller, which is why the contract lists both rather than merging
// them.
//
// Rungs 2-6 live ONCE, in [`read_setfit_apr_parts_within`], and
// [`load_setfit_apr_within`] CALLS it. One ladder, never two policies: a second
// parse-only path would be a second verification policy with its own tolerances.
// ===========================================================================

/// The contract's outer artifact size bound, in bytes (256 MiB).
///
/// `contracts/setfit-apr-v1.yaml`, equation `artifact_size_bounds`, constant
/// `max_artifact_bytes: 268435456`. It is DERIVED there, not chosen for
/// roundness: the pinned encoder payload is 22565376 f32 (90261504 bytes), the
/// pinned tokenizer is 466247 bytes and a three-label head is 4620 bytes, so a
/// legitimate artifact is about 90.8 MB and this clears it by ~2.9x — the same
/// headroom factor `MAX_BUNDLE_BYTES` uses.
///
/// # This is an outer RESOURCE bound, not a correctness check
///
/// Correctness is the per-entry size rule and the architecture-derived tensor
/// set. The cap exists so a hostile input cannot exhaust memory before either of
/// those can run — see [`read_setfit_apr_bytes_bounded`].
pub const MAX_ARTIFACT_BYTES: u64 = 268_435_456;

/// The largest encoder depth this loader will expand the per-layer templates over.
///
/// `expected(arch)` is `5 + 16 * num_layers + 3` names and `num_layers` arrives
/// from the DOCUMENT, which is attacker-controlled. Without this bound a document
/// claiming 2^60 layers would make the EXPANSION itself the allocation attack —
/// the exact failure the architecture-derived tensor set exists to prevent. The
/// contract records the requirement as `architecture_derived_tensor_set`
/// precondition 2 ("num_layers is bounded by the tensor-count limit, so the
/// expansion cannot itself become the allocation attack") without freezing a
/// number, so the number is derived here and stated: 256 is ~42x the pinned
/// model's 6, and 5 + 16*256 + 3 = 4104 names sits in the same order as the
/// bundle's `MAX_TENSOR_COUNT` of 4096.
pub const MAX_ENCODER_LAYERS: usize = 256;

/// Absolute tolerance for a probe embedding component.
///
/// CITED from `setfit-encoder-conformance-v1`'s frozen pooled-output family, not
/// re-derived here. A Phase 4 number would be a SECOND tolerance for the same
/// quantity, and the looser of two silently becomes the real one.
pub const PROBE_EMBEDDING_ABS_TOLERANCE: f64 = 7.629_394_53e-06;

/// Absolute tolerance for a probe logit.
pub const PROBE_LOGITS_ABS_TOLERANCE: f64 = 1.0e-5;

/// Absolute tolerance for a probe class probability.
pub const PROBE_PROBABILITIES_ABS_TOLERANCE: f64 = 1.0e-5;

/// The ladder's outer resource bounds, as one value.
///
/// # Why the bound is a parameter at all
///
/// The cap has to be shown BITING, and it bites only on inputs far larger than
/// any fixture. Materializing 256 MiB in a unit test would make the suite pay a
/// quarter of a gigabyte to learn that a comparison compares. So the bound VALUE
/// and the bound MECHANISM are falsified separately: the mechanism against a
/// deliberately tiny bound on a real artifact, the value against the contract it
/// is frozen in. This is `BundleLimits`'s shape (bundle.rs:138-181) and it is
/// kept for the same reason.
///
/// The field is private and the only value production can name is
/// [`Self::CONTRACTED`]; the shrinking constructor is `#[cfg(test)]`. A knob that
/// could weaken a bound in a shipped build would be worse than the attack it
/// tests for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ArtifactLimits {
    max_artifact_bytes: u64,
}

impl ArtifactLimits {
    /// The contracted bound — the only value a shipped build can construct.
    const CONTRACTED: Self = Self {
        max_artifact_bytes: MAX_ARTIFACT_BYTES,
    };

    /// A deliberately tiny bound, so the cap can be shown biting on a real artifact.
    #[cfg(test)]
    const fn tiny(max_artifact_bytes: u64) -> Self {
        Self { max_artifact_bytes }
    }
}

// ---------------------------------------------------------------------------
// The parsed document (contract equation `artifact_doc_schema`)
// ---------------------------------------------------------------------------

/// The `preprocessing` group of [`SetFitArtifactDoc`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetFitPreprocessingDoc {
    /// The pooling policy identifier the encoder applied.
    pub pooling: String,
    /// The normalization policy identifier the encoder applied.
    pub normalization: String,
    /// The L2 epsilon as an f32 BIT PATTERN in hex, never a JSON number.
    pub l2_epsilon_hex: String,
    /// The tokenizer's truncation bound.
    pub truncation_max_sequence_length: u32,
    /// The tokenizer's padding mode.
    pub padding_mode: String,
    /// The run's requested max sequence length.
    pub max_length: u32,
}

/// The `head` group of [`SetFitArtifactDoc`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetFitHeadDoc {
    /// The head's fitted feature dimension.
    pub n_features: usize,
    /// The number of labels the head discriminates.
    pub num_labels: usize,
}

/// One embedded probe record (contract equation `probe_policy`).
///
/// EVERY float is a bit-pattern hex string. Decimal text is not the identity on
/// `f32`, and a `null` from a non-finite value would be indistinguishable from a
/// missing one — hex keeps probe expectations exact and off the null path.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetFitProbeRecord {
    /// The probe string, verbatim.
    pub input: String,
    /// The pooled, normalized embedding, one bit-pattern hex string per component.
    pub embedding_hex: Vec<String>,
    /// One logit per label, in `ordered_labels` order.
    pub logits_hex: Vec<String>,
    /// One probability per label, in `ordered_labels` order.
    pub probabilities_hex: Vec<String>,
    /// The argmax label, compared EXACTLY.
    pub label: String,
}

/// The single value held at custom metadata key `"setfit"`.
///
/// The field list is EXACTLY [`SETFIT_ARTIFACT_DOC_FIELDS`] — the contract's
/// normative list (review B2) — and `deny_unknown_fields` makes an unknown key a
/// typed parse refusal rather than a silently ignored field. That the two agree
/// is asserted by a test, so an added or renamed field is a LOUD failure.
///
/// # The four opaque sub-documents
///
/// `requested_config`, `resolved_config`, `evidence` and `provenance` stay
/// `serde_json::Value` because `aprender-core` CANNOT NAME the `aprender-train`
/// types behind them: train depends on core, never the reverse. `architecture`
/// is the one core DOES own, so it is parsed into the typed
/// [`EncoderArchitecture`] the rebuild needs.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetFitArtifactDoc {
    /// Const `"setfit-apr-v1"`.
    pub schema: String,
    /// Const `1`.
    pub schema_version: u32,
    /// The `SetFitBundle` wire version this artifact was written from.
    pub bundle_schema_version: u32,
    /// The codec identifier that wrote the payload.
    pub format_id: String,
    /// The encoder architecture record — typed, because the rebuild needs it.
    pub architecture: EncoderArchitecture,
    /// Lowercase-hex SHA-256 of the tokenizer bytes, at the doc's first level.
    pub tokenizer_sha256: String,
    /// The pooling/normalization/truncation policy the encoder applied.
    pub preprocessing: SetFitPreprocessingDoc,
    /// The root seed every dropout stream derives from.
    pub root_seed: u64,
    /// The head's configuration.
    pub head: SetFitHeadDoc,
    /// Index `i` is the label of head weight row `i`.
    pub ordered_labels: Vec<String>,
    /// `to_value(SetFitTrainConfig)` — opaque here.
    pub requested_config: Value,
    /// `to_value(ResolvedConfigRecord)` — opaque here.
    pub resolved_config: Value,
    /// `to_value(EvidenceSummary)` — opaque here.
    pub evidence: Value,
    /// `to_value(ProvenanceRecord)` — opaque here (bundle field 20).
    pub provenance: Value,
    /// HF dotted name -> canonical tensor name, carried BY the artifact.
    pub hf_name_map: BTreeMap<String, String>,
    /// The six contract-resident probe records, in probe order.
    pub probes: Vec<SetFitProbeRecord>,
}

/// Everything rungs 2-6 recover, before any rebuild has happened.
///
/// A STRUCT and not a five-element tuple, deliberately: plan 04-05 maps this
/// field-by-field onto a `SetFitBundle`, and a tuple of five same-shaped
/// components is exactly where such a mapping goes wrong silently.
#[derive(Debug, Clone)]
pub struct SetFitAprParts {
    /// The parsed, `deny_unknown_fields` document.
    pub doc: SetFitArtifactDoc,
    /// Every encoder tensor keyed by its HF dotted name, `(shape, data)`.
    ///
    /// Keyed by HF name because that is what `SetFitMiniLm::from_bundle_parts`
    /// takes; the canonical names are inverted through the doc's OWN
    /// `hf_name_map`, never re-derived from a table that may have moved since
    /// the write.
    pub tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    /// The head's `K * d` weights, row-major; row `i` belongs to `ordered_labels[i]`.
    pub head_weights: Vec<f32>,
    /// The head's `K` intercepts.
    pub head_intercepts: Vec<f32>,
    /// The exact `tokenizer.json` bytes the artifact carries.
    pub tokenizer_bytes: Vec<u8>,
    /// Lowercase-hex SHA-256 of the artifact bytes these parts came from.
    pub artifact_sha256: String,
}

// ---------------------------------------------------------------------------
// The verified typestate (APR-04's consumer side)
// ---------------------------------------------------------------------------

/// A model that reached the END of the ladder, and the ONLY value a consumer may
/// classify with.
///
/// # Non-constructibility is the mechanism, not the documentation
///
/// Every field is private and there is NO public constructor, no `Default`, no
/// `Deserialize` and no builder. The only way to obtain one is
/// [`load_setfit_apr`], which means every rung — including the six-probe replay —
/// ran in THIS process on THESE bytes. `crates/aprender-train/tests/ui/
/// setfit_verified_model_constructed.rs` pins that as a COMPILE error from
/// outside the crate: a runtime rejection can be caught and ignored by a caller,
/// a non-compiling program cannot.
///
/// APR-04's obligation in one sentence: evaluation, registration, benchmarking,
/// prediction and serving accept only `ArtifactReloadedAndVerified`, and
/// out-of-crate code cannot mint the consumer-side witness type.
///
/// # What is deliberately absent
///
/// `classify` arrives in plan 04-04 and reads [`Self::head`]'s coefficients.
/// Nothing speculative is added here.
#[derive(Debug)]
pub struct VerifiedSetFitModel {
    /// The rebuilt encoder + tokenizer pair.
    model: SetFitMiniLm,
    /// The rebuilt classifier head. [`Self::ordered_labels`] reads its label
    /// vector — the one a classification will actually index by, rather than the
    /// doc's copy of it — and plan 04-04's `classify` reads its coefficients.
    head: MultinomialLogisticRegression,
    /// Everything APR-05's inspection recovers from the artifact alone.
    doc: SetFitArtifactDoc,
    /// The identity every Phase 4 response carries.
    artifact_sha256: String,
}

impl VerifiedSetFitModel {
    /// Lowercase-hex SHA-256 of the artifact these bytes were verified from.
    #[must_use]
    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    /// The labels the head's weight rows are indexed by.
    ///
    /// Read off the REBUILT HEAD, not off the document: this is the list a
    /// classification will actually index into, and a doc copy that had drifted
    /// from it would be a label map that describes a different model.
    #[must_use]
    pub fn ordered_labels(&self) -> &[String] {
        self.head.labels()
    }

    /// The whole recovered document — APR-05's inspection surface.
    ///
    /// Revisions, hashes, pooling/truncation policy, label order, head
    /// configuration, provenance (including the dataset fingerprint), seeds,
    /// update evidence and the schema version, all recovered from the artifact
    /// alone with no network and no sidecar file.
    #[must_use]
    pub fn doc_view(&self) -> &SetFitArtifactDoc {
        &self.doc
    }

    /// The rebuilt encoder + tokenizer pair.
    ///
    /// `pub(crate)`, for `setfit::classify` (plan 04-04). It is a READ borrow:
    /// it constructs nothing and mutates nothing, so the D-08 seal and this
    /// typestate's private constructor are both untouched. It exists because
    /// [`Self::embed`] deliberately calls the UNTRACED
    /// `SetFitMiniLm::encode_texts`, and `classify` must instead reach
    /// `encode_batch_traced` so the backend it reports is the one the encode
    /// that produced ITS embeddings returned (D-12).
    #[must_use]
    pub(crate) fn model(&self) -> &SetFitMiniLm {
        &self.model
    }

    /// The rebuilt classifier head. `pub(crate)`, on the same terms as
    /// [`Self::model`].
    #[must_use]
    pub(crate) fn head(&self) -> &MultinomialLogisticRegression {
        &self.head
    }

    /// The encoder's L2-normalized sentence embeddings — OPS-01's "embed" step.
    ///
    /// This is the SAME encode path rung 8's probe replay verified, so a caller
    /// reaching embeddings through the public API gets the vectors the artifact's
    /// own probe expectations were checked against.
    ///
    /// # Errors
    ///
    /// [`SetFitArtifactError::EmptyEmbedBatch`] for an empty input list, and
    /// [`SetFitArtifactError::EncodeFailed`] for anything the tokenizer or the
    /// encoder rejects. There is no panic path.
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, SetFitArtifactError> {
        if texts.is_empty() {
            return Err(SetFitArtifactError::EmptyEmbedBatch);
        }
        let borrowed: Vec<&str> = texts.iter().map(String::as_str).collect();
        let pooled =
            self.model
                .encode_texts(&borrowed)
                .map_err(|e| SetFitArtifactError::EncodeFailed {
                    reason: e.to_string(),
                })?;
        split_embedding_rows(&pooled, texts.len())
            .map_err(|reason| SetFitArtifactError::EncodeFailed { reason })
    }
}

/// Split a `[B, H]` pooled tensor into `B` owned rows, or say why it could not be.
///
/// ONE implementation of the shape rule for the whole crate. `embed` and
/// `classify` differ in WHICH encode produced the tensor — traced vs untraced —
/// but not in how a `[B, H]` tensor becomes rows, and two copies of that split
/// were two places for the bounds rule to drift apart.
///
/// Returns the reason as a `String` rather than a typed error precisely so each
/// caller keeps its OWN error enum: `embed` maps it to
/// `SetFitArtifactError::EncodeFailed`, `classify` to `ClassifyError::EncodeFailed`.
/// Sharing the rule does not mean sharing the refusal type.
pub(crate) fn split_embedding_rows(
    pooled: &crate::autograd::Tensor,
    expected_rows: usize,
) -> Result<Vec<Vec<f32>>, String> {
    let shape = pooled.shape().to_vec();
    let (rows, width) = match shape.as_slice() {
        [rows, width] => (*rows, *width),
        other => {
            return Err(format!(
                "the encoder produced shape {other:?}, expected [B, H]"
            ))
        }
    };
    if rows != expected_rows {
        return Err(format!(
            "{expected_rows} texts produced {rows} embedding rows"
        ));
    }
    let data = pooled.data();
    let mut out = Vec::with_capacity(rows);
    for row in 0..rows {
        let start = row.saturating_mul(width);
        let end = start.saturating_add(width);
        // `get` and not `[start..end]`: this is a codec whose contract is that
        // it does not panic, and a slice index is a panic path.
        let slice = data.get(start..end).ok_or_else(|| {
            format!(
                "row {row} spans {start}..{end} of a {}-element result",
                data.len()
            )
        })?;
        out.push(slice.to_vec());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The bounded read (contract equation `bounded_read`, review B5)
// ---------------------------------------------------------------------------

/// Read artifact bytes with the cap applied BEFORE the allocation it bounds.
///
/// THIS is the API every filesystem and stream adapter is required to call —
/// `apr-cli`, `aprender-serve` and the codec alike. Two adapters with two
/// bounded-read implementations are two places for the bound to be forgotten.
///
/// # Two checks, because one is not enough
///
/// (a) If `declared_len` is over cap the call returns WITHOUT TOUCHING the
/// reader, so a caller passing `fs::metadata(path)?.len()` is refused before
/// `fs::read` ever runs. (b) The read then goes through
/// `reader.take(MAX_ARTIFACT_BYTES + 1)` anyway, so a length that LIES — a FIFO,
/// a growing file, a filesystem reporting zero — still cannot exhaust memory.
/// Check (a) alone trusts metadata an attacker controls; check (b) alone reads
/// 256 MiB of garbage before refusing.
///
/// The `+ 1` is load-bearing: reading exactly the cap cannot distinguish "a legal
/// artifact of exactly the cap size" from "a larger stream truncated at the cap".
///
/// # Why [`load_setfit_apr`] keeps its own length check
///
/// Defense in depth, and not redundancy: the in-memory check CANNOT protect a
/// caller who already read the file from some other source. Review B5's finding
/// was precisely that the cap ran only after `fs::read` had already succeeded.
///
/// # Errors
///
/// [`SetFitArtifactError::ArtifactTooLarge`] naming which of the two checks
/// fired and what it observed, or [`SetFitArtifactError::ArtifactRead`] for an
/// I/O failure.
pub fn read_setfit_apr_bytes_bounded<R: std::io::Read>(
    reader: R,
    declared_len: Option<u64>,
) -> Result<Vec<u8>, SetFitArtifactError> {
    read_setfit_apr_bytes_bounded_within(reader, declared_len, &ArtifactLimits::CONTRACTED)
}

/// [`read_setfit_apr_bytes_bounded`] at a caller-chosen bound.
///
/// Module-private, and it stays that way: the shipped door above is the only one
/// that names a bound, and the only bound it names is the contracted one.
fn read_setfit_apr_bytes_bounded_within<R: std::io::Read>(
    reader: R,
    declared_len: Option<u64>,
    limits: &ArtifactLimits,
) -> Result<Vec<u8>, SetFitArtifactError> {
    use std::io::Read as _;

    let cap = limits.max_artifact_bytes;

    // (a) THE DECLARED LENGTH, BEFORE THE READER IS TOUCHED. A caller passing
    //     `fs::metadata(path)?.len()` is refused here, so `fs::read` never runs
    //     on a hostile multi-gigabyte file. Review B5: the cap belongs at the
    //     read, not only after it.
    if let Some(declared) = declared_len {
        if declared > cap {
            return Err(SetFitArtifactError::ArtifactTooLarge {
                what: "declared_length",
                observed: declared,
                cap,
            });
        }
    }

    // (b) THE READ ITSELF, BOUNDED AT cap + 1 ANYWAY, because check (a) trusts
    //     metadata an attacker controls. The reservation is the declared length
    //     CLAMPED to the cap, so a length that lies cannot pre-allocate past it,
    //     and an absent length reserves nothing at all.
    let ceiling = cap.saturating_add(1);
    let reserve = usize::try_from(declared_len.unwrap_or(0).min(cap)).unwrap_or(0);
    let mut bytes = Vec::with_capacity(reserve);
    let mut bounded = reader.take(ceiling);
    bounded
        .read_to_end(&mut bytes)
        .map_err(|e| SetFitArtifactError::ArtifactRead {
            reason: e.to_string(),
        })?;

    // (c) THE `+ 1` PAYS OFF HERE. Exactly `cap` bytes is a legal artifact of
    //     exactly the cap size; `cap + 1` is a larger stream truncated at the
    //     cap, and only reading one byte more makes the two distinguishable.
    let observed = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if observed > cap {
        return Err(SetFitArtifactError::ArtifactTooLarge {
            what: "stream",
            observed,
            cap,
        });
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// The two doors
// ---------------------------------------------------------------------------

/// Rungs 2-6 as a PARSE-ONLY door: no rebuild, no probe replay.
///
/// [`load_setfit_apr`] is implemented as this function followed by rungs 7-8, so
/// rungs 2-6 exist ONCE. Rung 1 bounds a SOURCE and therefore lives at the
/// boundary that acquires the bytes ([`read_setfit_apr_bytes_bounded`]); rung 2
/// is what this door runs for a caller who already holds a slice. Plan 04-05's codec consumes this door; landing it here
/// is what keeps the two from becoming two verification policies.
///
/// # Errors
///
/// The same typed [`SetFitArtifactError`] variants [`load_setfit_apr`] reports
/// for any rung 2-6 failure — by construction, because it is the same code.
pub fn read_setfit_apr_parts(bytes: &[u8]) -> Result<SetFitAprParts, SetFitArtifactError> {
    read_setfit_apr_parts_within(bytes, &ArtifactLimits::CONTRACTED)
}

/// [`read_setfit_apr_parts`] at a caller-chosen bound. Module-private.
fn read_setfit_apr_parts_within(
    bytes: &[u8],
    limits: &ArtifactLimits,
) -> Result<SetFitAprParts, SetFitArtifactError> {
    rung2_raw_length(bytes, limits)?;
    let reader = rung3_container(bytes)?;
    let doc = rung4_document(&reader)?;
    rung5_structure(&reader, &doc)?;
    let (tensors, head_weights, head_intercepts, tokenizer_bytes) =
        rung6_finite_payloads(&reader, &doc)?;
    Ok(SetFitAprParts {
        doc,
        tensors,
        head_weights,
        head_intercepts,
        tokenizer_bytes,
        artifact_sha256: artifact_sha256_hex(bytes),
    })
}

/// The ONE production door: bytes in, a verified model or a typed refusal out.
///
/// Runs rungs 2-8 in the contract's order, offline, after the caller has taken
/// rung 1 through [`read_setfit_apr_bytes_bounded`]. Nothing short of the whole
/// ladder produces a [`VerifiedSetFitModel`], and no consumer may add a second
/// minting path — a consumer with its own load path would be a second
/// verification policy with its own tolerances.
///
/// # Errors
///
/// One distinct [`SetFitArtifactError`] variant per corruption class, each naming
/// the rung and what it observed. "Invalid artifact" is not an admissible
/// diagnosis and this function never produces one.
pub fn load_setfit_apr(bytes: &[u8]) -> Result<VerifiedSetFitModel, SetFitArtifactError> {
    load_setfit_apr_within(bytes, &ArtifactLimits::CONTRACTED)
}

/// [`load_setfit_apr`] at a caller-chosen bound. Module-private.
fn load_setfit_apr_within(
    bytes: &[u8],
    limits: &ArtifactLimits,
) -> Result<VerifiedSetFitModel, SetFitArtifactError> {
    // RUNGS 2-6, through the very code the parse-only door runs. The call is the
    // point: two copies of this ladder would be two verification policies, and
    // the looser of the two would silently become the real one.
    let mut parts = read_setfit_apr_parts_within(bytes, limits)?;
    // RUNG 7. The tensor map moves in: `from_bundle_parts` drains it, and nothing
    // after this rung reads `parts.tensors`. Cloning here would duplicate ~90 MB on
    // a full pin — the same waste `from_named_tensors` was changed to take by value
    // to eliminate (encoder.rs).
    let tensors = std::mem::take(&mut parts.tensors);
    let (model, head) = rung7_rebuild(tensors, &parts)?;
    // RUNG 8 — the last word. Nothing classify-capable exists until it returns.
    rung8_replay_probes(&model, &head, &parts)?;
    Ok(VerifiedSetFitModel {
        model,
        head,
        artifact_sha256: parts.artifact_sha256,
        doc: parts.doc,
    })
}

// ---------------------------------------------------------------------------
// Rung 2: the raw length, before any parse
// ---------------------------------------------------------------------------

/// The in-memory door's own length check — DEFENSE IN DEPTH, not redundancy.
///
/// [`read_setfit_apr_bytes_bounded`] protects a caller who is about to read a
/// file. This protects a caller who already holds bytes from somewhere else, and
/// neither one subsumes the other.
fn rung2_raw_length(bytes: &[u8], limits: &ArtifactLimits) -> Result<(), SetFitArtifactError> {
    let observed = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if observed > limits.max_artifact_bytes {
        return Err(SetFitArtifactError::ArtifactTooLarge {
            what: "input_bytes",
            observed,
            cap: limits.max_artifact_bytes,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Rung 3: the container
// ---------------------------------------------------------------------------

/// Magic, header CRC, container version, row-major flag and footer CRC.
///
/// The footer CRC runs FIRST because it is a pure byte comparison over the whole
/// content: nothing in the file is interpreted until the file has been shown to
/// be the file that was written. CRC32 is NOT cryptographic, so this rung cannot
/// see semantic corruption — that is rung 8's job.
fn rung3_container(bytes: &[u8]) -> Result<AprV2Reader, SetFitArtifactError> {
    verify_footer_checksum(bytes)?;

    let reader = AprV2Reader::from_bytes(bytes).map_err(|e| {
        let what = match &e {
            V2FormatError::ChecksumMismatch => "header_checksum",
            V2FormatError::InvalidMagic(_) => "magic",
            _ => "container_parse",
        };
        SetFitArtifactError::ContainerIntegrity {
            what,
            reason: e.to_string(),
        }
    })?;

    let header = reader.header();
    if header.version != VERSION_V2 {
        return Err(SetFitArtifactError::ContainerIntegrity {
            what: "container_version",
            reason: format!(
                "the container declares version {:?}, this build reads {VERSION_V2:?}",
                header.version
            ),
        });
    }
    // LAYOUT-002. `AprV2Reader` only refuses an EXPLICIT column-major flag; this
    // artifact schema additionally REQUIRES the row-major flag to be present, so
    // an artifact that asserts nothing about its layout is refused rather than
    // assumed.
    if !header.flags.is_row_major() {
        return Err(SetFitArtifactError::ContainerIntegrity {
            what: "row_major_flag",
            reason: "LAYOUT_ROW_MAJOR is not set; setfit-apr-v1 is exclusively row-major and \
                     does not assume a layout an artifact declined to declare"
                .to_string(),
        });
    }
    Ok(reader)
}

/// The container's trailing CRC32 over everything that precedes it.
fn verify_footer_checksum(bytes: &[u8]) -> Result<(), SetFitArtifactError> {
    let split =
        bytes
            .len()
            .checked_sub(4)
            .ok_or_else(|| SetFitArtifactError::ContainerIntegrity {
                what: "footer_length",
                reason: format!(
                    "{} bytes cannot carry the container's 4-byte trailing checksum",
                    bytes.len()
                ),
            })?;
    let (content, footer) = bytes.split_at(split);
    let declared = u32::from_le_bytes([footer[0], footer[1], footer[2], footer[3]]);
    let observed = crate::format::crc32(content);
    if declared != observed {
        return Err(SetFitArtifactError::ContainerIntegrity {
            what: "footer_checksum",
            reason: format!(
                "the footer declares {declared:#010x} over {} content bytes; a recompute gives \
                 {observed:#010x}",
                content.len()
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Rung 4: the typed tag and the ONE document
// ---------------------------------------------------------------------------

/// D-04 explicit-tag detection, the one custom key, then `schema` /
/// `schema_version` BEFORE any other field is read.
fn rung4_document(reader: &AprV2Reader) -> Result<SetFitArtifactDoc, SetFitArtifactError> {
    let metadata = reader.metadata();
    // D-04: the whole detection rule. No tensor-name sniffing, no shape
    // inference — a SetFit-shaped tensor set without the tag is a plain APR.
    if metadata.model_type != MODEL_TYPE_TAG {
        return Err(SetFitArtifactError::NotASetFitArtifact {
            model_type: metadata.model_type.clone(),
        });
    }
    if metadata.custom.len() != 1 {
        return Err(SetFitArtifactError::ArtifactDocumentMissing {
            reason: format!(
                "the container carries {} custom metadata keys; exactly one ({CUSTOM_METADATA_KEY:?}) \
                 is reproducible, because AprV2Metadata.custom is a flattened HashMap whose \
                 iteration order is unspecified",
                metadata.custom.len()
            ),
        });
    }
    let value = metadata.custom.get(CUSTOM_METADATA_KEY).ok_or_else(|| {
        SetFitArtifactError::ArtifactDocumentMissing {
            reason: format!("the one custom metadata key is not {CUSTOM_METADATA_KEY:?}"),
        }
    })?;
    let object = value
        .as_object()
        .ok_or_else(|| SetFitArtifactError::ArtifactDocumentMissing {
            reason: format!("the {CUSTOM_METADATA_KEY:?} key does not hold a JSON object"),
        })?;

    // SCHEMA AND SCHEMA VERSION FIRST. Reading them off the raw `Value` before
    // the typed parse is what makes the contract's postcondition literally true:
    // a future schema is refused rather than partially interpreted by this build.
    let schema = object
        .get("schema")
        .and_then(Value::as_str)
        .ok_or_else(|| SetFitArtifactError::ArtifactDocumentMissing {
            reason: "the document has no first-level `schema` field".to_string(),
        })?;
    if schema != ARTIFACT_SCHEMA {
        return Err(SetFitArtifactError::UnsupportedSchema {
            got: schema.to_string(),
            supported: ARTIFACT_SCHEMA,
        });
    }
    let version = object
        .get("schema_version")
        .and_then(Value::as_u64)
        .ok_or_else(|| SetFitArtifactError::ArtifactDocumentMissing {
            reason: "the document has no first-level `schema_version` field".to_string(),
        })?;
    if version != u64::from(ARTIFACT_SCHEMA_VERSION) {
        return Err(SetFitArtifactError::UnsupportedSchemaVersion {
            got: version,
            supported: ARTIFACT_SCHEMA_VERSION,
        });
    }

    // Only now the typed, `deny_unknown_fields` parse. Deserializing from `&Value`
    // rather than `from_value(value.clone())` avoids deep-cloning the whole tree —
    // dominated by the probes' ~2,600 hex strings — purely to hand it to serde.
    // Same `deny_unknown_fields` behaviour, same error, same ordering: the schema
    // and schema_version pre-checks above still run first on the raw `Value`.
    SetFitArtifactDoc::deserialize(value).map_err(|e| SetFitArtifactError::ArtifactDocumentParse {
        detail: e.to_string(),
    })
}

// ---------------------------------------------------------------------------
// Rung 5: structure, derived from the artifact's OWN declared architecture
// ---------------------------------------------------------------------------

/// The architecture-derived tensor set, the per-entry size rule, the head
/// shapes, the carried name map and the tokenizer identity.
fn rung5_structure(
    reader: &AprV2Reader,
    doc: &SetFitArtifactDoc,
) -> Result<(), SetFitArtifactError> {
    check_declared_depth(doc)?;
    // ONE expansion of the sixteen per-layer templates for the whole rung. It was
    // computed three times — once inside `expected_tensor_names` and twice inside
    // `check_carried_name_map` — and each expansion allocates `5 + 16 * num_layers`
    // Strings for a map that cannot change between the calls. The bound checked
    // immediately above is what makes doing it once, here, safe to do eagerly.
    let derived = build_hf_name_map(doc.architecture.num_layers);
    check_tensor_name_set(reader, &derived)?;
    check_tensor_entries(reader)?;
    check_head_shapes(reader, doc)?;
    check_carried_name_map(doc, &derived)?;
    check_tokenizer_identity(reader, doc)
}

/// The expansion bound, BEFORE the expansion.
///
/// `expected(arch)` is `5 + 16 * num_layers + 3` names and `num_layers` arrives
/// from the DOCUMENT — attacker-controlled. Without this check a document
/// claiming 2^60 layers would make the EXPANSION the allocation attack, which is
/// precisely what the tensor-set rule exists to prevent (contract
/// `architecture_derived_tensor_set`, precondition 2).
fn check_declared_depth(doc: &SetFitArtifactDoc) -> Result<(), SetFitArtifactError> {
    let declared = u64::try_from(doc.architecture.num_layers).unwrap_or(u64::MAX);
    let cap = u64::try_from(MAX_ENCODER_LAYERS).unwrap_or(u64::MAX);
    if declared > cap {
        return Err(SetFitArtifactError::ArtifactTooLarge {
            what: "declared_layers",
            observed: declared,
            cap,
        });
    }
    Ok(())
}

/// `tensor_index_names == expected(doc.architecture)`, compared EXACTLY.
///
/// A mismatch names the MISSING and the UNEXPECTED tensors, not merely the two
/// counts: "104 != 103" is not a diagnosis. The head tensors are members of the
/// expected set (review B1), so a HEADLESS artifact cannot reach a prediction.
fn check_tensor_name_set(
    reader: &AprV2Reader,
    derived: &BTreeMap<String, String>,
) -> Result<(), SetFitArtifactError> {
    let expected = expected_tensor_names_from(derived);
    let names = reader.tensor_names();
    let observed: BTreeSet<String> = names.iter().map(|name| (*name).to_string()).collect();

    // A DUPLICATE NAME IS NOT A SET-EQUALITY QUESTION, so it has to be asked
    // separately: collapsing the index into a `BTreeSet` makes two entries called
    // `setfit.head.weight` indistinguishable from one, and `get_tensor` then
    // resolves to whichever the index happens to list first. The whole rung
    // reasons about names, so a name that does not identify one payload has to
    // die here rather than silently pick a winner.
    if observed.len() != names.len() {
        let mut seen: BTreeSet<&str> = BTreeSet::new();
        let mut duplicated: Vec<String> = names
            .iter()
            .filter(|name| !seen.insert(name))
            .map(|name| (*name).to_string())
            .collect();
        duplicated.sort();
        duplicated.dedup();
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "the index carries {} entries under {} distinct names; a duplicate name does not \
                 identify one payload: {}",
                names.len(),
                observed.len(),
                duplicated.join(", ")
            ),
        });
    }

    let missing: Vec<String> = expected.difference(&observed).cloned().collect();
    if !missing.is_empty() {
        return Err(SetFitArtifactError::IncompleteTensorSet { missing });
    }
    let unexpected: Vec<String> = observed.difference(&expected).cloned().collect();
    if !unexpected.is_empty() {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "{} tensor(s) outside the architecture-derived set: {}",
                unexpected.len(),
                unexpected.join(", ")
            ),
        });
    }
    Ok(())
}

/// `size == product(shape) * dtype_width`, on the DECLARED shape and dtype,
/// before any payload is decoded into a tensor.
fn check_tensor_entries(reader: &AprV2Reader) -> Result<(), SetFitArtifactError> {
    for entry in reader.tensor_index() {
        // dtype U8 makes the tokenizer's byte-exactness STRUCTURAL: there is no
        // float path it could be rounded through.
        let required = if entry.name == TOKENIZER_BLOB_TENSOR {
            TensorDType::U8
        } else {
            TensorDType::F32
        };
        if entry.dtype != required {
            return Err(SetFitArtifactError::InconsistentTensor {
                tensor: entry.name.clone(),
                reason: format!(
                    "declares dtype {} but {required} is the only dtype this name may carry",
                    entry.dtype
                ),
            });
        }
        if entry.shape.is_empty() {
            return Err(SetFitArtifactError::InconsistentTensor {
                tensor: entry.name.clone(),
                reason: "declares no shape at all".to_string(),
            });
        }
        let mut elements: u64 = 1;
        for dim in &entry.shape {
            let dim = u64::try_from(*dim).unwrap_or(u64::MAX);
            elements = elements.checked_mul(dim).ok_or_else(|| {
                SetFitArtifactError::InconsistentTensor {
                    tensor: entry.name.clone(),
                    reason: format!("shape {:?} overflows an element count", entry.shape),
                }
            })?;
        }
        // `u64::MAX` and not `0` on the (unreachable) conversion failure: a width
        // of zero makes `declared` zero, which turns this check into "the entry
        // must declare size 0" — a guard that fails OPEN for exactly the entry a
        // hostile artifact would want it to. Saturating upward refuses instead.
        let width = u64::try_from(required.bytes_per_element()).unwrap_or(u64::MAX);
        let declared =
            elements
                .checked_mul(width)
                .ok_or_else(|| SetFitArtifactError::InconsistentTensor {
                    tensor: entry.name.clone(),
                    reason: format!("shape {:?} overflows a byte count", entry.shape),
                })?;
        if entry.size != declared {
            return Err(SetFitArtifactError::InconsistentTensor {
                tensor: entry.name.clone(),
                reason: format!(
                    "declares shape {:?} ({elements} elements x {width} bytes = {declared}) but \
                     the index records size {}",
                    entry.shape, entry.size
                ),
            });
        }
    }
    Ok(())
}

/// The head must be the head of THIS label set and THIS encoder (review B1/T-04-44).
fn check_head_shapes(
    reader: &AprV2Reader,
    doc: &SetFitArtifactDoc,
) -> Result<(), SetFitArtifactError> {
    let num_labels = doc.ordered_labels.len();
    if num_labels < 2 {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "a classifier head needs at least two labels, the document declares {num_labels}"
            ),
        });
    }
    if doc.head.num_labels != num_labels {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "head.num_labels is {} but ordered_labels carries {num_labels}",
                doc.head.num_labels
            ),
        });
    }
    // A head fitted at a different width would refuse every probe at replay time
    // in a fresh process, long after the artifact shipped.
    if doc.head.n_features != doc.architecture.hidden {
        return Err(SetFitArtifactError::InconsistentTensorSet {
            reason: format!(
                "head.n_features {} does not match the encoder's hidden width {}",
                doc.head.n_features, doc.architecture.hidden
            ),
        });
    }

    for (name, expected) in [
        (HEAD_WEIGHT_TENSOR, vec![num_labels, doc.head.n_features]),
        (HEAD_BIAS_TENSOR, vec![num_labels]),
    ] {
        let entry =
            reader
                .get_tensor(name)
                .ok_or_else(|| SetFitArtifactError::IncompleteTensorSet {
                    missing: vec![name.to_string()],
                })?;
        if entry.shape != expected {
            return Err(SetFitArtifactError::InconsistentTensor {
                tensor: name.to_string(),
                reason: format!(
                    "declares shape {:?}; the document's label set and head configuration require \
                     {expected:?}, because row i belongs to ordered_labels[i]",
                    entry.shape
                ),
            });
        }
    }
    Ok(())
}

/// The carried `hf_name_map` must be usable as an INVERSION.
///
/// The map is written into the artifact precisely so `deserialize` inverts a map
/// the artifact carries rather than re-deriving names from a table that may have
/// moved since the write. For that to be safe it must be INJECTIVE (a collision
/// would OVERWRITE a tensor rather than fail) and it must be THE SAME MAP the
/// architecture derives — every PAIR, not merely the same set of canonical names.
///
/// # Why the DOMAIN is checked too, and not only the value set
///
/// An earlier form compared only `values()` against the derived value set. That
/// admits two shapes it should not. Junk keys (`{"a": "token_embd.weight", ...}`)
/// pass the value-set test and make rung 6 key the recovered tensor map by names
/// no rebuild can look up. Worse, a PERMUTED map — the right HF keys pointed at
/// each other's canonical names — also passes: rung 6 then loads the query
/// projection out of the key projection's payload, rung 7 rebuilds happily
/// because the two have identical shapes, and only rung 8's probe replay
/// disagrees, which reports a math divergence for what is a name-map defect. The
/// parse-only door (`read_setfit_apr_parts`, and therefore
/// `AprCodec::deserialize`) does not run rung 8 at all, so on that path the
/// swapped tensors travel out with no refusal whatsoever.
///
/// Comparing the whole map costs nothing extra: rung 5 already expanded the
/// derived map in order to check the value side.
fn check_carried_name_map(
    doc: &SetFitArtifactDoc,
    derived: &BTreeMap<String, String>,
) -> Result<(), SetFitArtifactError> {
    let distinct: BTreeSet<&String> = doc.hf_name_map.values().collect();
    if distinct.len() != doc.hf_name_map.len() {
        return Err(SetFitArtifactError::InconsistentNameMap {
            reason: format!(
                "{} HF names resolve to {} canonical names; a collision would overwrite a tensor \
                 rather than fail",
                doc.hf_name_map.len(),
                distinct.len()
            ),
        });
    }
    // The encoder half IS `build_hf_name_map`'s map. Comparing against it rather
    // than against `expected_tensor_names` with the three schema-owned names
    // removed back off restates no composition rule: a fourth schema-owned tensor
    // cannot silently require an edit here, and a collision between a schema name
    // and a canonical encoder name cannot drop a legitimate entry instead of
    // failing.
    if &doc.hf_name_map != derived {
        let missing: Vec<&String> = derived
            .keys()
            .filter(|hf| !doc.hf_name_map.contains_key(*hf))
            .collect();
        let unexpected: Vec<&String> = doc
            .hf_name_map
            .keys()
            .filter(|hf| !derived.contains_key(*hf))
            .collect();
        let misdirected: Vec<String> = derived
            .iter()
            .filter_map(|(hf, canonical)| {
                doc.hf_name_map
                    .get(hf)
                    .filter(|carried| *carried != canonical)
                    .map(|carried| format!("{hf} -> {carried} (derived {canonical})"))
            })
            .collect();
        return Err(SetFitArtifactError::InconsistentNameMap {
            reason: format!(
                "the carried map is not the architecture-derived map: missing {missing:?}, \
                 unexpected {unexpected:?}, misdirected {misdirected:?}"
            ),
        });
    }
    Ok(())
}

/// The tokenizer is paired BEFORE a tensor is installed.
///
/// An encoder rebuilt with a substituted tokenizer produces confidently wrong
/// embeddings for every input and looks structurally valid the whole time.
fn check_tokenizer_identity(
    reader: &AprV2Reader,
    doc: &SetFitArtifactDoc,
) -> Result<(), SetFitArtifactError> {
    let blob = reader
        .get_tensor_data(TOKENIZER_BLOB_TENSOR)
        .ok_or_else(|| SetFitArtifactError::InconsistentTensor {
            tensor: TOKENIZER_BLOB_TENSOR.to_string(),
            reason: "the index entry does not resolve to a payload inside the file".to_string(),
        })?;
    let observed = sha256_hex(blob);
    if observed != doc.tokenizer_sha256 {
        return Err(SetFitArtifactError::TokenizerHashMismatch {
            expected: doc.tokenizer_sha256.clone(),
            got: observed,
        });
    }
    // The doc's first-level digest and the architecture record's are two copies
    // of one fact, and two copies of one fact are two values that can disagree.
    if doc.architecture.tokenizer_sha256 != doc.tokenizer_sha256 {
        return Err(SetFitArtifactError::TokenizerHashMismatch {
            expected: doc.tokenizer_sha256.clone(),
            got: doc.architecture.tokenizer_sha256.clone(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Rung 6: the non-finite scan, and the payloads it produces
// ---------------------------------------------------------------------------

/// The four payload groups rungs 7-8 consume: HF-keyed tensors, head weights,
/// head intercepts, tokenizer bytes.
type LoadedPayloads = (
    BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    Vec<f32>,
    Vec<f32>,
    Vec<u8>,
);

/// Decode every `f32` the artifact carries and refuse the first non-finite one.
///
/// A `NaN` weight poisons every prediction and looks structurally valid the whole
/// time, so it dies here rather than at a tolerance comparison two rungs later.
fn rung6_finite_payloads(
    reader: &AprV2Reader,
    doc: &SetFitArtifactDoc,
) -> Result<LoadedPayloads, SetFitArtifactError> {
    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
    for (hf, canonical) in &doc.hf_name_map {
        let entry = reader.get_tensor(canonical).ok_or_else(|| {
            SetFitArtifactError::InconsistentNameMap {
                reason: format!(
                    "the map sends {hf} to {canonical}, which the index does not carry"
                ),
            }
        })?;
        let shape = entry.shape.clone();
        let data = decode_finite_f32(reader, canonical)?;
        tensors.insert(hf.clone(), (shape, data));
    }
    let head_weights = decode_finite_f32(reader, HEAD_WEIGHT_TENSOR)?;
    let head_intercepts = decode_finite_f32(reader, HEAD_BIAS_TENSOR)?;
    scan_probe_expectations(doc)?;
    let tokenizer_bytes = reader
        .get_tensor_data(TOKENIZER_BLOB_TENSOR)
        .ok_or_else(|| SetFitArtifactError::InconsistentTensor {
            tensor: TOKENIZER_BLOB_TENSOR.to_string(),
            reason: "the index entry does not resolve to a payload inside the file".to_string(),
        })?
        .to_vec();
    Ok((tensors, head_weights, head_intercepts, tokenizer_bytes))
}

/// One F32 payload, decoded and scanned, with the offending index named.
fn decode_finite_f32(reader: &AprV2Reader, name: &str) -> Result<Vec<f32>, SetFitArtifactError> {
    let data =
        reader
            .get_f32_tensor(name)
            .ok_or_else(|| SetFitArtifactError::InconsistentTensor {
                tensor: name.to_string(),
                reason: "the F32 payload does not resolve inside the file".to_string(),
            })?;
    for (index, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(SetFitArtifactError::NonFiniteValue {
                path: format!("{name}[{index}]"),
            });
        }
    }
    Ok(data)
}

/// Every probe EXPECTATION is decodable and finite.
///
/// A non-finite expectation would make its replay unfalsifiable: the comparator
/// is NaN-visible, so such a probe could never agree with anything — the artifact
/// is refused here instead, where the diagnosis names the field.
fn scan_probe_expectations(doc: &SetFitArtifactDoc) -> Result<(), SetFitArtifactError> {
    for (index, probe) in doc.probes.iter().enumerate() {
        for (field, values) in [
            ("embedding_hex", &probe.embedding_hex),
            ("logits_hex", &probe.logits_hex),
            ("probabilities_hex", &probe.probabilities_hex),
        ] {
            for (position, hex) in values.iter().enumerate() {
                let path = format!("probes.{index}.{field}[{position}]");
                let value = hex_to_f32(hex).ok_or_else(|| {
                    SetFitArtifactError::ArtifactDocumentParse {
                        detail: format!(
                            "{path} is {hex:?}, not eight lowercase hex digits of an f32 bit pattern"
                        ),
                    }
                })?;
                if !value.is_finite() {
                    return Err(SetFitArtifactError::NonFiniteValue { path });
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Rung 7: the rebuild
// ---------------------------------------------------------------------------

/// Rebuild the encoder + tokenizer pair and the classifier head from bytes alone.
///
/// `from_bundle_parts` re-checks the tokenizer digest itself (mod.rs:365-371).
/// That is deliberate duplication of rung 5's check, not an oversight: this
/// function is reachable from one place today and the pairing is too important to
/// depend on the caller having done it.
fn rung7_rebuild(
    tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    parts: &SetFitAprParts,
) -> Result<(SetFitMiniLm, MultinomialLogisticRegression), SetFitArtifactError> {
    let model = SetFitMiniLm::from_bundle_parts(
        &parts.tokenizer_bytes,
        &parts.doc.architecture,
        tensors,
        parts.doc.root_seed,
    )
    .map_err(|e| SetFitArtifactError::ArtifactRebuildFailed {
        what: "encoder",
        reason: e.to_string(),
    })?;
    let head = MultinomialLogisticRegression::from_stored_coefficients(
        parts.doc.ordered_labels.clone(),
        parts.doc.head.n_features,
        parts.head_weights.clone(),
        parts.head_intercepts.clone(),
    )
    .map_err(|e| SetFitArtifactError::ArtifactRebuildFailed {
        what: "head",
        reason: e.to_string(),
    })?;
    Ok((model, head))
}

// ---------------------------------------------------------------------------
// Rung 8: probe replay (D-11)
// ---------------------------------------------------------------------------

/// The ONE shape a probe divergence takes, so the rungs below cannot disagree on it.
fn probe_diverged(
    probe: usize,
    probe_id: &str,
    component: &'static str,
    index: usize,
    expected: String,
    observed: String,
    tolerance: &str,
) -> SetFitArtifactError {
    SetFitArtifactError::ProbeReplayFailed(Box::new(ProbeReplayDivergence {
        probe,
        probe_id: probe_id.to_string(),
        component,
        index,
        expected,
        observed,
        tolerance: tolerance.to_string(),
    }))
}

/// Decode one probe field's hex array. Rung 5 already proved it decodes.
fn decode_probe_field(values: &[String]) -> Vec<f32> {
    values.iter().filter_map(|hex| hex_to_f32(hex)).collect()
}

/// Replay all SIX probes through the rebuilt model, in the contract's order.
///
/// This is the production-side mirror of the train-time round trip: it catches
/// corrupted-but-checksummed states, wrong-loader states and platform math
/// divergence, none of which a checksum can see. "The served model is the
/// evaluated model" becomes a property this process RE-PROVES offline rather than
/// a training-time memory.
///
/// Order — counts, inputs, embeddings, logits, probabilities, labels — mirrors
/// `compare_probes` (verify.rs:476+). Counts come FIRST because every rung below
/// walks its pairs with `zip`, which STOPS at the shorter side: a partial
/// comparison that passed would be the worst possible outcome.
fn rung8_replay_probes(
    model: &SetFitMiniLm,
    head: &MultinomialLogisticRegression,
    parts: &SetFitAprParts,
) -> Result<(), SetFitArtifactError> {
    let doc = &parts.doc;
    let inputs = probe_inputs();
    if doc.probes.len() != PROBE_COUNT {
        return Err(probe_diverged(
            0,
            "<all>",
            "probe_count",
            0,
            PROBE_COUNT.to_string(),
            doc.probes.len().to_string(),
            "exact",
        ));
    }

    let width = doc.head.n_features;

    for (index, record) in doc.probes.iter().enumerate() {
        let id = PROBE_IDS[index];

        // Probe inputs are FIXED, SYNTHETIC and CONTRACT-RESIDENT (Pitfall 8):
        // an artifact carrying a dataset sentence here is refused, not replayed.
        if record.input != inputs[index] {
            return Err(probe_diverged(
                index,
                id,
                "input",
                0,
                format!("{:?}", inputs[index]),
                format!("{:?}", record.input),
                "exact",
            ));
        }

        let pooled = model.encode_texts(&[record.input.as_str()]).map_err(|e| {
            SetFitArtifactError::EncodeFailed {
                reason: format!("probe {index} ({id}): {e}"),
            }
        })?;
        if pooled.shape() != [1, width] {
            return Err(probe_diverged(
                index,
                id,
                "embedding_width",
                0,
                format!("{:?}", [1, width]),
                format!("{:?}", pooled.shape()),
                "exact",
            ));
        }
        let observed_embedding: Vec<f32> = pooled.data().to_vec();

        let expected_embedding = decode_probe_field(&record.embedding_hex);
        if expected_embedding.len() != observed_embedding.len() {
            return Err(probe_diverged(
                index,
                id,
                "embedding_width",
                0,
                expected_embedding.len().to_string(),
                observed_embedding.len().to_string(),
                "exact",
            ));
        }
        compare_probe_component(
            index,
            id,
            "embedding",
            &expected_embedding,
            &observed_embedding,
            PROBE_EMBEDDING_ABS_TOLERANCE,
        )?;

        // The logits are accumulated in `f64` in the SAME order the writer used
        // and then narrowed, so the recorded and the replayed logits cannot
        // describe two different computations.
        let observed_logits = replay_logits(head, &observed_embedding, index, id)?;
        let expected_logits = decode_probe_field(&record.logits_hex);
        if expected_logits.len() != observed_logits.len() {
            return Err(probe_diverged(
                index,
                id,
                "logit_count",
                0,
                expected_logits.len().to_string(),
                observed_logits.len().to_string(),
                "exact",
            ));
        }
        compare_probe_component(
            index,
            id,
            "logit",
            &expected_logits,
            &observed_logits,
            PROBE_LOGITS_ABS_TOLERANCE,
        )?;

        let observed_probabilities: Vec<f32> = head
            .predict_proba(&[observed_embedding.clone()])
            .map_err(|e| SetFitArtifactError::EncodeFailed {
                reason: format!("probe {index} ({id}): {e}"),
            })?
            .into_iter()
            .next()
            .ok_or_else(|| SetFitArtifactError::EncodeFailed {
                reason: format!("probe {index} ({id}): predict_proba returned no rows"),
            })?
            .into_iter()
            .map(|p| p as f32)
            .collect();
        let expected_probabilities = decode_probe_field(&record.probabilities_hex);
        if expected_probabilities.len() != observed_probabilities.len() {
            return Err(probe_diverged(
                index,
                id,
                "probability_count",
                0,
                expected_probabilities.len().to_string(),
                observed_probabilities.len().to_string(),
                "exact",
            ));
        }
        compare_probe_component(
            index,
            id,
            "probability",
            &expected_probabilities,
            &observed_probabilities,
            PROBE_PROBABILITIES_ABS_TOLERANCE,
        )?;

        // Labels are compared EXACTLY. A tolerance on a label is a category
        // error, and a "close enough" label is a wrong answer.
        let observed_label = head
            .predict(&[observed_embedding])
            .map_err(|e| SetFitArtifactError::EncodeFailed {
                reason: format!("probe {index} ({id}): {e}"),
            })?
            .into_iter()
            .next()
            .ok_or_else(|| SetFitArtifactError::EncodeFailed {
                reason: format!("probe {index} ({id}): predict returned no rows"),
            })?;
        if observed_label != record.label {
            return Err(probe_diverged(
                index,
                id,
                "label",
                0,
                record.label.clone(),
                observed_label,
                "exact",
            ));
        }
    }
    Ok(())
}

/// Recompute one probe's logits the way the writer recorded them.
///
/// Through the rebuilt head's `predict_logits` — THE single logit implementation,
/// and the very function `compute_probes` records with. A hand-written
/// accumulation here was two copies of one loop: the writer's and this one, whose
/// agreement was the whole property rung 8 exists to check. It also read the row
/// with `zip`, which STOPS at the shorter side, so an embedding narrower than the
/// head silently produced a truncated dot product instead of a refusal;
/// `predict_logits` reports a typed `FeatureDimMismatch` for exactly that input.
fn replay_logits(
    head: &MultinomialLogisticRegression,
    embedding: &[f32],
    probe: usize,
    probe_id: &str,
) -> Result<Vec<f32>, SetFitArtifactError> {
    let batch = [embedding.to_vec()];
    let rows = head
        .predict_logits(&batch)
        .map_err(|e| SetFitArtifactError::EncodeFailed {
            reason: format!("probe {probe} ({probe_id}): {e}"),
        })?;
    let row = rows
        .into_iter()
        .next()
        .ok_or_else(|| SetFitArtifactError::EncodeFailed {
            reason: format!("probe {probe} ({probe_id}): predict_logits returned no rows"),
        })?;
    Ok(row.into_iter().map(|z| z as f32).collect())
}

/// One tolerance-bounded component comparison, NaN-visible throughout.
fn compare_probe_component(
    probe: usize,
    probe_id: &str,
    component: &'static str,
    expected: &[f32],
    observed: &[f32],
    bound: f64,
) -> Result<(), SetFitArtifactError> {
    for (index, (want, got)) in expected.iter().zip(observed.iter()).enumerate() {
        let delta = f64::from((got - want).abs());
        if !within(delta, bound) {
            return Err(probe_diverged(
                probe,
                probe_id,
                component,
                index,
                format!("{want:e}"),
                format!("{got:e}"),
                &format!("{bound:e}"),
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// NaN-visible comparison (contract equation `probe_and_parity_tolerances`)
// ---------------------------------------------------------------------------

/// Whether `delta` is inside `bound`, with an INCOMPARABLE delta counting as OUTSIDE.
///
/// The contract MANDATES this exact form. A bare `delta <= bound` happens to
/// reject `NaN` in this direction, but the idiom is fragile under the obvious
/// refactor to `!(delta > bound)`, which ACCEPTS `NaN` silently. The explicit
/// `partial_cmp` form cannot be refactored into acceptance by accident. This is
/// verify.rs:337-342's `within`, kept identical so the train-time and load-time
/// comparators cannot disagree.
///
/// This is the ONE definition in `aprender-core`: `classify` calls it rather than
/// carrying a second copy, so the crate cannot end up with two comparators whose
/// divergence would be silent. `classify`'s
/// `within_is_nan_visible_in_both_argument_positions` scans THIS function's source
/// for the `partial_cmp` form, so the surviving copy is the guarded one.
pub(crate) fn within(delta: f64, bound: f64) -> bool {
    matches!(
        delta.partial_cmp(&bound),
        Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
    )
}

/// One `f32` recovered from the lowercase hex of its LITTLE-ENDIAN bit pattern.
///
/// The exact inverse of [`f32_bits_hex`]. Lowercase-only and length-exact on
/// purpose: this reads attacker-supplied text, and a lenient parser here would
/// accept documents the writer can never produce.
fn hex_to_f32(hex: &str) -> Option<f32> {
    if hex.len() != 8 {
        return None;
    }
    let mut bytes = [0u8; 4];
    for (index, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let hi = hex_nibble_value(pair[0])?;
        let lo = hex_nibble_value(pair[1])?;
        bytes[index] = (hi << 4) | lo;
    }
    Some(f32::from_bits(u32::from_le_bytes(bytes)))
}

/// One lowercase hex digit as its value, or `None`.
fn hex_nibble_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

// ===========================================================================
// Test fixtures
//
// `mod fixture`, `mod tests`, `mod nullable` and `mod determinism` are DIRECT
// children of `artifact`, not nested inside one `tests` module. That is forced
// by the acceptance criteria's filters: `cargo test` matches on the full test
// path, so `setfit::artifact::nullable` is a substring of
// `setfit::artifact::nullable::foo` but NOT of
// `setfit::artifact::tests::nullable::foo`. A nested layout would make both
// counted filters match ZERO tests — and a filter matching zero tests exits 0,
// which is precisely the CR-02 vacuous pass this phase exists to prevent.
// ===========================================================================

#[cfg(all(test, feature = "setfit"))]
pub(crate) mod fixture {
    //! The tiny fixture, in the PRODUCTION nullability shape.
    //!
    //! Two rules govern everything here, and both come from the contract rather
    //! than from convenience:
    //!
    //! 1. **The NAME TOPOLOGY is exact.** The fixture keeps the same per-layer
    //!    template set as the pinned model at reduced dimensions and layer
    //!    count, and it is judged by the SAME production rule. There is no
    //!    test-only schema exception.
    //! 2. **The NULLABILITY TOPOLOGY is exact too**, and this is the one an
    //!    earlier draft got wrong. [`fixture_view_full_pin_shape`] is the
    //!    DEFAULT because `architecture.vocab_remap: None` is what a pinned
    //!    MiniLM serializes (import.rs:501). The slice fixture sets `Some`
    //!    (import.rs:620), which emits NO null at that path — so a suite built
    //!    only on the slice shape earns every green count on the one input shape
    //!    at which the guard CANNOT fire, and an allowlist that omitted
    //!    `architecture.vocab_remap` would refuse every production artifact while
    //!    every test stayed green. That is the tests-pass/production-fails class
    //!    this phase exists to prevent.

    use super::*;

    use crate::setfit::{
        L2_EPS, MAX_SEQUENCE_LENGTH, NORMALIZATION_POLICY, PADDING_MODE, PINNED_ACTIVATION,
        PINNED_REVISION, POOLING_POLICY,
    };
    use serde_json::json;

    /// Reduced dimensions. `positions` is NOT reduced: the tokenizer truncates at
    /// [`MAX_SEQUENCE_LENGTH`], so `probe_truncation_boundary` produces a
    /// 256-position row and an encoder with fewer position rows would refuse it
    /// with `OversizeInput` before a single probe could be recorded.
    pub(super) const FIXTURE_HIDDEN: usize = 8;
    pub(super) const FIXTURE_HEADS: usize = 2;
    pub(super) const FIXTURE_HEAD_DIM: usize = FIXTURE_HIDDEN / FIXTURE_HEADS;
    pub(super) const FIXTURE_LAYERS: usize = 2;
    pub(super) const FIXTURE_INTERMEDIATE: usize = 16;
    pub(super) const FIXTURE_POSITIONS: usize = MAX_SEQUENCE_LENGTH;
    pub(super) const FIXTURE_TYPE_VOCAB: usize = 2;
    pub(super) const FIXTURE_LABELS: [&str; 3] = ["against", "favor", "neutral"];
    pub(super) const FIXTURE_ROOT_SEED: u64 = 0x0405_0000_0000_0002;
    pub(super) const FIXTURE_BUNDLE_SCHEMA_VERSION: u32 = 1;
    pub(super) const FIXTURE_FORMAT_ID: &str = "setfit-apr-v1-fixture";

    /// A tiny WordPiece vocabulary. Everything outside it becomes `[UNK]`, so
    /// every id the tokenizer can emit is `< TINY_VOCAB.len()` — which is what
    /// lets the fixture carry `vocab_remap: None` (the production shape) with a
    /// 48-row embedding table instead of the pin's 30522.
    pub(super) const TINY_VOCAB: [&str; 48] = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "ok",
        "few",
        "shot",
        "classification",
        "with",
        "contrastive",
        "pairs",
        "line",
        "one",
        "two",
        "tabbed",
        "spaced",
        "stance",
        "detection",
        "i",
        "firmly",
        "support",
        "this",
        "position",
        "el",
        "zorro",
        "cafe",
        "naive",
        "pi",
        "##s",
        "##ed",
        ".",
        ",",
        "!",
        "#",
        "@",
        ":",
        "/",
        "-",
        "=",
    ];

    /// A valid, self-contained `tokenizer.json` in the pinned file's exact shape
    /// (BertNormalizer + BertPreTokenizer + TemplateProcessing + WordPiece) with
    /// [`TINY_VOCAB`] substituted for the 30522-entry pin.
    ///
    /// Self-contained ON PURPOSE. Reading the committed
    /// `tests/fixtures/setfit/tokenizer.json` would work, but that path honours
    /// the `APRENDER_SETFIT_FIXTURES` override (tokenizer_tests.rs:41-49), so an
    /// environment that set it would silently change the artifact bytes and
    /// therefore the pinned golden hash — a gate whose expected value depends on
    /// an environment variable is not a gate.
    pub(super) fn tiny_tokenizer_json() -> Vec<u8> {
        let mut s = String::new();
        s.push_str(r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":["#);
        for (id, content) in ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
            .iter()
            .enumerate()
        {
            if id > 0 {
                s.push(',');
            }
            s.push_str(&format!(
                r#"{{"id":{id},"special":true,"content":"{content}","single_word":false,"lstrip":false,"rstrip":false,"normalized":false}}"#
            ));
        }
        s.push_str(
            r###"],"normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,"strip_accents":null,"lowercase":true},"pre_tokenizer":{"type":"BertPreTokenizer"},"post_processor":{"type":"TemplateProcessing","single":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}}],"pair":[{"SpecialToken":{"id":"[CLS]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},{"SpecialToken":{"id":"[SEP]","type_id":0}},{"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[SEP]","type_id":1}}],"special_tokens":{"[CLS]":{"id":"[CLS]","ids":[2],"tokens":["[CLS]"]},"[SEP]":{"id":"[SEP]","ids":[3],"tokens":["[SEP]"]}}},"decoder":{"type":"WordPiece","prefix":"##","cleanup":true},"model":{"type":"WordPiece","unk_token":"[UNK]","continuing_subword_prefix":"##","max_input_chars_per_word":100,"vocab":{"###,
        );
        for (id, token) in TINY_VOCAB.iter().enumerate() {
            if id > 0 {
                s.push(',');
            }
            s.push_str(&format!(r#""{token}":{id}"#));
        }
        s.push_str("}}}");
        s.into_bytes()
    }

    /// A deterministic, platform-independent filler.
    ///
    /// Every produced value is `k / 65536 - 0.5` for an integer `k`, so it is
    /// EXACTLY representable in `f32` on every target — the fixture's own bytes
    /// therefore cannot be a source of cross-platform hash drift.
    pub(super) struct Filler(u64);

    impl Filler {
        pub(super) fn new(seed: u64) -> Self {
            Self(seed | 1)
        }

        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let quantum = ((self.0 >> 40) & 0xFFFF) as f32 / 65536.0;
            quantum - 0.5
        }

        pub(super) fn vec(&mut self, n: usize) -> Vec<f32> {
            (0..n).map(|_| self.next()).collect()
        }
    }

    pub(super) fn fixture_architecture(vocab_remap: Option<Vec<u32>>) -> EncoderArchitecture {
        EncoderArchitecture {
            hidden: FIXTURE_HIDDEN,
            heads: FIXTURE_HEADS,
            head_dim: FIXTURE_HEAD_DIM,
            num_layers: FIXTURE_LAYERS,
            intermediate: FIXTURE_INTERMEDIATE,
            vocab: TINY_VOCAB.len(),
            positions: FIXTURE_POSITIONS,
            type_vocab_size: FIXTURE_TYPE_VOCAB,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            hidden_act: PINNED_ACTIVATION.to_string(),
            source_revision: PINNED_REVISION.to_string(),
            tokenizer_sha256: sha256_hex(&tiny_tokenizer_json()),
            vocab_remap,
        }
    }

    pub(super) fn fixture_tensors(
        arch: &EncoderArchitecture,
    ) -> BTreeMap<String, (Vec<usize>, Vec<f32>)> {
        let h = arch.hidden;
        let im = arch.intermediate;
        let mut f = Filler::new(0x0402_0001);
        let mut t: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();
        // Not `mut`: the closure captures nothing, so every mutation it performs
        // arrives through its two `&mut` parameters. `rustc` warns on the binding.
        let put = |t: &mut BTreeMap<String, (Vec<usize>, Vec<f32>)>,
                   f: &mut Filler,
                   name: String,
                   shape: Vec<usize>| {
            let n = shape.iter().product();
            t.insert(name, (shape, f.vec(n)));
        };

        put(
            &mut t,
            &mut f,
            "embeddings.word_embeddings.weight".to_string(),
            vec![arch.vocab, h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.position_embeddings.weight".to_string(),
            vec![arch.positions, h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.token_type_embeddings.weight".to_string(),
            vec![arch.type_vocab_size, h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.LayerNorm.weight".to_string(),
            vec![h],
        );
        put(
            &mut t,
            &mut f,
            "embeddings.LayerNorm.bias".to_string(),
            vec![h],
        );

        for n in 0..arch.num_layers {
            let p = format!("encoder.layer.{n}");
            for leaf in ["query", "key", "value"] {
                put(
                    &mut t,
                    &mut f,
                    format!("{p}.attention.self.{leaf}.weight"),
                    vec![h, h],
                );
                put(
                    &mut t,
                    &mut f,
                    format!("{p}.attention.self.{leaf}.bias"),
                    vec![h],
                );
            }
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.dense.weight"),
                vec![h, h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.dense.bias"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.LayerNorm.weight"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.attention.output.LayerNorm.bias"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.intermediate.dense.weight"),
                vec![im, h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.intermediate.dense.bias"),
                vec![im],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.output.dense.weight"),
                vec![h, im],
            );
            put(&mut t, &mut f, format!("{p}.output.dense.bias"), vec![h]);
            put(
                &mut t,
                &mut f,
                format!("{p}.output.LayerNorm.weight"),
                vec![h],
            );
            put(
                &mut t,
                &mut f,
                format!("{p}.output.LayerNorm.bias"),
                vec![h],
            );
        }
        t
    }

    /// `pair_config.budget` and `pair_config.hard_cap` are `null`: both are
    /// absent-by-default knobs, so both are routinely null on an HONEST artifact.
    pub(super) fn fixture_requested_config() -> Value {
        json!({
            "max_length": 256,
            "pair_config": { "budget": null, "hard_cap": null, "strategy": "all_pairs" },
            "requested_device": "cpu",
            "seed": 7
        })
    }

    /// `ResolvedConfigRecord` — one `String`, ZERO allowlisted paths, WALKED anyway.
    pub(super) fn fixture_resolved_config() -> Value {
        json!({ "resolved_device": "cpu" })
    }

    /// `epsilon_used` is `null` — the production shape (evidence.rs:655). The
    /// fractional `f64`s are here on purpose: they are what makes the
    /// number-formatting half of the round-trip stability proof non-vacuous.
    pub(super) fn fixture_evidence() -> Value {
        json!({
            "epsilon_used": null,
            "table_hash": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "per_class": {
                "against": { "support": 8, "mean_margin": 0.1 },
                "favor": { "support": 8, "mean_margin": 0.333_333_333_333_333_3 },
                "neutral": { "support": 8, "mean_margin": 2.220_446_049_250_313e-16 }
            }
        })
    }

    /// `ProvenanceRecord` — four `String`, one `u64`, one `u32`; ZERO allowlisted
    /// paths, WALKED anyway, and fully populated here so the full-pin fixture
    /// carries no null in this subtree at all.
    pub(super) fn fixture_provenance() -> Value {
        json!({
            "dataset_fingerprint": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
            "validation_split_fingerprint": "vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv",
            "selection_semantic_hash": "ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss",
            "selection_ledger_hash": "llllllllllllllllllllllllllllllllllllllllllllllllllllllllllllllll",
            "selection_root_seed": 11,
            "shots_per_class": 8
        })
    }

    fn fixture_view(vocab_remap: Option<Vec<u32>>) -> SetFitArtifactView {
        let architecture = fixture_architecture(vocab_remap);
        let tensors = fixture_tensors(&architecture);
        let mut head = Filler::new(0x0402_0002);
        let k = FIXTURE_LABELS.len();
        SetFitArtifactView {
            bundle_schema_version: FIXTURE_BUNDLE_SCHEMA_VERSION,
            format_id: FIXTURE_FORMAT_ID.to_string(),
            tokenizer_bytes: tiny_tokenizer_json(),
            head_weights: head.vec(k * architecture.hidden),
            head_intercepts: head.vec(k),
            head_n_features: architecture.hidden,
            architecture,
            tensors,
            pooling: POOLING_POLICY.to_string(),
            normalization: NORMALIZATION_POLICY.to_string(),
            l2_epsilon: L2_EPS,
            truncation_max_sequence_length: MAX_SEQUENCE_LENGTH as u32,
            padding_mode: PADDING_MODE.to_string(),
            max_length: MAX_SEQUENCE_LENGTH as u32,
            root_seed: FIXTURE_ROOT_SEED,
            ordered_labels: FIXTURE_LABELS.iter().map(|s| (*s).to_string()).collect(),
            requested_config: fixture_requested_config(),
            resolved_config: fixture_resolved_config(),
            evidence: fixture_evidence(),
            provenance: fixture_provenance(),
        }
    }

    /// THE DEFAULT BUILDER. `architecture.vocab_remap: None` — the FULL PIN.
    ///
    /// Every suite that touches the writer must exercise this shape by default,
    /// and every downstream plan that "builds the fixture artifact the same way"
    /// inherits it from here.
    /// `pub(crate)`, not `pub(super)`: `setfit::classify`'s suites (plan 04-04)
    /// build their model from THIS fixture. A second fixture assembled next door
    /// would be a second definition of "the artifact shape under test", free to
    /// drift from this one — the classify suites would then earn their green on
    /// a shape the writer never produces.
    pub(crate) fn fixture_view_full_pin_shape() -> SetFitArtifactView {
        fixture_view(None)
    }

    /// The slice topology, used ONLY by the tests that need `vocab_remap: Some`.
    ///
    /// The identity remap is the smallest one that is valid for this vocabulary:
    /// `slice_to_orig[i] == i`, so `slice_vocab() == dims.vocab` and every id the
    /// tiny tokenizer emits resolves.
    pub(super) fn fixture_view_slice_shape() -> SetFitArtifactView {
        let remap: Vec<u32> = (0..TINY_VOCAB.len())
            .map(|i| u32::try_from(i).expect("48 fits in u32"))
            .collect();
        fixture_view(Some(remap))
    }
}

#[cfg(all(test, feature = "setfit"))]
mod tests {
    //! Writer behaviour: the tensor set, the head tensors, the one-key document,
    //! the typed refusals and the probe records.

    use super::fixture::*;
    use super::*;

    use crate::format::v2::{AprV2Flags, AprV2Reader};

    /// Read the ONE custom document back out of written bytes.
    fn read_doc(bytes: &[u8]) -> JsonMap<String, Value> {
        let reader =
            AprV2Reader::from_bytes(bytes).expect("the writer emits a parseable container");
        reader
            .metadata()
            .custom
            .get(CUSTOM_METADATA_KEY)
            .expect("the one custom key is present")
            .as_object()
            .expect("the custom key holds a JSON object")
            .clone()
    }

    fn written(view: &SetFitArtifactView) -> Vec<u8> {
        write_setfit_apr(view).expect("the fixture view is writable")
    }

    #[test]
    fn writer_writes_exactly_the_architecture_derived_tensor_set() {
        let view = fixture_view_full_pin_shape();
        let bytes = written(&view);
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        let observed: BTreeSet<String> = reader
            .tensor_names()
            .into_iter()
            .map(str::to_string)
            .collect();
        let expected = expected_tensor_names(view.architecture.num_layers);
        assert_eq!(observed, expected);
        // The count is DERIVED (5 global + 16 per layer + 3 schema-owned), never
        // quoted: the same arithmetic gives 104 for the pinned six-layer model.
        assert_eq!(expected.len(), 5 + 16 * FIXTURE_LAYERS + 3);
    }

    #[test]
    fn the_head_is_two_named_f32_tensors_with_the_declared_shapes() {
        let view = fixture_view_full_pin_shape();
        let bytes = written(&view);
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");

        let weight = reader
            .get_tensor(HEAD_WEIGHT_TENSOR)
            .expect("setfit.head.weight is a first-class tensor entry");
        assert_eq!(
            weight.shape,
            vec![FIXTURE_LABELS.len(), view.head_n_features]
        );
        let bias = reader
            .get_tensor(HEAD_BIAS_TENSOR)
            .expect("setfit.head.bias is a first-class tensor entry");
        assert_eq!(bias.shape, vec![FIXTURE_LABELS.len()]);

        // Bit-exact round trip, not "close enough": the payload is raw LE f32.
        assert_eq!(
            reader.get_f32_tensor(HEAD_WEIGHT_TENSOR).expect("f32"),
            view.head_weights
        );
        assert_eq!(
            reader.get_f32_tensor(HEAD_BIAS_TENSOR).expect("f32"),
            view.head_intercepts
        );
    }

    #[test]
    fn metadata_declares_the_setfit_model_type_and_exactly_one_custom_key() {
        let bytes = written(&fixture_view_full_pin_shape());
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        assert_eq!(reader.metadata().model_type, MODEL_TYPE_TAG);
        assert_eq!(
            reader.metadata().custom.len(),
            1,
            "N custom keys serialize in HashMap iteration order and are not reproducible"
        );
        assert!(reader.metadata().custom.contains_key(CUSTOM_METADATA_KEY));
        assert_eq!(
            reader.metadata().created_at,
            None,
            "a timestamp would make two artifacts of the same run differ"
        );
    }

    #[test]
    fn the_doc_key_set_equals_the_contract_field_list_exactly() {
        let doc = read_doc(&written(&fixture_view_full_pin_shape()));
        let observed: BTreeSet<&str> = doc.keys().map(String::as_str).collect();
        let expected: BTreeSet<&str> = SETFIT_ARTIFACT_DOC_FIELDS.iter().copied().collect();
        assert_eq!(
            observed, expected,
            "an added, renamed or missing doc field must be a LOUD failure"
        );
        assert_eq!(doc["schema"], Value::String(ARTIFACT_SCHEMA.to_string()));
        assert_eq!(doc["schema_version"], Value::from(ARTIFACT_SCHEMA_VERSION));
    }

    #[test]
    fn the_doc_carries_the_first_level_identity_fields_and_the_nested_groups() {
        let view = fixture_view_full_pin_shape();
        let doc = read_doc(&written(&view));
        assert_eq!(
            doc["tokenizer_sha256"],
            Value::String(view.architecture.tokenizer_sha256.clone())
        );
        assert_eq!(
            doc["ordered_labels"],
            serde_json::to_value(&view.ordered_labels).expect("labels serialize")
        );
        let preprocessing = doc["preprocessing"].as_object().expect("object");
        assert_eq!(
            preprocessing["l2_epsilon_hex"],
            Value::String(f32_bits_hex(view.l2_epsilon)),
            "every doc-OWNED float is a bit-pattern hex string, never a JSON number"
        );
        assert_eq!(
            preprocessing["pooling"],
            Value::String(view.pooling.clone())
        );
        let head = doc["head"].as_object().expect("object");
        assert_eq!(head["n_features"], Value::from(view.head_n_features));
        assert_eq!(head["num_labels"], Value::from(view.ordered_labels.len()));
    }

    #[test]
    fn a_missing_encoder_tensor_is_a_typed_incomplete_tensor_set() {
        let mut view = fixture_view_full_pin_shape();
        view.tensors
            .remove("encoder.layer.1.output.LayerNorm.bias")
            .expect("the fixture carries it");
        let err = write_setfit_apr(&view).expect_err("a partial artifact must not be produced");
        assert!(
            matches!(&err, SetFitArtifactError::IncompleteTensorSet { missing }
                if missing == &vec!["encoder.layer.1.output.LayerNorm.bias".to_string()]),
            "got {err:?}"
        );
    }

    #[test]
    fn an_unmapped_hf_tensor_name_is_a_typed_refusal() {
        let mut view = fixture_view_full_pin_shape();
        view.tensors.insert(
            "encoder.layer.0.attention.self.rotary.weight".to_string(),
            (vec![1], vec![0.0]),
        );
        let err = write_setfit_apr(&view).expect_err("an unnamed tensor must not be dropped");
        assert!(
            matches!(&err, SetFitArtifactError::UnmappedTensorName { hf_name }
                if hf_name == "encoder.layer.0.attention.self.rotary.weight"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_non_finite_encoder_tensor_value_is_refused_before_serialization() {
        let mut view = fixture_view_full_pin_shape();
        let entry = view
            .tensors
            .get_mut("embeddings.LayerNorm.bias")
            .expect("present");
        entry.1[2] = f32::NAN;
        let err = write_setfit_apr(&view).expect_err("NaN must never reach the payload");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "tensors.embeddings.LayerNorm.bias[2]"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_non_finite_head_coefficient_is_refused_before_serialization() {
        let mut view = fixture_view_full_pin_shape();
        view.head_intercepts[1] = f32::INFINITY;
        let err = write_setfit_apr(&view).expect_err("+Inf must never reach the payload");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "head_intercepts[1]"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_head_arity_that_disagrees_with_the_label_set_is_refused() {
        let mut view = fixture_view_full_pin_shape();
        view.head_weights.push(0.5);
        let err = write_setfit_apr(&view).expect_err("K*d is not negotiable");
        assert!(
            matches!(&err, SetFitArtifactError::InconsistentTensorSet { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn a_head_feature_dimension_that_disagrees_with_the_encoder_width_is_refused() {
        let mut view = fixture_view_full_pin_shape();
        view.head_n_features = FIXTURE_HIDDEN + 1;
        view.head_weights = vec![0.25; FIXTURE_LABELS.len() * view.head_n_features];
        let err = write_setfit_apr(&view)
            .expect_err("a head that cannot consume this encoder's embedding is not shippable");
        assert!(
            matches!(&err, SetFitArtifactError::InconsistentTensorSet { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn the_tokenizer_blob_round_trips_byte_identically_and_matches_the_recorded_digest() {
        let view = fixture_view_full_pin_shape();
        let bytes = written(&view);
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        let entry = reader
            .get_tensor(TOKENIZER_BLOB_TENSOR)
            .expect("tokenizer.blob is a tensor entry");
        assert_eq!(entry.dtype, TensorDType::U8);
        assert_eq!(entry.shape, vec![view.tokenizer_bytes.len()]);
        let payload = reader
            .get_tensor_data(TOKENIZER_BLOB_TENSOR)
            .expect("payload");
        // BYTE-IDENTICAL: an artifact carrying only the digest could DETECT a
        // substituted tokenizer and could not REBUILD the right one.
        assert_eq!(payload, view.tokenizer_bytes.as_slice());
        let doc = read_doc(&bytes);
        assert_eq!(
            doc["tokenizer_sha256"],
            Value::String(sha256_hex(payload)),
            "the recorded digest must describe the bytes that travelled"
        );
    }

    #[test]
    fn a_tokenizer_digest_that_does_not_describe_the_bytes_is_refused() {
        let mut view = fixture_view_full_pin_shape();
        view.architecture.tokenizer_sha256 = "0".repeat(64);
        let err = write_setfit_apr(&view)
            .expect_err("a mis-paired tokenizer produces confidently wrong embeddings");
        assert!(
            matches!(&err, SetFitArtifactError::TokenizerHashMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn the_hf_name_map_is_written_total_and_injective_over_the_encoder_tensor_set() {
        let view = fixture_view_full_pin_shape();
        let doc = read_doc(&written(&view));
        let map = doc["hf_name_map"].as_object().expect("object");
        assert_eq!(map.len(), view.tensors.len(), "total over the tensor set");
        let canonical: BTreeSet<&str> = map.values().filter_map(Value::as_str).collect();
        assert_eq!(
            canonical.len(),
            map.len(),
            "injective: no two HF names collide"
        );
        for hf in view.tensors.keys() {
            assert!(map.contains_key(hf), "{hf} has no canonical entry");
        }
        // The map is WRITTEN INTO the artifact precisely so `deserialize` inverts
        // a map the artifact carries rather than re-deriving names from a table
        // that may have moved between the write and the read.
        assert_eq!(
            map["embeddings.position_embeddings.weight"],
            Value::String("position_embd.weight".to_string())
        );
    }

    #[test]
    fn the_six_probe_records_use_the_contract_resident_inputs_only() {
        let doc = read_doc(&written(&fixture_view_full_pin_shape()));
        let probes = doc["probes"].as_array().expect("array");
        assert_eq!(probes.len(), PROBE_COUNT);
        let expected = probe_inputs();
        for (index, probe) in probes.iter().enumerate() {
            let record = probe.as_object().expect("object");
            let keys: BTreeSet<&str> = record.keys().map(String::as_str).collect();
            assert_eq!(
                keys,
                [
                    "embedding_hex",
                    "input",
                    "label",
                    "logits_hex",
                    "probabilities_hex"
                ]
                .into_iter()
                .collect::<BTreeSet<&str>>()
            );
            assert_eq!(record["input"], Value::String(expected[index].clone()));
        }
        // T-04-04: no dataset text is embedded. The truncation probe is a
        // repetition of a contract-resident unit, not a corpus sample.
        assert!(expected[2].starts_with(PROBE_TRUNCATION_REPEAT_UNIT));
        assert_eq!(
            expected[2].len(),
            PROBE_TRUNCATION_REPEAT_UNIT.len() * PROBE_TRUNCATION_REPEAT_COUNT
        );
    }

    #[test]
    fn probe_expectations_are_bit_pattern_hex_and_the_label_comes_from_the_label_set() {
        let view = fixture_view_full_pin_shape();
        let doc = read_doc(&written(&view));
        let probes = doc["probes"].as_array().expect("array");
        for probe in probes {
            let record = probe.as_object().expect("object");
            let embedding = record["embedding_hex"].as_array().expect("array");
            assert_eq!(embedding.len(), view.architecture.hidden);
            for value in embedding {
                let hex = value.as_str().expect("hex string");
                assert_eq!(hex.len(), 8, "an f32 bit pattern is 8 hex characters");
                assert!(hex
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
            }
            for field in ["logits_hex", "probabilities_hex"] {
                let values = record[field].as_array().expect("array");
                assert_eq!(values.len(), view.ordered_labels.len());
            }
            let label = record["label"].as_str().expect("string");
            assert!(view.ordered_labels.iter().any(|l| l == label), "{label}");
        }
    }

    #[test]
    fn the_slice_shape_fixture_writes_under_the_same_production_rule() {
        let view = fixture_view_slice_shape();
        assert!(view.architecture.vocab_remap.is_some());
        let bytes = write_setfit_apr(&view).expect("the slice topology is writable");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        let observed: BTreeSet<String> = reader
            .tensor_names()
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(
            observed,
            expected_tensor_names(view.architecture.num_layers),
            "the same rule judges the fixture and the pin; there is no test-only branch"
        );
    }

    #[test]
    fn every_tensor_is_written_row_major_and_the_container_says_so() {
        let bytes = written(&fixture_view_full_pin_shape());
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        assert!(
            reader.header().flags.contains(AprV2Flags::LAYOUT_ROW_MAJOR),
            "LAYOUT-001/002: there is no GGUF import path into this writer"
        );
        assert!(!reader
            .header()
            .flags
            .contains(AprV2Flags::LAYOUT_COLUMN_MAJOR));
    }

    #[test]
    fn the_per_entry_structural_size_rule_holds_for_every_written_tensor() {
        let bytes = written(&fixture_view_full_pin_shape());
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        for entry in reader.tensor_index() {
            let elements: usize = entry.shape.iter().product();
            let width = if entry.dtype == TensorDType::U8 { 1 } else { 4 };
            assert_eq!(
                entry.size as usize,
                elements * width,
                "{}: declared size disagrees with declared shape",
                entry.name
            );
        }
    }

    #[test]
    fn artifact_sha256_hex_is_the_sha256_of_the_bytes_it_was_given() {
        let bytes = written(&fixture_view_full_pin_shape());
        assert_eq!(artifact_sha256_hex(&bytes), sha256_hex(&bytes));
        assert_eq!(artifact_sha256_hex(&bytes).len(), 64);
        assert_ne!(
            artifact_sha256_hex(&bytes),
            artifact_sha256_hex(&bytes[1..])
        );
    }

    #[test]
    fn canonical_name_for_hf_agrees_with_the_one_name_table() {
        let map = build_hf_name_map(FIXTURE_LAYERS);
        for (hf, canonical) in &map {
            assert_eq!(
                canonical_name_for_hf(hf, FIXTURE_LAYERS).as_ref(),
                Some(canonical),
                "{hf}"
            );
        }
        // A layer this architecture does not have has no canonical name, because
        // the expansion is a FUNCTION of num_layers rather than a fixed list.
        assert_eq!(
            canonical_name_for_hf(
                "encoder.layer.9.attention.self.query.weight",
                FIXTURE_LAYERS
            ),
            None
        );
        assert_eq!(
            canonical_name_for_hf("pooler.dense.weight", FIXTURE_LAYERS),
            None
        );
    }

    #[test]
    fn f32_bit_pattern_hex_matches_the_bundle_little_endian_precedent() {
        // bundle.rs:699-715 encodes `value.to_bits().to_le_bytes()`. 1.0f32 is
        // 0x3f800000, whose LE bytes are 00 00 80 3f. A big-endian rendering
        // would read "3f800000" and would silently disagree with the bundle
        // about the same number.
        assert_eq!(f32_bits_hex(1.0), "0000803f");
        assert_eq!(f32_bits_hex(0.0), "00000000");
        assert_eq!(f32_bits_hex(-2.0), "000000c0");
    }
}

#[cfg(all(test, feature = "setfit"))]
mod nullable {
    //! The four-path allowlist over five walked sub-documents.
    //!
    //! ACCEPT cases are not optional and are counted separately. A suite with
    //! zero accept cases would have passed with an allowlist containing only
    //! `evidence.epsilon_used` — an allowlist that refuses every production
    //! artifact — because every fixture used the slice shape, which sets
    //! `vocab_remap: Some(..)` and never reaches the failing path.

    use super::fixture::*;
    use super::*;

    use crate::format::v2::AprV2Reader;

    fn doc_of(view: &SetFitArtifactView) -> JsonMap<String, Value> {
        let bytes = write_setfit_apr(view).expect("writable");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        reader
            .metadata()
            .custom
            .get(CUSTOM_METADATA_KEY)
            .expect("one custom key")
            .as_object()
            .expect("object")
            .clone()
    }

    // ---- ACCEPT ----------------------------------------------------------

    #[test]
    fn accept_the_production_full_pin_shape_whose_vocab_remap_is_none() {
        let view = fixture_view_full_pin_shape();
        assert!(
            view.architecture.vocab_remap.is_none(),
            "the DEFAULT fixture must carry the production nullability shape"
        );
        let bytes = write_setfit_apr(&view)
            .expect("a guard that rejects the full pin refuses every real artifact");
        assert!(!bytes.is_empty());
        let doc = doc_of(&view);
        assert!(
            observed_null_paths(&doc).contains(&"architecture.vocab_remap".to_string()),
            "the accept case must actually REACH the allowlisted path"
        );
    }

    #[test]
    fn accept_a_view_whose_pair_config_budget_and_hard_cap_are_both_none() {
        let view = fixture_view_full_pin_shape();
        let doc = doc_of(&view);
        let observed = observed_null_paths(&doc);
        assert!(observed.contains(&"requested_config.pair_config.budget".to_string()));
        assert!(observed.contains(&"requested_config.pair_config.hard_cap".to_string()));
        assert!(disallowed_null_paths(&doc).is_empty());
    }

    #[test]
    fn accept_a_view_whose_evidence_epsilon_used_is_none() {
        let view = fixture_view_full_pin_shape();
        let doc = doc_of(&view);
        assert!(observed_null_paths(&doc).contains(&"evidence.epsilon_used".to_string()));
        assert!(write_setfit_apr(&view).is_ok());
    }

    #[test]
    fn the_full_pin_fixture_emits_exactly_the_four_allowlisted_nulls_and_no_others() {
        let doc = doc_of(&fixture_view_full_pin_shape());
        let observed: BTreeSet<String> = observed_null_paths(&doc).into_iter().collect();
        let allowed: BTreeSet<String> = NULLABLE_PATH_ALLOWLIST
            .iter()
            .map(|p| (*p).to_string())
            .collect();
        assert_eq!(observed, allowed);
    }

    // ---- REJECT ----------------------------------------------------------

    #[test]
    fn reject_a_null_inside_provenance_and_name_the_path() {
        // THE case that proves the FIFTH subtree is actually walked: an unwalked
        // subtree accepts every null silently, so a suite without this test
        // cannot tell a walked provenance from an unwalked one.
        let mut view = fixture_view_full_pin_shape();
        view.provenance["dataset_fingerprint"] = Value::Null;
        let err = write_setfit_apr(&view).expect_err("provenance is one of the five");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "provenance.dataset_fingerprint"),
            "got {err:?}"
        );
    }

    #[test]
    fn reject_a_null_at_resolved_config_resolved_device_and_name_the_path() {
        let mut view = fixture_view_full_pin_shape();
        view.resolved_config["resolved_device"] = Value::Null;
        let err = write_setfit_apr(&view).expect_err("resolved_config is one of the five");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "resolved_config.resolved_device"),
            "got {err:?}"
        );
    }

    #[test]
    fn reject_a_non_finite_f64_smuggled_into_evidence_as_a_null() {
        // This is the CR-03 signature end to end: `serde_json` renders every
        // non-finite f64 as `null` SILENTLY, so the value is already gone by the
        // time the walk sees it — which is exactly why the walk exists.
        let mut view = fixture_view_full_pin_shape();
        view.evidence["per_class"]["favor"]["mean_margin"] =
            serde_json::to_value(f64::NAN).expect("serde_json maps non-finite f64 to null");
        assert_eq!(
            view.evidence["per_class"]["favor"]["mean_margin"],
            Value::Null
        );
        let err = write_setfit_apr(&view).expect_err("a destroyed measurement must not ship");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "evidence.per_class.favor.mean_margin"),
            "got {err:?}"
        );
    }

    #[test]
    fn reject_a_null_at_architecture_hidden_act_and_name_the_path() {
        // `architecture` is derived from the TYPED `EncoderArchitecture`, whose
        // only `Option` is the allowlisted `vocab_remap` — so no view can smuggle
        // a null in here, which is a stronger guarantee than a check. The guard
        // is still exercised at the exact function `write_setfit_apr` calls, so
        // that a future `Option` on that record is caught the moment it lands.
        let mut doc = doc_of(&fixture_view_full_pin_shape());
        doc["architecture"]["hidden_act"] = Value::Null;
        let err = guard_subdocument_nulls(&doc).expect_err("architecture is one of the five");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "architecture.hidden_act"),
            "got {err:?}"
        );
    }

    #[test]
    fn the_walk_collects_every_offender_and_the_refusal_names_the_first() {
        // Two offenders, in two different sub-documents. The walk must SEE both
        // — a first-only walk could not report the set — while the refusal names
        // the first in document order, which is the order of WALKED_SUBDOCUMENTS.
        let mut view = fixture_view_full_pin_shape();
        view.evidence["table_hash"] = Value::Null;
        view.provenance["shots_per_class"] = Value::Null;
        let doc = {
            let mut doc = doc_of(&fixture_view_full_pin_shape());
            doc["evidence"]["table_hash"] = Value::Null;
            doc["provenance"]["shots_per_class"] = Value::Null;
            doc
        };
        assert_eq!(
            disallowed_null_paths(&doc),
            vec![
                "evidence.table_hash".to_string(),
                "provenance.shots_per_class".to_string(),
            ]
        );
        assert_eq!(
            first_unallowed_null_path(&doc),
            Some("evidence.table_hash".to_string())
        );
        let err = write_setfit_apr(&view).expect_err("two offenders is still a refusal");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "evidence.table_hash"),
            "got {err:?}"
        );
    }

    // ---- THE CONSTANTS THEMSELVES ---------------------------------------

    #[test]
    fn the_allowlist_constant_is_exactly_the_contracts_four_paths() {
        assert_eq!(NULLABLE_PATH_ALLOWLIST.len(), 4);
        assert_eq!(
            NULLABLE_PATH_ALLOWLIST,
            [
                "architecture.vocab_remap",
                "requested_config.pair_config.budget",
                "requested_config.pair_config.hard_cap",
                "evidence.epsilon_used",
            ],
            "a fifth entry or a missing entry is a silent widening of the guard"
        );
    }

    #[test]
    fn the_walk_covers_all_five_sub_documents() {
        assert_eq!(WALKED_SUBDOCUMENTS.len(), 5);
        assert_eq!(
            WALKED_SUBDOCUMENTS,
            [
                "architecture",
                "requested_config",
                "resolved_config",
                "evidence",
                "provenance",
            ]
        );
        // The two numbers differ on purpose: `resolved_config` and `provenance`
        // contribute ZERO allowlisted paths and are walked anyway.
        assert!(WALKED_SUBDOCUMENTS.len() > NULLABLE_PATH_ALLOWLIST.len());
        for name in WALKED_SUBDOCUMENTS {
            assert!(
                SETFIT_ARTIFACT_DOC_FIELDS.contains(&name),
                "{name} must be a doc field to be walkable"
            );
        }
    }

    #[test]
    fn the_walk_does_not_cover_the_containers_typed_metadata_nulls() {
        // `license`, `data_source` and `data_license` serialize as explicit
        // `null` by container design (FALSIFY-SHIP-022). Pointing the walk at
        // them would refuse every artifact this writer produces.
        let bytes = write_setfit_apr(&fixture_view_full_pin_shape()).expect("writable");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        assert_eq!(reader.metadata().license, None);
        assert_eq!(reader.metadata().data_source, None);
        assert_eq!(reader.metadata().data_license, None);
        for name in WALKED_SUBDOCUMENTS {
            assert!(!["license", "data_source", "data_license"].contains(&name));
        }
    }
}

#[cfg(all(test, feature = "setfit"))]
mod determinism {
    //! Determinism proven where it can actually break.

    use super::fixture::*;
    use super::*;

    use crate::format::v2::{AprV2Metadata, AprV2Reader};

    /// Set in the child process only.
    const CHILD_ENV: &str = "SETFIT_APR_DETERMINISM_CHILD";

    /// The child's single line of output.
    const CHILD_MARKER: &str = "SETFIT_APR_CHILD_SHA256=";

    /// The SHA-256 of the tiny fixture artifact IN ITS DEFAULT FULL-PIN
    /// NULLABILITY SHAPE (`architecture.vocab_remap: None`).
    ///
    /// The name states the shape on purpose: a later reader must be able to tell
    /// WHICH fixture this pins without reading the builder. A drift here is
    /// either a deliberate writer change (re-bless it, Ph1 D-13) or a real
    /// finding — the artifact hash IS the identity every Phase 4 response
    /// carries, so a platform that produces different bytes for this view has a
    /// parity defect, not a flaky test.
    const GOLDEN_SHA256_FIXTURE_VIEW_FULL_PIN_SHAPE: &str =
        "13e5c2965e95fc970c19a93f298a33b123f5c524a03c1e33e4a0e36967000bf4";

    #[test]
    fn two_writes_of_one_view_are_byte_identical_in_one_process() {
        let view = fixture_view_full_pin_shape();
        let first = write_setfit_apr(&view).expect("writable");
        let second = write_setfit_apr(&view).expect("writable");
        assert_eq!(first, second);
        assert_eq!(artifact_sha256_hex(&first), artifact_sha256_hex(&second));
    }

    #[test]
    fn metadata_survives_a_parse_and_re_serialize_byte_identically() {
        let bytes = write_setfit_apr(&fixture_view_full_pin_shape()).expect("writable");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        let once = reader.metadata().to_json().expect("metadata serializes");
        let reparsed = AprV2Metadata::from_json(&once).expect("metadata parses");
        let twice = reparsed.to_json().expect("metadata re-serializes");
        assert_eq!(once, twice, "write -> parse -> write must be the identity");
        assert_eq!(
            reparsed.custom.len(),
            1,
            "exactly one top-level custom key beyond the typed fields"
        );
    }

    #[test]
    fn all_five_sub_documents_round_trip_byte_identically_including_the_allowlisted_nulls() {
        let view = fixture_view_full_pin_shape();
        let bytes = write_setfit_apr(&view).expect("writable");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        let doc = reader
            .metadata()
            .custom
            .get(CUSTOM_METADATA_KEY)
            .expect("one custom key")
            .as_object()
            .expect("object")
            .clone();

        for name in WALKED_SUBDOCUMENTS {
            let sub = doc.get(name).expect("all five sub-documents are present");
            let once = serde_json::to_vec(sub).expect("serializes");
            let back: Value = serde_json::from_slice(&once).expect("parses");
            let twice = serde_json::to_vec(&back).expect("re-serializes");
            assert_eq!(once, twice, "{name} is not round-trip stable");
        }

        // The number-formatting half is only exercised if a fractional f64 is
        // actually present; assert that rather than assume it.
        let evidence = serde_json::to_string(&doc["evidence"]).expect("string");
        assert!(evidence.contains("0.3333333333333333"), "got {evidence}");

        // A `null -> None -> null` round trip is byte-STABLE, which is exactly
        // why closure alone cannot substitute for the write-time null scan: an
        // allowlisted null round-trips perfectly while carrying no value.
        assert!(observed_null_paths(&doc).contains(&"architecture.vocab_remap".to_string()));
        assert!(observed_null_paths(&doc).contains(&"evidence.epsilon_used".to_string()));
        assert!(doc.contains_key("provenance"));
    }

    #[test]
    fn the_fixture_artifact_hash_matches_the_committed_golden() {
        let bytes = write_setfit_apr(&fixture_view_full_pin_shape()).expect("writable");
        assert_eq!(
            artifact_sha256_hex(&bytes),
            GOLDEN_SHA256_FIXTURE_VIEW_FULL_PIN_SHAPE
        );
    }

    #[test]
    fn the_public_writer_path_can_never_produce_two_custom_keys() {
        // In-band negative: build metadata with TWO custom keys by hand and show
        // the resulting instability is what the one-key rule prevents. `custom`
        // is a `HashMap` with `RandomState`, so with >1 key the serialized order
        // is unspecified — three runs of one binary produced three orders when
        // this was measured. The public writer is then shown to produce exactly
        // one key, so it cannot reach that state at all.
        let mut two = AprV2Metadata {
            model_type: MODEL_TYPE_TAG.to_string(),
            ..Default::default()
        };
        two.custom.insert("setfit".to_string(), Value::from("a"));
        two.custom
            .insert("setfit_extra".to_string(), Value::from("b"));
        let rendered = String::from_utf8(two.to_json().expect("serializes")).expect("utf8");
        // Both keys are present; which comes FIRST is not specified anywhere.
        assert!(rendered.contains("setfit_extra"));
        assert_eq!(two.custom.len(), 2);

        let bytes = write_setfit_apr(&fixture_view_full_pin_shape()).expect("writable");
        let reader = AprV2Reader::from_bytes(&bytes).expect("parseable");
        assert_eq!(reader.metadata().custom.len(), 1);
    }

    #[test]
    fn cross_process_writes_produce_the_same_artifact_sha256() {
        let parent = artifact_sha256_hex(
            &write_setfit_apr(&fixture_view_full_pin_shape()).expect("parent write"),
        );
        let exe = std::env::current_exe().expect("the test binary knows its own path");
        let output = std::process::Command::new(exe)
            .arg("setfit::artifact::determinism::child_writes_the_fixture_and_prints_its_sha256")
            .arg("--exact")
            .arg("--nocapture")
            .arg("--test-threads=1")
            .env(CHILD_ENV, "1")
            .output()
            .expect("spawn the child test binary");

        // CLAUDE.md Verification rule 1: the status is read DIRECTLY off
        // `Output.status`. Reading it through a pipe would report the LAST
        // command's status and the assertion would be unreachable.
        assert!(
            output.status.success(),
            "child exited {:?}\nstdout: {}\nstderr: {}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8_lossy(&output.stdout);
        // libtest with `--nocapture` prints the test's stdout on the SAME line as
        // its `test <name> ... ` prefix, so the marker is searched for ANYWHERE
        // in the line rather than at its start. A `strip_prefix` here found
        // nothing and the test failed loudly — which is the correct behaviour for
        // a missing marker and is why the `expect` is not an `unwrap_or_default`.
        let child = stdout
            .lines()
            .find_map(|line| {
                line.find(CHILD_MARKER)
                    .map(|at| &line[at + CHILD_MARKER.len()..])
            })
            .map(str::trim)
            .expect("the child must print its marker line; a missing marker FAILS, never passes");
        assert_eq!(child.len(), 64, "the child printed {child:?}, not a sha256");
        assert_eq!(
            child, parent,
            "same view, two processes, two different HashMap RandomStates"
        );
    }

    #[test]
    fn child_writes_the_fixture_and_prints_its_sha256() {
        // A no-op unless the parent set the env var, so a plain `cargo test` run
        // does not spawn anything and this test cannot recurse.
        if std::env::var(CHILD_ENV).is_err() {
            return;
        }
        let bytes = write_setfit_apr(&fixture_view_full_pin_shape()).expect("child write");
        println!("{CHILD_MARKER}{}", artifact_sha256_hex(&bytes));
    }
}

#[cfg(all(test, feature = "setfit"))]
mod tamper {
    //! Take REAL writer-produced bytes apart and re-emit them through the SAME
    //! container writer with exactly ONE thing changed.
    //!
    //! Every negative in `mod ladder` and `mod probe` is an INDUCED CORRUPTION of
    //! a real artifact, never a hand-built fake: a fake can be wrong in ways the
    //! writer would never produce, so a loader that refused it would have proven
    //! nothing about the artifacts it will actually meet.
    //!
    //! The harness is only evidence if a round trip through it is the IDENTITY —
    //! otherwise a "tampered" artifact differs from the honest one in ways nobody
    //! chose, and every refusal below could be about the harness. That is asserted
    //! by `the_tamper_harness_re_emits_untouched_bytes_byte_identically`, which is
    //! the first test in `mod ladder` for exactly that reason.

    use super::*;

    use crate::format::v2::AprV2Reader;

    pub(super) struct Tampered {
        /// The container's typed metadata, with the custom document REMOVED (it
        /// lives in `doc` so a test can mutate it as a `serde_json::Map`).
        pub(super) metadata: AprV2Metadata,
        /// The one custom document.
        pub(super) doc: JsonMap<String, Value>,
        /// Every index entry as `(name, dtype, shape, payload)`.
        pub(super) tensors: Vec<(String, TensorDType, Vec<usize>, Vec<u8>)>,
    }

    impl Tampered {
        pub(super) fn of(bytes: &[u8]) -> Self {
            let reader = AprV2Reader::from_bytes(bytes).expect("real writer bytes parse");
            let mut metadata = reader.metadata().clone();
            let doc = metadata
                .custom
                .remove(CUSTOM_METADATA_KEY)
                .expect("the one custom key is present")
                .as_object()
                .expect("the custom key holds a JSON object")
                .clone();
            let tensors = reader
                .tensor_index()
                .iter()
                .map(|entry| {
                    let payload = reader
                        .get_tensor_data(&entry.name)
                        .expect("every index entry has a payload")
                        .to_vec();
                    (
                        entry.name.clone(),
                        entry.dtype,
                        entry.shape.clone(),
                        payload,
                    )
                })
                .collect();
            Self {
                metadata,
                doc,
                tensors,
            }
        }

        pub(super) fn emit(&self) -> Vec<u8> {
            let mut metadata = self.metadata.clone();
            metadata.custom.insert(
                CUSTOM_METADATA_KEY.to_string(),
                Value::Object(self.doc.clone()),
            );
            let mut writer = AprV2Writer::new(metadata);
            for (name, dtype, shape, payload) in &self.tensors {
                writer.add_tensor(name.clone(), *dtype, shape.clone(), payload.clone());
            }
            writer.write().expect("the harness re-emits a container")
        }

        /// Borrow one entry's `(shape, payload)` by name.
        pub(super) fn entry_mut(&mut self, name: &str) -> (&mut Vec<usize>, &mut Vec<u8>) {
            let found = self
                .tensors
                .iter_mut()
                .find(|(entry, ..)| entry == name)
                .unwrap_or_else(|| panic!("the fixture artifact carries {name}"));
            (&mut found.2, &mut found.3)
        }

        pub(super) fn drop_tensor(&mut self, name: &str) {
            let before = self.tensors.len();
            self.tensors.retain(|(entry, ..)| entry != name);
            assert_eq!(
                self.tensors.len() + 1,
                before,
                "the fixture artifact must carry {name} for dropping it to mean anything"
            );
        }

        /// Mutate one probe record in place.
        pub(super) fn probe_mut(&mut self, index: usize) -> &mut JsonMap<String, Value> {
            self.doc
                .get_mut("probes")
                .and_then(Value::as_array_mut)
                .and_then(|probes| probes.get_mut(index))
                .and_then(Value::as_object_mut)
                .expect("the document carries the six probe records")
        }
    }

    /// The honest artifact, produced by the real writer from the DEFAULT fixture.
    pub(super) fn honest_bytes() -> Vec<u8> {
        write_setfit_apr(&super::fixture::fixture_view_full_pin_shape())
            .expect("the fixture view is writable")
    }
}

#[cfg(all(test, feature = "setfit"))]
mod ladder {
    //! Rungs 1-6: the bounded read, the cap, the container, the document, the
    //! structure and the non-finite scan — each shown ABLE TO FAIL by induced
    //! corruption of real writer-produced bytes.

    use super::tamper::{honest_bytes, Tampered};
    use super::*;

    use serde_json::json;
    use std::io::Read;

    /// THE RUNG NUMBERING IS THE CONTRACT'S, AND THIS IS WHAT KEEPS IT THAT WAY.
    ///
    /// `load_validation_ladder`'s postcondition is that a refusal "names the RUNG"
    /// — a promise about a number an operator will look up in the contract, which
    /// is only worth anything if the two agree. They did not: this module numbered
    /// its own seven functions 1-7 by folding the bounded read out of the count,
    /// so every `rung N` it printed pointed one step up the contract's eight-rung
    /// ladder. `rung 4` on a corrupt tensor index resolved to "typed tag /
    /// document parse". Nothing was red, because nothing compared them.
    ///
    /// So this compares them. `include_str!` and not a runtime read, for the
    /// reason `thresholds.rs` gives: a test that silently skips when a file is
    /// absent is a test that reports success for having done nothing.
    ///
    /// # It scans the PRODUCTION half, and it learned that the hard way
    ///
    /// The first version scanned the whole file and failed on its own text: the
    /// `absent` list below spells `fn rung1_` as a literal, so `SRC.contains` saw
    /// the needle in the haystack of the guard itself. That is orchestrator note
    /// F-05's defect — a source assertion scanning its own needle — and cutting at
    /// the first test banner removes it structurally rather than by being careful
    /// about wording. Every symbol asserted below lives above that cut.
    #[test]
    fn the_rung_numbering_matches_the_contracts_eight_rung_ladder() {
        const CONTRACT: &str = include_str!("../../../../contracts/setfit-apr-v1.yaml");
        const WHOLE_FILE: &str = include_str!("artifact.rs");

        // The banner above `mod fixture`, which is the first test item in the
        // file. `find` returns THAT occurrence and not this one, because it comes
        // first — `production_source_is_cut_above_the_test_modules` pins it.
        let cut = WHOLE_FILE
            .find("// Test fixtures")
            .expect("the production half ends at the test banner");
        let src = &WHOLE_FILE[..cut];

        // The contract's `rungs:` block, read as the list of numbers it declares.
        let block = CONTRACT
            .split_once("\n    rungs:\n")
            .expect("load_validation_ladder declares a rungs block")
            .1;
        let declared: Vec<u32> = block
            .lines()
            .map_while(|line| line.strip_prefix("      "))
            .filter_map(|entry| entry.split_once(':'))
            .filter_map(|(key, _)| key.trim().parse::<u32>().ok())
            .collect();
        assert_eq!(
            declared,
            (1..=8).collect::<Vec<u32>>(),
            "the contract's ladder is eight consecutively numbered rungs; if that changed, this \
             module's function names and its error strings both have to move with it"
        );

        // Rung 1 bounds a SOURCE, so it has no `rungN_` function: it is the door
        // that acquires the bytes. Asserted by name so "there is no rung 1 here"
        // stays a deliberate statement rather than an omission.
        assert!(
            src.contains("pub fn read_setfit_apr_bytes_bounded<R: std::io::Read>("),
            "rung 1 is the bounded read at the acquisition boundary"
        );

        // Rungs 2-8 are functions, each named for the rung it IS.
        for (rung, suffix) in [
            (2, "raw_length"),
            (3, "container"),
            (4, "document"),
            (5, "structure"),
            (6, "finite_payloads"),
            (7, "rebuild"),
            (8, "replay_probes"),
        ] {
            let expected = format!("fn rung{rung}_{suffix}");
            assert!(
                src.contains(&expected),
                "rung {rung} of the contract's ladder must be implemented by `{expected}`; a \
                 function numbered differently from the rung it implements is how the printed \
                 diagnosis and the published ladder drifted apart in the first place"
            );
        }

        // And no function may carry a number the ladder does not have. `rung1_` is
        // the specific mistake this guard was written after: folding the bounded
        // read into the count is what shifted everything below it by one.
        for absent in ["fn rung0_", "fn rung1_", "fn rung9_"] {
            assert!(
                !src.contains(absent),
                "`{absent}` names a rung the contract's ladder does not declare"
            );
        }
    }

    /// The cut above is REAL: the production slice excludes this module.
    ///
    /// Without this, the three absence assertions above could pass for the worst
    /// possible reason — a cut that landed at the top of the file would make
    /// `src` empty and every `!contains` trivially true, while the `contains`
    /// assertions above would fail loudly enough that nobody would ever look. It
    /// pins BOTH directions: a production symbol is inside the slice and a test
    /// symbol is outside it.
    #[test]
    fn production_source_is_cut_above_the_test_modules() {
        const WHOLE_FILE: &str = include_str!("artifact.rs");
        let cut = WHOLE_FILE
            .find("// Test fixtures")
            .expect("the production half ends at the test banner");
        let src = &WHOLE_FILE[..cut];

        assert!(
            src.contains("pub fn write_setfit_apr("),
            "the production half must still contain the code being scanned"
        );
        assert!(
            !src.contains("fn the_rung_numbering_matches"),
            "the cut must remove this module, or every assertion above scans its own text"
        );
    }

    /// The largest byte count any test in this module is allowed to materialize.
    ///
    /// The cap boundary is exercised through the injected limit instead, so the
    /// suite never pays 256 MiB to learn that a comparison compares.
    const TEST_ALLOCATION_CEILING: usize = 1_048_576;

    /// A reader that PANICS the moment it is read from.
    ///
    /// It is the whole proof of the declared-length ordering: if
    /// `read_setfit_apr_bytes_bounded` consulted the stream before the declared
    /// length, this test would abort instead of returning a typed refusal.
    struct PanicOnRead;

    impl Read for PanicOnRead {
        fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
            panic!(
                "read_setfit_apr_bytes_bounded touched the reader; the declared-length refusal \
                 must run BEFORE the first read (review B5)"
            )
        }
    }

    /// A reader with far more bytes than the cap, which COUNTS what it handed over.
    ///
    /// Bounded rather than endless on purpose: an endless reader would HANG when
    /// the `take` is missing, and a hanging test is a worse signal than a failing
    /// one. The discriminating assertion is on `handed`, not on the refusal —
    /// without the `take` the refusal still fires, but only after the whole flood
    /// is resident.
    struct Flood<'a> {
        remaining: u64,
        handed: &'a mut u64,
    }

    impl Read for Flood<'_> {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let take = buf
                .len()
                .min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
            for byte in buf.iter_mut().take(take) {
                *byte = 0xAB;
            }
            self.remaining -= take as u64;
            *self.handed += take as u64;
            Ok(take)
        }
    }

    // -----------------------------------------------------------------------
    // The harness itself
    // -----------------------------------------------------------------------

    #[test]
    fn the_tamper_harness_re_emits_untouched_bytes_byte_identically() {
        let bytes = honest_bytes();
        assert!(
            bytes.len() < TEST_ALLOCATION_CEILING,
            "the fixture artifact is {} bytes; this suite must never materialize more than {}",
            bytes.len(),
            TEST_ALLOCATION_CEILING
        );
        assert_eq!(
            Tampered::of(&bytes).emit(),
            bytes,
            "a round trip through the tamper harness must be the IDENTITY, or every negative \
             below could be about the harness rather than about the corruption"
        );
    }

    #[test]
    fn the_honest_artifact_passes_every_rung_through_both_doors() {
        let bytes = honest_bytes();
        let parts = read_setfit_apr_parts(&bytes).expect("the honest artifact parses");
        assert_eq!(parts.doc.schema, ARTIFACT_SCHEMA);
        load_setfit_apr(&bytes).expect("the honest artifact loads");
    }

    // -----------------------------------------------------------------------
    // The bounded read (contract `bounded_read`, review B5)
    // -----------------------------------------------------------------------

    #[test]
    fn the_public_cap_constant_is_the_contracts_268435456() {
        assert_eq!(
            MAX_ARTIFACT_BYTES, 268_435_456,
            "contracts/setfit-apr-v1.yaml artifact_size_bounds.max_artifact_bytes"
        );
        assert_eq!(
            ArtifactLimits::CONTRACTED.max_artifact_bytes,
            MAX_ARTIFACT_BYTES
        );
    }

    #[test]
    fn a_declared_length_over_the_cap_is_refused_before_the_reader_is_touched() {
        let err = read_setfit_apr_bytes_bounded(PanicOnRead, Some(MAX_ARTIFACT_BYTES + 1))
            .expect_err("an over-cap declared length must be refused");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ArtifactTooLarge {
                    what: "declared_length",
                    observed,
                    cap
                } if *observed == MAX_ARTIFACT_BYTES + 1 && *cap == MAX_ARTIFACT_BYTES
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_lying_declared_length_cannot_make_the_reader_hand_over_more_than_cap_plus_one() {
        let limits = ArtifactLimits::tiny(64);
        let mut handed = 0_u64;
        let flood = Flood {
            remaining: 4096,
            handed: &mut handed,
        };
        // The declared length claims the stream is tiny. It is not.
        let err = read_setfit_apr_bytes_bounded_within(flood, Some(8), &limits)
            .expect_err("a stream over the cap must be refused however it was described");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ArtifactTooLarge {
                    what: "stream",
                    cap: 64,
                    ..
                }
            ),
            "got {err:?}"
        );
        assert!(
            handed <= 65,
            "the reader handed over {handed} bytes; the bound is cap + 1 = 65 and the `+ 1` is \
             what makes 'exactly the cap' distinguishable from 'truncated at the cap'"
        );
    }

    #[test]
    fn an_absent_declared_length_is_not_permission_to_read_unboundedly() {
        let limits = ArtifactLimits::tiny(64);
        let mut handed = 0_u64;
        let flood = Flood {
            remaining: 4096,
            handed: &mut handed,
        };
        let err = read_setfit_apr_bytes_bounded_within(flood, None, &limits)
            .expect_err("no declared length is treated as an over-cap CLAIM, not as permission");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ArtifactTooLarge { what: "stream", .. }
            ),
            "got {err:?}"
        );
        assert!(handed <= 65, "handed {handed}");
    }

    #[test]
    fn a_stream_of_exactly_the_cap_is_accepted_and_returned_whole() {
        let limits = ArtifactLimits::tiny(64);
        let source = vec![0x5A_u8; 64];
        let read = read_setfit_apr_bytes_bounded_within(source.as_slice(), Some(64), &limits)
            .expect("a stream of exactly the cap is legal");
        assert_eq!(read, source);
    }

    // -----------------------------------------------------------------------
    // Rung 2: the raw length, before any parse
    // -----------------------------------------------------------------------

    #[test]
    fn an_artifact_at_the_limit_loads_and_one_byte_of_limit_less_is_refused_before_any_parse() {
        let bytes = honest_bytes();
        let exact = u64::try_from(bytes.len()).expect("a fixture artifact fits in u64");
        load_setfit_apr_within(&bytes, &ArtifactLimits::tiny(exact))
            .expect("an artifact of exactly the limit passes rung 2");

        let err = load_setfit_apr_within(&bytes, &ArtifactLimits::tiny(exact - 1))
            .expect_err("one byte over the limit is refused");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ArtifactTooLarge {
                    what: "input_bytes",
                    observed,
                    cap
                } if *observed == exact && *cap == exact - 1
            ),
            "got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Rung 2: the container
    // -----------------------------------------------------------------------

    #[test]
    fn a_flipped_header_byte_is_a_typed_header_checksum_refusal() {
        let mut bytes = honest_bytes();
        // Byte 44 is inside the header's `reserved` region: covered by the header
        // CRC (which spans 0..40 and 44..64) and interpreted by nothing else, so
        // this isolates the checksum from every other header rule.
        bytes[44] ^= 0xFF;
        // The footer is recomputed, so the ONLY thing wrong with these bytes is
        // the header checksum.
        reseal_footer(&mut bytes);
        let err = load_setfit_apr(&bytes).expect_err("a corrupt header must not be believed");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ContainerIntegrity {
                    what: "header_checksum",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_flipped_payload_byte_without_a_reseal_is_a_typed_footer_checksum_refusal() {
        let mut bytes = honest_bytes();
        let last_payload = bytes.len() - 8;
        bytes[last_payload] ^= 0xFF;
        let err = load_setfit_apr(&bytes).expect_err("the footer CRC covers the whole content");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ContainerIntegrity {
                    what: "footer_checksum",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_truncated_artifact_is_a_typed_container_integrity_refusal() {
        let bytes = honest_bytes();
        for keep in [0_usize, 3, 63, bytes.len() - 1] {
            let err = load_setfit_apr(&bytes[..keep])
                .expect_err("a truncated artifact must not be partially interpreted");
            assert!(
                matches!(&err, SetFitArtifactError::ContainerIntegrity { .. }),
                "keeping {keep} bytes gave {err:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Rung 3: the typed tag and the one document
    // -----------------------------------------------------------------------

    #[test]
    fn a_setfit_shaped_tensor_set_without_the_typed_tag_is_refused() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered.metadata.model_type = "bert".to_string();
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("D-04 detection is explicit-tag-only, never tensor-name sniffing");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::NotASetFitArtifact { model_type } if model_type == "bert"
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn an_unknown_document_field_is_a_typed_parse_refusal() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .doc
            .insert("shadow_field".to_string(), json!("smuggled"));
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("deny_unknown_fields: an unknown key is a refusal, not a skipped field");
        assert!(
            matches!(&err, SetFitArtifactError::ArtifactDocumentParse { detail }
                if detail.contains("shadow_field")),
            "got {err:?}"
        );
    }

    #[test]
    fn schema_version_two_is_a_typed_unsupported_schema_version() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered.doc.insert("schema_version".to_string(), json!(2));
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("a future schema must be refused, never partially interpreted");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::UnsupportedSchemaVersion {
                    got: 2,
                    supported: 1
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_foreign_schema_identifier_is_a_typed_unsupported_schema() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .doc
            .insert("schema".to_string(), json!("setfit-apr-v9"));
        let err = load_setfit_apr(&tampered.emit()).expect_err("this build owns one schema id");
        assert!(
            matches!(&err, SetFitArtifactError::UnsupportedSchema { got, .. }
                if got == "setfit-apr-v9"),
            "got {err:?}"
        );
    }

    #[test]
    fn the_schema_check_runs_before_the_documents_other_fields_are_read() {
        // BOTH a foreign schema AND an unknown field. The schema refusal must
        // win, because the contract requires `schema`/`schema_version` to be
        // checked before any other field is read.
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .doc
            .insert("schema".to_string(), json!("setfit-apr-v9"));
        tampered.doc.insert("shadow_field".to_string(), json!(1));
        let err = load_setfit_apr(&tampered.emit()).expect_err("refused");
        assert!(
            matches!(&err, SetFitArtifactError::UnsupportedSchema { .. }),
            "the schema rung must speak first; got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Rung 4: the architecture-derived tensor set and the per-entry rules
    // -----------------------------------------------------------------------

    #[test]
    fn a_missing_encoder_tensor_is_a_typed_incomplete_tensor_set() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered.drop_tensor("blk.1.ffn_norm.bias");
        let err = load_setfit_apr(&tampered.emit()).expect_err("an incomplete encoder is refused");
        assert!(
            matches!(&err, SetFitArtifactError::IncompleteTensorSet { missing }
                if missing == &vec!["blk.1.ffn_norm.bias".to_string()]),
            "got {err:?}"
        );
    }

    #[test]
    fn a_headless_artifact_is_a_typed_incomplete_tensor_set_naming_the_head_tensor() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered.drop_tensor(HEAD_WEIGHT_TENSOR);
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("review B1: a headless artifact cannot load");
        assert!(
            matches!(&err, SetFitArtifactError::IncompleteTensorSet { missing }
                if missing == &vec![HEAD_WEIGHT_TENSOR.to_string()]),
            "got {err:?}"
        );
    }

    #[test]
    fn an_extra_tensor_is_a_typed_inconsistent_tensor_set() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered.tensors.push((
            "setfit.head.shadow".to_string(),
            TensorDType::F32,
            vec![2],
            vec![0_u8; 8],
        ));
        let err =
            load_setfit_apr(&tampered.emit()).expect_err("the expected set is compared EXACTLY");
        assert!(
            matches!(&err, SetFitArtifactError::InconsistentTensorSet { reason }
                if reason.contains("setfit.head.shadow")),
            "got {err:?}"
        );
    }

    #[test]
    fn the_expected_set_is_derived_from_the_documents_own_num_layers() {
        // The fixture is two layers. Claim three, and the SAME rule must now
        // demand sixteen layer-2 tensors the artifact does not carry — which is
        // only possible if the expected set is a FUNCTION of the parsed doc.
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .doc
            .get_mut("architecture")
            .and_then(Value::as_object_mut)
            .expect("the architecture sub-document")
            .insert("num_layers".to_string(), json!(3));
        let err = load_setfit_apr(&tampered.emit()).expect_err("refused");
        let SetFitArtifactError::IncompleteTensorSet { missing } = &err else {
            panic!("got {err:?}")
        };
        assert_eq!(missing.len(), 16, "one per layer template: {missing:?}");
        assert!(missing.iter().all(|name| name.starts_with("blk.2.")));
    }

    #[test]
    fn a_document_claiming_an_absurd_encoder_depth_is_refused_before_the_expansion() {
        // The expected-set builder is a FUNCTION of num_layers, so a document
        // claiming 2^40 layers would make the EXPANSION the allocation attack.
        // The bound must therefore run BEFORE the expansion, not after it.
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .doc
            .get_mut("architecture")
            .and_then(Value::as_object_mut)
            .expect("the architecture sub-document")
            .insert("num_layers".to_string(), json!(1_099_511_627_776_u64));
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("an absurd declared depth is refused, not expanded");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ArtifactTooLarge {
                    what: "declared_layers",
                    observed: 1_099_511_627_776,
                    cap
                } if *cap == MAX_ENCODER_LAYERS as u64
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_declared_size_that_disagrees_with_the_declared_shape_is_a_typed_inconsistent_tensor() {
        let mut tampered = Tampered::of(&honest_bytes());
        {
            let (shape, _) = tampered.entry_mut("token_embd_norm.bias");
            shape[0] += 1;
        }
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("size == product(shape) * dtype_width is structural");
        assert!(
            matches!(&err, SetFitArtifactError::InconsistentTensor { tensor, .. }
                if tensor == "token_embd_norm.bias"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_head_weight_shape_that_disagrees_with_the_label_set_is_a_typed_inconsistent_tensor() {
        // [3, 8] -> [8, 3]: the PRODUCT is unchanged, so the per-entry size rule
        // still holds and only the head-shape rule can catch this.
        let mut tampered = Tampered::of(&honest_bytes());
        {
            let (shape, _) = tampered.entry_mut(HEAD_WEIGHT_TENSOR);
            shape.reverse();
        }
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("row i of the head must belong to ordered_labels[i]");
        assert!(
            matches!(&err, SetFitArtifactError::InconsistentTensor { tensor, .. }
                if tensor == HEAD_WEIGHT_TENSOR),
            "got {err:?}"
        );
    }

    #[test]
    fn one_flipped_tokenizer_blob_byte_is_a_typed_tokenizer_hash_mismatch() {
        let mut tampered = Tampered::of(&honest_bytes());
        {
            let (_, payload) = tampered.entry_mut(TOKENIZER_BLOB_TENSOR);
            payload[7] ^= 0x01;
        }
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("the tokenizer is paired BEFORE a tensor is installed");
        assert!(
            matches!(&err, SetFitArtifactError::TokenizerHashMismatch { .. }),
            "got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Rung 5: the non-finite scan
    // -----------------------------------------------------------------------

    #[test]
    fn a_nan_bit_pattern_in_an_f32_payload_is_a_typed_non_finite_value() {
        let mut tampered = Tampered::of(&honest_bytes());
        {
            let (_, payload) = tampered.entry_mut("blk.0.attn_q.weight");
            // 0x7FC00000, little-endian: the quiet-NaN bit pattern.
            payload[0..4].copy_from_slice(&[0x00, 0x00, 0xC0, 0x7F]);
        }
        let err =
            load_setfit_apr(&tampered.emit()).expect_err("a NaN weight must never reach a rebuild");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "blk.0.attn_q.weight[0]"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_non_finite_head_coefficient_is_a_typed_non_finite_value() {
        let mut tampered = Tampered::of(&honest_bytes());
        {
            let (_, payload) = tampered.entry_mut(HEAD_BIAS_TENSOR);
            // 0x7F800000, little-endian: +Inf.
            payload[4..8].copy_from_slice(&[0x00, 0x00, 0x80, 0x7F]);
        }
        let err = load_setfit_apr(&tampered.emit()).expect_err("+Inf in the head is refused");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "setfit.head.bias[1]"),
            "got {err:?}"
        );
    }

    #[test]
    fn a_non_finite_probe_expectation_is_a_typed_non_finite_value() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .probe_mut(0)
            .get_mut("embedding_hex")
            .and_then(Value::as_array_mut)
            .expect("the probe records an embedding")[0] = json!("0000c07f");
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("a non-finite EXPECTATION would make every replay unfalsifiable");
        assert!(
            matches!(&err, SetFitArtifactError::NonFiniteValue { path }
                if path == "probes.0.embedding_hex[0]"),
            "got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // The parse-only door
    // -----------------------------------------------------------------------

    #[test]
    fn read_setfit_apr_parts_recovers_the_doc_the_tensors_the_head_and_the_tokenizer() {
        let view = super::fixture::fixture_view_full_pin_shape();
        let bytes = honest_bytes();
        let parts = read_setfit_apr_parts(&bytes).expect("the honest artifact parses");

        assert_eq!(parts.doc.schema, ARTIFACT_SCHEMA);
        assert_eq!(parts.doc.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(parts.doc.architecture, view.architecture);
        // HF-keyed, bit-exact, and the WHOLE set: the map came back through the
        // artifact's OWN hf_name_map inversion, not through a re-derivation.
        assert_eq!(parts.tensors, view.tensors);
        assert_eq!(parts.head_weights, view.head_weights);
        assert_eq!(parts.head_intercepts, view.head_intercepts);
        assert_eq!(parts.tokenizer_bytes, view.tokenizer_bytes);
        assert_eq!(parts.artifact_sha256, artifact_sha256_hex(&bytes));
    }

    #[test]
    fn both_doors_report_the_same_typed_variant_for_every_rung_two_to_six_corruption() {
        let honest = honest_bytes();
        let mut untagged = Tampered::of(&honest);
        untagged.metadata.model_type = "bert".to_string();
        let mut unknown_field = Tampered::of(&honest);
        unknown_field.doc.insert("shadow".to_string(), json!(1));
        let mut headless = Tampered::of(&honest);
        headless.drop_tensor(HEAD_BIAS_TENSOR);
        let mut nan = Tampered::of(&honest);
        {
            let (_, payload) = nan.entry_mut("blk.0.ffn_up.bias");
            payload[0..4].copy_from_slice(&[0x00, 0x00, 0xC0, 0x7F]);
        }

        let mut truncated = honest.clone();
        truncated.truncate(honest.len() - 1);

        for (what, bytes) in [
            ("truncated", truncated),
            ("untagged", untagged.emit()),
            ("unknown_field", unknown_field.emit()),
            ("headless", headless.emit()),
            ("nan_payload", nan.emit()),
        ] {
            let from_parts = read_setfit_apr_parts(&bytes)
                .err()
                .unwrap_or_else(|| panic!("{what} must be refused by the parse-only door"));
            let from_load = load_setfit_apr(&bytes)
                .err()
                .unwrap_or_else(|| panic!("{what} must be refused by the production door"));
            assert_eq!(
                from_parts, from_load,
                "{what}: ONE ladder, two doors — a divergence here is a second policy"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Helpers and unit-level rules
    // -----------------------------------------------------------------------

    #[test]
    fn the_document_type_carries_exactly_the_contracts_sixteen_fields() {
        let bytes = honest_bytes();
        let parts = read_setfit_apr_parts(&bytes).expect("parses");
        let value = serde_json::to_value(&parts.doc).expect("the doc re-serializes");
        let observed: BTreeSet<&str> = value
            .as_object()
            .expect("an object")
            .keys()
            .map(String::as_str)
            .collect();
        let expected: BTreeSet<&str> = SETFIT_ARTIFACT_DOC_FIELDS.iter().copied().collect();
        assert_eq!(
            observed, expected,
            "the parse struct and the contract's normative field list must not drift"
        );
    }

    #[test]
    fn the_hex_helpers_round_trip_every_stored_float_exactly() {
        for value in [
            0.0_f32,
            -0.0,
            1.0,
            -1.0,
            f32::MIN_POSITIVE,
            f32::MAX,
            L2_EPS_PROBE,
            0.333_333_34,
        ] {
            let hex = f32_bits_hex(value);
            assert_eq!(hex.len(), 8, "{value} rendered as {hex}");
            assert_eq!(
                hex_to_f32(&hex).map(f32::to_bits),
                Some(value.to_bits()),
                "hex is the identity on f32 and decimal text is not"
            );
        }
        // Strictness: the reader accepts nothing the writer cannot produce.
        assert_eq!(hex_to_f32(""), None);
        assert_eq!(hex_to_f32("0000803"), None);
        assert_eq!(hex_to_f32("0000803FF"), None);
        assert_eq!(hex_to_f32("0000803F".to_uppercase().as_str()), None);
        assert_eq!(hex_to_f32("zzzzzzzz"), None);
    }

    /// A stand-in for a small positive constant, kept local so this test does not
    /// depend on a `setfit` re-export that a later plan might move.
    const L2_EPS_PROBE: f32 = 1e-12;

    /// Recompute the container's trailing CRC32 over the (possibly tampered) content.
    ///
    /// A LEGITIMATE re-signing, exactly like the writer's own footer step: it is
    /// what lets a rung-3 header test and a rung-5/6 content test be about
    /// different rungs instead of both landing on the footer.
    fn reseal_footer(bytes: &mut [u8]) {
        let split = bytes.len() - 4;
        let checksum = crate::format::crc32(&bytes[..split]);
        bytes[split..].copy_from_slice(&checksum.to_le_bytes());
    }
}

#[cfg(all(test, feature = "setfit"))]
mod probe {
    //! Rungs 7-8 and the typestate: the rebuild, the six-probe replay, and the
    //! only value a consumer may classify with.

    use super::tamper::{honest_bytes, Tampered};
    use super::*;

    use serde_json::json;

    #[test]
    fn a_valid_artifact_yields_a_verified_model_carrying_the_artifacts_own_hash() {
        let bytes = honest_bytes();
        let model = load_setfit_apr(&bytes).expect("the honest artifact replays all six probes");
        assert_eq!(model.artifact_sha256(), artifact_sha256_hex(&bytes));
        assert_eq!(model.artifact_sha256().len(), 64);
    }

    #[test]
    fn ordered_labels_are_read_off_the_rebuilt_head() {
        let model = load_setfit_apr(&honest_bytes()).expect("loads");
        assert_eq!(model.ordered_labels(), model.doc_view().ordered_labels);
        assert_eq!(
            model.ordered_labels().len(),
            model.doc_view().head.num_labels
        );
    }

    #[test]
    fn a_transposed_encoder_tensor_is_a_typed_rebuild_failure() {
        // [vocab, hidden] -> [hidden, vocab]. The PRODUCT is unchanged, so the
        // per-entry size rule still holds, the name set is untouched and every
        // value is still finite — rungs 2-6 have nothing to say. Only the rebuild
        // knows what shape this name must have, which is what makes rung 7 a rung
        // rather than a formality.
        let mut tampered = Tampered::of(&honest_bytes());
        {
            let (shape, _) = tampered.entry_mut("token_embd.weight");
            shape.reverse();
        }
        let bytes = tampered.emit();
        // The parse-only door gets all the way through: this is genuinely a rung
        // BELOW it, not a rung it skipped.
        read_setfit_apr_parts(&bytes).expect("rungs 2-6 have no complaint about a transpose");
        let err = load_setfit_apr(&bytes).expect_err("the rebuild knows the required shape");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ArtifactRebuildFailed {
                    what: "encoder",
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_perturbed_probe_embedding_is_a_typed_replay_failure_naming_the_probe() {
        let mut tampered = Tampered::of(&honest_bytes());
        // Move ONE component far outside the contract's 7.63e-06 bound, and
        // re-sign the whole artifact, so this exercises rung 8 and not rung 3.
        let perturbed = f32_bits_hex(1.5);
        tampered
            .probe_mut(2)
            .get_mut("embedding_hex")
            .and_then(Value::as_array_mut)
            .expect("embedding_hex")[3] = json!(perturbed);
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("probe replay is the last word before a classify-capable value exists");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ProbeReplayFailed(divergence)
                    if divergence.probe == 2
                        && divergence.probe_id == PROBE_IDS[2]
                        && divergence.component == "embedding"
                        && divergence.index == 3
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_perturbed_probe_label_is_a_typed_replay_failure_compared_exactly() {
        let mut tampered = Tampered::of(&honest_bytes());
        let recorded = tampered.probe_mut(0)["label"]
            .as_str()
            .expect("the probe records a label")
            .to_string();
        let other = super::fixture::FIXTURE_LABELS
            .iter()
            .find(|label| **label != recorded)
            .expect("the fixture has three labels");
        tampered
            .probe_mut(0)
            .insert("label".to_string(), json!(other));
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("a label has no tolerance; a close-enough label is a wrong answer");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ProbeReplayFailed(divergence)
                    if divergence.probe == 0
                        && divergence.component == "label"
                        && divergence.tolerance == "exact"
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_probe_input_that_is_not_the_contracts_own_string_is_a_typed_replay_failure() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .probe_mut(4)
            .insert("input".to_string(), json!("a dataset sentence"));
        let err = load_setfit_apr(&tampered.emit())
            .expect_err("probe inputs are fixed, synthetic and contract-resident");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ProbeReplayFailed(divergence)
                    if divergence.probe == 4 && divergence.component == "input"
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn a_short_probe_array_is_a_typed_replay_failure_and_a_partial_replay_is_not_a_pass() {
        let mut tampered = Tampered::of(&honest_bytes());
        tampered
            .doc
            .get_mut("probes")
            .and_then(Value::as_array_mut)
            .expect("probes")
            .truncate(5);
        let err = load_setfit_apr(&tampered.emit()).expect_err("all six replay, or none passes");
        assert!(
            matches!(
                &err,
                SetFitArtifactError::ProbeReplayFailed(divergence)
                    if divergence.component == "probe_count"
                        && divergence.expected == PROBE_COUNT.to_string()
                        && divergence.observed == "5"
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn the_probe_comparator_is_nan_visible_in_both_argument_positions() {
        assert!(within(0.0, 1.0));
        assert!(within(1.0, 1.0), "the bound itself is INSIDE");
        assert!(!within(1.000_001, 1.0));
        assert!(
            !within(f64::NAN, 1.0),
            "a NaN delta means at least one side was non-finite — a divergence in every sense"
        );
        assert!(!within(1.0, f64::NAN), "and in the other argument position");
        assert!(!within(f64::NAN, f64::NAN));
        assert!(!within(f64::INFINITY, 1.0));
    }

    #[test]
    fn doc_view_recovers_every_apr_05_identity_field_from_the_artifact_alone() {
        let view = super::fixture::fixture_view_full_pin_shape();
        let model = load_setfit_apr(&honest_bytes()).expect("loads");
        let doc = model.doc_view();

        // Schema identity
        assert_eq!(doc.schema, ARTIFACT_SCHEMA);
        assert_eq!(doc.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(doc.bundle_schema_version, view.bundle_schema_version);
        assert_eq!(doc.format_id, view.format_id);
        // Revisions and hashes
        assert_eq!(
            doc.architecture.source_revision,
            view.architecture.source_revision
        );
        assert_eq!(
            doc.architecture.tokenizer_sha256,
            view.architecture.tokenizer_sha256
        );
        assert_eq!(doc.tokenizer_sha256, view.architecture.tokenizer_sha256);
        // Pooling / truncation policy
        assert_eq!(doc.preprocessing.pooling, view.pooling);
        assert_eq!(doc.preprocessing.normalization, view.normalization);
        assert_eq!(
            doc.preprocessing.l2_epsilon_hex,
            f32_bits_hex(view.l2_epsilon)
        );
        assert_eq!(
            doc.preprocessing.truncation_max_sequence_length,
            view.truncation_max_sequence_length
        );
        assert_eq!(doc.preprocessing.padding_mode, view.padding_mode);
        assert_eq!(doc.preprocessing.max_length, view.max_length);
        // Label order and head configuration
        assert_eq!(doc.ordered_labels, view.ordered_labels);
        assert_eq!(doc.head.n_features, view.head_n_features);
        assert_eq!(doc.head.num_labels, view.ordered_labels.len());
        // Seeds
        assert_eq!(doc.root_seed, view.root_seed);
        // Configuration and update evidence
        assert_eq!(doc.requested_config, view.requested_config);
        assert_eq!(doc.resolved_config, view.resolved_config);
        assert_eq!(doc.evidence, view.evidence);
        // Provenance, INCLUDING the data fingerprint APR-05 asks for by name
        assert_eq!(doc.provenance, view.provenance);
        assert_eq!(
            doc.provenance.get("dataset_fingerprint"),
            view.provenance.get("dataset_fingerprint")
        );
        assert!(doc.provenance.get("dataset_fingerprint").is_some());
        // The naming table and the probes travelled too
        assert_eq!(doc.hf_name_map.len(), view.tensors.len());
        assert_eq!(doc.probes.len(), PROBE_COUNT);
    }

    #[test]
    fn embed_returns_l2_normalized_rows_of_the_encoders_hidden_width() {
        let model = load_setfit_apr(&honest_bytes()).expect("loads");
        let width = model.doc_view().architecture.hidden;
        let texts = vec!["the quick brown fox".to_string(), "ok".to_string()];
        let rows = model
            .embed(&texts)
            .expect("embed is reachable through the public API");

        assert_eq!(rows.len(), texts.len());
        for row in &rows {
            assert_eq!(row.len(), width);
            assert!(row.iter().all(|value| value.is_finite()));
            let norm = f64::from(row.iter().map(|v| v * v).sum::<f32>()).sqrt();
            assert!(
                (norm - 1.0).abs() <= 1e-5,
                "pooled embeddings are L2-normalized; observed norm {norm}"
            );
        }
    }

    #[test]
    fn embed_on_an_empty_batch_is_a_typed_refusal_and_never_panics() {
        let model = load_setfit_apr(&honest_bytes()).expect("loads");
        let err = model
            .embed(&[])
            .expect_err("embedding nothing is a caller mistake, not an empty result");
        assert!(
            matches!(&err, SetFitArtifactError::EmptyEmbedBatch),
            "got {err:?}"
        );
    }

    /// APR-04's OTHER half: the typestate has no public constructor at all.
    ///
    /// `tests/ui/setfit_verified_model_constructed.rs` proves the struct literal
    /// is a compile error, and its own header says the ABSENT CONSTRUCTOR "is
    /// covered by a source assertion instead". Until this test existed it was not:
    /// the trybuild case deliberately carries ONE claim (two claims failing in
    /// different compiler passes cannot share a snapshot, since rustc aborts after
    /// the first), and the second claim was documented as mechanised without being
    /// mechanised anywhere. A compensating control named in a comment and present
    /// nowhere is the failure class this phase exists to catch, so here it is.
    ///
    /// It scans the `impl` block by slicing to the first column-0 `}` — the same
    /// shape `classify`'s guards use — so the assertions below cannot match their
    /// own text, which sits far below that block.
    #[test]
    fn verified_model_declares_no_public_constructor_and_no_minting_derive() {
        const SRC: &str = include_str!("artifact.rs");

        let marker = "impl VerifiedSetFitModel {";
        let start = SRC
            .find(marker)
            .expect("the typestate's inherent impl block");
        let body = &SRC[start + marker.len()..];
        let end = body
            .find("\n}")
            .expect("the impl block is closed at column 0");
        let block = &body[..end];

        // A `pub fn` that HANDS BACK the type is a minting path whatever it is
        // called, so the scan is on the return type and not on the name `new`.
        for forbidden in ["-> Self", "-> VerifiedSetFitModel"] {
            assert!(
                !block.contains(forbidden),
                "an inherent method of VerifiedSetFitModel returns `{forbidden}`; the ONLY value \
                 of this type must come from load_setfit_apr, which ran rungs 2-8 including \
                 the six-probe replay. A second minting path is a second verification policy."
            );
        }

        // The derive list, read off the declaration rather than trusted: `Default`
        // or `Deserialize` would each mint the type without a single rung running,
        // and neither needs a `pub fn` to do it.
        let decl = SRC
            .find("pub struct VerifiedSetFitModel {")
            .expect("the typestate declaration");
        let derive_line = SRC[..decl]
            .lines()
            .next_back()
            .filter(|line| line.contains("#[derive("))
            .expect("the declaration is preceded by its derive attribute");
        for forbidden in ["Default", "Deserialize", "Clone"] {
            assert!(
                !derive_line.contains(forbidden),
                "VerifiedSetFitModel derives `{forbidden}`; the derive list is `Debug` alone \
                 because Default and Deserialize each mint the witness with no rung run, and \
                 Clone would let one verified value become an unbounded supply of them."
            );
        }
    }
}
