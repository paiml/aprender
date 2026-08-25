//! The typed failure surface for the contrastive data protocol.
//!
//! # Contract: contrastive-pair-protocol-v1.yaml (`OBLIG-CPP-ERROR-TAXONOMY`)
//!
//! Every fallible path in this crate returns [`ContrastiveDataError`]. There is no
//! `unwrap()`, no `panic!` on caller input, and no sentinel return value: a caller that
//! feeds this crate untrusted bytes from object storage must be able to distinguish
//! "malformed row 12 of train" from "this dataset does not have enough examples left in
//! class 2 to supply 64 shots", and both from an internal arithmetic overflow.
//!
//! Messages follow the `data_tweeteval.rs` house style — they name the split, the index,
//! and BOTH the expected and the observed value, because a message that says only
//! "validation failed" turns a five-second fix into a bisect.
//!
//! # This enum is an exhaustive INITIAL design, not a permanently closed one
//!
//! Every variant below was derived up-front from DATA-02's failure classes plus the
//! degenerate-case, version-mismatch, arithmetic, and untrusted-input classes that plans
//! 02-05, 02-07 and 02-09 need, so that a downstream plan does not have to widen the
//! error surface mid-wave and force its siblings to rebase. That is a design review, not
//! a freeze.
//!
//! A downstream plan **MAY** add a variant. When it does, two things are mandatory:
//!
//! 1. the addition and its reason are recorded in that plan's `SUMMARY.md`, and
//! 2. `OBLIG-CPP-ERROR-TAXONOMY` in `contracts/contrastive-pair-protocol-v1.yaml` is
//!    extended to list it.
//!
//! The enum is `#[non_exhaustive]` so that adding a variant is not a breaking change for
//! an external consumer, and so that a `match` in `apr-cli` cannot silently go stale.

/// Every way the contrastive data protocol can refuse to proceed.
///
/// Grouped below by the boundary that raises them: split ingest, selection and manifest,
/// dataset attestation, pair construction, and version/arithmetic/plumbing.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ContrastiveDataError {
    // ---------------------------------------------------------------------------
    // DATA-02 — split ingest, the bytes -> typed boundary
    // ---------------------------------------------------------------------------
    /// A JSONL row did not parse, or parsed into something the schema rejects.
    #[error("{split} row {index} is malformed: {reason}")]
    MalformedRow {
        /// Split role name as declared by the caller (`train`, `validation`, ...).
        split: String,
        /// Zero-based row index within the split's buffer.
        index: usize,
        /// Why the row was rejected (parser message or schema violation).
        reason: String,
    },

    /// A row's bytes are not valid UTF-8, so they were never parsed.
    #[error("{split} row {index} is not valid UTF-8")]
    InvalidUtf8 {
        /// Split role name as declared by the caller.
        split: String,
        /// Zero-based row index within the split's buffer.
        index: usize,
    },

    /// A row's `input` field is empty or whitespace-only.
    #[error("{split} row {index} has empty text")]
    EmptyInput {
        /// Split role name as declared by the caller.
        split: String,
        /// Zero-based row index within the split's buffer.
        index: usize,
    },

    /// A row's numeric label is outside the declared label map.
    #[error("{split} row {index} has unknown label {label}")]
    UnknownLabel {
        /// Split role name as declared by the caller.
        split: String,
        /// Zero-based row index within the split's buffer.
        index: usize,
        /// The out-of-range label value read from the row.
        label: usize,
    },

    /// A row's `label_text` disagrees with `label_names[label]`.
    ///
    /// This is a distinct failure from [`Self::UnknownLabel`]: the numeric label is in
    /// range, but the human-readable text contradicts it, which is exactly what a
    /// tampered or hand-edited mirror looks like.
    #[error(
        "{split} row {index} label {label} text mismatch: expected {expected_text:?}, got {got_text:?}"
    )]
    LabelTextMismatch {
        /// Split role name as declared by the caller.
        split: String,
        /// Zero-based row index within the split's buffer.
        index: usize,
        /// The numeric label carried by the row.
        label: usize,
        /// The text the declared label map assigns to `label`.
        expected_text: String,
        /// The text the row actually carried.
        got_text: String,
    },

    /// The per-class row counts of a split do not match the declaration.
    #[error("{split} class-count contract failed: expected {expected:?}, got {got:?}")]
    InvalidClassCounts {
        /// Split role name as declared by the caller.
        split: String,
        /// Per-class counts the declaration requires.
        expected: Vec<usize>,
        /// Per-class counts actually observed.
        got: Vec<usize>,
    },

    /// The same row identifier appears twice inside one split.
    #[error("{split} contains duplicate id {id:?}")]
    DuplicateId {
        /// Split role name as declared by the caller.
        split: String,
        /// The identifier that appeared more than once.
        id: String,
    },

    /// A dataset profile declared by the caller conflicts with the one in the bytes.
    #[error("conflicting source role: caller declared {declared:?}, bytes embed {embedded:?}")]
    ConflictingSourceRole {
        /// Role the caller asserted when constructing the split.
        declared: String,
        /// Role embedded in the row payloads.
        embedded: String,
    },

    /// A row's embedded `source_split` is not the role being constructed.
    ///
    /// The typestate makes leakage inexpressible for a *library* caller; this variant is
    /// what stops honest-looking bytes from object storage becoming a `Split<Train>` the
    /// compiler is perfectly happy with (D-16).
    #[error("split role mismatch: expected {expected_role:?}, bytes embed {embedded_role:?}")]
    SplitRoleMismatch {
        /// Role being constructed.
        expected_role: String,
        /// Role found inside the row.
        embedded_role: String,
    },

    /// A row's recomputed content hash disagrees with the attested one.
    #[error("row {id:?} hash mismatch: expected {expected}, got {got}")]
    RowHashMismatch {
        /// The row identifier whose hash disagreed.
        id: String,
        /// Hex digest recorded in the attestation.
        expected: String,
        /// Hex digest recomputed from the supplied bytes.
        got: String,
    },

    /// Cross-split duplicate exclusion shrank a class pool below `shots_per_class`.
    ///
    /// Duplicate *content* is never fatal at prepare time (D-18, upheld verbatim by
    /// D-27) — it is excluded and recorded. This variant is the one real failure: after
    /// exclusion the pool can no longer supply the requested shots.
    #[error(
        "class {class_label} pool exhausted after cross-split duplicate exclusion: {pool} rows remain, {shots} shots requested"
    )]
    CrossSplitDuplicateUnderflow {
        /// The class whose pool ran short.
        class_label: usize,
        /// Rows remaining in the class pool after exclusion.
        pool: usize,
        /// Shots per class the selection requested.
        shots: usize,
    },

    // ---------------------------------------------------------------------------
    // Selection and manifest
    // ---------------------------------------------------------------------------
    /// `shots_per_class` is not one of the contracted values.
    ///
    /// Checked BEFORE any RNG draw, so an invalid request never consumes an ordinal and
    /// never produces a partially built selection.
    #[error("invalid shots_per_class {got}: allowed values are {allowed}")]
    InvalidShots {
        /// The requested shots-per-class value.
        got: usize,
        /// The contracted set, rendered for the message (e.g. `{8, 16, 32, 64}`).
        allowed: &'static str,
    },

    /// A selection manifest's recorded `semantic_hash` does not match its payload.
    #[error("selection semantic_hash mismatch: expected {expected}, got {got}")]
    SemanticHashMismatch {
        /// Digest recorded in the manifest envelope.
        expected: String,
        /// Digest recomputed from the canonical payload bytes.
        got: String,
    },

    /// Replaying a selection from its manifest did not reproduce the manifest.
    #[error("selection replay mismatch in field {field:?}")]
    SelectionReplayMismatch {
        /// The first field that disagreed (ordering, balance, membership, ...).
        field: String,
    },

    /// A pair endpoint names an identifier that is not in the selection.
    ///
    /// This is D-27's fail-closed span check for untrusted, replayed pair bytes.
    #[error("pair endpoint {id:?} is not in the selection (found in {found_in})")]
    EndpointNotInSelection {
        /// The offending identifier, named so the failure is diagnosable.
        id: String,
        /// Where the identifier WAS found, if anywhere (`validation`, `nowhere`, ...).
        found_in: String,
    },

    // ---------------------------------------------------------------------------
    // Dataset attestation (plan 02-06 boundary)
    // ---------------------------------------------------------------------------
    /// The attested dataset profile is not the one the consumer asked for.
    #[error("dataset profile mismatch: expected {expected:?}, got {got:?}")]
    ProfileMismatch {
        /// Profile the consumer requires.
        expected: String,
        /// Profile the attestation carries.
        got: String,
    },

    /// The attestation names a split role for which no bytes were supplied.
    #[error("attestation requires split {role:?} but no bytes were supplied for it")]
    MissingSplit {
        /// The split role that is absent.
        role: String,
    },

    /// A split's recomputed JSONL digest disagrees with the attested one.
    #[error("{split} split hash mismatch: expected {expected}, got {got}")]
    SplitHashMismatch {
        /// Split role name.
        split: String,
        /// Digest recorded in the attestation.
        expected: String,
        /// Digest recomputed from the supplied buffer.
        got: String,
    },

    /// The recomputed dataset fingerprint disagrees with the attested one.
    #[error("dataset fingerprint mismatch: expected {expected}, got {got}")]
    FingerprintMismatch {
        /// Fingerprint recorded in the attestation.
        expected: String,
        /// Fingerprint recomputed from the supplied buffers.
        got: String,
    },

    /// The recomputed cross-split exclusion record disagrees with the attested one.
    #[error("exclusion record mismatch: expected {expected}, got {got}")]
    ExclusionRecordMismatch {
        /// Exclusion record digest recorded in the attestation.
        expected: String,
        /// Exclusion record digest recomputed from the supplied buffers.
        got: String,
    },

    // ---------------------------------------------------------------------------
    // Pair construction
    // ---------------------------------------------------------------------------
    /// A pair was requested whose two endpoints are the same selected ordinal.
    ///
    /// D-12: unreachable through the sampler, because `CanonicalPair::new` is the sole
    /// constructor and it rejects equal endpoints. It IS reachable through the untrusted
    /// pair-ingest boundary, which is why the variant exists.
    #[error("self-pair rejected: both endpoints are selected ordinal {id}")]
    SelfPair {
        /// The selected-example ordinal that appeared on both sides.
        id: u64,
    },

    /// The layout admits no pairs of either kind.
    #[error(
        "no pair capacity: positive_capacity={positive_capacity}, negative_capacity={negative_capacity}"
    )]
    NoPairCapacity {
        /// Number of distinct same-class unordered pairs available.
        positive_capacity: u64,
        /// Number of distinct cross-class unordered pairs available.
        negative_capacity: u64,
    },

    /// An effective pair budget of zero was resolved or requested.
    #[error("pair budget must be greater than zero")]
    ZeroBudget,

    /// A pair hard cap of zero was configured.
    ///
    /// Distinct from [`Self::ZeroBudget`]: a zero cap means no budget can ever be
    /// satisfied, which is a configuration defect rather than a request defect.
    #[error("pair hard_cap must be greater than zero")]
    ZeroHardCap,

    /// An explicit budget above the configured hard cap.
    ///
    /// This FAILS rather than silently clamping: the cap exists for DoS control, and a
    /// user who typed a larger number deserves to be told it was refused, not to receive
    /// a quietly different dataset (`budget_resolution`).
    #[error("requested pair budget {budget} exceeds hard_cap {hard_cap}")]
    BudgetExceedsHardCap {
        /// Budget the caller requested.
        budget: u64,
        /// Configured hard cap.
        hard_cap: u64,
    },

    /// A budget exceeding the available UNIQUE pair capacity (D-11).
    ///
    /// Reserved for the unique-capacity check. The oversampling strategy draws with
    /// replacement, so `budget > capacity` is not an error there.
    #[error("requested pair budget {budget} exceeds unique pair capacity {capacity}")]
    BudgetExceedsCapacity {
        /// Budget the caller requested.
        budget: u64,
        /// Unique pair capacity available for the strategy.
        capacity: u64,
    },

    /// A pair was requested at an ordinal at or beyond the resolved budget.
    #[error("pair ordinal {ordinal} is out of range for budget {budget}")]
    OrdinalOutOfRange {
        /// The requested draw ordinal.
        ordinal: u64,
        /// The resolved effective budget.
        budget: u64,
    },

    /// A replayed pair record's target disagrees with its endpoints' classes.
    ///
    /// The 1.0/0.0 target is DERIVED from endpoint classes at emission and is never
    /// accepted from caller input; this variant is how that is enforced for bytes that
    /// claim otherwise.
    #[error(
        "pair ({lo:?}, {hi:?}) declares target {declared_target} but its endpoints derive {derived_target}"
    )]
    PairTargetMismatch {
        /// Lower canonical endpoint identifier.
        lo: String,
        /// Upper canonical endpoint identifier.
        hi: String,
        /// Target the untrusted record carried.
        declared_target: f32,
        /// Target derived from the endpoints' classes.
        derived_target: f32,
    },

    // ---------------------------------------------------------------------------
    // Version, arithmetic, plumbing
    // ---------------------------------------------------------------------------
    /// A serialized artifact declares a schema version this build does not support.
    #[error("unsupported schema version for {field}: got {got}, supported {supported}")]
    UnsupportedSchemaVersion {
        /// Which artifact or field carried the version (`selection`, `attestation`, ...).
        field: String,
        /// Version read from the artifact.
        got: u32,
        /// Version this build implements.
        supported: u32,
    },

    /// An artifact was produced under a content-normalization pipeline this build does
    /// not implement.
    ///
    /// Distinct from [`Self::UnsupportedSchemaVersion`] because the normalization version
    /// is a STRING tag rather than an integer, and because it changes what the exclusion
    /// record MEANS rather than what the artifact's fields are. Silently accepting a
    /// foreign tag would let an exclusion record computed under different collapsing rules
    /// be replayed as if it had been computed under these ones (D-17: the normalization is
    /// contracted and versioned so it cannot drift).
    #[error("unsupported content normalization version: got {got:?}, supported {supported:?}")]
    UnsupportedNormalizationVersion {
        /// Tag read from the artifact.
        got: String,
        /// Tag this build implements.
        supported: &'static str,
    },

    /// A versioned policy enum value this build does not implement.
    #[error("unsupported {policy} policy version: got {got}, supported {supported}")]
    UnsupportedPolicyVersion {
        /// Which policy (`singleton`, `degenerate`, ...).
        policy: String,
        /// Version read from the artifact.
        got: u32,
        /// Version this build implements.
        supported: u32,
    },

    /// A sampling-algorithm version this build does not implement.
    ///
    /// Separate from [`Self::UnsupportedPolicyVersion`] because changing the algorithm
    /// changes pair IDENTITIES, whereas changing a policy changes which pairs are legal.
    #[error("unsupported algorithm version: got {got}, supported {supported}")]
    UnsupportedAlgorithmVersion {
        /// Version read from the artifact.
        got: u32,
        /// Version this build implements.
        supported: u32,
    },

    /// A capacity or budget computation overflowed.
    ///
    /// Every step of the closed-form capacity math uses checked arithmetic, so a large
    /// class layout produces this typed error instead of a wrapped, plausible-looking
    /// capacity that would then silently under-sample.
    #[error("arithmetic overflow in {operation}")]
    ArithmeticOverflow {
        /// The operation that overflowed (`positive_capacity`, `default_budget`, ...).
        operation: String,
    },

    /// Canonical serialization or deserialization of an artifact failed.
    #[error("serialization failed in {context}: {detail}")]
    Serialization {
        /// What was being (de)serialized.
        context: String,
        /// The underlying serializer message.
        detail: String,
    },

    /// A caller-supplied sink or source failed.
    ///
    /// The crate performs NO filesystem or network access (D-04). This variant exists
    /// only so `dump_pairs<W: Write>` can surface the caller's own writer failure as a
    /// typed error rather than swallowing it.
    #[error("i/o failed in {context}: {detail}")]
    Io {
        /// What was being written or read.
        context: String,
        /// The underlying `std::io::Error` message.
        detail: String,
    },
}
