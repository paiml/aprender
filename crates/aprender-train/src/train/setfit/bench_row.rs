//! The benchmark row and the hashed run manifest (plan 05-05, EVAL-03 / EVAL-04).
//!
//! Contract: `setfit-benchmark-claims-v1` (authored in plan 05-05 task 1). Decisions: D-12
//! (method-tagged rows, never null-padded), D-14 (row-per-file plus a hashed manifest whose
//! pre-declared expectation set defines completeness), D-06 (no RNG anywhere in the claims
//! path).
//!
//! # Why this lives in the library and not in `apr-cli`
//!
//! The row schema, the digest discipline and the 40-cell ACTIVE expectation set ARE the claim.
//! (The two-method 80-cell scope is retained as deferred — see [`ExpectationScope`].) An
//! adapter is a filesystem shim over them, and a shim is the wrong place for a definition
//! that two commands, a report renderer and a GPU host all have to agree on.
//!
//! # The three properties that carry the weight
//!
//! 1. **Verify before return.** [`BenchRow::from_bytes`] and [`RunManifest::from_bytes`]
//!    recompute SHA-256 over the payload's canonical bytes and compare it to the envelope
//!    BEFORE handing back a value, so a caller cannot hold a row whose digest disagrees with
//!    its payload. The ORDER is the property: a digest verified after the value escapes
//!    protects nothing. This is [`SelectionManifest`]'s discipline
//!    (`aprender-contrastive-data/src/manifest.rs`), copied rather than re-invented.
//!
//! 2. **A missing evidence block is unrepresentable, not null-padded.** [`MethodEvidence`] is
//!    an externally-tagged enum whose variants carry no `Option` provenance fields. A LoRA row
//!    cannot present SetFit evidence, and neither can present its own block with the fields
//!    absent — which is the shape SAFE-03 exists to forbid and the reason "the block is
//!    missing" stays distinguishable from "the block is empty".
//!
//! 3. **Cells collide rather than merge.** [`RunManifest::record`] is idempotent on an
//!    identical digest (the resume path after a crash at cell 35) and a typed error on a
//!    differing one. Refusing the first would get manifests deleted and re-created, which
//!    erases the pre-declared expectation set that makes omission visible at all; allowing
//!    the second would let a re-run silently replace the run that produced a published
//!    number.
//!
//! # Canonical bytes are key-sorted, not declaration-ordered
//!
//! [`BenchRowPayload::to_canonical_bytes`] serializes through [`serde_json::Value`], whose
//! `Map` is `BTreeMap`-backed (no workspace crate enables `preserve_order`). The digest is
//! therefore independent of Rust field-declaration order, so adding a field in a different
//! position cannot silently change every historical digest, and an auditor can recompute the
//! digest from the JSON alone without owning this struct.
//!
//! [`SelectionManifest`]: aprender_contrastive_data::manifest::SelectionManifest

use core::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ===========================================================================================
// Contract-resident constants (Ph1 D-14: committed before the code that compares against them)
// ===========================================================================================

/// The claims contract this module implements.
pub const CLAIMS_CONTRACT_ID: &str = "setfit-benchmark-claims-v1";

/// The row schema version this build writes and reads.
pub const BENCH_ROW_SCHEMA_VERSION: u32 = 1;

/// The manifest schema version this build writes and reads.
pub const RUN_MANIFEST_SCHEMA_VERSION: u32 = 1;

/// THE ROW-VALIDITY DOMAIN: which method tags a row may carry AT ALL.
///
/// This answers "is this a representable row", and it is READ BY [`CellKey::is_contracted`].
/// It keeps BOTH methods after the 2.0.0 narrowing, deliberately. Narrowing it would make a
/// second method's row fail row validity, which would delete the two deferred-scope negatives
/// by making the rows they doctor unbuildable — a gate that looks tighter while proving less.
///
/// It is NOT the expectation-set domain. See [`ACTIVE_METHODS`], which is, and read both
/// comments together: the whole defect class here is ONE list being taken as the answer to TWO
/// different questions.
pub const BENCH_METHODS: [Method; 2] = [Method::Setfit, Method::Lora];

/// THE EXPECTATION-SET DOMAIN: which methods a COMPLETE RUN must contain a cell for.
///
/// This answers "what does complete mean", and it is what [`EXPECTED_CELLS`] and
/// [`RunManifest::expectation()`] derive from. Since the claims contract's 2.0.0 narrowing
/// (D-19, approved at a blocking human checkpoint) it names ONE method, matching
/// `equations.expectation_set.methods`, so the expectation set is 1 * 4 * 10 = 40 cells.
///
/// The two-method 80-cell product is not deleted: it is retained, marked deferred, at
/// `equations.expectation_set.deferred_two_method_scope` in the contract and reachable in code
/// only from the `#[cfg(test)]`-gated deferred scope. `D-ITEM-05-15` restores it, and
/// restoration is an edit to THIS constant plus the contract list it is pinned to — not a
/// re-derivation.
///
/// It is NOT the row-validity domain. See [`BENCH_METHODS`], which is.
pub const ACTIVE_METHODS: [Method; 1] = [Method::Setfit];

/// The four contracted shot counts, ASCENDING.
pub const BENCH_SHOTS: [u32; 4] = [8, 16, 32, 64];

/// The ten contracted seeds, ASCENDING.
///
/// Byte-equal to `few_shot_protocol.seeds` in `tweet-eval-stance-benchmark-v1.yaml`. 42 is
/// NOT among them (OBLIG-TWEET-EVAL-SEED-SET), and a tool that defaults a seed to 42 samples
/// outside the contract while appearing to honour it.
pub const BENCH_SEEDS: [u32; 10] = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53];

/// The size of the ACTIVE expectation set: `|ACTIVE_METHODS| * |shots| * |seeds|` = 40.
///
/// A PRODUCT, never a literal — so narrowing the method list moved this number automatically
/// and a hand-typed count beside an unchanged list is unrepresentable. Derived from
/// [`ACTIVE_METHODS`] (what a complete run must contain), NOT from [`BENCH_METHODS`] (what a
/// row may carry).
pub const EXPECTED_CELLS: usize = ACTIVE_METHODS.len() * BENCH_SHOTS.len() * BENCH_SEEDS.len();

/// Which expectation set a derivation or a verification is performed against.
///
/// TWO SCOPES, ONE IMPLEMENTATION (OPS-03). [`RunManifest::expectation_for`] builds both from
/// the same product, so the deferred scope cannot drift from the active one by being written
/// twice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectationScope {
    /// The ACTIVE scope: [`ACTIVE_METHODS`] x [`BENCH_SHOTS`] x [`BENCH_SEEDS`] = 40 cells.
    /// This is what every shipped door verifies against.
    Active,
    /// The DEFERRED two-method scope: [`BENCH_METHODS`] x shots x seeds = 80 cells, matching
    /// `equations.expectation_set.deferred_two_method_scope` in the claims contract.
    ///
    /// THE VARIANT ITSELF IS `#[cfg(test)]`-GATED, so in a production build it does not exist
    /// and no caller can name it — which is stronger than a variant that merely happens not to
    /// be constructed (T-05-11-07). Its only callers are the deferred-scope negatives that
    /// doctor a second method's rows, which is work `D-ITEM-05-15` restores.
    #[cfg(test)]
    DeferredTwoMethod,
}

impl ExpectationScope {
    /// The method axis this scope's product is built over.
    #[must_use]
    pub fn methods(self) -> &'static [Method] {
        match self {
            Self::Active => &ACTIVE_METHODS,
            #[cfg(test)]
            Self::DeferredTwoMethod => &BENCH_METHODS,
        }
    }
}

/// The split calibration diagnostics are measured on, and the only value a row may record.
pub const CALIBRATION_SPLIT: &str = "validation";

/// The contracted warmup count discarded before the warm-latency measurement.
pub const WARMUP_COUNT: u32 = 3;

/// Exact kernel high-water mark of a spawned child, macOS (`/usr/bin/time -l`).
pub const MECHANISM_CHILD_MAX_RSS_TIME_L: &str = "child_max_rss_time_l";

/// Exact kernel high-water mark of a spawned child, Linux (`/usr/bin/time -v` / `VmHWM`).
pub const MECHANISM_CHILD_MAX_RSS_VM_HWM: &str = "child_max_rss_vm_hwm";

/// The claims contract, embedded at compile time.
///
/// `include_str!` rather than a runtime read: a parity test that silently skips when the
/// contract is absent proves nothing, and a path that resolves differently under `cargo test`
/// and in the packaged crate is a defect waiting for a release.
#[cfg(test)]
pub(crate) const CLAIMS_CONTRACT_YAML: &str =
    include_str!("../../../../../contracts/setfit-benchmark-claims-v1.yaml");

/// Lowercase hex of the SHA-256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    let mut out = String::with_capacity(64);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

// ===========================================================================================
// Cell identity
// ===========================================================================================

/// Which method a row measures.
///
/// The wire form is lowercase, matching the contract's `expectation_set.methods`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    /// The SetFit path: a standalone APR carrying encoder, tokenizer and head.
    Setfit,
    /// The 9B LoRA baseline: a base model plus a trained adapter.
    Lora,
}

impl Method {
    /// The wire tag, which is also the first component of a cell key.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Setfit => "setfit",
            Self::Lora => "lora",
        }
    }

    /// Parse a wire tag. `None` for anything outside the two contracted methods.
    #[must_use]
    pub fn from_tag(tag: &str) -> Option<Self> {
        match tag {
            "setfit" => Some(Self::Setfit),
            "lora" => Some(Self::Lora),
            _ => None,
        }
    }

    /// Position in the contract's declared method order — the primary sort key.
    const fn order(self) -> u8 {
        match self {
            Self::Setfit => 0,
            Self::Lora => 1,
        }
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.tag())
    }
}

/// One cell of the benchmark matrix.
///
/// `Ord` is DERIVED over `(method, shots, seed)` in that field order, and `Method`'s own
/// `Ord` follows its declaration order — so the derived ordering IS the contract ordering
/// (method in declared order, then shots ascending, then seed ascending) rather than a second
/// definition that could drift from it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellKey {
    /// Which method.
    pub method: Method,
    /// Examples per class.
    pub shots: u32,
    /// The sampling seed.
    pub seed: u32,
}

impl CellKey {
    /// A cell key. No validation here — [`BenchRow::from_bytes`] and
    /// [`RunManifest::record`] are the doors that refuse an uncontracted cell.
    #[must_use]
    pub const fn new(method: Method, shots: u32, seed: u32) -> Self {
        Self { method, shots, seed }
    }

    /// The stable rendering `"<method>/s<shots>/seed<seed>"`.
    #[must_use]
    pub fn render(&self) -> String {
        format!("{}/s{}/seed{}", self.method.tag(), self.shots, self.seed)
    }

    /// Whether every component is contracted.
    #[must_use]
    pub fn is_contracted(&self) -> bool {
        BENCH_METHODS.contains(&self.method)
            && BENCH_SHOTS.contains(&self.shots)
            && BENCH_SEEDS.contains(&self.seed)
    }
}

impl fmt::Display for CellKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

// ===========================================================================================
// The row
// ===========================================================================================

/// Volatile metadata. NEVER part of the digest.
///
/// Re-running a cell on a newer build must not change the row's identity while its measured
/// content is identical, so the tool version sits outside the payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolatileBenchMetadata {
    /// Left empty by this build; a timestamp would make two writes of one row differ.
    pub created_at: String,
    /// The crate version that wrote the row.
    pub tool_version: String,
}

impl Default for VolatileBenchMetadata {
    fn default() -> Self {
        Self { created_at: String::new(), tool_version: env!("CARGO_PKG_VERSION").to_string() }
    }
}

/// Where the cell ran. Recorded per row because resource metrics are never pooled across
/// hosts (D-09): SetFit cells run CPU and LoRA cells run on the GPU host, and a report that
/// averaged across them would state something no measurement supports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HostIdentity {
    /// The host's name.
    pub hostname: String,
    /// Operating-system identifier.
    pub os: String,
    /// CPU architecture identifier.
    pub arch: String,
}

/// The quality half of a row (EVAL-01).
///
/// Every headline `f64` carries its IEEE-754 `_bits` sibling, following the `EvalRow`
/// convention: the lock hashes bits, and a decimal is a rendering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualityBlock {
    /// Official `F_avg = (F1_against + F1_favor) / 2` — the HEADLINE metric.
    pub f_avg: f64,
    /// `f_avg`'s IEEE-754 bits.
    pub f_avg_bits: u64,
    /// Three-class macro F1, published BESIDE `f_avg` and never instead of it.
    pub macro_f1: f64,
    /// `macro_f1`'s IEEE-754 bits.
    pub macro_f1_bits: u64,
    /// Per-class precision, in `ordered_labels` order.
    pub per_class_precision: Vec<f64>,
    /// Per-class recall, in `ordered_labels` order.
    pub per_class_recall: Vec<f64>,
    /// Per-class F1, in `ordered_labels` order.
    pub per_class_f1: Vec<f64>,
    /// Matthews correlation coefficient over the three classes.
    pub mcc: f64,
    /// `mcc`'s IEEE-754 bits.
    pub mcc_bits: u64,
    /// Row-major `[3][3]` counts, true label by predicted label.
    pub confusion_matrix: Vec<Vec<u64>>,
    /// How many rows the metrics were computed over.
    pub n_test_rows: u64,
    /// The labels the head indexes by, in row order — recorded explicitly so no metric
    /// depends on an implicit index order.
    pub ordered_labels: Vec<String>,
    /// Top-label ECE on the CANONICAL VALIDATION split.
    pub ece_top_label_validation: f64,
    /// Its IEEE-754 bits.
    pub ece_top_label_validation_bits: u64,
    /// UNNORMALISED multiclass Brier on the canonical validation split. Codomain `[0, 2]`,
    /// NOT `[0, 1]` — at K=2 it is exactly twice the binary score.
    pub brier_multiclass_validation: f64,
    /// Its IEEE-754 bits.
    pub brier_multiclass_validation_bits: u64,
    /// The fixed string [`CALIBRATION_SPLIT`]. Calibration is validation-only (D-07).
    pub calibration_split: String,
}

/// The resource half of a row (EVAL-05), with every measurement boundary recorded.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResourceBlock {
    /// Wall-clock of the training invocation.
    pub train_wall_ms: u64,
    /// ONE single-text classify in a dedicated fresh child process.
    pub cold_latency_ms: f64,
    /// Always `true` on a valid row: a cold latency measured in the process that just
    /// finished training is operationally WARM (populated caches, resident pages) and the
    /// contract forbids emitting it.
    pub cold_measured_in_child_process: bool,
    /// Median of ten single-text classifies after [`WARMUP_COUNT`] warmups.
    pub warm_latency_ms_median: f64,
    /// `n_test_rows` divided by the wall time of one full test-split batch pass.
    pub throughput_rows_per_sec: f64,
    /// The batch size that pass used. Throughput without a batch size is not comparable.
    pub throughput_batch_size: u32,
    /// Warmup classifies discarded before the warm measurement.
    pub warmup_count: u32,
    /// Peak RSS of the TRAINING process.
    pub train_peak_rss_bytes: u64,
    /// How `train_peak_rss_bytes` was obtained.
    pub train_peak_rss_mechanism: String,
    /// Peak RSS of the dedicated COLD-MEASUREMENT CHILD — a different process, a different
    /// number. A kernel high-water mark is process-CUMULATIVE, so one pooled figure would
    /// report the training peak while claiming to report the inference peak.
    pub inference_peak_rss_bytes: u64,
    /// How `inference_peak_rss_bytes` was obtained.
    pub inference_peak_rss_mechanism: String,
    /// Present exactly when a mechanism is a sampled one, which is a LOWER BOUND and is not
    /// comparable to a `child_max_rss_*` figure.
    pub peak_rss_sample_interval_hz: Option<u32>,
    /// Bytes of the artifact THIS method wrote (SetFit: the standalone APR; LoRA: the
    /// adapter alone).
    pub artifact_bytes: u64,
    /// Bytes a user must ship to serve this model. The ONLY field a cross-method size claim
    /// may be built on.
    pub deployable_total_bytes: u64,
}

/// The SetFit lock reference. `lock_hash` is a CLAIM about the file at `lock_record_path`;
/// the file is the evidence, and the report recomputes the digest from its bytes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchLockRef {
    /// Hex SHA-256 of the committed lock record's bytes.
    pub lock_hash: String,
    /// `"written"` or `"consumed"`, matching `apr eval`'s `LockRow` vocabulary.
    pub role: String,
    /// The selection rule the lock records.
    pub rule: String,
    /// Path to the committed lock record, RELATIVE to the benchmark directory.
    pub lock_record_path: String,
}

/// SetFit-side evidence. Every field is non-`Option`: a row cannot omit one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SetfitEvidence {
    /// The Ph3 D-12 evidence-table hash, exactly as the APR carries it.
    pub evidence_table_hash: String,
    /// SHA-256 of the `setfit-apr-v1` artifact the predictions came from.
    pub apr_artifact_sha256: String,
    /// The selection lock this cell wrote or consumed.
    pub lock: BenchLockRef,
}

/// LoRA-side evidence. Every field is non-`Option`: a row cannot omit one.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoraEvidence {
    /// SHA-256 of the base model the adapter applies to.
    pub base_model_sha256: String,
    /// Bytes of that base model. Mandatory: an adapter alone is not deployable, and the
    /// base model is not in the benchmark directory for a report to measure.
    pub base_model_bytes: u64,
    /// SHA-256 of the trained adapter.
    pub adapter_sha256: String,
    /// The epoch count the run was asked for.
    pub epochs_requested: u32,
    /// The epoch count the run actually finished.
    pub epochs_completed: u32,
    /// Always `true` on a valid row.
    pub early_stopping_disabled: bool,
    /// Always `0.0` on a valid row: no validation split was carved, so there was nothing to
    /// select on.
    pub val_split: f64,
    /// The attestation itself, which the two ledger fields below are what make checkable.
    pub no_selection_attestation: bool,
    /// Hex SHA-256 over the append-only candidate ledger's bytes.
    pub candidate_ledger_sha256: String,
    /// The ledger's line count, which the contract requires to be exactly 1.
    pub candidates_trained: u32,
    /// Path to the ledger, RELATIVE to the benchmark directory.
    pub candidate_ledger_path: String,
}

/// The method-specific evidence block (D-12).
///
/// EXTERNALLY TAGGED — the wire form is `{"setfit": {..}}` or `{"lora": {..}}`. One shared
/// null-padded superset would make "the block is missing" and "the block is empty" the same
/// bytes, and would let a row be shaped like SetFit evidence without being it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
pub enum MethodEvidence {
    /// A SetFit cell's evidence.
    Setfit(SetfitEvidence),
    /// A LoRA cell's evidence.
    Lora(LoraEvidence),
}

impl MethodEvidence {
    /// Which method this block belongs to.
    #[must_use]
    pub const fn method(&self) -> Method {
        match self {
            Self::Setfit(_) => Method::Setfit,
            Self::Lora(_) => Method::Lora,
        }
    }
}

/// The hashed part of a row: everything a claim is computed from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRowPayload {
    /// The row schema version.
    pub schema_version: u32,
    /// The contract this row is written against.
    pub contract_id: String,
    /// Which method — the tag that selects the evidence block.
    pub method: Method,
    /// Examples per class.
    pub shots: u32,
    /// The sampling seed.
    pub seed: u32,
    /// The pinned upstream dataset revision.
    pub dataset_revision: String,
    /// The whole-dataset fingerprint from the attested `PreparedDataset`.
    pub dataset_fingerprint: String,
    /// Encoder (SetFit) or base-model (LoRA) revision, as resolved.
    pub model_revision: String,
    /// Hex SHA-256 of the selection manifest this cell consumed — THE PAIRING KEY.
    pub selection_manifest_hash: String,
    /// `<device>:<implementation>:<kernel>`, READ FROM EXECUTION (Ph4 D-12), never echoed
    /// from configuration.
    pub backend_identity: String,
    /// Where the cell ran.
    pub host: HostIdentity,
    /// The quality metrics.
    pub quality: QualityBlock,
    /// The resource metrics and their boundaries.
    pub resource: ResourceBlock,
    /// The method-specific evidence.
    pub evidence: MethodEvidence,
}

impl BenchRowPayload {
    /// The bytes the digest is taken over.
    ///
    /// Serialized through [`serde_json::Value`] so key order is `BTreeMap`-canonical rather
    /// than declaration-ordered. See this module's header for why that matters.
    ///
    /// # Errors
    ///
    /// [`BenchRowError::Serialization`] if the payload cannot be serialized.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, BenchRowError> {
        canonical_bytes(self, "bench_row_payload")
    }

    /// This row's cell.
    #[must_use]
    pub const fn cell(&self) -> CellKey {
        CellKey::new(self.method, self.shots, self.seed)
    }
}

/// The on-disk row: digest, unhashed volatile block, payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchRow {
    /// Lowercase hex of `SHA-256(payload.to_canonical_bytes())`.
    pub semantic_hash: String,
    /// Volatile metadata. NEVER part of the digest.
    pub volatile: VolatileBenchMetadata,
    /// The hashed payload.
    pub payload: BenchRowPayload,
}

impl BenchRow {
    /// Seal a payload into a row, computing its digest.
    ///
    /// # Panics
    ///
    /// Never in practice: the digest is computed from an owned, already-typed payload, and a
    /// `serde_json` failure over it is not reachable. The fallback records an empty digest
    /// rather than panicking, which [`Self::from_bytes`] then refuses.
    #[must_use]
    pub fn new(payload: BenchRowPayload) -> Self {
        let semantic_hash =
            payload.to_canonical_bytes().map_or_else(|_| String::new(), |bytes| sha256_hex(&bytes));
        Self { semantic_hash, volatile: VolatileBenchMetadata::default(), payload }
    }

    /// The on-disk byte form: pretty JSON plus a terminating newline.
    ///
    /// Pretty because rows are reviewed in diffs; the DIGEST is over the payload's own
    /// canonical bytes, so the file's whitespace can be chosen for humans without weakening
    /// anything.
    ///
    /// # Errors
    ///
    /// [`BenchRowError::Serialization`] if the row cannot be serialized.
    pub fn to_file_bytes(&self) -> Result<Vec<u8>, BenchRowError> {
        pretty_file_bytes(self, "bench_row")
    }

    /// Parse a row, verifying schema version, digest, cell and evidence tag BEFORE returning.
    ///
    /// # Errors
    ///
    /// [`BenchRowError::UnsupportedSchemaVersion`], [`BenchRowError::SemanticHashMismatch`],
    /// [`BenchRowError::UncontractedCell`], [`BenchRowError::MissingSetfitEvidence`],
    /// [`BenchRowError::MissingLoraEvidence`] or [`BenchRowError::Serialization`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BenchRowError> {
        let row: Self = match serde_json::from_slice(bytes) {
            Ok(row) => row,
            // The strict parse failed. Classify STRUCTURALLY — by re-parsing the evidence
            // sub-value against the block its own `method` field demands — rather than by
            // matching serde's message, which is a rendering and not a fact.
            Err(error) => return Err(classify_row_parse_failure(bytes, &error)),
        };

        // Schema version FIRST: a foreign schema's digest convention is not necessarily
        // ours, so comparing digests across versions would compare two different questions.
        if row.payload.schema_version != BENCH_ROW_SCHEMA_VERSION {
            return Err(BenchRowError::UnsupportedSchemaVersion {
                got: row.payload.schema_version,
                supported: BENCH_ROW_SCHEMA_VERSION,
            });
        }

        let recomputed = sha256_hex(&row.payload.to_canonical_bytes()?);
        if recomputed != row.semantic_hash {
            return Err(BenchRowError::SemanticHashMismatch {
                expected: row.semantic_hash.clone(),
                got: recomputed,
            });
        }

        let cell = row.payload.cell();
        if !cell.is_contracted() {
            return Err(BenchRowError::UncontractedCell { cell: cell.render() });
        }

        // The tag and the block are two statements of the same fact; they must agree.
        // SAFE-03's spirit: a lora row may not present setfit evidence.
        if row.payload.evidence.method() != row.payload.method {
            return Err(match row.payload.method {
                Method::Setfit => BenchRowError::MissingSetfitEvidence {
                    detail: "the row declares method `setfit` but carries lora evidence"
                        .to_string(),
                },
                Method::Lora => BenchRowError::MissingLoraEvidence {
                    detail: "the row declares method `lora` but carries setfit evidence"
                        .to_string(),
                },
            });
        }

        Ok(row)
    }
}

/// The evidence-block half of a lenient re-parse, used only to CLASSIFY a strict failure.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceProbe {
    #[serde(default)]
    setfit: Option<serde_json::Value>,
    #[serde(default)]
    lora: Option<serde_json::Value>,
}

/// Turn a failed strict parse into the most specific typed refusal the bytes support.
///
/// A missing `lock` inside a setfit block and a missing `candidate_ledger_sha256` inside a
/// lora block are BOTH plain serde "missing field" errors, indistinguishable from an unknown
/// field at the envelope. Re-parsing the evidence sub-value against the block the row's own
/// `method` demands is what separates them — structurally, not by reading serde's prose.
fn classify_row_parse_failure(bytes: &[u8], error: &serde_json::Error) -> BenchRowError {
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(bytes) else {
        return BenchRowError::Serialization {
            context: "bench_row".to_string(),
            detail: error.to_string(),
        };
    };
    let Some(method) = value
        .get("payload")
        .and_then(|p| p.get("method"))
        .and_then(serde_json::Value::as_str)
        .and_then(Method::from_tag)
    else {
        return BenchRowError::Serialization {
            context: "bench_row".to_string(),
            detail: error.to_string(),
        };
    };
    let Some(evidence) = value.get("payload").and_then(|p| p.get("evidence")) else {
        return missing_evidence(method, "the payload carries no `evidence` block".to_string());
    };
    let Ok(probe) = serde_json::from_value::<EvidenceProbe>(evidence.clone()) else {
        return missing_evidence(
            method,
            "the `evidence` block names something other than exactly one of `setfit`/`lora`"
                .to_string(),
        );
    };
    match method {
        Method::Setfit => match probe.setfit {
            None => missing_evidence(method, "no `setfit` evidence block is present".to_string()),
            Some(block) => match serde_json::from_value::<SetfitEvidence>(block) {
                Ok(_) => BenchRowError::Serialization {
                    context: "bench_row".to_string(),
                    detail: error.to_string(),
                },
                Err(inner) => missing_evidence(method, inner.to_string()),
            },
        },
        Method::Lora => match probe.lora {
            None => missing_evidence(method, "no `lora` evidence block is present".to_string()),
            Some(block) => match serde_json::from_value::<LoraEvidence>(block) {
                Ok(_) => BenchRowError::Serialization {
                    context: "bench_row".to_string(),
                    detail: error.to_string(),
                },
                Err(inner) => missing_evidence(method, inner.to_string()),
            },
        },
    }
}

fn missing_evidence(method: Method, detail: String) -> BenchRowError {
    match method {
        Method::Setfit => BenchRowError::MissingSetfitEvidence { detail },
        Method::Lora => BenchRowError::MissingLoraEvidence { detail },
    }
}

// ===========================================================================================
// The run manifest
// ===========================================================================================

/// Whether a declared cell has been produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CellStatus {
    /// Declared, not yet produced.
    Pending,
    /// Produced, with `row_sha256` recorded.
    Complete,
}

/// One entry of the pre-declared expectation table.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CellEntry {
    /// Which method.
    pub method: Method,
    /// Examples per class.
    pub shots: u32,
    /// The sampling seed.
    pub seed: u32,
    /// Whether the cell has been produced.
    pub status: CellStatus,
    /// The row's content digest — `Some` exactly when `status` is `Complete`.
    pub row_sha256: Option<String>,
}

impl CellEntry {
    /// This entry's cell key.
    #[must_use]
    pub const fn cell(&self) -> CellKey {
        CellKey::new(self.method, self.shots, self.seed)
    }
}

/// The hashed part of the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifestPayload {
    /// The manifest schema version.
    pub schema_version: u32,
    /// The contract the expectation set is derived from.
    pub contract_id: String,
    /// All 40 declared ACTIVE cells, in the deterministic contract order.
    pub cells: Vec<CellEntry>,
}

impl RunManifestPayload {
    /// The bytes the digest is taken over.
    ///
    /// # Errors
    ///
    /// [`BenchRowError::Serialization`] if the payload cannot be serialized.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, BenchRowError> {
        canonical_bytes(self, "run_manifest_payload")
    }
}

/// What a [`RunManifest::record`] call did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordOutcome {
    /// The cell moved from pending to complete.
    Recorded,
    /// The cell was already complete with the IDENTICAL digest — the resume path.
    AlreadyRecorded,
}

/// The run manifest: digest, unhashed volatile block, the pre-declared expectation table.
///
/// Completeness is defined by THIS FILE and the contract, never by a directory listing. A
/// listing can only report what is present; it cannot report what is missing, because nothing
/// tells it what was expected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunManifest {
    /// Lowercase hex of `SHA-256(payload.to_canonical_bytes())`.
    pub semantic_hash: String,
    /// Volatile metadata. NEVER part of the digest.
    pub volatile: VolatileBenchMetadata,
    /// The hashed payload.
    pub payload: RunManifestPayload,
}

impl RunManifest {
    /// The contract-derived ACTIVE expectation set: 40 unique cells in the deterministic
    /// contract order (method in declared order, then shots ascending, then seed ascending).
    ///
    /// Derived IN CODE from the constants, never read from row hashes or a directory — which
    /// is what lets the manifest be declared BEFORE any cell runs.
    ///
    /// Derives from [`ACTIVE_METHODS`]. The deferred two-method scope is
    /// [`Self::expectation_for`] with [`ExpectationScope::DeferredTwoMethod`], which is
    /// `#[cfg(test)]`-gated and therefore unreachable from production code.
    #[must_use]
    pub fn expectation() -> Vec<CellKey> {
        Self::expectation_for(ExpectationScope::Active)
    }

    /// The expectation set of a given scope, in the deterministic contract order.
    ///
    /// One implementation for both scopes (OPS-03): a second, separately-written product would
    /// be a second definition of "the expectation set" and two definitions drift.
    #[must_use]
    pub fn expectation_for(scope: ExpectationScope) -> Vec<CellKey> {
        let mut methods = scope.methods().to_vec();
        let mut cells = Vec::with_capacity(methods.len() * BENCH_SHOTS.len() * BENCH_SEEDS.len());
        methods.sort_unstable_by_key(|m| m.order());
        let mut shots = BENCH_SHOTS;
        shots.sort_unstable();
        let mut seeds = BENCH_SEEDS;
        seeds.sort_unstable();
        for method in methods {
            for shot in shots {
                for seed in seeds {
                    cells.push(CellKey::new(method, shot, seed));
                }
            }
        }
        cells
    }

    /// Declare a fresh manifest: all 40 ACTIVE cells `Pending`, no digests yet.
    #[must_use]
    pub fn declare() -> Self {
        Self::declare_for(ExpectationScope::Active)
    }

    /// Declare a fresh manifest over an explicit scope.
    ///
    /// Production callers can only name [`ExpectationScope::Active`], because the deferred
    /// variant is `#[cfg(test)]`-gated — so this is not a widening door.
    #[must_use]
    pub fn declare_for(scope: ExpectationScope) -> Self {
        let cells = Self::expectation_for(scope)
            .into_iter()
            .map(|cell| CellEntry {
                method: cell.method,
                shots: cell.shots,
                seed: cell.seed,
                status: CellStatus::Pending,
                row_sha256: None,
            })
            .collect();
        Self::seal(RunManifestPayload {
            schema_version: RUN_MANIFEST_SCHEMA_VERSION,
            contract_id: CLAIMS_CONTRACT_ID.to_string(),
            cells,
        })
    }

    /// Seal a payload, computing its digest.
    #[must_use]
    pub fn seal(payload: RunManifestPayload) -> Self {
        let semantic_hash =
            payload.to_canonical_bytes().map_or_else(|_| String::new(), |bytes| sha256_hex(&bytes));
        Self { semantic_hash, volatile: VolatileBenchMetadata::default(), payload }
    }

    /// The on-disk byte form: pretty JSON plus a terminating newline.
    ///
    /// # Errors
    ///
    /// [`BenchRowError::Serialization`] if the manifest cannot be serialized.
    pub fn to_file_bytes(&self) -> Result<Vec<u8>, BenchRowError> {
        pretty_file_bytes(self, "run_manifest")
    }

    /// Parse a manifest, verifying schema version, digest and the expectation set BEFORE
    /// returning — and BEFORE any row byte is read.
    ///
    /// The two backstops are the point: a zero-cell manifest is an ERROR rather than an `Ok`
    /// over an empty aggregate, and a cell sequence that is not exactly the 40 contract-derived
    /// ACTIVE cells in contract order is an ERROR rather than a smaller definition of
    /// "complete". This second backstop is also what refuses a manifest DECLARING a cell for a
    /// method outside the active scope, before any row byte is read — no separate refusal is
    /// minted for that, because it is the same defect.
    ///
    /// # Errors
    ///
    /// [`BenchRowError::UnsupportedSchemaVersion`], [`BenchRowError::SemanticHashMismatch`],
    /// [`BenchRowError::EmptyExpectationSet`], [`BenchRowError::ExpectationSetMismatch`] or
    /// [`BenchRowError::Serialization`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BenchRowError> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|error| BenchRowError::Serialization {
                context: "run_manifest".to_string(),
                detail: error.to_string(),
            })?;

        if manifest.payload.schema_version != RUN_MANIFEST_SCHEMA_VERSION {
            return Err(BenchRowError::UnsupportedSchemaVersion {
                got: manifest.payload.schema_version,
                supported: RUN_MANIFEST_SCHEMA_VERSION,
            });
        }

        let recomputed = sha256_hex(&manifest.payload.to_canonical_bytes()?);
        if recomputed != manifest.semantic_hash {
            return Err(BenchRowError::SemanticHashMismatch {
                expected: manifest.semantic_hash.clone(),
                got: recomputed,
            });
        }

        if manifest.payload.cells.is_empty() {
            return Err(BenchRowError::EmptyExpectationSet);
        }

        // SEQUENCE equality, which subsumes set equality, cardinality and order in one
        // comparison — so a reordered manifest is caught by the same check as a truncated one.
        let declared: Vec<CellKey> = manifest.payload.cells.iter().map(CellEntry::cell).collect();
        if declared != Self::expectation() {
            return Err(BenchRowError::ExpectationSetMismatch {
                declared: declared.len(),
                expected: EXPECTED_CELLS,
            });
        }

        Ok(manifest)
    }

    /// How many declared cells are complete.
    #[must_use]
    pub fn completed(&self) -> usize {
        self.payload.cells.iter().filter(|entry| entry.status == CellStatus::Complete).count()
    }

    /// The recorded digest of `cell`, if it is complete.
    #[must_use]
    pub fn row_sha256(&self, cell: CellKey) -> Option<&str> {
        self.payload
            .cells
            .iter()
            .find(|entry| entry.cell() == cell)
            .and_then(|entry| entry.row_sha256.as_deref())
    }

    /// Record a produced row against its declared cell.
    ///
    /// IDEMPOTENT on an identical digest (the resume path) and a typed error on a differing
    /// one (the collision). A refused record does NOT partially apply.
    ///
    /// # Errors
    ///
    /// [`BenchRowError::UnknownCell`] if the cell is not in the expectation set;
    /// [`BenchRowError::CellAlreadyRecorded`] if it is complete with a different digest.
    pub fn record(
        &mut self,
        cell: CellKey,
        row_sha256: &str,
    ) -> Result<RecordOutcome, BenchRowError> {
        let Some(index) = self.payload.cells.iter().position(|entry| entry.cell() == cell) else {
            return Err(BenchRowError::UnknownCell { cell: cell.render() });
        };

        // Read the existing state first, so nothing is mutated on the refusal path.
        if self.payload.cells[index].status == CellStatus::Complete {
            let existing = self.payload.cells[index].row_sha256.clone().unwrap_or_default();
            if existing == row_sha256 {
                return Ok(RecordOutcome::AlreadyRecorded);
            }
            return Err(BenchRowError::CellAlreadyRecorded {
                cell: cell.render(),
                existing,
                incoming: row_sha256.to_string(),
            });
        }

        self.payload.cells[index].status = CellStatus::Complete;
        self.payload.cells[index].row_sha256 = Some(row_sha256.to_string());
        self.reseal();
        Ok(RecordOutcome::Recorded)
    }

    /// Recompute the envelope digest after a payload transition.
    fn reseal(&mut self) {
        self.semantic_hash = self
            .payload
            .to_canonical_bytes()
            .map_or_else(|_| String::new(), |bytes| sha256_hex(&bytes));
    }
}

// ===========================================================================================
// Shared serialization helpers
// ===========================================================================================

fn canonical_bytes<T: Serialize>(value: &T, context: &str) -> Result<Vec<u8>, BenchRowError> {
    let as_value = serde_json::to_value(value).map_err(|error| BenchRowError::Serialization {
        context: context.to_string(),
        detail: error.to_string(),
    })?;
    serde_json::to_vec(&as_value).map_err(|error| BenchRowError::Serialization {
        context: context.to_string(),
        detail: error.to_string(),
    })
}

fn pretty_file_bytes<T: Serialize>(value: &T, context: &str) -> Result<Vec<u8>, BenchRowError> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| {
        BenchRowError::Serialization { context: context.to_string(), detail: error.to_string() }
    })?;
    bytes.push(b'\n');
    Ok(bytes)
}

// ===========================================================================================
// Errors
// ===========================================================================================

/// Failure modes of row and manifest parsing and recording.
///
/// `#[non_exhaustive]`: the report gate (05-10) adds refusals of its own over the same
/// vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BenchRowError {
    /// The value could not be serialized, or the bytes could not be parsed.
    Serialization {
        /// What was being (de)serialized.
        context: String,
        /// The underlying diagnostic, including the parse position when serde has one.
        detail: String,
    },
    /// The payload declares a schema version this build does not implement.
    UnsupportedSchemaVersion {
        /// The version found.
        got: u32,
        /// The version this build writes and reads.
        supported: u32,
    },
    /// The recomputed digest disagrees with the envelope's.
    SemanticHashMismatch {
        /// The digest the envelope claims.
        expected: String,
        /// The digest the payload's own bytes produce.
        got: String,
    },
    /// The row names a `(method, shots, seed)` outside the contracted matrix.
    UncontractedCell {
        /// The offending cell, rendered.
        cell: String,
    },
    /// A row declaring `method: "setfit"` carries no USABLE setfit evidence block — absent
    /// entirely, present but naming another method, or present but not deserializable into
    /// the block this schema requires (a missing non-`Option` field, or a field this schema
    /// does not name).
    MissingSetfitEvidence {
        /// Which of those it was.
        detail: String,
    },
    /// The LoRA twin of [`Self::MissingSetfitEvidence`].
    MissingLoraEvidence {
        /// Which of those it was.
        detail: String,
    },
    /// The manifest declares no cells at all. A gate whose input is empty and which
    /// therefore finds no failures reports success; zero cells is a failure, not a clean run.
    EmptyExpectationSet,
    /// The manifest's cell sequence is not exactly the contract-derived expectation set in
    /// contract order.
    ExpectationSetMismatch {
        /// How many cells the manifest declares.
        declared: usize,
        /// How many the contract derives.
        expected: usize,
    },
    /// A record was attempted against a cell outside the expectation set.
    UnknownCell {
        /// The offending cell, rendered.
        cell: String,
    },
    /// The cell is already complete with a DIFFERENT digest. Both are named, because the
    /// contradiction is the finding.
    CellAlreadyRecorded {
        /// The offending cell, rendered.
        cell: String,
        /// The digest already recorded.
        existing: String,
        /// The digest being offered.
        incoming: String,
    },
}

impl BenchRowError {
    /// A stable discriminant string, so a test can assert that N refusals are N DISTINCT
    /// variants without matching on rendered prose.
    #[must_use]
    pub const fn variant_tag(&self) -> &'static str {
        match self {
            Self::Serialization { .. } => "serialization",
            Self::UnsupportedSchemaVersion { .. } => "unsupported_schema_version",
            Self::SemanticHashMismatch { .. } => "semantic_hash_mismatch",
            Self::UncontractedCell { .. } => "uncontracted_cell",
            Self::MissingSetfitEvidence { .. } => "missing_setfit_evidence",
            Self::MissingLoraEvidence { .. } => "missing_lora_evidence",
            Self::EmptyExpectationSet => "empty_expectation_set",
            Self::ExpectationSetMismatch { .. } => "expectation_set_mismatch",
            Self::UnknownCell { .. } => "unknown_cell",
            Self::CellAlreadyRecorded { .. } => "cell_already_recorded",
        }
    }
}

impl fmt::Display for BenchRowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Serialization { context, detail } => {
                write!(f, "benchmark {context} failed: {detail}")
            }
            Self::UnsupportedSchemaVersion { got, supported } => write!(
                f,
                "benchmark artifact declares schema version {got}, but this build implements \
                 {supported}; an artifact from a different schema is refused rather than \
                 partially interpreted",
            ),
            Self::SemanticHashMismatch { expected, got } => write!(
                f,
                "digest mismatch: the envelope claims {expected} but the payload's own bytes \
                 hash to {got}. These are not the bytes that were attested; re-run the cell \
                 rather than editing the row",
            ),
            Self::UncontractedCell { cell } => write!(
                f,
                "cell {cell} is outside the contracted matrix (methods setfit|lora, shots \
                 8|16|32|64, the ten contracted seeds — 42 is NOT one of them)",
            ),
            Self::MissingSetfitEvidence { detail } => write!(
                f,
                "a row declaring method `setfit` carries no usable setfit evidence block: \
                 {detail}. The block is not optional and is never null-padded (D-12)",
            ),
            Self::MissingLoraEvidence { detail } => write!(
                f,
                "a row declaring method `lora` carries no usable lora evidence block: \
                 {detail}. The block is not optional and is never null-padded (D-12)",
            ),
            Self::EmptyExpectationSet => write!(
                f,
                "the run manifest declares zero cells; completeness is defined by the \
                 manifest, so an empty expectation set would let a run that produced nothing \
                 report success. Re-declare it with `bench run --declare`",
            ),
            Self::ExpectationSetMismatch { declared, expected } => write!(
                f,
                "the run manifest declares {declared} cells in this order, but the contract \
                 derives {expected} in the order (method, shots ascending, seed ascending). \
                 Completeness is defined by the contract-derived set, never by whatever the \
                 manifest happens to list",
            ),
            Self::UnknownCell { cell } => write!(
                f,
                "cell {cell} is not in the declared expectation set, so there is no slot to \
                 record it in",
            ),
            Self::CellAlreadyRecorded { cell, existing, incoming } => write!(
                f,
                "cell {cell} is already recorded with digest {existing}, and {incoming} was \
                 offered. A differing re-record is refused rather than silently overwritten: \
                 the run that produced the published number must stay the run on disk",
            ),
        }
    }
}

impl std::error::Error for BenchRowError {}

#[cfg(test)]
#[path = "bench_row_tests.rs"]
mod bench_row_tests;
