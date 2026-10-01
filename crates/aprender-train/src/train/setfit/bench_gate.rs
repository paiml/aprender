//! The claims gate: fail-closed verification over a benchmark directory, then closed-form
//! aggregation (plan 05-10, EVAL-04).
//!
//! Contract: `setfit-benchmark-claims-v1` — equations `completeness_rule`, `pairing_rule`,
//! `selection_safety_evidence`, `no_selection_attestation` and `claims_statistics`. This module
//! implements that file field for field, in the rule order the file declares.
//!
//! # The one property this module exists for
//!
//! EVAL-04: *missing, selectively omitted, unmatched, or post-test-selected cells invalidate the
//! report*. That is a STRUCTURAL claim, so it is enforced structurally: [`aggregate`] takes a
//! [`VerifiedRunSet`], which has no public constructor and is reachable only out of
//! [`verify_run`]. There is no partial-data mode and no "aggregate what we have" path, because
//! neither can be expressed.
//!
//! # Verification order IS the property
//!
//! [`verify_run`] walks the contract's rules in the contract's order, and returns on the FIRST
//! failure:
//!
//! 1. the manifest's own digest, recomputed from its payload's canonical bytes;
//! 2. the manifest's expectation set equals the contract-derived ACTIVE 40, in contract order;
//! 3. every declared cell is `complete` with a recorded digest;
//! 4. per cell: the row file exists, [`BenchRow::from_bytes`] accepts it (schema first, then its
//!    own envelope digest — that ORDER belongs to `bench_row` and is documented there), the file
//!    bytes hash to the digest the MANIFEST recorded, and the payload agrees with the slot;
//! 5. per `(shots, seed)`: the two rows carry an identical `selection_manifest_hash`
//!    (DEFERRED — the active scope has one method, so this rule's domain is empty and it is
//!    reported as not exercised, never as satisfied);
//! 6. per row: provenance, THE SELECTION BINDING and THE PUBLISHED QUALITY, all three
//!    RECOMPUTED rather than trusted (below) — the lock/ledger digests at a row-supplied path
//!    that had to be validated; the `selection_manifest_hash` against the manifest at a
//!    GATE-DERIVED path the row cannot choose ([`selection_manifest_path`]), including the
//!    manifest's own declared shots and seed against the cell key; and `f_avg`, `macro_f1`,
//!    `mcc`, the three per-class vectors, `n_test_rows` and every `_bits` sibling against what
//!    the row's OWN `confusion_matrix` and `ordered_labels` determine in closed form
//!    ([`verify_quality_closed_form`]);
//! 7. per LoRA row: the no-selection attestation's conjuncts (DEFERRED, same reason).
//!
//! Steps 1 + 4 + 6 are ALSO available over a single declared cell as [`verify_cell`], which
//! deliberately excludes the set-level steps — see that function's own note for why a door
//! that inherited them could never pass at pilot time.
//!
//! Refusing early is not an optimisation. Aggregating over partially-verified rows would produce
//! a number, and a number is what a reader takes away.
//!
//! # Provenance is recomputed, never trusted
//!
//! A row's `lock_hash` and `candidate_ledger_sha256` are CLAIMS ABOUT FILES. This gate reads the
//! bytes at `lock.lock_record_path` and at `candidate_ledger_path` and recomputes their SHA-256,
//! and it counts the ledger's lines rather than reading `candidates_trained`. A row field that
//! disagrees with the file it names is a refusal naming BOTH the cell and the file.
//!
//! **AND THE PATH ITSELF IS RESOLVED AT A LOCATION THE ROW CANNOT CHOOSE.** Both of those
//! strings are producer-written, so both go through [`resolve_committed_evidence_path`], which
//! refuses an absolute, rooted, `..`-carrying or empty declaration before any filesystem call
//! and then refuses any target whose canonical form lands outside the canonical benchmark
//! directory. Until that door existed the recomputation above was true of whatever file the row
//! pointed at — including one the benchmark does not commit — which made the sentence
//! "provenance was recomputed from the committed lock bytes" printable on a run where it was
//! false. One helper serves BOTH arms, so a restored two-method scope inherits the enforcement
//! rather than re-opening the hole.
//!
//! # THE RESIDUAL, STATED RATHER THAN HIDDEN
//!
//! Recomputing the lock and ledger digests raises selection safety from a SELF-ASSERTED BOOLEAN
//! to APPEND-ONLY, RECOMPUTABLE EVIDENCE. It does not reach a cryptographic train-then-seal
//! credential, and it is not claimed to. The doctored negatives in this module's test file
//! prove detection of INCONSISTENT evidence; they prove nothing about TRUTHFUL provenance.
//!
//! **The former unqualified sentence here — "a producer that controls the rows AND the
//! lock/ledger files can still emit a mutually consistent forgery" — was RETIRED by 05-17 and
//! is not to be restored as written.** It conceded three attacks this gate now refuses: an
//! escaping evidence path (05-15's containment door), a doctored pairing key (05-16's manifest
//! recomputation at a gate-derived path), and a doctored quality metric (05-17's closed-form
//! cross-check). A residual that keeps conceding a refused attack teaches a reader to trust the
//! evidence LESS than it warrants, which is over-claiming with the sign flipped, and it is as
//! much a defect as overstating the gate.
//!
//! THREE RESIDUALS REMAIN, and they are stated precisely rather than dropped:
//!
//! 1. **The confusion MATRIX is producer-written.** The cross-check proves the published
//!    metrics agree with the matrix recorded beside them; it does NOT prove that matrix is the
//!    one the model produced. A producer who edits the matrix and recomputes the metrics from
//!    it emits a set nothing here can distinguish from a measurement. Closing this needs a new
//!    committed evidence artifact (per-row predictions) that no run writes today.
//! 2. **`ece_top_label_validation` and `brier_multiclass_validation` are not recomputable at
//!    all.** They need per-row probability vectors no committed file carries, so those two
//!    figures stay SELF-ASSERTED. Their `_bits` siblings ARE held against the `f64` beside
//!    them, which is an internal consistency claim and not a proof of the value.
//! 3. **`evidence_table_hash` and `apr_artifact_sha256`** remain claims about artifacts the
//!    index deliberately does not carry.
//!
//! This is the same statement `apr setfit bench report`'s `residual:` line and
//! `setfit-benchmark-claims-v1`'s `selection_safety_evidence.residual_risk` make. The three are
//! ONE fact written down three times, and gap 1 was findable precisely because three statements
//! of one fact had drifted apart — so a change to any of them is a change to all three.
//!
//! A SECOND RESIDUAL SINCE THE 2.0.0 NARROWING, stated rather than left to be inferred: the
//! active scope means this gate cannot be handed a second method's row AT ALL, so nothing it
//! verifies is a cross-method claim. That narrowing buys exactly one thing — the manifest no
//! longer declares cells that cannot exist — and it must not be read as having strengthened
//! anything. The two-method machinery below (pairing, the LoRA attestation, the paired delta)
//! is retained, contract-bound and unexercised, exactly as 05-04's paired-t is. It is repeated
//! here because a gate whose module doc MISDESCRIBES what it proves — in either direction — is
//! the exact failure this phase exists to prevent; the phase's own bar is "a gate could pass
//! while the claim is false", and its mirror image is "a disclosure could concede what the gate
//! refuses".
//!
//! # No RNG, and no second definition of the arithmetic
//!
//! Every statistic comes from the closed-form f64 surface plan 05-04 shipped in
//! `aprender::stats::hypothesis` — [`mean_f64`], [`sample_std_f64`], [`min_max_f64`],
//! [`paired_ci95_df9`] and [`ttest_rel_f64`], all built on the frozen [`T_CRIT_975_DF9`]. Nothing
//! is reimplemented here: a second mean is a second definition, and two definitions of one number
//! disagree eventually and invisibly. There is no `rand` import anywhere in this file, and a test
//! scans the non-comment source to keep it that way (D-06).
//!
//! # A degenerate interval is VISIBLE, never `null`
//!
//! Ten seeds that all move by the same amount is not exotic. `paired_ci95_df9` returns
//! [`AprenderError::ZeroVarianceDifferences`] for that input, and this module renders it as
//! [`Ci95`] carrying [`ZERO_VARIANCE_NULL_REASON`] and NO bounds. A non-finite `f64` would be
//! serialized by `serde_json` as `null`, which a reader parses as a MISSING measurement rather
//! than a degenerate one (Ph3 CR-03). Nothing in [`RunAggregate`] serializes to `null`.

use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};

use aprender::error::AprenderError;
use aprender::stats::hypothesis::{
    ci95_one_sample_df9, mean_f64, min_max_f64, paired_ci95_df9, sample_std_f64, ttest_rel_f64,
    PAIRED_DESIGN_N, T_CRIT_975_DF9,
};
use serde::{Deserialize, Serialize};

use super::bench_row::{
    sha256_hex, BenchRow, CellEntry, CellKey, CellStatus, ExpectationScope, Method, MethodEvidence,
    RunManifest, BENCH_METHODS, BENCH_SEEDS, BENCH_SHOTS, CALIBRATION_SPLIT, CLAIMS_CONTRACT_ID,
    EXPECTED_CELLS, MECHANISM_CHILD_MAX_RSS_TIME_L, MECHANISM_CHILD_MAX_RSS_VM_HWM, WARMUP_COUNT,
};
// THE RECOMPUTATION LIVES IN `bench_metrics`, NOT HERE (05-17). This module routes to it and
// compares; it authors no metric arithmetic of its own, for the same reason it holds no second
// definition of the mean.
use super::bench_metrics::quality_from_confusion_matrix;
use super::lock::SelectionRule;

// ===========================================================================================
// Directory layout — ONE spelling, shared with the adapters
// ===========================================================================================

/// Row files live here, relative to the benchmark directory.
pub const ROWS_DIR: &str = "rows";

/// Committed SetFit selection-lock records live here.
pub const LOCKS_DIR: &str = "locks";

/// Append-only LoRA candidate ledgers live here.
pub const LEDGER_DIR: &str = "ledger";

/// Committed selection manifests live here, one per-cell directory beneath.
pub const SELECTIONS_DIR: &str = "selections";

/// The per-cell selection manifest's filename.
pub const SELECTION_MANIFEST_FILE: &str = "selection-manifest.json";

/// The pre-declared expectation set lives here.
pub const RUN_MANIFEST_FILE: &str = "run-manifest.json";

/// The house cap for every evidence file this gate reads.
///
/// The gate never reads a model artifact; everything it opens is a row, a manifest, a lock
/// record or a ledger, all of which are small JSON. The cap is checked from the file's DECLARED
/// length before a byte is read, and again against the stream, so a length that lied is detected
/// rather than silently truncated into a payload that happens to parse.
pub const MAX_EVIDENCE_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// The two `role` values a committed lock record may carry, matching `apr eval`'s `LockRow`.
pub const LOCK_ROLES: [&str; 2] = ["written", "consumed"];

/// The ledger's contracted line count. More than one line IS the model selection the LoRA
/// attestation says did not happen.
pub const CONTRACTED_CANDIDATES_TRAINED: u32 = 1;

/// The value [`Ci95::null_reason`] carries when the paired differences have no variance.
pub const ZERO_VARIANCE_NULL_REASON: &str = "zero_variance";

/// The claims contract, embedded at compile time.
///
/// `include_str!` rather than a runtime read, on `bench_row`'s precedent: a cross-pin test that
/// silently skips when the contract is absent proves nothing.
#[cfg(test)]
pub(crate) const CLAIMS_CONTRACT_YAML: &str =
    include_str!("../../../../../contracts/setfit-benchmark-claims-v1.yaml");

/// This module's own source, for the no-RNG structural guard.
#[cfg(test)]
const BENCH_GATE_SOURCE: &str = include_str!("bench_gate.rs");

/// The row filename grammar, produced by ONE function.
///
/// `{method}-s{shots}-seed{seed}.json`. Two spellings of a filename are two filenames, and this
/// gate compares a manifest-recorded digest against the file this name resolves to — so a drift
/// between the writer's spelling and the reader's would make every cell look un-run while
/// reporting a missing-file error that names a path the writer never used. `apr-cli`'s
/// `setfit_bench::row_file_name` delegates here rather than restating it.
#[must_use]
pub fn row_file_name(cell: CellKey) -> String {
    format!("{}-s{}-seed{}.json", cell.method.tag(), cell.shots, cell.seed)
}

/// The committed selection manifest for one cell — `selections/s{shots}-seed{seed}/…`.
///
/// # THIS SIGNATURE IS THE POINT OF THE WHOLE FIX
///
/// The path is built from the CELL KEY and from nothing the row supplies. Compare it with the
/// lock and ledger paths, which are producer-written strings and therefore have to be dragged
/// through [`resolve_committed_evidence_path`]'s two stages before they may touch the
/// filesystem: there, validation is the only thing standing between a row and any file on the
/// host. Here there is nothing for an attacker to choose. A reviewer can settle that from the
/// signature alone — this function takes a [`CellKey`] and a [`Path`], and no [`BenchRow`] —
/// without reading the body, which is a strictly stronger guarantee than a correct body.
///
/// That is also what makes the RECOMPUTATION meaningful. Reading a manifest at a path the row
/// named and comparing its digest to a hash the same row named would compare a producer's two
/// statements against each other and call the agreement evidence.
///
/// Only the `(shots, seed)` half of the key is used, and deliberately: ONE selection is drawn
/// per `(shots, seed)` and BOTH methods consume it. That sharing IS the pairing design, and it
/// is why a restored second arm inherits this binding instead of needing its own.
///
/// Follows [`row_file_name`]'s precedent — one function produces the spelling, so a writer and
/// a reader cannot drift into two filenames for one file.
#[must_use]
pub fn selection_manifest_path(bench_dir: &Path, cell: CellKey) -> PathBuf {
    bench_dir
        .join(SELECTIONS_DIR)
        .join(format!("s{}-seed{}", cell.shots, cell.seed))
        .join(SELECTION_MANIFEST_FILE)
}

// ===========================================================================================
// Resource mechanism classes (review consensus item 2)
// ===========================================================================================

/// What KIND of number a peak-RSS mechanism produces.
///
/// The renderer needs this at the point of comparison, not in a methods paragraph: a
/// `sysinfo_sampled_*` figure is a LOWER BOUND whose bias is one-directional and whose magnitude
/// depends on the allocation pattern, so it cannot be corrected for and is not comparable to an
/// exact kernel high-water mark. A table that puts one beside the other is arithmetically true
/// and factually misleading, which is the class of defect PF-008 names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MechanismClass {
    /// The kernel's own high-water mark: `child_max_rss_*`, or `vm_hwm` for the training process.
    ExactKernelHighWaterMark,
    /// A poll at a finite rate. Can only UNDERSTATE, never overstate.
    SampledLowerBound,
    /// A mechanism string this build does not know. Never silently treated as comparable.
    Unrecognised,
}

impl MechanismClass {
    /// The stable tag, for rendering.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::ExactKernelHighWaterMark => "exact_kernel_high_water_mark",
            Self::SampledLowerBound => "sampled_lower_bound",
            Self::Unrecognised => "unrecognised",
        }
    }
}

/// Classify a mechanism string.
///
/// `vm_hwm` is the TRAINING process's own `/proc/self/status` high-water mark — an exact kernel
/// figure, taken in-process rather than off a child, which is why it is named separately from
/// the two `child_max_rss_*` strings the contract enumerates.
#[must_use]
pub fn mechanism_class(mechanism: &str) -> MechanismClass {
    if mechanism == MECHANISM_CHILD_MAX_RSS_TIME_L
        || mechanism == MECHANISM_CHILD_MAX_RSS_VM_HWM
        || mechanism == "vm_hwm"
    {
        MechanismClass::ExactKernelHighWaterMark
    } else if mechanism.starts_with("sysinfo_sampled") {
        MechanismClass::SampledLowerBound
    } else {
        MechanismClass::Unrecognised
    }
}

/// Whether two figures measured by these mechanisms may be placed side by side unqualified.
///
/// `Unrecognised` is never comparable to anything, including itself: an unknown mechanism is not
/// evidence that two numbers mean the same thing.
#[must_use]
pub fn mechanisms_are_comparable(left: &str, right: &str) -> bool {
    let (a, b) = (mechanism_class(left), mechanism_class(right));
    a == b && a != MechanismClass::Unrecognised
}

// ===========================================================================================
// Which KIND of evidence file a path names
// ===========================================================================================

/// Which kind of evidence file a path resolution or a bounded read is about.
///
/// Two refusals need this, and both are about honesty of diagnosis rather than about control
/// flow. [`read_evidence`] used to map every `NotFound` to
/// [`BenchGateError::RowFileMissing`] unconditionally, so an absent LOCK RECORD told the
/// operator to restore a ROW file and reported the wrong variant tag to any caller matching on
/// one. And [`resolve_committed_evidence_path`] has to name, in its refusal, which of a row's
/// two path fields left the benchmark directory.
///
/// Deliberately NOT `#[non_exhaustive]`: 05-16 adds a `SelectionManifest` variant, and a
/// non-exhaustive enum would let that addition compile while a `_` arm silently gave the new
/// kind somebody else's diagnosis. Every match on this type must be revisited by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceKind {
    /// The per-cell row file under [`ROWS_DIR`]. Its path is DERIVED by
    /// [`row_file_name`], never supplied by the row.
    Row,
    /// The committed SetFit selection-lock record under [`LOCKS_DIR`], named by
    /// `evidence.setfit.lock.lock_record_path`.
    Lock,
    /// The append-only LoRA candidate ledger under [`LEDGER_DIR`], named by
    /// `evidence.lora.candidate_ledger_path`.
    Ledger,
    /// The committed selection manifest under [`SELECTIONS_DIR`]. Like [`Self::Row`] and
    /// unlike the two above, its path is GATE-DERIVED — by [`selection_manifest_path`], from
    /// the cell key — so it never passes through [`resolve_committed_evidence_path`]. The kind
    /// exists so an ABSENT manifest names its own artifact rather than telling an operator to
    /// restore a row (WR-06's rule, applied to the field 05-16 adds).
    SelectionManifest,
}

impl EvidenceKind {
    /// The stable, human-facing name of this kind, as it appears in a refusal.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::Row => "row",
            Self::Lock => "lock record",
            Self::Ledger => "candidate ledger",
            Self::SelectionManifest => "selection manifest",
        }
    }
}

// ===========================================================================================
// Refusals
// ===========================================================================================

/// Why a benchmark directory is not a publishable run.
///
/// Every variant names the CELL, and every provenance variant also names the FILE whose bytes
/// disagreed — because "which cell" and "which file" are what a reader needs to re-run, and a
/// gate that says only "verification failed" gets its output ignored.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum BenchGateError {
    /// The manifest's envelope digest disagrees with its payload's own bytes.
    ManifestDigestMismatch {
        /// The digest the envelope claims.
        expected: String,
        /// The digest the payload's bytes produce.
        got: String,
    },
    /// The manifest declares no cells. Zero cells is a failure, not a clean run.
    EmptyExpectationSet,
    /// The manifest's cell sequence is not exactly the contract-derived 80 in contract order.
    ExpectationSetMismatch {
        /// How many cells the manifest declares.
        declared: usize,
        /// How many the contract derives.
        expected: usize,
    },
    /// A declared cell is still `pending`, or is `complete` with no recorded digest.
    IncompleteCell {
        /// The offending cell, rendered.
        cell: String,
        /// What the manifest actually records for it.
        detail: String,
    },
    /// The row file for a complete cell is not on disk.
    RowFileMissing {
        /// The offending cell, rendered.
        cell: String,
        /// The path that was expected to hold it.
        path: String,
    },
    /// A ROW-SUPPLIED evidence path names bytes outside the benchmark directory.
    ///
    /// The schema documents `lock_record_path` and `candidate_ledger_path` as RELATIVE to the
    /// benchmark directory. Until this variant existed, nothing enforced that: `Path::join`
    /// discards its base when the argument is absolute and never resolves `..`, so a row could
    /// point the gate at any file on the host and have its provenance "recomputed" from bytes
    /// the benchmark does not commit.
    EvidencePathEscape {
        /// The offending cell, rendered.
        cell: String,
        /// Which kind of evidence file the path was for ([`EvidenceKind::tag`]).
        kind: String,
        /// The declared string, VERBATIM — quoting it is what lets a reader see the shape.
        declared: String,
        /// Which bound it crossed, and how.
        detail: String,
    },
    /// A non-ROW evidence file a row names is not on disk.
    ///
    /// Distinct from [`Self::RowFileMissing`] because the REMEDY is different: a missing row
    /// file is restored or re-run, a missing lock record or candidate ledger is re-run or
    /// restored as that artifact. One unconditional mapping told the operator to restore the
    /// wrong thing and reported a variant tag that named the wrong artifact.
    EvidenceFileMissing {
        /// The offending cell, rendered.
        cell: String,
        /// Which kind of evidence file was absent ([`EvidenceKind::tag`]).
        kind: String,
        /// The path that was expected to hold it.
        path: String,
    },
    /// An evidence file could not be read (permissions, over the cap, a directory).
    EvidenceReadFailed {
        /// The offending cell, rendered.
        cell: String,
        /// The file.
        path: String,
        /// The underlying diagnostic.
        detail: String,
    },
    /// The row's OWN envelope digest disagrees with its payload — the bit-flip signature.
    RowDigestMismatch {
        /// The offending cell, rendered.
        cell: String,
        /// The row file.
        path: String,
        /// The digest the envelope claims.
        expected: String,
        /// The digest the payload's bytes produce.
        got: String,
    },
    /// The row's bytes are not a row this schema can accept — a trimmed block, an unknown
    /// field, a foreign schema version, an uncontracted cell.
    RowSchemaRefused {
        /// The offending cell, rendered.
        cell: String,
        /// The row file.
        path: String,
        /// The library's own typed diagnostic.
        detail: String,
    },
    /// The row file's bytes do not hash to the digest the MANIFEST recorded for that cell.
    ///
    /// Distinct from [`Self::RowDigestMismatch`]: that one says the row contradicts ITSELF, this
    /// one says a self-consistent row was substituted for the one the run recorded.
    RowManifestDigestMismatch {
        /// The offending cell, rendered.
        cell: String,
        /// The row file.
        path: String,
        /// The digest the manifest recorded.
        recorded: String,
        /// The digest the file's bytes produce.
        actual: String,
    },
    /// A row filed under one cell whose payload names another.
    RowSlotMismatch {
        /// The slot it was filed under.
        cell: String,
        /// The row file.
        path: String,
        /// The cell its payload names.
        payload_cell: String,
    },
    /// A `(shots, seed)` pair whose two rows consumed DIFFERENT selection manifests.
    ///
    /// PF-007's incomparable comparison: two methods evaluated on different sampled few-shot
    /// subsets are not two measurements of the same thing, and at 8 examples per class the
    /// subset is a larger source of variation than the method.
    UnpairedSelection {
        /// Examples per class.
        shots: u32,
        /// The seed.
        seed: u32,
        /// The SetFit row's pairing key.
        setfit_hash: String,
        /// The LoRA row's pairing key.
        lora_hash: String,
    },
    /// The row's pairing key disagrees with the selection manifest its own cell committed.
    ///
    /// `selection_manifest_hash` is THE PAIRING KEY: it is the whole remaining justification
    /// for keeping EVAL-02 open under D-19, because a future second arm pairs against it.
    /// Until this variant existed the gate opened no manifest at all, so the key was a
    /// 64-character string a producer typed and the forty committed manifests were inert
    /// files — `apr setfit bench report` returned 0 with the entire `selections/` directory
    /// deleted, and returned 0 with a row's key doctored to 64 zeros.
    SelectionManifestMismatch {
        /// The offending cell, rendered.
        cell: String,
        /// The manifest whose bytes were recomputed, at the GATE-DERIVED path.
        path: String,
        /// The digest the ROW claims.
        claimed: String,
        /// The digest the committed manifest's own bytes produce.
        recomputed: String,
    },
    /// A selection manifest that is internally valid but belongs to a DIFFERENT cell.
    ///
    /// Distinct from [`Self::SelectionManifestMismatch`] and NOT reachable by the same
    /// attack: a manifest transplanted from another cell seals correctly and hashes
    /// correctly, so a producer who also doctors the row's key makes the digest comparison
    /// AGREE. The manifest's own `payload.shots_per_class` and `payload.root_seed` are a
    /// second, independent statement of which cell drew it, and that is what refuses here.
    SelectionManifestCellMismatch {
        /// The slot the manifest was found under, rendered.
        cell: String,
        /// The manifest, at the GATE-DERIVED path.
        path: String,
        /// The `shots_per_class` the manifest itself declares.
        manifest_shots: u32,
        /// The `root_seed` the manifest itself declares.
        manifest_seed: u64,
    },
    /// A published quality figure that does not follow from the row's OWN confusion matrix.
    ///
    /// The row records `confusion_matrix` and `ordered_labels` beside every metric it
    /// publishes, and those counts determine `f_avg`, `macro_f1`, `mcc`, the three per-class
    /// vectors and `n_test_rows` IN CLOSED FORM. Until this variant existed the metrics were
    /// numbers a producer typed: verifier spot-check D moved `quality.f_avg` from 0.4579 to
    /// 0.99, repaired the row envelope digest, the manifest's `row_sha256` and the manifest
    /// envelope digest, and `apr setfit bench report` returned 0 with the published mean moving
    /// 0.4746 -> 0.5278.
    ///
    /// Also covers the `_bits` siblings, which are the row stating the same number twice in two
    /// encodings — an internal contradiction no external file is needed to detect.
    QualityCrossCheckMismatch {
        /// The offending cell, rendered.
        cell: String,
        /// Which published field disagreed, indexed where it is a vector element.
        field: String,
        /// What the row publishes.
        claimed: String,
        /// What the row's own confusion matrix produces.
        recomputed: String,
    },
    /// A row field disagrees with the committed file it names — FORGED PROVENANCE.
    ProvenanceMismatch {
        /// The offending cell, rendered.
        cell: String,
        /// The file whose bytes disagreed.
        file: String,
        /// What the row claims.
        claimed: String,
        /// What the file's bytes produce.
        recomputed: String,
        /// Which claim it was.
        detail: String,
    },
    /// A conjunct that forecloses post-test selection does not hold.
    PostTestSelection {
        /// The offending cell, rendered.
        cell: String,
        /// Which conjunct failed.
        conjunct: String,
        /// The observed state.
        detail: String,
    },
}

impl BenchGateError {
    /// A stable discriminant string, so a test can assert that N refusals are N DISTINCT
    /// variants without matching on rendered prose.
    #[must_use]
    pub const fn variant_tag(&self) -> &'static str {
        match self {
            Self::ManifestDigestMismatch { .. } => "manifest_digest_mismatch",
            Self::EmptyExpectationSet => "empty_expectation_set",
            Self::ExpectationSetMismatch { .. } => "expectation_set_mismatch",
            Self::IncompleteCell { .. } => "incomplete_cell",
            Self::RowFileMissing { .. } => "row_file_missing",
            Self::EvidencePathEscape { .. } => "evidence_path_escape",
            Self::EvidenceFileMissing { .. } => "evidence_file_missing",
            Self::EvidenceReadFailed { .. } => "evidence_read_failed",
            Self::RowDigestMismatch { .. } => "row_digest_mismatch",
            Self::RowSchemaRefused { .. } => "row_schema_refused",
            Self::RowManifestDigestMismatch { .. } => "row_manifest_digest_mismatch",
            Self::RowSlotMismatch { .. } => "row_slot_mismatch",
            Self::UnpairedSelection { .. } => "unpaired_selection",
            Self::SelectionManifestMismatch { .. } => "selection_manifest_mismatch",
            Self::SelectionManifestCellMismatch { .. } => "selection_manifest_cell_mismatch",
            Self::QualityCrossCheckMismatch { .. } => "quality_cross_check_mismatch",
            Self::ProvenanceMismatch { .. } => "provenance_mismatch",
            Self::PostTestSelection { .. } => "post_test_selection",
        }
    }

    /// The cell this refusal is about, when it is about one.
    #[must_use]
    pub fn cell(&self) -> Option<&str> {
        match self {
            Self::IncompleteCell { cell, .. }
            | Self::RowFileMissing { cell, .. }
            | Self::EvidencePathEscape { cell, .. }
            | Self::EvidenceFileMissing { cell, .. }
            | Self::EvidenceReadFailed { cell, .. }
            | Self::RowDigestMismatch { cell, .. }
            | Self::RowSchemaRefused { cell, .. }
            | Self::RowManifestDigestMismatch { cell, .. }
            | Self::RowSlotMismatch { cell, .. }
            | Self::SelectionManifestMismatch { cell, .. }
            | Self::SelectionManifestCellMismatch { cell, .. }
            | Self::QualityCrossCheckMismatch { cell, .. }
            | Self::ProvenanceMismatch { cell, .. }
            | Self::PostTestSelection { cell, .. } => Some(cell),
            Self::ManifestDigestMismatch { .. }
            | Self::EmptyExpectationSet
            | Self::ExpectationSetMismatch { .. }
            | Self::UnpairedSelection { .. } => None,
        }
    }
}

impl core::fmt::Display for BenchGateError {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ManifestDigestMismatch { expected, got } => write!(
                f,
                "the run manifest's envelope claims digest {expected} but its payload's own \
                 bytes hash to {got}. These are not the bytes that were attested; re-run the \
                 affected cells rather than editing the manifest",
            ),
            Self::EmptyExpectationSet => write!(
                f,
                "the run manifest declares zero cells. A gate whose input is empty finds no \
                 failures and would report success; zero cells is a failure, not a clean run. \
                 Re-declare the manifest with `apr setfit bench run --bench-dir <DIR>`",
            ),
            Self::ExpectationSetMismatch { declared, expected } => write!(
                f,
                "the run manifest declares {declared} cells, but {CLAIMS_CONTRACT_ID} derives \
                 {expected} in the order (method, shots ascending, seed ascending). \
                 Completeness is defined by the contract-derived set, never by whatever the \
                 manifest happens to list — otherwise a 12-cell expectation could be satisfied \
                 completely and published as a complete run",
            ),
            Self::IncompleteCell { cell, detail } => write!(
                f,
                "cell {cell} is not complete: {detail}. The report has no partial-data mode — \
                 run it with `apr setfit bench run --method {} --shots ... --seed ...`, or, if \
                 it ran on the other host, ingest its row with `--record <ROW_FILE>`",
                cell.split('/').next().unwrap_or("<method>"),
            ),
            Self::RowFileMissing { cell, path } => write!(
                f,
                "cell {cell} is recorded complete in the manifest, but {path} does not exist. \
                 A recorded digest with no bytes behind it is an omission the manifest cannot \
                 see; re-run the cell or restore the row file",
            ),
            Self::EvidencePathEscape { cell, kind, declared, detail } => write!(
                f,
                "cell {cell}: the declared {kind} path `{declared}` leaves the benchmark \
                 directory ({detail}). That field is RELATIVE to the benchmark directory by \
                 contract, and a path that leaves it names bytes the benchmark does not commit \
                 — so recomputing a digest from them would attest to evidence no reader can \
                 audit. Re-run the cell, or repoint the field at the committed file under the \
                 benchmark directory",
            ),
            Self::EvidenceFileMissing { cell, kind, path } => write!(
                f,
                "cell {cell}: the {kind} at {path} does not exist. The row's provenance is \
                 recomputed from that file's bytes, and there are none; re-run the cell or \
                 restore the {kind}",
            ),
            Self::EvidenceReadFailed { cell, path, detail } => {
                write!(f, "cell {cell}: {path} could not be read: {detail}",)
            }
            Self::RowDigestMismatch { cell, path, expected, got } => write!(
                f,
                "cell {cell}: {path} claims digest {expected} but its payload's own bytes hash \
                 to {got}. These are not the bytes that were attested; re-run the cell rather \
                 than editing the row",
            ),
            Self::RowSchemaRefused { cell, path, detail } => {
                write!(f, "cell {cell}: {path} is not a row this schema accepts: {detail}",)
            }
            Self::RowManifestDigestMismatch { cell, path, recorded, actual } => write!(
                f,
                "cell {cell}: the manifest recorded row digest {recorded}, but {path} hashes to \
                 {actual}. The row is internally consistent, so this is a SUBSTITUTION rather \
                 than a corruption: the run that produced the published number is no longer the \
                 run on disk",
            ),
            Self::RowSlotMismatch { cell, path, payload_cell } => write!(
                f,
                "cell {cell}: {path} carries a payload for {payload_cell}. The manifest slot \
                 and the row payload are two independent statements of the same fact; a \
                 disagreement is a refusal, not a relabelling",
            ),
            Self::UnpairedSelection { shots, seed, setfit_hash, lora_hash } => write!(
                f,
                "the (shots {shots}, seed {seed}) pair is NOT comparable: setfit consumed \
                 selection manifest {setfit_hash} and lora consumed {lora_hash}. Two methods \
                 evaluated on different sampled few-shot subsets are not two measurements of \
                 the same thing — at 8 examples per class the subset is a larger source of \
                 variation than the method. Re-run both cells against ONE selection manifest",
            ),
            Self::SelectionManifestMismatch { cell, path, claimed, recomputed } => write!(
                f,
                "cell {cell}: the row's `selection_manifest_hash` is {claimed}, but the \
                 committed selection manifest at {path} hashes to {recomputed}. THE PAIRING \
                 KEY IS RECOMPUTED FROM THE COMMITTED MANIFEST, never read off the row — and \
                 the manifest is opened at a path DERIVED FROM THE CELL KEY, which is what \
                 makes the recomputation mean anything. A row field that disagrees with the \
                 manifest its own cell committed is evidence of nothing. Re-run the cell \
                 rather than editing the row or the manifest",
            ),
            Self::SelectionManifestCellMismatch { cell, path, manifest_shots, manifest_seed } => {
                write!(
                    f,
                    "cell {cell}: the selection manifest at {path} declares \
                     `shots_per_class` = {manifest_shots} and `root_seed` = {manifest_seed}, \
                     which is not this cell. This is DISTINCT from a hash disagreement and is \
                     not reachable by the same attack: a manifest transplanted from another \
                     cell is internally valid and seals correctly, so a producer who also \
                     doctors the row's `selection_manifest_hash` makes the digest comparison \
                     AGREE. The manifest's own declared shots and seed are a second, \
                     independent statement of which cell drew it. Restore this cell's own \
                     manifest, or re-run the cell",
                )
            }
            Self::QualityCrossCheckMismatch { cell, field, claimed, recomputed } => write!(
                f,
                "cell {cell}: the published `quality.{field}` is {claimed}, but the row's OWN \
                 `confusion_matrix` and `ordered_labels` produce {recomputed}. These counts \
                 determine the metric in CLOSED FORM, through the same surfaces that computed \
                 it — so a published figure that disagrees with the matrix printed beside it is \
                 not a measurement. The comparison is exact IEEE-754 bit equality and needs no \
                 tolerance, because the recomputation routes to the same integer-sourced \
                 surfaces rather than re-deriving the arithmetic. Re-run the cell rather than \
                 editing either number",
            ),
            Self::ProvenanceMismatch { cell, file, claimed, recomputed, detail } => write!(
                f,
                "cell {cell}: {detail}. The row claims {claimed}; {file} produces {recomputed}. \
                 Selection safety is RECOMPUTED from committed bytes, never read from a row \
                 field — a field that disagrees with the file it names is evidence of nothing",
            ),
            Self::PostTestSelection { cell, conjunct, detail } => write!(
                f,
                "cell {cell}: the `{conjunct}` conjunct does not hold ({detail}). Each conjunct \
                 closes a different route to a post-hoc choice, and any one alone is satisfiable \
                 while the claim is false. Re-run the cell under the frozen published defaults",
            ),
        }
    }
}

impl std::error::Error for BenchGateError {}

// ===========================================================================================
// The verified set — no public constructor, on purpose
// ===========================================================================================

/// Eighty rows that passed every rule of `setfit-benchmark-claims-v1`, in contract order.
///
/// **There is no public constructor and no public field.** The only way to hold one is to have
/// called [`verify_run`] and had it return `Ok`, which is what makes [`aggregate`]'s signature a
/// proof rather than a convention: the statistics cannot be run over an unverified directory
/// even by a future caller inside this crate who has forgotten why. This is the same typestate
/// argument Phase 3 used for `SetFitRun` and Phase 4 for `ReloadedSetFitCredential` — proving a
/// runtime flag is always set needs whole-program reasoning; proving a value cannot be built
/// needs one look at the type.
#[derive(Debug, Clone)]
pub struct VerifiedRunSet {
    /// Cell-keyed rows in the contract order, exactly [`EXPECTED_CELLS`] of them.
    rows: Vec<(CellKey, BenchRow)>,
}

impl VerifiedRunSet {
    /// The verified rows, in the deterministic contract order.
    #[must_use]
    pub fn rows(&self) -> &[(CellKey, BenchRow)] {
        &self.rows
    }

    /// How many rows were verified. Always [`EXPECTED_CELLS`] for a value that exists.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Never true for a value that exists; present so `len` does not trip clippy's pairing lint.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The row for a cell, if the set holds one.
    #[must_use]
    pub fn row(&self, cell: CellKey) -> Option<&BenchRow> {
        self.rows.iter().find(|(key, _)| *key == cell).map(|(_, row)| row)
    }
}

// ===========================================================================================
// Bounded evidence reads
// ===========================================================================================

/// THE ONE DOOR a row-supplied path may reach the filesystem through.
///
/// `declared` is attacker-controlled in this gate's own threat model: the report's header says a
/// doctored cell would have been refused, so the adversary is a producer who writes rows. The
/// schema documents both path fields as RELATIVE to the benchmark directory, and until this
/// function existed that was an assertion with nothing behind it.
///
/// # Two stages, and the ORDER is the property
///
/// 1. **Syntactic, total, filesystem-independent.** Empty or whitespace-only, absolute, or
///    carrying a `ParentDir` / `RootDir` / `Prefix` component — refused before any `fs` call.
///    Running first is what stops a "that file does not exist" answer from MASKING an escape:
///    the verdict is the same whether or not the target happens to be present.
///    [`Component::CurDir`] is ALLOWED, because `./locks/x.json` names the same file as
///    `locks/x.json` and refusing it would be over-refusal, not safety.
/// 2. **Canonical containment.** Both the joined target and `bench_dir` are canonicalized and
///    compared with [`Path::starts_with`], which is COMPONENT-wise. `str::starts_with` would
///    accept `<tmp>/bench-evil/x.json` under a `bench_dir` of `<tmp>/bench`. Canonicalizing
///    the BASE too is not optional: on macOS `/tmp` is itself a symlink to `/private/tmp`, so
///    comparing a canonical target against a raw base would reject every legitimate path under
///    a temp directory. This stage is the only one that can see a SYMLINK, which stage 1
///    cannot.
///
/// The value returned is the JOINED path, not the canonical one, so refusals downstream keep
/// naming the file by the spelling the operator sees on disk.
///
/// # The residual, stated rather than hidden
///
/// Containment is checked and then the file is opened, so a symlink swapped between the two
/// calls would be read. That race needs write access to the benchmark directory DURING the
/// verification, which is strictly more than the producer-written-tree threat this gate is
/// built for; it is recorded here rather than left for a reader to discover.
///
/// # Errors
///
/// [`BenchGateError::EvidencePathEscape`] when the declared path leaves the benchmark
/// directory by either stage; [`BenchGateError::EvidenceFileMissing`] (or
/// [`BenchGateError::RowFileMissing`] for [`EvidenceKind::Row`]) when the target is absent —
/// an absent file is NOT an escape and must not be reported as one;
/// [`BenchGateError::EvidenceReadFailed`] for any other canonicalization failure.
fn resolve_committed_evidence_path(
    cell: CellKey,
    bench_dir: &Path,
    kind: EvidenceKind,
    declared: &str,
) -> Result<PathBuf, BenchGateError> {
    let escape = |detail: &str| BenchGateError::EvidencePathEscape {
        cell: cell.render(),
        kind: kind.tag().to_string(),
        declared: declared.to_string(),
        detail: detail.to_string(),
    };

    // ---- STAGE 1: SYNTACTIC, TOTAL, FILESYSTEM-INDEPENDENT --------------------------------
    if declared.trim().is_empty() {
        return Err(escape(
            "it is empty or whitespace-only, so it names no file at all and would resolve to \
             the benchmark directory itself",
        ));
    }
    let relative = Path::new(declared);
    if relative.is_absolute() {
        return Err(escape(
            "it is ABSOLUTE, and `Path::join` discards its base when the argument is absolute",
        ));
    }
    for component in relative.components() {
        match component {
            Component::ParentDir => {
                return Err(escape(
                    "it carries a `..` component, which `Path::join` never resolves — the climb \
                     happens at open time, after this gate has stopped looking",
                ))
            }
            Component::RootDir => return Err(escape("it carries a root component")),
            Component::Prefix(_) => return Err(escape("it carries a path prefix")),
            Component::CurDir | Component::Normal(_) => {}
        }
    }

    // ---- STAGE 2: CANONICAL CONTAINMENT ---------------------------------------------------
    let joined = bench_dir.join(relative);
    let canonical_base =
        fs::canonicalize(bench_dir).map_err(|error| BenchGateError::EvidenceReadFailed {
            cell: cell.render(),
            path: bench_dir.display().to_string(),
            detail: format!("the benchmark directory could not be canonicalized: {error}"),
        })?;
    let canonical_target = match fs::canonicalize(&joined) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Err(missing_evidence(cell, kind, &joined))
        }
        Err(error) => {
            return Err(BenchGateError::EvidenceReadFailed {
                cell: cell.render(),
                path: joined.display().to_string(),
                detail: error.to_string(),
            })
        }
    };
    if !canonical_target.starts_with(&canonical_base) {
        return Err(escape(&format!(
            "it resolves to {}, which is outside the benchmark directory {}",
            canonical_target.display(),
            canonical_base.display()
        )));
    }
    Ok(joined)
}

/// The ONE mapping from "this evidence file is absent" to a refusal that names its own kind.
///
/// Shared by [`read_evidence`] and [`resolve_committed_evidence_path`] so the two cannot
/// disagree about which artifact an operator is told to restore.
fn missing_evidence(cell: CellKey, kind: EvidenceKind, path: &Path) -> BenchGateError {
    match kind {
        EvidenceKind::Row => {
            BenchGateError::RowFileMissing { cell: cell.render(), path: path.display().to_string() }
        }
        // The SELECTION MANIFEST joins the two producer-named kinds here rather than the ROW,
        // even though its path is gate-derived like the row's: the REMEDY is what this match
        // selects (WR-06), and an absent manifest is restored or re-selected as a manifest,
        // never "restored as the row file".
        EvidenceKind::Lock | EvidenceKind::Ledger | EvidenceKind::SelectionManifest => {
            BenchGateError::EvidenceFileMissing {
                cell: cell.render(),
                kind: kind.tag().to_string(),
                path: path.display().to_string(),
            }
        }
    }
}

/// Read a small evidence file, refused from its DECLARED length before a byte is read.
fn read_evidence(
    cell: CellKey,
    kind: EvidenceKind,
    path: &Path,
) -> Result<Vec<u8>, BenchGateError> {
    let metadata = fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            missing_evidence(cell, kind, path)
        } else {
            BenchGateError::EvidenceReadFailed {
                cell: cell.render(),
                path: path.display().to_string(),
                detail: error.to_string(),
            }
        }
    })?;
    if !metadata.is_file() {
        return Err(BenchGateError::EvidenceReadFailed {
            cell: cell.render(),
            path: path.display().to_string(),
            detail: "not a regular file".to_string(),
        });
    }
    if metadata.len() > MAX_EVIDENCE_FILE_BYTES {
        return Err(BenchGateError::EvidenceReadFailed {
            cell: cell.render(),
            path: path.display().to_string(),
            detail: format!(
                "declares {} bytes against the evidence cap of {MAX_EVIDENCE_FILE_BYTES}; \
                 refused from its declared length, before a byte is read",
                metadata.len()
            ),
        });
    }
    let file = fs::File::open(path).map_err(|error| BenchGateError::EvidenceReadFailed {
        cell: cell.render(),
        path: path.display().to_string(),
        detail: error.to_string(),
    })?;
    let mut bytes = Vec::new();
    // `+ 1` so a length that LIED is detected rather than silently truncated into a payload
    // that happens to parse.
    file.take(MAX_EVIDENCE_FILE_BYTES + 1).read_to_end(&mut bytes).map_err(|error| {
        BenchGateError::EvidenceReadFailed {
            cell: cell.render(),
            path: path.display().to_string(),
            detail: error.to_string(),
        }
    })?;
    if bytes.len() as u64 > MAX_EVIDENCE_FILE_BYTES {
        return Err(BenchGateError::EvidenceReadFailed {
            cell: cell.render(),
            path: path.display().to_string(),
            detail: "the stream exceeded the cap, so its declared length lied".to_string(),
        });
    }
    Ok(bytes)
}

// ===========================================================================================
// verify_run — the boundary guard
// ===========================================================================================

/// Verify a benchmark directory against `setfit-benchmark-claims-v1`, in the contract's order.
///
/// `bench_dir` is the directory holding [`RUN_MANIFEST_FILE`], [`ROWS_DIR`], [`LOCKS_DIR`] and
/// [`LEDGER_DIR`]. It is the whole directory rather than just the rows directory because a row's
/// `lock_record_path` and `candidate_ledger_path` are RELATIVE TO THE BENCHMARK DIRECTORY —
/// a rows-only parameter could not resolve the very files this gate recomputes from.
///
/// That relativity is ENFORCED, not merely documented: both fields are resolved through
/// [`resolve_committed_evidence_path`], which refuses a declaration that is absolute, rooted,
/// empty or carries `..` before touching the filesystem, and then refuses any target whose
/// canonical form escapes the canonical `bench_dir` — which is the only stage that can see a
/// symlink. A declaration that leaves the tree is
/// [`BenchGateError::EvidencePathEscape`], naming the cell, the kind and the declared string
/// verbatim. This paragraph used to assert the convention while nothing checked it, and a run
/// pointing at `/tmp` exited 0 while printing that provenance had been recomputed from the
/// committed bytes.
///
/// # Errors
///
/// One [`BenchGateError`] naming the FIRST rule that failed and the cell (and, for provenance,
/// the file) it failed on. The walk stops there: aggregating over partially-verified rows would
/// produce a number, and a number is what a reader takes away.
pub fn verify_run(
    manifest: &RunManifest,
    bench_dir: &Path,
) -> Result<VerifiedRunSet, BenchGateError> {
    // TWO ARGUMENTS, AND NO SCOPE ARGUMENT (T-05-11-07). The public door admits no way to
    // widen the expectation set: it delegates with the ACTIVE scope and nothing else. The
    // scoped form below is `pub(crate)` and its only other caller is a deferred-scope test,
    // because `ExpectationScope::DeferredTwoMethod` is itself `#[cfg(test)]`-gated.
    verify_run_scoped(manifest, bench_dir, ExpectationScope::Active)
}

/// [`verify_run`] against an explicit scope.
///
/// Identical logic; the scope only chooses which expectation set step 2 compares against.
/// One implementation for both scopes (OPS-03) — a separate deferred-scope verifier would be
/// a second definition of the seven refusals and would drift from this one.
pub(crate) fn verify_run_scoped(
    manifest: &RunManifest,
    bench_dir: &Path,
    scope: ExpectationScope,
) -> Result<VerifiedRunSet, BenchGateError> {
    // ---- 1. THE MANIFEST'S OWN DIGEST -----------------------------------------------------
    // Recomputed here even though `RunManifest::from_bytes` already checked it, because a
    // manifest can also be built in memory (`declare()` + `record()`), and this gate must not
    // depend on which door its argument came through.
    verify_manifest_digest(manifest)?;
    verify_manifest_contract(manifest)?;

    // ---- 2. THE EXPECTATION SET, BEFORE ANY ROW BYTE IS READ -------------------------------
    // The two vacuity backstops. Without them a producer can declare almost nothing, satisfy it
    // completely, and publish a "complete" run — and a gate with no input reports success.
    if manifest.payload.cells.is_empty() {
        return Err(BenchGateError::EmptyExpectationSet);
    }
    let expectation = RunManifest::expectation_for(scope);
    let declared: Vec<CellKey> = manifest.payload.cells.iter().map(|e| e.cell()).collect();
    if declared != expectation {
        // THIS IS ALSO THE OUT-OF-SCOPE REFUSAL, and deliberately so. A manifest DECLARING a
        // cell for a method outside the active scope differs from the expectation set, so it
        // is refused HERE — before any row byte is read — by the backstop that already
        // existed. No new variant is minted for it: a variant reachable only from a test-only
        // constructor would be a guard over a path production cannot take.
        return Err(BenchGateError::ExpectationSetMismatch {
            declared: declared.len(),
            expected: expectation.len(),
        });
    }

    // ---- 3. EVERY EXPECTED CELL IS COMPLETE -----------------------------------------------
    // Also before any row byte is read: a `pending` entry in a pre-declared table is what makes
    // a selectively omitted cell VISIBLE, and reading rows first would report a file-level
    // symptom for a run-level omission.
    for entry in &manifest.payload.cells {
        verify_entry_complete(entry)?;
    }

    // ---- 4. PER CELL: FILE, SCHEMA + ENVELOPE DIGEST, MANIFEST DIGEST, SLOT ----------------
    // The loop BODY is `verify_row_evidence`, shared verbatim with the single-cell door
    // (OPS-03). Nothing moved between passes: it performs exactly the checks this body always
    // performed, in exactly that order, so the order in which refusals fire across rows is
    // unchanged.
    let rows_dir = bench_dir.join(ROWS_DIR);
    let mut rows: Vec<(CellKey, BenchRow)> = Vec::with_capacity(expectation.len());
    for entry in &manifest.payload.cells {
        let cell = entry.cell();
        rows.push((cell, verify_row_evidence(entry, &rows_dir, cell)?));
    }

    // ---- 5. PAIRING ------------------------------------------------------------------------
    verify_pairing(&rows)?;

    // ---- 6. PROVENANCE, RECOMPUTED FROM COMMITTED BYTES ------------------------------------
    for (cell, row) in &rows {
        verify_provenance(*cell, row, bench_dir)?;
    }

    // ---- 6b. THE SELECTION BINDING, RECOMPUTED FROM COMMITTED BYTES ------------------------
    // ITS OWN LOOP, AND IN STEP 6 RATHER THAN EARLIER. Step 5 is `verify_pairing`, the
    // DEFERRED cross-method rule; moving the binding above it would pre-empt the
    // deferred-scope `unpaired_selection` negative and silently change which variant that
    // negative observes, leaving it green while it had stopped testing what it names.
    for (cell, row) in &rows {
        verify_selection_binding(*cell, row, bench_dir)?;
    }

    // ---- 6c. THE PUBLISHED QUALITY, RECOMPUTED FROM THE ROW'S OWN CONFUSION MATRIX ---------
    // ITS OWN LOOP, AND AFTER 6b for the same reason 6b came after step 5: a check placed
    // earlier pre-empts the negatives that prove the later ones, leaving them green while they
    // had stopped testing what they name. This one reads NO file — the counts are already in
    // the row — so it is last among the step-6 recomputations by that rule alone.
    for (cell, row) in &rows {
        verify_quality_closed_form(*cell, row, bench_dir)?;
    }

    // ---- 7. THE LoRA NO-SELECTION ATTESTATION ----------------------------------------------
    for (cell, row) in &rows {
        if let MethodEvidence::Lora(evidence) = &row.payload.evidence {
            verify_lora_attestation(*cell, evidence)?;
        }
    }

    Ok(VerifiedRunSet { rows })
}

/// Step 4's loop body, as ONE named function — `verify_run`'s step-4 loop and the single-cell
/// door both call it, so there is exactly one definition of "is this row's evidence valid".
///
/// Performs, IN THIS ORDER: file present, `from_bytes` (schema, then envelope digest),
/// manifest-digest agreement, slot agreement. Returns the PARSED row, because both callers
/// need the value — `verify_run` accumulates it for steps 5 through 7, and the door hands it
/// to [`verify_provenance`].
///
/// EQUIVALENCE IS PROVEN BEHAVIOURALLY, not structurally. `bench_gate_tests.rs` feeds a table
/// of per-row defects to BOTH entry points and asserts the same variant tag comes back from
/// each. A test that merely asserted the door calls this function would restate the door's own
/// definition and could never go red.
///
/// # Errors
///
/// [`BenchGateError::RowFileMissing`], [`BenchGateError::EvidenceReadFailed`],
/// [`BenchGateError::RowSchemaRefused`], [`BenchGateError::RowDigestMismatch`],
/// [`BenchGateError::RowManifestDigestMismatch`] or [`BenchGateError::RowSlotMismatch`].
fn verify_row_evidence(
    entry: &CellEntry,
    rows_dir: &Path,
    cell: CellKey,
) -> Result<BenchRow, BenchGateError> {
    // The row's path is GATE-DERIVED — `row_file_name` builds it from the cell key — so it
    // never goes through `resolve_committed_evidence_path`, which exists for the strings a ROW
    // supplies. The kind is passed so an absent row file still reports `row_file_missing`.
    let path = rows_dir.join(row_file_name(cell));
    let bytes = read_evidence(cell, EvidenceKind::Row, &path)?;

    // `from_bytes` is the ONE door: it parses (schema) and then recomputes the envelope
    // digest, and it returns nothing on either failure. The two outcomes are separated here
    // because they are different defects with the same symptom — a trimmed block is an
    // omission, a digest mismatch is tampering.
    // Match the variant directly rather than dispatching on `variant_tag()` and
    // then re-matching: the string compare discarded the type information the
    // second match needed back, which forced an unreachable arm to stay total.
    // Matching once makes that arm impossible to write, and moves `expected`/
    // `got` out of the owned error instead of cloning them.
    let row = BenchRow::from_bytes(&bytes).map_err(|error| match error {
        super::bench_row::BenchRowError::SemanticHashMismatch { expected, got } => {
            BenchGateError::RowDigestMismatch {
                cell: cell.render(),
                path: path.display().to_string(),
                expected,
                got,
            }
        }
        other => BenchGateError::RowSchemaRefused {
            cell: cell.render(),
            path: path.display().to_string(),
            detail: other.to_string(),
        },
    })?;

    let recorded = entry.row_sha256.clone().unwrap_or_default();
    // WHICH DIGEST THE MANIFEST HOLDS, stated once. `emit_row` records `row.semantic_hash`
    // — the digest over the payload's CANONICAL COMPACT bytes — not a digest over the file.
    // That is deliberate and it is `bench_row_schema`'s own invariant: the file is PRETTY so
    // rows can be reviewed in diffs, and the digest is canonical so whitespace cannot change
    // a row's identity. Comparing the file's bytes here would refuse a byte-identical row
    // that had been reformatted, and would call it tampering.
    //
    // The check is not weakened by that: `from_bytes` above has already proven
    // `semantic_hash == sha256(canonical(payload))`, so a row whose payload differs at all
    // carries a different `semantic_hash` and is caught here.
    if row.semantic_hash != recorded {
        return Err(BenchGateError::RowManifestDigestMismatch {
            cell: cell.render(),
            path: path.display().to_string(),
            recorded,
            actual: row.semantic_hash.clone(),
        });
    }

    if row.payload.cell() != cell {
        return Err(BenchGateError::RowSlotMismatch {
            cell: cell.render(),
            path: path.display().to_string(),
            payload_cell: row.payload.cell().render(),
        });
    }

    // LAST, so every refusal that fired before this plan still fires in the same order.
    verify_contracted_row_constants(cell, &path, &row)?;

    Ok(row)
}

/// The contract-derived constants a row DECLARES, compared rather than carried.
///
/// `05-15-gate-input-surface.md` enumerated the gate's whole row-supplied input surface and
/// found four fields that the schema accepts as free-form values while the contract pins them
/// to one value each. Every one was checked only at EMISSION time — by `bench_metrics` or by
/// the CLI writer — and never when the gate READ a committed row, so a hand-edited row could
/// declare any of them and be published.
///
/// `contract_id` is the one that matters most and the one the enumeration existed to find:
/// nothing anywhere compared a row's declared contract against [`CLAIMS_CONTRACT_ID`], while
/// [`aggregate`] stamped the published payload with that constant regardless — so the
/// aggregate would assert a contract its inputs never claimed.
///
/// `throughput_batch_size` is deliberately NOT checked here: the contract pins no value for it
/// ("the batch size that pass used"), and the writer's constant is a producer choice. Comparing
/// against it would refuse a legitimately different batch size while calling it a contract
/// violation. The reasoning is recorded in the enumeration artifact rather than only here.
///
/// No new variant is minted. A row declaring a value the contract forbids is exactly "not a row
/// this schema accepts", which [`BenchGateError::RowSchemaRefused`] already says, and every
/// refusal names BOTH the declared value and the expected one.
///
/// # Errors
///
/// [`BenchGateError::RowSchemaRefused`].
fn verify_contracted_row_constants(
    cell: CellKey,
    path: &Path,
    row: &BenchRow,
) -> Result<(), BenchGateError> {
    let refuse = |detail: String| BenchGateError::RowSchemaRefused {
        cell: cell.render(),
        path: path.display().to_string(),
        detail,
    };

    if row.payload.contract_id != CLAIMS_CONTRACT_ID {
        return Err(refuse(format!(
            "the row declares contract `{}`, but this build verifies against \
             `{CLAIMS_CONTRACT_ID}`. A row written against a foreign contract is not a row this \
             schema accepts, and publishing it would stamp the aggregate with a contract its \
             inputs never claimed",
            row.payload.contract_id
        )));
    }
    if row.payload.quality.calibration_split != CALIBRATION_SPLIT {
        return Err(refuse(format!(
            "the row records `calibration_split` = `{}`, but the contract measures calibration \
             on `{CALIBRATION_SPLIT}` and only there (D-07). Diagnostics from another split are \
             a different measurement wearing this one's name",
            row.payload.quality.calibration_split
        )));
    }
    if row.payload.resource.warmup_count != WARMUP_COUNT {
        return Err(refuse(format!(
            "the row records `warmup_count` = {}, but the contract discards exactly \
             {WARMUP_COUNT} warmup classifies before the warm measurement. A different warmup \
             count produces a warm latency that is not comparable to the published ones",
            row.payload.resource.warmup_count
        )));
    }
    if !row.payload.resource.cold_measured_in_child_process {
        return Err(refuse(
            "the row records `cold_measured_in_child_process` = false. A cold latency measured \
             in the process that just finished training is operationally WARM — populated \
             caches, resident pages — and the contract pins this field to the literal true"
                .to_string(),
        ));
    }
    Ok(())
}

/// The manifest's own declared contract, compared against [`CLAIMS_CONTRACT_ID`].
///
/// The manifest half of the third enumerated field. Called by BOTH doors right after
/// [`verify_manifest_digest`], because a manifest can be built in memory as well as parsed and
/// no door may depend on which constructor its argument came through.
///
/// Reuses [`BenchGateError::RowSchemaRefused`] rather than minting a variant, matching the row
/// side so one defect has one tag. The `cell` field carries the sentinel `<run manifest>`: this
/// refusal is about the run-level artifact and not about any one cell, and saying so in the
/// field a reader already looks at is more honest than leaving a real cell key there.
///
/// # Errors
///
/// [`BenchGateError::RowSchemaRefused`].
fn verify_manifest_contract(manifest: &RunManifest) -> Result<(), BenchGateError> {
    if manifest.payload.contract_id != CLAIMS_CONTRACT_ID {
        return Err(BenchGateError::RowSchemaRefused {
            cell: "<run manifest>".to_string(),
            path: RUN_MANIFEST_FILE.to_string(),
            detail: format!(
                "the run manifest declares contract `{}`, but this build verifies against \
                 `{CLAIMS_CONTRACT_ID}`. The expectation set is DERIVED from the contract, so a \
                 manifest naming another one declares completeness against a definition this \
                 build does not implement",
                manifest.payload.contract_id
            ),
        });
    }
    Ok(())
}

/// Step 1, as a named function: the manifest's digest over its own canonical bytes.
///
/// Recomputed even though `RunManifest::from_bytes` already checked it, because a manifest can
/// also be built in memory (`declare()` + `record()`), and no door may depend on which
/// constructor its argument came through.
///
/// # Errors
///
/// [`BenchGateError::ManifestDigestMismatch`].
fn verify_manifest_digest(manifest: &RunManifest) -> Result<(), BenchGateError> {
    let recomputed = manifest
        .payload
        .to_canonical_bytes()
        .map_or_else(|_| String::new(), |bytes| sha256_hex(&bytes));
    if recomputed != manifest.semantic_hash {
        return Err(BenchGateError::ManifestDigestMismatch {
            expected: manifest.semantic_hash.clone(),
            got: recomputed,
        });
    }
    Ok(())
}

/// Step 3's PER-ENTRY rule: this entry is `Complete` and carries a non-empty row digest.
///
/// `verify_run` applies it to every entry (the sweep); the single-cell door applies it to its
/// one declared cell and to no other, which is exactly what lets the door pass at pilot time
/// when the other 39 entries are still `pending`.
///
/// # Errors
///
/// [`BenchGateError::IncompleteCell`].
fn verify_entry_complete(entry: &CellEntry) -> Result<(), BenchGateError> {
    match (entry.status, entry.row_sha256.as_deref()) {
        (CellStatus::Complete, Some(digest)) if !digest.is_empty() => Ok(()),
        (CellStatus::Complete, _) => Err(BenchGateError::IncompleteCell {
            cell: entry.cell().render(),
            detail: "the manifest marks it complete but records no row digest".to_string(),
        }),
        (CellStatus::Pending, _) => Err(BenchGateError::IncompleteCell {
            cell: entry.cell().render(),
            detail: "the manifest still lists it as `pending`, so this run is not \
                     complete and no aggregate over it is publishable"
                .to_string(),
        }),
    }
}

/// THE SINGLE-CELL VERIFICATION DOOR — `verify_run`'s steps 1 + 4 + 6, over ONE declared cell.
///
/// # Which steps it applies, and which it deliberately does not
///
/// APPLIES, in `verify_run`'s order:
/// 1. the manifest's own digest ([`verify_manifest_digest`]);
/// 3'. step 3's per-entry rule, TO THIS ONE ENTRY ONLY ([`verify_entry_complete`]);
/// 4. the row's file, schema, envelope digest, manifest-digest agreement and slot agreement
///    ([`verify_row_evidence`]);
/// 6. provenance recomputed from committed bytes ([`verify_provenance`]); the selection
///    binding recomputed from the committed manifest at a gate-derived path
///    ([`verify_selection_binding`]) — the row's `selection_manifest_hash` against the
///    recomputed `semantic_hash`, and the manifest's own declared shots and seed against this
///    cell's key; AND the published quality recomputed in closed form from the row's own
///    confusion matrix ([`verify_quality_closed_form`]).
///
/// DOES NOT APPLY: step 2 (expectation-set equality), step 3's SWEEP over every entry, step 5
/// (pairing) and step 7 (the second method's attestation).
///
/// # Why the set-level steps are excluded — this is the whole point of the door
///
/// At pilot time the manifest declares 40 cells with 39 still `pending`, which is PRECISELY
/// the state step 3's sweep refuses on. A door that inherited the set-level checks could
/// therefore never pass on the one cell it exists to check, and `bench report` over a copy
/// holding a single row refuses at completeness — BEFORE the row loop — so the pilot row's own
/// bytes are never read at all. This door reads them.
///
/// # It emits no statistic, and cannot
///
/// The return type is `()`, not the row and not an aggregate: no mean, no dispersion, no
/// interval, nothing a reader could mistake for a partial report (T-05-11-06). The report is
/// the only door that emits numbers. What this validates is exactly what those two stages
/// validate and no more — file present, schema parse, envelope digest, manifest-digest
/// agreement, slot agreement, the recomputed lock digest with its role and rule, the
/// selection binding recomputed from this cell's own committed manifest, and every published
/// QUALITY figure recomputed in closed form from the row's own confusion matrix. The
/// resource and size FIELDS are serde fields whose presence and parse are covered by
/// `BenchRow::from_bytes`; this door does not independently validate their values. The quality
/// fields ARE independently validated, and the two calibration diagnostics are the exception
/// inside that exception: their `_bits` siblings are held against the `f64` beside them, but
/// their VALUES are not recomputable from any committed file and are not claimed to be.
///
/// # Errors
///
/// [`BenchGateError::ExpectationSetMismatch`] if `cell` is not in the ACTIVE expectation set —
/// the EXISTING variant, because asking for a cell the contract does not declare is exactly an
/// expectation-set disagreement and no new variant is minted for it. Otherwise whatever
/// [`verify_manifest_digest`], [`verify_entry_complete`], [`verify_row_evidence`] or
/// [`verify_provenance`] returns, with the SAME variant tag `verify_run` would have produced
/// for the same defect.
pub fn verify_cell(
    manifest: &RunManifest,
    bench_dir: &Path,
    cell: CellKey,
) -> Result<(), BenchGateError> {
    // ---- 1. THE MANIFEST'S OWN DIGEST -----------------------------------------------------
    verify_manifest_digest(manifest)?;
    verify_manifest_contract(manifest)?;

    // ---- THE CELL MUST BE ONE THE CONTRACT DECLARES ---------------------------------------
    // Not step 2: this does NOT compare the manifest's whole cell list against the
    // expectation, which is what would refuse a 39-pending pilot manifest. It only refuses a
    // REQUEST for a cell outside the active scope — including a second method's cell.
    let Some(entry) = manifest.payload.cells.iter().find(|e| e.cell() == cell) else {
        return Err(BenchGateError::ExpectationSetMismatch {
            declared: 0,
            expected: EXPECTED_CELLS,
        });
    };

    // ---- 3'. STEP 3's PER-ENTRY RULE, FOR THIS ENTRY AND NO OTHER -------------------------
    // The status of every other entry is ignored, which is what lets this pass at pilot time.
    verify_entry_complete(entry)?;

    // ---- 4. THE ROW'S OWN EVIDENCE --------------------------------------------------------
    let rows_dir = bench_dir.join(ROWS_DIR);
    let row = verify_row_evidence(entry, &rows_dir, cell)?;

    // ---- 6. PROVENANCE, RECOMPUTED FROM COMMITTED BYTES ------------------------------------
    verify_provenance(cell, &row, bench_dir)?;

    // ---- 6b. THE SELECTION BINDING, RECOMPUTED FROM COMMITTED BYTES ------------------------
    // The single-cell door performs this too. A door whose own doc enumerates its coverage and
    // is WRONG about it is the artifact this phase exists to prevent, so the enumeration above
    // names it and this call is what makes the enumeration true.
    verify_selection_binding(cell, &row, bench_dir)?;

    // ---- 6c. THE PUBLISHED QUALITY, RECOMPUTED FROM THE ROW'S OWN CONFUSION MATRIX ---------
    // The single-cell door performs this too, and the enumeration above names it. A door whose
    // own printed scope under-describes what it enforces teaches a reader to trust it less than
    // the evidence warrants, which is the same defect as over-claiming with the sign flipped.
    verify_quality_closed_form(cell, &row, bench_dir)?;

    // Deliberately nothing returned. See the module note above.
    Ok(())
}

/// Every `(shots, seed)` pair's two rows must carry an identical `selection_manifest_hash`.
fn verify_pairing(rows: &[(CellKey, BenchRow)]) -> Result<(), BenchGateError> {
    let mut by_cell: BTreeMap<CellKey, &str> = BTreeMap::new();
    for (cell, row) in rows {
        by_cell.insert(*cell, row.payload.selection_manifest_hash.as_str());
    }
    // Iterated in the contract order rather than over the map, so the FIRST mismatch reported
    // is deterministic across runs.
    for shots in BENCH_SHOTS {
        for seed in BENCH_SEEDS {
            let setfit = by_cell.get(&CellKey::new(Method::Setfit, shots, seed));
            let lora = by_cell.get(&CellKey::new(Method::Lora, shots, seed));
            if let (Some(setfit_hash), Some(lora_hash)) = (setfit, lora) {
                if setfit_hash != lora_hash {
                    return Err(BenchGateError::UnpairedSelection {
                        shots,
                        seed,
                        setfit_hash: (*setfit_hash).to_string(),
                        lora_hash: (*lora_hash).to_string(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// One JSONL line of the LoRA candidate ledger, as much of it as this gate reads.
#[derive(Debug, Deserialize)]
struct LedgerLine {
    /// The selection manifest the invocation consumed. Must equal the row's pairing key.
    selection_manifest_hash: String,
}

/// Recompute a row's provenance from the committed bytes it names.
fn verify_provenance(
    cell: CellKey,
    row: &BenchRow,
    bench_dir: &Path,
) -> Result<(), BenchGateError> {
    match &row.payload.evidence {
        MethodEvidence::Setfit(evidence) => {
            let path = resolve_committed_evidence_path(
                cell,
                bench_dir,
                EvidenceKind::Lock,
                &evidence.lock.lock_record_path,
            )?;
            let bytes = read_evidence(cell, EvidenceKind::Lock, &path)?;
            let recomputed = sha256_hex(&bytes);
            if recomputed != evidence.lock.lock_hash {
                return Err(BenchGateError::ProvenanceMismatch {
                    cell: cell.render(),
                    file: path.display().to_string(),
                    claimed: evidence.lock.lock_hash.clone(),
                    recomputed,
                    detail: "the committed lock record's bytes do not hash to the row's \
                             `lock.lock_hash`"
                        .to_string(),
                });
            }
            // The lock's ROLE and RULE are what make it a selection lock rather than a file.
            if !LOCK_ROLES.contains(&evidence.lock.role.as_str()) {
                return Err(BenchGateError::PostTestSelection {
                    cell: cell.render(),
                    conjunct: "lock.role".to_string(),
                    detail: format!(
                        "the row records role `{}`; the vocabulary is {LOCK_ROLES:?}",
                        evidence.lock.role
                    ),
                });
            }
            if evidence.lock.rule != SelectionRule::MaxMetricLowestIndexTieBreak.tag() {
                return Err(BenchGateError::PostTestSelection {
                    cell: cell.render(),
                    conjunct: "lock.rule".to_string(),
                    detail: format!(
                        "the row records rule `{}`; this build commits under `{}`, and a lock \
                         written under an unrecognised rule is a selection nothing can replay",
                        evidence.lock.rule,
                        SelectionRule::MaxMetricLowestIndexTieBreak.tag()
                    ),
                });
            }
            Ok(())
        }
        MethodEvidence::Lora(evidence) => {
            // THE SAME HELPER, THE OTHER FIELD. One door for both arms is what makes the
            // deferred two-method scope INHERIT this fix instead of re-opening the hole when
            // `D-ITEM-05-15` restores it.
            let path = resolve_committed_evidence_path(
                cell,
                bench_dir,
                EvidenceKind::Ledger,
                &evidence.candidate_ledger_path,
            )?;
            let bytes = read_evidence(cell, EvidenceKind::Ledger, &path)?;
            let recomputed = sha256_hex(&bytes);
            if recomputed != evidence.candidate_ledger_sha256 {
                return Err(BenchGateError::ProvenanceMismatch {
                    cell: cell.render(),
                    file: path.display().to_string(),
                    claimed: evidence.candidate_ledger_sha256.clone(),
                    recomputed,
                    detail: "the committed candidate ledger's bytes do not hash to the row's \
                             `candidate_ledger_sha256`"
                        .to_string(),
                });
            }

            // THE LINE COUNT IS COUNTED, NOT READ. A ledger carrying a second candidate while
            // the row still claims one is the forgery this recomputation exists to catch.
            let lines: Vec<&[u8]> =
                bytes.split(|byte| *byte == b'\n').filter(|line| !line.is_empty()).collect();
            let counted = u32::try_from(lines.len()).unwrap_or(u32::MAX);
            if counted != evidence.candidates_trained {
                return Err(BenchGateError::ProvenanceMismatch {
                    cell: cell.render(),
                    file: path.display().to_string(),
                    claimed: format!("candidates_trained = {}", evidence.candidates_trained),
                    recomputed: format!("{counted} ledger line(s)"),
                    detail: "the committed candidate ledger's line count disagrees with the \
                             row's `candidates_trained`"
                        .to_string(),
                });
            }

            // Every line must name the SAME selection manifest the row does, so a ledger
            // transplanted from another cell cannot stand in for this one's.
            for (index, line) in lines.iter().enumerate() {
                let parsed: LedgerLine = serde_json::from_slice(line).map_err(|error| {
                    BenchGateError::ProvenanceMismatch {
                        cell: cell.render(),
                        file: path.display().to_string(),
                        claimed: "a JSONL candidate line".to_string(),
                        recomputed: format!("line {} did not parse: {error}", index + 1),
                        detail: "the committed candidate ledger is not the append-only JSONL the \
                                 contract requires"
                            .to_string(),
                    }
                })?;
                if parsed.selection_manifest_hash != row.payload.selection_manifest_hash {
                    return Err(BenchGateError::ProvenanceMismatch {
                        cell: cell.render(),
                        file: path.display().to_string(),
                        claimed: row.payload.selection_manifest_hash.clone(),
                        recomputed: parsed.selection_manifest_hash,
                        detail: format!(
                            "ledger line {} names a different selection manifest than the row",
                            index + 1
                        ),
                    });
                }
            }
            Ok(())
        }
    }
}

/// Recompute the row's PAIRING KEY from the selection manifest its own cell committed.
///
/// # Why this exists — verifier gap 2 (EVAL-02)
///
/// `payload.selection_manifest_hash` is documented in `bench_row` as THE PAIRING KEY, and
/// under D-19 it is the whole remaining justification for keeping EVAL-02 open: the active
/// scope has one method, so the key's only job is to let a FUTURE second arm pair against the
/// same sampled few-shot subset. Until this function existed nothing on the gate path ever
/// opened a selection manifest, so the property held in the DATA by construction and not by
/// enforcement — `apr setfit bench report` returned 0 with the entire `selections/` directory
/// deleted and returned 0 with a row's key doctored to 64 zeros, and a future arm would have
/// paired against a key nothing ever attested.
///
/// # The digest is NOT recomputed here, and that is deliberate
///
/// [`SelectionManifest::from_bytes`](aprender_contrastive_data::manifest::SelectionManifest::from_bytes)
/// verifies `sha256(payload.to_canonical_bytes()) == semantic_hash` BEFORE returning, so a
/// caller cannot hold an unsealed manifest. Recomputing it again in this module would be a
/// SECOND definition of a manifest's identity, and two definitions of one value disagree
/// eventually and invisibly — the same OPS-03 argument the module header makes about not
/// reimplementing a mean.
///
/// # The ORDER of the two comparisons, and why each one needs its own negative
///
/// 1. the recomputed `semantic_hash` against the row's claim — the common case;
/// 2. the manifest's own `payload.shots_per_class` / `payload.root_seed` against the cell key.
///
/// The hash check fires first because it is what an ordinary tamper trips. The cell-key check
/// is what REMAINS reachable once an attacker has already made the hashes agree: a manifest
/// transplanted from another cell seals correctly, so a producer who also doctors the row's
/// key satisfies step 1 completely. Placing the cell-key check first would be equally correct
/// and would simply make step 1 the unreachable one instead — which is why the test table
/// carries a transplant WITH the row hash doctored (reaching step 2) and the same transplant
/// WITHOUT it (reaching step 1), rather than one standing in for both.
///
/// The seed comparison widens the CELL KEY with [`u64::from`] rather than narrowing the
/// manifest: `root_seed` is `u64` and `CellKey::seed` is `u32`, and a `try_from` that failed
/// would silently change the question being asked from "are these the same seed" to "are these
/// the same seed, given it fits".
///
/// # Errors
///
/// [`BenchGateError::EvidenceFileMissing`] carrying [`EvidenceKind::SelectionManifest`] when
/// the manifest is absent; [`BenchGateError::EvidenceReadFailed`] when it cannot be read or
/// parsed; [`BenchGateError::SelectionManifestMismatch`] when the digests disagree (including
/// the manifest's own unsealed-payload refusal); [`BenchGateError::SelectionManifestCellMismatch`]
/// when the manifest belongs to a different cell.
fn verify_selection_binding(
    cell: CellKey,
    row: &BenchRow,
    bench_dir: &Path,
) -> Result<(), BenchGateError> {
    let path = selection_manifest_path(bench_dir, cell);
    let bytes = read_evidence(cell, EvidenceKind::SelectionManifest, &path)?;

    let manifest = aprender_contrastive_data::manifest::SelectionManifest::from_bytes(&bytes)
        .map_err(|error| match error {
            aprender_contrastive_data::ContrastiveDataError::SemanticHashMismatch {
                expected,
                got,
            } => BenchGateError::SelectionManifestMismatch {
                cell: cell.render(),
                path: path.display().to_string(),
                claimed: expected,
                recomputed: got,
            },
            other => BenchGateError::EvidenceReadFailed {
                cell: cell.render(),
                path: path.display().to_string(),
                detail: other.to_string(),
            },
        })?;

    // EXACT BYTE EQUALITY of the two 64-character lowercase hex strings. No truncation, no
    // `eq_ignore_ascii_case`, no prefix match: a key agreeing in its first N characters is a
    // different key, and every relaxation of this line is a way for a doctored row to pass.
    if manifest.semantic_hash != row.payload.selection_manifest_hash {
        return Err(BenchGateError::SelectionManifestMismatch {
            cell: cell.render(),
            path: path.display().to_string(),
            claimed: row.payload.selection_manifest_hash.clone(),
            recomputed: manifest.semantic_hash,
        });
    }

    if manifest.payload.shots_per_class != cell.shots
        || manifest.payload.root_seed != u64::from(cell.seed)
    {
        return Err(BenchGateError::SelectionManifestCellMismatch {
            cell: cell.render(),
            path: path.display().to_string(),
            manifest_shots: manifest.payload.shots_per_class,
            manifest_seed: manifest.payload.root_seed,
        });
    }

    Ok(())
}

/// Recompute a row's published quality from the row's OWN confusion matrix, and compare.
///
/// # This function computes nothing
///
/// It calls [`quality_from_confusion_matrix`], which routes to the same
/// `MultiClassMetrics` / `f1_average_for_classes` / `matthews_corrcoef` surfaces
/// `assemble_quality_block` routed to. A second metric computation HERE would give the phase
/// two definitions of `F_avg`, and two definitions of one number disagree eventually and
/// invisibly (OPS-03) — the same argument this module already makes about a second mean.
///
/// # The band is exact bits, and that was MEASURED before it was chosen
///
/// Over all forty committed rows the recomputation agrees BIT-IDENTICALLY on every field, so
/// the acceptance band is exact IEEE-754 bit equality and there is no epsilon to justify. Every
/// `f64` here is compared through `to_bits()` rather than `==`: the row carries the bits for
/// exactly this reason (the lock hashes bits and a decimal is a rendering), and
/// `clippy::float_cmp` is allowed workspace-wide for ML code, so the lint would not have caught
/// a sloppy comparison.
///
/// # What it does NOT prove
///
/// That the confusion matrix is the one the model produced. A producer who doctors the MATRIX
/// and recomputes the metrics from it emits a consistent forgery, and closing that needs an
/// evidence artifact no run commits today (per-row predictions). The report's `residual:` line
/// says so; see THE RESIDUAL, STATED RATHER THAN HIDDEN in this module's header.
///
/// `bench_dir` is used only so a shape refusal can NAME the row file an operator has to look
/// at, exactly as [`verify_provenance`] and [`verify_selection_binding`] use it.
///
/// # Errors
///
/// [`BenchGateError::QualityCrossCheckMismatch`] when a published figure or a `_bits` sibling
/// disagrees; [`BenchGateError::RowSchemaRefused`] when the matrix is not a shape any metric
/// can be computed from — ragged, mis-dimensioned against `ordered_labels`, totalling zero, or
/// above the recomputation's expansion cap.
fn verify_quality_closed_form(
    cell: CellKey,
    row: &BenchRow,
    bench_dir: &Path,
) -> Result<(), BenchGateError> {
    let quality = &row.payload.quality;
    let recomputed =
        quality_from_confusion_matrix(&quality.confusion_matrix, &quality.ordered_labels).map_err(
            |error| BenchGateError::RowSchemaRefused {
                cell: cell.render(),
                path: bench_dir.join(ROWS_DIR).join(row_file_name(cell)).display().to_string(),
                detail: error.to_string(),
            },
        )?;

    // A FIXED FIELD ORDER, so the first reported disagreement is deterministic across runs and
    // across machines. `n_test_rows` is first because it is the denominator every other number
    // divides by: a reader told "f_avg disagrees" when the row count is wrong would chase the
    // wrong field.
    if quality.n_test_rows != recomputed.n_rows {
        return Err(BenchGateError::QualityCrossCheckMismatch {
            cell: cell.render(),
            field: "n_test_rows".to_string(),
            claimed: quality.n_test_rows.to_string(),
            recomputed: recomputed.n_rows.to_string(),
        });
    }
    cross_check_f64(cell, "f_avg", quality.f_avg, recomputed.f_avg)?;
    cross_check_f64(cell, "macro_f1", quality.macro_f1, recomputed.macro_f1)?;
    cross_check_f64(cell, "mcc", quality.mcc, recomputed.mcc)?;
    cross_check_vector(
        cell,
        "per_class_precision",
        &quality.per_class_precision,
        &recomputed.per_class_precision,
    )?;
    cross_check_vector(
        cell,
        "per_class_recall",
        &quality.per_class_recall,
        &recomputed.per_class_recall,
    )?;
    cross_check_vector(cell, "per_class_f1", &quality.per_class_f1, &recomputed.per_class_f1)?;

    // ---- THE BITS SIBLINGS -----------------------------------------------------------------
    // ALL FIVE, including the two calibration ones whose VALUES no committed file makes
    // recomputable. A bits field is a claim the row makes about ITSELF, so it costs nothing to
    // hold — and holding it is not the same as proving the value, which is a distinction the
    // report's residual disclosure has to keep making.
    for (field, value, claimed_bits) in [
        ("f_avg_bits", quality.f_avg, quality.f_avg_bits),
        ("macro_f1_bits", quality.macro_f1, quality.macro_f1_bits),
        ("mcc_bits", quality.mcc, quality.mcc_bits),
        (
            "ece_top_label_validation_bits",
            quality.ece_top_label_validation,
            quality.ece_top_label_validation_bits,
        ),
        (
            "brier_multiclass_validation_bits",
            quality.brier_multiclass_validation,
            quality.brier_multiclass_validation_bits,
        ),
    ] {
        if claimed_bits != value.to_bits() {
            return Err(BenchGateError::QualityCrossCheckMismatch {
                cell: cell.render(),
                field: field.to_string(),
                claimed: format!("{claimed_bits:#018x}"),
                recomputed: format!("{:#018x} (the bits of the f64 beside it)", value.to_bits()),
            });
        }
    }
    Ok(())
}

/// Compare one published `f64` against its recomputation by IEEE-754 BITS, never by `==`.
fn cross_check_f64(
    cell: CellKey,
    field: &str,
    claimed: f64,
    recomputed: f64,
) -> Result<(), BenchGateError> {
    if claimed.to_bits() == recomputed.to_bits() {
        return Ok(());
    }
    Err(BenchGateError::QualityCrossCheckMismatch {
        cell: cell.render(),
        field: field.to_string(),
        claimed: render_f64(claimed),
        recomputed: render_f64(recomputed),
    })
}

/// Compare one published per-class vector, element by element and by bits.
///
/// A LENGTH disagreement is reported first and as its own message: a vector shorter than the
/// label map means index i of one is not index i of the other, which is a different defect from
/// a wrong value at a known index.
fn cross_check_vector(
    cell: CellKey,
    field: &str,
    claimed: &[f64],
    recomputed: &[f64],
) -> Result<(), BenchGateError> {
    if claimed.len() != recomputed.len() {
        return Err(BenchGateError::QualityCrossCheckMismatch {
            cell: cell.render(),
            field: field.to_string(),
            claimed: format!("{} element(s)", claimed.len()),
            recomputed: format!("{} element(s), one per declared label", recomputed.len()),
        });
    }
    for (index, (left, right)) in claimed.iter().zip(recomputed.iter()).enumerate() {
        cross_check_f64(cell, &format!("{field}[{index}]"), *left, *right)?;
    }
    Ok(())
}

/// An `f64` rendered with BOTH its decimal and its bits, because the two can disagree.
fn render_f64(value: f64) -> String {
    format!("{value:?} (bits {:#018x})", value.to_bits())
}

/// The six conjuncts of `no_selection_attestation`, each closing a different route to a
/// post-hoc choice.
///
/// The digest and the line count are checked in [`verify_provenance`]; the four self-reported
/// conjuncts are checked here, AFTER them, because the recomputed pair is what makes the
/// self-reported four more than a declaration.
fn verify_lora_attestation(
    cell: CellKey,
    evidence: &super::bench_row::LoraEvidence,
) -> Result<(), BenchGateError> {
    if evidence.epochs_completed != evidence.epochs_requested {
        return Err(BenchGateError::PostTestSelection {
            cell: cell.render(),
            conjunct: "epochs_completed == epochs_requested".to_string(),
            detail: format!(
                "requested {}, completed {} — a run that ended somewhere other than where it \
                 was told to ended at a checkpoint somebody chose",
                evidence.epochs_requested, evidence.epochs_completed
            ),
        });
    }
    if !evidence.early_stopping_disabled {
        return Err(BenchGateError::PostTestSelection {
            cell: cell.render(),
            conjunct: "early_stopping_disabled".to_string(),
            detail: "early stopping was enabled, so the run picked a checkpoint".to_string(),
        });
    }
    if evidence.val_split != 0.0 {
        return Err(BenchGateError::PostTestSelection {
            cell: cell.render(),
            conjunct: "val_split == 0.0".to_string(),
            detail: format!(
                "a validation split of {} was carved, which created something to select on",
                evidence.val_split
            ),
        });
    }
    if !evidence.no_selection_attestation {
        return Err(BenchGateError::PostTestSelection {
            cell: cell.render(),
            conjunct: "no_selection_attestation".to_string(),
            detail: "the row does not attest that no selection happened".to_string(),
        });
    }
    if evidence.candidates_trained != CONTRACTED_CANDIDATES_TRAINED {
        return Err(BenchGateError::PostTestSelection {
            cell: cell.render(),
            conjunct: "candidates_trained == 1".to_string(),
            detail: format!(
                "{} candidates were trained for this cell; a second candidate IS the \
                 uncontracted model selection the attestation says did not happen",
                evidence.candidates_trained
            ),
        });
    }
    Ok(())
}

// ===========================================================================================
// The aggregate
// ===========================================================================================

/// Mean, `(n-1)` std, min and max over one series.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeriesSummary {
    /// How many observations. Always [`PAIRED_DESIGN_N`] here.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// SAMPLE standard deviation, `(n-1)` denominator — the SetFit paper's convention, so the
    /// numbers are directly comparable to published results.
    pub std: f64,
    /// The smallest observation.
    pub min: f64,
    /// The largest observation.
    pub max: f64,
}

/// One seed's headline numbers, so a reader can recompute the summary by hand.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeedValue {
    /// The contracted seed.
    pub seed: u32,
    /// Official `F_avg`.
    pub f_avg: f64,
    /// `f_avg`'s IEEE-754 bits, carried so a decimal rendering can never become the compared
    /// value.
    pub f_avg_bits: u64,
    /// Three-class macro F1.
    pub macro_f1: f64,
    /// Matthews correlation coefficient.
    pub mcc: f64,
}

/// The quality summary for one `(method, shots)` group.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodShotQuality {
    /// Which method.
    pub method: Method,
    /// Examples per class.
    pub shots: u32,
    /// The HEADLINE metric's summary.
    pub f_avg: SeriesSummary,
    /// Macro F1's summary, published beside `f_avg` and never instead of it.
    pub macro_f1: SeriesSummary,
    /// MCC's summary.
    pub mcc: SeriesSummary,
    /// THE ACTIVE-SCOPE UNCERTAINTY: a 95% interval for `f_avg` over the ten contracted
    /// seeds, on the frozen `T_CRIT_975_DF9`.
    ///
    /// A SEED-DISPERSION interval at fixed data and protocol — how far the headline moves
    /// when only the sampling seed moves. NOT a population interval and NOT a comparison.
    /// The renderer must label it as such at the point of presentation; a bare interval
    /// beside a mean reads as a comparison the active scope did not make.
    pub f_avg_seed_ci95: Ci95,
    /// Every seed's own numbers, seed ascending.
    pub per_seed: Vec<SeedValue>,
}

/// One seed's paired delta.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeedDelta {
    /// The contracted seed.
    pub seed: u32,
    /// `F_avg(setfit) - F_avg(lora)` on the SAME selection manifest.
    pub delta: f64,
    /// The delta's IEEE-754 bits.
    pub delta_bits: u64,
}

/// A paired 95% interval, or an explicit statement that there is none.
///
/// Every numeric field is `skip_serializing_if = "Option::is_none"`, so the degenerate case
/// serializes as `{"null_reason": "zero_variance"}` and NOT as a set of `null` bounds. That
/// distinction is the whole point: `serde_json` renders a non-finite `f64` as `null`, and a
/// reader parses `null` as a MISSING measurement rather than a degenerate one (Ph3 CR-03).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ci95 {
    /// `d̄ − t·s_d/√n`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<f64>,
    /// `d̄ + t·s_d/√n`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<f64>,
    /// `t·s_d/√n`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub half_width: Option<f64>,
    /// `s_d/√n`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_err: Option<f64>,
    /// Present EXACTLY when there is no interval, stating why in one machine-readable word.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub null_reason: Option<String>,
}

impl Ci95 {
    /// Whether an interval exists.
    #[must_use]
    pub const fn is_present(&self) -> bool {
        self.low.is_some() && self.high.is_some()
    }
}

/// The paired comparison at one shot level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShotDelta {
    /// Examples per class.
    pub shots: u32,
    /// Ten per-seed deltas, seed ascending.
    pub per_seed_deltas: Vec<SeedDelta>,
    /// `d̄`, the mean paired difference. ALWAYS present, even when the interval is not: a point
    /// estimate over identical differences is well defined and is the honest thing to report.
    pub mean_delta: f64,
    /// `s_d`. Absent exactly when the differences have no variance.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub std_delta: Option<f64>,
    /// The interval, or the stated reason there is none.
    pub ci95: Ci95,
    /// The paired t-statistic. DETAIL ONLY — never claim language (D-08).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub t_statistic: Option<f64>,
    /// The two-tailed p-value. DETAIL ONLY — it may sit in the machine-readable payload and it
    /// may NOT appear in a verdict. A binary verdict is precisely where few-shot seed
    /// sensitivity hides: rankings that reverse across seeds become one word.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub p_value: Option<f64>,
}

/// The resource summary for one `(method, shots)` group, every figure carrying its boundary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MethodShotResource {
    /// Which method.
    pub method: Method,
    /// Examples per class.
    pub shots: u32,
    /// The distinct hosts these cells ran on, sorted. Resource figures are NEVER pooled across
    /// hosts (D-09); this field is what lets the renderer say so at the point of comparison.
    pub hosts: Vec<String>,
    /// The distinct `<device>:<implementation>:<kernel>` identities, sorted.
    pub backends: Vec<String>,
    /// Mean training wall clock, milliseconds.
    pub train_wall_ms: SeriesSummary,
    /// Cold latency: ONE classify in a dedicated fresh child.
    pub cold_latency_ms: SeriesSummary,
    /// Warm latency: median of ten after three warmups, against the reloaded model.
    pub warm_latency_ms_median: SeriesSummary,
    /// Rows per second over one full test-split batch pass.
    pub throughput_rows_per_sec: SeriesSummary,
    /// The batch size that pass used. Throughput without a batch size is not comparable.
    pub throughput_batch_sizes: Vec<u32>,
    /// Peak RSS of the TRAINING process.
    pub train_peak_rss_bytes: SeriesSummary,
    /// The distinct mechanisms `train_peak_rss_bytes` was measured by, sorted.
    pub train_peak_rss_mechanisms: Vec<String>,
    /// The class of those mechanisms, for the renderer's comparability note.
    pub train_peak_rss_mechanism_classes: Vec<MechanismClass>,
    /// Peak RSS of the dedicated COLD-MEASUREMENT CHILD — a different process, a different
    /// number. A kernel high-water mark is process-cumulative, so one pooled figure would
    /// report the training peak while claiming to report the inference peak.
    pub inference_peak_rss_bytes: SeriesSummary,
    /// The distinct mechanisms `inference_peak_rss_bytes` was measured by, sorted.
    pub inference_peak_rss_mechanisms: Vec<String>,
    /// The class of those mechanisms.
    pub inference_peak_rss_mechanism_classes: Vec<MechanismClass>,
    /// Bytes of the artifact THIS method wrote. For LoRA this is the ADAPTER ALONE and it is
    /// NOT a deployable size — see [`Self::deployable_total_bytes`].
    pub artifact_bytes: SeriesSummary,
    /// Bytes a user must ship to serve this model. The ONLY field a cross-method size claim may
    /// be built on (review consensus item 8).
    pub deployable_total_bytes: SeriesSummary,
}

/// Everything recomputed from a [`VerifiedRunSet`], and nothing else.
///
/// No number in here comes from anywhere but the stored rows plus closed-form arithmetic, which
/// is what EVAL-04's "exactly recompute" asks for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunAggregate {
    /// The contract these numbers are computed under.
    pub contract_id: String,
    /// Seeds per `(method, shots)` group.
    pub n_seeds: usize,
    /// Degrees of freedom of the paired design.
    pub degrees_of_freedom: usize,
    /// The frozen two-tailed 95% critical value the intervals were built with.
    pub t_crit_975_df9: f64,
    /// The cell keys, in the exact order this aggregate iterated them. Pinned by test.
    pub key_sequence: Vec<String>,
    /// The methods this aggregate actually covers, in contract order. Under the ACTIVE scope
    /// this is one method, and that is what makes `deltas` empty rather than defective.
    pub methods: Vec<String>,
    /// Quality per `(method, shots)`, in contract order.
    pub quality: Vec<MethodShotQuality>,
    /// The paired deltas per shot level, shots ascending.
    ///
    /// EMPTY, AND STRUCTURALLY ABSENT FROM THE SERIALIZED FORM, whenever fewer than two
    /// methods are present — which is every run under the active scope. Not an empty section
    /// and not a null: a delta key carrying `[]` still tells a reader a comparison was
    /// attempted, and that is the implication D-19 forbids.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deltas: Vec<ShotDelta>,
    /// Resource per `(method, shots)`, in contract order.
    pub resource: Vec<MethodShotResource>,
}

/// The invariant message used wherever a helper's `Option` cannot be `None`.
///
/// [`VerifiedRunSet`] holds exactly [`EXPECTED_CELLS`] rows covering every `(method, shots)`
/// group at all ten seeds — [`verify_run`] proved it before the value existed. `expect` rather
/// than a fallible signature on purpose: if the type cannot carry that guarantee, the typestate
/// is decoration.
const SERIES_INVARIANT: &str =
    "VerifiedRunSet holds exactly ten seeds per (method, shots) — verify_run proved it";

/// Summarise one series with the 05-04 closed-form helpers. No local mean or std.
fn summarise(values: &[f64]) -> SeriesSummary {
    let mean = mean_f64(values).expect(SERIES_INVARIANT);
    let std = sample_std_f64(values).expect(SERIES_INVARIANT);
    let (min, max) = min_max_f64(values).expect(SERIES_INVARIANT);
    SeriesSummary { n: values.len(), mean, std, min, max }
}

/// Distinct values of a string field across a group, sorted — so two runs emit the same order.
fn distinct_sorted(values: impl Iterator<Item = String>) -> Vec<String> {
    let mut out: Vec<String> = values.collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// Recompute every published number from a verified run set.
///
/// Deterministic by construction: the iteration order is the contract order (method in declared
/// order, then shots ascending, then seed ascending), every collection is a `Vec` or a
/// `BTreeMap`, and no `HashMap` is iterated anywhere. Two invocations over the same rows produce
/// bit-identical output, which a test asserts by comparing serialized `f64` bits rather than
/// rendered decimals.
///
/// The signature is the proof that the arithmetic cannot run on unverified data: [`VerifiedRunSet`]
/// has no public constructor.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn aggregate(verified: &VerifiedRunSet) -> RunAggregate {
    let mut key_sequence = Vec::with_capacity(verified.len());
    let mut quality = Vec::with_capacity(BENCH_METHODS.len() * BENCH_SHOTS.len());
    let mut resource = Vec::with_capacity(BENCH_METHODS.len() * BENCH_SHOTS.len());

    // Index once, in a BTreeMap keyed by the derived-Ord CellKey, so every lookup below is
    // deterministic and no hash iteration order can leak into a published number.
    let mut by_cell: BTreeMap<CellKey, &BenchRow> = BTreeMap::new();
    for (cell, row) in verified.rows() {
        by_cell.insert(*cell, row);
    }

    // THE METHODS THIS SET ACTUALLY CONTAINS, in contract order — read from the verified
    // rows rather than from a constant. Under the active scope that is one method, so no
    // group is emitted for a method nobody measured, and the deltas below are empty rather
    // than computed over an absent arm. Reading it from a constant would have produced a
    // group of empty series for the deferred method and called it data.
    let mut methods_present: Vec<Method> = BENCH_METHODS
        .iter()
        .copied()
        .filter(|m| verified.rows().iter().any(|(cell, _)| cell.method == *m))
        .collect();
    methods_present
        .sort_unstable_by_key(|m| BENCH_METHODS.iter().position(|b| b == m).unwrap_or(0));

    for method in methods_present.iter().copied() {
        for shots in BENCH_SHOTS {
            let mut f_avg = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut macro_f1 = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut mcc = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut per_seed = Vec::with_capacity(PAIRED_DESIGN_N);

            let mut train_wall = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut cold = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut warm = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut throughput = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut train_peak = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut inference_peak = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut artifact = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut deployable = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut hosts = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut backends = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut train_mechanisms = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut inference_mechanisms = Vec::with_capacity(PAIRED_DESIGN_N);
            let mut batch_sizes = Vec::with_capacity(PAIRED_DESIGN_N);

            for seed in BENCH_SEEDS {
                let cell = CellKey::new(method, shots, seed);
                key_sequence.push(cell.render());
                let Some(row) = by_cell.get(&cell) else {
                    // Unreachable through verify_run, which proved all 80 cells present. Skipped
                    // rather than panicked so a future in-crate caller gets a short series and a
                    // loud `expect` below rather than an abort here.
                    continue;
                };
                let q = &row.payload.quality;
                f_avg.push(q.f_avg);
                macro_f1.push(q.macro_f1);
                mcc.push(q.mcc);
                per_seed.push(SeedValue {
                    seed,
                    f_avg: q.f_avg,
                    f_avg_bits: q.f_avg_bits,
                    macro_f1: q.macro_f1,
                    mcc: q.mcc,
                });

                let r = &row.payload.resource;
                #[allow(clippy::cast_precision_loss)]
                {
                    train_wall.push(r.train_wall_ms as f64);
                    train_peak.push(r.train_peak_rss_bytes as f64);
                    inference_peak.push(r.inference_peak_rss_bytes as f64);
                    artifact.push(r.artifact_bytes as f64);
                    deployable.push(r.deployable_total_bytes as f64);
                }
                cold.push(r.cold_latency_ms);
                warm.push(r.warm_latency_ms_median);
                throughput.push(r.throughput_rows_per_sec);
                batch_sizes.push(r.throughput_batch_size);
                train_mechanisms.push(r.train_peak_rss_mechanism.clone());
                inference_mechanisms.push(r.inference_peak_rss_mechanism.clone());
                hosts.push(format!(
                    "{} ({}/{})",
                    row.payload.host.hostname, row.payload.host.os, row.payload.host.arch
                ));
                backends.push(row.payload.backend_identity.clone());
            }

            quality.push(MethodShotQuality {
                method,
                shots,
                f_avg: summarise(&f_avg),
                f_avg_seed_ci95: seed_dispersion_ci95(&f_avg),
                macro_f1: summarise(&macro_f1),
                mcc: summarise(&mcc),
                per_seed,
            });

            let train_mechanisms = distinct_sorted(train_mechanisms.into_iter());
            let inference_mechanisms = distinct_sorted(inference_mechanisms.into_iter());
            let mut batch_sizes_sorted = batch_sizes;
            batch_sizes_sorted.sort_unstable();
            batch_sizes_sorted.dedup();

            resource.push(MethodShotResource {
                method,
                shots,
                hosts: distinct_sorted(hosts.into_iter()),
                backends: distinct_sorted(backends.into_iter()),
                train_wall_ms: summarise(&train_wall),
                cold_latency_ms: summarise(&cold),
                warm_latency_ms_median: summarise(&warm),
                throughput_rows_per_sec: summarise(&throughput),
                throughput_batch_sizes: batch_sizes_sorted,
                train_peak_rss_bytes: summarise(&train_peak),
                train_peak_rss_mechanism_classes: mechanism_classes(&train_mechanisms),
                train_peak_rss_mechanisms: train_mechanisms,
                inference_peak_rss_bytes: summarise(&inference_peak),
                inference_peak_rss_mechanism_classes: mechanism_classes(&inference_mechanisms),
                inference_peak_rss_mechanisms: inference_mechanisms,
                artifact_bytes: summarise(&artifact),
                deployable_total_bytes: summarise(&deployable),
            });
        }
    }

    // A DELTA NEEDS TWO ARMS. With one method present there is no pair to difference, so no
    // delta is computed at all — rather than computed over an absent arm and reported with a
    // null reason, which is a comparison section wearing a caveat.
    let deltas: Vec<ShotDelta> = if methods_present.len() >= 2 {
        BENCH_SHOTS.iter().map(|shots| shot_delta(*shots, &by_cell)).collect()
    } else {
        Vec::new()
    };

    RunAggregate {
        contract_id: CLAIMS_CONTRACT_ID.to_string(),
        n_seeds: PAIRED_DESIGN_N,
        degrees_of_freedom: PAIRED_DESIGN_N - 1,
        t_crit_975_df9: T_CRIT_975_DF9,
        key_sequence,
        methods: methods_present.iter().map(|m| m.tag().to_string()).collect(),
        quality,
        deltas,
        resource,
    }
}

/// The ACTIVE-scope seed-dispersion interval for one series, in the typed [`Ci95`] shape.
///
/// Delegates to `aprender_core::stats::hypothesis::ci95_one_sample_df9`, which itself
/// delegates to the PAIRED helper against a zero comparator — so the mean and the (n-1) std
/// come from the one `moments_or_zero_variance` the paired path uses (OPS-03). There is no
/// second definition of either moment anywhere in this layer.
///
/// Ten identical seed scores produce the typed zero-variance shape with an explicit
/// no-interval reason. Never a NaN, and never a serde null a reader would take for a missing
/// measurement (CR-03).
fn seed_dispersion_ci95(values: &[f64]) -> Ci95 {
    match ci95_one_sample_df9(values) {
        Ok(ci) => Ci95 {
            low: Some(ci.low),
            high: Some(ci.high),
            half_width: Some(ci.half_width),
            std_err: Some(ci.std_err),
            null_reason: None,
        },
        Err(AprenderError::ZeroVarianceDifferences { .. }) => Ci95 {
            low: None,
            high: None,
            half_width: None,
            std_err: None,
            null_reason: Some(ZERO_VARIANCE_NULL_REASON.to_string()),
        },
        // Unreachable through a VerifiedRunSet, which guarantees ten seeds per group.
        // Reported with its reason rather than panicked: a gate that aborts tells a reader
        // less than one that says which number is missing and why.
        Err(other) => Ci95 {
            low: None,
            high: None,
            half_width: None,
            std_err: None,
            null_reason: Some(other.to_string()),
        },
    }
}

/// The distinct classes of a sorted mechanism list, deduplicated but order-preserving.
fn mechanism_classes(mechanisms: &[String]) -> Vec<MechanismClass> {
    let mut out: Vec<MechanismClass> =
        mechanisms.iter().map(|m| mechanism_class(m.as_str())).collect();
    out.dedup();
    out
}

/// The paired comparison at one shot level, computed with the 05-04 closed-form surface only.
fn shot_delta(shots: u32, by_cell: &BTreeMap<CellKey, &BenchRow>) -> ShotDelta {
    let mut setfit = Vec::with_capacity(PAIRED_DESIGN_N);
    let mut lora = Vec::with_capacity(PAIRED_DESIGN_N);
    let mut per_seed_deltas = Vec::with_capacity(PAIRED_DESIGN_N);

    for seed in BENCH_SEEDS {
        let s = by_cell.get(&CellKey::new(Method::Setfit, shots, seed));
        let l = by_cell.get(&CellKey::new(Method::Lora, shots, seed));
        if let (Some(s), Some(l)) = (s, l) {
            let sv = s.payload.quality.f_avg;
            let lv = l.payload.quality.f_avg;
            setfit.push(sv);
            lora.push(lv);
            let delta = sv - lv;
            per_seed_deltas.push(SeedDelta { seed, delta, delta_bits: delta.to_bits() });
        }
    }

    // ONE definition of the differences. `paired_ci95_df9` and `ttest_rel_f64` both take the two
    // series and derive `d̄`/`s_d` through the same private helper in `hypothesis.rs` (OPS-03),
    // so the interval and the statistic can never disagree about the moments.
    let ci = paired_ci95_df9(&setfit, &lora);
    let test = ttest_rel_f64(&setfit, &lora);

    match ci {
        Ok(ci) => ShotDelta {
            shots,
            per_seed_deltas,
            mean_delta: ci.mean_diff,
            std_delta: Some(ci.std_diff),
            ci95: Ci95 {
                low: Some(ci.low),
                high: Some(ci.high),
                half_width: Some(ci.half_width),
                std_err: Some(ci.std_err),
                null_reason: None,
            },
            t_statistic: test.as_ref().ok().map(|t| t.statistic),
            p_value: test.ok().map(|t| t.pvalue),
        },
        // THE DEGENERATE CASE, MADE VISIBLE. The point estimate is still reported — it is well
        // defined and it is what a reader wants — and the interval is STRUCTURALLY ABSENT with a
        // stated reason rather than serialized as a non-finite number that renders as `null`.
        Err(AprenderError::ZeroVarianceDifferences { constant_value, .. }) => ShotDelta {
            shots,
            per_seed_deltas,
            mean_delta: constant_value,
            std_delta: None,
            ci95: Ci95 {
                low: None,
                high: None,
                half_width: None,
                std_err: None,
                null_reason: Some(ZERO_VARIANCE_NULL_REASON.to_string()),
            },
            t_statistic: None,
            p_value: None,
        },
        // Unreachable through a VerifiedRunSet, which guarantees ten pairs at every shot level.
        // Reported as a degenerate interval with its own reason rather than panicking: a gate
        // that aborts tells a reader less than one that says which number is missing and why.
        Err(other) => ShotDelta {
            shots,
            per_seed_deltas,
            mean_delta: mean_f64(
                &setfit.iter().zip(lora.iter()).map(|(s, l)| s - l).collect::<Vec<f64>>(),
            )
            .unwrap_or_default(),
            std_delta: None,
            ci95: Ci95 {
                low: None,
                high: None,
                half_width: None,
                std_err: None,
                null_reason: Some(other.to_string()),
            },
            t_statistic: None,
            p_value: None,
        },
    }
}

#[cfg(test)]
#[path = "bench_gate_tests.rs"]
mod bench_gate_tests;
