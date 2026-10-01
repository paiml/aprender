//! The fresh-process door: artifact bytes plus the run's inputs, in — a sealed
//! [`SetFitCredential`] out (plan 04-16, D-11, D-16, APR-04, TRN-07).
//!
//! # The gap this closes
//!
//! `create_selection_lock`, `SelectionLock::mint_test_token` and
//! `CanonicalTestAccess::grant` are the three doors D-16/TRN-07's "a user can" tier runs
//! through. Before 04-17 they were typed against `SetFitRun<ArtifactReloadedAndVerified>`,
//! whose only producer is `SetFitRun::<HeadFitted>::verify_artifact` — a method that
//! consumes a live, just-trained run. A separate `apr eval --split test` process, holding
//! nothing but a `.apr` file, could not reach any of them.
//!
//! # What is NOT attempted here, and why that is settled rather than deferred
//!
//! This module does **not** rebuild a `SetFitRun<HeadFitted>` and re-enter
//! `verify_artifact`. That was 04-16's original design and rustc refused it, twice, in
//! terms that are not a matter of effort:
//!
//! * `E0063` — `HeadFittedEvidence` has seven fields, and five of them
//!   (`report`, `encode_ledger`, `encode_call_count`, `passed`, and `effective_lambda`
//!   only by recomputation) are records of optimizer work that happened in another
//!   process. They are not in the artifact because the artifact deliberately carries the
//!   evidence SUMMARY, not the evidence TABLE (bundle field 19).
//! * `E0451` — `tune::PassedEvidence`'s fields are private, and its only producer is
//!   `tune::validate_evidence`, which takes the full `UpdateEvidence` table. The artifact
//!   carries that table's HASH, which can corroborate a table but cannot supply one.
//!
//! The measurement is written up in `04-16-BLOCKED.md`. **No field of `HeadFittedEvidence`
//! or `PassedEvidence` is fabricated, defaulted or reconstructed anywhere in this module,
//! and none may be added later.** A reloaded run that answered `loss_trace_hash()` or
//! `step_count()` would be answering about a run this process never observed.
//!
//! # What IS produced, and why it is sufficient
//!
//! 04-CONTEXT.md **D-11** already defines a fresh process's verified state, and it is
//! narrower than the train-time one: *integrity + embedded self-test probes*. That value
//! exists and shipped in 04-03 — `aprender::setfit::VerifiedSetFitModel`, mintable only by
//! `load_setfit_apr`, which runs the whole eight-rung ladder including the six-probe
//! replay. And the three doors between them read exactly three values: the artifact hash,
//! the selection's semantic hash and the selection's ledger hash. Not one touches the
//! evidence table.
//!
//! So this door runs the production loader, checks that the artifact and the caller's own
//! `Selection`/`PreparedDataset` are describing the same run, and hands back a
//! [`ReloadedSetFitCredential`]. `credential.rs` seals it; `lock.rs` accepts it.
//!
//! # There is no second verification policy here
//!
//! Every rung this door depends on is run by `load_setfit_apr`, in `aprender-core`, in the
//! one place production runs them. This module adds no tolerance, no rebuild, no probe and
//! no comparison of model outputs. What it adds is a PROVENANCE gate — three equalities
//! between the artifact's record of its inputs and the objects the caller actually holds —
//! which is a check the loader cannot make, because the loader is never handed the dataset
//! or the selection.
//!
//! # Why there is no re-serialization gate
//!
//! An earlier draft of this plan ended with "the minted artifact hash must equal
//! `artifact_sha256_hex` of the input". That gate belonged to a design that rebuilt a run
//! and re-serialized it through the codec, where the re-serialized bytes could genuinely
//! differ from the file. Nothing here re-serializes anything: `VerifiedSetFitModel`'s
//! `artifact_sha256()` IS `artifact_sha256_hex` of the input slice, taken by the loader
//! over the bytes it was handed. Restating that equality would be a check comparing a
//! value with itself — the shape of assertion CLAUDE.md's verification discipline calls
//! out, and worse than no check because it reads like one.
//!
//! # Why the LEDGER hash is compared, and why the gate order is what it is
//!
//! **It is not live process state.** It is a value the persisted selection manifest
//! CARRIES: `Selection::replay` (aprender-contrastive-data `select.rs:480-503`) reads
//! `payload.ledger_hash`, rebuilds an `AccessLedger` from `payload.access_ledger`, refuses
//! any disagreement with `SelectionReplayMismatch { field: "access_ledger" }`, and then
//! assembles the `Selection` with that same recorded hash. Two processes loading one
//! manifest therefore hold one ledger hash. An earlier draft of this door declined the
//! comparison on the grounds that a reloading process is entitled to an audit trail of its
//! own; that is false for a manifest-loaded selection, and it would have talked a later
//! reader out of a check that works.
//!
//! **The three identifiers are NOT independent, and the order below is what makes all three
//! reachable.** The relationship was measured rather than assumed
//! (`apr_reload_the_three_recorded_identifiers_are_ordered_coarse_to_fine`):
//!
//! * `Selection::semantic_hash` is `SHA-256` of the WHOLE `SelectionPayload`
//!   (`select.rs::assemble`), and that payload embeds both `access_ledger` and
//!   `ledger_hash`. So a different ledger implies a different semantic hash, always.
//! * The access ledger records the dataset fingerprint and the purposes, and NOTHING that
//!   depends on the seed. So two selections that differ only in their root seed have the
//!   SAME ledger hash and different semantic hashes.
//! * The ledger's records embed the dataset fingerprint, so a different corpus implies a
//!   different ledger hash too.
//!
//! Checked coarse to fine — corpus, then ledger, then rows — each refusal is reached only
//! once every coarser identity has already agreed, so each one means exactly what it says.
//! Checked in the other order the fine gate absorbs the coarse ones: a selection taken under
//! a different audit trail would be reported as "different rows were selected", which is not
//! an opaque diagnosis but a WRONG one, and the ledger arm would be unreachable code
//! wearing a test.

use aprender::setfit::{load_setfit_apr, VerifiedSetFitModel};
use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::select::Selection;

use super::apr_codec::artifact_error;
use super::bundle::ProvenanceRecord;
use super::SetFitTrainError;

/// Failure modes of the fresh-process reload door.
///
/// The four are kept in their own enum, on the precedent [`super::verify::CodecError`] and
/// [`super::bundle::BundleError`] set: the module that owns an operation owns its failure
/// vocabulary, and `SetFitTrainError` wraps it. Every mismatch names BOTH values, because
/// "the artifact does not match your selection" is a diagnosis nobody can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum AprReloadError {
    /// The artifact's `provenance` sub-document is not a [`ProvenanceRecord`].
    ///
    /// The document survives `deny_unknown_fields` parsing at rung 4 as an opaque
    /// `serde_json::Value` — `aprender-core` cannot name this crate's types, which is why
    /// the four sub-documents stay opaque there. So the shape is checked HERE, and a
    /// disagreement fails closed rather than reaching the gate with a defaulted record.
    ProvenanceUnreadable {
        /// What `serde_json` said about the sub-document.
        reason: String,
    },
    /// The supplied selection selected different ROWS than the artifact records.
    SelectionSemanticHashMismatch {
        /// The hash the artifact's provenance records.
        recorded: String,
        /// The hash the supplied selection computes.
        supplied: String,
    },
    /// The supplied selection was taken under a different ACCESS LEDGER.
    ///
    /// Reached only when the corpus already matched, and raised BEFORE
    /// [`Self::SelectionSemanticHashMismatch`] because the semantic hash covers the ledger
    /// too (`select.rs::assemble` digests the whole payload, ledger fields included). Asked
    /// in the other order this arm is unreachable and a differently-audited selection is
    /// reported as a different DRAW — a wrong diagnosis rather than a coarse one.
    SelectionLedgerHashMismatch {
        /// The hash the artifact's provenance records.
        recorded: String,
        /// The hash the supplied selection carries.
        supplied: String,
    },
    /// The supplied dataset is not the corpus the artifact was trained on.
    DatasetFingerprintMismatch {
        /// The fingerprint the artifact's provenance records.
        recorded: String,
        /// The fingerprint the supplied dataset computes.
        supplied: String,
    },
}

impl core::fmt::Display for AprReloadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ProvenanceUnreadable { reason } => write!(
                f,
                "the artifact's `provenance` sub-document is not a provenance record: \
                 {reason}. It is bundle field 20, and no function of the other nineteen \
                 fields reproduces it, so a reload cannot proceed without it \
                 (contract setfit-apr-v1, requirement APR-01)",
            ),
            Self::SelectionSemanticHashMismatch { recorded, supplied } => write!(
                f,
                "this artifact was trained under the selection whose semantic hash is \
                 `{recorded}`, and the selection supplied hashes to `{supplied}`. The \
                 semantic hash covers the ordered ids, the label map and both content \
                 hashes, so a different value means DIFFERENT ROWS were selected — the \
                 lock this artifact would take would then record a selection decision \
                 nobody made (contract setfit-apr-v1, requirement TRN-07)",
            ),
            Self::SelectionLedgerHashMismatch { recorded, supplied } => write!(
                f,
                "this artifact was trained under the selection whose access-ledger hash is \
                 `{recorded}`, and the selection supplied carries `{supplied}`. The ledger \
                 hash travels IN the persisted selection manifest and `Selection::replay` \
                 refuses to assemble a selection whose records do not produce it, so two \
                 processes loading one manifest hold one value: a disagreement here is a \
                 different selection, not a different process \
                 (contract setfit-apr-v1, requirement TRN-07)",
            ),
            Self::DatasetFingerprintMismatch { recorded, supplied } => write!(
                f,
                "this artifact was trained on the dataset fingerprinted `{recorded}`, and \
                 the dataset supplied fingerprints to `{supplied}`. A canonical-test grant \
                 taken over a different corpus would admit rows this model has no claim to \
                 (contract setfit-apr-v1, requirement TRN-07)",
            ),
        }
    }
}

impl std::error::Error for AprReloadError {}

/// A fresh process's credential: a verified model, bound to the inputs it was trained on.
///
/// # What holding one proves
///
/// 1. The bytes passed every rung of `aprender-core`'s production ladder, including the
///    six-probe replay — so the model rebuilt from them reproduces the answers the writer
///    recorded, in THIS process, on THESE bytes.
/// 2. The artifact's recorded selection semantic hash, selection ledger hash and dataset
///    fingerprint all equal the values computed from the `Selection` and
///    `PreparedDataset` the caller actually holds.
///
/// It proves nothing about the optimizer run that produced the weights, and it exposes no
/// accessor that would imply otherwise. That asymmetry with
/// `SetFitRun<ArtifactReloadedAndVerified>` is D-11 as written, not an omission.
///
/// # Non-constructible outside this module
///
/// Every field is private and there is no constructor, no `Default` and no `Deserialize`:
/// [`reload_verified_run_from_apr`] is the only way one comes into existence, which is the
/// same argument `VerifiedSetFitModel` makes one layer down and `SelectionLock` makes one
/// layer up.
pub struct ReloadedSetFitCredential {
    /// The end of the production ladder. Retained so the process that earned the grant can
    /// actually classify with it rather than loading the file a second time.
    model: VerifiedSetFitModel,
    /// SHA-256 of the artifact bytes, lowercase hex — the loader's own digest of its input.
    artifact_hash: String,
    /// The supplied selection's semantic hash, hex. Equal to the artifact's record by the
    /// gate that ran before this value was stored.
    selection_semantic_hash: String,
    /// The supplied selection's access-ledger hash, RAW. Same equality, same gate.
    selection_ledger_hash: [u8; 32],
}

/// Hand-written, and it prints the three digests and nothing else.
///
/// A derive would recurse into [`VerifiedSetFitModel`], whose `Debug` renders the rebuilt
/// encoder — every tensor. 04-17 measured the same hazard on a 1.74 MiB retained buffer and
/// answered it the same way; a pinned MiniLM's tensor set is two orders of magnitude worse.
impl core::fmt::Debug for ReloadedSetFitCredential {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReloadedSetFitCredential")
            .field("artifact_hash", &self.artifact_hash)
            .field("selection_semantic_hash", &self.selection_semantic_hash)
            .field("selection_ledger_hash", &hex::encode(self.selection_ledger_hash))
            .finish()
    }
}

impl ReloadedSetFitCredential {
    /// SHA-256 of the artifact bytes this credential was minted from, lowercase hex.
    #[must_use]
    pub fn artifact_hash(&self) -> &str {
        &self.artifact_hash
    }

    /// The selection's semantic hash, hex.
    #[must_use]
    pub fn selection_semantic_hash(&self) -> &str {
        &self.selection_semantic_hash
    }

    /// The selection's access-ledger hash, raw.
    #[must_use]
    pub const fn selection_ledger_hash(&self) -> [u8; 32] {
        self.selection_ledger_hash
    }

    /// The verified model, for the classification the grant was earned to perform.
    ///
    /// A SHARED borrow: this door hands out no way to mutate or re-mint the model.
    #[must_use]
    pub const fn model(&self) -> &VerifiedSetFitModel {
        &self.model
    }

    /// Give up the credential and keep the model.
    ///
    /// For the caller that has finished with the lock chain and wants to classify without
    /// holding both. It CONSUMES, so a caller cannot keep presenting a credential whose
    /// model it has moved away.
    #[must_use]
    pub fn into_model(self) -> VerifiedSetFitModel {
        self.model
    }
}

/// Reload a `setfit-apr-v1` artifact into a credential the lock chain accepts.
///
/// The order below is the whole of the design and it is not interchangeable:
///
/// 1. **`load_setfit_apr` FIRST.** A consumer who cannot verify the artifact must never
///    reach a provenance comparison, let alone a credential. The full ladder — container
///    CRCs, the `deny_unknown_fields` document, the structural and finiteness checks, the
///    rebuild and the six-probe replay — runs before anything below it.
/// 2. The provenance identity gate, all three recorded identifiers, each its own typed
///    refusal naming both values.
/// 3. The credential, carrying the three values the doors read.
///
/// The `dataset` and `selection` are taken BY REFERENCE. The credential retains neither: the
/// three doors read three hashes, and `CanonicalTestAccess::grant` takes the dataset
/// separately — so consuming them here would take from the caller exactly the object it
/// needs next.
///
/// # The two hashes are read off the CALLER'S objects, not off the artifact
///
/// After the gate the two agree, so the choice looks free. It is not. The artifact records
/// hex STRINGS; `SetFitCredential::selection_ledger_hash` returns a `[u8; 32]`, and turning
/// the record back into one needs a decode that can fail — a failure mode arriving after the
/// credential exists, which is precisely the state the sealed trait's infallible accessors
/// are defined to make unrepresentable. Reading the digest off the `Selection` that just
/// satisfied the gate has no such step.
///
/// # Errors
///
/// [`SetFitTrainError::Codec`] carrying the core loader's typed
/// `SetFitArtifactError` whole, for any rung failure; and
/// [`SetFitTrainError::AprReload`] carrying one of [`AprReloadError`]'s four variants.
pub fn reload_verified_run_from_apr(
    bytes: &[u8],
    dataset: &PreparedDataset<Canonical>,
    selection: &Selection,
) -> Result<ReloadedSetFitCredential, SetFitTrainError> {
    // RUNG LADDER FIRST — all eight of them, in aprender-core, in the one place production
    // runs them. `artifact_error` is `apr_codec`'s own mapping, borrowed rather than
    // rewritten: a second spelling of "this core failure belongs to this codec" is a second
    // place for the format id to drift.
    let model =
        load_setfit_apr(bytes).map_err(|source| SetFitTrainError::Codec(artifact_error(source)))?;

    // Field 20, typed. `aprender-core` keeps it opaque because train depends on core and
    // never the reverse, so this is the first point at which it CAN be a record.
    // Deserialize by REFERENCE. `from_value` takes the `Value` by value, so the previous
    // spelling deep-cloned the provenance object purely to satisfy the signature; serde_json
    // implements `Deserializer` for `&Value`, so the error text and field handling are
    // unchanged and the copy disappears. `apr eval --split validation` reloads once per
    // `--candidate`, so this recurs per candidate.
    let recorded: ProvenanceRecord = serde::Deserialize::deserialize(&model.doc_view().provenance)
        .map_err(|error: serde_json::Error| AprReloadError::ProvenanceUnreadable {
            reason: error.to_string(),
        })?;

    // ---- The provenance identity gate: three checks, COARSE TO FINE, no rebuild yet. ----
    // The order is load-bearing, not stylistic. See this module's header: the three
    // identifiers nest, and the fine one absorbs the coarse ones if it is asked first.

    // COARSEST — the corpus. It digests the WHOLE dataset (`evaluate.rs`'s header draws the
    // same distinction), so a changed validation row is caught here rather than by a fourth
    // check.
    let supplied_dataset_fingerprint = dataset.validation_witness().dataset_fingerprint_hex();
    if recorded.dataset_fingerprint() != supplied_dataset_fingerprint {
        return Err(AprReloadError::DatasetFingerprintMismatch {
            recorded: recorded.dataset_fingerprint().to_string(),
            supplied: supplied_dataset_fingerprint,
        }
        .into());
    }

    // THE ACCESS TRAIL. Reached only once the corpus agrees, so this refusal means "the same
    // corpus, audited differently" and nothing else. The value is comparable across
    // processes because `Selection::replay` assembles from the manifest's own recorded
    // ledger hash after refusing a manifest whose records do not reproduce it.
    let supplied_ledger_digest = selection.ledger_hash();
    let supplied_ledger_hash = hex::encode(supplied_ledger_digest);
    if recorded.selection_ledger_hash() != supplied_ledger_hash {
        return Err(AprReloadError::SelectionLedgerHashMismatch {
            recorded: recorded.selection_ledger_hash().to_string(),
            supplied: supplied_ledger_hash,
        }
        .into());
    }

    // FINEST — which rows. `semantic_hash` is a digest of the entire selection payload, so
    // reaching this point means every coarser fact already matched and the disagreement is
    // the draw itself: a different seed, or a different shot count.
    let supplied_semantic_hash = hex::encode(selection.semantic_hash());
    if recorded.selection_semantic_hash() != supplied_semantic_hash {
        return Err(AprReloadError::SelectionSemanticHashMismatch {
            recorded: recorded.selection_semantic_hash().to_string(),
            supplied: supplied_semantic_hash,
        }
        .into());
    }

    Ok(ReloadedSetFitCredential {
        // The loader's digest of the slice it was handed. Not re-derived here: one hash of
        // one buffer, computed once, by the code that read it.
        artifact_hash: model.artifact_sha256().to_string(),
        selection_semantic_hash: supplied_semantic_hash,
        selection_ledger_hash: supplied_ledger_digest,
        model,
    })
}

#[cfg(test)]
#[path = "apr_reload_tests.rs"]
mod apr_reload_tests;
