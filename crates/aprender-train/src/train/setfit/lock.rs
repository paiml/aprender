//! The selection lock, and the canonical-test token it mints (D-14, TRN-07).
//!
//! Contract: `setfit-train-lifecycle-v1`, equations `selection_lock_commitment` and
//! `canonical_test_token_minting`. Requirement: TRN-07.
//!
//! # What this is the complement of
//!
//! Phase 2's access ledger records WHICH splits were touched. That is a different fact from the
//! one TRN-07 needs and the two must not be conflated (D-14): a ledger showing only train and
//! validation accesses is equally consistent with an honest run and with a run that peeked at
//! test in a previous process. The lock is the other half — it is created BEFORE canonical test
//! access is permitted and it commits the whole selection decision, so "this model was chosen
//! on validation alone" becomes a hash somebody can check rather than a claim.
//!
//! # Three things a caller cannot do here, and why each was possible before
//!
//! 1. **Hand over a chosen index.** [`SelectionLock::from_candidates`] takes the candidate SET
//!    and a [`SelectionRule`], and derives the winner. A lock recording only the winner is
//!    exactly as consistent with "we tried ten configurations, looked at test, and wrote down
//!    the one that won there" as it is with honest selection, so recording only the winner
//!    proves nothing the requirement is about.
//! 2. **Hand over an artifact hash to mint against.** [`SelectionLock::mint_test_token`] takes
//!    the verified run OBJECT and reads `artifact_hash()` off it. A `[u8; 32]` parameter lets a
//!    caller submit the LOCKED hash and then evaluate an entirely different artifact — which is
//!    precisely the sequence TRN-07 exists to block, and it would leave no trace.
//! 3. **Carry a token to a different model.** [`CanonicalTestAccess::grant`] re-checks the
//!    token's artifact identity against the model AT ACCESS TIME. Minting is a check at one
//!    instant; a token is a value that can be moved, cloned into a struct and used later.
//!
//! # Append-only, like the ledger
//!
//! There is no API that removes a candidate, and none that edits one. `ledger.rs` states the
//! reason and it transfers unchanged: an append-only log that can be rewritten is not evidence.
//! The integrity check ([`SelectionLock::verify_integrity`]) is what makes that structural
//! rather than conventional, and `mint_test_token` runs it before it does anything else.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
use aprender_contrastive_data::split::{Split, Test};

use super::credential::SetFitCredential;
use super::evaluate::{ValidationEvaluation, ValidationEvaluationWire, ValidationMetricKind};
use super::{ArtifactReloadedAndVerified, SetFitRun};

/// The canonical selection-lock schema version.
const LOCK_SCHEMA_VERSION: u32 = 1;

/// Maximum accepted length of a serialized selection lock, in bytes (1 MiB).
///
/// # The derivation, and where it is checked
///
/// A candidate is a FIXED-SHAPE record: a config label and an artifact hash, plus a nested
/// evaluation carrying that artifact hash again, two more fingerprints, a metric tag, a `u64` of
/// value bits and a row count. At the 64-character hex hashes this format actually uses, that is
/// on the order of 600 bytes once JSON keys and punctuation are counted, so 1 MiB admits roughly
/// 1,600 candidates. `lock_max_selection_lock_bytes_clears_a_realistic_sweep` MEASURES that
/// figure as the difference between two real locks rather than trusting this paragraph, because
/// a derivation nobody re-runs is a comment about a version of the wire form that used to exist.
///
/// A sweep in this phase commits tens of candidates, so the bound clears the realistic worst
/// case by about two orders of magnitude while still refusing a payload that claims millions of
/// candidates in order to make the allocation itself the attack.
///
/// The bound is INCLUSIVE — a payload of exactly this length is admitted — and it is enforced on
/// the RAW slice before serde is handed anything, following `bundle.rs`'s door discipline.
pub const MAX_SELECTION_LOCK_BYTES: u64 = 1_048_576;

/// One configuration that was tried, and what it measured on canonical validation.
///
/// # The artifact hash is READ OUT of the evaluation, never supplied beside it
///
/// A candidate whose `artifact_hash` field could disagree with the artifact its metric was
/// computed on would be a lie with two halves that individually look fine. So there is no such
/// field: the evaluation already commits the hash it read off the verified run, and
/// [`Self::artifact_hash`] READS THAT ONE. The two cannot disagree because there is literally
/// only one of them — a copy taken at construction would have been a second value that a later
/// edit could move independently, which is the shape this doc claims not to have.
///
/// `config_hash` IS caller-supplied, and deliberately so — it is a LABEL identifying which
/// configuration produced the candidate, not the identity that gates access. A sweep does not
/// keep ten verified runs alive at once; the evaluation is the portable evidence, and the hash
/// that gates test access is the one inside it.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionCandidate {
    config_hash: String,
    evaluation: ValidationEvaluation,
}

impl SelectionCandidate {
    /// Build a candidate from its validation evaluation.
    #[must_use]
    pub fn from_evaluation(config_hash: &str, evaluation: ValidationEvaluation) -> Self {
        Self { config_hash: config_hash.to_string(), evaluation }
    }

    /// The label identifying the configuration that produced this candidate.
    #[must_use]
    pub fn config_hash(&self) -> &str {
        &self.config_hash
    }

    /// The artifact this candidate's metric was computed with.
    ///
    /// Delegates to the evaluation rather than to a copy of it, so this accessor cannot drift
    /// from the hash the evaluator committed.
    #[must_use]
    pub fn artifact_hash(&self) -> &str {
        self.evaluation.artifact_hash()
    }

    /// The candidate's canonical-validation evaluation.
    #[must_use]
    pub const fn evaluation(&self) -> &ValidationEvaluation {
        &self.evaluation
    }
}

/// The closed set of deterministic selection rules.
///
/// A rule is part of the lock's canonical bytes, so changing which rule was applied changes the
/// lock hash. There is one member because there is one contracted rule; a second would be a
/// contract edit, which is the point of making this an enum rather than a closure.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SelectionRule {
    /// Highest metric value wins; an exact tie goes to the LOWEST candidate index.
    ///
    /// Deterministic in both halves. The comparison is `f64::total_cmp`, which is a total order
    /// over every bit pattern including NaN, so the rule cannot depend on the order the
    /// candidates happened to arrive in even for values `partial_cmp` refuses to order.
    MaxMetricLowestIndexTieBreak,
}

impl SelectionRule {
    /// The rule's stable tag.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::MaxMetricLowestIndexTieBreak => "max_metric_lowest_index_tie_break",
        }
    }

    /// Apply the rule to a NON-EMPTY candidate list, returning the winning index.
    fn apply(self, candidates: &[SelectionCandidate]) -> usize {
        match self {
            Self::MaxMetricLowestIndexTieBreak => {
                // The incumbent's VALUE is carried rather than re-read by index each iteration.
                // The re-read's `NEG_INFINITY` fallback fired once PER ITERATION and its failure
                // mode was inverted: had it ever been taken, every challenger would have won.
                // Seeding once moves the same unreachable arm to a place where taking it is
                // inert — an empty list skips the loop entirely and index 0 is returned, which
                // is the index `from_candidates` has already refused to produce.
                let mut best = 0_usize;
                let mut best_value =
                    candidates.first().map_or(f64::NEG_INFINITY, |c| c.evaluation().value());
                for (index, candidate) in candidates.iter().enumerate().skip(1) {
                    let challenger = candidate.evaluation().value();
                    // STRICTLY greater, so an exact tie leaves the earlier index in place.
                    if challenger.total_cmp(&best_value) == core::cmp::Ordering::Greater {
                        best = index;
                        best_value = challenger;
                    }
                }
                best
            }
        }
    }
}

impl core::fmt::Display for SelectionRule {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.tag())
    }
}

/// The committed record of a selection decision.
///
/// Every field below is inside the hash. See the module docs for what a caller cannot do.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionLock {
    schema_version: u32,
    rule: SelectionRule,
    candidates: Vec<SelectionCandidate>,
    chosen_index: usize,
    dataset_fingerprint: String,
    validation_split_fingerprint: String,
    selection_semantic_hash: String,
    ledger_hash: String,
    lock_hash: String,
}

impl SelectionLock {
    /// Commit a candidate set, APPLY the rule, and bind the whole record to a hash.
    ///
    /// `pub(super)` rather than `pub`: the selection semantic hash and the ledger hash are
    /// PROVENANCE, and the only honest source for them is a run that actually executed. A
    /// public four-argument door would let a caller supply provenance for a run that never
    /// happened, which is the same class of defect as a caller-supplied metric. The public door
    /// is [`SetFitRun::create_selection_lock`], which reads both off the run.
    ///
    /// There is deliberately NO `chosen` parameter — see the module docs.
    ///
    /// # Errors
    ///
    /// [`LockError::NoCandidates`] for an empty list; [`LockError::MetricKindMismatch`],
    /// [`LockError::ValidationSplitFingerprintMismatch`],
    /// [`LockError::DatasetFingerprintMismatch`] and [`LockError::DuplicateArtifactHash`] for a
    /// candidate set that is not internally comparable, each naming the offending index.
    pub(super) fn from_candidates(
        candidates: Vec<SelectionCandidate>,
        rule: SelectionRule,
        selection_semantic_hash: &str,
        ledger_hash: &str,
    ) -> Result<Self, LockError> {
        // Only the three facts the comparison and the record actually need are taken, rather
        // than cloning the whole reference evaluation: the clone existed to release the borrow
        // before `candidates` moves into `Self`, and then had two of its own fields re-allocated
        // at the struct literal below. These two Strings are moved into the record instead.
        let reference = candidates.first().ok_or(LockError::NoCandidates)?.evaluation();
        let reference_kind = reference.metric_kind();
        let reference_split_fingerprint = reference.validation_split_fingerprint().to_string();
        let reference_dataset_fingerprint = reference.dataset_fingerprint().to_string();

        // (0) Every candidate's value must be FINITE, checked before anything else because a
        //     non-finite value is a defect in the evaluation rather than a disagreement between
        //     evaluations. `f64::total_cmp` is a total order that ranks +NaN above +infinity, so
        //     an admitted NaN would be SELECTED rather than ignored; the rule cannot refuse it
        //     without ceasing to be a total order, so admission is the only place this can live.
        for (index, candidate) in candidates.iter().enumerate() {
            let value = candidate.evaluation().value();
            if !value.is_finite() {
                return Err(LockError::NonFiniteMetric { index, value_bits: value.to_bits() });
            }
        }

        // (1) Every candidate must be COMPARABLE with the first. Each check names the offending
        //     index, because "some candidate disagreed" is not investigable.
        for (index, candidate) in candidates.iter().enumerate().skip(1) {
            let evaluation = candidate.evaluation();
            if evaluation.metric_kind() != reference_kind {
                return Err(LockError::MetricKindMismatch {
                    index,
                    expected: reference_kind,
                    observed: evaluation.metric_kind(),
                });
            }
            if evaluation.validation_split_fingerprint() != reference_split_fingerprint {
                return Err(LockError::ValidationSplitFingerprintMismatch {
                    index,
                    expected: reference_split_fingerprint,
                    observed: evaluation.validation_split_fingerprint().to_string(),
                });
            }
            if evaluation.dataset_fingerprint() != reference_dataset_fingerprint {
                return Err(LockError::DatasetFingerprintMismatch {
                    index,
                    expected: reference_dataset_fingerprint,
                    observed: evaluation.dataset_fingerprint().to_string(),
                });
            }
        }

        // (2) One artifact cannot be two candidates. A duplicate would let a single model take
        //     two places in the ranking, which changes what a tie-break means.
        let mut seen: ArtifactIndex<'_> = ArtifactIndex::new();
        for (index, candidate) in candidates.iter().enumerate() {
            if let Some(&first_index) = seen.get(candidate.artifact_hash()) {
                return Err(LockError::DuplicateArtifactHash {
                    index,
                    first_index,
                    artifact_hash: candidate.artifact_hash().to_string(),
                });
            }
            seen.insert(candidate.artifact_hash(), index);
        }
        drop(seen);

        // (3) The RULE picks the winner. There is no parameter above that could have named one.
        let chosen_index = rule.apply(&candidates);

        let mut lock = Self {
            schema_version: LOCK_SCHEMA_VERSION,
            rule,
            candidates,
            chosen_index,
            dataset_fingerprint: reference_dataset_fingerprint,
            validation_split_fingerprint: reference_split_fingerprint,
            selection_semantic_hash: selection_semantic_hash.to_string(),
            ledger_hash: ledger_hash.to_string(),
            // Filled immediately below, from the record that now exists. Hashing a partially
            // built record and then finishing it would digest something the reader cannot see.
            lock_hash: String::new(),
        };
        lock.lock_hash = lock.recompute_lock_hash();
        Ok(lock)
    }

    /// The rule that was applied.
    #[must_use]
    pub const fn rule(&self) -> SelectionRule {
        self.rule
    }

    /// The committed schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Every candidate, in the order they were committed.
    #[must_use]
    pub fn candidates(&self) -> &[SelectionCandidate] {
        &self.candidates
    }

    /// The index the rule selected.
    #[must_use]
    pub const fn chosen_index(&self) -> usize {
        self.chosen_index
    }

    /// The winning candidate.
    ///
    /// # Panics
    ///
    /// Never: `chosen_index` is produced by [`SelectionRule::apply`] over a list this type has
    /// already refused to build empty, and no API mutates the candidate list afterwards.
    #[must_use]
    pub fn chosen(&self) -> &SelectionCandidate {
        self.candidates
            .get(self.chosen_index)
            .expect("chosen_index is derived from a non-empty candidate list that never shrinks")
    }

    /// The artifact hash a token may be minted against.
    #[must_use]
    pub fn chosen_artifact_hash(&self) -> &str {
        self.chosen().artifact_hash()
    }

    /// The dataset fingerprint every candidate agreed on.
    #[must_use]
    pub fn dataset_fingerprint(&self) -> &str {
        &self.dataset_fingerprint
    }

    /// The validation-split fingerprint every candidate agreed on.
    #[must_use]
    pub fn validation_split_fingerprint(&self) -> &str {
        &self.validation_split_fingerprint
    }

    /// The selection's semantic hash, read off the run that created the lock.
    #[must_use]
    pub fn selection_semantic_hash(&self) -> &str {
        &self.selection_semantic_hash
    }

    /// The access ledger's hash, read off the run that created the lock.
    #[must_use]
    pub fn ledger_hash(&self) -> &str {
        &self.ledger_hash
    }

    /// SHA-256 of [`Self::to_canonical_bytes`], recorded at construction.
    #[must_use]
    pub fn lock_hash(&self) -> &str {
        &self.lock_hash
    }

    /// Deterministic canonical serialization — compact JSON over structs with a fixed field
    /// order, no map to iterate and no wall-clock value to drift.
    ///
    /// # Panics
    ///
    /// Never: the wire form is integers, strings and unit enum variants, so `serde_json` has no
    /// failure mode to report. The metric value travels as its BIT PATTERN rather than as a
    /// float, which is what removes the one arm that could have failed.
    #[must_use]
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let wire = SelectionLockWire {
            schema_version: self.schema_version,
            rule: self.rule,
            candidates: self
                .candidates
                .iter()
                .map(|candidate| SelectionCandidateWire {
                    config_hash: candidate.config_hash.clone(),
                    artifact_hash: candidate.artifact_hash().to_string(),
                    evaluation: ValidationEvaluationWire::from(candidate.evaluation.clone()),
                })
                .collect(),
            chosen_index: self.chosen_index as u64,
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            validation_split_fingerprint: self.validation_split_fingerprint.clone(),
            selection_semantic_hash: self.selection_semantic_hash.clone(),
            ledger_hash: self.ledger_hash.clone(),
        };
        serde_json::to_vec(&wire)
            .expect("the lock's canonical form is integers, strings and unit variants")
    }

    /// Reconstruct a lock from canonical bytes a PRIOR process wrote (review B4, TRN-07).
    ///
    /// # Why this exists
    ///
    /// [`Self::to_canonical_bytes`] was public and nothing could read those bytes back, so a
    /// second process — `apr eval --split test` — had no lock to consume. The only shape that
    /// compiled was minting the lock INSIDE the test command, which satisfies "a lock existed
    /// before test access" with an object created one line earlier and therefore satisfies
    /// nothing. Validation evaluation writes the lock; test evaluation reads it here.
    ///
    /// # The trust model, stated exactly and no stronger
    ///
    /// This file is an INTEGRITY-CHECKED RECORD, not an unforgeable credential. `lock_hash` is
    /// the digest OF THESE VERY BYTES, so a hand-written file that is internally consistent is
    /// possible and its own integrity check will pass. What the mechanism guarantees is narrower
    /// and still worth having: canonical-test access cannot be reached without a recorded,
    /// hash-committing selection decision that names the candidates and the chosen artifact, and
    /// every downstream door re-checks identity against the REAL run and the REAL dataset —
    /// [`Self::mint_test_token`] reads the artifact hash off the run it is handed, and
    /// [`CanonicalTestAccess::grant`] compares the dataset fingerprint against the dataset it is
    /// handed. An edited lock still cannot make a mismatched artifact or corpus pass; it just
    /// fails one door later, typed. This is the wording of `setfit-apr-v1`'s
    /// `selection_lock_lifecycle` invariants, and it is deliberately not upgraded here.
    ///
    /// # This does not widen the CREATION door
    ///
    /// [`Self::from_candidates`] stays `pub(super)`: creating a lock still requires a run, whose
    /// selection semantic hash and ledger hash are read off it rather than accepted as
    /// arguments. This door only READS a record such a run already created — and it rebuilds it
    /// THROUGH that same constructor, so a file cannot carry a candidate set the constructor
    /// would have refused, nor a winner the rule would not have picked.
    ///
    /// # Order is load-bearing
    ///
    /// 1. The input-length bound, on the raw slice, before serde is handed anything.
    /// 2. The parse, where `deny_unknown_fields` refuses a key this reader does not know.
    /// 3. The schema version, before any content is believed.
    /// 4. The candidate rebuild, where each artifact hash is taken from the EVALUATION rather
    ///    than from the wire's second copy of it.
    /// 5. The rebuild THROUGH [`Self::from_candidates`] — every consistency invariant and the
    ///    selection rule itself, from the single implementation.
    /// 6. The recorded winner: in range, then equal to the rule's.
    /// 7. The canonical-form check, which is what makes the list above total.
    /// 8. [`Self::verify_integrity`], and honest about being structural: the hash was derived
    ///    from this very record, so it proves self-consistency, not provenance.
    ///
    /// This list is the eight steps the body performs, numbered as the body numbers them. A
    /// doc that collapsed them would be a second, shorter description of an ordered ladder —
    /// exactly the drift the rest of this file works to prevent.
    ///
    /// # Errors
    ///
    /// [`LockError::LockPayloadTooLarge`] naming the limit and the observed length,
    /// [`LockError::MalformedLockPayload`], [`LockError::UnsupportedLockSchemaVersion`] naming
    /// both versions, [`LockError::ChosenIndexOutOfRange`],
    /// [`LockError::NonCanonicalLockPayload`], plus every candidate-consistency failure
    /// [`Self::from_candidates`] reports.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, LockError> {
        // (1) The bound, on the RAW slice. Before serde, so a payload that would be expensive to
        //     parse is refused by a comparison rather than by an allocator.
        let observed = bytes.len() as u64;
        if observed > MAX_SELECTION_LOCK_BYTES {
            return Err(LockError::LockPayloadTooLarge {
                limit: MAX_SELECTION_LOCK_BYTES,
                observed,
            });
        }

        // (2) The parse.
        let wire: SelectionLockWire = serde_json::from_slice(bytes)
            .map_err(|error| LockError::MalformedLockPayload { detail: error.to_string() })?;

        // (3) The version, before any of the record's contents are believed.
        if wire.schema_version != LOCK_SCHEMA_VERSION {
            return Err(LockError::UnsupportedLockSchemaVersion {
                got: wire.schema_version,
                supported: LOCK_SCHEMA_VERSION,
            });
        }

        // (4) Rebuild the candidates. The artifact hash comes from the EVALUATION, exactly as
        //     `SelectionCandidate::artifact_hash` reads it, so the wire's readable top-level copy
        //     cannot become a second answer. A file whose two copies disagree is caught at step
        //     (6) rather than silently normalized to whichever one this loop happened to prefer.
        let recorded_index = wire.chosen_index;
        let candidates: Vec<SelectionCandidate> = wire
            .candidates
            .into_iter()
            .map(|candidate| {
                SelectionCandidate::from_evaluation(
                    &candidate.config_hash,
                    ValidationEvaluation::from_wire(candidate.evaluation),
                )
            })
            .collect();

        // (5) Rebuild THROUGH the constructor, so the finiteness, comparability, duplicate and
        //     non-empty invariants — and the RULE — are the same single implementation a freshly
        //     created lock went through. A reader that re-implemented them would be a second
        //     answer that could drift; one that skipped them would admit a file the constructor
        //     would have refused.
        let lock = Self::from_candidates(
            candidates,
            wire.rule,
            &wire.selection_semantic_hash,
            &wire.ledger_hash,
        )?;

        // (6) The recorded winner must name a candidate. Checked AFTER the rebuild so an empty
        //     candidate list is reported as `NoCandidates` — the truer statement — rather than as
        //     an out-of-range index into a list that does not exist.
        let count = lock.candidates.len() as u64;
        if recorded_index >= count {
            return Err(LockError::ChosenIndexOutOfRange {
                chosen_index: recorded_index,
                candidates: lock.candidates.len(),
            });
        }

        // (7) The canonical-form check, and it is what makes the steps above TOTAL. Everything
        //     the file said that this reconstruction did not preserve shows up here as a byte
        //     difference: a winner the rule did not pick, top-level fingerprints that disagree
        //     with candidate zero's, a candidate whose two artifact-hash copies differ, a drifted
        //     evaluation schema version, a duplicated JSON key, a pretty-printed rendering.
        //     Enumerating those as separate checks would be a list that drifts from the wire
        //     form; comparing against the canonical serialization cannot.
        //
        //     It is also what makes the recomputed `lock_hash` the SAME hash the writing process
        //     recorded, rather than a fresh opinion about a payload silently repaired on the way
        //     in — which would be a lock whose digest matched nothing anybody wrote down.
        let canonical = lock.to_canonical_bytes();
        if canonical != bytes {
            return Err(LockError::NonCanonicalLockPayload {
                observed_len: bytes.len(),
                canonical_len: canonical.len(),
                first_difference_at: first_difference(bytes, &canonical),
            });
        }

        // (8) Structural, and honest about being so: the hash was derived from this very record
        //     three lines ago, so this cannot fail today. It is here so this door cannot become
        //     the one path that returns a lock whose recorded hash was never derived from its
        //     contents. The TAMPER check is not this line — it is that an edited file's hash
        //     CHANGES, and that every downstream door re-checks identity against the real run.
        lock.verify_integrity()?;
        Ok(lock)
    }

    /// Recompute the lock hash from the record as it stands now.
    #[must_use]
    pub fn recompute_lock_hash(&self) -> String {
        hex::encode(Sha256::digest(self.to_canonical_bytes()))
    }

    /// Refuse a record whose contents no longer produce its recorded hash.
    ///
    /// # Errors
    ///
    /// [`LockError::LockHashMismatch`] naming both digests.
    pub fn verify_integrity(&self) -> Result<(), LockError> {
        let recomputed = self.recompute_lock_hash();
        if recomputed == self.lock_hash {
            return Ok(());
        }
        Err(LockError::LockHashMismatch { recorded: self.lock_hash.clone(), recomputed })
    }

    /// Mint a canonical-test token for the model this lock CHOSE.
    ///
    /// # It takes a CREDENTIAL OBJECT, not bytes
    ///
    /// A `[u8; 32]` parameter here would let a caller pass the locked hash and then evaluate a
    /// different artifact: the check would pass, the token would be valid, and nothing
    /// downstream would ever see the substitution. Reading `artifact_hash()` off the supplied
    /// credential makes the identity a property of the object being granted access.
    ///
    /// [`SetFitCredential`] is SEALED, so widening this parameter from the concrete
    /// `&SetFitRun<ArtifactReloadedAndVerified>` (04-17 G2) did not widen who can satisfy it:
    /// an out-of-crate type that returned three strings of its choosing would be exactly the
    /// byte-array parameter above wearing a struct, and it does not compile —
    /// `tests/ui/setfit_external_credential_impl.rs`.
    ///
    /// # Errors
    ///
    /// [`LockError::LockHashMismatch`] for a record that no longer hashes to its digest, and
    /// [`LockError::StaleLock`] naming BOTH hashes when the supplied model is not the locked
    /// one — which is what a lock-then-keep-tuning-then-test sequence produces.
    pub fn mint_test_token<C: SetFitCredential>(
        &self,
        model: &C,
    ) -> Result<CanonicalTestToken, LockError> {
        // FIRST, before the identity comparison: a record that no longer hashes to its digest is
        // not a lock, and comparing against a field somebody may have edited would make the
        // comparison a formality.
        self.verify_integrity()?;

        // READ OFF the model. There is no parameter here a caller could have supplied.
        let observed = model.artifact_hash();
        if self.chosen_artifact_hash() != observed {
            return Err(LockError::StaleLock {
                locked: self.chosen_artifact_hash().to_string(),
                observed,
            });
        }
        Ok(CanonicalTestToken {
            lock_hash: self.lock_hash.clone(),
            artifact_hash: observed,
            dataset_fingerprint: self.dataset_fingerprint.clone(),
            validation_split_fingerprint: self.validation_split_fingerprint.clone(),
        })
    }

    /// Overwrite the recorded chosen index WITHOUT re-deriving the hash.
    ///
    /// `#[cfg(test)]`, and nothing weaker. This is the forgery [`Self::verify_integrity`] exists
    /// to catch, and a shipped build must not contain a door that produces one. It is the lock's
    /// analogue of 03-08's `EchoCodec`: a negative that has only ever been described is not a
    /// negative (Ph1 D-24 / Ph2 D-25).
    #[cfg(test)]
    pub(super) fn forge_chosen_index_for_tests(&mut self, chosen_index: usize) {
        self.chosen_index = chosen_index;
    }
}

/// The lock's own door on the run.
///
/// # Why this block lives HERE and not beside the reproducibility accessors in `mod.rs`
///
/// `mod.rs`'s `impl SetFitRun<ArtifactReloadedAndVerified>` block is THE reproducibility
/// surface, and `verify_reproducibility_accessors_are_read_only_and_complete` counts its `pub
/// fn`s and requires every one to be exercised through a shared reference — an exhaustiveness
/// guard that a twelfth member would have to join. `create_selection_lock` is not a
/// reproducibility accessor: it takes arguments, it can fail, and it constructs a new object.
/// Putting it in that block would have meant either weakening the guard's meaning or adding a
/// candidate-building fixture to a test whose subject is read-only accessors.
///
/// Moving it here is NOT dodging the count. `lock_run_side_door_is_a_single_read_only_method`
/// counts this block too, so both surfaces are guarded and neither can grow unobserved.
impl SetFitRun<ArtifactReloadedAndVerified> {
    /// Commit this run's selection decision, BEFORE canonical test access is permitted (D-14).
    ///
    /// # This is the only public door to a lock
    ///
    /// The selection semantic hash and the access ledger's hash are PROVENANCE, and this method
    /// reads both off the run rather than accepting them — [`SelectionLock::from_candidates`] is
    /// `pub(super)` for exactly that reason. A public constructor taking provenance strings would
    /// let a caller mint a lock describing a run that never executed, which is the same class of
    /// defect as a caller-supplied metric value.
    ///
    /// # The creating run must be among the candidates
    ///
    /// A lock whose candidate set does not contain the model that created it is not the record of
    /// that model's selection, and a token minted from it could never match — so it is refused
    /// here rather than becoming an unmintable lock somebody debugs later.
    ///
    /// # Errors
    ///
    /// [`LockError::ChosenModelNotACandidate`] when this run's artifact is absent from the
    /// candidate set, plus every candidate-consistency failure `from_candidates` reports.
    pub fn create_selection_lock(
        &self,
        candidates: Vec<SelectionCandidate>,
        rule: SelectionRule,
    ) -> Result<SelectionLock, LockError> {
        self::create_selection_lock(self, candidates, rule)
    }
}

/// Commit a selection decision from any [`SetFitCredential`] — the ONE implementation.
///
/// # Why this is a free function and the method above delegates to it
///
/// The other two doors ([`SelectionLock::mint_test_token`], [`CanonicalTestAccess::grant`])
/// are already free-standing enough to take a generic parameter. `create_selection_lock` was
/// an inherent method on the train-time run, and 04-17 needed it reachable from a fresh
/// process too. Copying the body into a generic function would have produced TWO
/// candidate-membership checks — and the day one of them gained a condition the other did
/// not, a lock created one way would accept a candidate set the other way refused. OPS-03
/// says one implementation per operation; this is it, and the method above is a one-line
/// forward so every existing call site keeps its spelling and its behaviour.
///
/// # Errors
///
/// [`LockError::ChosenModelNotACandidate`] when the credential's artifact is absent from the
/// candidate set, plus every candidate-consistency failure
/// [`SelectionLock::from_candidates`] reports.
pub fn create_selection_lock<C: SetFitCredential>(
    model: &C,
    candidates: Vec<SelectionCandidate>,
    rule: SelectionRule,
) -> Result<SelectionLock, LockError> {
    let mine = model.artifact_hash();
    if !candidates.iter().any(|candidate| candidate.artifact_hash() == mine) {
        return Err(LockError::ChosenModelNotACandidate {
            artifact_hash: mine,
            candidates: candidates.len(),
        });
    }
    SelectionLock::from_candidates(
        candidates,
        rule,
        &model.selection_semantic_hash(),
        &hex::encode(model.selection_ledger_hash()),
    )
}

/// Proof that a lock was created and that its chosen artifact is a particular model.
///
/// Every field is private and there is no public constructor: reading a token's contents cannot
/// mint one, exactly as `SelectedId::ordinal` cannot mint a `SelectedId` (Phase 2 `select.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalTestToken {
    lock_hash: String,
    artifact_hash: String,
    dataset_fingerprint: String,
    validation_split_fingerprint: String,
}

impl CanonicalTestToken {
    /// The lock this token was minted from.
    #[must_use]
    pub fn lock_hash(&self) -> &str {
        &self.lock_hash
    }

    /// The artifact this token admits, read off the run at minting.
    #[must_use]
    pub fn artifact_hash(&self) -> &str {
        &self.artifact_hash
    }

    /// The dataset the selection was made on.
    #[must_use]
    pub fn dataset_fingerprint(&self) -> &str {
        &self.dataset_fingerprint
    }

    /// The validation split the selection was made on.
    #[must_use]
    pub fn validation_split_fingerprint(&self) -> &str {
        &self.validation_split_fingerprint
    }
}

/// The one door to the canonical test split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalTestAccess;

impl CanonicalTestAccess {
    /// Exchange a token, a model and the canonical dataset for a typed access witness.
    ///
    /// # The identity re-check is not redundant
    ///
    /// Minting checked the identity at ONE instant. A token is a value: it can be moved into a
    /// struct, cloned, returned from a helper and used against whatever model is in scope three
    /// functions later. Re-checking at access time is what makes the token's meaning "THIS
    /// artifact may see the test split" rather than "some artifact once matched a lock".
    ///
    /// # Why the parameter is the DATASET and not a `&Split<Test>`
    ///
    /// The phase-3 cleanup review found this door checking the artifact and nothing else while
    /// accepting a bare `&Split<Test>`. A valid token therefore admitted test rows from ANY
    /// corpus, and the grant it produced went on reporting the `lock_hash` of the corpus the lock
    /// was actually taken over — the model half of the substitution TRN-07 blocks was pinned and
    /// the data half was wide open.
    ///
    /// The repair had to be the ARGUMENT rather than an added assertion: a bare split carries no
    /// provenance, so there was nothing sound to compare it against. Taking the dataset gives
    /// this door the fingerprint the token has been carrying all along, and RETURNING
    /// `dataset.test()` is the load-bearing half — a version that checked the dataset and then
    /// admitted a separately-supplied split would verify one thing and hand out another.
    ///
    /// `Split<CompatibilityTest>` is a DIFFERENT type (Ph2 D-16/D-19) and the compatibility
    /// profile is a different `PreparedDataset` parameterisation with no `test()` at all, so a
    /// compatibility profile is now refused one step earlier — at the parameter, not inside.
    ///
    /// # The model parameter is a SEALED credential (04-17 G2)
    ///
    /// `model` widened from `&SetFitRun<ArtifactReloadedAndVerified>` to any
    /// [`SetFitCredential`] so that `apr eval --split test`, running in a process that never
    /// trained anything, can reach this door at all. The widening is safe for exactly one
    /// reason: the trait is sealed, so the set of types that satisfy it is this crate's to
    /// choose. An openly implementable trait here would be a caller-supplied artifact hash
    /// with extra steps.
    ///
    /// # Errors
    ///
    /// [`LockError::TokenModelMismatch`] naming both hashes, or
    /// [`LockError::TokenDatasetMismatch`] naming both dataset fingerprints.
    pub fn grant<'a, C: SetFitCredential>(
        token: CanonicalTestToken,
        model: &C,
        dataset: &'a PreparedDataset<Canonical>,
    ) -> Result<CanonicalTestGrant<'a>, LockError> {
        let model_artifact_hash = model.artifact_hash();
        if token.artifact_hash() != model_artifact_hash {
            return Err(LockError::TokenModelMismatch {
                token_artifact_hash: token.artifact_hash().to_string(),
                model_artifact_hash,
            });
        }
        let observed_dataset_fingerprint = dataset.validation_witness().dataset_fingerprint_hex();
        if token.dataset_fingerprint() != observed_dataset_fingerprint {
            return Err(LockError::TokenDatasetMismatch {
                token_dataset_fingerprint: token.dataset_fingerprint().to_string(),
                observed_dataset_fingerprint,
            });
        }
        Ok(CanonicalTestGrant { token, test: dataset.test() })
    }
}

/// A typed witness that canonical test access was EARNED by this model.
#[derive(Debug)]
pub struct CanonicalTestGrant<'a> {
    token: CanonicalTestToken,
    test: &'a Split<Test>,
}

impl<'a> CanonicalTestGrant<'a> {
    /// The canonical test split this grant admits.
    #[must_use]
    pub const fn test(&self) -> &'a Split<Test> {
        self.test
    }

    /// The token that was exchanged.
    #[must_use]
    pub const fn token(&self) -> &CanonicalTestToken {
        &self.token
    }

    /// The artifact this grant belongs to.
    #[must_use]
    pub fn artifact_hash(&self) -> &str {
        self.token.artifact_hash()
    }

    /// The lock this grant traces back to.
    #[must_use]
    pub fn lock_hash(&self) -> &str {
        self.token.lock_hash()
    }
}

// ===========================================================================================
// The canonical wire forms
// ===========================================================================================

/// One candidate's canonical form.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionCandidateWire {
    config_hash: String,
    artifact_hash: String,
    evaluation: ValidationEvaluationWire,
}

/// The lock's canonical form. A struct, not a map, so field order is fixed by the type.
///
/// `lock_hash` is deliberately absent: it is the SHA-256 OF these bytes, and a record that
/// contained its own digest could not be checked against one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SelectionLockWire {
    schema_version: u32,
    rule: SelectionRule,
    candidates: Vec<SelectionCandidateWire>,
    chosen_index: u64,
    dataset_fingerprint: String,
    validation_split_fingerprint: String,
    selection_semantic_hash: String,
    ledger_hash: String,
}

// ===========================================================================================
// Failures
// ===========================================================================================

/// Failure modes of the selection lock and the canonical-test token.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LockError {
    /// The candidate list was empty, so there was no selection to commit.
    NoCandidates,
    /// A candidate measured a different quantity from the first one.
    MetricKindMismatch {
        /// The offending candidate's index.
        index: usize,
        /// The metric the first candidate measured.
        expected: ValidationMetricKind,
        /// The metric this candidate measured.
        observed: ValidationMetricKind,
    },
    /// A candidate was evaluated on a different validation split.
    ValidationSplitFingerprintMismatch {
        /// The offending candidate's index.
        index: usize,
        /// The first candidate's validation-split fingerprint.
        expected: String,
        /// This candidate's validation-split fingerprint.
        observed: String,
    },
    /// A candidate was evaluated on a different dataset.
    DatasetFingerprintMismatch {
        /// The offending candidate's index.
        index: usize,
        /// The first candidate's dataset fingerprint.
        expected: String,
        /// This candidate's dataset fingerprint.
        observed: String,
    },
    /// A candidate's metric value is NaN or an infinity.
    ///
    /// Refused at admission rather than handled in the rule: `f64::total_cmp` is a TOTAL order
    /// and orders `+NaN` above `+infinity`, so a NaN candidate is not skipped by the comparison
    /// — it wins it. A total order cannot both stay total and reject a value, so the rejection
    /// belongs here (contract `selection_lock_commitment`).
    NonFiniteMetric {
        /// The offending candidate's index.
        index: usize,
        /// The non-finite value's IEEE-754 BITS.
        ///
        /// Bits rather than an `f64` for the same reason the wire form carries bits: this enum is
        /// `PartialEq + Eq`, `f64` is neither, and an `assert_eq!` against a variant holding a
        /// NaN could never match itself since `NaN != NaN`. The bits make the payload both
        /// comparable and exactly assertable, and `Display` renders them back.
        value_bits: u64,
    },
    /// Two candidates name the same artifact.
    DuplicateArtifactHash {
        /// The repeating candidate's index.
        index: usize,
        /// Where the same artifact first appeared.
        first_index: usize,
        /// The repeated artifact hash.
        artifact_hash: String,
    },
    /// The run creating the lock is not among the candidates it committed.
    ChosenModelNotACandidate {
        /// The creating run's artifact hash.
        artifact_hash: String,
        /// How many candidates were supplied.
        candidates: usize,
    },
    /// The supplied run is not the model this lock chose.
    StaleLock {
        /// The artifact hash the lock committed to.
        locked: String,
        /// The artifact hash read off the supplied run.
        observed: String,
    },
    /// A token was presented alongside a model it was not minted for.
    TokenModelMismatch {
        /// The artifact the token admits.
        token_artifact_hash: String,
        /// The artifact the supplied model actually is.
        model_artifact_hash: String,
    },
    /// A token was presented alongside a canonical dataset it was not minted over.
    ///
    /// The model half of the identity can be right while the CORPUS half is wrong: pinning the
    /// artifact and leaving the data free admits test rows from another dataset under a grant
    /// that still names the locked one.
    TokenDatasetMismatch {
        /// The dataset fingerprint the token (and therefore the lock) committed to.
        token_dataset_fingerprint: String,
        /// The dataset fingerprint of the dataset actually supplied.
        observed_dataset_fingerprint: String,
    },
    /// The lock record no longer produces its recorded hash.
    LockHashMismatch {
        /// The digest the lock carries.
        recorded: String,
        /// The digest its current contents produce.
        recomputed: String,
    },
    /// The serialized lock was longer than [`MAX_SELECTION_LOCK_BYTES`].
    ///
    /// Reported from the RAW slice, before serde is handed anything.
    LockPayloadTooLarge {
        /// The bound, in bytes.
        limit: u64,
        /// The input's length, in bytes.
        observed: u64,
    },
    /// The payload did not parse as a canonical selection lock.
    MalformedLockPayload {
        /// What the parser reported, including the offending key or byte position.
        detail: String,
    },
    /// The payload declares a schema version this reader does not implement.
    UnsupportedLockSchemaVersion {
        /// The version the payload declared.
        got: u32,
        /// The only version this reader implements.
        supported: u32,
    },
    /// The recorded winner names no candidate in the list the payload carried.
    ChosenIndexOutOfRange {
        /// The index the payload recorded.
        chosen_index: u64,
        /// How many candidates the payload carried.
        candidates: usize,
    },
    /// The payload is not the canonical serialization of the record it parses to.
    ///
    /// One refusal rather than a list, deliberately. Everything a file can say that a faithful
    /// reconstruction does not preserve — a winner the rule did not pick, top-level fingerprints
    /// disagreeing with candidate zero's, a candidate whose two artifact-hash copies differ, a
    /// drifted nested schema version, a duplicated JSON key, a pretty-printed rendering — is the
    /// same defect seen from a different angle: the bytes are not the canonical form of the
    /// record. Enumerating the angles would be a list that drifts from the wire form.
    NonCanonicalLockPayload {
        /// The payload's length, in bytes.
        observed_len: usize,
        /// The length of the canonical serialization of the record it parsed to.
        canonical_len: usize,
        /// The byte offset of the first difference, for locating the edit.
        first_difference_at: usize,
    },
}

impl core::fmt::Display for LockError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoCandidates => write!(
                f,
                "a selection lock needs at least one candidate; an empty list records no \
                 decision and therefore evidences nothing \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
            ),
            Self::MetricKindMismatch { index, expected, observed } => write!(
                f,
                "candidate {index} measured `{observed}` where the first candidate measured \
                 `{expected}`; a selection across two different quantities is not a selection \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
            ),
            Self::ValidationSplitFingerprintMismatch { index, expected, observed } => write!(
                f,
                "candidate {index} was evaluated on validation split `{observed}` where the \
                 first candidate used `{expected}`; comparing metrics computed on different \
                 rows ranks the splits, not the models \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
            ),
            Self::DatasetFingerprintMismatch { index, expected, observed } => write!(
                f,
                "candidate {index} was evaluated on dataset `{observed}` where the first \
                 candidate used `{expected}` \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
            ),
            Self::NonFiniteMetric { index, value_bits } => write!(
                f,
                "candidate {index} carries a non-finite metric value ({}); `f64::total_cmp` \
                 orders +NaN ABOVE +infinity, so admitting one would not make the ordering \
                 meaningless -- it would make that candidate WIN \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
                f64::from_bits(*value_bits),
            ),
            Self::DuplicateArtifactHash { index, first_index, artifact_hash } => write!(
                f,
                "candidate {index} names artifact `{artifact_hash}`, which candidate \
                 {first_index} already named; one artifact cannot be two candidates, and a \
                 duplicate would let one model occupy two places in the ranking \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
            ),
            Self::ChosenModelNotACandidate { artifact_hash, candidates } => write!(
                f,
                "the run creating this lock has artifact hash `{artifact_hash}`, which is not \
                 among the {candidates} candidates it committed; a lock that does not include \
                 the model that created it cannot be the record of that model's selection \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
            ),
            Self::StaleLock { locked, observed } => write!(
                f,
                "this lock committed artifact `{locked}` but the supplied run is `{observed}`; \
                 the model was tuned further after the lock was taken, so the lock is no longer \
                 a record of THIS model's selection and canonical test access is refused \
                 (contract setfit-train-lifecycle-v1, equation canonical_test_token_minting)",
            ),
            Self::TokenModelMismatch { token_artifact_hash, model_artifact_hash } => write!(
                f,
                "this token admits artifact `{token_artifact_hash}` but was presented with \
                 model `{model_artifact_hash}`; a token that travelled cannot be paired with a \
                 different model \
                 (contract setfit-train-lifecycle-v1, equation canonical_test_token_minting)",
            ),
            Self::TokenDatasetMismatch {
                token_dataset_fingerprint,
                observed_dataset_fingerprint,
            } => write!(
                f,
                "this token was minted over dataset `{token_dataset_fingerprint}` but was \
                 presented with dataset `{observed_dataset_fingerprint}`; the model may be the \
                 one the lock chose, but the corpus is not the one it was chosen on \
                 (contract setfit-train-lifecycle-v1, equation canonical_test_token_minting)",
            ),
            Self::LockHashMismatch { recorded, recomputed } => write!(
                f,
                "the lock record carries hash `{recorded}` but its contents now produce \
                 `{recomputed}`; the record was edited after it was committed, and an \
                 append-only log that can be rewritten is not evidence \
                 (contract setfit-train-lifecycle-v1, equation selection_lock_commitment)",
            ),
            Self::LockPayloadTooLarge { limit, observed } => write!(
                f,
                "the selection lock payload is {observed} bytes, over the {limit}-byte limit; \
                 refused on the raw input before it was parsed, so a payload claiming millions \
                 of candidates cannot make the allocation itself the attack \
                 (contract setfit-apr-v1, equation selection_lock_lifecycle)",
            ),
            Self::MalformedLockPayload { detail } => write!(
                f,
                "the selection lock payload did not parse: {detail}; a reader that guessed at a \
                 record it could not read would be believing a lock nobody wrote \
                 (contract setfit-apr-v1, equation selection_lock_lifecycle)",
            ),
            Self::UnsupportedLockSchemaVersion { got, supported } => write!(
                f,
                "the selection lock declares schema version {got}, but this reader implements \
                 version {supported}; a reader that met a version it did not know and continued \
                 would be interpreting fields that may have been re-meant \
                 (contract setfit-apr-v1, equation selection_lock_lifecycle)",
            ),
            Self::ChosenIndexOutOfRange { chosen_index, candidates } => write!(
                f,
                "the selection lock records chosen index {chosen_index}, but it carries only \
                 {candidates} candidates; the recorded winner names no candidate at all \
                 (contract setfit-apr-v1, equation selection_lock_lifecycle)",
            ),
            Self::NonCanonicalLockPayload { observed_len, canonical_len, first_difference_at } => {
                write!(
                f,
                "the payload is {observed_len} bytes but the record it parses to serializes to \
                 {canonical_len} canonical bytes, first differing at offset \
                 {first_difference_at}; the file therefore claims something the reconstruction \
                 does not reproduce -- a winner the rule did not pick, a fingerprint disagreeing \
                 with the candidates it summarizes, or a rendering that is not the canonical one \
                 -- and a lock whose bytes are not its canonical form has a hash nobody wrote \
                 down (contract setfit-apr-v1, equation selection_lock_lifecycle)",
            )
            }
        }
    }
}

impl std::error::Error for LockError {}

/// The byte offset of the first difference between two slices.
///
/// Falls back to the shorter length when one is a prefix of the other, which is the offset a
/// reader would look at first anyway. Only ever called on a pair already known to differ.
fn first_difference(left: &[u8], right: &[u8]) -> usize {
    left.iter()
        .zip(right.iter())
        .position(|(l, r)| l != r)
        .unwrap_or_else(|| left.len().min(right.len()))
}

/// A `BTreeMap` rather than a `HashMap` for the duplicate scan: the iteration order never
/// affects the reported index, because the scan reports the FIRST occurrence it recorded.
type ArtifactIndex<'a> = BTreeMap<&'a str, usize>;

#[cfg(test)]
#[path = "lock_tests.rs"]
mod lock_tests;
