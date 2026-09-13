//! Balanced few-shot selection and the ordered selected-ID manifest model.
//!
//! Partial Fisher-Yates over sorted per-class buckets, with the swap index for step *i*
//! drawn at ordinal *i* in that class's domain. The output order IS the draw order, which
//! is what makes the ordered manifest statable as a contract equation rather than as an
//! implementation detail.
//!
//! # One dataset value in, one selection out
//!
//! [`FewShotSelector::select`] takes a single `&PreparedDataset<Canonical>`. That one
//! argument supplies the training pool, the validation witness, the exclusion record and
//! the fingerprint, so there is no signature into which a caller could feed a training
//! split from one dataset and a validation split from another. Witness mixing is not
//! rejected here; it is unrepresentable. A compatibility dataset is a different type and
//! cannot be passed at all (D-19).
//!
//! # Every selected row carries its LABEL
//!
//! [`SelectedExample`] holds `(id, label, exact_hash, normalized_hash)` and [`SelectedId`]
//! is an opaque ordinal into one [`Selection`]. Downstream pair construction therefore
//! derives its targets from real per-row labels rather than from class-size totals, which
//! would silently mislabel any layout the totals happen to be symmetric in.
//!
//! # Determinism, and what it does and does not survive
//!
//! The ordered selection is a pure function of `(post-exclusion sorted pools, root_seed,
//! shots_per_class)`. It survives thread count, iteration order and permuted ingest order,
//! because the buckets sort before any draw and draw *i* is a pure function of *i*.
//!
//! The `semantic_hash` deliberately does NOT survive permuted ingest order, and that is
//! correct rather than a gap: the payload embeds `dataset_fingerprint`, which is partly a
//! digest of the split's canonical JSONL bytes *in ingest order*. A permuted file is
//! different bytes and therefore a different dataset for provenance purposes. The
//! permutation test below asserts both halves — same selection, different fingerprint —
//! so neither can regress unnoticed.

use core::num::NonZeroU64;
use std::collections::{BTreeMap, BTreeSet};

use sha2::{Digest, Sha256};

use crate::buckets::ClassBuckets;
use crate::error::ContrastiveDataError;
use crate::hash::{hex, CONTENT_NORMALIZATION_VERSION};
use crate::ledger::AccessLedger;
use crate::manifest::{
    SelectedExampleRecord, SelectionManifest, SelectionPayload, SELECTION_SCHEMA_VERSION,
    SUPPORTED_SELECTION_SCHEMA_VERSIONS,
};
use crate::prepared::{Canonical, DatasetProfile, PreparedDataset};
use crate::rng::{bounded, derive_key, domains};
use crate::split::{SplitRole, Train};

/// The selection-algorithm version. A change here changes selected IDENTITIES, which is
/// why it is versioned separately from the manifest schema.
pub const SELECTION_ALGORITHM_VERSION: u32 = 1;

/// The contracted shot counts.
const ALLOWED_SHOTS: [u32; 4] = [8, 16, 32, 64];

/// Rendered form of [`ALLOWED_SHOTS`] for the typed error.
const ALLOWED_SHOTS_TEXT: &str = "{8, 16, 32, 64}";

/// The access-ledger purpose recorded by a selection.
const SELECT_PURPOSE: &str = "select";

/// The access-ledger purpose recorded by a replay.
const REPLAY_PURPOSE: &str = "select-replay";

/// What a caller asks a selection for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SelectionConfig {
    /// The root seed. Every draw key is derived from this plus a domain string.
    pub root_seed: u64,
    /// Shots per class — one of 8, 16, 32, 64.
    pub shots_per_class: u32,
}

/// One selected row, with its LABEL and both content hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SelectedExample {
    /// The row identifier.
    pub id: String,
    /// The row's class label.
    pub label: usize,
    /// Exact SHA-256 of the raw input bytes.
    pub exact_hash: [u8; 32],
    /// `nfc-trim-ws-v1` normalized content hash.
    pub normalized_hash: [u8; 32],
}

/// An opaque ordinal into ONE [`Selection`].
///
/// The field is private and the constructor is private to this module, so a value of this
/// type can only have come from a `Selection`'s own accessors. That is what makes pair
/// endpoints structurally non-leaky downstream: there is no way to name a row that was
/// never selected.
///
/// An ordinal taken from one selection and applied to another is a programming error, not
/// a data error; the accessors panic with a named message rather than returning a
/// plausible row from the wrong selection.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SelectedId(u32);

impl SelectedId {
    /// The zero-based position of this row in its selection's ordered list.
    ///
    /// Exposed so pair bytes can be encoded and hashed by ORDINAL rather than by
    /// variable-length identifier — the pair manifest already commits the selection's
    /// semantic hash, so the ordinals are unambiguous inside it.
    ///
    /// This is an accessor, not a constructor: reading the number cannot mint a
    /// `SelectedId`, so the type remains proof of membership in the selection that
    /// produced it.
    pub fn ordinal(self) -> u32 {
        self.0
    }
}

/// A completed few-shot selection.
#[derive(Debug, Clone)]
pub struct Selection {
    ordered: Vec<SelectedExample>,
    by_id: BTreeMap<String, SelectedId>,
    by_class: BTreeMap<usize, Vec<SelectedId>>,
    class_sizes: Vec<(usize, u64)>,
    payload: SelectionPayload,
    semantic_hash: [u8; 32],
    ledger_hash: [u8; 32],
}

/// Balanced few-shot selection over one canonical prepared dataset.
#[derive(Debug, Clone, Copy)]
pub struct FewShotSelector;

impl FewShotSelector {
    /// Select `shots_per_class` examples from every declared class.
    ///
    /// Fail-closed first: an invalid shot count and an exhausted class pool are both
    /// rejected BEFORE any draw, so an invalid request never consumes an ordinal and never
    /// leaves a partially built selection behind.
    ///
    /// Then, per class in ASCENDING label order, a partial Fisher-Yates over that class's
    /// sorted post-exclusion pool: for step `i`, swap index
    /// `j = i + bounded(key, 0, i, pool_len - i)` where `key = derive_key(root_seed,
    /// "select/{label}")`. The first `shots_per_class` slots IN DRAW ORDER are that class's
    /// ordered selection, and the classes concatenated in ascending label order ARE the
    /// manifest order DATA-03 contracts.
    ///
    /// Each row's two hashes are copied from the split that already computed them — never
    /// re-derived here, so there is exactly one source of truth for a row's identity.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::InvalidShots`] for a shot count outside `{8, 16, 32, 64}`;
    /// [`ContrastiveDataError::CrossSplitDuplicateUnderflow`] when a class's post-exclusion
    /// pool cannot supply the requested shots (D-27's pool-exhaustion half);
    /// [`ContrastiveDataError::Serialization`] if the canonical payload cannot be built.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "few_shot_selection"
    )]
    pub fn select(
        dataset: &PreparedDataset<Canonical>,
        cfg: &SelectionConfig,
        ledger: &mut AccessLedger,
    ) -> Result<Selection, ContrastiveDataError> {
        let ordered = compute_ordered(dataset, cfg.root_seed, cfg.shots_per_class)?;

        let dataset_fingerprint = dataset.fingerprint().hex();
        ledger.record(
            Train::ROLE,
            Canonical::PROFILE,
            SELECT_PURPOSE,
            &dataset_fingerprint,
        );
        let ledger_hash = ledger.ledger_hash();

        let payload = SelectionPayload {
            schema_version: SELECTION_SCHEMA_VERSION,
            algorithm_version: SELECTION_ALGORITHM_VERSION,
            profile: Canonical::PROFILE.to_string(),
            dataset_fingerprint,
            validation_fingerprint: dataset.validation_witness().fingerprint_hex(),
            label_names: dataset.label_names().to_vec(),
            normalization_version: CONTENT_NORMALIZATION_VERSION.to_string(),
            root_seed: cfg.root_seed,
            shots_per_class: cfg.shots_per_class,
            ordered_examples: ordered
                .iter()
                .map(SelectedExampleRecord::from_example)
                .collect(),
            exclusions: dataset.exclusions().clone(),
            access_ledger: ledger.records().to_vec(),
            ledger_hash: hex(&ledger_hash),
        };

        Selection::assemble(ordered, payload, ledger_hash)
    }
}

/// The pure selection: no ledger, no payload, no hashing.
///
/// Split out because `Selection::replay` must recompute exactly this and compare, and it
/// cannot do so through `select` — `select` appends to the ledger, and a replay that
/// appended twice would produce a payload that disagrees with the manifest it is
/// validating.
pub(crate) fn compute_ordered(
    dataset: &PreparedDataset<Canonical>,
    root_seed: u64,
    shots_per_class: u32,
) -> Result<Vec<SelectedExample>, ContrastiveDataError> {
    if !ALLOWED_SHOTS.contains(&shots_per_class) {
        return Err(ContrastiveDataError::InvalidShots {
            got: shots_per_class as usize,
            allowed: ALLOWED_SHOTS_TEXT,
        });
    }
    let shots = shots_per_class as usize;

    let buckets = ClassBuckets::from_prepared(dataset);
    for (label, pool) in buckets.class_sizes() {
        if (pool as usize) < shots {
            return Err(ContrastiveDataError::CrossSplitDuplicateUnderflow {
                class_label: label,
                pool: pool as usize,
                shots,
            });
        }
    }

    let train = dataset.train();
    let labels = buckets.labels();
    let mut ordered = Vec::with_capacity(shots.saturating_mul(labels.len()));

    for label in labels {
        let key = derive_key(root_seed, &domains::select(label));
        let mut work: Vec<&str> = buckets.ids(label).iter().map(String::as_str).collect();
        let pool_len = work.len();

        for i in 0..shots {
            // Unreachable given the fail-closed check above (`pool_len >= shots > i`), but
            // typed rather than asserted so the invariant cannot be broken silently by a
            // future edit to the pre-check.
            let remaining = NonZeroU64::new((pool_len - i) as u64).ok_or_else(|| {
                ContrastiveDataError::ArithmeticOverflow {
                    operation: format!("selection_remaining_pool/class-{label}"),
                }
            })?;
            let j = i + bounded(&key, 0, i as u64, remaining) as usize;
            work.swap(i, j);
        }

        for id in work.into_iter().take(shots) {
            // Also unreachable: the bucket was built from this very split's rows.
            let missing = || ContrastiveDataError::SelectionReplayMismatch {
                field: format!("row_hashes/{id}"),
            };
            let exact_hash = *train.exact_hash_of(id).ok_or_else(missing)?;
            let normalized_hash = *train.normalized_hash_of(id).ok_or_else(missing)?;
            ordered.push(SelectedExample {
                id: id.to_string(),
                label,
                exact_hash,
                normalized_hash,
            });
        }
    }

    Ok(ordered)
}

impl Selection {
    /// Build the value from an ordered list and the payload it is attested by.
    pub(crate) fn assemble(
        ordered: Vec<SelectedExample>,
        payload: SelectionPayload,
        ledger_hash: [u8; 32],
    ) -> Result<Self, ContrastiveDataError> {
        let semantic_hash: [u8; 32] = Sha256::digest(payload.to_canonical_bytes()?).into();

        let mut by_id = BTreeMap::new();
        let mut by_class: BTreeMap<usize, Vec<SelectedId>> = BTreeMap::new();
        for (ordinal, example) in ordered.iter().enumerate() {
            let selected = SelectedId(ordinal as u32);
            by_id.insert(example.id.clone(), selected);
            by_class.entry(example.label).or_default().push(selected);
        }
        let class_sizes = by_class
            .iter()
            .map(|(label, ids)| (*label, ids.len() as u64))
            .collect();

        Ok(Self {
            ordered,
            by_id,
            by_class,
            class_sizes,
            payload,
            semantic_hash,
            ledger_hash,
        })
    }

    /// The ordered selected examples: classes ascending, draw order within a class.
    pub fn examples(&self) -> &[SelectedExample] {
        &self.ordered
    }

    /// The ordered selected ids — the list DATA-03 contracts and Phase 5 replays.
    pub fn ordered_ids(&self) -> Vec<&str> {
        self.ordered.iter().map(|row| row.id.as_str()).collect()
    }

    /// How many rows were selected in total.
    pub fn len(&self) -> usize {
        self.ordered.len()
    }

    /// Whether the selection is empty. Never true for a selection this crate produced —
    /// `shots_per_class` is at least 8 — but required beside `len`.
    pub fn is_empty(&self) -> bool {
        self.ordered.is_empty()
    }

    /// `SHA-256` of the canonical payload bytes.
    pub fn semantic_hash(&self) -> [u8; 32] {
        self.semantic_hash
    }

    /// The access-ledger hash as of the moment this selection was built — including the
    /// selection's own record.
    pub fn ledger_hash(&self) -> [u8; 32] {
        self.ledger_hash
    }

    /// The payload this selection is attested by. Retained, never rebuilt.
    pub fn payload(&self) -> &SelectionPayload {
        &self.payload
    }

    /// Hex fingerprint of the whole dataset the selection came from.
    pub fn dataset_fingerprint_hex(&self) -> &str {
        &self.payload.dataset_fingerprint
    }

    /// Hex fingerprint of the validation split alone.
    pub fn validation_fingerprint_hex(&self) -> &str {
        &self.payload.validation_fingerprint
    }

    /// The root seed this selection was drawn from.
    pub fn root_seed(&self) -> u64 {
        self.payload.root_seed
    }

    /// Shots per class.
    pub fn shots_per_class(&self) -> u32 {
        self.payload.shots_per_class
    }

    /// SELECTED rows per class, ascending by label. Every entry equals
    /// `shots_per_class`; the vector exists so a consumer can read the class set without
    /// walking the ordered list.
    pub fn class_sizes(&self) -> &[(usize, u64)] {
        &self.class_sizes
    }

    /// The ordinal of a selected id, or `None` when the id was not selected.
    ///
    /// This is the ONLY way to obtain a [`SelectedId`], which is what makes the type a
    /// proof of membership.
    pub fn selected_id(&self, id: &str) -> Option<SelectedId> {
        self.by_id.get(id).copied()
    }

    /// The id behind an ordinal.
    ///
    /// # Panics
    ///
    /// If `selected` came from a DIFFERENT selection and is out of range here.
    pub fn id_of(&self, selected: SelectedId) -> &str {
        self.example_of(selected).id.as_str()
    }

    /// The class label behind an ordinal.
    ///
    /// # Panics
    ///
    /// If `selected` came from a DIFFERENT selection and is out of range here.
    pub fn label_of(&self, selected: SelectedId) -> usize {
        self.example_of(selected).label
    }

    /// The full row behind an ordinal.
    ///
    /// # Panics
    ///
    /// If `selected` came from a DIFFERENT selection and is out of range here. A
    /// `SelectedId` is proof of membership in the selection that produced it, not in every
    /// selection, and returning a plausible row from the wrong one would be worse than a
    /// named panic.
    pub fn example_of(&self, selected: SelectedId) -> &SelectedExample {
        self.ordered
            .get(selected.0 as usize)
            .expect("SelectedId ordinals are produced only by the Selection they index")
    }

    /// Every selected ordinal of one class, ascending. An unknown label yields an empty
    /// slice.
    pub fn ids_in_class(&self, label: usize) -> &[SelectedId] {
        self.by_class.get(&label).map_or(&[], Vec::as_slice)
    }

    /// STRICT replay: the sole sanctioned path from manifest bytes back to a `Selection`.
    ///
    /// Consumes a CANONICAL prepared dataset. A compatibility dataset is not rejected here
    /// — it cannot be passed, because it is a different type (D-19).
    ///
    /// The ladder, in order, each step with its own typed error:
    ///
    /// 1. supported `schema_version`, then `algorithm_version` equal to this build's
    ///    ([`ContrastiveDataError::UnsupportedSchemaVersion`] /
    ///    [`ContrastiveDataError::UnsupportedAlgorithmVersion`]). A different algorithm
    ///    version is refused, never silently skipped past the recomputation below;
    /// 2. profile, dataset fingerprint, validation fingerprint, exclusion record;
    /// 3. membership of every id in the post-exclusion train pool, then id uniqueness,
    ///    then per-class balance, then class-ascending ordering, then both per-row hashes;
    /// 4. `semantic_hash`;
    /// 5. a full deterministic RECOMPUTATION of the ordered list.
    ///
    /// # Why step 4 hashes the manifest's OWN payload bytes
    ///
    /// The payload embeds the access ledger and its hash as they stood when `select`
    /// finished. Replay appends its own record, so by the time this check runs the LIVE
    /// ledger has diverged from the recorded one by construction. A rule that rebuilt the
    /// payload from the live ledger could therefore never pass — it would reject every
    /// honest manifest and take the CLI's replay path down with it. The digest is
    /// recomputed over `manifest.payload.to_canonical_bytes()` and nothing else.
    ///
    /// # Why step 5 exists at all
    ///
    /// Steps 1-4 accept any manifest that is internally consistent. A hand-edited manifest
    /// with recomputed hashes IS internally consistent. Step 5 is what makes the artifact
    /// evidence: a selection no seed could have produced is refused even when every digest
    /// in it agrees.
    ///
    /// # Errors
    ///
    /// The variant named by whichever step disagrees first; see the ladder above.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "selection_replay"
    )]
    pub fn replay(
        manifest: &SelectionManifest,
        dataset: &PreparedDataset<Canonical>,
        ledger: &mut AccessLedger,
    ) -> Result<Self, ContrastiveDataError> {
        let payload = &manifest.payload;
        check_versions(payload)?;
        check_provenance(payload, dataset)?;

        let buckets = ClassBuckets::from_prepared(dataset);
        check_membership(payload, dataset, &buckets)?;
        check_uniqueness(payload)?;
        check_class_balance(payload)?;
        check_class_ordering(payload)?;
        let recorded = rebuild_examples(payload, dataset)?;

        manifest.verify_digest()?;

        let recomputed = compute_ordered(dataset, payload.root_seed, payload.shots_per_class)?;
        if recomputed != recorded {
            return Err(ContrastiveDataError::SelectionReplayMismatch {
                field: "ordered_examples".to_string(),
            });
        }

        let ledger_hash = digest_from_hex(&payload.ledger_hash).ok_or_else(|| {
            ContrastiveDataError::SelectionReplayMismatch {
                field: "ledger_hash".to_string(),
            }
        })?;
        // The payload carries BOTH the persisted records and their digest, and nothing
        // above compares the two. Without this, a manifest whose `ledger_hash` does not
        // describe its own `access_ledger` replays clean and hands the unearned digest to
        // the returned `Selection` — where `SelectionManifest::from_selection` then treats
        // it as the ledger this selection was actually taken under.
        let mut persisted = AccessLedger::new();
        for record in &payload.access_ledger {
            persisted.record(
                &record.role,
                &record.profile,
                &record.purpose,
                &record.fingerprint_hex,
            );
        }
        if persisted.ledger_hash() != ledger_hash {
            return Err(ContrastiveDataError::SelectionReplayMismatch {
                field: "access_ledger".to_string(),
            });
        }

        ledger.record(
            Train::ROLE,
            Canonical::PROFILE,
            REPLAY_PURPOSE,
            &payload.dataset_fingerprint,
        );
        Self::assemble(recorded, payload.clone(), ledger_hash)
    }
}

/// Parse a 64-character LOWERCASE hex digest.
///
/// Uppercase is refused rather than accepted case-insensitively: every producer in this
/// crate renders through [`hex`], which is lowercase, so an uppercase digest cannot have
/// come from an honest writer. Accepting it would admit a manifest whose bytes differ from
/// the canonical form while its parsed value looks identical.
fn digest_from_hex(text: &str) -> Option<[u8; 32]> {
    let bytes = text.as_bytes();
    if bytes.len() != 64 {
        return None;
    }
    let nibble = |byte: u8| -> Option<u32> {
        if byte.is_ascii_uppercase() {
            return None;
        }
        char::from(byte).to_digit(16)
    };
    let mut digest = [0_u8; 32];
    for (index, slot) in digest.iter_mut().enumerate() {
        let high = nibble(bytes[index * 2])?;
        let low = nibble(bytes[index * 2 + 1])?;
        *slot = (high * 16 + low) as u8;
    }
    Some(digest)
}

/// Step 1: versions.
fn check_versions(payload: &SelectionPayload) -> Result<(), ContrastiveDataError> {
    if !SUPPORTED_SELECTION_SCHEMA_VERSIONS.contains(&payload.schema_version) {
        return Err(ContrastiveDataError::UnsupportedSchemaVersion {
            field: "selection".to_string(),
            got: payload.schema_version,
            supported: SELECTION_SCHEMA_VERSION,
        });
    }
    if payload.algorithm_version != SELECTION_ALGORITHM_VERSION {
        return Err(ContrastiveDataError::UnsupportedAlgorithmVersion {
            got: payload.algorithm_version,
            supported: SELECTION_ALGORITHM_VERSION,
        });
    }
    Ok(())
}

/// Step 2: profile, both fingerprints, exclusion record.
fn check_provenance(
    payload: &SelectionPayload,
    dataset: &PreparedDataset<Canonical>,
) -> Result<(), ContrastiveDataError> {
    if payload.profile != Canonical::PROFILE {
        return Err(ContrastiveDataError::ProfileMismatch {
            expected: Canonical::PROFILE.to_string(),
            got: payload.profile.clone(),
        });
    }
    let dataset_fingerprint = dataset.fingerprint().hex();
    if payload.dataset_fingerprint != dataset_fingerprint {
        return Err(ContrastiveDataError::FingerprintMismatch {
            expected: payload.dataset_fingerprint.clone(),
            got: dataset_fingerprint,
        });
    }
    let validation_fingerprint = dataset.validation_witness().fingerprint_hex();
    if payload.validation_fingerprint != validation_fingerprint {
        return Err(ContrastiveDataError::FingerprintMismatch {
            expected: payload.validation_fingerprint.clone(),
            got: validation_fingerprint,
        });
    }
    // The label MAP and the normalization tag are payload fields in their own right: the
    // dataset fingerprint is computed from the DATASET, so matching it says nothing about
    // what this payload wrote down. Without these two comparisons a manifest could carry
    // renamed classes — or claim a normalization version this build does not implement —
    // and still survive the full recomputation below, because neither value reaches the
    // draw. Every downstream consumer reads its class names from the payload, so a
    // silently wrong map is a mislabeled benchmark rather than a cosmetic defect.
    if payload.label_names.as_slice() != dataset.label_names() {
        return Err(ContrastiveDataError::SelectionReplayMismatch {
            field: "label_names".to_string(),
        });
    }
    if payload.normalization_version != CONTENT_NORMALIZATION_VERSION {
        return Err(ContrastiveDataError::UnsupportedNormalizationVersion {
            got: payload.normalization_version.clone(),
            supported: CONTENT_NORMALIZATION_VERSION,
        });
    }
    let recorded = hex(&payload.exclusions.hash());
    let actual = hex(&dataset.exclusions().hash());
    if recorded != actual {
        return Err(ContrastiveDataError::ExclusionRecordMismatch {
            expected: recorded,
            got: actual,
        });
    }
    Ok(())
}

/// Where an id lives, when it is not in the post-exclusion training pool.
fn locate(dataset: &PreparedDataset<Canonical>, id: &str) -> &'static str {
    if dataset.train().exact_hash_of(id).is_some() {
        return "train, but excluded from the selection pool";
    }
    if dataset.validation().exact_hash_of(id).is_some() {
        return "validation";
    }
    if dataset.test().exact_hash_of(id).is_some() {
        return "test";
    }
    "nowhere"
}

/// Step 3a: every recorded id is in the post-exclusion training pool.
fn check_membership(
    payload: &SelectionPayload,
    dataset: &PreparedDataset<Canonical>,
    buckets: &ClassBuckets,
) -> Result<(), ContrastiveDataError> {
    let pool: BTreeSet<&str> = buckets
        .labels()
        .into_iter()
        .flat_map(|label| buckets.ids(label).iter().map(String::as_str))
        .collect();
    for record in &payload.ordered_examples {
        if !pool.contains(record.id.as_str()) {
            return Err(ContrastiveDataError::EndpointNotInSelection {
                id: record.id.clone(),
                found_in: locate(dataset, &record.id).to_string(),
            });
        }
    }
    Ok(())
}

/// Step 3b: no repeated id.
fn check_uniqueness(payload: &SelectionPayload) -> Result<(), ContrastiveDataError> {
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for record in &payload.ordered_examples {
        if !seen.insert(record.id.as_str()) {
            return Err(ContrastiveDataError::DuplicateId {
                split: "selection".to_string(),
                id: record.id.clone(),
            });
        }
    }
    Ok(())
}

/// Step 3c: exactly `shots_per_class` rows of every declared class.
fn check_class_balance(payload: &SelectionPayload) -> Result<(), ContrastiveDataError> {
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for record in &payload.ordered_examples {
        *counts.entry(record.label).or_default() += 1;
    }
    let classes = payload.label_names.len();
    let got: Vec<usize> = (0..classes)
        .map(|label| counts.get(&label).copied().unwrap_or(0))
        .collect();
    let expected = vec![payload.shots_per_class as usize; classes];
    if got != expected || counts.len() != classes {
        return Err(ContrastiveDataError::InvalidClassCounts {
            split: "selection".to_string(),
            expected,
            got,
        });
    }
    Ok(())
}

/// Step 3d: labels never decrease, i.e. the classes are concatenated ascending.
fn check_class_ordering(payload: &SelectionPayload) -> Result<(), ContrastiveDataError> {
    let out_of_order = payload
        .ordered_examples
        .windows(2)
        .any(|pair| pair[1].label < pair[0].label);
    if out_of_order {
        return Err(ContrastiveDataError::SelectionReplayMismatch {
            field: "class_order".to_string(),
        });
    }
    Ok(())
}

/// Step 3e: both recorded hashes equal the split's own, and the typed list is rebuilt.
fn rebuild_examples(
    payload: &SelectionPayload,
    dataset: &PreparedDataset<Canonical>,
) -> Result<Vec<SelectedExample>, ContrastiveDataError> {
    let train = dataset.train();
    let mut rebuilt = Vec::with_capacity(payload.ordered_examples.len());
    for record in &payload.ordered_examples {
        let exact_hash = compare_row_hash(
            &record.id,
            &record.exact_hash,
            train.exact_hash_of(&record.id),
        )?;
        let normalized_hash = compare_row_hash(
            &record.id,
            &record.normalized_hash,
            train.normalized_hash_of(&record.id),
        )?;
        rebuilt.push(SelectedExample {
            id: record.id.clone(),
            label: record.label,
            exact_hash,
            normalized_hash,
        });
    }
    Ok(rebuilt)
}

/// One recorded hex digest against the split's own value.
fn compare_row_hash(
    id: &str,
    recorded_hex: &str,
    actual: Option<&[u8; 32]>,
) -> Result<[u8; 32], ContrastiveDataError> {
    // `actual` is always `Some` here: membership was checked before this step runs.
    let actual = actual
        .copied()
        .ok_or_else(|| ContrastiveDataError::RowHashMismatch {
            id: id.to_string(),
            expected: recorded_hex.to_string(),
            got: "row absent from the training split".to_string(),
        })?;
    if hex(&actual) == recorded_hex {
        return Ok(actual);
    }
    Err(ContrastiveDataError::RowHashMismatch {
        id: id.to_string(),
        expected: recorded_hex.to_string(),
        got: hex(&actual),
    })
}

#[cfg(test)]
pub(crate) mod test_corpus {
    //! A synthetic three-class corpus shared by every test in this crate that needs a
    //! prepared dataset.
    //!
    //! Training rows are emitted INTERLEAVED by class and with ids that descend within a
    //! class, so ingest order is neither grouped nor sorted. A bucketing implementation
    //! that merely preserved ingest order would produce different selections, and the
    //! sorting assertions would otherwise be vacuous.

    use crate::ledger::AccessLedger;
    use crate::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
    use crate::schema::LabeledExample;
    use crate::select::{FewShotSelector, Selection, SelectionConfig};
    use crate::split::SplitDeclaration;

    pub(crate) const LABEL_TEXTS: [&str; 3] = ["none", "against", "favor"];

    /// The ten contracted benchmark seeds.
    ///
    /// Cited verbatim from `BENCHMARK_SEEDS` in
    /// `crates/apr-cli/src/commands/data_tweeteval.rs`. Note that **42 is not among them**.
    ///
    /// Cited by SYMBOL, not by line: the original citation pinned a line number that was
    /// already wrong when written and has drifted twice since. A reference that rots
    /// silently is worse than a slightly less precise one.
    pub(crate) const CONTRACTED_SEEDS: [u64; 10] = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53];

    pub(crate) fn label_names() -> Vec<String> {
        LABEL_TEXTS.iter().map(|name| (*name).to_string()).collect()
    }

    fn row(id: String, input: String, label: usize, split: &str) -> LabeledExample {
        LabeledExample {
            id,
            input,
            label,
            label_text: LABEL_TEXTS[label].to_string(),
            source_split: split.to_string(),
        }
    }

    fn train_row(label: usize, index: usize) -> LabeledExample {
        row(
            format!("train:{label}-{index:03}"),
            format!("training post for class {label} item {index}"),
            label,
            "train",
        )
    }

    /// `(train, validation, test)` rows for a corpus with `per_class` training rows in
    /// each of the three classes.
    pub(crate) fn rows(
        per_class: usize,
    ) -> (
        Vec<LabeledExample>,
        Vec<LabeledExample>,
        Vec<LabeledExample>,
    ) {
        let mut train = Vec::with_capacity(per_class * 3);
        for step in 0..per_class {
            let index = per_class - 1 - step;
            for label in 0..3 {
                train.push(train_row(label, index));
            }
        }
        let validation = (0..3)
            .map(|label| {
                row(
                    format!("validation:{label}"),
                    format!("validation post for class {label}"),
                    label,
                    "validation",
                )
            })
            .collect();
        let test = (0..3)
            .map(|label| {
                row(
                    format!("test:{label}"),
                    format!("held-out post for class {label}"),
                    label,
                    "test",
                )
            })
            .collect();
        (train, validation, test)
    }

    pub(crate) fn declarations(train_counts: Vec<usize>) -> CanonicalDeclarations {
        let decl = |counts: Vec<usize>| SplitDeclaration {
            expected_class_counts: counts,
            label_names: label_names(),
        };
        CanonicalDeclarations {
            train: decl(train_counts),
            validation: decl(vec![1, 1, 1]),
            test: decl(vec![1, 1, 1]),
            label_names: label_names(),
        }
    }

    pub(crate) fn build(
        train: Vec<LabeledExample>,
        validation: Vec<LabeledExample>,
        test: Vec<LabeledExample>,
        decls: &CanonicalDeclarations,
        ledger: &mut AccessLedger,
    ) -> PreparedDataset<Canonical> {
        PreparedDataset::<Canonical>::from_labeled_rows(train, validation, test, decls, ledger)
            .expect("the synthetic corpus must be valid")
    }

    /// The standard corpus: `per_class` rows per class, no duplicates, no exclusions.
    pub(crate) fn dataset(
        per_class: usize,
        ledger: &mut AccessLedger,
    ) -> PreparedDataset<Canonical> {
        let (train, validation, test) = rows(per_class);
        build(
            train,
            validation,
            test,
            &declarations(vec![per_class; 3]),
            ledger,
        )
    }

    /// The same corpus with the validation class-0 row duplicating a training class-0
    /// row's content, so exactly one training id is excluded.
    pub(crate) fn dataset_with_cross_split_duplicate(
        per_class: usize,
        ledger: &mut AccessLedger,
    ) -> PreparedDataset<Canonical> {
        let (train, mut validation, test) = rows(per_class);
        validation[0].input = train_row(0, 0).input;
        build(
            train,
            validation,
            test,
            &declarations(vec![per_class; 3]),
            ledger,
        )
    }

    /// The same corpus with class 2 declared but absent from the training split.
    pub(crate) fn dataset_with_empty_class(
        per_class: usize,
        ledger: &mut AccessLedger,
    ) -> PreparedDataset<Canonical> {
        let (train, validation, test) = rows(per_class);
        let train = train.into_iter().filter(|row| row.label != 2).collect();
        build(
            train,
            validation,
            test,
            &declarations(vec![per_class, per_class, 0]),
            ledger,
        )
    }

    pub(crate) fn select(
        dataset: &PreparedDataset<Canonical>,
        root_seed: u64,
        shots_per_class: u32,
        ledger: &mut AccessLedger,
    ) -> Selection {
        FewShotSelector::select(
            dataset,
            &SelectionConfig {
                root_seed,
                shots_per_class,
            },
            ledger,
        )
        .expect("the synthetic corpus must support this selection")
    }

    /// A fresh dataset AND a fresh ledger, so two selections are comparable: a ledger that
    /// already carried a previous selection's record would produce a different payload.
    pub(crate) fn fresh_selection(
        per_class: usize,
        root_seed: u64,
        shots_per_class: u32,
    ) -> (Selection, AccessLedger) {
        let mut ledger = AccessLedger::new();
        let prepared = dataset(per_class, &mut ledger);
        let selection = select(&prepared, root_seed, shots_per_class, &mut ledger);
        (selection, ledger)
    }
}

#[cfg(test)]
mod select_tests {
    use super::test_corpus::{self, CONTRACTED_SEEDS};
    use super::{FewShotSelector, SelectionConfig};
    use crate::error::ContrastiveDataError;
    use crate::ledger::AccessLedger;
    use crate::prepared::PreparedDataset;
    use proptest::prelude::{prop_assert_eq, proptest};

    fn ordered(selection: &super::Selection) -> Vec<(String, usize)> {
        selection
            .examples()
            .iter()
            .map(|row| (row.id.clone(), row.label))
            .collect()
    }

    #[test]
    fn select_returns_exactly_shots_per_class_with_labels_and_hashes() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 13, 8, &mut ledger);

        assert_eq!(selection.len(), 24);
        assert!(!selection.is_empty());
        assert_eq!(selection.class_sizes(), [(0, 8), (1, 8), (2, 8)]);

        let ids = selection.ordered_ids();
        let mut unique = ids.clone();
        unique.sort_unstable();
        unique.dedup();
        assert_eq!(unique.len(), 24, "every selected id must be distinct");

        for row in selection.examples() {
            assert!(
                row.id.starts_with(&format!("train:{}-", row.label)),
                "row {row:?} must come from its own class's training pool"
            );
            assert_eq!(
                Some(&row.exact_hash),
                dataset.train().exact_hash_of(&row.id),
                "hashes are COPIED from the split, never re-derived"
            );
            assert_eq!(
                Some(&row.normalized_hash),
                dataset.train().normalized_hash_of(&row.id)
            );
        }
    }

    #[test]
    fn select_replays_identically_across_two_calls() {
        let (first, _) = test_corpus::fresh_selection(12, 13, 8);
        let (second, _) = test_corpus::fresh_selection(12, 13, 8);
        assert_eq!(ordered(&first), ordered(&second));
        assert_eq!(first.semantic_hash(), second.semantic_hash());
    }

    /// DATA-03's headline evidence, stated directly rather than only through the goldens:
    /// each of the TEN contracted seeds replays exactly.
    #[test]
    fn select_replays_identically_for_every_contracted_seed() {
        assert_eq!(CONTRACTED_SEEDS.len(), 10);
        assert!(
            !CONTRACTED_SEEDS.contains(&42),
            "42 is not a contracted seed — see data_tweeteval.rs:45"
        );
        let mut seen = Vec::new();
        for seed in CONTRACTED_SEEDS {
            let (first, _) = test_corpus::fresh_selection(12, seed, 8);
            let (second, _) = test_corpus::fresh_selection(12, seed, 8);
            assert_eq!(ordered(&first), ordered(&second), "seed {seed}");
            assert_eq!(first.semantic_hash(), second.semantic_hash(), "seed {seed}");
            seen.push(first.ordered_ids().join(","));
        }
        let mut distinct = seen.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "different seeds must produce different orderings"
        );
    }

    #[test]
    fn select_supports_all_four_contracted_shot_counts() {
        for shots in [8_u32, 16, 32, 64] {
            let (selection, _) = test_corpus::fresh_selection(64, 13, shots);
            assert_eq!(selection.len() as u32, shots * 3, "shots {shots}");
            assert!(selection
                .class_sizes()
                .iter()
                .all(|(_, size)| *size == u64::from(shots)));
        }
    }

    /// The ordered SELECTION survives a permuted ingest order; the dataset fingerprint,
    /// and therefore the semantic hash, deliberately does NOT. Both halves are asserted so
    /// neither can regress unnoticed.
    #[test]
    fn select_is_invariant_under_permuted_ingest_order() {
        let (train, validation, test) = test_corpus::rows(12);
        let decls = test_corpus::declarations(vec![12; 3]);

        let mut ledger_a = AccessLedger::new();
        let straight = test_corpus::build(
            train.clone(),
            validation.clone(),
            test.clone(),
            &decls,
            &mut ledger_a,
        );
        let selection_a = test_corpus::select(&straight, 29, 8, &mut ledger_a);

        let mut reversed = train;
        reversed.reverse();
        let mut ledger_b = AccessLedger::new();
        let permuted = test_corpus::build(reversed, validation, test, &decls, &mut ledger_b);
        let selection_b = test_corpus::select(&permuted, 29, 8, &mut ledger_b);

        assert_eq!(
            ordered(&selection_a),
            ordered(&selection_b),
            "buckets sort before any draw, so ingest order cannot reach the selection"
        );
        assert_ne!(
            straight.fingerprint().hex(),
            permuted.fingerprint().hex(),
            "a permuted file IS different bytes; provenance must say so"
        );
        assert_ne!(selection_a.semantic_hash(), selection_b.semantic_hash());
    }

    #[test]
    fn select_rejects_an_invalid_shot_count_before_any_draw() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let before = ledger.records().len();

        for shots in [0_u32, 1, 7, 9, 63, 65, 128] {
            let err = FewShotSelector::select(
                &dataset,
                &SelectionConfig {
                    root_seed: 13,
                    shots_per_class: shots,
                },
                &mut ledger,
            )
            .expect_err("an uncontracted shot count must be refused");
            match err {
                ContrastiveDataError::InvalidShots { got, allowed } => {
                    assert_eq!(got, shots as usize);
                    assert_eq!(allowed, "{8, 16, 32, 64}");
                }
                other => panic!("expected InvalidShots, got {other:?}"),
            }
        }
        assert_eq!(
            ledger.records().len(),
            before,
            "a refused request must not touch the ledger"
        );
    }

    /// D-27's pool-exhaustion half.
    #[test]
    fn select_rejects_a_class_pool_smaller_than_shots() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset_with_empty_class(8, &mut ledger);
        let err = FewShotSelector::select(
            &dataset,
            &SelectionConfig {
                root_seed: 13,
                shots_per_class: 8,
            },
            &mut ledger,
        )
        .expect_err("an exhausted class pool must be refused");
        match err {
            ContrastiveDataError::CrossSplitDuplicateUnderflow {
                class_label,
                pool,
                shots,
            } => {
                assert_eq!((class_label, pool, shots), (2, 0, 8));
                let message = ContrastiveDataError::CrossSplitDuplicateUnderflow {
                    class_label,
                    pool,
                    shots,
                }
                .to_string();
                assert!(message.contains("class 2"), "{message}");
                assert!(message.contains("0 rows remain"), "{message}");
                assert!(message.contains("8 shots"), "{message}");
            }
            other => panic!("expected CrossSplitDuplicateUnderflow, got {other:?}"),
        }
    }

    #[test]
    fn select_excludes_cross_split_duplicates_from_the_pool() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset_with_cross_split_duplicate(12, &mut ledger);
        let excluded = dataset.exclusions().excluded_train_ids().to_vec();
        assert_eq!(
            excluded.len(),
            1,
            "the fixture must exclude exactly one row"
        );

        let selection = test_corpus::select(&dataset, 13, 8, &mut ledger);
        for id in &excluded {
            assert!(
                selection.selected_id(id).is_none(),
                "excluded id {id:?} must be unselectable"
            );
        }
    }

    #[test]
    fn select_selected_ids_round_trip_and_labels_agree() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 37, 8, &mut ledger);

        for row in selection.examples() {
            let selected = selection
                .selected_id(&row.id)
                .expect("every selected row resolves to an ordinal");
            assert_eq!(selection.id_of(selected), row.id);
            assert_eq!(selection.label_of(selected), row.label);
            assert_eq!(selection.example_of(selected), row);
        }
        assert!(selection.selected_id("train:0-999").is_none());
        assert!(selection.selected_id("validation:0").is_none());
    }

    #[test]
    fn select_ids_in_class_concatenate_to_the_full_ordered_list() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 41, 8, &mut ledger);

        let mut rebuilt = Vec::new();
        for (label, _) in selection.class_sizes() {
            let ordinals = selection.ids_in_class(*label);
            assert!(
                ordinals.windows(2).all(|pair| pair[0] < pair[1]),
                "class {label} ordinals must ascend"
            );
            for selected in ordinals {
                assert_eq!(selection.label_of(*selected), *label);
                rebuilt.push(selection.id_of(*selected).to_string());
            }
        }
        let expected: Vec<String> = selection
            .ordered_ids()
            .into_iter()
            .map(str::to_string)
            .collect();
        assert_eq!(rebuilt, expected);
        assert!(selection.ids_in_class(99).is_empty());
    }

    /// D-19's evidence trail: a selection can only be produced from a dataset that HAS a
    /// validation witness, and the ledger records the profile it ran under.
    #[test]
    fn select_appends_one_access_record_naming_the_selection() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let ingest_records = ledger.records().len();
        let selection = test_corpus::select(&dataset, 43, 8, &mut ledger);

        assert_eq!(ledger.records().len(), ingest_records + 1);
        let record = ledger.records().last().expect("a record was just appended");
        assert!(record.purpose.contains("select"));
        assert_eq!(record.role, "train", "selection reads ONLY the train pool");
        assert_eq!(record.profile, "canonical");
        assert_eq!(record.fingerprint_hex, dataset.fingerprint().hex());
        assert_eq!(
            record.fingerprint_hex,
            dataset.validation_witness().dataset_fingerprint_hex(),
            "reachable only from a dataset that has a validation witness (D-19)"
        );
        assert_eq!(selection.dataset_fingerprint_hex(), record.fingerprint_hex);
    }

    #[test]
    fn select_ledger_hash_matches_the_live_ledger_immediately_after() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 47, 8, &mut ledger);
        assert_eq!(selection.ledger_hash(), ledger.ledger_hash());

        ledger.record("train", "canonical", "unrelated", "aa");
        assert_ne!(
            selection.ledger_hash(),
            ledger.ledger_hash(),
            "the retained hash describes the ledger AS OF selection, not the live one"
        );
    }

    /// Typed by construction rather than asserted: `select` takes exactly one dataset
    /// value, so there is no argument list into which a foreign validation split could be
    /// substituted.
    #[test]
    fn select_takes_exactly_one_dataset_value() {
        fn signature_check(
            dataset: &PreparedDataset<crate::prepared::Canonical>,
            cfg: &SelectionConfig,
            ledger: &mut AccessLedger,
        ) -> Result<super::Selection, ContrastiveDataError> {
            FewShotSelector::select(dataset, cfg, ledger)
        }
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = signature_check(
            &dataset,
            &SelectionConfig {
                root_seed: 53,
                shots_per_class: 8,
            },
            &mut ledger,
        )
        .expect("selection succeeds");
        assert_eq!(selection.len(), 24);
    }

    /// `SelectedId::ordinal` IS the position in the ordered list (plan 02-08 triage).
    ///
    /// `cargo mutants` found `SelectedId::ordinal -> 0` and `-> 1` both surviving. The
    /// ordinal is what the pair wire format encodes and what `SelfPair` names, so a
    /// constant ordinal would make every pair a self-pair on paper while the identifiers
    /// stayed correct — and nothing asserted the accessor.
    #[test]
    fn selected_id_ordinals_are_the_positions_in_the_ordered_list() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 13, 8, &mut ledger);
        assert_eq!(
            selection.len(),
            24,
            "pin the population before relating over it"
        );

        let mut seen = std::collections::BTreeSet::new();
        for (index, row) in selection.examples().iter().enumerate() {
            let selected = selection
                .selected_id(&row.id)
                .expect("every selected example resolves to an ordinal");
            assert_eq!(
                selected.ordinal() as usize,
                index,
                "the ordinal of {} must be its position",
                row.id
            );
            assert_eq!(selection.id_of(selected), row.id, "and it must round-trip");
            assert!(seen.insert(selected.ordinal()), "ordinals must be distinct");
        }
        assert_eq!(seen.len(), 24);
    }

    /// The validation fingerprint is the witness digest, not a placeholder (02-08 triage).
    ///
    /// `cargo mutants` found `validation_fingerprint_hex -> ""` and `-> "xyzzy"` surviving.
    /// That field is what a replay compares to prove the manifest describes a dataset with
    /// THIS validation split (D-19); an empty string would have compared equal to another
    /// empty string and the isolation evidence would have been two placeholders agreeing.
    #[test]
    fn the_validation_fingerprint_is_the_witness_digest_and_differs_from_the_dataset_one() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 13, 8, &mut ledger);

        let expected = dataset.validation_witness().fingerprint_hex();
        assert_eq!(selection.validation_fingerprint_hex(), expected);
        assert_eq!(expected.len(), 64, "SHA-256 rendered as lowercase hex");
        assert!(
            expected
                .chars()
                .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c)),
            "not lowercase hex: {expected}"
        );
        // Different domain tags over the same bytes, so a swap of the two fields is
        // detectable rather than invisible.
        assert_ne!(
            selection.validation_fingerprint_hex(),
            selection.dataset_fingerprint_hex()
        );
    }

    proptest! {
        /// Determinism over the seed space, not only over the ten contracted seeds.
        #[test]
        fn select_is_deterministic_for_any_seed(seed in 0_u64..u64::MAX) {
            let (first, _) = test_corpus::fresh_selection(12, seed, 8);
            let (second, _) = test_corpus::fresh_selection(12, seed, 8);
            prop_assert_eq!(first.ordered_ids(), second.ordered_ids());
            prop_assert_eq!(first.semantic_hash(), second.semantic_hash());
        }
    }
}

#[cfg(test)]
mod manifest_replay_tests {
    //! Strict-replay evidence: one round trip, one pre-populated-ledger case, and the
    //! TWELVE distinct rejections of the validation ladder.
    //!
    //! Twelve rather than eleven because `dataset_fingerprint` and `validation_fingerprint`
    //! hold different values — the whole-dataset digest and the validation split's own —
    //! so tampering with each is a genuinely different test rather than the same one twice.
    //!
    //! Every negative constructs a `SelectionManifest` VALUE directly rather than parsing
    //! one, because `from_bytes` verifies the digest and would reject most of these before
    //! `replay` ever ran. Replay must defend itself against a manifest that is already in
    //! memory.

    use super::test_corpus;
    use super::{Selection, SelectionConfig};
    use crate::error::ContrastiveDataError;
    use crate::ledger::AccessLedger;
    use crate::manifest::{SelectedExampleRecord, SelectionManifest};
    use crate::prepared::{Canonical, PreparedDataset};

    struct Fixture {
        dataset: PreparedDataset<Canonical>,
        ledger: AccessLedger,
        manifest: SelectionManifest,
        selection: Selection,
    }

    fn fixture() -> Fixture {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 31, 8, &mut ledger);
        let manifest =
            SelectionManifest::from_selection(&selection, &ledger).expect("the wrap succeeds");
        Fixture {
            dataset,
            ledger,
            manifest,
            selection,
        }
    }

    /// Recompute the envelope digest so a tampered payload is INTERNALLY CONSISTENT.
    fn reseal(manifest: &mut SelectionManifest) {
        use sha2::{Digest, Sha256};
        let digest: [u8; 32] = Sha256::digest(
            manifest
                .payload
                .to_canonical_bytes()
                .expect("payload serializes"),
        )
        .into();
        manifest.semantic_hash = crate::hash::hex(&digest);
    }

    fn reject(
        mutate: impl FnOnce(&mut SelectionManifest, &PreparedDataset<Canonical>),
    ) -> ContrastiveDataError {
        let mut fixture = fixture();
        mutate(&mut fixture.manifest, &fixture.dataset);
        Selection::replay(&fixture.manifest, &fixture.dataset, &mut fixture.ledger)
            .expect_err("the tampered manifest must be refused")
    }

    #[test]
    fn manifest_replay_round_trips_through_the_file_form() {
        let mut fixture = fixture();
        let bytes = fixture.manifest.to_file_bytes().expect("file bytes");
        let parsed = SelectionManifest::from_bytes(&bytes).expect("the digest verifies");

        let replayed = Selection::replay(&parsed, &fixture.dataset, &mut fixture.ledger)
            .expect("an honest manifest replays");

        assert_eq!(replayed.examples(), fixture.selection.examples());
        assert_eq!(replayed.semantic_hash(), fixture.selection.semantic_hash());
        assert_eq!(replayed.ledger_hash(), fixture.selection.ledger_hash());
        assert_eq!(replayed.ordered_ids(), fixture.selection.ordered_ids());
    }

    /// Checker warning 2, stated as a test: replay appends its OWN record, so the live
    /// ledger has already diverged from the recorded one. A digest rule that rebuilt the
    /// payload from the live ledger could never pass this.
    #[test]
    fn manifest_replay_succeeds_against_a_ledger_that_has_already_moved_on() {
        let mut fixture = fixture();
        fixture
            .ledger
            .record("train", "canonical", "unrelated-later-work", "aa");
        assert_ne!(
            crate::hash::hex(&fixture.ledger.ledger_hash()),
            fixture.manifest.payload.ledger_hash,
            "the fixture must actually have diverged, or this test is vacuous"
        );
        let before = fixture.ledger.records().len();

        let replayed = Selection::replay(&fixture.manifest, &fixture.dataset, &mut fixture.ledger)
            .expect("replay must not depend on the live ledger");

        assert_eq!(replayed.ordered_ids(), fixture.selection.ordered_ids());
        assert_eq!(fixture.ledger.records().len(), before + 1);
        assert_eq!(
            fixture
                .ledger
                .records()
                .last()
                .expect("a record was appended")
                .purpose,
            "select-replay"
        );
    }

    #[test]
    fn manifest_replay_rejects_a_compatibility_profile() {
        let err = reject(|manifest, _| manifest.payload.profile = "compatibility".to_string());
        match err {
            ContrastiveDataError::ProfileMismatch { expected, got } => {
                assert_eq!(expected, "canonical");
                assert_eq!(got, "compatibility");
            }
            other => panic!("expected ProfileMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_altered_dataset_fingerprint() {
        let err = reject(|manifest, _| manifest.payload.dataset_fingerprint = "ab".repeat(32));
        match err {
            ContrastiveDataError::FingerprintMismatch { expected, got } => {
                assert_eq!(expected, "ab".repeat(32));
                assert_ne!(got, expected);
            }
            other => panic!("expected FingerprintMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_altered_validation_fingerprint() {
        let err = reject(|manifest, dataset| {
            // Pin that the two fields really are different values, so this test cannot be
            // an accidental duplicate of the one above.
            assert_ne!(
                manifest.payload.dataset_fingerprint,
                manifest.payload.validation_fingerprint
            );
            assert_eq!(
                manifest.payload.dataset_fingerprint,
                dataset.fingerprint().hex()
            );
            manifest.payload.validation_fingerprint = "cd".repeat(32);
        });
        match err {
            ContrastiveDataError::FingerprintMismatch { expected, got } => {
                assert_eq!(expected, "cd".repeat(32));
                assert_ne!(got, expected);
            }
            other => panic!("expected FingerprintMismatch, got {other:?}"),
        }
    }

    /// A renamed CLASS survives every fingerprint, so only a direct comparison catches it.
    ///
    /// The dataset fingerprint is computed from the DATASET, not from the payload, and the
    /// numeric labels the recomputation compares are unchanged — so before the
    /// `label_names` rung existed this manifest replayed clean and every downstream reader
    /// of `Selection::payload().label_names` got the wrong class names.
    #[test]
    fn manifest_replay_rejects_a_renamed_label() {
        let err = reject(|manifest, dataset| {
            assert_eq!(
                manifest.payload.label_names,
                dataset.label_names().to_vec(),
                "the fixture must start in agreement, or this test proves nothing"
            );
            manifest.payload.label_names[1] = "opposed".to_string();
            reseal(manifest);
        });
        match err {
            ContrastiveDataError::SelectionReplayMismatch { field } => {
                assert_eq!(field, "label_names");
            }
            other => panic!("expected SelectionReplayMismatch on label_names, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_unsupported_normalization_version() {
        let err = reject(|manifest, _| {
            manifest.payload.normalization_version = "nfc-trim-ws-v2".to_string();
            reseal(manifest);
        });
        match err {
            ContrastiveDataError::UnsupportedNormalizationVersion { got, supported } => {
                assert_eq!(got, "nfc-trim-ws-v2");
                assert_eq!(supported, crate::hash::CONTENT_NORMALIZATION_VERSION);
            }
            other => panic!("expected UnsupportedNormalizationVersion, got {other:?}"),
        }
    }

    /// The payload carries the persisted records AND their digest; they must agree.
    #[test]
    fn manifest_replay_rejects_a_ledger_hash_that_does_not_describe_its_own_records() {
        let err = reject(|manifest, _| {
            manifest
                .payload
                .access_ledger
                .push(crate::ledger::AccessRecord {
                    role: "train".to_string(),
                    profile: "canonical".to_string(),
                    purpose: "fabricated".to_string(),
                    fingerprint_hex: "aa".repeat(32),
                });
            // `ledger_hash` is deliberately LEFT ALONE, which is exactly the tampering the
            // rung exists for: the envelope digest is resealed, so every other check passes.
            reseal(manifest);
        });
        match err {
            ContrastiveDataError::SelectionReplayMismatch { field } => {
                assert_eq!(field, "access_ledger");
            }
            other => panic!("expected SelectionReplayMismatch on access_ledger, got {other:?}"),
        }
    }

    /// Every producer renders through `hex`, which is lowercase; an uppercase digest cannot
    /// have come from one, so it is refused rather than parsed case-insensitively.
    #[test]
    fn manifest_replay_rejects_an_uppercase_ledger_hash() {
        let err = reject(|manifest, _| {
            manifest.payload.ledger_hash = manifest.payload.ledger_hash.to_uppercase();
            reseal(manifest);
        });
        match err {
            ContrastiveDataError::SelectionReplayMismatch { field } => {
                assert_eq!(field, "ledger_hash");
            }
            other => panic!("expected SelectionReplayMismatch on ledger_hash, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_altered_exclusion_record() {
        let err = reject(|manifest, _| {
            let mut other_ledger = AccessLedger::new();
            let other = test_corpus::dataset_with_cross_split_duplicate(12, &mut other_ledger);
            assert_ne!(
                other.exclusions(),
                &manifest.payload.exclusions,
                "the substituted record must actually differ"
            );
            manifest.payload.exclusions = other.exclusions().clone();
        });
        match err {
            ContrastiveDataError::ExclusionRecordMismatch { expected, got } => {
                assert_ne!(expected, got);
            }
            other => panic!("expected ExclusionRecordMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_id_outside_the_selection_pool() {
        let err = reject(|manifest, dataset| {
            let row = dataset.validation().rows()[0].clone();
            manifest.payload.ordered_examples[0] = SelectedExampleRecord {
                id: row.id,
                label: 0,
                exact_hash: manifest.payload.ordered_examples[0].exact_hash.clone(),
                normalized_hash: manifest.payload.ordered_examples[0].normalized_hash.clone(),
            };
        });
        match err {
            ContrastiveDataError::EndpointNotInSelection { id, found_in } => {
                assert_eq!(id, "validation:0");
                assert_eq!(found_in, "validation");
            }
            other => panic!("expected EndpointNotInSelection, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_a_duplicated_id() {
        let err = reject(|manifest, _| {
            manifest.payload.ordered_examples[1] = manifest.payload.ordered_examples[0].clone();
        });
        match err {
            ContrastiveDataError::DuplicateId { split, id } => {
                assert_eq!(split, "selection");
                assert!(id.starts_with("train:0-"), "{id}");
            }
            other => panic!("expected DuplicateId, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_unbalanced_class() {
        let err = reject(|manifest, _| {
            manifest.payload.ordered_examples.remove(0);
        });
        match err {
            ContrastiveDataError::InvalidClassCounts {
                split,
                expected,
                got,
            } => {
                assert_eq!(split, "selection");
                assert_eq!(expected, vec![8, 8, 8]);
                assert_eq!(got, vec![7, 8, 8]);
            }
            other => panic!("expected InvalidClassCounts, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_examples_swapped_across_classes() {
        let err = reject(|manifest, _| {
            let rows = &mut manifest.payload.ordered_examples;
            assert_eq!((rows[0].label, rows[8].label), (0, 1));
            rows.swap(0, 8);
        });
        match err {
            ContrastiveDataError::SelectionReplayMismatch { field } => {
                assert_eq!(field, "class_order");
            }
            other => panic!("expected SelectionReplayMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_a_tampered_row_hash() {
        let err = reject(|manifest, _| {
            manifest.payload.ordered_examples[3].exact_hash = "0".repeat(64);
        });
        match err {
            ContrastiveDataError::RowHashMismatch { id, expected, got } => {
                assert!(id.starts_with("train:0-"), "{id}");
                assert_eq!(expected, "0".repeat(64));
                assert_ne!(got, expected);
            }
            other => panic!("expected RowHashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_unsupported_schema_version() {
        let err = reject(|manifest, _| manifest.payload.schema_version = 99);
        match err {
            ContrastiveDataError::UnsupportedSchemaVersion {
                field,
                got,
                supported,
            } => {
                assert_eq!(field, "selection");
                assert_eq!((got, supported), (99, 1));
            }
            other => panic!("expected UnsupportedSchemaVersion, got {other:?}"),
        }
    }

    #[test]
    fn manifest_replay_rejects_an_unsupported_algorithm_version() {
        let err = reject(|manifest, _| manifest.payload.algorithm_version = 99);
        match err {
            ContrastiveDataError::UnsupportedAlgorithmVersion { got, supported } => {
                assert_eq!((got, supported), (99, 1));
            }
            other => panic!("expected UnsupportedAlgorithmVersion, got {other:?}"),
        }
    }

    /// The case a membership-plus-hash check alone would have ACCEPTED, and the whole
    /// reason replay recomputes the selection rather than merely auditing the manifest.
    ///
    /// Every earlier rung passes: the substituted row is in the pool, unique, correctly
    /// labelled, correctly hashed, the class counts and ordering are untouched, and the
    /// envelope digest is recomputed so the manifest is internally consistent. It is
    /// simply not a selection any seed could have produced.
    #[test]
    fn manifest_replay_rejects_a_consistent_but_unreachable_ordered_list() {
        let err = reject(|manifest, dataset| {
            let selected: Vec<&str> = manifest
                .payload
                .ordered_examples
                .iter()
                .map(|row| row.id.as_str())
                .collect();
            let substitute = dataset
                .train()
                .rows()
                .iter()
                .find(|row| row.label == 0 && !selected.contains(&row.id.as_str()))
                .expect("the pool is larger than the selection")
                .clone();
            manifest.payload.ordered_examples[0] = SelectedExampleRecord {
                exact_hash: crate::hash::hex(
                    dataset
                        .train()
                        .exact_hash_of(&substitute.id)
                        .expect("the row is in the split"),
                ),
                normalized_hash: crate::hash::hex(
                    dataset
                        .train()
                        .normalized_hash_of(&substitute.id)
                        .expect("the row is in the split"),
                ),
                id: substitute.id,
                label: 0,
            };
            reseal(manifest);
            manifest
                .verify_digest()
                .expect("the forgery is internally consistent — that is the point");
        });
        match err {
            ContrastiveDataError::SelectionReplayMismatch { field } => {
                assert_eq!(field, "ordered_examples");
            }
            other => panic!("expected SelectionReplayMismatch, got {other:?}"),
        }
    }

    /// The digest rung itself, reached only when every structural rung has passed.
    #[test]
    fn manifest_replay_rejects_a_digest_that_disagrees_with_its_payload() {
        let err = reject(|manifest, _| manifest.semantic_hash = "f".repeat(64));
        match err {
            ContrastiveDataError::SemanticHashMismatch { expected, got } => {
                assert_eq!(expected, "f".repeat(64));
                assert_ne!(got, expected);
            }
            other => panic!("expected SemanticHashMismatch, got {other:?}"),
        }
    }

    /// A replayed selection is a full-fidelity `Selection`, not a shell.
    #[test]
    fn manifest_replay_returns_a_usable_selection() {
        let mut fixture = fixture();
        let replayed = Selection::replay(&fixture.manifest, &fixture.dataset, &mut fixture.ledger)
            .expect("an honest manifest replays");

        assert_eq!(replayed.class_sizes(), fixture.selection.class_sizes());
        assert_eq!(replayed.root_seed(), 31);
        assert_eq!(replayed.shots_per_class(), 8);
        let first = replayed.ordered_ids()[0].to_string();
        let selected = replayed
            .selected_id(&first)
            .expect("the replayed selection indexes its own rows");
        assert_eq!(replayed.id_of(selected), first);
        assert_eq!(replayed.label_of(selected), 0);

        // And it is the SAME selection a fresh run produces from the same inputs.
        let mut fresh_ledger = AccessLedger::new();
        let fresh_dataset = test_corpus::dataset(12, &mut fresh_ledger);
        let fresh = super::FewShotSelector::select(
            &fresh_dataset,
            &SelectionConfig {
                root_seed: 31,
                shots_per_class: 8,
            },
            &mut fresh_ledger,
        )
        .expect("a fresh selection succeeds");
        assert_eq!(fresh.examples(), replayed.examples());
        assert_eq!(fresh.semantic_hash(), replayed.semantic_hash());
    }
}
