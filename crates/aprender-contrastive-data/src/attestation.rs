//! Dataset identity attestation and its re-derivation from supplied buffers.
//!
//! # Contract: contrastive-pair-protocol-v1.yaml (equation `dataset_attestation`)
//!
//! A [`DatasetAttestation`] carries profile, schema version, label map, per-split JSONL
//! SHA-256, per-split per-class counts, normalization version, the cross-split
//! exclusion-record digest, and the dataset fingerprint. `from_attested_bytes` re-derives
//! every one of those from the buffers the caller supplied and fails typed on the first
//! disagreement — an attestation that is merely *quoted back* proves nothing about the
//! bytes in hand.
//!
//! # The threat this closes
//!
//! Row-level checks are not enough. A consumer pointed at an output directory whose
//! `train.jsonl` came from one preparation and `validation.jsonl` from another would pass
//! every row-level gate: each file is individually well-formed, each row carries the right
//! role, each class count is internally consistent. What is broken is *split identity* —
//! the two files do not describe one dataset — and nothing a row can say detects it. The
//! attested per-split digests plus the whole-dataset fingerprint are what turn that mixed
//! directory into a typed [`ContrastiveDataError::SplitHashMismatch`] or
//! [`ContrastiveDataError::FingerprintMismatch`] instead of a silent success.
//!
//! # Order is part of the guarantee
//!
//! The ladder runs: parse the attestation -> schema version -> normalization version ->
//! profile -> role set -> **per-split SHA-256 BEFORE the buffer is parsed** -> the ordinary
//! ingest gate ladder (which is where per-class counts are checked) -> exclusion digest ->
//! dataset fingerprint. The split-hash check precedes parsing deliberately: a corrupted
//! buffer must be reported as "these are not the bytes you attested", not as "row 4 is
//! malformed". The second diagnosis sends a reader looking for a data-quality problem in a
//! file that is simply the wrong file.
//!
//! No value of type `PreparedDataset<P>` — and therefore no `Split<R>` accessor — exists
//! until every comparison has passed. Exposure-then-validate would let a caller read rows
//! out of a dataset that is about to be rejected.
//!
//! # There is no schema-version-1 migration, deliberately
//!
//! See [`SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS`].

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::ContrastiveDataError;
use crate::hash::{hex, CONTENT_NORMALIZATION_VERSION};
use crate::ledger::AccessLedger;
use crate::prepared::{Canonical, Compatibility, DatasetProfile, PreparedDataset};
use crate::split::{
    CompatibilityTest, Split, SplitDeclaration, SplitRole, Test, Train, Validation,
};

/// The attestation schema version this build WRITES.
pub const DATASET_ATTESTATION_SCHEMA_VERSION: u32 = 2;

/// Every attestation schema version this build ACCEPTS.
///
/// # Why there is no version-1 shim
///
/// Version 1 predates the cross-split exclusion record. A version-1 artifact therefore
/// does not say which training rows were removed from the selection pool, and no migration
/// could supply that: recomputing it from the version-1 splits would produce a value the
/// original preparation never attested to, and defaulting it to "nothing was excluded"
/// would assert something that is false for the canonical TweetEval data. Either way the
/// upgraded record would be an unattested guess wearing an attestation's clothes, which is
/// strictly worse than a refusal. A version-1 manifest is
/// [`ContrastiveDataError::UnsupportedSchemaVersion`] and the remedy is to re-prepare.
pub const SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS: &[u32] = &[2];

/// What one split's bytes must reproduce.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitAttestation {
    /// Lowercase hex SHA-256 of the split's canonical JSONL bytes.
    pub sha256: String,
    /// Per-class row counts, indexed by class label.
    pub class_counts: Vec<u64>,
}

/// The identity a prepared dataset attests to.
///
/// Field order here IS the serialized field order (`serde_json` emits struct fields in
/// declaration order) and every map is a `BTreeMap`, so [`DatasetAttestation::to_bytes`]
/// is canonical: two runs over the same dataset produce byte-identical output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetAttestation {
    /// Attestation schema version. Always [`DATASET_ATTESTATION_SCHEMA_VERSION`] on write.
    pub schema_version: u32,
    /// The dataset profile these splits belong to.
    pub profile: String,
    /// The declared label map, in label order.
    pub label_names: Vec<String>,
    /// The content-normalization version the exclusion record was computed under.
    pub normalization_version: String,
    /// Per-split digests and class counts, keyed by split role.
    pub splits: BTreeMap<String, SplitAttestation>,
    /// Lowercase hex SHA-256 of the cross-split exclusion record's canonical bytes.
    pub exclusion_hash: String,
    /// Lowercase hex dataset fingerprint.
    pub dataset_fingerprint: String,
}

impl DatasetAttestation {
    /// Build the attestation a prepared dataset warrants.
    pub fn from_prepared<P: AttestedProfile>(dataset: &PreparedDataset<P>) -> Self {
        P::attest(dataset)
    }

    /// Canonical, deterministic serialization.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if the record cannot be serialized.
    pub fn to_bytes(&self) -> Result<Vec<u8>, ContrastiveDataError> {
        serde_json::to_vec(self).map_err(|error| ContrastiveDataError::Serialization {
            context: "dataset_attestation".to_string(),
            detail: error.to_string(),
        })
    }

    /// Parse an attestation from untrusted bytes.
    ///
    /// `deny_unknown_fields` applies: an extra key is a schema change, and a schema change
    /// that deserializes silently is a data change nobody reviewed.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if the bytes are not a well-formed
    /// attestation.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContrastiveDataError> {
        serde_json::from_slice(bytes).map_err(|error| ContrastiveDataError::Serialization {
            context: "dataset_attestation".to_string(),
            detail: error.to_string(),
        })
    }
}

/// A profile that can be attested. Implemented only by [`Canonical`] and [`Compatibility`].
pub trait AttestedProfile: DatasetProfile + Sized {
    /// Every split role this profile emits. Membership is all that is asked of it, so the
    /// order is documentary rather than load-bearing.
    const ROLES: &'static [&'static str];

    /// Derive this profile's attestation from a prepared dataset.
    fn attest(dataset: &PreparedDataset<Self>) -> DatasetAttestation;
}

impl AttestedProfile for Canonical {
    const ROLES: &'static [&'static str] = &["train", "validation", "test"];

    fn attest(dataset: &PreparedDataset<Self>) -> DatasetAttestation {
        let mut splits = BTreeMap::new();
        splits.insert(Train::ROLE.to_string(), split_attestation(dataset.train()));
        splits.insert(
            Validation::ROLE.to_string(),
            split_attestation(dataset.validation()),
        );
        splits.insert(Test::ROLE.to_string(), split_attestation(dataset.test()));
        DatasetAttestation {
            schema_version: DATASET_ATTESTATION_SCHEMA_VERSION,
            profile: Self::PROFILE.to_string(),
            label_names: dataset.label_names().to_vec(),
            normalization_version: CONTENT_NORMALIZATION_VERSION.to_string(),
            splits,
            exclusion_hash: hex(&dataset.exclusions().hash()),
            dataset_fingerprint: dataset.fingerprint().hex(),
        }
    }
}

impl AttestedProfile for Compatibility {
    const ROLES: &'static [&'static str] = &["train", "compatibility_test"];

    fn attest(dataset: &PreparedDataset<Self>) -> DatasetAttestation {
        let mut splits = BTreeMap::new();
        splits.insert(Train::ROLE.to_string(), split_attestation(dataset.train()));
        splits.insert(
            CompatibilityTest::ROLE.to_string(),
            split_attestation(dataset.compatibility_test()),
        );
        DatasetAttestation {
            schema_version: DATASET_ATTESTATION_SCHEMA_VERSION,
            profile: Self::PROFILE.to_string(),
            label_names: dataset.label_names().to_vec(),
            normalization_version: CONTENT_NORMALIZATION_VERSION.to_string(),
            splits,
            exclusion_hash: hex(&dataset.exclusions().hash()),
            dataset_fingerprint: dataset.fingerprint().hex(),
        }
    }
}

/// One split's attested parts, taken from the split's own accessors.
fn split_attestation<R: SplitRole>(split: &Split<R>) -> SplitAttestation {
    SplitAttestation {
        sha256: hex(split.source_hash()),
        class_counts: split.class_counts().to_vec(),
    }
}

/// Gates that do not depend on the buffers: schema version, normalization version,
/// profile, and the role set.
fn preflight<P: AttestedProfile>(
    attestation_bytes: &[u8],
    buffers: &BTreeMap<String, Vec<u8>>,
) -> Result<DatasetAttestation, ContrastiveDataError> {
    let attestation = DatasetAttestation::from_bytes(attestation_bytes)?;

    if !SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS.contains(&attestation.schema_version) {
        return Err(ContrastiveDataError::UnsupportedSchemaVersion {
            field: "dataset_attestation".to_string(),
            got: attestation.schema_version,
            supported: DATASET_ATTESTATION_SCHEMA_VERSION,
        });
    }
    if attestation.normalization_version != CONTENT_NORMALIZATION_VERSION {
        return Err(ContrastiveDataError::UnsupportedNormalizationVersion {
            got: attestation.normalization_version,
            supported: CONTENT_NORMALIZATION_VERSION,
        });
    }
    if attestation.profile != P::PROFILE {
        return Err(ContrastiveDataError::ProfileMismatch {
            expected: P::PROFILE.to_string(),
            got: attestation.profile,
        });
    }
    // A role this profile does not emit has no business in either map. Left unchecked it
    // would be inert today and load-bearing the moment somebody iterated the maps instead
    // of the profile's own role list.
    for role in attestation.splits.keys().chain(buffers.keys()) {
        if !P::ROLES.contains(&role.as_str()) {
            return Err(ContrastiveDataError::ConflictingSourceRole {
                declared: P::PROFILE.to_string(),
                embedded: role.clone(),
            });
        }
    }
    Ok(attestation)
}

/// Verify one split's buffer against its attested digest, THEN run the ingest ladder.
///
/// The digest comparison is first on purpose — see the module doc.
fn verified_split<R: SplitRole>(
    attestation: &DatasetAttestation,
    buffers: &BTreeMap<String, Vec<u8>>,
) -> Result<Split<R>, ContrastiveDataError> {
    let role = R::ROLE;
    let attested =
        attestation
            .splits
            .get(role)
            .ok_or_else(|| ContrastiveDataError::MissingSplit {
                role: role.to_string(),
            })?;
    let buffer = buffers
        .get(role)
        .ok_or_else(|| ContrastiveDataError::MissingSplit {
            role: role.to_string(),
        })?;

    let digest: [u8; 32] = Sha256::digest(buffer).into();
    let got = hex(&digest);
    if got != attested.sha256 {
        return Err(ContrastiveDataError::SplitHashMismatch {
            split: role.to_string(),
            expected: attested.sha256.clone(),
            got,
        });
    }

    let decl = SplitDeclaration {
        // A count that cannot be represented as a `usize` on this target cannot be matched
        // by any real split, so saturating turns an absurd attested count into a
        // guaranteed InvalidClassCounts rather than a panic.
        expected_class_counts: attested
            .class_counts
            .iter()
            .map(|count| usize::try_from(*count).unwrap_or(usize::MAX))
            .collect(),
        label_names: attestation.label_names.clone(),
    };
    Split::<R>::from_jsonl_bytes(buffer, &decl)
}

/// The two derived comparisons that can only run once the dataset exists.
fn check_derived(
    attestation: &DatasetAttestation,
    exclusion_hash: &str,
    dataset_fingerprint: &str,
) -> Result<(), ContrastiveDataError> {
    if attestation.exclusion_hash != exclusion_hash {
        return Err(ContrastiveDataError::ExclusionRecordMismatch {
            expected: attestation.exclusion_hash.clone(),
            got: exclusion_hash.to_string(),
        });
    }
    if attestation.dataset_fingerprint != dataset_fingerprint {
        return Err(ContrastiveDataError::FingerprintMismatch {
            expected: attestation.dataset_fingerprint.clone(),
            got: dataset_fingerprint.to_string(),
        });
    }
    Ok(())
}

/// Copy a staging ledger into the caller's ledger.
///
/// Ingest is staged and only committed once every comparison has passed, so a rejected
/// dataset leaves no access record — the same invariant `from_labeled_rows` upholds by
/// failing before it records anything.
fn commit_ledger(staged: &AccessLedger, ledger: &mut AccessLedger) {
    for record in staged.records() {
        ledger.record(
            &record.role,
            &record.profile,
            &record.purpose,
            &record.fingerprint_hex,
        );
    }
}

impl PreparedDataset<Canonical> {
    /// The canonical profile's attested boundary.
    ///
    /// `splits` is keyed by split role — exactly the shape
    /// [`crate::prepared::PreparedJsonl::as_map`] hands back, so
    /// `from_labeled_rows -> encode_jsonl -> from_attested_bytes` is a closed round trip
    /// whose two ends must fingerprint identically.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::UnsupportedSchemaVersion`],
    /// [`ContrastiveDataError::UnsupportedNormalizationVersion`],
    /// [`ContrastiveDataError::ProfileMismatch`],
    /// [`ContrastiveDataError::ConflictingSourceRole`],
    /// [`ContrastiveDataError::MissingSplit`],
    /// [`ContrastiveDataError::SplitHashMismatch`], any gate-ladder variant (including
    /// [`ContrastiveDataError::InvalidClassCounts`] when the attested per-class counts
    /// disagree with the split contents), [`ContrastiveDataError::ExclusionRecordMismatch`],
    /// or [`ContrastiveDataError::FingerprintMismatch`].
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "dataset_attestation"
    )]
    pub fn from_attested_bytes(
        attestation_bytes: &[u8],
        splits: &BTreeMap<String, Vec<u8>>,
        ledger: &mut AccessLedger,
    ) -> Result<Self, ContrastiveDataError> {
        let attestation = preflight::<Canonical>(attestation_bytes, splits)?;

        let train = verified_split::<Train>(&attestation, splits)?;
        let validation = verified_split::<Validation>(&attestation, splits)?;
        let test = verified_split::<Test>(&attestation, splits)?;

        let mut staged = AccessLedger::new();
        let dataset = Self::from_validated_splits(
            train,
            validation,
            test,
            &attestation.label_names,
            &mut staged,
        );

        check_derived(
            &attestation,
            &hex(&dataset.exclusions().hash()),
            &dataset.fingerprint().hex(),
        )?;
        commit_ledger(&staged, ledger);
        Ok(dataset)
    }
}

impl PreparedDataset<Compatibility> {
    /// The compatibility profile's attested boundary.
    ///
    /// A canonical attestation fed here is [`ContrastiveDataError::ProfileMismatch`], and
    /// so is a compatibility attestation fed to the canonical constructor. The profile is
    /// a type parameter, so the two are separate functions the compiler keeps apart; this
    /// check is what stops untrusted BYTES from crossing between them (D-19).
    ///
    /// # Errors
    ///
    /// The same set as the canonical constructor.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "dataset_attestation"
    )]
    pub fn from_attested_bytes(
        attestation_bytes: &[u8],
        splits: &BTreeMap<String, Vec<u8>>,
        ledger: &mut AccessLedger,
    ) -> Result<Self, ContrastiveDataError> {
        let attestation = preflight::<Compatibility>(attestation_bytes, splits)?;

        let train = verified_split::<Train>(&attestation, splits)?;
        let compatibility_test = verified_split::<CompatibilityTest>(&attestation, splits)?;

        let mut staged = AccessLedger::new();
        let dataset = Self::from_validated_splits(
            train,
            compatibility_test,
            &attestation.label_names,
            &mut staged,
        );

        check_derived(
            &attestation,
            &hex(&dataset.exclusions().hash()),
            &dataset.fingerprint().hex(),
        )?;
        commit_ledger(&staged, ledger);
        Ok(dataset)
    }
}

#[cfg(test)]
mod attestation_tests {
    use super::{
        DatasetAttestation, DATASET_ATTESTATION_SCHEMA_VERSION,
        SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS,
    };
    use crate::error::ContrastiveDataError;
    use crate::ledger::AccessLedger;
    use crate::prepared::{
        Canonical, CanonicalDeclarations, Compatibility, CompatibilityDeclarations, PreparedDataset,
    };
    use crate::schema::LabeledExample;
    use crate::split::SplitDeclaration;
    use std::collections::BTreeMap;

    fn label_names() -> Vec<String> {
        vec![
            "none".to_string(),
            "against".to_string(),
            "favor".to_string(),
        ]
    }

    fn row(id: &str, input: &str, label: usize, split: &str) -> LabeledExample {
        LabeledExample {
            id: id.to_string(),
            input: input.to_string(),
            label,
            label_text: label_names()[label].clone(),
            source_split: split.to_string(),
        }
    }

    fn decl(counts: Vec<usize>) -> SplitDeclaration {
        SplitDeclaration {
            expected_class_counts: counts,
            label_names: label_names(),
        }
    }

    fn train_rows(tag: &str) -> Vec<LabeledExample> {
        vec![
            row("train:0", &format!("{tag} alpha post"), 0, "train"),
            row("train:1", &format!("{tag} beta post"), 1, "train"),
            row("train:2", &format!("{tag} gamma post"), 2, "train"),
            row("train:3", &format!("{tag} delta post"), 0, "train"),
        ]
    }

    fn validation_rows(tag: &str) -> Vec<LabeledExample> {
        vec![
            row(
                "validation:0",
                &format!("{tag} epsilon post"),
                0,
                "validation",
            ),
            row("validation:1", &format!("{tag} zeta post"), 1, "validation"),
        ]
    }

    fn test_rows(tag: &str) -> Vec<LabeledExample> {
        vec![
            row("test:0", &format!("{tag} eta post"), 2, "test"),
            row("test:1", &format!("{tag} theta post"), 1, "test"),
        ]
    }

    fn canonical_decls() -> CanonicalDeclarations {
        CanonicalDeclarations {
            train: decl(vec![2, 1, 1]),
            validation: decl(vec![1, 1, 0]),
            test: decl(vec![0, 1, 1]),
            label_names: label_names(),
        }
    }

    /// A canonical dataset plus the attestation and buffers a consumer would be handed.
    struct Attested {
        attestation: DatasetAttestation,
        buffers: BTreeMap<String, Vec<u8>>,
        fingerprint: String,
    }

    fn attested_canonical(tag: &str) -> Attested {
        let mut ledger = AccessLedger::new();
        let dataset = PreparedDataset::<Canonical>::from_labeled_rows(
            train_rows(tag),
            validation_rows(tag),
            test_rows(tag),
            &canonical_decls(),
            &mut ledger,
        )
        .expect("valid canonical corpus");
        let jsonl = dataset.encode_jsonl().expect("encode must succeed");
        Attested {
            attestation: DatasetAttestation::from_prepared(&dataset),
            buffers: jsonl.as_map().clone(),
            fingerprint: dataset.fingerprint().hex(),
        }
    }

    fn attested_compatibility() -> Attested {
        let mut ledger = AccessLedger::new();
        let compatibility_rows = vec![
            row("compatibility_test:0", "eta post", 2, "compatibility_test"),
            row(
                "compatibility_test:1",
                "theta post",
                1,
                "compatibility_test",
            ),
        ];
        let dataset = PreparedDataset::<Compatibility>::from_labeled_rows(
            train_rows("x"),
            compatibility_rows,
            &CompatibilityDeclarations {
                train: decl(vec![2, 1, 1]),
                compatibility_test: decl(vec![0, 1, 1]),
                label_names: label_names(),
            },
            &mut ledger,
        )
        .expect("valid compatibility corpus");
        let jsonl = dataset.encode_jsonl().expect("encode must succeed");
        Attested {
            attestation: DatasetAttestation::from_prepared(&dataset),
            buffers: jsonl.as_map().clone(),
            fingerprint: dataset.fingerprint().hex(),
        }
    }

    fn bytes_of(attestation: &DatasetAttestation) -> Vec<u8> {
        attestation.to_bytes().expect("attestation serializes")
    }

    fn open_canonical(
        attested: &Attested,
    ) -> Result<PreparedDataset<Canonical>, ContrastiveDataError> {
        let mut ledger = AccessLedger::new();
        PreparedDataset::<Canonical>::from_attested_bytes(
            &bytes_of(&attested.attestation),
            &attested.buffers,
            &mut ledger,
        )
    }

    // -----------------------------------------------------------------------------
    // The happy paths
    // -----------------------------------------------------------------------------

    #[test]
    fn attestation_accepts_a_self_consistent_canonical_set() {
        let attested = attested_canonical("a");
        let mut ledger = AccessLedger::new();
        let dataset = PreparedDataset::<Canonical>::from_attested_bytes(
            &bytes_of(&attested.attestation),
            &attested.buffers,
            &mut ledger,
        )
        .expect("a self-consistent attested set must be accepted");

        assert_eq!(dataset.fingerprint().hex(), attested.fingerprint);
        assert_eq!(dataset.train().rows().len(), 4);
        assert_eq!(ledger.records().len(), 3);
        assert!(ledger
            .records()
            .iter()
            .all(|record| record.profile == "canonical"));
    }

    #[test]
    fn attestation_accepts_a_self_consistent_compatibility_set() {
        let attested = attested_compatibility();
        let mut ledger = AccessLedger::new();
        let dataset = PreparedDataset::<Compatibility>::from_attested_bytes(
            &bytes_of(&attested.attestation),
            &attested.buffers,
            &mut ledger,
        )
        .expect("a self-consistent compatibility set must be accepted");

        assert_eq!(dataset.fingerprint().hex(), attested.fingerprint);
        assert_eq!(dataset.compatibility_test().rows().len(), 2);
        assert_eq!(ledger.records().len(), 2);
    }

    /// Checker warning 3 — the cross-path fingerprint reproduction.
    ///
    /// The two doors into a split derive `source_hash` DIFFERENTLY: `from_labeled_rows`
    /// hashes the canonical re-encoding of typed rows, while `from_attested_bytes` lands in
    /// `Split::from_jsonl_bytes`, which hashes the supplied buffer. This assertion is a
    /// genuine two-derivation agreement rather than a tautology, and it holds only because
    /// `encode_jsonl(parse_jsonl_bytes(b)?)? == b` for canonical input.
    #[test]
    fn attestation_round_trip_reproduces_the_dataset_fingerprint() {
        let attested = attested_canonical("round");
        let reopened = open_canonical(&attested).expect("round trip must succeed");
        assert_eq!(reopened.fingerprint().hex(), attested.fingerprint);
        assert_eq!(
            reopened.exclusions().excluded_train_ids(),
            Vec::<String>::new().as_slice()
        );
    }

    #[test]
    fn attestation_serialization_is_canonical_and_strict() {
        let attested = attested_canonical("canon");
        let first = bytes_of(&attested.attestation);
        let second = bytes_of(&attested.attestation);
        assert_eq!(first, second, "serialization must be deterministic");

        let restored = DatasetAttestation::from_bytes(&first).expect("round-trips");
        assert_eq!(restored, attested.attestation);

        let mut widened = String::from_utf8(first).expect("attestation bytes are UTF-8");
        widened.pop();
        widened.push_str(",\"extra\":1}");
        let err = DatasetAttestation::from_bytes(widened.as_bytes())
            .expect_err("an unknown field must be rejected");
        assert!(matches!(err, ContrastiveDataError::Serialization { .. }));
    }

    /// Vacuity guard for the version rejection below: while the supported set has exactly
    /// one member, the single `supported` field of `UnsupportedSchemaVersion` can carry it.
    /// If the set ever grows, this fails and forces the error to be widened rather than
    /// letting the message quietly under-report.
    #[test]
    fn attestation_supported_set_is_exactly_the_writing_version() {
        assert_eq!(
            SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS,
            &[DATASET_ATTESTATION_SCHEMA_VERSION]
        );
    }

    // -----------------------------------------------------------------------------
    // The nine rejections
    // -----------------------------------------------------------------------------

    #[test]
    fn attestation_rejects_a_compatibility_profile_at_the_canonical_door() {
        let mut attested = attested_canonical("p1");
        attested.attestation.profile = "compatibility".to_string();
        match open_canonical(&attested).expect_err("profile mismatch must be rejected") {
            ContrastiveDataError::ProfileMismatch { expected, got } => {
                assert_eq!(expected, "canonical");
                assert_eq!(got, "compatibility");
            }
            other => panic!("expected ProfileMismatch, got {other:?}"),
        }
    }

    #[test]
    fn attestation_rejects_a_canonical_profile_at_the_compatibility_door() {
        let mut attested = attested_compatibility();
        attested.attestation.profile = "canonical".to_string();
        let mut ledger = AccessLedger::new();
        let err = PreparedDataset::<Compatibility>::from_attested_bytes(
            &bytes_of(&attested.attestation),
            &attested.buffers,
            &mut ledger,
        )
        .expect_err("profile mismatch must be rejected");
        match err {
            ContrastiveDataError::ProfileMismatch { expected, got } => {
                assert_eq!(expected, "compatibility");
                assert_eq!(got, "canonical");
            }
            other => panic!("expected ProfileMismatch, got {other:?}"),
        }
    }

    #[test]
    fn attestation_rejects_an_unsupported_schema_version() {
        let mut attested = attested_canonical("v1");
        attested.attestation.schema_version = 1;
        let err = open_canonical(&attested).expect_err("version 1 must be refused");
        let message = err.to_string();
        match err {
            ContrastiveDataError::UnsupportedSchemaVersion {
                field,
                got,
                supported,
            } => {
                assert_eq!(field, "dataset_attestation");
                assert_eq!(got, 1);
                assert_eq!(supported, DATASET_ATTESTATION_SCHEMA_VERSION);
            }
            other => panic!("expected UnsupportedSchemaVersion, got {other:?}"),
        }
        // The message is read out of the constant, not hardcoded, so widening the
        // supported set cannot leave this assertion silently describing the old one.
        for version in SUPPORTED_DATASET_ATTESTATION_SCHEMA_VERSIONS {
            assert!(
                message.contains(&version.to_string()),
                "message must name supported version {version}: {message}"
            );
        }
    }

    #[test]
    fn attestation_rejects_a_missing_split_buffer() {
        let mut attested = attested_canonical("miss");
        attested.buffers.remove("validation");
        match open_canonical(&attested).expect_err("a missing buffer must be rejected") {
            ContrastiveDataError::MissingSplit { role } => assert_eq!(role, "validation"),
            other => panic!("expected MissingSplit, got {other:?}"),
        }
    }

    /// The mixed-directory case: every buffer is individually valid, but one came from a
    /// different preparation. Row-level checks alone accept this.
    #[test]
    fn attestation_rejects_a_split_taken_from_another_valid_dataset() {
        let mut attested = attested_canonical("left");
        let other = attested_canonical("right");
        let foreign = other
            .buffers
            .get("validation")
            .expect("the other dataset has a validation split")
            .clone();

        // The substituted buffer is a WELL-FORMED validation split of the right shape —
        // parsing it succeeds and every row-level gate passes. Only the attested digest
        // knows it belongs to a different dataset.
        assert_ne!(
            attested.buffers.get("validation"),
            Some(&foreign),
            "the fixture must actually differ, or this test proves nothing"
        );
        attested.buffers.insert("validation".to_string(), foreign);

        match open_canonical(&attested).expect_err("a mixed directory must be rejected") {
            ContrastiveDataError::SplitHashMismatch {
                split,
                expected,
                got,
            } => {
                assert_eq!(split, "validation");
                assert_ne!(expected, got);
            }
            other => panic!("expected SplitHashMismatch, got {other:?}"),
        }
    }

    /// The split-hash check must precede parsing: a buffer that is not even JSON has to be
    /// diagnosed as "not the bytes you attested", never as "row 0 is malformed".
    #[test]
    fn attestation_rejects_corrupt_bytes_before_it_tries_to_parse_them() {
        let mut attested = attested_canonical("corrupt");
        attested
            .buffers
            .insert("test".to_string(), b"not json at all\n".to_vec());
        match open_canonical(&attested).expect_err("corrupt bytes must be rejected") {
            ContrastiveDataError::SplitHashMismatch { split, .. } => assert_eq!(split, "test"),
            other => panic!("expected SplitHashMismatch before parsing, got {other:?}"),
        }
    }

    #[test]
    fn attestation_rejects_a_fingerprint_that_disagrees_with_the_buffers() {
        let mut attested = attested_canonical("fp");
        let forged = "0".repeat(64);
        attested.attestation.dataset_fingerprint = forged.clone();
        match open_canonical(&attested).expect_err("fingerprint mismatch must be rejected") {
            ContrastiveDataError::FingerprintMismatch { expected, got } => {
                assert_eq!(expected, forged);
                assert_eq!(got, attested.fingerprint);
            }
            other => panic!("expected FingerprintMismatch, got {other:?}"),
        }
    }

    #[test]
    fn attestation_rejects_an_exclusion_hash_that_disagrees_with_the_buffers() {
        let mut attested = attested_canonical("exc");
        let forged = "1".repeat(64);
        attested.attestation.exclusion_hash = forged.clone();
        match open_canonical(&attested).expect_err("exclusion mismatch must be rejected") {
            ContrastiveDataError::ExclusionRecordMismatch { expected, got } => {
                assert_eq!(expected, forged);
                assert_ne!(got, forged);
            }
            other => panic!("expected ExclusionRecordMismatch, got {other:?}"),
        }
    }

    #[test]
    fn attestation_rejects_class_counts_that_disagree_with_the_split_contents() {
        let mut attested = attested_canonical("counts");
        attested
            .attestation
            .splits
            .get_mut("train")
            .expect("the canonical attestation names train")
            .class_counts = vec![4, 0, 0];
        match open_canonical(&attested).expect_err("count disagreement must be rejected") {
            ContrastiveDataError::InvalidClassCounts {
                split,
                expected,
                got,
            } => {
                assert_eq!(split, "train");
                assert_eq!(expected, vec![4, 0, 0]);
                assert_eq!(got, vec![2, 1, 1]);
            }
            other => panic!("expected InvalidClassCounts, got {other:?}"),
        }
    }

    // -----------------------------------------------------------------------------
    // The remaining boundary properties
    // -----------------------------------------------------------------------------

    #[test]
    fn attestation_rejects_a_stale_normalization_version() {
        let mut attested = attested_canonical("norm");
        attested.attestation.normalization_version = "nfc-trim-ws-v0".to_string();
        match open_canonical(&attested).expect_err("a stale normalization must be refused") {
            ContrastiveDataError::UnsupportedNormalizationVersion { got, supported } => {
                assert_eq!(got, "nfc-trim-ws-v0");
                assert_eq!(supported, "nfc-trim-ws-v1");
            }
            other => panic!("expected UnsupportedNormalizationVersion, got {other:?}"),
        }
    }

    #[test]
    fn attestation_rejects_a_role_outside_the_profile() {
        let mut attested = attested_canonical("role");
        let train = attested
            .attestation
            .splits
            .get("train")
            .expect("train is attested")
            .clone();
        attested
            .attestation
            .splits
            .insert("compatibility_test".to_string(), train);
        match open_canonical(&attested).expect_err("a foreign role must be rejected") {
            ContrastiveDataError::ConflictingSourceRole { declared, embedded } => {
                assert_eq!(declared, "canonical");
                assert_eq!(embedded, "compatibility_test");
            }
            other => panic!("expected ConflictingSourceRole, got {other:?}"),
        }
    }

    #[test]
    fn attestation_leaves_no_access_record_when_it_rejects() {
        let mut attested = attested_canonical("ledger");
        attested.attestation.dataset_fingerprint = "2".repeat(64);
        let mut ledger = AccessLedger::new();
        PreparedDataset::<Canonical>::from_attested_bytes(
            &bytes_of(&attested.attestation),
            &attested.buffers,
            &mut ledger,
        )
        .expect_err("a forged fingerprint must be rejected");
        assert!(
            ledger.records().is_empty(),
            "a rejected dataset must leave no access record"
        );
    }

    /// The exclusion digest is not decoration: a real cross-split duplicate changes it, so
    /// an attestation carrying the clean value cannot describe the duplicated buffers.
    #[test]
    fn attestation_binds_the_exclusion_record_to_the_buffers() {
        let clean = attested_canonical("dup");
        let mut duplicated_validation = validation_rows("dup");
        duplicated_validation[0].input = "dup alpha post".to_string();
        let mut ledger = AccessLedger::new();
        let dirty = PreparedDataset::<Canonical>::from_labeled_rows(
            train_rows("dup"),
            duplicated_validation,
            test_rows("dup"),
            &canonical_decls(),
            &mut ledger,
        )
        .expect("a cross-split duplicate is not fatal");
        assert_eq!(dirty.exclusions().groups().len(), 1);

        let dirty_attestation = DatasetAttestation::from_prepared(&dirty);
        assert_ne!(
            dirty_attestation.exclusion_hash, clean.attestation.exclusion_hash,
            "the exclusion digest must move when a duplicate appears"
        );

        let dirty_buffers = dirty
            .encode_jsonl()
            .expect("encode must succeed")
            .as_map()
            .clone();
        let mut ledger = AccessLedger::new();
        let reopened = PreparedDataset::<Canonical>::from_attested_bytes(
            &bytes_of(&dirty_attestation),
            &dirty_buffers,
            &mut ledger,
        )
        .expect("the duplicated dataset attests to itself consistently");
        assert_eq!(reopened.exclusions().excluded_train_ids(), ["train:0"]);
    }
}
