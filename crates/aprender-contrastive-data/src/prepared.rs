//! The attested, profile-parameterized dataset a consumer must present before canonical
//! splits are exposed.
//!
//! The profile is a TYPE PARAMETER, not a runtime field: `PreparedDataset<Canonical>` and
//! `PreparedDataset<Compatibility>` are distinct types with distinct constructors, and
//! only the canonical one exposes a validation witness. Selection consumes
//! `&PreparedDataset<Canonical>`, so a compatibility dataset cannot be passed at all —
//! which is what makes DATA-06's "cannot be constructed" provable by `trybuild` rather
//! than merely rejected at runtime.
//!
//! # `PreparedDataset<Compatibility>` has no `validation_witness` method
//!
//! A compatibility-profile selection run is not rejected at runtime. **It does not
//! compile.** There is no value of type `PreparedDataset<Compatibility>` that can be
//! passed where `&PreparedDataset<Canonical>` is expected, and there is no
//! `validation_witness` to call on it — rustc reports "no method named", which is a
//! non-compiling program rather than an error value a caller could ignore.
//!
//! The profile also selects which splits EXIST, through [`DatasetProfile::Splits`]. That
//! is stronger than an optional field: a compatibility dataset does not merely leave its
//! validation split empty, it has no place to put one (D-19).
//!
//! # This is where the typestate meets the hashes
//!
//! `hash.rs` is a leaf that knows nothing about split roles. The constructors here are the
//! single point of assembly: they build one `SplitFingerprintInput` per split from that
//! split's own accessors, hand them to `DatasetFingerprint::compute` in ascending role
//! order, and build the witness's `SplitFingerprint` from the SAME per-split value that
//! went into the dataset input. That shared value is what makes the two digests provably
//! describe the same bytes under different domain tags.

use core::marker::PhantomData;
use std::collections::BTreeMap;

use crate::dedup::{coalesced_exclusions, ExclusionRecord};
use crate::error::ContrastiveDataError;
use crate::hash::{
    DatasetFingerprint, DatasetFingerprintInput, SplitFingerprint, SplitFingerprintInput,
    CONTENT_NORMALIZATION_VERSION,
};
use crate::ledger::AccessLedger;
use crate::schema::LabeledExample;
use crate::split::{
    CompatibilityTest, Split, SplitDeclaration, SplitRole, Test, Train, Validation,
};

/// A dataset profile. Implemented only by [`Canonical`] and [`Compatibility`].
pub trait DatasetProfile {
    /// The profile string recorded in fingerprints and in the access ledger.
    const PROFILE: &'static str;
    /// The splits this profile emits.
    ///
    /// This associated type is what makes D-19 structural: the compatibility profile does
    /// not hold an EMPTY validation slot, it has no slot. An `Option<Split<Validation>>`
    /// field would have left one, and would have forced an `expect` into every accessor
    /// on an invariant only the constructor knows.
    type Splits: core::fmt::Debug + Clone + PartialEq + Eq;
}

/// The canonical three-split profile: train, validation, test.
#[derive(Debug, Clone, Copy)]
pub struct Canonical;

/// The merged SetFit compatibility profile: train and a compatibility test split, and NO
/// validation split at all (D-19).
#[derive(Debug, Clone, Copy)]
pub struct Compatibility;

/// The canonical profile's splits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalSplits {
    train: Split<Train>,
    validation: Split<Validation>,
    test: Split<Test>,
}

/// The compatibility profile's splits. There is no validation field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilitySplits {
    train: Split<Train>,
    compatibility_test: Split<CompatibilityTest>,
}

impl DatasetProfile for Canonical {
    const PROFILE: &'static str = "canonical";
    type Splits = CanonicalSplits;
}

impl DatasetProfile for Compatibility {
    const PROFILE: &'static str = "compatibility";
    type Splits = CompatibilitySplits;
}

/// Per-split declarations for the canonical profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalDeclarations {
    /// Declaration for the training split.
    pub train: SplitDeclaration,
    /// Declaration for the validation split.
    pub validation: SplitDeclaration,
    /// Declaration for the test split.
    pub test: SplitDeclaration,
    /// The shared label map.
    pub label_names: Vec<String>,
}

/// Per-split declarations for the compatibility profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityDeclarations {
    /// Declaration for the training split.
    pub train: SplitDeclaration,
    /// Declaration for the merged compatibility test split.
    pub compatibility_test: SplitDeclaration,
    /// The shared label map.
    pub label_names: Vec<String>,
}

/// Canonical JSONL bytes for every split of a prepared dataset, keyed by role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedJsonl {
    splits: BTreeMap<String, Vec<u8>>,
}

impl PreparedJsonl {
    /// The bytes of one role, if the profile emits it.
    pub fn get(&self, role: &str) -> Option<&[u8]> {
        self.splits.get(role).map(Vec::as_slice)
    }

    /// Every role's bytes, in ascending role order.
    pub fn as_map(&self) -> &BTreeMap<String, Vec<u8>> {
        &self.splits
    }
}

/// An opaque proof that a canonical validation split exists in THIS dataset.
///
/// Constructible only inside this module and obtainable only from
/// `PreparedDataset<Canonical>`, so a witness cannot describe a dataset other than the one
/// it was taken from.
#[derive(Debug)]
pub struct ValidationWitness<'a> {
    validation: &'a Split<Validation>,
    split_fingerprint: SplitFingerprint,
    dataset_fingerprint: DatasetFingerprint,
}

impl ValidationWitness<'_> {
    /// Fingerprint over the VALIDATION SPLIT ALONE.
    ///
    /// Deliberately NOT the dataset fingerprint. A selection payload records both a
    /// dataset fingerprint and a validation fingerprint; if this returned the dataset's
    /// own value the second field would be a duplicate of the first and the rejection
    /// tests that distinguish them would collapse into one test.
    pub fn fingerprint_hex(&self) -> String {
        self.split_fingerprint.hex()
    }

    /// The whole dataset's fingerprint — what the access ledger records.
    pub fn dataset_fingerprint_hex(&self) -> String {
        self.dataset_fingerprint.hex()
    }

    /// The validation split this witness proves the existence of.
    pub fn validation(&self) -> &Split<Validation> {
        self.validation
    }
}

/// A validated, fingerprinted dataset of exactly one profile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedDataset<P: DatasetProfile> {
    splits: P::Splits,
    exclusions: ExclusionRecord,
    fingerprint: DatasetFingerprint,
    /// The declared label map, retained.
    ///
    /// Retained rather than reconstructed from row `label_text` values, because a class
    /// whose split happens to contain no rows would simply vanish from a reconstruction —
    /// and the selection payload contracts a LABEL MAP, not "the labels that happened to
    /// appear". The map is already absorbed into `fingerprint`, so retaining it adds no
    /// new identity, only access to one that was already committed to.
    label_names: Vec<String>,
    profile: PhantomData<P>,
}

impl PreparedDataset<Canonical> {
    /// Ingest three typed split row sets under the canonical profile.
    ///
    /// The CLI has already decoded its dataset-specific source format into typed rows —
    /// D-05 keeps paired `*_text.txt` / `*_labels.txt` decoding on the CLI side, so this
    /// crate never sees a dataset-specific format and never touches a byte it was not
    /// handed.
    ///
    /// Ordering matters twice, and both orders are ascending by role name because that is
    /// what `DatasetFingerprint::compute` debug-asserts: the fingerprint inputs and the
    /// dedup inputs. The ledger, by contrast, records in INGEST order (train, validation,
    /// test), because a log of what happened should read in the order it happened.
    ///
    /// # Errors
    ///
    /// Any gate-ladder variant from the split boundary. Nothing is recorded in the ledger
    /// on the failing path: a dataset that was rejected was never accessed.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "prepared_dataset_typestate"
    )]
    pub fn from_labeled_rows(
        train: Vec<LabeledExample>,
        validation: Vec<LabeledExample>,
        test: Vec<LabeledExample>,
        decls: &CanonicalDeclarations,
        ledger: &mut AccessLedger,
    ) -> Result<Self, ContrastiveDataError> {
        let train = Split::<Train>::from_rows(train, &decls.train)?;
        let validation = Split::<Validation>::from_rows(validation, &decls.validation)?;
        let test = Split::<Test>::from_rows(test, &decls.test)?;
        Ok(Self::from_validated_splits(
            train,
            validation,
            test,
            &decls.label_names,
            ledger,
        ))
    }

    /// Assemble from splits that have ALREADY passed the ingest ladder.
    ///
    /// The SINGLE assembly point. Both doors land here — `from_labeled_rows`, which
    /// ingests typed rows, and `attestation`'s `from_attested_bytes`, which ingests
    /// attested buffers through [`Split::from_jsonl_bytes`] — so the fingerprint, the
    /// exclusion record and the ledger records cannot differ by which door a caller used.
    /// Two assembly paths that agree today are two that will disagree eventually.
    ///
    /// Infallible: every rejection already happened in the ladder.
    pub(crate) fn from_validated_splits(
        train: Split<Train>,
        validation: Split<Validation>,
        test: Split<Test>,
        label_names: &[String],
        ledger: &mut AccessLedger,
    ) -> Self {
        let fingerprint = {
            let train_pairs = train.exact_hash_pairs();
            let validation_pairs = validation.exact_hash_pairs();
            let test_pairs = test.exact_hash_pairs();
            let splits = [
                fingerprint_input::<Test>(&test, &test_pairs),
                fingerprint_input::<Train>(&train, &train_pairs),
                fingerprint_input::<Validation>(&validation, &validation_pairs),
            ];
            DatasetFingerprint::compute(&DatasetFingerprintInput {
                profile: Canonical::PROFILE,
                label_names,
                normalization_version: CONTENT_NORMALIZATION_VERSION,
                splits: &splits,
            })
        };

        let exclusions = coalesced_exclusions(&[
            (Test::ROLE, test.rows()),
            (Train::ROLE, train.rows()),
            (Validation::ROLE, validation.rows()),
        ]);

        let fingerprint_hex = fingerprint.hex();
        for role in [Train::ROLE, Validation::ROLE, Test::ROLE] {
            ledger.record(role, Canonical::PROFILE, "ingest", &fingerprint_hex);
        }

        Self {
            splits: CanonicalSplits {
                train,
                validation,
                test,
            },
            exclusions,
            fingerprint,
            label_names: label_names.to_vec(),
            profile: PhantomData,
        }
    }

    /// The declared label map, in label order.
    pub fn label_names(&self) -> &[String] {
        &self.label_names
    }

    /// The training split — the only selection pool.
    pub fn train(&self) -> &Split<Train> {
        &self.splits.train
    }

    /// The validation split.
    pub fn validation(&self) -> &Split<Validation> {
        &self.splits.validation
    }

    /// The held-out test split.
    pub fn test(&self) -> &Split<Test> {
        &self.splits.test
    }

    /// A proof that this dataset has a validation split.
    ///
    /// **This method exists only on the canonical type.** Its absence on
    /// `PreparedDataset<Compatibility>` is DATA-06's compile-time gate:
    ///
    /// ```compile_fail
    /// use aprender_contrastive_data::prepared::{Compatibility, PreparedDataset};
    ///
    /// fn take_witness(dataset: &PreparedDataset<Compatibility>) {
    ///     let _ = dataset.validation_witness();
    /// }
    /// ```
    ///
    /// The same call on the canonical type compiles, which is what stops the block above
    /// from being green for an unrelated reason:
    ///
    /// ```
    /// use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
    ///
    /// fn take_witness(dataset: &PreparedDataset<Canonical>) {
    ///     let _ = dataset.validation_witness();
    /// }
    /// ```
    pub fn validation_witness(&self) -> ValidationWitness<'_> {
        ValidationWitness {
            validation: &self.splits.validation,
            split_fingerprint: split_fingerprint_of::<Validation>(&self.splits.validation),
            dataset_fingerprint: self.fingerprint.clone(),
        }
    }

    /// What cross-split duplication removed from the training pool.
    pub fn exclusions(&self) -> &ExclusionRecord {
        &self.exclusions
    }

    /// This dataset's identity.
    pub fn fingerprint(&self) -> &DatasetFingerprint {
        &self.fingerprint
    }

    /// Canonical JSONL bytes per split.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if a split cannot be re-encoded.
    pub fn encode_jsonl(&self) -> Result<PreparedJsonl, ContrastiveDataError> {
        let mut splits = BTreeMap::new();
        splits.insert(
            Train::ROLE.to_string(),
            crate::schema::encode_jsonl(self.splits.train.rows())?,
        );
        splits.insert(
            Validation::ROLE.to_string(),
            crate::schema::encode_jsonl(self.splits.validation.rows())?,
        );
        splits.insert(
            Test::ROLE.to_string(),
            crate::schema::encode_jsonl(self.splits.test.rows())?,
        );
        Ok(PreparedJsonl { splits })
    }
}

impl PreparedDataset<Compatibility> {
    /// Ingest the compatibility profile's two split row sets.
    ///
    /// Selection cannot consume the result:
    ///
    /// ```compile_fail
    /// use aprender_contrastive_data::prepared::{Canonical, Compatibility, PreparedDataset};
    ///
    /// fn selection(_dataset: &PreparedDataset<Canonical>) {}
    ///
    /// fn call(compatibility: &PreparedDataset<Compatibility>) {
    ///     selection(compatibility);
    /// }
    /// ```
    ///
    /// The canonical control, which must compile:
    ///
    /// ```
    /// use aprender_contrastive_data::prepared::{Canonical, PreparedDataset};
    ///
    /// fn selection(_dataset: &PreparedDataset<Canonical>) {}
    ///
    /// fn call(canonical: &PreparedDataset<Canonical>) {
    ///     selection(canonical);
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// Any gate-ladder variant from the split boundary.
    pub fn from_labeled_rows(
        train: Vec<LabeledExample>,
        compatibility_test: Vec<LabeledExample>,
        decls: &CompatibilityDeclarations,
        ledger: &mut AccessLedger,
    ) -> Result<Self, ContrastiveDataError> {
        let train = Split::<Train>::from_rows(train, &decls.train)?;
        let compatibility_test =
            Split::<CompatibilityTest>::from_rows(compatibility_test, &decls.compatibility_test)?;
        Ok(Self::from_validated_splits(
            train,
            compatibility_test,
            &decls.label_names,
            ledger,
        ))
    }

    /// Assemble from splits that have ALREADY passed the ingest ladder.
    ///
    /// The compatibility profile's single assembly point, for the same reason the
    /// canonical one has exactly one.
    pub(crate) fn from_validated_splits(
        train: Split<Train>,
        compatibility_test: Split<CompatibilityTest>,
        label_names: &[String],
        ledger: &mut AccessLedger,
    ) -> Self {
        let fingerprint = {
            let train_pairs = train.exact_hash_pairs();
            let compatibility_pairs = compatibility_test.exact_hash_pairs();
            let splits = [
                fingerprint_input::<CompatibilityTest>(&compatibility_test, &compatibility_pairs),
                fingerprint_input::<Train>(&train, &train_pairs),
            ];
            DatasetFingerprint::compute(&DatasetFingerprintInput {
                profile: Compatibility::PROFILE,
                label_names,
                normalization_version: CONTENT_NORMALIZATION_VERSION,
                splits: &splits,
            })
        };

        let exclusions = coalesced_exclusions(&[
            (CompatibilityTest::ROLE, compatibility_test.rows()),
            (Train::ROLE, train.rows()),
        ]);

        let fingerprint_hex = fingerprint.hex();
        for role in [Train::ROLE, CompatibilityTest::ROLE] {
            ledger.record(role, Compatibility::PROFILE, "ingest", &fingerprint_hex);
        }

        Self {
            splits: CompatibilitySplits {
                train,
                compatibility_test,
            },
            exclusions,
            fingerprint,
            label_names: label_names.to_vec(),
            profile: PhantomData,
        }
    }

    /// The declared label map, in label order.
    pub fn label_names(&self) -> &[String] {
        &self.label_names
    }

    /// The training split.
    pub fn train(&self) -> &Split<Train> {
        &self.splits.train
    }

    /// The merged compatibility test split.
    pub fn compatibility_test(&self) -> &Split<CompatibilityTest> {
        &self.splits.compatibility_test
    }

    /// What cross-split duplication removed from the training pool.
    pub fn exclusions(&self) -> &ExclusionRecord {
        &self.exclusions
    }

    /// This dataset's identity.
    pub fn fingerprint(&self) -> &DatasetFingerprint {
        &self.fingerprint
    }

    /// Canonical JSONL bytes per split.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if a split cannot be re-encoded.
    pub fn encode_jsonl(&self) -> Result<PreparedJsonl, ContrastiveDataError> {
        let mut splits = BTreeMap::new();
        splits.insert(
            Train::ROLE.to_string(),
            crate::schema::encode_jsonl(self.splits.train.rows())?,
        );
        splits.insert(
            CompatibilityTest::ROLE.to_string(),
            crate::schema::encode_jsonl(self.splits.compatibility_test.rows())?,
        );
        Ok(PreparedJsonl { splits })
    }
}

/// One split's raw parts, in the shape both fingerprints absorb.
///
/// `pairs` is passed in rather than built here so the caller controls its lifetime: the
/// dataset fingerprint needs every split's pairs alive at once.
fn fingerprint_input<'a, R: SplitRole>(
    split: &'a Split<R>,
    pairs: &'a [(&'a str, [u8; 32])],
) -> SplitFingerprintInput<'a> {
    SplitFingerprintInput {
        role: R::ROLE,
        source_hash: split.source_hash(),
        class_counts: split.class_counts(),
        rows: pairs,
    }
}

/// The single-split fingerprint of one split, built from THE SAME raw parts that went into
/// the dataset fingerprint — which is what makes the two digests provably describe the same
/// bytes under different domain tags rather than merely look related.
fn split_fingerprint_of<R: SplitRole>(split: &Split<R>) -> SplitFingerprint {
    let pairs = split.exact_hash_pairs();
    SplitFingerprint::compute(&fingerprint_input::<R>(split, &pairs))
}

#[cfg(test)]
mod prepared_tests {
    use super::{
        Canonical, CanonicalDeclarations, Compatibility, CompatibilityDeclarations, PreparedDataset,
    };
    use crate::dedup::coalesced_exclusions;
    use crate::error::ContrastiveDataError;
    use crate::ledger::AccessLedger;
    use crate::schema::{encode_jsonl, parse_jsonl_bytes, LabeledExample};
    use crate::split::SplitDeclaration;

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

    fn train_rows() -> Vec<LabeledExample> {
        vec![
            row("train:0", "alpha post", 0, "train"),
            row("train:1", "beta post", 1, "train"),
            row("train:2", "gamma post", 2, "train"),
            row("train:3", "delta post", 0, "train"),
        ]
    }

    fn validation_rows() -> Vec<LabeledExample> {
        vec![
            row("validation:0", "epsilon post", 0, "validation"),
            row("validation:1", "zeta post", 1, "validation"),
        ]
    }

    fn test_rows() -> Vec<LabeledExample> {
        vec![
            row("test:0", "eta post", 2, "test"),
            row("test:1", "theta post", 1, "test"),
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

    fn build_canonical(
        ledger: &mut AccessLedger,
    ) -> Result<PreparedDataset<Canonical>, ContrastiveDataError> {
        PreparedDataset::<Canonical>::from_labeled_rows(
            train_rows(),
            validation_rows(),
            test_rows(),
            &canonical_decls(),
            ledger,
        )
    }

    fn compatibility_rows() -> Vec<LabeledExample> {
        vec![
            row("compatibility_test:0", "eta post", 2, "compatibility_test"),
            row(
                "compatibility_test:1",
                "theta post",
                1,
                "compatibility_test",
            ),
        ]
    }

    fn compatibility_decls() -> CompatibilityDeclarations {
        CompatibilityDeclarations {
            train: decl(vec![2, 1, 1]),
            compatibility_test: decl(vec![0, 1, 1]),
            label_names: label_names(),
        }
    }

    #[test]
    fn prepared_canonical_binds_three_splits_and_records_three_accesses() {
        let mut ledger = AccessLedger::new();
        let dataset = build_canonical(&mut ledger).expect("valid canonical corpus");

        assert_eq!(dataset.train().rows().len(), 4);
        assert_eq!(dataset.validation().rows().len(), 2);
        assert_eq!(dataset.test().rows().len(), 2);
        assert_eq!(ledger.records().len(), 3);
        assert!(ledger
            .records()
            .iter()
            .all(|record| record.profile == "canonical"));
        let roles: Vec<&str> = ledger
            .records()
            .iter()
            .map(|record| record.role.as_str())
            .collect();
        assert_eq!(roles, vec!["train", "validation", "test"]);
        assert!(ledger
            .records()
            .iter()
            .all(|record| record.fingerprint_hex == dataset.fingerprint().hex()));
    }

    #[test]
    fn prepared_compatibility_binds_two_splits_and_records_two_accesses() {
        let mut ledger = AccessLedger::new();
        let dataset = PreparedDataset::<Compatibility>::from_labeled_rows(
            train_rows(),
            compatibility_rows(),
            &compatibility_decls(),
            &mut ledger,
        )
        .expect("valid compatibility corpus");

        assert_eq!(dataset.train().rows().len(), 4);
        assert_eq!(dataset.compatibility_test().rows().len(), 2);
        assert_eq!(ledger.records().len(), 2);
        assert!(ledger
            .records()
            .iter()
            .all(|record| record.profile == "compatibility"));
        let roles: Vec<&str> = ledger
            .records()
            .iter()
            .map(|record| record.role.as_str())
            .collect();
        assert_eq!(roles, vec!["train", "compatibility_test"]);
    }

    #[test]
    fn prepared_witness_describes_this_dataset_and_the_validation_split_separately() {
        let mut ledger = AccessLedger::new();
        let dataset = build_canonical(&mut ledger).expect("valid canonical corpus");
        let witness = dataset.validation_witness();

        assert_eq!(
            witness.dataset_fingerprint_hex(),
            dataset.fingerprint().hex()
        );
        assert_ne!(
            witness.fingerprint_hex(),
            witness.dataset_fingerprint_hex(),
            "the validation fingerprint must not be a second copy of the dataset fingerprint"
        );
        assert_eq!(witness.validation().rows().len(), 2);
    }

    #[test]
    fn prepared_the_two_construction_paths_fingerprint_identically() {
        let mut direct_ledger = AccessLedger::new();
        let direct = build_canonical(&mut direct_ledger).expect("direct path");

        let reparse = |rows: Vec<LabeledExample>, role: &str| {
            parse_jsonl_bytes(&encode_jsonl(&rows).expect("encode"), role).expect("parse")
        };
        let mut round_ledger = AccessLedger::new();
        let round_trip = PreparedDataset::<Canonical>::from_labeled_rows(
            reparse(train_rows(), "train"),
            reparse(validation_rows(), "validation"),
            reparse(test_rows(), "test"),
            &canonical_decls(),
            &mut round_ledger,
        )
        .expect("round-trip path");

        assert_eq!(direct.fingerprint().hex(), round_trip.fingerprint().hex());
        assert_eq!(direct.exclusions(), round_trip.exclusions());
    }

    #[test]
    fn prepared_encode_jsonl_round_trips_into_an_equal_dataset() {
        let mut ledger = AccessLedger::new();
        let dataset = build_canonical(&mut ledger).expect("valid canonical corpus");
        let encoded = dataset.encode_jsonl().expect("encode must succeed");

        assert_eq!(encoded.as_map().len(), 3);
        let take = |role: &str| {
            parse_jsonl_bytes(encoded.get(role).expect("role is present"), role).expect("parse")
        };
        let mut replay_ledger = AccessLedger::new();
        let replayed = PreparedDataset::<Canonical>::from_labeled_rows(
            take("train"),
            take("validation"),
            take("test"),
            &canonical_decls(),
            &mut replay_ledger,
        )
        .expect("replay must succeed");

        assert_eq!(replayed.fingerprint().hex(), dataset.fingerprint().hex());
        assert_eq!(replayed.exclusions(), dataset.exclusions());
    }

    /// D-27, half one: prepare-time duplicate CONTENT is excluded and recorded, and the
    /// construction SUCCEEDS.
    #[test]
    fn prepare_time_duplicate_content_is_excluded_not_fatal() {
        let mut validation = validation_rows();
        validation[0].input = "alpha post".to_string();

        let mut ledger = AccessLedger::new();
        let dataset = PreparedDataset::<Canonical>::from_labeled_rows(
            train_rows(),
            validation,
            test_rows(),
            &canonical_decls(),
            &mut ledger,
        )
        .expect("a cross-split duplicate must NOT be fatal at prepare time");

        assert_eq!(
            dataset.exclusions().excluded_train_ids(),
            ["train:0".to_string()]
        );
        assert_eq!(dataset.exclusions().groups().len(), 1);
        assert_eq!(dataset.exclusions().reduced_pools().get(&0), Some(&1));
    }

    /// D-27, half two: actual split-role SPAN is a typed error.
    #[test]
    fn split_role_span_is_fail_closed() {
        let mut ledger = AccessLedger::new();
        let err = PreparedDataset::<Canonical>::from_labeled_rows(
            train_rows(),
            validation_rows(),
            compatibility_rows(),
            &canonical_decls(),
            &mut ledger,
        )
        .expect_err("compatibility rows must not become a canonical test split");

        match err {
            ContrastiveDataError::SplitRoleMismatch {
                expected_role,
                embedded_role,
            } => {
                assert_eq!(expected_role, "test");
                assert_eq!(embedded_role, "compatibility_test");
            }
            other => panic!("expected SplitRoleMismatch, got {other:?}"),
        }
        assert!(
            ledger.records().is_empty(),
            "a rejected dataset must leave no access record"
        );
    }

    #[test]
    fn prepared_runs_dedup_over_its_own_typed_splits() {
        let mut validation = validation_rows();
        validation[0].input = "alpha post".to_string();
        let mut ledger = AccessLedger::new();
        let dataset = PreparedDataset::<Canonical>::from_labeled_rows(
            train_rows(),
            validation.clone(),
            test_rows(),
            &canonical_decls(),
            &mut ledger,
        )
        .expect("valid canonical corpus");

        let expected = coalesced_exclusions(&[
            ("test", &test_rows()),
            ("train", &train_rows()),
            ("validation", &validation),
        ]);
        assert_eq!(dataset.exclusions(), &expected);
    }

    #[test]
    fn prepared_fingerprint_separates_the_two_profiles() {
        let mut canonical_ledger = AccessLedger::new();
        let canonical = build_canonical(&mut canonical_ledger).expect("canonical");
        let mut compatibility_ledger = AccessLedger::new();
        let compatibility = PreparedDataset::<Compatibility>::from_labeled_rows(
            train_rows(),
            compatibility_rows(),
            &compatibility_decls(),
            &mut compatibility_ledger,
        )
        .expect("compatibility");

        assert_ne!(
            canonical.fingerprint().hex(),
            compatibility.fingerprint().hex()
        );
    }

    #[test]
    fn prepared_rejects_a_declaration_whose_counts_disagree() {
        let mut decls = canonical_decls();
        decls.validation = decl(vec![2, 0, 0]);
        let mut ledger = AccessLedger::new();
        let err = PreparedDataset::<Canonical>::from_labeled_rows(
            train_rows(),
            validation_rows(),
            test_rows(),
            &decls,
            &mut ledger,
        )
        .expect_err("class-count contract must fail");
        assert!(matches!(
            err,
            ContrastiveDataError::InvalidClassCounts { .. }
        ));
    }
}
