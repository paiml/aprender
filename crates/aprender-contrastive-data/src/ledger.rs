//! Append-only access ledger: which splits were touched, under which profile.
//!
//! The ledger has a deterministic canonical byte form and a `ledger_hash`, and both the
//! records and the hash are embedded in the selection manifest payload. A ledger that
//! exists only in memory is not evidence — the downstream selection-lock gate needs an
//! artifact it can read.
//!
//! # Phase 2 records; a later phase reads (D-16)
//!
//! Nothing in this phase consumes the ledger to make a decision. The persistence surface
//! (`to_canonical_bytes` / `from_bytes` / `ledger_hash`) exists now because a selection
//! lock cannot be enforced against an in-memory value that died with the process that
//! built it, and retrofitting a canonical byte form later would invalidate every manifest
//! produced before the retrofit.
//!
//! # Determinism
//!
//! The canonical form is compact JSON over structs only — no maps whose iteration order
//! could vary, no timestamps, no build metadata. Two runs that touched the same splits in
//! the same order produce byte-identical ledgers and therefore the same hash.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::ContrastiveDataError;

/// The canonical ledger schema version.
const LEDGER_SCHEMA_VERSION: u32 = 1;

/// One recorded split access.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccessRecord {
    /// The split role that was touched.
    pub role: String,
    /// The dataset profile under which it was touched.
    pub profile: String,
    /// Why it was touched (`ingest`, `select`, ...).
    pub purpose: String,
    /// Hex dataset fingerprint the access belonged to.
    pub fingerprint_hex: String,
}

/// The canonical wire form. A struct, not a map, so field order is fixed by the type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LedgerWire {
    schema_version: u32,
    records: Vec<AccessRecord>,
}

/// An append-only log of split accesses.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccessLedger {
    records: Vec<AccessRecord>,
}

impl AccessLedger {
    /// An empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one access. There is no removal API: an append-only log that can be
    /// rewritten is not evidence.
    pub fn record(&mut self, role: &str, profile: &str, purpose: &str, fingerprint_hex: &str) {
        self.records.push(AccessRecord {
            role: role.to_string(),
            profile: profile.to_string(),
            purpose: purpose.to_string(),
            fingerprint_hex: fingerprint_hex.to_string(),
        });
    }

    /// The recorded accesses, in call order.
    pub fn records(&self) -> &[AccessRecord] {
        &self.records
    }

    /// Deterministic canonical serialization — the artifact a later phase's selection
    /// lock reads.
    ///
    /// Compact JSON over a struct with a fixed field order and a `Vec` whose order IS the
    /// access order. There is no map to iterate and no timestamp to drift, so two runs
    /// that touched the same splits in the same order produce byte-identical output.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if the ledger cannot be serialized.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "access_ledger_persistence"
    )]
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ContrastiveDataError> {
        let wire = LedgerWire {
            schema_version: LEDGER_SCHEMA_VERSION,
            records: self.records.clone(),
        };
        serde_json::to_vec(&wire).map_err(|error| ContrastiveDataError::Serialization {
            context: "access_ledger".to_string(),
            detail: error.to_string(),
        })
    }

    /// Parse a canonical ledger.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] on malformed bytes, or
    /// [`ContrastiveDataError::UnsupportedSchemaVersion`] on a future schema.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContrastiveDataError> {
        let wire: LedgerWire =
            serde_json::from_slice(bytes).map_err(|error| ContrastiveDataError::Serialization {
                context: "access_ledger".to_string(),
                detail: error.to_string(),
            })?;
        if wire.schema_version != LEDGER_SCHEMA_VERSION {
            return Err(ContrastiveDataError::UnsupportedSchemaVersion {
                field: "access_ledger".to_string(),
                got: wire.schema_version,
                supported: LEDGER_SCHEMA_VERSION,
            });
        }
        Ok(Self {
            records: wire.records,
        })
    }

    /// SHA-256 of [`Self::to_canonical_bytes`].
    ///
    /// Total rather than fallible: the canonical form is a `u32` and a vector of structs
    /// of `String`s, which has no non-string map key and no non-finite float, so
    /// `serde_json` has no failure mode to report. The `expect` documents that reasoning
    /// at the one place it is relied upon rather than pushing a `Result` into every
    /// caller that only ever wants a digest.
    pub fn ledger_hash(&self) -> [u8; 32] {
        let bytes = self
            .to_canonical_bytes()
            .expect("AccessLedger canonical form is strings and integers; serialization is total");
        Sha256::digest(bytes).into()
    }
}

#[cfg(test)]
mod ledger_tests {
    use super::{AccessLedger, AccessRecord};
    use crate::error::ContrastiveDataError;
    use crate::hash::hex;

    fn populated() -> AccessLedger {
        let mut ledger = AccessLedger::new();
        ledger.record("train", "canonical", "ingest", "aa");
        ledger.record("validation", "canonical", "ingest", "aa");
        ledger.record("test", "canonical", "ingest", "aa");
        ledger
    }

    #[test]
    fn ledger_records_append_in_call_order() {
        let ledger = populated();
        let roles: Vec<&str> = ledger
            .records()
            .iter()
            .map(|record| record.role.as_str())
            .collect();
        assert_eq!(roles, vec!["train", "validation", "test"]);
        assert!(AccessLedger::new().records().is_empty());
    }

    #[test]
    fn ledger_canonical_bytes_are_deterministic() {
        let first = populated()
            .to_canonical_bytes()
            .expect("serialization must succeed");
        let second = populated()
            .to_canonical_bytes()
            .expect("serialization must succeed");
        assert_eq!(first, second);
        assert!(!first.is_empty());
    }

    #[test]
    fn ledger_round_trips_through_its_canonical_bytes() {
        let ledger = populated();
        let bytes = ledger
            .to_canonical_bytes()
            .expect("serialization must succeed");
        let restored = AccessLedger::from_bytes(&bytes).expect("round-trip must succeed");
        assert_eq!(restored.records(), ledger.records());
        assert_eq!(restored.ledger_hash(), ledger.ledger_hash());
    }

    #[test]
    fn ledger_hash_is_stable_across_two_identical_constructions() {
        assert_eq!(populated().ledger_hash(), populated().ledger_hash());
    }

    #[test]
    fn ledger_hash_changes_when_any_record_field_changes() {
        let baseline = populated().ledger_hash();
        let mutations: Vec<(&str, fn(&mut AccessLedger))> = vec![
            ("role", |ledger| {
                ledger.record("compatibility_test", "canonical", "ingest", "aa");
            }),
            ("profile", |ledger| {
                ledger.record("train", "compatibility", "ingest", "aa");
            }),
            ("purpose", |ledger| {
                ledger.record("train", "canonical", "select", "aa");
            }),
            ("fingerprint", |ledger| {
                ledger.record("train", "canonical", "ingest", "bb");
            }),
        ];
        let mut seen = vec![hex(&baseline)];
        for (field, mutate) in mutations {
            let mut ledger = AccessLedger::new();
            ledger.record("train", "canonical", "ingest", "aa");
            ledger.record("validation", "canonical", "ingest", "aa");
            mutate(&mut ledger);
            let mutated = hex(&ledger.ledger_hash());
            assert!(
                !seen.contains(&mutated),
                "ledger_hash must change when {field} changes"
            );
            seen.push(mutated);
        }
    }

    #[test]
    fn ledger_hash_equals_the_digest_of_its_canonical_bytes() {
        let ledger = populated();
        let bytes = ledger
            .to_canonical_bytes()
            .expect("serialization must succeed");
        let expected: [u8; 32] = <sha2::Sha256 as sha2::Digest>::digest(&bytes).into();
        assert_eq!(ledger.ledger_hash(), expected);
    }

    #[test]
    fn ledger_rejects_an_unknown_field() {
        let bytes = br#"{"schema_version":1,"records":[],"extra":true}"#;
        let err = AccessLedger::from_bytes(bytes).expect_err("unknown field must fail");
        assert!(matches!(err, ContrastiveDataError::Serialization { .. }));
    }

    #[test]
    fn ledger_rejects_a_future_schema_version() {
        let bytes = br#"{"schema_version":99,"records":[]}"#;
        let err = AccessLedger::from_bytes(bytes).expect_err("future schema must fail");
        match err {
            ContrastiveDataError::UnsupportedSchemaVersion {
                field,
                got,
                supported,
            } => {
                assert_eq!(field, "access_ledger");
                assert_eq!(got, 99);
                assert_eq!(supported, 1);
            }
            other => panic!("expected UnsupportedSchemaVersion, got {other:?}"),
        }
    }

    #[test]
    fn ledger_records_deserialize_into_the_shape_downstream_manifests_embed() {
        let record = AccessRecord {
            role: "train".to_string(),
            profile: "canonical".to_string(),
            purpose: "ingest".to_string(),
            fingerprint_hex: "aa".to_string(),
        };
        let json = serde_json::to_string(&record).expect("record serializes");
        assert_eq!(
            json,
            r#"{"role":"train","profile":"canonical","purpose":"ingest","fingerprint_hex":"aa"}"#
        );
        let restored: AccessRecord = serde_json::from_str(&json).expect("record round-trips");
        assert_eq!(restored, record);
    }
}
