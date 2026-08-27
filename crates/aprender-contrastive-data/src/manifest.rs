//! Canonical serialization and semantic hashing for every manifest in the protocol.
//!
//! The hashed canonical payload never contains its own digest and never contains volatile
//! metadata such as a creation timestamp; the digest and the volatile fields live in an
//! outer envelope. A payload that includes its own hash cannot be recomputed, and a
//! timestamp inside the hashed region makes two identical runs compare unequal.
//!
//! # The two objects, and why they are two
//!
//! [`SelectionPayload`] is **the hashed object**. It holds everything that decides what a
//! selection IS — schema and algorithm versions, profile, both fingerprints, the label
//! map, the normalization version, the seed, the shot count, the ordered examples, the
//! exclusion record, and the access ledger with its hash. It contains no digest of
//! itself.
//!
//! The outer envelope (added with `SelectionManifest`) holds the digest and the unhashed
//! volatile block. Splitting them is not tidiness: `semantic_hash == SHA256(payload)` is
//! simply false for any payload that carries `semantic_hash`, so the one-object version of
//! this schema described a digest nobody could recompute.
//!
//! # Canonical bytes
//!
//! Compact `serde_json` over structs with a fixed field order. Every keyed structure that
//! reaches these bytes is a `BTreeMap`, so nothing serialized here depends on hash
//! iteration order. Two identical runs produce byte-identical payloads.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::dedup::ExclusionRecord;
use crate::error::ContrastiveDataError;
use crate::hash::hex;
use crate::ledger::{AccessLedger, AccessRecord};
use crate::pairs::{
    LabeledPair, PairConfig, PairSampler, PairStrategy, SingletonPolicy, UntrustedPairRecord,
    DEGENERATE_POLICY_VERSION,
};
use crate::select::{SelectedExample, Selection};

/// The selection-manifest schema version this build writes.
pub const SELECTION_SCHEMA_VERSION: u32 = 1;

/// Every selection-manifest schema version this build can read.
pub const SUPPORTED_SELECTION_SCHEMA_VERSIONS: &[u32] = &[1];

/// One selected row in its serialized form.
///
/// The two digests are hex rather than byte arrays because this object is read by humans
/// during an audit at least as often as by a program, and a 32-element JSON array of
/// integers is not readable. The hex rendering is lowercase and fixed-width, so it is
/// still byte-stable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedExampleRecord {
    /// The row identifier.
    pub id: String,
    /// The row's class label.
    pub label: usize,
    /// Lowercase hex of the exact SHA-256 content hash.
    pub exact_hash: String,
    /// Lowercase hex of the `nfc-trim-ws-v1` normalized content hash.
    pub normalized_hash: String,
}

impl SelectedExampleRecord {
    /// Serialized form of a selected example.
    pub(crate) fn from_example(example: &SelectedExample) -> Self {
        Self {
            id: example.id.clone(),
            label: example.label,
            exact_hash: hex(&example.exact_hash),
            normalized_hash: hex(&example.normalized_hash),
        }
    }
}

/// Volatile, NEVER-hashed metadata.
///
/// # Why `created_at` is a caller-supplied string
///
/// This crate reads no clock, just as it opens no file (D-04). A timestamp minted inside
/// the library would be ambient input the caller cannot control, which is exactly what
/// makes a "deterministic" artifact irreproducible. `from_selection` therefore leaves
/// `created_at` empty and the CLI fills it in; the field is outside the hashed region, so
/// filling it changes no digest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VolatileMetadata {
    /// When the manifest was written, in whatever format the caller records. Never hashed.
    pub created_at: String,
    /// The tool version that wrote it. Never hashed.
    pub tool_version: String,
}

/// The HASHED object. It contains NO digest of itself.
///
/// # What the `exclusions` field proves, and what it does not
///
/// Stated in the `revision_verified` spirit: `exclusions` proves which training ids the
/// cross-split duplicate detector removed from the selection pool **for the dataset this
/// payload's `dataset_fingerprint` names**, under the normalization version recorded
/// beside it. It does NOT prove the upstream corpus is duplicate-free, does not prove the
/// detector saw every split a consumer might later add, and says nothing about
/// near-duplicates that neither hash relation catches. A replay compares it for equality
/// against a freshly computed record; that comparison is the only claim it supports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionPayload {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Selection-algorithm version. A change here changes selected identities.
    pub algorithm_version: u32,
    /// The dataset profile the selection ran under. Always `canonical`.
    pub profile: String,
    /// Hex fingerprint of the WHOLE dataset.
    pub dataset_fingerprint: String,
    /// Hex fingerprint of the VALIDATION SPLIT ALONE — a different value from the above.
    pub validation_fingerprint: String,
    /// The ordered label map.
    pub label_names: Vec<String>,
    /// The content-normalization version the hashes were taken under.
    pub normalization_version: String,
    /// The root seed the selection was drawn from.
    pub root_seed: u64,
    /// Shots per class.
    pub shots_per_class: u32,
    /// The ORDERED selected examples: classes ascending, draw order within a class.
    pub ordered_examples: Vec<SelectedExampleRecord>,
    /// What cross-split duplication removed from the pool.
    pub exclusions: ExclusionRecord,
    /// The PERSISTED access ledger as of the moment selection finished.
    pub access_ledger: Vec<AccessRecord>,
    /// Hex `ledger_hash` of that ledger.
    pub ledger_hash: String,
}

impl SelectionPayload {
    /// Deterministic canonical serialization — the bytes whose SHA-256 IS the semantic
    /// hash.
    ///
    /// Compact JSON over a struct with a fixed field order. There is no timestamp, no
    /// path, no hostname, and no hash-ordered map anywhere inside, so two identical runs
    /// produce byte-identical output.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if the payload cannot be serialized.
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "selection_canonical_payload"
    )]
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ContrastiveDataError> {
        serde_json::to_vec(self).map_err(|error| ContrastiveDataError::Serialization {
            context: "selection_payload".to_string(),
            detail: error.to_string(),
        })
    }
}

/// The OUTER envelope: digest, unhashed volatile block, payload.
///
/// This is the on-disk form of `selection-manifest.json`. A consumer writes
/// [`Self::to_file_bytes`] verbatim and composes no JSON of its own, so there is exactly
/// one serializer and no second place for the byte form to drift.
///
/// The three fields are public because forging one must be *possible* for the defence to
/// be meaningful: [`Selection::replay`](crate::select::Selection::replay) is what makes a
/// forged manifest useless, not the privacy of a struct field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionManifest {
    /// Lowercase hex of `SHA-256(payload.to_canonical_bytes())`.
    pub semantic_hash: String,
    /// Volatile metadata. NEVER part of the digest.
    pub volatile: VolatileMetadata,
    /// The hashed payload.
    pub payload: SelectionPayload,
}

impl SelectionManifest {
    /// Wrap the payload a [`Selection`](crate::select::Selection) already retains.
    ///
    /// This is a WRAP, not a rebuild. The selection carries the exact payload its
    /// `semantic_hash` was taken over, so re-deriving one here could only introduce a way
    /// for the two to disagree.
    ///
    /// # The ledger guard
    ///
    /// The payload embeds `access_ledger` and `ledger_hash` describing the ledger **as it
    /// stood when `select` finished**. If the caller has appended to that ledger since,
    /// the payload no longer attests the ledger being handed in, and wrapping it would
    /// publish a manifest whose persisted ledger is quietly stale. That is refused.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::SemanticHashMismatch`] when
    /// `ledger.ledger_hash() != sel.ledger_hash()`.
    pub fn from_selection(
        sel: &Selection,
        ledger: &AccessLedger,
    ) -> Result<Self, ContrastiveDataError> {
        let live = ledger.ledger_hash();
        if live != sel.ledger_hash() {
            return Err(ContrastiveDataError::SemanticHashMismatch {
                expected: hex(&sel.ledger_hash()),
                got: hex(&live),
            });
        }
        Ok(Self {
            semantic_hash: hex(&sel.semantic_hash()),
            volatile: VolatileMetadata {
                created_at: String::new(),
                tool_version: env!("CARGO_PKG_VERSION").to_string(),
            },
            payload: sel.payload().clone(),
        })
    }

    /// The on-disk byte form: pretty JSON plus a terminating newline.
    ///
    /// Pretty rather than compact because this file is reviewed in diffs; the DIGEST is
    /// taken over the payload's own compact canonical bytes, so the file's whitespace can
    /// be chosen for humans without weakening anything.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if the envelope cannot be serialized.
    pub fn to_file_bytes(&self) -> Result<Vec<u8>, ContrastiveDataError> {
        let mut bytes = serde_json::to_vec_pretty(self).map_err(|error| {
            ContrastiveDataError::Serialization {
                context: "selection_manifest".to_string(),
                detail: error.to_string(),
            }
        })?;
        bytes.push(b'\n');
        Ok(bytes)
    }

    /// Parse the full file form, verifying the digest BEFORE returning.
    ///
    /// A caller therefore cannot hold a `SelectionManifest` parsed from bytes whose digest
    /// does not match its payload.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] on malformed or extended JSON;
    /// [`ContrastiveDataError::SemanticHashMismatch`] when the digest disagrees.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ContrastiveDataError> {
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|error| ContrastiveDataError::Serialization {
                context: "selection_manifest".to_string(),
                detail: error.to_string(),
            })?;
        manifest.verify_digest()?;
        Ok(manifest)
    }

    /// Recompute `SHA-256(payload.to_canonical_bytes())` and compare it to the envelope.
    ///
    /// Over the manifest's OWN payload bytes — never over a payload rebuilt from live
    /// state. See `Selection::replay` for why a live rebuild would be unsatisfiable.
    pub(crate) fn verify_digest(&self) -> Result<(), ContrastiveDataError> {
        let digest: [u8; 32] = Sha256::digest(self.payload.to_canonical_bytes()?).into();
        let recomputed = hex(&digest);
        if recomputed == self.semantic_hash {
            return Ok(());
        }
        Err(ContrastiveDataError::SemanticHashMismatch {
            expected: self.semantic_hash.clone(),
            got: recomputed,
        })
    }
}

// ===========================================================================================
// The pair half (D-09): pairs are REPLAYED, not stored
// ===========================================================================================

/// The pair-replay-record schema version this build writes and reads.
pub const PAIR_REPLAY_SCHEMA_VERSION: u32 = 1;

/// The three declared deviation clauses, copied VERBATIM from
/// `OBLIG-CPP-DEVIATION-DECLARED` in `contracts/contrastive-pair-protocol-v1.yaml`.
///
/// They are copied rather than paraphrased so the contract and every manifest cannot drift
/// apart, and they are labelled an explicit APRENDER POLICY: none of the three is
/// attributable to SetFit (D-15 / PF-008). Clause 3 in particular says the opposite of what
/// the pinned implementation does — `setfit==1.1.3` INCLUDES the diagonal, contradicting
/// its own published documentation, and Aprender's exclusion follows the docs, not the code.
pub const PAIR_DEVIATION_CLAUSES: [&str; 3] = [
    "Pair IDENTITIES are SAMPLED from the pair space, not enumerated-then-shuffled, so \
     identities cannot match the reference's Python RNG and only counts are comparable.",
    "The per-epoch count is CAPPED above N by a configurable hard cap.",
    "SELF-PAIRS ARE EXCLUDED, whereas the pinned setfit 1.1.3 implementation INCLUDES the \
     diagonal (`shuffle_combinations` defaults to replacement=True, i.e. \
     `np.triu_indices(n, k=0)`), which contradicts SetFit's own published documentation; \
     the exclusion is therefore ours and matches the docs, not the pinned code.",
];

/// The ~200-byte record a pair stream regenerates from.
///
/// # Pairs are replayed, not stored (D-09)
///
/// Forty benchmark cells times ten seeds times many epochs of pair bytes is gigabytes of
/// derivable data. Only this record and the manifest hash are persisted; a serverless
/// consumer carries the record instead of a file. An explicit DUMP path
/// ([`dump_pairs`]) exists for audit and fixture generation, and it is the only thing that
/// ever writes pair bytes.
///
/// Every field is present even when it holds its default, because an absent field is
/// indistinguishable from a field that was never considered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairReplayRecord {
    /// Record schema version.
    pub schema_version: u32,
    /// Hex `semantic_hash` of the selection the pairs are drawn over.
    pub selection_hash: String,
    /// The root seed every draw key derives from.
    pub root_seed: u64,
    /// Wire name of the sampling strategy.
    pub strategy: String,
    /// Version tag of that strategy. A change here changes pair IDENTITIES.
    pub strategy_version: u32,
    /// Wire name of the singleton policy.
    pub singleton_policy: String,
    /// Version tag of that policy.
    pub singleton_policy_version: u32,
    /// Version tag of the degenerate-layout policy.
    pub degenerate_policy_version: u32,
    /// The RESOLVED effective budget — post-clamp, exactly what the stream emitted.
    ///
    /// The hard cap itself is deliberately NOT persisted: the resolved budget subsumes it
    /// for replay, and the record stays small (D-09).
    pub budget: u64,
    /// Whether the DEFAULT budget was clamped. A clamped run that looked unclamped in the
    /// artifact would defeat the point of recording it.
    pub default_was_clamped: bool,
    /// `both`, `positives_only` or `negatives_only`.
    pub emitted_kinds: String,
    /// How many classes held exactly one selected example (`OBLIG-CPP-SINGLETON-EXPLICIT`).
    pub affected_singleton_classes: u64,
    /// The three declared deviation clauses, verbatim.
    pub deviation: [String; 3],
}

impl PairReplayRecord {
    /// Describe a live sampler.
    ///
    /// # Why this takes ONLY the sampler
    ///
    /// The plan's interface sketch was `from_sampler(sampler, selection, cfg)`. A sampler
    /// already BORROWS its selection and already retains its resolved configuration, so the
    /// extra parameters could only ever disagree with it — and a replay record that
    /// describes a different selection or a different budget than the stream it attests is
    /// precisely the artifact this record exists to make impossible. One argument, one
    /// source of truth.
    pub fn from_sampler(sampler: &PairSampler<'_>) -> Self {
        let layout = sampler.layout();
        Self {
            schema_version: PAIR_REPLAY_SCHEMA_VERSION,
            selection_hash: hex(&sampler.selection().semantic_hash()),
            root_seed: layout.root_seed(),
            strategy: layout.strategy().as_str().to_string(),
            strategy_version: layout.strategy().strategy_version(),
            singleton_policy: layout.singleton_policy().as_str().to_string(),
            singleton_policy_version: layout.singleton_policy().policy_version(),
            degenerate_policy_version: DEGENERATE_POLICY_VERSION,
            budget: layout.budget(),
            default_was_clamped: layout.default_was_clamped(),
            emitted_kinds: layout.emitted_kinds().as_str().to_string(),
            affected_singleton_classes: layout.affected_singleton_classes(),
            deviation: PAIR_DEVIATION_CLAUSES.map(str::to_string),
        }
    }

    /// Deterministic canonical serialization — the bytes the manifest hash commits FIRST.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if the record cannot be serialized.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ContrastiveDataError> {
        serde_json::to_vec(self).map_err(|error| ContrastiveDataError::Serialization {
            context: "pair_replay_record".to_string(),
            detail: error.to_string(),
        })
    }

    /// Rebuild the configuration that produced the stream, refusing unknown versions.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::UnsupportedSchemaVersion`],
    /// [`ContrastiveDataError::UnsupportedAlgorithmVersion`],
    /// [`ContrastiveDataError::UnsupportedPolicyVersion`].
    #[provable_contracts_macros::contract(
        "contrastive-pair-protocol-v1",
        equation = "pair_manifest_replay"
    )]
    pub fn to_config(&self) -> Result<PairConfig, ContrastiveDataError> {
        if self.schema_version != PAIR_REPLAY_SCHEMA_VERSION {
            return Err(ContrastiveDataError::UnsupportedSchemaVersion {
                field: "pair_replay".to_string(),
                got: self.schema_version,
                supported: PAIR_REPLAY_SCHEMA_VERSION,
            });
        }
        let strategy = PairStrategy::Oversampling;
        if self.strategy != strategy.as_str()
            || self.strategy_version != strategy.strategy_version()
        {
            // A strategy this build does not implement changes pair IDENTITIES, not merely
            // which pairs are legal, which is why it is the ALGORITHM variant.
            return Err(ContrastiveDataError::UnsupportedAlgorithmVersion {
                got: self.strategy_version,
                supported: strategy.strategy_version(),
            });
        }
        let policy = SingletonPolicy::NegativesOnly;
        if self.singleton_policy != policy.as_str()
            || self.singleton_policy_version != policy.policy_version()
        {
            return Err(ContrastiveDataError::UnsupportedPolicyVersion {
                policy: "singleton".to_string(),
                got: self.singleton_policy_version,
                supported: policy.policy_version(),
            });
        }
        if self.degenerate_policy_version != DEGENERATE_POLICY_VERSION {
            return Err(ContrastiveDataError::UnsupportedPolicyVersion {
                policy: "degenerate".to_string(),
                got: self.degenerate_policy_version,
                supported: DEGENERATE_POLICY_VERSION,
            });
        }
        if self.budget == 0 {
            return Err(ContrastiveDataError::ZeroBudget);
        }
        // The RESOLVED budget is replayed, and the cap is set to exactly it. The record
        // does not persist the original cap (D-09) because the resolved budget subsumes it:
        // what a replay has to reproduce is the stream, and the stream is a function of the
        // budget that was actually used.
        //
        // A CLAMPED run must be replayed through the DEFAULT route, not as an explicit
        // budget. Both routes resolve to the same number — for a clamped record the closed
        // form exceeded the cap, so `min(closed_form, budget) == budget` — but an explicit
        // budget resolves with `default_was_clamped == false`, so
        // `from_sampler(to_config(record))` would silently re-describe a clamped artifact
        // as an unclamped one. Replaying with `budget: None` and the cap pinned at the
        // resolved value reproduces the flag as well as the stream. A forged record whose
        // layout does not actually clamp resolves to a different budget and is then caught
        // by `assert_record_describes`.
        let (budget, hard_cap) = if self.default_was_clamped {
            (None, Some(self.budget))
        } else {
            (Some(self.budget), Some(self.budget))
        };
        Ok(PairConfig {
            root_seed: self.root_seed,
            strategy,
            singleton_policy: policy,
            budget,
            hard_cap,
        })
    }
}

/// One pair's canonical byte encoding: `lo` and `hi` ordinals, then the target's bits.
///
/// Twelve little-endian bytes. Ordinals rather than identifier strings because the record
/// already commits `selection_hash`, so the ordinals are unambiguous inside the digest —
/// and because a fixed-width encoding needs no delimiter and therefore has no concatenation
/// ambiguity of its own.
fn pair_canonical_bytes(pair: &LabeledPair) -> [u8; 12] {
    let mut out = [0_u8; 12];
    out[0..4].copy_from_slice(&pair.pair.lo().ordinal().to_le_bytes());
    out[4..8].copy_from_slice(&pair.pair.hi().ordinal().to_le_bytes());
    out[8..12].copy_from_slice(&pair.target.to_bits().to_le_bytes());
    out
}

/// Refuse a record that does not describe this sampler.
///
/// Without this the hash would happily attest a stream using somebody else's header, which
/// is the exact failure the header-inside-the-digest design exists to prevent.
fn assert_record_describes(
    sampler: &PairSampler<'_>,
    record: &PairReplayRecord,
) -> Result<(), ContrastiveDataError> {
    let expected = hex(&sampler.selection().semantic_hash());
    if record.selection_hash != expected {
        return Err(ContrastiveDataError::SelectionReplayMismatch {
            field: "pair_replay.selection_hash".to_string(),
        });
    }
    if record.budget != sampler.budget() {
        return Err(ContrastiveDataError::SelectionReplayMismatch {
            field: "pair_replay.budget".to_string(),
        });
    }
    Ok(())
}

/// `SHA-256( record.to_canonical_bytes() ‖ 0x1E ‖ pair_0 ‖ pair_1 ‖ … )`, STREAMED.
///
/// # Why the header is INSIDE the digest (review finding F12)
///
/// Hashing pair bytes alone would let two different replay tuples that happen to emit the
/// same identities claim the same manifest hash — so the manifest could attest a stream
/// while saying nothing about how it was produced, which is a strictly weaker statement
/// than the one the hash is supposed to make. Two tests pin this: one pair of
/// configurations whose streams are identical because both resolve to the same budget by
/// different routes, and one pair of DIFFERENT selections whose ordinal streams are
/// literally byte-identical because they share a class layout.
///
/// The `0x1E` (ASCII record separator) removes the record/stream boundary ambiguity that
/// plain concatenation would leave.
///
/// # Streamed, never collected
///
/// Pairs are fed to the hasher as they are produced. Collecting them would reintroduce the
/// `O(budget)` memory the whole design exists to avoid, in the one place nobody would look
/// for it.
///
/// # Errors
///
/// [`ContrastiveDataError::SelectionReplayMismatch`] when the record does not describe this
/// sampler; anything [`PairReplayRecord::to_canonical_bytes`] or `pair_at` can raise.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "pair_manifest_hash"
)]
pub fn pair_manifest_hash(
    sampler: &PairSampler<'_>,
    record: &PairReplayRecord,
) -> Result<[u8; 32], ContrastiveDataError> {
    assert_record_describes(sampler, record)?;
    let mut hasher = Sha256::new();
    hasher.update(record.to_canonical_bytes()?);
    hasher.update([PAIR_MANIFEST_SEPARATOR]);
    for ordinal in 0..sampler.budget() {
        hasher.update(pair_canonical_bytes(&sampler.pair_at(ordinal)?));
    }
    Ok(hasher.finalize().into())
}

/// ASCII record separator, between the replay header and the streamed pairs.
const PAIR_MANIFEST_SEPARATOR: u8 = 0x1E;

/// Write one JSON line per pair, in stream order, to any byte sink.
///
/// This is D-09's explicit audit and fixture path — the only thing in the protocol that
/// ever writes pair bytes. It takes a `std::io::Write` rather than a path: bytes-out is
/// permitted by D-04 while the filesystem stays in `apr-cli`.
///
/// The line schema is [`UntrustedPairRecord`], the SAME type
/// [`parse_pair_dump`](crate::pairs::parse_pair_dump) reads, so the writer and the untrusted
/// reader cannot drift apart into two schemas that almost agree.
///
/// # Errors
///
/// [`ContrastiveDataError::Io`] when the sink fails — a caller's broken pipe is surfaced as
/// a typed error, never swallowed and never a panic;
/// [`ContrastiveDataError::Serialization`] if a record cannot be encoded.
pub fn dump_pairs<W: std::io::Write>(
    sampler: &PairSampler<'_>,
    mut writer: W,
) -> Result<(), ContrastiveDataError> {
    let selection = sampler.selection();
    let io = |context: &str, error: &std::io::Error| ContrastiveDataError::Io {
        context: context.to_string(),
        detail: error.to_string(),
    };
    for ordinal in 0..sampler.budget() {
        let labeled = sampler.pair_at(ordinal)?;
        let record = UntrustedPairRecord {
            lo: selection.id_of(labeled.pair.lo()).to_string(),
            hi: selection.id_of(labeled.pair.hi()).to_string(),
            target: labeled.target,
        };
        // Serialized into a per-line `Vec` and then written, NOT via `serde_json::to_writer`.
        // `to_writer` folds a sink failure into a `serde_json::Error`, so a broken pipe
        // mid-dump would surface as `Serialization` instead of `Io` — the user loses the
        // one signal that says "the destination died", and
        // `dump_pairs_surfaces_a_failing_sink_as_a_typed_io_error` fails. The allocation is
        // one small `Vec` per pair against a syscall the caller's `BufWriter` already
        // coalesces, which is the cheaper half to keep.
        let mut line =
            serde_json::to_vec(&record).map_err(|error| ContrastiveDataError::Serialization {
                context: format!("pair_dump/{ordinal}"),
                detail: error.to_string(),
            })?;
        line.push(b'\n');
        writer
            .write_all(&line)
            .map_err(|error| io(&format!("pair_dump/{ordinal}"), &error))?;
    }
    writer
        .flush()
        .map_err(|error| io("pair_dump/flush", &error))
}

#[cfg(test)]
mod payload_tests {
    use super::{SelectionPayload, SELECTION_SCHEMA_VERSION, SUPPORTED_SELECTION_SCHEMA_VERSIONS};
    use crate::hash::hex;
    use crate::ledger::AccessLedger;
    use crate::select::{test_corpus, SELECTION_ALGORITHM_VERSION};
    use sha2::{Digest, Sha256};

    fn built() -> (SelectionPayload, AccessLedger) {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 13, 8, &mut ledger);
        (selection.payload().clone(), ledger)
    }

    #[test]
    fn payload_bytes_contain_no_semantic_hash_key() {
        let (payload, _) = built();
        let bytes = payload.to_canonical_bytes().expect("payload serializes");
        let text = String::from_utf8(bytes).expect("canonical JSON is UTF-8");
        assert!(
            !text.contains("semantic_hash"),
            "the hashed payload must not carry its own digest — it would be unrecomputable"
        );
        // Vacuity guard: the payload really did serialize something substantial.
        assert!(
            text.contains("ordered_examples"),
            "payload text: {text:.120}"
        );
        assert!(text.contains("ledger_hash"));
    }

    #[test]
    fn payload_canonical_bytes_are_byte_stable_across_two_builds() {
        let (first, _) = built();
        let (second, _) = built();
        assert_eq!(first, second);
        assert_eq!(
            first.to_canonical_bytes().expect("first serializes"),
            second.to_canonical_bytes().expect("second serializes")
        );
    }

    /// Checker warning 1: these two fields hold DIFFERENT values. If they did not, two of
    /// the strict-replay rejection tests would be the same test wearing two names.
    #[test]
    fn payload_dataset_and_validation_fingerprints_differ() {
        let (payload, _) = built();
        assert_ne!(payload.dataset_fingerprint, payload.validation_fingerprint);
        assert_eq!(payload.dataset_fingerprint.len(), 64);
        assert_eq!(payload.validation_fingerprint.len(), 64);
    }

    /// D-16 / review finding F7: the ledger is PERSISTED inside the hashed payload, so a
    /// later phase's selection lock has an artifact to read.
    #[test]
    fn payload_carries_the_persisted_ledger_and_its_hash() {
        let (payload, ledger) = built();
        assert_eq!(payload.access_ledger, ledger.records());
        assert_eq!(payload.ledger_hash, hex(&ledger.ledger_hash()));
        // three ingest records plus the selection's own
        assert_eq!(payload.access_ledger.len(), 4);
        assert_eq!(payload.access_ledger[3].purpose, "select");
    }

    #[test]
    fn payload_records_the_versions_this_build_writes() {
        let (payload, _) = built();
        assert_eq!(payload.schema_version, SELECTION_SCHEMA_VERSION);
        assert_eq!(payload.algorithm_version, SELECTION_ALGORITHM_VERSION);
        assert!(SUPPORTED_SELECTION_SCHEMA_VERSIONS.contains(&payload.schema_version));
        assert_eq!(payload.profile, "canonical");
        assert_eq!(payload.normalization_version, "nfc-trim-ws-v1");
        assert_eq!(payload.label_names, test_corpus::label_names());
    }

    /// There is exactly ONE hashing rule for a selection, and this is it.
    #[test]
    fn payload_digest_is_the_selections_semantic_hash() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 17, 8, &mut ledger);
        let bytes = selection
            .payload()
            .to_canonical_bytes()
            .expect("payload serializes");
        let expected: [u8; 32] = Sha256::digest(&bytes).into();
        assert_eq!(selection.semantic_hash(), expected);
    }

    #[test]
    fn payload_round_trips_through_serde_with_unknown_fields_denied() {
        let (payload, _) = built();
        let bytes = payload.to_canonical_bytes().expect("payload serializes");
        let restored: SelectionPayload =
            serde_json::from_slice(&bytes).expect("canonical bytes round-trip");
        assert_eq!(restored, payload);

        let mut text = String::from_utf8(bytes).expect("canonical JSON is UTF-8");
        text.insert_str(1, r#""extra":true,"#);
        let err = serde_json::from_str::<SelectionPayload>(&text);
        assert!(err.is_err(), "deny_unknown_fields must reject an added key");
    }
}

/// The FROZEN golden corpus and the four committed selection goldens.
///
/// # No filesystem, by construction
///
/// Every byte here arrives through `include_bytes!`, whose paths resolve against THIS
/// SOURCE FILE at compile time. The verifier is therefore working-directory independent
/// and adds no `std::fs` to `src/` — which `make contrastive-data-boundary` bans outright,
/// with no `cfg(test)` exemption.
///
/// # What is algorithm-derived and what is capture-and-blessed
///
/// Stated plainly, because the two are not equally strong evidence:
///
/// * **Algorithm-derived.** [`GOLDEN_CASES`]'s `ordered_ids_sha256` values were computed
///   by an independent Python implementation written from the contract equations
///   (`rng_key_derivation`, `bounded_draw`, `few_shot_selection`) and from Salmon et al.
///   (2011), which never read this crate's source. They pin the SELECTION itself — which
///   rows, in which order — against a second implementation.
/// * **Capture-and-blessed.** The `*.payload.json` files are this crate's own canonical
///   serialization of those selections, written once by `tests/goldens_regenerate.rs`.
///   They pin the BYTE FORM against future drift; they do not independently corroborate
///   it. Their content is corroborated by the digests above.
///
/// Re-baselining is `cargo test -p aprender-contrastive-data --test goldens_regenerate --
/// --ignored`, and it must be a reviewed diff.
#[cfg(test)]
mod golden_tests {
    use super::{count_lines, SelectionManifest, SelectionPayload};
    use crate::hash::hex;
    use crate::ledger::AccessLedger;
    use crate::prepared::{Canonical, CanonicalDeclarations, PreparedDataset};
    use crate::schema::parse_jsonl_bytes;
    use crate::select::{FewShotSelector, Selection, SelectionConfig};
    use crate::split::SplitDeclaration;
    use sha2::{Digest, Sha256};

    pub(super) const TRAIN_JSONL: &[u8] =
        include_bytes!("../tests/goldens/golden_corpus_train.jsonl");
    pub(super) const VALIDATION_JSONL: &[u8] =
        include_bytes!("../tests/goldens/golden_corpus_validation.jsonl");
    pub(super) const TEST_JSONL: &[u8] =
        include_bytes!("../tests/goldens/golden_corpus_test.jsonl");
    const SHA256_MANIFEST: &[u8] = include_bytes!("../tests/goldens/manifest.sha256");

    /// `(seed, shots, first-32 dumped pair bytes)`.
    ///
    /// CAPTURE-AND-BLESSED, and labelled as such. Unlike `GOLDEN_CASES`'s ordered-id
    /// digests — which a second implementation written from the contract produced — these
    /// files are this crate's own dump of its own stream. They pin the BYTE FORM against
    /// drift; they do not independently corroborate it. What corroborates their CONTENT is
    /// `pair_goldens_agree_with_an_independent_rederivation`, which rebuilds the first
    /// pairs from `rng::bounded` plus a NAIVE enumeration of the triangle instead of
    /// calling the sampler's binary-search unranking.
    pub(super) const PAIR_GOLDEN_CASES: [(u64, u32, &[u8]); 2] = [
        (
            13,
            8,
            include_bytes!("../tests/goldens/pairs_seed13_shots8_first32.jsonl"),
        ),
        (
            17,
            8,
            include_bytes!("../tests/goldens/pairs_seed17_shots8_first32.jsonl"),
        ),
    ];

    /// How many pairs each committed pair golden holds.
    pub(super) const PAIR_GOLDEN_PREFIX: u64 = 32;

    /// `(seed, shots, golden payload bytes, independently derived ordered-id digest)`.
    const GOLDEN_CASES: [(u64, u32, &[u8], &str); 4] = [
        (
            13,
            8,
            include_bytes!("../tests/goldens/selection_seed13_shots8.payload.json"),
            "1c99eec4d905430e4b5d05471a01af99f27b4ef65707767f2765cb10ef57701c",
        ),
        (
            13,
            16,
            include_bytes!("../tests/goldens/selection_seed13_shots16.payload.json"),
            "1ea9826fd29e0a298c911097d973e3ca4eb7d804bd542b481264804cd658ffaa",
        ),
        (
            17,
            8,
            include_bytes!("../tests/goldens/selection_seed17_shots8.payload.json"),
            "7bb11c386e9622d151c83d6a3471b56a21da5c32fc86c9f5b62d97e91d64c763",
        ),
        (
            17,
            16,
            include_bytes!("../tests/goldens/selection_seed17_shots16.payload.json"),
            "ca7c7c4c291beb63b9e878428e46f6a9217c23d0742043dd9e7489398f9dd301",
        ),
    ];

    /// Every file the committed `manifest.sha256` covers, paired with its embedded bytes.
    ///
    /// `include_bytes!` CANNOT iterate a manifest at runtime — every digest it checks has to
    /// be embedded by name at compile time — so appending a line to `manifest.sha256`
    /// without adding a matching entry here would leave the new golden unverified while the
    /// suite stayed green. `golden_manifest_coverage_is_total` asserts the two counts are
    /// equal for exactly that reason, and `make contrastive-data-boundary` forbids the
    /// runtime-iteration escape hatch (`std::fs` under `src/`) outright.
    fn covered_files() -> Vec<(&'static str, &'static [u8])> {
        let mut files: Vec<(&str, &[u8])> = vec![
            ("golden_corpus_train.jsonl", TRAIN_JSONL),
            ("golden_corpus_validation.jsonl", VALIDATION_JSONL),
            ("golden_corpus_test.jsonl", TEST_JSONL),
        ];
        for (seed, shots, bytes, _) in GOLDEN_CASES {
            files.push((golden_name(seed, shots), bytes));
        }
        for (seed, shots, bytes) in PAIR_GOLDEN_CASES {
            files.push((pair_golden_name(seed, shots), bytes));
        }
        files
    }

    fn golden_name(seed: u64, shots: u32) -> &'static str {
        match (seed, shots) {
            (13, 8) => "selection_seed13_shots8.payload.json",
            (13, 16) => "selection_seed13_shots16.payload.json",
            (17, 8) => "selection_seed17_shots8.payload.json",
            (17, 16) => "selection_seed17_shots16.payload.json",
            other => panic!("no golden is committed for {other:?}"),
        }
    }

    pub(super) fn pair_golden_name(seed: u64, shots: u32) -> &'static str {
        match (seed, shots) {
            (13, 8) => "pairs_seed13_shots8_first32.jsonl",
            (17, 8) => "pairs_seed17_shots8_first32.jsonl",
            other => panic!("no pair golden is committed for {other:?}"),
        }
    }

    /// The golden corpus's declarations. FROZEN alongside the corpus files.
    pub(super) fn declarations() -> CanonicalDeclarations {
        let label_names = ["none", "against", "favor"]
            .iter()
            .map(|name| (*name).to_string())
            .collect::<Vec<String>>();
        let decl = |counts: Vec<usize>| SplitDeclaration {
            expected_class_counts: counts,
            label_names: label_names.clone(),
        };
        CanonicalDeclarations {
            train: decl(vec![20, 20, 20]),
            validation: decl(vec![1, 1, 1]),
            test: decl(vec![1, 1, 1]),
            label_names,
        }
    }

    pub(super) fn golden_dataset(ledger: &mut AccessLedger) -> PreparedDataset<Canonical> {
        let parse = |bytes: &[u8], role: &str| {
            parse_jsonl_bytes(bytes, role).expect("the golden corpus must parse")
        };
        PreparedDataset::<Canonical>::from_labeled_rows(
            parse(TRAIN_JSONL, "train"),
            parse(VALIDATION_JSONL, "validation"),
            parse(TEST_JSONL, "test"),
            &declarations(),
            ledger,
        )
        .expect("the golden corpus must be a valid canonical dataset")
    }

    pub(super) fn golden_selection(seed: u64, shots: u32) -> (Selection, AccessLedger) {
        let mut ledger = AccessLedger::new();
        let dataset = golden_dataset(&mut ledger);
        let selection = FewShotSelector::select(
            &dataset,
            &SelectionConfig {
                root_seed: seed,
                shots_per_class: shots,
            },
            &mut ledger,
        )
        .expect("the golden corpus must support 8 and 16 shots");
        (selection, ledger)
    }

    /// The corpus carries exactly one cross-split duplicate, so the goldens exercise a
    /// NON-EMPTY exclusion record rather than only the easy path.
    #[test]
    fn golden_corpus_has_the_shape_the_goldens_were_derived_from() {
        let mut ledger = AccessLedger::new();
        let dataset = golden_dataset(&mut ledger);
        assert_eq!(dataset.train().rows().len(), 60);
        assert_eq!(dataset.validation().rows().len(), 3);
        assert_eq!(dataset.test().rows().len(), 3);
        assert_eq!(
            dataset.exclusions().excluded_train_ids(),
            ["train:0-07".to_string()],
            "the frozen corpus must exclude exactly this row"
        );
        assert_eq!(dataset.exclusions().reduced_pools().get(&0), Some(&19));

        // Two rows carry a whitespace variant, so the goldens pin the NORMALIZED hash as a
        // value distinct from the exact one. Without them every recorded pair would be
        // identical and the goldens would say nothing about `nfc-trim-ws-v1`.
        let train = dataset.train();
        let differing = train
            .rows()
            .iter()
            .filter(|row| train.exact_hash_of(&row.id) != train.normalized_hash_of(&row.id))
            .count();
        assert_eq!(
            differing, 2,
            "the frozen corpus must contain exactly two whitespace-variant rows"
        );
    }

    /// Every non-empty, non-comment line of the embedded manifest, as `(name, digest)`.
    fn manifest_lines() -> Vec<(&'static str, &'static str)> {
        let text = core::str::from_utf8(SHA256_MANIFEST).expect("manifest.sha256 is UTF-8");
        text.lines()
            .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
            .map(|line| {
                let (digest, name) = line
                    .split_once("  ")
                    .expect("each manifest line is `<hex>  <name>`");
                (name, digest)
            })
            .collect()
    }

    #[test]
    fn golden_files_match_the_committed_sha256_manifest() {
        let recorded = manifest_lines();
        let files = covered_files();
        // Vacuity guard: pin the population BEFORE asserting a relation over it, so an
        // empty manifest cannot satisfy an empty comparison (02-04's lesson).
        assert_eq!(files.len(), 9);
        assert_eq!(recorded.len(), 9, "manifest.sha256 must cover all 9 files");

        for (name, bytes) in files {
            let digest = hex(&Sha256::digest(bytes).into());
            let expected = recorded
                .iter()
                .find(|(entry, _)| *entry == name)
                .unwrap_or_else(|| panic!("manifest.sha256 has no entry for {name}"))
                .1;
            assert_eq!(digest, expected, "digest drift in {name}");
        }
    }

    /// Coverage is TOTAL: the verifier embeds one digest check per manifest line.
    ///
    /// This is the assertion that stops a regenerated `manifest.sha256` from carrying a line
    /// nothing checks. `include_bytes!` resolves at compile time, so a new golden that is
    /// added to the manifest but not to [`covered_files`] would sit inside the integrity
    /// record and outside the integrity CHECK — which is worse than not being covered at
    /// all, because the manifest would claim it.
    #[test]
    fn golden_manifest_coverage_is_total() {
        let recorded = manifest_lines();
        let embedded = covered_files();
        assert!(!recorded.is_empty(), "an empty manifest verifies nothing");
        assert_eq!(
            embedded.len(),
            recorded.len(),
            "every manifest line needs a matching include_bytes! entry"
        );

        let mut embedded_names: Vec<&str> = embedded.iter().map(|(name, _)| *name).collect();
        let mut recorded_names: Vec<&str> = recorded.iter().map(|(name, _)| *name).collect();
        embedded_names.sort_unstable();
        recorded_names.sort_unstable();
        assert_eq!(embedded_names, recorded_names);
    }

    /// The four committed payload goldens are byte-identical to what this build produces.
    #[test]
    fn golden_selection_payload_bytes_match_the_committed_goldens() {
        for (seed, shots, expected, _) in GOLDEN_CASES {
            let (selection, _) = golden_selection(seed, shots);
            let produced = selection
                .payload()
                .to_canonical_bytes()
                .expect("payload serializes");
            assert_eq!(
                produced,
                expected,
                "golden {} drifted",
                golden_name(seed, shots)
            );
        }
    }

    /// The independently derived half: which rows, in which order.
    ///
    /// These digests came from a Python implementation of the contract equations that has
    /// never read this crate. If this test and the byte-golden test disagree, the byte
    /// golden is the one that was re-blessed.
    #[test]
    fn golden_ordered_ids_match_the_independently_derived_digests() {
        for (seed, shots, _, expected) in GOLDEN_CASES {
            let (selection, _) = golden_selection(seed, shots);
            let joined = selection.ordered_ids().join("\n");
            let digest = hex(&Sha256::digest(joined.as_bytes()).into());
            assert_eq!(
                digest, expected,
                "seed {seed} shots {shots}: ordered ids disagree with the reference derivation"
            );
            assert_eq!(selection.len() as u32, shots * 3);
        }
    }

    /// A golden payload must parse back into the payload it was written from, so a golden
    /// cannot be a well-formed file describing something else.
    #[test]
    fn golden_payloads_round_trip_into_equal_manifests() {
        for (seed, shots, bytes, _) in GOLDEN_CASES {
            let (selection, ledger) = golden_selection(seed, shots);
            let parsed: SelectionPayload =
                serde_json::from_slice(bytes).expect("a golden payload must parse");
            assert_eq!(&parsed, selection.payload());

            let manifest =
                SelectionManifest::from_selection(&selection, &ledger).expect("wrap succeeds");
            let round_tripped =
                SelectionManifest::from_bytes(&manifest.to_file_bytes().expect("file bytes"))
                    .expect("the file form verifies its own digest");
            assert_eq!(round_tripped.payload, parsed);
        }
    }

    /// The first 32 dumped pairs of each committed pair golden are byte-identical.
    #[test]
    fn pair_goldens_match_the_committed_first_32_dumps() {
        for (seed, shots, expected) in PAIR_GOLDEN_CASES {
            let (selection, _) = golden_selection(seed, shots);
            let cfg = crate::pairs::PairConfig {
                budget: Some(PAIR_GOLDEN_PREFIX),
                ..crate::pairs::PairConfig::new(seed)
            };
            let sampler = crate::pairs::PairSampler::new(&selection, &cfg)
                .expect("the golden corpus supports a 32-pair budget");
            let mut produced = Vec::new();
            super::dump_pairs(&sampler, &mut produced).expect("dumping to a Vec cannot fail");
            assert_eq!(
                produced,
                expected,
                "pair golden {} drifted",
                pair_golden_name(seed, shots)
            );
            assert_eq!(
                count_lines(&produced),
                PAIR_GOLDEN_PREFIX as usize,
                "the golden must hold exactly 32 lines"
            );
        }
    }

    /// The pair goldens are capture-and-blessed, so their CONTENT needs a second derivation.
    ///
    /// This rebuilds the first eight pairs from `rng::bounded` plus a NAIVE enumeration of
    /// the class triangle and a LINEAR scan of the weight prefixes — i.e. from the contract's
    /// text rather than from the sampler's binary-search unranking. It is not a second
    /// language, but it is a second implementation of the part that could plausibly be
    /// wrong, and the RNG primitives it stands on were themselves pinned by plan 02-05
    /// against constants a Python implementation produced.
    #[test]
    fn pair_goldens_agree_with_an_independent_rederivation() {
        use crate::pairs::{PairConfig, PairSampler};
        use crate::rng::{bounded, derive_key, domains};
        use core::num::NonZeroU64;

        let (selection, _) = golden_selection(13, 8);
        let sizes: Vec<u64> = selection.class_sizes().iter().map(|(_, n)| *n).collect();
        assert_eq!(sizes, vec![8, 8, 8]);
        let total: u64 = sizes.iter().sum();

        // Naive, deliberately quadratic reference structures.
        let pos_weights: Vec<u64> = sizes.iter().map(|n| n * (n - 1) / 2).collect();
        let neg_weights: Vec<u64> = sizes.iter().map(|n| n * (total - n)).collect();
        let offsets: Vec<u64> = (0..sizes.len())
            .map(|c| sizes[..c].iter().sum::<u64>())
            .collect();
        let scan = |weights: &[u64], target: u64| -> usize {
            let mut acc = 0;
            for (index, weight) in weights.iter().enumerate() {
                acc += weight;
                if target < acc {
                    return index;
                }
            }
            panic!("target {target} exceeds the total weight");
        };
        let nz = |n: u64| NonZeroU64::new(n).expect("non-zero by construction");

        let cfg = PairConfig {
            budget: Some(8),
            ..PairConfig::new(13)
        };
        let sampler = PairSampler::new(&selection, &cfg).expect("a legal budget");

        for ordinal in 0..8_u64 {
            let draw = ordinal / 2;
            let (class_a, member_a, class_b, member_b) = if ordinal % 2 == 0 {
                let key = derive_key(13, domains::PAIRS_POS_CLASS);
                let c = scan(
                    &pos_weights,
                    bounded(&key, 0, draw, nz(pos_weights.iter().sum())),
                );
                let rank_key = derive_key(13, domains::PAIRS_POS_RANK);
                let rank = bounded(&rank_key, 0, draw, nz(pos_weights[c]));
                // Naive triangular enumeration — the reference form of the unranking.
                let mut triangle = Vec::new();
                for i in 0..sizes[c] {
                    for j in (i + 1)..sizes[c] {
                        triangle.push((i, j));
                    }
                }
                let (i, j) = triangle[rank as usize];
                (c, i, c, j)
            } else {
                let key = derive_key(13, domains::PAIRS_NEG_CLASS);
                let j = scan(
                    &neg_weights,
                    bounded(&key, 0, draw, nz(neg_weights.iter().sum())),
                );
                let first_key = derive_key(13, domains::PAIRS_NEG_FIRST);
                let a = bounded(&first_key, 0, draw, nz(sizes[j]));
                let second_key = derive_key(13, domains::PAIRS_NEG_SECOND);
                let u = bounded(&second_key, 0, draw, nz(total - sizes[j]));
                let global = if u < offsets[j] { u } else { u + sizes[j] };
                let k = scan(&sizes, global);
                (j, a, k, global - offsets[k])
            };

            // Canonical order is by ORDINAL, not by identifier string — `train:1-10`
            // precedes `train:1-01` in the selection while following it lexically, so
            // sorting the ids here would have compared a different pair.
            let expect = |class: usize, member: u64| {
                let label = selection.class_sizes()[class].0;
                selection.ids_in_class(label)[member as usize]
            };
            let (want_a, want_b) = (expect(class_a, member_a), expect(class_b, member_b));
            let (want_lo, want_hi) = (want_a.min(want_b), want_a.max(want_b));
            let got = sampler.pair_at(ordinal).expect("below the budget");
            assert_eq!(
                (got.pair.lo(), got.pair.hi()),
                (want_lo, want_hi),
                "ordinal {ordinal} disagrees with the independent re-derivation: got \
                 ({:?}, {:?}), want ({:?}, {:?})",
                selection.id_of(got.pair.lo()),
                selection.id_of(got.pair.hi()),
                selection.id_of(want_lo),
                selection.id_of(want_hi)
            );
        }
    }
}

/// Newline count of a byte buffer.
///
/// Written as an explicit loop rather than an iterator chain because clippy's
/// `naive_bytecount` fires on the chain and suggests the `bytecount` crate — a new runtime
/// dependency the D-04 allowlist would (correctly) reject for a test-only line count.
#[cfg(test)]
fn count_lines(bytes: &[u8]) -> usize {
    let mut lines = 0;
    for byte in bytes {
        if *byte == b'\n' {
            lines += 1;
        }
    }
    lines
}

#[cfg(test)]
mod pair_manifest_tests {
    //! The pair half of the manifest: the replay record, the tuple-committing streamed
    //! hash, and the dump path that closes the loop through the untrusted validator.

    use super::{
        count_lines, dump_pairs, pair_manifest_hash, PairReplayRecord, PAIR_DEVIATION_CLAUSES,
        PAIR_REPLAY_SCHEMA_VERSION,
    };
    use crate::error::ContrastiveDataError;
    use crate::pairs::{
        parse_pair_dump, validate_pair_records, LabeledPair, PairConfig, PairSampler,
        DEFAULT_HARD_CAP, DEGENERATE_POLICY_VERSION,
    };
    use crate::select::{test_corpus, Selection};
    use std::io::{Error, ErrorKind, Write};

    fn selection() -> Selection {
        test_corpus::fresh_selection(12, 13, 8).0
    }

    fn sampler_with<'a>(sel: &'a Selection, cfg: &PairConfig) -> PairSampler<'a> {
        PairSampler::new(sel, cfg).expect("the synthetic corpus supports this configuration")
    }

    #[test]
    fn pair_replay_record_carries_every_version_tag_and_the_three_deviation_clauses() {
        let sel = selection();
        let sampler = sampler_with(&sel, &PairConfig::new(13));
        let record = PairReplayRecord::from_sampler(&sampler);

        assert_eq!(record.schema_version, PAIR_REPLAY_SCHEMA_VERSION);
        assert_eq!(record.strategy, "oversampling");
        assert_eq!(record.strategy_version, 1);
        assert_eq!(record.singleton_policy, "negatives_only");
        assert_eq!(record.singleton_policy_version, 1);
        assert_eq!(record.degenerate_policy_version, DEGENERATE_POLICY_VERSION);
        assert_eq!(record.budget, 384);
        assert!(!record.default_was_clamped);
        assert_eq!(record.emitted_kinds, "both");
        assert_eq!(record.affected_singleton_classes, 0);
        assert_eq!(record.root_seed, 13);
        assert_eq!(record.deviation.len(), 3);
        for (slot, clause) in PAIR_DEVIATION_CLAUSES.iter().enumerate() {
            assert_eq!(&record.deviation[slot], clause);
        }
        // The deviation is APRENDER's, and the clause that could be misread as SetFit's
        // behaviour says the opposite of what the pinned implementation does.
        assert!(record.deviation[2].contains("SELF-PAIRS ARE EXCLUDED"));
        assert!(record.deviation[2].contains("setfit 1.1.3 implementation INCLUDES"));

        // ~200 bytes is the D-09 claim; the clauses dominate, so state the real number.
        let bytes = record.to_canonical_bytes().expect("serializes");
        assert!(bytes.len() < 1_200, "record is {} bytes", bytes.len());
    }

    #[test]
    fn pair_replay_record_round_trips_and_regenerates_the_identical_hash() {
        let sel = selection();
        let sampler = sampler_with(&sel, &PairConfig::new(29));
        let record = PairReplayRecord::from_sampler(&sampler);
        let first = pair_manifest_hash(&sampler, &record).expect("hashes");
        // Vacuity guard: two all-zero digests are equal, which is exactly how the final
        // assertion would pass against a hash that computes nothing.
        assert_ne!(first, [0_u8; 32]);

        let wire = record.to_canonical_bytes().expect("serializes");
        let restored: PairReplayRecord = serde_json::from_slice(&wire).expect("round-trips");
        assert_eq!(restored, record);

        let rebuilt = sampler_with(&sel, &restored.to_config().expect("a supported record"));
        assert_eq!(rebuilt.budget(), sampler.budget());
        let regenerated: Vec<LabeledPair> = rebuilt.iter_from(0).expect("from zero").collect();
        let original: Vec<LabeledPair> = sampler.iter_from(0).expect("from zero").collect();
        assert_eq!(regenerated.len(), 384);
        assert_eq!(regenerated, original, "replay is pairwise identical");
        assert_eq!(
            pair_manifest_hash(&rebuilt, &restored).expect("hashes"),
            first
        );
    }

    #[test]
    fn pair_replay_record_refuses_unsupported_versions() {
        let sel = selection();
        let sampler = sampler_with(&sel, &PairConfig::new(29));
        let base = PairReplayRecord::from_sampler(&sampler);

        let mut wrong_schema = base.clone();
        wrong_schema.schema_version = 99;
        assert!(matches!(
            wrong_schema.to_config(),
            Err(ContrastiveDataError::UnsupportedSchemaVersion { .. })
        ));

        let mut wrong_strategy = base.clone();
        wrong_strategy.strategy = "undersampling".to_string();
        assert!(matches!(
            wrong_strategy.to_config(),
            Err(ContrastiveDataError::UnsupportedAlgorithmVersion { .. })
        ));

        let mut wrong_policy = base.clone();
        wrong_policy.singleton_policy_version = 2;
        assert!(matches!(
            wrong_policy.to_config(),
            Err(ContrastiveDataError::UnsupportedPolicyVersion { .. })
        ));

        let mut wrong_degenerate = base;
        wrong_degenerate.degenerate_policy_version = 7;
        assert!(matches!(
            wrong_degenerate.to_config(),
            Err(ContrastiveDataError::UnsupportedPolicyVersion { .. })
        ));
    }

    /// Review finding F12, the whole point of hashing the header first.
    ///
    /// Configuration A takes the DEFAULT budget under a hard cap of 200, so it is clamped.
    /// Configuration B asks for 200 explicitly under the default cap, so it is not. Both
    /// resolve to a budget of 200 over the SAME selection and seed, so their pair streams
    /// are byte-identical — the test asserts that first, or the hash comparison below would
    /// prove nothing. Their manifest hashes must nonetheless DIFFER.
    #[test]
    fn identical_pair_streams_with_different_replay_tuples_hash_differently() {
        let sel = selection();
        let clamped = sampler_with(
            &sel,
            &PairConfig {
                hard_cap: Some(200),
                ..PairConfig::new(31)
            },
        );
        let explicit = sampler_with(
            &sel,
            &PairConfig {
                budget: Some(200),
                ..PairConfig::new(31)
            },
        );

        assert_eq!(clamped.budget(), 200);
        assert_eq!(explicit.budget(), 200);
        let left: Vec<LabeledPair> = clamped.iter_from(0).expect("from zero").collect();
        let right: Vec<LabeledPair> = explicit.iter_from(0).expect("from zero").collect();
        assert_eq!(
            left, right,
            "the two streams must be IDENTICAL pair for pair"
        );
        assert_eq!(left.len(), 200);

        let clamped_record = PairReplayRecord::from_sampler(&clamped);
        let explicit_record = PairReplayRecord::from_sampler(&explicit);
        assert!(clamped_record.default_was_clamped);
        assert!(!explicit_record.default_was_clamped);
        assert_ne!(
            pair_manifest_hash(&clamped, &clamped_record).expect("hashes"),
            pair_manifest_hash(&explicit, &explicit_record).expect("hashes"),
            "a pairs-only hash would collide here"
        );
    }

    /// The sharper half of the same finding: two DIFFERENT selections that share a class
    /// layout produce literally the same pair bytes, because pairs are encoded by ORDINAL.
    /// Only `selection_hash` separates them, and it lives in the header.
    #[test]
    fn identical_pair_bytes_over_different_selections_hash_differently() {
        let small = test_corpus::fresh_selection(12, 41, 8).0;
        let large = test_corpus::fresh_selection(20, 41, 8).0;
        assert_ne!(small.semantic_hash(), large.semantic_hash());
        assert_eq!(small.class_sizes(), large.class_sizes());

        let cfg = PairConfig::new(41);
        let a = sampler_with(&small, &cfg);
        let b = sampler_with(&large, &cfg);

        let ordinals = |sampler: &PairSampler<'_>| -> Vec<(u32, u32, f32)> {
            sampler
                .iter_from(0)
                .expect("from zero")
                .map(|p| (p.pair.lo().ordinal(), p.pair.hi().ordinal(), p.target))
                .collect()
        };
        assert_eq!(
            ordinals(&a),
            ordinals(&b),
            "same layout and seed must give the same ORDINAL stream"
        );

        let record_a = PairReplayRecord::from_sampler(&a);
        let record_b = PairReplayRecord::from_sampler(&b);
        assert_ne!(record_a.selection_hash, record_b.selection_hash);
        assert_ne!(
            pair_manifest_hash(&a, &record_a).expect("hashes"),
            pair_manifest_hash(&b, &record_b).expect("hashes")
        );
    }

    #[test]
    fn pair_manifest_hash_refuses_a_record_that_describes_another_stream() {
        let sel = selection();
        let sampler = sampler_with(&sel, &PairConfig::new(43));
        let mut record = PairReplayRecord::from_sampler(&sampler);
        record.budget = 7;
        assert!(matches!(
            pair_manifest_hash(&sampler, &record),
            Err(ContrastiveDataError::SelectionReplayMismatch { .. })
        ));

        let mut foreign = PairReplayRecord::from_sampler(&sampler);
        foreign.selection_hash = "0".repeat(64);
        assert!(matches!(
            pair_manifest_hash(&sampler, &foreign),
            Err(ContrastiveDataError::SelectionReplayMismatch { .. })
        ));
    }

    #[test]
    fn dump_pairs_is_deterministic_and_round_trips_through_the_untrusted_validator() {
        let sel = selection();
        let cfg = PairConfig {
            budget: Some(64),
            ..PairConfig::new(47)
        };
        let sampler = sampler_with(&sel, &cfg);

        let mut first = Vec::new();
        dump_pairs(&sampler, &mut first).expect("dumping to a Vec cannot fail");
        let mut second = Vec::new();
        dump_pairs(&sampler, &mut second).expect("dumping to a Vec cannot fail");
        assert_eq!(first, second, "two dumps are byte-identical");
        assert_eq!(count_lines(&first), 64);

        let parsed = parse_pair_dump(&first).expect("the dump parses");
        assert_eq!(parsed.len(), 64);
        let validated = validate_pair_records(&parsed, &sel).expect("the dump validates");
        let expected: Vec<LabeledPair> = sampler.iter_from(0).expect("from zero").collect();
        assert_eq!(validated, expected, "the dump/ingest loop closes");
    }

    /// A sink that fails mid-stream is [`ContrastiveDataError::Io`], never a panic.
    #[test]
    fn dump_pairs_surfaces_a_failing_sink_as_a_typed_io_error() {
        struct FailAfter(usize);
        impl Write for FailAfter {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                if self.0 == 0 {
                    return Err(Error::new(ErrorKind::BrokenPipe, "sink closed mid-stream"));
                }
                self.0 -= 1;
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let sel = selection();
        let cfg = PairConfig {
            budget: Some(64),
            ..PairConfig::new(47)
        };
        let sampler = sampler_with(&sel, &cfg);
        match dump_pairs(&sampler, FailAfter(3)) {
            Err(ContrastiveDataError::Io { context, detail }) => {
                assert!(context.contains("pair"), "context {context:?}");
                assert!(
                    detail.contains("sink closed mid-stream"),
                    "detail {detail:?}"
                );
            }
            Ok(()) => panic!("a failing sink must not be swallowed"),
            Err(other) => panic!("expected Io, got {other:?}"),
        }
    }

    /// The hash STREAMS: a 24,576-pair budget is hashed without collecting anything.
    ///
    /// Measured structurally rather than asserted — the sampler's own retained-state report
    /// is unchanged across the hash, and it counts materialized pairs.
    #[test]
    fn pair_manifest_hash_streams_a_large_budget_without_materializing() {
        use crate::pairs::RetainedState;

        let (sel, _) = test_corpus::fresh_selection(70, 53, 64);
        let sampler = sampler_with(&sel, &PairConfig::new(53));
        assert_eq!(sampler.budget(), 24_576);
        assert_eq!(DEFAULT_HARD_CAP, 1_048_576);

        let before = sampler.state_report();
        let record = PairReplayRecord::from_sampler(&sampler);
        let digest = pair_manifest_hash(&sampler, &record).expect("hashes");
        let after = sampler.state_report();

        assert_eq!(before, after);
        assert_eq!(after.materialized_pairs, 0);
        assert_ne!(digest, [0_u8; 32]);
    }

    /// The twelve-byte pair encoding, field by field (plan 02-08 mutation triage).
    ///
    /// `cargo mutants` found `pair_canonical_bytes -> [0; 12]` and `-> [1; 12]` both
    /// surviving. Under either, the digest depends on the record and the pair COUNT and on
    /// NOTHING ABOUT THE PAIRS — the streamed-stream half of the attestation would attest
    /// nothing at all, and every existing hash test still passed because each of them
    /// varies the HEADER. The gap is precisely that no test varied the stream while holding
    /// the header fixed.
    #[test]
    fn the_pair_wire_encoding_is_lo_then_hi_then_target_bits_little_endian() {
        let sel = selection();
        let sampler = sampler_with(&sel, &PairConfig::new(13));
        let pair = sampler.pair_at(0).expect("ordinal 0 is inside any budget");

        let bytes = super::pair_canonical_bytes(&pair);
        assert_eq!(&bytes[0..4], &pair.pair.lo().ordinal().to_le_bytes());
        assert_eq!(&bytes[4..8], &pair.pair.hi().ordinal().to_le_bytes());
        assert_eq!(&bytes[8..12], &pair.target.to_bits().to_le_bytes());

        // Field ORDER is the part a re-derivation cannot check, so it is pinned against a
        // second pair whose ordinals differ: lo must move the first four bytes and hi the
        // second four, not the reverse.
        let other = sampler.pair_at(1).expect("ordinal 1 is inside this budget");
        assert_ne!(pair.pair, other.pair, "the two ordinals must differ");
        let other_bytes = super::pair_canonical_bytes(&other);
        assert_ne!(
            bytes, other_bytes,
            "two different pairs must not encode to the same twelve bytes"
        );
    }

    /// The same HEADER over two DIFFERENT streams must not collide (02-08 triage).
    ///
    /// The behavioural half of the test above. `assert_record_describes` compares only
    /// `selection_hash` and `budget`, so one record legitimately describes both samplers
    /// here — same selection, same budget, different pair seed. If the pair bytes did not
    /// reach the hasher, these two digests would be equal.
    #[test]
    fn two_streams_under_one_header_produce_different_manifest_hashes() {
        let sel = selection();
        let budget = 64;
        let cfg_of = |seed: u64| PairConfig {
            budget: Some(budget),
            ..PairConfig::new(seed)
        };
        let first = sampler_with(&sel, &cfg_of(13));
        let second = sampler_with(&sel, &cfg_of(17));

        // Vacuity guard: the streams must genuinely differ, or the digests would be equal
        // for an entirely honest reason.
        let stream = |s: &PairSampler<'_>| -> Vec<(u32, u32, u32)> {
            (0..budget)
                .map(|ordinal| {
                    let p = s.pair_at(ordinal).expect("inside the budget");
                    (
                        p.pair.lo().ordinal(),
                        p.pair.hi().ordinal(),
                        p.target.to_bits(),
                    )
                })
                .collect()
        };
        assert_ne!(stream(&first), stream(&second));

        let header = PairReplayRecord::from_sampler(&first);
        let a = pair_manifest_hash(&first, &header).expect("hashes");
        let b = pair_manifest_hash(&second, &header)
            .expect("the header describes both samplers: same selection, same budget");
        assert_ne!(
            a, b,
            "the streamed pairs must reach the digest, not just the header"
        );
    }
}

#[cfg(test)]
mod manifest_tests {
    use super::{SelectionManifest, VolatileMetadata};
    use crate::error::ContrastiveDataError;
    use crate::hash::hex;
    use crate::ledger::AccessLedger;
    use crate::select::test_corpus;

    fn wrapped() -> (SelectionManifest, AccessLedger) {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 23, 8, &mut ledger);
        let manifest =
            SelectionManifest::from_selection(&selection, &ledger).expect("the wrap succeeds");
        (manifest, ledger)
    }

    /// `created_at` lives OUTSIDE the hashed region, so two manifests of the same
    /// selection that differ only in when they were written are the same artifact.
    #[test]
    fn manifest_semantic_hash_ignores_volatile_metadata() {
        let (mut first, _) = wrapped();
        let (mut second, _) = wrapped();
        first.volatile = VolatileMetadata {
            created_at: "2026-08-09T00:00:00Z".to_string(),
            tool_version: "0.0.1-alpha".to_string(),
        };
        second.volatile = VolatileMetadata {
            created_at: "2031-01-01T12:34:56Z".to_string(),
            tool_version: "9.9.9".to_string(),
        };

        assert_ne!(first.volatile, second.volatile);
        assert_eq!(first.semantic_hash, second.semantic_hash);
        assert_eq!(first.payload, second.payload);
        first.verify_digest().expect("digest still verifies");
        second.verify_digest().expect("digest still verifies");
    }

    /// Checker warning 2: the payload attests the ledger as it stood when `select`
    /// finished, so a ledger that has grown since is not the one it describes.
    #[test]
    fn manifest_from_selection_refuses_a_ledger_that_has_drifted() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(12, &mut ledger);
        let selection = test_corpus::select(&dataset, 29, 8, &mut ledger);
        SelectionManifest::from_selection(&selection, &ledger).expect("the matching ledger wraps");

        ledger.record("train", "canonical", "something-else", "aa");
        let err = SelectionManifest::from_selection(&selection, &ledger)
            .expect_err("a drifted ledger must be refused");
        match err {
            ContrastiveDataError::SemanticHashMismatch { expected, got } => {
                assert_eq!(expected, hex(&selection.ledger_hash()));
                assert_eq!(got, hex(&ledger.ledger_hash()));
                assert_ne!(expected, got);
            }
            other => panic!("expected SemanticHashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_file_form_round_trips_every_payload_field() {
        let (manifest, _) = wrapped();
        let bytes = manifest.to_file_bytes().expect("file bytes");
        assert_eq!(
            bytes.last(),
            Some(&b'\n'),
            "the file form ends in a newline"
        );

        let restored = SelectionManifest::from_bytes(&bytes).expect("round-trip");
        assert_eq!(restored, manifest);
        assert_eq!(restored.payload, manifest.payload);

        // Two file forms differing ONLY in volatile metadata parse to equal payloads and
        // equal digests.
        let mut other = manifest.clone();
        other.volatile.created_at = "1999-12-31T23:59:59Z".to_string();
        let other_bytes = other.to_file_bytes().expect("file bytes");
        assert_ne!(other_bytes, bytes);
        let other_restored = SelectionManifest::from_bytes(&other_bytes).expect("round-trip");
        assert_eq!(other_restored.payload, restored.payload);
        assert_eq!(other_restored.semantic_hash, restored.semantic_hash);
    }

    #[test]
    fn manifest_from_bytes_rejects_a_digest_that_disagrees_with_its_payload() {
        let (mut manifest, _) = wrapped();
        let honest = manifest.semantic_hash.clone();
        manifest.semantic_hash = "0".repeat(64);
        let bytes = manifest.to_file_bytes().expect("file bytes");

        let err = SelectionManifest::from_bytes(&bytes)
            .expect_err("a disagreeing digest must be refused before the value is returned");
        match err {
            ContrastiveDataError::SemanticHashMismatch { expected, got } => {
                assert_eq!(expected, "0".repeat(64));
                assert_eq!(got, honest);
            }
            other => panic!("expected SemanticHashMismatch, got {other:?}"),
        }
    }

    #[test]
    fn manifest_from_bytes_rejects_an_unknown_envelope_field() {
        let (manifest, _) = wrapped();
        let mut text =
            String::from_utf8(manifest.to_file_bytes().expect("file bytes")).expect("UTF-8");
        text.insert_str(1, r#""rogue":1,"#);
        let err = SelectionManifest::from_bytes(text.as_bytes())
            .expect_err("deny_unknown_fields must reject an added envelope key");
        assert!(matches!(err, ContrastiveDataError::Serialization { .. }));
    }

    /// Review finding F7: the persisted ledger is a real ledger, not a decorative copy.
    #[test]
    fn manifest_persisted_ledger_reproduces_its_hash_through_access_ledger() {
        let (manifest, live) = wrapped();
        let records = serde_json::to_string(&manifest.payload.access_ledger)
            .expect("the persisted records serialize");
        let wire = format!(r#"{{"schema_version":1,"records":{records}}}"#);
        let rebuilt = AccessLedger::from_bytes(wire.as_bytes())
            .expect("the persisted records parse as a ledger");

        assert_eq!(rebuilt.records(), live.records());
        assert_eq!(hex(&rebuilt.ledger_hash()), manifest.payload.ledger_hash);
    }
}
