//! Exact and normalized content hashes plus the dataset fingerprint.
//!
//! Two hashes per row, for two different jobs (D-17): the exact SHA-256 over the raw
//! `input` bytes is identity and provenance; the normalized hash (`nfc-trim-ws-v1` — NFC,
//! trimmed, internal whitespace collapsed, deliberately NO casefolding) is leakage
//! detection.
//!
//! # This module is a LEAF
//!
//! It depends on nothing else in this crate except the error type. In particular it does
//! not know that typed split roles exist: both fingerprints take **raw parts**
//! ([`SplitFingerprintInput`], [`DatasetFingerprintInput`]) rather than a typed split, so
//! the hashing story is complete and testable on its own. The one place the typestate and
//! the hashes meet is the prepared-dataset constructor, which assembles these inputs from
//! its own splits.
//!
//! # One construction, two domain tags
//!
//! A dataset fingerprint and a single-split fingerprint absorb split parts through the
//! *same* private helper. They differ only in their domain-tag prefix, which is what makes
//! them differ even for a one-split dataset — so a split fingerprint can never be mistaken
//! for a dataset fingerprint by a consumer comparing hex strings.

use sha2::{Digest, Sha256};

/// The version tag of the normalization pipeline behind [`normalized_hash`].
///
/// Recorded in every manifest that depends on it. Changing any step of the pipeline is a
/// NEW TAG and a contract change, never an in-place edit: the exclusion record of every
/// previously produced manifest was computed under the old one.
pub const CONTENT_NORMALIZATION_VERSION: &str = "nfc-trim-ws-v1";

/// Domain tag for a whole-dataset fingerprint.
const DATASET_FP_DOMAIN: &[u8] = b"apr-dataset-fp-v1\0";

/// Domain tag for a single-split fingerprint.
const SPLIT_FP_DOMAIN: &[u8] = b"apr-split-fp-v1\0";

/// SHA-256 over the raw bytes of `input`. Identity and provenance.
///
/// This is the digest that appears in fingerprints and attestations, so it must describe
/// the bytes as stored — no trimming, no normalization, no case folding.
pub fn exact_hash(input: &str) -> [u8; 32] {
    Sha256::digest(input.as_bytes()).into()
}

/// SHA-256 over the `nfc-trim-ws-v1` normalization of `input`. Leakage detection.
///
/// The pipeline is exactly: NFC, then trim, then collapse every internal whitespace run to
/// a single `U+0020`. `split_whitespace` performs the last two steps in one pass and uses
/// the Unicode `White_Space` property, so a non-breaking space collapses like a plain one.
///
/// # There is deliberately NO casefolding (D-17)
///
/// Casefolding would collide legitimately distinct short posts — the corpus this protocol
/// was designed against is social-media length, where `"Yes"` and `"yes"` are routinely
/// different rows by different authors. A false leakage positive silently REMOVES a
/// training row and shrinks a class pool, which is a worse outcome than the retweet
/// variant this normalization is here to catch.
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "normalized_content_hash"
)]
pub fn normalized_hash(input: &str) -> [u8; 32] {
    use unicode_normalization::UnicodeNormalization;

    let composed: String = input.nfc().collect();
    let collapsed = composed.split_whitespace().collect::<Vec<_>>().join(" ");
    Sha256::digest(collapsed.as_bytes()).into()
}

/// Lowercase hex rendering of a digest.
pub fn hex(digest: &[u8; 32]) -> String {
    use core::fmt::Write as _;

    digest
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            // Writing into a String is infallible; the Result exists only to satisfy the
            // `Write` trait, and discarding it here keeps the signature total.
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// Absorb one length-prefixed field.
///
/// Every variable-length field is prefixed with its length so that two different
/// decompositions of the same concatenated bytes cannot produce the same digest. Without
/// it, `role="tr"` + `id="ain:0"` and `role="train"` + `id=":0"` would hash identically.
fn absorb_field(hasher: &mut Sha256, field: &[u8]) {
    hasher.update((field.len() as u64).to_le_bytes());
    hasher.update(field);
}

/// The ONE per-split absorption routine, shared by both fingerprint entry points.
///
/// "The same construction with a different domain tag" is a fact about this function
/// rather than a claim in a comment: `SplitFingerprint::compute` and
/// `DatasetFingerprint::compute` both call it, and neither has a private copy that could
/// drift.
fn absorb_split(hasher: &mut Sha256, input: &SplitFingerprintInput<'_>) {
    debug_assert!(
        input.rows.windows(2).all(|pair| pair[0].0 <= pair[1].0),
        "SplitFingerprintInput::rows must be sorted ascending by id before hashing"
    );

    absorb_field(hasher, input.role.as_bytes());
    absorb_field(hasher, input.source_hash);
    hasher.update((input.class_counts.len() as u64).to_le_bytes());
    for count in input.class_counts {
        hasher.update(count.to_le_bytes());
    }
    hasher.update((input.rows.len() as u64).to_le_bytes());
    for (id, row_hash) in input.rows {
        absorb_field(hasher, id.as_bytes());
        absorb_field(hasher, row_hash);
    }
}

/// Raw parts describing ONE split, in absorption order.
pub(crate) struct SplitFingerprintInput<'a> {
    /// The split's role name.
    pub role: &'a str,
    /// SHA-256 of the split's canonical JSONL bytes.
    pub source_hash: &'a [u8; 32],
    /// Per-class row counts, indexed by class label.
    pub class_counts: &'a [u64],
    /// `(id, exact_hash)` pairs, ALREADY sorted ascending by id.
    pub rows: &'a [(&'a str, [u8; 32])],
}

/// Raw parts describing a WHOLE dataset.
pub(crate) struct DatasetFingerprintInput<'a> {
    /// The dataset profile string.
    pub profile: &'a str,
    /// Ordered label names.
    pub label_names: &'a [String],
    /// The content-normalization version.
    pub normalization_version: &'a str,
    /// Per-split parts, ALREADY ordered by ascending role name.
    pub splits: &'a [SplitFingerprintInput<'a>],
}

/// Identity of a WHOLE dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetFingerprint([u8; 32]);

impl DatasetFingerprint {
    /// Lowercase hex rendering.
    pub fn hex(&self) -> String {
        hex(&self.0)
    }

    /// Absorb the profile, the ordered label names, the normalization version, and then
    /// every split's raw parts in ascending role order through [`absorb_split`].
    pub(crate) fn compute(input: &DatasetFingerprintInput<'_>) -> Self {
        debug_assert!(
            input
                .splits
                .windows(2)
                .all(|pair| pair[0].role <= pair[1].role),
            "DatasetFingerprintInput::splits must be ordered by ascending role name"
        );

        let mut hasher = Sha256::new();
        hasher.update(DATASET_FP_DOMAIN);
        absorb_field(&mut hasher, input.profile.as_bytes());
        hasher.update((input.label_names.len() as u64).to_le_bytes());
        for name in input.label_names {
            absorb_field(&mut hasher, name.as_bytes());
        }
        absorb_field(&mut hasher, input.normalization_version.as_bytes());
        hasher.update((input.splits.len() as u64).to_le_bytes());
        for split in input.splits {
            absorb_split(&mut hasher, split);
        }
        Self(hasher.finalize().into())
    }
}

/// Identity of ONE split alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitFingerprint([u8; 32]);

impl SplitFingerprint {
    /// Lowercase hex rendering.
    pub fn hex(&self) -> String {
        hex(&self.0)
    }

    /// Absorb the SAME raw parts a dataset fingerprint absorbs for this split, under a
    /// different domain tag.
    pub(crate) fn compute(input: &SplitFingerprintInput<'_>) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(SPLIT_FP_DOMAIN);
        absorb_split(&mut hasher, input);
        Self(hasher.finalize().into())
    }
}

#[cfg(test)]
mod hash_tests {
    use super::{
        exact_hash, hex, normalized_hash, DatasetFingerprint, DatasetFingerprintInput,
        SplitFingerprint, SplitFingerprintInput, CONTENT_NORMALIZATION_VERSION,
    };
    use proptest::prelude::{prop_assert_eq, proptest, Strategy};

    fn label_names() -> Vec<String> {
        vec![
            "none".to_string(),
            "against".to_string(),
            "favor".to_string(),
        ]
    }

    fn sample_rows() -> Vec<(&'static str, [u8; 32])> {
        let mut rows = vec![
            ("train:0", exact_hash("alpha")),
            ("train:1", exact_hash("beta")),
        ];
        rows.sort_by(|left, right| left.0.cmp(right.0));
        rows
    }

    #[test]
    fn hash_content_normalization_version_is_pinned() {
        assert_eq!(CONTENT_NORMALIZATION_VERSION, "nfc-trim-ws-v1");
    }

    #[test]
    fn hash_hex_is_lowercase_and_64_characters() {
        let rendered = hex(&exact_hash("anything"));
        assert_eq!(rendered.len(), 64);
        assert!(rendered.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(rendered, rendered.to_lowercase());
    }

    #[test]
    fn hash_exact_is_byte_sensitive_while_normalized_trims() {
        assert_ne!(exact_hash("text "), exact_hash("text"));
        assert_eq!(normalized_hash("text "), normalized_hash("text"));
        assert_eq!(normalized_hash("  text\n"), normalized_hash("text"));
    }

    #[test]
    fn hash_normalized_collapses_unicode_whitespace_runs() {
        assert_eq!(normalized_hash("a  b"), normalized_hash("a b"));
        assert_eq!(normalized_hash("a\u{00A0}b"), normalized_hash("a b"));
        assert_eq!(normalized_hash("a\t\nb"), normalized_hash("a b"));
    }

    #[test]
    fn hash_normalized_does_not_casefold() {
        assert_ne!(normalized_hash("Text"), normalized_hash("text"));
    }

    #[test]
    fn hash_normalized_applies_nfc() {
        assert_eq!(normalized_hash("e\u{0301}"), normalized_hash("\u{00E9}"));
        assert_ne!(exact_hash("e\u{0301}"), exact_hash("\u{00E9}"));
    }

    /// Half the generated pairs are deliberately EQUAL, because two independently drawn
    /// strings essentially never collide under SHA-256 and the implication would then be
    /// vacuously true for every case the property ever saw.
    fn pair_strategy() -> impl Strategy<Value = (String, String)> {
        (".{0,24}", ".{0,24}", proptest::bool::ANY).prop_map(|(left, right, identical)| {
            if identical {
                (left.clone(), left)
            } else {
                (left, right)
            }
        })
    }

    proptest! {
        #[test]
        fn hash_exact_collision_implies_normalized_collision((left, right) in pair_strategy()) {
            if exact_hash(&left) == exact_hash(&right) {
                prop_assert_eq!(normalized_hash(&left), normalized_hash(&right));
            }
        }
    }

    struct Parts {
        role: String,
        source_hash: [u8; 32],
        class_counts: Vec<u64>,
        rows: Vec<(&'static str, [u8; 32])>,
        profile: String,
        label_names: Vec<String>,
        normalization_version: String,
    }

    impl Parts {
        fn base() -> Self {
            Self {
                role: "train".to_string(),
                source_hash: exact_hash("train-bytes"),
                class_counts: vec![3, 4, 5],
                rows: sample_rows(),
                profile: "canonical".to_string(),
                label_names: label_names(),
                normalization_version: CONTENT_NORMALIZATION_VERSION.to_string(),
            }
        }

        fn split_input(&self) -> SplitFingerprintInput<'_> {
            SplitFingerprintInput {
                role: &self.role,
                source_hash: &self.source_hash,
                class_counts: &self.class_counts,
                rows: &self.rows,
            }
        }

        fn dataset_fingerprint(&self) -> DatasetFingerprint {
            let splits = [self.split_input()];
            DatasetFingerprint::compute(&DatasetFingerprintInput {
                profile: &self.profile,
                label_names: &self.label_names,
                normalization_version: &self.normalization_version,
                splits: &splits,
            })
        }
    }

    fn assert_fingerprint_changes(mutate: impl FnOnce(&mut Parts), field: &str) {
        let base = Parts::base();
        let baseline = base.dataset_fingerprint();
        let mut mutated = Parts::base();
        mutate(&mut mutated);
        assert_ne!(
            baseline.hex(),
            mutated.dataset_fingerprint().hex(),
            "dataset fingerprint must change when {field} changes"
        );
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_a_row_id() {
        // Mutating the LAST id keeps the ascending-by-id ordering that `compute`
        // debug-asserts, so this test exercises identity sensitivity rather than the
        // caller's ordering obligation.
        assert_fingerprint_changes(|parts| parts.rows[1].0 = "train:9", "a row id");
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_a_role() {
        assert_fingerprint_changes(|parts| parts.role = "test".to_string(), "a split role");
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_the_profile() {
        assert_fingerprint_changes(
            |parts| parts.profile = "compatibility".to_string(),
            "the profile",
        );
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_a_label_name() {
        assert_fingerprint_changes(
            |parts| parts.label_names[1] = "opposed".to_string(),
            "a label name",
        );
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_the_normalization_version() {
        assert_fingerprint_changes(
            |parts| parts.normalization_version = "nfc-trim-ws-v2".to_string(),
            "the normalization version",
        );
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_a_row_exact_hash() {
        assert_fingerprint_changes(
            |parts| parts.rows[0].1 = exact_hash("mutated"),
            "a row exact hash",
        );
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_a_source_hash() {
        assert_fingerprint_changes(
            |parts| parts.source_hash = exact_hash("other-bytes"),
            "a split source hash",
        );
    }

    #[test]
    fn hash_dataset_fingerprint_is_sensitive_to_a_class_count() {
        assert_fingerprint_changes(|parts| parts.class_counts[2] = 6, "a per-class count");
    }

    #[test]
    fn hash_split_and_dataset_fingerprints_differ_for_a_one_split_dataset() {
        let parts = Parts::base();
        let split = SplitFingerprint::compute(&parts.split_input());
        let dataset = parts.dataset_fingerprint();
        assert_ne!(
            split.hex(),
            dataset.hex(),
            "distinct domain tags must keep a split fingerprint distinguishable from a dataset fingerprint"
        );
    }

    #[test]
    fn hash_split_fingerprint_is_stable_across_two_computations() {
        let parts = Parts::base();
        assert_eq!(
            SplitFingerprint::compute(&parts.split_input()).hex(),
            SplitFingerprint::compute(&parts.split_input()).hex()
        );
    }
}
