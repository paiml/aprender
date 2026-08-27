//! Cross-split duplicate coalescing and the deterministic exclusion record.
//!
//! Duplicate groups are connected components over the union of exact-hash and
//! normalized-hash edges. Grouping independently by both keys would double-count — an
//! exact duplicate is necessarily also a normalized duplicate — and could decrement a
//! class pool twice.
//!
//! Prepare-time duplicate content is excluded and recorded, never fatal (D-18, upheld by
//! D-27); the typed error fires only when the reduced pool can no longer supply
//! `shots_per_class`.
//!
//! # Why coalescing is not an optimization
//!
//! `hash.rs` proves, as a property test, that an exact-hash collision implies a
//! normalized-hash collision. So the two edge kinds are not independent: every exact
//! duplicate appears in BOTH groupings. Emitting one group per key would remove the same
//! training row twice from the same class pool, understating the pool. A pool understated
//! near the boundary produces a `CrossSplitDuplicateUnderflow` at selection time — a
//! failure invented by the detector rather than present in the data.
//!
//! # Why nothing here returns `Err`
//!
//! [`coalesced_exclusions`] returns an [`ExclusionRecord`], not a `Result`, and that is a
//! decision rather than an omission (D-18, upheld verbatim by D-27). Hard-failing at
//! SELECTION time would make failures seed-dependent, so a subset of benchmark cells would
//! die and a completeness gate would reject the run for a reason unrelated to the method.
//! Hard-failing at PREPARE time would hand upstream data quality a veto over the whole
//! dataset. The only real failure is a reduced pool that can no longer supply the
//! requested shots, and that is raised where the shots are known: at selection.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::ContrastiveDataError;
use crate::hash::{exact_hash, normalized_hash, CONTENT_NORMALIZATION_VERSION};
use crate::schema::LabeledExample;
use crate::split::{SplitRole, Train};

/// Which detection kinds fired inside one duplicate component.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetectionKinds {
    /// At least one pair of members shares an exact content hash.
    pub exact: bool,
    /// At least one pair of members shares a normalized content hash.
    pub normalized: bool,
}

/// One connected component of duplicate content spanning at least two split roles.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuplicateGroup {
    /// `(split_role, id)` members, sorted ascending.
    pub members: Vec<(String, String)>,
    /// Which detection kinds fired inside this component.
    pub detected_by: DetectionKinds,
    /// True when the members do not all carry the same label — impossible to reconcile
    /// automatically, and therefore worth surfacing rather than silently excluding.
    pub label_conflict: bool,
}

/// The deterministic record of everything cross-split duplication removed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExclusionRecord {
    excluded_train_ids: Vec<String>,
    groups: Vec<DuplicateGroup>,
    reduced_pools: BTreeMap<usize, u64>,
    normalization_version: String,
}

impl ExclusionRecord {
    /// Training ids removed from the selection pool, sorted ascending.
    pub fn excluded_train_ids(&self) -> &[String] {
        &self.excluded_train_ids
    }

    /// Remaining training pool size per class label, after exclusion.
    pub fn reduced_pools(&self) -> &BTreeMap<usize, u64> {
        &self.reduced_pools
    }

    /// The duplicate components, sorted deterministically.
    pub fn groups(&self) -> &[DuplicateGroup] {
        &self.groups
    }

    /// Deterministic canonical serialization.
    ///
    /// # Errors
    ///
    /// [`ContrastiveDataError::Serialization`] if the record cannot be serialized.
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>, ContrastiveDataError> {
        serde_json::to_vec(self).map_err(|error| ContrastiveDataError::Serialization {
            context: "exclusion_record".to_string(),
            detail: error.to_string(),
        })
    }

    /// SHA-256 of [`Self::to_canonical_bytes`].
    ///
    /// Total for the same reason the ledger's hash is: the canonical form is strings,
    /// integers and booleans, with every map a `BTreeMap<usize, u64>` whose keys
    /// serialize as strings. `serde_json` has no failure mode to report here.
    pub fn hash(&self) -> [u8; 32] {
        let bytes = self
            .to_canonical_bytes()
            .expect("ExclusionRecord canonical form is strings, integers and bools");
        Sha256::digest(bytes).into()
    }
}

/// A minimal disjoint-set forest over row ordinals.
///
/// Union by size with path halving. The structure is a `Vec`, not a map, so nothing here
/// depends on hash iteration order (PF-006).
struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            size: vec![1; len],
        }
    }

    fn find(&mut self, mut node: usize) -> usize {
        while self.parent[node] != node {
            let grandparent = self.parent[self.parent[node]];
            self.parent[node] = grandparent;
            node = grandparent;
        }
        node
    }

    fn union(&mut self, left: usize, right: usize) {
        let (mut a, mut b) = (self.find(left), self.find(right));
        if a == b {
            return;
        }
        if self.size[a] < self.size[b] {
            core::mem::swap(&mut a, &mut b);
        }
        self.parent[b] = a;
        self.size[a] += self.size[b];
    }
}

/// One row, flattened across splits, with both of its content hashes.
struct FlatRow<'a> {
    role: &'a str,
    id: &'a str,
    label: usize,
    exact: [u8; 32],
    normalized: [u8; 32],
}

/// Coalesce cross-split duplicate content into connected components.
///
/// Deterministic and total. `splits` is `(role, rows)` for every split of one dataset.
///
/// Edges come from two sources — equal exact hashes and equal normalized hashes — and are
/// merged into ONE disjoint-set forest before any group is emitted. A component that spans
/// at least two distinct split roles is a duplicate group; a component confined to one role
/// is a within-split repetition, which is not evaluation leakage and is left alone.
///
/// Only TRAIN rows are removed, and only from the selection pool: the evaluation splits
/// keep every row they arrived with, because shrinking an evaluation split would change
/// what a reported score means (D-18).
#[provable_contracts_macros::contract(
    "contrastive-pair-protocol-v1",
    equation = "cross_split_exclusion"
)]
pub(crate) fn coalesced_exclusions(
    splits: &[(&'static str, &[LabeledExample])],
) -> ExclusionRecord {
    let flat: Vec<FlatRow<'_>> = splits
        .iter()
        .flat_map(|(role, rows)| {
            rows.iter().map(move |row| FlatRow {
                role,
                id: row.id.as_str(),
                label: row.label,
                exact: exact_hash(&row.input),
                normalized: normalized_hash(&row.input),
            })
        })
        .collect();

    // Both edge kinds go into ONE forest. Bucketing by hash uses BTreeMap so the union
    // order — and therefore nothing observable, but also nothing accidental — is fixed.
    let mut forest = DisjointSet::new(flat.len());
    for key in [
        |row: &FlatRow<'_>| row.exact,
        |row: &FlatRow<'_>| row.normalized,
    ] {
        let mut buckets: BTreeMap<[u8; 32], Vec<usize>> = BTreeMap::new();
        for (index, row) in flat.iter().enumerate() {
            buckets.entry(key(row)).or_default().push(index);
        }
        for members in buckets.values() {
            for pair in members.windows(2) {
                forest.union(pair[0], pair[1]);
            }
        }
    }

    let mut components: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for index in 0..flat.len() {
        let root = forest.find(index);
        components.entry(root).or_default().push(index);
    }

    let mut groups: Vec<DuplicateGroup> = Vec::new();
    let mut excluded_train_ids: BTreeSet<String> = BTreeSet::new();
    for members in components.values() {
        let roles: BTreeSet<&str> = members.iter().map(|index| flat[*index].role).collect();
        if roles.len() < 2 {
            continue;
        }

        let detected_by = DetectionKinds {
            exact: shares_a_key(members, &flat, |row| row.exact),
            normalized: shares_a_key(members, &flat, |row| row.normalized),
        };
        let labels: BTreeSet<usize> = members.iter().map(|index| flat[*index].label).collect();
        let mut member_pairs: Vec<(String, String)> = members
            .iter()
            .map(|index| (flat[*index].role.to_string(), flat[*index].id.to_string()))
            .collect();
        member_pairs.sort();

        for index in members {
            if flat[*index].role == Train::ROLE {
                excluded_train_ids.insert(flat[*index].id.to_string());
            }
        }

        groups.push(DuplicateGroup {
            members: member_pairs,
            detected_by,
            label_conflict: labels.len() > 1,
        });
    }
    groups.sort_by(|left, right| left.members.cmp(&right.members));

    let mut reduced_pools: BTreeMap<usize, u64> = BTreeMap::new();
    for row in flat.iter().filter(|row| row.role == Train::ROLE) {
        let entry = reduced_pools.entry(row.label).or_insert(0);
        if !excluded_train_ids.contains(row.id) {
            *entry += 1;
        }
    }

    ExclusionRecord {
        excluded_train_ids: excluded_train_ids.into_iter().collect(),
        groups,
        reduced_pools,
        normalization_version: CONTENT_NORMALIZATION_VERSION.to_string(),
    }
}

/// True when at least two members of the component share the given hash.
fn shares_a_key(
    members: &[usize],
    flat: &[FlatRow<'_>],
    key: impl Fn(&FlatRow<'_>) -> [u8; 32],
) -> bool {
    let mut seen: BTreeSet<[u8; 32]> = BTreeSet::new();
    members.iter().any(|index| !seen.insert(key(&flat[*index])))
}

#[cfg(test)]
mod dedup_tests {
    use super::coalesced_exclusions;
    use crate::hash::CONTENT_NORMALIZATION_VERSION;
    use crate::schema::LabeledExample;

    fn row(id: &str, input: &str, label: usize, split: &str) -> LabeledExample {
        LabeledExample {
            id: id.to_string(),
            input: input.to_string(),
            label,
            label_text: ["none", "against", "favor"][label].to_string(),
            source_split: split.to_string(),
        }
    }

    fn train_base() -> Vec<LabeledExample> {
        vec![
            row("train:0", "alpha post", 0, "train"),
            row("train:1", "beta post", 1, "train"),
            row("train:2", "gamma post", 2, "train"),
        ]
    }

    /// Fixture A — a train row byte-identical to a validation row, same label.
    #[test]
    fn dedup_fixture_a_exact_duplicate_is_one_group_and_one_decrement() {
        let train = train_base();
        let validation = vec![row("validation:0", "alpha post", 0, "validation")];
        let record = coalesced_exclusions(&[("train", &train), ("validation", &validation)]);

        assert_eq!(record.excluded_train_ids(), ["train:0".to_string()]);
        assert_eq!(
            record.groups().len(),
            1,
            "an exact duplicate is also a normalized duplicate; it must not be two groups"
        );
        let group = &record.groups()[0];
        assert!(group.detected_by.exact, "exact edge must be recorded");
        assert!(
            group.detected_by.normalized,
            "an exact duplicate always co-fires the normalized edge"
        );
        assert!(!group.label_conflict);
        assert_eq!(
            group.members,
            vec![
                ("train".to_string(), "train:0".to_string()),
                ("validation".to_string(), "validation:0".to_string()),
            ]
        );
        assert_eq!(record.reduced_pools().get(&0), Some(&0));
        assert_eq!(record.reduced_pools().get(&1), Some(&1));
        assert_eq!(record.reduced_pools().get(&2), Some(&1));
    }

    /// Fixture B — differs from a test row only by trailing whitespace.
    #[test]
    fn dedup_fixture_b_whitespace_variant_is_normalized_only() {
        let train = train_base();
        let test = vec![row("test:0", "beta post  ", 1, "test")];
        let record = coalesced_exclusions(&[("train", &train), ("test", &test)]);

        assert_eq!(record.excluded_train_ids(), ["train:1".to_string()]);
        assert_eq!(record.groups().len(), 1);
        let group = &record.groups()[0];
        assert!(
            !group.detected_by.exact,
            "the bytes differ, so no exact edge exists"
        );
        assert!(group.detected_by.normalized);
        assert_eq!(record.reduced_pools().get(&1), Some(&0));
    }

    /// Fixture C — duplicate content across splits with DIFFERENT labels. Real data has
    /// none, so this path can only be reached synthetically.
    #[test]
    fn dedup_fixture_c_label_conflict_is_flagged() {
        let train = train_base();
        let validation = vec![row("validation:0", "gamma post", 0, "validation")];
        let record = coalesced_exclusions(&[("train", &train), ("validation", &validation)]);

        assert_eq!(record.excluded_train_ids(), ["train:2".to_string()]);
        assert_eq!(record.groups().len(), 1);
        assert!(record.groups()[0].label_conflict);
    }

    /// Fixture D — a three-way chain. `train:0` equals `validation:0` exactly, and
    /// `validation:0` equals `test:0` only after normalization. Union-find must merge all
    /// three into ONE component and decrement the train pool exactly once.
    #[test]
    fn dedup_fixture_d_three_way_chain_is_one_component() {
        let train = train_base();
        let validation = vec![row("validation:0", "alpha post", 0, "validation")];
        let test = vec![row("test:0", "  alpha   post ", 0, "test")];
        let record = coalesced_exclusions(&[
            ("test", &test),
            ("train", &train),
            ("validation", &validation),
        ]);

        assert_eq!(record.groups().len(), 1, "the chain is ONE component");
        let group = &record.groups()[0];
        assert_eq!(group.members.len(), 3);
        assert!(group.detected_by.exact);
        assert!(group.detected_by.normalized);
        assert_eq!(record.excluded_train_ids(), ["train:0".to_string()]);
        assert_eq!(record.reduced_pools().get(&0), Some(&0));
    }

    #[test]
    fn dedup_no_duplicates_leaves_the_pools_intact() {
        let train = train_base();
        let validation = vec![row("validation:0", "delta post", 0, "validation")];
        let record = coalesced_exclusions(&[("train", &train), ("validation", &validation)]);

        assert!(record.excluded_train_ids().is_empty());
        assert!(record.groups().is_empty());
        assert_eq!(record.reduced_pools().get(&0), Some(&1));
        assert_eq!(record.reduced_pools().get(&1), Some(&1));
        assert_eq!(record.reduced_pools().get(&2), Some(&1));
        assert!(!record.to_canonical_bytes().expect("serializes").is_empty());
    }

    #[test]
    fn dedup_records_the_normalization_version() {
        let train = train_base();
        let record = coalesced_exclusions(&[("train", &train)]);
        let json = String::from_utf8(record.to_canonical_bytes().expect("serializes"))
            .expect("canonical bytes are UTF-8");
        assert!(json.contains(CONTENT_NORMALIZATION_VERSION));
    }

    /// A record that reaches a manifest reaches a hash, so its content must not depend on
    /// the order the caller happened to collect rows in.
    #[test]
    fn dedup_is_order_independent_in_both_record_and_hash() {
        let mut train = train_base();
        train.push(row("train:3", "alpha post", 0, "train"));
        let validation = vec![
            row("validation:0", "alpha post", 0, "validation"),
            row("validation:1", "beta post ", 1, "validation"),
        ];
        let forward = coalesced_exclusions(&[("train", &train), ("validation", &validation)]);

        let mut permuted_train = train.clone();
        permuted_train.reverse();
        let mut permuted_validation = validation.clone();
        permuted_validation.reverse();
        let backward = coalesced_exclusions(&[
            ("validation", &permuted_validation),
            ("train", &permuted_train),
        ]);

        assert_eq!(forward, backward);
        assert_eq!(forward.hash(), backward.hash());
    }

    #[test]
    fn dedup_hash_changes_when_the_excluded_set_changes() {
        let train = train_base();
        let clean = coalesced_exclusions(&[("train", &train)]);
        let validation = vec![row("validation:0", "alpha post", 0, "validation")];
        let dirty = coalesced_exclusions(&[("train", &train), ("validation", &validation)]);
        assert_ne!(clean.hash(), dirty.hash());
    }

    #[test]
    fn dedup_within_split_duplicate_content_is_not_a_cross_split_group() {
        // Two train rows with identical content are NOT evaluation leakage; only content
        // spanning two split roles is.
        let train = vec![
            row("train:0", "alpha post", 0, "train"),
            row("train:1", "alpha post", 0, "train"),
        ];
        let record = coalesced_exclusions(&[("train", &train)]);
        assert!(record.groups().is_empty());
        assert!(record.excluded_train_ids().is_empty());
        assert_eq!(record.reduced_pools().get(&0), Some(&2));
    }
}
