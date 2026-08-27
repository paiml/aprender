//! Sorted per-class buckets over a selection pool.
//!
//! Sorted `Vec`s throughout, never hash-map iteration: any collection whose order
//! reaches a hash or a manifest must have a defined order, or determinism becomes a
//! property of the allocator (PF-006).
//!
//! # What a bucket is, exactly
//!
//! One entry per DECLARED class label — including labels whose pool is empty — holding
//! the training ids of that class in ascending lexicographic order, with every id in the
//! dataset's [`ExclusionRecord`](crate::dedup::ExclusionRecord) removed.
//!
//! Declaring the empty buckets matters. If a label with no remaining rows simply vanished
//! from the map, selection would silently produce a manifest with fewer classes than the
//! label map declares, and the per-class balance check would pass on it. Keeping the empty
//! bucket turns that case into `CrossSplitDuplicateUnderflow`, naming the class.
//!
//! # Exclusions are applied HERE, once
//!
//! Cross-split duplicate content is excluded from the *selection pool* and recorded, never
//! fatal at prepare time (D-18, upheld by D-27). This module is the single place that
//! subtraction happens, so no selection path can forget it and no path can apply it twice.

use std::collections::{BTreeMap, BTreeSet};

use crate::prepared::{Canonical, PreparedDataset};

/// Per-class training-id pools, sorted and post-exclusion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassBuckets {
    buckets: BTreeMap<usize, Vec<String>>,
}

impl ClassBuckets {
    /// Build the buckets of a canonical dataset's training split, minus its exclusions.
    ///
    /// Only `PreparedDataset<Canonical>` has buckets to build: a compatibility dataset is
    /// a different type and cannot be passed (D-19).
    pub fn from_prepared(dataset: &PreparedDataset<Canonical>) -> Self {
        let train = dataset.train();
        let excluded: BTreeSet<&str> = dataset
            .exclusions()
            .excluded_train_ids()
            .iter()
            .map(String::as_str)
            .collect();

        // Every DECLARED label gets a bucket, empty or not — see the module docs.
        let mut buckets: BTreeMap<usize, Vec<String>> = (0..train.class_counts().len())
            .map(|label| (label, Vec::new()))
            .collect();

        for row in train.rows() {
            if excluded.contains(row.id.as_str()) {
                continue;
            }
            buckets.entry(row.label).or_default().push(row.id.clone());
        }

        for ids in buckets.values_mut() {
            ids.sort_unstable();
        }

        Self { buckets }
    }

    /// `(label, pool_size)` for every declared class, ascending by label.
    pub fn class_sizes(&self) -> Vec<(usize, u64)> {
        self.buckets
            .iter()
            .map(|(label, ids)| (*label, ids.len() as u64))
            .collect()
    }

    /// Every declared class label, ascending.
    pub fn labels(&self) -> Vec<usize> {
        self.buckets.keys().copied().collect()
    }

    /// One class's sorted, post-exclusion pool.
    ///
    /// An undeclared label yields an empty slice rather than an error: the caller's own
    /// fail-closed check is the place that turns "no rows" into a typed failure, and two
    /// places deciding that would eventually disagree.
    pub(crate) fn ids(&self, label: usize) -> &[String] {
        self.buckets.get(&label).map_or(&[], Vec::as_slice)
    }
}

#[cfg(test)]
mod buckets_tests {
    use super::ClassBuckets;
    use crate::ledger::AccessLedger;
    use crate::select::test_corpus;

    #[test]
    fn buckets_sort_ids_and_declare_every_label() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset(4, &mut ledger);
        let buckets = ClassBuckets::from_prepared(&dataset);

        assert_eq!(buckets.labels(), vec![0, 1, 2]);
        assert_eq!(buckets.class_sizes(), vec![(0, 4), (1, 4), (2, 4)]);
        for label in buckets.labels() {
            let ids = buckets.ids(label);
            assert_eq!(ids.len(), 4);
            assert!(
                ids.windows(2).all(|pair| pair[0] < pair[1]),
                "class {label} pool must be strictly ascending, got {ids:?}"
            );
        }
    }

    /// The training rows are handed in interleaved by class, so a bucket that merely
    /// preserved ingest order would come out unsorted. This is the vacuity guard for the
    /// assertion above.
    #[test]
    fn buckets_do_real_sorting_work() {
        let (train, ..) = test_corpus::rows(4);
        let ingest_order: Vec<&str> = train
            .iter()
            .filter(|row| row.label == 0)
            .map(|row| row.id.as_str())
            .collect();
        let mut sorted = ingest_order.clone();
        sorted.sort_unstable();
        assert_ne!(
            ingest_order, sorted,
            "the fixture must NOT already be sorted, or the sort assertion proves nothing"
        );
    }

    #[test]
    fn buckets_omit_every_excluded_training_id() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset_with_cross_split_duplicate(4, &mut ledger);
        let excluded = dataset.exclusions().excluded_train_ids().to_vec();
        assert_eq!(
            excluded.len(),
            1,
            "the fixture must produce exactly one exclusion, or this test is vacuous"
        );

        let buckets = ClassBuckets::from_prepared(&dataset);
        let all: Vec<&String> = buckets
            .labels()
            .into_iter()
            .flat_map(|label| buckets.ids(label).iter())
            .collect();
        for id in &excluded {
            assert!(
                !all.contains(&id),
                "excluded id {id:?} must not appear in any bucket"
            );
        }
        assert_eq!(buckets.class_sizes(), vec![(0, 3), (1, 4), (2, 4)]);
    }

    #[test]
    fn buckets_keep_a_declared_label_whose_pool_is_empty() {
        let mut ledger = AccessLedger::new();
        let dataset = test_corpus::dataset_with_empty_class(4, &mut ledger);
        let buckets = ClassBuckets::from_prepared(&dataset);

        assert_eq!(buckets.labels(), vec![0, 1, 2]);
        assert_eq!(buckets.ids(2), Vec::<String>::new().as_slice());
        assert_eq!(buckets.class_sizes(), vec![(0, 4), (1, 4), (2, 0)]);
    }
}
