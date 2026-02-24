// =========================================================================
// FALSIFY-KF: KFold cross-validation contract (aprender model_selection)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-KF-* tests for KFold
//   Why 2: model_selection tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for cross-validation yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: KFold was "obviously correct" (index arithmetic)
//
// References:
//   - Stone (1974) "Cross-Validatory Choice and Assessment of Predictions"
// =========================================================================

use super::*;

/// FALSIFY-KF-001: K-Fold produces exactly K splits
#[test]
fn falsify_kf_001_produces_k_splits() {
    let kfold = KFold::new(5);
    let splits = kfold.split(100);

    assert_eq!(
        splits.len(),
        5,
        "FALSIFIED KF-001: splits={}, expected 5",
        splits.len()
    );
}

/// FALSIFY-KF-002: Every sample appears in exactly one test fold
#[test]
fn falsify_kf_002_every_sample_in_one_test_fold() {
    let kfold = KFold::new(5);
    let splits = kfold.split(20);

    let mut test_counts = vec![0usize; 20];
    for (_train, test) in &splits {
        for &idx in test {
            test_counts[idx] += 1;
        }
    }

    for (i, &count) in test_counts.iter().enumerate() {
        assert_eq!(
            count, 1,
            "FALSIFIED KF-002: sample {i} appeared in {count} test folds (expected 1)"
        );
    }
}

/// FALSIFY-KF-003: Train + test indices cover all samples per fold
#[test]
fn falsify_kf_003_train_test_cover_all() {
    let n = 17; // non-divisible by K to test remainder handling
    let kfold = KFold::new(4);
    let splits = kfold.split(n);

    for (fold_idx, (train, test)) in splits.iter().enumerate() {
        let mut all: Vec<usize> = train.iter().chain(test.iter()).copied().collect();
        all.sort_unstable();
        all.dedup();

        assert_eq!(
            all.len(),
            n,
            "FALSIFIED KF-003: fold {fold_idx} covers {} samples, expected {n}",
            all.len()
        );
    }
}

/// FALSIFY-KF-004: Train and test sets are disjoint within each fold
#[test]
fn falsify_kf_004_train_test_disjoint() {
    let kfold = KFold::new(3);
    let splits = kfold.split(30);

    for (fold_idx, (train, test)) in splits.iter().enumerate() {
        use std::collections::HashSet;
        let train_set: HashSet<usize> = train.iter().copied().collect();
        let test_set: HashSet<usize> = test.iter().copied().collect();

        let overlap: Vec<usize> = train_set.intersection(&test_set).copied().collect();
        assert!(
            overlap.is_empty(),
            "FALSIFIED KF-004: fold {fold_idx} has {}-sample overlap between train/test",
            overlap.len()
        );
    }
}

mod kf_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-KF-001-prop: KFold produces exactly K splits for random K/n
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_kf_001_prop_k_splits(
            k in 2..=10usize,
            n in 10..=50usize,
        ) {
            let k = k.min(n); // K cannot exceed n
            let kfold = KFold::new(k);
            let splits = kfold.split(n);
            prop_assert_eq!(
                splits.len(),
                k,
                "FALSIFIED KF-001-prop: splits={} != k={}",
                splits.len(), k
            );
        }
    }

    /// FALSIFY-KF-002-prop: Every sample appears in exactly one test fold
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        #[test]
        fn falsify_kf_002_prop_sample_coverage(
            k in 2..=5usize,
            n in 10..=30usize,
        ) {
            let k = k.min(n);
            let kfold = KFold::new(k);
            let splits = kfold.split(n);

            let mut test_counts = vec![0usize; n];
            for (_train, test) in &splits {
                for &idx in test {
                    test_counts[idx] += 1;
                }
            }

            for (i, &count) in test_counts.iter().enumerate() {
                prop_assert_eq!(
                    count, 1,
                    "FALSIFIED KF-002-prop: sample {} appeared {} times",
                    i, count
                );
            }
        }
    }
}
