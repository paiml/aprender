// SHIP-TWO-001 — `random-forest-v1` algorithm-level PARTIAL discharge
// for FALSIFY-RF-001..004.
//
// Contract: `contracts/random-forest-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Four random-forest gates:
//
// - RF-001 (predictions in label range): every predict(x) ∈ training_labels.
// - RF-002 (deterministic with seed): predict(X, seed=s) ≡ predict(X, seed=s).
// - RF-003 (ensemble size): |forest.trees| = n_estimators.
// - RF-004 (prediction length): |predict(X)| = |X|.
//
// All four are pure invariants over (predictions, training_labels,
// n_estimators, m). No kernel selection, no SIMD path.
//
// In-module reference: `forest_majority_vote` — pure majority vote
// across a fixed `Vec<Vec<usize>>` of per-tree predictions, ties
// broken by smallest class id (deterministic).

/// Minimum legal value of n_estimators.
pub const AC_RF_003_MIN_ESTIMATORS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RfVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference majority-vote ensemble.
// -----------------------------------------------------------------------------

/// Compute majority-vote prediction for a single sample given B per-tree
/// predictions. Ties broken by smallest class id (deterministic).
#[must_use]
pub fn majority_vote(per_tree_preds: &[usize], n_classes: usize) -> usize {
    let mut counts = vec![0_usize; n_classes];
    for &p in per_tree_preds {
        if p < n_classes {
            counts[p] += 1;
        }
    }
    let mut best = 0_usize;
    let mut best_count = 0_usize;
    for (k, &c) in counts.iter().enumerate() {
        if c > best_count {
            best = k;
            best_count = c;
        }
    }
    best
}

/// Predict over a batch given a forest of B trees, each tree's
/// per-sample predictions in `tree_preds[b][i]`.
#[must_use]
pub fn forest_predict(tree_preds: &[Vec<usize>], n_classes: usize) -> Vec<usize> {
    if tree_preds.is_empty() {
        return Vec::new();
    }
    let n_samples = tree_preds[0].len();
    let mut out = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let per_tree: Vec<usize> = tree_preds.iter().map(|t| t[i]).collect();
        out.push(majority_vote(&per_tree, n_classes));
    }
    out
}

// -----------------------------------------------------------------------------
// Verdict 1: RF-001 — predictions ∈ training labels.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_predictions_in_labels(
    predictions: &[usize],
    training_labels: &[usize],
) -> RfVerdict {
    if training_labels.is_empty() {
        return if predictions.is_empty() {
            RfVerdict::Pass
        } else {
            RfVerdict::Fail
        };
    }
    let label_set: std::collections::HashSet<usize> =
        training_labels.iter().copied().collect();
    for &p in predictions {
        if !label_set.contains(&p) {
            return RfVerdict::Fail;
        }
    }
    RfVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 2: RF-002 — deterministic with seed.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_seed_determinism(
    predictions_run1: &[usize],
    predictions_run2: &[usize],
) -> RfVerdict {
    if predictions_run1.len() != predictions_run2.len() {
        return RfVerdict::Fail;
    }
    if predictions_run1 == predictions_run2 {
        RfVerdict::Pass
    } else {
        RfVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: RF-003 — ensemble size.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_ensemble_size(actual: usize, expected_n_estimators: usize) -> RfVerdict {
    if expected_n_estimators < AC_RF_003_MIN_ESTIMATORS {
        return RfVerdict::Fail;
    }
    if actual == expected_n_estimators {
        RfVerdict::Pass
    } else {
        RfVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: RF-004 — prediction length.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_prediction_length(predictions_len: usize, samples_len: usize) -> RfVerdict {
    if predictions_len == samples_len {
        RfVerdict::Pass
    } else {
        RfVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_estimators_is_one() {
        assert_eq!(AC_RF_003_MIN_ESTIMATORS, 1);
    }

    // -------------------------------------------------------------------------
    // Section 2: RF-001 Pass band.
    // -------------------------------------------------------------------------
    #[test]
    fn rf001_pass_all_predictions_in_labels() {
        let preds = vec![0_usize, 1, 0, 2, 1];
        let labels = vec![0_usize, 1, 2];
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Pass
        );
    }

    #[test]
    fn rf001_pass_empty_predictions() {
        let preds: Vec<usize> = vec![];
        let labels = vec![0_usize, 1, 2];
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Pass
        );
    }

    #[test]
    fn rf001_pass_single_class_problem() {
        let preds = vec![5_usize, 5, 5, 5];
        let labels = vec![5_usize];
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Pass
        );
    }

    #[test]
    fn rf001_pass_unsorted_label_set() {
        let preds = vec![7_usize, 3, 9];
        let labels = vec![9_usize, 3, 7]; // labels can be any usize, in any order
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: RF-001 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn rf001_fail_one_unseen_label() {
        let preds = vec![0_usize, 1, 5]; // 5 is OOD
        let labels = vec![0_usize, 1, 2];
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Fail
        );
    }

    #[test]
    fn rf001_fail_all_unseen() {
        let preds = vec![10_usize, 20, 30];
        let labels = vec![0_usize, 1, 2];
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Fail
        );
    }

    #[test]
    fn rf001_fail_predictions_with_no_training_labels() {
        // Empty training set but predictions exist — uninitialized
        // leaf node regression.
        let preds = vec![0_usize];
        let labels: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: RF-002 Pass band — determinism.
    // -------------------------------------------------------------------------
    #[test]
    fn rf002_pass_identical_runs() {
        let r1 = vec![0_usize, 1, 0, 2];
        let r2 = vec![0_usize, 1, 0, 2];
        assert_eq!(
            verdict_from_seed_determinism(&r1, &r2),
            RfVerdict::Pass
        );
    }

    #[test]
    fn rf002_pass_empty_both() {
        let r: Vec<usize> = vec![];
        assert_eq!(
            verdict_from_seed_determinism(&r, &r),
            RfVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: RF-002 Fail band.
    // -------------------------------------------------------------------------
    #[test]
    fn rf002_fail_one_off() {
        let r1 = vec![0_usize, 1, 0, 2];
        let r2 = vec![0_usize, 1, 0, 1]; // last differs
        assert_eq!(
            verdict_from_seed_determinism(&r1, &r2),
            RfVerdict::Fail
        );
    }

    #[test]
    fn rf002_fail_length_mismatch() {
        let r1 = vec![0_usize, 1, 2];
        let r2 = vec![0_usize, 1];
        assert_eq!(
            verdict_from_seed_determinism(&r1, &r2),
            RfVerdict::Fail
        );
    }

    #[test]
    fn rf002_fail_completely_different() {
        let r1 = vec![0_usize, 0, 0];
        let r2 = vec![1_usize, 1, 1];
        assert_eq!(
            verdict_from_seed_determinism(&r1, &r2),
            RfVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: RF-003 — ensemble size.
    // -------------------------------------------------------------------------
    #[test]
    fn rf003_pass_match() {
        assert_eq!(
            verdict_from_ensemble_size(100, 100),
            RfVerdict::Pass
        );
    }

    #[test]
    fn rf003_pass_minimum_one() {
        assert_eq!(verdict_from_ensemble_size(1, 1), RfVerdict::Pass);
    }

    #[test]
    fn rf003_fail_off_by_one() {
        assert_eq!(verdict_from_ensemble_size(99, 100), RfVerdict::Fail);
        assert_eq!(verdict_from_ensemble_size(101, 100), RfVerdict::Fail);
    }

    #[test]
    fn rf003_fail_zero_estimators() {
        // n_estimators=0 is illegal per contract (must be >= 1).
        assert_eq!(verdict_from_ensemble_size(0, 0), RfVerdict::Fail);
    }

    #[test]
    fn rf003_fail_early_termination() {
        // The contract failure mode: "Tree construction loop off-by-one
        // or early termination".
        assert_eq!(verdict_from_ensemble_size(50, 100), RfVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: RF-004 — prediction length.
    // -------------------------------------------------------------------------
    #[test]
    fn rf004_pass_match() {
        assert_eq!(verdict_from_prediction_length(10, 10), RfVerdict::Pass);
    }

    #[test]
    fn rf004_pass_zero() {
        assert_eq!(verdict_from_prediction_length(0, 0), RfVerdict::Pass);
    }

    #[test]
    fn rf004_fail_predictions_too_few() {
        assert_eq!(verdict_from_prediction_length(9, 10), RfVerdict::Fail);
    }

    #[test]
    fn rf004_fail_predictions_too_many() {
        assert_eq!(verdict_from_prediction_length(11, 10), RfVerdict::Fail);
    }

    #[test]
    fn rf004_fail_loop_skipping() {
        // Prediction loop skipping every other sample.
        assert_eq!(verdict_from_prediction_length(50, 100), RfVerdict::Fail);
    }

    #[test]
    fn rf004_fail_loop_duplicating() {
        assert_eq!(verdict_from_prediction_length(200, 100), RfVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Domain — majority vote.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_majority_vote_basic() {
        // 5 trees, classes {0, 1, 2}: 3 vote 1, 1 votes 0, 1 votes 2.
        let votes = vec![1_usize, 0, 1, 1, 2];
        assert_eq!(majority_vote(&votes, 3), 1);
    }

    #[test]
    fn domain_majority_vote_tie_breaks_smallest() {
        // Two-way tie at counts 2+2: must return smallest.
        let votes = vec![1_usize, 1, 0, 0];
        assert_eq!(majority_vote(&votes, 2), 0);
    }

    #[test]
    fn domain_majority_vote_unanimous() {
        let votes = vec![3_usize; 7];
        assert_eq!(majority_vote(&votes, 4), 3);
    }

    #[test]
    fn domain_forest_predict_three_samples() {
        // 3 trees, 3 samples; tree predictions per row.
        let trees = vec![
            vec![0_usize, 1, 1], // tree 0
            vec![0_usize, 0, 1], // tree 1
            vec![1_usize, 1, 1], // tree 2
        ];
        let preds = forest_predict(&trees, 2);
        // Sample 0: votes [0, 0, 1] → 0.
        // Sample 1: votes [1, 0, 1] → 1.
        // Sample 2: votes [1, 1, 1] → 1.
        assert_eq!(preds, vec![0, 1, 1]);
    }

    #[test]
    fn domain_forest_predict_empty_forest() {
        let preds = forest_predict(&[], 3);
        assert!(preds.is_empty());
    }

    // -------------------------------------------------------------------------
    // Section 9: Sweep — n_estimators boundary.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_ensemble_size_pass_band() {
        for n in [1_usize, 10, 100, 1000] {
            assert_eq!(
                verdict_from_ensemble_size(n, n),
                RfVerdict::Pass,
                "n={n}"
            );
        }
    }

    #[test]
    fn sweep_ensemble_size_fail_band() {
        let test_cases = [(99_usize, 100), (101, 100), (0, 100), (1000, 999)];
        for (actual, expected) in test_cases {
            assert_eq!(
                verdict_from_ensemble_size(actual, expected),
                RfVerdict::Fail,
                "actual={actual} expected={expected}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 10: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_uninitialized_leaf_node() {
        // RF-001 if_fails: "Tree leaf prediction not constrained to
        // training labels". Simulate by including label 99 in preds
        // with training labels {0, 1, 2}.
        let preds = vec![0_usize, 1, 99]; // 99 came from uninitialized leaf
        let labels = vec![0_usize, 1, 2];
        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Fail
        );
    }

    #[test]
    fn realistic_thread_dependent_ordering() {
        // RF-002 if_fails: "Random state not properly seeded or
        // thread-dependent ordering". Two runs disagree.
        let r1 = vec![0_usize, 1, 2, 0, 1];
        let r2 = vec![1_usize, 0, 2, 0, 1]; // first two flipped
        assert_eq!(
            verdict_from_seed_determinism(&r1, &r2),
            RfVerdict::Fail
        );
    }

    #[test]
    fn realistic_off_by_one_tree_loop() {
        // RF-003 if_fails: "Tree construction loop off-by-one".
        assert_eq!(verdict_from_ensemble_size(99, 100), RfVerdict::Fail);
        assert_eq!(verdict_from_ensemble_size(101, 100), RfVerdict::Fail);
    }

    #[test]
    fn realistic_prediction_skipping_samples() {
        // RF-004 if_fails: "Prediction loop skipping or duplicating
        // samples".
        assert_eq!(verdict_from_prediction_length(50, 100), RfVerdict::Fail);
        assert_eq!(verdict_from_prediction_length(200, 100), RfVerdict::Fail);
    }

    #[test]
    fn realistic_full_forest_pipeline_two_class() {
        // 3-tree forest, 2-class problem, 5 samples; verify all 4 gates
        // hold simultaneously.
        let labels = vec![0_usize, 1];
        let trees = vec![
            vec![0_usize, 0, 1, 1, 0],
            vec![0_usize, 1, 1, 1, 0],
            vec![0_usize, 0, 1, 1, 1],
        ];
        let preds = forest_predict(&trees, 2);

        assert_eq!(
            verdict_from_predictions_in_labels(&preds, &labels),
            RfVerdict::Pass
        );
        let preds_again = forest_predict(&trees, 2);
        assert_eq!(
            verdict_from_seed_determinism(&preds, &preds_again),
            RfVerdict::Pass
        );
        assert_eq!(
            verdict_from_ensemble_size(trees.len(), 3),
            RfVerdict::Pass
        );
        assert_eq!(
            verdict_from_prediction_length(preds.len(), 5),
            RfVerdict::Pass
        );
    }
}
