// SHIP-TWO-001 — `apr-data-pipeline-v1` algorithm-level PARTIAL
// discharge for FALSIFY-DATA-001..005.
//
// Contract: `contracts/apr-data-pipeline-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Five data-pipeline gates:
//
// - DATA-001 (split preserves all samples): train + val + test == n.
// - DATA-002 (no cross-contamination): split sets are pairwise disjoint.
// - DATA-003 (DataLoader yields all samples): N samples / batch_size yields
//   exactly N samples (last partial batch not dropped).
// - DATA-004 (preprocessing idempotent for special tokens):
//   preprocess(preprocess(x)) has same special-token count as
//   preprocess(x).
// - DATA-005 (validation is read-only): file SHA-256 unchanged.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: DATA-001 — split preserves all samples.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_split_preserves_all(
    train_count: usize,
    val_count: usize,
    test_count: usize,
    original_count: usize,
) -> DataVerdict {
    if original_count == 0 {
        return DataVerdict::Fail;
    }
    if train_count + val_count + test_count == original_count {
        DataVerdict::Pass
    } else {
        DataVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: DATA-002 — no cross-contamination.
// -----------------------------------------------------------------------------

/// Pass iff `train_ids`, `val_ids`, `test_ids` are pairwise disjoint.
#[must_use]
pub fn verdict_from_no_cross_contamination(
    train_ids: &[u64],
    val_ids: &[u64],
    test_ids: &[u64],
) -> DataVerdict {
    let train: std::collections::HashSet<u64> = train_ids.iter().copied().collect();
    let val: std::collections::HashSet<u64> = val_ids.iter().copied().collect();
    let test: std::collections::HashSet<u64> = test_ids.iter().copied().collect();

    if train.intersection(&val).next().is_some() {
        return DataVerdict::Fail;
    }
    if train.intersection(&test).next().is_some() {
        return DataVerdict::Fail;
    }
    if val.intersection(&test).next().is_some() {
        return DataVerdict::Fail;
    }
    // Also reject internal duplicates (within-split shuffling bug).
    if train.len() != train_ids.len() {
        return DataVerdict::Fail;
    }
    if val.len() != val_ids.len() {
        return DataVerdict::Fail;
    }
    if test.len() != test_ids.len() {
        return DataVerdict::Fail;
    }
    DataVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 3: DATA-003 — DataLoader yields all samples.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_dataloader_yields_all(
    yielded_count: usize,
    expected_count: usize,
) -> DataVerdict {
    if expected_count == 0 {
        return DataVerdict::Fail;
    }
    if yielded_count == expected_count {
        DataVerdict::Pass
    } else {
        DataVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: DATA-004 — preprocessing idempotent for special tokens.
// -----------------------------------------------------------------------------

/// `count_after_one_pass` is special-token count after preprocess(x).
/// `count_after_two_passes` is count after preprocess(preprocess(x)).
/// Pass iff equal (idempotent).
#[must_use]
pub fn verdict_from_preprocessing_idempotent(
    count_after_one_pass: usize,
    count_after_two_passes: usize,
) -> DataVerdict {
    if count_after_one_pass == count_after_two_passes {
        DataVerdict::Pass
    } else {
        DataVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: DATA-005 — validation is read-only.
// -----------------------------------------------------------------------------

/// Pass iff `sha_before == sha_after` (same hash). Hashes are 32 bytes.
#[must_use]
pub fn verdict_from_validation_readonly(
    sha_before: &[u8; 32],
    sha_after: &[u8; 32],
) -> DataVerdict {
    if sha_before == sha_after {
        DataVerdict::Pass
    } else {
        DataVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: DATA-001 — split preserves all samples.
    // -------------------------------------------------------------------------
    #[test]
    fn data001_pass_80_10_10_split_of_1000() {
        assert_eq!(
            verdict_from_split_preserves_all(800, 100, 100, 1000),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data001_pass_70_15_15_with_remainder() {
        // 30 / 7 = 4r2; spread as 22/4/4.
        assert_eq!(
            verdict_from_split_preserves_all(22, 4, 4, 30),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data001_pass_train_only() {
        assert_eq!(
            verdict_from_split_preserves_all(1000, 0, 0, 1000),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data001_fail_one_sample_lost() {
        // Rounding error: 800 + 100 + 99 = 999 != 1000.
        assert_eq!(
            verdict_from_split_preserves_all(800, 100, 99, 1000),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data001_fail_one_extra_sample() {
        assert_eq!(
            verdict_from_split_preserves_all(800, 100, 101, 1000),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data001_fail_zero_original() {
        assert_eq!(
            verdict_from_split_preserves_all(0, 0, 0, 0),
            DataVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: DATA-002 — no cross-contamination.
    // -------------------------------------------------------------------------
    #[test]
    fn data002_pass_disjoint_splits() {
        let train = vec![1_u64, 2, 3, 4, 5];
        let val = vec![6_u64, 7];
        let test = vec![8_u64, 9, 10];
        assert_eq!(
            verdict_from_no_cross_contamination(&train, &val, &test),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data002_pass_empty_val_test() {
        let train = vec![1_u64, 2, 3];
        assert_eq!(
            verdict_from_no_cross_contamination(&train, &[], &[]),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data002_fail_train_val_overlap() {
        let train = vec![1_u64, 2, 3];
        let val = vec![3_u64, 4]; // 3 leaks
        let test = vec![5_u64];
        assert_eq!(
            verdict_from_no_cross_contamination(&train, &val, &test),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data002_fail_train_test_overlap() {
        let train = vec![1_u64, 2, 3];
        let val = vec![4_u64];
        let test = vec![3_u64, 5]; // 3 leaks
        assert_eq!(
            verdict_from_no_cross_contamination(&train, &val, &test),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data002_fail_val_test_overlap() {
        let train = vec![1_u64];
        let val = vec![2_u64, 3];
        let test = vec![3_u64, 4]; // 3 leaks
        assert_eq!(
            verdict_from_no_cross_contamination(&train, &val, &test),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data002_fail_internal_duplicate_in_train() {
        let train = vec![1_u64, 2, 2, 3]; // 2 duplicated
        assert_eq!(
            verdict_from_no_cross_contamination(&train, &[], &[]),
            DataVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: DATA-003 — DataLoader yields all samples.
    // -------------------------------------------------------------------------
    #[test]
    fn data003_pass_full_iteration() {
        // 97 samples / batch_size 10 = 9 full + 1 partial of 7. Total 97.
        assert_eq!(
            verdict_from_dataloader_yields_all(97, 97),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data003_pass_evenly_divisible() {
        // 100 samples / batch_size 10 = 10 full batches.
        assert_eq!(
            verdict_from_dataloader_yields_all(100, 100),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data003_pass_single_sample() {
        assert_eq!(
            verdict_from_dataloader_yields_all(1, 1),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data003_fail_last_partial_batch_dropped() {
        // The classic bug: drop_last=true silently dropped the partial.
        // 97 samples / batch_size 10, drop_last → 90 yielded.
        assert_eq!(
            verdict_from_dataloader_yields_all(90, 97),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data003_fail_extra_samples() {
        assert_eq!(
            verdict_from_dataloader_yields_all(101, 100),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data003_fail_zero_expected() {
        assert_eq!(
            verdict_from_dataloader_yields_all(0, 0),
            DataVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: DATA-004 — preprocessing idempotent.
    // -------------------------------------------------------------------------
    #[test]
    fn data004_pass_idempotent_one_cls_one_sep() {
        // After 1 pass: 1 [CLS] + 1 [SEP] = 2 special tokens.
        // After 2 passes: still 2.
        assert_eq!(
            verdict_from_preprocessing_idempotent(2, 2),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data004_pass_zero_special_tokens() {
        assert_eq!(
            verdict_from_preprocessing_idempotent(0, 0),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data004_fail_doubled_after_second_pass() {
        // The contract failure: special tokens prepended/appended on each call.
        assert_eq!(
            verdict_from_preprocessing_idempotent(2, 4),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data004_fail_zero_then_two() {
        assert_eq!(
            verdict_from_preprocessing_idempotent(0, 2),
            DataVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: DATA-005 — validation is read-only.
    // -------------------------------------------------------------------------
    #[test]
    fn data005_pass_sha_unchanged() {
        let sha = [0xAB_u8; 32];
        assert_eq!(
            verdict_from_validation_readonly(&sha, &sha),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data005_pass_specific_sha() {
        let sha_before: [u8; 32] = [
            0x12, 0x34, 0x56, 0x78, 0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05,
            0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11,
            0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19,
        ];
        let sha_after = sha_before;
        assert_eq!(
            verdict_from_validation_readonly(&sha_before, &sha_after),
            DataVerdict::Pass
        );
    }

    #[test]
    fn data005_fail_one_byte_changed() {
        let sha_before = [0x00_u8; 32];
        let mut sha_after = sha_before;
        sha_after[5] = 0xFF;
        assert_eq!(
            verdict_from_validation_readonly(&sha_before, &sha_after),
            DataVerdict::Fail
        );
    }

    #[test]
    fn data005_fail_completely_different() {
        let sha_before = [0x00_u8; 32];
        let sha_after = [0xFF_u8; 32];
        assert_eq!(
            verdict_from_validation_readonly(&sha_before, &sha_after),
            DataVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Sweep — split fractions.
    // -------------------------------------------------------------------------
    #[test]
    fn sweep_split_fractions_sum_to_n() {
        // For each split size, sum should equal original.
        let cases = [
            (1000_usize, (700, 200, 100)),
            (123, (100, 13, 10)),
            (1, (1, 0, 0)),
        ];
        for (n, (t, v, ts)) in cases {
            assert_eq!(
                verdict_from_split_preserves_all(t, v, ts, n),
                DataVerdict::Pass,
                "n={n} t={t} v={v} ts={ts}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — contract regression scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_rounding_loses_sample_caught() {
        // DATA-001 if_fails: "Rounding error loses samples at partition
        // boundaries".
        assert_eq!(
            verdict_from_split_preserves_all(699, 200, 100, 1000),
            DataVerdict::Fail
        );
    }

    #[test]
    fn realistic_shuffle_duplicates_caught() {
        // DATA-002 if_fails: "Shuffle produces duplicates across partitions".
        let train = vec![1_u64, 2];
        let val = vec![1_u64]; // 1 leaked into val
        assert_eq!(
            verdict_from_no_cross_contamination(&train, &val, &[]),
            DataVerdict::Fail
        );
    }

    #[test]
    fn realistic_dataloader_drop_last_bug_caught() {
        // DATA-003 if_fails: "Last partial batch dropped (common
        // DataLoader bug)".
        // 97 samples, batch=10, drop_last=true → 90.
        assert_eq!(
            verdict_from_dataloader_yields_all(90, 97),
            DataVerdict::Fail
        );
    }

    #[test]
    fn realistic_double_special_tokens_caught() {
        // DATA-004 if_fails: "Special tokens prepended/appended on
        // each call".
        assert_eq!(
            verdict_from_preprocessing_idempotent(2, 4),
            DataVerdict::Fail
        );
    }

    #[test]
    fn realistic_validation_writes_file_caught() {
        // DATA-005 if_fails: "Validation writes normalized data back
        // to input file".
        let before = [0x12_u8; 32];
        let after = [0x34_u8; 32];
        assert_eq!(
            verdict_from_validation_readonly(&before, &after),
            DataVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_data_pipeline_passes_all_5_gates() {
        // 1000-sample dataset with 80/10/10 split.
        let n = 1000_usize;
        let train_ids: Vec<u64> = (0..800).collect();
        let val_ids: Vec<u64> = (800..900).collect();
        let test_ids: Vec<u64> = (900..1000).collect();

        // Gate 1: split preserves all.
        assert_eq!(
            verdict_from_split_preserves_all(800, 100, 100, n),
            DataVerdict::Pass
        );
        // Gate 2: no cross-contamination.
        assert_eq!(
            verdict_from_no_cross_contamination(&train_ids, &val_ids, &test_ids),
            DataVerdict::Pass
        );
        // Gate 3: DataLoader yields all.
        assert_eq!(
            verdict_from_dataloader_yields_all(n, n),
            DataVerdict::Pass
        );
        // Gate 4: preprocessing idempotent.
        assert_eq!(
            verdict_from_preprocessing_idempotent(2, 2),
            DataVerdict::Pass
        );
        // Gate 5: validation read-only.
        let sha = [0xAB_u8; 32];
        assert_eq!(
            verdict_from_validation_readonly(&sha, &sha),
            DataVerdict::Pass
        );
    }
}
