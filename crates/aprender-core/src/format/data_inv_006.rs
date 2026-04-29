// SHIP-TWO-001 MODEL-2 — `dataset-thestack-python-v1` (C-DATA-THESTACK-PYTHON)
// algorithm-level PARTIAL discharge for INV-DATA-006.
//
// Contract: `contracts/dataset-thestack-python-v1.yaml` v1.0.0 PROPOSED.
// Sibling to [`super::data_inv_004`] (#1146, train range + val floor).
//
// ## What INV-DATA-006 says
//
//   description: Train and val splits are DISJOINT by file sha256. No
//                file appears in both train and val shards.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given two sorted lists of file SHA-256s (one for train
// shards, one for val shards), Pass iff:
//
//   1. Both sets are non-empty.
//   2. Both sets are internally deduplicated (a file MUST NOT appear
//      twice in the same split — a subtler bug class than cross-split
//      leakage but symptomatically similar).
//   3. The intersection of train and val is empty.
//
// Composes with [`super::ship_010::verdict_from_sha256_match`] for hash
// format validation: every hash in either list is a 64-char lowercase
// hex string.
//
// Future implementations (the actual `apr ingest` shard splitter) cannot:
// - Emit train AND val with overlapping file sha256s (eval-set leakage).
// - Emit duplicate files within a single split (also a leakage class).
// - Emit empty train OR empty val (no work done).

use super::ship_010::{verdict_from_sha256_match, Ship010Verdict};

/// Binary verdict for `INV-DATA-006`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataInv006Verdict {
    /// `train_file_sha256s ∩ val_file_sha256s = ∅`. Splits are disjoint
    /// by file SHA-256, both non-empty, both internally deduplicated,
    /// every hash canonical lowercase 64-char hex.
    Pass,
    /// One or more of:
    /// - Either split is empty (caller error).
    /// - Some hash is malformed (delegated to ship_010 → Fail).
    /// - Train has a duplicate within itself.
    /// - Val has a duplicate within itself.
    /// - At least one file SHA-256 appears in both train and val.
    Fail,
}

/// Pure verdict function for `INV-DATA-006`.
///
/// Inputs:
/// - `train_file_sha256s`: per-file SHA-256 hex strings for train-split files.
/// - `val_file_sha256s`: per-file SHA-256 hex strings for val-split files.
///
/// Both slices are expected to be in canonical lowercase hex.
///
/// Pass iff:
/// 1. Both slices are non-empty.
/// 2. Every hash is a valid SHA-256 (composes with [`super::ship_010`]).
/// 3. No hash appears more than once within `train_file_sha256s`.
/// 4. No hash appears more than once within `val_file_sha256s`.
/// 5. No hash appears in both `train_file_sha256s` and `val_file_sha256s`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Disjoint splits — `Pass`:
/// ```
/// use aprender::format::data_inv_006::{
///     verdict_from_split_file_sha256s, DataInv006Verdict,
/// };
/// let train = vec!["0".repeat(64), "1".repeat(64), "2".repeat(64)];
/// let val = vec!["a".repeat(64), "b".repeat(64)];
/// assert_eq!(
///     verdict_from_split_file_sha256s(&train, &val),
///     DataInv006Verdict::Pass,
/// );
/// ```
///
/// Eval-set leakage — `Fail`:
/// ```
/// use aprender::format::data_inv_006::{
///     verdict_from_split_file_sha256s, DataInv006Verdict,
/// };
/// let leaked = "0".repeat(64);
/// let train = vec!["1".repeat(64), leaked.clone(), "2".repeat(64)];
/// let val = vec![leaked, "a".repeat(64)];
/// assert_eq!(
///     verdict_from_split_file_sha256s(&train, &val),
///     DataInv006Verdict::Fail,
/// );
/// ```
#[must_use]
pub fn verdict_from_split_file_sha256s(
    train_file_sha256s: &[String],
    val_file_sha256s: &[String],
) -> DataInv006Verdict {
    if train_file_sha256s.is_empty() || val_file_sha256s.is_empty() {
        return DataInv006Verdict::Fail;
    }
    // Format-validate every hash via ship_010 (single source of truth).
    for h in train_file_sha256s.iter().chain(val_file_sha256s.iter()) {
        if matches!(verdict_from_sha256_match(h, h), Ship010Verdict::Fail) {
            return DataInv006Verdict::Fail;
        }
    }
    // Internal-duplicate detection within each split.
    if has_internal_duplicate(train_file_sha256s) {
        return DataInv006Verdict::Fail;
    }
    if has_internal_duplicate(val_file_sha256s) {
        return DataInv006Verdict::Fail;
    }
    // Cross-split intersection check (the load-bearing leakage rule).
    let train_set: std::collections::HashSet<&String> = train_file_sha256s.iter().collect();
    for v in val_file_sha256s {
        if train_set.contains(v) {
            return DataInv006Verdict::Fail;
        }
    }
    DataInv006Verdict::Pass
}

fn has_internal_duplicate(hashes: &[String]) -> bool {
    let mut seen = std::collections::HashSet::with_capacity(hashes.len());
    for h in hashes {
        if !seen.insert(h.as_str()) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(c: char) -> String {
        std::iter::repeat(c).take(64).collect()
    }

    // -------------------------------------------------------------------------
    // Section 1: Pass band — well-formed disjoint splits.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_disjoint_splits() {
        let train = vec![h('0'), h('1'), h('2')];
        let val = vec![h('a'), h('b')];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Pass
        );
    }

    #[test]
    fn pass_minimal_each_one_file() {
        let train = vec![h('0')];
        let val = vec![h('a')];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Pass
        );
    }

    #[test]
    fn pass_realistic_scale_1k_train_50_val() {
        // ~95/5 split at 1050 files total.
        let train: Vec<String> = (0..1000).map(|i| format!("{:064x}", i)).collect();
        let val: Vec<String> = (1000..1050).map(|i| format!("{:064x}", i)).collect();
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — eval-set leakage (intersection non-empty).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_file_in_both_splits() {
        let leaked = h('0');
        let train = vec![h('1'), leaked.clone(), h('2')];
        let val = vec![leaked, h('a')];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    #[test]
    fn fail_all_files_in_both_splits() {
        // Identical sets — total leakage.
        let same = vec![h('0'), h('1'), h('2')];
        assert_eq!(
            verdict_from_split_file_sha256s(&same, &same),
            DataInv006Verdict::Fail
        );
    }

    #[test]
    fn fail_leakage_at_each_position() {
        for bad_idx in [0_usize, 2, 4] {
            let leaked = h('f');
            let mut train = vec![h('0'), h('1'), h('2'), h('3'), h('4')];
            train[bad_idx] = leaked.clone();
            let val = vec![leaked, h('a')];
            assert_eq!(
                verdict_from_split_file_sha256s(&train, &val),
                DataInv006Verdict::Fail,
                "leakage at index {bad_idx} must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — internal duplicates within a single split.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_train_internal_duplicate() {
        let train = vec![h('0'), h('1'), h('0')]; // h('0') twice
        let val = vec![h('a')];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    #[test]
    fn fail_val_internal_duplicate() {
        let train = vec![h('0')];
        let val = vec![h('a'), h('a')]; // h('a') twice
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — empty splits.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_train() {
        let train: Vec<String> = vec![];
        let val = vec![h('a')];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    #[test]
    fn fail_empty_val() {
        let train = vec![h('0')];
        let val: Vec<String> = vec![];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    #[test]
    fn fail_both_empty() {
        let train: Vec<String> = vec![];
        let val: Vec<String> = vec![];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — format violations (delegated to ship_010).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_uppercase_hex_in_train() {
        let train = vec!["A".repeat(64)];
        let val = vec![h('a')];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    #[test]
    fn fail_too_short_hash_in_val() {
        let train = vec![h('0')];
        let val = vec!["a".repeat(63)];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    #[test]
    fn fail_non_hex_character() {
        let mut bad = "0".repeat(63);
        bad.push('z');
        let train = vec![bad];
        let val = vec![h('a')];
        assert_eq!(
            verdict_from_split_file_sha256s(&train, &val),
            DataInv006Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry — swap train/val should still detect leakage.
    // -------------------------------------------------------------------------
    #[test]
    fn symmetry_under_swap() {
        let train = vec![h('0'), h('1'), h('2')];
        let val = vec![h('a'), h('b')];
        let v1 = verdict_from_split_file_sha256s(&train, &val);
        let v2 = verdict_from_split_file_sha256s(&val, &train);
        assert_eq!(v1, v2, "verdict must be symmetric under split swap");
        assert_eq!(v1, DataInv006Verdict::Pass);
    }

    #[test]
    fn symmetry_leakage_under_swap() {
        let leaked = h('f');
        let a = vec![h('0'), leaked.clone()];
        let b = vec![leaked, h('a')];
        assert_eq!(
            verdict_from_split_file_sha256s(&a, &b),
            DataInv006Verdict::Fail
        );
        assert_eq!(
            verdict_from_split_file_sha256s(&b, &a),
            DataInv006Verdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Internal-duplicate helper — direct test of the private fn.
    // -------------------------------------------------------------------------
    #[test]
    fn helper_has_internal_duplicate_detects() {
        let dupes = vec![h('0'), h('1'), h('0')];
        assert!(has_internal_duplicate(&dupes));
    }

    #[test]
    fn helper_has_internal_duplicate_clean_returns_false() {
        let clean = vec![h('0'), h('1'), h('2')];
        assert!(!has_internal_duplicate(&clean));
    }

    #[test]
    fn helper_has_internal_duplicate_empty_returns_false() {
        let empty: Vec<String> = vec![];
        assert!(!has_internal_duplicate(&empty));
    }
}
