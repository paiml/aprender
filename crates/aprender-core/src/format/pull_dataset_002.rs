// SHIP-TWO-001 MODEL-2 — `apr-cli-pull-dataset-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-PULL-DATASET-002.
//
// Contract: `contracts/apr-cli-pull-dataset-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pull (P1.1).
//
// ## What FALSIFY-APR-PULL-DATASET-002 says
//
//   rule: include glob filters correctly
//   prediction: "`apr pull dataset <repo> --include 'data/train-00000-of-00880.parquet'
//                --output /tmp/test-pull` produces exactly 1 parquet file"
//   if_fails:   "include glob semantics broken; risks pulling entire 314 GB repo"
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`matched_count`, `expected_count`), Pass iff:
//
//   matched_count == expected_count AND expected_count > 0
//
// Strict equality matches the contract's "exactly 1 parquet file"
// wording — any drift (over-match because glob is too broad,
// under-match because no-match silently returns 0) trips the gate.
// `expected_count > 0` refuses the degenerate case where a test
// expects zero files; that's the no-match-fail-fast scenario which
// belongs to FALSIFY-APR-PULL-DATASET-003, not this gate.

/// Binary verdict for `FALSIFY-APR-PULL-DATASET-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDataset002Verdict {
    /// Caller's expected_count is positive AND `matched_count ==
    /// expected_count` exactly.
    Pass,
    /// One or more of:
    /// - `expected_count == 0` (caller error — wrong gate; use
    ///   FALSIFY-003 for no-match-fail-fast scenarios).
    /// - `matched_count != expected_count` (glob over-matched or
    ///   under-matched; risks pulling 314 GB OR silent no-op).
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-PULL-DATASET-002`.
///
/// Inputs:
/// - `matched_count`: number of files actually downloaded by
///   `apr pull dataset <repo> --include <glob>`.
/// - `expected_count`: number of files the include glob ought to
///   match (the contract's example uses exactly 1).
///
/// Pass iff:
/// 1. `expected_count > 0`,
/// 2. `matched_count == expected_count`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Glob expects 1 parquet, scanner found 1 — `Pass`:
/// ```
/// use aprender::format::pull_dataset_002::{
///     verdict_from_include_glob_match_count, PullDataset002Verdict,
/// };
/// let v = verdict_from_include_glob_match_count(1, 1);
/// assert_eq!(v, PullDataset002Verdict::Pass);
/// ```
///
/// Glob over-matched (1 expected, 5 found — pulled too much) — `Fail`:
/// ```
/// use aprender::format::pull_dataset_002::{
///     verdict_from_include_glob_match_count, PullDataset002Verdict,
/// };
/// let v = verdict_from_include_glob_match_count(5, 1);
/// assert_eq!(v, PullDataset002Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_include_glob_match_count(
    matched_count: u64,
    expected_count: u64,
) -> PullDataset002Verdict {
    if expected_count == 0 {
        return PullDataset002Verdict::Fail;
    }
    if matched_count == expected_count {
        PullDataset002Verdict::Pass
    } else {
        PullDataset002Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — exact match at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_exact_one_file() {
        // Contract example: glob 'data/train-00000-of-00880.parquet' → 1.
        let v = verdict_from_include_glob_match_count(1, 1);
        assert_eq!(v, PullDataset002Verdict::Pass);
    }

    #[test]
    fn pass_exact_multiple_files() {
        // Multi-file glob: 'data/train-0000*-of-00880.parquet' → 10.
        let v = verdict_from_include_glob_match_count(10, 10);
        assert_eq!(v, PullDataset002Verdict::Pass);
    }

    #[test]
    fn pass_full_repo_match() {
        // Full-repo include glob 'data/*.parquet' → 880.
        let v = verdict_from_include_glob_match_count(880, 880);
        assert_eq!(v, PullDataset002Verdict::Pass);
    }

    #[test]
    fn pass_huge_match() {
        // Sanity at large scale.
        let v = verdict_from_include_glob_match_count(1_000_000, 1_000_000);
        assert_eq!(v, PullDataset002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — over-match (pulled too much / 314 GB risk).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_off_by_one_high() {
        let v = verdict_from_include_glob_match_count(2, 1);
        assert_eq!(
            v,
            PullDataset002Verdict::Fail,
            "+1 over-match must Fail (glob too broad)"
        );
    }

    #[test]
    fn fail_glob_pulled_full_repo_when_expected_one() {
        // Worst case: glob matched all 880 files when expected 1.
        let v = verdict_from_include_glob_match_count(880, 1);
        assert_eq!(v, PullDataset002Verdict::Fail);
    }

    #[test]
    fn fail_glob_5x_overmatch() {
        let v = verdict_from_include_glob_match_count(50, 10);
        assert_eq!(v, PullDataset002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — under-match (silent no-op risk).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_off_by_one_low() {
        let v = verdict_from_include_glob_match_count(0, 1);
        assert_eq!(
            v,
            PullDataset002Verdict::Fail,
            "0 found when 1 expected must Fail (silent no-op)"
        );
    }

    #[test]
    fn fail_under_match_partial() {
        // Expected 100 files, only 87 came back.
        let v = verdict_from_include_glob_match_count(87, 100);
        assert_eq!(v, PullDataset002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — caller errors (zero expected).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_expected_zero_matched() {
        // Caller-error gate: expected==0 belongs to FALSIFY-003,
        // not -002. Refuse here.
        let v = verdict_from_include_glob_match_count(0, 0);
        assert_eq!(
            v,
            PullDataset002Verdict::Fail,
            "expected==0 must Fail (use FALSIFY-003 for no-match scenarios)"
        );
    }

    #[test]
    fn fail_zero_expected_with_matches() {
        // Pathological: expected 0, scanner returned 5.
        let v = verdict_from_include_glob_match_count(5, 0);
        assert_eq!(v, PullDataset002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — matched count around expected=10.
    // -------------------------------------------------------------------------
    #[test]
    fn matched_count_sweep_at_fixed_expected() {
        let expected = 10_u64;
        let probes: Vec<(u64, PullDataset002Verdict)> = vec![
            (0, PullDataset002Verdict::Fail),
            (1, PullDataset002Verdict::Fail),
            (5, PullDataset002Verdict::Fail),
            (9, PullDataset002Verdict::Fail),
            (10, PullDataset002Verdict::Pass),
            (11, PullDataset002Verdict::Fail),
            (100, PullDataset002Verdict::Fail),
            (u64::MAX, PullDataset002Verdict::Fail),
        ];
        for (matched, expected_verdict) in probes {
            let v = verdict_from_include_glob_match_count(matched, expected);
            assert_eq!(
                v, expected_verdict,
                "matched={matched} expected={expected} verdict {expected_verdict:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Symmetry / equality property.
    // -------------------------------------------------------------------------
    #[test]
    fn verdict_pass_iff_equal_at_canonical_sizes() {
        for size in [1_u64, 10, 100, 1_000, 10_000, 1_000_000] {
            let v_pass = verdict_from_include_glob_match_count(size, size);
            assert_eq!(v_pass, PullDataset002Verdict::Pass, "size={size}");

            let v_fail_high = verdict_from_include_glob_match_count(size + 1, size);
            assert_eq!(v_fail_high, PullDataset002Verdict::Fail, "size+1");

            if size > 0 {
                let v_fail_low = verdict_from_include_glob_match_count(size - 1, size);
                assert_eq!(v_fail_low, PullDataset002Verdict::Fail, "size-1");
            }
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — codeparrot/github-code-clean shard counts.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_codeparrot_single_shard() {
        // Memory: P1.4 codeparrot pull. Single training shard.
        let v = verdict_from_include_glob_match_count(1, 1);
        assert_eq!(v, PullDataset002Verdict::Pass);
    }

    #[test]
    fn pass_codeparrot_python_subset() {
        // Hypothetical: 30 shards for the Python+permissive subset.
        let v = verdict_from_include_glob_match_count(30, 30);
        assert_eq!(v, PullDataset002Verdict::Pass);
    }

    #[test]
    fn fail_codeparrot_full_repo_when_subset_expected() {
        // Worst-case 314 GB pull: expected 30 (one Python subset),
        // got 880 (full repo).
        let v = verdict_from_include_glob_match_count(880, 30);
        assert_eq!(
            v,
            PullDataset002Verdict::Fail,
            "expected 30 got 880 — full-repo pull must Fail"
        );
    }
}
