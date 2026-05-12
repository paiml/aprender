// SHIP-TWO-001 MODEL-2 — `apr-cli-pull-dataset-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-PULL-DATASET-004.
//
// Contract: `contracts/apr-cli-pull-dataset-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pull (P1.1).
//
// ## What FALSIFY-APR-PULL-DATASET-004 says
//
//   rule: license allowlist drops disallowed rows
//   prediction: After --license-allowlist mit,apache-2.0, output
//               parquet contains only rows where license ∈ {mit, apache-2.0}
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a post-filter scan of the output parquet
// producing (`output_rows`, `disallowed_rows_in_output`,
// `allowlist_size`), Pass iff:
//
//   output_rows > 0 AND
//   allowlist_size > 0 AND
//   disallowed_rows_in_output == 0 AND
//   disallowed_rows_in_output <= output_rows
//
// Zero-tolerance on disallowed rows: a single row with a license
// outside the allowlist in the output parquet means the filter
// failed. A non-empty allowlist is required (an empty allowlist
// would silently pass an empty-output check); a non-empty output
// is required (an empty output would silently pass — but means
// the filter rejected everything, which is FALSIFY-003's domain).

/// Binary verdict for `FALSIFY-APR-PULL-DATASET-004`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PullDataset004Verdict {
    /// Output parquet is non-empty, allowlist is non-empty, AND
    /// every output row's license is in the allowlist.
    Pass,
    /// One or more of:
    /// - `output_rows == 0` (caller error — empty output; use
    ///   FALSIFY-003 for no-match scenarios).
    /// - `allowlist_size == 0` (caller error — empty allowlist
    ///   would trivially reject everything).
    /// - `disallowed_rows_in_output > 0` (filter failed —
    ///   disallowed-license row leaked through).
    /// - `disallowed_rows_in_output > output_rows` (counter
    ///   corruption — partition violation).
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-PULL-DATASET-004`.
///
/// Inputs:
/// - `output_rows`: total rows in the post-filter output parquet.
/// - `disallowed_rows_in_output`: count of those rows whose
///   `license` field is NOT in the configured allowlist.
/// - `allowlist_size`: number of distinct licenses in the
///   `--license-allowlist` flag.
///
/// Pass iff:
/// 1. `output_rows > 0`,
/// 2. `allowlist_size > 0`,
/// 3. `disallowed_rows_in_output == 0`,
/// 4. `disallowed_rows_in_output <= output_rows` (counter sanity).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 1M output rows, allowlist {mit, apache-2.0}, zero disallowed —
/// `Pass`:
/// ```
/// use aprender::format::pull_dataset_004::{
///     verdict_from_license_allowlist_filter, PullDataset004Verdict,
/// };
/// let v = verdict_from_license_allowlist_filter(1_000_000, 0, 2);
/// assert_eq!(v, PullDataset004Verdict::Pass);
/// ```
///
/// One disallowed row leaked through — `Fail`:
/// ```
/// use aprender::format::pull_dataset_004::{
///     verdict_from_license_allowlist_filter, PullDataset004Verdict,
/// };
/// let v = verdict_from_license_allowlist_filter(1_000_000, 1, 2);
/// assert_eq!(v, PullDataset004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_license_allowlist_filter(
    output_rows: u64,
    disallowed_rows_in_output: u64,
    allowlist_size: u64,
) -> PullDataset004Verdict {
    if output_rows == 0 || allowlist_size == 0 {
        return PullDataset004Verdict::Fail;
    }
    if disallowed_rows_in_output > output_rows {
        return PullDataset004Verdict::Fail;
    }
    if disallowed_rows_in_output == 0 {
        PullDataset004Verdict::Pass
    } else {
        PullDataset004Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — clean filter, canonical scales.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_canonical_mit_apache_allowlist() {
        // Contract example: --license-allowlist mit,apache-2.0
        // (size 2), 1M output rows, all whitelisted.
        let v = verdict_from_license_allowlist_filter(1_000_000, 0, 2);
        assert_eq!(v, PullDataset004Verdict::Pass);
    }

    #[test]
    fn pass_single_license_allowlist() {
        // Strict policy: only Apache-2.0.
        let v = verdict_from_license_allowlist_filter(500_000, 0, 1);
        assert_eq!(v, PullDataset004Verdict::Pass);
    }

    #[test]
    fn pass_large_allowlist() {
        // Permissive policy: many licenses.
        let v = verdict_from_license_allowlist_filter(2_000_000, 0, 8);
        assert_eq!(v, PullDataset004Verdict::Pass);
    }

    #[test]
    fn pass_minimal_one_row() {
        let v = verdict_from_license_allowlist_filter(1, 0, 1);
        assert_eq!(v, PullDataset004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — disallowed rows leaked through (zero-tolerance).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_disallowed_in_million() {
        let v = verdict_from_license_allowlist_filter(1_000_000, 1, 2);
        assert_eq!(
            v,
            PullDataset004Verdict::Fail,
            "one disallowed row must Fail (no tolerance — license sovereignty)"
        );
    }

    #[test]
    fn fail_handful_of_disallowed() {
        let v = verdict_from_license_allowlist_filter(1_000_000, 7, 2);
        assert_eq!(v, PullDataset004Verdict::Fail);
    }

    #[test]
    fn fail_half_disallowed() {
        // Catastrophic: half the output is disallowed.
        let v = verdict_from_license_allowlist_filter(1_000_000, 500_000, 2);
        assert_eq!(v, PullDataset004Verdict::Fail);
    }

    #[test]
    fn fail_all_disallowed() {
        // Filter inverted entirely.
        let v = verdict_from_license_allowlist_filter(1_000_000, 1_000_000, 2);
        assert_eq!(v, PullDataset004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — caller errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_output_rows() {
        // Empty output → no-match-fail-fast belongs to FALSIFY-003.
        let v = verdict_from_license_allowlist_filter(0, 0, 2);
        assert_eq!(
            v,
            PullDataset004Verdict::Fail,
            "zero output rows must Fail (use FALSIFY-003 for that case)"
        );
    }

    #[test]
    fn fail_zero_allowlist_size() {
        // Empty allowlist would trivially reject everything; refuse
        // to give it a Pass even with zero disallowed.
        let v = verdict_from_license_allowlist_filter(1_000_000, 0, 0);
        assert_eq!(
            v,
            PullDataset004Verdict::Fail,
            "empty allowlist must Fail (config error)"
        );
    }

    #[test]
    fn fail_zero_output_zero_allowlist() {
        let v = verdict_from_license_allowlist_filter(0, 0, 0);
        assert_eq!(v, PullDataset004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — counter / partition violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_disallowed_exceeds_output() {
        // Counter corruption: disallowed > output.
        let v = verdict_from_license_allowlist_filter(100, 101, 2);
        assert_eq!(
            v,
            PullDataset004Verdict::Fail,
            "disallowed > output must Fail (partition violation)"
        );
    }

    #[test]
    fn fail_huge_disallowed_with_smaller_output() {
        let v = verdict_from_license_allowlist_filter(100, u64::MAX, 2);
        assert_eq!(v, PullDataset004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep — disallowed count from 0 to output+1.
    // -------------------------------------------------------------------------
    #[test]
    fn disallowed_sweep_at_fixed_output() {
        let output = 1000_u64;
        let allowlist = 2_u64;
        let probes: Vec<(u64, PullDataset004Verdict)> = vec![
            (0, PullDataset004Verdict::Pass),
            (1, PullDataset004Verdict::Fail),
            (2, PullDataset004Verdict::Fail),
            (10, PullDataset004Verdict::Fail),
            (500, PullDataset004Verdict::Fail),
            (999, PullDataset004Verdict::Fail),
            (1000, PullDataset004Verdict::Fail),
            (1001, PullDataset004Verdict::Fail), // partition violation
        ];
        for (disallowed, expected) in probes {
            let v = verdict_from_license_allowlist_filter(output, disallowed, allowlist);
            assert_eq!(
                v, expected,
                "output={output} disallowed={disallowed} allowlist={allowlist} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Domain — zero-tolerance property at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_disallowed_is_exactly_zero() {
        let allowlist = 2_u64;
        for output in [1_u64, 100, 10_000, 1_000_000] {
            let v_pass = verdict_from_license_allowlist_filter(output, 0, allowlist);
            assert_eq!(v_pass, PullDataset004Verdict::Pass, "output={output}");

            let v_fail = verdict_from_license_allowlist_filter(output, 1, allowlist);
            assert_eq!(
                v_fail,
                PullDataset004Verdict::Fail,
                "output={output} with one disallowed"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — codeparrot apache-2.0/mit allowlist.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_codeparrot_python_apache_only() {
        // codeparrot/github-code-clean Python permissive, --license-allowlist apache-2.0.
        // ~700k rows post-filter, all Apache-2.0.
        let v = verdict_from_license_allowlist_filter(700_000, 0, 1);
        assert_eq!(v, PullDataset004Verdict::Pass);
    }

    #[test]
    fn fail_codeparrot_gpl_leakage() {
        // Worst case: 100 GPL rows leaked into Apache-2.0-only pull.
        // License sovereignty broken.
        let v = verdict_from_license_allowlist_filter(700_000, 100, 1);
        assert_eq!(
            v,
            PullDataset004Verdict::Fail,
            "GPL leakage in Apache-only pull must Fail"
        );
    }
}
