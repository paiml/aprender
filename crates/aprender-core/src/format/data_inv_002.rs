// SHIP-TWO-001 MODEL-2 — `dataset-thestack-python-v1` (C-DATA-THESTACK-PYTHON)
// algorithm-level PARTIAL discharge for INV-DATA-002.
//
// Contract: `contracts/dataset-thestack-python-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pipeline (§26.2), AC-SHIP2-002.
//
// ## What INV-DATA-002 says
//
//   description: No file in any output shard matches any
//                `pii_scrub.patterns` regex (applied to full file
//                content, not just header).
//   falsifier:   Stream-scan all shards; regex-match every record
//                content against every pattern. Any match → FAIL.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a stream-scan that produces three counters
// (`scanned_files`, `pattern_count`, `total_pattern_matches`), Pass iff:
//
//   scanned_files > 0 AND
//   pattern_count > 0 AND
//   total_pattern_matches == 0
//
// AND `total_pattern_matches <= scanned_files * pattern_count`
// (counter sanity, computed via `checked_mul` to prevent silent
// wraparound at corpora near `u64::MAX`). The contract falsifier
// ("Any match → FAIL") admits no tolerance band: one PII match
// poisons the entire MODEL-2 corpus from a privacy/legal-sovereignty
// standpoint, so `total_pattern_matches > 0` is dispositive.

/// Binary verdict for `INV-DATA-002`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataInv002Verdict {
    /// Stream-scan visited at least one file with at least one
    /// configured pattern, and zero (file × pattern) matches were
    /// observed.
    Pass,
    /// One or more of:
    /// - `scanned_files == 0` (caller error — vacuous Pass refused).
    /// - `pattern_count == 0` (caller error — empty PII pattern set
    ///   would make the scan trivially pass; refuse explicitly).
    /// - `total_pattern_matches > 0` (one match is enough).
    /// - `total_pattern_matches > scanned_files * pattern_count`
    ///   (counter corruption — partition violation).
    /// - Multiplication overflow on `scanned_files * pattern_count`
    ///   (treated as a corpus-too-large caller error).
    Fail,
}

/// Pure verdict function for `INV-DATA-002`.
///
/// Inputs:
/// - `scanned_files`: total file count visited by the stream-scan.
/// - `pattern_count`: number of PII regex patterns applied to each
///   file's content.
/// - `total_pattern_matches`: total count of (file × pattern) matches
///   observed (one file matching two patterns counts as 2).
///
/// Pass iff:
/// 1. `scanned_files > 0`,
/// 2. `pattern_count > 0`,
/// 3. `total_pattern_matches == 0`,
/// 4. `total_pattern_matches <= scanned_files.checked_mul(pattern_count)`
///    (counter sanity; overflow → Fail).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 4M files × 32 patterns, zero matches — `Pass`:
/// ```
/// use aprender::format::data_inv_002::{
///     verdict_from_pii_scrub_scan, DataInv002Verdict,
/// };
/// let v = verdict_from_pii_scrub_scan(4_000_000, 32, 0);
/// assert_eq!(v, DataInv002Verdict::Pass);
/// ```
///
/// One PII match in millions — `Fail` (one is enough):
/// ```
/// use aprender::format::data_inv_002::{
///     verdict_from_pii_scrub_scan, DataInv002Verdict,
/// };
/// let v = verdict_from_pii_scrub_scan(4_000_000, 32, 1);
/// assert_eq!(v, DataInv002Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_pii_scrub_scan(
    scanned_files: u64,
    pattern_count: u64,
    total_pattern_matches: u64,
) -> DataInv002Verdict {
    if scanned_files == 0 || pattern_count == 0 {
        return DataInv002Verdict::Fail;
    }
    let max_possible = match scanned_files.checked_mul(pattern_count) {
        Some(v) => v,
        None => return DataInv002Verdict::Fail,
    };
    if total_pattern_matches > max_possible {
        return DataInv002Verdict::Fail;
    }
    if total_pattern_matches == 0 {
        DataInv002Verdict::Pass
    } else {
        DataInv002Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — clean corpora at canonical sizes/pattern counts.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_4m_files_32_patterns_zero_matches() {
        let v = verdict_from_pii_scrub_scan(4_000_000, 32, 0);
        assert_eq!(v, DataInv002Verdict::Pass);
    }

    #[test]
    fn pass_one_file_one_pattern_zero_matches() {
        let v = verdict_from_pii_scrub_scan(1, 1, 0);
        assert_eq!(v, DataInv002Verdict::Pass);
    }

    #[test]
    fn pass_realistic_csn_python_size() {
        let v = verdict_from_pii_scrub_scan(455_000, 16, 0);
        assert_eq!(v, DataInv002Verdict::Pass);
    }

    #[test]
    fn pass_realistic_codeparrot_size() {
        let v = verdict_from_pii_scrub_scan(8_000_000, 32, 0);
        assert_eq!(v, DataInv002Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — at least one match fails the gate (no tolerance).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_match_in_million_files() {
        let v = verdict_from_pii_scrub_scan(1_000_000, 32, 1);
        assert_eq!(
            v,
            DataInv002Verdict::Fail,
            "one PII match must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_a_handful_of_matches() {
        let v = verdict_from_pii_scrub_scan(1_000_000, 32, 7);
        assert_eq!(v, DataInv002Verdict::Fail);
    }

    #[test]
    fn fail_match_per_file_per_pattern() {
        // Worst case: every (file, pattern) pair matches.
        let v = verdict_from_pii_scrub_scan(1_000, 32, 32_000);
        assert_eq!(v, DataInv002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — caller errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_scanned_files() {
        // Vacuous Pass refused.
        let v = verdict_from_pii_scrub_scan(0, 32, 0);
        assert_eq!(
            v,
            DataInv002Verdict::Fail,
            "zero scanned files must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_zero_pattern_count() {
        // Empty PII pattern set would trivially pass — refuse.
        let v = verdict_from_pii_scrub_scan(1_000_000, 0, 0);
        assert_eq!(
            v,
            DataInv002Verdict::Fail,
            "zero patterns must Fail (empty PII set is a defect)"
        );
    }

    #[test]
    fn fail_both_zero() {
        let v = verdict_from_pii_scrub_scan(0, 0, 0);
        assert_eq!(v, DataInv002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — counter / partition violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_matches_exceed_max_possible() {
        // 100 files × 4 patterns = 400 max possible matches; 401 must Fail.
        let v = verdict_from_pii_scrub_scan(100, 4, 401);
        assert_eq!(
            v,
            DataInv002Verdict::Fail,
            "matches > files * patterns must Fail"
        );
    }

    #[test]
    fn fail_matches_huge_with_small_corpus() {
        // Counter rollover style: u64::MAX matches in 100-file scan.
        let v = verdict_from_pii_scrub_scan(100, 4, u64::MAX);
        assert_eq!(v, DataInv002Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Overflow protection — checked_mul on (files * patterns).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_files_times_patterns_overflow() {
        // files * patterns overflows u64.
        let huge = u64::MAX / 2 + 1;
        let v = verdict_from_pii_scrub_scan(huge, 4, 0);
        assert_eq!(
            v,
            DataInv002Verdict::Fail,
            "overflow in files * patterns must Fail (not silently wrap)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — match-count sweep at fixed dims.
    // -------------------------------------------------------------------------
    #[test]
    fn match_count_sweep_at_fixed_dims() {
        let files = 1000_u64;
        let patterns = 4_u64;
        let max_possible = files * patterns; // 4000
        let probes: Vec<(u64, DataInv002Verdict)> = vec![
            (0, DataInv002Verdict::Pass),
            (1, DataInv002Verdict::Fail),
            (10, DataInv002Verdict::Fail),
            (1000, DataInv002Verdict::Fail),
            (max_possible, DataInv002Verdict::Fail),
            (max_possible + 1, DataInv002Verdict::Fail),
        ];
        for (matches, expected) in probes {
            let v = verdict_from_pii_scrub_scan(files, patterns, matches);
            assert_eq!(
                v, expected,
                "files={files} patterns={patterns} matches={matches} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — zero-tolerance property at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_matches_is_exactly_zero() {
        let pattern_count = 32_u64;
        for scanned in [1_u64, 100, 10_000, 1_000_000] {
            let v_pass = verdict_from_pii_scrub_scan(scanned, pattern_count, 0);
            assert_eq!(v_pass, DataInv002Verdict::Pass, "scanned={scanned}");

            let v_fail = verdict_from_pii_scrub_scan(scanned, pattern_count, 1);
            assert_eq!(
                v_fail,
                DataInv002Verdict::Fail,
                "scanned={scanned} with one PII match"
            );
        }
    }
}
