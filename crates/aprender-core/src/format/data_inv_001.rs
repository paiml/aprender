// SHIP-TWO-001 MODEL-2 — `dataset-thestack-python-v1` (C-DATA-THESTACK-PYTHON)
// algorithm-level PARTIAL discharge for INV-DATA-001.
//
// Contract: `contracts/dataset-thestack-python-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pipeline (§26.2), AC-SHIP2-002.
//
// ## What INV-DATA-001 says
//
//   description: Every file in every output shard has an SPDX license
//                identifier present in `license_whitelist.spdx_allow`.
//                No file with an unknown or disallowed license reaches
//                a shard.
//   falsifier:   Stream-scan all shards; for each record, check
//                `metadata.license` against the whitelist. Any
//                non-whitelist license → FAIL.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a stream-scan that produces
// (`scanned_files`, `non_whitelisted_count`), Pass iff:
//
//   scanned_files > 0 AND non_whitelisted_count == 0
//
// AND `non_whitelisted_count <= scanned_files` (sanity check; counter
// arithmetic must respect the partition). Any non-zero
// non-whitelisted count → Fail (one disallowed file is enough). A
// zero-files scan → Fail (caller error: nothing was actually
// validated, and we refuse to vacuously pass an empty corpus).

/// Binary verdict for `INV-DATA-001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataInv001Verdict {
    /// Stream-scan visited at least one file AND every visited file
    /// had a whitelisted SPDX license.
    Pass,
    /// One or more of:
    /// - `scanned_files == 0` (caller error — scan produced no work).
    /// - `non_whitelisted_count > 0` (at least one disallowed license).
    /// - `non_whitelisted_count > scanned_files` (counter corruption —
    ///   the non-whitelisted partition cannot exceed the total).
    Fail,
}

/// Pure verdict function for `INV-DATA-001`.
///
/// Inputs:
/// - `scanned_files`: total file count visited by the stream-scan.
/// - `non_whitelisted_count`: number of those files whose
///   `metadata.license` was not in `license_whitelist.spdx_allow`.
///
/// Pass iff:
/// 1. `scanned_files > 0`,
/// 2. `non_whitelisted_count == 0`,
/// 3. `non_whitelisted_count <= scanned_files` (counter sanity).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 565M-token corpus, 4M files, all whitelisted — `Pass`:
/// ```
/// use aprender::format::data_inv_001::{
///     verdict_from_license_whitelist_scan, DataInv001Verdict,
/// };
/// let v = verdict_from_license_whitelist_scan(4_000_000, 0);
/// assert_eq!(v, DataInv001Verdict::Pass);
/// ```
///
/// One disallowed file in 4M — `Fail` (one is enough):
/// ```
/// use aprender::format::data_inv_001::{
///     verdict_from_license_whitelist_scan, DataInv001Verdict,
/// };
/// let v = verdict_from_license_whitelist_scan(4_000_000, 1);
/// assert_eq!(v, DataInv001Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_license_whitelist_scan(
    scanned_files: u64,
    non_whitelisted_count: u64,
) -> DataInv001Verdict {
    if scanned_files == 0 {
        return DataInv001Verdict::Fail;
    }
    if non_whitelisted_count > scanned_files {
        return DataInv001Verdict::Fail;
    }
    if non_whitelisted_count == 0 {
        DataInv001Verdict::Pass
    } else {
        DataInv001Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — clean corpora at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_4m_files_zero_disallowed() {
        let v = verdict_from_license_whitelist_scan(4_000_000, 0);
        assert_eq!(v, DataInv001Verdict::Pass);
    }

    #[test]
    fn pass_one_file_zero_disallowed() {
        let v = verdict_from_license_whitelist_scan(1, 0);
        assert_eq!(v, DataInv001Verdict::Pass);
    }

    #[test]
    fn pass_realistic_csn_python_size() {
        // CSN-Python: ~455k docs after filtering.
        let v = verdict_from_license_whitelist_scan(455_000, 0);
        assert_eq!(v, DataInv001Verdict::Pass);
    }

    #[test]
    fn pass_realistic_codeparrot_size() {
        // codeparrot/github-code-clean Python permissive: millions of files.
        let v = verdict_from_license_whitelist_scan(8_000_000, 0);
        assert_eq!(v, DataInv001Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — at least one disallowed file fails the gate.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_disallowed_in_million() {
        // Strict invariant: one disallowed license is enough to fail
        // the entire corpus. No tolerance band.
        let v = verdict_from_license_whitelist_scan(1_000_000, 1);
        assert_eq!(
            v,
            DataInv001Verdict::Fail,
            "one disallowed license must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_half_disallowed() {
        let v = verdict_from_license_whitelist_scan(1_000_000, 500_000);
        assert_eq!(v, DataInv001Verdict::Fail);
    }

    #[test]
    fn fail_all_disallowed() {
        let v = verdict_from_license_whitelist_scan(1_000_000, 1_000_000);
        assert_eq!(v, DataInv001Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — caller / counter errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_scanned_files() {
        // Vacuous Pass refused: scanning zero files validates nothing.
        let v = verdict_from_license_whitelist_scan(0, 0);
        assert_eq!(
            v,
            DataInv001Verdict::Fail,
            "zero scanned files must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_zero_scanned_with_nonzero_disallowed() {
        // Counter corruption: non-whitelisted > scanned.
        let v = verdict_from_license_whitelist_scan(0, 5);
        assert_eq!(v, DataInv001Verdict::Fail);
    }

    #[test]
    fn fail_non_whitelisted_exceeds_scanned() {
        // Counter corruption — partition violation.
        let v = verdict_from_license_whitelist_scan(100, 101);
        assert_eq!(
            v,
            DataInv001Verdict::Fail,
            "non_whitelisted > scanned must Fail (partition violation)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Boundary sweep — disallowed count from 0 to scanned.
    // -------------------------------------------------------------------------
    #[test]
    fn disallowed_sweep_at_fixed_scanned() {
        let scanned = 1000_u64;
        let probes: Vec<(u64, DataInv001Verdict)> = vec![
            (0, DataInv001Verdict::Pass),
            (1, DataInv001Verdict::Fail),
            (2, DataInv001Verdict::Fail),
            (10, DataInv001Verdict::Fail),
            (500, DataInv001Verdict::Fail),
            (999, DataInv001Verdict::Fail),
            (1000, DataInv001Verdict::Fail),
            (1001, DataInv001Verdict::Fail), // partition violation
        ];
        for (disallowed, expected) in probes {
            let v = verdict_from_license_whitelist_scan(scanned, disallowed);
            assert_eq!(
                v, expected,
                "scanned={scanned} disallowed={disallowed} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 5: Domain — strict zero-tolerance for license violations.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_disallowed_is_exactly_zero() {
        // Property: at any non-zero scanned count, Pass iff
        // disallowed == 0. No fractional/percentage tolerance.
        for scanned in [1_u64, 10, 100, 1_000, 10_000, 100_000, 1_000_000] {
            let pass_v = verdict_from_license_whitelist_scan(scanned, 0);
            assert_eq!(pass_v, DataInv001Verdict::Pass, "scanned={scanned}");

            let fail_v = verdict_from_license_whitelist_scan(scanned, 1);
            assert_eq!(
                fail_v,
                DataInv001Verdict::Fail,
                "scanned={scanned} with one violation"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic scale — u64::MAX edge cases (counter overflow
    // domain).
    // -------------------------------------------------------------------------
    #[test]
    fn pass_u64_max_scanned_zero_disallowed() {
        // Implausible but well-formed: huge scan, zero violations.
        let v = verdict_from_license_whitelist_scan(u64::MAX, 0);
        assert_eq!(v, DataInv001Verdict::Pass);
    }

    #[test]
    fn fail_u64_max_disallowed_with_smaller_scanned() {
        // Counter-rollover-style error: disallowed = u64::MAX, scanned
        // = 100. Partition violated → Fail.
        let v = verdict_from_license_whitelist_scan(100, u64::MAX);
        assert_eq!(v, DataInv001Verdict::Fail);
    }
}
