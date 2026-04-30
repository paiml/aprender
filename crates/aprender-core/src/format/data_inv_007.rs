// SHIP-TWO-001 MODEL-2 — `dataset-thestack-python-v1` (C-DATA-THESTACK-PYTHON)
// algorithm-level PARTIAL discharge for INV-DATA-007.
//
// Contract: `contracts/dataset-thestack-python-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pipeline (§26.2), AC-SHIP2-002.
//
// ## What INV-DATA-007 says
//
//   description: For every file content byte in every shard, decoding
//                as UTF-8 succeeds AND the decoded string survives
//                NFC normalization round-trip
//                (nfc(bytes) == nfc(nfc(bytes))).
//   falsifier:   Stream-scan all shards; attempt UTF-8 decode + NFC
//                round-trip. Any failure → FAIL. Reason: MODEL-2
//                tokenizer (C-TOK-BPE-001 INV-TOK-003) mandates NFC;
//                feeding non-NFC bytes at pretrain corrupts the
//                vocabulary distribution.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a stream-scan that produces three counters
// (`scanned_files`, `utf8_decode_failures`, `nfc_roundtrip_failures`),
// Pass iff:
//
//   scanned_files > 0 AND
//   utf8_decode_failures == 0 AND
//   nfc_roundtrip_failures == 0 AND
//   (utf8_decode_failures + nfc_roundtrip_failures) <= scanned_files
//
// AND the partition arithmetic does not overflow on `checked_add`.
// This composes two zero-tolerance invariants into one verdict so
// drift in either dimension (a bad UTF-8 byte sequence OR a non-NFC
// pre-existing string) trips the gate. Contract falsifier admits no
// tolerance band.

/// Binary verdict for `INV-DATA-007`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataInv007Verdict {
    /// Stream-scan visited at least one file AND every visited file
    /// (1) decoded as UTF-8 successfully AND (2) survived NFC
    /// round-trip without modification.
    Pass,
    /// One or more of:
    /// - `scanned_files == 0` (caller error — vacuous Pass refused).
    /// - `utf8_decode_failures > 0` (one is enough).
    /// - `nfc_roundtrip_failures > 0` (one is enough).
    /// - `utf8_decode_failures + nfc_roundtrip_failures > scanned_files`
    ///   (counter corruption — partition violation).
    /// - Overflow in `utf8 + nfc` `checked_add`.
    Fail,
}

/// Pure verdict function for `INV-DATA-007`.
///
/// Inputs:
/// - `scanned_files`: total file count visited by the stream-scan.
/// - `utf8_decode_failures`: number of files where UTF-8 decode
///   failed.
/// - `nfc_roundtrip_failures`: number of files where the decoded
///   string did NOT equal `nfc(nfc(bytes))`.
///
/// Pass iff:
/// 1. `scanned_files > 0`,
/// 2. `utf8_decode_failures == 0`,
/// 3. `nfc_roundtrip_failures == 0`,
/// 4. `utf8_decode_failures.checked_add(nfc_roundtrip_failures) <= scanned_files`
///    (counter sanity; overflow → Fail).
///
/// Otherwise `Fail`.
///
/// Note: a single file can fail both UTF-8 and NFC; the counters are
/// not required to be disjoint, but their sum cannot exceed the total
/// scanned. Invalid-UTF-8 files cannot meaningfully contribute to
/// `nfc_roundtrip_failures` in a real scanner since NFC operates on
/// decoded strings — but the verdict is robust to either ordering of
/// detection.
///
/// # Examples
///
/// 565M-token corpus, 4M files, all clean — `Pass`:
/// ```
/// use aprender::format::data_inv_007::{
///     verdict_from_utf8_nfc_scan, DataInv007Verdict,
/// };
/// let v = verdict_from_utf8_nfc_scan(4_000_000, 0, 0);
/// assert_eq!(v, DataInv007Verdict::Pass);
/// ```
///
/// One non-NFC string in 4M — `Fail` (one is enough):
/// ```
/// use aprender::format::data_inv_007::{
///     verdict_from_utf8_nfc_scan, DataInv007Verdict,
/// };
/// let v = verdict_from_utf8_nfc_scan(4_000_000, 0, 1);
/// assert_eq!(v, DataInv007Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_utf8_nfc_scan(
    scanned_files: u64,
    utf8_decode_failures: u64,
    nfc_roundtrip_failures: u64,
) -> DataInv007Verdict {
    if scanned_files == 0 {
        return DataInv007Verdict::Fail;
    }
    let combined = match utf8_decode_failures.checked_add(nfc_roundtrip_failures) {
        Some(v) => v,
        None => return DataInv007Verdict::Fail,
    };
    if combined > scanned_files {
        return DataInv007Verdict::Fail;
    }
    if utf8_decode_failures == 0 && nfc_roundtrip_failures == 0 {
        DataInv007Verdict::Pass
    } else {
        DataInv007Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — clean corpora at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_4m_files_zero_failures() {
        let v = verdict_from_utf8_nfc_scan(4_000_000, 0, 0);
        assert_eq!(v, DataInv007Verdict::Pass);
    }

    #[test]
    fn pass_one_file_zero_failures() {
        let v = verdict_from_utf8_nfc_scan(1, 0, 0);
        assert_eq!(v, DataInv007Verdict::Pass);
    }

    #[test]
    fn pass_realistic_csn_python_size() {
        // CSN-Python: ~455k docs after filtering.
        let v = verdict_from_utf8_nfc_scan(455_000, 0, 0);
        assert_eq!(v, DataInv007Verdict::Pass);
    }

    #[test]
    fn pass_realistic_codeparrot_size() {
        let v = verdict_from_utf8_nfc_scan(8_000_000, 0, 0);
        assert_eq!(v, DataInv007Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — UTF-8 decode failures (one is enough).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_utf8_failure_in_million() {
        let v = verdict_from_utf8_nfc_scan(1_000_000, 1, 0);
        assert_eq!(
            v,
            DataInv007Verdict::Fail,
            "one utf8 failure must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_all_utf8_failures() {
        let v = verdict_from_utf8_nfc_scan(1_000_000, 1_000_000, 0);
        assert_eq!(v, DataInv007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — NFC round-trip failures (one is enough).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_nfc_failure_in_million() {
        let v = verdict_from_utf8_nfc_scan(1_000_000, 0, 1);
        assert_eq!(
            v,
            DataInv007Verdict::Fail,
            "one nfc failure must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_all_nfc_failures() {
        let v = verdict_from_utf8_nfc_scan(1_000_000, 0, 1_000_000);
        assert_eq!(v, DataInv007Verdict::Fail);
    }

    #[test]
    fn fail_both_utf8_and_nfc_each_one() {
        let v = verdict_from_utf8_nfc_scan(1_000_000, 1, 1);
        assert_eq!(v, DataInv007Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — caller / counter errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_scanned_files() {
        // Vacuous Pass refused: scanning zero files validates nothing.
        let v = verdict_from_utf8_nfc_scan(0, 0, 0);
        assert_eq!(
            v,
            DataInv007Verdict::Fail,
            "zero scanned files must Fail (vacuous Pass refused)"
        );
    }

    #[test]
    fn fail_zero_scanned_with_failures() {
        let v = verdict_from_utf8_nfc_scan(0, 5, 5);
        assert_eq!(v, DataInv007Verdict::Fail);
    }

    #[test]
    fn fail_combined_exceeds_scanned() {
        // Counter corruption: utf8=600k + nfc=600k = 1.2M > scanned=1M.
        let v = verdict_from_utf8_nfc_scan(1_000_000, 600_000, 600_000);
        assert_eq!(
            v,
            DataInv007Verdict::Fail,
            "utf8 + nfc > scanned must Fail (partition violation)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Overflow protection — checked_add on (utf8 + nfc).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_utf8_plus_nfc_overflow() {
        // utf8 + nfc would overflow u64.
        let huge = u64::MAX / 2 + 1;
        let v = verdict_from_utf8_nfc_scan(u64::MAX, huge, huge);
        // huge + huge overflows → Fail.
        assert_eq!(
            v,
            DataInv007Verdict::Fail,
            "overflow in utf8 + nfc must Fail (not silently wrap)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — failure counts from 0 to scanned.
    // -------------------------------------------------------------------------
    #[test]
    fn utf8_sweep_at_fixed_scanned_zero_nfc() {
        let scanned = 1000_u64;
        let probes: Vec<(u64, DataInv007Verdict)> = vec![
            (0, DataInv007Verdict::Pass),
            (1, DataInv007Verdict::Fail),
            (10, DataInv007Verdict::Fail),
            (500, DataInv007Verdict::Fail),
            (1000, DataInv007Verdict::Fail),
            (1001, DataInv007Verdict::Fail), // partition violation
        ];
        for (utf8, expected) in probes {
            let v = verdict_from_utf8_nfc_scan(scanned, utf8, 0);
            assert_eq!(
                v, expected,
                "scanned={scanned} utf8={utf8} expected {expected:?}"
            );
        }
    }

    #[test]
    fn nfc_sweep_at_fixed_scanned_zero_utf8() {
        let scanned = 1000_u64;
        let probes: Vec<(u64, DataInv007Verdict)> = vec![
            (0, DataInv007Verdict::Pass),
            (1, DataInv007Verdict::Fail),
            (10, DataInv007Verdict::Fail),
            (500, DataInv007Verdict::Fail),
            (1000, DataInv007Verdict::Fail),
            (1001, DataInv007Verdict::Fail),
        ];
        for (nfc, expected) in probes {
            let v = verdict_from_utf8_nfc_scan(scanned, 0, nfc);
            assert_eq!(
                v, expected,
                "scanned={scanned} nfc={nfc} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — both dimensions zero-tolerance simultaneously.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_both_failures_are_exactly_zero() {
        // Property: at any non-zero scanned count, Pass iff
        // utf8 == 0 AND nfc == 0.
        for scanned in [1_u64, 100, 10_000, 1_000_000] {
            // Pass case
            let v_pass = verdict_from_utf8_nfc_scan(scanned, 0, 0);
            assert_eq!(v_pass, DataInv007Verdict::Pass, "scanned={scanned}");

            // utf8 violation
            let v_u = verdict_from_utf8_nfc_scan(scanned, 1, 0);
            assert_eq!(v_u, DataInv007Verdict::Fail);

            // nfc violation
            let v_n = verdict_from_utf8_nfc_scan(scanned, 0, 1);
            assert_eq!(v_n, DataInv007Verdict::Fail);

            // both
            let v_b = verdict_from_utf8_nfc_scan(scanned, 1, 1);
            assert_eq!(v_b, DataInv007Verdict::Fail);
        }
    }
}
