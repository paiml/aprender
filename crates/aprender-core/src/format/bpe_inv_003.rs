// SHIP-TWO-001 MODEL-2 — `tokenizer-bpe-v1` (C-TOK-BPE-001)
// algorithm-level PARTIAL discharge for INV-BPE-003.
//
// Contract: `contracts/tokenizer-bpe-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 tokenizer pipeline (§26.3), AC-SHIP2-003.
// Also FALSIFY-SHIP-012 directly.
//
// ## What INV-BPE-003 says
//
//   description: Round-trip: decode(encode(text)) byte-equals text
//                for every document in the 10K-doc held-out corpus,
//                with NFC pre-applied to both sides of the
//                comparison.
//   falsifier:   Tokenize each of the 10_000 held-out docs,
//                detokenize, byte-compare to nfc(original). Any
//                non-zero diff bytes fails. This is the
//                SHIP-TWO-001 FALSIFY-SHIP-012 gate directly.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a sample-scan that produces
// (`docs_scanned`, `roundtrip_failures`), Pass iff:
//
//   docs_scanned >= AC_BPE_INV_003_REQUIRED_DOCS (10_000) AND
//   roundtrip_failures == 0 AND
//   roundtrip_failures <= docs_scanned
//
// Zero-tolerance on round-trip failures: a single byte-mismatch
// between `decode(encode(nfc(text)))` and `nfc(text)` indicates
// non-injective tokenization, which corrupts MODEL-2's loss target
// at training time. Floor of 10K docs matches the contract's
// statistical-power requirement; below that, rare round-trip
// failures (e.g., an emoji edge case in 1-in-50K docs) escape
// detection.

/// Required minimum number of held-out documents to scan.
///
/// Per contract `INV-BPE-003`: "10K-doc held-out corpus". A scan
/// over fewer docs lacks coverage of rare round-trip-failing tokens
/// (emoji ZWJ sequences, RTL combining marks, control sequences).
/// Drift to 1K would let those slip; drift to 100K would over-tax
/// every smoke run.
pub const AC_BPE_INV_003_REQUIRED_DOCS: u64 = 10_000;

/// Binary verdict for `INV-BPE-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpeInv003Verdict {
    /// Sample-scan visited at least 10K docs AND zero round-trip
    /// failures observed.
    Pass,
    /// One or more of:
    /// - `docs_scanned < 10_000` (insufficient statistical power).
    /// - `roundtrip_failures > 0` (one is enough — non-injective
    ///   tokenization).
    /// - `roundtrip_failures > docs_scanned` (counter corruption).
    Fail,
}

/// Pure verdict function for `INV-BPE-003`.
///
/// Inputs:
/// - `docs_scanned`: number of held-out documents the round-trip
///   scan evaluated.
/// - `roundtrip_failures`: number of those documents where
///   `decode(encode(nfc(text)))` did NOT byte-equal `nfc(text)`.
///
/// Pass iff:
/// 1. `docs_scanned >= 10_000`,
/// 2. `roundtrip_failures == 0`,
/// 3. `roundtrip_failures <= docs_scanned` (counter sanity).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 10K held-out docs, zero failures — `Pass`:
/// ```
/// use aprender::format::bpe_inv_003::{
///     verdict_from_roundtrip_scan, BpeInv003Verdict,
/// };
/// let v = verdict_from_roundtrip_scan(10_000, 0);
/// assert_eq!(v, BpeInv003Verdict::Pass);
/// ```
///
/// One round-trip failure in 10K docs — `Fail`:
/// ```
/// use aprender::format::bpe_inv_003::{
///     verdict_from_roundtrip_scan, BpeInv003Verdict,
/// };
/// let v = verdict_from_roundtrip_scan(10_000, 1);
/// assert_eq!(v, BpeInv003Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_roundtrip_scan(
    docs_scanned: u64,
    roundtrip_failures: u64,
) -> BpeInv003Verdict {
    if docs_scanned < AC_BPE_INV_003_REQUIRED_DOCS {
        return BpeInv003Verdict::Fail;
    }
    if roundtrip_failures > docs_scanned {
        return BpeInv003Verdict::Fail;
    }
    if roundtrip_failures == 0 {
        BpeInv003Verdict::Pass
    } else {
        BpeInv003Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 10K minimum doc count.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_docs_is_10_000() {
        assert_eq!(AC_BPE_INV_003_REQUIRED_DOCS, 10_000);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — clean tokenizer, sufficient sample.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_exact_floor() {
        let v = verdict_from_roundtrip_scan(10_000, 0);
        assert_eq!(v, BpeInv003Verdict::Pass);
    }

    #[test]
    fn pass_above_floor() {
        let v = verdict_from_roundtrip_scan(10_001, 0);
        assert_eq!(v, BpeInv003Verdict::Pass);
    }

    #[test]
    fn pass_at_realistic_csn_python_size() {
        // CSN-Python: ~455k docs after filtering. Scanning all of
        // them is well above the 10K floor.
        let v = verdict_from_roundtrip_scan(455_000, 0);
        assert_eq!(v, BpeInv003Verdict::Pass);
    }

    #[test]
    fn pass_at_huge_sample() {
        let v = verdict_from_roundtrip_scan(10_000_000, 0);
        assert_eq!(v, BpeInv003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — round-trip failures (zero-tolerance).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_roundtrip_failure_in_10k() {
        let v = verdict_from_roundtrip_scan(10_000, 1);
        assert_eq!(
            v,
            BpeInv003Verdict::Fail,
            "one failure must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_handful_of_failures() {
        let v = verdict_from_roundtrip_scan(10_000, 7);
        assert_eq!(v, BpeInv003Verdict::Fail);
    }

    #[test]
    fn fail_all_failed() {
        let v = verdict_from_roundtrip_scan(10_000, 10_000);
        assert_eq!(v, BpeInv003Verdict::Fail);
    }

    #[test]
    fn fail_one_in_million() {
        // Even at huge sample sizes, one failure trips the gate.
        let v = verdict_from_roundtrip_scan(1_000_000, 1);
        assert_eq!(v, BpeInv003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — sample size too small.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_docs_scanned() {
        let v = verdict_from_roundtrip_scan(0, 0);
        assert_eq!(v, BpeInv003Verdict::Fail);
    }

    #[test]
    fn fail_just_below_floor() {
        let v = verdict_from_roundtrip_scan(9_999, 0);
        assert_eq!(
            v,
            BpeInv003Verdict::Fail,
            "9_999 docs lacks contract statistical power"
        );
    }

    #[test]
    fn fail_one_doc_zero_failures() {
        // Single doc passes round-trip but fails statistical floor.
        let v = verdict_from_roundtrip_scan(1, 0);
        assert_eq!(v, BpeInv003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — counter / partition violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_failures_exceed_docs() {
        // Counter corruption: failures > scanned.
        let v = verdict_from_roundtrip_scan(10_000, 10_001);
        assert_eq!(v, BpeInv003Verdict::Fail);
    }

    #[test]
    fn fail_huge_failures_with_smaller_scan() {
        let v = verdict_from_roundtrip_scan(10_000, u64::MAX);
        assert_eq!(v, BpeInv003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep — failure-count sweep at fixed 10K scan.
    // -------------------------------------------------------------------------
    #[test]
    fn failure_count_sweep_at_floor() {
        let scanned = 10_000_u64;
        let probes: Vec<(u64, BpeInv003Verdict)> = vec![
            (0, BpeInv003Verdict::Pass),
            (1, BpeInv003Verdict::Fail),
            (2, BpeInv003Verdict::Fail),
            (10, BpeInv003Verdict::Fail),
            (5_000, BpeInv003Verdict::Fail),
            (9_999, BpeInv003Verdict::Fail),
            (10_000, BpeInv003Verdict::Fail),
            (10_001, BpeInv003Verdict::Fail), // partition violation
        ];
        for (failures, expected) in probes {
            let v = verdict_from_roundtrip_scan(scanned, failures);
            assert_eq!(
                v, expected,
                "scanned={scanned} failures={failures} expected {expected:?}"
            );
        }
    }

    #[test]
    fn docs_scanned_sweep_at_zero_failures() {
        let probes: Vec<(u64, BpeInv003Verdict)> = vec![
            (0, BpeInv003Verdict::Fail),
            (1, BpeInv003Verdict::Fail),
            (1_000, BpeInv003Verdict::Fail),
            (9_999, BpeInv003Verdict::Fail),
            (10_000, BpeInv003Verdict::Pass),
            (10_001, BpeInv003Verdict::Pass),
            (100_000, BpeInv003Verdict::Pass),
            (10_000_000, BpeInv003Verdict::Pass),
        ];
        for (scanned, expected) in probes {
            let v = verdict_from_roundtrip_scan(scanned, 0);
            assert_eq!(
                v, expected,
                "scanned={scanned} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — zero-tolerance property at canonical sizes.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_failures_is_exactly_zero() {
        for scanned in [10_000_u64, 50_000, 100_000, 1_000_000] {
            let v_pass = verdict_from_roundtrip_scan(scanned, 0);
            assert_eq!(v_pass, BpeInv003Verdict::Pass, "scanned={scanned}");

            let v_fail = verdict_from_roundtrip_scan(scanned, 1);
            assert_eq!(
                v_fail,
                BpeInv003Verdict::Fail,
                "scanned={scanned} with one failure"
            );
        }
    }
}
