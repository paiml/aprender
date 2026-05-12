// SHIP-TWO-001 — `apr-cli-operations-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OPS-006.
//
// Contract: `contracts/apr-cli-operations-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What FALSIFY-OPS-006 says
//
//   rule: Tokenizer encode/decode roundtrip
//   prediction: Random UTF-8 strings survive tokenize/detokenize
//               roundtrip
//   test: Generate 1000 random UTF-8 strings, encode then decode,
//         assert equality
//   if_fails: Tokenizer drops or corrupts characters
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given (`docs_scanned`, `roundtrip_failures`),
// Pass iff:
//
//   docs_scanned >= AC_OPS_006_REQUIRED_DOCS (1000) AND
//   roundtrip_failures == 0 AND
//   roundtrip_failures <= docs_scanned
//
// Same shape as `bpe_inv_003` (BPE round-trip) but with a smaller
// (1000-doc) sample size and broader scope (any UTF-8 input,
// not just held-out corpus).

/// Required minimum number of random UTF-8 strings to scan.
///
/// Per contract `OPS-006`: "Generate 1000 random UTF-8 strings".
/// Drift to 100 would lose statistical power for rare-codepoint
/// regressions; drift to 10000 would over-tax smoke runs.
pub const AC_OPS_006_REQUIRED_DOCS: u64 = 1_000;

/// Binary verdict for `FALSIFY-OPS-006`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ops006Verdict {
    /// Sample-scan visited >= 1000 random UTF-8 strings AND zero
    /// round-trip failures observed.
    Pass,
    /// One or more of:
    /// - `docs_scanned < 1000` (insufficient sample size).
    /// - `roundtrip_failures > 0` (tokenizer dropped or corrupted
    ///   characters; one is enough).
    /// - `roundtrip_failures > docs_scanned` (counter corruption).
    Fail,
}

/// Pure verdict function for `FALSIFY-OPS-006`.
///
/// Inputs:
/// - `docs_scanned`: number of random UTF-8 strings the
///   roundtrip-test harness evaluated.
/// - `roundtrip_failures`: number of those where
///   `decode(encode(s)) != s`.
///
/// Pass iff:
/// 1. `docs_scanned >= 1000`,
/// 2. `roundtrip_failures == 0`,
/// 3. `roundtrip_failures <= docs_scanned`.
///
/// Otherwise `Fail`.
#[must_use]
pub fn verdict_from_random_roundtrip_scan(
    docs_scanned: u64,
    roundtrip_failures: u64,
) -> Ops006Verdict {
    if docs_scanned < AC_OPS_006_REQUIRED_DOCS {
        return Ops006Verdict::Fail;
    }
    if roundtrip_failures > docs_scanned {
        return Ops006Verdict::Fail;
    }
    if roundtrip_failures == 0 {
        Ops006Verdict::Pass
    } else {
        Ops006Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 1000-doc sample floor.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_required_docs_is_1000() {
        assert_eq!(AC_OPS_006_REQUIRED_DOCS, 1_000);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — clean tokenizer, sufficient sample.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_exact_floor() {
        let v = verdict_from_random_roundtrip_scan(1_000, 0);
        assert_eq!(v, Ops006Verdict::Pass);
    }

    #[test]
    fn pass_above_floor() {
        let v = verdict_from_random_roundtrip_scan(10_000, 0);
        assert_eq!(v, Ops006Verdict::Pass);
    }

    #[test]
    fn pass_at_huge_sample() {
        let v = verdict_from_random_roundtrip_scan(1_000_000, 0);
        assert_eq!(v, Ops006Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — roundtrip failures (zero-tolerance).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_one_failure_in_1000() {
        let v = verdict_from_random_roundtrip_scan(1_000, 1);
        assert_eq!(
            v,
            Ops006Verdict::Fail,
            "one roundtrip failure must Fail (no tolerance)"
        );
    }

    #[test]
    fn fail_handful_of_failures() {
        let v = verdict_from_random_roundtrip_scan(1_000, 7);
        assert_eq!(v, Ops006Verdict::Fail);
    }

    #[test]
    fn fail_one_in_million() {
        let v = verdict_from_random_roundtrip_scan(1_000_000, 1);
        assert_eq!(v, Ops006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — sample size too small.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_docs() {
        let v = verdict_from_random_roundtrip_scan(0, 0);
        assert_eq!(v, Ops006Verdict::Fail);
    }

    #[test]
    fn fail_just_below_floor() {
        let v = verdict_from_random_roundtrip_scan(999, 0);
        assert_eq!(v, Ops006Verdict::Fail);
    }

    #[test]
    fn fail_one_doc() {
        let v = verdict_from_random_roundtrip_scan(1, 0);
        assert_eq!(v, Ops006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Counter / partition violations.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_failures_exceed_docs() {
        let v = verdict_from_random_roundtrip_scan(1_000, 1_001);
        assert_eq!(v, Ops006Verdict::Fail);
    }

    #[test]
    fn fail_huge_failures_with_smaller_scan() {
        let v = verdict_from_random_roundtrip_scan(1_000, u64::MAX);
        assert_eq!(v, Ops006Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep.
    // -------------------------------------------------------------------------
    #[test]
    fn failure_count_sweep_at_1000_scan() {
        let scanned = 1_000_u64;
        let probes: Vec<(u64, Ops006Verdict)> = vec![
            (0, Ops006Verdict::Pass),
            (1, Ops006Verdict::Fail),
            (10, Ops006Verdict::Fail),
            (500, Ops006Verdict::Fail),
            (999, Ops006Verdict::Fail),
            (1_000, Ops006Verdict::Fail),
            (1_001, Ops006Verdict::Fail), // partition
        ];
        for (failures, expected) in probes {
            let v = verdict_from_random_roundtrip_scan(scanned, failures);
            assert_eq!(
                v, expected,
                "scanned={scanned} failures={failures} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Domain — zero-tolerance property.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_iff_failures_is_exactly_zero() {
        for scanned in [1_000_u64, 5_000, 10_000, 100_000] {
            let v_pass = verdict_from_random_roundtrip_scan(scanned, 0);
            assert_eq!(v_pass, Ops006Verdict::Pass, "scanned={scanned}");

            let v_fail = verdict_from_random_roundtrip_scan(scanned, 1);
            assert_eq!(
                v_fail,
                Ops006Verdict::Fail,
                "scanned={scanned} with one failure"
            );
        }
    }
}
