// SHIP-TWO-001 MODEL-2 — `dataset-thestack-python-v1` (C-DATA-THESTACK-PYTHON)
// algorithm-level PARTIAL discharge for INV-DATA-003.
//
// Contract: `contracts/dataset-thestack-python-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pipeline (§26.2), AC-SHIP2-002.
//
// ## What INV-DATA-003 says
//
//   description: No two files in the corpus have Jaccard similarity
//                on 5-line shingles ≥ 0.85 (dedup survived its own
//                rule).
//   falsifier:   Sample 10,000 random file pairs; compute MinHash
//                Jaccard; if any pair ≥ 0.85 → FAIL.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a sample-scan that produces
// (`pairs_sampled`, `max_observed_jaccard`), Pass iff:
//
//   pairs_sampled >= AC_DATA_INV_003_MIN_PAIRS (10_000) AND
//   max_observed_jaccard.is_finite() AND
//   0.0 <= max_observed_jaccard < AC_DATA_INV_003_JACCARD_THRESHOLD (0.85)
//
// AND the verdict is exclusive at the threshold: a pair that
// observes Jaccard exactly 0.85 → Fail (matching the contract's
// `≥ 0.85 → FAIL` wording character-for-character). NaN → Fail
// (cannot reason about an undefined similarity). Negative or > 1.0 →
// Fail (out-of-domain).

/// Minimum sample size required to give the verdict statistical
/// power.
///
/// Per contract `INV-DATA-003`: the falsifier "Sample 10,000 random
/// file pairs". A scan that produced fewer than 10k pairs lacks the
/// statistical floor the contract assumes. Drift to 1k would let
/// near-duplicate clusters slip through; drift to 100k would impose
/// unwarranted compute on every smoke run.
pub const AC_DATA_INV_003_MIN_PAIRS: u64 = 10_000;

/// Jaccard threshold above which two files are considered
/// near-duplicates.
///
/// Per contract: ≥ 0.85 → Fail. The verdict therefore Pass iff
/// `max_observed_jaccard < 0.85` (strict). Drift to 0.80 would
/// over-tighten and reject sibling files; drift to 0.90 would let
/// clear duplicates pass.
pub const AC_DATA_INV_003_JACCARD_THRESHOLD: f64 = 0.85;

/// Binary verdict for `INV-DATA-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataInv003Verdict {
    /// Sample-scan visited at least 10,000 pairs, the observed max
    /// Jaccard is finite and in `[0.0, 0.85)`.
    Pass,
    /// One or more of:
    /// - `pairs_sampled < AC_DATA_INV_003_MIN_PAIRS` (insufficient
    ///   statistical power).
    /// - `max_observed_jaccard` is NaN (undefined).
    /// - `max_observed_jaccard < 0.0` or `> 1.0` (out-of-domain).
    /// - `max_observed_jaccard >= AC_DATA_INV_003_JACCARD_THRESHOLD`
    ///   (near-duplicate observed).
    Fail,
}

/// Pure verdict function for `INV-DATA-003`.
///
/// Inputs:
/// - `pairs_sampled`: number of random file pairs the dedup scan
///   evaluated.
/// - `max_observed_jaccard`: maximum MinHash Jaccard similarity
///   observed across those pairs.
///
/// Pass iff:
/// 1. `pairs_sampled >= 10_000`,
/// 2. `max_observed_jaccard.is_finite()` (rules out NaN and ±∞),
/// 3. `0.0 <= max_observed_jaccard < 0.85`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 10k pairs, max 0.42 — `Pass`:
/// ```
/// use aprender::format::data_inv_003::{
///     verdict_from_jaccard_max, DataInv003Verdict,
/// };
/// let v = verdict_from_jaccard_max(10_000, 0.42);
/// assert_eq!(v, DataInv003Verdict::Pass);
/// ```
///
/// 10k pairs, max exactly 0.85 — `Fail` (strict threshold):
/// ```
/// use aprender::format::data_inv_003::{
///     verdict_from_jaccard_max, DataInv003Verdict,
/// };
/// let v = verdict_from_jaccard_max(10_000, 0.85);
/// assert_eq!(v, DataInv003Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_jaccard_max(
    pairs_sampled: u64,
    max_observed_jaccard: f64,
) -> DataInv003Verdict {
    if pairs_sampled < AC_DATA_INV_003_MIN_PAIRS {
        return DataInv003Verdict::Fail;
    }
    if !max_observed_jaccard.is_finite() {
        return DataInv003Verdict::Fail;
    }
    if !(0.0..=1.0).contains(&max_observed_jaccard) {
        return DataInv003Verdict::Fail;
    }
    if max_observed_jaccard >= AC_DATA_INV_003_JACCARD_THRESHOLD {
        return DataInv003Verdict::Fail;
    }
    DataInv003Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — constants match contract.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_min_pairs_is_10_000() {
        assert_eq!(AC_DATA_INV_003_MIN_PAIRS, 10_000);
    }

    #[test]
    fn provenance_jaccard_threshold_is_0_85() {
        assert!((AC_DATA_INV_003_JACCARD_THRESHOLD - 0.85).abs() < 1e-12);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — typical clean-corpus values.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_typical_low_jaccard() {
        let v = verdict_from_jaccard_max(10_000, 0.42);
        assert_eq!(v, DataInv003Verdict::Pass);
    }

    #[test]
    fn pass_zero_jaccard() {
        let v = verdict_from_jaccard_max(10_000, 0.0);
        assert_eq!(v, DataInv003Verdict::Pass);
    }

    #[test]
    fn pass_just_below_threshold() {
        // Strict <: 0.8499... must Pass.
        let v = verdict_from_jaccard_max(10_000, 0.849_999);
        assert_eq!(v, DataInv003Verdict::Pass);
    }

    #[test]
    fn pass_at_huge_sample_size() {
        let v = verdict_from_jaccard_max(1_000_000, 0.5);
        assert_eq!(v, DataInv003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — strict threshold (0.85 itself fails).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_at_exact_threshold() {
        // Contract says ≥ 0.85 → FAIL. 0.85 itself must Fail.
        let v = verdict_from_jaccard_max(10_000, 0.85);
        assert_eq!(
            v,
            DataInv003Verdict::Fail,
            "exact 0.85 must Fail (strict threshold per contract)"
        );
    }

    #[test]
    fn fail_above_threshold() {
        let v = verdict_from_jaccard_max(10_000, 0.90);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    #[test]
    fn fail_at_one() {
        // Identical files → Jaccard 1.0.
        let v = verdict_from_jaccard_max(10_000, 1.0);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    #[test]
    fn fail_just_above_threshold() {
        // 0.85 + 1 ULP must Fail.
        let just_above = f64::from_bits(0.85_f64.to_bits() + 1);
        let v = verdict_from_jaccard_max(10_000, just_above);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — sample size too small.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_pairs_sampled() {
        let v = verdict_from_jaccard_max(0, 0.0);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    #[test]
    fn fail_just_below_min_pairs() {
        let v = verdict_from_jaccard_max(9_999, 0.0);
        assert_eq!(
            v,
            DataInv003Verdict::Fail,
            "9_999 pairs lacks contractual statistical power"
        );
    }

    #[test]
    fn pass_at_exact_min_pairs() {
        // 10_000 is inclusive floor.
        let v = verdict_from_jaccard_max(10_000, 0.5);
        assert_eq!(v, DataInv003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — domain violations (NaN, ±inf, negative, > 1.0).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_nan_jaccard() {
        let v = verdict_from_jaccard_max(10_000, f64::NAN);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    #[test]
    fn fail_positive_infinity() {
        let v = verdict_from_jaccard_max(10_000, f64::INFINITY);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    #[test]
    fn fail_negative_infinity() {
        let v = verdict_from_jaccard_max(10_000, f64::NEG_INFINITY);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    #[test]
    fn fail_negative_jaccard() {
        let v = verdict_from_jaccard_max(10_000, -0.01);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    #[test]
    fn fail_above_one() {
        let v = verdict_from_jaccard_max(10_000, 1.01);
        assert_eq!(v, DataInv003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary sweep around the 0.85 threshold.
    // -------------------------------------------------------------------------
    #[test]
    fn jaccard_sweep_around_threshold() {
        let probes: Vec<(f64, DataInv003Verdict)> = vec![
            (0.0, DataInv003Verdict::Pass),
            (0.50, DataInv003Verdict::Pass),
            (0.80, DataInv003Verdict::Pass),
            (0.84, DataInv003Verdict::Pass),
            (0.849, DataInv003Verdict::Pass),
            (0.85, DataInv003Verdict::Fail), // exact threshold → Fail
            (0.851, DataInv003Verdict::Fail),
            (0.90, DataInv003Verdict::Fail),
            (1.00, DataInv003Verdict::Fail),
        ];
        for (jaccard, expected) in probes {
            let v = verdict_from_jaccard_max(10_000, jaccard);
            assert_eq!(v, expected, "jaccard={jaccard} expected {expected:?}");
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic scale — sample size sweep at fixed safe Jaccard.
    // -------------------------------------------------------------------------
    #[test]
    fn pairs_sweep_at_fixed_safe_jaccard() {
        let safe_jaccard = 0.5_f64;
        let probes: Vec<(u64, DataInv003Verdict)> = vec![
            (0, DataInv003Verdict::Fail),
            (1, DataInv003Verdict::Fail),
            (1_000, DataInv003Verdict::Fail),
            (9_999, DataInv003Verdict::Fail),
            (10_000, DataInv003Verdict::Pass),
            (10_001, DataInv003Verdict::Pass),
            (100_000, DataInv003Verdict::Pass),
            (1_000_000_000, DataInv003Verdict::Pass),
        ];
        for (pairs, expected) in probes {
            let v = verdict_from_jaccard_max(pairs, safe_jaccard);
            assert_eq!(v, expected, "pairs={pairs} expected {expected:?}");
        }
    }
}
