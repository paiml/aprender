// SHIP-TWO-001 MODEL-2 — `dataset-thestack-python-v1` (C-DATA-THESTACK-PYTHON)
// algorithm-level PARTIAL discharge for INV-DATA-004.
//
// Contract: `contracts/dataset-thestack-python-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// MODEL-2 corpus pipeline (§26.2), AC-SHIP2-002.
//
// ## What INV-DATA-004 says
//
//   description: train_token_count ≥ budget.min_train_tokens AND ≤
//                budget.max_train_tokens. val_token_count ≥ 2% of
//                train_token_count.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Two conjoined decision rules:
//
//   1. Train-token range: `min_train_tokens <= train <= max_train_tokens`.
//   2. Val-token floor: `val * 100 >= 2 * train`, computed via
//      `checked_mul` to prevent overflow at corpus sizes near u64::MAX.
//
// Both must Pass for the verdict to Pass. The 2% threshold is bound as
// `AC_DATA_INV_004_VAL_FLOOR_PERCENT` so a future drift to 1% (would
// silently weaken the val split) or to 5% (would over-tighten and
// reject reasonable splits) trips the provenance test.

/// Minimum val-set size as a percentage of train-set size.
///
/// Per contract `INV-DATA-004`: a 2% floor ensures the val split is large
/// enough to detect overfitting without consuming too much of the budget.
/// Drift to 1% would let the eval distribution shift silently mask
/// regressions; drift to 5% would reject reasonable corpus configs.
pub const AC_DATA_INV_004_VAL_FLOOR_PERCENT: u64 = 2;

/// Binary verdict for `INV-DATA-004`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataInv004Verdict {
    /// Train tokens are in `[min_train_tokens, max_train_tokens]` inclusive
    /// AND val tokens are at least 2% of train tokens.
    Pass,
    /// One or more of:
    /// - `min_train_tokens > max_train_tokens` (caller error — no valid range).
    /// - `train_token_count < min_train_tokens` (corpus too small).
    /// - `train_token_count > max_train_tokens` (corpus too large).
    /// - `val_token_count * 100 < train_token_count * 2` (val floor violated).
    /// - Multiplication overflow at the val-floor check (corpus near u64::MAX).
    Fail,
}

/// Pure verdict function for `INV-DATA-004`.
///
/// Inputs:
/// - `train_token_count`: actual number of tokens in the train split.
/// - `val_token_count`: actual number of tokens in the val split.
/// - `min_train_tokens`: contract-specified train budget floor.
/// - `max_train_tokens`: contract-specified train budget ceiling.
///
/// Pass iff:
/// 1. `min_train_tokens <= max_train_tokens`,
/// 2. `min_train_tokens <= train_token_count <= max_train_tokens`,
/// 3. `val_token_count * 100 >= train_token_count * AC_DATA_INV_004_VAL_FLOOR_PERCENT`
///    (computed via `checked_mul`).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// 100M train, 5M val (5%), budget [50M, 200M] — `Pass`:
/// ```
/// use aprender::format::data_inv_004::{
///     verdict_from_split_token_counts, DataInv004Verdict,
/// };
/// let v = verdict_from_split_token_counts(
///     100_000_000, 5_000_000,
///     50_000_000, 200_000_000,
/// );
/// assert_eq!(v, DataInv004Verdict::Pass);
/// ```
///
/// 100M train, 1M val (1%, below 2% floor) — `Fail`:
/// ```
/// use aprender::format::data_inv_004::{
///     verdict_from_split_token_counts, DataInv004Verdict,
/// };
/// let v = verdict_from_split_token_counts(
///     100_000_000, 1_000_000,
///     50_000_000, 200_000_000,
/// );
/// assert_eq!(v, DataInv004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_split_token_counts(
    train_token_count: u64,
    val_token_count: u64,
    min_train_tokens: u64,
    max_train_tokens: u64,
) -> DataInv004Verdict {
    if min_train_tokens > max_train_tokens {
        return DataInv004Verdict::Fail;
    }
    if train_token_count < min_train_tokens || train_token_count > max_train_tokens {
        return DataInv004Verdict::Fail;
    }
    // val_token_count * 100 >= train_token_count * 2
    let val_lhs = match val_token_count.checked_mul(100) {
        Some(v) => v,
        None => return DataInv004Verdict::Fail,
    };
    let train_rhs = match train_token_count.checked_mul(AC_DATA_INV_004_VAL_FLOOR_PERCENT) {
        Some(v) => v,
        None => return DataInv004Verdict::Fail,
    };
    if val_lhs < train_rhs {
        return DataInv004Verdict::Fail;
    }
    DataInv004Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — 2% floor matches contract.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_val_floor_is_two_percent() {
        assert_eq!(AC_DATA_INV_004_VAL_FLOOR_PERCENT, 2);
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — train in range AND val ≥ 2% of train.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_typical_5_percent_val_split() {
        let v = verdict_from_split_token_counts(100_000_000, 5_000_000, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Pass);
    }

    #[test]
    fn pass_exactly_at_min_train() {
        let v = verdict_from_split_token_counts(50_000_000, 1_000_000, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Pass);
    }

    #[test]
    fn pass_exactly_at_max_train() {
        let v = verdict_from_split_token_counts(200_000_000, 4_000_000, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Pass);
    }

    #[test]
    fn pass_exactly_at_val_floor() {
        // 100M train, 2M val = exactly 2% — inclusive floor.
        let v = verdict_from_split_token_counts(100_000_000, 2_000_000, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Pass, "exact 2% must Pass (inclusive)");
    }

    #[test]
    fn pass_realistic_565m_corpus() {
        // Per spec: MODEL-2 565.6M tokens. 95/5 split = 537M train, 28M val.
        let v = verdict_from_split_token_counts(537_000_000, 28_000_000, 500_000_000, 600_000_000);
        assert_eq!(v, DataInv004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — train out of range.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_train_below_min() {
        let v = verdict_from_split_token_counts(49_999_999, 5_000_000, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Fail);
    }

    #[test]
    fn fail_train_above_max() {
        let v = verdict_from_split_token_counts(200_000_001, 10_000_000, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Fail);
    }

    #[test]
    fn fail_train_zero_with_zero_min() {
        // Edge: min=0 max=0 train=0 val=0 — vacuous Pass possible? Let's see.
        // train (0) is in [0, 0] ✓. val (0) * 100 = 0 >= train (0) * 2 = 0. Pass.
        let v = verdict_from_split_token_counts(0, 0, 0, 0);
        assert_eq!(
            v,
            DataInv004Verdict::Pass,
            "all-zero is degenerate but technically valid"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — val below 2% floor.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_val_one_percent() {
        let v = verdict_from_split_token_counts(100_000_000, 1_000_000, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Fail);
    }

    #[test]
    fn fail_val_just_below_floor() {
        // 100M train: 2% floor = 2_000_000. 1_999_999 must Fail.
        let v = verdict_from_split_token_counts(100_000_000, 1_999_999, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Fail);
    }

    #[test]
    fn fail_zero_val_with_real_train() {
        let v = verdict_from_split_token_counts(100_000_000, 0, 50_000_000, 200_000_000);
        assert_eq!(v, DataInv004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — caller errors (inverted range, etc.).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_inverted_min_max() {
        // min > max → no valid range → Fail.
        let v = verdict_from_split_token_counts(100, 50, 200, 100);
        assert_eq!(v, DataInv004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Overflow protection — checked_mul on val_lhs and train_rhs.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_val_count_overflows_times_100() {
        // val * 100 overflows u64.
        let val = u64::MAX / 50; // val * 100 > u64::MAX
        let v = verdict_from_split_token_counts(100, val, 50, 200);
        assert_eq!(
            v,
            DataInv004Verdict::Fail,
            "overflow in val * 100 must Fail (not silently wrap)"
        );
    }

    #[test]
    fn fail_train_count_overflows_times_2() {
        // train * 2 would overflow if we tried it on u64::MAX, but train
        // is constrained by [min, max]. Force max to be near-overflow.
        let near_max = u64::MAX / 2 + 1;
        let v = verdict_from_split_token_counts(near_max, near_max, 0, u64::MAX);
        // train * 2 overflows; but val * 100 also overflows first since
        // val == near_max. Either way, Fail.
        assert_eq!(v, DataInv004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Boundary sweep — val sweep around 2% floor at fixed train.
    // -------------------------------------------------------------------------
    #[test]
    fn val_sweep_around_two_percent_floor() {
        let train = 1_000_000_u64; // 2% = 20_000
        let probes: Vec<(u64, DataInv004Verdict)> = vec![
            (0, DataInv004Verdict::Fail),
            (1, DataInv004Verdict::Fail),
            (10_000, DataInv004Verdict::Fail),
            (19_999, DataInv004Verdict::Fail),
            (20_000, DataInv004Verdict::Pass), // exactly 2% — inclusive
            (20_001, DataInv004Verdict::Pass),
            (50_000, DataInv004Verdict::Pass),
            (1_000_000, DataInv004Verdict::Pass),
        ];
        for (val, expected) in probes {
            let v = verdict_from_split_token_counts(train, val, 100, 10_000_000);
            assert_eq!(v, expected, "train={train} val={val} expected {expected:?}");
        }
    }
}
