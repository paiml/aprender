//! Fixed-order, f64-accumulating reductions — the trainer's ONLY reduction door (D-13).
//!
//! Contract: `setfit-train-lifecycle-v1` (authored in plan 03-06). Requirement: TRN-06.
//!
//! # Why not `par_iter().sum()`
//!
//! A rayon reduction is not bitwise reproducible even at a FIXED thread count, because
//! work-stealing changes which partial sums are combined with which — and floating-point
//! addition is not associative, so a different reduction tree is a different number. Pinning
//! the thread count does not fix it; it only makes the nondeterminism less frequent, which
//! is worse than making it obvious. TRN-06's "bitwise at any thread count" therefore
//! resolves to: every trainer-side reduction runs in index order, sequentially, here.
//!
//! # Why f64 accumulation over f32 inputs
//!
//! Two reasons, and the second is the load-bearing one for this phase.
//!
//! 1. Cancellation. Summing `[1e8, 1.0, -1e8]` in f32 loses the `1.0` entirely: `1e8 + 1.0`
//!    is `1e8` again, because 1.0 is far below the ULP of 1e8. In f64 the term survives.
//!    `reduce_recovers_the_term_f32_accumulation_destroys` pins exactly that.
//! 2. Near-frozen parameters. D-10's SetFit-identity gate discriminates on
//!    `||delta_theta|| / max(||theta_init||, s_class)`, and for a parameter that barely
//!    moved the delta norm is a sum of many tiny squares against a large base. That is the
//!    regime where f32 accumulation silently reports zero movement — which would fail a
//!    legitimate run, or pass an illegitimate one, depending on which side of the epsilon
//!    the noise landed.
//!
//! Every function here returns `f64`. Narrowing back to f32 is the caller's decision at the
//! APR boundary, made once and visibly.

/// Sum `values` in index order, accumulating in f64.
///
/// An empty slice sums to `0.0`, which is the additive identity and not a special case.
#[must_use]
pub fn sum_in_index_order(values: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for &value in values {
        total += f64::from(value);
    }
    total
}

/// Arithmetic mean of `values`, accumulated in index order in f64.
///
/// An EMPTY slice returns `0.0` rather than NaN. A mean of nothing is not a number, but the
/// callers here are metric accumulators over batches that a validated config cannot make
/// empty (`batch_size` is non-zero, `epochs` is non-zero), and returning NaN would poison a
/// loss trace through every subsequent comparison rather than staying local. The choice is
/// stated rather than implied so nobody reads `0.0` as "the mean was measured to be zero".
#[must_use]
pub fn mean_in_index_order(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let count = values.len() as f64;
    sum_in_index_order(values) / count
}

/// Sum already-f64 `values` in index order.
///
/// # Why a second `sum` rather than narrowing to the f32 door
///
/// The door D-13 names is about ORDER, not about the input width, and the guarantee here is
/// exactly the one above: sequential, index-order, f64 accumulation. Plan 03-09's macro-F1
/// averages per-class RATIOS that were computed in f64 from integer counts; pushing them
/// through [`sum_in_index_order`] would mean narrowing each ratio to f32 first, which discards
/// precision the ratio already has for no reason other than to reuse a signature. Two
/// functions with one reduction rule beats one function that silently rounds its input.
#[must_use]
pub fn sum_f64_in_index_order(values: &[f64]) -> f64 {
    let mut total = 0.0_f64;
    for &value in values {
        total += value;
    }
    total
}

/// Arithmetic mean of already-f64 `values`, accumulated in index order.
///
/// An EMPTY slice returns `0.0`, for the same stated reason as [`mean_in_index_order`]: a NaN
/// would poison every later comparison rather than staying local, and the callers here average
/// over a DECLARED label map, which a validated dataset cannot make empty.
#[must_use]
pub fn mean_f64_in_index_order(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)]
    let count = values.len() as f64;
    sum_f64_in_index_order(values) / count
}

/// Euclidean (L2) norm of `values`, accumulated in index order in f64.
///
/// The squares are accumulated in f64 BEFORE the square root, so a parameter whose elements
/// are individually near f32 epsilon still contributes: squaring in f32 first would flush
/// those terms to zero and report a delta norm of exactly 0 for a parameter that did move.
#[must_use]
pub fn l2_norm_in_index_order(values: &[f32]) -> f64 {
    let mut total = 0.0_f64;
    for &value in values {
        let widened = f64::from(value);
        total += widened * widened;
    }
    total.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_sum_matches_its_golden_values() {
        assert_eq!(sum_in_index_order(&[]), 0.0);
        assert_eq!(sum_in_index_order(&[1.0, 2.0, 3.0, 4.0]), 10.0);
        assert_eq!(sum_in_index_order(&[-1.5, 1.5]), 0.0);
        assert_eq!(sum_in_index_order(&[0.5; 8]), 4.0);
    }

    #[test]
    fn reduce_mean_matches_its_golden_values() {
        assert_eq!(mean_in_index_order(&[]), 0.0);
        assert_eq!(mean_in_index_order(&[2.0, 4.0]), 3.0);
        assert_eq!(mean_in_index_order(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[test]
    fn reduce_f64_sum_and_mean_match_their_golden_values() {
        assert_eq!(sum_f64_in_index_order(&[]), 0.0);
        assert_eq!(sum_f64_in_index_order(&[1.0, 2.0, 3.0, 4.0]), 10.0);
        assert_eq!(mean_f64_in_index_order(&[]), 0.0);
        assert_eq!(mean_f64_in_index_order(&[2.0, 4.0]), 3.0);
        assert_eq!(mean_f64_in_index_order(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    /// The f64 door PRESERVES what narrowing to the f32 door would discard.
    ///
    /// `1/3` is not representable in either width, but the f32 rounding of it differs from the
    /// f64 one in the 8th decimal place — so a macro-F1 routed through the f32 signature would
    /// report a different number for the same counts. Both values are asserted exactly, so this
    /// distinguishes the two accumulators rather than observing that they are close.
    #[test]
    fn reduce_f64_door_preserves_what_narrowing_to_f32_would_discard() {
        let ratio = 1.0_f64 / 3.0;
        let narrowed = f64::from(ratio as f32);
        assert_ne!(ratio, narrowed, "the fixture must actually round");
        assert_eq!(sum_f64_in_index_order(&[ratio]), ratio);
        assert_eq!(sum_in_index_order(&[ratio as f32]), narrowed);
    }

    #[test]
    fn reduce_l2_norm_matches_its_golden_values() {
        assert_eq!(l2_norm_in_index_order(&[]), 0.0);
        assert_eq!(l2_norm_in_index_order(&[3.0, 4.0]), 5.0);
        assert_eq!(l2_norm_in_index_order(&[1.0, 1.0, 1.0, 1.0]), 2.0);
        assert_eq!(l2_norm_in_index_order(&[-3.0, -4.0]), 5.0);
    }

    /// THE reason these functions accumulate in f64.
    ///
    /// In f32, `1e8 + 1.0 == 1e8` — the ULP of 1e8 is 8.0, so the middle term vanishes and
    /// the sum is exactly 0.0. In f64 it survives and the sum is exactly 1.0. Both results
    /// are asserted as EXACT values, so this test distinguishes the two accumulators rather
    /// than merely observing that they are close.
    #[test]
    fn reduce_recovers_the_term_f32_accumulation_destroys() {
        let series = [1e8_f32, 1.0, -1e8];

        // What an f32 accumulator would have produced, computed here so the claim is
        // demonstrated rather than asserted from memory.
        let mut f32_total = 0.0_f32;
        for &value in &series {
            f32_total += value;
        }
        assert_eq!(f32_total, 0.0, "f32 accumulation is expected to lose the 1.0");

        assert_eq!(
            sum_in_index_order(&series),
            1.0,
            "f64 index-order accumulation must preserve the term f32 loses",
        );
    }

    /// The same cancellation regime, at the magnitude D-10's epsilon actually discriminates.
    #[test]
    fn reduce_l2_norm_survives_a_near_frozen_parameter() {
        // 1024 elements each 1e-7 — an f32 delta that is real but tiny. The exact norm is
        // sqrt(1024 * 1e-14) = 32 * 1e-7 = 3.2e-6.
        let deltas = [1e-7_f32; 1024];
        let norm = l2_norm_in_index_order(&deltas);
        assert!(norm > 0.0, "a parameter that moved must not report a zero delta norm");
        assert!((norm - 3.2e-6).abs() < 1e-12, "expected 3.2e-6, got {norm}",);
    }

    /// Index order is what makes the result reproducible: permuting the inputs is allowed to
    /// change the answer, and pretending otherwise would hide the property being relied on.
    #[test]
    fn reduce_is_a_pure_function_of_the_slice_in_order() {
        let series = [1e8_f32, 1.0, -1e8];
        let repeated = sum_in_index_order(&series);
        assert_eq!(repeated, sum_in_index_order(&series), "the same input twice is the same");

        let permuted = [1e8_f32, -1e8, 1.0];
        assert_eq!(
            sum_in_index_order(&permuted),
            1.0,
            "this particular permutation happens to agree",
        );

        // And one that does NOT agree, so the order-dependence is on the record as a
        // measured fact rather than a caveat. The ULP of 1e30 in f64 is about 1.5e14, so
        // `1.0 + 1e30` discards the 1.0 and the sum is exactly 0.0; cancelling FIRST keeps
        // it. Widening the accumulator moves this threshold, it does not remove it — which
        // is exactly why index order has to be fixed rather than merely hoped for, and why
        // a `par_iter().sum()` whose tree changes run to run cannot be reproducible.
        let hostile = [1.0_f32, 1e30, -1e30];
        assert_eq!(sum_in_index_order(&hostile), 0.0);
        let hostile_reordered = [1e30_f32, -1e30, 1.0];
        assert_eq!(sum_in_index_order(&hostile_reordered), 1.0);
    }

    #[test]
    fn reduce_handles_a_single_element() {
        assert_eq!(sum_in_index_order(&[7.5]), 7.5);
        assert_eq!(mean_in_index_order(&[7.5]), 7.5);
        assert_eq!(l2_norm_in_index_order(&[-7.5]), 7.5);
    }
}
