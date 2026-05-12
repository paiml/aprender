// SHIP-TWO-001 — `apr-cli-operations-v1` algorithm-level PARTIAL
// discharge for FALSIFY-OPS-004.
//
// Contract: `contracts/apr-cli-operations-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What FALSIFY-OPS-004 says
//
//   rule: Progress percentage monotonically increasing
//   prediction: Progress never decreases during model loading
//   test: Capture all progress events during pull, verify monotonicity
//   if_fails: Progress percentage regression (confuses users)
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given a sequence of progress percentages observed
// during a model pull/load operation, Pass iff:
//
//   sequence is non-empty AND
//   every value is finite AND in [0.0, 100.0] AND
//   for every adjacent pair (a, b): a <= b (monotonically
//   non-decreasing)
//
// Non-strict `<=` allows the same value to be emitted twice in a
// row (typical with throttled progress reporting). Strict `<` would
// reject normal progress streams.

/// Binary verdict for `FALSIFY-OPS-004`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ops004Verdict {
    /// Sequence is non-empty, all values are finite + in
    /// `[0.0, 100.0]`, AND every adjacent pair is non-decreasing.
    Pass,
    /// One or more of:
    /// - `progress_sequence.is_empty()` (caller error — no
    ///   progress events captured).
    /// - Any value is NaN, ±∞, or outside `[0.0, 100.0]`.
    /// - Any adjacent pair `(a, b)` has `a > b` (progress
    ///   regressed).
    Fail,
}

/// Pure verdict function for `FALSIFY-OPS-004`.
///
/// Inputs:
/// - `progress_sequence`: percentages (in [0, 100]) emitted by
///   the apr puller in chronological order.
///
/// Pass iff:
/// 1. `!progress_sequence.is_empty()`,
/// 2. Every value is finite,
/// 3. Every value is in `[0.0, 100.0]`,
/// 4. For every `i > 0`: `progress_sequence[i-1] <=
///    progress_sequence[i]`.
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Strictly increasing — `Pass`:
/// ```
/// use aprender::format::ops_004::{
///     verdict_from_progress_monotonicity, Ops004Verdict,
/// };
/// let v = verdict_from_progress_monotonicity(&[0.0, 25.0, 50.0, 75.0, 100.0]);
/// assert_eq!(v, Ops004Verdict::Pass);
/// ```
///
/// Regression at index 2 — `Fail`:
/// ```
/// use aprender::format::ops_004::{
///     verdict_from_progress_monotonicity, Ops004Verdict,
/// };
/// let v = verdict_from_progress_monotonicity(&[0.0, 50.0, 30.0, 100.0]);
/// assert_eq!(v, Ops004Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_progress_monotonicity(progress_sequence: &[f64]) -> Ops004Verdict {
    if progress_sequence.is_empty() {
        return Ops004Verdict::Fail;
    }
    for &p in progress_sequence {
        if !p.is_finite() {
            return Ops004Verdict::Fail;
        }
        if !(0.0..=100.0).contains(&p) {
            return Ops004Verdict::Fail;
        }
    }
    for w in progress_sequence.windows(2) {
        if w[0] > w[1] {
            return Ops004Verdict::Fail;
        }
    }
    Ops004Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — typical clean progress sequences.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_strictly_increasing() {
        let v = verdict_from_progress_monotonicity(&[0.0, 25.0, 50.0, 75.0, 100.0]);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    #[test]
    fn pass_with_repeats() {
        // Throttled emission: 25.0 → 25.0 → 50.0 (same-value adjacent
        // pairs are allowed under non-strict <=).
        let v = verdict_from_progress_monotonicity(&[0.0, 25.0, 25.0, 50.0, 100.0]);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    #[test]
    fn pass_single_event() {
        // Single 0.0 (or 100.0) sequence is trivially monotonic.
        let v = verdict_from_progress_monotonicity(&[0.0]);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    #[test]
    fn pass_starts_at_nonzero() {
        // Resume scenario: progress starts at 50% (cached prefix).
        let v = verdict_from_progress_monotonicity(&[50.0, 75.0, 100.0]);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    #[test]
    fn pass_realistic_dense_progress() {
        // Realistic 1% emission cadence.
        let dense: Vec<f64> = (0..=100).map(f64::from).collect();
        let v = verdict_from_progress_monotonicity(&dense);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — progress regression.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_simple_regression() {
        let v = verdict_from_progress_monotonicity(&[0.0, 50.0, 30.0, 100.0]);
        assert_eq!(
            v,
            Ops004Verdict::Fail,
            "regression at index 2 must Fail"
        );
    }

    #[test]
    fn fail_drop_to_zero() {
        // Progress reset mid-stream.
        let v = verdict_from_progress_monotonicity(&[0.0, 50.0, 0.0, 100.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    #[test]
    fn fail_one_ulp_regression() {
        let v = verdict_from_progress_monotonicity(&[
            50.0,
            f64::from_bits(50.0_f64.to_bits() - 1),
        ]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    #[test]
    fn fail_regression_at_end() {
        // 99 → 50 at the last pair.
        let v = verdict_from_progress_monotonicity(&[0.0, 25.0, 50.0, 99.0, 50.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — empty sequence (caller error).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_sequence() {
        let v = verdict_from_progress_monotonicity(&[]);
        assert_eq!(
            v,
            Ops004Verdict::Fail,
            "empty sequence must Fail (no progress observed)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — domain violations (NaN, ±∞).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_nan_in_sequence() {
        let v = verdict_from_progress_monotonicity(&[0.0, 25.0, f64::NAN, 50.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    #[test]
    fn fail_positive_infinity() {
        let v = verdict_from_progress_monotonicity(&[0.0, f64::INFINITY, 100.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    #[test]
    fn fail_negative_infinity() {
        let v = verdict_from_progress_monotonicity(&[f64::NEG_INFINITY, 50.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — out-of-range values.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_negative_percentage() {
        let v = verdict_from_progress_monotonicity(&[-1.0, 50.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    #[test]
    fn fail_above_100_percent() {
        let v = verdict_from_progress_monotonicity(&[0.0, 50.0, 101.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    #[test]
    fn fail_huge_value() {
        let v = verdict_from_progress_monotonicity(&[0.0, 1e10]);
        assert_eq!(v, Ops004Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Boundary cases — at exactly 0.0 and 100.0.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_at_floor_0_percent() {
        let v = verdict_from_progress_monotonicity(&[0.0]);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    #[test]
    fn pass_at_ceiling_100_percent() {
        let v = verdict_from_progress_monotonicity(&[100.0]);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    #[test]
    fn pass_full_range_two_events() {
        let v = verdict_from_progress_monotonicity(&[0.0, 100.0]);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — apr pull dataset progress scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_apr_pull_clean_stream() {
        // 1KB granularity progress on 1MB file.
        let mut events = Vec::new();
        for i in 0..=1024 {
            events.push(f64::from(i) * 100.0 / 1024.0);
        }
        let v = verdict_from_progress_monotonicity(&events);
        assert_eq!(v, Ops004Verdict::Pass);
    }

    #[test]
    fn fail_apr_pull_retry_resets_progress() {
        // Realistic regression: HF endpoint 503; puller retries
        // and emits progress=0 mid-stream.
        let v = verdict_from_progress_monotonicity(&[0.0, 30.0, 60.0, 0.0, 30.0, 100.0]);
        assert_eq!(
            v,
            Ops004Verdict::Fail,
            "retry reset must Fail (confuses users)"
        );
    }

    #[test]
    fn fail_apr_pull_concurrent_writer_overlap() {
        // Multi-thread regression: progress events arrive out of
        // order due to lockless emission.
        let v = verdict_from_progress_monotonicity(&[0.0, 25.0, 50.0, 40.0, 75.0]);
        assert_eq!(v, Ops004Verdict::Fail);
    }
}
