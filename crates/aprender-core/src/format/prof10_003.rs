// SHIP-TWO-001 perf — `profile-graph-vs-per-op-methodology-v1` (F-PROFILE-010)
// algorithm-level PARTIAL discharge for FALSIFY-PROF10-003.
//
// Contract: `contracts/profile-graph-vs-per-op-methodology-v1.yaml` v1.0.0
// PROPOSED. Spec: docs/specifications/aprender-monorepo-consolidation.md
// (perf gate). First binding for the F-PROFILE-010 contract surface.
//
// ## What FALSIFY-PROF10-003 says
//
//   name: Sanity: hotspot sum < graphed decode time in modern models
//   method: On Qwen2.5-Coder-1.5B Q4_K_M, ungraphed kernel sum per token
//           should EXCEED graphed decode per token (because ungraphed
//           includes launch overhead that graph eliminates). Check the
//           inequality holds.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: `ungraphed_us_per_token > graphed_us_per_token` (strict).
// Both inputs must be non-zero.
//
// Future implementations cannot:
// - Report `ungraphed == graphed` (graph-capture amortization broken).
// - Report `ungraphed < graphed` (timing measurement broken — graphed
//   path should always be at most equal, never *better* than per-op
//   sum which already excludes launch overhead beneficial only for
//   amortization).
// - Report zero on either side (caller error / measurement failure).

/// Binary verdict for `FALSIFY-PROF10-003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prof10_003Verdict {
    /// Ungraphed per-token time strictly exceeds graphed per-token time.
    /// CUDA graph capture is correctly amortizing launch overhead.
    Pass,
    /// One or more of:
    /// - `ungraphed_us_per_token == 0` (measurement failure).
    /// - `graphed_us_per_token == 0` (measurement failure).
    /// - `ungraphed_us_per_token <= graphed_us_per_token` (graph capture
    ///   not amortizing as expected, or per-op timing instrumentation
    ///   broken — both are regression classes the gate catches).
    Fail,
}

/// Pure verdict function for `FALSIFY-PROF10-003`.
///
/// Inputs:
/// - `ungraphed_us_per_token`: sum of per-op kernel compute times, in
///   microseconds, when running with `SKIP_CUDA_GRAPH=1`.
/// - `graphed_us_per_token`: end-to-end graphed decode time per token,
///   in microseconds, when running with default CUDA graph capture.
///
/// Pass iff both inputs are non-zero AND
/// `ungraphed_us_per_token > graphed_us_per_token` (strict).
///
/// # Examples
///
/// Modern model — graph amortization wins — `Pass`:
/// ```
/// use aprender::format::prof10_003::{
///     verdict_from_decode_time_pair, Prof10_003Verdict,
/// };
/// // Qwen2.5-Coder-1.5B representative: ungraphed=2400µs, graphed=1800µs.
/// let v = verdict_from_decode_time_pair(2400, 1800);
/// assert_eq!(v, Prof10_003Verdict::Pass);
/// ```
///
/// Equal times — graph capture not amortizing — `Fail`:
/// ```
/// use aprender::format::prof10_003::{
///     verdict_from_decode_time_pair, Prof10_003Verdict,
/// };
/// let v = verdict_from_decode_time_pair(1800, 1800);
/// assert_eq!(v, Prof10_003Verdict::Fail);
/// ```
#[must_use]
pub const fn verdict_from_decode_time_pair(
    ungraphed_us_per_token: u64,
    graphed_us_per_token: u64,
) -> Prof10_003Verdict {
    if ungraphed_us_per_token == 0 || graphed_us_per_token == 0 {
        return Prof10_003Verdict::Fail;
    }
    if ungraphed_us_per_token > graphed_us_per_token {
        Prof10_003Verdict::Pass
    } else {
        Prof10_003Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Pass band — graph capture amortizing properly.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_qwen_1_5b_representative() {
        // Qwen2.5-Coder-1.5B at ~440 tok/s graphed, ~300 tok/s ungraphed.
        // Implies graphed≈2273µs/tok, ungraphed≈3333µs/tok.
        let v = verdict_from_decode_time_pair(3333, 2273);
        assert_eq!(v, Prof10_003Verdict::Pass);
    }

    #[test]
    fn pass_clear_gap() {
        let v = verdict_from_decode_time_pair(10_000, 5_000);
        assert_eq!(v, Prof10_003Verdict::Pass);
    }

    #[test]
    fn pass_just_above_equal_one_microsecond() {
        let v = verdict_from_decode_time_pair(1801, 1800);
        assert_eq!(v, Prof10_003Verdict::Pass, "1µs gap must Pass (strict >)");
    }

    #[test]
    fn pass_realistic_7b_q4k_model() {
        // Qwen2.5-Coder-7B Q4_K: ~225 tok/s graphed (CUDA), ~140 tok/s ungraphed.
        // graphed≈4444µs/tok, ungraphed≈7142µs/tok.
        let v = verdict_from_decode_time_pair(7142, 4444);
        assert_eq!(v, Prof10_003Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 2: Fail band — equal times (no amortization).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_equal_times() {
        let v = verdict_from_decode_time_pair(1800, 1800);
        assert_eq!(
            v,
            Prof10_003Verdict::Fail,
            "equal times implies graph capture didn't amortize → Fail"
        );
    }

    #[test]
    fn fail_equal_at_typical_4090_speed() {
        let v = verdict_from_decode_time_pair(2273, 2273);
        assert_eq!(v, Prof10_003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — ungraphed < graphed (impossible/bug).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_graphed_slower_than_ungraphed() {
        // Mathematically impossible on a working graph capture — graphed
        // amortizes launch overhead. Catches measurement-instrumentation
        // bugs.
        let v = verdict_from_decode_time_pair(1800, 2400);
        assert_eq!(v, Prof10_003Verdict::Fail);
    }

    #[test]
    fn fail_ungraphed_one_microsecond_below() {
        let v = verdict_from_decode_time_pair(1799, 1800);
        assert_eq!(v, Prof10_003Verdict::Fail, "1µs below must Fail (strict >)");
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — measurement failures (zero values).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_ungraphed_zero() {
        let v = verdict_from_decode_time_pair(0, 1800);
        assert_eq!(v, Prof10_003Verdict::Fail);
    }

    #[test]
    fn fail_graphed_zero() {
        let v = verdict_from_decode_time_pair(2400, 0);
        assert_eq!(v, Prof10_003Verdict::Fail);
    }

    #[test]
    fn fail_both_zero() {
        let v = verdict_from_decode_time_pair(0, 0);
        assert_eq!(v, Prof10_003Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Boundary sweep around equality.
    // -------------------------------------------------------------------------
    #[test]
    fn boundary_sweep_around_equality() {
        let graphed = 1800_u64;
        let probes: Vec<(u64, Prof10_003Verdict)> = vec![
            (0, Prof10_003Verdict::Fail),
            (1, Prof10_003Verdict::Fail), // 1 < 1800
            (1798, Prof10_003Verdict::Fail),
            (1799, Prof10_003Verdict::Fail),
            (1800, Prof10_003Verdict::Fail), // equal
            (1801, Prof10_003Verdict::Pass),
            (2000, Prof10_003Verdict::Pass),
            (3600, Prof10_003Verdict::Pass), // 2× graphed
            (u64::MAX, Prof10_003Verdict::Pass),
        ];
        for (ungraphed, expected) in probes {
            let v = verdict_from_decode_time_pair(ungraphed, graphed);
            assert_eq!(
                v, expected,
                "ungraphed={ungraphed} graphed={graphed} expected {expected:?}"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic gap sweep — typical graph speedup ratios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_gap_sweep() {
        // Modern models typically see 1.1×–2× speedup from graph capture
        // depending on kernel count and per-launch overhead.
        let graphed = 1800_u64;
        for ratio_pct in [105_u64, 110, 115, 125, 150, 200, 300] {
            let ungraphed = graphed * ratio_pct / 100;
            let v = verdict_from_decode_time_pair(ungraphed, graphed);
            assert_eq!(
                v,
                Prof10_003Verdict::Pass,
                "{ratio_pct}% ratio (ungraphed={ungraphed}) must Pass"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Const evaluability — verdict is `pub const fn`.
    // -------------------------------------------------------------------------
    #[test]
    fn const_eval_works_in_static_context() {
        const PASS: Prof10_003Verdict = verdict_from_decode_time_pair(2400, 1800);
        const FAIL_EQUAL: Prof10_003Verdict = verdict_from_decode_time_pair(1800, 1800);
        const FAIL_INVERTED: Prof10_003Verdict = verdict_from_decode_time_pair(1000, 2000);
        const FAIL_ZERO: Prof10_003Verdict = verdict_from_decode_time_pair(0, 1800);
        assert_eq!(PASS, Prof10_003Verdict::Pass);
        assert_eq!(FAIL_EQUAL, Prof10_003Verdict::Fail);
        assert_eq!(FAIL_INVERTED, Prof10_003Verdict::Fail);
        assert_eq!(FAIL_ZERO, Prof10_003Verdict::Fail);
    }
}
