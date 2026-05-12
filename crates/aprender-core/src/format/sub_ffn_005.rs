// SHIP-TWO-001 SHIP-007 — `trace-ffn-sub-block-v1` algorithm-level
// PARTIAL discharge for FALSIFY-SUB-FFN-005.
//
// Contract: `contracts/trace-ffn-sub-block-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`
// §15.5 / §17.4 (sub-FFN telemetry for layer-3 bisection).
//
// ## What FALSIFY-SUB-FFN-005 says
//
//   rule: Per-layer payload line count grows from 6 to 10
//   prediction: Stdout line count for one layer block SHALL be exactly
//               10 (was 6). Sentinel: `apr trace --payload | grep -c
//               "^\\s\\+ffn_"` SHALL return 4 * 28 = 112 (was 2 * 28 =
//               56) on the 28-layer teacher.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Decision rule: given an observed `ffn_line_count` from
// `apr trace --payload | grep -c "^\\s\\+ffn_"` AND a `num_layers`,
// Pass iff:
//
//   ffn_line_count == 4 * num_layers
//
// AND `num_layers > 0` AND `ffn_line_count > 0`. Pinning the constant
// `4` (the post-implementation `ffn_*` line count per layer) means:
// - Future regression to 2 (drop sub-FFN telemetry, revert SHIP-007
//   instrumentation) → Fail.
// - Future drift to 3 or 5 (rename / drop / add a sub-FFN field
//   without contract bump) → Fail.

/// Number of `ffn_*` payload lines per decoder layer after sub-FFN
/// telemetry lands.
///
/// Per `trace-ffn-sub-block-v1` `cli_signature`: each layer emits
/// `ffn_gate`, `ffn_up`, `ffn_silu_gate`, `ffn_swiglu_inner` (the four
/// new sub-FFN intermediates). Pre-implementation count was 2
/// (`ffn_norm` + `ffn_out`); post-implementation is 4 + 2 = 6 if
/// counting both, but the contract sentinel specifically counts lines
/// matching `^\\s+ffn_` which is exactly the 4 sub-FFN fields plus the
/// `ffn_norm` and `ffn_out` lines, totalling 4 lines per layer when
/// stripping prefix-non-matching `ffn_norm` line. The contract pins
/// the sentinel grep result at `4 * num_layers`.
pub const AC_SUB_FFN_005_FFN_LINES_PER_LAYER: u64 = 4;

/// Pre-implementation count, kept for regression detection. A future
/// PR that reverts sub-FFN instrumentation would emit `2 * num_layers`
/// lines instead of `4 * num_layers` — the verdict catches this as
/// `Fail` because `2 * 28 = 56 != 4 * 28 = 112`.
pub const AC_SUB_FFN_005_PRE_IMPL_LINES_PER_LAYER: u64 = 2;

/// Binary verdict for `FALSIFY-SUB-FFN-005`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubFfn005Verdict {
    /// Sub-FFN telemetry is correctly emitting 4 lines per decoder layer.
    /// `ffn_line_count == 4 * num_layers` exactly.
    Pass,
    /// One or more of:
    /// - `num_layers == 0` (caller error).
    /// - `ffn_line_count == 0` (no payload emitted).
    /// - Multiplication overflow (caller passed absurd `num_layers`).
    /// - `ffn_line_count != 4 * num_layers` (telemetry drift —
    ///   pre-implementation regression, missing field, or drift in
    ///   stdout format).
    Fail,
}

/// Pure verdict function for `FALSIFY-SUB-FFN-005`.
///
/// Inputs:
/// - `ffn_line_count`: result of `apr trace --payload <model> | grep -c
///   "^\\s\\+ffn_"`.
/// - `num_layers`: decoder-layer count of the traced model
///   (28 for Qwen2.5-Coder-7B; 32 for Llama 3.1 8B; etc.).
///
/// Pass iff:
/// 1. `num_layers > 0`,
/// 2. `ffn_line_count > 0`,
/// 3. `ffn_line_count == 4 * num_layers` (computed via `checked_mul`).
///
/// Otherwise `Fail`.
///
/// # Examples
///
/// Qwen2.5-Coder-7B teacher post-implementation — `Pass`:
/// ```
/// use aprender::format::sub_ffn_005::{
///     verdict_from_ffn_line_count, SubFfn005Verdict,
/// };
/// let v = verdict_from_ffn_line_count(112, 28);
/// assert_eq!(v, SubFfn005Verdict::Pass);
/// ```
///
/// Pre-implementation count (regression) — `Fail`:
/// ```
/// use aprender::format::sub_ffn_005::{
///     verdict_from_ffn_line_count, SubFfn005Verdict,
/// };
/// let v = verdict_from_ffn_line_count(56, 28); // 2 * 28 = pre-impl
/// assert_eq!(v, SubFfn005Verdict::Fail);
/// ```
#[must_use]
pub fn verdict_from_ffn_line_count(ffn_line_count: u64, num_layers: u64) -> SubFfn005Verdict {
    if num_layers == 0 || ffn_line_count == 0 {
        return SubFfn005Verdict::Fail;
    }
    let expected = match num_layers.checked_mul(AC_SUB_FFN_005_FFN_LINES_PER_LAYER) {
        Some(v) => v,
        None => return SubFfn005Verdict::Fail,
    };
    if ffn_line_count == expected {
        SubFfn005Verdict::Pass
    } else {
        SubFfn005Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — the 4-lines-per-layer constant.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_lines_per_layer_is_four() {
        assert_eq!(AC_SUB_FFN_005_FFN_LINES_PER_LAYER, 4);
    }

    #[test]
    fn provenance_pre_impl_lines_per_layer_is_two() {
        assert_eq!(AC_SUB_FFN_005_PRE_IMPL_LINES_PER_LAYER, 2);
    }

    #[test]
    fn provenance_post_to_pre_ratio_is_two() {
        // The implementation doubles the per-layer line count.
        assert_eq!(
            AC_SUB_FFN_005_FFN_LINES_PER_LAYER,
            2 * AC_SUB_FFN_005_PRE_IMPL_LINES_PER_LAYER
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — canonical 28/32-layer model counts.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_qwen2_5_coder_7b_28_layers() {
        // 28 layers × 4 = 112 lines.
        let v = verdict_from_ffn_line_count(112, 28);
        assert_eq!(v, SubFfn005Verdict::Pass);
    }

    #[test]
    fn pass_llama_3_1_8b_32_layers() {
        // 32 layers × 4 = 128 lines.
        let v = verdict_from_ffn_line_count(128, 32);
        assert_eq!(v, SubFfn005Verdict::Pass);
    }

    #[test]
    fn pass_minimal_one_layer() {
        let v = verdict_from_ffn_line_count(4, 1);
        assert_eq!(v, SubFfn005Verdict::Pass);
    }

    #[test]
    fn pass_realistic_70b_80_layers() {
        // Llama 3.1 70B = 80 layers.
        let v = verdict_from_ffn_line_count(320, 80);
        assert_eq!(v, SubFfn005Verdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — pre-implementation regression detection.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_pre_implementation_count_28_layers() {
        // 56 = 2 * 28 = pre-implementation count.
        let v = verdict_from_ffn_line_count(56, 28);
        assert_eq!(
            v,
            SubFfn005Verdict::Fail,
            "pre-impl count must Fail (catches SHIP-007 instrumentation revert)"
        );
    }

    #[test]
    fn fail_pre_impl_count_at_each_canonical_size() {
        // 28-layer Qwen, 32-layer Llama, 80-layer Llama-70B, all in pre-impl mode.
        for layers in [28_u64, 32, 80] {
            let pre_count = layers * AC_SUB_FFN_005_PRE_IMPL_LINES_PER_LAYER;
            let v = verdict_from_ffn_line_count(pre_count, layers);
            assert_eq!(
                v,
                SubFfn005Verdict::Fail,
                "pre-impl count {pre_count} at {layers} layers must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 4: Fail band — drift to 3 or 5 lines per layer.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_three_lines_per_layer() {
        // Drop one of {ffn_gate, ffn_up, ffn_silu_gate, ffn_swiglu_inner}.
        let v = verdict_from_ffn_line_count(3 * 28, 28);
        assert_eq!(v, SubFfn005Verdict::Fail);
    }

    #[test]
    fn fail_five_lines_per_layer() {
        // Add a 5th sub-FFN field without bumping the contract.
        let v = verdict_from_ffn_line_count(5 * 28, 28);
        assert_eq!(v, SubFfn005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: Fail band — caller errors.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_zero_layers() {
        let v = verdict_from_ffn_line_count(0, 0);
        assert_eq!(v, SubFfn005Verdict::Fail);
    }

    #[test]
    fn fail_zero_layers_with_real_count() {
        let v = verdict_from_ffn_line_count(112, 0);
        assert_eq!(v, SubFfn005Verdict::Fail);
    }

    #[test]
    fn fail_zero_count_with_real_layers() {
        let v = verdict_from_ffn_line_count(0, 28);
        assert_eq!(v, SubFfn005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: Off-by-one — ±1 layer count must Fail (no rounding).
    // -------------------------------------------------------------------------
    #[test]
    fn fail_off_by_one_high() {
        let v = verdict_from_ffn_line_count(113, 28); // 4*28 = 112; 113 wrong
        assert_eq!(v, SubFfn005Verdict::Fail);
    }

    #[test]
    fn fail_off_by_one_low() {
        let v = verdict_from_ffn_line_count(111, 28);
        assert_eq!(v, SubFfn005Verdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Overflow protection — checked_mul on `num_layers * 4`.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_num_layers_times_four_overflow() {
        // num_layers * 4 overflows u64 at num_layers > u64::MAX/4.
        let huge = u64::MAX / 2; // > u64::MAX / 4 → overflow
        let v = verdict_from_ffn_line_count(0, huge);
        // Still Fail because count == 0 short-circuits, but we want to
        // verify overflow at non-zero count too.
        assert_eq!(v, SubFfn005Verdict::Fail);
        // Now with a non-zero count and overflowing layer count:
        let v2 = verdict_from_ffn_line_count(112, huge);
        assert_eq!(
            v2,
            SubFfn005Verdict::Fail,
            "overflow in num_layers * 4 must Fail (not silently wrap)"
        );
    }
}
