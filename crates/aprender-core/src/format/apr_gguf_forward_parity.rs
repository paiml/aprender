// SHIP-TWO-001 SHIP-007 — `apr-vs-gguf-forward-parity-v1` algorithm-level
// PARTIAL discharge.
//
// Contract: `contracts/apr-vs-gguf-forward-parity-v1.yaml` v1.0.0 PROPOSED.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md` §27 / §28
// (P3 binding criterion DECIDED: layer-3 APR/GGUF ffn_swigl ratio = 18.23×).
//
// The contract states that the canonical 7B teacher
// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) loaded into BOTH formats MUST
// produce per-layer ffn_swigl std within Q4K tolerance:
//
//   r_i = apr_layer[i].ffn_swigl_stats.std / gguf_layer[i].ffn_swigl_stats.std
//   binding: r_i ∈ [0.5, 2.0] for ALL i ∈ [0, 28)
//
// Today this FAILS at layer 3 (ratio = 18.23×) because APR's forward path
// uses `helpers::f32_matmul` while GGUF uses Q4K-fused matmul. PR E (in
// flight) replaces the f32_matmul path with the Q4K-aware fused kernel;
// once that lands, the live test passes and the contract flips to ACTIVE.
//
// What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`): the *decision rule*
// — "all 28 ratios within [0.5, 2.0]" — cannot be silently weakened. Future
// drift in the threshold or the layer count would trip the mutation survey
// tests at the bottom of this file.

/// Number of decoder layers in the canonical 7B teacher
/// (`paiml/qwen2.5-coder-7b-apache-q4k-v1`).
///
/// Derivation: Qwen2.5-Coder-7B-Instruct architecture: hidden_dim=3584,
/// num_layers=28, num_heads=28, num_kv_heads=4 (GQA-7:1). The literal `28`
/// is bound here so that drift (e.g., a regression that loads only 27 of
/// 28 layers, or a fork that adds a 29th) is caught at test time.
pub const AC_APR_GGUF_PARITY_NUM_LAYERS: usize = 28;

/// Lower bound of the `r_i` ratio band, inclusive.
///
/// Q4K tolerance: 2× variance ≈ 1.4× std → ±100% std band → [0.5, 2.0].
/// Symmetric in log-space: `log2(0.5) = -1`, `log2(2.0) = +1`.
pub const AC_APR_GGUF_PARITY_MIN_RATIO: f32 = 0.5;

/// Upper bound of the `r_i` ratio band, inclusive.
pub const AC_APR_GGUF_PARITY_MAX_RATIO: f32 = 2.0;

/// Binary verdict for FALSIFY-APR-GGUF-PARITY / per-layer ffn_swigl ratio gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AprGgufForwardParityVerdict {
    /// All `AC_APR_GGUF_PARITY_NUM_LAYERS` ratios fall within
    /// `[AC_APR_GGUF_PARITY_MIN_RATIO, AC_APR_GGUF_PARITY_MAX_RATIO]`.
    /// SHIP-007 is closed; 5 MODEL-1 PARTIALs discharge.
    Pass,
    /// One or more of:
    /// - Slice length ≠ `AC_APR_GGUF_PARITY_NUM_LAYERS`
    /// - At least one ratio outside `[MIN, MAX]`
    /// - At least one ratio is non-finite (NaN, ±∞)
    /// - At least one ratio is ≤ 0 (would imply broken std denominator)
    Fail,
}

/// Pure verdict function for the per-layer ffn_swigl ratio gate.
///
/// Returns [`AprGgufForwardParityVerdict::Pass`] iff:
/// 1. `ratios.len() == AC_APR_GGUF_PARITY_NUM_LAYERS` (exactly 28),
/// 2. every ratio is finite,
/// 3. every ratio is `> 0.0` (a non-positive ratio implies a broken std
///    denominator, not a math gap; conservative `Fail`),
/// 4. every ratio is in `[MIN, MAX]` inclusive.
///
/// Otherwise returns `Fail`. Bounds are inclusive — an exact-2.0 ratio
/// passes; `2.0 + 1 ULP` fails.
#[must_use]
pub fn verdict_from_per_layer_ratios(ratios: &[f32]) -> AprGgufForwardParityVerdict {
    if ratios.len() != AC_APR_GGUF_PARITY_NUM_LAYERS {
        return AprGgufForwardParityVerdict::Fail;
    }
    for &r in ratios {
        if !r.is_finite() || r <= 0.0 {
            return AprGgufForwardParityVerdict::Fail;
        }
        if r < AC_APR_GGUF_PARITY_MIN_RATIO || r > AC_APR_GGUF_PARITY_MAX_RATIO {
            return AprGgufForwardParityVerdict::Fail;
        }
    }
    AprGgufForwardParityVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ones() -> Vec<f32> {
        vec![1.0_f32; AC_APR_GGUF_PARITY_NUM_LAYERS]
    }

    // -------------------------------------------------------------------------
    // Section 1: Provenance pin — bound thresholds match the contract spec.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_num_layers_is_twenty_eight() {
        assert_eq!(AC_APR_GGUF_PARITY_NUM_LAYERS, 28);
    }

    #[test]
    fn provenance_min_ratio_is_half() {
        assert_eq!(AC_APR_GGUF_PARITY_MIN_RATIO, 0.5);
    }

    #[test]
    fn provenance_max_ratio_is_two() {
        assert_eq!(AC_APR_GGUF_PARITY_MAX_RATIO, 2.0);
    }

    #[test]
    fn provenance_bounds_symmetric_in_log_space() {
        // |log2(0.5)| == |log2(2.0)| == 1 — the contract's symmetric-in-log
        // invariant. Catches accidental [0.5, 1.5] or [0.6, 2.0] drift.
        let lower = AC_APR_GGUF_PARITY_MIN_RATIO.log2();
        let upper = AC_APR_GGUF_PARITY_MAX_RATIO.log2();
        assert!((lower - (-1.0)).abs() < 1e-6);
        assert!((upper - 1.0).abs() < 1e-6);
        assert!((lower + upper).abs() < 1e-6, "symmetric in log space");
    }

    // -------------------------------------------------------------------------
    // Section 2: Pass band — clearly-in-bounds inputs Pass.
    // -------------------------------------------------------------------------
    #[test]
    fn pass_all_ones() {
        assert_eq!(
            verdict_from_per_layer_ratios(&ones()),
            AprGgufForwardParityVerdict::Pass
        );
    }

    #[test]
    fn pass_all_clearly_in_band() {
        for &r in &[0.6_f32, 0.8, 1.1, 1.5, 1.9] {
            let v: Vec<f32> = vec![r; AC_APR_GGUF_PARITY_NUM_LAYERS];
            assert_eq!(
                verdict_from_per_layer_ratios(&v),
                AprGgufForwardParityVerdict::Pass,
                "ratio {r} should pass"
            );
        }
    }

    #[test]
    fn pass_inclusive_lower_boundary() {
        let v = vec![AC_APR_GGUF_PARITY_MIN_RATIO; AC_APR_GGUF_PARITY_NUM_LAYERS];
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Pass,
            "exact 0.5 must pass (inclusive lower)"
        );
    }

    #[test]
    fn pass_inclusive_upper_boundary() {
        let v = vec![AC_APR_GGUF_PARITY_MAX_RATIO; AC_APR_GGUF_PARITY_NUM_LAYERS];
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Pass,
            "exact 2.0 must pass (inclusive upper)"
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: Fail band — clearly-out-of-bounds inputs Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_all_below_band() {
        let v = vec![0.4_f32; AC_APR_GGUF_PARITY_NUM_LAYERS];
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_all_above_band() {
        let v = vec![2.1_f32; AC_APR_GGUF_PARITY_NUM_LAYERS];
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_layer_3_at_18_23x_today() {
        // Mirrors the §27 binding criterion observation: layer 3 today
        // produces ratio = 18.23×. A single-layer Fail in any position
        // must Fail the whole verdict (no per-layer carve-out).
        let mut v = ones();
        v[3] = 18.23_f32;
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail,
            "layer-3 ratio = 18.23× must fail the gate (today's state)"
        );
    }

    /// Next f32 above `x` via bit manipulation. Avoids `f32::EPSILON` which
    /// is the ULP at 1.0, not at arbitrary x — `2.0 + EPSILON` rounds back
    /// to 2.0 because the ULP at 2.0 is ≈ 2.38e-7, not 1.19e-7.
    fn next_up_f32(x: f32) -> f32 {
        f32::from_bits(x.to_bits() + 1)
    }
    fn next_down_f32(x: f32) -> f32 {
        f32::from_bits(x.to_bits() - 1)
    }

    #[test]
    fn fail_just_above_upper_boundary() {
        // 2.0 + 1 ULP → outside band.
        let above = next_up_f32(AC_APR_GGUF_PARITY_MAX_RATIO);
        assert!(above > AC_APR_GGUF_PARITY_MAX_RATIO);
        let mut v = ones();
        v[5] = above;
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_just_below_lower_boundary() {
        // 0.5 - 1 ULP → outside band.
        let below = next_down_f32(AC_APR_GGUF_PARITY_MIN_RATIO);
        assert!(below < AC_APR_GGUF_PARITY_MIN_RATIO);
        let mut v = ones();
        v[27] = below;
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Length drift — wrong-count slices Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_empty_slice() {
        assert_eq!(
            verdict_from_per_layer_ratios(&[]),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_too_few_layers() {
        let v = vec![1.0_f32; AC_APR_GGUF_PARITY_NUM_LAYERS - 1];
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_too_many_layers() {
        let v = vec![1.0_f32; AC_APR_GGUF_PARITY_NUM_LAYERS + 1];
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: Non-finite / non-positive — domain violations Fail.
    // -------------------------------------------------------------------------
    #[test]
    fn fail_nan_in_any_position() {
        for pos in [0_usize, 13, 27] {
            let mut v = ones();
            v[pos] = f32::NAN;
            assert_eq!(
                verdict_from_per_layer_ratios(&v),
                AprGgufForwardParityVerdict::Fail,
                "NaN at position {pos} must fail"
            );
        }
    }

    #[test]
    fn fail_positive_infinity() {
        let mut v = ones();
        v[7] = f32::INFINITY;
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_negative_infinity() {
        let mut v = ones();
        v[7] = f32::NEG_INFINITY;
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_zero_ratio() {
        // r=0 implies apr.std=0 (everything dead) — domain violation.
        let mut v = ones();
        v[10] = 0.0;
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    #[test]
    fn fail_negative_ratio() {
        // r<0 cannot arise from std/std (both ≥0); conservative Fail.
        let mut v = ones();
        v[10] = -1.0;
        assert_eq!(
            verdict_from_per_layer_ratios(&v),
            AprGgufForwardParityVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Single-layer mutation sweep — flipping any one layer fails.
    // -------------------------------------------------------------------------
    #[test]
    fn single_layer_mutation_at_each_index_fails() {
        for i in 0..AC_APR_GGUF_PARITY_NUM_LAYERS {
            let mut v = ones();
            v[i] = 100.0;
            assert_eq!(
                verdict_from_per_layer_ratios(&v),
                AprGgufForwardParityVerdict::Fail,
                "single bad layer at index {i} must fail the whole gate"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Monotonicity sweep — gate flips exactly once across the
    //            ratio sweep [0.0, 0.5, 0.5+ε, 0.6, 1.0, 1.4, 2.0, 2.0+ε].
    // -------------------------------------------------------------------------
    #[test]
    fn monotonicity_sweep_flips_exactly_at_boundaries() {
        let probes: Vec<(f32, AprGgufForwardParityVerdict)> = vec![
            (0.0, AprGgufForwardParityVerdict::Fail),
            (0.4, AprGgufForwardParityVerdict::Fail),
            (
                next_down_f32(AC_APR_GGUF_PARITY_MIN_RATIO),
                AprGgufForwardParityVerdict::Fail,
            ),
            (
                AC_APR_GGUF_PARITY_MIN_RATIO,
                AprGgufForwardParityVerdict::Pass,
            ),
            (0.6, AprGgufForwardParityVerdict::Pass),
            (1.0, AprGgufForwardParityVerdict::Pass),
            (1.4, AprGgufForwardParityVerdict::Pass),
            (
                AC_APR_GGUF_PARITY_MAX_RATIO,
                AprGgufForwardParityVerdict::Pass,
            ),
            (
                next_up_f32(AC_APR_GGUF_PARITY_MAX_RATIO),
                AprGgufForwardParityVerdict::Fail,
            ),
            (2.5, AprGgufForwardParityVerdict::Fail),
        ];
        for (r, expected) in probes {
            let v = vec![r; AC_APR_GGUF_PARITY_NUM_LAYERS];
            assert_eq!(
                verdict_from_per_layer_ratios(&v),
                expected,
                "ratio {r} expected {expected:?}"
            );
        }
    }
}
