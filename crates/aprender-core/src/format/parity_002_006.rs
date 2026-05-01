// SHIP-TWO-001 — `apr-vs-gguf-forward-parity-v1` algorithm-level
// PARTIAL discharge for FALSIFY-APR-GGUF-PARITY-002..006.
// Closes 6/6 sweep (001 already bound separately).
//
// Contract: `contracts/apr-vs-gguf-forward-parity-v1.yaml`.
// Spec: SHIP-007 sample-size-parity gate per §37.5.

// ===========================================================================
// PARITY-002 — layer 3 ffn_swigl std ratio in [0.5, 2.0]
// ===========================================================================

pub const AC_PARITY_002_RATIO_LOW: f64 = 0.5;
pub const AC_PARITY_002_RATIO_HIGH: f64 = 2.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity002Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-GGUF-PARITY-002`.
///
/// Pass iff `ratio` is finite AND in `[0.5, 2.0]` inclusive.
#[must_use]
pub fn verdict_from_ffn_swigl_ratio(ratio: f64) -> Parity002Verdict {
    if !ratio.is_finite() {
        return Parity002Verdict::Fail;
    }
    if (AC_PARITY_002_RATIO_LOW..=AC_PARITY_002_RATIO_HIGH).contains(&ratio) {
        Parity002Verdict::Pass
    } else {
        Parity002Verdict::Fail
    }
}

// ===========================================================================
// PARITY-003 — layer 3 ffn_gate std ratio in [0.7, 1.4]
// ===========================================================================

pub const AC_PARITY_003_RATIO_LOW: f64 = 0.7;
pub const AC_PARITY_003_RATIO_HIGH: f64 = 1.4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity003Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-GGUF-PARITY-003`.
///
/// Pass iff `ratio` is finite AND in `[0.7, 1.4]` inclusive.
/// Tighter band than 002 because gate matmul precision is the
/// pinned root cause.
#[must_use]
pub fn verdict_from_ffn_gate_ratio(ratio: f64) -> Parity003Verdict {
    if !ratio.is_finite() {
        return Parity003Verdict::Fail;
    }
    if (AC_PARITY_003_RATIO_LOW..=AC_PARITY_003_RATIO_HIGH).contains(&ratio) {
        Parity003Verdict::Pass
    } else {
        Parity003Verdict::Fail
    }
}

// ===========================================================================
// PARITY-004 + 005 — pv validate / cargo test exit-code-zero
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity004005Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-GGUF-PARITY-004` (pv
/// validate) AND `005` (cargo test f32_path_unchanged).
///
/// Pass iff `exit_code == 0`.
#[must_use]
pub fn verdict_from_pv_or_cargo_exit(exit_code: i32) -> Parity004005Verdict {
    if exit_code == 0 {
        Parity004005Verdict::Pass
    } else {
        Parity004005Verdict::Fail
    }
}

// ===========================================================================
// PARITY-006 — apr trace --payload emits ffn_swigl >= 28 lines
// ===========================================================================

pub const AC_PARITY_006_MIN_LINES: u64 = 28;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parity006Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-APR-GGUF-PARITY-006`.
///
/// Pass iff `ffn_swigl_line_count >= 28` (one per layer of the
/// canonical 28-layer Qwen2.5-Coder-7B teacher).
#[must_use]
pub fn verdict_from_ffn_swigl_line_count(ffn_swigl_line_count: u64) -> Parity006Verdict {
    if ffn_swigl_line_count >= AC_PARITY_006_MIN_LINES {
        Parity006Verdict::Pass
    } else {
        Parity006Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // PARITY-002 -----------------------------------------------------------------
    #[test]
    fn p002_provenance_band() {
        assert_eq!(AC_PARITY_002_RATIO_LOW, 0.5);
        assert_eq!(AC_PARITY_002_RATIO_HIGH, 2.0);
    }

    #[test]
    fn p002_pass_at_unity() {
        assert_eq!(verdict_from_ffn_swigl_ratio(1.0), Parity002Verdict::Pass);
    }

    #[test]
    fn p002_pass_at_low_boundary() {
        assert_eq!(verdict_from_ffn_swigl_ratio(0.5), Parity002Verdict::Pass);
    }

    #[test]
    fn p002_pass_at_high_boundary() {
        assert_eq!(verdict_from_ffn_swigl_ratio(2.0), Parity002Verdict::Pass);
    }

    #[test]
    fn p002_fail_below_floor() {
        assert_eq!(verdict_from_ffn_swigl_ratio(0.49), Parity002Verdict::Fail);
    }

    #[test]
    fn p002_fail_above_ceiling() {
        assert_eq!(verdict_from_ffn_swigl_ratio(2.01), Parity002Verdict::Fail);
    }

    #[test]
    fn p002_fail_18_23x_ship_007_baseline() {
        // Per memory `2026-04-26 session SHIP-007`: layer-3 ffn_swigl
        // ratio was 17× (later refined to 18.23×) — must Fail.
        assert_eq!(verdict_from_ffn_swigl_ratio(18.23), Parity002Verdict::Fail);
    }

    #[test]
    fn p002_fail_nan() {
        assert_eq!(verdict_from_ffn_swigl_ratio(f64::NAN), Parity002Verdict::Fail);
    }

    #[test]
    fn p002_fail_infinity() {
        assert_eq!(verdict_from_ffn_swigl_ratio(f64::INFINITY), Parity002Verdict::Fail);
    }

    // PARITY-003 -----------------------------------------------------------------
    #[test]
    fn p003_provenance_band() {
        assert_eq!(AC_PARITY_003_RATIO_LOW, 0.7);
        assert_eq!(AC_PARITY_003_RATIO_HIGH, 1.4);
    }

    #[test]
    fn p003_pass_at_unity() {
        assert_eq!(verdict_from_ffn_gate_ratio(1.0), Parity003Verdict::Pass);
    }

    #[test]
    fn p003_pass_at_low_boundary() {
        assert_eq!(verdict_from_ffn_gate_ratio(0.7), Parity003Verdict::Pass);
    }

    #[test]
    fn p003_pass_at_high_boundary() {
        assert_eq!(verdict_from_ffn_gate_ratio(1.4), Parity003Verdict::Pass);
    }

    #[test]
    fn p003_fail_below_floor() {
        assert_eq!(verdict_from_ffn_gate_ratio(0.69), Parity003Verdict::Fail);
    }

    #[test]
    fn p003_fail_above_ceiling() {
        assert_eq!(verdict_from_ffn_gate_ratio(1.41), Parity003Verdict::Fail);
    }

    #[test]
    fn p003_band_is_tighter_than_p002() {
        // p003's band [0.7, 1.4] is strictly inside p002's [0.5, 2.0]
        // because gate matmul is the pinned root cause.
        assert!(AC_PARITY_003_RATIO_LOW > AC_PARITY_002_RATIO_LOW);
        assert!(AC_PARITY_003_RATIO_HIGH < AC_PARITY_002_RATIO_HIGH);
    }

    // PARITY-004 + 005 ----------------------------------------------------------
    #[test]
    fn p004_005_pass_exit_zero() {
        assert_eq!(verdict_from_pv_or_cargo_exit(0), Parity004005Verdict::Pass);
    }

    #[test]
    fn p004_005_fail_exit_one() {
        assert_eq!(verdict_from_pv_or_cargo_exit(1), Parity004005Verdict::Fail);
    }

    #[test]
    fn p004_005_fail_panic_101() {
        assert_eq!(verdict_from_pv_or_cargo_exit(101), Parity004005Verdict::Fail);
    }

    // PARITY-006 ----------------------------------------------------------------
    #[test]
    fn p006_provenance_28_lines() {
        assert_eq!(AC_PARITY_006_MIN_LINES, 28);
    }

    #[test]
    fn p006_pass_exact_28_lines() {
        assert_eq!(verdict_from_ffn_swigl_line_count(28), Parity006Verdict::Pass);
    }

    #[test]
    fn p006_pass_above_28() {
        assert_eq!(verdict_from_ffn_swigl_line_count(56), Parity006Verdict::Pass);
    }

    #[test]
    fn p006_fail_27_lines() {
        assert_eq!(verdict_from_ffn_swigl_line_count(27), Parity006Verdict::Fail);
    }

    #[test]
    fn p006_fail_zero_lines() {
        assert_eq!(verdict_from_ffn_swigl_line_count(0), Parity006Verdict::Fail);
    }
}
