// SHIP-TWO-001 — `qwen3-moe-forward-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QW3-MOE-FORWARD-001..004 (closes 4/4).
//
// Contract: `contracts/qwen3-moe-forward-v1.yaml`.
// Spec: M32d Qwen3-MoE forward parity per memory `2026-04-28
// session distillation track complete`.

// ===========================================================================
// QW3-MOE-FORWARD-001 — baseline error message exact match
// ===========================================================================

/// Expected baseline error message at the M32a-precursor commit
/// `15d504cfe`. Per contract: `apr run` against a qwen3_moe GGUF
/// must emit exactly this string when dense-FFN tensor lookup is
/// reached architecture-blind.
pub const AC_QW3_MOE_001_EXPECTED_BASELINE_ERROR: &[u8] =
    b"Invalid shape: Tensor 'blk.0.ffn_up.weight' not found";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3MoeForward001Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-QW3-MOE-FORWARD-001`.
///
/// Pass iff `stderr_or_stdout` contains the canonical baseline
/// error string. This is a **regression sentinel**: when M32b
/// lands, this gate flips polarity (the verdict pinned here
/// represents the pre-M32b reproduction state).
#[must_use]
pub fn verdict_from_baseline_error_string(output: &[u8]) -> Qw3MoeForward001Verdict {
    if contains_subseq(output, AC_QW3_MOE_001_EXPECTED_BASELINE_ERROR) {
        Qw3MoeForward001Verdict::Pass
    } else {
        Qw3MoeForward001Verdict::Fail
    }
}

// ===========================================================================
// QW3-MOE-FORWARD-002 — UnsupportedOperation forward-pass error
// ===========================================================================

/// Expected M32b error string after arch-aware load is wired but
/// before forward is implemented.
pub const AC_QW3_MOE_002_UNSUPPORTED_OP: &[u8] = b"moe_forward_pass";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3MoeForward002Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-QW3-MOE-FORWARD-002`.
///
/// Pass iff:
/// - output does NOT contain dense-FFN error (`ffn_up.weight not
///   found`), AND
/// - output contains `moe_forward_pass` (the M32b-state error).
#[must_use]
pub fn verdict_from_m32b_unsupported_state(output: &[u8]) -> Qw3MoeForward002Verdict {
    let has_dense_error = contains_subseq(
        output,
        b"ffn_up.weight",
    );
    let has_unsupported = contains_subseq(output, AC_QW3_MOE_002_UNSUPPORTED_OP);
    if !has_dense_error && has_unsupported {
        Qw3MoeForward002Verdict::Pass
    } else {
        Qw3MoeForward002Verdict::Fail
    }
}

// ===========================================================================
// QW3-MOE-FORWARD-003 — apr run produces tokens (exit 0 + non-whitespace)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3MoeForward003Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-QW3-MOE-FORWARD-003`.
///
/// Pass iff `exit_code == 0 AND` stdout contains at least one
/// non-whitespace byte (any token; correctness not yet asserted).
#[must_use]
pub fn verdict_from_apr_run_produces_tokens(
    exit_code: i32,
    stdout: &[u8],
) -> Qw3MoeForward003Verdict {
    if exit_code != 0 {
        return Qw3MoeForward003Verdict::Fail;
    }
    if stdout.iter().any(|b| !b.is_ascii_whitespace()) {
        Qw3MoeForward003Verdict::Pass
    } else {
        Qw3MoeForward003Verdict::Fail
    }
}

// ===========================================================================
// QW3-MOE-FORWARD-004 — cosine similarity vs HF FP16 > 0.99
// ===========================================================================

pub const AC_QW3_MOE_004_MIN_COSINE: f64 = 0.99;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qw3MoeForward004Verdict {
    Pass,
    Fail,
}

/// Pure verdict function for `FALSIFY-QW3-MOE-FORWARD-004`.
///
/// Pass iff `cosine_similarity` is finite AND `> 0.99` strict.
#[must_use]
pub fn verdict_from_hf_fp16_cosine(cosine_similarity: f64) -> Qw3MoeForward004Verdict {
    if !cosine_similarity.is_finite() {
        return Qw3MoeForward004Verdict::Fail;
    }
    if !(-1.0..=1.0).contains(&cosine_similarity) {
        return Qw3MoeForward004Verdict::Fail;
    }
    if cosine_similarity > AC_QW3_MOE_004_MIN_COSINE {
        Qw3MoeForward004Verdict::Pass
    } else {
        Qw3MoeForward004Verdict::Fail
    }
}

// ===========================================================================
// Shared primitive
// ===========================================================================

#[must_use]
fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    // QW3-MOE-FORWARD-001 -------------------------------------------------------
    #[test]
    fn q001_provenance_baseline_error_string() {
        assert_eq!(
            AC_QW3_MOE_001_EXPECTED_BASELINE_ERROR,
            b"Invalid shape: Tensor 'blk.0.ffn_up.weight' not found"
        );
    }

    #[test]
    fn q001_pass_baseline_error_present() {
        let output = b"thread 'main' panicked at 'Invalid shape: Tensor 'blk.0.ffn_up.weight' not found'";
        assert_eq!(verdict_from_baseline_error_string(output), Qw3MoeForward001Verdict::Pass);
    }

    #[test]
    fn q001_fail_no_error() {
        assert_eq!(verdict_from_baseline_error_string(b"clean output"), Qw3MoeForward001Verdict::Fail);
    }

    // QW3-MOE-FORWARD-002 -------------------------------------------------------
    #[test]
    fn q002_pass_m32b_state() {
        // No dense-FFN error AND has moe_forward_pass error.
        let output = b"RealizarError::UnsupportedOperation { operation: \"moe_forward_pass\" }";
        assert_eq!(verdict_from_m32b_unsupported_state(output), Qw3MoeForward002Verdict::Pass);
    }

    #[test]
    fn q002_fail_still_dense_error() {
        // Both errors present (regression to pre-M32b state).
        let output = b"ffn_up.weight not found ... moe_forward_pass also referenced";
        assert_eq!(verdict_from_m32b_unsupported_state(output), Qw3MoeForward002Verdict::Fail);
    }

    #[test]
    fn q002_fail_no_unsupported_message() {
        let output = b"some other error";
        assert_eq!(verdict_from_m32b_unsupported_state(output), Qw3MoeForward002Verdict::Fail);
    }

    // QW3-MOE-FORWARD-003 -------------------------------------------------------
    #[test]
    fn q003_pass_token_emitted() {
        assert_eq!(verdict_from_apr_run_produces_tokens(0, b"4"), Qw3MoeForward003Verdict::Pass);
    }

    #[test]
    fn q003_pass_real_response() {
        assert_eq!(
            verdict_from_apr_run_produces_tokens(0, b"The answer is 4."),
            Qw3MoeForward003Verdict::Pass
        );
    }

    #[test]
    fn q003_fail_exit_nonzero() {
        assert_eq!(verdict_from_apr_run_produces_tokens(1, b"4"), Qw3MoeForward003Verdict::Fail);
    }

    #[test]
    fn q003_fail_whitespace_only_stdout() {
        assert_eq!(
            verdict_from_apr_run_produces_tokens(0, b"   \n  \t  "),
            Qw3MoeForward003Verdict::Fail
        );
    }

    #[test]
    fn q003_fail_empty_stdout() {
        assert_eq!(verdict_from_apr_run_produces_tokens(0, &[]), Qw3MoeForward003Verdict::Fail);
    }

    // QW3-MOE-FORWARD-004 -------------------------------------------------------
    #[test]
    fn q004_provenance_min_cosine_is_0_99() {
        assert_eq!(AC_QW3_MOE_004_MIN_COSINE, 0.99);
    }

    #[test]
    fn q004_pass_perfect_match() {
        assert_eq!(verdict_from_hf_fp16_cosine(1.0), Qw3MoeForward004Verdict::Pass);
    }

    #[test]
    fn q004_pass_above_floor() {
        assert_eq!(verdict_from_hf_fp16_cosine(0.995), Qw3MoeForward004Verdict::Pass);
    }

    #[test]
    fn q004_fail_at_exact_floor() {
        // Strict >: 0.99 itself fails.
        assert_eq!(verdict_from_hf_fp16_cosine(0.99), Qw3MoeForward004Verdict::Fail);
    }

    #[test]
    fn q004_fail_below_floor() {
        assert_eq!(verdict_from_hf_fp16_cosine(0.98), Qw3MoeForward004Verdict::Fail);
    }

    #[test]
    fn q004_fail_negative_correlation() {
        assert_eq!(verdict_from_hf_fp16_cosine(-0.5), Qw3MoeForward004Verdict::Fail);
    }

    #[test]
    fn q004_fail_nan() {
        assert_eq!(verdict_from_hf_fp16_cosine(f64::NAN), Qw3MoeForward004Verdict::Fail);
    }

    #[test]
    fn q004_fail_above_one() {
        // Cosine cannot exceed 1.0 by definition.
        assert_eq!(verdict_from_hf_fp16_cosine(1.001), Qw3MoeForward004Verdict::Fail);
    }
}
