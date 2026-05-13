// SHIP-TWO-001 — `apr-format-safety-v1` algorithm-level PARTIAL
// discharge for FALSIFY-FMT-001..008.
//
// Contract: `contracts/apr-format-safety-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Eight format-safety gates (the security surface of apr-cli):
//
// - FMT-001 (magic byte panic safety on empty input).
// - FMT-002 (magic byte panic safety on truncated < 4 bytes).
// - FMT-003 (header allocation bounded — reject u64::MAX tensor count).
// - FMT-004 (provenance enforcement — exit 5 when enforced + missing hash).
// - FMT-005 (dtype coercion preserves shape — Q4_0 of [768, 3072] is still [768, 3072]).
// - FMT-006 (truncation detected — file size != header expected).
// - FMT-007 (strict import rejects NaN tensors).
// - FMT-008 (no silent F32 → F16 overflow — must warn or error).

/// GGUF magic bytes.
pub const AC_FMT_GGUF_MAGIC: [u8; 4] = *b"GGUF";

/// APR v2 magic bytes.
pub const AC_FMT_APR_V2_MAGIC: [u8; 4] = *b"APR\x02";

/// SafeTensors uses an 8-byte LE u64 header length prefix; we accept
/// the first 8 bytes as the discriminant.
pub const AC_FMT_SAFETENSORS_HEADER_LEN_BYTES: usize = 8;

/// Maximum tensor count (sanity cap to prevent OOM from claimed counts
/// in headers). Realistic 7B models have ~339 tensors; cap is 4M.
pub const AC_FMT_003_MAX_TENSOR_COUNT: u64 = 4_000_000;

/// Provenance-error exit code.
pub const AC_FMT_004_PROVENANCE_EXIT_CODE: i32 = 5;

/// F16 maximum representable magnitude (≈ 65504).
pub const AC_FMT_008_F16_MAX: f32 = 65504.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    Gguf,
    Apr,
    SafeTensors,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FmtVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// Detect format from leading bytes. Never panics on empty/truncated input.
#[must_use]
pub fn detect_format(bytes: &[u8]) -> ModelFormat {
    if bytes.len() < 4 {
        return ModelFormat::Unknown;
    }
    if bytes[0..4] == AC_FMT_GGUF_MAGIC {
        return ModelFormat::Gguf;
    }
    if bytes[0..4] == AC_FMT_APR_V2_MAGIC {
        return ModelFormat::Apr;
    }
    // SafeTensors: first 8 bytes form a LE u64 header length. We
    // require >= 8 bytes AND the implied length must not exceed
    // an absurd cap.
    if bytes.len() >= AC_FMT_SAFETENSORS_HEADER_LEN_BYTES {
        let mut buf = [0_u8; 8];
        buf.copy_from_slice(&bytes[0..8]);
        let header_len = u64::from_le_bytes(buf);
        if header_len > 0 && header_len < (1 << 32) {
            return ModelFormat::SafeTensors;
        }
    }
    ModelFormat::Unknown
}

// -----------------------------------------------------------------------------
// Verdict 1: FMT-001 — magic byte detection on empty input.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_empty_input_safety(bytes: &[u8]) -> FmtVerdict {
    if !bytes.is_empty() {
        return FmtVerdict::Pass; // not the empty case
    }
    if detect_format(bytes) == ModelFormat::Unknown {
        FmtVerdict::Pass
    } else {
        FmtVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: FMT-002 — magic byte detection on truncated < 4 bytes.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_truncated_input_safety(bytes: &[u8]) -> FmtVerdict {
    if bytes.len() >= 4 {
        return FmtVerdict::Pass; // not the truncated case
    }
    if detect_format(bytes) == ModelFormat::Unknown {
        FmtVerdict::Pass
    } else {
        FmtVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: FMT-003 — header allocation bounded.
// -----------------------------------------------------------------------------

/// Pass iff `claimed_tensor_count <= AC_FMT_003_MAX_TENSOR_COUNT`.
#[must_use]
pub fn verdict_from_tensor_count_bound(claimed_tensor_count: u64) -> FmtVerdict {
    if claimed_tensor_count <= AC_FMT_003_MAX_TENSOR_COUNT {
        FmtVerdict::Pass
    } else {
        FmtVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: FMT-004 — provenance enforcement.
// -----------------------------------------------------------------------------

/// `enforce` is the --enforce-provenance flag (default true).
/// `has_hash` indicates the model has base_model_hash metadata.
/// `exit_code` is what apr returned.
#[must_use]
pub fn verdict_from_provenance_enforcement(
    enforce: bool,
    has_hash: bool,
    exit_code: i32,
) -> FmtVerdict {
    match (enforce, has_hash) {
        // Enforced + missing hash MUST be exit 5.
        (true, false) => {
            if exit_code == AC_FMT_004_PROVENANCE_EXIT_CODE {
                FmtVerdict::Pass
            } else {
                FmtVerdict::Fail
            }
        }
        // Otherwise: provenance check is OK regardless of exit code,
        // EXCEPT exit_code 5 is reserved — must NOT be used spuriously.
        (true, true) | (false, _) => {
            if exit_code == AC_FMT_004_PROVENANCE_EXIT_CODE {
                FmtVerdict::Fail
            } else {
                FmtVerdict::Pass
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: FMT-005 — dtype coercion preserves shape.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_coercion_shape_preserved(
    input_shape: &[usize],
    output_shape: &[usize],
) -> FmtVerdict {
    if input_shape == output_shape {
        FmtVerdict::Pass
    } else {
        FmtVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 6: FMT-006 — truncation detected.
// -----------------------------------------------------------------------------

/// Pass iff actual_size == expected_size. Truncation OR over-extension
/// must Fail.
#[must_use]
pub fn verdict_from_truncation_detected(actual_size: u64, expected_size: u64) -> FmtVerdict {
    if expected_size == 0 {
        return FmtVerdict::Fail;
    }
    if actual_size == expected_size {
        FmtVerdict::Pass
    } else {
        FmtVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 7: FMT-007 — strict import rejects NaN tensors.
// -----------------------------------------------------------------------------

/// Pass iff:
///   - strict=false (no requirement),
///     OR
///   - strict=true AND no NaN/Inf in tensor.
#[must_use]
pub fn verdict_from_strict_rejects_nan(strict: bool, tensor: &[f32]) -> FmtVerdict {
    if !strict {
        return FmtVerdict::Pass;
    }
    for &v in tensor {
        if !v.is_finite() {
            return FmtVerdict::Fail;
        }
    }
    FmtVerdict::Pass
}

// -----------------------------------------------------------------------------
// Verdict 8: FMT-008 — no silent F32 → F16 overflow.
// -----------------------------------------------------------------------------

/// Pass iff:
///   - F32 value within F16 range, OR
///   - F32 value out of range AND `warning_emitted=true`.
#[must_use]
pub fn verdict_from_no_silent_f32_to_f16_overflow(
    f32_value: f32,
    warning_emitted: bool,
) -> FmtVerdict {
    if !f32_value.is_finite() {
        // NaN/Inf is a separate case; FMT-007 covers it.
        return FmtVerdict::Fail;
    }
    if f32_value.abs() <= AC_FMT_008_F16_MAX {
        // Within range — no warning required.
        FmtVerdict::Pass
    } else if warning_emitted {
        FmtVerdict::Pass
    } else {
        FmtVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_gguf_magic() {
        assert_eq!(&AC_FMT_GGUF_MAGIC, b"GGUF");
    }

    #[test]
    fn provenance_apr_v2_magic() {
        assert_eq!(&AC_FMT_APR_V2_MAGIC, b"APR\x02");
    }

    #[test]
    fn provenance_max_tensor_count_4m() {
        assert_eq!(AC_FMT_003_MAX_TENSOR_COUNT, 4_000_000);
    }

    #[test]
    fn provenance_provenance_exit_code_5() {
        assert_eq!(AC_FMT_004_PROVENANCE_EXIT_CODE, 5);
    }

    #[test]
    fn provenance_f16_max_65504() {
        assert_eq!(AC_FMT_008_F16_MAX, 65504.0);
    }

    // -------------------------------------------------------------------------
    // Section 2: detect_format reference behavior.
    // -------------------------------------------------------------------------
    #[test]
    fn detect_empty_unknown() {
        assert_eq!(detect_format(&[]), ModelFormat::Unknown);
    }

    #[test]
    fn detect_short_unknown() {
        assert_eq!(detect_format(b"GGU"), ModelFormat::Unknown);
        assert_eq!(detect_format(b"AP"), ModelFormat::Unknown);
    }

    #[test]
    fn detect_gguf() {
        let mut bytes = Vec::from(b"GGUF" as &[u8]);
        bytes.extend_from_slice(&[0_u8; 24]);
        assert_eq!(detect_format(&bytes), ModelFormat::Gguf);
    }

    #[test]
    fn detect_apr_v2() {
        let mut bytes = Vec::from(b"APR\x02" as &[u8]);
        bytes.extend_from_slice(&[0_u8; 24]);
        assert_eq!(detect_format(&bytes), ModelFormat::Apr);
    }

    #[test]
    fn detect_safetensors() {
        // SafeTensors header: 8-byte LE u64 = 100 (header length).
        let mut bytes = vec![100_u8, 0, 0, 0, 0, 0, 0, 0]; // 100 little-endian
        bytes.extend_from_slice(&[0_u8; 200]);
        assert_eq!(detect_format(&bytes), ModelFormat::SafeTensors);
    }

    // -------------------------------------------------------------------------
    // Section 3: FMT-001 — empty input.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt001_pass_empty_returns_unknown() {
        assert_eq!(verdict_from_empty_input_safety(&[]), FmtVerdict::Pass);
    }

    #[test]
    fn fmt001_pass_non_empty_skipped() {
        // Non-empty: verdict trivially passes (gate doesn't apply).
        assert_eq!(
            verdict_from_empty_input_safety(b"GGUF"),
            FmtVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: FMT-002 — truncated < 4 bytes.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt002_pass_three_bytes() {
        assert_eq!(
            verdict_from_truncated_input_safety(b"GGU"),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt002_pass_one_byte() {
        assert_eq!(verdict_from_truncated_input_safety(b"G"), FmtVerdict::Pass);
    }

    #[test]
    fn fmt002_pass_full_4_bytes_skipped() {
        // ≥ 4 bytes: verdict trivially passes (gate doesn't apply).
        let bytes = b"GGUF\x00\x00\x00\x00";
        assert_eq!(
            verdict_from_truncated_input_safety(bytes),
            FmtVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: FMT-003 — tensor count bounded.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt003_pass_realistic_7b_count() {
        assert_eq!(verdict_from_tensor_count_bound(339), FmtVerdict::Pass);
    }

    #[test]
    fn fmt003_pass_at_cap() {
        assert_eq!(
            verdict_from_tensor_count_bound(4_000_000),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt003_fail_above_cap() {
        assert_eq!(
            verdict_from_tensor_count_bound(4_000_001),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt003_fail_u64_max() {
        assert_eq!(verdict_from_tensor_count_bound(u64::MAX), FmtVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: FMT-004 — provenance enforcement.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt004_pass_enforced_missing_hash_exit_5() {
        assert_eq!(
            verdict_from_provenance_enforcement(true, false, 5),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt004_fail_enforced_missing_hash_exit_0() {
        // Bug: provenance silently passed.
        assert_eq!(
            verdict_from_provenance_enforcement(true, false, 0),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt004_fail_enforced_missing_hash_exit_1() {
        // Bug: returned generic error not the reserved 5.
        assert_eq!(
            verdict_from_provenance_enforcement(true, false, 1),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt004_pass_enforced_with_hash_exit_0() {
        // Provenance OK, normal success.
        assert_eq!(
            verdict_from_provenance_enforcement(true, true, 0),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt004_pass_not_enforced() {
        assert_eq!(
            verdict_from_provenance_enforcement(false, false, 0),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt004_fail_spurious_exit_5() {
        // Bug: emitted exit 5 when provenance was OK.
        assert_eq!(
            verdict_from_provenance_enforcement(true, true, 5),
            FmtVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: FMT-005 — dtype coercion preserves shape.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt005_pass_qwen_ffn_shape_preserved() {
        let s = vec![768_usize, 3072];
        assert_eq!(
            verdict_from_coercion_shape_preserved(&s, &s),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt005_pass_4d_shape() {
        let s = vec![1_usize, 4, 64, 64];
        assert_eq!(
            verdict_from_coercion_shape_preserved(&s, &s),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt005_fail_block_size_dimension_added() {
        let input = vec![768_usize, 3072];
        let output = vec![768_usize, 3072, 32]; // bug: block_size dim
        assert_eq!(
            verdict_from_coercion_shape_preserved(&input, &output),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt005_fail_dimension_dropped() {
        let input = vec![768_usize, 3072];
        let output = vec![768_usize];
        assert_eq!(
            verdict_from_coercion_shape_preserved(&input, &output),
            FmtVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 8: FMT-006 — truncation detection.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt006_pass_exact_match() {
        assert_eq!(
            verdict_from_truncation_detected(8_000_000_000, 8_000_000_000),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt006_fail_truncated_at_50pct() {
        assert_eq!(
            verdict_from_truncation_detected(4_000_000_000, 8_000_000_000),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt006_fail_over_extended() {
        assert_eq!(
            verdict_from_truncation_detected(8_000_000_001, 8_000_000_000),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt006_fail_zero_expected() {
        assert_eq!(
            verdict_from_truncation_detected(100, 0),
            FmtVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 9: FMT-007 — strict rejects NaN.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt007_pass_strict_clean_tensor() {
        let t = vec![1.0_f32, 2.0, -3.5];
        assert_eq!(
            verdict_from_strict_rejects_nan(true, &t),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt007_pass_non_strict_nan_allowed() {
        let t = vec![1.0_f32, f32::NAN];
        assert_eq!(
            verdict_from_strict_rejects_nan(false, &t),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt007_fail_strict_with_nan() {
        let t = vec![1.0_f32, f32::NAN, 3.0];
        assert_eq!(
            verdict_from_strict_rejects_nan(true, &t),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt007_fail_strict_with_inf() {
        let t = vec![1.0_f32, f32::INFINITY];
        assert_eq!(
            verdict_from_strict_rejects_nan(true, &t),
            FmtVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 10: FMT-008 — no silent F32 → F16 overflow.
    // -------------------------------------------------------------------------
    #[test]
    fn fmt008_pass_within_f16_range_no_warning() {
        assert_eq!(
            verdict_from_no_silent_f32_to_f16_overflow(1000.0, false),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt008_pass_at_f16_max() {
        assert_eq!(
            verdict_from_no_silent_f32_to_f16_overflow(65504.0, false),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt008_pass_overflow_with_warning() {
        assert_eq!(
            verdict_from_no_silent_f32_to_f16_overflow(100_000.0, true),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn fmt008_fail_silent_overflow() {
        // The contract failure mode: F32::MAX → F16::INFINITY without warning.
        assert_eq!(
            verdict_from_no_silent_f32_to_f16_overflow(f32::MAX, false),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt008_fail_negative_overflow_no_warning() {
        assert_eq!(
            verdict_from_no_silent_f32_to_f16_overflow(-100_000.0, false),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn fmt008_fail_nan() {
        // NaN is a separate gate but must Fail here too.
        assert_eq!(
            verdict_from_no_silent_f32_to_f16_overflow(f32::NAN, true),
            FmtVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 11: Realistic — full security-surface scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_adversarial_truncated_gguf() {
        // 3-byte file with partial GGUF magic.
        assert_eq!(
            verdict_from_truncated_input_safety(b"GGU"),
            FmtVerdict::Pass
        );
        assert_eq!(detect_format(b"GGU"), ModelFormat::Unknown);
    }

    #[test]
    fn realistic_adversarial_u64_max_tensor_count() {
        // Crafted GGUF with claimed 2^64 tensors.
        assert_eq!(
            verdict_from_tensor_count_bound(u64::MAX),
            FmtVerdict::Fail
        );
    }

    #[test]
    fn realistic_provenance_violation_blocks_ship() {
        // Real-world: someone tries to import a model without provenance.
        assert_eq!(
            verdict_from_provenance_enforcement(true, false, 5),
            FmtVerdict::Pass
        );
    }

    #[test]
    fn realistic_full_security_pipeline() {
        // Synthetic apr import of a real Qwen2.5-Coder-7B-Q4_K model.
        let bytes = {
            let mut b = Vec::from(b"GGUF" as &[u8]);
            b.extend_from_slice(&[0_u8; 100]);
            b
        };
        // FMT-001/002:
        assert_eq!(
            verdict_from_empty_input_safety(&bytes),
            FmtVerdict::Pass
        );
        assert_eq!(
            verdict_from_truncated_input_safety(&bytes),
            FmtVerdict::Pass
        );
        // Format detected.
        assert_eq!(detect_format(&bytes), ModelFormat::Gguf);
        // FMT-003:
        assert_eq!(verdict_from_tensor_count_bound(339), FmtVerdict::Pass);
        // FMT-004:
        assert_eq!(
            verdict_from_provenance_enforcement(true, true, 0),
            FmtVerdict::Pass
        );
        // FMT-005:
        let s = vec![768_usize, 3072];
        assert_eq!(
            verdict_from_coercion_shape_preserved(&s, &s),
            FmtVerdict::Pass
        );
        // FMT-006:
        assert_eq!(
            verdict_from_truncation_detected(7_950_000_000, 7_950_000_000),
            FmtVerdict::Pass
        );
        // FMT-007:
        let clean_tensor = vec![1.0_f32; 100];
        assert_eq!(
            verdict_from_strict_rejects_nan(true, &clean_tensor),
            FmtVerdict::Pass
        );
        // FMT-008:
        assert_eq!(
            verdict_from_no_silent_f32_to_f16_overflow(1.0, false),
            FmtVerdict::Pass
        );
    }
}
