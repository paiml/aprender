// SHIP-TWO-001 — `apr-model-diagnostics-v1` algorithm-level PARTIAL
// discharge for FALSIFY-DIAG-001..005.
//
// Contract: `contracts/apr-model-diagnostics-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Five model-diagnostics gates:
//
// - DIAG-001 (hex byte offset correctness): hex output at offset O for
//   N bytes byte-equals raw[O..O+N].
// - DIAG-002 (fingerprint format-independence): GGUF fingerprint ==
//   SafeTensors fingerprint for same model.
// - DIAG-003 (oracle never misidentifies unknown architecture):
//   randomized tensor names ⇒ FamilyDetection::Unknown.
// - DIAG-004 (compatibility check has no false positives): model size
//   > VRAM ⇒ incompatible.
// - DIAG-005 (NaN fault has actionable remediation): identifies layer
//   AND emits remediation hint.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FamilyDetection {
    Known(u32),
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: DIAG-001 — hex byte offset correctness.
// -----------------------------------------------------------------------------

/// Pass iff `displayed_bytes == raw_bytes[offset..offset + displayed_bytes.len()]`.
#[must_use]
pub fn verdict_from_hex_byte_offset(
    raw_bytes: &[u8],
    offset: usize,
    displayed_bytes: &[u8],
) -> DiagVerdict {
    let end = offset.saturating_add(displayed_bytes.len());
    if end > raw_bytes.len() {
        return DiagVerdict::Fail;
    }
    if &raw_bytes[offset..end] == displayed_bytes {
        DiagVerdict::Pass
    } else {
        DiagVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: DIAG-002 — fingerprint format-independence.
// -----------------------------------------------------------------------------

/// Pass iff `gguf_fingerprint == safetensors_fingerprint`.
#[must_use]
pub fn verdict_from_fingerprint_format_independent(
    gguf_fingerprint: &[u8; 32],
    safetensors_fingerprint: &[u8; 32],
) -> DiagVerdict {
    if gguf_fingerprint == safetensors_fingerprint {
        DiagVerdict::Pass
    } else {
        DiagVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: DIAG-003 — oracle returns Unknown for randomized arch.
// -----------------------------------------------------------------------------

/// Pass iff `detection == FamilyDetection::Unknown` when input has no
/// recognizable architecture markers.
#[must_use]
pub fn verdict_from_oracle_unknown_for_random(detection: FamilyDetection) -> DiagVerdict {
    match detection {
        FamilyDetection::Unknown => DiagVerdict::Pass,
        FamilyDetection::Known(_) => DiagVerdict::Fail,
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: DIAG-004 — compatibility check has no false positives.
// -----------------------------------------------------------------------------

/// Pass iff `is_compatible == false` when model_size_mb > vram_mb.
#[must_use]
pub fn verdict_from_compat_no_false_positive(
    model_size_mb: u64,
    vram_mb: u64,
    reported_compatible: bool,
) -> DiagVerdict {
    let actually_compatible = model_size_mb <= vram_mb;
    if actually_compatible == reported_compatible {
        DiagVerdict::Pass
    } else if !actually_compatible && reported_compatible {
        // FALSE POSITIVE: model exceeds VRAM but reported compatible.
        DiagVerdict::Fail
    } else {
        // false negative: incompatible reported when actually OK —
        // also Fail (under-reports capability).
        DiagVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: DIAG-005 — NaN fault has actionable remediation.
// -----------------------------------------------------------------------------

/// `fault_layer_index` is the layer where NaN was detected (None if
/// not located). `remediation` is the hint string emitted.
///
/// Pass iff layer is identified AND remediation is non-empty.
#[must_use]
pub fn verdict_from_nan_fault_remediation(
    fault_layer_index: Option<usize>,
    remediation: &str,
) -> DiagVerdict {
    if fault_layer_index.is_none() {
        return DiagVerdict::Fail;
    }
    if remediation.trim().is_empty() {
        return DiagVerdict::Fail;
    }
    DiagVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: DIAG-001 — hex byte offset.
    // -------------------------------------------------------------------------
    #[test]
    fn diag001_pass_offset_matches() {
        let raw: Vec<u8> = (0..=255).collect();
        let displayed = &raw[64..96]; // 32 bytes at offset 64
        assert_eq!(
            verdict_from_hex_byte_offset(&raw, 64, displayed),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag001_pass_offset_zero() {
        let raw = vec![1_u8, 2, 3, 4];
        assert_eq!(
            verdict_from_hex_byte_offset(&raw, 0, &raw[0..4]),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag001_fail_displayed_corrupted() {
        let raw: Vec<u8> = (0..=10).collect();
        let displayed = vec![100_u8, 101, 102];
        assert_eq!(
            verdict_from_hex_byte_offset(&raw, 0, &displayed),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag001_fail_offset_out_of_range() {
        let raw = vec![1_u8, 2, 3];
        let displayed = vec![1_u8, 2, 3];
        assert_eq!(
            verdict_from_hex_byte_offset(&raw, 5, &displayed),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag001_fail_displayed_extends_past_eof() {
        let raw = vec![1_u8, 2];
        let displayed = vec![1_u8, 2, 3, 4];
        assert_eq!(
            verdict_from_hex_byte_offset(&raw, 0, &displayed),
            DiagVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: DIAG-002 — fingerprint format-independence.
    // -------------------------------------------------------------------------
    #[test]
    fn diag002_pass_identical() {
        let fp = [0xAB_u8; 32];
        assert_eq!(
            verdict_from_fingerprint_format_independent(&fp, &fp),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag002_fail_one_bit_differs() {
        let fp1 = [0xAB_u8; 32];
        let mut fp2 = fp1;
        fp2[15] ^= 0x01;
        assert_eq!(
            verdict_from_fingerprint_format_independent(&fp1, &fp2),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag002_fail_completely_different() {
        let fp1 = [0x00_u8; 32];
        let fp2 = [0xFF_u8; 32];
        assert_eq!(
            verdict_from_fingerprint_format_independent(&fp1, &fp2),
            DiagVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: DIAG-003 — oracle Unknown for random.
    // -------------------------------------------------------------------------
    #[test]
    fn diag003_pass_unknown() {
        assert_eq!(
            verdict_from_oracle_unknown_for_random(FamilyDetection::Unknown),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag003_fail_known_qwen2() {
        assert_eq!(
            verdict_from_oracle_unknown_for_random(FamilyDetection::Known(0xC0DE)),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag003_fail_known_llama() {
        assert_eq!(
            verdict_from_oracle_unknown_for_random(FamilyDetection::Known(0xBEEF)),
            DiagVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: DIAG-004 — compat no false positives.
    // -------------------------------------------------------------------------
    #[test]
    fn diag004_pass_fits_in_vram() {
        // 4GB model, 24GB VRAM, reported compatible.
        assert_eq!(
            verdict_from_compat_no_false_positive(4096, 24576, true),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag004_pass_exceeds_vram_correctly_incompatible() {
        // 30GB model, 24GB VRAM, correctly reported incompatible.
        assert_eq!(
            verdict_from_compat_no_false_positive(30000, 24576, false),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag004_fail_false_positive() {
        // The exact regression: model exceeds VRAM, reported compatible.
        assert_eq!(
            verdict_from_compat_no_false_positive(30000, 24576, true),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag004_fail_false_negative() {
        // Model fits but reported incompatible.
        assert_eq!(
            verdict_from_compat_no_false_positive(4096, 24576, false),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag004_pass_at_exact_vram() {
        assert_eq!(
            verdict_from_compat_no_false_positive(24576, 24576, true),
            DiagVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: DIAG-005 — NaN fault remediation.
    // -------------------------------------------------------------------------
    #[test]
    fn diag005_pass_layer_and_remediation_present() {
        assert_eq!(
            verdict_from_nan_fault_remediation(
                Some(5),
                "Detected NaN at layer 5 attention. Remediation: re-run with --strict to reject; verify upstream training stability."
            ),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag005_pass_layer_zero_short_remediation() {
        assert_eq!(
            verdict_from_nan_fault_remediation(Some(0), "Re-quantize."),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn diag005_fail_layer_not_located() {
        assert_eq!(
            verdict_from_nan_fault_remediation(None, "Re-quantize."),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag005_fail_empty_remediation() {
        assert_eq!(
            verdict_from_nan_fault_remediation(Some(5), ""),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag005_fail_whitespace_only_remediation() {
        assert_eq!(
            verdict_from_nan_fault_remediation(Some(5), "   \n\t  "),
            DiagVerdict::Fail
        );
    }

    #[test]
    fn diag005_fail_both_missing() {
        assert_eq!(
            verdict_from_nan_fault_remediation(None, ""),
            DiagVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full diagnostics pipeline.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_diag_post_fix_full_pipeline() {
        // Synthesize a diagnostics run on Qwen2.5-Coder-7B-Q4_K_M.
        let raw_header: Vec<u8> = vec![
            b'G', b'G', b'U', b'F', // magic
            0x03, 0x00, 0x00, 0x00, // version 3
        ];
        let displayed = &raw_header[0..4];
        // DIAG-001:
        assert_eq!(
            verdict_from_hex_byte_offset(&raw_header, 0, displayed),
            DiagVerdict::Pass
        );

        // DIAG-002 (fingerprint stability):
        let fp = [0xCA_u8; 32];
        assert_eq!(
            verdict_from_fingerprint_format_independent(&fp, &fp),
            DiagVerdict::Pass
        );

        // DIAG-003 (Qwen2.5 known, but on randomized model returns
        // Unknown):
        assert_eq!(
            verdict_from_oracle_unknown_for_random(FamilyDetection::Unknown),
            DiagVerdict::Pass
        );

        // DIAG-004 (4GB Q4 fits 24GB VRAM):
        assert_eq!(
            verdict_from_compat_no_false_positive(4096, 24576, true),
            DiagVerdict::Pass
        );

        // DIAG-005 (no NaN ⇒ no fault to report; this gate only
        // applies when NaN exists, so we test the in-fault path):
        assert_eq!(
            verdict_from_nan_fault_remediation(
                Some(5),
                "Re-train with gradient clipping enabled."
            ),
            DiagVerdict::Pass
        );
    }

    #[test]
    fn realistic_diag_pre_fix_all_5_failures() {
        // Pre-fix: hex bytes corrupted, fingerprints differ, oracle
        // misidentifies, compat says yes when no, no remediation.
        let raw = vec![1_u8, 2, 3, 4];
        let bad_displayed = vec![99_u8, 99, 99];

        assert_eq!(
            verdict_from_hex_byte_offset(&raw, 0, &bad_displayed),
            DiagVerdict::Fail
        );
        assert_eq!(
            verdict_from_fingerprint_format_independent(
                &[0x00_u8; 32],
                &[0xFF_u8; 32]
            ),
            DiagVerdict::Fail
        );
        assert_eq!(
            verdict_from_oracle_unknown_for_random(FamilyDetection::Known(0xC0DE)),
            DiagVerdict::Fail
        );
        assert_eq!(
            verdict_from_compat_no_false_positive(30000, 24576, true),
            DiagVerdict::Fail
        );
        assert_eq!(
            verdict_from_nan_fault_remediation(None, ""),
            DiagVerdict::Fail
        );
    }
}
