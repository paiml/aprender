// SHIP-TWO-001 — `model-format-conversion-v1` algorithm-level PARTIAL
// discharge for FALSIFY-CONV-001..009 (closes 9/9 sweep).
//
// Contract: `contracts/model-format-conversion-v1.yaml`.

// ===========================================================================
// CONV-001 — Tensor count preserved across conversion
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv001Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_tensor_count_preserved(src_count: u64, dst_count: u64) -> Conv001Verdict {
    if src_count == 0 { return Conv001Verdict::Fail; }
    if src_count == dst_count { Conv001Verdict::Pass } else { Conv001Verdict::Fail }
}

// ===========================================================================
// CONV-002 — Quantization error bounded: max abs error < 0.5 (Q4_0)
// ===========================================================================

pub const AC_CONV_002_MAX_ABS_ERROR: f32 = 0.5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_quantization_error(observed_max_abs_error: f32) -> Conv002Verdict {
    if !observed_max_abs_error.is_finite() || observed_max_abs_error < 0.0 {
        return Conv002Verdict::Fail;
    }
    if observed_max_abs_error < AC_CONV_002_MAX_ABS_ERROR {
        Conv002Verdict::Pass
    } else {
        Conv002Verdict::Fail
    }
}

// ===========================================================================
// CONV-003 — Merge architecture compatibility: different name sets → Err
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeOutcome { OkSameNames, ErrDifferentNames, OkSilentDrop }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv003Verdict { Pass, Fail }

/// Pass iff:
///   - identical name sets ⇒ OkSameNames is acceptable
///   - different name sets ⇒ MUST be ErrDifferentNames (silent drop is Fail)
#[must_use]
pub fn verdict_from_merge_compatibility(
    names_match: bool,
    outcome: MergeOutcome,
) -> Conv003Verdict {
    match (names_match, outcome) {
        (true,  MergeOutcome::OkSameNames)        => Conv003Verdict::Pass,
        (false, MergeOutcome::ErrDifferentNames)  => Conv003Verdict::Pass,
        _ => Conv003Verdict::Fail,
    }
}

// ===========================================================================
// CONV-004 — Format detection by content not extension
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedFormat { Gguf, Safetensors, Apr, Unknown }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv004Verdict { Pass, Fail }

#[must_use]
pub fn detect_by_magic(bytes: &[u8]) -> DetectedFormat {
    if bytes.len() < 4 { return DetectedFormat::Unknown; }
    if &bytes[0..4] == b"GGUF" { return DetectedFormat::Gguf; }
    if &bytes[0..4] == b"APR\0" || &bytes[0..4] == b"APRN" { return DetectedFormat::Apr; }
    // safetensors files start with a u64 little-endian header length;
    // the bytes immediately after are JSON `{`. Use the JSON sentinel.
    if bytes.len() >= 9 && bytes[8] == b'{' { return DetectedFormat::Safetensors; }
    DetectedFormat::Unknown
}

/// Pass iff `detect_by_magic(bytes)` returns the expected format and
/// is independent of the extension (the extension is only used as a
/// disambiguation tie-breaker, never as the primary signal).
#[must_use]
pub fn verdict_from_content_detection(
    bytes: &[u8],
    expected: DetectedFormat,
) -> Conv004Verdict {
    let detected = detect_by_magic(bytes);
    if detected == expected && detected != DetectedFormat::Unknown {
        Conv004Verdict::Pass
    } else {
        Conv004Verdict::Fail
    }
}

// ===========================================================================
// CONV-005 — Export-import roundtrip preserves tensor names + shapes
// ===========================================================================

#[derive(Debug, Clone)]
pub struct TensorMeta { pub name: String, pub shape: Vec<u64> }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_roundtrip_fidelity(src: &[TensorMeta], roundtrip: &[TensorMeta]) -> Conv005Verdict {
    if src.is_empty() || src.len() != roundtrip.len() { return Conv005Verdict::Fail; }
    // Sort both by name to be order-independent.
    let mut src_sorted: Vec<&TensorMeta> = src.iter().collect();
    let mut rt_sorted: Vec<&TensorMeta> = roundtrip.iter().collect();
    src_sorted.sort_by(|a, b| a.name.cmp(&b.name));
    rt_sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for (a, b) in src_sorted.iter().zip(rt_sorted.iter()) {
        if a.name != b.name { return Conv005Verdict::Fail; }
        if a.shape != b.shape { return Conv005Verdict::Fail; }
    }
    Conv005Verdict::Pass
}

// ===========================================================================
// CONV-006 — Atomic write: interrupted export leaves no file
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptOutcome { TargetAbsent, PartialFileLeftBehind, FullFileWritten }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_atomic_write(
    write_completed: bool,
    outcome: InterruptOutcome,
) -> Conv006Verdict {
    match (write_completed, outcome) {
        (true,  InterruptOutcome::FullFileWritten) => Conv006Verdict::Pass,
        (false, InterruptOutcome::TargetAbsent)    => Conv006Verdict::Pass,
        _ => Conv006Verdict::Fail,
    }
}

// ===========================================================================
// CONV-007 — APR tokenizer embedding present in metadata
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv007Verdict { Pass, Fail }

/// Pass iff the first 64 KiB of the APR file contains either the
/// substring `tokenizer.merges` or `tokenizer.vocabulary`.
#[must_use]
pub fn verdict_from_tokenizer_embedded(first_64k: &[u8]) -> Conv007Verdict {
    if first_64k.is_empty() { return Conv007Verdict::Fail; }
    let needles: [&[u8]; 2] = [b"tokenizer.merges", b"tokenizer.vocabulary"];
    for needle in needles {
        if first_64k.windows(needle.len()).any(|w| w == needle) {
            return Conv007Verdict::Pass;
        }
    }
    Conv007Verdict::Fail
}

// ===========================================================================
// CONV-008 — Streaming Q4K preserves tensor names + finite values
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv008Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_streaming_q4k(
    src_names: &[String],
    dst_names: &[String],
    nonfinite_count: u64,
    quant_type: &str,
) -> Conv008Verdict {
    if src_names.is_empty() { return Conv008Verdict::Fail; }
    if src_names.len() != dst_names.len() { return Conv008Verdict::Fail; }
    let mut a = src_names.to_vec();
    let mut b = dst_names.to_vec();
    a.sort();
    b.sort();
    if a != b { return Conv008Verdict::Fail; }
    if nonfinite_count != 0 { return Conv008Verdict::Fail; }
    if quant_type != "q4_k" { return Conv008Verdict::Fail; }
    Conv008Verdict::Pass
}

// ===========================================================================
// CONV-009 — Streaming threshold gate: small/non-APR rejected; ≥ threshold accepts
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conv009Verdict { Pass, Fail }

#[must_use]
pub fn qualifies_for_streaming_q4k(file_size_bytes: u64, is_apr: bool, threshold_bytes: u64) -> bool {
    if !is_apr { return false; }
    if threshold_bytes == 0 { return false; }
    file_size_bytes >= threshold_bytes
}

#[must_use]
pub fn verdict_from_threshold_gate(
    file_size: u64,
    is_apr: bool,
    threshold: u64,
    expected_qualifies: bool,
) -> Conv009Verdict {
    if qualifies_for_streaming_q4k(file_size, is_apr, threshold) == expected_qualifies {
        Conv009Verdict::Pass
    } else {
        Conv009Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CONV-001
    #[test] fn conv001_pass_match() { assert_eq!(verdict_from_tensor_count_preserved(100, 100), Conv001Verdict::Pass); }
    #[test] fn conv001_fail_drop() { assert_eq!(verdict_from_tensor_count_preserved(100, 99), Conv001Verdict::Fail); }
    #[test] fn conv001_fail_dup() { assert_eq!(verdict_from_tensor_count_preserved(100, 200), Conv001Verdict::Fail); }
    #[test] fn conv001_fail_zero_src() { assert_eq!(verdict_from_tensor_count_preserved(0, 0), Conv001Verdict::Fail); }

    // CONV-002
    #[test] fn conv002_pass_normal() { assert_eq!(verdict_from_quantization_error(0.1), Conv002Verdict::Pass); }
    #[test] fn conv002_pass_at_boundary_epsilon() { assert_eq!(verdict_from_quantization_error(0.4999), Conv002Verdict::Pass); }
    #[test] fn conv002_fail_at_threshold() { assert_eq!(verdict_from_quantization_error(0.5), Conv002Verdict::Fail); }
    #[test] fn conv002_fail_above() { assert_eq!(verdict_from_quantization_error(1.0), Conv002Verdict::Fail); }
    #[test] fn conv002_fail_negative() { assert_eq!(verdict_from_quantization_error(-0.1), Conv002Verdict::Fail); }
    #[test] fn conv002_fail_nan() { assert_eq!(verdict_from_quantization_error(f32::NAN), Conv002Verdict::Fail); }

    // CONV-003
    #[test] fn conv003_pass_same_names_ok() {
        assert_eq!(verdict_from_merge_compatibility(true, MergeOutcome::OkSameNames), Conv003Verdict::Pass);
    }
    #[test] fn conv003_pass_diff_names_err() {
        assert_eq!(verdict_from_merge_compatibility(false, MergeOutcome::ErrDifferentNames), Conv003Verdict::Pass);
    }
    #[test] fn conv003_fail_silent_drop() {
        assert_eq!(verdict_from_merge_compatibility(false, MergeOutcome::OkSilentDrop), Conv003Verdict::Fail);
    }
    #[test] fn conv003_fail_same_names_returned_err() {
        assert_eq!(verdict_from_merge_compatibility(true, MergeOutcome::ErrDifferentNames), Conv003Verdict::Fail);
    }

    // CONV-004
    #[test] fn conv004_pass_gguf_magic() {
        let bytes = b"GGUF\0\0\0\0";
        assert_eq!(verdict_from_content_detection(bytes, DetectedFormat::Gguf), Conv004Verdict::Pass);
    }
    #[test] fn conv004_pass_apr_magic() {
        let bytes = b"APR\0\0\0\0\0";
        assert_eq!(verdict_from_content_detection(bytes, DetectedFormat::Apr), Conv004Verdict::Pass);
    }
    #[test] fn conv004_pass_apr_v1_magic() {
        let bytes = b"APRN\0\0\0\0";
        assert_eq!(verdict_from_content_detection(bytes, DetectedFormat::Apr), Conv004Verdict::Pass);
    }
    #[test] fn conv004_fail_unknown() {
        let bytes = b"\0\0\0\0\0\0\0\0\0";
        assert_eq!(verdict_from_content_detection(bytes, DetectedFormat::Gguf), Conv004Verdict::Fail);
    }
    #[test] fn conv004_fail_short() {
        assert_eq!(verdict_from_content_detection(b"GG", DetectedFormat::Gguf), Conv004Verdict::Fail);
    }

    // CONV-005
    fn meta(name: &str, shape: &[u64]) -> TensorMeta {
        TensorMeta { name: name.to_string(), shape: shape.to_vec() }
    }

    #[test] fn conv005_pass_match() {
        let s = vec![meta("a", &[2, 3]), meta("b", &[5])];
        let r = vec![meta("a", &[2, 3]), meta("b", &[5])];
        assert_eq!(verdict_from_roundtrip_fidelity(&s, &r), Conv005Verdict::Pass);
    }
    #[test] fn conv005_pass_reordered() {
        let s = vec![meta("a", &[2, 3]), meta("b", &[5])];
        let r = vec![meta("b", &[5]), meta("a", &[2, 3])];
        assert_eq!(verdict_from_roundtrip_fidelity(&s, &r), Conv005Verdict::Pass);
    }
    #[test] fn conv005_fail_shape_drift() {
        let s = vec![meta("a", &[2, 3])];
        let r = vec![meta("a", &[2, 4])];
        assert_eq!(verdict_from_roundtrip_fidelity(&s, &r), Conv005Verdict::Fail);
    }
    #[test] fn conv005_fail_name_drift() {
        let s = vec![meta("a", &[2])];
        let r = vec![meta("b", &[2])];
        assert_eq!(verdict_from_roundtrip_fidelity(&s, &r), Conv005Verdict::Fail);
    }
    #[test] fn conv005_fail_count_drift() {
        let s = vec![meta("a", &[2])];
        let r = vec![meta("a", &[2]), meta("b", &[3])];
        assert_eq!(verdict_from_roundtrip_fidelity(&s, &r), Conv005Verdict::Fail);
    }

    // CONV-006
    #[test] fn conv006_pass_full_write() {
        assert_eq!(verdict_from_atomic_write(true, InterruptOutcome::FullFileWritten), Conv006Verdict::Pass);
    }
    #[test] fn conv006_pass_interrupted_no_file() {
        assert_eq!(verdict_from_atomic_write(false, InterruptOutcome::TargetAbsent), Conv006Verdict::Pass);
    }
    #[test] fn conv006_fail_partial_left() {
        // Atomicity violation.
        assert_eq!(verdict_from_atomic_write(false, InterruptOutcome::PartialFileLeftBehind), Conv006Verdict::Fail);
    }
    #[test] fn conv006_fail_complete_but_no_file() {
        assert_eq!(verdict_from_atomic_write(true, InterruptOutcome::TargetAbsent), Conv006Verdict::Fail);
    }

    // CONV-007
    #[test] fn conv007_pass_merges() {
        let payload = b"some metadata tokenizer.merges = ['ab' 'cd']";
        assert_eq!(verdict_from_tokenizer_embedded(payload), Conv007Verdict::Pass);
    }
    #[test] fn conv007_pass_vocabulary() {
        let payload = b"tokenizer.vocabulary: {...}";
        assert_eq!(verdict_from_tokenizer_embedded(payload), Conv007Verdict::Pass);
    }
    #[test] fn conv007_fail_missing() {
        let payload = b"only model weights, no tokenizer metadata";
        assert_eq!(verdict_from_tokenizer_embedded(payload), Conv007Verdict::Fail);
    }
    #[test] fn conv007_fail_empty() {
        assert_eq!(verdict_from_tokenizer_embedded(&[]), Conv007Verdict::Fail);
    }

    // CONV-008
    #[test] fn conv008_pass_canonical() {
        let names = vec!["a".to_string(), "b".to_string()];
        assert_eq!(
            verdict_from_streaming_q4k(&names, &names, 0, "q4_k"),
            Conv008Verdict::Pass
        );
    }
    #[test] fn conv008_pass_reordered_same_set() {
        let src = vec!["a".to_string(), "b".to_string()];
        let dst = vec!["b".to_string(), "a".to_string()];
        assert_eq!(verdict_from_streaming_q4k(&src, &dst, 0, "q4_k"), Conv008Verdict::Pass);
    }
    #[test] fn conv008_fail_dropped_tensor() {
        let src = vec!["a".to_string(), "b".to_string()];
        let dst = vec!["a".to_string()];
        assert_eq!(verdict_from_streaming_q4k(&src, &dst, 0, "q4_k"), Conv008Verdict::Fail);
    }
    #[test] fn conv008_fail_nonfinite() {
        let names = vec!["a".to_string()];
        assert_eq!(verdict_from_streaming_q4k(&names, &names, 1, "q4_k"), Conv008Verdict::Fail);
    }
    #[test] fn conv008_fail_wrong_quant_type() {
        let names = vec!["a".to_string()];
        assert_eq!(verdict_from_streaming_q4k(&names, &names, 0, "q8_0"), Conv008Verdict::Fail);
    }

    // CONV-009
    #[test] fn conv009_pass_below_threshold_apr() {
        // 100 MB below 4 GiB threshold + apr → does NOT qualify.
        assert!(!qualifies_for_streaming_q4k(100_000_000, true, 4 * 1024 * 1024 * 1024));
        assert_eq!(
            verdict_from_threshold_gate(100_000_000, true, 4 * 1024 * 1024 * 1024, false),
            Conv009Verdict::Pass
        );
    }

    #[test] fn conv009_pass_above_threshold_apr() {
        let big = 5_u64 * 1024 * 1024 * 1024;
        assert!(qualifies_for_streaming_q4k(big, true, 4 * 1024 * 1024 * 1024));
        assert_eq!(
            verdict_from_threshold_gate(big, true, 4 * 1024 * 1024 * 1024, true),
            Conv009Verdict::Pass
        );
    }

    #[test] fn conv009_pass_above_threshold_non_apr_rejected() {
        // GGUF / non-APR file above threshold should NOT qualify.
        let big = 10_u64 * 1024 * 1024 * 1024;
        assert!(!qualifies_for_streaming_q4k(big, false, 4 * 1024 * 1024 * 1024));
        assert_eq!(
            verdict_from_threshold_gate(big, false, 4 * 1024 * 1024 * 1024, false),
            Conv009Verdict::Pass
        );
    }

    #[test] fn conv009_fail_qualifies_when_should_not() {
        assert_eq!(
            verdict_from_threshold_gate(100, false, 1000, true),
            Conv009Verdict::Fail
        );
    }

    // Provenance
    #[test] fn provenance_max_error() {
        assert!((AC_CONV_002_MAX_ABS_ERROR - 0.5).abs() < f32::EPSILON);
    }
}
