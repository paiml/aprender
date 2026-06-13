// PMAT-540 Phase 5: Tests for forward_error/QA helper functions

#[cfg(test)]
mod forward_error_tests {
    use super::*;

    // ========================================================================
    // strip_quant_suffix
    // ========================================================================

    #[test]
    fn strip_quant_suffix_q4k() {
        assert_eq!(strip_quant_suffix("model-q4k"), "model");
    }

    #[test]
    fn strip_quant_suffix_q4_k_m() {
        assert_eq!(strip_quant_suffix("model-q4_k_m"), "model");
    }

    #[test]
    fn strip_quant_suffix_q6k() {
        assert_eq!(strip_quant_suffix("model-q6k"), "model");
    }

    #[test]
    fn strip_quant_suffix_f16() {
        assert_eq!(strip_quant_suffix("model-f16"), "model");
    }

    #[test]
    fn strip_quant_suffix_f32() {
        assert_eq!(strip_quant_suffix("model-f32"), "model");
    }

    #[test]
    fn strip_quant_suffix_no_suffix() {
        assert_eq!(strip_quant_suffix("model"), "model");
    }

    #[test]
    fn strip_quant_suffix_preserves_non_quant() {
        assert_eq!(strip_quant_suffix("qwen2.5-coder-7b"), "qwen2.5-coder-7b");
    }

    // ========================================================================
    // compute_argmax
    // ========================================================================

    #[test]
    fn compute_argmax_basic() {
        assert_eq!(compute_argmax(&[1.0, 3.0, 2.0]), Some(1));
    }

    #[test]
    fn compute_argmax_first() {
        assert_eq!(compute_argmax(&[9.0, 1.0, 2.0]), Some(0));
    }

    #[test]
    fn compute_argmax_last() {
        assert_eq!(compute_argmax(&[1.0, 2.0, 99.0]), Some(2));
    }

    #[test]
    fn compute_argmax_empty() {
        assert_eq!(compute_argmax(&[]), None);
    }

    #[test]
    fn compute_argmax_single() {
        assert_eq!(compute_argmax(&[42.0]), Some(0));
    }

    #[test]
    fn compute_argmax_negative() {
        assert_eq!(compute_argmax(&[-3.0, -1.0, -2.0]), Some(1));
    }

    // ========================================================================
    // hf_cache_dir_matches
    // ========================================================================

    #[test]
    fn hf_cache_dir_matches_standard() {
        assert!(hf_cache_dir_matches("models--Qwen--Qwen2.5-Coder-7B", "qwen2.5-coder"));
    }

    #[test]
    fn hf_cache_dir_matches_case_insensitive() {
        assert!(hf_cache_dir_matches("models--Meta--Llama-3-8B", "llama-3"));
    }

    #[test]
    fn hf_cache_dir_matches_no_prefix() {
        assert!(!hf_cache_dir_matches("Qwen--Qwen2.5", "qwen2.5"));
    }

    #[test]
    fn hf_cache_dir_matches_no_match() {
        assert!(!hf_cache_dir_matches("models--OpenAI--Whisper", "llama"));
    }

    // ========================================================================
    // detect_size_from_filename (word boundary)
    // ========================================================================

    #[test]
    fn detect_size_from_filename_3b_with_boundary() {
        assert_eq!(detect_size_from_filename("model-3b-chat"), Some("3b"));
    }

    #[test]
    fn detect_size_from_filename_7b_dot_gguf() {
        assert_eq!(detect_size_from_filename("llama-7b.gguf"), Some("7b"));
    }

    #[test]
    fn detect_size_from_filename_0_5b() {
        assert_eq!(detect_size_from_filename("model-0.5b-instruct"), Some("0.5b"));
    }

    #[test]
    fn detect_size_from_filename_no_match_hex() {
        // Random hex like "3bF2" should NOT match "3b"
        assert_eq!(detect_size_from_filename("tmp3bF2a1"), None);
    }

    #[test]
    fn detect_size_from_filename_no_match_hex_prefixed_digit() {
        // Reproducer for CI flake (workspace-test #25276156029): random hex like
        // ".tmp97b...gguf" has "7b" at a position where the BEFORE-char is a
        // digit ('9'). Without rejecting digit-prefixed matches we mis-detected
        // the empty file as "7b". Must return None.
        assert_eq!(detect_size_from_filename(".tmp97b1234.gguf"), None);
        // 132b pattern — '3' before, '2b' is the candidate
        assert_eq!(detect_size_from_filename(".tmp132b.gguf"), None);
        // 'model3b' (letter prefix) is still valid — see _3b_standalone test
        assert_eq!(detect_size_from_filename("model3b"), Some("3b"));
    }

    #[test]
    fn detect_size_from_filename_14b() {
        assert_eq!(detect_size_from_filename("model-14b-chat.gguf"), Some("14b"));
    }

    #[test]
    fn detect_size_from_filename_underscore_variant() {
        assert_eq!(detect_size_from_filename("model_0_5b"), Some("0.5b"));
    }

    // ========================================================================
    // estimate_size_from_file
    // ========================================================================

    #[test]
    fn estimate_size_from_file_zero_bytes() {
        let file = tempfile::NamedTempFile::with_suffix(".gguf").expect("temp");
        assert_eq!(estimate_size_from_file(file.path()), "0.5b");
    }

    #[test]
    fn estimate_size_from_file_nonexistent() {
        // Non-existent file → metadata fails → unwrap_or(0) → "0.5b"
        assert_eq!(
            estimate_size_from_file(std::path::Path::new("/nonexistent/model.gguf")),
            "0.5b"
        );
    }

    // ========================================================================
    // run_format_parity_gate: non-GGUF SKIP (regression — previously FAILed,
    // making `apr qa <safetensors>` fail overall even though the gate can't
    // meaningfully run without a GGUF primary.)
    // ========================================================================

    #[cfg(feature = "inference")]
    #[test]
    fn format_parity_skips_safetensors_primary() {
        use super::QaConfig;
        use std::io::Write;

        // Minimal SafeTensors file: 8-byte LE length of empty JSON metadata
        // `{}` = 2 bytes → header reads `0200 0000 0000 0000` then `{}`.
        let mut tmp = tempfile::NamedTempFile::with_suffix(".safetensors").expect("temp");
        tmp.write_all(&2u64.to_le_bytes()).expect("write len");
        tmp.write_all(b"{}").expect("write meta");
        tmp.flush().expect("flush");

        let config = QaConfig::default();
        let result = super::run_format_parity_gate(tmp.path(), &config).expect("gate ran");

        assert!(result.skipped, "non-GGUF primary must SKIP, not FAIL/PASS");
        assert!(result.passed, "skipped gates count as passed in summary");
        assert!(
            result.message.contains("Non-GGUF"),
            "skip reason should say Non-GGUF, got: {}",
            result.message
        );
    }

    #[cfg(feature = "inference")]
    #[test]
    fn format_parity_skips_apr_primary() {
        use super::QaConfig;
        use std::io::Write;

        // APR v2 magic: b"APR\0" followed by u32 LE version
        let mut tmp = tempfile::NamedTempFile::with_suffix(".apr").expect("temp");
        tmp.write_all(b"APR\0").expect("write magic");
        tmp.write_all(&2u32.to_le_bytes()).expect("write version");
        tmp.flush().expect("flush");

        let config = QaConfig::default();
        let result = super::run_format_parity_gate(tmp.path(), &config).expect("gate ran");

        assert!(result.skipped, "APR primary must SKIP, not FAIL");
        assert!(result.passed, "skipped gates count as passed");
        assert!(result.message.contains("Non-GGUF"), "got: {}", result.message);
    }

    // ========================================================================
    // PMAT-743: format-parity discovery must ignore apr's own conversion
    // artifacts (`*.converted*.safetensors`) — they are circular references,
    // not independent SafeTensors, and are frequently stale / double-converted.
    // ========================================================================

    #[test]
    fn synthetic_artifact_detection() {
        assert!(is_synthetic_conversion_artifact("m-q4_k_m.converted.safetensors"));
        assert!(is_synthetic_conversion_artifact(
            "m-q4_k_m.converted.converted.safetensors"
        ));
        // Genuine HF SafeTensors never contain a `.converted.` segment.
        assert!(!is_synthetic_conversion_artifact("model.safetensors"));
        assert!(!is_synthetic_conversion_artifact(
            "model-00001-of-00002.safetensors"
        ));
        assert!(!is_synthetic_conversion_artifact("qwen2.5-coder.safetensors"));
    }

    #[test]
    fn snapshot_discovery_prefers_genuine_over_converted() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path();
        // A genuine shard alongside two synthetic conversion artifacts.
        for name in [
            "model-00001-of-00002.safetensors",
            "m-q4_k_m.converted.safetensors",
            "m-q4_k_m.converted.converted.safetensors",
        ] {
            std::fs::write(p.join(name), b"x").expect("write");
        }
        let found = super::find_safetensors_in_snapshot(p).expect("a genuine shard exists");
        assert_eq!(
            found.file_name().unwrap().to_str().unwrap(),
            "model-00001-of-00002.safetensors",
            "must pick the genuine shard, never a .converted artifact"
        );
    }

    #[test]
    fn snapshot_discovery_skips_when_only_converted_artifacts() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path();
        // Only synthetic artifacts exist (GGUF-only model whose cache was polluted).
        for name in [
            "m-q4_k_m.converted.safetensors",
            "m-q4_k_m.converted.converted.safetensors",
        ] {
            std::fs::write(p.join(name), b"x").expect("write");
        }
        assert!(
            super::find_safetensors_in_snapshot(p).is_none(),
            "must find NO independent reference (artifacts ignored), not pick a circular one"
        );
    }
}
