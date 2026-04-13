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
}
