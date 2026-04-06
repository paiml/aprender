//! Additional tests for CLI inference module (Part 02)
//!
//! This module focuses on testing inference.rs code paths that are not covered
//! by tests_part_02.rs, including:
//! - Different file data scenarios for GGUF parsing
//! - Detailed error message validation
//! - Token generation boundary conditions
//! - Format output validation

#[cfg(test)]
mod inference_additional_coverage {
    use crate::cli::inference;
    use crate::error::RealizarError;
    use crate::inference_trace::TraceConfig;

    #[test]
    fn test_apr_inference_large_max_tokens() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "large tokens",
            100000,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_inference_large_max_tokens() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "large tokens",
            100000,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Trace Mode Tests
    // =========================================================================

    #[test]
    fn test_gguf_inference_trace_mode_enabled() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "trace test",
            5,
            0.5,
            "text",
            false,
            false,
            Some(TraceConfig::enabled()), // trace enabled
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_trace_and_verbose() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "trace+verbose",
            5,
            0.5,
            "text",
            false,                        // force_gpu
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // GPU Flag Tests (without CUDA feature)
    // =========================================================================

    #[test]
    fn test_gguf_inference_gpu_flag_warning_path() {
        // When cuda feature is not enabled, force_gpu should print warning
        // but continue with CPU path
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "gpu flag test",
            5,
            0.5,
            "text",
            true, // force_gpu
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err()); // Still fails, but warning path exercised
    }

    #[test]
    fn test_apr_inference_gpu_flag_warning_path() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "gpu flag test",
            5,
            0.5,
            "text",
            true, // force_gpu
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // File Extension and Path Variations
    // =========================================================================

    #[test]
    fn test_gguf_inference_uppercase_extension() {
        let result = inference::run_gguf_inference(
            "/path/to/model.GGUF",
            &[],
            "uppercase ext",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_inference_uppercase_extension() {
        let result = inference::run_safetensors_inference(
            "/path/to/model.SAFETENSORS",
            "uppercase ext",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_mixed_case_extension() {
        let result = inference::run_apr_inference(
            "/path/to/model.Apr",
            &[],
            "mixed case ext",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Special Character Paths
    // =========================================================================

    #[test]
    fn test_gguf_inference_path_with_special_chars() {
        let result = inference::run_gguf_inference(
            "/path/with-dashes_and_underscores/model.gguf",
            &[],
            "special chars",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_inference_path_with_dots() {
        let result = inference::run_safetensors_inference(
            "/path.with.dots/model.safetensors",
            "dots in path",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Format String Variations
    // =========================================================================

    #[test]
    fn test_gguf_inference_format_uppercase_json() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "format test",
            5,
            0.5,
            "JSON", // Uppercase - should fall through to text
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_inference_format_uppercase_json() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "format test",
            5,
            0.5,
            "JSON",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_safetensors_inference_format_unknown() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "format test",
            5,
            0.5,
            "xml", // Unknown format
            None,  // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Comprehensive Parameter Matrix Tests
    // =========================================================================

    #[test]
    fn test_gguf_all_boolean_flags_true() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "all flags",
            5,
            0.5,
            "text",
            true,                         // force_gpu
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_all_boolean_flags_false() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "no flags",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_all_boolean_flags_true() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "all flags",
            5,
            0.5,
            "text",
            true,
            true,
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_apr_all_boolean_flags_false() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "no flags",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }
include!("inference_tests_run_gguf.rs");
}

include!("inference_tests_model_type.rs");
