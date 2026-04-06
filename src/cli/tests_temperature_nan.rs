//! Additional tests for CLI inference module (Part 02)
//!
//! This module adds comprehensive test coverage for inference.rs code paths
//! including configuration building, CLI argument validation, and edge cases.

#[cfg(test)]
mod inference_coverage {
    use crate::cli::inference;
    use crate::inference_trace::TraceConfig;

    #[test]
    fn test_temperature_nan() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "nan temp",
            5,
            f32::NAN,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // String Parameter Edge Cases
    // =========================================================================

    #[test]
    fn test_format_with_whitespace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "format test",
            5,
            0.5,
            " json ", // Whitespace around format
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_format_case_sensitivity() {
        for format in &["JSON", "Json", "TEXT", "Text"] {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "case test",
                5,
                0.5,
                format,
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // API Contract Tests (Public Interface)
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_public_api() {
        // Verify public function signature matches expected API
        let _: crate::error::Result<()> = inference::run_gguf_inference(
            "",                           // model_ref: &str
            &[],                          // file_data: &[u8]
            "",                           // prompt: &str
            0,                            // max_tokens: usize
            0.0,                          // temperature: f32
            "",                           // format: &str
            true,                         // force_gpu: bool
            true,                         // verbose: bool
            Some(TraceConfig::enabled()), // trace_config
        );
    }

    #[test]
    fn test_run_safetensors_inference_public_api() {
        let _: crate::error::Result<()> = inference::run_safetensors_inference(
            "",   // model_ref: &str
            "",   // prompt: &str
            0,    // max_tokens: usize
            0.0,  // temperature: f32
            "",   // format: &str
            None, // trace_config
        );
    }

    #[test]
    fn test_run_apr_inference_public_api() {
        let _: crate::error::Result<()> = inference::run_apr_inference(
            "",    // model_ref: &str
            &[],   // file_data: &[u8]
            "",    // prompt: &str
            0,     // max_tokens: usize
            0.0,   // temperature: f32
            "",    // format: &str
            true,  // force_gpu: bool
            false, // verbose: bool
            None,  // trace_config
        );
    }
include!("tests_run_gguf_02.rs");
include!("tests_prompt.rs");
}

include!("tests_run_gguf.rs");
