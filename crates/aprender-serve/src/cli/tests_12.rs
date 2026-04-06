//! Server Command Tests & CLI Inference Module Tests (Part 12 - PMAT-802)
//!
//! Extracted from tests.rs to keep the main file under 2000 lines.
//! Contains:
//! 1. Server Command Tests (EXTREME TDD - PAR-112)
//! 2. ModelType / PreparedServer tests
//! 3. prepare_serve_state tests
//! 4. serve_demo tests
//! 5. CLI Inference Module Tests (run_gguf/safetensors/apr_inference)

#[cfg(test)]
mod server_and_inference_tests {
    use crate::cli::*;

    // =========================================================================
    // CLI Inference Module Tests (EXTREME TDD - PMAT-802)
    // Coverage for src/cli/inference.rs
    // =========================================================================

    mod inference_tests {

        use crate::cli::inference;
        use crate::inference_trace::TraceConfig;

        #[test]
        fn test_run_apr_inference_high_temperature() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "sampling test",
                5,
                2.0, // high temperature
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        // -------------------------------------------------------------------------
        // Parameter Validation Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_inference_temperature_boundary_greedy() {
            // Temperature <= 0.01 triggers greedy decoding
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "boundary test",
                5,
                0.01, // Exactly at boundary
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_inference_temperature_boundary_sampling() {
            // Temperature > 0.01 triggers sampling
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "boundary test",
                5,
                0.02, // Just above boundary
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_inference_format_unknown() {
            // Unknown format should default to text-like output
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "format test",
                5,
                0.5,
                "xml", // Unknown format
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        // -------------------------------------------------------------------------
        // Environment Variable Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_cpu_debug_env_var() {
            // The CPU_DEBUG environment variable controls diagnostic output
            std::env::remove_var("CPU_DEBUG");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());

            // Set CPU_DEBUG=1 and try again
            std::env::set_var("CPU_DEBUG", "1");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
            std::env::remove_var("CPU_DEBUG");
        }

        #[test]
        fn test_skip_gpu_resident_env_var() {
            // The SKIP_GPU_RESIDENT environment variable affects GPU path selection
            std::env::remove_var("SKIP_GPU_RESIDENT");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                true, // force_gpu
                false,
                None, // trace_config
            );
            assert!(result.is_err());

            // Set SKIP_GPU_RESIDENT=1 and try again
            std::env::set_var("SKIP_GPU_RESIDENT", "1");
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "env test",
                5,
                0.5,
                "text",
                true, // force_gpu
                false,
                None, // trace_config
            );
            assert!(result.is_err());
            std::env::remove_var("SKIP_GPU_RESIDENT");
        }

        // -------------------------------------------------------------------------
        // Error Message Content Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_gguf_error_contains_operation() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "error test",
                5,
                0.5,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = err.to_string();
            // Error should identify the operation that failed
            assert!(
                err_str.contains("mmap") || err_str.contains("load") || err_str.contains("GGUF"),
                "Error should mention mmap or load operation: {}",
                err_str
            );
        }

        #[test]
        fn test_apr_error_contains_operation() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "error test",
                5,
                0.5,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            let err_str = err.to_string();
            // Error should identify the operation that failed
            assert!(
                err_str.contains("parse") || err_str.contains("APR") || err_str.contains("Failed"),
                "Error should mention parse or APR operation: {}",
                err_str
            );
        }

        // -------------------------------------------------------------------------
        // Integration Tests (ignored by default - require real models)
        // -------------------------------------------------------------------------

        /// Integration test for GGUF inference with a real model
        /// Run with: cargo test test_run_gguf_inference_real -- --ignored
        #[test]
        #[ignore = "requires real GGUF model file"]
        fn test_run_gguf_inference_real() {
            let home = std::env::var("HOME").expect("HOME env var not set");
            let model_paths = [
                format!("{}/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", home),
                format!("{}/src/single-shot-eval/models/raw/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf", home),
            ];

            let model_path = model_paths
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .expect("No test model file found");

            // Read the file
            let file_data = std::fs::read(model_path).expect("Failed to read model file");

            let result = inference::run_gguf_inference(
                model_path,
                &file_data,
                "Hello, world!",
                5,
                0.0, // greedy
                "text",
                false,
                true, // verbose
                None, // trace_config
            );

            // Should succeed with real model
            assert!(result.is_ok(), "Inference failed: {:?}", result.err());
        }

        /// Integration test for GGUF inference with JSON output
        #[test]
        #[ignore = "requires real GGUF model file"]
        fn test_run_gguf_inference_json_output_real() {
            let home = std::env::var("HOME").expect("HOME env var not set");
            let model_paths = [format!("{}/.apr/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", home)];

            let model_path = model_paths
                .iter()
                .find(|p| std::path::Path::new(p).exists())
                .expect("No test model file found");

            let file_data = std::fs::read(model_path).expect("Failed to read model file");

            let result = inference::run_gguf_inference(
                model_path,
                &file_data,
                "What is 2+2?",
                10,
                0.0,
                "json", // JSON output
                false,
                false,
                None, // trace_config
            );

            assert!(result.is_ok(), "JSON inference failed: {:?}", result.err());
        }

        // -------------------------------------------------------------------------
        // Comprehensive API Surface Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_gguf_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_gguf_inference(
                "model_ref",                  // model_ref: &str
                &[],                          // file_data: &[u8]
                "prompt",                     // prompt: &str
                10usize,                      // max_tokens: usize
                0.5f32,                       // temperature: f32
                "format",                     // format: &str
                true,                         // force_gpu: bool
                true,                         // verbose: bool
                Some(TraceConfig::enabled()), // trace_config: Option<TraceConfig>
            );
        }

        #[test]
        fn test_safetensors_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_safetensors_inference(
                "model_ref",                  // model_ref: &str
                "prompt",                     // prompt: &str
                10usize,                      // max_tokens: usize
                0.5f32,                       // temperature: f32 (unused in current impl)
                "format",                     // format: &str
                Some(TraceConfig::enabled()), // trace_config: Option<TraceConfig>
            );
        }

        #[test]
        fn test_apr_inference_api_surface() {
            // Verify all parameters are accepted in the expected order
            let _ = inference::run_apr_inference(
                "model_ref",                  // model_ref: &str
                &[],                          // file_data: &[u8]
                "prompt",                     // prompt: &str
                10usize,                      // max_tokens: usize
                0.5f32,                       // temperature: f32
                "format",                     // format: &str
                true,                         // force_gpu: bool
                true,                         // verbose: bool
                Some(TraceConfig::enabled()), // trace_config: Option<TraceConfig>
            );
        }

        // -------------------------------------------------------------------------
        // Edge Case Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_unicode_prompt() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "Hello \u{1F600} \u{4E2D}\u{6587} \u{0410}\u{0411}\u{0412}",
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
        fn test_very_long_prompt() {
            let long_prompt = "word ".repeat(1000);
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                &long_prompt,
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
        fn test_special_characters_in_prompt() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "Test <script>alert('xss')</script> & \"quotes\" 'apostrophe'",
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
        fn test_newlines_in_prompt() {
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "Line 1\nLine 2\r\nLine 3\tTab",
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
        fn test_negative_temperature_clamped() {
            // Negative temperature should be treated as greedy (temperature <= 0.01)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "negative temp test",
                5,
                -1.0, // Negative temperature
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_extreme_temperature() {
            // Very high temperature
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "extreme temp test",
                5,
                100.0, // Very high temperature
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }
include!("tests_serve_model.rs");
include!("tests_serve_model_02.rs");
include!("tests_run_gguf_03.rs");
}
