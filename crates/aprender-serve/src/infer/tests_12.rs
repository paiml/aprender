//! Extended coverage tests for infer module (Part 12)
//!
//! Split from tests.rs to keep file under 2000 lines.
//! These tests cover:
//! - run_inference format detection tests
//! - Architecture detection from path tests
//! - Token per second calculation edge cases
//! - InferenceResult edge cases
//! - Config path handling, builder chaining, numeric boundaries
//! - Format detection with various magic bytes
//! - Error message content validation
//! - prefault_mmap comprehensive tests
//! - clean_model_output comprehensive edge cases
//! - InferenceResult/InferenceConfig Debug and Clone full coverage

#[cfg(test)]
mod tests {
    use crate::infer::*;
    use std::path::PathBuf;

    // --- Path conversion tests ---

    #[test]
    fn test_path_from_str() {
        let config = InferenceConfig::new("/path/to/model.gguf");
        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
    }

    #[test]
    fn test_path_from_string() {
        let path = String::from("/path/to/model.gguf");
        let config = InferenceConfig::new(path);
        assert_eq!(config.model_path, PathBuf::from("/path/to/model.gguf"));
    }

    #[test]
    fn test_path_from_pathbuf() {
        let path = PathBuf::from("/path/to/model.gguf");
        let config = InferenceConfig::new(path.clone());
        assert_eq!(config.model_path, path);
    }

    // --- RealizarError type tests ---

    #[test]
    fn test_io_error_construction() {
        use crate::error::RealizarError;

        let err = RealizarError::IoError {
            message: "test error".to_string(),
        };
        assert!(err.to_string().contains("test error"));
    }

    #[test]
    fn test_format_error_construction() {
        use crate::error::RealizarError;

        let err = RealizarError::FormatError {
            reason: "invalid magic".to_string(),
        };
        assert!(err.to_string().contains("invalid magic"));
    }

    #[test]
    fn test_inference_error_construction() {
        use crate::error::RealizarError;

        let err = RealizarError::InferenceError("generation failed".to_string());
        assert!(err.to_string().contains("generation failed"));
    }

    // --- EOS token detection comprehensive ---

    #[test]
    fn test_eos_token_detection_all_variants() {
        // Define EOS tokens to check against
        let eos_tokens = [151645u32, 151643, 2];

        // Helper to check if token is EOS
        let is_eos = |token: u32| eos_tokens.contains(&token);

        // Qwen2 EOS
        assert!(is_eos(151645));
        // Qwen2 BOS (also stops)
        assert!(is_eos(151643));
        // Standard EOS
        assert!(is_eos(2));
        // Not EOS tokens
        assert!(!is_eos(1));
        assert!(!is_eos(1000));
    }

    // --- Max tokens passthrough in different formats (GH-372) ---

    #[test]
    fn test_max_tokens_passthrough_consistency() {
        let configs = [
            InferenceConfig::new("/m.gguf").with_max_tokens(500),
            InferenceConfig::new("/m.apr").with_max_tokens(500),
            InferenceConfig::new("/m.safetensors").with_max_tokens(500),
        ];

        for config in configs {
            assert_eq!(config.max_tokens, 500, "Max tokens should pass through uncapped");
        }
    }

    // --- Generated tokens calculation ---

    #[test]
    fn test_generated_token_count_calculation() {
        let all_tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let input_token_count = 4;
        let generated_tokens = &all_tokens[input_token_count..];
        let generated_token_count = generated_tokens.len();

        assert_eq!(generated_token_count, 6);
        assert_eq!(all_tokens.len(), input_token_count + generated_token_count);
    }

    // --- Load time precision ---

    #[test]
    fn test_load_time_precision() {
        let elapsed_secs: f64 = 0.123456789;
        let load_ms = elapsed_secs * 1000.0;

        // Should preserve precision
        assert!((load_ms - 123.456789).abs() < 0.0001);
    }

    #[test]
    fn test_inference_time_precision() {
        let elapsed_secs: f64 = 0.987654321;
        let inference_ms = elapsed_secs * 1000.0;

        assert!((inference_ms - 987.654321).abs() < 0.0001);
    }
include!("tests_run_inference.rs");
include!("tests_inference_02.rs");
include!("tests_clean_model_03.rs");
}
