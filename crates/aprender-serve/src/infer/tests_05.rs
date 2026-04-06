//! Infer Module Tests Part 05 - T-COV-95 Coverage Bridge (B5)
//!
//! Tests for:
//! - validate_model_path: traversal, wrong extension, nonexistent, directory
//! - qtype_to_dtype_str: all 14+ match arms
//! - run_inference: dispatch with mock backend, file too small, unknown format
//! - InferenceConfig builder
//!
//! Refs PMAT-802: Protocol T-COV-95 Batch B5

#[cfg(test)]
mod tests {
    use crate::infer::*;
    use std::path::PathBuf;

    // =========================================================================
    // validate_model_path Tests
    // =========================================================================

    #[test]
    fn test_validate_model_path_traversal_double_dot() {
        let path = PathBuf::from("/home/user/../../../etc/passwd");
        let result = crate::infer::validate_model_path(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("traversal") || err.contains(".."));
    }

    #[test]
    fn test_validate_model_path_traversal_middle() {
        let path = PathBuf::from("/models/../secret.gguf");
        let result = crate::infer::validate_model_path(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_model_path_wrong_extension_txt() {
        let path = PathBuf::from("/tmp/model.txt");
        let result = crate::infer::validate_model_path(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("extension") || err.contains("Invalid"));
    }

    #[test]
    fn test_validate_model_path_wrong_extension_py() {
        let path = PathBuf::from("/tmp/model.py");
        let result = crate::infer::validate_model_path(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_model_path_no_extension() {
        let path = PathBuf::from("/tmp/model");
        let result = crate::infer::validate_model_path(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_model_path_nonexistent_file() {
        let path = PathBuf::from("/nonexistent/path/model.gguf");
        let result = crate::infer::validate_model_path(&path);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not found") || err.contains("File"));
    }

    #[test]
    fn test_validate_model_path_directory_not_file() {
        // /tmp is a directory, not a file
        // But it has no valid extension, so it will fail on extension check first
        let path = PathBuf::from("/tmp");
        let result = crate::infer::validate_model_path(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_model_path_valid_extensions() {
        // These are valid extensions but files don't exist
        for ext in &["gguf", "safetensors", "apr", "bin"] {
            let path = PathBuf::from(format!("/nonexistent/model.{}", ext));
            let result = crate::infer::validate_model_path(&path);
            // Should fail on "not found", not "invalid extension"
            assert!(result.is_err());
            let err = result.unwrap_err().to_string();
            assert!(
                err.contains("not found") || err.contains("File"),
                "Expected 'not found' for .{} but got: {}",
                ext,
                err
            );
        }
    }

    // =========================================================================
    // qtype_to_dtype_str: all match arms
    // =========================================================================

    #[test]
    fn test_qtype_to_dtype_str_all_known_types() {
        assert_eq!(crate::infer::qtype_to_dtype_str(0), "F32");
        assert_eq!(crate::infer::qtype_to_dtype_str(1), "F16");
        assert_eq!(crate::infer::qtype_to_dtype_str(2), "Q4_0");
        assert_eq!(crate::infer::qtype_to_dtype_str(3), "Q4_1");
        assert_eq!(crate::infer::qtype_to_dtype_str(6), "Q5_0");
        assert_eq!(crate::infer::qtype_to_dtype_str(7), "Q5_1");
        assert_eq!(crate::infer::qtype_to_dtype_str(8), "Q8_0");
        assert_eq!(crate::infer::qtype_to_dtype_str(9), "Q8_1");
        assert_eq!(crate::infer::qtype_to_dtype_str(10), "Q2_K");
        assert_eq!(crate::infer::qtype_to_dtype_str(11), "Q3_K");
        assert_eq!(crate::infer::qtype_to_dtype_str(12), "Q4_K");
        assert_eq!(crate::infer::qtype_to_dtype_str(13), "Q5_K");
        assert_eq!(crate::infer::qtype_to_dtype_str(14), "Q6_K");
        assert_eq!(crate::infer::qtype_to_dtype_str(16), "IQ2_XXS");
        assert_eq!(crate::infer::qtype_to_dtype_str(17), "IQ2_XS");
        assert_eq!(crate::infer::qtype_to_dtype_str(30), "BF16");
    }

    #[test]
    fn test_qtype_to_dtype_str_unknown_values() {
        assert_eq!(crate::infer::qtype_to_dtype_str(4), "Unknown");
        assert_eq!(crate::infer::qtype_to_dtype_str(5), "Unknown");
        assert_eq!(crate::infer::qtype_to_dtype_str(15), "Unknown");
        assert_eq!(crate::infer::qtype_to_dtype_str(18), "Unknown");
        assert_eq!(crate::infer::qtype_to_dtype_str(29), "Unknown");
        assert_eq!(crate::infer::qtype_to_dtype_str(31), "Unknown");
        assert_eq!(crate::infer::qtype_to_dtype_str(100), "Unknown");
        assert_eq!(crate::infer::qtype_to_dtype_str(u32::MAX), "Unknown");
    }

    // =========================================================================
    // run_inference: dispatch with mock backend
    // =========================================================================

    #[test]
    fn test_run_inference_mock_backend() {
        let config = InferenceConfig::new("/mock/model.gguf")
            .with_prompt("Hello world")
            .with_max_tokens(5)
            .with_mock_backend();
        let result = run_inference(&config);
        assert!(result.is_ok());
        let res = result.unwrap();
        assert!(!res.text.is_empty());
        assert!(res.generated_token_count > 0);
    }

    #[test]
    fn test_run_inference_mock_backend_with_temperature() {
        let config = InferenceConfig::new("/mock/model.gguf")
            .with_prompt("Test")
            .with_temperature(0.7)
            .with_mock_backend();
        let result = run_inference(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_inference_mock_backend_with_top_k() {
        let config = InferenceConfig::new("/mock/model.gguf")
            .with_prompt("Test")
            .with_top_k(40)
            .with_mock_backend();
        let result = run_inference(&config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_inference_nonexistent_path() {
        let config = InferenceConfig::new("/nonexistent/model.gguf").with_prompt("Hello");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_traversal_attack() {
        let config = InferenceConfig::new("/etc/../etc/passwd").with_prompt("Hello");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_inference_wrong_extension() {
        let config = InferenceConfig::new("/tmp/test.txt").with_prompt("Hello");
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    // =========================================================================
    // InferenceConfig builder methods
    // =========================================================================

    #[test]
    fn test_inference_config_new_defaults() {
        let config = InferenceConfig::new("model.gguf");
        assert_eq!(config.model_path, PathBuf::from("model.gguf"));
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < 1e-6);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.trace);
        assert!(!config.verbose);
    }

    #[test]
    fn test_inference_config_with_prompt() {
        let config = InferenceConfig::new("model.gguf").with_prompt("Hello");
        assert_eq!(config.prompt, Some("Hello".to_string()));
    }

    #[test]
    fn test_inference_config_with_input_tokens() {
        let config = InferenceConfig::new("model.gguf").with_input_tokens(vec![1, 2, 3]);
        assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_inference_config_with_max_tokens() {
        let config = InferenceConfig::new("model.gguf").with_max_tokens(100);
        assert_eq!(config.max_tokens, 100);
    }

    #[test]
    fn test_inference_config_without_gpu() {
        let config = InferenceConfig::new("model.gguf").without_gpu();
        assert!(config.no_gpu);
    }

    #[test]
    fn test_inference_config_with_verbose() {
        let config = InferenceConfig::new("model.gguf").with_verbose(true);
        assert!(config.verbose);
    }

    #[test]
    fn test_inference_config_with_trace() {
        let config = InferenceConfig::new("model.gguf").with_trace(true);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_with_trace_output() {
        let config = InferenceConfig::new("model.gguf").with_trace_output("/tmp/trace.json");
        assert_eq!(config.trace_output, Some(PathBuf::from("/tmp/trace.json")));
    }

    #[test]
    fn test_inference_config_chained_builders() {
        let config = InferenceConfig::new("model.gguf")
            .with_prompt("Hello")
            .with_max_tokens(50)
            .with_temperature(0.7)
            .with_top_k(40)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true);
        assert_eq!(config.prompt, Some("Hello".to_string()));
        assert_eq!(config.max_tokens, 50);
        assert!((config.temperature - 0.7).abs() < 1e-6);
        assert_eq!(config.top_k, 40);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
    }

    // =========================================================================
    // InferenceResult: field verification
    // =========================================================================

    #[test]
    fn test_inference_result_debug() {
        let result = InferenceResult {
            text: "hello".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 10.0,
            tok_per_sec: 200.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("hello"));
        assert!(debug.contains("GGUF"));
    }

    #[test]
    fn test_inference_result_clone() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: "APR".to_string(),
            used_gpu: false,
        };
        let cloned = result.clone();
        assert_eq!(cloned.text, result.text);
        assert_eq!(cloned.format, result.format);
    }
}
