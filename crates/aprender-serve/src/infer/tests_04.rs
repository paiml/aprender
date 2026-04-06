//! Infer Module Tests Part 04 - Helper Functions Coverage
//!
//! Tests for:
//! - clean_model_output function
//! - run_mock_inference function
//! - InferenceResult struct
//! - InferenceConfig edge cases
//!
//! Refs PMAT-802: Protocol T-COV-95

#[cfg(test)]
mod tests {
    use crate::infer::*;
    use std::path::PathBuf;

    // =========================================================================
    // clean_model_output Tests
    // =========================================================================

    #[test]
    fn test_clean_model_output_no_markers() {
        let output = clean_model_output("Hello, world!");
        assert_eq!(output, "Hello, world!");
    }

    #[test]
    fn test_clean_model_output_im_start_assistant() {
        let output = clean_model_output("<|im_start|>assistant\nHello!");
        assert_eq!(output, "Hello!");
    }

    #[test]
    fn test_clean_model_output_im_end() {
        let output = clean_model_output("Response text<|im_end|>");
        assert_eq!(output, "Response text");
    }

    #[test]
    fn test_clean_model_output_endoftext() {
        let output = clean_model_output("Final text<|endoftext|>");
        assert_eq!(output, "Final text");
    }

    #[test]
    fn test_clean_model_output_multiple_markers() {
        let output = clean_model_output("<|im_start|>assistant\nContent<|im_end|><|endoftext|>");
        assert_eq!(output, "Content");
    }

    #[test]
    fn test_clean_model_output_all_markers() {
        let raw = "<|im_start|>assistant\n<|im_start|>User said<|im_end|><|endoftext|>";
        let output = clean_model_output(raw);
        assert_eq!(output, "User said");
    }

    #[test]
    fn test_clean_model_output_empty_string() {
        let output = clean_model_output("");
        assert_eq!(output, "");
    }

    #[test]
    fn test_clean_model_output_only_markers() {
        let output = clean_model_output("<|im_start|><|im_end|>");
        assert_eq!(output, "");
    }

    #[test]
    fn test_clean_model_output_trims_whitespace() {
        let output = clean_model_output("  text with spaces  ");
        assert_eq!(output, "text with spaces");
    }

    #[test]
    fn test_clean_model_output_preserves_internal_newlines() {
        let output = clean_model_output("line1\nline2\nline3");
        assert_eq!(output, "line1\nline2\nline3");
    }

    // =========================================================================
    // run_mock_inference Tests
    // =========================================================================

    #[test]
    fn test_mock_inference_basic() {
        let config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("Hello world")
            .with_max_tokens(10);
        let mut mock_config = config;
        mock_config.use_mock_backend = true;

        let result = run_mock_inference(&mock_config).expect("mock inference should succeed");

        assert_eq!(result.format, "Mock");
        assert!(!result.used_gpu);
        assert!(result.text.contains("mock response for: Hello world"));
    }

    #[test]
    fn test_mock_inference_no_prompt() {
        let config = InferenceConfig::new("/test/model.gguf").with_max_tokens(5);
        let mut mock_config = config;
        mock_config.use_mock_backend = true;

        let result = run_mock_inference(&mock_config).expect("mock inference should succeed");

        assert!(result.text.contains("(no prompt)"));
        assert_eq!(result.input_token_count, 1); // BOS token
    }

    #[test]
    fn test_mock_inference_with_input_tokens() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_input_tokens(vec![10, 20, 30])
            .with_max_tokens(5);
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");

        assert_eq!(result.input_token_count, 3);
        assert!(result.tokens.starts_with(&[10, 20, 30]));
    }

    #[test]
    fn test_mock_inference_generated_tokens() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("test")
            .with_max_tokens(5);
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");

        assert_eq!(result.generated_token_count, 5);
        // Generated tokens should be 100, 101, 102, 103, 104
        let generated = &result.tokens[result.input_token_count..];
        assert_eq!(generated, &[100, 101, 102, 103, 104]);
    }

    #[test]
    fn test_mock_inference_max_tokens_capped() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("test")
            .with_max_tokens(100); // Will be capped to 32
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");

        // Mock caps at 32 tokens
        assert_eq!(result.generated_token_count, 32);
    }

    #[test]
    fn test_mock_inference_timing() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("test")
            .with_max_tokens(5);
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");

        // Mock timing: 10ms load, inference_ms = 50 + (tokens * 2)
        assert!((result.load_ms - 10.0).abs() < 0.001);
        // For 5 tokens: 50 + 10 = 60ms
        assert!((result.inference_ms - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_mock_inference_tok_per_sec() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("test")
            .with_max_tokens(5);
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");

        // tok_per_sec = 5 tokens / 0.06 seconds = ~83.33 tok/s
        // Formula: inference_ms = 50 + (5 * 2) = 60ms
        let expected = 5.0 / 0.060;
        assert!((result.tok_per_sec - expected).abs() < 0.01);
    }

    #[test]
    fn test_mock_inference_prompt_tokenization() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("one two three four five")
            .with_max_tokens(3);
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");

        // Mock tokenizes by splitting on whitespace: 5 words = 5 input tokens
        assert_eq!(result.input_token_count, 5);
    }

    // =========================================================================
    // InferenceConfig Builder Tests
    // =========================================================================

    #[test]
    fn test_inference_config_default_values() {
        let config = InferenceConfig::new("/test/model.gguf");

        assert_eq!(config.model_path, PathBuf::from("/test/model.gguf"));
        assert!(config.prompt.is_none());
        assert!(config.input_tokens.is_none());
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < 0.001);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
        assert!(!config.verbose);
        assert!(!config.use_mock_backend);
    }

    #[test]
    fn test_inference_config_with_prompt() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("Hello");
        assert_eq!(config.prompt, Some("Hello".to_string()));
    }

    #[test]
    fn test_inference_config_with_input_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_input_tokens(vec![1, 2, 3]);
        assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
    }

    #[test]
    fn test_inference_config_with_max_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(64);
        assert_eq!(config.max_tokens, 64);
    }

    #[test]
    fn test_inference_config_with_temperature() {
        let config = InferenceConfig::new("/model.gguf").with_temperature(0.7);
        assert!((config.temperature - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_inference_config_with_top_k() {
        let config = InferenceConfig::new("/model.gguf").with_top_k(50);
        assert_eq!(config.top_k, 50);
    }

    #[test]
    fn test_inference_config_without_gpu() {
        let config = InferenceConfig::new("/model.gguf").without_gpu();
        assert!(config.no_gpu);
    }

    #[test]
    fn test_inference_config_with_verbose() {
        let config = InferenceConfig::new("/model.gguf").with_verbose(true);
        assert!(config.verbose);
    }

    #[test]
    fn test_inference_config_with_trace() {
        let config = InferenceConfig::new("/model.gguf").with_trace(true);
        assert!(config.trace);
    }

    #[test]
    fn test_inference_config_with_trace_output() {
        let config = InferenceConfig::new("/model.gguf").with_trace_output("/tmp/trace.json");
        assert_eq!(config.trace_output, Some(PathBuf::from("/tmp/trace.json")));
    }

    #[test]
    fn test_inference_config_chained_builders() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(50)
            .with_temperature(0.5)
            .with_top_k(40)
            .without_gpu()
            .with_verbose(true)
            .with_trace(true)
            .with_trace_output("/output/trace.json");

        assert_eq!(config.prompt, Some("Test prompt".to_string()));
        assert_eq!(config.max_tokens, 50);
        assert!((config.temperature - 0.5).abs() < 0.001);
        assert_eq!(config.top_k, 40);
        assert!(config.no_gpu);
        assert!(config.verbose);
        assert!(config.trace);
        assert_eq!(
            config.trace_output,
            Some(PathBuf::from("/output/trace.json"))
        );
    }

    #[test]
    fn test_inference_config_clone() {
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("test")
            .with_max_tokens(20);

        let cloned = config.clone();
        assert_eq!(cloned.prompt, Some("test".to_string()));
        assert_eq!(cloned.max_tokens, 20);
    }

    #[test]
    fn test_inference_config_debug() {
        let config = InferenceConfig::new("/model.gguf");
        let debug_str = format!("{:?}", config);
        assert!(debug_str.contains("InferenceConfig"));
        assert!(debug_str.contains("model.gguf"));
    }

    // =========================================================================
    // InferenceResult Tests
    // =========================================================================

    #[test]
    fn test_inference_result_clone() {
        let result = InferenceResult {
            text: "output text".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 50.0,
            tok_per_sec: 40.0,
            load_ms: 10.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };

        let cloned = result.clone();
        assert_eq!(cloned.text, "output text");
        assert_eq!(cloned.tokens, vec![1, 2, 3]);
        assert_eq!(cloned.input_token_count, 1);
        assert_eq!(cloned.generated_token_count, 2);
        assert!((cloned.inference_ms - 50.0).abs() < 0.001);
        assert!((cloned.tok_per_sec - 40.0).abs() < 0.001);
        assert!((cloned.load_ms - 10.0).abs() < 0.001);
        assert_eq!(cloned.format, "GGUF");
        assert!(cloned.used_gpu);
    }

    #[test]
    fn test_inference_result_debug() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "Mock".to_string(),
            used_gpu: false,
        };

        let debug_str = format!("{:?}", result);
        assert!(debug_str.contains("InferenceResult"));
        assert!(debug_str.contains("Mock"));
    }

    // =========================================================================
    // run_inference with mock Tests
    // =========================================================================

    #[test]
    fn test_run_inference_mock_backend() {
        let mut config = InferenceConfig::new("/test/nonexistent.gguf")
            .with_prompt("Test prompt")
            .with_max_tokens(5);
        config.use_mock_backend = true;

        // Should succeed without accessing disk
        let result = run_inference(&config).expect("mock inference should succeed");
        assert_eq!(result.format, "Mock");
    }

    // =========================================================================
    // Edge Cases
    // =========================================================================

    #[test]
    fn test_inference_config_empty_prompt() {
        let config = InferenceConfig::new("/model.gguf").with_prompt("");
        assert_eq!(config.prompt, Some(String::new()));
    }

    #[test]
    fn test_inference_config_zero_max_tokens() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(0);
        assert_eq!(config.max_tokens, 0);
    }

    #[test]
    fn test_inference_config_zero_temperature() {
        let config = InferenceConfig::new("/model.gguf").with_temperature(0.0);
        assert!((config.temperature - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_inference_config_zero_top_k() {
        let config = InferenceConfig::new("/model.gguf").with_top_k(0);
        assert_eq!(config.top_k, 0);
    }

    #[test]
    fn test_mock_inference_zero_max_tokens() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("test")
            .with_max_tokens(0);
        config.use_mock_backend = true;

        // Mock returns error for max_tokens == 0
        let result = run_mock_inference(&config);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("max_tokens"));
    }

    #[test]
    fn test_mock_inference_empty_prompt() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("")
            .with_max_tokens(3);
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");
        // Empty string split by whitespace gives 0 tokens, falls back to BOS
        assert_eq!(result.input_token_count, 0);
    }

    #[test]
    fn test_mock_inference_whitespace_only_prompt() {
        let mut config = InferenceConfig::new("/test/model.gguf")
            .with_prompt("   ")
            .with_max_tokens(3);
        config.use_mock_backend = true;

        let result = run_mock_inference(&config).expect("mock inference should succeed");
        // Whitespace-only prompt gives 0 words when split
        assert_eq!(result.input_token_count, 0);
    }

    // =========================================================================
    // prefault_mmap Test (simple coverage)
    // =========================================================================

    #[test]
    fn test_prefault_mmap_empty() {
        let data: [u8; 0] = [];
        prefault_mmap(&data);
        // Should not panic
    }

    #[test]
    fn test_prefault_mmap_small() {
        let data = [1u8, 2, 3, 4, 5];
        prefault_mmap(&data);
        // Should not panic
    }

    #[test]
    fn test_prefault_mmap_page_size() {
        let data = vec![0u8; 4096]; // Exactly one page
        prefault_mmap(&data);
        // Should not panic
    }

    #[test]
    fn test_prefault_mmap_multiple_pages() {
        let data = vec![0u8; 4096 * 3 + 100]; // 3+ pages
        prefault_mmap(&data);
        // Should not panic
    }
}
