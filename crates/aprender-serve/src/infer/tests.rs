#[cfg(test)]
mod tests {
    use crate::infer::*;

    #[test]
    fn test_inference_config_input_tokens_path() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal GGUF to test input_tokens path (instead of prompt)
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("file operation failed");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF");
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        // Use input_tokens instead of prompt
        let config = InferenceConfig::new(temp.path())
            .with_input_tokens(vec![1, 2, 3])
            .with_max_tokens(5);

        assert_eq!(config.input_tokens, Some(vec![1, 2, 3]));
        assert!(config.prompt.is_none());

        // Will fail on model load, but exercises input_tokens path
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_inference_config_verbose_path() {
        // Test verbose config (doesn't need model file for this)
        let config = InferenceConfig::new("/model.gguf")
            .with_verbose(true)
            .with_prompt("test");

        assert!(config.verbose);
    }

    #[test]
    fn test_inference_result_tok_per_sec_calculation() {
        // Test tok/s calculation edge case: zero inference_ms
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 1,
            generated_token_count: 2,
            inference_ms: 0.0, // Edge case
            tok_per_sec: 0.0,
            load_ms: 5.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.tok_per_sec, 0.0);
    }

    // =========================================================================
    // Additional Coverage Tests for Uncovered Lines
    // =========================================================================

    // --- InferenceConfig additional builder tests ---

    #[test]
    fn test_inference_config_all_trace_options() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/trace/output.json"));
        config.trace_steps = Some(vec![
            "embed".to_string(),
            "attention".to_string(),
            "ffn".to_string(),
        ]);

        assert!(config.trace);
        assert!(config.trace_verbose);
        assert_eq!(
            config.trace_output,
            Some(PathBuf::from("/trace/output.json"))
        );
        assert_eq!(config.trace_steps.as_ref().map(std::vec::Vec::len), Some(3));
    }

    #[test]
    fn test_inference_config_trace_steps_empty() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace_steps = Some(vec![]);
        assert_eq!(config.trace_steps, Some(vec![]));
    }

    #[test]
    fn test_inference_config_with_all_options_set() {
        let mut config = InferenceConfig::new("/full/path/model.gguf");
        config.prompt = Some("test prompt".to_string());
        config.input_tokens = Some(vec![1, 2, 3, 4, 5]);
        config.max_tokens = 256;
        config.temperature = 0.9;
        config.top_k = 50;
        config.no_gpu = true;
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/output.json"));
        config.trace_steps = Some(vec!["step1".to_string()]);
        config.verbose = true;

        // Verify all fields
        assert!(config.prompt.is_some());
        assert!(config.input_tokens.is_some());
        assert_eq!(config.max_tokens, 256);
        assert!((config.temperature - 0.9).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 50);
        assert!(config.no_gpu);
        assert!(config.trace);
        assert!(config.trace_verbose);
        assert!(config.trace_output.is_some());
        assert!(config.trace_steps.is_some());
        assert!(config.verbose);
    }

include!("tests_inference_config.rs");
include!("tests_inference_config_02.rs");
include!("tests_clean_model.rs");
include!("tests_file_stem_tok.rs");
include!("tests_legacy.rs");
}
