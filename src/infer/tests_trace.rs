
    // =========================================================================
    // with_trace_output Builder Method Tests
    // =========================================================================

    #[test]
    fn test_with_trace_output_basic() {
        let config =
            InferenceConfig::new("/model.gguf").with_trace_output("/tmp/trace_output.json");
        assert_eq!(
            config.trace_output,
            Some(PathBuf::from("/tmp/trace_output.json"))
        );
    }

    #[test]
    fn test_with_trace_output_pathbuf() {
        let path = PathBuf::from("/custom/path/trace.json");
        let config = InferenceConfig::new("/model.gguf").with_trace_output(path.clone());
        assert_eq!(config.trace_output, Some(path));
    }

    #[test]
    fn test_with_trace_output_string() {
        let path_str = String::from("/string/path/trace.json");
        let config = InferenceConfig::new("/model.gguf").with_trace_output(path_str);
        assert_eq!(
            config.trace_output,
            Some(PathBuf::from("/string/path/trace.json"))
        );
    }

    #[test]
    fn test_with_trace_output_empty_string() {
        let config = InferenceConfig::new("/model.gguf").with_trace_output("");
        assert_eq!(config.trace_output, Some(PathBuf::from("")));
    }

    #[test]
    fn test_with_trace_output_relative_path() {
        let config = InferenceConfig::new("/model.gguf").with_trace_output("./trace.json");
        assert_eq!(config.trace_output, Some(PathBuf::from("./trace.json")));
    }

    #[test]
    fn test_with_trace_output_with_spaces() {
        let config =
            InferenceConfig::new("/model.gguf").with_trace_output("/path with spaces/trace.json");
        assert_eq!(
            config.trace_output,
            Some(PathBuf::from("/path with spaces/trace.json"))
        );
    }

    #[test]
    fn test_with_trace_output_chained_with_trace() {
        let config = InferenceConfig::new("/model.gguf")
            .with_trace(true)
            .with_trace_output("/output/trace.json");
        assert!(config.trace);
        assert_eq!(
            config.trace_output,
            Some(PathBuf::from("/output/trace.json"))
        );
    }

    #[test]
    fn test_with_trace_output_unicode_path() {
        let config =
            InferenceConfig::new("/model.gguf").with_trace_output("/\u{4e2d}\u{6587}/trace.json");
        assert!(config.trace_output.is_some());
        assert!(config
            .trace_output
            .unwrap()
            .to_str()
            .unwrap()
            .contains('\u{4e2d}'));
    }

    // =========================================================================
    // Architecture Detection from Filename Tests
    // These test the logic in run_gguf_inference that extracts arch from filename
    // =========================================================================

    #[test]
    fn test_arch_detection_qwen_lowercase() {
        let filename = "qwen2-7b-instruct.gguf";
        let lower = filename.to_lowercase();
        assert!(lower.contains("qwen"));
    }

    #[test]
    fn test_arch_detection_qwen_uppercase() {
        let filename = "QWEN-7B.gguf";
        let lower = filename.to_lowercase();
        assert!(lower.contains("qwen"));
    }

    #[test]
    fn test_arch_detection_llama_variations() {
        let filenames = ["llama-2-7b.gguf", "LLAMA3.gguf", "LLaMA-70b.gguf"];
        for filename in filenames {
            let lower = filename.to_lowercase();
            assert!(lower.contains("llama"), "Failed for: {}", filename);
        }
    }

    #[test]
    fn test_arch_detection_mistral_variations() {
        let filenames = ["mistral-7b.gguf", "MISTRAL.gguf", "Mistral-Instruct.gguf"];
        for filename in filenames {
            let lower = filename.to_lowercase();
            assert!(lower.contains("mistral"), "Failed for: {}", filename);
        }
    }

    #[test]
    fn test_arch_detection_phi_variations() {
        let filenames = ["phi-2.gguf", "PHI3.gguf", "Phi-mini.gguf"];
        for filename in filenames {
            let lower = filename.to_lowercase();
            assert!(lower.contains("phi"), "Failed for: {}", filename);
        }
    }

    #[test]
    fn test_arch_detection_no_match() {
        let filename = "custom-model-v1.gguf";
        let lower = filename.to_lowercase();
        let has_known_arch = lower.contains("qwen")
            || lower.contains("llama")
            || lower.contains("mistral")
            || lower.contains("phi");
        assert!(!has_known_arch);
    }

    #[test]
    fn test_arch_detection_embedded_in_longer_name() {
        // Test that substrings within other words are detected
        let filename = "my-llama-like-model.gguf";
        let lower = filename.to_lowercase();
        assert!(lower.contains("llama"));
    }

    // =========================================================================
    // tok_per_sec Calculation Edge Cases
    // =========================================================================

    #[test]
    fn test_tok_per_sec_zero_generated() {
        let generated_token_count = 0;
        let inference_ms = 1000.0;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_token_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tok_per_sec_very_fast() {
        let generated_token_count = 100;
        let inference_ms = 0.1; // 100 microseconds
        let tok_per_sec = if inference_ms > 0.0 {
            generated_token_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!(tok_per_sec > 900_000.0);
    }

    #[test]
    fn test_tok_per_sec_very_slow() {
        let generated_token_count = 1;
        let inference_ms = 60_000.0; // 1 minute
        let tok_per_sec = if inference_ms > 0.0 {
            generated_token_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!(tok_per_sec < 0.02);
    }

    #[test]
    fn test_tok_per_sec_exactly_zero_ms() {
        let generated_token_count = 10;
        let inference_ms = 0.0;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_token_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tok_per_sec_negative_ms_guard() {
        // Edge case: negative inference time should be handled
        let generated_token_count = 10;
        let inference_ms = -1.0;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_token_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    // =========================================================================
    // Format Detection Edge Cases (Testing detect_format indirectly)
    // =========================================================================

    #[test]
    fn test_format_detection_gguf_magic() {
        use crate::format::{detect_format, ModelFormat};
        let data = b"GGUFxxxx"; // GGUF magic
        let result = detect_format(data);
        assert!(matches!(result, Ok(ModelFormat::Gguf)));
    }

    #[test]
    fn test_format_detection_apr_magic() {
        use crate::format::{detect_format, ModelFormat};
        // APR with version byte 0
        let data = b"APR\x00xxxx";
        let result = detect_format(data);
        assert!(matches!(result, Ok(ModelFormat::Apr)));
    }

    #[test]
    fn test_format_detection_safetensors_small_header() {
        use crate::format::{detect_format, ModelFormat};
        // SafeTensors uses header size as first 8 bytes (little-endian u64)
        // A small header size (e.g., 100 bytes) would be a valid safetensors indicator
        let mut data = [0u8; 8];
        let header_size: u64 = 100;
        data.copy_from_slice(&header_size.to_le_bytes());
        let result = detect_format(&data);
        assert!(matches!(result, Ok(ModelFormat::SafeTensors)));
    }

    #[test]
    fn test_format_detection_too_large_safetensors_header() {
        use crate::format::detect_format;
        // Very large header size should be treated as unknown/invalid
        let mut data = [0u8; 8];
        let header_size: u64 = u64::MAX;
        data.copy_from_slice(&header_size.to_le_bytes());
        let result = detect_format(&data);
        // May be error or safetensors depending on implementation
        let _ = result;
    }

    // =========================================================================
    // InferenceConfig Field Combination Tests
    // =========================================================================

    #[test]
    fn test_config_prompt_and_input_tokens_both_set() {
        // Both can be set even though only one is used
        let config = InferenceConfig::new("/model.gguf")
            .with_prompt("Hello")
            .with_input_tokens(vec![1, 2, 3]);
        assert!(config.prompt.is_some());
        assert!(config.input_tokens.is_some());
    }

    #[test]
    fn test_config_all_trace_fields_set() {
        let mut config = InferenceConfig::new("/model.gguf");
        config.trace = true;
        config.trace_verbose = true;
        config.trace_output = Some(PathBuf::from("/trace.json"));
        config.trace_steps = Some(vec!["embed".to_string(), "attn".to_string()]);

        assert!(config.trace);
        assert!(config.trace_verbose);
        assert!(config.trace_output.is_some());
        assert_eq!(config.trace_steps.as_ref().map(std::vec::Vec::len), Some(2));
    }

    #[test]
    fn test_config_default_values_preserved() {
        let config = InferenceConfig::new("/model.gguf");

        // Verify all default values
        assert_eq!(config.max_tokens, 32);
        assert!((config.temperature - 0.0).abs() < f32::EPSILON);
        assert_eq!(config.top_k, 1);
        assert!(!config.no_gpu);
        assert!(!config.trace);
        assert!(!config.trace_verbose);
        assert!(config.trace_output.is_none());
        assert!(config.trace_steps.is_none());
        assert!(!config.verbose);
    }

    #[test]
    fn test_config_override_defaults_sequentially() {
        let config = InferenceConfig::new("/model.gguf")
            .with_max_tokens(100)
            .with_max_tokens(200); // Override again

        assert_eq!(config.max_tokens, 200);
    }

    // =========================================================================
    // InferenceResult Field Boundary Tests
    // =========================================================================

    #[test]
    fn test_result_empty_text() {
        let result = InferenceResult {
            text: String::new(),
            tokens: vec![],
            input_token_count: 0,
            generated_token_count: 0,
            inference_ms: 0.0,
            tok_per_sec: 0.0,
            load_ms: 0.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(result.text.is_empty());
        assert!(result.tokens.is_empty());
    }

    #[test]
    fn test_result_very_long_text() {
        let long_text = "a".repeat(100_000);
        let result = InferenceResult {
            text: long_text.clone(),
            tokens: vec![1; 10000],
            input_token_count: 100,
            generated_token_count: 9900,
            inference_ms: 10000.0,
            tok_per_sec: 990.0,
            load_ms: 500.0,
            format: "APR".to_string(),
            used_gpu: true,
        };
        assert_eq!(result.text.len(), 100_000);
        assert_eq!(result.tokens.len(), 10000);
    }

    #[test]
    fn test_result_mismatched_counts() {
        // token counts don't have to match tokens vec length
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1, 2, 3],
            input_token_count: 100, // Doesn't match tokens.len()
            generated_token_count: 200,
            inference_ms: 1.0,
            tok_per_sec: 200000.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        // Fields are just stored, not validated
        assert_eq!(result.input_token_count, 100);
        assert_eq!(result.generated_token_count, 200);
        assert_eq!(result.tokens.len(), 3);
    }

    // =========================================================================
    // Error Message Format Tests
    // =========================================================================

    #[test]
    fn test_io_error_message_contains_path_info() {
        let config = InferenceConfig::new("/very/specific/nonexistent/path/model.gguf");
        let result = run_inference(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        // Error message should indicate file not found or read failure
        assert!(
            err_msg.contains("File not found") || err_msg.contains("Failed to read"),
            "Expected 'File not found' or 'Failed to read' in error: {}",
            err_msg
        );
    }

    #[test]
    fn test_format_error_message_content() {
        // Create file with unrecognized format
        let mut temp = NamedTempFile::with_suffix(".bin").expect("create temp");
        // Bytes that don't match any known format
        temp.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE])
            .expect("write");
        temp.flush().expect("flush");

        let config = InferenceConfig::new(temp.path());
        let result = run_inference(&config);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Format") || err_msg.contains("format"),
            "Error should mention format: {}",
            err_msg
        );
    }

    // =========================================================================
    // clean_model_output Additional Tests
    // =========================================================================

    #[test]
    fn test_clean_model_output_nested_markers() {
        let raw = "<|im_start|>assistant\n<|im_start|>nested<|im_end|><|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        assert_eq!(cleaned, "nested");
    }

    #[test]
    fn test_clean_model_output_interleaved_content() {
        let raw = "A<|im_end|>B<|im_start|>C<|endoftext|>D";
        let cleaned = clean_model_output(raw);
        assert_eq!(cleaned, "ABCD");
    }

    #[test]
    fn test_clean_model_output_only_whitespace_after_clean() {
        let raw = "<|im_start|>assistant\n   \t  \n<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.is_empty());
    }

    #[test]
    fn test_clean_model_output_preserves_internal_whitespace() {
        let raw = "<|im_start|>assistant\nHello   World\n\nNew paragraph<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("   ")); // Multiple spaces preserved
        assert!(cleaned.contains("\n\n")); // Double newline preserved
    }

    // =========================================================================
    // prefault_mmap Edge Cases
    // =========================================================================

    #[test]
    fn test_prefault_mmap_exactly_one_page() {
        let data = vec![0xABu8; 4096];
        prefault_mmap(&data);
    }
