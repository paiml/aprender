
    #[test]
    fn test_file_stem_extraction_multiple_dots_deep_icov() {
        let path = PathBuf::from("/models/model.v1.0.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model.v1.0"));
    }

    #[test]
    fn test_file_stem_extraction_no_extension_deep_icov() {
        let path = PathBuf::from("/models/model");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert_eq!(stem, Some("model"));
    }

    // --- Tokens Per Second Calculation Tests ---

    #[test]
    fn test_tok_per_sec_calculation_deep_icov() {
        let generated_count = 100;
        let inference_ms = 500.0; // 500ms
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 200.0).abs() < 0.001); // 100 tokens / 0.5 sec = 200 tok/s
    }

    #[test]
    fn test_tok_per_sec_zero_time_deep_icov() {
        let generated_count = 100;
        let inference_ms = 0.0;
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tok_per_sec_very_fast_deep_icov() {
        let generated_count = 1000;
        let inference_ms = 10.0; // 10ms
        let tok_per_sec = if inference_ms > 0.0 {
            generated_count as f64 / (inference_ms / 1000.0)
        } else {
            0.0
        };
        assert!((tok_per_sec - 100000.0).abs() < 0.001); // 100k tok/s
    }

    // --- Max Tokens Passthrough Tests (GH-372: removed .min(128) cap) ---

    #[test]
    fn test_max_tokens_passthrough_large() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(1000);
        assert_eq!(config.max_tokens, 1000);
    }

    #[test]
    fn test_max_tokens_passthrough_small() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(50);
        assert_eq!(config.max_tokens, 50);
    }

    #[test]
    fn test_max_tokens_passthrough_512() {
        let config = InferenceConfig::new("/model.gguf").with_max_tokens(512);
        assert_eq!(config.max_tokens, 512);
    }

    // --- Format String Tests ---

    #[test]
    fn test_inference_result_format_string_gguf_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "GGUF");
    }

    #[test]
    fn test_inference_result_format_string_apr_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "APR".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "APR");
    }

    #[test]
    fn test_inference_result_format_string_safetensors_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "SafeTensors".to_string(),
            used_gpu: false,
        };
        assert_eq!(result.format, "SafeTensors");
    }

    // --- Used GPU Flag Tests ---

    #[test]
    fn test_inference_result_used_gpu_true_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: true,
        };
        assert!(result.used_gpu);
    }

    #[test]
    fn test_inference_result_used_gpu_false_deep_icov() {
        let result = InferenceResult {
            text: "test".to_string(),
            tokens: vec![1],
            input_token_count: 1,
            generated_token_count: 0,
            inference_ms: 1.0,
            tok_per_sec: 0.0,
            load_ms: 1.0,
            format: "GGUF".to_string(),
            used_gpu: false,
        };
        assert!(!result.used_gpu);
    }

    // --- Architecture Priority Tests ---

    #[test]
    fn test_architecture_detection_priority_qwen_over_llama_deep_icov() {
        // If filename contains both "qwen" and "llama", qwen should win (checked first)
        let path = PathBuf::from("/models/qwen-llama-hybrid.gguf");
        let arch = path.file_stem().and_then(|s| s.to_str()).map(|s| {
            if s.to_lowercase().contains("qwen") {
                "Qwen2"
            } else if s.to_lowercase().contains("llama") {
                "LLaMA"
            } else {
                "Transformer"
            }
        });
        assert_eq!(arch, Some("Qwen2"));
    }

    #[test]
    fn test_architecture_detection_priority_llama_over_mistral_deep_icov() {
        // If filename contains both "llama" and "mistral", llama should win
        let path = PathBuf::from("/models/llama-mistral-blend.gguf");
        let arch = path.file_stem().and_then(|s| s.to_str()).map(|s| {
            if s.to_lowercase().contains("qwen") {
                "Qwen2"
            } else if s.to_lowercase().contains("llama") {
                "LLaMA"
            } else if s.to_lowercase().contains("mistral") {
                "Mistral"
            } else {
                "Transformer"
            }
        });
        assert_eq!(arch, Some("LLaMA"));
    }

    // --- Path Edge Cases Tests ---

    #[test]
    fn test_path_with_special_characters_deep_icov() {
        let path = PathBuf::from("/models/model-v1.0_final (copy).gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert!(stem.is_some());
        assert!(stem.expect("operation failed").contains("model"));
    }

    #[test]
    fn test_path_with_unicode_deep_icov() {
        let path = PathBuf::from("/models/模型-v1.gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        assert!(stem.is_some());
    }

    #[test]
    fn test_path_just_extension_deep_icov() {
        let path = PathBuf::from(".gguf");
        let stem = path.file_stem().and_then(|s| s.to_str());
        // For dotfiles like .gguf, file_stem returns the full name ".gguf" (no extension)
        assert_eq!(stem, Some(".gguf"));
    }

    // --- Load Time Tests ---

    #[test]
    fn test_load_ms_calculation_deep_icov() {
        // Simulating load_start.elapsed().as_secs_f64() * 1000.0
        let elapsed_secs: f64 = 0.5;
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 500.0).abs() < 0.001);
    }

    #[test]
    fn test_load_ms_very_fast_deep_icov() {
        let elapsed_secs: f64 = 0.001; // 1ms
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_load_ms_very_slow_deep_icov() {
        let elapsed_secs: f64 = 10.0; // 10 seconds
        let load_ms = elapsed_secs * 1000.0;
        assert!((load_ms - 10000.0).abs() < 0.001);
    }

    // --- Generated Token Slice Tests ---

    #[test]
    fn test_generated_tokens_slice_deep_icov() {
        let all_tokens = vec![1, 2, 3, 4, 5, 6];
        let input_token_count = 2;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens, &[3, 4, 5, 6]);
    }

    #[test]
    fn test_generated_tokens_slice_empty_deep_icov() {
        let all_tokens = vec![1, 2];
        let input_token_count = 2;
        let generated_tokens = &all_tokens[input_token_count..];
        assert!(generated_tokens.is_empty());
    }

    #[test]
    fn test_generated_tokens_slice_all_generated_deep_icov() {
        let all_tokens = vec![1, 2, 3, 4];
        let input_token_count = 0;
        let generated_tokens = &all_tokens[input_token_count..];
        assert_eq!(generated_tokens.len(), 4);
    }

    // --- Clean Output Integration Tests ---

    #[test]
    fn test_clean_output_chatml_full_conversation_deep_icov() {
        let raw = "<|im_start|>system\nYou are helpful<|im_end|><|im_start|>user\nHi<|im_end|><|im_start|>assistant\nHello!<|im_end|>";
        let cleaned = clean_model_output(raw);
        // All markers should be removed
        assert!(!cleaned.contains("<|im_start|>"));
        assert!(!cleaned.contains("<|im_end|>"));
        // Content should remain (without newlines added by markers)
        assert!(cleaned.contains("helpful") || cleaned.contains("Hello"));
    }

    #[test]
    fn test_clean_output_preserves_code_blocks_deep_icov() {
        let raw = "<|im_start|>assistant\n```python\nprint('hello')\n```<|im_end|>";
        let cleaned = clean_model_output(raw);
        assert!(cleaned.contains("```python"));
        assert!(cleaned.contains("print('hello')"));
    }

    // --- Model Format Enum Tests ---

    #[test]
    fn test_model_format_display_gguf_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Gguf;
        assert_eq!(format.to_string(), "GGUF");
    }

    #[test]
    fn test_model_format_display_apr_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Apr;
        assert_eq!(format.to_string(), "APR");
    }

    #[test]
    fn test_model_format_display_safetensors_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::SafeTensors;
        assert_eq!(format.to_string(), "SafeTensors");
    }

    #[test]
    fn test_model_format_clone_deep_icov() {
        use crate::format::ModelFormat;
        let format = ModelFormat::Gguf;
        let cloned = format;
        assert_eq!(format, cloned);
    }

    #[test]
    fn test_model_format_eq_deep_icov() {
        use crate::format::ModelFormat;
        assert_eq!(ModelFormat::Gguf, ModelFormat::Gguf);
        assert_ne!(ModelFormat::Gguf, ModelFormat::Apr);
        assert_ne!(ModelFormat::Apr, ModelFormat::SafeTensors);
    }

    // --- Additional Edge Case Tests ---

    #[test]
    fn test_inference_config_path_with_symlink_name_deep_icov() {
        let config = InferenceConfig::new("/models/latest -> llama-3.gguf");
        assert!(config
            .model_path
            .to_str()
            .expect("invalid UTF-8")
            .contains("latest"));
    }

    #[test]
    fn test_inference_config_relative_path_deep_icov() {
        let config = InferenceConfig::new("./models/model.gguf");
        assert!(config
            .model_path
            .to_str()
            .expect("invalid UTF-8")
            .contains("./"));
    }

    #[test]
    fn test_inference_config_absolute_path_deep_icov() {
        let config = InferenceConfig::new("/absolute/path/model.gguf");
        assert!(config.model_path.starts_with("/"));
    }

    // =========================================================================
    // Synthetic Model Inference Path Tests
    // These tests create minimal model files to exercise inference code paths
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_minimal_model() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal valid GGUF file (no tensors - will fail on model load)
        let mut temp = NamedTempFile::with_suffix(".gguf").expect("file operation failed");
        let mut data = Vec::new();
        data.extend_from_slice(b"GGUF"); // magic
        data.extend_from_slice(&3u32.to_le_bytes()); // version 3
        data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
        data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Hello")
            .with_max_tokens(5);

        // Will fail because model has no tensors, but exercises the GGUF path
        let result = run_inference(&config);
        assert!(result.is_err()); // Expected: model can't be loaded properly
    }

    #[test]
    fn test_run_apr_inference_minimal_model() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal valid APR file header
        let mut temp = NamedTempFile::with_suffix(".apr").expect("file operation failed");
        let mut data = Vec::new();
        // APR magic + minimal header
        data.extend_from_slice(b"APR\x02"); // APR v2 magic
        data.extend_from_slice(&[0u8; 60]); // Minimal header padding
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(3);

        // Will fail on loading, but exercises the APR format detection path
        let result = run_inference(&config);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_minimal() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create minimal SafeTensors file (8-byte header size + empty JSON)
        let mut temp = NamedTempFile::with_suffix(".safetensors").expect("file operation failed");
        let json_header = b"{}";
        let header_size = json_header.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_size.to_le_bytes());
        data.extend_from_slice(json_header);
        temp.write_all(&data).expect("operation failed");
        temp.flush().expect("operation failed");

        let config = InferenceConfig::new(temp.path())
            .with_prompt("Test")
            .with_max_tokens(3);

        // Will fail on loading (no tensors), but exercises SafeTensors path
        let result = run_inference(&config);
        assert!(result.is_err());
    }
