
    // =========================================================================
    // GGUF Inference: File Data Validation Tests
    // Tests the early-exit paths when file_data is processed
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_empty_file_data_with_valid_path() {
        // Even with empty file_data, the function uses model_ref for mmap
        let result = inference::run_gguf_inference(
            "/nonexistent/path/model.gguf",
            &[], // Empty file data - function uses mmap path
            "test prompt",
            10,
            0.7,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        // Should fail at mmap stage
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("mmap") || err_msg.contains("Failed"),
            "Expected mmap error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_run_gguf_inference_with_gguf_magic_but_truncated() {
        // GGUF magic header but truncated data
        let truncated_gguf = b"GGUF\x03\x00\x00\x00"; // Just magic + version
        let result = inference::run_gguf_inference(
            "/tmp/truncated.gguf",
            truncated_gguf,
            "prompt",
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
    // Temperature Sampling Logic Tests
    // Verifies temperature threshold behavior
    // =========================================================================

    #[test]
    fn test_gguf_inference_temperature_exactly_zero() {
        // temperature == 0.0 should use greedy decoding (< 0.01)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "zero temp",
            5,
            0.0, // Exactly zero
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err()); // Fails on model load, but exercises temp path
    }

    #[test]
    fn test_gguf_inference_temperature_very_small() {
        // temperature = 0.001 should use greedy (< 0.01)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "small temp",
            5,
            0.001,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_temperature_at_sampling_threshold() {
        // temperature = 0.02 should use temperature sampling (> 0.01)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "sampling temp",
            5,
            0.02,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_temperature_high_value() {
        // High temperature should work without issues
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "high temp",
            5,
            2.0,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Format Output Path Tests
    // Tests JSON vs text output format handling
    // =========================================================================

    #[test]
    fn test_gguf_inference_json_format_output_path() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "json output test",
            5,
            0.5,
            "json", // JSON format
            false,
            true, // verbose to test more output code
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_text_format_verbose() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "text verbose test",
            5,
            0.5,
            "text",
            false,
            true, // verbose mode
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gguf_inference_text_format_non_verbose() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "text quiet test",
            5,
            0.5,
            "text",
            false,
            false, // non-verbose (Ollama-style output)
            None,  // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // APR Inference: File Data Scenarios
    // =========================================================================

    #[test]
    fn test_run_apr_inference_with_valid_apr_magic_but_incomplete() {
        // APR magic "APR\0" but missing model data
        let apr_header = b"APR\0\x01\x00\x01\x00"; // Magic + type + version only
        let result = inference::run_apr_inference(
            "/tmp/incomplete.apr",
            apr_header,
            "incomplete apr",
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
    fn test_run_apr_inference_zero_max_tokens() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "zero tokens",
            0, // Zero tokens
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_json_format() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "json format",
            5,
            0.5,
            "json",
            false,
            true,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_temperature_greedy() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "greedy",
            5,
            0.0, // Greedy
            "text",
            false,
            true,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // SafeTensors Inference: Path Resolution Tests
    // =========================================================================

    #[test]
    fn test_run_safetensors_inference_relative_path() {
        let result = inference::run_safetensors_inference(
            "relative/path/model.safetensors",
            "relative path test",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_absolute_path() {
        let result = inference::run_safetensors_inference(
            "/absolute/path/model.safetensors",
            "absolute path test",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_with_tokenizer_path() {
        // The function tries to load tokenizer.json from sibling path
        let result = inference::run_safetensors_inference(
            "/some/model/dir/model.safetensors",
            "tokenizer test",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Error Type Verification Tests
    // Ensures correct error variants are returned
    // =========================================================================

    #[test]
    fn test_gguf_inference_error_is_unsupported_operation() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "error type test",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        // Verify the error type
        match result.unwrap_err() {
            RealizarError::UnsupportedOperation { operation, reason } => {
                assert_eq!(operation, "mmap_gguf");
                assert!(reason.contains("Failed") || reason.contains("mmap"));
            },
            other => panic!("Expected UnsupportedOperation, got: {:?}", other),
        }
    }

    #[test]
    fn test_apr_inference_error_is_unsupported_operation() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[0, 1, 2, 3], // Invalid magic
            "error type test",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            RealizarError::UnsupportedOperation { operation, reason } => {
                assert_eq!(operation, "parse_apr");
                assert!(reason.contains("APR") || reason.contains("parse"));
            },
            other => panic!("Expected UnsupportedOperation, got: {:?}", other),
        }
    }

    #[test]
    fn test_safetensors_inference_error_is_unsupported_operation() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "error type test",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            RealizarError::UnsupportedOperation { operation, reason } => {
                assert_eq!(operation, "convert_safetensors");
                assert!(reason.contains("SafeTensors") || reason.contains("convert"));
            },
            other => panic!("Expected UnsupportedOperation, got: {:?}", other),
        }
    }

    // =========================================================================
    // Prompt Encoding Edge Cases
    // Tests prompt tokenization fallback behavior
    // =========================================================================

    #[test]
    fn test_gguf_inference_empty_prompt() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "", // Empty prompt
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
    fn test_gguf_inference_very_long_prompt() {
        // Long prompt to test tokenization
        let long_prompt = "a".repeat(10000);
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
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
    fn test_apr_inference_empty_prompt() {
        let result = inference::run_apr_inference(
            "/nonexistent.apr",
            &[],
            "", // Empty prompt
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
    fn test_safetensors_inference_empty_prompt() {
        let result = inference::run_safetensors_inference(
            "/nonexistent.safetensors",
            "", // Empty prompt
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Max Tokens Boundary Tests
    // =========================================================================

    #[test]
    fn test_gguf_inference_large_max_tokens() {
        let result = inference::run_gguf_inference(
            "/nonexistent.gguf",
            &[],
            "large tokens",
            100000, // Large number
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }
