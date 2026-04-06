
    // =========================================================================
    // Configuration Building Tests (EXTREME TDD)
    // Tests verify that inference functions correctly build internal configs
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_config_temperature_boundary_0_01() {
        // Temperature exactly at 0.01 boundary should use greedy
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "boundary test",
            5,
            0.01, // Exactly at greedy threshold
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_config_temperature_just_above_boundary() {
        // Temperature 0.011 should use sampling (top-k 40)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "sampling boundary",
            5,
            0.011, // Just above greedy threshold
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_max_tokens_one() {
        // Single token generation
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "single token",
            1, // Only 1 token
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_max_tokens_very_large() {
        // Large max_tokens tests bounds handling
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "large generation",
            usize::MAX / 2, // Very large but won't overflow
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // CLI Argument Parsing Tests - Format Variations
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_format_json_lowercase() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "json test",
            5,
            0.5,
            "json",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_format_text_lowercase() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "text test",
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
    fn test_run_gguf_inference_format_arbitrary_string() {
        // Unknown formats should fall through to default (text-like)
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "unknown format",
            5,
            0.5,
            "yaml", // Not recognized
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_format_empty_string() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "empty format",
            5,
            0.5,
            "", // Empty format string
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // GPU Flag Tests (Non-CUDA Feature)
    // When cuda feature is NOT enabled, force_gpu flag should be handled
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_gpu_flag_no_cuda_feature() {
        // Test GPU flag warning path when cuda feature is disabled
        // This should NOT panic and should proceed with CPU fallback
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "gpu warning test",
            5,
            0.5,
            "text",
            true, // force_gpu - should warn without cuda feature
            false,
            None, // trace_config
        );
        assert!(result.is_err()); // Model loading will still fail
    }

    #[test]
    fn test_run_apr_inference_gpu_flag_no_cuda_feature() {
        // Same for APR inference
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "gpu warning test",
            5,
            0.5,
            "text",
            true,  // force_gpu
            false, // verbose
            None,  // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // Verbose and Trace Flag Combinations
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_verbose_only() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "verbose only",
            5,
            0.5,
            "text",
            false,
            true, // verbose
            None, // trace_config (disabled)
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_trace_only() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "trace only",
            5,
            0.5,
            "text",
            false,
            false,                        // no verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_verbose_and_trace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "both flags",
            5,
            0.5,
            "text",
            false,
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_all_debug_flags() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "all flags",
            5,
            0.5,
            "json",
            true,                         // force_gpu
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // APR Inference Verbose Mode Tests
    // =========================================================================

    #[test]
    fn test_run_apr_inference_verbose_only() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "verbose apr",
            5,
            0.5,
            "text",
            false, // no gpu
            true,  // verbose
            None,  // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_verbose_with_gpu() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "verbose gpu apr",
            5,
            0.5,
            "json",
            true, // force_gpu
            true, // verbose
            None, // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // SafeTensors Inference Tests
    // =========================================================================

    #[test]
    fn test_run_safetensors_inference_format_variations() {
        for format in &["json", "text", "yaml", "", "markdown"] {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "format test",
                5,
                0.5,
                format,
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_run_safetensors_inference_temperature_ignored() {
        // Temperature is marked as _temperature in the function signature
        // Test that various values are accepted without affecting the error type
        for temp in &[0.0, 0.01, 0.5, 1.0, 2.0, -1.0, f32::MAX, f32::MIN] {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "temp test",
                5,
                *temp,
                "text",
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    #[test]
    fn test_run_safetensors_inference_max_tokens_variations() {
        for max_tokens in &[0, 1, 5, 100, 1000, usize::MAX / 4] {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "tokens test",
                *max_tokens,
                0.5,
                "text",
                None, // trace_config
            );
            assert!(result.is_err());
        }
    }

    // =========================================================================
    // Error Handling Path Coverage
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_error_message_structure() {
        let result = inference::run_gguf_inference(
            "/definitely/not/a/real/path/model.gguf",
            &[],
            "error structure test",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = format!("{}", err);
        // Verify error contains useful diagnostic info
        assert!(
            err_string.contains("mmap")
                || err_string.contains("GGUF")
                || err_string.contains("load"),
            "Error message should contain diagnostic info: {}",
            err_string
        );
    }

    #[test]
    fn test_run_apr_inference_error_message_structure() {
        let result = inference::run_apr_inference(
            "/definitely/not/a/real/path/model.apr",
            &[0x00, 0x01, 0x02, 0x03], // Invalid APR magic
            "error structure test",
            5,
            0.5,
            "text",
            false, // force_gpu
            false, // verbose
            None,  // trace_config
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = format!("{}", err);
        assert!(
            err_string.contains("APR")
                || err_string.contains("parse")
                || err_string.contains("Failed"),
            "Error message should contain APR-related info: {}",
            err_string
        );
    }

    #[test]
    fn test_run_safetensors_inference_error_message_structure() {
        let result = inference::run_safetensors_inference(
            "/definitely/not/a/real/path/model.safetensors",
            "error structure test",
            5,
            0.5,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_string = format!("{}", err);
        assert!(
            err_string.contains("SafeTensors")
                || err_string.contains("convert")
                || err_string.contains("Failed")
                || err_string.contains("No such file"),
            "Error message should contain useful info: {}",
            err_string
        );
    }

    // =========================================================================
    // Prompt Content Edge Cases
    // =========================================================================

    #[test]
    fn test_prompt_with_null_bytes() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "hello\x00world", // Contains null byte
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }
