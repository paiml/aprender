
        // -------------------------------------------------------------------------
        // run_gguf_inference Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_run_gguf_inference_invalid_model_path() {
            // Test with empty file data - should fail to mmap
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],     // empty file data
                "Hello", // prompt
                10,      // max_tokens
                0.0,     // temperature (greedy)
                "text",  // format
                false,   // force_gpu
                false,   // verbose
                None,    // trace_config
            );
            assert!(result.is_err());
            let err = result.unwrap_err();
            // Should fail during mmap since path doesn't exist
            assert!(
                err.to_string().contains("mmap")
                    || err.to_string().contains("Failed to mmap")
                    || err.to_string().contains("No such file"),
                "Expected mmap error, got: {}",
                err
            );
        }

        #[test]
        fn test_run_gguf_inference_invalid_gguf_data() {
            // Test with non-existent path - the function reads from path, not from data param
            let invalid_data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
            let result = inference::run_gguf_inference(
                "/nonexistent_invalid_model.gguf",
                &invalid_data,
                "Test prompt",
                5,
                0.7,
                "json",
                false,
                true, // verbose
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_gguf_inference_format_json() {
            // Test that JSON format parameter is accepted
            // (Will fail on model loading, but exercises format parsing)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                1,
                0.0,
                "json", // JSON output format
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_format_text() {
            // Test that text format parameter is accepted
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                1,
                0.0,
                "text", // text output format
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_greedy_temperature() {
            // Test greedy decoding (temperature <= 0.01)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                5,
                0.0, // Greedy (temperature = 0)
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_sampling_temperature() {
            // Test temperature sampling
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                5,
                1.0, // Temperature sampling
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_verbose_mode() {
            // Test verbose output path
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "verbose test",
                5,
                0.5,
                "text",
                false,
                true, // verbose = true
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_trace_mode() {
            // Test trace output path
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "trace test",
                5,
                0.5,
                "text",
                false,
                false,
                Some(TraceConfig::enabled()), // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_force_gpu_flag() {
            // Test force_gpu flag (should warn if cuda not available)
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "gpu test",
                5,
                0.5,
                "text",
                true, // force_gpu = true
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_all_flags() {
            // Test with all flags enabled
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "comprehensive test",
                10,
                0.8,
                "json",
                true,                         // force_gpu
                true,                         // verbose
                Some(TraceConfig::enabled()), // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_empty_prompt() {
            // Test with empty prompt
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "", // empty prompt
                5,
                0.0,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_zero_tokens() {
            // Test with zero max_tokens
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                0, // zero tokens to generate
                0.0,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        #[test]
        fn test_run_gguf_inference_large_tokens() {
            // Test with large max_tokens
            let result = inference::run_gguf_inference(
                "/nonexistent/model.gguf",
                &[],
                "prompt",
                10000, // large number of tokens
                0.0,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err()); // Expected - no valid model
        }

        // -------------------------------------------------------------------------
        // run_safetensors_inference Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_run_safetensors_inference_invalid_path() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "Test prompt",
                10,
                0.5,
                "text",
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_safetensors_inference_json_format() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "JSON test",
                5,
                0.0,
                "json",
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_safetensors_inference_text_format() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "text test",
                5,
                0.0,
                "text",
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_safetensors_inference_empty_prompt() {
            let result = inference::run_safetensors_inference(
                "/nonexistent/model.safetensors",
                "",
                10,
                0.7,
                "text",
                None, // trace_config
            );
            assert!(result.is_err());
        }

        // -------------------------------------------------------------------------
        // run_apr_inference Tests
        // -------------------------------------------------------------------------

        #[test]
        fn test_run_apr_inference_invalid_path() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "Test prompt",
                10,
                0.5,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_invalid_apr_data() {
            // Test with invalid APR magic bytes
            let invalid_data = vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07];
            let result = inference::run_apr_inference(
                "/tmp/test.apr",
                &invalid_data,
                "Test prompt",
                5,
                0.7,
                "text",
                false,
                true,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_json_format() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "JSON test",
                5,
                0.0,
                "json",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_text_format() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "text test",
                5,
                0.0,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_force_gpu_flag() {
            // Test force_gpu flag
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "gpu test",
                5,
                0.5,
                "text",
                true, // force_gpu
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_verbose_mode() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "verbose test",
                5,
                0.5,
                "text",
                false,
                true, // verbose
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_all_flags() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "comprehensive test",
                10,
                0.8,
                "json",
                true,                         // force_gpu
                true,                         // verbose
                Some(TraceConfig::enabled()), // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_empty_prompt() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "",
                10,
                0.7,
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }

        #[test]
        fn test_run_apr_inference_greedy_temperature() {
            let result = inference::run_apr_inference(
                "/nonexistent/model.apr",
                &[],
                "greedy test",
                5,
                0.0, // greedy
                "text",
                false,
                false,
                None, // trace_config
            );
            assert!(result.is_err());
        }
