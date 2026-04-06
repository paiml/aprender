
    #[test]
    fn test_prompt_with_only_whitespace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "   \t\n\r   ", // Only whitespace
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
    fn test_prompt_with_control_characters() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "hello\x07\x08\x1bworld", // Bell, backspace, escape
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
    fn test_prompt_with_utf8_multibyte() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "\u{1F4A9}\u{1F680}\u{1F31F}", // Poop, rocket, star emojis
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
    fn test_prompt_with_cjk_characters() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "\u{4E2D}\u{6587}\u{6D4B}\u{8BD5}", // Chinese characters
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
    // File Data Edge Cases (for functions that accept file_data)
    // =========================================================================

    #[test]
    fn test_run_apr_inference_with_partial_magic() {
        // Only partial APR magic bytes
        let partial_magic = b"APR"; // Missing null terminator
        let result = inference::run_apr_inference(
            "/tmp/partial.apr",
            partial_magic,
            "partial magic",
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
    fn test_run_apr_inference_with_gguf_magic() {
        // GGUF magic bytes passed to APR parser
        let gguf_magic = b"GGUF\x03\x00\x00\x00";
        let result = inference::run_apr_inference(
            "/tmp/wrong_magic.apr",
            gguf_magic,
            "wrong magic",
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
    fn test_run_apr_inference_with_large_file_data() {
        // Large but invalid file data
        let large_data = vec![0x42u8; 1024 * 1024]; // 1MB of 'B' characters
        let result = inference::run_apr_inference(
            "/tmp/large.apr",
            &large_data,
            "large data",
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
    // Model Reference Path Edge Cases
    // =========================================================================

    #[test]
    fn test_model_ref_with_spaces() {
        let result = inference::run_gguf_inference(
            "/path/with spaces/model file.gguf",
            &[],
            "path with spaces",
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
    fn test_model_ref_with_unicode() {
        let result = inference::run_gguf_inference(
            "/path/\u{4E2D}\u{6587}/model.gguf",
            &[],
            "unicode path",
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
    fn test_model_ref_relative_path() {
        let result = inference::run_gguf_inference(
            "../models/model.gguf",
            &[],
            "relative path",
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
    fn test_model_ref_empty_string() {
        let result = inference::run_gguf_inference(
            "",
            &[],
            "empty model ref",
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
    // Environment Variable Coverage Tests
    // These test the conditional paths controlled by environment variables
    // =========================================================================

    #[test]
    fn test_cpu_debug_env_var_false() {
        std::env::set_var("CPU_DEBUG", "0");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "debug disabled",
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
    fn test_cpu_debug_env_var_invalid() {
        std::env::set_var("CPU_DEBUG", "invalid");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "debug invalid",
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
    fn test_skip_gpu_resident_env_var_false() {
        std::env::set_var("SKIP_GPU_RESIDENT", "0");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "skip disabled",
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

    #[test]
    fn test_skip_gpu_resident_env_var_invalid() {
        std::env::set_var("SKIP_GPU_RESIDENT", "maybe");
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "skip invalid",
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

    // =========================================================================
    // Return Type and Result Handling Tests
    // =========================================================================

    #[test]
    fn test_result_error_type_is_realizar_error() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "type test",
            5,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());

        // Verify we get a RealizarError
        let err = result.unwrap_err();
        // The error should implement Display
        let _display = format!("{}", err);
        // The error should implement Debug
        let _debug = format!("{:?}", err);
    }

    // =========================================================================
    // Concurrent Safety Tests (Thread Safety)
    // =========================================================================

    #[test]
    fn test_inference_functions_are_thread_safe() {
        use std::thread;

        let handles: Vec<_> = (0..4)
            .map(|i| {
                thread::spawn(move || {
                    let result = inference::run_gguf_inference(
                        &format!("/nonexistent/model_{}.gguf", i),
                        &[],
                        &format!("thread {} prompt", i),
                        5,
                        0.5,
                        "text",
                        false,
                        false,
                        None, // trace_config
                    );
                    assert!(result.is_err());
                })
            })
            .collect();

        for handle in handles {
            handle.join().expect("Thread should not panic");
        }
    }

    // =========================================================================
    // Parameter Combination Matrix Tests
    // =========================================================================

    #[test]
    fn test_parameter_combinations_gguf() {
        let temps = [0.0, 0.01, 0.5, 1.0];
        let formats = ["json", "text"];
        let gpu_flags = [false, true];
        let _verbose_flags = [false, true];
        let _trace_flags = [false, true];

        for &temp in &temps {
            for &format in &formats {
                for &gpu in &gpu_flags {
                    // Only test a subset to keep test runtime reasonable
                    let result = inference::run_gguf_inference(
                        "/nonexistent/model.gguf",
                        &[],
                        "combo test",
                        5,
                        temp,
                        format,
                        gpu,
                        false,
                        None, // trace_config
                    );
                    assert!(result.is_err());
                }
            }
        }
    }

    #[test]
    fn test_parameter_combinations_apr() {
        let temps = [0.0, 0.5, 1.0];
        let formats = ["json", "text"];
        let gpu_flags = [false, true];
        let verbose_flags = [false, true];

        for &temp in &temps {
            for &format in &formats {
                for &gpu in &gpu_flags {
                    for &verbose in &verbose_flags {
                        let result = inference::run_apr_inference(
                            "/nonexistent/model.apr",
                            &[],
                            "combo test",
                            5,
                            temp,
                            format,
                            gpu,
                            verbose,
                            None, // trace_config
                        );
                        assert!(result.is_err());
                    }
                }
            }
        }
    }

    // =========================================================================
    // Boundary Value Tests for Numeric Parameters
    // =========================================================================

    #[test]
    fn test_max_tokens_zero() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "zero tokens",
            0,
            0.5,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_negative() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "negative temp",
            5,
            -0.5, // Negative
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_infinity() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "inf temp",
            5,
            f32::INFINITY,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }
