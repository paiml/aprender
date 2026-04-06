
    // =========================================================================
    // display_model_info: GGUF format detection via magic bytes
    // =========================================================================

    #[test]
    fn test_display_model_info_gguf_valid_minimal() {
        // Build a valid GGUF file using GGUFBuilder
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        let result = display_model_info("test_model.gguf", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_safetensors_valid_minimal() {
        // SafeTensors has a JSON header with tensor metadata
        // Minimal valid safetensors: 8-byte LE header_size + JSON header + data
        let header = r#"{"test":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut data = Vec::new();
        data.extend_from_slice(&header_len.to_le_bytes());
        data.extend_from_slice(header_bytes);
        data.extend_from_slice(&[0u8; 16]); // tensor data
        let result = display_model_info("model.safetensors", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_with_known_type_codes() {
        // Test various known APR type codes
        let type_codes: &[(u16, &str)] = &[
            (0x0001, "LinearRegression"),
            (0x0003, "DecisionTree"),
            (0x0009, "KNN"),
            (0x0020, "NeuralSequential"),
        ];
        for &(type_code, _name) in type_codes {
            let mut data = vec![0u8; 16];
            data[0..4].copy_from_slice(b"APR\0");
            data[4..6].copy_from_slice(&type_code.to_le_bytes());
            data[6..8].copy_from_slice(&1u16.to_le_bytes());
            let result = display_model_info("model.apr", &data);
            assert!(result.is_ok(), "Failed for type_code 0x{:04X}", type_code);
        }
    }

    #[test]
    fn test_display_model_info_unknown_extension_unknown_magic() {
        let data = vec![0xFF, 0xFE, 0xFD, 0xFC, 0xFB, 0xFA, 0xF9, 0xF8];
        let result = display_model_info("model.unknown", &data);
        assert!(result.is_ok()); // Should show "Unknown (8 bytes)"
    }

    #[test]
    fn test_display_model_info_gguf_magic_but_wrong_extension() {
        // GGUF magic bytes with .bin extension - should detect via magic
        use crate::gguf::test_factory::GGUFBuilder;
        let data = GGUFBuilder::new()
            .architecture("llama")
            .hidden_dim("llama", 32)
            .num_layers("llama", 1)
            .num_heads("llama", 4)
            .build();
        let result = display_model_info("model.bin", &data);
        // The GGUF magic check is: file_data.starts_with(GGUF_MAGIC)
        // OR model_ref.ends_with(".gguf")
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_apr_magic_but_bin_extension() {
        let mut data = vec![0u8; 16];
        data[0..4].copy_from_slice(b"APR\0");
        data[4..6].copy_from_slice(&0x0005u16.to_le_bytes()); // GradientBoosting
        data[6..8].copy_from_slice(&1u16.to_le_bytes());
        let result = display_model_info("test.bin", &data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_display_model_info_single_byte() {
        let result = display_model_info("tiny.bin", &[0x42]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // parse_cargo_bench_output: additional edge cases
    // =========================================================================

    #[test]
    fn test_parse_cargo_bench_output_no_test_keyword() {
        // Line with bench: and ns/iter but no "test" keyword
        let output = "benchmark_foo ... bench: 500 ns/iter (+/- 25)";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_incomplete_bench_line() {
        // "test" and "bench:" present but no time value
        let output = "test benchmark_foo ... bench:";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_parse_cargo_bench_output_with_commas_in_number() {
        let output = "test large_bench ... bench:  1,234,567 ns/iter (+/- 1,000)";
        let results = parse_cargo_bench_output(output, Some("perf"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0]["time_ns"], 1234567);
        assert_eq!(results[0]["suite"], "perf");
    }

    #[test]
    fn test_parse_cargo_bench_output_suite_none_vs_some() {
        let output = "test bench_a ... bench:      100 ns/iter (+/- 5)";
        let with_suite = parse_cargo_bench_output(output, Some("my_suite"));
        let without_suite = parse_cargo_bench_output(output, None);
        assert_eq!(with_suite[0]["suite"], "my_suite");
        assert!(without_suite[0]["suite"].is_null());
    }

    #[test]
    fn test_parse_cargo_bench_output_only_whitespace() {
        let output = "   \n   \n   \n";
        let results = parse_cargo_bench_output(output, None);
        assert!(results.is_empty());
    }

    // =========================================================================
    // run_model_command: error paths (async)
    // =========================================================================

    #[tokio::test]
    async fn test_run_model_command_nonexistent_file() {
        let result = run_model_command(
            "/nonexistent/path/model.gguf",
            Some("hello"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_pacha_registry_uri() {
        let result = run_model_command(
            "pacha://model:latest",
            Some("test"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        // Should return Ok (prints message about registry support)
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_model_command_hf_registry_uri() {
        let result = run_model_command(
            "hf://meta-llama/Llama-3",
            Some("test"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_model_command_colon_registry_uri() {
        // model:tag format triggers registry path
        let result = run_model_command(
            "llama3:8b",
            Some("test"),
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_model_command_no_prompt() {
        // No prompt = interactive mode message, but file read still fails
        let result = run_model_command(
            "/nonexistent/model.gguf",
            None,
            10,
            0.0,
            "text",
            None,
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_with_verbose() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            None,
            false,
            false,
            true, // verbose
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_with_system_prompt() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            Some("You are helpful"),
            false,
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_raw_mode() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            None,
            true, // raw mode
            false,
            false,
            None,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_model_command_with_trace() {
        let result = run_model_command(
            "/nonexistent/model.gguf",
            Some("test"),
            10,
            0.5,
            "text",
            None,
            false,
            false,
            false,
            Some(None), // trace enabled
        )
        .await;
        assert!(result.is_err());
    }

    // =========================================================================
    // run_chat_command: error paths
    // =========================================================================

    #[tokio::test]
    async fn test_run_chat_command_nonexistent_model() {
        let result = run_chat_command("/nonexistent/model.gguf", None, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_run_chat_command_pacha_uri() {
        let result = run_chat_command("pacha://model:v1", None, None).await;
        // Should return Ok (prints "Registry URIs require --features registry")
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_run_chat_command_hf_uri() {
        let result = run_chat_command("hf://meta-llama/Llama-3", None, None).await;
        assert!(result.is_ok());
    }

    // =========================================================================
    // run_gguf_inference: nonexistent path errors
    // =========================================================================

    #[test]
    fn test_run_gguf_inference_nonexistent_path() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test prompt",
            10,
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_with_force_gpu_no_cuda() {
        // force_gpu=true but no CUDA feature - should warn and fail on file
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test",
            5,
            0.0,
            "text",
            true, // force_gpu
            false,
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_gguf_inference_json_format() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test",
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
    fn test_run_gguf_inference_with_trace() {
        let result = inference::run_gguf_inference(
            "/nonexistent/model.gguf",
            &[],
            "test",
            5,
            0.0,
            "text",
            false,
            true,                         // verbose
            Some(TraceConfig::enabled()), // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_safetensors_inference_nonexistent() {
        let result = inference::run_safetensors_inference(
            "/nonexistent/model.safetensors",
            "test prompt",
            10,
            0.0,
            "text",
            None, // trace_config
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_run_apr_inference_nonexistent() {
        let result = inference::run_apr_inference(
            "/nonexistent/model.apr",
            &[],
            "test prompt",
            10,
            0.0,
            "text",
            false, // force_gpu
            false, // verbose
            None,  // trace_config
        );
        assert!(result.is_err());
    }

    // =========================================================================
    // is_local_file_path: comprehensive edge cases
    // =========================================================================

    #[test]
    fn test_is_local_file_path_empty_string() {
        assert!(!is_local_file_path(""));
    }

    #[test]
    fn test_is_local_file_path_just_dot_slash() {
        assert!(is_local_file_path("./"));
    }

    #[test]
    fn test_is_local_file_path_nested_gguf() {
        assert!(is_local_file_path("path/to/deep/model.gguf"));
    }

    #[test]
    fn test_is_local_file_path_apr_without_path() {
        assert!(is_local_file_path("model.apr"));
    }
