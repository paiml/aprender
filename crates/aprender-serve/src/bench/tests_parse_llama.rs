
    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_full_output() {
        let output = r#"Hello world",
```

llama_perf_sampler_print:    sampling time =       0.14 ms /     6 runs   (    0.02 ms per token, 42553.19 tokens per second)
llama_perf_context_print:        load time =    1349.68 ms
llama_perf_context_print: prompt eval time =       5.00 ms /     1 tokens (    5.00 ms per token,   200.00 tokens per second)
llama_perf_context_print:        eval time =      22.60 ms /     5 runs   (    4.52 ms per token,   221.28 tokens per second)
llama_perf_context_print:       total time =      27.60 ms /     6 tokens"#;

        let result = LlamaCppBackend::parse_cli_output(output);
        assert!(result.is_ok());
        let response = result.expect("test");
        // TTFT = prompt eval time (time to first token)
        assert!((response.ttft_ms - 5.0).abs() < 0.1);
        // Total time from parse
        assert!((response.total_time_ms - 27.60).abs() < 0.1);
        // Tokens generated = eval runs (5 tokens generated after prompt)
        assert_eq!(response.tokens_generated, 5);
        // Text should contain the generated content
        assert!(response.text.contains("Hello world"));
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_parse_llama_cli_extract_generated_text() {
        let output =
            "The answer is 42.\n\nllama_perf_context_print: total time = 100.0 ms / 10 tokens";
        let text = LlamaCppBackend::extract_generated_text(output);
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    #[cfg(feature = "bench-http")]
    fn test_llama_cpp_backend_build_command() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: Some("/path/to/model.gguf".to_string()),
            n_gpu_layers: 99,
            ctx_size: 4096,
            threads: 8,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let args = backend.build_cli_args(&request);

        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"/path/to/model.gguf".to_string()));
        assert!(args.contains(&"-p".to_string()));
        assert!(args.contains(&"Hello".to_string()));
        assert!(args.contains(&"-n".to_string()));
        assert!(args.contains(&"10".to_string()));
        assert!(args.contains(&"-ngl".to_string()));
        assert!(args.contains(&"99".to_string()));
        assert!(args.contains(&"-c".to_string()));
        assert!(args.contains(&"4096".to_string()));
        assert!(args.contains(&"-t".to_string()));
        assert!(args.contains(&"8".to_string()));
    }

    #[cfg(feature = "bench-http")]
    #[test]
    fn test_llama_cpp_backend_no_model_path_error() {
        let config = LlamaCppConfig {
            binary_path: "/path/to/llama-cli".to_string(),
            model_path: None, // No model
            n_gpu_layers: 0,
            ctx_size: 2048,
            threads: 4,
        };
        let backend = LlamaCppBackend::new(config);
        let request = InferenceRequest {
            prompt: "Hello".to_string(),
            max_tokens: 10,
            temperature: 0.7,
            stop: vec![],
        };
        let result = backend.inference(&request);
        assert!(result.is_err());
    }

    // ========================================================================
    // Benchmark Matrix Tests (EXTREME TDD)
    // ========================================================================

    #[test]
    fn test_compute_backend_type_display() {
        assert_eq!(format!("{}", ComputeBackendType::Cpu), "cpu");
        assert_eq!(format!("{}", ComputeBackendType::Wgpu), "wgpu");
        assert_eq!(format!("{}", ComputeBackendType::Cuda), "cuda");
    }

    #[test]
    fn test_compute_backend_type_from_str() {
        assert_eq!(
            ComputeBackendType::parse("cpu"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("CPU"),
            Some(ComputeBackendType::Cpu)
        );
        assert_eq!(
            ComputeBackendType::parse("wgpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("gpu"),
            Some(ComputeBackendType::Wgpu)
        );
        assert_eq!(
            ComputeBackendType::parse("cuda"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(
            ComputeBackendType::parse("nvidia"),
            Some(ComputeBackendType::Cuda)
        );
        assert_eq!(ComputeBackendType::parse("unknown"), None);
    }

    #[test]
    fn test_compute_backend_type_all() {
        let all = ComputeBackendType::all();
        assert_eq!(all.len(), 3);
        assert!(all.contains(&ComputeBackendType::Cpu));
        assert!(all.contains(&ComputeBackendType::Wgpu));
        assert!(all.contains(&ComputeBackendType::Cuda));
    }

    #[test]
    fn test_matrix_benchmark_entry_unavailable() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda);
        assert!(!entry.available);
        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cuda);
        assert!(entry.notes.contains("not available"));
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples() {
        let latencies = vec![100.0, 105.0, 110.0, 95.0, 102.0];
        let throughputs = vec![50.0, 48.0, 52.0, 49.0, 51.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &latencies,
            &throughputs,
            150.0,
        );

        assert!(entry.available);
        assert_eq!(entry.runtime, RuntimeType::LlamaCpp);
        assert_eq!(entry.backend, ComputeBackendType::Wgpu);
        assert_eq!(entry.model, "phi-2");
        assert_eq!(entry.samples, 5);
        assert!((entry.p50_latency_ms - 102.0).abs() < 1.0); // Median
        assert!((entry.cold_start_ms - 150.0).abs() < 0.1);
        assert!(entry.throughput_tps > 0.0);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_empty_samples() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model",
            &[],
            &[],
            0.0,
        );
        assert!(!entry.available);
    }

    #[test]
    fn test_matrix_benchmark_entry_with_notes() {
        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cuda)
                .with_notes("GPU layers: 99");
        assert_eq!(entry.notes, "GPU layers: 99");
    }

    #[test]
    fn test_benchmark_matrix_creation() {
        let hardware = HardwareSpec::default();
        let matrix = BenchmarkMatrix::new("phi-2", hardware);

        assert_eq!(matrix.model, "phi-2");
        assert_eq!(matrix.version, "1.1");
        assert!(matrix.methodology.contains("Hoefler"));
        assert!(matrix.entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_add_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry1 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        );
        matrix.add_entry(entry1);

        assert_eq!(matrix.entries.len(), 1);

        // Add another entry
        let entry2 = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0, 82.0, 78.0],
            &[60.0, 61.0, 59.0],
            120.0,
        );
        matrix.add_entry(entry2);

        assert_eq!(matrix.entries.len(), 2);

        // Replace existing entry
        let entry1_updated = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0, 92.0, 88.0],
            &[55.0, 56.0, 54.0],
            95.0,
        );
        matrix.add_entry(entry1_updated);

        assert_eq!(matrix.entries.len(), 2); // Still 2, replaced
    }

    #[test]
    fn test_benchmark_matrix_get_entry() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        );
        matrix.add_entry(entry);

        let found = matrix.get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu);
        assert!(found.is_some());
        assert_eq!(found.expect("test").runtime, RuntimeType::Realizar);

        let not_found = matrix.get_entry(RuntimeType::LlamaCpp, ComputeBackendType::Cuda);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));

        let realizar_entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert_eq!(realizar_entries.len(), 2);

        let llama_entries = matrix.entries_for_runtime(RuntimeType::LlamaCpp);
        assert_eq!(llama_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[55.0],
            95.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Wgpu,
            "phi-2",
            &[80.0],
            &[60.0],
            90.0,
        ));

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 2);

        let wgpu_entries = matrix.entries_for_backend(ComputeBackendType::Wgpu);
        assert_eq!(wgpu_entries.len(), 1);

        let cuda_entries = matrix.entries_for_backend(ComputeBackendType::Cuda);
        assert!(cuda_entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[80.0, 82.0, 78.0], // Faster
            &[55.0],
            95.0,
        ));

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.expect("test").runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
            "phi-2",
            &[90.0],
            &[70.0, 71.0, 69.0], // Higher throughput
            95.0,
        ));

        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Cpu);
        assert!(highest.is_some());
        assert_eq!(highest.expect("test").runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 110.0, 105.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let table = matrix.to_markdown_table();
        assert!(table.contains("| Runtime | Backend |"));
        assert!(table.contains("| **realizar** |"));
        assert!(table.contains("| - | - |")); // Unavailable entry
    }

    #[test]
    fn test_benchmark_matrix_json_roundtrip() {
        let hardware = HardwareSpec::default();
        let mut matrix = BenchmarkMatrix::new("phi-2", hardware);

        matrix.add_entry(MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "phi-2",
            &[100.0, 102.0, 98.0],
            &[50.0, 51.0, 49.0],
            100.0,
        ));

        let json = matrix.to_json().expect("serialization should succeed");
        assert!(json.contains("\"model\": \"phi-2\""));

        let parsed = BenchmarkMatrix::from_json(&json).expect("deserialization should succeed");
        assert_eq!(parsed.model, "phi-2");
        assert_eq!(parsed.entries.len(), 1);
    }
