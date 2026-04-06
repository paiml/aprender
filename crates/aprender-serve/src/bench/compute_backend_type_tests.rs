
    // =========================================================================
    // ComputeBackendType Tests
    // =========================================================================

    #[test]
    fn test_compute_backend_type_display() {
        assert_eq!(format!("{}", ComputeBackendType::Cpu), "cpu");
        assert_eq!(format!("{}", ComputeBackendType::Wgpu), "wgpu");
        assert_eq!(format!("{}", ComputeBackendType::Cuda), "cuda");
    }

    #[test]
    fn test_compute_backend_type_parse() {
        assert_eq!(
            ComputeBackendType::parse("cpu"),
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
        assert_eq!(
            ComputeBackendType::parse("CPU"),
            Some(ComputeBackendType::Cpu)
        ); // case-insensitive
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
    fn test_compute_backend_type_clone_eq() {
        let backend = ComputeBackendType::Cuda;
        assert_eq!(backend, backend.clone());
    }

    #[test]
    fn test_compute_backend_type_serialize() {
        let json = serde_json::to_string(&ComputeBackendType::Wgpu).unwrap();
        assert!(json.contains("Wgpu"));
    }

    // =========================================================================
    // MatrixBenchmarkEntry Tests
    // =========================================================================

    #[test]
    fn test_matrix_benchmark_entry_default() {
        let entry = MatrixBenchmarkEntry::default();
        assert_eq!(entry.runtime, RuntimeType::Realizar);
        assert_eq!(entry.backend, ComputeBackendType::Cpu);
        assert!(!entry.available);
        assert_eq!(entry.samples, 0);
    }

    #[test]
    fn test_matrix_benchmark_entry_unavailable() {
        let entry = MatrixBenchmarkEntry::unavailable(RuntimeType::Vllm, ComputeBackendType::Cuda);
        assert_eq!(entry.runtime, RuntimeType::Vllm);
        assert_eq!(entry.backend, ComputeBackendType::Cuda);
        assert!(!entry.available);
        assert!(entry.notes.contains("not available"));
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples() {
        let latencies = vec![10.0, 12.0, 11.0, 13.0, 9.0];
        let throughputs = vec![100.0, 90.0, 95.0, 85.0, 105.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Wgpu,
            "llama-7b",
            &latencies,
            &throughputs,
            50.0,
        );

        assert_eq!(entry.runtime, RuntimeType::LlamaCpp);
        assert_eq!(entry.backend, ComputeBackendType::Wgpu);
        assert_eq!(entry.model, "llama-7b");
        assert!(entry.available);
        assert_eq!(entry.samples, 5);
        assert!((entry.cold_start_ms - 50.0).abs() < 0.01);
        // p50 of [9, 10, 11, 12, 13] should be around 11
        assert!(entry.p50_latency_ms > 0.0);
        // Average throughput should be 95
        assert!((entry.throughput_tps - 95.0).abs() < 0.01);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_empty_samples() {
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Ollama,
            ComputeBackendType::Cpu,
            "model",
            &[],
            &[],
            0.0,
        );
        assert!(!entry.available);
    }

    #[test]
    fn test_matrix_benchmark_entry_from_samples_empty_throughput() {
        let latencies = vec![10.0, 12.0];
        let entry = MatrixBenchmarkEntry::from_samples(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
            "model",
            &latencies,
            &[],
            0.0,
        );
        assert!(entry.available);
        assert_eq!(entry.throughput_tps, 0.0);
    }

    #[test]
    fn test_matrix_benchmark_entry_with_notes() {
        let entry = MatrixBenchmarkEntry::default().with_notes("GPU layers: 32");
        assert_eq!(entry.notes, "GPU layers: 32");
    }

    #[test]
    fn test_matrix_benchmark_entry_serialize() {
        let entry = MatrixBenchmarkEntry {
            runtime: RuntimeType::LlamaCpp,
            backend: ComputeBackendType::Cuda,
            model: "phi-2".to_string(),
            available: true,
            p50_latency_ms: 25.5,
            p99_latency_ms: 45.0,
            throughput_tps: 200.0,
            cold_start_ms: 100.0,
            samples: 30,
            cv_at_stop: 0.03,
            notes: "test".to_string(),
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("phi-2"));
        assert!(json.contains("200"));
    }

    // =========================================================================
    // BenchmarkMatrix Tests
    // =========================================================================

    fn make_hardware_spec() -> HardwareSpec {
        HardwareSpec {
            cpu: "Intel i7".to_string(),
            gpu: Some("RTX 4090".to_string()),
            memory_gb: 32,
            storage: "NVMe SSD".to_string(),
        }
    }

    #[test]
    fn test_benchmark_matrix_new() {
        let hw = make_hardware_spec();
        let matrix = BenchmarkMatrix::new("llama-7b", hw);

        assert_eq!(matrix.model, "llama-7b");
        assert_eq!(matrix.version, "1.1");
        assert!((matrix.cv_threshold - 0.05).abs() < 0.001);
        assert!(matrix.entries.is_empty());
    }

    #[test]
    fn test_benchmark_matrix_add_entry() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let entry =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cpu);
        matrix.add_entry(entry);

        assert_eq!(matrix.entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_add_entry_replaces_existing() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let entry1 =
            MatrixBenchmarkEntry::unavailable(RuntimeType::Realizar, ComputeBackendType::Cpu);
        matrix.add_entry(entry1);

        let entry2 = MatrixBenchmarkEntry {
            runtime: RuntimeType::Realizar,
            backend: ComputeBackendType::Cpu,
            available: true,
            p50_latency_ms: 10.0,
            ..Default::default()
        };
        matrix.add_entry(entry2);

        assert_eq!(matrix.entries.len(), 1);
        assert!(matrix.entries[0].available);
    }

    #[test]
    fn test_benchmark_matrix_get_entry() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let entry = MatrixBenchmarkEntry::unavailable(RuntimeType::Vllm, ComputeBackendType::Cuda);
        matrix.add_entry(entry);

        let found = matrix.get_entry(RuntimeType::Vllm, ComputeBackendType::Cuda);
        assert!(found.is_some());

        let not_found = matrix.get_entry(RuntimeType::Realizar, ComputeBackendType::Cpu);
        assert!(not_found.is_none());
    }

    #[test]
    fn test_benchmark_matrix_entries_for_runtime() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cpu,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Vllm,
            ComputeBackendType::Cuda,
        ));

        let realizar_entries = matrix.entries_for_runtime(RuntimeType::Realizar);
        assert_eq!(realizar_entries.len(), 2);

        let vllm_entries = matrix.entries_for_runtime(RuntimeType::Vllm);
        assert_eq!(vllm_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_entries_for_backend() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cuda,
        ));
        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::LlamaCpp,
            ComputeBackendType::Cpu,
        ));

        let cuda_entries = matrix.entries_for_backend(ComputeBackendType::Cuda);
        assert_eq!(cuda_entries.len(), 2);

        let cpu_entries = matrix.entries_for_backend(ComputeBackendType::Cpu);
        assert_eq!(cpu_entries.len(), 1);
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry1 = MatrixBenchmarkEntry::default();
        entry1.runtime = RuntimeType::Realizar;
        entry1.backend = ComputeBackendType::Cpu;
        entry1.available = true;
        entry1.p50_latency_ms = 20.0;
        matrix.add_entry(entry1);

        let mut entry2 = MatrixBenchmarkEntry::default();
        entry2.runtime = RuntimeType::LlamaCpp;
        entry2.backend = ComputeBackendType::Cpu;
        entry2.available = true;
        entry2.p50_latency_ms = 15.0; // faster
        matrix.add_entry(entry2);

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cpu);
        assert!(fastest.is_some());
        assert_eq!(fastest.unwrap().runtime, RuntimeType::LlamaCpp);
    }

    #[test]
    fn test_benchmark_matrix_fastest_for_backend_none_available() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Realizar,
            ComputeBackendType::Cuda,
        ));

        let fastest = matrix.fastest_for_backend(ComputeBackendType::Cuda);
        assert!(fastest.is_none());
    }

    #[test]
    fn test_benchmark_matrix_highest_throughput_for_backend() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry1 = MatrixBenchmarkEntry::default();
        entry1.runtime = RuntimeType::Realizar;
        entry1.backend = ComputeBackendType::Wgpu;
        entry1.available = true;
        entry1.throughput_tps = 100.0;
        matrix.add_entry(entry1);

        let mut entry2 = MatrixBenchmarkEntry::default();
        entry2.runtime = RuntimeType::Ollama;
        entry2.backend = ComputeBackendType::Wgpu;
        entry2.available = true;
        entry2.throughput_tps = 150.0; // higher
        matrix.add_entry(entry2);

        let highest = matrix.highest_throughput_for_backend(ComputeBackendType::Wgpu);
        assert!(highest.is_some());
        assert_eq!(highest.unwrap().runtime, RuntimeType::Ollama);
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        let mut entry = MatrixBenchmarkEntry::default();
        entry.runtime = RuntimeType::Realizar;
        entry.backend = ComputeBackendType::Cpu;
        entry.available = true;
        entry.p50_latency_ms = 10.5;
        entry.p99_latency_ms = 25.0;
        entry.throughput_tps = 95.5;
        entry.cold_start_ms = 50.0;
        entry.samples = 30;
        entry.cv_at_stop = 0.045;
        matrix.add_entry(entry);

        let md = matrix.to_markdown_table();
        assert!(md.contains("| Runtime |"));
        assert!(md.contains("realizar"));
        assert!(md.contains("cpu"));
        assert!(md.contains("10.5ms"));
        assert!(md.contains("95.5 tok/s"));
    }

    #[test]
    fn test_benchmark_matrix_to_markdown_table_unavailable() {
        let hw = make_hardware_spec();
        let mut matrix = BenchmarkMatrix::new("model", hw);

        matrix.add_entry(MatrixBenchmarkEntry::unavailable(
            RuntimeType::Vllm,
            ComputeBackendType::Cuda,
        ));

        let md = matrix.to_markdown_table();
        assert!(md.contains("vllm"));
        assert!(md.contains("| - | - | - |"));
    }

    #[test]
    fn test_benchmark_matrix_to_json() {
        let hw = make_hardware_spec();
        let matrix = BenchmarkMatrix::new("llama-7b", hw);

        let json = matrix.to_json().unwrap();
        assert!(json.contains("llama-7b"));
        assert!(json.contains("version"));
    }

    #[test]
    fn test_benchmark_matrix_from_json() {
        let hw = make_hardware_spec();
        let matrix = BenchmarkMatrix::new("phi-2", hw);
        let json = matrix.to_json().unwrap();

        let parsed = BenchmarkMatrix::from_json(&json).unwrap();
        assert_eq!(parsed.model, "phi-2");
    }

    // =========================================================================
    // MatrixBenchmarkConfig Tests
    // =========================================================================

    #[test]
    fn test_matrix_benchmark_config_default() {
        let config = MatrixBenchmarkConfig::default();
        assert_eq!(config.runtimes.len(), 3);
        assert!(config.runtimes.contains(&RuntimeType::Realizar));
        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.max_tokens, 50);
        assert_eq!(config.min_samples, 30);
        assert_eq!(config.max_samples, 200);
    }

    #[test]
    fn test_matrix_benchmark_config_debug() {
        let config = MatrixBenchmarkConfig::default();
        let debug = format!("{:?}", config);
        assert!(debug.contains("MatrixBenchmarkConfig"));
    }

    // =========================================================================
    // BackendSummary Tests
    // =========================================================================

    #[test]
    fn test_backend_summary_serialize() {
        let summary = BackendSummary {
            backend: ComputeBackendType::Cuda,
            available_runtimes: 2,
            fastest_runtime: Some("realizar".to_string()),
            fastest_p50_ms: 15.0,
            highest_throughput_runtime: Some("llama-cpp".to_string()),
            highest_throughput_tps: 200.0,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(json.contains("Cuda")); // Serializes as enum variant name
        assert!(json.contains("realizar"));
    }
