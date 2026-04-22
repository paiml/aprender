    // ── dispatch_profiling_commands ─────────────────────────────────────

    #[test]
    fn test_dispatch_profiling_returns_none_for_non_extended() {
        let cli = make_cli(Commands::List);
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_none(), "Non-extended command should not match profiling dispatcher");
    }

    #[test]
    fn test_dispatch_profiling_returns_none_for_chat() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Chat {
            file: PathBuf::from("model.apr"),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            no_gpu: true,
            gpu: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
            backend: None,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_none(), "Chat should not be handled by profiling dispatcher");
    }

    #[test]
    fn test_dispatch_profiling_bench_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Bench {
            file: PathBuf::from("/tmp/nonexistent_bench_model.apr"),
            warmup: 1,
            iterations: 1,
            max_tokens: 10,
            prompt: None,
            fast: true,
            brick: None,
            percentiles: vec![50.0, 95.0, 99.0],
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "Bench should be handled by profiling dispatcher");
        assert!(result.expect("should be some").is_err(), "Bench with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_profiling_profile_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Profile {
            file: PathBuf::from("/tmp/nonexistent_profile_model.apr"),
            granular: false,
            format: "text".to_string(),
            focus: None,
            detect_naive: false,
            threshold: 0.01,
            compare_hf: None,
            energy: false,
            perf_grade: false,
            callgraph: false,
            fail_on_naive: false,
            output: None,
            ci: false,
            assert_throughput: None,
            assert_p99: None,
            assert_p50: None,
            warmup: 3,
            measure: 10,
            tokens: 32,
            ollama: false,
            no_gpu: true,
            compare: None,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "Profile should be handled by profiling dispatcher");
        assert!(result.expect("should be some").is_err(), "Profile with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_profiling_eval_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Eval {
            file: PathBuf::from("/tmp/nonexistent_eval_model.apr"),
            dataset: "test".to_string(),
            text: None,
            max_tokens: 32,
            threshold: 90.0,
            task: None,
            data: None,
            model_size: None,
            num_classes: 5,
            generate_card: false,
            device: "cpu".to_string(),
            samples: 1,
            temperature: 0.0,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "Eval should be handled by profiling dispatcher");
        assert!(result.expect("should be some").is_err(), "Eval with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_profiling_qa_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Qa {
            file: PathBuf::from("/tmp/nonexistent_qa_model.apr"),
            assert_tps: None,
            assert_speedup: None,
            assert_gpu_speedup: None,
            skip_golden: true,
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            skip_ptx_parity: true,
            safetensors_path: None,
            iterations: 1,
            warmup: 0,
            max_tokens: 10,
            json: false,
            verbose: false,
            min_executed: Some(0),
            previous_report: None,
            regression_threshold: Some(0.0),
            skip_gpu_state: true,
            skip_metadata: true,
            skip_capability: true,
            assert_classifier_head: false,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "QA should be handled by profiling dispatcher");
    }

    // ── dispatch_train_command ──────────────────────────────────────────
