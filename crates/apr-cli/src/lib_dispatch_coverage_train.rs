    // ── dispatch_train_command ──────────────────────────────────────────

    #[cfg(feature = "training")]
    #[test]
    fn test_dispatch_analysis_train_routing() {
        // Train Plan with nonexistent data — tests train routing path
        let cli = make_cli(Commands::Extended(ExtendedCommands::Training(
            TrainingCommands::Train {
                command: TrainCommands::Plan {
                    data: Some(PathBuf::from("/tmp/nonexistent_train_data.jsonl")),
                    model_size: "tiny".to_string(),
                    model_path: None,
                    num_classes: 5,
                    task: "classify".to_string(),
                    config: None,
                    output: PathBuf::from("/tmp/train_plan_out"),
                    strategy: "auto".to_string(),
                    budget: 100,
                    scout: false,
                    max_epochs: 3,
                    learning_rate: Some(1e-4),
                    lora_rank: Some(8),
                    batch_size: Some(4),
                    val_data: None,
                    test_data: None,
                    format: "text".to_string(),
                },
            },
        )));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Train should be handled by analysis dispatcher (delegates to dispatch_train_command)");
    }

    #[cfg(feature = "training")]
    #[test]
    fn test_dispatch_train_sweep() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Training(
            TrainingCommands::Train {
                command: TrainCommands::Sweep {
                    config: PathBuf::from("/tmp/nonexistent_sweep_config.toml"),
                    strategy: "grid".to_string(),
                    num_configs: 5,
                    output_dir: PathBuf::from("/tmp/sweep_out"),
                    seed: 42,
                },
            },
        )));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Train Sweep should be handled by analysis dispatcher");
    }

    #[cfg(feature = "training")]
    #[test]
    fn test_dispatch_train_cluster_status() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Training(
            TrainingCommands::Train {
                command: TrainCommands::ClusterStatus {
                    cluster: PathBuf::from("/tmp/nonexistent_cluster_config.yaml"),
                },
            },
        )));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Train ClusterStatus should be handled by analysis dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_inspection_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_inspection_routes_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Inspect should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_routes_debug() {
        let cli = make_cli(Commands::Debug {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            drama: false,
            hex: false,
            strings: false,
            limit: 256,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Debug should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_routes_validate() {
        let cli = make_cli(Commands::Validate {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            strict: false,
            quality: false,
            min_score: None,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Validate should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_routes_lint() {
        let cli = make_cli(Commands::Lint {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Lint should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_returns_none_for_export() {
        let cli = make_cli(Commands::Export {
            file: Some(PathBuf::from("/tmp/nonexistent_pmat540.apr")),
            format: "gguf".to_string(),
            output: None,
            quantize: None,
            list_formats: false,
            batch: None,
            json: false,
            plan: false,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_none(), "Export should NOT be handled by inspection dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_diagnostic_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_diagnostic_routes_trace() {
        let cli = make_cli(Commands::Trace {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            layer: None,
            reference: None,
            json: false,
            verbose: false,
            payload: false,
            diff: false,
            interactive: false,
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_some(), "Trace should be handled by diagnostic dispatcher");
    }

    #[test]
    fn test_dispatch_diagnostic_routes_tensors() {
        let cli = make_cli(Commands::Tensors {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            stats: false,
            filter: None,
            limit: 0,
            json: false,
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_some(), "Tensors should be handled by diagnostic dispatcher");
    }

    #[test]
    fn test_dispatch_diagnostic_routes_diff() {
        let cli = make_cli(Commands::Diff {
            file1: PathBuf::from("/tmp/nonexistent_a.apr"),
            file2: PathBuf::from("/tmp/nonexistent_b.apr"),
            weights: false,
            values: false,
            filter: None,
            limit: 10,
            transpose_aware: false,
            json: false,
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_some(), "Diff should be handled by diagnostic dispatcher");
    }

    #[test]
    fn test_dispatch_diagnostic_returns_none_for_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_none(), "Inspect should NOT be handled by diagnostic dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_format_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_format_routes_import() {
        let cli = make_cli(Commands::Import {
            source: "/tmp/nonexistent_pmat540.safetensors".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
            allow_no_config: true,
        });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_some(), "Import should be handled by format dispatcher");
    }

    #[test]
    fn test_dispatch_format_routes_convert() {
        let cli = make_cli(Commands::Convert {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            quantize: None,
            compress: None,
            output: PathBuf::from("/tmp/nonexistent_pmat540_out.apr"),
            force: false,
        });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_some(), "Convert should be handled by format dispatcher");
    }

    #[test]
    fn test_dispatch_format_routes_export() {
        let cli = make_cli(Commands::Export {
            file: Some(PathBuf::from("/tmp/nonexistent_pmat540.apr")),
            format: "gguf".to_string(),
            output: None,
            quantize: None,
            list_formats: false,
            batch: None,
            json: false,
            plan: false,
        });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_some(), "Export should be handled by format dispatcher");
    }

    #[test]
    fn test_dispatch_format_returns_none_for_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_none(), "Inspect should NOT be handled by format dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_runtime_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_runtime_routes_check() {
        let cli = make_cli(Commands::Check {
            file: PathBuf::from("/tmp/nonexistent_pmat540.gguf"),
            no_gpu: true,
            json: false,
        });
        let result = dispatch_runtime_commands(&cli);
        assert!(result.is_some(), "Check should be handled by runtime dispatcher");
    }

    #[test]
    fn test_dispatch_runtime_returns_none_for_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        let result = dispatch_runtime_commands(&cli);
        assert!(result.is_none(), "Inspect should NOT be handled by runtime dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_core_command routing coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_core_routes_inspect_via_inspection() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        // dispatch_core_command delegates to dispatch_inspection_commands
        let result = dispatch_core_command(&cli);
        assert!(result.is_some(), "Inspect should be routed through core → inspection");
    }

    #[test]
    fn test_dispatch_core_routes_tensors_via_diagnostic() {
        let cli = make_cli(Commands::Tensors {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            stats: false,
            filter: None,
            limit: 0,
            json: false,
        });
        let result = dispatch_core_command(&cli);
        assert!(result.is_some(), "Tensors should be routed through core → diagnostic");
    }

    #[test]
    fn test_dispatch_core_routes_import_via_format() {
        let cli = make_cli(Commands::Import {
            source: "/tmp/nonexistent_pmat540.safetensors".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
            allow_no_config: true,
        });
        let result = dispatch_core_command(&cli);
        assert!(result.is_some(), "Import should be routed through core → format");
    }

    #[test]
    fn test_dispatch_core_returns_none_for_extended() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Chat {
            file: PathBuf::from("/tmp/nonexistent_pmat540.gguf"),
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
        let result = dispatch_core_command(&cli);
        assert!(result.is_none(), "Chat (extended) should NOT be handled by core dispatcher");
    }
