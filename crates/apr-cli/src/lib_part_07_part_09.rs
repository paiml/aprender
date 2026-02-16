
    /// Test extract_model_paths: Tree is diagnostic (exempt)
    #[test]
    fn test_extract_paths_tree_exempt() {
        let cmd = Commands::Tree {
            file: PathBuf::from("model.apr"),
            filter: None,
            format: "ascii".to_string(),
            sizes: false,
            depth: None,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Tree is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Flow is diagnostic (exempt)
    #[test]
    fn test_extract_paths_flow_exempt() {
        let cmd = Commands::Flow {
            file: PathBuf::from("model.apr"),
            layer: None,
            component: "full".to_string(),
            verbose: false,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Flow is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Publish is diagnostic (exempt)
    #[test]
    fn test_extract_paths_publish_exempt() {
        let cmd = Commands::Publish {
            directory: PathBuf::from("/tmp/models"),
            repo_id: "org/repo".to_string(),
            model_name: None,
            license: "mit".to_string(),
            pipeline_tag: "text-generation".to_string(),
            library_name: None,
            tags: None,
            message: None,
            dry_run: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Publish is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Tune is diagnostic (exempt)
    #[test]
    fn test_extract_paths_tune_exempt() {
        let cmd = Commands::Tune {
            file: Some(PathBuf::from("model.apr")),
            method: "auto".to_string(),
            rank: None,
            vram: 16.0,
            plan: false,
            model: None,
            freeze_base: false,
            train_data: None,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Tune is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Pull is diagnostic (exempt)
    #[test]
    fn test_extract_paths_pull_exempt() {
        let cmd = Commands::Pull {
            model_ref: "hf://org/repo".to_string(),
            force: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Pull is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Rm is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rm_exempt() {
        let cmd = Commands::Rm {
            model_ref: "model-name".to_string(),
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Rm is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Canary is diagnostic (exempt)
    #[test]
    fn test_extract_paths_canary_exempt() {
        let cmd = Commands::Canary {
            command: CanaryCommands::Check {
                file: PathBuf::from("model.apr"),
                canary: PathBuf::from("canary.json"),
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Canary is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Oracle is diagnostic (exempt)
    #[test]
    fn test_extract_paths_oracle_exempt() {
        let cmd = Commands::Oracle {
            source: Some("model.gguf".to_string()),
            family: None,
            size: None,
            compliance: false,
            tensors: false,
            stats: false,
            explain: false,
            kernels: false,
            validate: false,
            full: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Oracle is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Showcase is diagnostic (exempt)
    #[test]
    fn test_extract_paths_showcase_exempt() {
        let cmd = Commands::Showcase {
            auto_verify: false,
            step: None,
            tier: "small".to_string(),
            model_dir: PathBuf::from("./models"),
            baseline: "llama-cpp,ollama".to_string(),
            zram: false,
            runs: 30,
            gpu: false,
            json: false,
            verbose: false,
            quiet: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Showcase is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta Convert returns source path
    #[test]
    fn test_extract_paths_rosetta_convert() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Convert {
                source: PathBuf::from("model.gguf"),
                target: PathBuf::from("out.safetensors"),
                quantize: None,
                verify: false,
                json: false,
                tokenizer: None,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Rosetta Chain returns source path
    #[test]
    fn test_extract_paths_rosetta_chain() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Chain {
                source: PathBuf::from("model.gguf"),
                formats: vec!["safetensors".to_string(), "apr".to_string()],
                work_dir: PathBuf::from("/tmp"),
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Rosetta Verify returns source path
    #[test]
    fn test_extract_paths_rosetta_verify() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Verify {
                source: PathBuf::from("model.apr"),
                intermediate: "safetensors".to_string(),
                tolerance: 1e-5,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Rosetta CompareInference returns both paths
    #[test]
    fn test_extract_paths_rosetta_compare_inference() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::CompareInference {
                model_a: PathBuf::from("model_a.gguf"),
                model_b: PathBuf::from("model_b.apr"),
                prompt: "test".to_string(),
                max_tokens: 5,
                temperature: 0.0,
                tolerance: 0.1,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(
            paths,
            vec![PathBuf::from("model_a.gguf"), PathBuf::from("model_b.apr")]
        );
    }

    /// Test extract_model_paths: Rosetta Inspect is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_inspect_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Inspect {
                file: PathBuf::from("model.gguf"),
                hexdump: false,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta Inspect is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta DiffTensors is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_diff_tensors_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::DiffTensors {
                model_a: PathBuf::from("a.gguf"),
                model_b: PathBuf::from("b.apr"),
                mismatches_only: false,
                show_values: 0,
                filter: None,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta DiffTensors is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta Fingerprint is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_fingerprint_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::Fingerprint {
                model: PathBuf::from("model.gguf"),
                model_b: None,
                output: None,
                filter: None,
                verbose: false,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta Fingerprint is a diagnostic command (exempt)"
        );
    }

    /// Test extract_model_paths: Rosetta ValidateStats is diagnostic (exempt)
    #[test]
    fn test_extract_paths_rosetta_validate_stats_exempt() {
        let cmd = Commands::Rosetta {
            action: RosettaCommands::ValidateStats {
                model: PathBuf::from("model.apr"),
                reference: None,
                fingerprints: None,
                threshold: 3.0,
                strict: false,
                json: false,
            },
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Rosetta ValidateStats is a diagnostic command (exempt)"
        );
    }

    // =========================================================================
    // validate_model_contract: additional edge cases
    // =========================================================================

    /// Test validate_model_contract: multiple non-existent paths all skipped
    #[test]
    fn test_validate_contract_multiple_nonexistent() {
        let paths = vec![
            PathBuf::from("/tmp/nonexistent_a.apr"),
            PathBuf::from("/tmp/nonexistent_b.gguf"),
            PathBuf::from("/tmp/nonexistent_c.safetensors"),
        ];
        let result = validate_model_contract(&paths);
        assert!(result.is_ok(), "All non-existent paths should be skipped");
    }

    /// Test validate_model_contract: mix of non-existent paths
    #[test]
    fn test_validate_contract_mixed_nonexistent() {
        let paths = vec![
            PathBuf::from("/tmp/does_not_exist_xyz.apr"),
            PathBuf::from("/tmp/also_missing_123.gguf"),
        ];
        let result = validate_model_contract(&paths);
        assert!(
            result.is_ok(),
            "Mixed non-existent paths should all be skipped"
        );
    }

    // =========================================================================
    // execute_command: error path tests (file not found)
    // =========================================================================

    /// Helper: create a Cli struct with the given command and default flags
    fn make_cli(command: Commands) -> Cli {
        Cli {
            command: Box::new(command),
            json: false,
            verbose: false,
            quiet: false,
            offline: false,
            skip_contract: true, // Skip contract to test command dispatch errors
        }
    }

    /// Test execute_command: Inspect with non-existent file returns error
    #[test]
    fn test_execute_inspect_file_not_found() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_model_inspect_test.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Inspect should fail with non-existent file"
        );
    }

    /// Test execute_command: Debug with non-existent file returns error
    #[test]
    fn test_execute_debug_file_not_found() {
        let cli = make_cli(Commands::Debug {
            file: PathBuf::from("/tmp/nonexistent_model_debug_test.apr"),
            drama: false,
            hex: false,
            strings: false,
            limit: 256,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Debug should fail with non-existent file");
    }

    /// Test execute_command: Validate with non-existent file returns error
    #[test]
    fn test_execute_validate_file_not_found() {
        let cli = make_cli(Commands::Validate {
            file: PathBuf::from("/tmp/nonexistent_model_validate_test.apr"),
            quality: false,
            strict: false,
            min_score: None,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Validate should fail with non-existent file"
        );
    }

    /// Test execute_command: Diff with non-existent files returns error
    #[test]
    fn test_execute_diff_file_not_found() {
        let cli = make_cli(Commands::Diff {
            file1: PathBuf::from("/tmp/nonexistent_model_diff1.apr"),
            file2: PathBuf::from("/tmp/nonexistent_model_diff2.apr"),
            weights: false,
            values: false,
            filter: None,
            limit: 10,
            transpose_aware: false,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Diff should fail with non-existent files");
    }

    /// Test execute_command: Tensors with non-existent file returns error
    #[test]
    fn test_execute_tensors_file_not_found() {
        let cli = make_cli(Commands::Tensors {
            file: PathBuf::from("/tmp/nonexistent_model_tensors_test.apr"),
            stats: false,
            filter: None,
            limit: 0,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Tensors should fail with non-existent file"
        );
    }

    /// Test execute_command: Lint with non-existent file returns error
    #[test]
    fn test_execute_lint_file_not_found() {
        let cli = make_cli(Commands::Lint {
            file: PathBuf::from("/tmp/nonexistent_model_lint_test.apr"),
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Lint should fail with non-existent file");
    }

    /// Test execute_command: Trace with non-existent file returns error
    #[test]
    fn test_execute_trace_file_not_found() {
        let cli = make_cli(Commands::Trace {
            file: PathBuf::from("/tmp/nonexistent_model_trace_test.apr"),
            layer: None,
            reference: None,
            json: false,
            verbose: false,
            payload: false,
            diff: false,
            interactive: false,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Trace should fail with non-existent file");
    }
