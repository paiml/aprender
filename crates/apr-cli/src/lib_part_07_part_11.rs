
    /// Test verbose inheritance end-to-end with -v after the subcommand.
    /// Because clap uses `global = true` + `short = 'v'` on both Cli and
    /// Run, -v placed after the subcommand sets the global verbose field.
    #[test]
    fn test_verbose_inheritance_run_local() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "-v"];
        let cli = parse_cli(args).expect("Failed to parse");
        // -v after the subcommand still sets the global flag due to global = true
        match *cli.command {
            Commands::Run { verbose, .. } => {
                let effective = verbose || cli.verbose;
                assert!(effective, "effective verbose should be true");
            }
            _ => panic!("Expected Run command"),
        }
    }

    // =========================================================================
    // Edge case: conflicting flags (--gpu vs --no-gpu)
    // =========================================================================

    /// Test that --gpu and --no-gpu conflict (Run command)
    #[test]
    fn test_parse_run_gpu_nogpu_conflict() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--gpu",
            "--no-gpu",
        ];
        let result = parse_cli(args);
        assert!(result.is_err(), "--gpu and --no-gpu should conflict in Run");
    }

    /// Test parsing 'apr run' with --gpu flag alone
    #[test]
    fn test_parse_run_gpu_only() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "--gpu"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { gpu, no_gpu, .. } => {
                assert!(gpu);
                assert!(!no_gpu);
            }
            _ => panic!("Expected Run command"),
        }
    }

    // =========================================================================
    // Missing required args error tests
    // =========================================================================

    /// Test that 'apr serve' without FILE fails
    #[test]
    fn test_missing_serve_file() {
        let args = vec!["apr", "serve"];
        let result = parse_cli(args);
        assert!(result.is_err(), "serve requires FILE");
    }

    /// Test that 'apr diff' with only one file fails
    #[test]
    fn test_missing_diff_second_file() {
        let args = vec!["apr", "diff", "model1.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "diff requires two files");
    }

    /// Test that 'apr export' without output fails
    #[test]
    fn test_missing_export_output() {
        // Output is now optional at parse level (validated at runtime)
        // so parse succeeds, but execution should fail with ValidationFailed
        let args = vec!["apr", "export", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_ok(), "export parses without -o (validated at runtime)");
    }

    /// Test that 'apr convert' without output fails
    #[test]
    fn test_missing_convert_output() {
        let args = vec!["apr", "convert", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "convert requires -o/--output");
    }

    /// Test that 'apr merge' with fewer than 2 files fails
    #[test]
    fn test_missing_merge_files() {
        let args = vec!["apr", "merge", "model1.apr", "-o", "out.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "merge requires at least 2 files");
    }

    /// Test that 'apr publish' without repo_id fails
    #[test]
    fn test_missing_publish_repo_id() {
        let args = vec!["apr", "publish", "/tmp/models"];
        let result = parse_cli(args);
        assert!(result.is_err(), "publish requires REPO_ID");
    }

    /// Test that 'apr pull' without model_ref fails
    #[test]
    fn test_missing_pull_model_ref() {
        let args = vec!["apr", "pull"];
        let result = parse_cli(args);
        assert!(result.is_err(), "pull requires MODEL");
    }

    /// Test that 'apr rm' without model_ref fails
    #[test]
    fn test_missing_rm_model_ref() {
        let args = vec!["apr", "rm"];
        let result = parse_cli(args);
        assert!(result.is_err(), "rm requires MODEL");
    }

    /// Test that 'apr compare-hf' without --hf fails
    #[test]
    fn test_missing_compare_hf_hf_arg() {
        let args = vec!["apr", "compare-hf", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "compare-hf requires --hf");
    }

    /// Test that 'apr canary create' without --input fails
    #[test]
    fn test_missing_canary_create_input() {
        let args = vec![
            "apr",
            "canary",
            "create",
            "model.apr",
            "--output",
            "canary.json",
        ];
        let result = parse_cli(args);
        assert!(result.is_err(), "canary create requires --input");
    }

    /// Test that 'apr canary check' without --canary fails
    #[test]
    fn test_missing_canary_check_canary() {
        let args = vec!["apr", "canary", "check", "model.apr"];
        let result = parse_cli(args);
        assert!(result.is_err(), "canary check requires --canary");
    }

    // =========================================================================
    // execute_command: contract gate integration
    // =========================================================================

    /// Test that execute_command with skip_contract=false and non-existent paths
    /// still works because non-existent paths are skipped in validate_model_contract
    #[test]
    fn test_execute_with_contract_gate_nonexistent() {
        let cli = Cli {
            command: Box::new(Commands::Inspect {
                file: PathBuf::from("/tmp/nonexistent_contract_test.apr"),
                vocab: false,
                filters: false,
                weights: false,
                json: false,
            }),
            json: false,
            verbose: false,
            quiet: false,
            offline: false,
            skip_contract: false, // Contract enabled, but paths don't exist
        };
        // The contract gate should pass (non-existent paths are skipped),
        // but the command itself should fail (file not found)
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Should still fail from command execution, not contract"
        );
    }

    /// Test that execute_command dispatches List even with contract enabled
    #[test]
    fn test_execute_list_with_contract_enabled() {
        let cli = Cli {
            command: Box::new(Commands::List),
            json: false,
            verbose: false,
            quiet: false,
            offline: false,
            skip_contract: false, // Contract enabled
        };
        let result = execute_command(&cli);
        assert!(result.is_ok(), "List should succeed with contract enabled");
    }

    // =========================================================================
    // Rosetta command execution error paths
    // =========================================================================

    /// Test execute_command: Rosetta inspect with non-existent file returns error
    #[test]
    fn test_execute_rosetta_inspect_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Rosetta {
            action: RosettaCommands::Inspect {
                file: PathBuf::from("/tmp/nonexistent_rosetta_inspect.gguf"),
                hexdump: false,
                json: false,
            },
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Rosetta inspect should fail with non-existent file"
        );
    }

    /// Test execute_command: Rosetta convert with non-existent source returns error
    #[test]
    fn test_execute_rosetta_convert_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Rosetta {
            action: RosettaCommands::Convert {
                source: PathBuf::from("/tmp/nonexistent_rosetta_convert.gguf"),
                target: PathBuf::from("/tmp/out.safetensors"),
                quantize: None,
                verify: false,
                json: false,
                tokenizer: None,
            },
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Rosetta convert should fail with non-existent source"
        );
    }

    /// Test execute_command: Rosetta fingerprint with non-existent file returns error
    #[test]
    fn test_execute_rosetta_fingerprint_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Rosetta {
            action: RosettaCommands::Fingerprint {
                model: PathBuf::from("/tmp/nonexistent_rosetta_fingerprint.gguf"),
                model_b: None,
                output: None,
                filter: None,
                verbose: false,
                json: false,
            },
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Rosetta fingerprint should fail with non-existent file"
        );
    }

    /// Test execute_command: Bench with non-existent file returns error
    #[test]
    fn test_execute_bench_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Bench {
            file: PathBuf::from("/tmp/nonexistent_model_bench_test.gguf"),
            warmup: 1,
            iterations: 1,
            max_tokens: 1,
            prompt: None,
            fast: false,
            brick: None,
        }));
        let result = execute_command(&cli);
        assert!(result.is_err(), "Bench should fail with non-existent file");
    }

    /// Test execute_command: Eval with non-existent file returns error
    #[test]
    fn test_execute_eval_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Eval {
            file: PathBuf::from("/tmp/nonexistent_model_eval_test.gguf"),
            dataset: "wikitext-2".to_string(),
            text: None,
            max_tokens: 32,
            threshold: 20.0,
        }));
        let result = execute_command(&cli);
        assert!(result.is_err(), "Eval should fail with non-existent file");
    }

    /// Test execute_command: Profile with non-existent file returns error
    #[test]
    fn test_execute_profile_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Profile {
            file: PathBuf::from("/tmp/nonexistent_model_profile_test.apr"),
            granular: false,
            format: "human".to_string(),
            focus: None,
            detect_naive: false,
            threshold: 10.0,
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
            no_gpu: false,
            compare: None,
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Profile should fail with non-existent file"
        );
    }

    /// Test execute_command: CompareHf with non-existent file returns error
    #[test]
    fn test_execute_compare_hf_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::CompareHf {
            file: PathBuf::from("/tmp/nonexistent_model_compare_hf_test.apr"),
            hf: "openai/whisper-tiny".to_string(),
            tensor: None,
            threshold: 1e-5,
            json: false,
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "CompareHf should fail with non-existent file"
        );
    }

    /// Test execute_command: Canary check with non-existent file returns error
    #[test]
    fn test_execute_canary_check_file_not_found() {
        let cli = make_cli(Commands::Canary {
            command: CanaryCommands::Check {
                file: PathBuf::from("/tmp/nonexistent_canary_check.apr"),
                canary: PathBuf::from("/tmp/nonexistent_canary.json"),
            },
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Canary check should fail with non-existent file"
        );
    }

    /// Test execute_command: Publish with non-existent directory returns error
    #[test]
    fn test_execute_publish_dir_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Publish {
            directory: PathBuf::from("/tmp/nonexistent_publish_dir_test"),
            repo_id: "test/test".to_string(),
            model_name: None,
            license: "mit".to_string(),
            pipeline_tag: "text-generation".to_string(),
            library_name: None,
            tags: None,
            message: None,
            dry_run: true, // Use dry_run to avoid actual upload
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Publish should fail with non-existent directory"
        );
    }

    // =========================================================================
    // Default value verification tests
    // =========================================================================

    /// Test Run command defaults
    #[test]
    fn test_parse_run_defaults() {
        let args = vec!["apr", "run", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                max_tokens,
                stream,
                format,
                no_gpu,
                gpu,
                offline,
                benchmark,
                trace,
                trace_payload,
                trace_verbose,
                trace_level,
                profile,
                chat,
                verbose,
                prompt,
                input,
                language,
                task,
                trace_steps,
                trace_output,
                ..
            } => {
                assert_eq!(max_tokens, 32);
                assert!(!stream);
                assert_eq!(format, "text");
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!offline);
                assert!(!benchmark);
                assert!(!trace);
                assert!(!trace_payload);
                assert!(!trace_verbose);
                assert_eq!(trace_level, "basic");
                assert!(!profile);
                assert!(!chat);
                assert!(!verbose);
                assert!(prompt.is_none());
                assert!(input.is_none());
                assert!(language.is_none());
                assert!(task.is_none());
                assert!(trace_steps.is_none());
                assert!(trace_output.is_none());
            }
            _ => panic!("Expected Run command"),
        }
    }
