
    /// Test parsing 'apr rosetta inspect' command
    #[test]
    fn test_parse_rosetta_inspect() {
        let args = vec!["apr", "rosetta", "inspect", "model.gguf", "--json"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tools(ToolCommands::Rosetta { action })) => match action {
                RosettaCommands::Inspect { file, json, .. } => {
                    assert_eq!(file, PathBuf::from("model.gguf"));
                    assert!(json);
                }
                _ => panic!("Expected Inspect subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing 'apr rosetta convert' command
    #[test]
    fn test_parse_rosetta_convert() {
        let args = vec![
            "apr",
            "rosetta",
            "convert",
            "model.gguf",
            "model.safetensors",
            "--verify",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tools(ToolCommands::Rosetta { action })) => match action {
                RosettaCommands::Convert {
                    source,
                    target,
                    verify,
                    ..
                } => {
                    assert_eq!(source, PathBuf::from("model.gguf"));
                    assert_eq!(target, PathBuf::from("model.safetensors"));
                    assert!(verify);
                }
                _ => panic!("Expected Convert subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing 'apr rosetta chain' command
    #[test]
    fn test_parse_rosetta_chain() {
        let args = vec![
            "apr",
            "rosetta",
            "chain",
            "model.gguf",
            "safetensors",
            "apr",
            "--work-dir",
            "/tmp/rosetta",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tools(ToolCommands::Rosetta { action })) => match action {
                RosettaCommands::Chain {
                    source,
                    formats,
                    work_dir,
                    ..
                } => {
                    assert_eq!(source, PathBuf::from("model.gguf"));
                    assert_eq!(formats, vec!["safetensors", "apr"]);
                    assert_eq!(work_dir, PathBuf::from("/tmp/rosetta"));
                }
                _ => panic!("Expected Chain subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing 'apr rosetta verify' command
    #[test]
    fn test_parse_rosetta_verify() {
        let args = vec![
            "apr",
            "rosetta",
            "verify",
            "model.apr",
            "--intermediate",
            "gguf",
            "--tolerance",
            "1e-4",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tools(ToolCommands::Rosetta { action })) => match action {
                RosettaCommands::Verify {
                    source,
                    intermediate,
                    tolerance,
                    ..
                } => {
                    assert_eq!(source, PathBuf::from("model.apr"));
                    assert_eq!(intermediate, "gguf");
                    assert!((tolerance - 1e-4).abs() < f32::EPSILON);
                }
                _ => panic!("Expected Verify subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    // =========================================================================
    // PMAT-237: Contract gate tests
    // =========================================================================

    /// Test that --skip-contract global flag is parsed
    #[test]
    fn test_parse_skip_contract_flag() {
        let args = vec!["apr", "--skip-contract", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.skip_contract);
    }

    /// Test that --skip-contract defaults to false
    #[test]
    fn test_skip_contract_default_false() {
        let args = vec!["apr", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(!cli.skip_contract);
    }

    /// Test extract_model_paths: diagnostic commands return empty vec
    #[test]
    fn test_extract_paths_diagnostic_exempt() {
        // Diagnostic commands should return no paths (exempt from validation)
        let diagnostic_commands = vec![
            Commands::Inspect {
                file: PathBuf::from("m.apr"),
                vocab: false,
                filters: false,
                weights: false,
                json: false,
            },
            Commands::Debug {
                file: PathBuf::from("m.apr"),
                drama: false,
                hex: false,
                strings: false,
                limit: 256,
            },
            Commands::Validate {
                file: PathBuf::from("m.apr"),
                quality: false,
                strict: false,
                min_score: None,
            },
            Commands::Tensors {
                file: PathBuf::from("m.apr"),
                stats: false,
                filter: None,
                limit: 0,
                json: false,
            },
            Commands::Lint {
                file: PathBuf::from("m.apr"),
            },
            Commands::Extended(ExtendedCommands::Qa {
                file: PathBuf::from("m.apr"),
                assert_tps: None,
                assert_speedup: None,
                assert_gpu_speedup: None,
                skip_golden: false,
                skip_throughput: false,
                skip_ollama: false,
                skip_gpu_speedup: false,
                skip_contract: false,
                skip_format_parity: false,
                skip_ptx_parity: false,
                safetensors_path: None,
                iterations: 10,
                warmup: 3,
                max_tokens: 32,
                json: false,
                verbose: false,
                min_executed: None,
                previous_report: None,
                regression_threshold: None,
                skip_gpu_state: false,
                skip_metadata: false,
                skip_capability: false,
            }),
            Commands::Extended(ExtendedCommands::Hex {
                file: PathBuf::from("m.apr"),
                tensor: None,
                limit: 64,
                stats: false,
                list: false,
                json: false,
                header: false,
                blocks: false,
                distribution: false,
                contract: false,
                entropy: false,
                raw: false,
                offset: "0".to_string(),
                width: 16,
                slice: None,
            }),
            Commands::Extended(ExtendedCommands::Tree {
                file: PathBuf::from("m.apr"),
                filter: None,
                format: "ascii".to_string(),
                sizes: false,
                depth: None,
            }),
            Commands::Extended(ExtendedCommands::Flow {
                file: PathBuf::from("m.apr"),
                layer: None,
                component: "full".to_string(),
                verbose: false,
                json: false,
            }),
            Commands::Explain {
                code: None,
                file: None,
                tensor: None,
            },
            Commands::List,
        ];
        for cmd in &diagnostic_commands {
            let paths = extract_model_paths(cmd);
            assert!(
                paths.is_empty(),
                "Diagnostic command should be exempt: {cmd:?}"
            );
        }
    }

    /// Test extract_model_paths: action commands return file paths
    #[test]
    fn test_extract_paths_action_commands() {
        let serve_cmd = Commands::Serve {
            file: PathBuf::from("model.gguf"),
            port: 8080,
            host: "127.0.0.1".to_string(),
            no_cors: false,
            no_metrics: false,
            no_gpu: false,
            gpu: false,
            batch: false,
            trace: false,
            trace_level: "basic".to_string(),
            profile: false,
        };
        let paths = extract_model_paths(&serve_cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);

        let bench_cmd = Commands::Extended(ExtendedCommands::Bench {
            file: PathBuf::from("model.apr"),
            warmup: 3,
            iterations: 5,
            max_tokens: 32,
            prompt: None,
            fast: false,
            brick: None,
        });
        let paths = extract_model_paths(&bench_cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Run with hf:// URL returns empty
    #[test]
    fn test_extract_paths_run_hf_url() {
        let cmd = Commands::Run {
            source: "hf://org/repo".to_string(),
            positional_prompt: None,
            input: None,
            prompt: None,
            max_tokens: 32,
            stream: false,
            language: None,
            task: None,
            format: "text".to_string(),
            no_gpu: false,
            gpu: false,
            offline: false,
            benchmark: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            trace_payload: false,
            profile: false,
            chat: false,
            verbose: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "hf:// URLs should not be validated locally"
        );
    }

    /// Test extract_model_paths: Merge returns multiple files
    #[test]
    fn test_extract_paths_merge_multiple() {
        let cmd = Commands::Merge {
            files: vec![
                PathBuf::from("a.apr"),
                PathBuf::from("b.apr"),
                PathBuf::from("c.apr"),
            ],
            strategy: "average".to_string(),
            output: PathBuf::from("merged.apr"),
            weights: None,
            base_model: None,
            drop_rate: 0.9,
            density: 0.2,
            seed: 42,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths.len(), 3);
    }

    /// Test validate_model_contract: non-existent path is skipped (Ok)
    #[test]
    fn test_validate_contract_nonexistent_skipped() {
        let paths = vec![PathBuf::from("nonexistent_model_xyz.apr")];
        let result = validate_model_contract(&paths);
        assert!(result.is_ok(), "Non-existent paths should be skipped");
    }

    /// Test validate_model_contract: empty paths is Ok
    #[test]
    fn test_validate_contract_empty_paths() {
        let result = validate_model_contract(&[]);
        assert!(result.is_ok());
    }

    // =========================================================================
    // Parse tests for all remaining command variants
    // =========================================================================

    /// Test parsing 'apr publish' command with all options
    #[test]
    fn test_parse_publish_command() {
        let args = vec![
            "apr",
            "publish",
            "/tmp/models",
            "paiml/whisper-apr-tiny",
            "--model-name",
            "Whisper Tiny",
            "--license",
            "apache-2.0",
            "--pipeline-tag",
            "automatic-speech-recognition",
            "--library-name",
            "whisper-apr",
            "--tags",
            "whisper,tiny,asr",
            "--message",
            "Initial release",
            "--dry-run",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tools(ToolCommands::Publish {
                directory,
                repo_id,
                model_name,
                license,
                pipeline_tag,
                library_name,
                tags,
                message,
                dry_run,
            })) => {
                assert_eq!(directory, PathBuf::from("/tmp/models"));
                assert_eq!(repo_id, "paiml/whisper-apr-tiny");
                assert_eq!(model_name, Some("Whisper Tiny".to_string()));
                assert_eq!(license, "apache-2.0");
                assert_eq!(pipeline_tag, "automatic-speech-recognition");
                assert_eq!(library_name, Some("whisper-apr".to_string()));
                assert_eq!(
                    tags,
                    Some(vec![
                        "whisper".to_string(),
                        "tiny".to_string(),
                        "asr".to_string()
                    ])
                );
                assert_eq!(message, Some("Initial release".to_string()));
                assert!(dry_run);
            }
            _ => panic!("Expected Publish command"),
        }
    }

    /// Test parsing 'apr publish' with defaults
    #[test]
    fn test_parse_publish_defaults() {
        let args = vec!["apr", "publish", "./models", "org/repo"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tools(ToolCommands::Publish {
                license,
                pipeline_tag,
                dry_run,
                model_name,
                library_name,
                tags,
                message,
                ..
            })) => {
                assert_eq!(license, "mit");
                assert_eq!(pipeline_tag, "text-generation");
                assert!(!dry_run);
                assert!(model_name.is_none());
                assert!(library_name.is_none());
                assert!(tags.is_none());
                assert!(message.is_none());
            }
            _ => panic!("Expected Publish command"),
        }
    }
