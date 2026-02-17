
    /// Test parsing rosetta fingerprint subcommand
    #[test]
    fn test_parse_rosetta_fingerprint() {
        let args = vec![
            "apr",
            "rosetta",
            "fingerprint",
            "model.gguf",
            "model2.apr",
            "--output",
            "fingerprints.json",
            "--filter",
            "encoder",
            "--verbose",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Rosetta { action }) => match action {
                RosettaCommands::Fingerprint {
                    model,
                    model_b,
                    output,
                    filter,
                    verbose,
                    json,
                } => {
                    assert_eq!(model, PathBuf::from("model.gguf"));
                    assert_eq!(model_b, Some(PathBuf::from("model2.apr")));
                    assert_eq!(output, Some(PathBuf::from("fingerprints.json")));
                    assert_eq!(filter, Some("encoder".to_string()));
                    assert!(verbose);
                    assert!(json);
                }
                _ => panic!("Expected Fingerprint subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing rosetta validate-stats subcommand
    #[test]
    fn test_parse_rosetta_validate_stats() {
        let args = vec![
            "apr",
            "rosetta",
            "validate-stats",
            "model.apr",
            "--reference",
            "ref.gguf",
            "--fingerprints",
            "fp.json",
            "--threshold",
            "5.0",
            "--strict",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Rosetta { action }) => match action {
                RosettaCommands::ValidateStats {
                    model,
                    reference,
                    fingerprints,
                    threshold,
                    strict,
                    json,
                } => {
                    assert_eq!(model, PathBuf::from("model.apr"));
                    assert_eq!(reference, Some(PathBuf::from("ref.gguf")));
                    assert_eq!(fingerprints, Some(PathBuf::from("fp.json")));
                    assert!((threshold - 5.0).abs() < f32::EPSILON);
                    assert!(strict);
                    assert!(json);
                }
                _ => panic!("Expected ValidateStats subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    // =========================================================================
    // Global flag tests
    // =========================================================================

    /// Test global --offline flag
    #[test]
    fn test_global_offline_flag() {
        let args = vec!["apr", "--offline", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.offline);
    }

    /// Test global --quiet flag
    #[test]
    fn test_global_quiet_flag() {
        let args = vec!["apr", "--quiet", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.quiet);
    }

    /// Test multiple global flags combined
    #[test]
    fn test_multiple_global_flags() {
        let args = vec![
            "apr",
            "--verbose",
            "--json",
            "--offline",
            "--quiet",
            "--skip-contract",
            "inspect",
            "model.apr",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.verbose);
        assert!(cli.json);
        assert!(cli.offline);
        assert!(cli.quiet);
        assert!(cli.skip_contract);
    }

    /// Test global flags default to false
    #[test]
    fn test_global_flags_default_false() {
        let args = vec!["apr", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(!cli.verbose);
        assert!(!cli.json);
        assert!(!cli.offline);
        assert!(!cli.quiet);
        assert!(!cli.skip_contract);
    }

    // =========================================================================
    // extract_model_paths: additional command variants
    // =========================================================================

    /// Test extract_model_paths: Export returns file path
    #[test]
    fn test_extract_paths_export() {
        let cmd = Commands::Export {
            file: Some(PathBuf::from("model.apr")),
            format: "gguf".to_string(),
            output: Some(PathBuf::from("out.gguf")),
            quantize: None,
            list_formats: false,
            batch: None,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Convert returns file path
    #[test]
    fn test_extract_paths_convert() {
        let cmd = Commands::Convert {
            file: PathBuf::from("model.apr"),
            quantize: Some("q4k".to_string()),
            compress: None,
            output: PathBuf::from("out.apr"),
            force: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Check returns file path
    #[test]
    fn test_extract_paths_check() {
        let cmd = Commands::Check {
            file: PathBuf::from("model.gguf"),
            no_gpu: false,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Trace returns file path
    #[test]
    fn test_extract_paths_trace() {
        let cmd = Commands::Trace {
            file: PathBuf::from("model.apr"),
            layer: None,
            reference: None,
            json: false,
            verbose: false,
            payload: false,
            diff: false,
            interactive: false,
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Probar returns file path
    #[test]
    fn test_extract_paths_probar() {
        let cmd = Commands::Extended(ExtendedCommands::Probar {
            file: PathBuf::from("model.apr"),
            output: PathBuf::from("./probar-export"),
            format: "both".to_string(),
            golden: None,
            layer: None,
        });
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: CompareHf returns file path
    #[test]
    fn test_extract_paths_compare_hf() {
        let cmd = Commands::Extended(ExtendedCommands::CompareHf {
            file: PathBuf::from("model.apr"),
            hf: "openai/whisper-tiny".to_string(),
            tensor: None,
            threshold: 1e-5,
            json: false,
        });
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Chat returns file path
    #[test]
    fn test_extract_paths_chat() {
        let cmd = Commands::Extended(ExtendedCommands::Chat {
            file: PathBuf::from("model.gguf"),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            no_gpu: false,
            gpu: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
        });
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Eval returns file path
    #[test]
    fn test_extract_paths_eval() {
        let cmd = Commands::Extended(ExtendedCommands::Eval {
            file: PathBuf::from("model.gguf"),
            dataset: "wikitext-2".to_string(),
            text: None,
            max_tokens: 512,
            threshold: 20.0,
        });
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Profile returns file path
    #[test]
    fn test_extract_paths_profile() {
        let cmd = Commands::Extended(ExtendedCommands::Profile {
            file: PathBuf::from("model.apr"),
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
        });
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Import with hf:// URL returns empty (non-local)
    #[test]
    fn test_extract_paths_import_hf_url() {
        let cmd = Commands::Import {
            source: "hf://openai/whisper-tiny".to_string(),
            output: Some(PathBuf::from("whisper.apr")),
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
            allow_no_config: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "hf:// URLs should not be validated locally for import"
        );
    }

    /// Test extract_model_paths: Import with non-existent local path returns empty
    #[test]
    fn test_extract_paths_import_nonexistent_local() {
        let cmd = Commands::Import {
            source: "/tmp/nonexistent_model_abc123.gguf".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
            allow_no_config: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(
            paths.is_empty(),
            "Non-existent local paths return empty for import"
        );
    }

    /// Test extract_model_paths: Tui with file returns file
    #[test]
    fn test_extract_paths_tui_with_file() {
        let cmd = Commands::Tui {
            file: Some(PathBuf::from("model.apr")),
        };
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.apr")]);
    }

    /// Test extract_model_paths: Tui without file returns empty
    #[test]
    fn test_extract_paths_tui_no_file() {
        let cmd = Commands::Tui { file: None };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty());
    }

    /// Test extract_model_paths: Cbtop with model_path returns it
    #[test]
    fn test_extract_paths_cbtop_with_model_path() {
        let cmd = Commands::Extended(ExtendedCommands::Cbtop {
            model: None,
            attach: None,
            model_path: Some(PathBuf::from("model.gguf")),
            headless: false,
            json: false,
            output: None,
            ci: false,
            throughput: None,
            brick_score: None,
            warmup: 10,
            iterations: 100,
            speculative: false,
            speculation_k: 4,
            draft_model: None,
            concurrent: 1,
            simulated: false,
        });
        let paths = extract_model_paths(&cmd);
        assert_eq!(paths, vec![PathBuf::from("model.gguf")]);
    }

    /// Test extract_model_paths: Cbtop without model_path returns empty
    #[test]
    fn test_extract_paths_cbtop_no_model_path() {
        let cmd = Commands::Extended(ExtendedCommands::Cbtop {
            model: Some("qwen2.5-coder".to_string()),
            attach: None,
            model_path: None,
            headless: false,
            json: false,
            output: None,
            ci: false,
            throughput: None,
            brick_score: None,
            warmup: 10,
            iterations: 100,
            speculative: false,
            speculation_k: 4,
            draft_model: None,
            concurrent: 1,
            simulated: false,
        });
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty());
    }

    /// Test extract_model_paths: Diff is diagnostic (exempt)
    #[test]
    fn test_extract_paths_diff_exempt() {
        let cmd = Commands::Diff {
            file1: PathBuf::from("a.apr"),
            file2: PathBuf::from("b.apr"),
            weights: false,
            values: false,
            filter: None,
            limit: 10,
            transpose_aware: false,
            json: false,
        };
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Diff is a diagnostic command (exempt)");
    }

    /// Test extract_model_paths: Hex is diagnostic (exempt)
    #[test]
    fn test_extract_paths_hex_exempt() {
        let cmd = Commands::Extended(ExtendedCommands::Hex {
            file: PathBuf::from("model.apr"),
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
        });
        let paths = extract_model_paths(&cmd);
        assert!(paths.is_empty(), "Hex is a diagnostic command (exempt)");
    }
