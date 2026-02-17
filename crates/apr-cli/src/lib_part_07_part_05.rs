
    /// Test parsing 'apr export' command
    #[test]
    fn test_parse_export_command() {
        let args = vec![
            "apr",
            "export",
            "model.apr",
            "--format",
            "gguf",
            "-o",
            "model.gguf",
            "--quantize",
            "int4",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Export {
                file,
                format,
                output,
                quantize,
                ..
            } => {
                assert_eq!(file, Some(PathBuf::from("model.apr")));
                assert_eq!(format, "gguf");
                assert_eq!(output, Some(PathBuf::from("model.gguf")));
                assert_eq!(quantize, Some("int4".to_string()));
            }
            _ => panic!("Expected Export command"),
        }
    }

    /// Test parsing 'apr export' with defaults
    #[test]
    fn test_parse_export_defaults() {
        let args = vec!["apr", "export", "model.apr", "-o", "out.safetensors"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Export {
                format, quantize, ..
            } => {
                assert_eq!(format, "safetensors");
                assert!(quantize.is_none());
            }
            _ => panic!("Expected Export command"),
        }
    }

    /// Test parsing 'apr convert' command with all options
    #[test]
    fn test_parse_convert_command() {
        let args = vec![
            "apr",
            "convert",
            "model.apr",
            "--quantize",
            "q4k",
            "--compress",
            "zstd",
            "-o",
            "model-q4k.apr",
            "--force",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Convert {
                file,
                quantize,
                compress,
                output,
                force,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(quantize, Some("q4k".to_string()));
                assert_eq!(compress, Some("zstd".to_string()));
                assert_eq!(output, PathBuf::from("model-q4k.apr"));
                assert!(force);
            }
            _ => panic!("Expected Convert command"),
        }
    }

    /// Test parsing 'apr convert' with defaults
    #[test]
    fn test_parse_convert_defaults() {
        let args = vec!["apr", "convert", "model.apr", "-o", "out.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Convert {
                quantize,
                compress,
                force,
                ..
            } => {
                assert!(quantize.is_none());
                assert!(compress.is_none());
                assert!(!force);
            }
            _ => panic!("Expected Convert command"),
        }
    }

    /// Test parsing 'apr oracle' command with source
    #[test]
    fn test_parse_oracle_command_with_source() {
        let args = vec![
            "apr",
            "oracle",
            "model.gguf",
            "--compliance",
            "--tensors",
            "--stats",
            "--explain",
            "--kernels",
            "--validate",
            "--full",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Oracle {
                source,
                compliance,
                tensors,
                stats,
                explain,
                kernels,
                validate,
                full,
                family,
                size,
            }) => {
                assert_eq!(source, Some("model.gguf".to_string()));
                assert!(compliance);
                assert!(tensors);
                assert!(stats);
                assert!(explain);
                assert!(kernels);
                assert!(validate);
                assert!(full);
                assert!(family.is_none());
                assert!(size.is_none());
            }
            _ => panic!("Expected Oracle command"),
        }
    }

    /// Test parsing 'apr oracle' with --family flag
    #[test]
    fn test_parse_oracle_family_mode() {
        let args = vec!["apr", "oracle", "--family", "qwen2", "--size", "7b"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Oracle {
                source,
                family,
                size,
                ..
            }) => {
                assert!(source.is_none());
                assert_eq!(family, Some("qwen2".to_string()));
                assert_eq!(size, Some("7b".to_string()));
            }
            _ => panic!("Expected Oracle command"),
        }
    }

    /// Test parsing 'apr oracle' with hf:// URI
    #[test]
    fn test_parse_oracle_hf_uri() {
        let args = vec!["apr", "oracle", "hf://Qwen/Qwen2.5-Coder-1.5B"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Oracle { source, .. }) => {
                assert_eq!(source, Some("hf://Qwen/Qwen2.5-Coder-1.5B".to_string()));
            }
            _ => panic!("Expected Oracle command"),
        }
    }

    /// Test parsing 'apr canary create' subcommand
    #[test]
    fn test_parse_canary_create() {
        let args = vec![
            "apr",
            "canary",
            "create",
            "model.apr",
            "--input",
            "audio.wav",
            "--output",
            "canary.json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Canary { command } => match command {
                CanaryCommands::Create {
                    file,
                    input,
                    output,
                } => {
                    assert_eq!(file, PathBuf::from("model.apr"));
                    assert_eq!(input, PathBuf::from("audio.wav"));
                    assert_eq!(output, PathBuf::from("canary.json"));
                }
                _ => panic!("Expected Create subcommand"),
            },
            _ => panic!("Expected Canary command"),
        }
    }

    /// Test parsing 'apr canary check' subcommand
    #[test]
    fn test_parse_canary_check() {
        let args = vec![
            "apr",
            "canary",
            "check",
            "model.apr",
            "--canary",
            "canary.json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Canary { command } => match command {
                CanaryCommands::Check { file, canary } => {
                    assert_eq!(file, PathBuf::from("model.apr"));
                    assert_eq!(canary, PathBuf::from("canary.json"));
                }
                _ => panic!("Expected Check subcommand"),
            },
            _ => panic!("Expected Canary command"),
        }
    }

    /// Test parsing 'apr compare-hf' command
    #[test]
    fn test_parse_compare_hf_command() {
        let args = vec![
            "apr",
            "compare-hf",
            "model.apr",
            "--hf",
            "openai/whisper-tiny",
            "--tensor",
            "encoder.0",
            "--threshold",
            "1e-3",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::CompareHf {
                file,
                hf,
                tensor,
                threshold,
                json,
            }) => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(hf, "openai/whisper-tiny");
                assert_eq!(tensor, Some("encoder.0".to_string()));
                assert!((threshold - 1e-3).abs() < f64::EPSILON);
                assert!(json);
            }
            _ => panic!("Expected CompareHf command"),
        }
    }

    /// Test parsing 'apr compare-hf' with defaults
    #[test]
    fn test_parse_compare_hf_defaults() {
        let args = vec![
            "apr",
            "compare-hf",
            "model.apr",
            "--hf",
            "openai/whisper-tiny",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::CompareHf {
                tensor,
                threshold,
                json,
                ..
            }) => {
                assert!(tensor.is_none());
                assert!((threshold - 1e-5).abs() < f64::EPSILON);
                assert!(!json);
            }
            _ => panic!("Expected CompareHf command"),
        }
    }

    /// Test parsing 'apr pull' command
    #[test]
    fn test_parse_pull_command() {
        let args = vec!["apr", "pull", "hf://Qwen/Qwen2.5-Coder-1.5B", "--force"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Pull { model_ref, force } => {
                assert_eq!(model_ref, "hf://Qwen/Qwen2.5-Coder-1.5B");
                assert!(force);
            }
            _ => panic!("Expected Pull command"),
        }
    }

    /// Test parsing 'apr pull' without force
    #[test]
    fn test_parse_pull_defaults() {
        let args = vec!["apr", "pull", "qwen2.5-coder"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Pull { model_ref, force } => {
                assert_eq!(model_ref, "qwen2.5-coder");
                assert!(!force);
            }
            _ => panic!("Expected Pull command"),
        }
    }

    /// Test parsing 'apr tune' command with all options
    #[test]
    fn test_parse_tune_command() {
        let args = vec![
            "apr",
            "tune",
            "model.apr",
            "--method",
            "lora",
            "--rank",
            "16",
            "--vram",
            "24.0",
            "--plan",
            "--model",
            "7B",
            "--freeze-base",
            "--train-data",
            "data.jsonl",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tune {
                file,
                method,
                rank,
                vram,
                plan,
                model,
                freeze_base,
                train_data,
                json,
            }) => {
                assert_eq!(file, Some(PathBuf::from("model.apr")));
                assert_eq!(method, "lora");
                assert_eq!(rank, Some(16));
                assert!((vram - 24.0).abs() < f64::EPSILON);
                assert!(plan);
                assert_eq!(model, Some("7B".to_string()));
                assert!(freeze_base);
                assert_eq!(train_data, Some(PathBuf::from("data.jsonl")));
                assert!(json);
            }
            _ => panic!("Expected Tune command"),
        }
    }

    /// Test parsing 'apr tune' with defaults (no file)
    #[test]
    fn test_parse_tune_defaults() {
        let args = vec!["apr", "tune"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tune {
                file,
                method,
                rank,
                vram,
                plan,
                model,
                freeze_base,
                train_data,
                json,
            }) => {
                assert!(file.is_none());
                assert_eq!(method, "auto");
                assert!(rank.is_none());
                assert!((vram - 16.0).abs() < f64::EPSILON);
                assert!(!plan);
                assert!(model.is_none());
                assert!(!freeze_base);
                assert!(train_data.is_none());
                assert!(!json);
            }
            _ => panic!("Expected Tune command"),
        }
    }

    /// Test parsing 'apr check' command
    #[test]
    fn test_parse_check_command() {
        let args = vec!["apr", "check", "model.apr", "--no-gpu"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Check { file, no_gpu, .. } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(no_gpu);
            }
            _ => panic!("Expected Check command"),
        }
    }

    /// Test parsing 'apr check' with defaults
    #[test]
    fn test_parse_check_defaults() {
        let args = vec!["apr", "check", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Check { no_gpu, .. } => {
                assert!(!no_gpu);
            }
            _ => panic!("Expected Check command"),
        }
    }

    /// Test parsing 'apr lint' command
    #[test]
    fn test_parse_lint_command() {
        let args = vec!["apr", "lint", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Lint { file } => {
                assert_eq!(file, PathBuf::from("model.apr"));
            }
            _ => panic!("Expected Lint command"),
        }
    }
