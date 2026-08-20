
    /// Test parsing 'apr eval' command with all options
    #[test]
    fn test_parse_eval_command() {
        let args = vec![
            "apr",
            "eval",
            "model.gguf",
            "--dataset",
            "lambada",
            "--text",
            "The quick brown fox",
            "--max-tokens",
            "256",
            "--threshold",
            "15.5",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Eval {
                file,
                dataset,
                text,
                max_tokens,
                threshold,
                task,
                data,
                model_size,
                num_classes,
                generate_card,
                ..
            }) => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert_eq!(dataset, "lambada");
                assert_eq!(text, Some("The quick brown fox".to_string()));
                assert_eq!(max_tokens, 256);
                assert!((threshold - 15.5).abs() < f32::EPSILON);
                assert!(task.is_none());
                assert!(data.is_none());
                assert!(model_size.is_none());
                assert_eq!(num_classes, 5);
                assert!(!generate_card);
            }
            _ => panic!("Expected Eval command"),
        }
    }

    /// Test parsing 'apr eval' with defaults
    #[test]
    fn test_parse_eval_defaults() {
        let args = vec!["apr", "eval", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Eval {
                dataset,
                text,
                max_tokens,
                threshold,
                ..
            }) => {
                assert_eq!(dataset, "wikitext-2");
                assert!(text.is_none());
                assert_eq!(max_tokens, 512);
                assert!((threshold - 20.0).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Eval command"),
        }
    }

    /// Test parsing 'apr flow' command with options
    #[test]
    fn test_parse_flow_command() {
        let args = vec![
            "apr",
            "flow",
            "model.apr",
            "--layer",
            "encoder.0",
            "--component",
            "encoder",
            "-v",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Flow {
                file,
                layer,
                component,
                verbose,
                json: _,
            }) => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(layer, Some("encoder.0".to_string()));
                assert_eq!(component, "encoder");
                assert!(verbose);
            }
            _ => panic!("Expected Flow command"),
        }
    }

    /// Test parsing 'apr flow' with defaults
    #[test]
    fn test_parse_flow_defaults() {
        let args = vec!["apr", "flow", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Flow {
                component,
                verbose,
                layer,
                ..
            }) => {
                assert_eq!(component, "full");
                assert!(!verbose);
                assert!(layer.is_none());
            }
            _ => panic!("Expected Flow command"),
        }
    }

    /// Test parsing 'apr hex' command with all options
    #[test]
    fn test_parse_hex_command() {
        let args = vec![
            "apr",
            "hex",
            "model.apr",
            "--tensor",
            "embed.weight",
            "--limit",
            "128",
            "--stats",
            "--list",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Hex {
                file,
                tensor,
                limit,
                stats,
                list,
                json,
                ..
            }) => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(tensor, Some("embed.weight".to_string()));
                assert_eq!(limit, 128);
                assert!(stats);
                assert!(list);
                assert!(json);
            }
            _ => panic!("Expected Hex command"),
        }
    }

    /// Test parsing 'apr hex' with defaults
    #[test]
    fn test_parse_hex_defaults() {
        let args = vec!["apr", "hex", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Hex {
                limit,
                stats,
                list,
                json,
                tensor,
                ..
            }) => {
                assert_eq!(limit, 64);
                assert!(!stats);
                assert!(!list);
                assert!(!json);
                assert!(tensor.is_none());
            }
            _ => panic!("Expected Hex command"),
        }
    }

    /// Test parsing 'apr tree' command with options
    #[test]
    fn test_parse_tree_command() {
        let args = vec![
            "apr",
            "tree",
            "model.apr",
            "--filter",
            "encoder",
            "--format",
            "mermaid",
            "--sizes",
            "--depth",
            "3",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tree {
                file,
                filter,
                format,
                sizes,
                depth,
            }) => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(filter, Some("encoder".to_string()));
                assert_eq!(format, crate::commands::tree::TreeFormat::Mermaid);
                assert!(sizes);
                assert_eq!(depth, Some(3));
            }
            _ => panic!("Expected Tree command"),
        }
    }

    /// Test parsing 'apr tree' with defaults
    #[test]
    fn test_parse_tree_defaults() {
        let args = vec!["apr", "tree", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Tree {
                format,
                sizes,
                depth,
                filter,
                ..
            }) => {
                assert_eq!(format, crate::commands::tree::TreeFormat::Ascii);
                assert!(!sizes);
                assert!(depth.is_none());
                assert!(filter.is_none());
            }
            _ => panic!("Expected Tree command"),
        }
    }

    /// Test parsing 'apr probar tensor' command with options.
    ///
    /// GH-876 Milestone 1: the existing flat `apr probar <FILE>` became
    /// `apr probar tensor <FILE>`. The tensor subcommand preserves all
    /// the original flags (--output / --format / --golden / --layer /
    /// --assert / --tolerance).
    #[test]
    fn test_parse_probar_command() {
        use TestSubcommand;
        let args = vec![
            "apr",
            "probar",
            "tensor",
            "model.apr",
            "--output",
            "/tmp/probar",
            "--format",
            "json",
            "--golden",
            "/refs/golden",
            "--layer",
            "layer.0",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Test { command }) => match command {
                TestSubcommand::Tensor {
                    file,
                    output,
                    format,
                    golden,
                    layer,
                    assert,
                    tolerance,
                } => {
                    assert_eq!(file, PathBuf::from("model.apr"));
                    assert_eq!(output, PathBuf::from("/tmp/probar"));
                    assert_eq!(format, "json");
                    assert_eq!(golden, Some(PathBuf::from("/refs/golden")));
                    assert_eq!(layer, Some("layer.0".to_string()));
                    assert!(!assert);
                    assert!((tolerance - 0.98).abs() < 0.01);
                },
            },
            _ => panic!("Expected Probar command"),
        }
    }

    /// Test parsing 'apr probar tensor' with defaults
    #[test]
    fn test_parse_probar_defaults() {
        use TestSubcommand;
        let args = vec!["apr", "probar", "tensor", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Test { command }) => match command {
                TestSubcommand::Tensor {
                    output,
                    format,
                    golden,
                    layer,
                    ..
                } => {
                    assert_eq!(output, PathBuf::from("./probar-export"));
                    assert_eq!(format, "both");
                    assert!(golden.is_none());
                    assert!(layer.is_none());
                },
            },
            _ => panic!("Expected Probar command"),
        }
    }

    /// Test parsing 'apr debug' command with all flags
    #[test]
    fn test_parse_debug_command() {
        let args = vec![
            "apr",
            "debug",
            "model.apr",
            "--drama",
            "--hex",
            "--strings",
            "--limit",
            "512",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Debug {
                file,
                action,
                drama,
                hex,
                strings,
                limit,
            } => {
                assert_eq!(file, Some(PathBuf::from("model.apr")));
                assert!(action.is_none(), "no subcommand was given");
                assert!(drama);
                assert!(hex);
                assert!(strings);
                assert_eq!(limit, 512);
            }
            _ => panic!("Expected Debug command"),
        }
    }

    /// Test parsing 'apr debug' with defaults
    #[test]
    fn test_parse_debug_defaults() {
        let args = vec!["apr", "debug", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Debug {
                drama,
                hex,
                strings,
                limit,
                ..
            } => {
                assert!(!drama);
                assert!(!hex);
                assert!(!strings);
                assert_eq!(limit, 256);
            }
            _ => panic!("Expected Debug command"),
        }
    }

    /// Test parsing 'apr tui' command with file
    #[test]
    fn test_parse_tui_command_with_file() {
        let args = vec!["apr", "tui", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tui { file } => {
                assert_eq!(file, Some(PathBuf::from("model.apr")));
            }
            _ => panic!("Expected Tui command"),
        }
    }

    /// Test parsing 'apr tui' without file (optional)
    #[test]
    fn test_parse_tui_command_no_file() {
        let args = vec!["apr", "tui"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tui { file } => {
                assert!(file.is_none());
            }
            _ => panic!("Expected Tui command"),
        }
    }

    /// Test parsing 'apr import' command with all options
    #[test]
    fn test_parse_import_command() {
        let args = vec![
            "apr",
            "import",
            "hf://openai/whisper-tiny",
            "--output",
            "whisper.apr",
            "--arch",
            "whisper",
            "--quantize",
            "int8",
            "--strict",
            "--preserve-q4k",
            "--tokenizer",
            "/path/to/tokenizer.json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Import {
                source,
                output,
                arch,
                quantize,
                strict,
                preserve_q4k,
                tokenizer,
                enforce_provenance,
                allow_no_config,
            } => {
                assert_eq!(source, "hf://openai/whisper-tiny");
                assert_eq!(output, Some(PathBuf::from("whisper.apr")));
                assert_eq!(arch, "whisper");
                assert_eq!(quantize, Some("int8".to_string()));
                assert!(strict);
                assert!(preserve_q4k);
                assert_eq!(tokenizer, Some(PathBuf::from("/path/to/tokenizer.json")));
                assert!(!enforce_provenance);
                assert!(!allow_no_config);
            }
            _ => panic!("Expected Import command"),
        }
    }

    /// Test parsing 'apr import' with defaults
    #[test]
    fn test_parse_import_defaults() {
        let args = vec!["apr", "import", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Import {
                arch,
                quantize,
                strict,
                preserve_q4k,
                output,
                tokenizer,
                ..
            } => {
                assert_eq!(arch, "auto");
                assert!(quantize.is_none());
                assert!(!strict);
                assert!(!preserve_q4k);
                assert!(output.is_none());
                assert!(tokenizer.is_none());
            }
            _ => panic!("Expected Import command"),
        }
    }
