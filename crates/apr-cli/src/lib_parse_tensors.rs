
    /// Test parsing 'apr tensors' command with all options
    #[test]
    fn test_parse_tensors_command() {
        let args = vec![
            "apr",
            "tensors",
            "model.apr",
            "--stats",
            "--filter",
            "encoder",
            "--limit",
            "20",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Tensors {
                file,
                stats,
                filter,
                limit,
                json,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(stats);
                assert_eq!(filter, Some("encoder".to_string()));
                assert_eq!(limit, 20);
                assert!(json);
            }
            _ => panic!("Expected Tensors command"),
        }
    }

    /// Test parsing 'apr explain' command with code
    #[test]
    fn test_parse_explain_with_code() {
        let args = vec!["apr", "explain", "E001"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Explain {
                code_or_file,
                file,
                tensor,
                ..
            } => {
                assert_eq!(code_or_file, Some("E001".to_string()));
                assert!(file.is_none());
                assert!(tensor.is_none());
            }
            _ => panic!("Expected Explain command"),
        }
    }

    /// Test parsing 'apr explain' with tensor and file
    #[test]
    fn test_parse_explain_with_tensor_and_file() {
        let args = vec![
            "apr",
            "explain",
            "--file",
            "model.apr",
            "--tensor",
            "embed.weight",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Explain {
                code_or_file,
                file,
                tensor,
            } => {
                assert!(code_or_file.is_none());
                assert_eq!(file, Some(PathBuf::from("model.apr")));
                assert_eq!(tensor, Some("embed.weight".to_string()));
            }
            _ => panic!("Expected Explain command"),
        }
    }

    /// Test parsing 'apr trace' command with all options
    #[test]
    fn test_parse_trace_command() {
        let args = vec![
            "apr",
            "trace",
            "model.apr",
            "--layer",
            "layer.0",
            "--reference",
            "ref.apr",
            "--json",
            "-v",
            "--payload",
            "--diff",
            "--interactive",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Trace {
                file,
                layer,
                reference,
                json,
                verbose,
                payload,
                diff,
                interactive,
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(layer, Some("layer.0".to_string()));
                assert_eq!(reference, Some(PathBuf::from("ref.apr")));
                assert!(json);
                assert!(verbose);
                assert!(payload);
                assert!(diff);
                assert!(interactive);
            }
            _ => panic!("Expected Trace command"),
        }
    }

    /// Test parsing 'apr validate' with min-score
    #[test]
    fn test_parse_validate_with_min_score() {
        let args = vec!["apr", "validate", "model.apr", "--min-score", "80"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Validate {
                min_score, strict, ..
            } => {
                assert_eq!(min_score, Some(80));
                assert!(!strict);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    /// Test parsing 'apr diff' with all options
    #[test]
    fn test_parse_diff_with_all_options() {
        let args = vec![
            "apr",
            "diff",
            "a.apr",
            "b.apr",
            "--values",
            "--filter",
            "embed",
            "--limit",
            "5",
            "--transpose-aware",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Diff {
                file1,
                file2,
                values,
                filter,
                limit,
                transpose_aware,
                json,
                ..
            } => {
                assert_eq!(file1, PathBuf::from("a.apr"));
                assert_eq!(file2, PathBuf::from("b.apr"));
                assert!(values);
                assert_eq!(filter, Some("embed".to_string()));
                assert_eq!(limit, 5);
                assert!(transpose_aware);
                assert!(json);
            }
            _ => panic!("Expected Diff command"),
        }
    }

    /// Test parsing 'apr run' with --chat flag
    #[test]
    fn test_parse_run_with_chat_flag() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "Hello world",
            "--chat",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                chat,
                prompt,
                source,
                ..
            } => {
                assert!(chat);
                assert_eq!(prompt, Some("Hello world".to_string()));
                assert_eq!(source, "model.gguf");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --trace-payload shorthand
    #[test]
    fn test_parse_run_with_trace_payload() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--trace-payload",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                trace_payload,
                trace,
                trace_level,
                ..
            } => {
                assert!(trace_payload);
                // trace itself should default to false (trace_payload is separate flag)
                assert!(!trace);
                assert_eq!(trace_level, "basic");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: Positional prompt is parsed as second argument
    #[test]
    fn test_parse_run_positional_prompt() {
        let args = vec!["apr", "run", "model.gguf", "What is 2+2?"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                source,
                positional_prompt,
                prompt,
                ..
            } => {
                assert_eq!(source, "model.gguf");
                assert_eq!(positional_prompt, Some("What is 2+2?".to_string()));
                assert_eq!(prompt, None);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: --prompt flag still works and takes precedence
    #[test]
    fn test_parse_run_flag_prompt_overrides_positional() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "positional text",
            "--prompt",
            "flag text",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                positional_prompt,
                prompt,
                ..
            } => {
                assert_eq!(positional_prompt, Some("positional text".to_string()));
                assert_eq!(prompt, Some("flag text".to_string()));
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: Positional prompt with -n short flag for max_tokens
    #[test]
    fn test_parse_run_positional_prompt_with_n_flag() {
        let args = vec!["apr", "run", "model.gguf", "What is 2+2?", "-n", "64"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                source,
                positional_prompt,
                max_tokens,
                ..
            } => {
                assert_eq!(source, "model.gguf");
                assert_eq!(positional_prompt, Some("What is 2+2?".to_string()));
                assert_eq!(max_tokens, 64);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// GH-217: No prompt provided (neither positional nor flag)
    #[test]
    fn test_parse_run_no_prompt() {
        let args = vec!["apr", "run", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                positional_prompt,
                prompt,
                ..
            } => {
                assert_eq!(positional_prompt, None);
                assert_eq!(prompt, None);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with local verbose flag
    #[test]
    fn test_parse_run_with_local_verbose() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "hi", "-v"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { verbose, .. } => {
                assert!(verbose);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with all trace options
    #[test]
    fn test_parse_run_with_full_trace() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--trace",
            "--trace-steps",
            "Tokenize,Embed,Attention",
            "--trace-verbose",
            "--trace-output",
            "/tmp/trace.json",
            "--trace-level",
            "layer",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                trace,
                trace_steps,
                trace_verbose,
                trace_output,
                trace_level,
                ..
            } => {
                assert!(trace);
                assert_eq!(
                    trace_steps,
                    Some(vec![
                        "Tokenize".to_string(),
                        "Embed".to_string(),
                        "Attention".to_string()
                    ])
                );
                assert!(trace_verbose);
                assert_eq!(trace_output, Some(PathBuf::from("/tmp/trace.json")));
                assert_eq!(trace_level, "layer");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --benchmark and --profile flags
    #[test]
    fn test_parse_run_benchmark_and_profile() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--benchmark",
            "--profile",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                benchmark, profile, ..
            } => {
                assert!(benchmark);
                assert!(profile);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --no-gpu flag
    #[test]
    fn test_parse_run_no_gpu() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "--no-gpu"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { no_gpu, .. } => {
                assert!(no_gpu);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --offline flag
    #[test]
    fn test_parse_run_offline() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "test", "--offline"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { offline, .. } => {
                assert!(offline);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr run' with --stream and --format options
    #[test]
    fn test_parse_run_stream_and_format() {
        let args = vec![
            "apr",
            "run",
            "model.gguf",
            "--prompt",
            "test",
            "--stream",
            "-f",
            "json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run { stream, format, .. } => {
                assert!(stream);
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Run command"),
        }
    }
