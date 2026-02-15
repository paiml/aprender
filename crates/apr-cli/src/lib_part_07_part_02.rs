
    /// Parse CLI args on a thread with 16 MB stack.
    /// Clap's parser for 34 subcommands exceeds the default test-thread
    /// stack in debug builds.
    fn parse_cli(args: Vec<&'static str>) -> Result<Cli, clap::error::Error> {
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || Cli::try_parse_from(args))
            .expect("spawn thread")
            .join()
            .expect("join thread")
    }

    /// Test CLI parsing with clap's debug_assert
    #[test]
    fn test_cli_parsing_valid() {
        use clap::CommandFactory;
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| Cli::command().debug_assert())
            .expect("spawn")
            .join()
            .expect("join");
    }

    /// Test parsing 'apr inspect' command
    #[test]
    fn test_parse_inspect_command() {
        let args = vec!["apr", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Inspect { file, .. } => {
                assert_eq!(file, PathBuf::from("model.apr"));
            }
            _ => panic!("Expected Inspect command"),
        }
    }

    /// Test parsing 'apr inspect' with flags
    #[test]
    fn test_parse_inspect_with_flags() {
        let args = vec!["apr", "inspect", "model.apr", "--vocab", "--json"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Inspect {
                file, vocab, json, ..
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(vocab);
                assert!(json);
            }
            _ => panic!("Expected Inspect command"),
        }
    }

    /// Test parsing 'apr serve' command
    #[test]
    fn test_parse_serve_command() {
        let args = vec!["apr", "serve", "model.apr", "--port", "3000"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve { file, port, .. } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert_eq!(port, 3000);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    /// Test parsing 'apr run' command
    #[test]
    fn test_parse_run_command() {
        let args = vec![
            "apr",
            "run",
            "hf://openai/whisper-tiny",
            "--prompt",
            "Hello",
            "--max-tokens",
            "64",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                source,
                prompt,
                max_tokens,
                ..
            } => {
                assert_eq!(source, "hf://openai/whisper-tiny");
                assert_eq!(prompt, Some("Hello".to_string()));
                assert_eq!(max_tokens, 64);
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr chat' command
    #[test]
    fn test_parse_chat_command() {
        let args = vec![
            "apr",
            "chat",
            "model.gguf",
            "--temperature",
            "0.5",
            "--top-p",
            "0.95",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Chat {
                file,
                temperature,
                top_p,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert!((temperature - 0.5).abs() < f32::EPSILON);
                assert!((top_p - 0.95).abs() < f32::EPSILON);
            }
            _ => panic!("Expected Chat command"),
        }
    }

    /// Test parsing 'apr validate' command with quality flag
    #[test]
    fn test_parse_validate_with_quality() {
        let args = vec!["apr", "validate", "model.apr", "--quality", "--strict"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Validate {
                file,
                quality,
                strict,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(quality);
                assert!(strict);
            }
            _ => panic!("Expected Validate command"),
        }
    }

    /// Test parsing 'apr diff' command
    #[test]
    fn test_parse_diff_command() {
        let args = vec!["apr", "diff", "model1.apr", "model2.apr", "--weights"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Diff {
                file1,
                file2,
                weights,
                ..
            } => {
                assert_eq!(file1, PathBuf::from("model1.apr"));
                assert_eq!(file2, PathBuf::from("model2.apr"));
                assert!(weights);
            }
            _ => panic!("Expected Diff command"),
        }
    }

    /// Test parsing 'apr bench' command
    #[test]
    fn test_parse_bench_command() {
        let args = vec![
            "apr",
            "bench",
            "model.gguf",
            "--warmup",
            "5",
            "--iterations",
            "10",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Bench {
                file,
                warmup,
                iterations,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert_eq!(warmup, 5);
                assert_eq!(iterations, 10);
            }
            _ => panic!("Expected Bench command"),
        }
    }

    /// Test parsing 'apr cbtop' command with CI flags
    #[test]
    fn test_parse_cbtop_ci_mode() {
        let args = vec![
            "apr",
            "cbtop",
            "--headless",
            "--ci",
            "--throughput",
            "100.0",
            "--brick-score",
            "90",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Cbtop {
                headless,
                ci,
                throughput,
                brick_score,
                ..
            } => {
                assert!(headless);
                assert!(ci);
                assert_eq!(throughput, Some(100.0));
                assert_eq!(brick_score, Some(90));
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// Test parsing 'apr qa' command
    #[test]
    fn test_parse_qa_command() {
        let args = vec![
            "apr",
            "qa",
            "model.gguf",
            "--assert-tps",
            "50.0",
            "--skip-ollama",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Qa {
                file,
                assert_tps,
                skip_ollama,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert_eq!(assert_tps, Some(50.0));
                assert!(skip_ollama);
            }
            _ => panic!("Expected Qa command"),
        }
    }

    /// Test global --verbose flag
    #[test]
    fn test_global_verbose_flag() {
        let args = vec!["apr", "--verbose", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.verbose);
    }

    /// Test global --json flag
    #[test]
    fn test_global_json_flag() {
        let args = vec!["apr", "--json", "inspect", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.json);
    }

    /// Test parsing 'apr list' command (alias 'ls')
    #[test]
    fn test_parse_list_command() {
        let args = vec!["apr", "list"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(matches!(*cli.command, Commands::List));
    }

    /// Test parsing 'apr ls' alias
    #[test]
    fn test_parse_ls_alias() {
        let args = vec!["apr", "ls"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(matches!(*cli.command, Commands::List));
    }

    /// Test parsing 'apr rm' command (alias 'remove')
    #[test]
    fn test_parse_rm_command() {
        let args = vec!["apr", "rm", "model-name"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rm { model_ref } => {
                assert_eq!(model_ref, "model-name");
            }
            _ => panic!("Expected Rm command"),
        }
    }

    /// Test invalid command fails parsing
    #[test]
    fn test_invalid_command() {
        let args = vec!["apr", "invalid-command"];
        let result = parse_cli(args);
        assert!(result.is_err());
    }

    /// Test missing required argument fails
    #[test]
    fn test_missing_required_arg() {
        let args = vec!["apr", "inspect"]; // Missing FILE
        let result = parse_cli(args);
        assert!(result.is_err());
    }

    /// Test parsing 'apr merge' with multiple files and weights
    #[test]
    fn test_parse_merge_command() {
        let args = vec![
            "apr",
            "merge",
            "model1.apr",
            "model2.apr",
            "--strategy",
            "weighted",
            "--weights",
            "0.7,0.3",
            "-o",
            "merged.apr",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Merge {
                files,
                strategy,
                output,
                weights,
                ..
            } => {
                assert_eq!(files.len(), 2);
                assert_eq!(strategy, "weighted");
                assert_eq!(output, PathBuf::from("merged.apr"));
                assert_eq!(weights, Some(vec![0.7, 0.3]));
            }
            _ => panic!("Expected Merge command"),
        }
    }

    /// Test parsing 'apr showcase' command
    #[test]
    fn test_parse_showcase_command() {
        let args = vec![
            "apr",
            "showcase",
            "--tier",
            "medium",
            "--gpu",
            "--auto-verify",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Showcase {
                tier,
                gpu,
                auto_verify,
                ..
            } => {
                assert_eq!(tier, "medium");
                assert!(gpu);
                assert!(auto_verify);
            }
            _ => panic!("Expected Showcase command"),
        }
    }

    /// Test parsing 'apr profile' with all options
    #[test]
    fn test_parse_profile_command() {
        let args = vec![
            "apr",
            "profile",
            "model.apr",
            "--granular",
            "--detect-naive",
            "--fail-on-naive",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Profile {
                file,
                granular,
                detect_naive,
                fail_on_naive,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.apr"));
                assert!(granular);
                assert!(detect_naive);
                assert!(fail_on_naive);
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test parsing 'apr profile' with CI assertions (PMAT-192, GH-180)
    #[test]
    fn test_parse_profile_ci_mode() {
        let args = vec![
            "apr",
            "profile",
            "model.gguf",
            "--ci",
            "--assert-throughput",
            "100",
            "--assert-p99",
            "50",
            "--format",
            "json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Profile {
                file,
                ci,
                assert_throughput,
                assert_p99,
                format,
                ..
            } => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert!(ci);
                assert_eq!(assert_throughput, Some(100.0));
                assert_eq!(assert_p99, Some(50.0));
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Profile command"),
        }
    }
