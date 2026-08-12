
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

    /// `apr code -p "hi" --model X` must RUN model X.
    ///
    /// With `trailing_var_arg` the option landed in the prompt vector and the
    /// run silently fell back to auto-discovery — a different model than the
    /// one named, with no warning.
    #[test]
    fn test_parse_code_model_after_prompt_is_honoured() {
        let args = vec!["apr", "code", "-p", "hi", "--model", "/tmp/named.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Code {
                model,
                prompt,
                print,
                ..
            } => {
                assert!(print, "-p must still set print");
                assert_eq!(
                    model,
                    Some(PathBuf::from("/tmp/named.gguf")),
                    "--model written after the prompt must be honoured, not swallowed"
                );
                assert_eq!(
                    prompt,
                    vec!["hi".to_string()],
                    "the prompt must not absorb the option"
                );
            }
            _ => panic!("Expected Code command"),
        }
    }

    /// A misspelled option after the prompt must be a parse error, not silence.
    #[test]
    fn test_parse_code_unknown_option_after_prompt_is_rejected() {
        let args = vec!["apr", "code", "-p", "hi", "--totally-bogus-flag-xyz"];
        let err = parse_cli(args)
            .expect_err("an unknown option after the prompt must fail to parse");
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::UnknownArgument,
            "expected UnknownArgument, got {:?}",
            err.kind()
        );
    }

    /// Multi-word prompts must still collect into the positional vector.
    #[test]
    fn test_parse_code_multiword_prompt_still_collects() {
        let args = vec!["apr", "code", "-p", "fix", "the", "auth", "bug"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Code { prompt, .. } => {
                assert_eq!(prompt, vec!["fix", "the", "auth", "bug"]);
            }
            _ => panic!("Expected Code command"),
        }
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

    /// Test parsing 'apr serve run' command
    #[test]
    fn test_parse_serve_command() {
        let args = vec!["apr", "serve", "run", "model.apr", "--port", "3000"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                command: ServeCommands::Run { ref file, port, .. },
            } => {
                assert_eq!(*file, PathBuf::from("model.apr"));
                assert_eq!(port, 3000);
            }
            _ => panic!("Expected Serve Run command"),
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

    /// Test parsing 'apr run --stream' flag (NDJSON token stream).
    #[test]
    fn test_parse_run_stream_flag() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "Hi", "--stream"];
        let cli = parse_cli(args).expect("Failed to parse --stream");
        match *cli.command {
            Commands::Run {
                source,
                stream,
                ..
            } => {
                assert_eq!(source, "model.gguf");
                assert!(stream, "--stream must set stream=true");
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Default for --stream is false when omitted.
    #[test]
    fn test_parse_run_stream_default_false() {
        let args = vec!["apr", "run", "model.gguf", "--prompt", "Hi"];
        let cli = parse_cli(args).expect("Failed to parse default");
        match *cli.command {
            Commands::Run { stream, .. } => assert!(!stream, "--stream defaults to false"),
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
            Commands::Extended(ExtendedCommands::Chat {
                file,
                temperature,
                top_p,
                ..
            }) => {
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
            Commands::Extended(ExtendedCommands::Bench {
                file,
                warmup,
                iterations,
                ..
            }) => {
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
            Commands::Extended(ExtendedCommands::Cbtop {
                headless,
                ci,
                throughput,
                brick_score,
                ..
            }) => {
                assert!(headless);
                assert!(ci);
                assert_eq!(throughput, Some(100.0));
                assert_eq!(brick_score, Some(90));
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// #2397 finding 1: `--iterations 0` must be rejected where it is typed.
    /// Appending it to any `cbtop --ci` invocation used to turn the gate green
    /// because a brick with zero samples scores a perfect 100/A.
    #[test]
    fn test_parse_cbtop_rejects_zero_iterations() {
        let args = vec![
            "apr",
            "cbtop",
            "--headless",
            "--simulated",
            "--ci",
            "--iterations",
            "0",
        ];
        assert!(
            parse_cli(args).is_err(),
            "cbtop accepted --iterations 0 at parse time"
        );

        // One iteration is the smallest honest run and must still parse.
        let ok = vec!["apr", "cbtop", "--headless", "--iterations", "1"];
        let cli = parse_cli(ok).expect("--iterations 1 should parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Cbtop { iterations, .. }) => {
                assert_eq!(iterations, 1);
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// #2397 finding 4: `--json` and `--output` document "requires --headless",
    /// so the parser must enforce it. Without the constraint the flag was
    /// silently dropped and cbtop entered the interactive TUI instead.
    #[test]
    fn test_parse_cbtop_json_requires_headless() {
        assert!(
            parse_cli(vec!["apr", "cbtop", "--json"]).is_err(),
            "cbtop --json was accepted without --headless"
        );
        assert!(
            parse_cli(vec!["apr", "cbtop", "--output", "r.json"]).is_err(),
            "cbtop --output was accepted without --headless"
        );

        let cli = parse_cli(vec!["apr", "cbtop", "--headless", "--json"])
            .expect("--json --headless should parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Cbtop { headless, json, .. }) => {
                assert!(headless);
                assert!(json);
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
            Commands::Extended(ExtendedCommands::Qa {
                file,
                assert_tps,
                skip_ollama,
                ..
            }) => {
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
                assert_eq!(output, Some(PathBuf::from("merged.apr")));
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
            Commands::Extended(ExtendedCommands::Tools(ToolCommands::Showcase {
                tier,
                gpu,
                auto_verify,
                ..
            })) => {
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
            Commands::Extended(ExtendedCommands::Profile {
                file,
                granular,
                detect_naive,
                fail_on_naive,
                ..
            }) => {
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
            Commands::Extended(ExtendedCommands::Profile {
                file,
                ci,
                assert_throughput,
                assert_p99,
                format,
                ..
            }) => {
                assert_eq!(file, PathBuf::from("model.gguf"));
                assert!(ci);
                assert_eq!(assert_throughput, Some(100.0));
                assert_eq!(assert_p99, Some(50.0));
                assert_eq!(format, "json");
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Dogfood 0.63.0 #2377 finding 10: `apr typical-p-lint --help` printed
    /// its only required flag with an empty description, so the (non-obvious,
    /// otherwise undocumented) observation schema had no route to the user.
    /// Every other command in the 16-command lint family documents its flag.
    ///
    /// A family-wide ratchet, not a one-line fix: a new `*-lint` command that
    /// forgets its flag doc turns this red.
    #[test]
    fn every_lint_command_documents_its_flags() {
        use clap::CommandFactory;
        let offenders = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let mut root = Cli::command();
                root.build();
                let mut bad: Vec<String> = Vec::new();
                for sub in root.get_subcommands() {
                    if !sub.get_name().ends_with("-lint") {
                        continue;
                    }
                    for arg in sub.get_arguments() {
                        if arg.get_id() == "help" || arg.get_id() == "version" {
                            continue;
                        }
                        let documented = arg
                            .get_help()
                            .is_some_and(|h| !h.to_string().trim().is_empty());
                        if !documented {
                            bad.push(format!("{} --{}", sub.get_name(), arg.get_id()));
                        }
                    }
                }
                bad
            })
            .expect("spawn")
            .join()
            .expect("join");
        assert!(
            offenders.is_empty(),
            "lint commands with an undocumented flag: {offenders:?}"
        );
    }

    /// FALSIFIER (#2394 finding 15): `apr tree --format <garbage>` must be
    /// rejected, not silently rendered as ascii.
    ///
    /// `apr tree model.apr --format bogusvalue` printed the ascii tree and
    /// exited 0 — no error, no warning. The dispatcher parsed the string with
    /// `.unwrap_or(TreeFormat::Ascii)`, so `--format josn` in a pipeline
    /// produced a tree where JSON was expected and every downstream check saw
    /// success. `TreeFormat::from_str` had always returned
    /// `Err("Unknown format: …")`; the error was thrown away one call up.
    #[test]
    fn test_tree_rejects_an_unknown_format_value() {
        let err = parse_cli(vec!["apr", "tree", "m.apr", "--format", "bogusvalue"])
            .expect_err("--format bogusvalue must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("bogusvalue"),
            "the error must quote the value it rejected, got: {msg}"
        );
    }

    /// ...while every documented spelling still parses, so the guard rejects
    /// typos rather than rejecting everything.
    #[test]
    fn test_tree_accepts_every_documented_format() {
        for value in ["ascii", "text", "dot", "graphviz", "mermaid", "md", "json"] {
            let args: Vec<&'static str> = vec!["apr", "tree", "m.apr", "--format", value];
            assert!(
                parse_cli(args).is_ok(),
                "--format {value} is documented and must parse"
            );
        }
        // And the default (no --format at all) still works.
        assert!(parse_cli(vec!["apr", "tree", "m.apr"]).is_ok());
    }
