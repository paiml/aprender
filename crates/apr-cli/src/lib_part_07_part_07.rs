
    /// Test parsing 'apr run' with --input and --language for ASR
    #[test]
    fn test_parse_run_asr_options() {
        let args = vec![
            "apr",
            "run",
            "hf://openai/whisper-tiny",
            "--input",
            "audio.wav",
            "--language",
            "en",
            "--task",
            "transcribe",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Run {
                input,
                language,
                task,
                ..
            } => {
                assert_eq!(input, Some(PathBuf::from("audio.wav")));
                assert_eq!(language, Some("en".to_string()));
                assert_eq!(task, Some("transcribe".to_string()));
            }
            _ => panic!("Expected Run command"),
        }
    }

    /// Test parsing 'apr remove' alias for 'rm'
    #[test]
    fn test_parse_remove_alias() {
        let args = vec!["apr", "remove", "my-model"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Rm { model_ref } => {
                assert_eq!(model_ref, "my-model");
            }
            _ => panic!("Expected Rm command"),
        }
    }

    /// Test parsing 'apr qa' with all skip flags
    #[test]
    fn test_parse_qa_all_skip_flags() {
        let args = vec![
            "apr",
            "qa",
            "model.gguf",
            "--skip-golden",
            "--skip-throughput",
            "--skip-ollama",
            "--skip-gpu-speedup",
            "--skip-contract",
            "--skip-format-parity",
            "--safetensors-path",
            "model.safetensors",
            "--iterations",
            "20",
            "--warmup",
            "5",
            "--max-tokens",
            "64",
            "--json",
            "-v",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Qa {
                skip_golden,
                skip_throughput,
                skip_ollama,
                skip_gpu_speedup,
                skip_contract,
                skip_format_parity,
                safetensors_path,
                iterations,
                warmup,
                max_tokens,
                json,
                verbose,
                ..
            }) => {
                assert!(skip_golden);
                assert!(skip_throughput);
                assert!(skip_ollama);
                assert!(skip_gpu_speedup);
                assert!(skip_contract);
                assert!(skip_format_parity);
                assert_eq!(safetensors_path, Some(PathBuf::from("model.safetensors")));
                assert_eq!(iterations, 20);
                assert_eq!(warmup, 5);
                assert_eq!(max_tokens, 64);
                assert!(json);
                assert!(verbose);
            }
            _ => panic!("Expected Qa command"),
        }
    }

    /// Test parsing 'apr serve' with all options
    #[test]
    fn test_parse_serve_all_options() {
        let args = vec![
            "apr",
            "serve",
            "model.apr",
            "--port",
            "9090",
            "--host",
            "0.0.0.0",
            "--no-cors",
            "--no-metrics",
            "--no-gpu",
            "--batch",
            "--trace",
            "--trace-level",
            "layer",
            "--profile",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                port,
                host,
                no_cors,
                no_metrics,
                no_gpu,
                batch,
                trace,
                trace_level,
                profile,
                ..
            } => {
                assert_eq!(port, 9090);
                assert_eq!(host, "0.0.0.0");
                assert!(no_cors);
                assert!(no_metrics);
                assert!(no_gpu);
                assert!(batch);
                assert!(trace);
                assert_eq!(trace_level, "layer");
                assert!(profile);
            }
            _ => panic!("Expected Serve command"),
        }
    }

    /// Test parsing 'apr bench' with all options
    #[test]
    fn test_parse_bench_all_options() {
        let args = vec![
            "apr",
            "bench",
            "model.gguf",
            "--warmup",
            "10",
            "--iterations",
            "20",
            "--max-tokens",
            "64",
            "--prompt",
            "The quick brown fox",
            "--fast",
            "--brick",
            "attention",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Bench {
                warmup,
                iterations,
                max_tokens,
                prompt,
                fast,
                brick,
                ..
            }) => {
                assert_eq!(warmup, 10);
                assert_eq!(iterations, 20);
                assert_eq!(max_tokens, 64);
                assert_eq!(prompt, Some("The quick brown fox".to_string()));
                assert!(fast);
                assert_eq!(brick, Some("attention".to_string()));
            }
            _ => panic!("Expected Bench command"),
        }
    }

    /// Test parsing 'apr cbtop' with speculative decoding flags
    #[test]
    fn test_parse_cbtop_speculative() {
        let args = vec![
            "apr",
            "cbtop",
            "--model-path",
            "model.gguf",
            "--speculative",
            "--speculation-k",
            "8",
            "--draft-model",
            "draft.gguf",
            "--concurrent",
            "4",
            "--simulated",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Cbtop {
                model_path,
                speculative,
                speculation_k,
                draft_model,
                concurrent,
                simulated,
                ..
            }) => {
                assert_eq!(model_path, Some(PathBuf::from("model.gguf")));
                assert!(speculative);
                assert_eq!(speculation_k, 8);
                assert_eq!(draft_model, Some(PathBuf::from("draft.gguf")));
                assert_eq!(concurrent, 4);
                assert!(simulated);
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// Test parsing 'apr profile' with energy and perf-grade flags
    #[test]
    fn test_parse_profile_energy_perf() {
        let args = vec![
            "apr",
            "profile",
            "model.apr",
            "--energy",
            "--perf-grade",
            "--callgraph",
            "--compare-hf",
            "openai/whisper-tiny",
            "--output",
            "/tmp/flame.svg",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Profile {
                energy,
                perf_grade,
                callgraph,
                compare_hf,
                output,
                ..
            }) => {
                assert!(energy);
                assert!(perf_grade);
                assert!(callgraph);
                assert_eq!(compare_hf, Some("openai/whisper-tiny".to_string()));
                assert_eq!(output, Some(PathBuf::from("/tmp/flame.svg")));
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test parsing 'apr chat' with all trace options
    #[test]
    fn test_parse_chat_with_trace() {
        let args = vec![
            "apr",
            "chat",
            "model.gguf",
            "--system",
            "You are a helpful assistant.",
            "--inspect",
            "--trace",
            "--trace-steps",
            "Tokenize,Decode",
            "--trace-verbose",
            "--trace-output",
            "/tmp/chat-trace.json",
            "--trace-level",
            "payload",
            "--profile",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Chat {
                system,
                inspect,
                trace,
                trace_steps,
                trace_verbose,
                trace_output,
                trace_level,
                profile,
                ..
            }) => {
                assert_eq!(system, Some("You are a helpful assistant.".to_string()));
                assert!(inspect);
                assert!(trace);
                assert_eq!(
                    trace_steps,
                    Some(vec!["Tokenize".to_string(), "Decode".to_string()])
                );
                assert!(trace_verbose);
                assert_eq!(trace_output, Some(PathBuf::from("/tmp/chat-trace.json")));
                assert_eq!(trace_level, "payload");
                assert!(profile);
            }
            _ => panic!("Expected Chat command"),
        }
    }

    /// Test parsing 'apr showcase' with step and all options
    #[test]
    fn test_parse_showcase_with_step() {
        let args = vec![
            "apr", "showcase", "--step", "bench", "--tier", "tiny", "--zram", "--runs", "50",
            "--json", "-v", "-q",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Showcase {
                step,
                tier,
                zram,
                runs,
                json,
                verbose,
                quiet,
                ..
            }) => {
                assert_eq!(step, Some("bench".to_string()));
                assert_eq!(tier, "tiny");
                assert!(zram);
                assert_eq!(runs, 50);
                assert!(json);
                assert!(verbose);
                assert!(quiet);
            }
            _ => panic!("Expected Showcase command"),
        }
    }

    /// Test parsing rosetta compare-inference subcommand
    #[test]
    fn test_parse_rosetta_compare_inference() {
        let args = vec![
            "apr",
            "rosetta",
            "compare-inference",
            "model_a.gguf",
            "model_b.apr",
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "10",
            "--temperature",
            "0.5",
            "--tolerance",
            "0.05",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Rosetta { action }) => match action {
                RosettaCommands::CompareInference {
                    model_a,
                    model_b,
                    prompt,
                    max_tokens,
                    temperature,
                    tolerance,
                    json,
                } => {
                    assert_eq!(model_a, PathBuf::from("model_a.gguf"));
                    assert_eq!(model_b, PathBuf::from("model_b.apr"));
                    assert_eq!(prompt, "What is 2+2?");
                    assert_eq!(max_tokens, 10);
                    assert!((temperature - 0.5).abs() < f32::EPSILON);
                    assert!((tolerance - 0.05).abs() < f32::EPSILON);
                    assert!(json);
                }
                _ => panic!("Expected CompareInference subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }

    /// Test parsing rosetta diff-tensors subcommand
    #[test]
    fn test_parse_rosetta_diff_tensors() {
        let args = vec![
            "apr",
            "rosetta",
            "diff-tensors",
            "ref.gguf",
            "test.apr",
            "--mismatches-only",
            "--show-values",
            "5",
            "--filter",
            "lm_head",
            "--json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Rosetta { action }) => match action {
                RosettaCommands::DiffTensors {
                    model_a,
                    model_b,
                    mismatches_only,
                    show_values,
                    filter,
                    json,
                } => {
                    assert_eq!(model_a, PathBuf::from("ref.gguf"));
                    assert_eq!(model_b, PathBuf::from("test.apr"));
                    assert!(mismatches_only);
                    assert_eq!(show_values, 5);
                    assert_eq!(filter, Some("lm_head".to_string()));
                    assert!(json);
                }
                _ => panic!("Expected DiffTensors subcommand"),
            },
            _ => panic!("Expected Rosetta command"),
        }
    }
