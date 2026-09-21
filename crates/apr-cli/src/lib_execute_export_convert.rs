
    /// Test execute_command: Export with non-existent file returns error
    #[test]
    fn test_execute_export_file_not_found() {
        let cli = make_cli(Commands::Export {
            file: Some(PathBuf::from("/tmp/nonexistent_model_export_test.apr")),
            format: "safetensors".to_string(),
            output: Some(PathBuf::from("/tmp/out.safetensors")),
            quantize: None,
            list_formats: false,
            batch: None,
            json: false,
            plan: false,
                force: true,
            });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Export should fail with non-existent file");
    }

    /// Test execute_command: Convert with non-existent file returns error
    #[test]
    fn test_execute_convert_file_not_found() {
        let cli = make_cli(Commands::Convert {
            file: PathBuf::from("/tmp/nonexistent_model_convert_test.apr"),
            quantize: None,
            compress: None,
            output: PathBuf::from("/tmp/out.apr"),
            force: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Convert should fail with non-existent file"
        );
    }

    /// Test execute_command: Hex with non-existent file returns error
    #[test]
    fn test_execute_hex_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Hex {
            file: PathBuf::from("/tmp/nonexistent_model_hex_test.apr"),
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
            offset: String::new(),
            width: 16,
            slice: None,
        }));
        let result = execute_command(&cli);
        assert!(result.is_err(), "Hex should fail with non-existent file");
    }

    /// Test execute_command: Tree with non-existent file returns error
    #[test]
    fn test_execute_tree_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Tree {
            file: PathBuf::from("/tmp/nonexistent_model_tree_test.apr"),
            filter: None,
            format: crate::commands::tree::TreeFormat::Ascii,
            sizes: false,
            depth: None,
        }));
        let result = execute_command(&cli);
        assert!(result.is_err(), "Tree should fail with non-existent file");
    }

    /// Test execute_command: Flow with non-existent file returns error
    #[test]
    fn test_execute_flow_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Flow {
            file: PathBuf::from("/tmp/nonexistent_model_flow_test.apr"),
            layer: None,
            component: "full".to_string(),
            verbose: false,
            json: false,
        }));
        let result = execute_command(&cli);
        assert!(result.is_err(), "Flow should fail with non-existent file");
    }

    /// Test execute_command: Probar with non-existent file returns error
    /// (via the GH-876 tensor subcommand)
    #[test]
    fn test_execute_probar_file_not_found() {
        use TestSubcommand;
        let cli = make_cli(Commands::Extended(ExtendedCommands::Test {
            command: TestSubcommand::Tensor {
                file: PathBuf::from("/tmp/nonexistent_model_probar_test.apr"),
                output: PathBuf::from("/tmp/probar-out"),
                format: "both".to_string(),
                golden: None,
                layer: None,
                assert: false,
                tolerance: 0.98,
            },
        }));
        let result = execute_command(&cli);
        assert!(result.is_err(), "Probar should fail with non-existent file");
    }

    /// Test execute_command: Check with non-existent file returns error
    #[test]
    fn test_execute_check_file_not_found() {
        let cli = make_cli(Commands::Check {
            file: PathBuf::from("/tmp/nonexistent_model_check_test.apr"),
            no_gpu: true,
            json: false,
        });
        let result = execute_command(&cli);
        assert!(result.is_err(), "Check should fail with non-existent file");
    }

    /// Test execute_command: List succeeds (no file needed)
    #[test]
    fn test_execute_list_succeeds() {
        let cli = make_cli(Commands::List);
        // List should succeed even if cache is empty
        let result = execute_command(&cli);
        assert!(result.is_ok(), "List should succeed without arguments");
    }

    /// Test execute_command: Explain without args returns validation error
    #[test]
    fn test_execute_explain_no_args() {
        let cli = make_cli(Commands::Explain {
            code_or_file: None,
            file: None,
            tensor: None,
            kernel: false,
            json: false,
            verbose: false,
            proof_status: false,
        });
        // Explain with no args returns error (shows usage guidance)
        let result = execute_command(&cli);
        assert!(result.is_err(), "Explain with no args should return error");
    }

    /// Test execute_command: Explain with code succeeds
    #[test]
    fn test_execute_explain_with_code() {
        let cli = make_cli(Commands::Explain {
            code_or_file: Some("E001".to_string()),
            file: None,
            tensor: None,
            kernel: false,
            json: false,
            verbose: false,
            proof_status: false,
        });
        let result = execute_command(&cli);
        // Should succeed even for unknown error codes (it prints "unknown error code")
        assert!(result.is_ok(), "Explain with error code should succeed");
    }

    /// Test execute_command: Tune --plan without file succeeds
    #[test]
    fn test_execute_tune_plan_no_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Tune {
            file: None,
            method: "auto".to_string(),
            rank: None,
            vram: 16.0,
            plan: true,
            model: Some("7B".to_string()),
            freeze_base: false,
            train_data: None,
            json: false,
            task: None,
            budget: 10,
            strategy: "tpe".to_string(),
            scheduler: "asha".to_string(),
            scout: false,
            data: None,
            num_classes: 5,
            model_size: None,
            from_scout: None,
            max_epochs: 20,
            time_limit: None,
        }));
        let result = execute_command(&cli);
        // Tune with --plan and --model should succeed without a file
        assert!(
            result.is_ok(),
            "Tune --plan --model 7B should succeed without file"
        );
    }

    /// Test execute_command: Qa with non-existent file and all skips still succeeds
    /// because QA gates are individually skipped. With no gates enabled, it just
    /// prints summary and returns Ok.
    #[test]
    fn test_execute_qa_all_skips_succeeds() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Qa {
            file: PathBuf::from("/tmp/nonexistent_model_qa_test.gguf"),
            assert_tps: None,
            assert_speedup: None,
            assert_gpu_speedup: None,
            skip_golden: true,
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            skip_ptx_parity: true,
            safetensors_path: None,
            iterations: 1,
            warmup: 0,
            max_tokens: 1,
            json: false,
            verbose: false,
            min_executed: None,
            previous_report: None,
            regression_threshold: None,
            skip_gpu_state: false,
            skip_metadata: true,
            skip_capability: true,
            assert_classifier_head: false,
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_ok(),
            "Qa with all gates skipped should succeed even with non-existent file"
        );
    }

    /// Test execute_command: Qa with non-existent file and gates enabled returns error
    #[test]
    fn test_execute_qa_with_gates_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Qa {
            file: PathBuf::from("/tmp/nonexistent_model_qa_gates_test.gguf"),
            assert_tps: None,
            assert_speedup: None,
            assert_gpu_speedup: None,
            skip_golden: false, // Gate enabled
            skip_throughput: true,
            skip_ollama: true,
            skip_gpu_speedup: true,
            skip_contract: true,
            skip_format_parity: true,
            skip_ptx_parity: true,
            safetensors_path: None,
            iterations: 1,
            warmup: 0,
            max_tokens: 1,
            json: false,
            verbose: false,
            min_executed: None,
            previous_report: None,
            regression_threshold: None,
            skip_gpu_state: false,
            skip_metadata: true,
            skip_capability: true,
            assert_classifier_head: false,
        }));
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Qa with golden gate enabled should fail with non-existent file"
        );
    }

    /// Test execute_command: Import with invalid source returns error
    #[test]
    fn test_execute_import_invalid_source() {
        let cli = make_cli(Commands::Import {
            source: "/tmp/nonexistent_model_import_test.gguf".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
            allow_no_config: false,
        });
        let result = execute_command(&cli);
        assert!(
            result.is_err(),
            "Import should fail with non-existent source file"
        );
    }

    // =========================================================================
    // --chat: the prompt reaches realizar RAW; the flag carries the intent (#3672)
    // =========================================================================
    //
    // These replace three tests that re-implemented the old ChatML pre-wrap inline and
    // asserted on their own copy: they could not fail, and they pinned the double
    // template as the contract.

    /// An instruct source auto-enables the template, and the prompt is NOT pre-wrapped:
    /// realizar applies the model's own template exactly once.
    #[test]
    fn test_3672_instruct_source_sets_the_flag_and_leaves_the_prompt_raw() {
        let prompt = "What is the meaning of life?".to_string();
        let (run_prompt, chat) =
            run_prompt_and_chat(Some(&prompt), None, "Qwen2.5-0.5B-Instruct-f16.gguf", false);
        assert_eq!(run_prompt.as_deref(), Some("What is the meaning of life?"));
        assert!(chat, "an instruct source must ask realizar for the chat template");
        assert!(
            !run_prompt.unwrap_or_default().contains("<|im_start|>"),
            "a pre-wrapped prompt is templated a second time by realizar (#3672)"
        );
    }

    /// `--chat` on a source whose name says nothing still asks for the template.
    #[test]
    fn test_3672_explicit_chat_flag_on_a_plain_source() {
        let prompt = "Hello".to_string();
        let (run_prompt, chat) = run_prompt_and_chat(Some(&prompt), None, "model.gguf", true);
        assert_eq!(run_prompt.as_deref(), Some("Hello"));
        assert!(chat);
    }

    /// A base model with no `--chat` gets neither a wrap nor the flag.
    #[test]
    fn test_3672_base_source_without_chat_is_untouched() {
        let prompt = "Hello world".to_string();
        let (run_prompt, chat) = run_prompt_and_chat(Some(&prompt), None, "smollm-base.gguf", false);
        assert_eq!(run_prompt.as_deref(), Some("Hello world"));
        assert!(!chat);
    }

    /// No prompt: nothing to template; `-p` wins over the positional prompt.
    #[test]
    fn test_3672_prompt_precedence_and_absence() {
        let (none, chat) = run_prompt_and_chat(None, None, "x-instruct.gguf", false);
        assert!(none.is_none() && !chat, "the name heuristic needs a prompt to template");
        let (p, pos) = ("from -p".to_string(), "positional".to_string());
        let (got, _) = run_prompt_and_chat(Some(&p), Some(&pos), "m.gguf", false);
        assert_eq!(got.as_deref(), Some("from -p"));
    }

    /// #3743: the three spellings of a chat prompt reach realizar identically.
    ///
    /// `--prompt P --chat`, positional `P --chat` and `-i file --chat` (same text) must
    /// build the same realizar config: the raw text, and `force_chat_template` set. At
    /// 0.69.0 the first two were pre-wrapped in ChatML and the third was not, so realizar
    /// templated them twice (24 -> 56 prompt tokens on Qwen3.5-0.8B) and the model read
    /// zero-width-escaped special tokens. The wrap-restored mutant turns this RED.
    #[cfg(feature = "inference")]
    #[test]
    fn test_3743_three_prompt_shapes_build_the_same_realizar_config() {
        const P: &str = "What is 2+2? Answer with one number.";
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("prompt.txt");
        std::fs::write(&file, P).expect("write prompt");
        let file_arg: &'static str = Box::leak(file.to_string_lossy().into_owned().into_boxed_str());

        let shapes: [(&str, Vec<&'static str>); 3] = [
            ("--prompt P --chat", vec!["apr", "run", "m.gguf", "--prompt", P, "--chat"]),
            ("positional P --chat", vec!["apr", "run", "m.gguf", P, "--chat"]),
            ("-i file --chat", vec!["apr", "run", "m.gguf", "-i", file_arg, "--chat"]),
        ];
        let built: Vec<(String, Option<String>, bool)> = shapes
            .iter()
            .map(|(name, argv)| {
                let argv = argv.clone();
                // clap's recursive `Commands` parse overflows the 2 MiB test stack in debug.
                let (source, prompt, positional, input, chat) = std::thread::Builder::new()
                    .stack_size(16 * 1024 * 1024)
                    .spawn(move || {
                        use clap::Parser;
                        match *crate::Cli::try_parse_from(argv).expect("parse").command {
                            Commands::Run {
                                source, prompt, positional_prompt, input, chat, ..
                            } => (source, prompt, positional_prompt, input, chat),
                            _ => panic!("expected `run`"),
                        }
                    })
                    .expect("spawn")
                    .join()
                    .expect("join");
                // Exactly what dispatch_run does with those fields.
                let (run_prompt, chat_template) =
                    run_prompt_and_chat(prompt.as_ref(), positional.as_ref(), &source, chat);
                let options = crate::commands::run::RunOptions {
                    prompt: run_prompt,
                    chat_template,
                    ..crate::commands::run::RunOptions::default()
                };
                let config = crate::commands::run::realizar_config(
                    std::path::Path::new(&source),
                    input.as_ref(),
                    &options,
                )
                .expect("config");
                ((*name).to_string(), config.prompt, config.force_chat_template)
            })
            .collect();

        for (name, prompt, force) in &built {
            assert_eq!(prompt.as_deref(), Some(P), "{name}: realizar must get the raw text");
            assert!(*force, "{name}: --chat must reach realizar as force_chat_template");
        }
    }

    // =========================================================================
    // --trace-payload shorthand logic
    // =========================================================================

    /// Test trace-payload shorthand enables trace and sets level to payload
    #[test]
    fn test_trace_payload_shorthand_logic() {
        let trace = false;
        let trace_payload = true;
        let trace_level = "basic".to_string();

        let effective_trace = trace || trace_payload;
        let effective_trace_level = if trace_payload {
            "payload"
        } else {
            trace_level.as_str()
        };

        assert!(effective_trace);
        assert_eq!(effective_trace_level, "payload");
    }

    /// Test that without --trace-payload, trace settings are preserved
    #[test]
    fn test_no_trace_payload_preserves_settings() {
        let trace = true;
        let trace_payload = false;
        let trace_level = "layer".to_string();

        let effective_trace = trace || trace_payload;
        let effective_trace_level = if trace_payload {
            "payload"
        } else {
            trace_level.as_str()
        };

        assert!(effective_trace);
        assert_eq!(effective_trace_level, "layer");
    }

    /// Test that neither trace nor trace_payload results in no trace
    #[test]
    fn test_no_trace_no_trace_payload() {
        let trace = false;
        let trace_payload = false;
        let trace_level = "basic".to_string();

        let effective_trace = trace || trace_payload;
        let effective_trace_level = if trace_payload {
            "payload"
        } else {
            trace_level.as_str()
        };

        assert!(!effective_trace);
        assert_eq!(effective_trace_level, "basic");
    }

    // =========================================================================
    // Verbose flag inheritance (local vs global)
    // =========================================================================

    /// Test that local verbose flag overrides global false
    #[test]
    fn test_verbose_local_true_global_false() {
        let local_verbose = true;
        let global_verbose = false;
        let effective_verbose = local_verbose || global_verbose;
        assert!(effective_verbose);
    }

    /// Test that global verbose flag takes effect when local is false
    #[test]
    fn test_verbose_local_false_global_true() {
        let local_verbose = false;
        let global_verbose = true;
        let effective_verbose = local_verbose || global_verbose;
        assert!(effective_verbose);
    }

    /// Test that both verbose false means not verbose
    #[test]
    fn test_verbose_both_false() {
        let local_verbose = false;
        let global_verbose = false;
        let effective_verbose = local_verbose || global_verbose;
        assert!(!effective_verbose);
    }

    /// Test that both verbose true means verbose
    #[test]
    fn test_verbose_both_true() {
        let local_verbose = true;
        let global_verbose = true;
        let effective_verbose = local_verbose || global_verbose;
        assert!(effective_verbose);
    }

    /// Test verbose inheritance end-to-end via global flag and Run command.
    /// Note: clap with `global = true` and matching short flag `-v` means
    /// the global verbose flag propagates to both the Cli struct and the
    /// Run subcommand's local verbose field.
    #[test]
    fn test_verbose_inheritance_run_global() {
        let args = vec!["apr", "--verbose", "run", "model.gguf", "--prompt", "test"];
        let cli = parse_cli(args).expect("Failed to parse");
        assert!(cli.verbose);
        match *cli.command {
            Commands::Run { verbose, .. } => {
                // With global = true, clap propagates to both levels
                // effective_verbose = local || global = always true
                let effective = verbose || cli.verbose;
                assert!(effective);
            }
            _ => panic!("Expected Run command"),
        }
    }
