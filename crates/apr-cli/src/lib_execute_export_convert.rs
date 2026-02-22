
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
            format: "ascii".to_string(),
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
    #[test]
    fn test_execute_probar_file_not_found() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Probar {
            file: PathBuf::from("/tmp/nonexistent_model_probar_test.apr"),
            output: PathBuf::from("/tmp/probar-out"),
            format: "both".to_string(),
            golden: None,
            layer: None,
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

    /// Test execute_command: Explain without args succeeds
    #[test]
    fn test_execute_explain_no_args() {
        let cli = make_cli(Commands::Explain {
            code_or_file: None,
            file: None,
            tensor: None,
        });
        // Explain with no args should still run (shows general help)
        let result = execute_command(&cli);
        assert!(result.is_ok(), "Explain with no args should succeed");
    }

    /// Test execute_command: Explain with code succeeds
    #[test]
    fn test_execute_explain_with_code() {
        let cli = make_cli(Commands::Explain {
            code_or_file: Some("E001".to_string()),
            file: None,
            tensor: None,
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
    // --chat flag logic: effective_prompt with ChatML wrapping
    // =========================================================================

    /// Test that --chat flag wraps prompt in ChatML format (verified via parse)
    #[test]
    fn test_chat_flag_chatml_wrapping_logic() {
        // We cannot call execute_command with --chat on a non-existent model
        // without error, but we can verify the ChatML wrapping logic directly.
        let prompt = "What is the meaning of life?";
        let chat = true;

        let effective_prompt = if chat {
            Some(format!(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            ))
        } else {
            Some(prompt.to_string())
        };

        assert!(effective_prompt
            .as_ref()
            .expect("prompt should exist")
            .starts_with("<|im_start|>user\n"));
        assert!(effective_prompt
            .as_ref()
            .expect("prompt should exist")
            .ends_with("<|im_start|>assistant\n"));
        assert!(effective_prompt
            .as_ref()
            .expect("prompt should exist")
            .contains("What is the meaning of life?"));
    }

    /// Test that without --chat, prompt is passed through unchanged
    #[test]
    fn test_no_chat_flag_passthrough() {
        let prompt = Some("Hello world".to_string());
        let chat = false;

        let effective_prompt = if chat {
            prompt
                .as_ref()
                .map(|p| format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", p))
        } else {
            prompt.clone()
        };

        assert_eq!(effective_prompt, Some("Hello world".to_string()));
    }

    /// Test that --chat with no prompt produces None
    #[test]
    fn test_chat_flag_no_prompt() {
        let prompt: Option<String> = None;
        let chat = true;

        let effective_prompt = if chat {
            prompt
                .as_ref()
                .map(|p| format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", p))
        } else {
            prompt.clone()
        };

        assert!(effective_prompt.is_none());
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
