
    // ════════════════════════════════════════════════════════════════════
    // Coverage tests for dispatch_model_commands, dispatch_analysis_commands,
    // dispatch_profiling_commands, dispatch_train_command (PMAT coverage gap)
    // ════════════════════════════════════════════════════════════════════

    // ── dispatch_model_commands ──────────────────────────────────────────

    #[test]
    fn test_dispatch_model_commands_returns_none_for_inspect() {
        // Inspect is an inspection command, not a model management command
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("model.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
            quality: false,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_none(), "Inspect should not be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_returns_none_for_list_command() {
        // List IS a model command — should return Some
        let cli = make_cli(Commands::List);
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "List should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_pull_nonexistent() {
        let cli = make_cli(Commands::Pull {
            model_ref: "nonexistent-model-that-does-not-exist-xyz123".to_string(),
            repo: None,
            force: false,
            dry_run: false,
            revision: None,
            offline: false,
            include: vec![],
            output: None,
            verify: false,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Pull should be handled by dispatch_model_commands");
        // Pull may succeed (network) or fail (no network) — we just test routing
    }

    #[test]
    fn test_dispatch_model_commands_rm_nonexistent() {
        let cli = make_cli(Commands::Rm {
            model_ref: "nonexistent-model-xyz789".to_string(),
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Rm should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_tui_none_file() {
        let cli = make_cli(Commands::Tui { file: None });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Tui should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_tui_with_file() {
        let cli = make_cli(Commands::Tui {
            file: Some(PathBuf::from("/tmp/nonexistent_tui_model.apr")),
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Tui with file should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_merge_nonexistent() {
        let cli = make_cli(Commands::Merge {
            files: vec![
                PathBuf::from("/tmp/nonexistent_merge_a.apr"),
                PathBuf::from("/tmp/nonexistent_merge_b.apr"),
            ],
            strategy: "average".to_string(),
            output: Some(PathBuf::from("/tmp/merged_out.apr")),
            weights: None,
            base_model: None,
            drop_rate: 0.9,
            density: 0.2,
            seed: 42,
            plan: false,
                force: true,
            });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Merge should be handled by dispatch_model_commands");
        // merge with nonexistent files should error
        assert!(result.expect("should be some").is_err());
    }

    #[test]
    fn test_dispatch_model_commands_merge_plan_mode() {
        let cli = make_cli(Commands::Merge {
            files: vec![
                PathBuf::from("/tmp/nonexistent_merge_plan_a.apr"),
                PathBuf::from("/tmp/nonexistent_merge_plan_b.apr"),
            ],
            strategy: "slerp".to_string(),
            output: None, // plan mode doesn't need output
            weights: Some(vec![0.5, 0.5]),
            base_model: None,
            drop_rate: 0.9,
            density: 0.2,
            seed: 42,
            plan: true,
                force: true,
            });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Merge plan should be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_returns_none_for_validate() {
        let cli = make_cli(Commands::Validate {
            file: PathBuf::from("model.apr"),
            quality: false,
            strict: false,
            min_score: None,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_none(), "Validate should not be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_returns_none_for_debug() {
        let cli = make_cli(Commands::Debug {
            file: Some(PathBuf::from("model.apr")),
            action: None,
            drama: false,
            hex: false,
            strings: false,
            limit: 256,
        });
        let result = dispatch_model_commands(&cli);
        assert!(result.is_none(), "Debug should not be handled by dispatch_model_commands");
    }

    #[test]
    fn test_dispatch_model_commands_prune_nonexistent() {
        let cli = make_cli(Commands::ModelOps(ModelOpsCommands::Prune {
            file: PathBuf::from("/tmp/nonexistent_prune_model.apr"),
            method: "magnitude".to_string(),
            target_ratio: 0.5,
            sparsity: 0.0,
            output: Some(PathBuf::from("/tmp/pruned_out.apr")),
            remove_layers: None,
            analyze: false,
            plan: false,
            calibration: None,
        }));
        let result = dispatch_model_commands(&cli);
        assert!(result.is_some(), "Prune should be handled by dispatch_model_commands");
        assert!(result.expect("should be some").is_err(), "Prune with nonexistent file should error");
    }

    // ── dispatch_analysis_commands ──────────────────────────────────────

    #[test]
    fn test_dispatch_analysis_returns_none_for_non_extended() {
        let cli = make_cli(Commands::List);
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_none(), "Non-extended command should not match analysis dispatcher");
    }

    #[test]
    fn test_dispatch_analysis_returns_none_for_chat() {
        // Chat is Extended but not an analysis command
        let cli = make_cli(Commands::Extended(ExtendedCommands::Chat {
            file: PathBuf::from("model.apr"),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            no_gpu: true,
            gpu: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
            backend: None,
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_none(), "Chat command should not be handled by analysis dispatcher");
    }

    #[test]
    fn test_dispatch_analysis_tree_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Tree {
            file: PathBuf::from("/tmp/nonexistent_tree_model.apr"),
            filter: None,
            format: crate::commands::tree::TreeFormat::Ascii,
            sizes: false,
            depth: None,
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Tree should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Tree with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_hex_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Hex {
            file: PathBuf::from("/tmp/nonexistent_hex_model.apr"),
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
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Hex should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Hex with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_flow_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Flow {
            file: PathBuf::from("/tmp/nonexistent_flow_model.apr"),
            layer: None,
            component: "full".to_string(),
            verbose: false,
            json: false,
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Flow should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Flow with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_qualify_nonexistent_file() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Qualify {
            file: PathBuf::from("/tmp/nonexistent_qualify_model.apr"),
            tier: "basic".to_string(),
            timeout: 30,
            json: false,
            verbose: false,
            skip: None,
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Qualify should be handled by analysis dispatcher");
        assert!(result.expect("should be some").is_err(), "Qualify with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_analysis_diagnose_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Diagnose {
            checkpoint_dir: PathBuf::from("/tmp/nonexistent_diagnose_dir"),
            data: None,
            model_size: None,
            num_classes: 5,
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Diagnose should be handled by analysis dispatcher");
    }

    // ── apr showcase --step ─────────────────────────────────────────────

    /// `--step bogus` used to map to `None`, indistinguishable from "no --step
    /// given", so the CLI answered "No step specified" to a command line that
    /// plainly specified one.
    #[test]
    fn test_showcase_unknown_step_names_the_offending_value() {
        let err = parse_showcase_step("bogus").expect_err("unknown step must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("bogus"),
            "error must quote what the user typed, got: {msg}"
        );
        assert!(
            !msg.contains("No step specified"),
            "must not claim no step was specified, got: {msg}"
        );
        for expected in ["import", "gguf", "convert", "apr", "all"] {
            assert!(
                msg.contains(expected),
                "error should list '{expected}', got: {msg}"
            );
        }
    }

    #[test]
    fn test_showcase_every_documented_step_parses() {
        // The list printed by `print_available_steps`, plus `chat`.
        for name in [
            "import",
            "gguf",
            "convert",
            "apr",
            "brick",
            "bench",
            "chat",
            "visualize",
            "zram",
            "cuda",
            "all",
        ] {
            assert!(
                parse_showcase_step(name).is_ok(),
                "'{name}' is advertised and must parse"
            );
        }
    }

    #[test]
    fn test_showcase_step_typo_is_rejected_not_silently_ignored() {
        // A near-miss is the realistic case: `--step ggug` for `gguf`.
        assert!(parse_showcase_step("ggug").is_err());
        assert!(parse_showcase_step("GGUF").is_err(), "matching is exact");
        assert!(parse_showcase_step("").is_err());
    }

    /// `apr probar tensor --format bogus` used to be silently rewritten to
    /// `both` by `.unwrap_or(ExportFormat::Both)`, so a typo produced a
    /// different export than requested and exited 0.
    #[test]
    fn test_dispatch_analysis_probar_rejects_unknown_format() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Probar {
            command: ProbarSubcommand::Tensor {
                file: PathBuf::from("/tmp/nonexistent_probar_model.apr"),
                output: PathBuf::from("/tmp/nonexistent_probar_out"),
                format: "bogus".to_string(),
                golden: None,
                layer: None,
                assert: false,
                tolerance: 0.98,
            },
        }));
        let result = dispatch_analysis_commands(&cli);
        let err = result
            .expect("Probar should be handled by analysis dispatcher")
            .expect_err("unknown --format must be rejected, not silently defaulted");
        let msg = err.to_string();
        assert!(
            msg.contains("Unknown format") && msg.contains("bogus"),
            "error must name the rejected value, got: {msg}"
        );
    }

    /// The valid values must still reach probar (which then fails on the
    /// missing model, not on the format).
    #[test]
    fn test_dispatch_analysis_probar_accepts_known_formats() {
        for format in ["json", "png", "both", "all"] {
            let cli = make_cli(Commands::Extended(ExtendedCommands::Probar {
                command: ProbarSubcommand::Tensor {
                    file: PathBuf::from("/tmp/nonexistent_probar_model.apr"),
                    output: PathBuf::from("/tmp/nonexistent_probar_out"),
                    format: format.to_string(),
                    golden: None,
                    layer: None,
                    assert: false,
                    tolerance: 0.98,
                },
            }));
            let err = dispatch_analysis_commands(&cli)
                .expect("handled")
                .expect_err("missing model file still errors");
            assert!(
                !err.to_string().contains("Unknown format"),
                "'{format}' must be accepted, got: {err}"
            );
        }
    }

    #[test]
    fn test_dispatch_analysis_compare_hf_offline_rejected() {
        let mut cli = make_cli(Commands::Extended(ExtendedCommands::CompareHf {
            file: PathBuf::from("/tmp/nonexistent_compare_hf.apr"),
            hf: "org/repo".to_string(),
            tensor: None,
            threshold: 0.01,
            json: false,
        }));
        cli.offline = true;
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "CompareHf should be handled by analysis dispatcher");
        let err = result.expect("should be some");
        assert!(err.is_err(), "CompareHf in offline mode should be rejected");
    }

    // ── dispatch_profiling_commands ─────────────────────────────────────

    #[test]
    fn test_dispatch_profiling_returns_none_for_non_extended() {
        let cli = make_cli(Commands::List);
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_none(), "Non-extended command should not match profiling dispatcher");
    }

    #[test]
    fn test_dispatch_profiling_returns_none_for_chat() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Chat {
            file: PathBuf::from("model.apr"),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            no_gpu: true,
            gpu: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
            backend: None,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_none(), "Chat should not be handled by profiling dispatcher");
    }

    // ── ptx: dead command on a default `cargo install aprender` (#2399 f1) ──

    /// On a build without the PTX analyzer, `apr ptx <file>` must say the
    /// command is unavailable and how to get it — not report the file.
    ///
    /// The feature check used to run AFTER path resolution, so
    /// `apr ptx /nonexistent.ptx` exited 3 with "File not found", sending the
    /// user off to fix a path for a command this binary can never run.
    #[cfg(not(feature = "trueno-explain"))]
    #[test]
    fn ptx_without_analyzer_reports_the_feature_not_the_path() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Ptx {
            file: Some(PathBuf::from("/nonexistent_ptx_source_xyz.ptx")),
            kernel: None,
            strict: false,
            bugs: false,
            json: false,
            verbose: false,
        }));
        let err = dispatch_profiling_commands(&cli)
            .expect("Ptx must be handled by the profiling dispatcher")
            .expect_err("ptx must fail when the analyzer is not compiled in");
        let msg = err.to_string();
        assert!(
            !msg.contains("File not found"),
            "the missing path is not the reason ptx failed, got: {msg}"
        );
        assert!(
            msg.contains("cargo install aprender --features ptx"),
            "the error must name a remedy the user of an installed binary can run, got: {msg}"
        );
        assert!(
            matches!(err, crate::error::CliError::FeatureDisabled(_)),
            "an unavailable command must not exit as if the input were missing \
             (FileNotFound is exit 3); got {err:?}"
        );
    }

    /// Same message with no file argument at all — the two shapes a user tries.
    #[cfg(not(feature = "trueno-explain"))]
    #[test]
    fn ptx_without_analyzer_reports_remedy_for_kernel_form_too() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Ptx {
            file: None,
            kernel: Some("Q4KGemv".to_string()),
            strict: false,
            bugs: false,
            json: false,
            verbose: false,
        }));
        let err = dispatch_profiling_commands(&cli)
            .expect("Ptx must be handled by the profiling dispatcher")
            .expect_err("ptx must fail when the analyzer is not compiled in");
        assert!(
            err.to_string()
                .contains("cargo install aprender --features ptx"),
            "got: {err}"
        );
    }

    #[test]
    fn test_dispatch_profiling_bench_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Bench {
            file: PathBuf::from("/tmp/nonexistent_bench_model.apr"),
            warmup: 1,
            iterations: 1,
            max_tokens: 10,
            prompt: None,
            fast: true,
            brick: None,
            percentiles: vec![50.0, 95.0, 99.0],
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "Bench should be handled by profiling dispatcher");
        assert!(result.expect("should be some").is_err(), "Bench with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_profiling_profile_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Profile {
            file: PathBuf::from("/tmp/nonexistent_profile_model.apr"),
            granular: false,
            format: "text".to_string(),
            focus: None,
            detect_naive: false,
            threshold: 0.01,
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
            no_gpu: true,
            compare: None,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "Profile should be handled by profiling dispatcher");
        assert!(result.expect("should be some").is_err(), "Profile with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_profiling_eval_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Eval {
            file: PathBuf::from("/tmp/nonexistent_eval_model.apr"),
            dataset: "test".to_string(),
            text: None,
            max_tokens: 32,
            threshold: 90.0,
            task: None,
            data: None,
            model_size: None,
            num_classes: 5,
            generate_card: false,
            device: "cpu".to_string(),
            samples: 1,
            temperature: 0.0,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "Eval should be handled by profiling dispatcher");
        assert!(result.expect("should be some").is_err(), "Eval with nonexistent file should error");
    }

    #[test]
    fn test_dispatch_profiling_qa_nonexistent() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Qa {
            file: PathBuf::from("/tmp/nonexistent_qa_model.apr"),
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
            max_tokens: 10,
            json: false,
            verbose: false,
            min_executed: Some(0),
            previous_report: None,
            regression_threshold: Some(0.0),
            skip_gpu_state: true,
            skip_metadata: true,
            skip_capability: true,
            assert_classifier_head: false,
        }));
        let result = dispatch_profiling_commands(&cli);
        assert!(result.is_some(), "QA should be handled by profiling dispatcher");
    }

    // ── dispatch_train_command ──────────────────────────────────────────

    #[cfg(feature = "training")]
    #[test]
    fn test_dispatch_analysis_train_routing() {
        // Train Plan with nonexistent data — tests train routing path
        let cli = make_cli(Commands::Extended(ExtendedCommands::Train {
            command: TrainCommands::Plan {
                data: Some(PathBuf::from("/tmp/nonexistent_train_data.jsonl")),
                model_size: "tiny".to_string(),
                model_path: None,
                num_classes: 5,
                task: "classify".to_string(),
                config: None,
                output: PathBuf::from("/tmp/train_plan_out"),
                strategy: "auto".to_string(),
                budget: 100,
                scout: false,
                max_epochs: 3,
                learning_rate: Some(1e-4),
                lora_rank: Some(8),
                batch_size: Some(4),
                val_data: None,
                test_data: None,
                format: "text".to_string(),
            },
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Train should be handled by analysis dispatcher (delegates to dispatch_train_command)");
    }

    #[cfg(feature = "training")]
    #[test]
    fn test_dispatch_train_sweep() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Train {
            command: TrainCommands::Sweep {
                config: PathBuf::from("/tmp/nonexistent_sweep_config.toml"),
                strategy: "grid".to_string(),
                num_configs: 5,
                output_dir: PathBuf::from("/tmp/sweep_out"),
                seed: 42,
            },
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Train Sweep should be handled by analysis dispatcher");
    }

    #[cfg(feature = "training")]
    #[test]
    fn test_dispatch_train_cluster_status() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Train {
            command: TrainCommands::ClusterStatus {
                cluster: PathBuf::from("/tmp/nonexistent_cluster_config.yaml"),
            },
        }));
        let result = dispatch_analysis_commands(&cli);
        assert!(result.is_some(), "Train ClusterStatus should be handled by analysis dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_inspection_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_inspection_routes_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
            quality: false,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Inspect should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_routes_debug() {
        let cli = make_cli(Commands::Debug {
            file: Some(PathBuf::from("/tmp/nonexistent_pmat540.apr")),
            action: None,
            drama: false,
            hex: false,
            strings: false,
            limit: 256,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Debug should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_routes_validate() {
        let cli = make_cli(Commands::Validate {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            strict: false,
            quality: false,
            min_score: None,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Validate should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_routes_lint() {
        let cli = make_cli(Commands::Lint {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
                strict: false,
        });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_some(), "Lint should be handled by inspection dispatcher");
    }

    #[test]
    fn test_dispatch_inspection_returns_none_for_export() {
        let cli = make_cli(Commands::Export {
            file: Some(PathBuf::from("/tmp/nonexistent_pmat540.apr")),
            format: "gguf".to_string(),
            output: None,
            quantize: None,
            list_formats: false,
            batch: None,
            json: false,
            plan: false,
                force: true,
            });
        let result = dispatch_inspection_commands(&cli);
        assert!(result.is_none(), "Export should NOT be handled by inspection dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_diagnostic_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_diagnostic_routes_trace() {
        let cli = make_cli(Commands::Trace {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            layer: None,
            reference: None,
            json: false,
            verbose: false,
            payload: false,
            diff: false,
            interactive: false,
            save_tensor: None,
            save_tensor_dir: None,
            save_tensor_layers: "0..1".to_string(),
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_some(), "Trace should be handled by diagnostic dispatcher");
    }

    #[test]
    fn test_dispatch_diagnostic_routes_tensors() {
        let cli = make_cli(Commands::Tensors {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            stats: false,
            filter: None,
            limit: 0,
            json: false,
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_some(), "Tensors should be handled by diagnostic dispatcher");
    }

    #[test]
    fn test_dispatch_diagnostic_routes_diff() {
        let cli = make_cli(Commands::Diff {
            file1: PathBuf::from("/tmp/nonexistent_a.apr"),
            file2: PathBuf::from("/tmp/nonexistent_b.apr"),
            weights: false,
            values: false,
            filter: None,
            limit: 10,
            transpose_aware: false,
            json: false,
            quant_roundtrip: false,
            threshold: 0.95,
            no_threshold: false,
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_some(), "Diff should be handled by diagnostic dispatcher");
    }

    #[test]
    fn test_dispatch_diagnostic_returns_none_for_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
            quality: false,
        });
        let result = dispatch_diagnostic_commands(&cli);
        assert!(result.is_none(), "Inspect should NOT be handled by diagnostic dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_format_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_format_routes_import() {
        let cli = make_cli(Commands::Import {
            source: "/tmp/nonexistent_pmat540.safetensors".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
            allow_no_config: true,
        });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_some(), "Import should be handled by format dispatcher");
    }

    #[test]
    fn test_dispatch_format_routes_convert() {
        let cli = make_cli(Commands::Convert {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            quantize: None,
            compress: None,
            output: PathBuf::from("/tmp/nonexistent_pmat540_out.apr"),
            force: false,
        });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_some(), "Convert should be handled by format dispatcher");
    }

    #[test]
    fn test_dispatch_format_routes_export() {
        let cli = make_cli(Commands::Export {
            file: Some(PathBuf::from("/tmp/nonexistent_pmat540.apr")),
            format: "gguf".to_string(),
            output: None,
            quantize: None,
            list_formats: false,
            batch: None,
            json: false,
            plan: false,
                force: true,
            });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_some(), "Export should be handled by format dispatcher");
    }

    #[test]
    fn test_dispatch_format_returns_none_for_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
            quality: false,
        });
        let result = dispatch_format_commands(&cli);
        assert!(result.is_none(), "Inspect should NOT be handled by format dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_runtime_commands coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_runtime_routes_check() {
        let cli = make_cli(Commands::Check {
            file: PathBuf::from("/tmp/nonexistent_pmat540.gguf"),
            no_gpu: true,
            json: false,
        });
        let result = dispatch_runtime_commands(&cli);
        assert!(result.is_some(), "Check should be handled by runtime dispatcher");
    }

    #[test]
    fn test_dispatch_runtime_returns_none_for_inspect() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
            quality: false,
        });
        let result = dispatch_runtime_commands(&cli);
        assert!(result.is_none(), "Inspect should NOT be handled by runtime dispatcher");
    }

    // ════════════════════════════════════════════════════════════════════
    // PMAT-540 Phase 2: dispatch_core_command routing coverage
    // ════════════════════════════════════════════════════════════════════

    #[test]
    fn test_dispatch_core_routes_inspect_via_inspection() {
        let cli = make_cli(Commands::Inspect {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            vocab: false,
            filters: false,
            weights: false,
            json: false,
            quality: false,
        });
        // dispatch_core_command delegates to dispatch_inspection_commands
        let result = dispatch_core_command(&cli);
        assert!(result.is_some(), "Inspect should be routed through core → inspection");
    }

    #[test]
    fn test_dispatch_core_routes_tensors_via_diagnostic() {
        let cli = make_cli(Commands::Tensors {
            file: PathBuf::from("/tmp/nonexistent_pmat540.apr"),
            stats: false,
            filter: None,
            limit: 0,
            json: false,
        });
        let result = dispatch_core_command(&cli);
        assert!(result.is_some(), "Tensors should be routed through core → diagnostic");
    }

    #[test]
    fn test_dispatch_core_routes_import_via_format() {
        let cli = make_cli(Commands::Import {
            source: "/tmp/nonexistent_pmat540.safetensors".to_string(),
            output: None,
            arch: "auto".to_string(),
            quantize: None,
            strict: false,
            preserve_q4k: false,
            tokenizer: None,
            enforce_provenance: false,
            allow_no_config: true,
        });
        let result = dispatch_core_command(&cli);
        assert!(result.is_some(), "Import should be routed through core → format");
    }

    #[test]
    fn test_dispatch_core_returns_none_for_extended() {
        let cli = make_cli(Commands::Extended(ExtendedCommands::Chat {
            file: PathBuf::from("/tmp/nonexistent_pmat540.gguf"),
            temperature: 0.7,
            top_p: 0.9,
            max_tokens: 512,
            system: None,
            inspect: false,
            no_gpu: true,
            gpu: false,
            trace: false,
            trace_steps: None,
            trace_verbose: false,
            trace_output: None,
            trace_level: "basic".to_string(),
            profile: false,
            backend: None,
        }));
        let result = dispatch_core_command(&cli);
        assert!(result.is_none(), "Chat (extended) should NOT be handled by core dispatcher");
    }
