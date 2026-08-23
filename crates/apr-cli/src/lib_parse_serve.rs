
    /// Test Serve Plan command with local file path
    #[test]
    fn test_parse_serve_plan_local() {
        let args = vec!["apr", "serve", "plan", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                command:
                    ServeCommands::Plan {
                        ref model,
                        gpu,
                        batch_size,
                        seq_len,
                        ref format,
                        ref quant,
                    },
            } => {
                assert_eq!(model, "model.gguf");
                assert!(!gpu);
                assert_eq!(batch_size, 1);
                assert_eq!(seq_len, 4096);
                assert_eq!(format, "text");
                assert!(quant.is_none());
            }
            _ => panic!("Expected Serve Plan command"),
        }
    }

    /// Test Serve Plan command with HuggingFace URL
    #[test]
    fn test_parse_serve_plan_hf_url() {
        let args = vec![
            "apr", "serve", "plan",
            "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "--gpu", "--quant", "Q4_K_M",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                command:
                    ServeCommands::Plan {
                        ref model,
                        gpu,
                        batch_size,
                        seq_len,
                        ref format,
                        ref quant,
                    },
            } => {
                assert_eq!(model, "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF");
                assert!(gpu);
                assert_eq!(batch_size, 1);
                assert_eq!(seq_len, 4096);
                assert_eq!(format, "text");
                assert_eq!(quant.as_deref(), Some("Q4_K_M"));
            }
            _ => panic!("Expected Serve Plan command"),
        }
    }

    /// Test Serve Plan with JSON output and custom batch/seq
    #[test]
    fn test_parse_serve_plan_options() {
        let args = vec![
            "apr", "serve", "plan", "meta-llama/Llama-3.2-1B",
            "--gpu", "--batch-size", "4", "--seq-len", "2048",
            "--format", "json",
        ];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                command:
                    ServeCommands::Plan {
                        ref model,
                        gpu,
                        batch_size,
                        seq_len,
                        ref format,
                        ref quant,
                    },
            } => {
                assert_eq!(model, "meta-llama/Llama-3.2-1B");
                assert!(gpu);
                assert_eq!(batch_size, 4);
                assert_eq!(seq_len, 2048);
                assert_eq!(format, "json");
                assert!(quant.is_none());
            }
            _ => panic!("Expected Serve Plan command"),
        }
    }

    /// Test Serve Run command defaults
    #[test]
    fn test_parse_serve_defaults() {
        let args = vec!["apr", "serve", "run", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Serve {
                command:
                    ServeCommands::Run {
                        port,
                        ref host,
                        no_cors,
                        no_metrics,
                        no_gpu,
                        gpu,
                        batch,
                        trace,
                        ref trace_level,
                        profile,
                        ..
                    },
            } => {
                assert_eq!(port, 8080);
                assert_eq!(host, "127.0.0.1");
                assert!(!no_cors);
                assert!(!no_metrics);
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!batch);
                assert!(!trace);
                assert_eq!(trace_level, "basic");
                assert!(!profile);
            }
            _ => panic!("Expected Serve Run command"),
        }
    }

    /// Test Bench command defaults
    #[test]
    fn test_parse_bench_defaults() {
        let args = vec!["apr", "bench", "model.gguf"];
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
                assert_eq!(warmup, 3);
                assert_eq!(iterations, 5);
                assert_eq!(max_tokens, 32);
                assert!(prompt.is_none());
                assert!(!fast);
                assert!(brick.is_none());
            }
            _ => panic!("Expected Bench command"),
        }
    }

    /// Test Cbtop command defaults
    #[test]
    fn test_parse_cbtop_defaults() {
        let args = vec!["apr", "cbtop"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Cbtop {
                model,
                attach,
                model_path,
                headless,
                json,
                output,
                ci,
                throughput,
                brick_score,
                warmup,
                iterations,
                speculative,
                speculation_k,
                draft_model,
                concurrent,
                simulated,
            }) => {
                assert!(model.is_none());
                assert!(attach.is_none());
                assert!(model_path.is_none());
                assert!(!headless);
                assert!(!json);
                assert!(output.is_none());
                assert!(!ci);
                assert!(throughput.is_none());
                assert!(brick_score.is_none());
                assert_eq!(warmup, 10);
                assert_eq!(iterations, 100);
                assert!(!speculative);
                assert_eq!(speculation_k, 4);
                assert!(draft_model.is_none());
                assert_eq!(concurrent, 1);
                assert!(!simulated);
            }
            _ => panic!("Expected Cbtop command"),
        }
    }

    /// Test Profile command defaults
    #[test]
    fn test_parse_profile_defaults() {
        let args = vec!["apr", "profile", "model.apr"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Profile {
                granular,
                format,
                focus,
                detect_naive,
                threshold,
                compare_hf,
                energy,
                perf_grade,
                callgraph,
                fail_on_naive,
                output,
                ci,
                assert_throughput,
                assert_p99,
                assert_p50,
                warmup,
                measure,
                ..
            }) => {
                assert!(!granular);
                assert_eq!(format, "human");
                assert!(focus.is_none());
                assert!(!detect_naive);
                assert!((threshold - 10.0).abs() < f64::EPSILON);
                assert!(compare_hf.is_none());
                assert!(!energy);
                assert!(!perf_grade);
                assert!(!callgraph);
                assert!(!fail_on_naive);
                assert!(output.is_none());
                assert!(!ci);
                assert!(assert_throughput.is_none());
                assert!(assert_p99.is_none());
                assert!(assert_p50.is_none());
                assert_eq!(warmup, 3);
                assert_eq!(measure, 10);
            }
            _ => panic!("Expected Profile command"),
        }
    }

    /// Test Qa command defaults
    #[test]
    fn test_parse_qa_defaults() {
        let args = vec!["apr", "qa", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Qa {
                assert_tps,
                assert_speedup,
                assert_gpu_speedup,
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
                assert!(assert_tps.is_none());
                assert!(assert_speedup.is_none());
                assert!(assert_gpu_speedup.is_none());
                assert!(!skip_golden);
                assert!(!skip_throughput);
                assert!(!skip_ollama);
                assert!(!skip_gpu_speedup);
                assert!(!skip_contract);
                assert!(!skip_format_parity);
                assert!(safetensors_path.is_none());
                assert_eq!(iterations, 10);
                assert_eq!(warmup, 3);
                assert_eq!(max_tokens, 32);
                assert!(!json);
                assert!(!verbose);
            }
            _ => panic!("Expected Qa command"),
        }
    }

    /// Test Chat command defaults
    #[test]
    fn test_parse_chat_defaults() {
        let args = vec!["apr", "chat", "model.gguf"];
        let cli = parse_cli(args).expect("Failed to parse");
        match *cli.command {
            Commands::Extended(ExtendedCommands::Chat {
                temperature,
                top_p,
                max_tokens,
                system,
                inspect,
                no_gpu,
                gpu,
                trace,
                trace_steps,
                trace_verbose,
                trace_output,
                trace_level,
                profile,
                ..
            }) => {
                assert!((temperature - 0.7).abs() < f32::EPSILON);
                assert!((top_p - 0.9).abs() < f32::EPSILON);
                assert_eq!(max_tokens, 512);
                assert!(system.is_none());
                assert!(!inspect);
                assert!(!no_gpu);
                assert!(!gpu);
                assert!(!trace);
                assert!(trace_steps.is_none());
                assert!(!trace_verbose);
                assert!(trace_output.is_none());
                assert_eq!(trace_level, "basic");
                assert!(!profile);
            }
            _ => panic!("Expected Chat command"),
        }
    }


    // ========================================================================
    // #2583: `apr serve run --backend` must validate like `apr run` / `apr chat`
    // ========================================================================

    /// #2583 falsifier: `apr serve run --backend nonsense` must be REJECTED and
    /// the error must name the valid values.
    ///
    /// Before the fix, `ServeCommands::Run::backend` was declared
    /// `#[arg(long, value_name = "BACKEND")]` with no `value_parser`
    /// (serve_commands.rs:73), so this parsed successfully, `ServerConfig.backend`
    /// carried `"nonsense"`, and the server started on whatever backend it would
    /// have chosen anyway. The identical typo on `apr run` / `apr chat` has been
    /// a hard parse error since PMAT-488.
    #[test]
    fn test_serve_run_rejects_unknown_backend_2583() {
        let err = parse_cli(vec![
            "apr",
            "serve",
            "run",
            "model.apr",
            "--backend",
            "nonsense",
        ])
        .expect_err("`apr serve run --backend nonsense` must NOT parse (#2583)");
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::InvalidValue,
            "unknown backend must be an InvalidValue parse error, got {:?}",
            err.kind()
        );
        let rendered = err.to_string();
        for value in crate::BACKEND_VALUES {
            assert!(
                rendered.contains(value),
                "error must name the valid backend `{value}`; got:\n{rendered}"
            );
        }
    }

    /// Every value in `BACKEND_VALUES` must still be accepted by `apr serve run`.
    /// A "validation" that rejected the *valid* values too would satisfy the
    /// rejection falsifier above while breaking every real invocation.
    #[test]
    fn test_serve_run_accepts_known_backends_2583() {
        for value in crate::BACKEND_VALUES {
            let cli = parse_cli(vec!["apr", "serve", "run", "model.apr", "--backend", value])
                .unwrap_or_else(|e| panic!("`apr serve run --backend {value}` must parse: {e}"));
            match *cli.command {
                Commands::Serve {
                    command: ServeCommands::Run { ref backend, .. },
                } => assert_eq!(backend.backend.as_deref(), Some(value)),
                _ => panic!("Expected Serve Run command"),
            }
        }
    }

    /// Regression guard for the two sites that already validated: sharing one
    /// declaration must not silently drop validation from `apr run` / `apr chat`.
    #[test]
    fn test_run_and_chat_still_reject_unknown_backend_2583() {
        for args in [
            vec!["apr", "run", "model.apr", "--backend", "nonsense"],
            vec!["apr", "chat", "model.apr", "--backend", "nonsense"],
        ] {
            let label = args[1];
            let err = match parse_cli(args.clone()) {
                Ok(_) => panic!("`apr {label} --backend nonsense` must NOT parse"),
                Err(e) => e,
            };
            assert_eq!(
                err.kind(),
                clap::error::ErrorKind::InvalidValue,
                "`apr {label} --backend nonsense` must be InvalidValue, got {:?}",
                err.kind()
            );
        }
    }

    /// #2583 invariant, asserted against the built command tree rather than the
    /// source text: every command that offers the inference `--backend` override
    /// must advertise the SAME fixed value set.
    ///
    /// `apr serve run` advertised none, which is how it accepted anything.
    #[test]
    fn test_backend_possible_values_identical_across_commands_2583() {
        use clap::CommandFactory;

        // Same 16 MB stack as `parse_cli`: building the command tree for the
        // full subcommand set blows the default test-thread stack in debug.
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let root = crate::Cli::command();
                let possible_values = |path: &[&str]| -> Vec<String> {
                    let mut cmd = root.clone();
                    for name in path {
                        cmd = cmd
                            .find_subcommand(name)
                            .unwrap_or_else(|| panic!("subcommand `{name}` not found"))
                            .clone();
                    }
                    let arg = cmd
                        .get_arguments()
                        .find(|a| a.get_id() == "backend")
                        .unwrap_or_else(|| {
                            panic!("`--backend` not found on `{}`", path.join(" "))
                        });
                    arg.get_possible_values()
                        .iter()
                        .map(|p| p.get_name().to_string())
                        .collect()
                };

                let expected: Vec<String> =
                    crate::BACKEND_VALUES.iter().map(|s| (*s).to_string()).collect();
                for path in [vec!["run"], vec!["chat"], vec!["serve", "run"]] {
                    assert_eq!(
                        possible_values(&path),
                        expected,
                        "`apr {}` must accept exactly {expected:?} for --backend (#2583)",
                        path.join(" ")
                    );
                }
            })
            .expect("spawn thread")
            .join()
            .expect("join thread");
    }

    /// #2583 anti-duplication ratchet.
    ///
    /// The defect was not "one site forgot a `value_parser`" — it was that the
    /// declaration is copied by hand, so every new consumer is one more chance
    /// to forget one. Two hand-written copies had already spawned a third,
    /// unvalidated site. The backend value list is now wired into clap in exactly
    /// one place (`BackendArg`); every consumer flattens it. This fails if a
    /// second hand-written wiring is added instead.
    ///
    /// Both needles are assembled with `concat!` so this file's own source text
    /// — doc comment, code and failure message alike — cannot match them. The
    /// first draft scanned for the literals and found itself three times.
    /// Every `.rs` file under `crates/apr-cli/src`, recursively.
    fn apr_cli_src_rs_files_2583() -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        let mut stack = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")];
        while let Some(dir) = stack.pop() {
            let entries = std::fs::read_dir(&dir)
                .unwrap_or_else(|e| panic!("read_dir {}: {e}", dir.display()));
            for entry in entries {
                let path = entry.expect("dir entry").path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().and_then(std::ffi::OsStr::to_str) == Some("rs") {
                    files.push(path);
                }
            }
        }
        files
    }

    #[test]
    fn test_backend_value_parser_wired_exactly_once_2583() {
        let arg_attr = concat!("#[", "arg(");
        let wiring = concat!("value_parser", " = BACKEND_VALUES");
        let mut sites: Vec<String> = Vec::new();
        for path in apr_cli_src_rs_files_2583() {
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            let hits = text
                .lines()
                .enumerate()
                .filter(|(_, line)| line.contains(arg_attr) && line.contains(wiring));
            sites.extend(hits.map(|(i, _)| format!("{}:{}", path.display(), i + 1)));
        }
        assert_eq!(
            sites.len(),
            1,
            "`{wiring}` must appear in exactly one `{arg_attr}..)]` (the shared \
             `BackendArg`); every other command flattens it. Found {}: {sites:#?}",
            sites.len()
        );
    }

    // ========================================================================
    // #2583 follow-up: the ratchet above is blind to a backend flag that never
    // mentions `BACKEND_VALUES`. `apr finetune --gpu-backend` was exactly that
    // — a live, undisclosed instance of the same silent-wrong-backend defect,
    // one command over, that the source-text scan cannot see because it
    // legitimately carries a different value set.
    // ========================================================================

    /// Every backend-selecting flag in the whole built command tree, as
    /// `(path, arg-id, advertised values)`.
    ///
    /// The universe is the CLAP TREE, not a hand-written list of commands: a new
    /// subcommand adds itself here whether or not anyone remembers to. That is
    /// the property the source-text ratchet lacks.
    fn discover_backend_args_2583() -> Vec<(String, String, Vec<String>)> {
        use clap::CommandFactory;
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(|| {
                let mut found = Vec::new();
                let mut stack = vec![(String::new(), crate::Cli::command())];
                while let Some((prefix, cmd)) = stack.pop() {
                    for arg in cmd.get_arguments() {
                        let id = arg.get_id().to_string();
                        if !id.contains("backend") {
                            continue;
                        }
                        let values: Vec<String> = arg
                            .get_possible_values()
                            .iter()
                            .map(|p| p.get_name().to_string())
                            .collect();
                        found.push((prefix.trim().to_string(), id, values));
                    }
                    for sub in cmd.get_subcommands() {
                        if sub.get_name() == "help" {
                            continue;
                        }
                        stack.push((format!("{prefix} {}", sub.get_name()), sub.clone()));
                    }
                }
                found.sort();
                found
            })
            .expect("spawn thread")
            .join()
            .expect("join thread")
    }

    /// #2583 universal ratchet: EVERY backend-selecting flag anywhere in the
    /// command tree must advertise a fixed value set to clap.
    ///
    /// This is the invariant the three inference sites share with
    /// `apr finetune --gpu-backend` and `apr distill --backend` even though the
    /// value sets differ, so it — unlike the `BACKEND_VALUES` source scan — can
    /// see all five. `apr finetune --gpu-backend` advertised `[]` on the branch
    /// that first fixed `apr serve run`.
    #[test]
    fn test_every_backend_arg_advertises_its_values_2583() {
        let unvalidated: Vec<String> = discover_backend_args_2583()
            .into_iter()
            .filter(|(_, _, values)| values.is_empty())
            .map(|(path, id, _)| format!("apr {path} --{}", id.replace('_', "-")))
            .collect();
        assert!(
            unvalidated.is_empty(),
            "these backend flags accept ANY string, so a typo silently selects the \
             default backend (#2583). Give each a `value_parser` naming its own \
             value set: {unvalidated:#?}"
        );
    }

    /// The discovered universe itself is pinned: a NEW backend flag must be an
    /// explicit decision (which value list?), not an inherited default.
    #[test]
    fn test_backend_arg_universe_is_pinned_2583() {
        // Discovered by the ratchet, NOT by grepping apr-cli: `cgp profile
        // compare` lives in `aprender-cgp` and `rag transcribe` names its own
        // three values. A source scan of this crate finds neither.
        let expected: Vec<(String, String, Vec<String>)> = [
            ("cgp profile compare", "backends", &cgp::cli::CGP_BACKEND_VALUES[..]),
            ("chat", "backend", &crate::BACKEND_VALUES[..]),
            ("distill", "backend", &crate::DISTILL_BACKEND_VALUES[..]),
            (
                "finetune",
                "gpu_backend",
                &crate::FINETUNE_GPU_BACKEND_VALUES[..],
            ),
            ("rag transcribe", "backend", &["cpu", "gpu", "cuda"][..]),
            ("run", "backend", &crate::BACKEND_VALUES[..]),
            ("serve run", "backend", &crate::BACKEND_VALUES[..]),
        ]
        .into_iter()
        .map(|(path, id, values)| {
            (
                path.to_string(),
                id.to_string(),
                values.iter().map(|s| (*s).to_string()).collect(),
            )
        })
        .collect();
        assert_eq!(
            discover_backend_args_2583(),
            expected,
            "the set of backend-selecting flags changed; pick this one's value list \
             deliberately and update the table (#2583)"
        );
    }

    /// #2583 falsifier for the surviving live instance: a typo'd
    /// `--gpu-backend` must be rejected, not routed to the `_ =>` ("auto") arm
    /// of `gpu_backend_notice`, where it is indistinguishable from the flag
    /// being omitted.
    #[cfg(feature = "training")]
    #[test]
    fn test_finetune_rejects_unknown_gpu_backend_2583() {
        let err = parse_cli(vec![
            "apr",
            "finetune",
            "model.apr",
            "--gpu-backend",
            "cudaa",
        ])
        .expect_err("`apr finetune --gpu-backend cudaa` must NOT parse (#2583)");
        assert_eq!(
            err.kind(),
            clap::error::ErrorKind::InvalidValue,
            "unknown gpu-backend must be an InvalidValue parse error, got {:?}",
            err.kind()
        );
        let rendered = err.to_string();
        for value in crate::FINETUNE_GPU_BACKEND_VALUES {
            assert!(
                rendered.contains(value),
                "error must name the valid backend `{value}`; got:\n{rendered}"
            );
        }
    }

    /// …and the valid values, including the `auto` default that is absent from
    /// `BACKEND_VALUES`, must still parse. An over-rejecting "fix" — e.g.
    /// reusing `BackendArg` here — satisfies the falsifier above while breaking
    /// every real `apr finetune` invocation.
    #[cfg(feature = "training")]
    #[test]
    fn test_finetune_accepts_its_own_gpu_backends_2583() {
        for value in crate::FINETUNE_GPU_BACKEND_VALUES {
            let cli = parse_cli(vec!["apr", "finetune", "model.apr", "--gpu-backend", value])
                .unwrap_or_else(|e| {
                    panic!("`apr finetune --gpu-backend {value}` must parse: {e}")
                });
            match *cli.command {
                Commands::ModelOps(ModelOpsCommands::Finetune {
                    ref gpu_backend, ..
                }) => assert_eq!(gpu_backend, value),
                _ => panic!("Expected Finetune command"),
            }
        }
        // The default must be a member of the set it is now validated against.
        let cli = parse_cli(vec!["apr", "finetune", "model.apr"]).expect("bare finetune parses");
        match *cli.command {
            Commands::ModelOps(ModelOpsCommands::Finetune {
                ref gpu_backend, ..
            }) => assert_eq!(gpu_backend, "auto"),
            _ => panic!("Expected Finetune command"),
        }
    }

    /// `apr distill --backend` — the earlier report called this the same defect
    /// class. Re-measured: it is NOT. `distill::run()` rejects an unknown value
    /// by name before any I/O (commands/distill.rs:464-503, covered by
    /// `distill_run_unknown_backend_errors`), so it never silently ran the
    /// default. What it lacked was clap-level advertisement, closed here.
    #[cfg(feature = "training")]
    #[test]
    fn test_distill_backend_values_2583() {
        for value in crate::DISTILL_BACKEND_VALUES {
            parse_cli(vec!["apr", "distill", "t.apr", "--backend", value])
                .unwrap_or_else(|e| panic!("`apr distill --backend {value}` must parse: {e}"));
        }
        let err = parse_cli(vec!["apr", "distill", "t.apr", "--backend", "cda"])
            .expect_err("`apr distill --backend cda` must NOT parse");
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidValue);
    }
