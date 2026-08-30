/// Dispatch core commands (run, serve, inspection, format operations).
///
/// Delegates to sub-dispatchers to keep cyclomatic complexity below 10 per function.
fn dispatch_core_command(cli: &Cli) -> Option<Result<(), CliError>> {
    contract_pre_side_effect_classification!();
    contract_pre_dispatch_completeness!();
    contract_pre_output_format_fidelity!();
    // Try runtime commands first (check, run, serve)
    if let Some(result) = dispatch_runtime_commands(cli) {
        return Some(result);
    }

    // Try inspection commands (inspect, debug, validate, lint, explain, canary)
    if let Some(result) = dispatch_inspection_commands(cli) {
        return Some(result);
    }

    // Try diagnostic commands (trace, tensors, diff)
    if let Some(result) = dispatch_diagnostic_commands(cli) {
        return Some(result);
    }

    // Try format commands (export, import, convert, quantize)
    if let Some(result) = dispatch_format_commands(cli) {
        return Some(result);
    }

    // Try model management commands (merge, finetune, prune, distill, pull, list, rm, tui)
    if let Some(result) = dispatch_model_commands(cli) {
        contract_post_side_effect_classification!(&());
        contract_post_output_format_fidelity!(&());
        return Some(result);
    }

    // Sibling CLIs whose whole command surface used to be reachable only from
    // their own binary (trueno-rag, trueno-zram).
    if let Some(result) = dispatch_sibling_cli_commands(cli) {
        return Some(result);
    }

    // Monorepo management (publish, shims, audit, archive) — dev-only
    #[cfg(feature = "dev")]
    if let Commands::Mono(command) = cli.command.as_ref() {
        return Some(crate::commands::mono::run(command.clone()));
    }

    contract_post_side_effect_classification!(&());
    contract_post_output_format_fidelity!(&());
    None
}

/// Dispatch the commands whose implementation lives in a sibling crate's CLI
/// library: `apr rag` -> `aprender_rag_cli`, `apr zram` -> `aprender_zram_cli`.
///
/// Each arm calls the SAME `dispatch` function the standalone binary calls, so
/// `apr rag index` and `trueno-rag index` cannot drift apart. Before this, both
/// command enums lived in a `main.rs`, which is importable by nothing -- the
/// separate binary was the only way to reach any of it.
fn dispatch_sibling_cli_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    match cli.command.as_ref() {
        Commands::Rag(command) => Some(
            aprender_rag_cli::dispatch(command.clone())
                .map_err(|e| CliError::ValidationFailed(format!("rag: {e}"))),
        ),
        Commands::Zram(command) => {
            let format = if cli.json {
                aprender_zram_cli::output::OutputFormat::Json
            } else {
                aprender_zram_cli::output::OutputFormat::Table
            };
            Some(
                aprender_zram_cli::dispatch(command, format)
                    .map_err(|e| CliError::ValidationFailed(format!("zram: {e}"))),
            )
        }
        Commands::Sim(command) => {
            // #2493 and #2527 each converted simular to clap independently and
            // chose different type names; the batch takes #2527's (richer case
            // table, and it owns the parser ban). Its root is `Cli` with an
            // Option<Commands>, not `Args` with a bare `command`.
            let code = simular::cli::run_cli(simular::cli::Cli {
                command: Some(command.clone()),
            });
            Some(if code == std::process::ExitCode::SUCCESS {
                Ok(())
            } else {
                Err(CliError::ValidationFailed("sim command failed".to_string()))
            })
        }
        Commands::Cgp(command) => Some(
            cgp::cli::dispatch(command.clone(), cli.json)
                .map_err(|e| CliError::ValidationFailed(format!("cgp: {e}"))),
        ),
        Commands::Pv(command) => Some(
            aprender_contracts_cli::dispatch(command.clone())
                .map_err(|e| CliError::ValidationFailed(format!("pv: {e}"))),
        ),
        _ => None,
    }
}

/// Dispatch runtime commands: check, run, serve.
fn dispatch_runtime_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    Some(match cli.command.as_ref() {
        // GH-685: forward cli.verbose to check
        Commands::Check { file, no_gpu, json } => crate::error::resolve_model_path(file)
            .and_then(|r| commands::check::run(&r, *no_gpu, *json || cli.json, cli.verbose)),
        Commands::Run {
            source,
            positional_prompt,
            input,
            prompt,
            max_tokens,
            stream,
            language,
            task,
            format,
            no_gpu,
            gpu,
            offline,
            benchmark,
            trace,
            trace_steps,
            trace_verbose,
            trace_output,
            trace_level,
            trace_payload,
            profile,
            temperature,
            top_k,
            top_p,
            seed,
            repeat_penalty,
            repeat_last_n,
            chat,
            split_prompt,
            batch_jsonl,
            verbose,
            backend: BackendArg { backend },
        } => {
            // GH-614: --backend cpu forces CPU-only inference
            let backend_forces_cpu = backend.as_deref() == Some("cpu");
            if let Some(ref b) = backend {
                if b != "cpu" {
                    eprintln!("Backend override: {b}");
                }
            }
            // FALSIFY-BACKEND-CUDA-HONESTY-001: refuse `--backend cuda` on a build
            // that has no CUDA compiled in, instead of silently serving wgpu/CPU.
            //
            // The CUDA generate path is behind `#[cfg(feature = "cuda")]`
            // (aprender-serve/src/infer/gguf_gpu_generate.rs:356). On a build without
            // that feature the whole block VANISHES, control falls through to the
            // GH-559 wgpu fallback, and the run prints:
            //     Backend override: cuda
            //     Backend: wgpu (Vulkan)
            // wgpu then fails its own cpu-parity gate (cosine 0.884 < 0.99) and
            // degrades again — ~20 tok/s where CUDA gives ~400. Measured 2026-07-27
            // on an RTX 4090 with nvcc 12.8 present, so this is NOT a
            // missing-hardware case; it is a build that cannot honour the flag
            // reporting success anyway.
            //
            // This silently invalidates any measurement taken through it. The
            // Pillar-4 decode beat run against such a binary reports
            // `ratio_median=0.070x` and a BEAT-REGRESSION panic — a fabricated 14x
            // regression with nothing wrong in apr's decode path.
            //
            // A 20x silent downgrade is never what the caller asked for. Fail.
            //
            // NOTE: this checks build capability only. When CUDA *is* compiled in
            // but fails at runtime (e.g. the Blackwell sm_121 JIT), the GH-559
            // wgpu fallback is deliberate and stays.
            if backend.as_deref() == Some("cuda") && !cfg!(feature = "cuda") {
                return Some(Err(CliError::ValidationFailed(
                    "--backend cuda requested, but this `apr` was built WITHOUT the \
`cuda` feature, so the CUDA backend does not exist in this binary. \
Refusing to silently fall back to wgpu/CPU: that path is ~20x slower \
(~20 tok/s vs ~400) and makes any throughput measurement taken through it \
meaningless. Rebuild the ROOT facade with CUDA: `cargo build --release \
--features cuda` (build the root, not `-p apr-cli`: BOTH packages define a \
binary named `apr`, and only the root's cuda = [\"cli\", \"apr-cli/cuda\"] \
chain enables this path). To run on this build anyway, pass `--backend cpu` \
or drop `--backend`."
                        .to_string(),
                )));
            }
            // PERF-021: `apr run` is the surface #2696 was MEASURED through —
            // 15.7 tok/s decode, 0.099x llama.cpp — and it was the surface with
            // no guard. The jidoka refusal landed only on `apr serve`, one
            // command over from where the defect was recorded.
            //
            // Placed ABOVE `effective_no_gpu` and above the `batch_jsonl` early
            // return below: that return bypasses `dispatch_run` entirely, so a
            // check any lower is skipped by `apr run --gpu --batch-jsonl f.jsonl`.
            if let Err(e) = crate::accel::ensure_available(
                *gpu && !*no_gpu,
                &crate::accel::asked_flag(*gpu, backend.as_deref()),
            ) {
                return Some(Err(e));
            }

            // GH-326: --gpu overrides --no-gpu when both specified
            let effective_no_gpu = if *gpu {
                false
            } else {
                *no_gpu || backend_forces_cpu
            };

            // PERF-062 / #2790: LATCH THE REQUEST BEFORE IT IS COLLAPSED.
            //
            // The line above is where the fact dies: after it, `--gpu` and a
            // bare `apr run` are the same `false`, so nothing downstream can
            // report that an accelerator was asked for and refused. That is
            // why the F2 fallback was unreportable rather than merely
            // unreported — see `crate::compute_latch`.
            #[cfg(feature = "inference")]
            crate::compute_latch::latch_request(realizar::infer::ComputeRequest::from_flags(
                *gpu,
                *no_gpu,
                backend.as_deref(),
            ));

            // Batch JSONL mode: load model once, process all prompts
            #[cfg(feature = "inference")]
            if let Some(ref batch_file) = batch_jsonl {
                return Some(run::run_batch(
                    source,
                    batch_file,
                    *max_tokens,
                    *temperature,
                    *top_k,
                    effective_no_gpu,
                    *verbose || cli.verbose,
                ));
            }

            // GH-240: merge global --json flag into output format
            let effective_format = if cli.json { "json" } else { format.as_str() };
            dispatch_run(
                source,
                positional_prompt.as_ref(),
                input.as_deref(),
                prompt.as_ref(),
                *max_tokens,
                *stream,
                language.as_deref(),
                task.as_deref(),
                effective_format,
                effective_no_gpu,
                *offline,
                *benchmark,
                *verbose || cli.verbose,
                *trace,
                *trace_payload,
                trace_steps.as_deref(),
                *trace_verbose,
                trace_output.clone(),
                trace_level.as_str(),
                *profile,
                *chat,
                // PMAT-496: Sampling parameters — no longer silently dropped
                *temperature,
                *top_k,
                *top_p,
                *seed,
                *repeat_penalty,
                *repeat_last_n,
                *split_prompt,
            )
        }

        Commands::Serve { command } => dispatch_serve_command(command, cli),

        // PMAT-182: apr code — sovereign coding assistant
        
        Commands::Code {
            model,
            project,
            resume,
            prompt,
            print,
            max_turns,
            manifest,
            emit_trace,
            output_format,
            input_format,
        } => dispatch_code_command(CodeArgs {
            model,
            project,
            resume,
            prompt,
            print: *print,
            max_turns: *max_turns,
            manifest,
            emit_trace,
            output_format: *output_format,
            input_format: *input_format,
        }),

        _ => return None,
    })
}

/// Borrowed view of the parsed `apr code` flags, so [`dispatch_code_command`]
/// takes one argument instead of ten.
struct CodeArgs<'a> {
    model: &'a Option<PathBuf>,
    project: &'a Path,
    resume: &'a Option<Option<String>>,
    prompt: &'a [String],
    print: bool,
    max_turns: u32,
    manifest: &'a Option<PathBuf>,
    emit_trace: &'a Option<PathBuf>,
    output_format: crate::CodeOutputFormat,
    input_format: crate::CodeInputFormat,
}

/// Dispatch `apr code` (PMAT-182): the sovereign coding assistant.
///
/// Split out of `dispatch_runtime_commands` so the start-up guard below does
/// not add to that already-oversized match arm's cognitive complexity.
fn dispatch_code_command(args: CodeArgs<'_>) -> Result<(), CliError> {
    // #2607: `apr code` with NO arguments and a stdin that is not a terminal
    // used to auto-discover the largest local GGUF and spawn an `apr serve`
    // child for it. Print help instead — and print it BEFORE `cmd_code` runs,
    // so nothing is discovered and nothing is spawned. `cmd_code` refuses the
    // same shape on its own (it is a public library API); this exists so the
    // operator sees the real clap help for the subcommand, not a bare error.
    if batuta::agent::code::CodeInvocation::from_args(
        args.prompt,
        args.print,
        args.model.as_ref(),
        args.manifest.as_ref(),
        args.resume.as_ref(),
    )
    .wants_help()
    {
        print_code_help_and_exit();
    }
    batuta::agent::code::cmd_code(
        args.model.clone(),
        args.project.to_path_buf(),
        args.resume.clone(),
        args.prompt.to_vec(),
        args.print,
        args.max_turns,
        args.manifest.clone(),
        args.emit_trace.clone(),
        // PMAT-CODE-OUTPUT-FORMAT-001 / PMAT-CODE-INPUT-FORMAT-001: forward as
        // `&str` so the orchestrate crate need not depend on the apr-cli
        // ValueEnum types.
        match args.output_format {
            crate::CodeOutputFormat::Text => "text",
            crate::CodeOutputFormat::Json => "json",
        },
        match args.input_format {
            crate::CodeInputFormat::Text => "text",
            crate::CodeInputFormat::Json => "json",
        },
    )
    .map_err(|e| CliError::Aprender(e.to_string()))
}

/// #2607: render the real `apr code --help` and leave, without running the
/// agent.
///
/// The exit status is clap's usage-error code (2), the same one
/// `apr code --nonsense` produces, so a script cannot mistake "I did nothing,
/// here is how to use me" for a completed run. Help goes to stderr for the
/// same reason: stdout of `apr code` is the assistant's answer.
fn print_code_help_and_exit() -> ! {
    let mut root = <Cli as clap::CommandFactory>::command();
    if let Some(sub) = root.find_subcommand_mut("code") {
        eprintln!("{}", sub.render_help());
    }
    eprintln!("{}", batuta::agent::code::NO_ARG_NON_INTERACTIVE);
    std::process::exit(2);
}

/// Dispatch `apr debug`: either the file dump or a debug subcommand.
///
/// aprender#2377 finding 3: `embed-viz-lint`'s help documented
/// `apr debug embed-viz` and no such subcommand existed. `file` is now optional
/// because a subcommand supplies its own input, and `apr debug` with neither
/// must REFUSE rather than dump nothing and exit 0.
fn dispatch_debug(
    cli: &Cli,
    file: Option<&Path>,
    action: Option<&DebugCommands>,
    flags: (bool, bool, bool, usize),
) -> Result<(), CliError> {
    if let Some(DebugCommands::EmbedViz {
        model,
        tensor,
        projection,
        seed,
        limit,
        tokens,
        output,
        force,
    }) = action
    {
        return commands::embed_viz::run(&commands::embed_viz::EmbedVizArgs {
            model: model.clone(),
            tensor: tensor.clone(),
            projection: *projection,
            seed: *seed,
            limit: *limit,
            tokens: tokens.clone(),
            output: output.clone(),
            force: *force,
        });
    }
    let file = file.ok_or_else(|| {
        CliError::ValidationFailed(
            "apr debug: needs a model FILE (`apr debug model.apr`) or a subcommand \
             (`apr debug embed-viz --model model.apr`)"
                .to_string(),
        )
    })?;
    let (drama, hex, strings, limit) = flags;
    let (j, verb) = (cli.json, cli.verbose);
    crate::pipe::with_stdin_support(file, |p| {
        debug::run(p, drama, hex, strings, limit, j, verb)
    })
}

/// Dispatch inspection commands: inspect, debug, validate, lint, explain, canary.
#[allow(clippy::many_single_char_names)]
fn dispatch_inspection_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    contract_pre_no_side_effects!();
    contract_pre_idempotent_inspection!();
    contract_pre_idempotent_output!();
    let result = match cli.command.as_ref() {
        Commands::Inspect {
            file,
            vocab,
            filters,
            weights,
            json,
            quality,
        } => {
            let (v, f, w, j, q) = (*vocab, *filters, *weights, *json || cli.json, *quality);
            crate::pipe::with_stdin_support(file, |p| inspect::run(p, v, f, w, j, q))
        }

        // GH-685: forward cli.verbose to debug
        Commands::Debug {
            file,
            action,
            drama,
            hex,
            strings,
            limit,
        } => dispatch_debug(
            cli,
            file.as_deref(),
            action.as_ref(),
            (*drama, *hex, *strings, *limit),
        ),

        Commands::Validate {
            file,
            quality,
            strict,
            min_score,
        } => {
            let (q, s, ms, j, sc) = (*quality, *strict, *min_score, cli.json, cli.skip_contract);
            crate::pipe::with_stdin_support(file, |p| validate::run(p, q, s, ms, j, sc))
        }

        Commands::ValidateManifest {
            file,
            artifact,
            live,
        } => {
            let live_check = *live && !cli.offline;
            validate_manifest::run(file, artifact.as_deref(), cli.json, live_check)
        }

        Commands::Lint { file, strict } => {
            let (j, q, st) = (cli.json, cli.quiet, *strict);
            crate::pipe::with_stdin_support(file, |p| lint::run(p, j, q, st))
        }

        Commands::BeatRun { contract, measured } => {
            commands::beat_run::run(contract, *measured, cli.json)
        }

        Commands::Manifest { files, output } => {
            // CRUX-G-05 — SHA-256 manifest of the input file set.
            commands::manifest::run(files, output, cli.json)
        }
        Commands::Explain {
            code_or_file,
            file,
            tensor,
            kernel,
            json,
            verbose,
            proof_status,
        } => explain::run(
            code_or_file.clone(),
            file.clone(),
            tensor.as_deref(),
            *kernel,
            *json || cli.json,
            *verbose || cli.verbose,
            *proof_status,
        ),
        Commands::Canary { command } => canary::run(command.clone()),

        _ => return None,
    };
    contract_post_no_side_effects!(&());
    contract_post_idempotent_output!(&());
    Some(result)
}

/// Dispatch diagnostic commands: trace, tensors, diff.
fn dispatch_diagnostic_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    Some(match cli.command.as_ref() {
        Commands::Trace {
            file,
            layer,
            reference,
            json,
            verbose,
            payload,
            diff,
            interactive,
            save_tensor,
            save_tensor_dir,
            save_tensor_layers,
        } => crate::error::resolve_model_path(file).and_then(|r| {
            // SHIP-007 layer-0 stage diff: when --save-tensor is set on a
            // .apr file, dispatch to the end-to-end save-tensor wrapper
            // (PR-A clap → PR-B plan → PR-C-real step1+2 wrapper). For
            // .gguf/.safetensors and the common no-flag case, fall through
            // to the existing trace path.
            #[cfg(feature = "inference")]
            if let Some(stages) = save_tensor.as_deref() {
                let ext_lower = r
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase);
                match ext_lower.as_deref() {
                    Some("apr") => {
                        return crate::commands::trace_save_tensor::run_save_tensor_apr(
                            &r,
                            stages,
                            save_tensor_dir.as_deref(),
                            save_tensor_layers,
                        );
                    }
                    Some("gguf") => {
                        // M-MOE-SUB-2 step (a) CLI completion: GGUF dispatches
                        // to the MoE-traced wireup if the arch is qwen3_moe;
                        // dense-GGUF will be wired in SHIP-007 PR-E.
                        return crate::commands::trace_save_tensor::run_save_tensor_gguf_moe(
                            &r,
                            stages,
                            save_tensor_dir.as_deref(),
                            save_tensor_layers,
                        );
                    }
                    _ => {
                        eprintln!(
                            "apr trace --save-tensor: only .apr and .gguf (qwen3_moe arch) \
                             supported today; .safetensors will be wired in SHIP-007 PR-E \
                             (got {})",
                            r.display()
                        );
                    }
                }
            }
            trace::run(
                &r,
                layer.as_deref(),
                reference.as_deref(),
                *json || cli.json,
                *verbose || cli.verbose,
                *payload,
                *diff,
                *interactive,
            )
        }),

        Commands::Tensors {
            file,
            stats,
            filter,
            limit,
            json,
        } => {
            let (s, f, j, l) = (
                *stats,
                filter.as_deref().map(str::to_owned),
                *json || cli.json,
                *limit,
            );
            crate::pipe::with_stdin_support(file, |p| tensors::run(p, s, f.as_deref(), j, l))
        }

        Commands::Diff {
            file1,
            file2,
            weights,
            values,
            filter,
            limit,
            transpose_aware,
            json,
            quant_roundtrip,
            threshold,
            no_threshold,
        } => {
            if *quant_roundtrip {
                // CRUX-B-20: per-tensor quant roundtrip error report.
                crate::error::resolve_model_path(file1).and_then(|r1| {
                    crate::error::resolve_model_path(file2).and_then(|r2| {
                        dispatch_quant_roundtrip(
                            &r1,
                            &r2,
                            *threshold,
                            *no_threshold,
                            *json || cli.json,
                        )
                    })
                })
            } else {
                crate::error::resolve_model_path(file1).and_then(|r1| {
                    crate::error::resolve_model_path(file2).and_then(|r2| {
                        diff::run(
                            &r1,
                            &r2,
                            *weights,
                            *values,
                            filter.as_deref(),
                            *limit,
                            *transpose_aware,
                            *json || cli.json,
                        )
                    })
                })
            }
        }

        _ => return None,
    })
}

/// CRUX-B-20 — render an `apr diff --quant-roundtrip` report.
fn dispatch_quant_roundtrip(
    reference: &std::path::Path,
    quantized: &std::path::Path,
    threshold: f32,
    no_threshold: bool,
    json: bool,
) -> Result<(), CliError> {
    use commands::diff_quant_roundtrip::{build_report, render_tsv};

    // GH-2391: `any_below_threshold` is an OR of `cosine < threshold`. Against a
    // NaN threshold every term is false, so the CRUX-B-20 exit-code gate reports
    // a clean roundtrip for a quantization it never checked.
    commands::threshold_arg::guard_f32("--threshold", threshold, commands::threshold_arg::COSINE)?;

    let report = build_report(reference, quantized, threshold)?;

    if json {
        let serialized = serde_json::to_string_pretty(&report).map_err(|e| {
            CliError::ValidationFailed(format!("serialize quant-roundtrip JSON: {e}"))
        })?;
        println!("{serialized}");
    } else {
        print!("{}", render_tsv(&report));
    }

    // Threshold gate per contract crux-B-20 invariant
    // "exit code ≠ 0 if any tensor cosine < 0.95 (unless --no-threshold)".
    if report.any_below_threshold && !no_threshold {
        return Err(CliError::ValidationFailed(format!(
            "quant-roundtrip: at least one tensor below cosine threshold {threshold}",
        )));
    }
    Ok(())
}

/// Dispatch format operation commands: export, import, convert, quantize.
fn dispatch_format_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    Some(match cli.command.as_ref() {
        Commands::Export {
            file,
            format,
            output,
            quantize,
            list_formats,
            batch,
            json,
            plan,
            force,
        } => {
            match file
                .as_ref()
                .map(|f| crate::error::resolve_model_path(f))
                .transpose()
            {
                Ok(resolved) => export::run(
                    resolved.as_deref(),
                    format,
                    output.as_deref(),
                    quantize.as_deref(),
                    *list_formats,
                    batch.as_deref(),
                    *json || cli.json,
                    *plan,
                    *force,
                ),
                Err(e) => Err(e),
            }
        }
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
            // GH-666: Reject network sources when --offline is set
            if cli.offline
                && (source.starts_with("hf://")
                    || source.starts_with("http://")
                    || source.starts_with("https://"))
            {
                return Some(Err(crate::error::CliError::NetworkError(format!(
                    "Cannot import from '{}' in --offline mode. Use a local file path.",
                    source
                ))));
            }
            import::run(
                source,
                output.as_deref(),
                Some(arch.as_str()),
                quantize.as_deref(),
                *strict,
                *preserve_q4k,
                tokenizer.as_ref(),
                *enforce_provenance,
                *allow_no_config,
                cli.json,
            )
        }
        Commands::Convert {
            file,
            quantize,
            compress,
            output,
            force,
        } => crate::error::resolve_model_path(file).and_then(|r| {
            convert::run(
                &r,
                quantize.as_deref(),
                compress.as_deref(),
                output,
                *force,
                cli.json,
            )
        }),
        Commands::Stamp {
            file,
            license,
            data_source,
            data_license,
            hf_architecture,
            hf_model_type,
            architecture,
            tokenizer_dir,
            output,
            force,
        } => crate::error::resolve_model_path(file).and_then(|r| {
            stamp::run(
                &r,
                license.as_deref(),
                data_source.as_deref(),
                data_license.as_deref(),
                hf_architecture.as_deref(),
                hf_model_type.as_deref(),
                architecture.as_deref(),
                tokenizer_dir.as_deref(),
                output,
                *force,
                cli.json,
            )
        }),
        Commands::Compile {
            file,
            output,
            target,
            quantize,
            release,
            strip,
            lto,
            list_targets,
        } => {
            match file
                .as_ref()
                .map(|f| crate::error::resolve_model_path(f))
                .transpose()
            {
                Ok(resolved) => compile::run(
                    resolved.as_deref(),
                    output.as_deref(),
                    target.as_deref(),
                    quantize.as_deref(),
                    *release,
                    *strip,
                    *lto,
                    *list_targets,
                    cli.json,
                ),
                Err(e) => Err(e),
            }
        }
        Commands::Quantize {
            file,
            scheme,
            output,
            format,
            batch,
            plan,
            force,
        } => crate::error::resolve_model_path(file).and_then(|r| {
            quantize::run(
                &r,
                scheme,
                output.as_deref(),
                format.as_deref(),
                batch.as_deref(),
                *plan,
                *force,
                cli.json,
            )
        }),

        _ => return None,
    })
}

/// Dispatch model management commands: merge, finetune, prune, distill, pull, list, rm, tui.
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
fn dispatch_model_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    contract_pre_output_path_validation!();
    contract_pre_rm_confirmation_gate!();
    Some(match cli.command.as_ref() {
        Commands::Merge {
            files,
            strategy,
            output,
            weights,
            base_model,
            drop_rate,
            density,
            seed,
            plan,
            force,
        } => {
            let resolved: std::result::Result<Vec<std::path::PathBuf>, _> = files
                .iter()
                .map(|f| crate::error::resolve_model_path(f))
                .collect();
            match resolved {
                Ok(r) => merge::run(
                    &r,
                    strategy,
                    output.as_deref(),
                    weights.clone(),
                    base_model.clone(),
                    *drop_rate,
                    *density,
                    *seed,
                    cli.json,
                    *plan,
                    *force,
                ),
                Err(e) => Err(e),
            }
        }
        #[cfg(feature = "training")]
        Commands::Gpu { json } => gpu::run(*json || cli.json),
        #[cfg(feature = "training")]
        Commands::ModelOps(ModelOpsCommands::Finetune {
            file,
            method,
            rank,
            vram,
            plan,
            data,
            output,
            adapter,
            merge,
            epochs,
            learning_rate,
            model_size,
            task,
            num_classes,
            checkpoint_format,
            oversample,
            max_seq_len,
            quantize_nf4,
            gpus,
            gpu_backend,
            role,
            bind,
            coordinator,
            expect_workers,
            wait_gpu,
            adapters,
            adapters_config,
            experimental_mps,
            gpu_share,
            profile,
        }) => {
            if *profile {
                eprintln!("StepProfiler enabled for finetune (PMAT-486)");
            }
            finetune::run(
                file.as_deref(),
                method,
                *rank,
                *vram,
                *plan,
                data.as_deref(),
                output.as_deref(),
                adapter.as_deref(),
                *merge,
                *epochs,
                *learning_rate,
                model_size.as_deref(),
                task.as_deref(),
                *num_classes,
                checkpoint_format,
                *oversample,
                *max_seq_len,
                *quantize_nf4,
                gpus.as_deref(),
                gpu_backend,
                role.as_deref(),
                bind.as_deref(),
                coordinator.as_deref(),
                *expect_workers,
                *wait_gpu,
                adapters,
                adapters_config.as_deref(),
                cli.json,
                *experimental_mps,
                *gpu_share,
            )
        }
        Commands::ModelOps(ModelOpsCommands::Prune {
            file,
            method,
            target_ratio,
            sparsity,
            output,
            remove_layers,
            analyze,
            plan,
            calibration,
        }) => crate::error::resolve_model_path(file).and_then(|r| {
            prune::run(
                &r,
                method,
                *target_ratio,
                *sparsity,
                output.as_deref(),
                remove_layers.as_deref(),
                *analyze,
                *plan,
                calibration.as_deref(),
                cli.json,
            )
        }),
        Commands::ModelOps(ModelOpsCommands::Distill {
            teacher,
            student,
            data,
            output,
            strategy,
            temperature,
            alpha,
            epochs,
            plan,
            config,
            stage,
            backend,
            dataset,
        }) => distill::run(
            teacher.as_deref(),
            student.as_deref(),
            data.as_deref(),
            output.as_deref(),
            strategy,
            *temperature,
            *alpha,
            *epochs,
            *plan,
            config.as_deref(),
            stage.as_deref(),
            backend.as_str(),
            dataset.as_deref(),
            cli.json,
        ),
        Commands::Pull {
            model_ref,
            repo,
            force,
            dry_run,
            revision,
            offline,
            include,
            output,
            verify,
        } => {
            // CRUX-A-20: `pull` declares its OWN `--offline` in addition to the
            // clap global, and the dataset / `--verify` branches below never
            // enter `pull::run`. Arm the enforcement scope here, around all
            // three branches, from the variant's own flag — so the refusal does
            // not depend on clap continuing to populate the global as well.
            let _offline_scope = crate::commands::offline::scope(*offline);
            // SHIP-TWO-001 §26.8: when first positional is the literal
            // "dataset", treat the second positional as the HF dataset
            // repo and dispatch to the dataset puller. Otherwise fall
            // through to the existing model puller (backward compat).
            if model_ref == "dataset" {
                match repo.as_deref() {
                    // Issue #1410 / FALSIFY-PULL-DATASET-009: thread `dry_run`
                    // through to the dataset puller. Previously dropped on
                    // the floor, so `apr pull dataset --dry-run` performed
                    // full downloads in violation of the contract.
                    Some(r) => pull::run_dataset(
                        r,
                        include,
                        revision.as_deref(),
                        output.as_deref(),
                        *dry_run,
                    ),
                    None => Err(crate::error::CliError::ValidationFailed(
                        "apr pull dataset <REPO>: REPO argument required".to_string(),
                    )),
                }
            } else if *verify {
                // Verify-only: no network I/O, no download. Resolves the cache
                // directory for the reference and re-hashes what is on disk.
                crate::commands::pull::resolve_cache_dir_for_ref(model_ref)
                    .and_then(|dir| crate::commands::pull_verify::run_verify(&dir))
            } else {
                pull::run(
                    model_ref,
                    *force,
                    *dry_run,
                    revision.as_deref(),
                    *offline,
                    cli.json,
                )
            }
        }
        Commands::Registry { command } => crate::commands::registry::run(command.clone()),
        Commands::List => pull::list(cli.json, cli.quiet),
        Commands::Rm { model_ref } => pull::remove(model_ref, cli.json),
        Commands::Tui { file } => tui::run(file.clone()),
        Commands::Mcp {} => mcp::run(),

        _ => return None,
    })
}
