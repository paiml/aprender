/// Dispatch analysis commands (cbtop, probar, compare-hf, hex, tree, flow, oracle).
///
/// Returns `None` if the command is not an analysis command, allowing the caller
/// to try other sub-dispatchers.
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "side_effect_classification")]
fn dispatch_analysis_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    let Commands::Extended(ref ext) = *cli.command.as_ref() else {
        return None;
    };
    let result = match ext {
        #[cfg(feature = "training")]
        ExtendedCommands::Monitor {
            dir,
            refresh_ms,
            compact,
            json,
            format,
        } => commands::monitor::run(dir.as_deref(), *refresh_ms, *compact, *json || cli.json, format),

        #[cfg(feature = "training")]
        ExtendedCommands::Runs { command } => dispatch_runs_command(command, cli),
        #[cfg(feature = "training")]
        ExtendedCommands::Experiment { command } => dispatch_experiment_command(command, cli),

        ExtendedCommands::Cbtop {
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
        } => dispatch_cbtop(
            model.as_deref(),
            attach.as_deref(),
            model_path.as_deref(),
            *headless,
            *json || cli.json,
            output.as_deref(),
            *ci,
            *throughput,
            *brick_score,
            *warmup,
            *iterations,
            *speculative,
            *speculation_k,
            draft_model.as_deref(),
            *concurrent,
            *simulated,
        ),

        ExtendedCommands::Forensics(ForensicsCommands::Probar {
            file,
            output,
            format,
            golden,
            layer,
            assert,
            tolerance,
        }) => crate::error::resolve_model_path(file).and_then(|r| {
            probar::run(
                &r,
                output,
                format.parse().unwrap_or(probar::ExportFormat::Both),
                golden.as_deref(),
                layer.as_deref(),
                *assert,
                *tolerance,
            )
        }),

        ExtendedCommands::Forensics(ForensicsCommands::CompareHf {
            file,
            hf,
            tensor,
            threshold,
            json,
        }) => {
            // GH-663: Reject compare-hf in --offline mode (requires HuggingFace download)
            if cli.offline {
                return Some(Err(crate::error::CliError::NetworkError(
                    "Cannot run compare-hf in --offline mode (requires HuggingFace download).".to_string(),
                )));
            }
            crate::error::resolve_model_path(file).and_then(|r| {
                compare_hf::run(&r, hf, tensor.as_deref(), *threshold, *json || cli.json)
            })
        }

        ExtendedCommands::Lint(LintCommands::OllamaChatLint {
            response_file,
            stream,
        }) => commands::ollama_chat::run(response_file, *stream, cli.json),

        ExtendedCommands::Lint(LintCommands::DrySamplingLint { observation_file }) => {
            commands::dry_sampling_lint::run(observation_file, cli.json)
        }

        ExtendedCommands::Lint(LintCommands::AwqLint { observation_file }) => {
            commands::awq_lint::run(commands::awq_lint::AwqLintArgs {
                observation_file: observation_file.to_string_lossy().to_string(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::Lint(LintCommands::OomLint {
            report_file,
            stderr_file,
        }) => commands::oom_lint::run(report_file, stderr_file.as_deref(), cli.json),

        ExtendedCommands::Lint(LintCommands::ToolUseLint { observation_file }) => {
            commands::tool_use_lint::run(observation_file, cli.json)
        }

        ExtendedCommands::Lint(LintCommands::GbnfLint { observation_file }) => {
            commands::gbnf_lint::run(observation_file, cli.json)
        }


        ExtendedCommands::Forensics(ForensicsCommands::Hex {
            file,
            tensor,
            limit,
            stats,
            list,
            json,
            header,
            blocks,
            distribution,
            contract,
            entropy,
            raw,
            offset,
            width,
            slice,
        }) => crate::error::resolve_model_path(file).and_then(|r| {
            dispatch_hex(
                &r,
                tensor.as_deref(),
                *limit,
                *stats,
                *list,
                *json || cli.json,
                *header,
                *blocks,
                *distribution,
                *contract,
                *entropy,
                *raw,
                offset,
                *width,
                slice.as_deref(),
            )
        }),

        ExtendedCommands::Forensics(ForensicsCommands::Tree {
            file,
            filter,
            format,
            sizes,
            depth,
        }) => crate::error::resolve_model_path(file).and_then(|resolved| {
            let tree_format = if cli.json {
                tree::TreeFormat::Json
            } else {
                format.parse().unwrap_or(tree::TreeFormat::Ascii)
            };
            tree::run(&resolved, filter.as_deref(), tree_format, *sizes, *depth)
        }),

        ExtendedCommands::Forensics(ForensicsCommands::Flow {
            file,
            layer,
            component,
            verbose,
            json,
        }) => crate::error::resolve_model_path(file).and_then(|resolved| {
            flow::run(
                &resolved,
                layer.as_deref(),
                component.parse().unwrap_or(flow::FlowComponent::Full),
                *verbose || cli.verbose,
                *json || cli.json,
            )
        }),

        ExtendedCommands::Forensics(ForensicsCommands::Qualify {
            file,
            tier,
            timeout,
            json,
            verbose,
            skip,
        }) => crate::error::resolve_model_path(file).and_then(|resolved| {
            qualify::run(
                &resolved,
                tier,
                *timeout,
                *json || cli.json,
                *verbose || cli.verbose,
                skip.as_deref(),
            )
        }),

        ExtendedCommands::Tools(ToolCommands::Oracle {
            source,
            family,
            size,
            compliance,
            tensors,
            stats,
            explain,
            kernels,
            validate,
            full,
        }) => oracle::run(
            source.as_ref(),
            family.as_ref(),
            size.as_ref(),
            *compliance,
            *tensors,
            cli.json,
            cli.verbose,
            cli.offline,
            oracle::OracleFlags {
                stats: *stats,
                explain: *explain,
                kernels: *kernels,
                validate: *validate,
                full: *full,
            },
        ),

        #[cfg(feature = "training")]
        ExtendedCommands::Training(TrainingCommands::Train { command }) => {
            dispatch_train_command(command, cli)
        }
        #[cfg(feature = "training")]
        ExtendedCommands::Training(TrainingCommands::Pretrain {
            dataset,
            tokenizer,
            run_dir,
            mode,
            lr,
            num_steps,
            warmup_steps,
            batch_size,
            seq_length,
            steps_per_epoch,
            seed,
            target_val_loss,
            vocab_size,
            synthetic,
            device,
            allow_shard_cycle,
        }) => commands::pretrain::run(
            dataset,
            tokenizer,
            run_dir,
            *mode,
            *lr,
            *num_steps,
            *warmup_steps,
            *batch_size,
            *seq_length,
            *steps_per_epoch,
            *seed,
            *target_val_loss,
            *vocab_size,
            *synthetic,
            device,
            *allow_shard_cycle,
            cli.json,
        ),
        ExtendedCommands::Training(TrainingCommands::Tokenize { command }) => {
            dispatch_tokenize_command(command, cli)
        }
        ExtendedCommands::Training(TrainingCommands::Data { command }) => {
            dispatch_data_command(command, cli.json)
        }
        ExtendedCommands::Training(TrainingCommands::Pipeline { command }) => {
            dispatch_pipeline_command(command, cli)
        }

        ExtendedCommands::Training(TrainingCommands::Diagnose {
            checkpoint_dir,
            data,
            model_size,
            num_classes,
        }) => diagnose::run(
            checkpoint_dir,
            data.as_deref(),
            model_size.as_deref(),
            *num_classes,
            cli.json,
        ),

        _ => return None,
    };
    Some(result)
}



/// Dispatch extended commands (analysis, profiling, QA, benchmarks).
///
/// Delegates to [`dispatch_analysis_commands`] and [`dispatch_profiling_commands`]
/// sub-dispatchers to keep cyclomatic complexity below 10 per function.
fn dispatch_extended_command(cli: &Cli) -> Result<(), CliError> {
    contract_pre_feature_gated_dispatch!();
    // Try analysis commands first (cbtop, probar, compare-hf, hex, tree, flow, oracle)
    if let Some(result) = dispatch_analysis_commands(cli) {
        return result;
    }

    // Try profiling/QA commands (profile, bench, eval, qa, parity, ptx, ptx-map, tune)
    if let Some(result) = dispatch_profiling_commands(cli) {
        return result;
    }

    // Remaining extended commands handled directly
    let Commands::Extended(ref ext) = *cli.command.as_ref() else {
        unreachable!("dispatch_core_command handles all non-extended variants");
    };
    match ext {
        ExtendedCommands::Chat {
            file,
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
            backend,
        } => {
            if let Some(ref b) = backend {
                eprintln!("Backend override: {b}");
            }
            // GH-326: --gpu overrides --no-gpu when both specified
            let effective_no_gpu = if *gpu { false } else { *no_gpu };
            chat::run(
            file,
            *temperature,
            *top_p,
            *max_tokens,
            system.as_deref(),
            *inspect,
            effective_no_gpu,
            *trace,
            trace_steps.as_deref(),
            *trace_verbose,
            trace_output.clone(),
            trace_level.as_str(),
            *profile,
        )},

        ExtendedCommands::Tools(ToolCommands::Showcase {
            auto_verify,
            step,
            tier,
            model_dir,
            baseline,
            zram,
            runs,
            gpu,
            json,
            verbose,
            quiet,
        }) => dispatch_showcase(
            *auto_verify,
            step.as_deref(),
            tier,
            model_dir,
            baseline,
            *zram,
            *runs,
            *gpu,
            *json || cli.json,
            *verbose || cli.verbose,
            *quiet,
        ),

        ExtendedCommands::Tools(ToolCommands::Rosetta { action }) => {
            dispatch_rosetta(action, cli.json)
        }

        ExtendedCommands::Tools(ToolCommands::Publish {
            directory,
            repo_id,
            model_name,
            license,
            pipeline_tag,
            library_name,
            tags,
            message,
            dry_run,
            plan,
        }) => publish::execute(
            directory,
            repo_id,
            model_name.as_deref(),
            license,
            pipeline_tag,
            library_name.as_deref(),
            tags.as_ref().map_or(&[], std::vec::Vec::as_slice),
            message.as_deref(),
            *dry_run || *plan,
            cli.verbose,
            None,
            &[],
        ),

        ExtendedCommands::Tools(ToolCommands::Encrypt {
            file,
            output,
            key_file,
            force,
        }) => crate::error::resolve_model_path(file)
            .and_then(|r| eval::run_encrypt(&r, output, key_file.as_deref(), *force, cli.json)),

        ExtendedCommands::Tools(ToolCommands::Decrypt {
            file,
            output,
            key_file,
            force,
        }) => crate::error::resolve_model_path(file)
            .and_then(|r| eval::run_decrypt(&r, output, key_file.as_deref(), *force, cli.json)),

        // All other extended commands handled by sub-dispatchers above
        _ => unreachable!("all extended commands handled by sub-dispatchers"),
    }
}

include!("dispatch_helpers.rs");

include!("dispatch_profiling.rs");
