/// Dispatch analysis commands (cbtop, probar, compare-hf, hex, tree, flow, oracle).
///
/// Returns `None` if the command is not an analysis command, allowing the caller
/// to try other sub-dispatchers.
fn dispatch_analysis_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    let Commands::Extended(ref ext) = *cli.command.as_ref() else {
        return None;
    };
    let result = match ext {
        ExtendedCommands::Monitor {
            dir,
            refresh_ms,
            compact,
            json,
            format,
        } => commands::monitor::run(dir.as_deref(), *refresh_ms, *compact, *json, format),

        ExtendedCommands::Runs { command } => match command {
            RunsCommands::Ls {
                dir, global, status, json, limit,
            } => commands::runs::run_ls(dir, *global, status, *json, *limit),
            RunsCommands::Show {
                run_id, dir, global, json,
            } => commands::runs::run_show(run_id, dir, *global, *json),
        },

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
            *json,
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

        ExtendedCommands::Probar {
            file,
            output,
            format,
            golden,
            layer,
        } => probar::run(
            file,
            output,
            format.parse().unwrap_or(probar::ExportFormat::Both),
            golden.as_deref(),
            layer.as_deref(),
        ),

        ExtendedCommands::CompareHf {
            file,
            hf,
            tensor,
            threshold,
            json,
        } => compare_hf::run(file, hf, tensor.as_deref(), *threshold, *json || cli.json),

        ExtendedCommands::Hex {
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
        } => dispatch_hex(
            file,
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
        ),

        ExtendedCommands::Tree {
            file,
            filter,
            format,
            sizes,
            depth,
        } => {
            // GH-248: Global --json flag overrides tree format
            let tree_format = if cli.json {
                tree::TreeFormat::Json
            } else {
                format.parse().unwrap_or(tree::TreeFormat::Ascii)
            };
            tree::run(file, filter.as_deref(), tree_format, *sizes, *depth)
        }

        ExtendedCommands::Flow {
            file,
            layer,
            component,
            verbose,
            json,
        } => flow::run(
            file,
            layer.as_deref(),
            component.parse().unwrap_or(flow::FlowComponent::Full),
            *verbose || cli.verbose,
            *json || cli.json,
        ),

        ExtendedCommands::Qualify {
            file,
            tier,
            timeout,
            json,
            verbose,
            skip,
        } => qualify::run(
            file,
            tier,
            *timeout,
            *json || cli.json,
            *verbose || cli.verbose,
            skip.as_deref(),
        ),

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

        ExtendedCommands::Train { command } => dispatch_train_command(command, cli),

        ExtendedCommands::Tokenize { command } => dispatch_tokenize_command(command, cli),

        ExtendedCommands::Data { command } => dispatch_data_command(command, cli.json),

        ExtendedCommands::Pipeline { command } => dispatch_pipeline_command(command, cli),

        ExtendedCommands::Diagnose {
            checkpoint_dir,
            data,
            model_size,
            num_classes,
        } => diagnose::run(
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

/// Dispatch `apr data` subcommands to alimentar-backed implementations.
fn dispatch_data_command(command: &DataCommands, json: bool) -> std::result::Result<(), CliError> {
    match command {
        DataCommands::Audit {
            file,
            num_classes,
            input_column,
            label_column,
            preamble_prefix,
        } => data::run_audit(
            file,
            *num_classes,
            input_column,
            label_column,
            preamble_prefix.as_deref(),
            json,
        ),
        DataCommands::Split {
            file,
            train,
            val,
            test,
            label_column,
            seed,
            output,
        } => data::run_split(
            file,
            label_column,
            *train,
            *val,
            *test,
            *seed,
            output,
            json,
        ),
        DataCommands::Balance {
            file,
            strategy,
            label_column,
            num_classes,
            seed,
            output,
        } => data::run_balance(
            file,
            label_column,
            strategy,
            *num_classes,
            *seed,
            output.as_deref(),
            json,
        ),
    }
}

/// Dispatch `apr train` subcommands to entrenar-backed implementations.
fn dispatch_train_command(command: &TrainCommands, cli: &Cli) -> std::result::Result<(), CliError> {
    match command {
        TrainCommands::Plan {
            data,
            model_size,
            model_path,
            num_classes,
            task,
            config,
            output,
            strategy,
            budget,
            scout,
            max_epochs,
            learning_rate,
            lora_rank,
            batch_size,
            val_data,
            test_data,
            format,
        } => train::run_plan(
            data.as_deref(),
            model_size,
            model_path.as_deref(),
            *num_classes,
            task,
            config.as_deref(),
            output,
            strategy,
            *budget,
            *scout,
            *max_epochs,
            *learning_rate,
            *lora_rank,
            *batch_size,
            val_data.as_deref(),
            test_data.as_deref(),
            format,
            cli.json,
        ),
        TrainCommands::Apply {
            plan,
            config,
            task,
            data,
            model_size,
            model_path,
            num_classes,
            output,
            strategy,
            budget,
            scout,
            max_epochs,
            learning_rate,
            lora_rank,
            batch_size,
        } => train::run_apply(
            plan.as_deref(),
            config.as_deref(),
            task,
            data.as_deref(),
            model_size,
            model_path.as_deref(),
            *num_classes,
            output,
            strategy,
            *budget,
            *scout,
            *max_epochs,
            *learning_rate,
            *lora_rank,
            *batch_size,
            cli.json,
        ),
    }
}

/// Dispatch `apr tokenize` subcommands.
fn dispatch_tokenize_command(
    command: &TokenizeCommands,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    match command {
        TokenizeCommands::Plan {
            data,
            vocab_size,
            algorithm,
            output,
            format,
        } => tokenize::run_plan(data, *vocab_size, algorithm, output, format, cli.json),
        TokenizeCommands::Apply {
            data,
            vocab_size,
            algorithm,
            output,
            max_lines,
        } => tokenize::run_apply(data, *vocab_size, algorithm, output, *max_lines, cli.json),
    }
}

/// Dispatch `apr pipeline` subcommands — wraps forjar DAG engine.
fn dispatch_pipeline_command(
    command: &PipelineCommands,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    match command {
        PipelineCommands::Plan {
            manifest,
            machine,
            tag,
            cost,
        } => pipeline::run_plan(
            manifest,
            machine.as_deref(),
            tag.as_deref(),
            *cost,
            cli.json,
        ),
        PipelineCommands::Apply {
            manifest,
            machine,
            tag,
            parallel,
            keep_going,
        } => pipeline::run_apply(
            manifest,
            machine.as_deref(),
            tag.as_deref(),
            *parallel,
            *keep_going,
            cli.json,
        ),
        PipelineCommands::Status { manifest } => pipeline::run_status(manifest, cli.json),
        PipelineCommands::Validate { manifest } => pipeline::run_validate(manifest, cli.json),
    }
}

/// Dispatch profiling and QA commands (profile, bench, eval, qa, parity, ptx, ptx-map, tune).
///
/// Returns `None` if the command is not a profiling command, allowing the caller
/// to try other sub-dispatchers.
fn dispatch_profiling_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    let Commands::Extended(ref ext) = *cli.command.as_ref() else {
        return None;
    };
    let result = match ext {
        ExtendedCommands::Profile {
            file,
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
            tokens,
            ollama,
            no_gpu,
            compare,
        } => dispatch_profile(
            file,
            *granular,
            format,
            focus.as_deref(),
            *detect_naive,
            *threshold,
            compare_hf.as_deref(),
            *energy,
            *perf_grade,
            *callgraph,
            *fail_on_naive,
            output.as_deref(),
            *ci,
            *assert_throughput,
            *assert_p99,
            *assert_p50,
            *warmup,
            *measure,
            *tokens,
            *ollama,
            *no_gpu,
            compare.as_deref(),
        ),

        ExtendedCommands::Bench {
            file,
            warmup,
            iterations,
            max_tokens,
            prompt,
            fast,
            brick,
        } => bench::run(
            file,
            *warmup,
            *iterations,
            *max_tokens,
            prompt.as_deref(),
            *fast,
            brick.as_deref(),
            cli.json,
        ),

        ExtendedCommands::Eval {
            file,
            dataset,
            text,
            max_tokens,
            threshold,
            task,
            data,
            model_size,
            num_classes,
            generate_card,
        } => match task.as_deref() {
            Some("classify") => eval::run_classify_eval(
                file,
                data.as_deref(),
                model_size.as_deref(),
                *num_classes,
                *generate_card,
                cli.json,
            ),
            Some("code") => eval::run_code_eval(
                file,
                data.as_deref(),
                *max_tokens,
                *threshold,
                cli.json,
            ),
            Some("plan") => eval::run_eval_plan(
                file,
                dataset,
                data.as_deref(),
                *max_tokens,
                *threshold,
                cli.json,
            ),
            _ => eval::run(
                file,
                dataset,
                text.as_deref(),
                Some(*max_tokens),
                Some(*threshold),
                cli.json,
            ),
        }

        ExtendedCommands::Qa {
            file,
            assert_tps,
            assert_speedup,
            assert_gpu_speedup,
            skip_golden,
            skip_throughput,
            skip_ollama,
            skip_gpu_speedup,
            skip_contract,
            skip_format_parity,
            skip_ptx_parity,
            safetensors_path,
            iterations,
            warmup,
            max_tokens,
            json,
            verbose,
            min_executed,
            previous_report,
            regression_threshold,
            skip_gpu_state,
            skip_metadata,
            skip_capability,
            assert_classifier_head,
        } => qa::run(
            file,
            *assert_tps,
            *assert_speedup,
            *assert_gpu_speedup,
            *skip_golden,
            *skip_throughput,
            *skip_ollama,
            *skip_gpu_speedup,
            *skip_contract,
            *skip_format_parity,
            *skip_ptx_parity,
            safetensors_path.clone(),
            *iterations,
            *warmup,
            *max_tokens,
            *json || cli.json,
            *verbose || cli.verbose,
            *min_executed,
            previous_report.clone(),
            *regression_threshold,
            *skip_gpu_state,
            *skip_metadata,
            *skip_capability,
            *assert_classifier_head,
        ),

        ExtendedCommands::Parity {
            file,
            prompt,
            assert,
        } => commands::parity::run(file, prompt, *assert, cli.verbose),

        ExtendedCommands::PtxMap {
            file,
            kernel,
            reverse,
            json,
            verbose,
            prefill,
        } => commands::ptx_map::run(
            file,
            kernel.as_deref(),
            reverse.as_deref(),
            *json || cli.json,
            *verbose || cli.verbose,
            *prefill,
        ),

        ExtendedCommands::Ptx {
            file,
            kernel,
            strict,
            bugs,
            json,
            verbose,
        } => ptx_explain::run(
            file.as_deref(),
            kernel.as_deref(),
            *strict,
            *bugs,
            *json || cli.json,
            *verbose || cli.verbose,
        ),

        ExtendedCommands::Tune {
            file,
            method,
            rank,
            vram,
            plan,
            model,
            freeze_base,
            train_data,
            json,
            task,
            budget,
            strategy,
            scheduler,
            scout,
            data,
            num_classes,
            model_size,
            from_scout,
            max_epochs,
            time_limit,
        } => {
            // Route to classify tune if --task classify is specified
            if task.as_deref() == Some("classify") {
                tune::run_classify_tune(
                    file.as_deref(),
                    *budget,
                    strategy,
                    scheduler,
                    *scout,
                    data.as_deref().or(train_data.as_deref()),
                    *num_classes,
                    model_size.as_deref().or(model.as_deref()),
                    from_scout.as_deref(),
                    *max_epochs,
                    time_limit.as_deref(),
                    *json || cli.json,
                )
            } else {
                tune::run(
                    file.as_deref(),
                    method.parse().unwrap_or(tune::TuneMethod::Auto),
                    *rank,
                    *vram,
                    *plan,
                    model.as_deref(),
                    *freeze_base,
                    train_data.as_deref(),
                    *json || cli.json,
                )
            }
        }

        _ => return None,
    };
    Some(result)
}

/// Dispatch extended commands (analysis, profiling, QA, benchmarks).
///
/// Delegates to [`dispatch_analysis_commands`] and [`dispatch_profiling_commands`]
/// sub-dispatchers to keep cyclomatic complexity below 10 per function.
fn dispatch_extended_command(cli: &Cli) -> Result<(), CliError> {
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
            gpu: _,
            trace,
            trace_steps,
            trace_verbose,
            trace_output,
            trace_level,
            profile,
        } => chat::run(
            file,
            *temperature,
            *top_p,
            *max_tokens,
            system.as_deref(),
            *inspect,
            *no_gpu,
            *trace,
            trace_steps.as_deref(),
            *trace_verbose,
            trace_output.clone(),
            trace_level.as_str(),
            *profile,
        ),

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
            *json,
            *verbose,
            *quiet,
        ),

        ExtendedCommands::Tools(ToolCommands::Rosetta { action }) => dispatch_rosetta(action, cli.json),

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
        ),

        // All other extended commands handled by sub-dispatchers above
        _ => unreachable!("all extended commands handled by sub-dispatchers"),
    }
}
