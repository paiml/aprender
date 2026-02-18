
/// Dispatch core commands (run, serve, inspection, format operations).
///
/// Delegates to sub-dispatchers to keep cyclomatic complexity below 10 per function.
fn dispatch_core_command(cli: &Cli) -> Option<Result<(), CliError>> {
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
    dispatch_model_commands(cli)
}

/// Dispatch runtime commands: check, run, serve.
fn dispatch_runtime_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    Some(match cli.command.as_ref() {
        Commands::Check { file, no_gpu, json } => commands::check::run(file, *no_gpu, *json),
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
            gpu: _,
            offline,
            benchmark,
            trace,
            trace_steps,
            trace_verbose,
            trace_output,
            trace_level,
            trace_payload,
            profile,
            chat,
            verbose,
        } => {
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
                *no_gpu,
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
            )
        }

        Commands::Serve {
            file,
            port,
            host,
            no_cors,
            no_metrics,
            no_gpu,
            gpu,
            batch,
            trace,
            trace_level,
            profile,
        } => dispatch_serve(
            file,
            *port,
            host,
            *no_cors,
            *no_metrics,
            *no_gpu,
            *gpu,
            *batch,
            *trace,
            trace_level,
            *profile,
            cli.verbose,
        ),

        _ => return None,
    })
}

/// Dispatch inspection commands: inspect, debug, validate, lint, explain, canary.
#[allow(clippy::many_single_char_names)]
fn dispatch_inspection_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    Some(match cli.command.as_ref() {
        Commands::Inspect {
            file,
            vocab,
            filters,
            weights,
            json,
        } => {
            let (v, f, w, j) = (*vocab, *filters, *weights, *json || cli.json);
            crate::pipe::with_stdin_support(file, |p| inspect::run(p, v, f, w, j))
        }

        Commands::Debug {
            file,
            drama,
            hex,
            strings,
            limit,
        } => {
            let (d, h, s, l, j) = (*drama, *hex, *strings, *limit, cli.json);
            crate::pipe::with_stdin_support(file, |p| debug::run(p, d, h, s, l, j))
        }

        Commands::Validate {
            file,
            quality,
            strict,
            min_score,
        } => {
            let (q, s, ms, j) = (*quality, *strict, *min_score, cli.json);
            crate::pipe::with_stdin_support(file, |p| validate::run(p, q, s, ms, j))
        }

        Commands::Lint { file } => {
            let j = cli.json;
            crate::pipe::with_stdin_support(file, |p| lint::run(p, j))
        }
        Commands::Explain { code, file, tensor } => {
            explain::run(code.clone(), file.clone(), tensor.as_deref())
        }
        Commands::Canary { command } => canary::run(command.clone()),

        _ => return None,
    })
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
        } => trace::run(
            file,
            layer.as_deref(),
            reference.as_deref(),
            *json || cli.json,
            *verbose || cli.verbose,
            *payload,
            *diff,
            *interactive,
        ),

        Commands::Tensors {
            file,
            stats,
            filter,
            limit,
            json,
        } => {
            let (s, f, j, l) = (*stats, filter.as_deref().map(str::to_owned), *json || cli.json, *limit);
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
        } => diff::run(
            file1,
            file2,
            *weights,
            *values,
            filter.as_deref(),
            *limit,
            *transpose_aware,
            *json || cli.json,
        ),

        _ => return None,
    })
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
        } => export::run(
            file.as_deref(),
            format,
            output.as_deref(),
            quantize.as_deref(),
            *list_formats,
            batch.as_deref(),
            *json || cli.json,
        ),
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
        } => import::run(
            source,
            output.as_deref(),
            Some(arch.as_str()),
            quantize.as_deref(),
            *strict,
            *preserve_q4k,
            tokenizer.as_ref(),
            *enforce_provenance,
            *allow_no_config,
        ),
        Commands::Convert {
            file,
            quantize,
            compress,
            output,
            force,
        } => convert::run(
            file,
            quantize.as_deref(),
            compress.as_deref(),
            output,
            *force,
            cli.json,
        ),
        Commands::Quantize {
            file,
            scheme,
            output,
            format,
            batch,
            plan,
            force,
        } => quantize::run(file, scheme, output.as_deref(), format.as_deref(), batch.as_deref(), *plan, *force, cli.json),

        _ => return None,
    })
}

/// Dispatch model management commands: merge, finetune, prune, distill, pull, list, rm, tui.
fn dispatch_model_commands(cli: &Cli) -> Option<Result<(), CliError>> {
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
        } => merge::run(files, strategy, output, weights.clone(), base_model.clone(), *drop_rate, *density, *seed, cli.json),
        Commands::Finetune {
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
        } => finetune::run(
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
            cli.json,
        ),
        Commands::Prune {
            file,
            method,
            target_ratio,
            sparsity,
            output,
            remove_layers,
            analyze,
            plan,
            calibration,
        } => prune::run(
            file,
            method,
            *target_ratio,
            *sparsity,
            output.as_deref(),
            remove_layers.as_deref(),
            *analyze,
            *plan,
            calibration.as_deref(),
            cli.json,
        ),
        Commands::Distill {
            teacher,
            student,
            data,
            output,
            strategy,
            temperature,
            alpha,
            epochs,
            plan,
        } => distill::run(
            teacher,
            student.as_deref(),
            data.as_deref(),
            output.as_deref(),
            strategy,
            *temperature,
            *alpha,
            *epochs,
            *plan,
            cli.json,
        ),
        Commands::Pull { model_ref, force } => pull::run(model_ref, *force),
        Commands::List => pull::list(cli.json),
        Commands::Rm { model_ref } => pull::remove(model_ref),
        Commands::Tui { file } => tui::run(file.clone()),

        _ => return None,
    })
}
