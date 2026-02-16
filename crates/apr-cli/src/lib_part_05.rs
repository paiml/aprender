
/// Dispatch core commands (run, serve, inspection, format operations).
#[allow(clippy::too_many_lines)]
fn dispatch_core_command(cli: &Cli) -> Option<Result<(), CliError>> {
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
        } => debug::run(file, *drama, *hex, *strings, *limit, cli.json),

        Commands::Validate {
            file,
            quality,
            strict,
            min_score,
        } => {
            let (q, s, ms, j) = (*quality, *strict, *min_score, cli.json);
            crate::pipe::with_stdin_support(file, |p| validate::run(p, q, s, ms, j))
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

        Commands::Lint { file } => {
            let j = cli.json;
            crate::pipe::with_stdin_support(file, |p| lint::run(p, j))
        }
        Commands::Explain { code, file, tensor } => {
            explain::run(code.clone(), file.clone(), tensor.clone())
        }
        Commands::Canary { command } => canary::run(command.clone()),
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
        Commands::Pull { model_ref, force } => pull::run(model_ref, *force),
        Commands::List => pull::list(cli.json),
        Commands::Rm { model_ref } => pull::remove(model_ref),
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
        ),
        Commands::Merge {
            files,
            strategy,
            output,
            weights,
            base_model,
            drop_rate,
            density,
            seed,
        } => merge::run(files, strategy, output, weights.clone(), base_model.clone(), *drop_rate, *density, *seed),
        Commands::Quantize {
            file,
            scheme,
            output,
            format,
            batch,
            plan,
            force,
        } => quantize::run(file, scheme, output.as_deref(), format.as_deref(), batch.as_deref(), *plan, *force, cli.json),
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
        Commands::Tui { file } => tui::run(file.clone()),
        _ => return None,
    })
}
