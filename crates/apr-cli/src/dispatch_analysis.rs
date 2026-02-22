/// Dispatch analysis commands (cbtop, probar, compare-hf, hex, tree, flow, oracle).
///
/// Returns `None` if the command is not an analysis command, allowing the caller
/// to try other sub-dispatchers.
fn dispatch_analysis_commands(cli: &Cli) -> Option<Result<(), CliError>> {
    let Commands::Extended(ref ext) = *cli.command.as_ref() else {
        return None;
    };
    let result = match ext {
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

        _ => return None,
    };
    Some(result)
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
        } => eval::run(
            file,
            dataset,
            text.as_deref(),
            Some(*max_tokens),
            Some(*threshold),
            cli.json,
        ),

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
        } => tune::run(
            file.as_deref(),
            method.parse().unwrap_or(tune::TuneMethod::Auto),
            *rank,
            *vram,
            *plan,
            model.as_deref(),
            *freeze_base,
            train_data.as_deref(),
            *json || cli.json,
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
        }) => publish::execute(
            directory,
            repo_id,
            model_name.as_deref(),
            license,
            pipeline_tag,
            library_name.as_deref(),
            tags.as_ref().map_or(&[], std::vec::Vec::as_slice),
            message.as_deref(),
            *dry_run,
            cli.verbose,
        ),

        // All other extended commands handled by sub-dispatchers above
        _ => unreachable!("all extended commands handled by sub-dispatchers"),
    }
}
