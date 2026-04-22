// Profiling / QA dispatch (profile, bench, eval, qa, parity, ptx, ptx-map, tune)
// extracted from `dispatch_analysis.rs` to keep the PMAT-689 file-size invariant.
//
// Inlined into `dispatch_analysis.rs` via `include!()`, so items share the parent
// module's imports and file-private scope.

/// Dispatch profiling and QA commands (profile, bench, eval, qa, parity, ptx, ptx-map, tune).
///
/// Returns `None` if the command is not a profiling command, allowing the caller
/// to try other sub-dispatchers.
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "side_effect_classification")]
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
        } => {
            crate::error::resolve_model_path(file).and_then(|r| {
                dispatch_profile(
                    &r,
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
                )
            })
        }

        ExtendedCommands::Bench {
            file,
            warmup,
            iterations,
            max_tokens,
            prompt,
            fast,
            brick,
            percentiles,
        } => crate::error::resolve_model_path(file).and_then(|r| {
            bench::run(
                &r,
                *warmup,
                *iterations,
                *max_tokens,
                prompt.as_deref(),
                *fast,
                brick.as_deref(),
                cli.json,
                percentiles,
            )
        }),

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
            device,
            samples,
            temperature,
        } => crate::error::resolve_model_path(file).and_then(|r| {
            match task.as_deref() {
                #[cfg(feature = "training")]
                Some("classify") => eval::run_classify_eval(
                    &r,
                    data.as_deref(),
                    model_size.as_deref(),
                    *num_classes,
                    *generate_card,
                    cli.json,
                ),
                Some("code") => {
                    eval::run_code_eval(&r, data.as_deref(), *max_tokens, *threshold, cli.json)
                }
                Some("humaneval") => eval::run_humaneval(
                    &r,
                    data.as_deref(),
                    &[1, 10, 100],
                    cli.json,
                    device,
                    *samples,
                    *temperature,
                ),
                Some("mbpp") => eval::run_mbpp(
                    &r,
                    data.as_deref(),
                    &[1, 10, 100],
                    cli.json,
                    device,
                    *samples,
                    *temperature,
                ),
                Some("contamination") => eval::run_contamination(
                    &r,
                    data.as_deref(),
                    None,
                    *threshold / 100.0,
                    cli.json,
                ),
                Some("compare") => eval::run_compare(&r, data.as_deref(), None, cli.json),
                Some("verify") => eval::run_verify(&r, cli.json),
                Some("correlation") => eval::run_correlation(&r, data.as_deref(), cli.json),
                Some("human") => eval::run_human_eval(&r, data.as_deref(), cli.json),
                Some("plan") => eval::run_eval_plan(
                    &r,
                    dataset,
                    data.as_deref(),
                    *max_tokens,
                    *threshold,
                    cli.json,
                ),
                _ => eval::run(
                    &r,
                    dataset,
                    text.as_deref(),
                    Some(*max_tokens),
                    Some(*threshold),
                    cli.json,
                ),
            }
        }),

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
        // GH-636: pass cli.json to parity — was dropping the flag
        } => crate::error::resolve_model_path(file)
            .and_then(|r| commands::parity::run(&r, prompt, *assert, cli.verbose, cli.json)),

        ExtendedCommands::PtxMap {
            file,
            kernel,
            reverse,
            json,
            verbose,
            prefill,
        } => crate::error::resolve_model_path(file).and_then(|r| {
            commands::ptx_map::run(
                &r,
                kernel.as_deref(),
                reverse.as_deref(),
                *json || cli.json,
                *verbose || cli.verbose,
                *prefill,
            )
        }),

        ExtendedCommands::Ptx {
            file,
            kernel,
            strict,
            bugs,
            json,
            verbose,
        } => {
            match file
                .as_ref()
                .map(|f| crate::error::resolve_model_path(f))
                .transpose()
            {
                Ok(resolved) => {
                    #[cfg(feature = "full")]
                    { commands::ptx_explain::run(
                        resolved.as_deref(),
                        kernel.as_deref(),
                        *strict,
                        *bugs,
                        *json || cli.json,
                        *verbose || cli.verbose,
                    ) }
                    #[cfg(not(feature = "full"))]
                    { Err(CliError::Aprender("ptx command requires --features full".into())) }
                }
                Err(e) => Err(e),
            }
        }

        #[cfg(feature = "training")]
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
        } => dispatch_tune_command(
            file.as_deref(),
            method,
            *rank,
            *vram,
            *plan,
            model.as_deref(),
            *freeze_base,
            train_data.as_deref(),
            *json || cli.json,
            task.as_deref(),
            *budget,
            strategy,
            scheduler,
            *scout,
            data.as_deref(),
            *num_classes,
            model_size.as_deref(),
            from_scout.as_deref(),
            *max_epochs,
            time_limit.as_deref(),
        ),

        _ => return None,
    };
    Some(result)
}
