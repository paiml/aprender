/// Dispatch analysis commands (cbtop, probar, compare-hf, hex, tree, flow, oracle).
///
/// Returns `None` if the command is not an analysis command, allowing the caller
/// to try other sub-dispatchers.
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
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
        } => commands::monitor::run(
            dir.as_deref(),
            *refresh_ms,
            *compact,
            *json || cli.json,
            format,
        ),

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

        // GH-876 Milestone 1: Probar is now a subcommand container.
        // The existing flat-args behavior moved under `apr probar tensor <FILE>`.
        ExtendedCommands::Probar { command } => match command {
            ProbarSubcommand::Tensor {
                file,
                output,
                format,
                golden,
                layer,
                assert,
                tolerance,
            } => crate::error::resolve_model_path(file).and_then(|r| {
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
        },

        ExtendedCommands::CompareHf {
            file,
            hf,
            tensor,
            threshold,
            json,
        } => {
            // GH-663: Reject compare-hf in --offline mode (requires HuggingFace download)
            if cli.offline {
                return Some(Err(crate::error::CliError::NetworkError(
                    "Cannot run compare-hf in --offline mode (requires HuggingFace download)."
                        .to_string(),
                )));
            }
            crate::error::resolve_model_path(file).and_then(|r| {
                compare_hf::run(&r, hf, tensor.as_deref(), *threshold, *json || cli.json)
            })
        }

        ExtendedCommands::OllamaChatLint {
            response_file,
            stream,
        } => commands::ollama_chat::run(response_file, *stream, cli.json),

        ExtendedCommands::OllamaToolsLint {
            response_file,
            request_file,
            stream,
        } => commands::ollama_tools_lint::run(
            response_file,
            request_file.as_deref(),
            *stream,
            cli.json,
        ),

        ExtendedCommands::DrySamplingLint { observation_file } => {
            commands::dry_sampling_lint::run(observation_file, cli.json)
        }

        ExtendedCommands::AwqLint { observation_file } => {
            commands::awq_lint::run(commands::awq_lint::AwqLintArgs {
                observation_file: observation_file.to_string_lossy().to_string(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::Fp8Lint { observation_file } => {
            commands::fp8_lint::run(commands::fp8_lint::Fp8LintArgs {
                observation_file: observation_file.to_string_lossy().into_owned(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::Nf4Lint { observation_file } => {
            commands::nf4_lint::run(commands::nf4_lint::Nf4LintArgs {
                observation_file: observation_file.to_string_lossy().to_string(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::GptqLint { observation_file } => {
            commands::gptq_lint::run(commands::gptq_lint::GptqLintArgs {
                observation_file: observation_file.to_string_lossy().to_string(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::OomLint {
            report_file,
            stderr_file,
        } => commands::oom_lint::run(report_file, stderr_file.as_deref(), cli.json),

        ExtendedCommands::KvTimelineLint {
            timeline_file,
            preempt_threshold,
        } => commands::kv_timeline_lint::run(timeline_file, *preempt_threshold, cli.json),

        ExtendedCommands::GpuMemtraceLint { trace_file } => {
            commands::gpu_memtrace_lint::run(trace_file, cli.json)
        }

        ExtendedCommands::ExplainTokenLint {
            jsonl_file,
            tolerance,
            require_greedy,
        } => commands::explain_token_lint::run(jsonl_file, *tolerance, *require_greedy, cli.json),

        ExtendedCommands::CheckFiniteLint {
            error_file,
            list_file,
            min_layers,
        } => commands::check_finite_lint::run(
            error_file.as_deref(),
            list_file.as_deref(),
            *min_layers,
            cli.json,
        ),

        ExtendedCommands::NcclDiagLint {
            diag_file,
            exit_code,
            require_doc_link,
        } => commands::nccl_diag_lint::run(diag_file, *exit_code, *require_doc_link, cli.json),

        ExtendedCommands::ReactTraceLint {
            trace_file,
            max_iterations,
            require_grammar,
        } => {
            commands::react_trace_lint::run(trace_file, *max_iterations, *require_grammar, cli.json)
        }

        ExtendedCommands::DdpMetricsLint {
            metrics_1gpu_file,
            metrics_ngpu_file,
            world_size,
            scaling_floor,
            loss_tolerance,
        } => commands::ddp_metrics_lint::run(
            metrics_1gpu_file,
            metrics_ngpu_file,
            *world_size,
            *scaling_floor,
            *loss_tolerance,
            cli.json,
        ),

        ExtendedCommands::AudioInspectLint {
            json_file,
            expected_sample_rate,
            expected_channels,
        } => commands::audio_inspect_lint::run(
            json_file,
            *expected_sample_rate,
            *expected_channels,
            cli.json,
        ),

        ExtendedCommands::AttnParityLint {
            parity_file,
            provenance_file,
            head_dim_error_file,
            tol_abs,
            tol_cos,
        } => commands::attn_parity_lint::run(
            parity_file.as_deref(),
            provenance_file.as_deref(),
            head_dim_error_file.as_deref(),
            *tol_abs,
            *tol_cos,
            cli.json,
        ),

        ExtendedCommands::AttnVizLint {
            attn_file,
            html_file,
            expected_heatmaps,
            tolerance,
            epsilon,
        } => commands::attn_viz_lint::run(
            attn_file.as_deref(),
            html_file.as_deref(),
            *expected_heatmaps,
            *tolerance,
            *epsilon,
            cli.json,
        ),

        ExtendedCommands::EmbedVizLint {
            csv_file,
            expected_vocab_size,
            csv_file_b,
        } => commands::embed_viz_lint::run(
            csv_file,
            *expected_vocab_size,
            csv_file_b.as_deref(),
            cli.json,
        ),

        ExtendedCommands::HangTraceLint {
            trace_dir,
            mode,
            world_size,
            exit_code,
            expected_exit_code,
        } => match mode.as_str() {
            "timeout" | "success" => {
                let m = if mode == "timeout" {
                    commands::hang_trace_lint::HangMode::Timeout
                } else {
                    commands::hang_trace_lint::HangMode::Success
                };
                commands::hang_trace_lint::run(
                    trace_dir,
                    m,
                    *world_size,
                    *exit_code,
                    *expected_exit_code,
                    cli.json,
                )
            }
            other => Err(crate::error::CliError::ValidationFailed(format!(
                "apr hang-trace-lint --mode must be `timeout` or `success` (got `{other}`)"
            ))),
        },

        ExtendedCommands::OtlpLint {
            otlp_file,
            require_apr_span,
            require_genai_attrs,
            expect_trace_id,
        } => commands::otlp_lint::run(
            otlp_file,
            *require_apr_span,
            *require_genai_attrs,
            expect_trace_id.as_deref(),
            cli.json,
        ),

        ExtendedCommands::PrometheusLint {
            metrics_file,
            content_type,
            require_k07_metrics,
        } => commands::prometheus_lint::run(
            metrics_file,
            content_type.as_deref(),
            *require_k07_metrics,
            cli.json,
        ),

        ExtendedCommands::ToolUseLint { observation_file } => {
            commands::tool_use_lint::run(observation_file, cli.json)
        }

        ExtendedCommands::GbnfLint { observation_file } => {
            commands::gbnf_lint::run(observation_file, cli.json)
        }

        ExtendedCommands::TypicalPLint { observation_file } => {
            commands::typical_p_lint::run(observation_file, cli.json)
        }

        ExtendedCommands::GradNorm {
            history_file,
            max_grad_norm,
            spike_window,
            spike_multiplier,
        } => commands::grad_norm::run(
            history_file,
            *max_grad_norm,
            *spike_window,
            *spike_multiplier,
            cli.json,
        ),

        ExtendedCommands::RegistryQuotaLint { observation_file } => {
            commands::registry_quota_lint::run(
                commands::registry_quota_lint::RegistryQuotaLintArgs {
                    observation_file: observation_file.to_string_lossy().to_string(),
                    json: cli.json,
                },
            )
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::ImatrixLint { observation_file } => {
            commands::imatrix_lint::run(commands::imatrix_lint::ImatrixLintArgs {
                observation_file: observation_file.to_string_lossy().to_string(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::EmbeddingsLint { observation_file } => {
            commands::embeddings_lint::run(commands::embeddings_lint::EmbeddingsLintArgs {
                observation_file: observation_file.to_string_lossy().into_owned(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::UnifiedSearchLint { observation_file } => {
            commands::unified_search_lint::run(
                commands::unified_search_lint::UnifiedSearchLintArgs {
                    observation_file: observation_file.to_string_lossy().to_string(),
                    json: cli.json,
                },
            )
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::RmGcLint { observation_file } => {
            commands::rm_gc_lint::run(commands::rm_gc_lint::RmGcLintArgs {
                observation_file: observation_file.to_string_lossy().to_string(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        ExtendedCommands::SharedCacheLint { observation_file } => {
            commands::shared_cache_lint::run(commands::shared_cache_lint::SharedCacheLintArgs {
                observation_file: observation_file.to_string_lossy().to_string(),
                json: cli.json,
            })
            .map_err(crate::error::CliError::Aprender)
        }

        // CRUX-K-11: Modelfile DSL parser.
        ExtendedCommands::Modelfile { command } => match command {
            ModelfileSubcommand::Parse { file, format } => {
                crate::commands::modelfile::run_parse(file, format)
            }
        },

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
        } => crate::error::resolve_model_path(file).and_then(|r| {
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

        ExtendedCommands::Tree {
            file,
            filter,
            format,
            sizes,
            depth,
        } => crate::error::resolve_model_path(file).and_then(|resolved| {
            let tree_format = if cli.json {
                tree::TreeFormat::Json
            } else {
                format.parse().unwrap_or(tree::TreeFormat::Ascii)
            };
            tree::run(&resolved, filter.as_deref(), tree_format, *sizes, *depth)
        }),

        ExtendedCommands::Flow {
            file,
            layer,
            component,
            verbose,
            json,
        } => crate::error::resolve_model_path(file).and_then(|resolved| {
            flow::run(
                &resolved,
                layer.as_deref(),
                component.parse().unwrap_or(flow::FlowComponent::Full),
                *verbose || cli.verbose,
                *json || cli.json,
            )
        }),

        ExtendedCommands::Qualify {
            file,
            tier,
            timeout,
            json,
            verbose,
            skip,
        } => crate::error::resolve_model_path(file).and_then(|resolved| {
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
        ExtendedCommands::Train { command } => dispatch_train_command(command, cli),
        #[cfg(feature = "training")]
        ExtendedCommands::Pretrain {
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
            init,
            force_under_provisioned,
            val_shard,
        } => commands::pretrain::run(
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
            init.as_deref(),
            *force_under_provisioned,
            val_shard.as_deref(),
            cli.json,
        ),
        ExtendedCommands::Tokenize { command } => dispatch_tokenize_command(command, cli),
        ExtendedCommands::Data { command } => dispatch_data_command(command, cli.json),
        ExtendedCommands::Pipeline { command } => dispatch_pipeline_command(command, cli),
        ExtendedCommands::Ppl { log_probs_file } => commands::ppl::run(log_probs_file, cli.json),

        ExtendedCommands::QuantPreservationLint { reference, requant } => {
            // CRUX-B-19 — validate dequant→requant metadata preservation.
            commands::quant_preservation::run(reference, requant, cli.json)
        }

        ExtendedCommands::Shard {
            file,
            max_shard_size,
            output,
        } => dispatch_shard(file, max_shard_size, output, cli.json),

        ExtendedCommands::Unshard { input, output } => dispatch_unshard(input, output, cli.json),

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

/// CRUX-B-05 — split a safetensors file into shards + weight-map index.
fn dispatch_shard(
    file: &std::path::Path,
    max_shard_size: &str,
    output: &std::path::Path,
    json: bool,
) -> Result<(), CliError> {
    match commands::shard::run_shard(file, max_shard_size, output) {
        Ok(report) => {
            if json {
                let shard_names: Vec<String> = report
                    .shard_files
                    .iter()
                    .map(|p| {
                        p.file_name().map_or_else(
                            || p.display().to_string(),
                            |n| n.to_string_lossy().into_owned(),
                        )
                    })
                    .collect();
                let shards_json: Vec<String> =
                    shard_names.iter().map(|n| format!("\"{n}\"")).collect();
                println!(
                    "{{\"status\":\"ok\",\"command\":\"shard\",\"input\":\"{}\",\"output_dir\":\"{}\",\"index\":\"{}\",\"shard_count\":{},\"tensor_count\":{},\"total_size\":{},\"shards\":[{}]}}",
                    file.display(),
                    output.display(),
                    report.index_path.display(),
                    report.shard_files.len(),
                    report.tensor_count,
                    report.total_size,
                    shards_json.join(",")
                );
            } else {
                println!("APR Shard");
                println!("  input:        {}", file.display());
                println!("  output dir:   {}", output.display());
                println!("  index:        {}", report.index_path.display());
                println!("  tensor count: {}", report.tensor_count);
                println!("  shard count:  {}", report.shard_files.len());
                println!("  total size:   {} bytes", report.total_size);
            }
            Ok(())
        }
        Err(e) => Err(CliError::ValidationFailed(format!("apr shard failed: {e}"))),
    }
}

/// CRUX-B-05 — reconstruct a single safetensors file from a sharded directory.
fn dispatch_unshard(
    input: &std::path::Path,
    output: &std::path::Path,
    json: bool,
) -> Result<(), CliError> {
    match commands::shard::run_unshard(input, output) {
        Ok(report) => {
            if json {
                println!(
                    "{{\"status\":\"ok\",\"command\":\"unshard\",\"input_dir\":\"{}\",\"output\":\"{}\",\"shard_count\":{},\"tensor_count\":{},\"total_size\":{}}}",
                    input.display(),
                    report.output.display(),
                    report.shard_count,
                    report.tensor_count,
                    report.total_size,
                );
            } else {
                println!("APR Unshard");
                println!("  input dir:    {}", input.display());
                println!("  output:       {}", report.output.display());
                println!("  shard count:  {}", report.shard_count);
                println!("  tensor count: {}", report.tensor_count);
                println!("  total size:   {} bytes", report.total_size);
            }
            Ok(())
        }
        Err(e) => Err(CliError::ValidationFailed(format!(
            "apr unshard failed: {e}"
        ))),
    }
}

#[cfg(feature = "training")]
/// Dispatch `apr runs` subcommands.
fn dispatch_runs_command(command: &RunsCommands, cli: &Cli) -> std::result::Result<(), CliError> {
    match command {
        RunsCommands::Ls {
            dir,
            global,
            status,
            json,
            limit,
        } => commands::runs::run_ls(dir, *global, status, *json || cli.json, *limit),
        RunsCommands::Show {
            run_id,
            dir,
            global,
            json,
        } => commands::runs::run_show(run_id, dir, *global, *json || cli.json),
        RunsCommands::Diff {
            run_a,
            run_b,
            dir,
            global,
            json,
        } => commands::runs::run_diff(run_a, run_b, dir, *global, *json || cli.json),
    }
}

#[cfg(feature = "training")]
/// Dispatch `apr experiment` subcommands.
fn dispatch_experiment_command(
    command: &ExperimentCommands,
    cli: &Cli,
) -> std::result::Result<(), CliError> {
    match command {
        ExperimentCommands::View { db, global, json } => {
            commands::experiment::experiment_view(db, *global, *json || cli.json)
        }
    }
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
        } => data::run_split(file, label_column, *train, *val, *test, *seed, output, json),
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
        DataCommands::Decontaminate {
            file,
            reference,
            ngram,
            threshold,
            json: json_flag,
        } => data::run_decontaminate(file, reference, *ngram, *threshold, *json_flag || json),
    }
}

#[cfg(feature = "training")]
/// Dispatch `apr train` subcommands to entrenar-backed implementations.
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
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
            distributed,
            world_size,
            rank,
            coordinator_addr,
            deterministic,
            seed,
            profile,
            profile_interval,
        } => {
            if *profile {
                eprintln!(
                    "StepProfiler enabled (report every {} steps)",
                    profile_interval
                );
            }
            train::run_apply(
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
                *distributed,
                *world_size,
                *rank,
                coordinator_addr.as_deref(),
                *deterministic,
                *seed,
            )
        }
        TrainCommands::Watch {
            config,
            max_restarts,
            heartbeat_timeout,
            backoff_initial,
            backoff_max,
        } => train::run_watch(
            config,
            *max_restarts,
            *heartbeat_timeout,
            *backoff_initial,
            *backoff_max,
            cli.json,
        ),
        TrainCommands::Sweep {
            config,
            strategy,
            num_configs,
            output_dir,
            seed,
        } => train::run_sweep(config, strategy, *num_configs, output_dir, *seed, cli.json),
        TrainCommands::Halving {
            sweep_dir,
            rounds,
            steps_per_round,
            source_width,
            target_width,
            output,
        } => train::run_halving(
            sweep_dir,
            *rounds,
            *steps_per_round,
            *source_width,
            *target_width,
            output,
            cli.json,
        ),
        TrainCommands::Archive {
            checkpoint_dir,
            output,
            release_version,
            notes,
        } => train::run_archive(
            checkpoint_dir,
            output,
            release_version.as_deref(),
            notes.as_deref(),
            cli.json,
        ),
        TrainCommands::Submit {
            cluster,
            model,
            adapters,
            rank,
            epochs,
            budget_mb,
            dry_run,
        } => train::run_submit(
            cluster, model, adapters, *rank, *epochs, *budget_mb, *dry_run, cli.json,
        ),
        TrainCommands::ClusterStatus { cluster } => train::run_cluster_status(cluster, cli.json),
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
        TokenizeCommands::Train {
            corpus,
            vocab_size,
            min_frequency,
            output,
            normalization,
        } => tokenize::run_train(
            corpus,
            *vocab_size,
            *min_frequency,
            output,
            normalization,
            cli.json,
        ),
        TokenizeCommands::ImportHf {
            input,
            output,
            include_added_tokens,
        } => tokenize::run_import_hf(input, output, *include_added_tokens, cli.json),
        #[cfg(feature = "training")]
        TokenizeCommands::EncodeCorpus {
            corpus,
            tokenizer,
            output,
            shard_tokens,
            content_field,
            normalization,
            eos_policy,
            num_workers,
            quiet,
            progress_interval_docs,
            progress_interval_seconds,
            estimate_only,
            estimate_sample_docs,
        } => tokenize::run_encode_corpus(
            corpus.as_slice(),
            tokenizer,
            output,
            *shard_tokens,
            content_field,
            normalization,
            eos_policy,
            *num_workers,
            tokenize::ProgressConfig {
                quiet: *quiet,
                interval_docs: *progress_interval_docs,
                interval_seconds: *progress_interval_seconds,
            },
            tokenize::EstimateConfig {
                enabled: *estimate_only,
                sample_docs: *estimate_sample_docs,
            },
            cli.json,
        ),
        #[cfg(feature = "training")]
        TokenizeCommands::RepairManifest {
            output,
            tokenizer,
            json,
        } => tokenize::run_repair_manifest(output, tokenizer.as_deref(), *json || cli.json),
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

#[cfg(feature = "training")]
/// Dispatch tune command — routes between classify-tune and general tune.
#[allow(clippy::too_many_arguments)]
fn dispatch_tune_command(
    file: Option<&Path>,
    method: &str,
    rank: Option<u32>,
    vram: f64,
    plan: bool,
    model: Option<&str>,
    freeze_base: bool,
    train_data: Option<&Path>,
    json: bool,
    task: Option<&str>,
    budget: usize,
    strategy: &str,
    scheduler: &str,
    scout: bool,
    data: Option<&Path>,
    num_classes: usize,
    model_size: Option<&str>,
    from_scout: Option<&Path>,
    max_epochs: usize,
    time_limit: Option<&str>,
) -> std::result::Result<(), CliError> {
    if task == Some("classify") {
        tune::run_classify_tune(
            file,
            budget,
            strategy,
            scheduler,
            scout,
            data.or(train_data),
            num_classes,
            model_size.or(model),
            from_scout,
            max_epochs,
            time_limit,
            json,
        )
    } else {
        tune::run(
            file,
            method.parse().unwrap_or(tune::TuneMethod::Auto),
            rank,
            vram,
            plan,
            model_size.or(model),
            freeze_base,
            train_data,
            json,
        )
    }
}

/// What `apr ptx` says when this binary was built without the PTX analyzer.
///
/// `cargo install aprender` builds default features, which do not include
/// `trueno-explain`, so every `apr ptx` invocation fails while `apr --help`
/// still advertises the command (#2399 finding 1). The old text — "ptx command
/// requires --features full" — named a compile flag rather than something the
/// user of an installed binary can act on, and pointed at `full`, which drags
/// in CUDA and training rather than the one crate `ptx` actually needs.
#[cfg(not(feature = "trueno-explain"))]
fn ptx_unavailable_message() -> String {
    "apr ptx needs the PTX analyzer, which this build does not include. \
     Reinstall with it: cargo install aprender --features ptx"
        .to_string()
}

/// Dispatch profiling and QA commands (profile, bench, eval, qa, parity, ptx, ptx-map, tune).
///
/// Returns `None` if the command is not a profiling command, allowing the caller
/// to try other sub-dispatchers.
#[provable_contracts_macros::contract(
    "apr-cli-operations-v1",
    equation = "side_effect_classification"
)]
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
        } => crate::error::resolve_model_path(file).and_then(|r| {
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
        }),

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
        } => crate::error::resolve_model_path(file).and_then(|r| match task.as_deref() {
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
            Some("contamination") => {
                eval::run_contamination(&r, data.as_deref(), None, *threshold / 100.0, cli.json)
            }
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
            // #2399 finding 1: the feature check must come FIRST. It used to run
            // after path resolution, so on a default build `apr ptx missing.ptx`
            // answered "File not found" (exit 3) — a user could spend a long time
            // fixing a path for a command this binary cannot run at all.
            #[cfg(not(feature = "trueno-explain"))]
            {
                let _ = (file, kernel, strict, bugs, json, verbose);
                Err(CliError::FeatureDisabled(ptx_unavailable_message()))
            }
            #[cfg(feature = "trueno-explain")]
            {
                match file
                    .as_ref()
                    .map(|f| crate::error::resolve_model_path(f))
                    .transpose()
                {
                    Ok(resolved) => commands::ptx_explain::run(
                        resolved.as_deref(),
                        kernel.as_deref(),
                        *strict,
                        *bugs,
                        *json || cli.json,
                        *verbose || cli.verbose,
                    ),
                    Err(e) => Err(e),
                }
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
            )
        }

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

        ExtendedCommands::Rerank {
            model,
            input_ids,
            token_type_ids,
            query,
            passage,
            passages,
            sort,
            top_k,
            vocab,
            hidden_dim,
            num_layers,
            num_heads,
            intermediate_dim,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            num_labels,
            with_pooler,
            raw_logit,
            json,
        } => commands::rerank::run(
            model,
            input_ids.as_deref(),
            token_type_ids.as_deref(),
            query.as_deref(),
            passage.as_deref(),
            passages,
            *sort,
            *top_k,
            vocab.as_deref(),
            *hidden_dim,
            *num_layers,
            *num_heads,
            *intermediate_dim,
            *vocab_size,
            *max_position_embeddings,
            *type_vocab_size,
            *num_labels,
            *with_pooler,
            *raw_logit,
            *json,
        ),

        ExtendedCommands::Embed {
            model,
            text,
            text_file,
            vocab,
            pool,
            normalize,
            hidden_dim,
            num_layers,
            num_heads,
            intermediate_dim,
            vocab_size,
            max_position_embeddings,
            type_vocab_size,
            json,
        } => commands::embed::run(
            model,
            text,
            text_file.as_deref(),
            vocab,
            pool,
            *normalize,
            *hidden_dim,
            *num_layers,
            *num_heads,
            *intermediate_dim,
            *vocab_size,
            *max_position_embeddings,
            *type_vocab_size,
            *json,
        ),

        // All other extended commands handled by sub-dispatchers above
        _ => unreachable!("all extended commands handled by sub-dispatchers"),
    }
}
