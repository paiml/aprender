// Helper dispatchers (runs, experiment, data, train, tokenize, pipeline, tune)
// extracted from `dispatch_analysis.rs` to keep the PMAT-689 file-size invariant.
//
// Inlined into `dispatch_analysis.rs` via `include!()`, so items share the parent
// module's imports and file-private scope.

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
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "side_effect_classification")]
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
                eprintln!("StepProfiler enabled (report every {} steps)", profile_interval);
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
        TokenizeCommands::EncodeCorpus {
            corpus,
            tokenizer,
            output,
            shard_tokens,
            content_field,
            normalization,
            eos_policy,
        } => tokenize::run_encode_corpus(
            corpus,
            tokenizer,
            output,
            *shard_tokens,
            content_field,
            normalization,
            eos_policy,
            cli.json,
        ),
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
