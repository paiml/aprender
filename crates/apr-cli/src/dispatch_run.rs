
/// Dispatch `apr run` — extracted to reduce cognitive complexity of `execute_command`
#[allow(clippy::too_many_arguments)]
fn dispatch_run(
    source: &str,
    positional_prompt: Option<&String>,
    input: Option<&Path>,
    prompt: Option<&String>,
    max_tokens: usize,
    stream: bool,
    language: Option<&str>,
    task: Option<&str>,
    format: &str,
    no_gpu: bool,
    offline: bool,
    benchmark: bool,
    verbose: bool,
    trace: bool,
    trace_payload: bool,
    trace_steps: Option<&[String]>,
    trace_verbose: bool,
    trace_output: Option<PathBuf>,
    trace_level: &str,
    profile: bool,
    chat: bool,
    // PMAT-496: Sampling parameters
    temperature: f32,
    top_k: usize,
    top_p: Option<f32>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    split_prompt: bool,
) -> Result<(), CliError> {
    let effective_trace = trace || trace_payload;
    let effective_trace_level = if trace_payload {
        "payload"
    } else {
        trace_level
    };
    let merged_prompt = prompt.or(positional_prompt).cloned();
    // GH-638: Auto-detect chat template from model name when --chat not explicit.
    // Instruct/Chat models (Qwen-Instruct, LLaMA-Instruct, Mistral-Instruct, etc.)
    // need ChatML wrapping for correct output. Without it, the model ignores the
    // prompt structure and produces garbled responses.
    let use_chat = chat || {
        let src_lower = source.to_lowercase();
        merged_prompt.is_some()
            && (src_lower.contains("instruct") || src_lower.contains("chat"))
    };
    let effective_prompt = if use_chat {
        merged_prompt
            .as_ref()
            .map(|p| format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n"))
    } else {
        merged_prompt
    };

    run::run(
        source,
        input,
        effective_prompt.as_deref(),
        max_tokens,
        stream,
        language,
        task,
        format,
        no_gpu,
        offline,
        benchmark,
        verbose,
        effective_trace,
        trace_steps,
        trace_verbose,
        trace_output,
        effective_trace_level,
        profile,
        temperature,
        top_k,
        top_p,
        seed,
        repeat_penalty,
        repeat_last_n,
        split_prompt,
    )
}

/// Build server config and launch serve.
#[allow(clippy::too_many_arguments)]
fn dispatch_serve(
    file: &Path,
    port: u16,
    host: &str,
    no_cors: bool,
    no_metrics: bool,
    no_gpu: bool,
    gpu: bool,
    batch: bool,
    trace: bool,
    trace_level: &str,
    profile: bool,
    verbose: bool,
    backend: &Option<String>,
    otlp_endpoint: &Option<String>,
    context_length: usize,
    no_fp8_cache: bool,
    ollama_compat: bool,
) -> Result<(), CliError> {
    if let Some(ref endpoint) = otlp_endpoint {
        eprintln!("OTLP tracing enabled → {endpoint}");
        eprintln!("  Spans exported as W3C Trace Context (PMAT-485)");
    }
    let config = serve::ServerConfig {
        port: if ollama_compat && port == 8080 { 11434 } else { port },
        host: host.to_owned(),
        cors: !no_cors,
        metrics: !no_metrics,
        no_gpu,
        gpu,
        batch,
        trace,
        trace_level: trace_level.to_owned(),
        profile,
        verbose,
        backend: backend.clone(),
        otlp_endpoint: otlp_endpoint.clone(),
        context_length,
        no_fp8_cache,
        ollama_compat,
        ..Default::default()
    };
    serve::run(file, &config)
}

/// Route `apr serve` subcommands: plan or run.
fn dispatch_serve_command(command: &ServeCommands, cli: &Cli) -> Result<(), CliError> {
    match command {
        ServeCommands::Plan {
            model,
            gpu,
            batch_size,
            seq_len,
            format,
            quant,
        } => {
            // GH-630: Thread cli.json through to serve plan
            let effective_format = if cli.json { "json" } else { format.as_str() };
            commands::serve_plan::run_serve_plan(
                model, *gpu, *batch_size, *seq_len, effective_format, quant.as_deref(),
            )
        }
        ServeCommands::Run {
            file,
            port,
            host,
            no_cors,
            no_metrics,
            no_gpu,
            gpu,
            gpu_layers,
            list_devices,
            batch,
            trace,
            trace_level,
            profile,
            backend: BackendArg { backend },
            otlp_endpoint,
            context_length,
            no_fp8_cache,
            ollama_compat,
        } => {
            // PERF-021: answer "what can this BUILD dispatch to" without needing
            // a model or a port — the question a user hitting #2696 had no way
            // to ask.
            if *list_devices {
                return crate::commands::serve::list_devices();
            }
            // clap guarantees `file` is present unless --list-devices short-
            // circuited above, so this cannot be None here.
            let Some(file) = file.as_ref() else {
                return Err(CliError::InvalidInput(
                    "serve run needs a model file".to_string(),
                ));
            };
            crate::error::resolve_model_path(file).and_then(|r| {
            dispatch_serve(
                &r,
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
                backend,
                otlp_endpoint,
                *context_length,
                *no_fp8_cache,
                *ollama_compat,
            )
        })
        },
    }
}

/// Parse hex offset and run hex inspection.
#[allow(clippy::too_many_arguments)]
fn dispatch_hex(
    file: &Path,
    tensor: Option<&str>,
    limit: usize,
    stats: bool,
    list: bool,
    json: bool,
    header: bool,
    blocks: bool,
    distribution: bool,
    contract: bool,
    entropy: bool,
    raw: bool,
    offset: &str,
    width: usize,
    slice: Option<&str>,
) -> Result<(), CliError> {
    let parsed_offset = hex::parse_hex_offset(offset).map_err(CliError::InvalidFormat)?;
    hex::run(&hex::HexOptions {
        file: file.to_path_buf(),
        tensor: tensor.map(String::from),
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
        offset: parsed_offset,
        width,
        slice: slice.map(String::from),
    })
}

/// Dispatch a rosetta subcommand.
fn dispatch_rosetta(action: &RosettaCommands, global_json: bool) -> Result<(), CliError> {
    match action {
        RosettaCommands::Inspect {
            file,
            hexdump,
            json,
        } => rosetta::run_inspect(file, *hexdump, *json || global_json),
        RosettaCommands::Convert {
            source,
            target,
            quantize,
            verify,
            json,
            tokenizer,
        } => rosetta::run_convert(
            source,
            target,
            quantize.as_deref(),
            *verify,
            *json || global_json,
            tokenizer.as_deref(),
        ),
        RosettaCommands::Chain {
            source,
            formats,
            work_dir,
            json,
        } => rosetta::run_chain(source, formats, work_dir, *json || global_json),
        RosettaCommands::Verify {
            source,
            intermediate,
            tolerance,
            json,
        } => rosetta::run_verify(source, intermediate, *tolerance, *json || global_json),
        RosettaCommands::CompareInference {
            model_a,
            model_b,
            prompt,
            max_tokens,
            temperature,
            tolerance,
            json,
        } => rosetta::run_compare_inference(
            model_a,
            model_b,
            prompt,
            *max_tokens,
            *temperature,
            *tolerance,
            *json || global_json,
        ),
        RosettaCommands::DiffTensors {
            model_a,
            model_b,
            mismatches_only,
            show_values,
            filter,
            json,
        } => rosetta::run_diff_tensors(
            model_a,
            model_b,
            *mismatches_only,
            *show_values,
            filter.as_deref(),
            *json || global_json,
        ),
        RosettaCommands::Fingerprint {
            model,
            model_b,
            output,
            filter,
            verbose,
            json,
        } => rosetta::run_fingerprint(
            model,
            model_b.as_ref().map(std::path::PathBuf::as_path),
            output.as_ref().map(std::path::PathBuf::as_path),
            filter.as_deref(),
            *verbose,
            *json || global_json,
        ),
        RosettaCommands::ValidateStats {
            model,
            reference,
            fingerprints,
            threshold,
            strict,
            json,
        } => rosetta::run_validate_stats(
            model,
            reference.as_ref().map(std::path::PathBuf::as_path),
            fingerprints.as_ref().map(std::path::PathBuf::as_path),
            *threshold,
            *strict,
            *json || global_json,
        ),
    }
}

/// Execute the CLI command and return the result.
pub fn execute_command(cli: &Cli) -> Result<(), CliError> {
    contract_pre_contract_gate_enforcement!();
    // CRUX-A-20: record `--offline` once, here, for the whole process. Every
    // outbound-network helper consults `commands::offline::guard`, so no
    // command can disarm the control by forgetting to forward a parameter.
    // `--offline` is a clap global, so it is seen in either position
    // (`apr --offline pull ...` and `apr pull --offline ...`).
    crate::commands::offline::latch(cli.offline);
    // #2401: same shape, same reason. `--quiet` / `--verbose` are clap
    // globals advertised on all 104 subcommands and were read by exactly two
    // of them, because reading them meant threading a parameter through every
    // command. Record the level once, here, and let the crate-wide `println!`
    // shadow in `crate::verbosity` consult it.
    crate::verbosity::latch(cli.quiet, cli.verbose, cli.json);
    // PMAT-237: Contract gate — refuse to operate on corrupt models
    let gated_paths = extract_model_paths(&cli.command);
    // #2401: and the other half — `--verbose` was byte-inert on 13 of 16
    // sampled commands. Report the dispatcher's own resolution instead of
    // inventing per-command chatter for all 104: which model paths were
    // extracted, and what the contract gate did with them.
    if crate::verbosity::is_verbose() {
        for line in crate::verbosity::preamble_lines(
            env!("CARGO_PKG_VERSION"),
            cli.offline,
            cli.skip_contract,
            &gated_paths,
        ) {
            vprintln!("{line}");
        }
    }
    if !cli.skip_contract {
        validate_model_contract(&gated_paths)?;
    }

    dispatch_core_command(cli).unwrap_or_else(|| dispatch_extended_command(cli))
}
