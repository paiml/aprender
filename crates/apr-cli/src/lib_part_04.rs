
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
) -> Result<(), CliError> {
    let effective_trace = trace || trace_payload;
    let effective_trace_level = if trace_payload {
        "payload"
    } else {
        trace_level
    };
    let merged_prompt = prompt.or(positional_prompt).cloned();
    let effective_prompt = if chat {
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
) -> Result<(), CliError> {
    let config = serve::ServerConfig {
        port,
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
        ..Default::default()
    };
    serve::run(file, &config)
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
    // PMAT-237: Contract gate — refuse to operate on corrupt models
    if !cli.skip_contract {
        let paths = extract_model_paths(&cli.command);
        validate_model_contract(&paths)?;
    }

    dispatch_core_command(cli).unwrap_or_else(|| dispatch_extended_command(cli))
}
