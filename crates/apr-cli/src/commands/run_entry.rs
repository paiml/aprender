/// Run command entry point
///
/// Per Section 9.2 (Sovereign AI), the `offline` flag enforces strict network isolation:
/// - When `true`, all network access is blocked at the type level
/// - Production deployments MUST use `--offline` mode
#[allow(clippy::too_many_arguments)]
#[provable_contracts_macros::contract(
    "apr-cli-command-safety-v1",
    equation = "long_running_graceful"
)]
pub(crate) fn run(
    source: &str,
    input: Option<&Path>,
    prompt: Option<&str>,
    max_tokens: usize,
    stream: bool,
    language: Option<&str>,
    task: Option<&str>,
    output_format: &str,
    no_gpu: bool,
    offline: bool,
    benchmark: bool,
    verbose: bool,
    trace: bool,
    trace_steps: Option<&[String]>,
    trace_verbose: bool,
    trace_output: Option<PathBuf>,
    trace_level: &str,
    profile: bool,
    // PMAT-496: Sampling parameters — previously silently dropped
    temperature: f32,
    top_k: usize,
    top_p: Option<f32>,
    seed: u64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    split_prompt: bool,
) -> Result<()> {
    // GH-516: Warn on --language/--task since whisper integration is not yet wired up
    if language.is_some() {
        eprintln!("Warning: --language is not yet supported for inference. Flag ignored.");
    }
    if task.is_some() {
        eprintln!("Warning: --task is not yet supported for inference. Flag ignored.");
    }

    // GH-240: Suppress header/source in JSON mode for clean machine-parseable output
    if output_format != "json" {
        if offline {
            println!("{}", "=== APR Run (OFFLINE MODE) ===".cyan().bold());
            eprintln!(
                "{}",
                "Network access disabled. Only local/cached models allowed.".yellow()
            );
        } else {
            println!("{}", "=== APR Run ===".cyan().bold());
        }
        println!();
        println!("Source: {source}");
    }

    // Setup trace config if tracing enabled (APR-TRACE-001)
    if trace {
        print_trace_config(
            trace_level,
            trace_steps,
            trace_verbose,
            trace_output.as_ref(),
            profile,
        );
    }

    // Kept for the chrome-trace writer below: `RunOptions` takes ownership of
    // `trace_output`, and `--trace-level chrome` must honour the path the user
    // asked for instead of inventing `trace-<epoch>.json` in the CWD.
    let requested_trace_output = trace_output.clone();

    let options = RunOptions {
        input: input.map(Path::to_path_buf),
        prompt: prompt.map(String::from),
        max_tokens,
        output_format: output_format.to_string(),
        force: false,
        no_gpu,
        offline,
        benchmark,
        verbose,
        trace,
        trace_steps: trace_steps.map(<[std::string::String]>::to_vec),
        trace_verbose,
        trace_output,
        trace_level: trace_level.to_string(),
        profile,
        temperature,
        top_k,
        top_p,
        seed,
        repeat_penalty,
        repeat_last_n,
        split_prompt,
        stream,
    };

    let result = run_model(source, &options)?;

    if trace && trace_level == "layer" {
        print_layer_trace(&result, max_tokens);
    }

    if trace && trace_level == "payload" {
        print_payload_trace(&result, max_tokens);
    }

    // F-CLIPARITY-01 / PMAT-386: Chrome trace JSON output
    // Integrates layer trace + brick profile into chrome://tracing format.
    // Usage: apr run model.gguf "prompt" --trace --trace-level chrome --profile
    if trace && trace_level == "chrome" {
        print_chrome_trace(
            &result,
            source,
            max_tokens,
            profile,
            requested_trace_output.as_deref(),
        );
    }

    if profile && trace_level != "chrome" {
        print_roofline_profile(&result, max_tokens);
    }

    print_run_output(
        &result,
        source,
        output_format,
        max_tokens,
        benchmark,
        stream,
    )?;

    Ok(())
}

/// F-CLIPARITY-01 / PMAT-386: Chrome trace JSON output.
/// Integrates layer trace + brick profile into chrome://tracing format.
///
/// Destination: `--trace-output FILE` when the caller gave one, otherwise
/// `trace-{timestamp}.json` in the CWD (matches Candle's `--tracing` output).
///
/// The path argument exists because `--trace-level chrome` used to ignore
/// `--trace-output` unconditionally: the chrome JSON landed in an auto-named
/// file in the working directory while the path the user asked for was left
/// holding the summary stub, so a scripted consumer silently read the wrong
/// file.
fn print_chrome_trace(
    result: &super::run::RunResult,
    source: &str,
    max_tokens: usize,
    include_profile: bool,
    trace_output: Option<&Path>,
) {
    use std::time::{SystemTime, UNIX_EPOCH};

    let filename = match trace_output {
        Some(p) => p.to_path_buf(),
        None => {
            let timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            PathBuf::from(format!("trace-{timestamp}.json"))
        }
    };

    let trace = build_chrome_trace(result, source, max_tokens, include_profile);

    match std::fs::write(
        &filename,
        serde_json::to_string_pretty(&trace).unwrap_or_default(),
    ) {
        Ok(()) => eprintln!(
            "Chrome trace written to: {} (load in chrome://tracing)",
            filename.display()
        ),
        Err(e) => eprintln!("Failed to write chrome trace: {e}"),
    }
}

/// Build the chrome://tracing document for a completed run.
///
/// Split out of [`print_chrome_trace`] so the document can be asserted on
/// directly. The tests used to assert against a *copy* of this logic kept in
/// the test file, which proved only that the copy agreed with itself.
///
/// `metadata.timing_model` is `"derived"`: apart from the run's total wall
/// clock, the per-event durations here are a fixed split of that total, not
/// measured spans. Real per-operation timing comes from the brick profiler
/// (`apr profile --granular`).
pub(crate) fn build_chrome_trace(
    result: &super::run::RunResult,
    source: &str,
    max_tokens: usize,
    include_profile: bool,
) -> serde_json::Value {
    let mut events = Vec::new();
    let mut ts_us: u64 = 0;

    // Model load event
    let load_dur = (result.duration_secs * 1_000_000.0) as u64;
    events.push(serde_json::json!({
        "name": "model_load",
        "cat": "lifecycle",
        "ph": "X",
        "ts": 0,
        "dur": load_dur / 10, // ~10% of total is load
        "pid": 1,
        "tid": 1,
        "args": {"source": source, "max_tokens": max_tokens}
    }));
    ts_us = load_dur / 10;

    // Contract: apr-chrome-trace-v1.yaml — trace_event_categories equation
    // Required categories: tokenize, embed, layer, sample, decode

    // Tokenize event
    let tokenize_dur = load_dur / 100; // ~1% of total
    events.push(serde_json::json!({
        "name": "tokenize",
        "cat": "tokenize",
        "ph": "X",
        "ts": ts_us,
        "dur": tokenize_dur,
        "pid": 1, "tid": 1,
        "args": {"source": source}
    }));
    ts_us += tokenize_dur;

    // Embed event
    let embed_dur = load_dur / 100;
    events.push(serde_json::json!({
        "name": "embed",
        "cat": "embed",
        "ph": "X",
        "ts": ts_us,
        "dur": embed_dur,
        "pid": 1, "tid": 1
    }));
    ts_us += embed_dur;

    // Token generation events (decode + sample per token)
    if let Some(count) = result.tokens_generated {
        let gen_dur = load_dur - ts_us;
        let per_token = if count > 0 {
            gen_dur / count as u64
        } else {
            gen_dur
        };
        for i in 0..count {
            let token_start = ts_us + (i as u64 * per_token);
            // Layer forward pass (~90% of per-token time)
            let layer_dur = per_token * 9 / 10;
            events.push(serde_json::json!({
                "name": format!("layer_{}", i % 28),
                "cat": "layer",
                "ph": "X",
                "ts": token_start,
                "dur": layer_dur,
                "pid": 1, "tid": 1,
                "args": {"token_idx": i, "layer": i % 28}
            }));
            // Sample step (~10% of per-token time)
            events.push(serde_json::json!({
                "name": "sample",
                "cat": "sample",
                "ph": "X",
                "ts": token_start + layer_dur,
                "dur": per_token - layer_dur,
                "pid": 1, "tid": 1,
                "args": {"token_idx": i}
            }));
            // Decode event (instant marker)
            events.push(serde_json::json!({
                "name": format!("token_{}", i),
                "cat": "decode",
                "ph": "X",
                "ts": token_start,
                "dur": per_token,
                "pid": 1, "tid": 1,
                "args": {"token_idx": i}
            }));
        }
    }

    serde_json::json!({
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "source": source,
            "tool": "apr run --trace --trace-level chrome",
            "max_tokens": max_tokens,
            "tok_per_sec": result.tok_per_sec,
            "include_profile": include_profile,
            // Honesty marker: only the run total is measured; the per-event
            // durations below are a fixed split of it.
            "timing_model": "derived"
        }
    })
}

/// Print trace configuration when tracing is enabled.
fn print_trace_config(
    trace_level: &str,
    trace_steps: Option<&[String]>,
    trace_verbose: bool,
    trace_output: Option<&PathBuf>,
    profile: bool,
) {
    eprintln!("{}", "Inference tracing enabled (APR-TRACE-001)".cyan());
    eprintln!("  Trace level: {}", trace_level);
    if let Some(steps) = trace_steps {
        eprintln!("  Trace steps: {}", steps.join(", "));
    }
    if trace_verbose {
        eprintln!("  Verbose mode enabled");
    }
    if let Some(path) = trace_output {
        eprintln!("  Output: {}", path.display());
    }
    if profile {
        eprintln!("  Roofline profiling enabled");
    }
}

/// Print the final run output (benchmark, stream, or batch mode).
///
/// # Streaming mode (`--stream`)
///
/// When `stream` is true, output becomes a JSONL stream:
/// - One `{"event":"token", "index":N, "token_id":U, "text":"..."}` line per
///   generated token, in order.
/// - One terminal `{"event":"final", ...}` line carrying the same fields the
///   `--json` output mode emits today (model, text, tokens, tok_per_sec, ...).
///
/// # Implementation note
///
/// The current realizar `run_inference()` API returns the full token sequence
/// only after generation completes — there is no per-token callback hook
/// today. This function therefore emits all token events post-hoc just before
/// the final blob. The JSONL wire contract is identical to what a true
/// streaming implementation would produce; when realizar grows a callback the
/// emit point can move into the decode loop without touching consumers.
fn print_run_output(
    result: &RunResult,
    source: &str,
    output_format: &str,
    max_tokens: usize,
    benchmark: bool,
    stream: bool,
) -> Result<()> {
    // --stream takes precedence — emit JSONL stream. This implies json-style
    // structured output regardless of --format. (--stream --json is the same
    // as --stream alone.)
    if stream && !benchmark {
        return print_stream_output(result, source, max_tokens);
    }

    // GH-240/GH-250: JSON output mode with accurate token counts
    if output_format == "json" && !benchmark {
        let json = build_final_json(result, source, max_tokens);
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
        return Ok(());
    }

    if benchmark {
        print_benchmark_results(result, source, output_format, max_tokens);
    } else {
        println!();
        println!("{}", "Output:".green().bold());
        println!("{}", result.text);
    }

    if !benchmark {
        println!();
        println!(
            "Completed in {:.2}s {}",
            result.duration_secs,
            if result.cached {
                "(cached)".dimmed()
            } else {
                "(downloaded)".dimmed()
            }
        );
    }
    Ok(())
}

/// Build the terminal JSON blob shared by `--json` and `--stream` final events.
fn build_final_json(result: &RunResult, source: &str, max_tokens: usize) -> serde_json::Value {
    let tokens_generated = result.tokens_generated.unwrap_or(0);
    let tok_per_sec = result.tok_per_sec.unwrap_or_else(|| {
        if result.duration_secs > 0.0 {
            tokens_generated as f64 / result.duration_secs
        } else {
            0.0
        }
    });
    // GH-250: Include generated token IDs for parity checking
    let tokens_json = result.generated_tokens.as_deref().unwrap_or(&[]);
    serde_json::json!({
        "model": source,
        "text": result.text,
        "tokens": tokens_json,
        "tokens_generated": tokens_generated,
        "max_tokens": max_tokens,
        "tok_per_sec": (tok_per_sec * 10.0).round() / 10.0,
        "inference_time_ms": (result.duration_secs * 1000.0 * 100.0).round() / 100.0,
        "used_gpu": result.used_gpu.unwrap_or(false),
        "cached": result.cached,
    })
}

/// Emit one JSON line per generated token plus a terminal `final` blob.
///
/// Wire format (one JSON object per line, NDJSON):
/// ```text
/// {"event":"token","index":0,"token_id":1234,"text":""}
/// {"event":"token","index":1,"token_id":5678,"text":""}
/// ...
/// {"event":"final","model":"...","text":"...","tokens":[...],"tok_per_sec":42.0,...}
/// ```
///
/// Per-token `text` carries that token's own decoded text, so a consumer can
/// render the reply as the events arrive. It falls back to an empty string
/// only when no tokenizer could be resolved for the model; the token id is
/// always present and exact, and the terminal `final` event always carries the
/// authoritative full text.
fn print_stream_output(result: &RunResult, source: &str, max_tokens: usize) -> Result<()> {
    use std::io::Write;
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    write_stream_output(&mut out, result, source, max_tokens)?;
    out.flush()?;
    Ok(())
}

/// Write the stream NDJSON to a generic `Write` sink. Extracted from
/// [`print_stream_output`] for direct testing without stdout capture.
pub(crate) fn write_stream_output<W: std::io::Write>(
    out: &mut W,
    result: &RunResult,
    source: &str,
    max_tokens: usize,
) -> std::io::Result<()> {
    if let Some(tokens) = result.generated_tokens.as_deref() {
        let texts = result.token_texts.as_deref().unwrap_or(&[]);
        for (index, token_id) in tokens.iter().copied().enumerate() {
            let evt = serde_json::json!({
                "event": "token",
                "index": index as u32,
                "token_id": token_id,
                "text": texts.get(index).map_or("", String::as_str),
            });
            writeln!(out, "{}", serde_json::to_string(&evt).unwrap_or_default())?;
        }
    }

    let mut final_blob = build_final_json(result, source, max_tokens);
    if let Some(obj) = final_blob.as_object_mut() {
        obj.insert(
            "event".to_string(),
            serde_json::Value::String("final".to_string()),
        );
    }
    writeln!(
        out,
        "{}",
        serde_json::to_string(&final_blob).unwrap_or_default()
    )
}

/// Batch inference: load model once, process JSONL prompts.
///
/// Eliminates per-invocation model load + CUDA JIT overhead by keeping the
/// model resident across all prompts. Input/output are JSONL.
#[cfg(feature = "inference")]
pub(crate) fn run_batch(
    source: &str,
    batch_file: &Path,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    no_gpu: bool,
    verbose: bool,
) -> Result<()> {
    use realizar::{run_batch_inference, BatchInferenceConfig};

    // Resolve model path (same logic as regular run)
    let model_source = ModelSource::parse(source)?;
    let model_path = resolve_model(&model_source, false, false)?;

    let config = BatchInferenceConfig {
        model_path,
        max_tokens,
        temperature,
        top_k,
        no_gpu,
        verbose,
        stop_tokens: vec![],
    };

    let file = std::fs::File::open(batch_file)
        .map_err(|_| CliError::FileNotFound(batch_file.to_path_buf()))?;
    let reader = std::io::BufReader::new(file);
    let stdout = std::io::stdout();
    let writer = std::io::BufWriter::new(stdout.lock());

    let stats = run_batch_inference(&config, reader, writer)
        .map_err(|e| CliError::InferenceFailed(format!("Batch inference failed: {e}")))?;

    eprintln!(
        "[batch] Summary: {} prompts, {} ok, {} failed, {:.1} total tokens, {:.1}s model load",
        stats.total_prompts,
        stats.successful,
        stats.failed,
        stats.total_tokens_generated,
        stats.model_load_ms / 1000.0,
    );

    if stats.failed > 0 {
        eprintln!(
            "Warning: {} of {} prompts failed",
            stats.failed, stats.total_prompts
        );
    }

    Ok(())
}

/// Print benchmark results with optional JSON output.
fn print_benchmark_results(
    result: &RunResult,
    source: &str,
    output_format: &str,
    max_tokens: usize,
) {
    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let tok_per_sec = if result.duration_secs > 0.0 {
        tokens_generated as f64 / result.duration_secs
    } else {
        0.0
    };

    println!();
    println!("{}", "=== Benchmark Results ===".cyan().bold());
    println!("tok/s: {:.1}", tok_per_sec);
    println!("tokens: {}", tokens_generated);
    println!("latency: {:.2}ms", result.duration_secs * 1000.0);
    println!("model: {}", source);
    println!();

    if output_format == "json" {
        println!(
            r#"{{"tok_s": {:.1}, "tokens": {}, "latency_ms": {:.2}}}"#,
            tok_per_sec,
            tokens_generated,
            result.duration_secs * 1000.0
        );
    }
}
