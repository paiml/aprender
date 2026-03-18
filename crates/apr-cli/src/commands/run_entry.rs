
/// Run command entry point
///
/// Per Section 9.2 (Sovereign AI), the `offline` flag enforces strict network isolation:
/// - When `true`, all network access is blocked at the type level
/// - Production deployments MUST use `--offline` mode
#[allow(clippy::too_many_arguments)]
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
    };

    let result = run_model(source, &options)?;

    if trace && trace_level == "layer" {
        print_layer_trace(&result, max_tokens);
    }

    if trace && trace_level == "payload" {
        print_payload_trace(&result, max_tokens);
    }

    if profile {
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
fn print_run_output(
    result: &RunResult,
    source: &str,
    output_format: &str,
    max_tokens: usize,
    benchmark: bool,
    stream: bool,
) -> Result<()> {
    // GH-240/GH-250: JSON output mode with accurate token counts
    if output_format == "json" && !benchmark {
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
        let json = serde_json::json!({
            "model": source,
            "text": result.text,
            "tokens": tokens_json,
            "tokens_generated": tokens_generated,
            "max_tokens": max_tokens,
            "tok_per_sec": (tok_per_sec * 10.0).round() / 10.0,
            "inference_time_ms": (result.duration_secs * 1000.0 * 100.0).round() / 100.0,
            "used_gpu": result.used_gpu.unwrap_or(false),
            "cached": result.cached,
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
        return Ok(());
    }

    if benchmark {
        print_benchmark_results(result, source, output_format, max_tokens);
    } else if stream {
        for word in result.text.split_whitespace() {
            print!("{word} ");
            std::io::Write::flush(&mut std::io::stdout())?;
        }
        println!();
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

/// Batch inference: load model once, process JSONL prompts.
///
/// Eliminates per-invocation model load + CUDA JIT overhead by keeping the
/// model resident across all prompts. Input/output are JSONL.
#[cfg(feature = "inference")]
pub(crate) fn run_batch(
    source: &str,
    batch_file: &Path,
    max_tokens: usize,
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
        temperature: 0.0,
        top_k: 1,
        no_gpu,
        verbose,
        stop_tokens: vec![],
    };

    let file = std::fs::File::open(batch_file).map_err(|_| {
        CliError::FileNotFound(batch_file.to_path_buf())
    })?;
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
