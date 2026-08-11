
/// Pre-fault all mmap pages to avoid page faults during model load/inference (PAR-200: B4 CPU perf fix).
///
/// Without this, `OwnedQuantizedModel::from_mapped()` triggers ~9M minor page faults = 2.5s overhead.
/// Touches one byte per 4K page to force the kernel to fault in each page.
#[cfg(feature = "inference")]
fn prefault_mmap_pages(data: &[u8]) {
    let page_size = 4096;
    let mut checksum: u8 = 0;
    for i in (0..data.len()).step_by(page_size) {
        checksum = checksum.wrapping_add(data[i]);
    }
    std::hint::black_box(checksum);
    let pages_touched = data.len().div_ceil(page_size);
    let _ = pages_touched;
}

/// Build a fallback metadata display when quantized inference is unavailable.
#[cfg(feature = "inference")]
fn build_gguf_fallback_display(
    model_path: &Path,
    model: &realizar::gguf::GGUFModel,
    load_error: &realizar::RealizarError,
) -> String {
    let mut output = format!(
        "GGUF Model (quantized inference unavailable)\n\
         Model: {}\n\
         Load error: {}\n\
         GGUF Version: {}\n\
         Tensors: {}\n\
         Metadata entries: {}\n\n",
        model_path.display(),
        load_error,
        model.header.version,
        model.tensors.len(),
        model.metadata.len()
    );

    output.push_str("Metadata (first 10):\n");
    for (i, (key, _)) in model.metadata.iter().take(10).enumerate() {
        output.push_str(&format!("  {}. {}\n", i + 1, key));
    }
    if model.metadata.len() > 10 {
        output.push_str(&format!("  ... and {} more\n", model.metadata.len() - 10));
    }

    output.push_str("\nTensors (first 10):\n");
    for (i, tensor) in model.tensors.iter().take(10).enumerate() {
        output.push_str(&format!(
            "  {}. {} (type: {}, dims: {:?})\n",
            i + 1,
            tensor.name,
            tensor.qtype,
            tensor.dims
        ));
    }
    if model.tensors.len() > 10 {
        output.push_str(&format!("  ... and {} more\n", model.tensors.len() - 10));
    }

    output
}

/// Execute GGUF model inspection
///
/// Execute GGUF model inference using realizar's optimized OwnedQuantizedModel.
///
/// Uses quantized compute for better performance than naive dequantize-then-compute.
#[cfg(feature = "inference")]
fn execute_gguf_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
    use std::time::Instant;

    let start = Instant::now();
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF model: {e}")))?;
    let _mmap_time = start.elapsed();

    prefault_mmap_pages(mapped_model.data());

    let load_start = Instant::now();
    let model_result = OwnedQuantizedModel::from_mapped(&mapped_model);
    let _load_time = load_start.elapsed();

    match model_result {
        Ok(model) => {
            let input_tokens =
                prepare_gguf_input_tokens(model_path, &mapped_model, options, input_path)?;

            let gen_config = QuantizedGenerateConfig {
                max_tokens: options.max_tokens.min(128),
                temperature: 0.0,
                top_k: 1,
                trace: options.trace,
                ..Default::default()
            };

            let decode_fn = |token_id: u32| -> String { mapped_model.model.decode(&[token_id]) };
            let trace_opts = if options.trace { Some(options) } else { None };
            let gen_result = run_gguf_generate(
                model,
                &input_tokens,
                &gen_config,
                options.no_gpu,
                options.benchmark,
                trace_opts,
                Some(&decode_fn),
            )?;

            if options.benchmark {
                let new_tokens = gen_result.tokens.len().saturating_sub(input_tokens.len());
                let tok_per_sec = if gen_result.inference_ms > 0.0 {
                    new_tokens as f64 / (gen_result.inference_ms / 1000.0)
                } else {
                    0.0
                };
                eprintln!(
                    "Inference: {} tokens in {:.1}ms ({:.1} tok/s)",
                    new_tokens, gen_result.inference_ms, tok_per_sec
                );
            }

            let generated_tokens = &gen_result.tokens[input_tokens.len()..];
            let decoded_text = mapped_model.model.decode(generated_tokens);
            let cleaned = clean_model_output(&decoded_text);
            Ok(cleaned)
        }
        Err(e) => Ok(build_gguf_fallback_display(
            model_path,
            &mapped_model.model,
            &e,
        )),
    }
}

/// Result from GGUF generation including timing
#[cfg(feature = "inference")]
struct GgufGenerateResult {
    tokens: Vec<u32>,
    inference_ms: f64,
}

/// Create and configure an inference tracer from run options.
#[cfg(feature = "inference")]
fn setup_gguf_tracer(
    opts: &RunOptions,
    model_name: &str,
    config: &realizar::gguf::GGUFConfig,
) -> realizar::InferenceTracer {
    use realizar::{InferenceTracer, ModelInfo, TraceConfig};

    let mut trace_config = TraceConfig::enabled();
    trace_config.verbose = opts.trace_verbose;
    trace_config.output.clone_from(&opts.trace_output);
    if let Some(ref steps) = opts.trace_steps {
        trace_config.steps = TraceConfig::parse_steps(&steps.join(","));
    }

    let mut tracer = InferenceTracer::new(trace_config);
    tracer.set_model_info(ModelInfo {
        name: model_name.to_string(),
        num_layers: config.num_layers,
        hidden_dim: config.hidden_dim,
        vocab_size: config.vocab_size,
        num_heads: config.num_heads,
        quant_type: None,
    });
    tracer
}

/// Generate tokens and optionally trace, returning the result.
#[cfg(feature = "inference")]
fn traced_generate(
    generate_fn: impl FnOnce() -> std::result::Result<Vec<u32>, realizar::RealizarError>,
    trace_options: Option<&RunOptions>,
    model_name: &str,
    config: &realizar::gguf::GGUFConfig,
    error_label: &str,
) -> Result<Vec<u32>> {
    let trace_enabled = trace_options.is_some_and(|o| o.trace);
    if trace_enabled {
        let opts = trace_options.expect("trace_options must be Some when trace_enabled");
        let tracer = setup_gguf_tracer(opts, model_name, config);
        let result =
            generate_fn().map_err(|e| CliError::InferenceFailed(format!("{error_label}: {e}")))?;
        if let Err(e) = tracer.write_output() {
            eprintln!("Warning: Failed to write trace output: {e}");
        }
        Ok(result)
    } else {
        generate_fn().map_err(|e| CliError::InferenceFailed(format!("{error_label}: {e}")))
    }
}

/// Run GGUF generation with GPU-resident path for optimal performance (PAR-200)
/// Supports inference tracing when `trace_options` is provided (APR-TRACE-001)
#[cfg(feature = "inference")]
#[allow(clippy::too_many_arguments)]
fn run_gguf_generate(
    model: realizar::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    no_gpu: bool,
    benchmark: bool,
    trace_options: Option<&RunOptions>,
    decode_fn: Option<&dyn Fn(u32) -> String>,
) -> Result<GgufGenerateResult> {
    #[cfg(feature = "cuda")]
    if !no_gpu {
        use realizar::gguf::OwnedQuantizedModelCuda;
        let verbose = trace_options.is_some_and(|o| o.verbose);
        if verbose || benchmark {
            eprintln!("Initializing CUDA GPU 0 (GPU-resident mode)...");
        }
        let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
            .map_err(|e| CliError::InferenceFailed(format!("CUDA init failed: {e}")))?;

        if benchmark {
            eprintln!("Warmup (3 iterations)...");
            for _ in 0..3 {
                let _ = cuda_model.generate_gpu_resident(input_tokens, gen_config);
            }
        }

        let infer_start = Instant::now();
        let config = cuda_model.model().config().clone();
        let tokens = traced_generate(
            || cuda_model.generate_gpu_resident(input_tokens, gen_config),
            trace_options,
            "GGUF Model (GPU)",
            &config,
            "GPU generation failed",
        )?;

        return Ok(GgufGenerateResult {
            tokens,
            inference_ms: infer_start.elapsed().as_secs_f64() * 1000.0,
        });
    }

    #[allow(unused_variables)]
    let _ = benchmark;
    let infer_start = Instant::now();
    let config = model.config().clone();
    let tokens = traced_generate(
        || model.generate_with_cache(input_tokens, gen_config),
        trace_options,
        "GGUF Model",
        &config,
        "Generation failed",
    )?;

    Ok(GgufGenerateResult {
        tokens,
        inference_ms: infer_start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Parse input features from file or stdin
#[cfg(feature = "inference")]
fn parse_input_features(input_path: Option<&PathBuf>) -> Result<Vec<f32>> {
    let input_text = if let Some(path) = input_path {
        std::fs::read_to_string(path)?
    } else {
        // Read from stdin
        use std::io::Read;
        let mut buffer = String::new();
        std::io::stdin().read_to_string(&mut buffer)?;
        buffer
    };

    // Parse as JSON array or comma-separated values
    if input_text.trim().starts_with('[') {
        // JSON array
        serde_json::from_str(&input_text)
            .map_err(|e| CliError::InvalidFormat(format!("Failed to parse JSON input: {e}")))
    } else {
        // CSV or space-separated
        input_text
            .split([',', ' ', '\n', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim()
                    .parse::<f32>()
                    .map_err(|e| CliError::InvalidFormat(format!("Invalid float: {s} - {e}")))
            })
            .collect()
    }
}

/// Format prediction output based on options
#[cfg(feature = "inference")]
fn format_prediction_output(
    output: &[f32],
    inference_time: std::time::Duration,
    options: &RunOptions,
) -> Result<String> {
    let inference_ms = inference_time.as_secs_f64() * 1000.0;

    match options.output_format.as_str() {
        "json" => {
            let result = serde_json::json!({
                "predictions": output,
                "inference_time_ms": inference_ms
            });
            serde_json::to_string_pretty(&result)
                .map_err(|e| CliError::InvalidFormat(format!("JSON serialization failed: {e}")))
        }
        _ => {
            // Default text format
            let mut result = String::new();
            result.push_str("Predictions:\n");
            for (i, &val) in output.iter().enumerate() {
                result.push_str(&format!("  [{}]: {:.6}\n", i, val));
            }
            result.push_str(&format!("\nInference time: {:.2}ms", inference_ms));
            Ok(result)
        }
    }
}

/// Print layer-level trace timing breakdown.
///
/// Only reports measurable wall-clock totals. Per-layer breakdown is not
/// available without BrickProfiler instrumentation — use `apr profile --granular`
/// for real per-operation telemetry.
///
/// PMAT-480: Layer trace shows the 8-step inference state machine with
/// per-step timing. When realizar provides TensorStats (min/max/mean/std/
/// NaN/Inf counts), those are printed per layer. Otherwise falls back to
/// aggregate timing from RunResult.
fn print_layer_trace(result: &RunResult, max_tokens: usize) {
    eprint!("{}", render_layer_trace(result, max_tokens));
}

/// Fixed share of per-token wall time attributed to each step of the 8-step
/// state machine. These are ASSUMPTIONS, not measurements — see
/// [`render_layer_trace`].
fn layer_trace_share(step: &str) -> f64 {
    match step {
        "TRANSFORMER" => 0.85,
        "LM_HEAD" => 0.08,
        "SAMPLE" => 0.02,
        _ => 0.017,
    }
}

/// Render the layer-trace table.
///
/// # These numbers are ESTIMATES and the table says so
///
/// The per-step figures are `wall_ms / tokens * <fixed share>` — a constant
/// split of one measured total, identical for every model and every prompt.
/// The table used to head that column `Time` and print the values bare, so it
/// read as a measurement: it always "proved" TRANSFORMER was 85% of the run,
/// while the real `[BRICK-PROFILE]` block a few lines above the same output
/// reported FFN 42% / Qkv 21% / LmHead 4.5% for that identical run. TOKENIZE,
/// EMBED and DECODE came out equal to the hundredth of a millisecond in every
/// run, which is the tell.
///
/// Two things follow, and both are in the rendered output now:
///
/// 1. The column is `Est. Time`, each value is prefixed `~`, and the share used
///    to derive it is printed next to it. A reader cannot mistake a derived
///    number for a measured one.
/// 2. `TOTAL` is labelled wall-clock **including model load**, and its rate is
///    labelled end-to-end. Reporting `1.0 tok/s` next to the profiler's
///    `19.2 tok/s` for the same run, with neither labelled, is a 19x
///    contradiction inside one screen of output.
///
/// Returned as a `String` so the rendering is directly assertable; the caller
/// prints it to stderr.
fn render_layer_trace(result: &RunResult, max_tokens: usize) -> String {
    use std::fmt::Write as _;

    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let total_ms = result.duration_secs * 1000.0;
    let tok_per_sec = if result.duration_secs > 0.0 {
        tokens_generated as f64 / result.duration_secs
    } else {
        0.0
    };

    // 8-step inference state machine trace
    let steps = [
        ("TOKENIZE", "Text → Token IDs"),
        ("EMBED", "Token IDs → Vectors"),
        ("TRANSFORMER", "Vectors → Vectors (×N layers)"),
        ("LM_HEAD", "Hidden → Logits"),
        ("SAMPLE", "Logits → Token ID"),
        ("DECODE", "Token ID → Text"),
    ];

    let per_token_ms = if tokens_generated > 0 {
        total_ms / tokens_generated as f64
    } else {
        total_ms
    };

    let mut out = String::new();
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", "=== Layer Trace (APR-TRACE-001) ===".cyan().bold());
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        "ESTIMATED — per-step values are a fixed share of measured wall time,".yellow()
    );
    let _ = writeln!(
        out,
        "  {}",
        "NOT per-step measurements. Same ratios for every model and prompt.".yellow()
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {:<16} {:<12} {:<8} {}",
        "Step".bold(),
        "Est. Time".bold(),
        "Share".bold(),
        "Description".bold()
    );
    let _ = writeln!(out, "  {}", "─".repeat(66));

    for (name, desc) in &steps {
        let share = layer_trace_share(name);
        let _ = writeln!(
            out,
            "  {:<16} ~{:>8.2}ms  {:>6.1}%  {}",
            name,
            per_token_ms * share,
            share * 100.0,
            desc.dimmed()
        );
    }

    let _ = writeln!(out, "  {}", "─".repeat(66));
    let _ = writeln!(
        out,
        "  {:<16} {:>9.2}ms  measured wall clock, incl. model load",
        "TOTAL", total_ms
    );
    let _ = writeln!(
        out,
        "  {:<16} {:>9.1} tok/s  end-to-end ({} tokens / total wall clock);",
        "RATE", tok_per_sec, tokens_generated
    );
    let _ = writeln!(
        out,
        "  {:<16} {}",
        "",
        "decode-only throughput is the [BRICK-PROFILE] figure, which excludes load".dimmed()
    );
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "  {}",
        "For measured per-brick µs timing: `apr profile <model> --granular`.".dimmed()
    );
    let _ = writeln!(out);
    out
}

/// Print payload trace with activation statistics (TensorStats per layer).
///
/// PMAT-480: Shows min/max/mean/std and NaN/Inf detection per layer.
/// When realizar provides real TensorStats, uses those. Otherwise reports
/// that payload-level data requires BrickProfiler or REALIZE_TRACE=1.
fn print_payload_trace(result: &RunResult, max_tokens: usize) {
    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let total_ms = result.duration_secs * 1000.0;

    eprintln!();
    eprintln!("{}", "=== Payload Trace (APR-TRACE-001) ===".cyan().bold());
    eprintln!();
    eprintln!("  Total inference: {:.2} ms", total_ms);
    eprintln!("  Tokens generated: {}", tokens_generated);
    eprintln!();

    // TensorStats header
    eprintln!(
        "  {:<24} {:>8} {:>8} {:>8} {:>8} {:>5} {:>5}",
        "Layer".bold(),
        "Min".bold(),
        "Max".bold(),
        "Mean".bold(),
        "Std".bold(),
        "NaN".bold(),
        "Inf".bold(),
    );
    eprintln!("  {}", "─".repeat(72));

    // Payload-level tensor stats require integration with realizar's
    // InferenceTrace. When not available, show guidance.
    eprintln!(
        "  {}",
        "Per-layer TensorStats require REALIZE_TRACE=1 or `apr profile --granular`.".yellow()
    );
    eprintln!(
        "  {}",
        "This enables NaN/Inf detection at the exact layer of occurrence.".dimmed()
    );
    eprintln!();
}

/// Print roofline profiling analysis (PMAT-480).
///
/// Estimates compute vs memory boundedness from throughput. For real
/// per-brick µs timing, use `apr profile <model> --granular` which
/// integrates with trueno's BrickProfiler.
fn print_roofline_profile(result: &RunResult, max_tokens: usize) {
    let tokens_generated = result.tokens_generated.unwrap_or(max_tokens);
    let total_ms = result.duration_secs * 1000.0;
    let tok_per_sec = if result.duration_secs > 0.0 {
        tokens_generated as f64 / result.duration_secs
    } else {
        0.0
    };

    // Roofline classification based on Ivanov et al. (2021):
    // M=1 decode is memory-bandwidth bound. High tok/s implies GPU
    // compute is engaged (batched prefill or tensor cores).
    let (compute_pct, memory_pct, bottleneck, recommendation) = if tok_per_sec > 50.0 {
        (
            65,
            35,
            "Compute (GPU tensor cores engaged)",
            "Efficient — GPU-accelerated path active",
        )
    } else if tok_per_sec > 20.0 {
        (
            40,
            60,
            "Mixed (memory bandwidth limited)",
            "Try quantized model (Q4K) for less data movement",
        )
    } else if tok_per_sec > 5.0 {
        (
            20,
            80,
            "Memory bandwidth (DRAM → cache)",
            "Enable GPU with --gpu, or use smaller quantization",
        )
    } else {
        (
            10,
            90,
            "Memory bandwidth (CPU, no SIMD saturation)",
            "Model too large for CPU — use GPU or smaller model",
        )
    };

    eprintln!();
    eprintln!("{}", "=== Roofline Profile (PMAT-480) ===".cyan().bold());
    eprintln!();
    eprintln!("  Throughput:     {tok_per_sec:.1} tok/s");
    eprintln!("  Latency:        {total_ms:.1} ms ({tokens_generated} tokens)");
    eprintln!("  Per-token:      {:.2} ms", total_ms / tokens_generated.max(1) as f64);
    eprintln!("  GPU used:       {}", result.used_gpu.map_or("unknown", |g| if g { "yes" } else { "no" }));
    eprintln!();
    eprintln!("  {}", "Roofline Classification".bold());
    eprintln!("  Compute bound:  {compute_pct}%");
    eprintln!("  Memory bound:   {memory_pct}%");
    eprintln!("  Bottleneck:     {bottleneck}");
    eprintln!("  Recommendation: {recommendation}");
    eprintln!();
    eprintln!(
        "  {}",
        "For per-brick µs timing: `apr profile <model> --granular`".dimmed()
    );
    eprintln!(
        "  {}",
        "For live monitoring: `apr cbtop <model> --brick-score`".dimmed()
    );
    eprintln!();
}
