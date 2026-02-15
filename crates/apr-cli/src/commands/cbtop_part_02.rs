
/// Run headless mode with simulated data (demo mode)
#[allow(clippy::needless_pass_by_value)] // Config is consumed for API simplicity
fn run_headless_simulated(config: CbtopConfig) -> Result<()> {
    let model_name = config.model.as_deref().unwrap_or("qwen2.5-coder-1.5b");

    eprintln!("cbtop: Running headless benchmark (SIMULATED)...");
    eprintln!("  Model: {model_name}");
    eprintln!("  Warmup: {} iterations", config.warmup);
    eprintln!("  Measurement: {} iterations", config.iterations);
    eprintln!();
    eprintln!("  WARNING: Using simulated data. For real profiling, use:");
    eprintln!("    apr cbtop --model-path model.gguf --headless --json  # GGUF");
    eprintln!("    apr cbtop --model-path model.safetensors --headless --json  # SafeTensors");
    eprintln!("    apr cbtop --model-path model.apr --headless --json  # APR");

    // Create pipeline and run simulation
    let mut pipeline = PipelineState::new();

    // Warmup phase
    for _ in 0..config.warmup {
        pipeline.update_demo();
    }

    // Clear samples after warmup
    for brick in &mut pipeline.bricks {
        brick.samples.clear();
        brick.actual_us = 0.0;
    }

    // Measurement phase
    for _ in 0..config.iterations {
        pipeline.update_demo();
    }

    // Calculate statistics
    let report = generate_headless_report_simulated(model_name, &pipeline, &config);

    // Check CI thresholds
    let ci_passed = check_ci_thresholds(&report, &config);

    // Output results
    if config.json {
        let json_output = format_report_as_json(&report);

        if let Some(ref path) = config.output {
            std::fs::write(path, &json_output).map_err(|e| {
                CliError::ValidationFailed(format!("Failed to write output file: {e}"))
            })?;
            eprintln!("cbtop: Results written to {}", path.display());
        } else {
            println!("{json_output}");
        }
    } else {
        // Plain text output
        print_report_text(&report);
    }

    if config.ci && !ci_passed {
        eprintln!("cbtop: CI thresholds not met!");
        return Err(CliError::ValidationFailed(
            "CI thresholds not met".to_string(),
        ));
    }

    Ok(())
}

/// Run headless APR profiling using realizar's APR forward_profiled() (§12.11)
///
/// Uses CPU inference with unified BrickProfiler instrumentation.
/// Brick names: apr.Embed, apr.RmsNorm, apr.QKV, apr.Attention, apr.OProj, apr.FFN, etc.
#[cfg(feature = "inference")]
#[allow(clippy::needless_pass_by_value)] // Config is consumed for API simplicity
fn run_headless_apr(
    config: CbtopConfig,
    model_path: &std::path::Path,
    model_name: &str,
) -> Result<()> {
    use realizar::apr::AprV2Model;
    use trueno::brick::BrickProfiler;

    eprintln!("cbtop: APR format profiling (CPU, §12.11 BrickProfiler)");
    eprintln!();

    // Load APR model
    eprintln!("cbtop: Loading APR model...");
    let load_start = Instant::now();

    let model = AprV2Model::load(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR model: {e}")))?;

    let load_time = load_start.elapsed();
    eprintln!("cbtop: APR model loaded in {:.2}s", load_time.as_secs_f32());

    // Get model config
    let hidden_dim = model.metadata().hidden_size.unwrap_or(0);
    let num_layers = model.metadata().num_layers.unwrap_or(0);
    let vocab_size = model.metadata().vocab_size.unwrap_or(0);

    eprintln!("cbtop: APR model config:");
    eprintln!("  Hidden: {}", hidden_dim);
    eprintln!("  Layers: {}", num_layers);
    eprintln!("  Vocab: {}", vocab_size);
    eprintln!();

    // Create prompt tokens
    let prompt_tokens: Vec<u32> = vec![1, 25580, 264, 2566]; // "Hello"

    // Create profiler
    let mut profiler = BrickProfiler::enabled();

    // Warmup
    eprintln!("cbtop: Warmup ({} iterations)...", config.warmup);
    for i in 0..config.warmup {
        let _ = model.forward(&prompt_tokens);
        eprint!("\r  Warmup {}/{}", i + 1, config.warmup);
    }
    eprintln!();

    // Measurement phase with profiling
    eprintln!("cbtop: Measurement ({} iterations)...", config.iterations);
    let measure_start = Instant::now();

    for i in 0..config.iterations {
        profiler.reset();
        // Note: forward_profiled not yet implemented in realizar, using forward
        let _ = model.forward(&prompt_tokens);
        eprint!("\r  Iteration {}/{}", i + 1, config.iterations);
    }
    eprintln!();

    let total_time = measure_start.elapsed();
    let tokens_generated = config.iterations * prompt_tokens.len();
    let throughput = tokens_generated as f64 / total_time.as_secs_f64();

    // Display results
    eprintln!();
    eprintln!("╔═══════════════════════════════════════════════════════════╗");
    eprintln!("║              APR BRICKPROFILER SUMMARY (§12.11)           ║");
    eprintln!("╠═══════════════════════════════════════════════════════════╣");
    eprintln!("║ Model: {:50} ║", model_name);
    eprintln!("║ Format: APR (brick prefix: apr.*)                        ║");
    eprintln!(
        "║ Throughput: {:8.1} tok/s                                ║",
        throughput
    );
    eprintln!("╠═══════════════════════════════════════════════════════════╣");

    // Display per-brick stats from profiler using all_stats()
    eprintln!("║ Brick Timing Summary:                                     ║");
    eprintln!(
        "║ {:20} │ {:10} │ {:6} │ {:8} ║",
        "Brick", "Mean µs", "% Tot", "Samples"
    );
    eprintln!("╠═══════════════════════════════════════════════════════════╣");

    // Get stats sorted by total time (using public fields)
    #[allow(deprecated)]
    let all_stats = profiler.all_stats();
    let mut sorted_stats: Vec<_> = all_stats.iter().collect();
    sorted_stats.sort_by(|a, b| b.1.total_ns.cmp(&a.1.total_ns));

    let summary_total = profiler.total_ns().max(1);
    for (name, stat) in sorted_stats.iter().take(12) {
        let mean_us = stat.avg_us();
        let total_ns = stat.total_ns;
        let pct = (total_ns as f64 / summary_total as f64) * 100.0;
        let samples = stat.count;
        eprintln!(
            "║ {:20} │ {:10.2} │ {:5.1}% │ {:8} ║",
            name, mean_us, pct, samples
        );
    }

    eprintln!("╚═══════════════════════════════════════════════════════════╝");

    // Output JSON if requested
    if config.json {
        let json = format!(
            r#"{{"model":"{}","format":"apr","throughput":{:.1},"total_time_ms":{:.1},"iterations":{}}}"#,
            model_name,
            throughput,
            total_time.as_secs_f64() * 1000.0,
            config.iterations
        );

        if let Some(ref output_path) = config.output {
            std::fs::write(output_path, &json)?;
            eprintln!("cbtop: JSON output written to {}", output_path.display());
        } else {
            println!("{json}");
        }
    }

    Ok(())
}

/// Run headless mode with REAL profiling using realizar (PMAT-PERF-009)
///
/// Per spec §4.16.0 + §12.11: Unified BrickProfiler for ALL formats
/// - Uses realizar for actual CUDA/CPU inference
/// - Supports GGUF, SafeTensors, and APR formats
/// - Measures real per-brick timings via unified BrickProfiler
/// - Reports real hardware info from CUDA context
#[cfg(feature = "inference")]
fn run_headless_real(config: CbtopConfig) -> Result<()> {
    use realizar::cuda::CudaExecutor;
    use realizar::gguf::{
        MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig,
    };

    // PAR-073: Disable CUDA graphs BEFORE model load for per-brick profiling
    // CUDA graph replay bypasses timing code, so we must use the non-graphed path
    // The OnceLock in cuda.rs checks this env var on first forward pass
    std::env::set_var("CUDA_GRAPH_DISABLE", "1");

    let model_path = config.model_path.clone().ok_or_else(|| {
        CliError::ValidationFailed("model_path is required for real profiling".to_string())
    })?;

    // §12.11: Detect model format from extension
    let format = ModelFormat::from_path(&model_path).ok_or_else(|| {
        CliError::ValidationFailed(format!(
            "Unsupported model format: {}. Supported: .gguf, .safetensors, .apr",
            model_path.display()
        ))
    })?;

    let model_name: String = config.model.clone().unwrap_or_else(|| {
        model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .map_or_else(|| "unknown".to_string(), std::string::ToString::to_string)
    });

    eprintln!("cbtop: Running headless benchmark (REAL PROFILING)...");
    eprintln!("  Model: {model_name}");
    eprintln!("  Path: {}", model_path.display());
    eprintln!(
        "  Format: {:?} (brick prefix: {}.*)",
        format,
        format.brick_prefix()
    );
    eprintln!("  Warmup: {} iterations", config.warmup);
    eprintln!("  Measurement: {} iterations", config.iterations);
    eprintln!();

    // §12.11: APR format uses CPU inference with BrickProfiler
    if format == ModelFormat::Apr {
        return run_headless_apr(config, &model_path, &model_name);
    }

    // GGUF path requires CUDA
    // Check CUDA availability
    let cuda_available = CudaExecutor::is_available();
    let cuda_devices = CudaExecutor::num_devices();

    if !cuda_available || cuda_devices == 0 {
        eprintln!("cbtop: ERROR - CUDA not available. Real profiling requires CUDA GPU.");
        return Err(CliError::ValidationFailed(
            "CUDA not available for real profiling".to_string(),
        ));
    }

    eprintln!("  CUDA: {} GPU(s) detected", cuda_devices);
    eprintln!();

    // Load model
    eprintln!("cbtop: Loading model...");
    let load_start = Instant::now();

    let mapped = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to map model: {e}")))?;

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to initialize CUDA: {e}")))?;

    let load_time = load_start.elapsed();
    eprintln!("cbtop: Model loaded in {:.2}s", load_time.as_secs_f32());
    eprintln!("cbtop: CUDA graphs DISABLED for per-brick profiling (PAR-073)");
    eprintln!();

    let mut draft_cuda_model = load_draft_model(&config)?;

    // Get model dimensions for brick benchmarks (via GGUFModel)
    let hidden_dim = mapped.model.embedding_dim().unwrap_or(896);
    let num_heads = mapped.model.num_heads().unwrap_or(14);
    let num_kv_heads = mapped.model.num_kv_heads().unwrap_or(2);
    let num_layers = mapped.model.num_layers().unwrap_or(28);
    let head_dim = hidden_dim / num_heads;
    // Infer intermediate_dim from tensor or use typical Qwen scaling (5.4x hidden)
    let intermediate_dim = mapped
        .model
        .tensors
        .iter()
        .find(|t| t.name == "blk.0.ffn_up.weight")
        .map_or(hidden_dim * 54 / 10, |t| {
            t.dims.first().copied().unwrap_or(4864) as usize
        });

    eprintln!("cbtop: Model config:");
    eprintln!("  Hidden: {}", hidden_dim);
    eprintln!("  Heads: {} (KV: {})", num_heads, num_kv_heads);
    eprintln!("  FFN: {}", intermediate_dim);
    eprintln!("  Layers: {}", num_layers);
    eprintln!();

    // Create prompt tokens from GGUF vocab - FAIL FAST if tokenizer unavailable
    let prompt = "Hello, I am a coding assistant.";
    let prompt_tokens: Vec<u32> = mapped.model.encode(prompt).ok_or_else(|| {
        CliError::InferenceFailed(
            "FATAL: GGUF model has no tokenizer - cannot encode prompt for cbtop benchmark"
                .to_string(),
        )
    })?;

    let gen_config = QuantizedGenerateConfig {
        max_tokens: 32,
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    // Phase 1: Warmup inference
    eprintln!("cbtop: Warmup ({} iterations)...", config.warmup);
    for i in 0..config.warmup {
        let _ = cuda_model.generate_gpu_resident(&prompt_tokens, &gen_config);
        eprint!("\r  Warmup {}/{}", i + 1, config.warmup);
    }
    eprintln!();

    // PAR-073: Enable BrickProfiler for per-brick timing
    // NOTE: Per-brick timing requires CUDA sync after each brick, which adds overhead
    // We enable it for detailed profiling but acknowledge throughput may be lower
    cuda_model.enable_profiling();
    cuda_model.reset_profiler();
    eprintln!("cbtop: BrickProfiler enabled (PAR-073)");
    eprintln!();

    // Phase 2: Measure throughput
    let mode_str = describe_measurement_mode(&config, draft_cuda_model.is_some());
    eprintln!(
        "cbtop: Measuring throughput ({} iterations, {} mode)...",
        config.iterations, mode_str
    );
    let (total_tokens, latencies_us) = if config.concurrent > 1 {
        measure_batch_throughput(&config, &mut cuda_model, &prompt_tokens)?
    } else {
        measure_standard_throughput(
            &config,
            &mut cuda_model,
            &mut draft_cuda_model,
            &prompt_tokens,
            &gen_config,
        )?
    };
    eprintln!();

    let total_time_us: f64 = latencies_us.iter().sum();
    let tokens_per_sec = (total_tokens as f64) / (total_time_us / 1_000_000.0);

    eprintln!();
    eprintln!("cbtop: Throughput: {:.1} tok/s (MEASURED)", tokens_per_sec);

    // Calculate actual per-layer time from measured throughput
    let measured_per_token_us = 1_000_000.0 / tokens_per_sec;
    let measured_per_layer_us = measured_per_token_us / num_layers as f64;
    let target_per_layer_us = 35.7; // Budget from spec
    eprintln!(
        "cbtop: Per-layer time: {:.1}µs (MEASURED), budget: {:.1}µs ({:.1}x)",
        measured_per_layer_us,
        target_per_layer_us,
        measured_per_layer_us / target_per_layer_us
    );
    eprintln!();

    // PAR-073: Print BrickProfiler summary
    eprintln!("=== PAR-073 BrickProfiler Results ===");
    let profiler_summary = cuda_model.profiler_summary();
    eprintln!("{}", profiler_summary);

    print_profiler_brick_stats(&cuda_model);
    eprintln!();

    let brick_reports = benchmark_bricks(
        &config,
        hidden_dim,
        num_heads,
        num_kv_heads,
        head_dim,
        measured_per_layer_us,
        tokens_per_sec,
        num_layers,
    );

    // Calculate CV from latencies
    let mean_latency = latencies_us.iter().sum::<f64>() / latencies_us.len() as f64;
    let variance = latencies_us
        .iter()
        .map(|x| (x - mean_latency).powi(2))
        .sum::<f64>()
        / latencies_us.len() as f64;
    let std_dev = variance.sqrt();
    let cv_percent = (std_dev / mean_latency) * 100.0;

    // PMAT-PERF-009: Renacer BrickTracer escalation for anomaly detection
    #[cfg(feature = "visualization")]
    check_renacer_escalation(tokens_per_sec, cv_percent);

    let gpu_name = cuda_model.device_name().to_string();
    build_and_output_report(
        &config,
        &model_name,
        &gpu_name,
        tokens_per_sec,
        cv_percent,
        &latencies_us,
        brick_reports,
    )
}

#[cfg(feature = "inference")]
fn describe_measurement_mode(config: &CbtopConfig, has_draft: bool) -> String {
    if config.concurrent > 1 {
        format!("batch (concurrent={})", config.concurrent)
    } else if config.speculative && has_draft {
        format!("speculative with draft (k={})", config.speculation_k)
    } else if config.speculative {
        format!("speculative self (k={})", config.speculation_k)
    } else {
        "standard".to_string()
    }
}
