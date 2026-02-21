
/// SafeTensors format benchmark
/// GH-192: Now supports CUDA GPU acceleration
#[cfg(feature = "inference")]
fn run_safetensors_benchmark(
    path: &Path,
    config: &BenchConfig,
    use_cuda: bool,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<BenchResult> {
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    if use_cuda {
        return run_safetensors_cuda_benchmark(path, config, tracer);
    }

    bench_log(config, &"Loading SafeTensors model (CPU)...".yellow().to_string());
    let start = Instant::now();

    let transformer = SafetensorsToAprConverter::convert(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load SafeTensors: {e}")))?;

    bench_log_ready(config, start.elapsed(), "");
    let prompt_tokens = resolve_safetensors_tokens(path, &config.prompt);

    run_forward_warmup(config, &transformer, &prompt_tokens);

    bench_log(config, &"Running benchmark (forward pass)...".yellow().to_string());
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let total_tokens = config.iterations * prompt_tokens.len();
    let budget_us = config.max_tokens as u64 * 100_000;

    for i in 0..config.iterations {
        let traced = tracer.trace("bench_safetensors_cpu_iter", budget_us, || {
            transformer.forward(&prompt_tokens)
        });
        let _ = traced.result;
        let iter_time = Duration::from_micros(traced.duration_us);
        iteration_times.push(iter_time);
        bench_log_iter(config, i, iter_time, None);
    }
    let first_token_time = iteration_times.first().copied().unwrap_or(Duration::ZERO);
    bench_log_done(config);

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// Resolve prompt tokens from sibling tokenizer.json or fallback.
#[cfg(feature = "inference")]
fn resolve_safetensors_tokens(path: &Path, prompt: &str) -> Vec<u32> {
    use realizar::apr::AprV2Model;
    if let Some(tokenizer) =
        AprV2Model::load_tokenizer_from_path(&path.with_file_name("tokenizer.json"))
    {
        tokenizer.encode(prompt)
    } else {
        vec![151643, 9707, 11, 358, 1079, 264, 11761, 18328, 13, 9842]
    }
}

/// Run forward-pass warmup for SafeTensors CPU benchmark.
#[cfg(feature = "inference")]
fn run_forward_warmup(
    config: &BenchConfig,
    transformer: &realizar::apr_transformer::AprTransformer,
    prompt_tokens: &[u32],
) {
    bench_log(config, &"Running warmup...".yellow().to_string());
    for i in 0..config.warmup {
        let _ = transformer.forward(prompt_tokens);
        bench_log_iter(config, i, Duration::ZERO, None);
    }
    bench_log_done(config);
}

/// Log a message if not in quiet mode.
#[cfg(feature = "inference")]
fn bench_log(config: &BenchConfig, msg: &str) {
    if !config.quiet {
        eprintln!("{msg}");
    }
}

/// Log "Model ready" with timing.
#[cfg(feature = "inference")]
fn bench_log_ready(config: &BenchConfig, elapsed: Duration, suffix: &str) {
    if !config.quiet {
        eprintln!("{} in {:.2}s{suffix}", "Model ready".green(), elapsed.as_secs_f32());
        eprintln!();
    }
}

/// Log iteration progress.
#[cfg(feature = "inference")]
fn bench_log_iter(config: &BenchConfig, i: usize, time: Duration, tokens: Option<usize>) {
    if config.quiet {
        return;
    }
    if let Some(tok) = tokens {
        eprint!("  Iteration {}/{}: {} tokens in {:.2}s\r", i + 1, config.iterations, tok, time.as_secs_f32());
    } else if time > Duration::ZERO {
        eprint!("  Iteration {}/{}: {:.2}s\r", i + 1, config.iterations, time.as_secs_f32());
    } else {
        eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
    }
    std::io::Write::flush(&mut std::io::stderr()).ok();
}

/// Log benchmark phase completion.
#[cfg(feature = "inference")]
fn bench_log_done(config: &BenchConfig) {
    if !config.quiet {
        eprintln!("  Complete        ");
        eprintln!();
    }
}

/// SafeTensors CUDA benchmark (GH-192)
/// Uses SafeTensorsCudaModel for direct GPU loading
#[cfg(feature = "inference")]
fn run_safetensors_cuda_benchmark(
    path: &Path,
    config: &BenchConfig,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<BenchResult> {
    use realizar::safetensors_cuda::SafeTensorsCudaModel;

    bench_log(config, &"Loading SafeTensors model (GPU)...".yellow().to_string());
    let start = Instant::now();

    let mut model = SafeTensorsCudaModel::load(path, 0)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load SafeTensors CUDA: {e}")))?;

    let prompt_tokens = resolve_safetensors_tokens(path, &config.prompt);
    let eos_token: u32 = 151645;

    bench_log_ready(config, start.elapsed(), " (GPU device 0)");

    bench_log(config, &"Running warmup (GPU)...".yellow().to_string());
    for i in 0..config.warmup {
        model.reset_kv_cache();
        let _ = model.generate(&prompt_tokens, config.max_tokens.min(16), eos_token);
        bench_log_iter(config, i, Duration::ZERO, None);
    }
    bench_log_done(config);

    let (iteration_times, total_tokens, first_token_time) =
        run_safetensors_cuda_measurement(&mut model, &prompt_tokens, eos_token, config, tracer)?;

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// Run measurement iterations for SafeTensors CUDA benchmark.
#[cfg(feature = "inference")]
fn run_safetensors_cuda_measurement(
    model: &mut realizar::safetensors_cuda::SafeTensorsCudaModel,
    prompt_tokens: &[u32],
    eos_token: u32,
    config: &BenchConfig,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<(Vec<Duration>, usize, Duration)> {
    bench_log(config, &"Running benchmark (GPU)...".yellow().to_string());
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let budget_us = config.max_tokens as u64 * 100_000;

    for i in 0..config.iterations {
        model.reset_kv_cache();
        let traced = tracer.trace("bench_safetensors_gpu_iter", budget_us, || {
            model
                .generate(prompt_tokens, config.max_tokens.min(32), eos_token)
                .unwrap_or_else(|_| prompt_tokens.to_vec())
        });
        let output = traced.result;
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());
        let iter_time = Duration::from_micros(traced.duration_us);
        iteration_times.push(iter_time);
        total_tokens += tokens_generated;
        if i == 0 {
            first_token_time =
                Duration::from_secs_f64(iter_time.as_secs_f64() / tokens_generated.max(1) as f64);
        }
        bench_log_iter(config, i, iter_time, Some(tokens_generated));
    }
    bench_log_done(config);

    Ok((iteration_times, total_tokens, first_token_time))
}

/// Run warmup iterations for CUDA benchmark.
#[cfg(feature = "inference")]
fn run_cuda_warmup(
    cuda_model: &mut realizar::gguf::OwnedQuantizedModelCuda,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    config: &BenchConfig,
) -> Result<()> {
    if !config.quiet {
        eprintln!("{}", "Running warmup (GPU)...".yellow());
    }
    for i in 0..config.warmup {
        cuda_model
            .generate_gpu_resident(prompt_tokens, gen_config)
            .map_err(|e| {
                eprintln!("\n  Warmup error: {e}");
                CliError::ValidationFailed(format!("GPU warmup failed: {e}"))
            })?;
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }
    Ok(())
}

/// Run measurement iterations for CUDA benchmark.
#[cfg(feature = "inference")]
fn run_cuda_measurement(
    cuda_model: &mut realizar::gguf::OwnedQuantizedModelCuda,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    config: &BenchConfig,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<(Vec<Duration>, usize, Duration)> {
    if !config.quiet {
        eprintln!("{}", "Running benchmark (GPU)...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let budget_us = config.max_tokens as u64 * 100_000;

    for i in 0..config.iterations {
        let traced = tracer.trace("bench_gpu_iter", budget_us, || {
            cuda_model
                .generate_gpu_resident(prompt_tokens, gen_config)
        });
        let output = traced.result.map_err(|e| {
            eprintln!("\n  Generation error: {e}");
            CliError::ValidationFailed(format!("GPU generation failed: {e}"))
        })?;
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());

        let iter_time = Duration::from_micros(traced.duration_us);
        iteration_times.push(iter_time);
        total_tokens += tokens_generated;

        if i == 0 {
            first_token_time =
                Duration::from_secs_f64(iter_time.as_secs_f64() / tokens_generated.max(1) as f64);
        }

        if !config.quiet {
            eprint!(
                "  Iteration {}/{}: {} tokens in {:.2}s\r",
                i + 1,
                config.iterations,
                tokens_generated,
                iter_time.as_secs_f32()
            );
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!();
        eprintln!();
    }
    Ok((iteration_times, total_tokens, first_token_time))
}

/// CUDA GPU-accelerated benchmark path
#[cfg(feature = "inference")]
fn run_cuda_benchmark(
    _gguf: &realizar::gguf::GGUFModel,
    _model_bytes: &[u8],
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    config: &BenchConfig,
    start: Instant,
    model_path: &Path,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<BenchResult> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    if !config.quiet {
        eprintln!("{}", "Initializing CUDA model...".cyan());
    }

    let mapped = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to map model: {e}")))?;

    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let mut cuda_model = OwnedQuantizedModelCuda::new(model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to initialize CUDA: {e}")))?;

    let load_time = start.elapsed();
    if !config.quiet {
        eprintln!(
            "{} in {:.2}s (GPU device 0)",
            "Model ready".green(),
            load_time.as_secs_f32()
        );
        eprintln!();
    }

    run_cuda_warmup(&mut cuda_model, prompt_tokens, gen_config, config)?;

    let (iteration_times, total_tokens, first_token_time) =
        run_cuda_measurement(&mut cuda_model, prompt_tokens, gen_config, config, tracer)?;

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// CPU-based benchmark fallback path
#[cfg(feature = "inference")]
fn run_cpu_benchmark(
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    config: &BenchConfig,
    start: Instant,
    path: &Path,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<BenchResult> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to mmap model: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    bench_log_ready(config, start.elapsed(), " (CPU)");

    run_cpu_warmup(&model, prompt_tokens, gen_config, config);

    let (iteration_times, total_tokens, first_token_time) =
        run_cpu_measurement(&model, prompt_tokens, gen_config, config, tracer);

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// Run CPU warmup iterations.
#[cfg(feature = "inference")]
fn run_cpu_warmup(
    model: &realizar::gguf::OwnedQuantizedModel,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    config: &BenchConfig,
) {
    bench_log(config, &"Running warmup (CPU)...".yellow().to_string());
    for i in 0..config.warmup {
        let _ = model.generate_with_cache(prompt_tokens, gen_config);
        bench_log_iter(config, i, Duration::ZERO, None);
    }
    bench_log_done(config);
}

/// Run CPU measurement iterations.
#[cfg(feature = "inference")]
fn run_cpu_measurement(
    model: &realizar::gguf::OwnedQuantizedModel,
    prompt_tokens: &[u32],
    gen_config: &realizar::gguf::QuantizedGenerateConfig,
    config: &BenchConfig,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> (Vec<Duration>, usize, Duration) {
    bench_log(config, &"Running benchmark (CPU)...".yellow().to_string());
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let budget_us = config.max_tokens as u64 * 100_000;

    for i in 0..config.iterations {
        let traced = tracer.trace("bench_cpu_iter", budget_us, || {
            model
                .generate_with_cache(prompt_tokens, gen_config)
                .unwrap_or_default()
        });
        let output = traced.result;
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());
        let iter_time = Duration::from_micros(traced.duration_us);
        iteration_times.push(iter_time);
        total_tokens += tokens_generated;
        if i == 0 {
            first_token_time =
                Duration::from_secs_f64(iter_time.as_secs_f64() / tokens_generated.max(1) as f64);
        }
        bench_log_iter(config, i, iter_time, Some(tokens_generated));
    }
    bench_log_done(config);

    (iteration_times, total_tokens, first_token_time)
}

/// Calculate benchmark statistics from iteration timings
#[cfg(feature = "inference")]
fn calculate_benchmark_stats(
    iteration_times: Vec<Duration>,
    total_tokens: usize,
    first_token_time: Duration,
    config: &BenchConfig,
) -> Result<BenchResult> {
    let total_time: Duration = iteration_times.iter().sum();
    // GH-254: Guard against 0.0 tok/s (division by zero or zero tokens)
    let tokens_per_second = if total_tokens == 0 || total_time.as_secs_f64() <= 0.0 {
        0.0
    } else {
        total_tokens as f64 / total_time.as_secs_f64()
    };
    let mean_time = total_time / config.iterations as u32;

    let mut sorted_times = iteration_times.clone();
    sorted_times.sort();
    let median_time = sorted_times[config.iterations / 2];

    let mean_ms = mean_time.as_secs_f64() * 1000.0;
    let variance: f64 = iteration_times
        .iter()
        .map(|t| {
            let diff = t.as_secs_f64() * 1000.0 - mean_ms;
            diff * diff
        })
        .sum::<f64>()
        / config.iterations as f64;
    let std_dev = Duration::from_secs_f64(variance.sqrt() / 1000.0);

    // GH-254: Use same threshold as run() (10 tok/s per spec H12)
    let passed = tokens_per_second >= 10.0;

    Ok(BenchResult {
        total_tokens,
        total_time,
        tokens_per_second,
        time_to_first_token: first_token_time,
        iteration_times,
        mean_time,
        median_time,
        std_dev,
        passed,
    })
}
