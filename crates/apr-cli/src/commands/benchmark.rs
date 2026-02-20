
/// Run warmup iterations with progress display.
fn run_bench_warmup<F: FnMut()>(config: &BenchConfig, count: usize, mut f: F) {
    if !config.quiet {
        eprintln!("{}", "Running warmup...".yellow());
    }
    for i in 0..count {
        f();
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, count);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }
}

/// Print benchmark progress for a single iteration.
fn print_bench_progress(config: &BenchConfig, i: usize, tokens: usize, time: Duration) {
    if !config.quiet {
        eprint!(
            "  Iteration {}/{}: {} tokens in {:.2}s\r",
            i + 1,
            config.iterations,
            tokens,
            time.as_secs_f32()
        );
        std::io::Write::flush(&mut std::io::stderr()).ok();
    }
}

fn print_results(result: &BenchResult) {
    output::section("Results");
    println!();

    // Throughput (the key metric)
    let throughput_str = format!("{:.1} tok/s", result.tokens_per_second);
    if result.passed {
        println!(
            "{} {} {}",
            "Throughput:".white().bold(),
            throughput_str.green().bold(),
            "(PASS: >= 10 tok/s)".green()
        );
    } else {
        println!(
            "{} {} {}",
            "Throughput:".white().bold(),
            throughput_str.red().bold(),
            "(FAIL: < 10 tok/s)".red()
        );
    }

    println!();
    output::kv("Total tokens", result.total_tokens);
    output::kv(
        "Total time",
        format!("{:.2}s", result.total_time.as_secs_f32()),
    );
    output::kv(
        "Time to first token",
        format!("{:.0}ms", result.time_to_first_token.as_secs_f64() * 1000.0),
    );
    println!();
    output::kv(
        "Mean iteration time",
        format!("{:.2}s", result.mean_time.as_secs_f32()),
    );
    output::kv(
        "Median iteration time",
        format!("{:.2}s", result.median_time.as_secs_f32()),
    );
    output::kv(
        "Std deviation",
        format!("{:.0}ms", result.std_dev.as_secs_f64() * 1000.0),
    );
    println!();

    // Performance grade (Dean & Ghemawat 2025 style)
    let grade = if result.tokens_per_second >= 100.0 {
        "A+ (Excellent)".green()
    } else if result.tokens_per_second >= 50.0 {
        "A (Very Good)".green()
    } else if result.tokens_per_second >= 20.0 {
        "B (Good)".blue()
    } else if result.tokens_per_second >= 10.0 {
        "C (Acceptable)".yellow()
    } else {
        "F (Below Threshold)".red()
    };
    output::kv("Performance Grade", grade);
}

/// Realizar-based benchmark for model inference
///
/// Supports all formats: GGUF, APR, and SafeTensors.
/// Automatically uses CUDA GPU acceleration when available for GGUF.
#[cfg(feature = "inference")]
fn run_realizar_benchmark(path: &Path, config: &BenchConfig) -> Result<BenchResult> {
    use realizar::cuda::CudaExecutor;
    use realizar::format::{detect_format, ModelFormat};

    // Read first 8 bytes for format detection
    let header_bytes = std::fs::read(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;

    // Detect format
    let format = detect_format(&header_bytes[..8.min(header_bytes.len())])
        .map_err(|e| CliError::ValidationFailed(format!("Failed to detect format: {e}")))?;

    if !config.quiet {
        eprintln!("{} {}", "Format:".cyan().bold(), format.to_string().green());
    }

    // Check CUDA availability (only used for GGUF currently)
    let cuda_available = CudaExecutor::is_available();
    let cuda_devices = CudaExecutor::num_devices();

    if !config.quiet {
        if cuda_available && cuda_devices > 0 {
            eprintln!(
                "{} {} GPU(s) detected",
                "CUDA:".cyan().bold(),
                cuda_devices.to_string().green()
            );
        } else {
            eprintln!(
                "{} {}",
                "CUDA:".cyan().bold(),
                "Not available (CPU mode)".yellow()
            );
        }
    }

    // Route to format-specific benchmark
    // GH-192: All formats now support CUDA acceleration
    let use_cuda = cuda_available && cuda_devices > 0;
    let tracer = renacer::brick_tracer::BrickTracer::new_local();
    match format {
        ModelFormat::Gguf => run_gguf_benchmark(path, config, use_cuda, &tracer),
        ModelFormat::Apr => run_apr_benchmark(path, config, use_cuda, &tracer),
        ModelFormat::SafeTensors => run_safetensors_benchmark(path, config, use_cuda, &tracer),
    }
}

/// GGUF format benchmark (supports GPU acceleration)
#[cfg(feature = "inference")]
fn run_gguf_benchmark(
    path: &Path,
    config: &BenchConfig,
    use_cuda: bool,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<BenchResult> {
    use realizar::gguf::{GGUFModel, QuantizedGenerateConfig};

    if !config.quiet {
        eprintln!("{}", "Loading GGUF model...".yellow());
    }
    let start = Instant::now();

    // Load model for tokenization
    let model_bytes = std::fs::read(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to read model: {e}")))?;
    let gguf = GGUFModel::from_bytes(&model_bytes)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to parse GGUF: {e}")))?;

    // Tokenize prompt
    let prompt_tokens: Vec<u32> = gguf
        .encode(&config.prompt)
        .unwrap_or_else(|| vec![151643, 9707, 11, 358, 1079, 264, 11761, 18328, 13, 9842]);

    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens.min(128),
        temperature: 0.0,
        top_k: 1,
        ..Default::default()
    };

    if use_cuda {
        match run_cuda_benchmark(
            &gguf,
            &model_bytes,
            &prompt_tokens,
            &gen_config,
            config,
            start,
            path,
            tracer,
        ) {
            Ok(result) => Ok(result),
            Err(e) => {
                // GH-284: Fall back to CPU on CUDA capability mismatch (e.g. missing QkNorm kernel)
                if !config.quiet {
                    eprintln!(
                        "{}",
                        format!("CUDA init failed, falling back to CPU: {e}").yellow()
                    );
                }
                let cpu_start = Instant::now();
                run_cpu_benchmark(&prompt_tokens, &gen_config, config, cpu_start, path, tracer)
            }
        }
    } else {
        run_cpu_benchmark(&prompt_tokens, &gen_config, config, start, path, tracer)
    }
}

/// Resolve prompt tokens from APR model's tokenizer, with fallbacks.
#[cfg(feature = "inference")]
fn resolve_apr_prompt_tokens(path: &Path, prompt: &str) -> Vec<u32> {
    use realizar::apr::AprV2Model;

    if let Some(tokenizer) =
        AprV2Model::load_tokenizer_from_path(&path.with_file_name("tokenizer.json"))
    {
        tokenizer.encode(prompt)
    } else if let Some((vocab, _, _)) = AprV2Model::load_tokenizer_from_sibling(path) {
        let token_to_id: std::collections::HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        prompt
            .split_whitespace()
            .filter_map(|w| token_to_id.get(w).copied())
            .collect()
    } else {
        vec![1, 2, 3, 4, 5]
    }
}

/// Handle fallback when generation produced zero new tokens (GH-254).
///
/// Returns adjusted `total_tokens` for forward-pass throughput reporting.
#[cfg(feature = "inference")]
fn handle_zero_generation_fallback(
    generation_failed: bool,
    total_tokens: usize,
    iterations: usize,
    prompt_len: usize,
    _quiet: bool,
) -> usize {
    if generation_failed && total_tokens == 0 {
        eprintln!(
            "{}",
            "Note: Generation produced 0 new tokens, reporting forward-pass throughput.".yellow()
        );
        iterations * prompt_len
    } else {
        total_tokens
    }
}

/// APR format benchmark
/// GH-192: Now supports CUDA GPU acceleration and uses KV cache for O(n) generation
#[cfg(feature = "inference")]
fn run_apr_benchmark(
    path: &Path,
    config: &BenchConfig,
    use_cuda: bool,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<BenchResult> {
    use realizar::apr_transformer::{AprTransformer, GenerateConfig};

    if use_cuda {
        let result = run_apr_cuda_benchmark(path, config, tracer)?;
        if result.total_tokens > 0 {
            return Ok(result);
        }
        if !config.quiet {
            eprintln!(
                "{}",
                "GPU generated 0 tokens, falling back to CPU...".yellow()
            );
        }
    }

    if !config.quiet {
        eprintln!("{}", "Loading APR model (CPU)...".yellow());
    }
    let start = Instant::now();

    let transformer = AprTransformer::from_apr_file(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let load_time = start.elapsed();
    if !config.quiet {
        eprintln!(
            "{} in {:.2}s",
            "Model ready".green(),
            load_time.as_secs_f32()
        );
        eprintln!();
    }

    let prompt_tokens = resolve_apr_prompt_tokens(path, &config.prompt);

    let gen_config = GenerateConfig {
        max_tokens: config.max_tokens.min(32),
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
    };

    // Warmup (untraced)
    run_bench_warmup(config, config.warmup, || {
        let _ = transformer.generate_with_cache(&prompt_tokens, &gen_config);
    });

    // Measurement
    if !config.quiet {
        eprintln!("{}", "Running benchmark...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let mut generation_failed = false;
    let budget_us = config.max_tokens as u64 * 100_000;

    for i in 0..config.iterations {
        let traced = tracer.trace("bench_apr_iter", budget_us, || {
            transformer
                .generate_with_cache(&prompt_tokens, &gen_config)
                .unwrap_or_else(|_| prompt_tokens.clone())
        });
        let output = traced.result;
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());

        let iter_time = Duration::from_micros(traced.duration_us);
        iteration_times.push(iter_time);
        total_tokens += tokens_generated;

        if i == 0 {
            first_token_time =
                Duration::from_secs_f64(iter_time.as_secs_f64() / tokens_generated.max(1) as f64);
            if tokens_generated == 0 {
                generation_failed = true;
            }
        }

        print_bench_progress(config, i, tokens_generated, iter_time);
    }
    if !config.quiet {
        eprintln!();
    }

    // GH-254: If generation produced 0 new tokens, fall back to forward-pass throughput
    total_tokens = handle_zero_generation_fallback(
        generation_failed,
        total_tokens,
        config.iterations,
        prompt_tokens.len(),
        config.quiet,
    );
    if !config.quiet {
        eprintln!();
    }

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// APR format CUDA benchmark (GH-192)
#[cfg(feature = "inference")]
fn run_apr_cuda_benchmark(
    path: &Path,
    config: &BenchConfig,
    tracer: &renacer::brick_tracer::BrickTracer,
) -> Result<BenchResult> {
    use realizar::apr::{AprV2Model, AprV2ModelCuda};

    if !config.quiet {
        eprintln!("{}", "Loading APR model (GPU)...".yellow());
    }
    let start = Instant::now();

    // First load APR model (CPU), then wrap with CUDA
    let cpu_model = AprV2Model::load(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let tensor_count = cpu_model.tensor_count();

    // Use embedded tokenizer if available, else fall back to sibling/default
    let prompt_tokens: Vec<u32> = if let Some(tokenizer) = cpu_model.load_embedded_bpe_tokenizer() {
        tokenizer.encode(&config.prompt)
    } else {
        resolve_apr_prompt_tokens(path, &config.prompt)
    };

    let mut model = AprV2ModelCuda::new(cpu_model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to init APR CUDA: {e}")))?;

    let eos_token: u32 = 151645;

    let load_time = start.elapsed();
    if !config.quiet {
        eprintln!(
            "{} in {:.2}s ({} tensors, GPU device 0)",
            "Model ready".green(),
            load_time.as_secs_f32(),
            tensor_count
        );
        eprintln!();
    }

    // Warmup (untraced)
    run_bench_warmup(config, config.warmup, || {
        model.reset_kv_cache();
        let _ = model.generate_cuda_with_cache(&prompt_tokens, config.max_tokens.min(16), eos_token);
    });

    // Measurement
    if !config.quiet {
        eprintln!("{}", "Running benchmark (GPU)...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let budget_us = config.max_tokens as u64 * 100_000;

    for i in 0..config.iterations {
        model.reset_kv_cache();
        let traced = tracer.trace("bench_apr_gpu_iter", budget_us, || {
            model
                .generate_cuda_with_cache(&prompt_tokens, config.max_tokens.min(32), eos_token)
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

        print_bench_progress(config, i, tokens_generated, iter_time);
    }
    if !config.quiet {
        eprintln!();
        eprintln!();
    }

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}
