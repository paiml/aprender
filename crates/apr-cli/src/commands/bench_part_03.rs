
/// SafeTensors format benchmark
/// GH-192: Now supports CUDA GPU acceleration
#[cfg(feature = "inference")]
fn run_safetensors_benchmark(
    path: &Path,
    config: &BenchConfig,
    use_cuda: bool,
) -> Result<BenchResult> {
    use realizar::apr::AprV2Model;
    use realizar::safetensors_infer::SafetensorsToAprConverter;

    if use_cuda {
        return run_safetensors_cuda_benchmark(path, config);
    }

    if !config.quiet {
        eprintln!("{}", "Loading SafeTensors model (CPU)...".yellow());
    }
    let start = Instant::now();

    // Convert SafeTensors to AprTransformer
    let transformer = SafetensorsToAprConverter::convert(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load SafeTensors: {e}")))?;

    let load_time = start.elapsed();
    if !config.quiet {
        eprintln!(
            "{} in {:.2}s",
            "Model ready".green(),
            load_time.as_secs_f32()
        );
        eprintln!();
    }

    // Try to load tokenizer from sibling tokenizer.json file
    let prompt_tokens: Vec<u32> = if let Some(tokenizer) =
        AprV2Model::load_tokenizer_from_path(&path.with_file_name("tokenizer.json"))
    {
        tokenizer.encode(&config.prompt)
    } else {
        // Fallback tokens for Qwen2
        vec![151643, 9707, 11, 358, 1079, 264, 11761, 18328, 13, 9842]
    };

    // Warmup
    if !config.quiet {
        eprintln!("{}", "Running warmup...".yellow());
    }
    for i in 0..config.warmup {
        let _ = transformer.forward(&prompt_tokens);
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }

    // Measurement (forward pass only - no generation for SafeTensors)
    if !config.quiet {
        eprintln!("{}", "Running benchmark (forward pass)...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let total_tokens = config.iterations * prompt_tokens.len();

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        let _ = transformer.forward(&prompt_tokens);

        let iter_time = iter_start.elapsed();
        iteration_times.push(iter_time);

        if !config.quiet {
            eprint!(
                "  Iteration {}/{}: {:.2}s\r",
                i + 1,
                config.iterations,
                iter_time.as_secs_f32()
            );
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    let first_token_time = iteration_times.first().copied().unwrap_or(Duration::ZERO);
    if !config.quiet {
        eprintln!();
        eprintln!();
    }

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// SafeTensors CUDA benchmark (GH-192)
/// Uses SafeTensorsCudaModel for direct GPU loading
#[cfg(feature = "inference")]
fn run_safetensors_cuda_benchmark(path: &Path, config: &BenchConfig) -> Result<BenchResult> {
    use realizar::apr::AprV2Model;
    use realizar::safetensors_cuda::SafeTensorsCudaModel;

    if !config.quiet {
        eprintln!("{}", "Loading SafeTensors model (GPU)...".yellow());
    }
    let start = Instant::now();

    // Load SafeTensors directly to GPU (PMAT-116)
    let mut model = SafeTensorsCudaModel::load(path, 0)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load SafeTensors CUDA: {e}")))?;

    // Try to load tokenizer from sibling tokenizer.json file
    let prompt_tokens: Vec<u32> = if let Some(tokenizer) =
        AprV2Model::load_tokenizer_from_path(&path.with_file_name("tokenizer.json"))
    {
        tokenizer.encode(&config.prompt)
    } else {
        // Fallback tokens for Qwen2
        vec![151643, 9707, 11, 358, 1079, 264, 11761, 18328, 13, 9842]
    };

    // EOS token for Qwen2 models
    let eos_token: u32 = 151645;

    let load_time = start.elapsed();
    if !config.quiet {
        eprintln!(
            "{} in {:.2}s (GPU device 0)",
            "Model ready".green(),
            load_time.as_secs_f32()
        );
        eprintln!();
    }

    // Warmup - reset KV cache for each iteration
    if !config.quiet {
        eprintln!("{}", "Running warmup (GPU)...".yellow());
    }
    for i in 0..config.warmup {
        model.reset_kv_cache();
        let _ = model.generate(&prompt_tokens, config.max_tokens.min(16), eos_token);
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }

    // Measurement - reset KV cache for each iteration
    if !config.quiet {
        eprintln!("{}", "Running benchmark (GPU)...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        model.reset_kv_cache();
        let output = model
            .generate(&prompt_tokens, config.max_tokens.min(32), eos_token)
            .unwrap_or_else(|_| prompt_tokens.clone());
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());

        let iter_time = iter_start.elapsed();
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

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
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
) -> Result<BenchResult> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, OwnedQuantizedModelCuda};

    if !config.quiet {
        eprintln!("{}", "Initializing CUDA model...".cyan());
    }

    // Create MappedGGUFModel from path (memory-mapped for efficiency)
    let mapped = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to map model: {e}")))?;

    // Create OwnedQuantizedModel from mapped model
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    // Wrap with CUDA executor for GPU acceleration
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

    // Warmup with CUDA
    if !config.quiet {
        eprintln!("{}", "Running warmup (GPU)...".yellow());
    }
    for i in 0..config.warmup {
        // Try GPU-resident path for better performance
        match cuda_model.generate_gpu_resident(prompt_tokens, gen_config) {
            Ok(_) => {}
            Err(e) => {
                eprintln!("\n  Warmup error: {e}");
                return Err(CliError::ValidationFailed(format!(
                    "GPU warmup failed: {e}"
                )));
            }
        }
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }

    // Measurement with CUDA
    if !config.quiet {
        eprintln!("{}", "Running benchmark (GPU)...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        // Use GPU-resident path for maximum performance
        let output = match cuda_model.generate_gpu_resident(prompt_tokens, gen_config) {
            Ok(tokens) => tokens,
            Err(e) => {
                eprintln!("\n  Generation error: {e}");
                return Err(CliError::ValidationFailed(format!(
                    "GPU generation failed: {e}"
                )));
            }
        };
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());

        let iter_time = iter_start.elapsed();
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
) -> Result<BenchResult> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    // Use memory-mapped loading for CPU path
    let mapped = MappedGGUFModel::from_path(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to mmap model: {e}")))?;
    let model = OwnedQuantizedModel::from_mapped(&mapped)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to create model: {e}")))?;

    let load_time = start.elapsed();
    if !config.quiet {
        eprintln!(
            "{} in {:.2}s (CPU)",
            "Model ready".green(),
            load_time.as_secs_f32()
        );
        eprintln!();
    }

    // Warmup
    if !config.quiet {
        eprintln!("{}", "Running warmup (CPU)...".yellow());
    }
    for i in 0..config.warmup {
        let _ = model.generate_with_cache(prompt_tokens, gen_config);
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }

    // Measurement
    if !config.quiet {
        eprintln!("{}", "Running benchmark (CPU)...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        let output = model
            .generate_with_cache(prompt_tokens, gen_config)
            .unwrap_or_default();
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());

        let iter_time = iter_start.elapsed();
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

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
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
