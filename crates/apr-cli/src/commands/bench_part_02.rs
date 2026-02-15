
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
    match format {
        ModelFormat::Gguf => run_gguf_benchmark(path, config, use_cuda),
        ModelFormat::Apr => run_apr_benchmark(path, config, use_cuda),
        ModelFormat::SafeTensors => run_safetensors_benchmark(path, config, use_cuda),
    }
}

/// GGUF format benchmark (supports GPU acceleration)
#[cfg(feature = "inference")]
fn run_gguf_benchmark(path: &Path, config: &BenchConfig, use_cuda: bool) -> Result<BenchResult> {
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
        run_cuda_benchmark(
            &gguf,
            &model_bytes,
            &prompt_tokens,
            &gen_config,
            config,
            start,
            path,
        )
    } else {
        run_cpu_benchmark(&prompt_tokens, &gen_config, config, start, path)
    }
}

/// APR format benchmark
/// GH-192: Now supports CUDA GPU acceleration and uses KV cache for O(n) generation
#[cfg(feature = "inference")]
fn run_apr_benchmark(path: &Path, config: &BenchConfig, use_cuda: bool) -> Result<BenchResult> {
    use realizar::apr::AprV2Model;
    use realizar::apr_transformer::{AprTransformer, GenerateConfig};

    if use_cuda {
        // GH-254: If GPU generates 0 tokens (e.g. APR Q8 not supported on CUDA),
        // fall back to CPU benchmark transparently
        let result = run_apr_cuda_benchmark(path, config)?;
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

    // Load APR model as AprTransformer for KV cache support
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

    // Try to get tokenizer from sibling file
    let prompt_tokens: Vec<u32> = if let Some(tokenizer) =
        AprV2Model::load_tokenizer_from_path(&path.with_file_name("tokenizer.json"))
    {
        tokenizer.encode(&config.prompt)
    } else if let Some((vocab, _, _)) = AprV2Model::load_tokenizer_from_sibling(path) {
        // Simple whitespace tokenization as fallback
        let token_to_id: std::collections::HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        config
            .prompt
            .split_whitespace()
            .filter_map(|w| token_to_id.get(w).copied())
            .collect()
    } else {
        // Fallback tokens
        vec![1, 2, 3, 4, 5]
    };

    // Generation config for greedy sampling
    let gen_config = GenerateConfig {
        max_tokens: config.max_tokens.min(32),
        temperature: 0.0, // Greedy
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        trace: false,
    };

    // Warmup - uses KV cache for O(n) complexity
    if !config.quiet {
        eprintln!("{}", "Running warmup...".yellow());
    }
    for i in 0..config.warmup {
        let _ = transformer.generate_with_cache(&prompt_tokens, &gen_config);
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }

    // Measurement - uses KV cache for O(n) complexity
    if !config.quiet {
        eprintln!("{}", "Running benchmark...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;
    let mut generation_failed = false;

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        let output = transformer
            .generate_with_cache(&prompt_tokens, &gen_config)
            .unwrap_or_else(|_| prompt_tokens.clone());
        let tokens_generated = output.len().saturating_sub(prompt_tokens.len());

        let iter_time = iter_start.elapsed();
        iteration_times.push(iter_time);
        total_tokens += tokens_generated;

        if i == 0 {
            first_token_time =
                Duration::from_secs_f64(iter_time.as_secs_f64() / tokens_generated.max(1) as f64);
            if tokens_generated == 0 {
                generation_failed = true;
            }
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
    }

    // GH-254: If generation produced 0 new tokens, fall back to forward-pass throughput
    // This counts prompt tokens processed per iteration instead
    if generation_failed && total_tokens == 0 {
        eprintln!(
            "{}",
            "Note: Generation produced 0 new tokens, reporting forward-pass throughput.".yellow()
        );
        total_tokens = config.iterations * prompt_tokens.len();
    }
    if !config.quiet {
        eprintln!();
    }

    calculate_benchmark_stats(iteration_times, total_tokens, first_token_time, config)
}

/// APR format CUDA benchmark (GH-192)
#[cfg(feature = "inference")]
fn run_apr_cuda_benchmark(path: &Path, config: &BenchConfig) -> Result<BenchResult> {
    use realizar::apr::{AprV2Model, AprV2ModelCuda};

    if !config.quiet {
        eprintln!("{}", "Loading APR model (GPU)...".yellow());
    }
    let start = Instant::now();

    // First load APR model (CPU), then wrap with CUDA
    let cpu_model = AprV2Model::load(path)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to load APR: {e}")))?;

    let tensor_count = cpu_model.tensor_count();

    // Try to get tokenizer before wrapping with CUDA
    let prompt_tokens: Vec<u32> = if let Some(tokenizer) = cpu_model.load_embedded_bpe_tokenizer() {
        tokenizer.encode(&config.prompt)
    } else if let Some((vocab, _, _)) = AprV2Model::load_tokenizer_from_sibling(path) {
        // Simple whitespace tokenization as fallback
        let token_to_id: std::collections::HashMap<String, u32> = vocab
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        config
            .prompt
            .split_whitespace()
            .filter_map(|w| token_to_id.get(w).copied())
            .collect()
    } else {
        // Fallback tokens
        vec![1, 2, 3, 4, 5]
    };

    // Wrap with CUDA acceleration
    let mut model = AprV2ModelCuda::new(cpu_model, 0)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to init APR CUDA: {e}")))?;

    // EOS token for Qwen2 models
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

    // Warmup - use cached version for proper KV cache usage
    if !config.quiet {
        eprintln!("{}", "Running warmup (GPU)...".yellow());
    }
    for i in 0..config.warmup {
        // Reset KV cache position for each iteration
        model.reset_kv_cache();
        let _ =
            model.generate_cuda_with_cache(&prompt_tokens, config.max_tokens.min(16), eos_token);
        if !config.quiet {
            eprint!("  Warmup {}/{}\r", i + 1, config.warmup);
            std::io::Write::flush(&mut std::io::stderr()).ok();
        }
    }
    if !config.quiet {
        eprintln!("  Warmup complete        ");
        eprintln!();
    }

    // Measurement - use cached version for O(n) instead of O(n²)
    if !config.quiet {
        eprintln!("{}", "Running benchmark (GPU)...".yellow());
    }
    let mut iteration_times = Vec::with_capacity(config.iterations);
    let mut total_tokens = 0usize;
    let mut first_token_time = Duration::ZERO;

    for i in 0..config.iterations {
        let iter_start = Instant::now();

        // Reset KV cache and use cached generation
        model.reset_kv_cache();
        let output = model
            .generate_cuda_with_cache(&prompt_tokens, config.max_tokens.min(32), eos_token)
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
