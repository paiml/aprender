
/// Try GGUF GPU generation. Takes model by value to avoid expensive clone (~1GB).
/// Returns `Ok(result)` on GPU success, `Err(model)` to return model for CPU fallback.
#[cfg(feature = "cuda")]
fn try_gguf_gpu_generate(
    model: crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    verbose: bool,
) -> std::result::Result<Result<(Vec<u32>, bool)>, crate::gguf::OwnedQuantizedModel> {
    use crate::gguf::OwnedQuantizedModelCuda;

    let mut cuda_model = match OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048) {
        Ok(m) => m,
        Err(e) => {
            if verbose {
                eprintln!("Backend: CPU (GPU unavailable: {})", e);
            }
            // Model is preserved inside CudaInitError for CPU fallback
            return Err(e.into_model());
        },
    };

    if verbose {
        eprintln!(
            "Backend: GPU ({}, {} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        );
    }

    if !validate_gpu_first_token(&mut cuda_model, gen_config) {
        // Validation failed — extract model back for CPU fallback
        return Err(cuda_model.into_model());
    }

    // Reuse existing CUDA model — generate_gpu_resident() creates fresh KV cache
    // and resets GPU KV positions internally, so validation doesn't "consume" it.
    let result = cuda_model
        .generate_gpu_resident(input_tokens, gen_config)
        .map(|tokens| (tokens, true))
        .map_err(|e| RealizarError::InferenceError(format!("GPU generation failed: {}", e)));
    Ok(result)
}

/// Run GGUF generation with GPU or CPU
#[allow(unused_variables)] // config used only in CUDA feature
fn run_gguf_generate(
    model: crate::gguf::OwnedQuantizedModel,
    input_tokens: &[u32],
    gen_config: &crate::gguf::QuantizedGenerateConfig,
    config: &InferenceConfig,
) -> Result<(Vec<u32>, bool)> {
    let has_legacy_quant = model_has_legacy_quant(&model);

    // GPU path: pass model by value (zero-clone) — model is returned on failure for CPU fallback
    #[cfg(feature = "cuda")]
    let model = if !config.no_gpu && !has_legacy_quant {
        match try_gguf_gpu_generate(model, input_tokens, gen_config, config.verbose) {
            Ok(result) => return result,
            Err(returned_model) => returned_model, // GPU failed, use returned model for CPU
        }
    } else {
        model
    };

    log_cpu_backend(config.verbose, has_legacy_quant);
    let tokens = model
        .generate_with_cache(input_tokens, gen_config)
        .map_err(|e| RealizarError::InferenceError(format!("CPU generation failed: {}", e)))?;
    Ok((tokens, false))
}

/// Run APR model inference (PAR-302, PMAT-APR-CUDA-001)
///
/// Uses AprV2ModelCuda for GPU acceleration when available, falls back to
/// AprTransformer (CPU with proper RoPE and SwiGLU) otherwise.
/// PMAT-237: APR inference now uses PreparedTokens (compile-time enforced chat template).
/// Previously bypassed PreparedTokens entirely via prepare_apr_input_tokens().
fn run_apr_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    if config.verbose {
        eprintln!("Loading APR model: {}", config.model_path.display());
    }

    let load_start = Instant::now();
    let input_tokens = prepared.tokens();
    let input_token_count = prepared.input_count();

    // Try GPU path first
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        if let Some(result) =
            try_apr_cuda_inference(config, input_tokens, input_token_count, load_start)
        {
            return result;
        }
    }

    // CPU fallback: AprTransformer with RoPE and SwiGLU
    run_apr_cpu_inference(config, input_tokens, input_token_count, load_start)
}

/// GH-336: Map APR architecture string to chat template hint.
///
/// Uses contract-driven normalize_architecture() instead of contains() heuristics.
fn apr_arch_to_template_hint(apr_arch: &str, _model_name: &str) -> &'static str {
    crate::tensor_names::normalize_architecture(apr_arch)
}

/// Metadata captured from the model config before it is moved into CUDA.
#[cfg(feature = "cuda")]
struct AprCudaModelInfo {
    arch: String,
    num_layers: usize,
    vocab_size: usize,
    hidden_dim: usize,
}

/// Load an APR model and initialize it on CUDA, returning None on any failure.
#[cfg(feature = "cuda")]
fn load_apr_cuda_model(
    model_path: &std::path::Path,
    verbose: bool,
) -> Option<(crate::gguf::OwnedQuantizedModelCuda, AprCudaModelInfo)> {
    use crate::apr::MappedAprModel;
    use crate::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    let mapped = MappedAprModel::from_path(model_path).map_err(|e| {
        if verbose { eprintln!("[APR-CUDA] MappedAprModel::from_path failed: {}", e); }
    }).ok()?;

    let model = OwnedQuantizedModel::from_apr(&mapped).map_err(|e| {
        if verbose { eprintln!("[APR-CUDA] OwnedQuantizedModel::from_apr failed: {}", e); }
    }).ok()?;

    if model_has_legacy_quant(&model) {
        return None;
    }

    let info = AprCudaModelInfo {
        arch: model.config.architecture.clone(),
        num_layers: model.config.num_layers,
        vocab_size: model.config.vocab_size,
        hidden_dim: model.config.hidden_dim,
    };

    let cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 2048).map_err(|e| {
        if verbose { eprintln!("Backend: CPU (GPU unavailable: {})", e); }
    }).ok()?;

    Some((cuda_model, info))
}

#[cfg(feature = "cuda")]
fn log_apr_cuda_info(
    info: &AprCudaModelInfo,
    cuda_model: &crate::gguf::OwnedQuantizedModelCuda,
    load_ms: f64,
) {
    eprintln!(
        "Architecture: {} ({} layers, vocab_size={})",
        info.arch, info.num_layers, info.vocab_size
    );
    eprintln!(
        "Config: hidden_size={}, quant=CUDA+KVCache, threads=1 (GPU)",
        info.hidden_dim
    );
    eprintln!("Model loaded in {:.1}ms", load_ms);
    eprintln!(
        "Backend: GPU ({}, {} MB VRAM)",
        cuda_model.device_name(),
        cuda_model.vram_mb()
    );
}

/// Try APR CUDA inference, returning None to fall through to CPU.
///
/// Converts APR Q4K model to `OwnedQuantizedModel` and uses the proven GGUF CUDA
/// pipeline (same path as `try_gguf_gpu_generate`). The previous wgpu path used
/// `AprF32ToGpuAdapter` which only reads F32 fields — empty for Q4K models → garbage.
#[cfg(feature = "cuda")]
fn try_apr_cuda_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Option<Result<InferenceResult>> {
    use crate::gguf::QuantizedGenerateConfig;

    let (mut cuda_model, info) = load_apr_cuda_model(&config.model_path, config.verbose)?;

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        log_apr_cuda_info(&info, &cuda_model, load_ms);
    }

    // GH-330: EOS from model config (Design by Contract)
    let model_eos: Vec<u32> = cuda_model.model().config.eos_token_id.into_iter().collect();
    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: 0.0,
        top_k: 1,
        stop_tokens: model_eos,
        trace: false,
            ..Default::default()
    };

    if !validate_gpu_first_token(&mut cuda_model, &gen_config) {
        return None;
    }

    let infer_start = Instant::now();

    let tokens = match cuda_model.generate_gpu_resident(input_tokens, &gen_config) {
        Ok(t) => t,
        Err(e) => {
            let msg = e.to_string();
            // GH-278: Fall back to CPU for unsupported architectures (GPT-2 has no SwiGLU/RMSNorm)
            if msg.contains("not supported") || msg.contains("architecture") {
                if config.verbose {
                    eprintln!("[APR-CUDA] GPU-resident not supported, falling back to CPU: {msg}");
                }
                return None;
            }
            return Some(Err(RealizarError::InferenceError(format!(
                "GPU generation failed: {}",
                e
            ))));
        },
    };

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Some(Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: true,
    }))
}

/// Log APR model info when verbose mode is enabled.
fn log_apr_model_info(
    verbose: bool,
    validated: &crate::apr_transformer::AprTransformer,
    load_ms: f64,
) {
    if verbose {
        let arch = &validated.config.architecture;
        let thread_count = rayon::current_num_threads();
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            arch, validated.config.num_layers, validated.config.vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, context_length={}, quant=F32 (dequantized), threads={}",
            validated.config.hidden_dim, validated.config.context_length, thread_count
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!("Backend: CPU (SIMD-accelerated)");
    }
}

/// Run APR inference on CPU with KV-cache (PMAT-103)
fn run_apr_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Result<InferenceResult> {
    use crate::apr_transformer::AprTransformer;

    // GH-278: AprTransformer only supports LLaMA-style models (RoPE + SwiGLU).
    // For GPT-2 and other architectures, use OwnedQuantizedModel which supports
    // learned position embeddings, LayerNorm, GELU, etc.
    let validated = match AprTransformer::from_apr_file_validated(&config.model_path) {
        Ok(t) => {
            // Check if architecture needs OwnedQuantizedModel (GPT-2, etc.)
            let arch = t.config.architecture.to_lowercase();
            if arch.contains("gpt2") || arch.contains("gpt-2") {
                return run_apr_quantized_cpu_inference(config, input_tokens, input_token_count, load_start);
            }
            t
        },
        Err(_) => {
            return run_apr_quantized_cpu_inference(config, input_tokens, input_token_count, load_start);
        }
    };
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    log_apr_model_info(config.verbose, &validated, load_ms);

    let infer_start = Instant::now();
    let mut all_tokens = input_tokens.to_vec();
    let mut cache = crate::apr_transformer::AprKVCache::new(&validated.config);

    // Prefill: populate KV cache
    for (pos, &token) in input_tokens.iter().enumerate() {
        let _ = validated.forward_with_cache(token, &mut cache, pos)?;
    }

    // Generate with KV cache (O(1) per token)
    let mut position = input_tokens.len();
    for _ in 0..config.max_tokens {
        let last_token = *all_tokens.last().unwrap_or(&1);
        let logits = validated.forward_with_cache(last_token, &mut cache, position)?;

        let next_token = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i as u32);

        // GH-330: EOS from model config (Design by Contract)
        if next_token == 0 || validated.config.eos_token_id == Some(next_token) {
            break;
        }

        all_tokens.push(next_token);
        position += 1;
    }

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &all_tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens: all_tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: false,
    })
}

/// GH-278: CPU inference for APR models using OwnedQuantizedModel
///
/// Used for architectures not supported by AprTransformer (GPT-2, etc.).
/// AprTransformer only supports LLaMA-style (RoPE + SwiGLU).
fn run_apr_quantized_cpu_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
    load_start: Instant,
) -> Result<InferenceResult> {
    use crate::apr::MappedAprModel;
    use crate::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};

    let mapped = MappedAprModel::from_path(&config.model_path)?;
    let model = OwnedQuantizedModel::from_apr(&mapped)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Architecture: {} ({} layers, vocab_size={})",
            model.config.architecture, model.config.num_layers, model.config.vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, quant=Q4_K (OwnedQuantizedModel CPU), threads={}",
            model.config.hidden_dim,
            rayon::current_num_threads()
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!("Backend: CPU (OwnedQuantizedModel fallback for non-LLaMA arch)");
    }

    let gen_config = QuantizedGenerateConfig {
        max_tokens: config.max_tokens,
        temperature: config.temperature,
        top_k: config.top_k,
        trace: config.trace,
        ..Default::default()
    };

    let infer_start = Instant::now();
    let tokens = model.generate_with_cache(input_tokens, &gen_config)?;
    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "APR".to_string(),
        used_gpu: false,
    })
}

/// Decode APR output tokens using available tokenizer (GH-156)
fn decode_apr_tokens(model_path: &std::path::Path, tokens: &[u32]) -> String {
    use crate::apr::AprV2Model;

    let text = if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        tokenizer.decode(tokens)
    } else if let Some(tokenizer) = find_fallback_tokenizer(model_path) {
        tokenizer.decode(tokens)
    } else {
        format!("[{} tokens generated, tokenizer not found]", tokens.len())
    };
    clean_model_output(&text)
}

/// Compute tokens per second from count and elapsed milliseconds
fn tok_per_sec(count: usize, ms: f64) -> f64 {
    if ms > 0.0 {
        count as f64 / (ms / 1000.0)
    } else {
        0.0
    }
}

/// Run SafeTensors model inference (PAR-301, PMAT-129)
///
/// PMAT-236: Accepts `PreparedTokens` (compile-time enforced chat template).
/// Previously, this function raw-encoded prompts WITHOUT chat template,
/// producing garbage output for instruct models.
fn run_safetensors_inference(
    config: &InferenceConfig,
    prepared: &PreparedTokens,
) -> Result<InferenceResult> {
    if config.verbose {
        eprintln!("Loading SafeTensors model: {}", config.model_path.display());
    }

    // PMAT-236: Use PreparedTokens (chat template already applied by prepare_tokens)
    let input_tokens = prepared.tokens().to_vec();
    let input_token_count = prepared.input_count();

    // PMAT-129: Try GPU path first
    #[cfg(feature = "cuda")]
    if !config.no_gpu {
        if let Some(result) =
            try_safetensors_cuda_inference(config, &input_tokens, input_token_count)
        {
            return result;
        }
    }

    // CPU fallback: SafeTensors → AprTransformer conversion
    run_safetensors_cpu_inference(config, &input_tokens, input_token_count)
}

/// Try SafeTensors CUDA inference, returning None to fall through to CPU
#[cfg(feature = "cuda")]
fn try_safetensors_cuda_inference(
    config: &InferenceConfig,
    input_tokens: &[u32],
    input_token_count: usize,
) -> Option<Result<InferenceResult>> {
    use crate::safetensors_cuda::SafeTensorsCudaModel;

    let load_start = Instant::now();
    let mut cuda_model = match SafeTensorsCudaModel::load(&config.model_path, 0) {
        Ok(m) => m,
        Err(e) => {
            if config.verbose {
                eprintln!("Backend: CPU (GPU init failed: {})", e);
            }
            return None;
        },
    };

    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    if config.verbose {
        eprintln!(
            "Architecture: SafeTensors ({} layers, vocab_size={})",
            cuda_model.config().num_layers,
            cuda_model.config().vocab_size
        );
        eprintln!(
            "Config: hidden_size={}, context_length={}, quant=F16/BF16, threads=1 (GPU)",
            cuda_model.config().hidden_dim,
            cuda_model.config().context_length
        );
        eprintln!("Model loaded in {:.1}ms", load_ms);
        eprintln!(
            "Backend: GPU ({}, {} MB VRAM)",
            cuda_model.device_name(),
            cuda_model.vram_mb()
        );
    }

    let infer_start = Instant::now();
    // GH-330: EOS from model config (Design by Contract)
    let eos_id = cuda_model.model().config.eos_token_id.unwrap_or(0);
    let tokens = match cuda_model.generate(input_tokens, config.max_tokens, eos_id) {
        Ok(t) => t,
        Err(e) => {
            return Some(Err(RealizarError::InferenceError(format!(
                "GPU generation failed: {}",
                e
            ))))
        },
    };

    let inference_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens = &tokens[input_token_count..];
    let text = decode_apr_tokens(&config.model_path, generated_tokens);
    let generated_token_count = generated_tokens.len();

    Some(Ok(InferenceResult {
        text,
        tokens,
        input_token_count,
        generated_token_count,
        inference_ms,
        tok_per_sec: tok_per_sec(generated_token_count, inference_ms),
        load_ms,
        format: "SafeTensors".to_string(),
        used_gpu: true,
    }))
}
