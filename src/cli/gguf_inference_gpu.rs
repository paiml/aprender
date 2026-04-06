
/// Run GGUF inference with CUDA GPU acceleration
///
/// Uses OwnedQuantizedModel with CUDA backend for high-performance inference.
/// Called when --gpu flag is specified and CUDA feature is enabled.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run_gguf_inference_gpu(
    mapped: &crate::gguf::MappedGGUFModel,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    load_start: std::time::Instant,
    verbose: bool,
) -> Result<()> {
    use crate::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig};
    use std::time::Instant;

    if verbose {
        println!("Backend: CUDA (GPU)");
        println!("Creating quantized model with CUDA acceleration...");
    }

    // Create owned quantized model (required for CUDA - can't use borrowed mmap data)
    let quantized_model = OwnedQuantizedModel::from_mapped(mapped).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "create_quantized".to_string(),
            reason: format!("Failed to create quantized model: {e}"),
        }
    })?;

    // Get config info before wrapping
    let vocab_size = quantized_model.config.vocab_size;
    let hidden_dim = quantized_model.config.hidden_dim;
    let num_layers = quantized_model.layers.len();

    // PAR-046: Create OwnedQuantizedModelCuda wrapper for actual GPU acceleration
    // The previous implementation used OwnedQuantizedModel.enable_cuda() which only
    // initialized the executor but forward_cached still used CPU code paths.
    let max_seq_len = 256 + max_tokens; // Allow for prompt + generation
    let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(quantized_model, 0, max_seq_len)
        .map_err(|e| e.error)?;
    if verbose {
        println!("  CUDA enabled on GPU: {}", cuda_model.device_name());
    }

    let load_time = load_start.elapsed();
    if verbose {
        println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    }

    // Tokenize prompt using GGUF vocabulary
    let mut prompt_tokens: Vec<u32> = mapped
        .model
        .encode(prompt)
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());

    // Prepend BOS token if available
    if let Some(bos) = mapped.model.bos_token_id() {
        prompt_tokens.insert(0, bos);
    }
    let prompt_len = prompt_tokens.len();

    // Get EOS token for stopping
    let eos_token_id = mapped.model.eos_token_id();

    if verbose {
        print_gpu_model_info(
            vocab_size,
            hidden_dim,
            num_layers,
            prompt_len,
            mapped.model.bos_token_id(),
            eos_token_id,
            temperature,
        );
    }

    // PAR-046: Use CUDA-accelerated generation with GPU-resident KV cache
    // This calls generate_cuda_with_cache -> forward_single_cuda_with_cache -> GPU kernels
    let gen_start = Instant::now();

    // Build stop tokens list
    let mut stop_tokens = Vec::new();
    if let Some(eos) = eos_token_id {
        stop_tokens.push(eos);
    }

    // Configure CUDA generation
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature <= 0.01 { 1 } else { 40 },
        stop_tokens,
        trace: false,
            ..Default::default()
    };

    // PAR-047: Use generate_full_cuda_with_cache for maximum GPU acceleration
    // This path uses:
    // - GPU matmul for QKV, output projection, and FFN
    // - GPU incremental_attention_gpu with GQA support (PAR-021)
    // - Proper SwiGLU activation (PAR-015)
    // PAR-057: Use GPU-resident path for maximum performance (pre-uploads weights, minimal syncs)
    // Falls back to generate_full_cuda_with_cache if architecture not supported
    // PAR-058: Test GPU-resident vs standard CUDA path
    // PHASE-13: Skip GPU-resident if SKIP_GPU_RESIDENT=1 (for debugging)
    let skip_gpu_resident = std::env::var("SKIP_GPU_RESIDENT")
        .map(|v| v == "1")
        .unwrap_or(false);
    let generated = if cuda_model.supports_gpu_resident() && !skip_gpu_resident {
        if verbose {
            println!("Using GPU-resident path (pre-uploaded weights, ~2 syncs/token)");
        }
        cuda_model
            .generate_gpu_resident(&prompt_tokens, &gen_config)
            .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
                operation: "generate_gpu_resident".to_string(),
                reason: format!("GPU-resident generation failed: {e}"),
            })?
    } else {
        if verbose {
            println!("Using standard CUDA path");
        }
        cuda_model
            .generate_full_cuda_with_cache(&prompt_tokens, &gen_config)
            .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
                operation: "generate_full_cuda_with_cache".to_string(),
                reason: format!("CUDA generation failed: {e}"),
            })?
    };
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using GGUF vocabulary, replacing SentencePiece markers with spaces
    let output_text = mapped
        .model
        .decode(&generated[prompt_len..])
        .replace('▁', " ");

    print_inference_output(
        "GGUF (CUDA)",
        prompt,
        &output_text,
        tokens_generated,
        gen_time.as_secs_f64() * 1000.0,
        tokens_per_sec,
        temperature,
        format,
        verbose,
    );

    Ok(())
}

/// Run SafeTensors inference with performance timing
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
pub fn run_safetensors_inference(
    model_ref: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    trace_config: Option<crate::inference_trace::TraceConfig>,
) -> Result<()> {
    use crate::apr::AprV2Model;
    use crate::inference_trace::{InferenceTracer, ModelInfo, TraceStep};
    use crate::safetensors_infer::SafetensorsToAprConverter;
    use std::path::Path;
    use std::time::Instant;

    // APR-TRACE-001: Create tracer from config
    let mut tracer = trace_config.map_or_else(InferenceTracer::disabled, InferenceTracer::new);

    let load_start = Instant::now();
    let model_path = Path::new(model_ref);

    // Convert SafeTensors to AprTransformer (F32 weights)
    let transformer = SafetensorsToAprConverter::convert(model_path).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "convert_safetensors".to_string(),
            reason: format!("Failed to convert SafeTensors: {e}"),
        }
    })?;

    // APR-TRACE-001: Set model info
    tracer.set_model_info(ModelInfo {
        name: model_ref.to_string(),
        num_layers: transformer.config.num_layers,
        hidden_dim: transformer.config.hidden_dim,
        vocab_size: transformer.config.vocab_size,
        num_heads: transformer.config.num_heads,
        quant_type: Some("SafeTensors F32".to_string()),
    });

    let load_time = load_start.elapsed();
    println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    println!(
        "Architecture: {} ({} layers, vocab_size={})",
        transformer.config.architecture,
        transformer.config.num_layers,
        transformer.config.vocab_size
    );

    // APR-TRACE-001: Trace tokenization
    tracer.start_step(TraceStep::Tokenize);

    // Use proper tokenizer from sibling tokenizer.json
    let prompt_tokens = AprV2Model::encode_text(model_path, prompt).unwrap_or_else(|| {
        // Fallback: simple char tokenization
        prompt.chars().map(|c| c as u32).collect()
    });
    let prompt_len = prompt_tokens.len();

    tracer.trace_encode(prompt, &prompt_tokens, transformer.config.vocab_size);

    println!("Prompt tokens: {}", prompt_len);
    println!("Temperature: {:.1}", temperature);
    println!();

    // APR-TRACE-001: Trace embedding
    tracer.start_step(TraceStep::Embed);
    tracer.trace_embed(prompt_len, transformer.config.hidden_dim, None);

    // APR-TRACE-001: Trace transformer blocks (high-level, generation is a black box)
    tracer.start_step(TraceStep::TransformerBlock);

    // Run inference
    let gen_start = Instant::now();
    let generated = transformer.generate(&prompt_tokens, max_tokens)?;
    let gen_time = gen_start.elapsed();

    // Record transformer block completion (aggregate timing)
    tracer.trace_layer(
        transformer.config.num_layers - 1,
        0,
        None,
        1,
        transformer.config.hidden_dim,
    );

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using proper tokenizer
    let output_tokens = &generated[prompt_len..];
    let output_text = if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        tokenizer.decode(output_tokens)
    } else {
        output_tokens
            .iter()
            .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
            .collect()
    };

    // APR-TRACE-001: Trace decode for each output token
    for (i, &token) in output_tokens.iter().enumerate() {
        tracer.start_step(TraceStep::Decode);
        let decoded = output_text
            .chars()
            .nth(i.min(output_text.len().saturating_sub(1)))
            .map_or_else(|| format!("<{token}>"), |c| c.to_string());
        tracer.trace_decode(i, token, &decoded, transformer.config.vocab_size);
    }

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "format": "SafeTensors",
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
                "temperature": temperature,
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        },
        _ => {
            println!(
                "Generated ({tokens_generated} tokens in {:.2}ms):",
                gen_time.as_secs_f64() * 1000.0
            );
            println!("{output_text}");
            println!();
            println!("Performance: {:.1} tok/s", tokens_per_sec);
        },
    }

    // APR-TRACE-001: Write trace output if enabled
    if tracer.is_enabled() {
        if let Err(e) = tracer.write_output() {
            eprintln!("[TRACE] Warning: Failed to write trace output: {}", e);
        }
    }

    Ok(())
}
