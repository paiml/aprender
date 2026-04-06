/// Detect whether an APR file at `model_ref` path contains Q4K/Q6K quantized tensors.
///
/// Returns true if any tensor has dtype Q4_K or Q6_K, meaning the
/// Q4K GPU path (ALB-095) should be used instead of F32 dequant.
#[cfg(feature = "cuda")]
pub fn is_apr_q4k(model_ref: &str) -> bool {
    use crate::apr::AprV2Model;
    use std::path::Path;

    let path = Path::new(model_ref);
    if let Ok(model) = AprV2Model::load(path) {
        return model
            .tensor_names()
            .iter()
            .filter_map(|name| model.get_tensor(name))
            .any(|t| matches!(t.dtype.as_str(), "Q4_K" | "q4_k" | "Q6_K" | "q6_k"));
    }
    false
}

/// Run APR Q4K inference with CUDA GPU acceleration (ALB-095).
///
/// Uploads raw Q4K bytes from APR directly to GPU (17 GB fits 24 GB VRAM).
/// Hybrid CPU/GPU: embedding, RMSNorm, RoPE, attention on CPU;
/// all projection GEMVs on GPU via Q4K kernels.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run_apr_inference_gpu_q4k(
    model_ref: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    verbose: bool,
    trace_config: Option<crate::inference_trace::TraceConfig>,
) -> Result<()> {
    use crate::apr::AprV2Model;
    use crate::cuda::CudaExecutor;
    use crate::gpu::adapters::apr_q4k::{
        forward_token_apr_q4k, parse_apr_q4k_config, upload_apr_q4k_weights,
    };
    use crate::inference_trace::{InferenceTracer, ModelInfo, TraceStep};
    use std::path::Path;
    use std::time::Instant;

    let mut tracer = trace_config.map_or_else(InferenceTracer::disabled, InferenceTracer::new);

    let load_start = Instant::now();

    if verbose {
        println!("Backend: CUDA (GPU Q4K — ALB-095)");
        println!("Loading APR Q4K model...");
    }

    // Load APR model via mmap (zero-copy)
    let model_path = Path::new(model_ref);
    let model = AprV2Model::load(model_path).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "load_apr_q4k".to_string(),
            reason: format!("Failed to load APR model: {e}"),
        }
    })?;

    // Parse config from APR metadata
    let config = parse_apr_q4k_config(&model)?;

    if verbose {
        println!(
            "Model: {} layers, hidden={}, heads={}/{}, vocab={}",
            config.num_layers, config.hidden_dim, config.num_heads, config.num_kv_heads, config.vocab_size
        );
        if let Some(ne) = config.num_experts {
            println!(
                "MoE: {} experts, top-{}, intermediate={}",
                ne,
                config.num_experts_per_tok.unwrap_or(0),
                config.moe_intermediate_size.unwrap_or(0)
            );
        }
    }

    tracer.set_model_info(ModelInfo {
        name: model_ref.to_string(),
        num_layers: config.num_layers,
        hidden_dim: config.hidden_dim,
        vocab_size: config.vocab_size,
        num_heads: config.num_heads,
        quant_type: Some("APR Q4K (GPU)".to_string()),
    });

    // Initialize CUDA executor and upload weights
    let mut executor = CudaExecutor::new(0).map_err(|e| {
        crate::error::RealizarError::GpuError {
            reason: format!("CUDA init failed: {e}"),
        }
    })?;

    let upload_result = upload_apr_q4k_weights(&model, &mut executor)?;

    if verbose {
        println!(
            "Uploaded {} tensors ({} Q4K, {} F32) — {:.1} MB VRAM",
            upload_result.num_tensors,
            upload_result.num_q4k_tensors,
            upload_result.num_f32_tensors,
            upload_result.total_bytes as f64 / (1024.0 * 1024.0)
        );
    }

    // Extract F32 weights needed on CPU: embedding, output_norm, per-layer norms
    // #170: Use find_tensor_name for GGUF/SafeTensors/HF name resolution
    let embed_name = model.find_tensor_name(&[
        "model.embed_tokens.weight",
        "embed_tokens.weight",
        "transformer.wte.weight",
        "tok_embeddings.weight",
        "token_embd.weight",
    ]).map_err(|e| crate::error::RealizarError::FormatError {
        reason: format!("Missing embedding weight: {e}"),
    })?;
    let embedding_weight = model.get_tensor_f32(&embed_name).map_err(|e| {
        crate::error::RealizarError::FormatError {
            reason: format!("Missing embedding weight: {e}"),
        }
    })?;

    let norm_name = model.find_tensor_name(&[
        "model.norm.weight",
        "norm.weight",
        "transformer.ln_f.weight",
        "output_norm.weight",
    ]).map_err(|e| crate::error::RealizarError::FormatError {
        reason: format!("Missing output norm weight: {e}"),
    })?;
    let output_norm_weight = model.get_tensor_f32(&norm_name).map_err(|e| {
        crate::error::RealizarError::FormatError {
            reason: format!("Missing output norm weight: {e}"),
        }
    })?;

    let mut layer_norm_weights: Vec<(Vec<f32>, Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>)> =
        Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        let attn_norm_name = model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.input_layernorm.weight"),
            &format!("layers.{layer_idx}.input_layernorm.weight"),
            &format!("blk.{layer_idx}.attn_norm.weight"),
        ]).map_err(|e| crate::error::RealizarError::FormatError {
            reason: format!("Missing attn norm weight layer {layer_idx}: {e}"),
        })?;
        let attn_norm = model.get_tensor_f32(&attn_norm_name).map_err(|e| {
            crate::error::RealizarError::FormatError {
                reason: format!("Missing attn norm weight layer {layer_idx}: {e}"),
            }
        })?;
        let ffn_norm_name = model.find_tensor_name(&[
            &format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
            &format!("layers.{layer_idx}.post_attention_layernorm.weight"),
            &format!("blk.{layer_idx}.ffn_norm.weight"),
        ]).map_err(|e| crate::error::RealizarError::FormatError {
            reason: format!("Missing FFN norm weight layer {layer_idx}: {e}"),
        })?;
        let ffn_norm = model.get_tensor_f32(&ffn_norm_name).map_err(|e| {
            crate::error::RealizarError::FormatError {
                reason: format!("Missing FFN norm weight layer {layer_idx}: {e}"),
            }
        })?;
        // QK-norm weights (Qwen3-style, optional)
        let q_norm = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.q_norm.weight"))
            .ok();
        let k_norm = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.k_norm.weight"))
            .ok();
        layer_norm_weights.push((attn_norm, ffn_norm, q_norm, k_norm));
    }

    // PMAT-315: Extract QKV biases (required for Qwen2, Phi; no-op for LLaMA/Mistral)
    let mut layer_qkv_biases: Vec<(Option<Vec<f32>>, Option<Vec<f32>>, Option<Vec<f32>>)> =
        Vec::with_capacity(config.num_layers);
    for layer_idx in 0..config.num_layers {
        let q_bias = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.q_proj.bias"))
            .ok();
        let k_bias = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.k_proj.bias"))
            .ok();
        let v_bias = model
            .get_tensor_f32(&format!("model.layers.{layer_idx}.self_attn.v_proj.bias"))
            .ok();
        layer_qkv_biases.push((q_bias, k_bias, v_bias));
    }

    // Release mmap pages — weights are on GPU now (advisory, non-fatal)
    let _ = model.release_cpu_pages();

    let load_time = load_start.elapsed();
    if verbose {
        println!("Model loaded in {:.2}s", load_time.as_secs_f64());
    }

    // Load tokenizer for encoding AND decoding
    let tokenizer = AprV2Model::load_tokenizer(model_path);

    // Tokenize — prefer BpeTokenizer over encode_text (APR Q4K files lack embedded tokenizer)
    tracer.start_step(TraceStep::Tokenize);
    let prompt_tokens = if let Some(ref tok) = tokenizer {
        tok.encode(prompt)
    } else {
        AprV2Model::encode_text(model_path, prompt)
            .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect())
    };
    let prompt_len = prompt_tokens.len();
    tracer.trace_encode(prompt, &prompt_tokens, config.vocab_size);

    if verbose {
        println!("Prompt tokens: {}", prompt_len);
        println!("Temperature: {:.1}", temperature);
        println!();
    }

    tracer.start_step(TraceStep::Embed);
    tracer.trace_embed(prompt_len, config.hidden_dim, None);
    tracer.start_step(TraceStep::TransformerBlock);

    // Initialize KV cache
    let mut kv_cache_k: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers];
    let mut kv_cache_v: Vec<Vec<f32>> = vec![Vec::new(); config.num_layers];

    // Generate tokens
    let gen_start = Instant::now();
    let mut output_tokens: Vec<u32> = Vec::new();

    // Process prompt tokens (prefill) — discard logits except last
    let mut last_logits = Vec::new();
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        last_logits = forward_token_apr_q4k(
            &mut executor,
            &config,
            &embedding_weight,
            &output_norm_weight,
            &layer_norm_weights,
            &layer_qkv_biases,
            &mut kv_cache_k,
            &mut kv_cache_v,
            token_id,
            pos,
        )?;
    }

    // Sample first token from prefill logits
    let mut next_token = if temperature <= 0.01 {
        argmax(&last_logits)
    } else {
        sample_with_temperature(&last_logits, temperature, 40)
    };
    output_tokens.push(next_token);

    if verbose {
        if let Some(ref tok) = tokenizer {
            let text = tok.decode(&[next_token]);
            print!("{text}");
            use std::io::Write;
            let _ = std::io::stdout().flush();
        }
    }

    // Autoregressive generation
    for step in 0..max_tokens.saturating_sub(1) {
        // EOS check
        if next_token == 0 || next_token == 2 {
            break;
        }

        let position = prompt_len + step;
        let logits = forward_token_apr_q4k(
            &mut executor,
            &config,
            &embedding_weight,
            &output_norm_weight,
            &layer_norm_weights,
            &layer_qkv_biases,
            &mut kv_cache_k,
            &mut kv_cache_v,
            next_token,
            position,
        )?;

        next_token = if temperature <= 0.01 {
            argmax(&logits)
        } else {
            sample_with_temperature(&logits, temperature, 40)
        };

        output_tokens.push(next_token);

        // Streaming decode
        if verbose {
            if let Some(ref tok) = tokenizer {
                let text = tok.decode(&[next_token]);
                print!("{text}");
                use std::io::Write;
                let _ = std::io::stdout().flush();
            }
        }
    }

    let gen_time = gen_start.elapsed();
    let tokens_generated = output_tokens.len();
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    tracer.trace_layer(config.num_layers - 1, 0, None, 1, config.hidden_dim);

    // Decode full output
    let generated_text = if let Some(ref tok) = tokenizer {
        tok.decode(&output_tokens)
    } else {
        decode_apr_output_tokens(model_path, &output_tokens)
    };

    // Output
    if verbose {
        println!();
    }

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "format": "APR Q4K",
                "backend": "CUDA (Q4K GEMV)",
                "prompt": prompt,
                "generated_text": generated_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time.as_secs_f64() * 1000.0,
                "tokens_per_second": tokens_per_sec,
                "temperature": temperature,
                "vram_mb": upload_result.total_bytes as f64 / (1024.0 * 1024.0),
            });
            println!(
                "{}",
                serde_json::to_string_pretty(&json).unwrap_or_default()
            );
        },
        _ => {
            if verbose {
                println!(
                    "Generated ({tokens_generated} tokens in {:.2}ms):",
                    gen_time.as_secs_f64() * 1000.0
                );
                println!("Performance: {:.1} tok/s (GPU Q4K)", tokens_per_sec);
            } else {
                println!("{generated_text}");
            }
        },
    }

    if tracer.is_enabled() {
        if let Err(e) = tracer.write_output() {
            eprintln!("[TRACE] Warning: Failed to write trace output: {}", e);
        }
    }

    Ok(())
}

/// Greedy argmax over logits.
#[cfg(feature = "cuda")]
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(idx, _)| idx as u32)
}

/// Sample a token with temperature and top-k.
#[cfg(feature = "cuda")]
pub(crate) fn sample_with_temperature(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    // Apply temperature
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    // Top-k filtering
    let mut indexed: Vec<(usize, f32)> = scaled.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top = &indexed[..top_k.min(indexed.len())];

    // Softmax
    let max_val = top[0].1;
    let exp_vals: Vec<(usize, f32)> = top.iter().map(|&(i, v)| (i, (v - max_val).exp())).collect();
    let sum: f32 = exp_vals.iter().map(|(_, v)| v).sum();

    // Random sample
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;
    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    let r = (hasher.finish() as f32 / u64::MAX as f32) * sum;

    let mut cumsum = 0.0f32;
    for &(idx, val) in &exp_vals {
        cumsum += val;
        if cumsum >= r {
            return idx as u32;
        }
    }
    exp_vals.last().map_or(0, |&(i, _)| i as u32)
}

/// Run APR inference with CUDA GPU acceleration (PMAT-106)
///
/// Uses the APR F32 GPU adapter to convert weights to GpuModel format
/// for high-performance inference.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn run_apr_inference_gpu(
    model_ref: &str,
    file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    verbose: bool,
    trace_config: Option<crate::inference_trace::TraceConfig>,
) -> Result<()> {
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprTransformer;
    use crate::gpu::adapters::AprF32ToGpuAdapter;
    use crate::gpu::GpuGenerateConfig;
    use crate::inference_trace::{InferenceTracer, ModelInfo, TraceStep};
    use std::path::Path;
    use std::time::Instant;

    // APR-TRACE-001: Create tracer
    let mut tracer = trace_config.map_or_else(InferenceTracer::disabled, InferenceTracer::new);

    let load_start = Instant::now();

    if verbose {
        println!("Backend: CUDA (GPU)");
        println!("Loading APR model for GPU inference...");
    }

    // Load APR as F32 transformer
    let transformer = AprTransformer::from_apr_bytes(file_data).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "parse_apr".to_string(),
            reason: format!("Failed to parse APR: {e}"),
        }
    })?;

    // APR-TRACE-001: Set model info
    tracer.set_model_info(ModelInfo {
        name: model_ref.to_string(),
        num_layers: transformer.config.num_layers,
        hidden_dim: transformer.config.hidden_dim,
        vocab_size: transformer.config.vocab_size,
        num_heads: transformer.config.num_heads,
        quant_type: Some("APR F32 (GPU)".to_string()),
    });

    // Convert to GpuModel using F32 adapter
    let mut gpu_model = AprF32ToGpuAdapter::to_gpu_model(&transformer).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "apr_to_gpu".to_string(),
            reason: format!("Failed to convert APR to GPU format: {e}"),
        }
    })?;

    // Debug: Compare CPU vs GPU weights and matmul outputs
    if verbose {
        print_gpu_debug_weights(&transformer, &gpu_model);
    }

    let load_time = load_start.elapsed();
    if verbose {
        println!("Model loaded in {:.2}ms", load_time.as_secs_f64() * 1000.0);
    }

    // NOTE: Chat template is applied by the caller (mod.rs) before calling this function.
    // The `prompt` parameter already contains the formatted conversation with chat markers.

    // APR-TRACE-001: Trace tokenization
    tracer.start_step(TraceStep::Tokenize);

    // Use proper tokenizer from sibling tokenizer.json or embedded vocab
    let model_path = Path::new(model_ref);
    let prompt_tokens = AprV2Model::encode_text(model_path, prompt)
        .or_else(|| {
            // ALB-107: entrenar checkpoints lack embedded tokenizer.
            // Fall back to sibling tokenizer.json (same as decode path).
            AprV2Model::load_tokenizer(model_path).map(|tok| tok.encode(prompt))
        })
        .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());
    let prompt_len = prompt_tokens.len();

    tracer.trace_encode(prompt, &prompt_tokens, transformer.config.vocab_size);

    if verbose {
        println!("Prompt tokens: {}", prompt_len);
        // F-REGR-231 DEBUG: Show all token IDs
        eprintln!("[DEBUG-TOKENS] All tokens: {:?}", &prompt_tokens);
        println!("Temperature: {:.1}", temperature);
        println!();
    }

    // APR-TRACE-001: Trace embedding
    tracer.start_step(TraceStep::Embed);
    tracer.trace_embed(prompt_len, transformer.config.hidden_dim, None);

    // APR-TRACE-001: Trace transformer blocks (high-level, GPU generation is a black box)
    tracer.start_step(TraceStep::TransformerBlock);

    // Configure generation
    let gen_config = GpuGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature <= 0.01 { 1 } else { 40 },
        stop_tokens: vec![],
        trace: false,
    };

    // Convert prompt tokens to usize for GpuModel
    let prompt_tokens_usize: Vec<usize> = prompt_tokens.iter().map(|&t| t as usize).collect();

    // Run inference
    let gen_start = Instant::now();
    let generated = gpu_model
        .generate(&prompt_tokens_usize, &gen_config)
        .map_err(|e| crate::error::RealizarError::UnsupportedOperation {
            operation: "gpu_generate".to_string(),
            reason: format!("GPU generation failed: {e}"),
        })?;
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

    // Decode output (convert usize back to u32)
    let output_tokens: Vec<u32> = generated[prompt_len..].iter().map(|&t| t as u32).collect();
    let output_text = decode_apr_output_tokens(model_path, &output_tokens);

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
                "format": "APR",
                "backend": "CUDA",
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
            if verbose {
                println!(
                    "Generated ({tokens_generated} tokens in {:.2}ms):",
                    gen_time.as_secs_f64() * 1000.0
                );
                println!("{output_text}");
                println!();
                println!("Performance: {:.1} tok/s (GPU)", tokens_per_sec);
            } else {
                println!("{output_text}");
            }
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
