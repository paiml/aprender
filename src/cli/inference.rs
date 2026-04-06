//! Inference runner functions for CLI commands
//!
//! Contains run_gguf_inference, run_gguf_inference_gpu, run_safetensors_inference,
//! and run_apr_inference - extracted from main.rs (PMAT-802).

#![allow(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_pass_by_value)]

use crate::error::Result;

/// Sample the next token from logits using temperature scaling
fn sample_next_token(logits: &[f32], temperature: f32) -> u32 {
    use crate::gguf::OwnedQuantizedModel;
    if temperature <= 0.01 {
        // Greedy decoding
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx as u32)
    } else {
        // Temperature sampling with top-k=40
        OwnedQuantizedModel::sample_topk(logits, temperature, 40)
    }
}

/// Print inference results in the requested format
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
fn print_inference_output(
    model_ref: &str,
    prompt: &str,
    output_text: &str,
    tokens_generated: usize,
    gen_time_ms: f64,
    tokens_per_sec: f64,
    temperature: f32,
    format: &str,
    verbose: bool,
) {
    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "prompt": prompt,
                "generated_text": output_text,
                "tokens_generated": tokens_generated,
                "generation_time_ms": gen_time_ms,
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
                println!("Generated ({tokens_generated} tokens in {gen_time_ms:.2}ms):");
                println!("{prompt}{output_text}");
                println!();
                println!("Performance: {tokens_per_sec:.1} tok/s");
            } else {
                // Ollama-style clean output: just the response
                println!("{output_text}");
            }
        },
    }
}

/// Print model architecture info when verbose mode is enabled
/// Decode tokens one at a time with optional tracing
fn decode_tokens_with_cache(
    model: &crate::gguf::OwnedQuantizedModel,
    cache: &mut crate::gguf::OwnedQuantizedKVCache,
    initial_logits: Vec<f32>,
    all_tokens: &mut Vec<u32>,
    prompt_len: usize,
    max_tokens: usize,
    temperature: f32,
    eos_token_id: Option<u32>,
    tracer: &mut crate::inference_trace::InferenceTracer,
    _num_layers: usize,
    vocab: Option<&[String]>,
) -> Result<()> {
    use crate::inference_trace::TraceStep;

    let mut logits = initial_logits;
    let config = model.config();

    for i in 0..max_tokens {
        if i > 0 {
            let position = prompt_len + i - 1;
            let last_token = *all_tokens.last().expect("all_tokens should not be empty");

            // Trace transformer block
            tracer.start_step(TraceStep::TransformerBlock);
            logits = model.forward_cached(last_token, cache, position)?;

            // Trace layer stats (use last position's hidden state approximation from logits)
            if tracer.is_enabled() {
                tracer.trace_layer(
                    config.num_layers - 1, // Report final layer
                    i,
                    Some(&logits[..config.hidden_dim.min(logits.len())]),
                    1,
                    config.hidden_dim,
                );
            }
        }

        // Trace LM head
        tracer.start_step(TraceStep::LmHead);
        tracer.trace_lm_head(i, &logits, config.vocab_size);

        // Trace sampling
        tracer.start_step(TraceStep::Sample);
        let next_token = sample_next_token(&logits, temperature);
        tracer.trace_sample(i, &logits, next_token, temperature, 40);

        // Trace decode
        tracer.start_step(TraceStep::Decode);
        let decoded_text = vocab
            .and_then(|v| v.get(next_token as usize))
            .cloned()
            .unwrap_or_else(|| format!("<{}>", next_token));
        tracer.trace_decode(i, next_token, &decoded_text, config.vocab_size);

        if eos_token_id.is_some_and(|eos| next_token == eos) {
            break;
        }

        all_tokens.push(next_token);
    }
    Ok(())
}

/// Print GPU model info when verbose mode is enabled
#[allow(dead_code)]
fn print_gpu_model_info(
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    prompt_len: usize,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    temperature: f32,
) {
    println!("Vocab size: {vocab_size}, Hidden dim: {hidden_dim}, Layers: {num_layers}");
    println!("Prompt tokens: {prompt_len} (BOS={bos_token_id:?}, EOS={eos_token_id:?})");
    println!("Temperature: {temperature:.1}");
    println!();
}

fn print_model_info(
    mapped: &crate::gguf::MappedGGUFModel,
    config: &crate::gguf::GGUFConfig,
    prompt_len: usize,
    temperature: f32,
) {
    println!(
        "Architecture: {:?}, Hidden: {}, Layers: {}, Heads: {}/{} (KV)",
        mapped.model.architecture(),
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.num_kv_heads
    );
    println!(
        "Prompt tokens: {} (BOS={:?}, EOS={:?})",
        prompt_len,
        mapped.model.bos_token_id(),
        mapped.model.eos_token_id()
    );
    println!("Temperature: {temperature:.1}");
    println!();
}

/// PAR-051: Print diagnostic info for CPU debug mode
#[allow(clippy::never_loop)]
fn print_cpu_debug_info(
    logits: &[f32],
    prompt_tokens: &[u32],
    model: &crate::gguf::OwnedQuantizedModel,
    mapped: &crate::gguf::MappedGGUFModel,
) {
    let mut top5: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    top5.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top5.truncate(5);
    eprintln!("[PAR-051] Prompt tokens: {:?}", prompt_tokens);
    eprintln!("[PAR-051] Logits top5 after prefill: {:?}", top5);
    let greedy_token = top5[0].0 as u32;
    let decoded = mapped.model.decode(&[greedy_token]);
    eprintln!(
        "[PAR-051] Greedy next token: {} = {:?}",
        greedy_token, decoded
    );

    let last_token = prompt_tokens[prompt_tokens.len() - 1];
    let embed = model.embed(&[last_token]);
    eprintln!(
        "[PAR-051] Last token {} embed[0..5]: {:?}",
        last_token,
        &embed[..5.min(embed.len())]
    );
    eprintln!("[PAR-051] embed sum: {:.6}", embed.iter().sum::<f32>());

    let cfg = model.config();
    eprintln!(
        "[PAR-051] Config: hidden={}, heads={}/{}, layers={}, vocab={}, eps={:e}",
        cfg.hidden_dim, cfg.num_heads, cfg.num_kv_heads, cfg.num_layers, cfg.vocab_size, cfg.eps
    );

    let logit_2 = logits.get(29906).copied().unwrap_or(f32::NAN);
    eprintln!("[PAR-051] Logit for token 29906 ('2'): {:.6}", logit_2);
}

pub fn run_gguf_inference(
    model_ref: &str,
    _file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    force_gpu: bool,
    verbose: bool,
    trace_config: Option<crate::inference_trace::TraceConfig>,
) -> Result<()> {
    use crate::gguf::{MappedGGUFModel, OwnedQuantizedKVCache};
    use crate::inference_trace::{InferenceTracer, ModelInfo, TraceStep};
    use std::time::Instant;

    // Create tracer from config (APR-TRACE-001)
    let mut tracer = trace_config.map_or_else(InferenceTracer::disabled, InferenceTracer::new);

    // Handle --gpu flag warning when CUDA not available
    #[cfg(not(feature = "cuda"))]
    if force_gpu {
        eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
        eprintln!("Build with: cargo build --features cuda");
        eprintln!();
    }
    // Suppress unused warning when cuda feature not enabled
    #[cfg(not(feature = "cuda"))]
    let _ = force_gpu;

    let load_start = Instant::now();

    // Load model using memory-mapped file (same path as working examples)
    let mapped = MappedGGUFModel::from_path(model_ref).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "mmap_gguf".to_string(),
            reason: format!("Failed to mmap GGUF: {e}"),
        }
    })?;

    // GPU path: Use OwnedQuantizedModel with CUDA acceleration
    #[cfg(feature = "cuda")]
    if force_gpu {
        return run_gguf_inference_gpu(
            &mapped,
            prompt,
            max_tokens,
            temperature,
            format,
            load_start,
            verbose,
        );
    }

    // PAR-126: Five-Whys fix - use OwnedQuantizedModel for fast CPU inference
    // Root cause analysis:
    //   Why-1: CPU path was 14 tok/s vs Ollama's 200 tok/s
    //   Why-2: Old mmap-based transformers use per-matmul allocations
    //   Why-3: Each of 196 matmuls per token allocates/frees Vec
    //   Why-4: Vec allocation overhead + cache pollution from mmap page faults
    //   Why-5: OwnedQuantizedModel copies weights to RAM but uses _into methods
    // Solution: Use OwnedQuantizedModel - slower loading but faster inference
    let model = crate::gguf::OwnedQuantizedModel::from_mapped(&mapped).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "load_model".to_string(),
            reason: format!("Failed to load model: {e}"),
        }
    })?;

    let load_time = load_start.elapsed();
    if verbose {
        println!("Backend: CPU (AVX2 + SIMD)");
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

    // Debug: show model info and encoded tokens
    let config = model.config();
    if verbose {
        print_model_info(&mapped, config, prompt_len, temperature);
    }

    // APR-TRACE-001: Set model info for tracer
    tracer.set_model_info(ModelInfo {
        name: model_ref.to_string(),
        num_layers: config.num_layers,
        hidden_dim: config.hidden_dim,
        vocab_size: config.vocab_size,
        num_heads: config.num_heads,
        quant_type: Some("GGUF".to_string()),
    });

    // Get vocabulary for decode tracing
    let vocab = mapped.model.vocabulary();
    let vocab_ref = vocab.as_deref();

    // Run inference with KV cache for O(n) per-token cost
    let gen_start = Instant::now();
    let max_seq_len = prompt_tokens.len() + max_tokens;
    let mut cache = OwnedQuantizedKVCache::from_config(config, max_seq_len);
    let mut all_tokens = prompt_tokens.clone();

    // APR-TRACE-001: Trace tokenization
    tracer.start_step(TraceStep::Tokenize);
    tracer.trace_encode(prompt, &prompt_tokens, config.vocab_size);

    // Prefill: process prompt tokens to populate KV cache
    // We process all tokens but keep the logits from the last one
    tracer.start_step(TraceStep::Embed);
    let mut logits: Vec<f32> = vec![];
    for (pos, &token_id) in prompt_tokens.iter().enumerate() {
        logits = model.forward_cached(token_id, &mut cache, pos)?;
    }
    // Trace embed with first hidden state approximation
    tracer.trace_embed(
        prompt_tokens.len(),
        config.hidden_dim,
        Some(&logits[..config.hidden_dim.min(logits.len())]),
    );

    // PAR-051: Diagnostic output (gated by CPU_DEBUG=1 environment variable)
    if std::env::var("CPU_DEBUG").is_ok_and(|v| v == "1") {
        print_cpu_debug_info(&logits, &prompt_tokens, &model, &mapped);
    }

    // Decode: generate new tokens one at a time
    decode_tokens_with_cache(
        &model,
        &mut cache,
        logits,
        &mut all_tokens,
        prompt_tokens.len(),
        max_tokens,
        temperature,
        eos_token_id,
        &mut tracer,
        config.num_layers,
        vocab_ref,
    )?;

    let generated = all_tokens;
    let gen_time = gen_start.elapsed();

    let tokens_generated = generated.len() - prompt_len;
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
        model_ref,
        prompt,
        &output_text,
        tokens_generated,
        gen_time.as_secs_f64() * 1000.0,
        tokens_per_sec,
        temperature,
        format,
        verbose,
    );

    // APR-TRACE-001: Write trace output if enabled
    if tracer.is_enabled() {
        if let Err(e) = tracer.write_output() {
            eprintln!("[TRACE] Warning: Failed to write trace output: {}", e);
        }
    }

    Ok(())
}

include!("gguf_inference_gpu.rs");
include!("apr_inference.rs");
include!("apr_gpu_inference.rs");
