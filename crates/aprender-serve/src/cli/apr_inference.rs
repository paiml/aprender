
/// Run APR inference with performance timing
///
/// Supports both CPU and GPU backends (PMAT-106).
// serde_json::json!() uses infallible unwrap
#[allow(clippy::disallowed_methods)]
pub fn run_apr_inference(
    model_ref: &str,
    file_data: &[u8],
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    format: &str,
    force_gpu: bool,
    verbose: bool,
    trace_config: Option<crate::inference_trace::TraceConfig>,
) -> Result<()> {
    use crate::apr::{AprV2Model, MappedAprModel};
    use crate::gguf::{OwnedQuantizedModel, QuantizedGenerateConfig};
    use crate::inference_trace::{InferenceTracer, ModelInfo, TraceStep};
    use std::path::Path;
    use std::time::Instant;

    // GH-479: CPU path now loads via MappedAprModel + OwnedQuantizedModel (per-tensor
    // scratch dequant from GH-478) instead of eager F32 AprTransformer. file_data is
    // unused on both paths but kept in the signature for caller compatibility.
    let _ = file_data;

    // APR-TRACE-001: Create tracer from config
    let mut tracer = trace_config
        .clone()
        .map_or_else(InferenceTracer::disabled, InferenceTracer::new);

    // Handle --gpu flag warning when CUDA not available
    #[cfg(not(feature = "cuda"))]
    if force_gpu {
        eprintln!("Warning: --gpu flag requires 'cuda' feature. Falling back to CPU.");
        eprintln!("Build with: cargo build --features cuda");
        eprintln!();
    }
    #[cfg(not(feature = "cuda"))]
    let _ = (force_gpu, verbose);

    // #170: GPU path for APR models
    // Use OwnedQuantizedModel::from_apr → OwnedQuantizedModelCuda (same path as GGUF).
    // Both AprF32ToGpuAdapter and forward_token_apr_q4k produce garbage on GPU.
    #[cfg(feature = "cuda")]
    if force_gpu {
        use crate::gguf::OwnedQuantizedModelCuda;

        let model_path = Path::new(model_ref);

        // Load APR via MappedAprModel for weight data
        let mapped = MappedAprModel::from_path(model_path).map_err(|e| {
            crate::error::RealizarError::FormatError { reason: format!("APR load failed: {e}") }
        })?;
        let model = OwnedQuantizedModel::from_apr(&mapped).map_err(|e| {
            crate::error::RealizarError::FormatError { reason: format!("APR→GGUF failed: {e}") }
        })?;

        // PMAT-785: fail-closed quant gate before GPU-resident construction. An APR
        // model carrying a quant type without a verified GPU GEMV kernel would be
        // silently decoded as Q4_K on the GPU → garbage. Fall through to the CPU
        // path below (loud) instead of shipping GPU garbage from `apr run --gpu`.
        if model.has_gpu_unsupported_quant() {
            eprintln!(
                "Backend: CPU — APR model uses a quant type without a verified GPU \
                 kernel (PMAT-785). Running on CPU to avoid silent Q4_K-decode garbage."
            );
        } else {
            let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 4096)
                .map_err(|e| e.error)?;

            // Tokenize using embedded APR tokenizer (same as CPU path)
            let input_ids = crate::apr::AprV2Model::encode_text(model_path, prompt)
                .or_else(|| {
                    crate::apr::AprV2Model::load_tokenizer(model_path).map(|t| t.encode(prompt))
                })
                .unwrap_or_else(|| prompt.chars().map(|c| c as u32).collect());

            let gen_config = QuantizedGenerateConfig {
                max_tokens,
                temperature,
                top_k: 1,
                stop_tokens: vec![151645, 151643],
                trace: false,
                ..Default::default()
            };
            let output_ids = cuda_model.generate_gpu_resident(&input_ids, &gen_config)?;
            // Decode using APR tokenizer
            let output_text = crate::apr::AprV2Model::load(model_path)
                .ok()
                .and_then(|m| m.load_embedded_tokenizer()).map_or_else(|| output_ids.iter().map(|&id| format!("[{}]", id)).collect(), |tok| tok.decode(&output_ids));
            print!("{output_text}");
            return Ok(());
        }
    }

    let load_start = Instant::now();

    // GH-479: Load APR via MappedAprModel + OwnedQuantizedModel — per-tensor scratch
    // dequant avoids eager F32 inflation (GH-478). Replaces AprTransformer::from_apr_bytes.
    let model_path_for_load = Path::new(model_ref);
    let mapped = MappedAprModel::from_path(model_path_for_load).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "parse_apr".to_string(),
            reason: format!("Failed to map APR file: {e}"),
        }
    })?;
    let model = OwnedQuantizedModel::from_apr(&mapped).map_err(|e| {
        crate::error::RealizarError::UnsupportedOperation {
            operation: "parse_apr".to_string(),
            reason: format!("Failed to load APR as OwnedQuantizedModel: {e}"),
        }
    })?;

    // APR-TRACE-001: Set model info
    tracer.set_model_info(ModelInfo {
        name: model_ref.to_string(),
        num_layers: model.config.num_layers,
        hidden_dim: model.config.hidden_dim,
        vocab_size: model.config.vocab_size,
        num_heads: model.config.num_heads,
        quant_type: Some("APR scratch-dequant".to_string()),
    });

    let load_time = load_start.elapsed();
    if verbose {
        println!("Backend: CPU (per-tensor scratch dequant)");
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

    tracer.trace_encode(prompt, &prompt_tokens, model.config.vocab_size);

    if verbose {
        println!("Prompt tokens: {}", prompt_len);
        println!("Temperature: {:.1}", temperature);
        println!();
    }

    // APR-TRACE-001: Trace embedding (approximation - we don't have direct access)
    tracer.start_step(TraceStep::Embed);
    tracer.trace_embed(prompt_len, model.config.hidden_dim, None);

    // APR-TRACE-001: Trace transformer blocks (high-level, generation is a black box)
    tracer.start_step(TraceStep::TransformerBlock);

    // GH-479: OwnedQuantizedModel::generate_with_cache — O(n) autoregressive decode
    // with per-tensor scratch dequant on each matmul (no eager F32 inflation).
    let gen_config = QuantizedGenerateConfig {
        max_tokens,
        temperature,
        top_k: if temperature == 0.0 { 1 } else { 40 },
        trace: trace_config.is_some(),
        ..Default::default()
    };
    let gen_start = Instant::now();
    let generated = model.generate_with_cache(&prompt_tokens, &gen_config)?;
    let gen_time = gen_start.elapsed();

    // Record transformer block completion (aggregate timing)
    tracer.trace_layer(
        model.config.num_layers - 1,
        0,
        None,
        1,
        model.config.hidden_dim,
    );

    let tokens_generated = generated.len().saturating_sub(prompt_len);
    let tokens_per_sec = if gen_time.as_secs_f64() > 0.0 {
        tokens_generated as f64 / gen_time.as_secs_f64()
    } else {
        0.0
    };

    // Decode output using proper tokenizer (PMAT-171)
    let output_tokens = &generated[prompt_len..];
    let output_text = decode_apr_output_tokens(model_path, output_tokens);

    // APR-TRACE-001: Trace decode for each output token
    for (i, &token) in output_tokens.iter().enumerate() {
        tracer.start_step(TraceStep::Decode);
        let decoded = output_text
            .chars()
            .nth(i.min(output_text.len().saturating_sub(1)))
            .map_or_else(|| format!("<{token}>"), |c| c.to_string());
        tracer.trace_decode(i, token, &decoded, model.config.vocab_size);
    }

    match format {
        "json" => {
            let json = serde_json::json!({
                "model": model_ref,
                "format": "APR",
                "backend": "CPU",
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
                println!("Performance: {:.1} tok/s", tokens_per_sec);
            } else {
                // Clean output: just the response
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

/// Decode APR output tokens using the best available tokenizer (PMAT-171)
///
/// Tries embedded vocabulary first, then external tokenizer.json, then ASCII fallback.
fn decode_apr_output_tokens(model_path: &std::path::Path, output_tokens: &[u32]) -> String {
    use crate::apr::AprV2Model;

    let model = AprV2Model::load(model_path).ok();
    if let Some(ref m) = model {
        if let Some(simple_tok) = m.load_embedded_tokenizer() {
            return AprV2Model::decode_tokens(&simple_tok.id_to_token, output_tokens);
        }
        if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
            return tokenizer.decode(output_tokens);
        }
    } else if let Some(tokenizer) = AprV2Model::load_tokenizer(model_path) {
        return tokenizer.decode(output_tokens);
    }
    // Ultimate fallback: simple ASCII
    output_tokens
        .iter()
        .map(|&t| char::from_u32(t.min(127)).unwrap_or('?'))
        .collect()
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
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature).collect();

    let mut indexed: Vec<(usize, f32)> = scaled.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top = &indexed[..top_k.min(indexed.len())];

    let max_val = top[0].1;
    let exp_vals: Vec<(usize, f32)> = top.iter().map(|&(i, v)| (i, (v - max_val).exp())).collect();
    let sum: f32 = exp_vals.iter().map(|(_, v)| v).sum();

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
