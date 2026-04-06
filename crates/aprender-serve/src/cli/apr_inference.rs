
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
    use crate::apr::AprV2Model;
    use crate::apr_transformer::AprTransformer;
    use crate::inference_trace::{InferenceTracer, ModelInfo, TraceStep};
    use std::path::Path;
    use std::time::Instant;

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
        use crate::apr::MappedAprModel;
        use crate::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda, QuantizedGenerateConfig};
        use std::path::Path;

        let model_path = Path::new(model_ref);

        // Load APR via MappedAprModel for weight data
        let mapped = MappedAprModel::from_path(model_path).map_err(|e| {
            crate::error::RealizarError::FormatError { reason: format!("APR load failed: {e}") }
        })?;
        let model = OwnedQuantizedModel::from_apr(&mapped).map_err(|e| {
            crate::error::RealizarError::FormatError { reason: format!("APR→GGUF failed: {e}") }
        })?;
        let mut cuda_model = OwnedQuantizedModelCuda::with_max_seq_len(model, 0, 4096)
            .map_err(|e| e.error)?;

        // Tokenize using embedded APR tokenizer (same as CPU path)
        let input_ids = crate::apr::AprV2Model::encode_text(model_path, prompt)
            .or_else(|| crate::apr::AprV2Model::load_tokenizer(model_path).map(|t| t.encode(prompt)))
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

    let load_start = Instant::now();

    // Load APR transformer (CPU path)
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
        quant_type: Some("APR F32".to_string()),
    });

    let load_time = load_start.elapsed();
    if verbose {
        println!("Backend: CPU (AVX2 + SIMD)");
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
        println!("Temperature: {:.1} (using greedy decoding)", temperature);
        println!();
    }

    // APR-TRACE-001: Trace embedding (approximation - we don't have direct access)
    tracer.start_step(TraceStep::Embed);
    tracer.trace_embed(prompt_len, transformer.config.hidden_dim, None);

    // APR-TRACE-001: Trace transformer blocks (high-level, generation is a black box)
    tracer.start_step(TraceStep::TransformerBlock);

    // Run inference with timing
    // PMAT-103 FIX: Use generate_with_cache for O(n) instead of O(n²) complexity
    let gen_config = crate::apr_transformer::GenerateConfig {
        max_tokens,
        temperature,
        ..Default::default()
    };
    let gen_start = Instant::now();
    let generated = transformer.generate_with_cache(&prompt_tokens, &gen_config)?;
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
        tracer.trace_decode(i, token, &decoded, transformer.config.vocab_size);
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

/// Print verbose debug weight comparison between CPU and GPU models
#[cfg(feature = "cuda")]
fn print_gpu_debug_weights(
    transformer: &crate::apr_transformer::AprTransformer,
    gpu_model: &crate::gpu::GpuModel,
) {
    let has_gate = transformer
        .layers
        .first()
        .is_some_and(|l| l.ffn_gate_weight.is_some());
    eprintln!(
        "[DEBUG-SwiGLU] APR transformer has gate weight: {}",
        has_gate
    );
    if has_gate {
        let gate_len = transformer.layers[0]
            .ffn_gate_weight
            .as_ref()
            .map_or(0, Vec::len);
        eprintln!(
            "[DEBUG-SwiGLU] Gate weight elements: {} (expected: {}x{}={})",
            gate_len,
            transformer.config.hidden_dim,
            transformer.config.intermediate_dim,
            transformer.config.hidden_dim * transformer.config.intermediate_dim
        );
    }

    let has_gpu_gate = gpu_model
        .block_weights
        .first()
        .is_some_and(|b| b.ffn_gate_weight.is_some());
    eprintln!("[DEBUG-SwiGLU] GpuModel has gate weight: {}", has_gpu_gate);

    if let Some(layer0) = transformer.layers.first() {
        eprintln!(
            "[DEBUG-WEIGHT] CPU qkv_weight first 5: {:?}",
            &layer0.qkv_weight[0..5.min(layer0.qkv_weight.len())]
        );
        eprintln!(
            "[DEBUG-WEIGHT] GPU qkv_weight first 5: {:?}",
            &gpu_model.block_weights[0].qkv_weight
                [0..5.min(gpu_model.block_weights[0].qkv_weight.len())]
        );
        eprintln!(
            "[DEBUG-WEIGHT] CPU fc1 (up) first 5: {:?}",
            &layer0.ffn_up_weight[0..5.min(layer0.ffn_up_weight.len())]
        );
        eprintln!(
            "[DEBUG-WEIGHT] GPU fc1 (up) first 5: {:?}",
            &gpu_model.block_weights[0].ffn_fc1_weight
                [0..5.min(gpu_model.block_weights[0].ffn_fc1_weight.len())]
        );

        let hidden_dim = transformer.config.hidden_dim;
        if transformer.token_embedding.len() < hidden_dim {
            eprintln!("[DEBUG] token_embedding too short ({} < {hidden_dim}), skipping weight debug", transformer.token_embedding.len());
            return;
        }
        let test_embedding = &transformer.token_embedding[0..hidden_dim];

        // CPU matmul: y = x @ W where W is [out_dim, in_dim] (transposed internally)
        let cpu_qkv_dim = layer0.qkv_weight.len() / hidden_dim;
        let mut cpu_qkv = vec![0.0f32; cpu_qkv_dim];
        for o in 0..cpu_qkv_dim {
            let w_start = o * hidden_dim;
            let mut sum = 0.0f32;
            for i in 0..hidden_dim {
                sum += test_embedding[i] * layer0.qkv_weight[w_start + i];
            }
            cpu_qkv[o] = sum;
        }

        // GPU matmul: y = x @ W_t where W_t is [in_dim, out_dim] (already transposed)
        let gpu_qkv_weight = &gpu_model.block_weights[0].qkv_weight;
        let gpu_qkv_dim = gpu_qkv_weight.len() / hidden_dim;
        let mut gpu_qkv = vec![0.0f32; gpu_qkv_dim];
        for j in 0..gpu_qkv_dim {
            let mut sum = 0.0f32;
            for i in 0..hidden_dim {
                sum += test_embedding[i] * gpu_qkv_weight[i * gpu_qkv_dim + j];
            }
            gpu_qkv[j] = sum;
        }

        eprintln!(
            "[DEBUG-MATMUL] CPU QKV first 5: {:?}",
            &cpu_qkv[0..5.min(cpu_qkv.len())]
        );
        eprintln!(
            "[DEBUG-MATMUL] GPU QKV first 5: {:?}",
            &gpu_qkv[0..5.min(gpu_qkv.len())]
        );

        let max_diff: f32 = cpu_qkv
            .iter()
            .zip(gpu_qkv.iter())
            .map(|(c, g)| (c - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[DEBUG-MATMUL] Max diff: {}", max_diff);
        if max_diff > 0.01 {
            eprintln!("[DEBUG-MATMUL] WARNING: CPU vs GPU QKV mismatch!");
        }
    }
}
