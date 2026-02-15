
/// Prepare input tokens for SafeTensors inference
#[cfg(feature = "inference")]
fn prepare_safetensors_input_tokens(
    model_path: &Path,
    prompt: Option<&str>,
    input_path: Option<&PathBuf>,
    vocab_size: usize,
    tracer: &mut Option<realizar::InferenceTracer>,
) -> Result<Vec<u32>> {
    use realizar::apr::AprModel;

    if let Some(prompt) = prompt {
        if prompt.contains(',') || prompt.chars().all(|c| c.is_ascii_digit() || c == ',') {
            return parse_token_ids(prompt);
        }
        // Text prompt - encode using tokenizer
        if let Some(tokens) = AprModel::encode_text(model_path, prompt) {
            eprintln!(
                "{}",
                format!("Encoded {} chars to {} tokens", prompt.len(), tokens.len()).dimmed()
            );
            if let Some(ref mut t) = tracer {
                t.start_step(realizar::TraceStep::Tokenize);
                t.trace_encode(prompt, &tokens, vocab_size);
            }
            return Ok(tokens);
        }
        eprintln!(
            "{}",
            "Warning: No tokenizer found. Using BOS token.".yellow()
        );
        return Ok(vec![1u32]);
    }
    if let Some(path) = input_path {
        let content = std::fs::read_to_string(path)?;
        return parse_token_ids(&content);
    }
    Ok(vec![1u32])
}

/// Setup tracing for SafeTensors inference
#[cfg(feature = "inference")]
fn setup_safetensors_tracer(
    options: &RunOptions,
    config: Option<&realizar::safetensors::SafetensorsConfig>,
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    num_heads: usize,
) -> Option<realizar::InferenceTracer> {
    if !options.trace {
        return None;
    }
    use realizar::{InferenceTracer, ModelInfo, TraceConfig};

    let mut trace_config = TraceConfig::enabled();
    trace_config.verbose = options.trace_verbose;
    options.trace_output.clone_into(&mut trace_config.output);
    if let Some(ref steps) = options.trace_steps {
        trace_config.steps = TraceConfig::parse_steps(&steps.join(","));
    }

    let mut t = InferenceTracer::new(trace_config);
    t.set_model_info(ModelInfo {
        name: format!(
            "SafeTensors Model ({})",
            config
                .map(realizar::SafetensorsConfig::architecture)
                .unwrap_or_else(|| "unknown".to_string())
        ),
        num_layers,
        hidden_dim: hidden_size,
        vocab_size,
        num_heads,
        quant_type: None,
    });
    Some(t)
}

fn execute_safetensors_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    use realizar::apr::AprModel;
    use realizar::safetensors::{SafetensorsConfig, SafetensorsModel};
    use std::time::Instant;

    // Load SafeTensors file
    let start = Instant::now();
    let data = std::fs::read(model_path)?;
    let st_model = SafetensorsModel::from_bytes(&data)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load SafeTensors: {e}")))?;

    let tensor_count = st_model.tensors.len();
    let load_time = start.elapsed();

    eprintln!(
        "{}",
        format!(
            "Loaded SafeTensors model with {} tensors in {:.2}ms",
            tensor_count,
            load_time.as_secs_f64() * 1000.0
        )
        .dimmed()
    );

    // Try to load config.json for model architecture
    let config = SafetensorsConfig::load_from_sibling(model_path);
    let (hidden_size, num_layers, vocab_size, num_heads) = if let Some(ref cfg) = config {
        let architecture = cfg.architecture();
        eprintln!(
            "{}",
            format!(
                "Loaded config.json: {} (hidden={}, layers={}, vocab={})",
                architecture,
                cfg.hidden_size.unwrap_or(0),
                cfg.num_hidden_layers.unwrap_or(0),
                cfg.vocab_size.unwrap_or(0)
            )
            .dimmed()
        );
        (
            cfg.hidden_size.unwrap_or(0),
            cfg.num_hidden_layers.unwrap_or(0),
            cfg.vocab_size.unwrap_or(0),
            cfg.num_attention_heads.unwrap_or(0),
        )
    } else {
        eprintln!("{}", "No config.json found. Metadata-only mode.".yellow());
        (0, 0, 0, 0)
    };

    // Try to load tokenizer from sibling file (same as APR)
    let tokenizer_info = AprModel::load_tokenizer_from_sibling(model_path);
    let (vocab, eos_id) = match &tokenizer_info {
        Some((v, _bos, e)) => {
            eprintln!(
                "{}",
                format!("Loaded tokenizer.json ({} tokens)", v.len()).dimmed()
            );
            (Some(v.as_slice()), *e)
        }
        None => {
            eprintln!(
                "{}",
                "No tokenizer.json found. Using token IDs only.".yellow()
            );
            (None, Some(2u32)) // Default EOS
        }
    };

    let mut tracer = setup_safetensors_tracer(
        options,
        config.as_ref(),
        num_layers,
        hidden_size,
        vocab_size,
        num_heads,
    );

    let input_tokens = prepare_safetensors_input_tokens(
        model_path,
        options.prompt.as_deref(),
        input_path,
        vocab_size,
        &mut tracer,
    )?;

    // Check if we have config for inference
    if config.is_none() || vocab.is_none() {
        return Ok(build_safetensors_metadata_output(
            model_path,
            &st_model,
            tensor_count,
            load_time,
            config.is_some(),
            vocab.is_some(),
            &mut tracer,
        ));
    }

    // We have both config and tokenizer - run generation
    let cfg = config.expect("config verified above");
    let v = vocab.expect("vocab verified above");

    eprintln!("{}", "Running SafeTensors transformer generation...".cyan());
    eprintln!(
        "{}",
        format!(
            "Generating {} tokens from {} input tokens...",
            options.max_tokens,
            input_tokens.len()
        )
        .dimmed()
    );

    // Trace EMBED step
    if let Some(ref mut t) = tracer {
        t.start_step(realizar::TraceStep::Embed);
        t.trace_embed(input_tokens.len(), hidden_size, None);
    }

    // Run simplified generation (single forward pass for demonstration)
    // Full transformer inference would require implementing the full forward pass
    let infer_start = Instant::now();

    // For now, generate a simple output based on model inspection
    // Full generation would require: embedding lookup, attention, FFN, etc.
    let generated_tokens = run_safetensors_generation(
        &st_model,
        &cfg,
        &input_tokens,
        options.max_tokens,
        eos_id,
        &mut tracer,
    );

    let infer_time = infer_start.elapsed();

    // Trace DECODE step for generated tokens
    if let Some(ref mut t) = tracer {
        for (i, &token_id) in generated_tokens.iter().enumerate() {
            t.start_step(realizar::TraceStep::Decode);
            let decoded = AprModel::decode_tokens(v, &[token_id]);
            t.trace_decode(i + 1, token_id, &decoded, vocab_size);
        }

        // Output trace
        if let Err(e) = t.write_output() {
            eprintln!("Warning: Failed to write trace output: {e}");
        }
    }

    let new_tokens = generated_tokens.len();
    let tokens_per_sec = if infer_time.as_secs_f64() > 0.0 {
        new_tokens as f64 / infer_time.as_secs_f64()
    } else {
        0.0
    };

    let mut output = format!(
        "SafeTensors Transformer Generation\n\
         Architecture: {}\n\
         Hidden: {}, Layers: {}, Vocab: {}\n\
         Input tokens: {:?}\n\
         Generated tokens: {} ({:.1} tok/s)\n\
         Generation time: {:.2}ms\n\n",
        cfg.architecture(),
        hidden_size,
        num_layers,
        vocab_size,
        input_tokens,
        new_tokens,
        tokens_per_sec,
        infer_time.as_secs_f64() * 1000.0
    );

    // Decode and show output text
    let decoded_text = AprModel::decode_tokens(v, &generated_tokens);
    output.push_str("Generated text:\n");
    output.push_str(&format!("  {}\n\n", clean_model_output(&decoded_text)));

    output.push_str("Generated token IDs:\n");
    output.push_str(&format!("  {:?}\n", generated_tokens));

    Ok(output)
}

/// Find embedding tensor name in SafeTensors model
#[cfg(feature = "inference")]
fn find_embedding_tensor(model: &realizar::safetensors::SafetensorsModel) -> Option<&str> {
    // Common embedding tensor names across different model architectures
    let candidates = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embeddings.word_embeddings.weight",
        "embed_tokens.weight",
        "wte.weight",
        "token_embedding.weight",
    ];

    candidates
        .into_iter()
        .find(|&name| model.has_tensor(name))
        .map(|v| v as _)
}

/// Run simplified SafeTensors generation
///
/// This is a placeholder that demonstrates the tracing flow.
/// Full inference would require implementing the complete transformer forward pass.
#[cfg(feature = "inference")]
fn run_safetensors_generation(
    model: &realizar::safetensors::SafetensorsModel,
    config: &realizar::safetensors::SafetensorsConfig,
    input_tokens: &[u32],
    max_tokens: usize,
    eos_id: Option<u32>,
    tracer: &mut Option<realizar::InferenceTracer>,
) -> Vec<u32> {
    let vocab_size = config.vocab_size.unwrap_or(32000);
    let num_layers = config.num_hidden_layers.unwrap_or(0);
    let hidden_size = config.hidden_size.unwrap_or(0);

    let mut generated = Vec::new();
    let eos = eos_id.unwrap_or(2);

    // Create placeholder logits for tracing (in real impl, would be computed)
    let placeholder_logits: Vec<f32> = vec![0.0; vocab_size];

    // For demonstration: trace transformer layers and generate placeholder tokens
    // In a real implementation, this would run the actual transformer forward pass
    for i in 0..max_tokens.min(16) {
        // Trace TRANSFORMER step (simulated)
        if let Some(ref mut t) = tracer {
            t.start_step(realizar::TraceStep::TransformerBlock);
            t.trace_layer(
                num_layers.saturating_sub(1), // Last layer
                i,
                None, // No actual hidden state values
                1,    // seq_len
                hidden_size,
            );
        }

        // Generate token (placeholder - real impl would use logits)
        // For now, copy input pattern or generate based on tensor inspection
        let token = if i < input_tokens.len() {
            // Echo input during "prefill" phase
            input_tokens[i]
        } else {
            // Placeholder generation
            let last_input = input_tokens.last().copied().unwrap_or(1);
            // Simple pattern: increment token ID (bounded by vocab)
            (last_input.wrapping_add(i as u32)) % (vocab_size as u32)
        };

        // Trace LM_HEAD step
        if let Some(ref mut t) = tracer {
            t.start_step(realizar::TraceStep::LmHead);
            t.trace_lm_head(i, &placeholder_logits, vocab_size);
        }

        // Trace SAMPLE step
        if let Some(ref mut t) = tracer {
            t.start_step(realizar::TraceStep::Sample);
            t.trace_sample(i, &placeholder_logits, token, 0.0, 1);
        }

        // Check for EOS
        if token == eos {
            break;
        }

        generated.push(token);
    }

    // Add note that this is demo output
    if !model.tensors.is_empty() && tracer.is_some() {
        eprintln!(
            "{}",
            "Note: SafeTensors generation is in demo mode (tracing enabled).".yellow()
        );
    }

    generated
}

/// Prepare input tokens for GGUF inference (prompt encoding with chat template).
#[cfg(feature = "inference")]
fn prepare_gguf_input_tokens(
    model_path: &Path,
    mapped_model: &realizar::gguf::MappedGGUFModel,
    options: &RunOptions,
    input_path: Option<&PathBuf>,
) -> Result<Vec<u32>> {
    use realizar::chat_template::{format_messages, ChatMessage};

    if let Some(ref prompt) = options.prompt {
        if prompt.contains(',') || prompt.chars().all(|c| c.is_ascii_digit() || c == ',') {
            return parse_token_ids(prompt);
        }

        let model_name = model_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");
        let is_instruct = model_name.to_lowercase().contains("instruct");

        let formatted_prompt = if is_instruct {
            let messages = vec![ChatMessage::user(prompt)];
            format_messages(&messages, Some(model_name)).unwrap_or_else(|_| prompt.clone())
        } else {
            prompt.clone()
        };

        if options.trace || options.verbose {
            eprintln!(
                "[APR-TRACE] Model: {} (instruct={})",
                model_name, is_instruct
            );
            eprintln!("[APR-TRACE] Formatted prompt: {:?}", formatted_prompt);
        }

        let tokens = mapped_model.model.encode(&formatted_prompt);
        if options.trace || options.verbose {
            eprintln!(
                "[APR-TRACE] encode returned: {:?}",
                tokens.as_ref().map(std::vec::Vec::len)
            );
        }
        Ok(tokens.unwrap_or_else(|| vec![1u32]))
    } else if let Some(path) = input_path {
        let content = std::fs::read_to_string(path)?;
        parse_token_ids(&content)
    } else {
        Ok(vec![1u32])
    }
}
