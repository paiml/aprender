
/// Resolve tokenizer info into vocab slice, BOS ID, and EOS ID.
///
/// Returns default EOS=2 when no tokenizer is available.
#[cfg(feature = "inference")]
fn resolve_tokenizer_info(
    tokenizer_info: Option<&(Vec<String>, Option<u32>, Option<u32>)>,
) -> (Option<&[String]>, Option<u32>, Option<u32>) {
    match tokenizer_info {
        Some((v, b, e)) => {
            eprintln!(
                "{}",
                format!("Loaded tokenizer ({} tokens)", v.len()).dimmed()
            );
            (Some(v.as_slice()), *b, *e)
        }
        None => {
            eprintln!(
                "{}",
                "No tokenizer.json found. Using token IDs only.".yellow()
            );
            (None, None, Some(2u32))
        }
    }
}

/// Execute APR model inference (APR v2 format)
///
/// APR v2 now supports transformer inference with forward() and generate() methods.
/// For transformer models with proper metadata, runs autoregressive generation.
/// Supports GPU acceleration via `AprV2ModelCuda` when `--gpu` is specified.
#[cfg(feature = "inference")]
fn execute_apr_inference(
    model_path: &Path,
    input_path: Option<&PathBuf>,
    options: &RunOptions,
) -> Result<String> {
    use realizar::apr::AprModel;
    use std::time::Instant;

    // Check if GPU should be used
    #[cfg(feature = "cuda")]
    let use_gpu = !options.no_gpu && realizar::apr::AprV2ModelCuda::is_available();
    #[cfg(not(feature = "cuda"))]
    let use_gpu = false;

    // Load the APR v2 model
    let start = Instant::now();
    let model = AprModel::load(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load APR model: {e}")))?;
    let load_time = start.elapsed();

    // Display model info
    let model_type = model
        .metadata()
        .model_type
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let architecture = model
        .metadata()
        .architecture
        .clone()
        .unwrap_or_else(|| "unknown".to_string());

    eprintln!(
        "{}",
        format!(
            "Loaded {} model (arch: {}, {} tensors, ~{} parameters) in {:.2}ms",
            model_type,
            architecture,
            model.tensor_count(),
            model.estimated_parameters(),
            load_time.as_secs_f64() * 1000.0
        )
        .dimmed()
    );

    // Check if this is a transformer model
    if model.metadata().is_transformer() {
        eprintln!("{}", "Running transformer generation...".cyan());

        let tokenizer_info = AprModel::load_tokenizer_from_sibling(model_path);
        let (vocab, _bos_id, eos_id) = resolve_tokenizer_info(tokenizer_info.as_ref());

        let input_tokens =
            prepare_apr_input_tokens(model_path, options.prompt.as_deref(), input_path)?;

        let max_new_tokens = options.max_tokens;

        // Capture vocab_size before potential model move
        let vocab_size = model.metadata().vocab_size.unwrap_or(0);

        // Report backend selection
        let backend_label = if use_gpu { "GPU" } else { "CPU" };
        eprintln!(
            "{}",
            format!(
                "Generating {} tokens from {} input tokens ({} backend)...",
                max_new_tokens,
                input_tokens.len(),
                backend_label
            )
            .dimmed()
        );

        let mut tracer =
            setup_apr_tracer(options, &architecture, &model, vocab_size, &input_tokens);

        // Run generation (GPU or CPU path)
        let infer_start = Instant::now();
        let output_tokens = if use_gpu {
            // GPU path via AprV2ModelCuda
            #[cfg(feature = "cuda")]
            {
                let mut cuda_model = realizar::apr::AprV2ModelCuda::new(model, 0)
                    .map_err(|e| CliError::ModelLoadFailed(format!("CUDA init failed: {e}")))?;
                eprintln!(
                    "{}",
                    format!(
                        "Using GPU: {} ({} MB VRAM)",
                        cuda_model.device_name(),
                        cuda_model.vram_mb()
                    )
                    .green()
                );
                cuda_model
                    .generate_cuda(&input_tokens, max_new_tokens, eos_id.unwrap_or(2))
                    .map_err(|e| CliError::InferenceFailed(format!("GPU generation failed: {e}")))?
            }
            #[cfg(not(feature = "cuda"))]
            {
                // Fallback to CPU (should not reach here due to use_gpu check)
                model
                    .generate(&input_tokens, max_new_tokens, eos_id)
                    .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?
            }
        } else {
            // CPU path
            model
                .generate(&input_tokens, max_new_tokens, eos_id)
                .map_err(|e| CliError::InferenceFailed(format!("Generation failed: {e}")))?
        };
        let infer_time = infer_start.elapsed();

        trace_apr_decode_steps(
            &mut tracer,
            &output_tokens[input_tokens.len()..],
            vocab,
            vocab_size,
        );

        return Ok(format_apr_inference_output(
            &architecture,
            vocab_size,
            &input_tokens,
            &output_tokens,
            infer_time,
            vocab,
        ));
    }

    Ok(format_non_transformer_output(
        &model,
        &model_type,
        &architecture,
        load_time,
    ))
}

/// Trace DECODE steps for each generated token (APR-TRACE-001).
#[cfg(feature = "inference")]
fn trace_apr_decode_steps(
    tracer: &mut Option<realizar::InferenceTracer>,
    generated: &[u32],
    vocab: Option<&[String]>,
    vocab_size: usize,
) {
    let Some(ref mut t) = tracer else { return };
    for (i, &token_id) in generated.iter().enumerate() {
        t.start_step(realizar::TraceStep::Decode);
        let decoded = vocab
            .map(|v| realizar::apr::AprModel::decode_tokens(v, &[token_id]))
            .unwrap_or_else(|| format!("<token_{}>", token_id));
        t.trace_decode(i + 1, token_id, &decoded, vocab_size);
    }
    if let Err(e) = t.write_output() {
        eprintln!("Warning: Failed to write trace output: {e}");
    }
}

/// Format output for non-transformer models (metadata display).
#[cfg(feature = "inference")]
fn format_non_transformer_output(
    model: &realizar::apr::AprModel,
    model_type: &str,
    architecture: &str,
    load_time: std::time::Duration,
) -> String {
    let tensor_names = model.tensor_names();
    let mut output = format!(
        "APR v2 Model: {}\nArchitecture: {}\nTensors: {}\nLoad time: {:.2}ms\n\n",
        model_type,
        architecture,
        model.tensor_count(),
        load_time.as_secs_f64() * 1000.0
    );
    output.push_str("Available tensors:\n");
    for name in tensor_names.iter().take(20) {
        output.push_str(&format!("  - {name}\n"));
    }
    if tensor_names.len() > 20 {
        output.push_str(&format!("  ... and {} more\n", tensor_names.len() - 20));
    }
    output.push_str("\nNote: Model missing transformer config. Add hidden_size, num_layers, num_heads, vocab_size to metadata.");
    output
}

/// Parse token IDs from input string (JSON array or comma-separated)
#[cfg(feature = "inference")]
fn parse_token_ids(input: &str) -> Result<Vec<u32>> {
    let trimmed = input.trim();
    if trimmed.starts_with('[') {
        // JSON array
        serde_json::from_str(trimmed)
            .map_err(|e| CliError::InvalidFormat(format!("Failed to parse token IDs: {e}")))
    } else {
        // Comma or space separated
        trimmed
            .split([',', ' ', '\n', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim()
                    .parse::<u32>()
                    .map_err(|e| CliError::InvalidFormat(format!("Invalid token ID: {s} - {e}")))
            })
            .collect()
    }
}

/// Setup inference tracer if tracing is enabled (APR-TRACE-001).
#[cfg(feature = "inference")]
fn setup_apr_tracer(
    options: &RunOptions,
    architecture: &str,
    model: &realizar::apr::AprModel,
    vocab_size: usize,
    input_tokens: &[u32],
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
        name: format!("APR Model ({architecture})"),
        num_layers: model.metadata().num_layers.unwrap_or(0),
        hidden_dim: model.metadata().hidden_size.unwrap_or(0),
        vocab_size,
        num_heads: model.metadata().num_heads.unwrap_or(0),
        quant_type: None,
    });

    t.start_step(realizar::TraceStep::Tokenize);
    t.trace_encode(
        options.prompt.as_deref().unwrap_or(""),
        input_tokens,
        vocab_size,
    );

    Some(t)
}

/// Format the output string for APR transformer inference results.
#[cfg(feature = "inference")]
fn format_apr_inference_output(
    architecture: &str,
    vocab_size: usize,
    input_tokens: &[u32],
    output_tokens: &[u32],
    infer_time: std::time::Duration,
    vocab: Option<&[String]>,
) -> String {
    use realizar::apr::AprModel;
    use std::fmt::Write;

    let new_tokens = output_tokens.len().saturating_sub(input_tokens.len());
    let tokens_per_sec = if infer_time.as_secs_f64() > 0.0 {
        new_tokens as f64 / infer_time.as_secs_f64()
    } else {
        0.0
    };

    let mut output = format!(
        "APR v2 Transformer Generation\n\
         Architecture: {architecture}\n\
         Vocab size: {vocab_size}\n\
         Input tokens: {input_tokens:?}\n\
         Generated tokens: {new_tokens} new ({} total)\n\
         Generation time: {:.2}ms ({tokens_per_sec:.1} tok/s)\n\n",
        output_tokens.len(),
        infer_time.as_secs_f64() * 1000.0,
    );

    if let Some(v) = vocab {
        let generated_tokens = &output_tokens[input_tokens.len()..];
        let decoded_text = AprModel::decode_tokens(v, generated_tokens);
        output.push_str("Generated text:\n");
        output.push_str(&format!("  {decoded_text}\n\n"));
    }

    output.push_str("Output tokens:\n");
    output.push_str(&format!("  {output_tokens:?}\n"));

    if new_tokens > 0 {
        output.push_str("\nGenerated token IDs:\n  ");
        for (i, &tok) in output_tokens.iter().skip(input_tokens.len()).enumerate() {
            if i > 0 {
                output.push_str(", ");
            }
            write!(output, "{tok}").expect("write to String cannot fail");
        }
        output.push('\n');
    }

    output
}

/// Prepare input tokens from prompt text, file, or default BOS token.
///
/// Handles three input sources: text prompt (encode via tokenizer or parse IDs),
/// file path (parse token IDs from file), or default (BOS token).
#[cfg(feature = "inference")]
fn prepare_apr_input_tokens(
    model_path: &Path,
    prompt: Option<&str>,
    input_path: Option<&PathBuf>,
) -> Result<Vec<u32>> {
    use realizar::apr::AprModel;

    if let Some(prompt) = prompt {
        if prompt.contains(',') || prompt.chars().all(|c| c.is_ascii_digit() || c == ',') {
            return parse_token_ids(prompt);
        }
        // Text prompt — encode using tokenizer
        if let Some(tokens) = AprModel::encode_text(model_path, prompt) {
            eprintln!(
                "{}",
                format!("Encoded {} chars to {} tokens", prompt.len(), tokens.len()).dimmed()
            );
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

    Ok(vec![1u32]) // Default: BOS token
}

/// Clean model output by stripping ChatML markers and extra tokens
#[cfg(feature = "inference")]
fn clean_model_output(raw: &str) -> String {
    let mut cleaned = raw.to_string();
    // Strip ChatML markers commonly present in instruct model output
    let markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<|im_end|>",
        "<|im_start|>",
        "<|endoftext|>",
    ];
    for marker in markers {
        cleaned = cleaned.replace(marker, "");
    }
    cleaned.trim().to_string()
}

/// Execute SafeTensors model inference
///
/// Build metadata-only output when config or tokenizer is missing
#[cfg(feature = "inference")]
fn build_safetensors_metadata_output(
    model_path: &Path,
    st_model: &realizar::safetensors::SafetensorsModel,
    tensor_count: usize,
    load_time: std::time::Duration,
    has_config: bool,
    has_vocab: bool,
    tracer: &mut Option<realizar::InferenceTracer>,
) -> String {
    let tensor_names: Vec<&str> = st_model.tensor_names();
    let mut output = format!(
        "SafeTensors Model (metadata only)\n\
         Model: {}\n\
         Tensors: {}\n\
         Load time: {:.2}ms\n\n",
        model_path.display(),
        tensor_count,
        load_time.as_secs_f64() * 1000.0
    );

    if !has_config {
        output.push_str("Note: No config.json found - cannot run inference.\n");
        output.push_str("      Place config.json in the same directory as the model.\n\n");
        if let Some(ref mut t) = tracer {
            t.record_execution_failed("Initialization Failure", "Missing config.json");
        }
    }
    if !has_vocab {
        output.push_str("Note: No tokenizer.json found - cannot encode/decode text.\n");
        output.push_str("      Place tokenizer.json in the same directory as the model.\n\n");
    }

    output.push_str("Tensor names (first 15):\n");
    for (i, name) in tensor_names.iter().take(15).enumerate() {
        if let Some(info) = st_model.get_tensor_info(name) {
            output.push_str(&format!(
                "  {}. {} ({:?}, {:?})\n",
                i + 1,
                name,
                info.dtype,
                info.shape
            ));
        }
    }

    if tensor_names.len() > 15 {
        output.push_str(&format!("  ... and {} more\n", tensor_names.len() - 15));
    }

    if let Some(ref mut t) = tracer {
        if let Err(e) = t.write_output() {
            eprintln!("Warning: Failed to write trace output: {e}");
        }
    }

    output
}
