
/// Handle POST /v1/completions for GPU inference.
///
/// GH-284: Now async with `spawn_blocking` to avoid blocking the tokio runtime.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
async fn handle_gpu_completion(
    cuda: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tok_info: Arc<Option<SafeTensorsTokenizerInfo>>,
    req: GpuCompletionRequest,
    cpu_state: Arc<std::sync::Mutex<AprServerState>>,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let start = Instant::now();
    let tok_ref = tok_info.as_ref().as_ref();
    let input_tokens = encode_prompt(tok_ref, &req.prompt);
    let eos_id = eos_token_id(tok_ref, 2);
    let max_tokens = req.max_tokens.min(128);
    let prompt = req.prompt.clone();

    // GH-284: Run GPU generation off the async runtime
    let cuda_clone = cuda.clone();
    let input_clone = input_tokens.clone();
    let result = tokio::task::spawn_blocking(move || {
        run_gpu_generation(&cuda_clone, &input_clone, max_tokens, eos_id)
    })
    .await;

    let gen_start = Instant::now();
    let output_tokens = match result {
        Ok(Ok(t)) => t,
        Ok(Err(gpu_err)) => {
            // GH-261: Per-request CPU fallback
            eprintln!("[GPU->CPU FALLBACK] {gpu_err}");
            let s = match cpu_state.lock() {
                Ok(guard) => guard.clone(),
                Err(_) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": format!("GPU failed: {gpu_err}; CPU state corrupted")
                        })),
                    )
                        .into_response();
                }
            };

            // CPU fallback also in spawn_blocking
            let result = tokio::task::spawn_blocking(move || {
                run_apr_cpu_inference(&s, &prompt, max_tokens, 0.0)
            })
            .await;

            match result {
                Ok(Ok(out)) => {
                    return Json(serde_json::json!({
                        "text": out.text,
                        "tokens_generated": out.tokens_generated,
                        "latency_ms": out.gen_duration.as_millis() as u64,
                        "tok_per_sec": compute_tok_per_sec(out.tokens_generated, out.gen_duration),
                        "compute_mode": "cpu-fallback"
                    }))
                    .into_response();
                }
                Ok(Err(cpu_err)) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": format!("GPU failed: {gpu_err}; CPU fallback also failed: {cpu_err}")
                        })),
                    )
                        .into_response();
                }
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({
                            "error": format!("GPU failed: {gpu_err}; CPU task failed: {e}")
                        })),
                    )
                        .into_response();
                }
            }
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": format!("GPU task failed: {e}")})),
            )
                .into_response();
        }
    };
    let gen_time = gen_start.elapsed();

    let new_tokens = extract_new_tokens(&output_tokens, input_tokens.len());
    let text = decode_tokens(tok_info.as_ref().as_ref(), new_tokens);

    Json(GpuCompletionResponse {
        text,
        tokens_generated: new_tokens.len(),
        latency_ms: start.elapsed().as_millis() as u64,
        tok_per_sec: compute_tok_per_sec(new_tokens.len(), gen_time),
    })
    .into_response()
}

/// Handle POST /v1/chat/completions for GPU inference (PAR-302).
///
/// GH-284: True per-token SSE streaming. GPU generates all tokens in
/// `spawn_blocking`, then streams them as individual SSE events.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
async fn handle_gpu_chat_completion(
    cuda: Arc<std::sync::Mutex<realizar::apr::AprV2ModelCuda>>,
    tok_info: Arc<Option<SafeTensorsTokenizerInfo>>,
    req: serde_json::Value,
    cpu_state: Arc<std::sync::Mutex<AprServerState>>,
) -> axum::response::Response {
    use axum::{
        response::{
            sse::{Event, Sse},
            IntoResponse,
        },
        Json,
    };

    // GH-283: Validate model name before processing
    if let Ok(s) = cpu_state.lock() {
        if let Some(err_response) = validate_request_model(&req, &s.model_name) {
            return err_response;
        }
    }

    let messages = req.get("messages").and_then(|m| m.as_array());
    let stream_mode = req
        .get("stream")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let max_tokens = req
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(32) as usize;
    let temperature = req
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0) as f32;

    let Some(msgs) = messages else {
        return Json(serde_json::json!({"error": "Missing messages"})).into_response();
    };

    let prompt = format_chatml(msgs);
    let start = Instant::now();
    let tok_ref = tok_info.as_ref().as_ref();
    let input_tokens = encode_prompt(tok_ref, &prompt);
    let eos_id = eos_token_id(tok_ref, 151_645);
    let max_tokens_clamped = max_tokens.min(256);

    // GH-284: Run GPU generation in spawn_blocking
    let cuda_clone = cuda.clone();
    let input_clone = input_tokens.clone();
    let gen_start = Instant::now();
    let gen_result = tokio::task::spawn_blocking(move || {
        run_gpu_generation(&cuda_clone, &input_clone, max_tokens_clamped, eos_id)
    })
    .await;

    let output_tokens = match gen_result {
        Ok(Ok(t)) => t,
        Ok(Err(gpu_err)) => {
            // GH-261: Per-request CPU fallback
            eprintln!("[GPU->CPU FALLBACK] {gpu_err}");
            let s = match cpu_state.lock() {
                Ok(guard) => guard.clone(),
                Err(_) => {
                    return Json(serde_json::json!({
                        "error": format!("GPU failed: {gpu_err}; CPU state corrupted")
                    }))
                    .into_response();
                }
            };

            let result = tokio::task::spawn_blocking(move || {
                run_apr_cpu_inference(&s, &prompt, max_tokens_clamped, temperature)
            })
            .await;

            match result {
                Ok(Ok(out)) => {
                    let request_id = generate_request_id();
                    let created = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_secs();
                    return Json(serde_json::json!({
                        "id": request_id,
                        "object": "chat.completion",
                        "created": created,
                        "model": "apr-cpu-fallback",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": out.text}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": out.input_token_count,
                            "completion_tokens": out.tokens_generated,
                            "total_tokens": out.input_token_count + out.tokens_generated
                        },
                        "_apr_metrics": {
                            "latency_ms": start.elapsed().as_millis() as u64,
                            "tok_per_sec": compute_tok_per_sec(out.tokens_generated, out.gen_duration),
                            "compute_mode": "cpu-fallback"
                        }
                    }))
                    .into_response();
                }
                Ok(Err(cpu_err)) => {
                    return Json(serde_json::json!({
                        "error": format!("GPU failed: {gpu_err}; CPU fallback also failed: {cpu_err}")
                    }))
                    .into_response();
                }
                Err(e) => {
                    return Json(serde_json::json!({
                        "error": format!("GPU failed: {gpu_err}; CPU task failed: {e}")
                    }))
                    .into_response();
                }
            }
        }
        Err(e) => {
            return Json(serde_json::json!({"error": format!("GPU task failed: {e}")}))
                .into_response();
        }
    };
    let elapsed = gen_start.elapsed();

    let new_tokens = extract_new_tokens(&output_tokens, input_tokens.len());
    eprintln!(
        "[APR GPU CHAT DEBUG] Input tokens: {}, Output tokens: {}, New tokens: {}",
        input_tokens.len(),
        output_tokens.len(),
        new_tokens.len()
    );

    let tokens_generated = new_tokens.len();
    let tok_per_sec = compute_tok_per_sec(tokens_generated, elapsed);
    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // GH-283: Use actual model name in response
    let response_model = cpu_state
        .lock()
        .ok()
        .map_or_else(|| "apr-gpu".to_string(), |s| s.model_name.clone());

    if stream_mode {
        // GH-284: True per-token SSE streaming — send each token as a separate event
        let new_tokens_owned: Vec<u32> = new_tokens.to_vec();

        let stream = futures_util::stream::unfold(
            (Some(new_tokens_owned.into_iter()), tok_info, request_id, created, response_model),
            |(maybe_iter, tok_info, request_id, created, model_name)| async move {
                let mut iter = maybe_iter?;
                match iter.next() {
                    Some(token_id) => {
                        let text = decode_single_token((*tok_info).as_ref(), token_id);
                        let chunk = serde_json::json!({
                            "id": &request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": &model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": serde_json::Value::Null
                            }]
                        });
                        let event = Event::default().data(chunk.to_string());
                        Some((
                            Ok::<_, std::convert::Infallible>(event),
                            (Some(iter), tok_info, request_id, created, model_name),
                        ))
                    }
                    None => {
                        // All tokens sent — yield [DONE] and end stream
                        let event = Event::default().data("[DONE]");
                        Some((
                            Ok::<_, std::convert::Infallible>(event),
                            (None, tok_info, request_id, created, model_name),
                        ))
                    }
                }
            },
        );

        Sse::new(stream).into_response()
    } else {
        let output_text = decode_tokens(tok_info.as_ref().as_ref(), new_tokens);
        Json(serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": &response_model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": input_tokens.len(),
                "completion_tokens": tokens_generated,
                "total_tokens": input_tokens.len() + tokens_generated
            },
            "_apr_metrics": {"latency_ms": start.elapsed().as_millis(), "tok_per_sec": tok_per_sec}
        }))
        .into_response()
    }
}

/// Print GPU server startup banner.
fn print_gpu_server_banner(bind_addr: &str) {
    println!();
    println!(
        "{}",
        format!("APR GPU Inference Server listening on http://{bind_addr}")
            .green()
            .bold()
    );
    println!();
    println!("{}", "Endpoints:".cyan());
    println!("  GET  /health              - Health check");
    println!("  POST /v1/completions      - GPU text generation");
    println!("  POST /v1/chat/completions - GPU chat completions (PAR-302)");
    println!();
    println!("{}", "Press Ctrl+C to stop".dimmed());
}

// ============================================================================
// GGUF server handlers
// ============================================================================

/// Start GGUF model inference server with Ollama-parity performance
///
/// Uses realizar's full inference API for text generation, streaming, and batch inference.
/// Achieves Ollama-parity: 100+ tok/s CPU, 500+ tok/s GPU.
/// With --gpu --batch flags: 800+ tok/s (2.8x Ollama) via batched GPU inference.
#[cfg(feature = "inference")]
fn start_gguf_server(model_path: &Path, config: &ServerConfig) -> Result<()> {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};

    println!("{}", "Loading GGUF model (mmap)...".dimmed());
    let mapped_model = MappedGGUFModel::from_path(model_path)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to load GGUF: {e}")))?;

    println!(
        "{}",
        format!(
            "GGUF loaded: {} tensors, {} metadata entries",
            mapped_model.model.tensors.len(),
            mapped_model.model.metadata.len()
        )
        .dimmed()
    );

    println!("{}", "Building quantized inference model...".dimmed());
    let quantized_model = OwnedQuantizedModel::from_mapped(&mapped_model)
        .map_err(|e| CliError::ModelLoadFailed(format!("Failed to build quantized model: {e}")))?;

    println!(
        "{}",
        format!(
            "Model ready: {} layers, vocab_size={}, hidden_dim={}",
            quantized_model.layers().len(),
            quantized_model.config().vocab_size,
            quantized_model.config().hidden_dim
        )
        .green()
    );

    let vocab = extract_gguf_vocab(&mapped_model, quantized_model.config().vocab_size);

    #[cfg(feature = "cuda")]
    if config.gpu && config.batch {
        return start_gguf_server_gpu_batched(quantized_model, vocab, config);
    }

    #[cfg(feature = "cuda")]
    if config.gpu && !config.no_gpu {
        return start_gguf_server_cuda(quantized_model, vocab, &mapped_model, config);
    }

    run_cpu_server(quantized_model, vocab, config)
}

/// Extract vocabulary from GGUF model, falling back to placeholder tokens.
///
/// GH-226: When the GGUF lacks `tokenizer.ggml.tokens` metadata, the placeholder
/// vocabulary must include `<unk>` so that `BPETokenizer::new` can find it.
/// Without this, the serve path fails with "Unknown token '<unk>' not in vocabulary"
/// and the server exits before binding — causing "Server failed to become ready".
fn extract_gguf_vocab(
    mapped_model: &realizar::gguf::MappedGGUFModel,
    vocab_size: usize,
) -> Vec<String> {
    mapped_model.model.vocabulary().unwrap_or_else(|| {
        eprintln!("Warning: No vocabulary in GGUF, using placeholder tokens");
        let mut vocab: Vec<String> = (0..vocab_size).map(|i| format!("token{i}")).collect();
        // GH-226: Ensure <unk> is present for BPETokenizer compatibility.
        // Use slot 0 (standard convention for unknown token).
        if !vocab.is_empty() {
            vocab[0] = "<unk>".to_string();
        }
        vocab
    })
}

/// Start GGUF server with CUDA acceleration (PAR-111).
#[cfg(all(feature = "inference", feature = "cuda"))]
fn start_gguf_server_cuda(
    quantized_model: realizar::gguf::OwnedQuantizedModel,
    vocab: Vec<String>,
    mapped_model: &realizar::gguf::MappedGGUFModel,
    config: &ServerConfig,
) -> Result<()> {
    use realizar::api::{create_router, AppState};
    use realizar::gguf::{OwnedQuantizedModel, OwnedQuantizedModelCuda};

    println!(
        "{}",
        "Enabling optimized CUDA acceleration (PAR-111)...".cyan()
    );

    match OwnedQuantizedModelCuda::new(quantized_model, 0) {
        Ok(mut cuda_model) => {
            preload_gpu_weights(&mut cuda_model);
            println!("{}", "CUDA optimized model ready".green());

            let state = AppState::with_cuda_model_and_vocab(cuda_model, vocab)
                .map_err(|e| CliError::InferenceFailed(format!("Failed to create state: {e}")))?
                .with_verbose(config.verbose);

            let app = create_router(state);
            run_server_async(app, &config.bind_addr(), "CUDA-optimized")
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("CUDA init failed, falling back to CPU: {e}").yellow()
            );
            let quantized_model = OwnedQuantizedModel::from_mapped(mapped_model).map_err(|e| {
                CliError::ModelLoadFailed(format!("Failed to rebuild quantized model: {e}"))
            })?;
            let vocab = extract_gguf_vocab(mapped_model, quantized_model.config().vocab_size);
            run_cpu_server(quantized_model, vocab, config)
        }
    }
}

/// Pre-upload model weights to GPU for maximum performance.
#[cfg(all(feature = "inference", feature = "cuda"))]
fn preload_gpu_weights(cuda_model: &mut realizar::gguf::OwnedQuantizedModelCuda) {
    println!("  Initializing GPU on device 0...");
    match cuda_model.preload_weights_gpu() {
        Ok(bytes) => {
            println!(
                "{}",
                format!("  Pre-uploaded {} MB weights to GPU", bytes / (1024 * 1024)).green()
            );
        }
        Err(e) => {
            eprintln!(
                "{}",
                format!("  Warning: Weight preload failed, using on-demand: {e}").yellow()
            );
        }
    }
}

include!("handlers_part_03_include_01.rs");
