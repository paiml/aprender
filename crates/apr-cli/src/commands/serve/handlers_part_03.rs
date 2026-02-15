
/// Handle POST /v1/completions for GPU inference.
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn handle_gpu_completion(
    cuda: &std::sync::Mutex<realizar::apr::AprV2ModelCuda>,
    tok_info: Option<&SafeTensorsTokenizerInfo>,
    req: &GpuCompletionRequest,
    cpu_state: &std::sync::Mutex<AprServerState>,
) -> axum::response::Response {
    use axum::{http::StatusCode, response::IntoResponse, Json};

    let start = Instant::now();
    let input_tokens = encode_prompt(tok_info, &req.prompt);
    let eos_id = eos_token_id(tok_info, 2);

    let gen_start = Instant::now();
    let output_tokens =
        match run_gpu_generation(cuda, &input_tokens, req.max_tokens.min(128), eos_id) {
            Ok(t) => t,
            Err(gpu_err) => {
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
                match run_apr_cpu_inference(
                    &s,
                    &req.prompt,
                    req.max_tokens.min(128),
                    0.0,
                ) {
                    Ok(out) => {
                        return Json(serde_json::json!({
                            "text": out.text,
                            "tokens_generated": out.tokens_generated,
                            "latency_ms": out.gen_duration.as_millis() as u64,
                            "tok_per_sec": compute_tok_per_sec(out.tokens_generated, out.gen_duration),
                            "compute_mode": "cpu-fallback"
                        }))
                        .into_response();
                    }
                    Err(cpu_err) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            Json(serde_json::json!({
                                "error": format!("GPU failed: {gpu_err}; CPU fallback also failed: {cpu_err}")
                            })),
                        )
                            .into_response();
                    }
                }
            }
        };
    let gen_time = gen_start.elapsed();

    let new_tokens = extract_new_tokens(&output_tokens, input_tokens.len());
    let text = decode_tokens(tok_info, new_tokens);

    Json(GpuCompletionResponse {
        text,
        tokens_generated: new_tokens.len(),
        latency_ms: start.elapsed().as_millis() as u64,
        tok_per_sec: compute_tok_per_sec(new_tokens.len(), gen_time),
    })
    .into_response()
}

/// Handle POST /v1/chat/completions for GPU inference (PAR-302).
#[cfg(feature = "inference")]
#[allow(clippy::disallowed_methods)] // serde_json::json!() uses infallible unwrap
fn handle_gpu_chat_completion(
    cuda: &std::sync::Mutex<realizar::apr::AprV2ModelCuda>,
    tok_info: Option<&SafeTensorsTokenizerInfo>,
    req: &serde_json::Value,
    cpu_state: &std::sync::Mutex<AprServerState>,
) -> axum::response::Response {
    use axum::{
        response::{
            sse::{Event, Sse},
            IntoResponse,
        },
        Json,
    };
    use futures_util::stream;

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
    let input_tokens = encode_prompt(tok_info, &prompt);
    let eos_id = eos_token_id(tok_info, 151_645);

    let gen_start = Instant::now();
    let output_tokens = match run_gpu_generation(cuda, &input_tokens, max_tokens.min(256), eos_id) {
        Ok(t) => t,
        Err(gpu_err) => {
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
            match run_apr_cpu_inference(&s, &prompt, max_tokens.min(256), temperature) {
                Ok(out) => {
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
                Err(cpu_err) => {
                    return Json(serde_json::json!({
                        "error": format!("GPU failed: {gpu_err}; CPU fallback also failed: {cpu_err}")
                    }))
                    .into_response();
                }
            }
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

    let output_text = decode_tokens(tok_info, new_tokens);
    let tokens_generated = new_tokens.len();
    let tok_per_sec = compute_tok_per_sec(tokens_generated, elapsed);
    let request_id = generate_request_id();
    let created = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if stream_mode {
        let response = serde_json::json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": "apr-gpu",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": output_text}, "finish_reason": "stop"}]
        });
        let stream = stream::once(async move {
            Ok::<_, std::convert::Infallible>(Event::default().data(response.to_string()))
        });
        Sse::new(stream).into_response()
    } else {
        Json(serde_json::json!({
            "id": request_id,
            "object": "chat.completion",
            "created": created,
            "model": "apr-gpu",
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
            quantized_model.layers.len(),
            quantized_model.config.vocab_size,
            quantized_model.config.hidden_dim
        )
        .green()
    );

    let vocab = extract_gguf_vocab(&mapped_model, quantized_model.config.vocab_size);

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
            let vocab = extract_gguf_vocab(mapped_model, quantized_model.config.vocab_size);
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

/// Run an axum server with graceful shutdown and standard banner.
#[cfg(feature = "inference")]
fn run_server_async(app: axum::Router, bind_addr: &str, label: &str) -> Result<()> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|e| CliError::InferenceFailed(format!("Failed to create runtime: {e}")))?;

    let bind_addr = bind_addr.to_string();
    let label = label.to_string();

    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind_addr)
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Failed to bind: {e}")))?;

        println!();
        println!(
            "{}",
            format!("{label} server listening on http://{bind_addr}")
                .green()
                .bold()
        );
        println!();
        println!("{}", "Endpoints:".cyan());
        println!("  GET  /health              - Health check");
        println!("  GET  /metrics             - Prometheus metrics");
        println!("  POST /generate            - Text generation");
        println!("  POST /v1/completions      - OpenAI-compatible");
        println!("  POST /v1/chat/completions - Chat completions");
        println!();
        println!("{}", "Press Ctrl+C to stop".dimmed());

        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .map_err(|e| CliError::InferenceFailed(format!("Server error: {e}")))?;

        Ok(())
    })
}
