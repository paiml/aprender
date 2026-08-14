
/// Process a batch of requests (PARITY-053)
///
/// Processes all requests in the batch and sends results via their oneshot channels.
/// Uses tokio::spawn to process requests concurrently within the batch.
#[cfg(feature = "gpu")]
async fn process_batch(
    model: &std::sync::Arc<crate::gguf::OwnedQuantizedModelCachedSync>,
    config: &BatchConfig,
    batch: &mut Vec<ContinuousBatchRequest>,
) {
    use std::time::Instant;

    if batch.is_empty() {
        return;
    }

    let batch_size = batch.len();
    let batch_start = Instant::now();

    // PARITY-095: Use configurable GPU batch threshold
    // GPU GEMM wins at batch >= gpu_threshold (default 32, from IMP-600 analysis)
    let gpu_threshold = config.gpu_threshold;

    // Use true GPU batch inference if batch is large enough and GPU cache is warm
    if batch_size >= gpu_threshold && model.is_gpu_cache_warm() {
        // PARITY-094: True batch inference with GPU FFN
        // Collect all prompts
        let prompts: Vec<Vec<u32>> = batch.iter().map(|r| r.prompt_tokens.clone()).collect();

        // Use first request's config (batch inference assumes similar parameters)
        let first = &batch[0];
        let gen_config = crate::gguf::QuantizedGenerateConfig {
            max_tokens: first.max_tokens,
            temperature: first.temperature,
            top_k: first.top_k,
            stop_tokens: Vec::new(),
            trace: false,
            ..Default::default()
        };

        // Run batch generation with GPU FFN (PARITY-021)
        let results = model.batch_generate_gpu(&prompts, &gen_config);

        let total_latency_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
        let per_request_latency_ms = total_latency_ms / batch_size as f64;

        // Send responses
        match results {
            Ok(all_token_ids) => {
                for (request, token_ids) in batch.drain(..).zip(all_token_ids) {
                    let response = ContinuousBatchResponse {
                        token_ids,
                        prompt_len: request.prompt_tokens.len(),
                        batched: true,
                        batch_size,
                        latency_ms: per_request_latency_ms,
                    };
                    let _ = request.response_tx.send(response);
                }
            },
            Err(_) => {
                // Fallback: return prompts unchanged on error
                for request in batch.drain(..) {
                    let response = ContinuousBatchResponse {
                        token_ids: request.prompt_tokens.clone(),
                        prompt_len: request.prompt_tokens.len(),
                        batched: false,
                        batch_size,
                        latency_ms: per_request_latency_ms,
                    };
                    let _ = request.response_tx.send(response);
                }
            },
        }
    } else {
        // Concurrent single-request processing (for small batches or no GPU cache)
        let mut handles = Vec::with_capacity(batch_size);

        for request in batch.drain(..) {
            let model = model.clone();
            let handle = tokio::spawn(async move {
                let start = Instant::now();

                // Build generation config
                let gen_config = crate::gguf::QuantizedGenerateConfig {
                    max_tokens: request.max_tokens,
                    temperature: request.temperature,
                    top_k: request.top_k,
                    stop_tokens: Vec::new(),
                    trace: false,
            ..Default::default()
                };

                // Generate
                let result = model.generate_with_cache(&request.prompt_tokens, &gen_config);

                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

                // Send response
                let response = match result {
                    Ok(token_ids) => ContinuousBatchResponse {
                        token_ids,
                        prompt_len: request.prompt_tokens.len(),
                        batched: false,
                        batch_size: 1,
                        latency_ms,
                    },
                    Err(_) => ContinuousBatchResponse {
                        token_ids: request.prompt_tokens.clone(),
                        prompt_len: request.prompt_tokens.len(),
                        batched: false,
                        batch_size: 1,
                        latency_ms,
                    },
                };

                // Send response (ignore if receiver dropped)
                let _ = request.response_tx.send(response);
            });

            handles.push(handle);
        }

        // Wait for all to complete
        for handle in handles {
            let _ = handle.await;
        }
    }
}

/// GPU warmup handler (PARITY-022)
/// POST /v1/gpu/warmup - Warmup GPU cache for batch inference
#[cfg(feature = "gpu")]
pub async fn gpu_warmup_handler(
    State(state): State<AppState>,
) -> Result<Json<GpuWarmupResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(cached_model) = state.cached_model() {
        match cached_model.warmup_gpu_cache() {
            Ok((memory_bytes, num_layers)) => Ok(Json(GpuWarmupResponse {
                success: true,
                memory_bytes,
                num_layers,
                message: format!(
                    "GPU cache warmed up: {} layers, {:.2} GB",
                    num_layers,
                    memory_bytes as f64 / 1e9
                ),
            })),
            Err(e) => Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: format!("GPU warmup failed: {e}"),
                }),
            )),
        }
    } else {
        Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "No GPU-capable model loaded. Use with_cached_model() to enable."
                    .to_string(),
            }),
        ))
    }
}

/// GPU warmup handler stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub async fn gpu_warmup_handler(
    State(_state): State<AppState>,
) -> Result<Json<GpuWarmupResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "GPU feature not enabled. Build with --features gpu".to_string(),
        }),
    ))
}

/// GPU status handler (PARITY-022)
/// GET /v1/gpu/status - Check GPU cache status
#[cfg(feature = "gpu")]
pub async fn gpu_status_handler(
    State(state): State<AppState>,
) -> Result<Json<GpuStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(cached_model) = state.cached_model() {
        Ok(Json(GpuStatusResponse {
            cache_ready: cached_model.is_gpu_cache_warm(),
            cache_memory_bytes: cached_model.gpu_cache_memory(),
            batch_threshold: 32, // GPU GEMM threshold from IMP-600
            recommended_min_batch: 32,
        }))
    } else {
        Ok(Json(GpuStatusResponse {
            cache_ready: false,
            cache_memory_bytes: 0,
            batch_threshold: 32,
            recommended_min_batch: 32,
        }))
    }
}

/// GPU status handler stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub async fn gpu_status_handler(
    State(_state): State<AppState>,
) -> Result<Json<GpuStatusResponse>, (StatusCode, Json<ErrorResponse>)> {
    Ok(Json(GpuStatusResponse {
        cache_ready: false,
        cache_memory_bytes: 0,
        batch_threshold: 32,
        recommended_min_batch: 32,
    }))
}

/// Encode a batch of prompts with the server's tokenizer.
///
/// aprender#2465(3): this replaced `p.bytes().map(|b| b as u32)`, which answered
/// from UTF-8 byte VALUES. A prompt "世界" became six ids (0xE4,0xB8,0x96,0xE7,0x95,
/// 0x8C) naming six unrelated vocabulary entries, and ASCII was wrong too — "Hello"
/// became five ids [72,101,108,108,111] whatever the vocabulary actually says. Every
/// completion this route returned was therefore computed from a token sequence the
/// model was never given, while the response looked exactly like a real one, and two
/// routes on one server disagreed about the tokenization of one string.
///
/// An empty prompt is refused rather than handed to the model as an empty context.
#[cfg(feature = "gpu")]
fn encode_batch_prompts(tokenizer: &BPETokenizer, prompts: &[String]) -> Result<Vec<Vec<u32>>, ApiErr> {
    let mut encoded = Vec::with_capacity(prompts.len());
    for (idx, prompt) in prompts.iter().enumerate() {
        let ids = tokenizer.encode(prompt);
        if ids.is_empty() {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Prompt {idx} is empty: nothing to generate from"),
            ));
        }
        encoded.push(ids);
    }
    Ok(encoded)
}

/// Decode a batch of generated sequences with the same tokenizer that encoded them.
///
/// aprender#2465(3), decode half: `tokens.iter().map(|&t| t as u8 as char)` read the
/// id as a byte and the byte as a codepoint, so every id above 255 wrapped and every
/// real token came back as one mojibake character. An id outside the vocabulary is
/// now an honest error, not a substituted character.
#[cfg(feature = "gpu")]
fn decode_batch_results(
    tokenizer: &BPETokenizer,
    generated: Vec<Vec<u32>>,
    prompts_tokens: &[Vec<u32>],
) -> Result<Vec<GpuBatchResult>, ApiErr> {
    let mut results = Vec::with_capacity(generated.len());
    for (idx, tokens) in generated.into_iter().enumerate() {
        let prompt_len = prompts_tokens.get(idx).map_or(0, Vec::len);
        let num_generated = tokens.len().saturating_sub(prompt_len);
        let text = tokenizer.decode(&tokens).map_err(|e| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to decode generated tokens: {e}"),
            )
        })?;
        results.push(GpuBatchResult {
            index: idx,
            token_ids: tokens,
            text,
            num_generated,
        });
    }
    Ok(results)
}

/// GPU batch completions handler (PARITY-022)
/// POST /v1/batch/completions - GPU-accelerated batch inference
#[cfg(feature = "gpu")]
pub async fn gpu_batch_completions_handler(
    State(state): State<AppState>,
    Json(request): Json<GpuBatchRequest>,
) -> Result<Json<GpuBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    use std::time::Instant;

    if request.prompts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Prompts array cannot be empty".to_string(),
            }),
        ));
    }

    let Some(cached_model) = state.cached_model() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorResponse {
                error: "No GPU-capable model loaded".to_string(),
            }),
        ));
    };

    // Check if GPU cache is ready
    let gpu_ready = cached_model.is_gpu_cache_warm();
    let batch_size = request.prompts.len();

    // aprender#2465(3): tokenize with the SAME tokenizer `/tokenize` and
    // `/v1/completions` resolve — see `encode_batch_prompts`. A route that cannot
    // resolve one fails here with the status that names why, and never falls back
    // to answering from byte values.
    let tokenizer = state.get_tokenizer(None).map_err(|e| {
        (
            super::model_resolution_status(&e),
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;
    let prompts_tokens = encode_batch_prompts(&tokenizer, &request.prompts)?;

    // Create generation config
    let gen_config = crate::gguf::QuantizedGenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: request.top_k,
        stop_tokens: vec![],
        trace: false,
            ..Default::default()
    };

    let start = Instant::now();

    // Decide GPU vs CPU path based on cache readiness and batch size
    let gpu_threshold = 32;
    let use_gpu = gpu_ready && batch_size >= gpu_threshold;

    let results = if use_gpu {
        // GPU batch inference path
        match cached_model.batch_generate_gpu(&prompts_tokens, &gen_config) {
            Ok(generated) => generated,
            Err(e) => {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: format!("GPU batch generation failed: {e}"),
                    }),
                ));
            },
        }
    } else {
        // CPU sequential path (fallback)
        let mut results = Vec::with_capacity(batch_size);
        for prompt in &prompts_tokens {
            match cached_model.generate_with_cache(prompt, &gen_config) {
                Ok(tokens) => results.push(tokens),
                Err(e) => {
                    return Err((
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(ErrorResponse {
                            error: format!("Generation failed: {e}"),
                        }),
                    ));
                },
            }
        }
        results
    };

    let elapsed = start.elapsed();
    let total_tokens: usize = results.iter().map(Vec::len).sum();
    let throughput_tps = total_tokens as f64 / elapsed.as_secs_f64();

    // Build response — decoded by the same tokenizer that encoded the prompts.
    let batch_results = decode_batch_results(&tokenizer, results, &prompts_tokens)?;

    Ok(Json(GpuBatchResponse {
        results: batch_results,
        stats: GpuBatchStats {
            batch_size,
            gpu_used: use_gpu,
            total_tokens,
            processing_time_ms: elapsed.as_secs_f64() * 1000.0,
            throughput_tps,
        },
    }))
}

/// GPU batch completions handler stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub async fn gpu_batch_completions_handler(
    State(_state): State<AppState>,
    Json(_request): Json<GpuBatchRequest>,
) -> Result<Json<GpuBatchResponse>, (StatusCode, Json<ErrorResponse>)> {
    Err((
        StatusCode::SERVICE_UNAVAILABLE,
        Json(ErrorResponse {
            error: "GPU feature not enabled. Build with --features gpu".to_string(),
        }),
    ))
}

/// Models list handler - returns available models in multi-model mode
pub async fn models_handler(
    State(state): State<AppState>,
) -> Result<Json<ModelsResponse>, (StatusCode, Json<ErrorResponse>)> {
    if let Some(registry) = &state.registry {
        let models = registry.list();
        Ok(Json(ModelsResponse { models }))
    } else {
        // Single model mode - return the single model info
        Ok(Json(ModelsResponse {
            models: vec![ModelInfo {
                id: "default".to_string(),
                name: "Default Model".to_string(),
                description: "Single model deployment".to_string(),
                // aprender#2376(6): was the literal "unknown" while /realize/model
                // hardcoded "gguf" — two endpoints on one server contradicting each
                // other about one model. Both now read the resident backend.
                format: state.model_format().to_string(),
                loaded: state.model_loaded(),
            }],
        }))
    }
}

/// Tokenize text handler
pub async fn tokenize_handler(
    State(state): State<AppState>,
    Json(request): Json<TokenizeRequest>,
) -> Result<Json<TokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    // aprender#2376(1): tokenizing needs a tokenizer, not the dense f32 Model.
    // Asking get_model() for one made this route unreachable on every
    // `apr serve run model.gguf`, where the weights are in `quantized_model`.
    let tokenizer = state.get_tokenizer(request.model_id.as_deref()).map_err(|e| {
        (
            super::model_resolution_status(&e),
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    let token_ids = tokenizer.encode(&request.text);
    let num_tokens = token_ids.len();

    Ok(Json(TokenizeResponse {
        token_ids,
        num_tokens,
    }))
}

// ── generate_handler backend dispatch ────────────────────────────────

/// Generate text handler
#[cfg(feature = "cuda")]
fn try_cuda_generate(
    state: &AppState,
    request: &GenerateRequest,
    cancel: &CancelToken,
) -> Result<Option<GenerateResponse>, ApiErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let cuda_model_lock = match state.cuda_model() {
        Some(l) => l,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_tokens = prompt_ids.len();

    let q_config = QuantizedGenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: if request.temperature == 0.0 {
            1
        } else {
            request.top_k
        },
        stop_tokens: vec![eos_id(&tokenizer, state.model_eos_token_id())],
        trace: false,
        cancel: cancel.clone(),
        ..Default::default()
    };

    let mut cuda_model = cuda_model_lock.write().map_err(|_| {
        api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to acquire CUDA model lock",
        )
    })?;
    let generated = cuda_model
        .generate_gpu_resident(&prompt_ids, &q_config)
        .map_err(|e| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("CUDA generation failed: {e}"),
            )
        })?;
    let text = tokenizer
        .decode(&generated)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Some(GenerateResponse {
        num_generated: generated.len().saturating_sub(prompt_tokens),
        token_ids: generated,
        text,
    }))
}
