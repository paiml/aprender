
fn try_quantized_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<Option<GenerateResponse>, ApiErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let quantized_model = match state.quantized_model() {
        Some(m) => m,
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
        trace: state.is_trace_enabled(),
        ..Default::default()
    };

    // GH-95: Use generate_with_cache for O(n) autoregressive generation.
    // The previous .generate() call was O(n²) — it reprocessed the entire
    // token sequence on every step. With KV cache, only the new token is
    // processed each step. Contract: gguf-cpu-cache-v1.yaml
    let generated = quantized_model
        .generate_with_cache(&prompt_ids, &q_config)
        .map_err(|e| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("CPU generation failed: {e}"),
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

#[cfg(feature = "cuda")]
async fn try_apr_q4k_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<Option<GenerateResponse>, ApiErr> {
    use super::apr_q4k_scheduler::AprQ4kRequest;

    let q4k_tx = match state.apr_q4k_tx() {
        Some(tx) => tx,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_ids_copy = prompt_ids.clone();

    let (response_tx, response_rx) = tokio::sync::oneshot::channel();

    // ALB-109: Get EOS token IDs from model config or tokenizer.
    // Qwen3 uses 151643 (<|endoftext|>), not the default 0/2.
    let eos_ids = state.model_eos_ids();

    q4k_tx
        .send(AprQ4kRequest {
            prompt_ids,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            eos_ids,
            response_tx,
        })
        .await
        .map_err(|_| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Q4K inference thread not available",
            )
        })?;

    let result = response_rx.await.map_err(|_| {
        api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Q4K inference thread dropped response",
        )
    })?;

    let resp = result.map_err(|e| {
        api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Q4K generation failed: {e}"),
        )
    })?;

    // Build full token sequence (prompt + generated) and decode
    let mut all_tokens = prompt_ids_copy;
    all_tokens.extend_from_slice(&resp.output_tokens);
    let text = tokenizer
        .decode(&all_tokens)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(Some(GenerateResponse {
        num_generated: resp.tokens_generated,
        token_ids: all_tokens,
        text,
    }))
}

fn try_apr_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<Option<GenerateResponse>, ApiErr> {
    use crate::apr_transformer::GenerateConfig;

    let apr_transformer = match state.apr_transformer() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;
    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt_tokens = prompt_ids.len();

    let gen_config = GenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        ..Default::default()
    };

    let generated = apr_transformer
        .generate_with_cache(&prompt_ids, &gen_config)
        .map_err(|e| {
            api_err(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("APR generation failed: {e}"),
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

fn registry_generate(
    state: &AppState,
    request: &GenerateRequest,
) -> Result<GenerateResponse, ApiErr> {
    let (model, tokenizer) = state
        .get_model(request.model_id.as_deref())
        .map_err(|e| api_err(StatusCode::NOT_FOUND, e))?;

    let prompt_ids = tokenize_prompt(&tokenizer, &request.prompt)?;
    let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();

    let strategy = match request.strategy.as_str() {
        "greedy" => SamplingStrategy::Greedy,
        "top_k" => SamplingStrategy::TopK { k: request.top_k },
        "top_p" => SamplingStrategy::TopP { p: request.top_p },
        other => {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Invalid strategy: {other}"),
            ))
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);
    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    let generated = model
        .generate(&prompt, &config)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let token_ids: Vec<u32> = generated
        .iter()
        .map(|&id| {
            u32::try_from(id).map_err(|_| {
                api_err(
                    StatusCode::BAD_REQUEST,
                    format!("Token ID {id} exceeds u32 range"),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let text = tokenizer
        .decode(&token_ids)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    Ok(GenerateResponse {
        num_generated: generated.len() - prompt.len(),
        token_ids,
        text,
    })
}

pub async fn generate_handler(
    State(state): State<AppState>,
    Json(request): Json<GenerateRequest>,
) -> Result<Json<GenerateResponse>, ApiErr> {
    use std::time::Instant;
    let start = Instant::now();

    if state.is_verbose() {
        eprintln!(
            "[VERBOSE] POST /generate prompt={:?} max_tokens={}",
            &request.prompt.chars().take(50).collect::<String>(),
            request.max_tokens
        );
    }

    #[cfg(feature = "cuda")]
    if let Some(resp) = try_cuda_generate(&state, &request)? {
        state
            .metrics
            .record_success(resp.num_generated, start.elapsed());
        return Ok(Json(resp));
    }

    if let Some(resp) = try_quantized_generate(&state, &request)? {
        state
            .metrics
            .record_success(resp.num_generated, start.elapsed());
        return Ok(Json(resp));
    }

    #[cfg(feature = "cuda")]
    if let Some(resp) = try_apr_q4k_generate(&state, &request).await? {
        state
            .metrics
            .record_success(resp.num_generated, start.elapsed());
        return Ok(Json(resp));
    }

    if let Some(resp) = try_apr_generate(&state, &request)? {
        state
            .metrics
            .record_success(resp.num_generated, start.elapsed());
        return Ok(Json(resp));
    }

    let resp = registry_generate(&state, &request)?;
    state
        .metrics
        .record_success(resp.num_generated, start.elapsed());
    Ok(Json(resp))
}

/// Batch tokenize handler
pub async fn batch_tokenize_handler(
    State(state): State<AppState>,
    Json(request): Json<BatchTokenizeRequest>,
) -> Result<Json<BatchTokenizeResponse>, (StatusCode, Json<ErrorResponse>)> {
    if request.texts.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: "Texts array cannot be empty".to_string(),
            }),
        ));
    }

    // Get tokenizer (use default model)
    let (_model, tokenizer) = state.get_model(None).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                error: e.to_string(),
            }),
        )
    })?;

    // Tokenize all texts
    let results: Vec<TokenizeResponse> = request
        .texts
        .iter()
        .map(|text| {
            let token_ids = tokenizer.encode(text);
            let num_tokens = token_ids.len();
            TokenizeResponse {
                token_ids,
                num_tokens,
            }
        })
        .collect();

    Ok(Json(BatchTokenizeResponse { results }))
}

// ── batch_generate_handler backend dispatch ─────────────────────────

/// Batch generate handler
#[cfg(feature = "cuda")]
fn try_cuda_batch_generate(
    state: &AppState,
    request: &BatchGenerateRequest,
) -> Result<Option<Vec<GenerateResponse>>, ApiErr> {
    use crate::gguf::QuantizedGenerateConfig;

    let cuda_model_lock = match state.cuda_model() {
        Some(l) => l,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;

    let q_config = QuantizedGenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        top_k: if request.temperature == 0.0 {
            1
        } else {
            request.top_k
        },
        stop_tokens: vec![eos_id(&tokenizer, state.model_eos_token_id())],
        trace: state.is_trace_enabled(),
        ..Default::default()
    };

    let mut results = Vec::with_capacity(request.prompts.len());
    let mut cuda_model = cuda_model_lock.write().map_err(|_| {
        api_err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to acquire CUDA model lock",
        )
    })?;

    for prompt_text in &request.prompts {
        let prompt_ids = tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Prompt '{prompt_text}' tokenizes to empty sequence"),
            ));
        }
        let prompt_tokens = prompt_ids.len();
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
        results.push(GenerateResponse {
            num_generated: generated.len().saturating_sub(prompt_tokens),
            token_ids: generated,
            text,
        });
    }

    Ok(Some(results))
}

fn try_apr_batch_generate(
    state: &AppState,
    request: &BatchGenerateRequest,
) -> Result<Option<Vec<GenerateResponse>>, ApiErr> {
    use crate::apr_transformer::GenerateConfig;

    let apr_transformer = match state.apr_transformer() {
        Some(m) => m,
        None => return Ok(None),
    };
    let tokenizer = require_tok(state)?;

    let gen_config = GenerateConfig {
        max_tokens: request.max_tokens,
        temperature: request.temperature,
        ..Default::default()
    };

    let mut results = Vec::with_capacity(request.prompts.len());

    for prompt_text in &request.prompts {
        let prompt_ids = tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Prompt '{prompt_text}' tokenizes to empty sequence"),
            ));
        }
        let prompt_tokens = prompt_ids.len();
        let generated = apr_transformer
            .generate_with_cache(&prompt_ids, &gen_config)
            .map_err(|e| {
                api_err(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("APR generation failed: {e}"),
                )
            })?;
        let text = tokenizer
            .decode(&generated)
            .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        results.push(GenerateResponse {
            num_generated: generated.len().saturating_sub(prompt_tokens),
            token_ids: generated,
            text,
        });
    }

    Ok(Some(results))
}

fn registry_batch_generate(
    state: &AppState,
    request: &BatchGenerateRequest,
) -> Result<Vec<GenerateResponse>, ApiErr> {
    let (model, tokenizer) = state
        .get_model(None)
        .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;

    let strategy = match request.strategy.as_str() {
        "greedy" => SamplingStrategy::Greedy,
        "top_k" => SamplingStrategy::TopK { k: request.top_k },
        "top_p" => SamplingStrategy::TopP { p: request.top_p },
        other => {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Invalid strategy: {other}"),
            ))
        },
    };

    let mut config = GenerationConfig::default()
        .with_max_tokens(request.max_tokens)
        .with_temperature(request.temperature);
    config.strategy = strategy;
    if let Some(seed) = request.seed {
        config = config.with_seed(seed);
    }

    let mut results = Vec::with_capacity(request.prompts.len());

    for prompt_text in &request.prompts {
        let prompt_ids = tokenizer.encode(prompt_text);
        if prompt_ids.is_empty() {
            return Err(api_err(
                StatusCode::BAD_REQUEST,
                format!("Prompt '{prompt_text}' tokenizes to empty sequence"),
            ));
        }
        let prompt: Vec<usize> = prompt_ids.iter().map(|&id| id as usize).collect();
        let generated = model
            .generate(&prompt, &config)
            .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let token_ids: Vec<u32> = generated
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| {
                    api_err(
                        StatusCode::BAD_REQUEST,
                        format!("Token ID {id} exceeds u32 range"),
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let text = tokenizer
            .decode(&token_ids)
            .map_err(|e| api_err(StatusCode::INTERNAL_SERVER_ERROR, e))?;
        results.push(GenerateResponse {
            num_generated: generated.len() - prompt.len(),
            token_ids,
            text,
        });
    }

    Ok(results)
}

pub async fn batch_generate_handler(
    State(state): State<AppState>,
    Json(request): Json<BatchGenerateRequest>,
) -> Result<Json<BatchGenerateResponse>, ApiErr> {
    if request.prompts.is_empty() {
        return Err(api_err(
            StatusCode::BAD_REQUEST,
            "Prompts array cannot be empty",
        ));
    }

    #[cfg(feature = "cuda")]
    if let Some(results) = try_cuda_batch_generate(&state, &request)? {
        return Ok(Json(BatchGenerateResponse { results }));
    }

    if let Some(results) = try_apr_batch_generate(&state, &request)? {
        return Ok(Json(BatchGenerateResponse { results }));
    }

    let results = registry_batch_generate(&state, &request)?;
    Ok(Json(BatchGenerateResponse { results }))
}
